//! Adds user guidance that arrives while a run is in progress.
//!
//! Clients push messages into a [`SteerQueue`] at any time. [`SteerOperation`]
//! adds them to the conversation at the next point between turns, after
//! every pending tool call has been answered, so the model sees them on its
//! next request without the run having to stop.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_trait::async_trait;
use goose_provider_types::conversation::message::Message;
use goose_provider_types::conversation::{Conversation, EffectiveRole};
use goose_provider_types::maybe_send::{MaybeSend, MaybeSync};

use crate::operation::{
    applied, ends_turn, last_effective_role, messages_since_kickoff, not_applicable, Emitter,
    Operation, OperationResult,
};

#[derive(Clone, Default)]
pub struct SteerQueue(Arc<Mutex<VecDeque<Message>>>);

impl SteerQueue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&self, message: Message) {
        self.0.lock().unwrap().push_back(message);
    }

    pub fn is_empty(&self) -> bool {
        self.0.lock().unwrap().is_empty()
    }

    fn drain(&self) -> Vec<Message> {
        self.0.lock().unwrap().drain(..).collect()
    }
}

/// Called once for each steering message after it is added to the conversation.
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub trait SteerHook<S>: MaybeSend + MaybeSync {
    async fn on_steer(&self, session: &S, message: &Message);
}

pub struct SteerOperation<S> {
    queue: SteerQueue,
    hook: Option<Arc<dyn SteerHook<S>>>,
    drained: Mutex<Vec<Message>>,
}

impl<S> SteerOperation<S> {
    pub fn new(queue: SteerQueue) -> Self {
        Self {
            queue,
            hook: None,
            drained: Mutex::default(),
        }
    }

    pub fn with_hook(mut self, hook: Arc<dyn SteerHook<S>>) -> Self {
        self.hook = Some(hook);
        self
    }

    fn take_drained<E: From<Message>>(&self) -> Vec<E> {
        std::mem::take(&mut *self.drained.lock().unwrap())
            .into_iter()
            .map(E::from)
            .collect()
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl<S, E> Operation<S, E> for SteerOperation<S>
where
    S: MaybeSend + MaybeSync + 'static,
    E: From<Message> + MaybeSend + 'static,
{
    fn name(&self) -> &'static str {
        "steer"
    }

    async fn finalize_cancellation(
        &self,
        _session: &S,
        _conversation: &Conversation,
        _emit: &Emitter,
    ) -> Vec<E> {
        self.take_drained()
    }

    async fn run(
        &self,
        session: &S,
        conversation: &Conversation,
        emit: &Emitter,
    ) -> Result<OperationResult<E>> {
        let messages = messages_since_kickoff(conversation)?;
        let between_turns =
            ends_turn(messages) || last_effective_role(messages)? == EffectiveRole::Tool;
        if !between_turns {
            return not_applicable();
        }

        let pending = self.queue.drain();
        if pending.is_empty() {
            return not_applicable();
        }

        let drained: Vec<Message> = pending
            .into_iter()
            .map(|message| emit.message(message.with_steer()))
            .collect();
        *self.drained.lock().unwrap() = drained.clone();
        if let Some(hook) = &self.hook {
            for message in &drained {
                hook.on_steer(session, message).await;
            }
        }
        applied(self.take_drained())
    }
}
