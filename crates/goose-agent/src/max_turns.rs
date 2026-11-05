//! Ends the run once the agent has used its budget of model turns since the
//! last user message, and tells the model how much budget is left once it
//! has used half.

use anyhow::Result;
use async_trait::async_trait;
use goose_provider_types::conversation::message::Message;
use goose_provider_types::conversation::Conversation;
use goose_provider_types::maybe_send::{MaybeSend, MaybeSync};

use crate::operation::{
    assistant_turn_count, ends_turn, messages_since_kickoff, not_applicable, yielded_with, Emitter,
    Operation, OperationResult,
};

#[cfg(not(target_arch = "wasm32"))]
type LimitMessage = dyn Fn() -> Message + Send + Sync;

#[cfg(target_arch = "wasm32")]
type LimitMessage = dyn Fn() -> Message;

pub struct MaxTurnsOperation {
    max_turns: u32,
    limit_message: Option<Box<LimitMessage>>,
}

impl MaxTurnsOperation {
    /// A budget of 0 ends the run before the first model call.
    pub fn new(max_turns: u32) -> Self {
        Self {
            max_turns,
            limit_message: None,
        }
    }

    /// The message to add to the conversation when the budget runs out.
    pub fn with_limit_message(
        mut self,
        message: impl Fn() -> Message + MaybeSend + MaybeSync + 'static,
    ) -> Self {
        self.limit_message = Some(Box::new(message));
        self
    }
}

fn turn_budget_part(turns_taken: u32, max_turns: u32) -> Option<String> {
    if max_turns == 0 || turns_taken.saturating_mul(2) < max_turns {
        return None;
    }
    Some(format!(
        "<turn-budget>{turns_taken}/{max_turns} used</turn-budget>"
    ))
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl<S, E> Operation<S, E> for MaxTurnsOperation
where
    S: MaybeSend + MaybeSync + 'static,
    E: From<Message> + MaybeSend + 'static,
{
    fn name(&self) -> &'static str {
        "max_turns"
    }

    async fn moim_parts(&self, _session: &S, conversation: &Conversation) -> Result<Vec<String>> {
        let turns_taken = assistant_turn_count(messages_since_kickoff(conversation)?);
        Ok(turn_budget_part(turns_taken, self.max_turns)
            .into_iter()
            .collect())
    }

    async fn run(
        &self,
        _session: &S,
        conversation: &Conversation,
        emit: &Emitter,
    ) -> Result<OperationResult<E>> {
        let messages = messages_since_kickoff(conversation)?;
        // A reply that finished on its own did not run out of budget.
        if ends_turn(messages) || assistant_turn_count(messages) < self.max_turns {
            return not_applicable();
        }
        let message = self
            .limit_message
            .as_ref()
            .map(|build| emit.message(build()).into());
        yielded_with(message)
    }
}
