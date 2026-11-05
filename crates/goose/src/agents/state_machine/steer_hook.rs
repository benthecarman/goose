use async_trait::async_trait;
use goose_agent::steer::SteerHook;

use crate::conversation::message::Message;
use crate::hooks::{HookContext, HookEvent, HookManager};
use crate::session::Session;

pub struct GooseSteerHook {
    hook_manager: HookManager,
}

impl GooseSteerHook {
    pub fn new(hook_manager: HookManager) -> Self {
        Self { hook_manager }
    }
}

#[async_trait]
impl SteerHook<Session> for GooseSteerHook {
    async fn on_steer(&self, session: &Session, message: &Message) {
        let context = HookContext::new(HookEvent::UserPromptSubmit, &session.id)
            .with_message(message.agent_visible_content().as_concat_text());
        self.hook_manager
            .emit(HookEvent::UserPromptSubmit, context)
            .await;
    }
}
