use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_trait::async_trait;
use goose_agent::{
    events::AgentEvent,
    operation::{ConversationEffect, Emitter, Operation, OperationResult, RunStatus},
    steer::{SteerHook, SteerOperation, SteerQueue},
};
use goose_provider_types::conversation::{message::Message, Conversation};
use rmcp::model::{CallToolRequestParams, CallToolResult, ContentBlock};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[derive(Default)]
struct RecordingHook {
    seen: Mutex<Vec<String>>,
}

#[async_trait]
impl SteerHook<()> for RecordingHook {
    async fn on_steer(&self, _session: &(), message: &Message) {
        self.seen.lock().unwrap().push(message.as_concat_text());
    }
}

struct Outcome {
    status: Option<RunStatus>,
    appended: Vec<Message>,
    emitted: Vec<String>,
}

async fn run(operation: &SteerOperation<()>, messages: Vec<Message>) -> Result<Outcome> {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let emit = Emitter::new(tx, CancellationToken::new());
    let conversation = Conversation::new_unvalidated(messages);
    let result =
        Operation::<(), ConversationEffect>::run(operation, &(), &conversation, &emit).await?;
    let emitted = std::iter::from_fn(|| rx.try_recv().ok())
        .filter_map(|event| match event {
            AgentEvent::Message(message) => Some(message.as_concat_text()),
            _ => None,
        })
        .collect();
    let (status, appended) = match result {
        OperationResult::NotApplicable => (None, Vec::new()),
        OperationResult::Applied(result) => (
            Some(result.status),
            result
                .effects
                .into_iter()
                .map(|effect| match effect {
                    ConversationEffect::AppendMessage(message) => message,
                    _ => panic!("expected an appended message"),
                })
                .collect(),
        ),
    };
    Ok(Outcome {
        status,
        appended,
        emitted,
    })
}

fn tool_request() -> Message {
    Message::assistant().with_tool_request("call", Ok(CallToolRequestParams::new("look")))
}

fn tool_response() -> Message {
    Message::user().with_tool_response(
        "call",
        Ok(CallToolResult::success(vec![ContentBlock::text("done")])),
    )
}

fn queue_with(texts: &[&str]) -> SteerQueue {
    let queue = SteerQueue::new();
    for text in texts {
        queue.push(Message::user().with_text(*text));
    }
    queue
}

#[tokio::test]
async fn does_nothing_without_queued_messages() -> Result<()> {
    let outcome = run(
        &SteerOperation::new(SteerQueue::new()),
        vec![
            Message::user().with_text("hi"),
            tool_request(),
            tool_response(),
        ],
    )
    .await?;

    assert!(outcome.status.is_none());
    Ok(())
}

#[tokio::test]
async fn waits_while_a_tool_call_is_unanswered() -> Result<()> {
    let queue = queue_with(&["stop, do this instead"]);

    let outcome = run(
        &SteerOperation::new(queue.clone()),
        vec![Message::user().with_text("hi"), tool_request()],
    )
    .await?;

    assert!(outcome.status.is_none());
    assert!(!queue.is_empty());
    Ok(())
}

#[tokio::test]
async fn adds_queued_messages_after_tool_results_in_order() -> Result<()> {
    let queue = queue_with(&["first", "second"]);
    let hook = Arc::new(RecordingHook::default());
    let operation = SteerOperation::new(queue.clone()).with_hook(hook.clone());

    let outcome = run(
        &operation,
        vec![
            Message::user().with_text("hi"),
            tool_request(),
            tool_response(),
        ],
    )
    .await?;

    assert_eq!(outcome.status, Some(RunStatus::Continuing));
    let texts: Vec<_> = outcome
        .appended
        .iter()
        .map(Message::as_concat_text)
        .collect();
    assert_eq!(texts, ["first", "second"]);
    assert!(outcome
        .appended
        .iter()
        .all(|message| message.metadata.steer));
    assert_eq!(outcome.emitted, ["first", "second"]);
    assert_eq!(*hook.seen.lock().unwrap(), ["first", "second"]);
    assert!(queue.is_empty());
    Ok(())
}

#[tokio::test]
async fn adds_queued_messages_after_a_finished_reply() -> Result<()> {
    let outcome = run(
        &SteerOperation::new(queue_with(&["one more thing"])),
        vec![
            Message::user().with_text("hi"),
            Message::assistant().with_text("done"),
        ],
    )
    .await?;

    assert_eq!(outcome.status, Some(RunStatus::Continuing));
    assert_eq!(outcome.appended.len(), 1);
    Ok(())
}
