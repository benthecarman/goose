use anyhow::Result;
use goose_agent::{
    max_turns::MaxTurnsOperation,
    operation::{ConversationEffect, Emitter, Operation, OperationResult, RunStatus},
};
use goose_provider_types::conversation::{message::Message, Conversation};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

fn after_turns(turns: usize) -> Conversation {
    let mut messages = vec![Message::user().with_text("go")];
    for turn in 0..turns {
        messages.push(Message::assistant().with_text(format!("turn {turn}")));
        messages.push(
            Message::user()
                .with_text("continue")
                .with_visibility(false, true),
        );
    }
    Conversation::new_unvalidated(messages)
}

async fn run(
    operation: &MaxTurnsOperation,
    conversation: &Conversation,
) -> Result<OperationResult<ConversationEffect>> {
    let (tx, _rx) = mpsc::unbounded_channel();
    let emit = Emitter::new(tx, CancellationToken::new());
    Operation::<(), ConversationEffect>::run(operation, &(), conversation, &emit).await
}

async fn moim(operation: &MaxTurnsOperation, conversation: &Conversation) -> Result<Vec<String>> {
    Operation::<(), ConversationEffect>::moim_parts(operation, &(), conversation).await
}

#[tokio::test]
async fn continues_under_the_budget() -> Result<()> {
    let result = run(&MaxTurnsOperation::new(3), &after_turns(2)).await?;

    assert!(matches!(result, OperationResult::NotApplicable));
    Ok(())
}

#[tokio::test]
async fn yields_without_a_message_at_the_budget_by_default() -> Result<()> {
    let OperationResult::Applied(result) = run(&MaxTurnsOperation::new(3), &after_turns(3)).await?
    else {
        panic!("expected the operation to apply");
    };

    assert_eq!(result.status, RunStatus::Yielded);
    assert!(result.effects.is_empty());
    Ok(())
}

#[tokio::test]
async fn yields_with_the_limit_message_when_set() -> Result<()> {
    let operation = MaxTurnsOperation::new(3)
        .with_limit_message(|| Message::assistant().with_text("out of turns"));

    let OperationResult::Applied(result) = run(&operation, &after_turns(3)).await? else {
        panic!("expected the operation to apply");
    };

    assert_eq!(result.status, RunStatus::Yielded);
    let [ConversationEffect::AppendMessage(message)] = result.effects.as_slice() else {
        panic!("expected one appended message");
    };
    assert_eq!(message.as_concat_text(), "out of turns");
    Ok(())
}

#[tokio::test]
async fn does_not_claim_the_limit_when_the_last_reply_finished_the_turn() -> Result<()> {
    let mut messages = after_turns(2).messages().to_vec();
    messages.push(Message::assistant().with_text("all done"));
    let operation = MaxTurnsOperation::new(3)
        .with_limit_message(|| Message::assistant().with_text("out of turns"));

    let result = run(&operation, &Conversation::new_unvalidated(messages)).await?;

    assert!(matches!(result, OperationResult::NotApplicable));
    Ok(())
}

#[tokio::test]
async fn tells_the_model_its_budget_once_half_is_used() -> Result<()> {
    let operation = MaxTurnsOperation::new(4);

    assert!(moim(&operation, &after_turns(1)).await?.is_empty());
    assert_eq!(
        moim(&operation, &after_turns(2)).await?,
        ["<turn-budget>2/4 used</turn-budget>"]
    );
    Ok(())
}
