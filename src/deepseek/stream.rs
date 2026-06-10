use std::error::Error as StdError;

use futures_util::StreamExt;
use reqwest_eventsource::{Event, EventSource};
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::{DeepSeekError, FinishReason, StreamEvent, ToolCallDelta, UsageData};

pub(super) enum StreamAttemptOutcome {
    Complete,
    Cancelled,
    ShouldRetry,
}

pub(super) async fn run_stream_attempt(
    mut event_source: EventSource,
    tx: &mpsc::UnboundedSender<Result<StreamEvent, DeepSeekError>>,
    cancellation_token: &CancellationToken,
    attempt: u32,
    max_retries: u32,
) -> StreamAttemptOutcome {
    let mut saw_finish = false;
    let mut events_sent: u32 = 0;

    loop {
        let event = tokio::select! {
            () = cancellation_token.cancelled() => return StreamAttemptOutcome::Cancelled,
            event = event_source.next() => event,
        };

        let Some(event) = event else {
            break;
        };

        match event {
            Ok(Event::Open) => {}
            Ok(Event::Message(message)) => {
                if message.data.trim() == "[DONE]" {
                    break;
                }
                match parse_chat_completion_chunk(&message.data) {
                    Ok(updates) => {
                        for update in updates {
                            if matches!(update, StreamEvent::Finished(_)) {
                                saw_finish = true;
                            }
                            events_sent += 1;
                            if tx.send(Ok(update)).is_err() {
                                return StreamAttemptOutcome::Cancelled;
                            }
                        }
                    }
                    Err(error) => {
                        let _ = tx.send(Err(error));
                        return StreamAttemptOutcome::Cancelled;
                    }
                }
            }
            Err(error) => {
                if events_sent == 0 && attempt < max_retries && is_retryable_transport_error(&error)
                {
                    tracing::warn!(
                        error = ?error,
                        attempt,
                        max_retries,
                        "retryable transport error, will retry"
                    );
                    return StreamAttemptOutcome::ShouldRetry;
                }
                tracing::error!(
                    error = ?error,
                    events_sent,
                    attempt,
                    "non-retryable stream error or max retries exceeded"
                );
                let _ = tx.send(Err(error.into()));
                return StreamAttemptOutcome::Cancelled;
            }
        }
    }

    if !saw_finish && !cancellation_token.is_cancelled() {
        let _ = tx.send(Err(DeepSeekError::InvalidResponse(
            "stream ended before a finish reason was received".to_string(),
        )));
    }
    StreamAttemptOutcome::Complete
}

/// Returns true when a transport error is safe to retry before any events are emitted.
///
/// Only `Transport`-variant errors carrying network-level conditions qualify - these
/// arise from stale pooled connections, server-side shutdowns, and TCP disconnects.
/// Parse errors, status errors, and redirect loops are not retryable.
///
/// Two broad categories are accepted:
/// - `is_connect()` / `is_request()`: server dropped the connection before any HTTP
///   response arrived (e.g. `hyper::Error(IncompleteMessage)` on a stale pool reuse).
/// - Body/decode errors whose source chain contains a retryable IO kind
///   (`BrokenPipe`, `ConnectionReset`, `UnexpectedEof`, `ConnectionAborted`): TCP dropped
///   mid-stream, which is safe to restart only while `events_sent == 0` (guarded by
///   the caller).
pub(crate) fn is_retryable_transport_error(error: &reqwest_eventsource::Error) -> bool {
    let reqwest_eventsource::Error::Transport(ref reqwest_error) = *error else {
        return false;
    };

    if reqwest_error.is_connect() || reqwest_error.is_request() {
        return true;
    }

    let mut current: Option<&(dyn StdError + 'static)> = (reqwest_error as &dyn StdError).source();
    while let Some(err) = current {
        if let Some(io) = err.downcast_ref::<std::io::Error>() {
            return matches!(
                io.kind(),
                std::io::ErrorKind::BrokenPipe
                    | std::io::ErrorKind::ConnectionReset
                    | std::io::ErrorKind::UnexpectedEof
                    | std::io::ErrorKind::ConnectionAborted
            );
        }
        current = err.source();
    }
    false
}

#[derive(Debug, Deserialize)]
struct ChatCompletionChunk {
    choices: Vec<ChatChoice>,
    #[serde(default)]
    usage: Option<ChatCompletionUsage>,
}

#[derive(Debug, Default, Deserialize)]
struct ChatCompletionUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    delta: ChatDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct ChatDelta {
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ChatToolCallDelta>,
}

#[derive(Debug, Deserialize)]
struct ChatToolCallDelta {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    function: Option<ChatToolCallFunctionDelta>,
}

#[derive(Debug, Default, Deserialize)]
struct ChatToolCallFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

pub(crate) fn parse_chat_completion_chunk(
    payload: &str,
) -> Result<Vec<StreamEvent>, DeepSeekError> {
    let chunk: ChatCompletionChunk = serde_json::from_str(payload)?;
    let Some(choice) = chunk.choices.into_iter().next() else {
        return Err(DeepSeekError::InvalidResponse(
            "chat completion chunk did not include any choices".to_string(),
        ));
    };

    let mut updates = Vec::new();

    if let Some(reasoning) = choice
        .delta
        .reasoning_content
        .filter(|value| !value.is_empty())
    {
        updates.push(StreamEvent::Thought(reasoning));
    }

    if let Some(content) = choice.delta.content.filter(|value| !value.is_empty()) {
        updates.push(StreamEvent::Message(content));
    }

    for tool_call in choice.delta.tool_calls {
        updates.push(StreamEvent::ToolCallDelta(ToolCallDelta::new(
            tool_call.index,
            tool_call.id,
            tool_call
                .function
                .as_ref()
                .and_then(|function| function.name.clone()),
            tool_call.function.and_then(|function| function.arguments),
        )));
    }

    if let Some(finish_reason) = choice.finish_reason {
        updates.push(StreamEvent::Finished(FinishReason::from_api(
            &finish_reason,
        )));
    }

    if let Some(usage) = chunk.usage {
        updates.push(StreamEvent::Usage(UsageData {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
        }));
    }

    Ok(updates)
}
