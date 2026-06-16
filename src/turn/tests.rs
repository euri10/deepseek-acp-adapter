#![allow(clippy::indexing_slicing)]
use super::{ModelRequestSettings, handle_prompt_request, stream_model_turn};
use crate::acp::{
    ToolCallRequester, handle_delete_session_request, handle_new_session_request,
    handle_set_session_config_option_request,
};
use crate::session::{DEFAULT_MAX_TURN_REQUESTS, ReasoningEffort, SessionStore};
use crate::test_store;
use crate::tools::{
    AdapterToolRegistry, EmptyToolRegistry, ToolContext, ToolEdit, ToolExecution, ToolRegistry,
};
use agent_client_protocol::schema::{
    CancelNotification, ContentBlock, DeleteSessionRequest, PromptRequest, SessionNotification,
    SessionUpdate, SetSessionConfigOptionRequest, StopReason, ToolCallContent, ToolCallStatus,
    ToolKind,
};
use deepseek_acp_adapter::deepseek::{
    ChatMessage, ChatRequest, DeepSeekError, FinishReason, LlmClient, StreamEvent,
    ToolCall as DeepSeekToolCall, ToolCallDelta, ToolDefinition,
};
use deepseek_acp_adapter::error::AdapterError;
use futures_util::future::BoxFuture;
use futures_util::stream::{self, BoxStream};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

struct FakeLlmClient {
    requests: Arc<Mutex<Vec<ChatRequest>>>,
    streams: Mutex<VecDeque<Vec<FakeStreamStep>>>,
}

impl FakeLlmClient {
    fn new(events: Vec<Result<StreamEvent, DeepSeekError>>) -> Self {
        Self::with_steps(events.into_iter().map(FakeStreamStep::Event).collect())
    }

    fn with_steps(steps: Vec<FakeStreamStep>) -> Self {
        Self::with_streams(vec![steps])
    }

    fn with_streams(streams: Vec<Vec<FakeStreamStep>>) -> Self {
        Self {
            requests: Arc::new(Mutex::new(Vec::new())),
            streams: Mutex::new(VecDeque::from(streams)),
        }
    }

    fn requests(&self) -> Arc<Mutex<Vec<ChatRequest>>> {
        Arc::clone(&self.requests)
    }
}

impl LlmClient for FakeLlmClient {
    fn stream_chat(
        &self,
        request: ChatRequest,
        cancellation_token: CancellationToken,
    ) -> Result<BoxStream<'static, Result<StreamEvent, DeepSeekError>>, DeepSeekError> {
        self.requests
            .lock()
            .map_err(|error| DeepSeekError::InvalidResponse(error.to_string()))?
            .push(request);
        let steps = self
            .streams
            .lock()
            .map_err(|error| DeepSeekError::InvalidResponse(error.to_string()))?
            .pop_front()
            .ok_or_else(|| {
                DeepSeekError::InvalidResponse(
                    "fake client stream was requested too many times".to_string(),
                )
            })?;

        Ok(Box::pin(stream::unfold(
            (VecDeque::from(steps), cancellation_token),
            |(mut steps, cancellation_token)| async move {
                let step = steps.pop_front()?;
                match step {
                    FakeStreamStep::Event(event) => Some((event, (steps, cancellation_token))),
                    FakeStreamStep::WaitForCancel => {
                        cancellation_token.cancelled().await;
                        None
                    }
                }
            },
        )))
    }
}

enum FakeStreamStep {
    Event(Result<StreamEvent, DeepSeekError>),
    WaitForCancel,
}

struct PendingLlmClient {
    started: Arc<Notify>,
}

impl PendingLlmClient {
    fn new(started: Arc<Notify>) -> Self {
        Self { started }
    }
}

impl LlmClient for PendingLlmClient {
    fn stream_chat(
        &self,
        _request: ChatRequest,
        _cancellation_token: CancellationToken,
    ) -> Result<BoxStream<'static, Result<StreamEvent, DeepSeekError>>, DeepSeekError> {
        self.started.notify_one();
        Ok(Box::pin(stream::pending::<
            Result<StreamEvent, DeepSeekError>,
        >()))
    }
}

struct FakeToolRegistry {
    definitions: Vec<ToolDefinition>,
    result: ToolExecution,
    calls: Arc<Mutex<Vec<DeepSeekToolCall>>>,
}

impl FakeToolRegistry {
    fn new() -> Self {
        Self {
            definitions: vec![ToolDefinition::new(
                "echo",
                "Echo a message",
                serde_json::json!({
                    "type": "object",
                    "properties": { "message": { "type": "string" } },
                }),
            )],
            result: ToolExecution::completed(
                "tool says hi",
                serde_json::json!({ "message": "tool says hi" }),
            ),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn calls(&self) -> Arc<Mutex<Vec<DeepSeekToolCall>>> {
        Arc::clone(&self.calls)
    }
}

impl ToolRegistry for FakeToolRegistry {
    fn definitions(
        &self,
        _context: &ToolContext,
        _store: &SessionStore,
    ) -> Result<Vec<ToolDefinition>, AdapterError> {
        Ok(self.definitions.clone())
    }

    fn kind(&self, _name: &str) -> ToolKind {
        ToolKind::Other
    }

    fn execute<'a>(
        &'a self,
        call: &'a DeepSeekToolCall,
        _context: &'a ToolContext,
        _store: &'a SessionStore,
        _connection: Option<&'a dyn ToolCallRequester>,
        _cancellation_token: CancellationToken,
    ) -> BoxFuture<'a, ToolExecution> {
        Box::pin(async move {
            self.calls
                .lock()
                .map(|mut calls| calls.push(call.clone()))
                .ok();
            self.result.clone()
        })
    }
}

fn assert_diff_tool_update(
    notification: &SessionNotification,
) -> Result<(), agent_client_protocol::Error> {
    let SessionUpdate::ToolCallUpdate(update) = &notification.update else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected tool call update")
        );
    };
    let Some(content) = &update.fields.content else {
        return Err(
            agent_client_protocol::Error::internal_error().data("missing tool call update content")
        );
    };
    let Some(ToolCallContent::Diff(diff)) = content.first() else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected diff tool call content")
        );
    };
    assert_eq!(diff.path, PathBuf::from("src/lib.rs"));
    assert_eq!(diff.old_text, Some("old text".to_string()));
    assert_eq!(diff.new_text, "new text");

    let Some(locations) = &update.fields.locations else {
        return Err(agent_client_protocol::Error::internal_error()
            .data("missing tool call update locations"));
    };
    let Some(location) = locations.first() else {
        return Err(
            agent_client_protocol::Error::internal_error().data("missing tool call location")
        );
    };
    assert_eq!(location.path, PathBuf::from("src/lib.rs"));
    assert_eq!(location.line, Some(7));

    Ok(())
}

#[test_log::test(tokio::test)]
async fn prompt_uses_updated_session_model_and_reasoning()
-> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let session = handle_new_session_request(
        &store,
        &agent_client_protocol::schema::NewSessionRequest::new("/tmp"),
    )?;
    handle_set_session_config_option_request(
        &store,
        &SetSessionConfigOptionRequest::new(
            session.session_id.clone(),
            crate::SESSION_CONFIG_MODEL_ID,
            "deepseek-v4-flash",
        ),
    )?;
    handle_set_session_config_option_request(
        &store,
        &SetSessionConfigOptionRequest::new(
            session.session_id.clone(),
            crate::SESSION_CONFIG_REASONING_EFFORT_ID,
            "max",
        ),
    )?;

    let client = FakeLlmClient::new(vec![Ok(StreamEvent::Finished(FinishReason::EndTurn))]);
    let requests = client.requests();

    let response = handle_prompt_request(
        &store,
        &client,
        &EmptyToolRegistry,
        None,
        PromptRequest::new(session.session_id, vec![ContentBlock::from("hi")]),
        DEFAULT_MAX_TURN_REQUESTS,
        |_| Ok(()),
    )
    .await?;

    assert_eq!(response.stop_reason, StopReason::EndTurn);
    let request_guard = requests
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    assert_eq!(request_guard.len(), 1);
    assert_eq!(request_guard[0].model(), Some("deepseek-v4-flash"));
    assert_eq!(request_guard[0].reasoning_effort(), Some("max"));

    Ok(())
}

#[test_log::test(tokio::test)]
async fn prompt_streams_updates_and_stores_history() -> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let session = handle_new_session_request(
        &store,
        &agent_client_protocol::schema::NewSessionRequest::new("/tmp"),
    )?;
    let client = FakeLlmClient::new(vec![
        Ok(StreamEvent::Thought("thinking".to_string())),
        Ok(StreamEvent::Message("hello".to_string())),
        Ok(StreamEvent::Message(" world".to_string())),
        Ok(StreamEvent::Finished(FinishReason::EndTurn)),
    ]);
    let requests = client.requests();
    let mut notifications = Vec::new();

    let response = handle_prompt_request(
        &store,
        &client,
        &EmptyToolRegistry,
        None,
        PromptRequest::new(session.session_id.clone(), vec![ContentBlock::from("hi")]),
        DEFAULT_MAX_TURN_REQUESTS,
        |notification| {
            notifications.push(notification);
            Ok(())
        },
    )
    .await?;

    assert_eq!(response.stop_reason, StopReason::EndTurn);
    assert_eq!(notifications.len(), 4);
    let SessionUpdate::SessionInfoUpdate(session_info_update) = &notifications[0].update else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected session info update")
        );
    };
    assert!(session_info_update.title.is_value());
    assert!(session_info_update.updated_at.is_value());
    let SessionUpdate::AgentThoughtChunk(thought_chunk) = &notifications[1].update else {
        return Err(agent_client_protocol::Error::internal_error().data("expected thought chunk"));
    };
    let SessionUpdate::AgentMessageChunk(first_message_chunk) = &notifications[2].update else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected first message chunk")
        );
    };
    let SessionUpdate::AgentMessageChunk(second_message_chunk) = &notifications[3].update else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected second message chunk")
        );
    };
    assert!(thought_chunk.message_id.is_some());
    assert!(first_message_chunk.message_id.is_some());
    assert_eq!(
        first_message_chunk.message_id,
        second_message_chunk.message_id
    );
    assert_ne!(thought_chunk.message_id, first_message_chunk.message_id);

    let request_guard = requests
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    assert_eq!(request_guard.len(), 1);
    assert_eq!(request_guard[0].messages()[0].content(), "hi");
    drop(request_guard);

    let state_guard = store
        .state
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let stored = state_guard
        .sessions
        .get(&session.session_id)
        .ok_or_else(|| {
            agent_client_protocol::Error::internal_error().data("missing stored session")
        })?;
    assert_eq!(stored.history.len(), 2);
    assert_eq!(stored.history[0].content(), "hi");
    assert_eq!(stored.history[1].content(), "hello world");

    Ok(())
}

#[test_log::test(tokio::test)]
async fn prompt_does_not_emit_plan_from_plain_text() -> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let session = handle_new_session_request(
        &store,
        &agent_client_protocol::schema::NewSessionRequest::new("/tmp"),
    )?;
    let client = FakeLlmClient::new(vec![Ok(StreamEvent::Finished(FinishReason::EndTurn))]);
    let mut notifications = Vec::new();

    let response = handle_prompt_request(
        &store,
        &client,
        &EmptyToolRegistry,
        None,
        PromptRequest::new(
            session.session_id,
            vec![ContentBlock::from(
                "first sentence. second sentence. third sentence.",
            )],
        ),
        DEFAULT_MAX_TURN_REQUESTS,
        |notification| {
            notifications.push(notification);
            Ok(())
        },
    )
    .await?;

    assert_eq!(response.stop_reason, StopReason::EndTurn);
    assert_eq!(notifications.len(), 1);
    assert!(
        !notifications
            .iter()
            .any(|notification| matches!(notification.update, SessionUpdate::Plan(_)))
    );

    Ok(())
}

#[test_log::test(tokio::test)]
async fn prompt_emits_explicit_plan_update_from_tool_call()
-> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let session = handle_new_session_request(
        &store,
        &agent_client_protocol::schema::NewSessionRequest::new("/tmp"),
    )?;
    let client = FakeLlmClient::with_streams(vec![
        vec![
            FakeStreamStep::Event(Ok(StreamEvent::ToolCallDelta(ToolCallDelta::new(
                0,
                Some("call-plan".to_string()),
                Some("update_plan".to_string()),
                Some(
                    serde_json::json!({
                        "entries": [
                            {
                                "content": "Inspect the failing tests",
                                "priority": "high",
                                "status": "in_progress",
                            },
                            {
                                "content": "Land the fix",
                                "priority": "medium",
                                "status": "pending",
                            },
                        ]
                    })
                    .to_string(),
                ),
            )))),
            FakeStreamStep::Event(Ok(StreamEvent::Finished(FinishReason::ToolCalls))),
        ],
        vec![
            FakeStreamStep::Event(Ok(StreamEvent::Message("plan updated".to_string()))),
            FakeStreamStep::Event(Ok(StreamEvent::Finished(FinishReason::EndTurn))),
        ],
    ]);
    let mut notifications = Vec::new();

    let response = handle_prompt_request(
        &store,
        &client,
        &AdapterToolRegistry,
        None,
        PromptRequest::new(session.session_id, vec![ContentBlock::from("make a plan")]),
        DEFAULT_MAX_TURN_REQUESTS,
        |notification| {
            notifications.push(notification);
            Ok(())
        },
    )
    .await?;

    assert_eq!(response.stop_reason, StopReason::EndTurn);
    assert!(matches!(
        notifications[0].update,
        SessionUpdate::SessionInfoUpdate(_)
    ));
    assert!(matches!(
        notifications[1].update,
        SessionUpdate::ToolCall(_)
    ));
    assert!(matches!(
        notifications[2].update,
        SessionUpdate::ToolCallUpdate(_)
    ));
    let SessionUpdate::Plan(plan) = &notifications[3].update else {
        return Err(agent_client_protocol::Error::internal_error().data("expected plan update"));
    };
    assert_eq!(plan.entries.len(), 2);
    assert_eq!(plan.entries[0].content, "Inspect the failing tests");
    assert_eq!(
        plan.entries[0].priority,
        agent_client_protocol::schema::PlanEntryPriority::High
    );
    assert_eq!(
        plan.entries[0].status,
        agent_client_protocol::schema::PlanEntryStatus::InProgress
    );
    assert_eq!(plan.entries[1].content, "Land the fix");
    assert_eq!(
        plan.entries[1].priority,
        agent_client_protocol::schema::PlanEntryPriority::Medium
    );
    assert_eq!(
        plan.entries[1].status,
        agent_client_protocol::schema::PlanEntryStatus::Pending
    );
    assert!(matches!(
        notifications[4].update,
        SessionUpdate::AgentMessageChunk(_)
    ));
    assert!(
        notifications
            .iter()
            .any(|notification| matches!(notification.update, SessionUpdate::Plan(_)))
    );

    Ok(())
}

#[test_log::test(tokio::test)]
async fn prompt_omits_unchanged_title_in_session_info_update()
-> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let session = handle_new_session_request(
        &store,
        &agent_client_protocol::schema::NewSessionRequest::new("/tmp"),
    )?;

    let first_client = FakeLlmClient::new(vec![Ok(StreamEvent::Finished(FinishReason::EndTurn))]);
    handle_prompt_request(
        &store,
        &first_client,
        &EmptyToolRegistry,
        None,
        PromptRequest::new(
            session.session_id.clone(),
            vec![ContentBlock::from("first prompt")],
        ),
        DEFAULT_MAX_TURN_REQUESTS,
        |_| Ok(()),
    )
    .await?;

    let second_client = FakeLlmClient::new(vec![Ok(StreamEvent::Finished(FinishReason::EndTurn))]);
    let mut notifications = Vec::new();
    let response = handle_prompt_request(
        &store,
        &second_client,
        &EmptyToolRegistry,
        None,
        PromptRequest::new(
            session.session_id,
            vec![ContentBlock::from("second prompt")],
        ),
        DEFAULT_MAX_TURN_REQUESTS,
        |notification| {
            notifications.push(notification);
            Ok(())
        },
    )
    .await?;

    assert_eq!(response.stop_reason, StopReason::EndTurn);
    assert_eq!(notifications.len(), 1);
    let SessionUpdate::SessionInfoUpdate(session_info_update) = &notifications[0].update else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected session info update")
        );
    };
    assert!(session_info_update.title.is_undefined());
    assert!(session_info_update.updated_at.is_value());

    Ok(())
}

#[test_log::test(tokio::test)]
async fn cancel_notification_stops_active_prompt() -> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let session = handle_new_session_request(
        &store,
        &agent_client_protocol::schema::NewSessionRequest::new("/tmp"),
    )?;
    let session_id = session.session_id.clone();
    let client = Arc::new(FakeLlmClient::with_steps(vec![
        FakeStreamStep::Event(Ok(StreamEvent::Message("partial".to_string()))),
        FakeStreamStep::WaitForCancel,
    ]));
    let (notification_tx, mut notification_rx) =
        tokio::sync::mpsc::unbounded_channel::<SessionNotification>();

    let prompt_store = store.clone();
    let prompt_session_id = session_id.clone();
    let prompt_client = Arc::clone(&client);
    let prompt_task = tokio::spawn(async move {
        handle_prompt_request(
            &prompt_store,
            prompt_client.as_ref(),
            &EmptyToolRegistry,
            None,
            PromptRequest::new(prompt_session_id, vec![ContentBlock::from("cancel me")]),
            DEFAULT_MAX_TURN_REQUESTS,
            |notification| {
                notification_tx
                    .send(notification)
                    .map_err(agent_client_protocol::Error::into_internal_error)?;
                Ok(())
            },
        )
        .await
    });

    let plan_notification = notification_rx.recv().await.ok_or_else(|| {
        agent_client_protocol::Error::internal_error().data("missing session info update")
    })?;
    let SessionUpdate::SessionInfoUpdate(session_info_update) = &plan_notification.update else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected session info update")
        );
    };
    assert!(session_info_update.title.is_value());
    assert!(session_info_update.updated_at.is_value());

    let notification = notification_rx
        .recv()
        .await
        .ok_or_else(|| agent_client_protocol::Error::internal_error().data("missing update"))?;
    let SessionUpdate::AgentMessageChunk(chunk) = notification.update else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected agent message chunk")
        );
    };
    let ContentBlock::Text(text) = chunk.content else {
        return Err(agent_client_protocol::Error::internal_error().data("expected text chunk"));
    };
    assert_eq!(text.text, "partial");

    store.cancel_active_turn(&CancelNotification::new(session_id.clone()).session_id)?;
    let response = prompt_task
        .await
        .map_err(agent_client_protocol::Error::into_internal_error)??;

    assert_eq!(response.stop_reason, StopReason::Cancelled);
    let guard = store
        .state
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let session = guard
        .sessions
        .get(&session_id)
        .ok_or_else(|| agent_client_protocol::Error::internal_error().data("missing session"))?;
    assert!(session.active_turn.is_none());
    assert!(session.history.is_empty());

    Ok(())
}

#[test_log::test(tokio::test)]
async fn delete_session_cancels_prompt_without_failing_cleanup()
-> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let session = handle_new_session_request(
        &store,
        &agent_client_protocol::schema::NewSessionRequest::new("/tmp"),
    )?;
    let session_id = session.session_id.clone();
    let started = Arc::new(Notify::new());
    let client = Arc::new(PendingLlmClient::new(Arc::clone(&started)));

    let prompt_store = store.clone();
    let prompt_session_id = session_id.clone();
    let prompt_client = Arc::clone(&client);
    let prompt_task = tokio::spawn(async move {
        handle_prompt_request(
            &prompt_store,
            prompt_client.as_ref(),
            &EmptyToolRegistry,
            None,
            PromptRequest::new(prompt_session_id, vec![ContentBlock::from("delete me")]),
            DEFAULT_MAX_TURN_REQUESTS,
            |_| Ok(()),
        )
        .await
    });

    started.notified().await;
    let delete_response =
        handle_delete_session_request(&store, &DeleteSessionRequest::new(session_id.clone()))?;
    assert_eq!(
        serde_json::to_value(&delete_response)
            .map_err(agent_client_protocol::Error::into_internal_error)?,
        serde_json::json!({})
    );

    let response = tokio::time::timeout(std::time::Duration::from_secs(1), prompt_task)
        .await
        .map_err(|error| agent_client_protocol::Error::internal_error().data(error.to_string()))?
        .map_err(agent_client_protocol::Error::into_internal_error)??;

    assert_eq!(response.stop_reason, StopReason::Cancelled);

    let guard = store
        .state
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    assert!(!guard.sessions.contains_key(&session_id));

    Ok(())
}

#[test_log::test(tokio::test)]
async fn stream_model_turn_respects_cancellation_token() -> Result<(), agent_client_protocol::Error>
{
    let started = Arc::new(Notify::new());
    let client = PendingLlmClient::new(Arc::clone(&started));
    let cancellation_token = CancellationToken::new();
    let task_token = cancellation_token.clone();
    let session_id = agent_client_protocol::schema::SessionId::new("session-cancel");
    let messages: Vec<ChatMessage> = Vec::new();
    let tool_definitions: Vec<ToolDefinition> = Vec::new();

    let turn_task = tokio::spawn(async move {
        let mut notify = |_| Ok(());
        stream_model_turn(
            &client,
            &messages,
            &tool_definitions,
            ModelRequestSettings {
                model: "deepseek-v4-pro",
                reasoning_effort: Some(ReasoningEffort::High),
            },
            task_token,
            &session_id,
            &mut notify,
        )
        .await
    });

    started.notified().await;
    cancellation_token.cancel();

    let turn = tokio::time::timeout(std::time::Duration::from_secs(1), turn_task)
        .await
        .map_err(|error| agent_client_protocol::Error::internal_error().data(error.to_string()))?
        .map_err(agent_client_protocol::Error::into_internal_error)??;

    assert_eq!(turn.stop_reason, StopReason::Cancelled);
    assert_eq!(turn.assistant_text, "");
    assert!(turn.tool_calls.is_empty());

    Ok(())
}

#[test_log::test(tokio::test)]
async fn prompt_executes_tool_calls_and_replays_results() -> Result<(), agent_client_protocol::Error>
{
    let store = test_store();
    let session = handle_new_session_request(
        &store,
        &agent_client_protocol::schema::NewSessionRequest::new("/tmp"),
    )?;
    let client = FakeLlmClient::with_streams(vec![
        vec![
            FakeStreamStep::Event(Ok(StreamEvent::ToolCallDelta(ToolCallDelta::new(
                0,
                Some("call-1".to_string()),
                Some("echo".to_string()),
                Some("{\"message\":\"".to_string()),
            )))),
            FakeStreamStep::Event(Ok(StreamEvent::ToolCallDelta(ToolCallDelta::new(
                0,
                None,
                None,
                Some("hi\"}".to_string()),
            )))),
            FakeStreamStep::Event(Ok(StreamEvent::Finished(FinishReason::ToolCalls))),
        ],
        vec![
            FakeStreamStep::Event(Ok(StreamEvent::Message("done".to_string()))),
            FakeStreamStep::Event(Ok(StreamEvent::Finished(FinishReason::EndTurn))),
        ],
    ]);
    let requests = client.requests();
    let mut registry = FakeToolRegistry::new();
    registry.result.edit = Some(ToolEdit {
        path: PathBuf::from("src/lib.rs"),
        old_text: Some("old text".to_string()),
        new_text: "new text".to_string(),
        line: 7,
    });
    let tool_calls = registry.calls();
    let mut notifications = Vec::new();

    let response = handle_prompt_request(
        &store,
        &client,
        &registry,
        None,
        PromptRequest::new(
            session.session_id.clone(),
            vec![ContentBlock::from("use tool")],
        ),
        DEFAULT_MAX_TURN_REQUESTS,
        |notification| {
            notifications.push(notification);
            Ok(())
        },
    )
    .await?;

    assert_eq!(response.stop_reason, StopReason::EndTurn);
    let SessionUpdate::SessionInfoUpdate(session_info_update) = &notifications[0].update else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected session info update")
        );
    };
    assert!(session_info_update.title.is_value());
    assert!(session_info_update.updated_at.is_value());
    assert!(matches!(
        notifications[1].update,
        SessionUpdate::ToolCall(_)
    ));
    assert!(matches!(
        notifications[2].update,
        SessionUpdate::ToolCallUpdate(_)
    ));
    assert_diff_tool_update(&notifications[2])?;
    assert!(matches!(
        notifications[3].update,
        SessionUpdate::AgentMessageChunk(_)
    ));

    let tool_call_guard = tool_calls
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    assert_eq!(tool_call_guard.len(), 1);
    assert_eq!(tool_call_guard[0].id(), "call-1");
    assert_eq!(tool_call_guard[0].name(), "echo");
    assert_eq!(tool_call_guard[0].arguments(), "{\"message\":\"hi\"}");
    drop(tool_call_guard);

    let request_guard = requests
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    assert_eq!(request_guard.len(), 2);
    assert_eq!(request_guard[0].tools().len(), 1);
    let replayed = request_guard[1].messages();
    assert_eq!(replayed.len(), 3);
    assert_eq!(replayed[0].content(), "use tool");
    assert_eq!(replayed[1].tool_calls()[0].id(), "call-1");
    assert_eq!(
        replayed[2].role(),
        deepseek_acp_adapter::deepseek::MessageRole::Tool
    );
    assert_eq!(replayed[2].tool_call_id(), Some("call-1"));
    assert_eq!(replayed[2].content(), "tool says hi");

    Ok(())
}

#[test_log::test(tokio::test)]
async fn prompt_tool_loop_stops_at_max_turn_requests() -> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let session = handle_new_session_request(
        &store,
        &agent_client_protocol::schema::NewSessionRequest::new("/tmp"),
    )?;
    let limit = DEFAULT_MAX_TURN_REQUESTS.get();
    let mut streams = (0..limit)
        .map(|index| {
            vec![
                FakeStreamStep::Event(Ok(StreamEvent::ToolCallDelta(ToolCallDelta::new(
                    0,
                    Some(format!("call-{index}")),
                    Some("echo".to_string()),
                    Some("{}".to_string()),
                )))),
                FakeStreamStep::Event(Ok(StreamEvent::Finished(FinishReason::ToolCalls))),
            ]
        })
        .collect::<Vec<_>>();
    streams.push(vec![
        FakeStreamStep::Event(Ok(StreamEvent::Message("done".to_string()))),
        FakeStreamStep::Event(Ok(StreamEvent::Finished(FinishReason::EndTurn))),
    ]);
    let client = FakeLlmClient::with_streams(streams);
    let requests = client.requests();
    let registry = FakeToolRegistry::new();

    let response = handle_prompt_request(
        &store,
        &client,
        &registry,
        None,
        PromptRequest::new(session.session_id.clone(), vec![ContentBlock::from("loop")]),
        DEFAULT_MAX_TURN_REQUESTS,
        |_| Ok(()),
    )
    .await?;

    assert_eq!(response.stop_reason, StopReason::MaxTurnRequests);
    let request_guard = requests
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    assert_eq!(request_guard.len(), limit);
    drop(request_guard);

    let guard = store
        .state
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let record = guard
        .sessions
        .get(&session.session_id)
        .ok_or_else(|| agent_client_protocol::Error::internal_error().data("missing session"))?;
    assert_eq!(record.history.len(), 1 + (limit * 2));

    Ok(())
}

#[test_log::test(tokio::test)]
async fn prompt_replays_history_on_next_turn() -> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let session = handle_new_session_request(
        &store,
        &agent_client_protocol::schema::NewSessionRequest::new("/tmp"),
    )?;
    let first_client = FakeLlmClient::new(vec![
        Ok(StreamEvent::Message("first answer".to_string())),
        Ok(StreamEvent::Finished(FinishReason::EndTurn)),
    ]);
    handle_prompt_request(
        &store,
        &first_client,
        &EmptyToolRegistry,
        None,
        PromptRequest::new(
            session.session_id.clone(),
            vec![ContentBlock::from("first")],
        ),
        DEFAULT_MAX_TURN_REQUESTS,
        |_| Ok(()),
    )
    .await?;

    let second_client =
        FakeLlmClient::new(vec![Ok(StreamEvent::Finished(FinishReason::MaxTokens))]);
    let second_requests = second_client.requests();
    let response = handle_prompt_request(
        &store,
        &second_client,
        &EmptyToolRegistry,
        None,
        PromptRequest::new(session.session_id, vec![ContentBlock::from("second")]),
        DEFAULT_MAX_TURN_REQUESTS,
        |_| Ok(()),
    )
    .await?;

    assert_eq!(response.stop_reason, StopReason::MaxTokens);
    let request_guard = second_requests
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let messages = request_guard[0].messages();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].content(), "first");
    assert_eq!(messages[1].content(), "first answer");
    assert_eq!(messages[2].content(), "second");

    Ok(())
}

#[test_log::test(tokio::test)]
async fn report_tool_call_generates_correct_notification()
-> Result<(), agent_client_protocol::Error> {
    let session_id = agent_client_protocol::schema::SessionId::new("report-test");
    let call = DeepSeekToolCall::new(
        "call-rtc",
        "write_file",
        serde_json::json!({"path": "f"}).to_string(),
    );
    let mut notifications = Vec::new();
    super::report_tool_call(
        &session_id,
        &mut |n| {
            notifications.push(n);
            Ok(())
        },
        &call,
        ToolKind::Edit,
    )?;
    assert_eq!(notifications.len(), 1);
    let SessionUpdate::ToolCall(ref tc) = notifications[0].update else {
        return Err(agent_client_protocol::Error::internal_error().data("expected ToolCall"));
    };
    assert_eq!(tc.tool_call_id.0.as_ref(), "call-rtc");
    assert_eq!(tc.status, ToolCallStatus::Pending);
    Ok(())
}

#[test_log::test(tokio::test)]
async fn report_tool_result_with_edit_generates_diff_and_location()
-> Result<(), agent_client_protocol::Error> {
    let session_id = agent_client_protocol::schema::SessionId::new("report-result");
    let call = DeepSeekToolCall::new("call-rt", "write_file", "{}");
    let exec = ToolExecution {
        content: "ok".to_string(),
        raw_output: serde_json::json!({"x": 1}),
        success: true,
        edit: Some(ToolEdit {
            path: std::path::PathBuf::from("/tmp/f.txt"),
            old_text: Some("prev".to_string()),
            new_text: "next".to_string(),
            line: 3,
        }),
    };
    let mut notifications = Vec::new();
    super::report_tool_result(
        &session_id,
        &mut |n| {
            notifications.push(n);
            Ok(())
        },
        &call,
        &exec,
    )?;
    assert_eq!(notifications.len(), 1);
    let SessionUpdate::ToolCallUpdate(ref update) = notifications[0].update else {
        return Err(agent_client_protocol::Error::internal_error().data("expected ToolCallUpdate"));
    };
    assert_eq!(update.tool_call_id.0.as_ref(), "call-rt");
    assert_eq!(update.fields.status, Some(ToolCallStatus::Completed));
    assert!(update.fields.locations.is_some());
    let Some(ref locations) = update.fields.locations else {
        return Err(agent_client_protocol::Error::internal_error().data("missing locations"));
    };
    assert_eq!(locations[0].path, std::path::PathBuf::from("/tmp/f.txt"));
    assert_eq!(locations[0].line, Some(3));
    // Diff content
    let Some(ref content) = update.fields.content else {
        return Err(agent_client_protocol::Error::internal_error().data("missing content"));
    };
    let Some(ToolCallContent::Diff(diff)) = content.first() else {
        return Err(agent_client_protocol::Error::internal_error().data("expected Diff"));
    };
    assert_eq!(diff.new_text, "next");
    assert_eq!(diff.old_text, Some("prev".to_string()));
    Ok(())
}

#[test]
fn helper_raw_input_and_finish_reason_cover_branches() {
    use agent_client_protocol::schema::StopReason;
    use deepseek_acp_adapter::deepseek::FinishReason;

    let valid_raw_input = DeepSeekToolCall::new(
        "valid-raw-input",
        "echo",
        serde_json::json!({ "a": 1 }).to_string(),
    );
    assert_eq!(
        super::tool_raw_input(&valid_raw_input),
        serde_json::json!({ "a": 1 })
    );
    let invalid_raw_input = DeepSeekToolCall::new("invalid-raw-input", "echo", "not json");
    assert_eq!(
        super::tool_raw_input(&invalid_raw_input),
        serde_json::json!("not json")
    );

    assert_eq!(
        crate::stop_reason_from_finish(&FinishReason::EndTurn),
        StopReason::EndTurn
    );
    assert_eq!(
        crate::stop_reason_from_finish(&FinishReason::ToolCalls),
        StopReason::EndTurn
    );
    assert_eq!(
        crate::stop_reason_from_finish(&FinishReason::Other("rate_limit".to_string())),
        StopReason::EndTurn
    );
    assert_eq!(
        crate::stop_reason_from_finish(&FinishReason::MaxTokens),
        StopReason::MaxTokens
    );
    assert_eq!(
        crate::stop_reason_from_finish(&FinishReason::Refusal),
        StopReason::Refusal
    );
}

#[test]
fn tool_call_title_read_file() {
    let call = DeepSeekToolCall::new("c1", "read_file", r#"{"path":"src/lib.rs"}"#);
    assert_eq!(super::tool_call_title(&call), "Read: src/lib.rs");
}

#[test]
fn tool_call_title_write_file() {
    let call = DeepSeekToolCall::new("c2", "write_file", r#"{"path":"Cargo.toml"}"#);
    assert_eq!(super::tool_call_title(&call), "Write: Cargo.toml");
}

#[test]
fn tool_call_title_edit_file() {
    let call = DeepSeekToolCall::new("c3", "edit_file", r#"{"path":"src/main.rs"}"#);
    assert_eq!(super::tool_call_title(&call), "Edit: src/main.rs");
}

#[test]
fn tool_call_title_list_dir() {
    let call = DeepSeekToolCall::new("c4", "list_dir", r#"{"path":"src/"}"#);
    assert_eq!(super::tool_call_title(&call), "List: src/");
}

#[test]
fn tool_call_title_grep() {
    let call = DeepSeekToolCall::new("c5", "grep", r#"{"pattern":"fn main"}"#);
    assert_eq!(super::tool_call_title(&call), "Search: fn main");
}

#[test]
fn tool_call_title_glob() {
    let call = DeepSeekToolCall::new("c6", "glob", r#"{"pattern":"*.rs"}"#);
    assert_eq!(super::tool_call_title(&call), "Glob: *.rs");
}

#[test]
fn tool_call_title_run_command() {
    let call = DeepSeekToolCall::new("c7", "run_command", r#"{"command":"ls -la"}"#);
    assert_eq!(super::tool_call_title(&call), "ls -la");
}

#[test]
fn tool_call_title_run_command_complex() {
    let call = DeepSeekToolCall::new(
        "c8",
        "run_command",
        r#"{"command":"pwd && sed -n '1,220p' /home/user/file.txt"}"#,
    );
    assert_eq!(
        super::tool_call_title(&call),
        "pwd && sed -n '1,220p' /home/user/file.txt"
    );
}

#[test]
fn tool_call_title_fallback_to_name_when_no_known_args() {
    let call = DeepSeekToolCall::new("c9", "custom_tool", r#"{"foo":"bar"}"#);
    assert_eq!(super::tool_call_title(&call), "custom_tool");
}

#[test]
fn tool_call_title_fallback_to_name_when_invalid_json() {
    let call = DeepSeekToolCall::new("c10", "some_tool", "not json at all");
    assert_eq!(super::tool_call_title(&call), "some_tool");
}

#[test]
fn tool_call_title_prefers_command_over_path() {
    // Args with both command and path should use command (higher priority).
    let call = DeepSeekToolCall::new(
        "c11",
        "run_command",
        r#"{"command":"cargo build","path":"src/"}"#,
    );
    assert_eq!(super::tool_call_title(&call), "cargo build");
}

#[test]
fn tool_call_title_empty_string_filtered_out() {
    let call = DeepSeekToolCall::new("c12", "run_command", r#"{"command":""}"#);
    assert_eq!(super::tool_call_title(&call), "run_command");
}
