//! Prompt-turn orchestration isolated from ACP transport wiring.

use std::num::NonZeroUsize;

use agent_client_protocol::schema::{
    ContentChunk, Diff, PromptRequest, PromptResponse, SessionId, SessionNotification,
    SessionUpdate, StopReason, ToolCall as AcpToolCall, ToolCallContent, ToolCallLocation,
    ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields, ToolKind,
};
use deepseek_acp_adapter::deepseek::{
    ChatMessage, ChatRequest, FinishReason, LlmClient, StreamEvent, ToolCall as DeepSeekToolCall,
    ToolDefinition,
};
use futures_util::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::acp::ToolCallRequester;
use crate::tools::{ToolContext, ToolExecution, ToolRegistry};
use crate::{
    PendingToolCalls, ReasoningEffort, SessionStore, plan_from_prompt, session_notification,
    stop_reason_from_finish, text_from_prompt,
};

/// Stable model settings applied to each streamed LLM request in a prompt turn.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ModelRequestSettings<'a> {
    /// Selected model identifier.
    pub(crate) model: &'a str,
    /// Reasoning effort requested from the model.
    pub(crate) reasoning_effort: ReasoningEffort,
}

struct PromptTurnEnvironment<'a> {
    store: &'a SessionStore,
    llm_client: &'a dyn LlmClient,
    tool_registry: &'a dyn ToolRegistry,
    connection: Option<&'a dyn ToolCallRequester>,
    tool_context: ToolContext,
    request: PromptRequest,
    cancellation_token: CancellationToken,
    max_turn_requests: NonZeroUsize,
}

/// Run the full prompt-turn lifecycle for a single ACP `session/prompt` request.
///
/// This keeps ACP request translation in [`crate::acp`] while moving model
/// streaming, tool-call execution, cancellation handling, plan streaming, and
/// history updates into a dedicated module.
///
/// # Errors
///
/// Returns an ACP protocol error when the prompt is invalid, session setup
/// fails, a streamed model event fails, a tool notification fails, or the
/// session store cannot be updated.
pub(crate) async fn handle_prompt_request(
    store: &SessionStore,
    llm_client: &dyn LlmClient,
    tool_registry: &dyn ToolRegistry,
    connection: Option<&dyn ToolCallRequester>,
    request: PromptRequest,
    max_turn_requests: NonZeroUsize,
    mut notify: impl FnMut(SessionNotification) -> Result<(), agent_client_protocol::Error>,
) -> Result<PromptResponse, agent_client_protocol::Error> {
    let user_text = text_from_prompt(&request.prompt)?;
    let user_message = ChatMessage::user(user_text.clone());
    let session_id = request.session_id.clone();
    let cancellation_token = CancellationToken::new();

    let turn_setup = store.begin_turn(
        &request.session_id,
        cancellation_token.clone(),
        user_message,
    )?;

    let plan = plan_from_prompt(&user_text);
    if !plan.entries.is_empty() {
        notify(session_notification(
            session_id.clone(),
            SessionUpdate::Plan(plan),
        ))?;
    }

    let result = run_prompt_turn(
        PromptTurnEnvironment {
            store,
            llm_client,
            tool_registry,
            connection,
            tool_context: turn_setup.tool_context,
            request,
            cancellation_token: cancellation_token.clone(),
            max_turn_requests,
        },
        turn_setup.messages,
        ModelRequestSettings {
            model: &turn_setup.model,
            reasoning_effort: turn_setup.reasoning_effort,
        },
        &mut notify,
    )
    .await;
    store.clear_active_turn(&session_id)?;
    result
}

async fn run_prompt_turn(
    env: PromptTurnEnvironment<'_>,
    mut messages: Vec<ChatMessage>,
    model_settings: ModelRequestSettings<'_>,
    notify: &mut impl FnMut(SessionNotification) -> Result<(), agent_client_protocol::Error>,
) -> Result<PromptResponse, agent_client_protocol::Error> {
    let tool_definitions = env
        .tool_registry
        .definitions(&env.tool_context, env.store)?;

    let mut stop_reason = StopReason::MaxTurnRequests;

    for _ in 0..env.max_turn_requests.get() {
        let turn = stream_model_turn(
            env.llm_client,
            &messages,
            &tool_definitions,
            model_settings,
            env.cancellation_token.clone(),
            &env.request.session_id,
            notify,
        )
        .await?;

        if turn.stop_reason == StopReason::Cancelled {
            stop_reason = StopReason::Cancelled;
            break;
        }

        messages.push(if turn.tool_calls.is_empty() {
            ChatMessage::assistant(turn.assistant_text.clone())
        } else {
            ChatMessage::assistant_with_tool_calls(
                turn.assistant_text.clone(),
                turn.tool_calls.clone(),
            )
        });

        if !matches!(turn.finish_reason, FinishReason::ToolCalls) || turn.tool_calls.is_empty() {
            stop_reason = turn.stop_reason;
            break;
        }

        for tool_call in &turn.tool_calls {
            let tool_kind = env.tool_registry.kind(tool_call.name());
            report_tool_call(&env.request.session_id, notify, tool_call, tool_kind)?;
            let tool_result = env
                .tool_registry
                .execute(
                    tool_call,
                    &env.tool_context,
                    env.store,
                    env.connection,
                    env.cancellation_token.clone(),
                )
                .await;
            report_tool_result(&env.request.session_id, notify, tool_call, &tool_result)?;
            messages.push(ChatMessage::tool_result(
                tool_call.id(),
                tool_result.content_for_model(),
            ));
        }
    }

    if stop_reason != StopReason::Cancelled {
        env.store.save_history(&env.request.session_id, messages)?;
    }

    Ok(PromptResponse::new(stop_reason))
}

/// Stream a single LLM turn, collecting assistant text and pending tool calls.
///
/// # Errors
///
/// Returns an ACP protocol error when the underlying LLM stream fails, when a
/// streamed tool-call delta cannot be assembled into a complete call, or when
/// a session update notification fails.
pub(crate) async fn stream_model_turn(
    llm_client: &dyn LlmClient,
    messages: &[ChatMessage],
    tool_definitions: &[ToolDefinition],
    model_settings: ModelRequestSettings<'_>,
    cancellation_token: CancellationToken,
    session_id: &SessionId,
    notify: &mut impl FnMut(SessionNotification) -> Result<(), agent_client_protocol::Error>,
) -> Result<ModelTurn, agent_client_protocol::Error> {
    let mut stream = llm_client
        .stream_chat(
            ChatRequest::new(messages.to_vec())
                .with_tools(tool_definitions.to_vec())
                .with_model(model_settings.model)
                .with_reasoning_effort(model_settings.reasoning_effort.id()),
            cancellation_token.clone(),
        )
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let mut assistant_text = String::new();
    let mut stop_reason = StopReason::EndTurn;
    let mut finish_reason = FinishReason::EndTurn;
    let mut tool_calls = PendingToolCalls::default();

    loop {
        let event = tokio::select! {
            () = cancellation_token.cancelled() => {
                stop_reason = StopReason::Cancelled;
                break;
            }
            event = stream.next() => event,
        };

        let Some(event) = event else {
            if cancellation_token.is_cancelled() {
                stop_reason = StopReason::Cancelled;
            }
            break;
        };

        match event.map_err(agent_client_protocol::Error::into_internal_error)? {
            StreamEvent::Thought(chunk) => notify(session_notification(
                session_id.clone(),
                SessionUpdate::AgentThoughtChunk(ContentChunk::new(chunk.into())),
            ))?,
            StreamEvent::Message(chunk) => {
                assistant_text.push_str(&chunk);
                notify(session_notification(
                    session_id.clone(),
                    SessionUpdate::AgentMessageChunk(ContentChunk::new(chunk.into())),
                ))?;
            }
            StreamEvent::ToolCallDelta(delta) => tool_calls.push(&delta),
            StreamEvent::Finished(reason) => {
                stop_reason = stop_reason_from_finish(&reason);
                finish_reason = reason;
            }
        }
    }

    let tool_calls = tool_calls.finish()?;

    Ok(ModelTurn {
        assistant_text,
        tool_calls,
        finish_reason,
        stop_reason,
    })
}

/// Result of a single streamed model turn.
#[derive(Debug)]
pub(crate) struct ModelTurn {
    /// Aggregated assistant text from the stream.
    pub(crate) assistant_text: String,
    /// Fully assembled tool calls emitted by the model.
    pub(crate) tool_calls: Vec<DeepSeekToolCall>,
    /// Raw finish reason reported by `DeepSeek`.
    pub(crate) finish_reason: FinishReason,
    /// ACP stop reason derived for the client.
    pub(crate) stop_reason: StopReason,
}

fn report_tool_call(
    session_id: &SessionId,
    notify: &mut impl FnMut(SessionNotification) -> Result<(), agent_client_protocol::Error>,
    call: &DeepSeekToolCall,
    kind: ToolKind,
) -> Result<(), agent_client_protocol::Error> {
    notify(session_notification(
        session_id.clone(),
        SessionUpdate::ToolCall(
            AcpToolCall::new(call.id().to_string(), call.name().to_string())
                .kind(kind)
                .status(ToolCallStatus::Pending)
                .raw_input(tool_raw_input(call)),
        ),
    ))
}

fn report_tool_result(
    session_id: &SessionId,
    notify: &mut impl FnMut(SessionNotification) -> Result<(), agent_client_protocol::Error>,
    call: &DeepSeekToolCall,
    result: &ToolExecution,
) -> Result<(), agent_client_protocol::Error> {
    let mut fields = ToolCallUpdateFields::new()
        .status(result.status())
        .content(tool_call_update_content(result))
        .raw_output(result.raw_output.clone());

    if let Some(edit) = &result.edit {
        fields = fields.locations(vec![
            ToolCallLocation::new(edit.path.clone()).line(edit.line),
        ]);
    }

    notify(session_notification(
        session_id.clone(),
        SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(call.id().to_string(), fields)),
    ))
}

fn tool_call_update_content(result: &ToolExecution) -> Vec<ToolCallContent> {
    match &result.edit {
        Some(edit) => vec![ToolCallContent::from(
            Diff::new(edit.path.clone(), edit.new_text.clone()).old_text(edit.old_text.clone()),
        )],
        None => vec![ToolCallContent::from(result.content.clone())],
    }
}

/// Parse a tool call's raw JSON arguments for ACP notifications.
///
/// Invalid JSON is preserved as a plain string to keep notifications lossless.
#[must_use]
pub(crate) fn tool_raw_input(call: &DeepSeekToolCall) -> serde_json::Value {
    serde_json::from_str(call.arguments())
        .unwrap_or_else(|_| serde_json::Value::String(call.arguments().to_string()))
}

#[cfg(test)]
mod tests {
    use super::{ModelRequestSettings, handle_prompt_request, stream_model_turn};
    use crate::acp::{
        ToolCallRequester, handle_new_session_request, handle_set_session_config_option_request,
    };
    use crate::session::{DEFAULT_MAX_TURN_REQUESTS, ReasoningEffort, SessionStore};
    use crate::test_store;
    use crate::tools::{EmptyToolRegistry, ToolContext, ToolEdit, ToolExecution, ToolRegistry};
    use agent_client_protocol::schema::{
        CancelNotification, ContentBlock, PromptRequest, SessionNotification, SessionUpdate,
        SetSessionConfigOptionRequest, StopReason, ToolCallContent, ToolCallStatus, ToolKind,
    };
    use deepseek_acp_adapter::deepseek::{
        ChatMessage, ChatRequest, DeepSeekError, FinishReason, LlmClient, StreamEvent,
        ToolCall as DeepSeekToolCall, ToolCallDelta, ToolDefinition,
    };
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
        ) -> Result<Vec<ToolDefinition>, agent_client_protocol::Error> {
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
            return Err(agent_client_protocol::Error::internal_error()
                .data("missing tool call update content"));
        };
        let Some(ToolCallContent::Diff(diff)) = content.first() else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("expected diff tool call content"));
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
    async fn prompt_streams_updates_and_stores_history() -> Result<(), agent_client_protocol::Error>
    {
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
        assert!(matches!(notifications[0].update, SessionUpdate::Plan(_)));
        assert!(matches!(
            notifications[1].update,
            SessionUpdate::AgentThoughtChunk(_)
        ));
        assert!(matches!(
            notifications[2].update,
            SessionUpdate::AgentMessageChunk(_)
        ));
        assert!(matches!(
            notifications[3].update,
            SessionUpdate::AgentMessageChunk(_)
        ));

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
            agent_client_protocol::Error::internal_error().data("missing plan update")
        })?;
        assert!(matches!(plan_notification.update, SessionUpdate::Plan(_)));

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
        let session = guard.sessions.get(&session_id).ok_or_else(|| {
            agent_client_protocol::Error::internal_error().data("missing session")
        })?;
        assert!(session.active_turn.is_none());
        assert!(session.history.is_empty());

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn stream_model_turn_respects_cancellation_token()
    -> Result<(), agent_client_protocol::Error> {
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
                    reasoning_effort: ReasoningEffort::High,
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
            .map_err(|error| {
                agent_client_protocol::Error::internal_error().data(error.to_string())
            })?
            .map_err(agent_client_protocol::Error::into_internal_error)??;

        assert_eq!(turn.stop_reason, StopReason::Cancelled);
        assert_eq!(turn.assistant_text, "");
        assert!(turn.tool_calls.is_empty());

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn prompt_executes_tool_calls_and_replays_results()
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
        assert!(matches!(notifications[0].update, SessionUpdate::Plan(_)));
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
    async fn prompt_tool_loop_stops_at_max_turn_requests()
    -> Result<(), agent_client_protocol::Error> {
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
        let record = guard.sessions.get(&session.session_id).ok_or_else(|| {
            agent_client_protocol::Error::internal_error().data("missing session")
        })?;
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
            return Err(
                agent_client_protocol::Error::internal_error().data("expected ToolCallUpdate")
            );
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
}
