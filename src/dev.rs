//! Dev/smoke-test harness for local adapter development.
//!
//! Mocks the LLM client and permission requester so the full ACP adapter
//! pipeline can be exercised without hitting the `DeepSeek` API.

// stdout is the JSON-RPC wire; this harness is the only place where
// developer-facing output is intentionally printed to stdout.
#![allow(clippy::print_stdout)]

use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;

use agent_client_protocol::schema::{
    ContentBlock, ContentChunk, InitializeRequest, InitializeResponse, NewSessionRequest,
    NewSessionResponse, PermissionOptionKind, ProtocolVersion, RequestPermissionOutcome,
    RequestPermissionRequest, RequestPermissionResponse, SelectedPermissionOutcome,
    SessionNotification, SessionUpdate, StopReason,
};
use agent_client_protocol::util::MatchDispatch;
use agent_client_protocol::{AcpAgent, Client, ConnectTo, SessionMessage};
use clap::ValueEnum;
use deepseek_acp_adapter::deepseek::{
    ChatRequest, DeepSeekClient, DeepSeekError, FinishReason, LlmClient, StreamEvent,
};
use futures_util::future::BoxFuture;
use futures_util::stream::{self, BoxStream};
use tokio_util::sync::CancellationToken;

use crate::acp::{PermissionRequester, handle_new_session_request};
use crate::session::{AdapterState, PermissionDecision, SessionStore, request_tool_permission};
use crate::tools::ToolContext;

/// Mock or real backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum Backend {
    /// Connect to the real `DeepSeek` API.
    Real,
    /// Use a mock LLM client that returns canned responses.
    Mock,
}

impl Backend {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Real => "real",
            Self::Mock => "mock",
        }
    }
}

/// Build the appropriate LLM client for a backend.
///
/// # Errors
///
/// Returns an ACP internal error if the `DeepSeek` environment variables are
/// missing in real mode.
pub(crate) fn llm_client_for_backend(
    backend: Backend,
) -> Result<Arc<dyn LlmClient>, agent_client_protocol::Error> {
    match backend {
        Backend::Real => Ok(Arc::new(
            DeepSeekClient::from_env()
                .map_err(agent_client_protocol::Error::into_internal_error)?,
        )),
        Backend::Mock => Ok(Arc::new(MockLlmClient)),
    }
}

/// Build a dev agent pointing back at this adapter executable.
///
/// # Errors
///
/// Returns an ACP error if the agent config cannot be parsed.
pub(crate) fn build_dev_agent(
    executable: &Path,
    backend: Backend,
) -> Result<AcpAgent, agent_client_protocol::Error> {
    let command = executable.to_string_lossy();
    let agent_config = serde_json::json!({
        "type": "stdio",
        "name": "deepseek-acp-adapter-dev",
        "command": command,
        "args": [
            "serve",
            "--backend",
            backend.as_str(),
        ],
        "env": [],
    });

    AcpAgent::from_str(&agent_config.to_string())
}

/// Run a smoke test end-to-end: init → new session → prompt → stop reason.
///
/// # Errors
///
/// Returns an ACP error for any mis-handshakes or protocol-level failures.
pub(crate) async fn run_smoke_flow(
    transport: impl ConnectTo<Client> + 'static,
    prompt: String,
) -> Result<DevSmokeResult, agent_client_protocol::Error> {
    Client
        .builder()
        .name("deepseek-acp-adapter-dev-client")
        .on_receive_request(
            async move |request: RequestPermissionRequest, responder, _cx| {
                let outcome =
                    request
                        .options
                        .first()
                        .map_or(RequestPermissionOutcome::Cancelled, |option| {
                            RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                                option.option_id.clone(),
                            ))
                        });

                responder.respond(RequestPermissionResponse::new(outcome))
            },
            agent_client_protocol::on_receive_request!(),
        )
        .connect_with(transport, async move |cx| {
            let initialize_response = cx
                .send_request(InitializeRequest::new(ProtocolVersion::LATEST))
                .block_task()
                .await?;
            let new_session_response = cx
                .send_request(NewSessionRequest::new(std::env::current_dir().map_err(
                    |error| {
                        agent_client_protocol::Error::internal_error()
                            .data(format!("failed to get current directory: {error}"))
                    },
                )?))
                .block_task()
                .await?;
            let mut session = cx.attach_session(new_session_response.clone(), Vec::new())?;
            session.send_prompt(prompt.as_str())?;

            let mut updates = Vec::new();
            let mut response_text = String::new();
            loop {
                match session.read_update().await? {
                    SessionMessage::SessionMessage(dispatch) => {
                        MatchDispatch::new(dispatch)
                            .if_notification(async |notification: SessionNotification| {
                                updates.push(format!("{:?}", notification.update));
                                if let SessionUpdate::AgentMessageChunk(ContentChunk {
                                    content: ContentBlock::Text(text),
                                    ..
                                }) = notification.update
                                {
                                    response_text.push_str(&text.text);
                                }
                                Ok(())
                            })
                            .await
                            .otherwise_ignore()?;
                    }
                    SessionMessage::StopReason(stop_reason) => {
                        return Ok(DevSmokeResult {
                            initialize_response,
                            new_session_response,
                            updates,
                            response_text,
                            stop_reason,
                        });
                    }
                    _ => {
                        return Err(agent_client_protocol::Error::internal_error()
                            .data("unexpected session message variant"));
                    }
                }
            }
        })
        .await
}

/// Print a human-readable summary of the dev smoke test result.
pub(crate) fn print_dev_smoke_result(result: &DevSmokeResult) {
    println!("initialize response: {:?}", result.initialize_response);
    println!("new session response: {:?}", result.new_session_response);

    for update in &result.updates {
        println!("session update: {update}");
    }

    println!("stop reason: {:?}", result.stop_reason);
    println!("response text: {}", result.response_text);
}

/// Captured output from a [`run_smoke_flow`] invocation.
#[derive(Debug, Clone)]
pub(crate) struct DevSmokeResult {
    pub(crate) initialize_response: InitializeResponse,
    pub(crate) new_session_response: NewSessionResponse,
    pub(crate) updates: Vec<String>,
    pub(crate) response_text: String,
    pub(crate) stop_reason: StopReason,
}

/// Mock LLM client that echoes the user prompt as a canned response.
#[derive(Debug, Default)]
pub(crate) struct MockLlmClient;

impl LlmClient for MockLlmClient {
    fn stream_chat(
        &self,
        request: ChatRequest,
        _cancellation_token: CancellationToken,
    ) -> Result<BoxStream<'static, Result<StreamEvent, DeepSeekError>>, DeepSeekError> {
        let prompt = request.messages().last().map_or_else(
            || "mock prompt".to_owned(),
            |message| message.content().to_owned(),
        );
        let response_text = format!("mock response to: {prompt}");

        let events = vec![
            Ok(StreamEvent::Thought("mock reasoning".to_owned())),
            Ok(StreamEvent::Message(response_text)),
            Ok(StreamEvent::Finished(FinishReason::EndTurn)),
        ];

        Ok(Box::pin(stream::iter(events)))
    }
}

/// Mock permission requester that always grants `allow_always`.
#[derive(Debug, Default)]
pub(crate) struct MockPermissionRequester;

impl PermissionRequester for MockPermissionRequester {
    fn request_permission(
        &self,
        request: RequestPermissionRequest,
    ) -> BoxFuture<'_, Result<RequestPermissionResponse, agent_client_protocol::Error>> {
        let outcome = request
            .options
            .iter()
            .find(|option| option.kind == PermissionOptionKind::AllowAlways)
            .map_or(RequestPermissionOutcome::Cancelled, |option| {
                RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                    option.option_id.clone(),
                ))
            });

        Box::pin(async move { Ok(RequestPermissionResponse::new(outcome)) })
    }
}

/// Quick smoke check that the permission gating pipeline works end-to-end.
///
/// Creates an in-memory session, runs a tool call through
/// [`request_tool_permission`], and verifies the `allow_always` decision is
/// cached.
///
/// # Errors
///
/// Returns an ACP internal error if the permission gate does not produce the
/// expected outcome.
pub(crate) async fn exercise_permission_gate_smoke() -> Result<(), agent_client_protocol::Error> {
    let store = SessionStore::new(Arc::new(std::sync::Mutex::new(AdapterState::default())));
    let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
    let context = ToolContext {
        session_id: session.session_id.clone(),
        cwd: std::env::current_dir().map_err(|error| {
            agent_client_protocol::Error::internal_error()
                .data(format!("failed to get current directory: {error}"))
        })?,
        additional_directories: Vec::new(),
        client_capabilities: None,
    };
    let call = deepseek_acp_adapter::deepseek::ToolCall::new(
        "dev-permission-call",
        "write_file",
        serde_json::json!({ "path": "smoke.txt" }).to_string(),
    );
    let decision = request_tool_permission(
        &store,
        &context,
        &call,
        agent_client_protocol::schema::ToolKind::Edit,
        &MockPermissionRequester,
    )
    .await?;

    if !matches!(decision, PermissionDecision::AllowAlways) {
        return Err(agent_client_protocol::Error::internal_error()
            .data("permission gate smoke check did not allow always"));
    }

    if !store.is_always_allowed(&session.session_id, "write_file")? {
        return Err(agent_client_protocol::Error::internal_error()
            .data("permission gate smoke check did not cache allow_always"));
    }

    Ok(())
}

#[cfg(test)]
// Test assertions legitimately use indexing to access elements by position; replacing
// every `slice[i]` with `.get(i).unwrap()` adds noise without safety benefit in tests.
#[allow(clippy::indexing_slicing)]
mod tests {
    use super::{
        Backend, DevSmokeResult, MockLlmClient, MockPermissionRequester, build_dev_agent,
        exercise_permission_gate_smoke, llm_client_for_backend, print_dev_smoke_result,
        run_smoke_flow,
    };
    use crate::acp::{
        PermissionRequester, build_initialize_response, serve_with_transport_and_state_dir,
    };
    use crate::session::DEFAULT_MAX_TURN_REQUESTS;
    use crate::tools::EmptyToolRegistry;
    use agent_client_protocol::Channel;
    use agent_client_protocol::schema::{
        McpServer, PermissionOption, PermissionOptionKind, ProtocolVersion,
        RequestPermissionOutcome, RequestPermissionRequest, SessionId, StopReason, ToolCallStatus,
        ToolCallUpdate, ToolCallUpdateFields, ToolKind,
    };
    use deepseek_acp_adapter::deepseek::{
        ChatMessage, ChatRequest, FinishReason, LlmClient, StreamEvent,
    };
    use futures_util::StreamExt;
    use std::sync::{Arc, Mutex};
    use tokio_util::sync::CancellationToken;

    fn test_store() -> crate::session::SessionStore {
        crate::session::SessionStore::new(Arc::new(Mutex::new(
            crate::session::AdapterState::default(),
        )))
    }

    #[test_log::test]
    fn build_dev_agent_uses_backend_and_executable_path() -> Result<(), agent_client_protocol::Error>
    {
        let agent = build_dev_agent(
            std::path::Path::new("/tmp/deepseek-acp-adapter"),
            Backend::Mock,
        )?;

        let McpServer::Stdio(stdio) = agent.server() else {
            return Err(
                agent_client_protocol::Error::internal_error().data("expected stdio transport")
            );
        };

        assert_eq!(
            stdio.command,
            std::path::PathBuf::from("/tmp/deepseek-acp-adapter")
        );
        assert_eq!(stdio.args, vec!["serve", "--backend", "mock"]);
        assert!(stdio.env.is_empty());

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn dev_smoke_flow_runs_initialize_new_and_prompt()
    -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let llm_client: Arc<dyn LlmClient> = Arc::new(MockLlmClient);
        let tool_registry: Arc<dyn crate::tools::ToolRegistry> = Arc::new(EmptyToolRegistry);
        let (client_transport, server_transport) = Channel::duplex();
        let server_state = Arc::clone(&store.state);
        let server_client = Arc::clone(&llm_client);
        let server_tools = Arc::clone(&tool_registry);
        let state_dir =
            std::env::temp_dir().join(format!("deepseek-acp-smoke-test-{}", uuid::Uuid::new_v4()));

        let server = tokio::spawn(async move {
            serve_with_transport_and_state_dir(
                server_transport,
                server_state,
                server_client,
                server_tools,
                DEFAULT_MAX_TURN_REQUESTS,
                Some(state_dir),
            )
            .await
        });

        let result = run_smoke_flow(client_transport, "smoke prompt".to_string()).await?;

        assert_eq!(
            result.initialize_response.protocol_version,
            ProtocolVersion::LATEST
        );
        assert!(
            result
                .new_session_response
                .session_id
                .0
                .starts_with("session-")
        );
        assert_eq!(result.stop_reason, StopReason::EndTurn);
        assert_eq!(result.response_text, "mock response to: smoke prompt");
        assert!(
            result
                .updates
                .iter()
                .any(|update| update.contains("AgentMessageChunk"))
        );

        server.abort();

        Ok(())
    }

    #[test_log::test]
    fn backend_as_str_covers_real_and_mock() {
        assert_eq!(Backend::Real.as_str(), "real");
        assert_eq!(Backend::Mock.as_str(), "mock");
    }

    #[test_log::test]
    fn print_dev_smoke_result_is_callable() {
        let result = DevSmokeResult {
            initialize_response: build_initialize_response(ProtocolVersion::LATEST),
            new_session_response: agent_client_protocol::schema::NewSessionResponse::new(
                "session-1",
            ),
            updates: vec!["update-1".to_string()],
            response_text: "response".to_string(),
            stop_reason: StopReason::EndTurn,
        };

        print_dev_smoke_result(&result);
    }

    #[test_log::test(tokio::test)]
    async fn permission_gate_smoke_helper_runs() -> Result<(), agent_client_protocol::Error> {
        exercise_permission_gate_smoke().await
    }

    #[test_log::test(tokio::test)]
    async fn mock_backend_client_streams_expected_response()
    -> Result<(), agent_client_protocol::Error> {
        let client = llm_client_for_backend(Backend::Mock)?;
        let mut stream = client
            .stream_chat(
                deepseek_acp_adapter::deepseek::ChatRequest::new(vec![ChatMessage::user("hello")]),
                CancellationToken::new(),
            )
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let mut events = Vec::new();

        while let Some(item) = stream.next().await {
            events.push(item.map_err(agent_client_protocol::Error::into_internal_error)?);
        }

        assert_eq!(
            events,
            vec![
                StreamEvent::Thought("mock reasoning".to_string()),
                StreamEvent::Message("mock response to: hello".to_string()),
                StreamEvent::Finished(FinishReason::EndTurn),
            ]
        );

        Ok(())
    }

    #[test_log::test]
    fn llm_client_for_backend_real_uses_process_environment()
    -> Result<(), agent_client_protocol::Error> {
        assert!(crate::init_tracing().is_err());
        assert!(std::env::var("DEEPSEEK_API_KEY").is_ok());

        let client = deepseek_acp_adapter::deepseek::DeepSeekClient::from_env()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let config = client.config();
        assert!(!config.base_url().is_empty());
        assert!(!config.model().is_empty());

        let client = llm_client_for_backend(Backend::Real)?;
        drop(client);

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn mock_client_uses_default_prompt_when_no_messages()
    -> Result<(), agent_client_protocol::Error> {
        let client = MockLlmClient;
        let mut stream = client
            .stream_chat(ChatRequest::new(Vec::new()), CancellationToken::new())
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.map_err(agent_client_protocol::Error::into_internal_error)?);
        }
        assert_eq!(events.len(), 3);
        let StreamEvent::Message(text) = &events[1] else {
            return Err(agent_client_protocol::Error::internal_error().data("expected message"));
        };
        assert!(text.contains("mock prompt"));
        Ok(())
    }

    #[test_log::test]
    fn build_dev_agent_uses_real_backend_args() -> Result<(), agent_client_protocol::Error> {
        let agent = build_dev_agent(
            std::path::Path::new("/tmp/deepseek-acp-adapter"),
            Backend::Real,
        )?;

        let McpServer::Stdio(stdio) = agent.server() else {
            return Err(
                agent_client_protocol::Error::internal_error().data("expected stdio transport")
            );
        };
        assert_eq!(
            stdio.command,
            std::path::PathBuf::from("/tmp/deepseek-acp-adapter")
        );
        assert_eq!(stdio.args, vec!["serve", "--backend", "real"]);
        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn mock_permission_requester_grants_allow_always()
    -> Result<(), agent_client_protocol::Error> {
        let requester = MockPermissionRequester;
        let request = RequestPermissionRequest::new(
            SessionId::new("test"),
            ToolCallUpdate::new(
                "call-1",
                ToolCallUpdateFields::new()
                    .kind(ToolKind::Edit)
                    .status(ToolCallStatus::Pending)
                    .title("write_file"),
            ),
            vec![
                PermissionOption::new("allow_once", "Allow once", PermissionOptionKind::AllowOnce),
                PermissionOption::new(
                    "allow_always",
                    "Allow always",
                    PermissionOptionKind::AllowAlways,
                ),
            ],
        );

        let response = requester.request_permission(request).await?;
        let RequestPermissionOutcome::Selected(selected) = response.outcome else {
            return Err(
                agent_client_protocol::Error::internal_error().data("expected Selected outcome")
            );
        };
        assert_eq!(selected.option_id.0.as_ref(), "allow_always");
        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn mock_permission_requester_cancelled_when_no_allow_always()
    -> Result<(), agent_client_protocol::Error> {
        let requester = MockPermissionRequester;
        // Only allow_once — no allow_always option
        let request = RequestPermissionRequest::new(
            SessionId::new("test"),
            ToolCallUpdate::new(
                "call-2",
                ToolCallUpdateFields::new()
                    .kind(ToolKind::Edit)
                    .status(ToolCallStatus::Pending)
                    .title("write_file"),
            ),
            vec![PermissionOption::new(
                "allow_once",
                "Allow once",
                PermissionOptionKind::AllowOnce,
            )],
        );

        let response = requester.request_permission(request).await?;
        assert!(matches!(
            response.outcome,
            RequestPermissionOutcome::Cancelled
        ));
        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn mock_permission_requester_cancelled_with_empty_options()
    -> Result<(), agent_client_protocol::Error> {
        let requester = MockPermissionRequester;
        let request = RequestPermissionRequest::new(
            SessionId::new("test"),
            ToolCallUpdate::new(
                "call-3",
                ToolCallUpdateFields::new()
                    .kind(ToolKind::Edit)
                    .status(ToolCallStatus::Pending)
                    .title("write_file"),
            ),
            vec![],
        );

        let response = requester.request_permission(request).await?;
        assert!(matches!(
            response.outcome,
            RequestPermissionOutcome::Cancelled
        ));
        Ok(())
    }

    #[test_log::test]
    fn dev_smoke_result_clones() {
        let original = DevSmokeResult {
            initialize_response: build_initialize_response(ProtocolVersion::LATEST),
            new_session_response: agent_client_protocol::schema::NewSessionResponse::new(
                "session-clone",
            ),
            updates: vec!["clone-me".to_string()],
            response_text: "cloned text".to_string(),
            stop_reason: StopReason::EndTurn,
        };
        let cloned = original.clone();
        assert_eq!(
            cloned.initialize_response.protocol_version,
            ProtocolVersion::LATEST
        );
        assert_eq!(
            cloned.new_session_response.session_id.0.as_ref(),
            "session-clone"
        );
        assert_eq!(cloned.updates, vec!["clone-me"]);
        assert_eq!(cloned.response_text, "cloned text");
        assert_eq!(cloned.stop_reason, StopReason::EndTurn);
    }

    #[test_log::test]
    fn print_dev_smoke_result_handles_empty_updates() {
        let result = DevSmokeResult {
            initialize_response: build_initialize_response(ProtocolVersion::LATEST),
            new_session_response: agent_client_protocol::schema::NewSessionResponse::new(
                "session-empty",
            ),
            updates: vec![],
            response_text: String::new(),
            stop_reason: StopReason::EndTurn,
        };
        // Should not panic with empty updates vec
        print_dev_smoke_result(&result);
    }
}
