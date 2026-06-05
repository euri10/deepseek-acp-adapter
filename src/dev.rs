//! Dev/smoke-test harness for local adapter development.
//!
//! Mocks the LLM client and permission requester so the full ACP adapter
//! pipeline can be exercised without hitting the `DeepSeek` API.

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
