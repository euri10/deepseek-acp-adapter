//! Command-line entrypoint for the `DeepSeek` `ACP` adapter.

#![forbid(unsafe_code)]
#![deny(
    warnings,
    missing_docs,
    clippy::all,
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::unimplemented
)]
// `#[must_use]` on every internal binary helper is noise at this stage.
#![allow(clippy::must_use_candidate)]

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::{error::Error, process::ExitCode};

use agent_client_protocol::schema::{
    AgentAuthCapabilities, AgentCapabilities, AuthenticateRequest, AuthenticateResponse,
    CancelNotification, ClientCapabilities, ContentBlock, ContentChunk, InitializeRequest,
    InitializeResponse, NewSessionRequest, NewSessionResponse, PermissionOption,
    PermissionOptionKind, PromptCapabilities, PromptRequest, PromptResponse, ProtocolVersion,
    ReadTextFileRequest, ReadTextFileResponse, RequestPermissionOutcome, RequestPermissionRequest,
    RequestPermissionResponse, SelectedPermissionOutcome, SessionId, SessionMode, SessionModeId,
    SessionModeState, SessionNotification, SessionUpdate, SetSessionModeRequest,
    SetSessionModeResponse, StopReason, ToolCall as AcpToolCall, ToolCallContent, ToolCallStatus,
    ToolCallUpdate, ToolCallUpdateFields, ToolKind,
};
use agent_client_protocol::util::MatchDispatch;
use agent_client_protocol::{AcpAgent, Agent, Client, ConnectTo, SessionMessage, Stdio};
use clap::{Parser, Subcommand, ValueEnum};
use deepseek_acp_adapter::deepseek::{
    ChatMessage, ChatRequest, DeepSeekClient, DeepSeekError, FinishReason, LlmClient, StreamEvent,
    ToolCall as DeepSeekToolCall, ToolCallDelta, ToolDefinition,
};
use futures_util::StreamExt;
use futures_util::future::BoxFuture;
use futures_util::stream::{self, BoxStream};
use globset::{Glob, GlobSetBuilder};
use grep::regex::RegexMatcher;
use grep::searcher::sinks::UTF8;
use grep::searcher::{BinaryDetection, SearcherBuilder};
use ignore::WalkBuilder;
use ignore::gitignore::GitignoreBuilder;
use serde::Deserialize;
use serde_json::Value;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

type AdapterResult<T> = Result<T, Box<dyn Error + Send + Sync + 'static>>;
const MAX_TURN_REQUESTS: usize = 25;
const PERMISSION_ALLOW_ONCE_OPTION_ID: &str = "allow_once";
const PERMISSION_ALLOW_ALWAYS_OPTION_ID: &str = "allow_always";
const PERMISSION_REJECT_ONCE_OPTION_ID: &str = "reject_once";
const PERMISSION_REJECT_ALWAYS_OPTION_ID: &str = "reject_always";
const SESSION_MODE_ASK_ID: &str = "ask";
const SESSION_MODE_ACCEPT_EDITS_ID: &str = "accept-edits";
const SESSION_MODE_YOLO_ID: &str = "yolo";

#[derive(Debug, Parser)]
#[command(
    name = "deepseek-acp-adapter",
    version,
    about = "ACP stdio adapter for DeepSeek-backed coding sessions"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, PartialEq, Eq, Subcommand)]
enum Command {
    /// Run the ACP server over standard input and output.
    Serve {
        #[arg(long, value_enum, default_value_t = Backend::Real)]
        backend: Backend,
    },
    #[command(hide = true)]
    Dev {
        #[arg(long, value_enum, default_value_t = Backend::Mock)]
        backend: Backend,
        #[arg(long, default_value = "Hello from the dev smoke test.")]
        prompt: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Backend {
    Real,
    Mock,
}

impl Backend {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Real => "real",
            Self::Mock => "mock",
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> AdapterResult<()> {
    init_tracing()?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    runtime.block_on(async {
        match Cli::parse().command {
            Command::Serve { backend } => serve(backend).await,
            Command::Dev { backend, prompt } => dev(backend, prompt).await,
        }
    })?;

    Ok(())
}

fn init_tracing() -> AdapterResult<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .try_init()?;
    Ok(())
}

async fn serve(backend: Backend) -> Result<(), agent_client_protocol::Error> {
    let llm_client = llm_client_for_backend(backend)?;
    let tool_registry = Arc::new(ReadOnlyToolRegistry);
    let state = Arc::new(Mutex::new(AdapterState::default()));
    serve_with_transport(Stdio::new(), state, llm_client, tool_registry).await
}

async fn dev(backend: Backend, prompt: String) -> Result<(), agent_client_protocol::Error> {
    let agent = build_dev_agent(
        &std::env::current_exe().map_err(|error| {
            agent_client_protocol::Error::internal_error()
                .data(format!("failed to locate current executable: {error}"))
        })?,
        backend,
    )?;
    let result = run_smoke_flow(agent, prompt).await?;
    print_dev_smoke_result(&result);
    exercise_permission_gate_smoke().await?;
    Ok(())
}

async fn exercise_permission_gate_smoke() -> Result<(), agent_client_protocol::Error> {
    let state = Arc::new(Mutex::new(AdapterState::default()));
    let session = handle_new_session_request(&state, &NewSessionRequest::new("/tmp"))?;
    let context = ToolContext {
        session_id: session.session_id.clone(),
        cwd: std::env::current_dir().map_err(|error| {
            agent_client_protocol::Error::internal_error()
                .data(format!("failed to get current directory: {error}"))
        })?,
        additional_directories: Vec::new(),
        client_capabilities: None,
    };
    let call = DeepSeekToolCall::new(
        "dev-permission-call",
        "write_file",
        serde_json::json!({ "path": "smoke.txt" }).to_string(),
    );
    let decision = request_tool_permission(
        &state,
        &context,
        &call,
        ToolKind::Edit,
        &MockPermissionRequester,
    )
    .await?;

    if !matches!(decision, PermissionDecision::AllowAlways) {
        return Err(agent_client_protocol::Error::internal_error()
            .data("permission gate smoke check did not allow always"));
    }

    let guard = state
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let stored = guard.sessions.get(&session.session_id).ok_or_else(|| {
        agent_client_protocol::Error::internal_error().data("missing permission smoke session")
    })?;
    if !stored.permission_allow_always.contains("write_file") {
        return Err(agent_client_protocol::Error::internal_error()
            .data("permission gate smoke check did not cache allow_always"));
    }

    Ok(())
}

fn llm_client_for_backend(
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

fn build_dev_agent(
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

async fn run_smoke_flow(
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

fn print_dev_smoke_result(result: &DevSmokeResult) {
    println!("initialize response: {:?}", result.initialize_response);
    println!("new session response: {:?}", result.new_session_response);

    for update in &result.updates {
        println!("session update: {update}");
    }

    println!("stop reason: {:?}", result.stop_reason);
    println!("response text: {}", result.response_text);
}

#[derive(Debug, Clone)]
struct DevSmokeResult {
    initialize_response: InitializeResponse,
    new_session_response: NewSessionResponse,
    updates: Vec<String>,
    response_text: String,
    stop_reason: StopReason,
}

#[derive(Debug, Default)]
struct MockLlmClient;

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

#[derive(Debug, Default)]
struct MockPermissionRequester;

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

async fn serve_with_transport(
    transport: impl ConnectTo<Agent> + 'static,
    state: Arc<Mutex<AdapterState>>,
    llm_client: Arc<dyn LlmClient>,
    tool_registry: Arc<dyn ToolRegistry>,
) -> Result<(), agent_client_protocol::Error> {
    let initialize_state = Arc::clone(&state);
    let new_session_state = Arc::clone(&state);
    let set_mode_state = Arc::clone(&state);
    let prompt_state = Arc::clone(&state);
    let prompt_client = Arc::clone(&llm_client);
    let prompt_tools = Arc::clone(&tool_registry);
    let cancel_state = Arc::clone(&state);

    Agent
        .builder()
        .name("deepseek-acp-adapter")
        .on_receive_request(
            async move |request: InitializeRequest, responder, _cx| {
                responder.respond(handle_initialize_request(&initialize_state, request)?)
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |_request: AuthenticateRequest, responder, _cx| {
                responder.respond(handle_authenticate_request())
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: NewSessionRequest, responder, _cx| {
                responder.respond(handle_new_session_request(&new_session_state, &request)?)
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: SetSessionModeRequest, responder, _cx| {
                responder.respond(handle_set_session_mode_request(&set_mode_state, &request)?)
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: PromptRequest, responder, cx| {
                let state = Arc::clone(&prompt_state);
                let client = Arc::clone(&prompt_client);
                let tools = Arc::clone(&prompt_tools);
                let connection = cx.clone();

                cx.spawn(async move {
                    let result = handle_prompt_request(
                        &state,
                        client.as_ref(),
                        tools.as_ref(),
                        Some(&connection as &dyn ReadTextFileRequester),
                        request,
                        |notification| connection.send_notification(notification),
                    )
                    .await;

                    responder.respond_with_result(result)?;
                    Ok(())
                })?;

                Ok(())
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_notification(
            async move |notification: CancelNotification, _cx| {
                handle_cancel_notification(&cancel_state, &notification)
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_to(transport)
        .await
}

fn handle_initialize_request(
    state: &Arc<Mutex<AdapterState>>,
    request: InitializeRequest,
) -> Result<InitializeResponse, agent_client_protocol::Error> {
    record_client_capabilities(state, request.client_capabilities)?;
    Ok(build_initialize_response(request.protocol_version))
}

fn handle_authenticate_request() -> AuthenticateResponse {
    AuthenticateResponse::new()
}

fn handle_new_session_request(
    state: &Arc<Mutex<AdapterState>>,
    request: &NewSessionRequest,
) -> Result<NewSessionResponse, agent_client_protocol::Error> {
    validate_session_paths(request)?;

    let session_id = format!("session-{}", Uuid::new_v4());
    let mut guard = state
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    guard.sessions.insert(
        session_id.clone().into(),
        SessionRecord {
            cwd: request.cwd.clone(),
            additional_directories: request.additional_directories.clone(),
            history: Vec::new(),
            active_turn: None,
            mode: PermissionPosture::Ask,
            permission_allow_always: HashSet::new(),
        },
    );

    Ok(NewSessionResponse::new(session_id).modes(default_session_modes()))
}

fn handle_set_session_mode_request(
    state: &Arc<Mutex<AdapterState>>,
    request: &SetSessionModeRequest,
) -> Result<SetSessionModeResponse, agent_client_protocol::Error> {
    let Some(mode) = PermissionPosture::from_mode_id(&request.mode_id) else {
        return Err(agent_client_protocol::Error::invalid_params()
            .data(format!("unsupported session mode: {}", request.mode_id.0)));
    };

    let mut guard = state
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let session = guard.sessions.get_mut(&request.session_id).ok_or_else(|| {
        agent_client_protocol::Error::invalid_params()
            .data(format!("unknown session id: {}", request.session_id.0))
    })?;
    session.mode = mode;

    Ok(SetSessionModeResponse::new())
}

async fn handle_prompt_request(
    state: &Arc<Mutex<AdapterState>>,
    llm_client: &dyn LlmClient,
    tool_registry: &dyn ToolRegistry,
    connection: Option<&dyn ReadTextFileRequester>,
    request: PromptRequest,
    mut notify: impl FnMut(SessionNotification) -> Result<(), agent_client_protocol::Error>,
) -> Result<PromptResponse, agent_client_protocol::Error> {
    let user_text = text_from_prompt(&request.prompt)?;
    let user_message = ChatMessage::user(user_text.clone());
    let session_id = request.session_id.clone();
    let cancellation_token = CancellationToken::new();
    let (messages, tool_context) = {
        let mut guard = state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let client_capabilities = guard.client_capabilities.clone();
        let session = guard.sessions.get_mut(&request.session_id).ok_or_else(|| {
            agent_client_protocol::Error::invalid_params()
                .data(format!("unknown session id: {}", request.session_id.0))
        })?;
        if session.active_turn.is_some() {
            return Err(
                agent_client_protocol::Error::invalid_request().data(format!(
                    "session {} already has an active turn",
                    request.session_id.0
                )),
            );
        }
        session.active_turn = Some(cancellation_token.clone());

        let mut messages = session.history.clone();
        messages.push(user_message.clone());
        (
            messages,
            ToolContext {
                session_id: session_id.clone(),
                cwd: session.cwd.clone(),
                additional_directories: session.additional_directories.clone(),
                client_capabilities,
            },
        )
    };

    let result = run_prompt_turn(
        PromptTurnEnvironment {
            state,
            llm_client,
            tool_registry,
            connection,
            tool_context,
            request,
            cancellation_token: cancellation_token.clone(),
        },
        messages,
        &mut notify,
    )
    .await;
    clear_active_turn(state, &session_id)?;
    result
}

struct PromptTurnEnvironment<'a> {
    state: &'a Arc<Mutex<AdapterState>>,
    llm_client: &'a dyn LlmClient,
    tool_registry: &'a dyn ToolRegistry,
    connection: Option<&'a dyn ReadTextFileRequester>,
    tool_context: ToolContext,
    request: PromptRequest,
    cancellation_token: CancellationToken,
}

async fn run_prompt_turn(
    env: PromptTurnEnvironment<'_>,
    mut messages: Vec<ChatMessage>,
    notify: &mut impl FnMut(SessionNotification) -> Result<(), agent_client_protocol::Error>,
) -> Result<PromptResponse, agent_client_protocol::Error> {
    let tool_definitions = env.tool_registry.definitions();
    let mut stop_reason = StopReason::EndTurn;
    let mut exhausted_turns = true;

    for _ in 0..MAX_TURN_REQUESTS {
        let turn = stream_model_turn(
            env.llm_client,
            &messages,
            &tool_definitions,
            env.cancellation_token.clone(),
            &env.request.session_id,
            notify,
        )
        .await?;

        if turn.stop_reason == StopReason::Cancelled {
            stop_reason = StopReason::Cancelled;
            exhausted_turns = false;
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
            exhausted_turns = false;
            break;
        }

        for tool_call in &turn.tool_calls {
            report_tool_call(&env.request.session_id, notify, tool_call)?;
            let tool_result = env
                .tool_registry
                .execute(tool_call, &env.tool_context, env.connection)
                .await;
            report_tool_result(&env.request.session_id, notify, tool_call, &tool_result)?;
            messages.push(ChatMessage::tool_result(
                tool_call.id(),
                tool_result.content_for_model(),
            ));
        }
    }

    if exhausted_turns {
        stop_reason = StopReason::MaxTurnRequests;
    }

    if stop_reason != StopReason::Cancelled {
        let mut guard = env
            .state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let session = guard
            .sessions
            .get_mut(&env.request.session_id)
            .ok_or_else(|| {
                agent_client_protocol::Error::invalid_params()
                    .data(format!("unknown session id: {}", env.request.session_id.0))
            })?;
        session.history = messages;
    }

    Ok(PromptResponse::new(stop_reason))
}

async fn stream_model_turn(
    llm_client: &dyn LlmClient,
    messages: &[ChatMessage],
    tool_definitions: &[ToolDefinition],
    cancellation_token: CancellationToken,
    session_id: &SessionId,
    notify: &mut impl FnMut(SessionNotification) -> Result<(), agent_client_protocol::Error>,
) -> Result<ModelTurn, agent_client_protocol::Error> {
    let mut stream = llm_client
        .stream_chat(
            ChatRequest::new(messages.to_vec()).with_tools(tool_definitions.to_vec()),
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

#[derive(Debug)]
struct ModelTurn {
    assistant_text: String,
    tool_calls: Vec<DeepSeekToolCall>,
    finish_reason: FinishReason,
    stop_reason: StopReason,
}

#[derive(Debug, Clone)]
struct ToolContext {
    session_id: SessionId,
    cwd: PathBuf,
    additional_directories: Vec<PathBuf>,
    client_capabilities: Option<ClientCapabilities>,
}

trait ReadTextFileRequester: Send + Sync {
    fn read_text_file(
        &self,
        request: ReadTextFileRequest,
    ) -> BoxFuture<'_, Result<ReadTextFileResponse, agent_client_protocol::Error>>;
}

pub(crate) trait PermissionRequester: Send + Sync {
    fn request_permission(
        &self,
        request: RequestPermissionRequest,
    ) -> BoxFuture<'_, Result<RequestPermissionResponse, agent_client_protocol::Error>>;
}

impl ReadTextFileRequester for agent_client_protocol::ConnectionTo<Agent> {
    fn read_text_file(
        &self,
        request: ReadTextFileRequest,
    ) -> BoxFuture<'_, Result<ReadTextFileResponse, agent_client_protocol::Error>> {
        Box::pin(async move { self.send_request(request).block_task().await })
    }
}

impl ReadTextFileRequester for agent_client_protocol::ConnectionTo<Client> {
    fn read_text_file(
        &self,
        request: ReadTextFileRequest,
    ) -> BoxFuture<'_, Result<ReadTextFileResponse, agent_client_protocol::Error>> {
        Box::pin(async move { self.send_request(request).block_task().await })
    }
}

impl PermissionRequester for agent_client_protocol::ConnectionTo<Agent> {
    fn request_permission(
        &self,
        request: RequestPermissionRequest,
    ) -> BoxFuture<'_, Result<RequestPermissionResponse, agent_client_protocol::Error>> {
        Box::pin(async move { self.send_request(request).block_task().await })
    }
}

impl PermissionRequester for agent_client_protocol::ConnectionTo<Client> {
    fn request_permission(
        &self,
        request: RequestPermissionRequest,
    ) -> BoxFuture<'_, Result<RequestPermissionResponse, agent_client_protocol::Error>> {
        Box::pin(async move { self.send_request(request).block_task().await })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PermissionDecision {
    AllowOnce,
    AllowAlways,
    AllowByMode,
    RejectOnce,
    RejectAlways,
    Cancelled,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum PermissionPosture {
    #[default]
    Ask,
    AcceptEdits,
    Yolo,
}

impl PermissionPosture {
    fn mode_id(self) -> SessionModeId {
        match self {
            Self::Ask => SessionModeId::new(SESSION_MODE_ASK_ID),
            Self::AcceptEdits => SessionModeId::new(SESSION_MODE_ACCEPT_EDITS_ID),
            Self::Yolo => SessionModeId::new(SESSION_MODE_YOLO_ID),
        }
    }

    const fn allows_without_prompt(self, kind: ToolKind) -> bool {
        match self {
            Self::Ask => false,
            Self::AcceptEdits => matches!(kind, ToolKind::Edit),
            Self::Yolo => !matches!(
                kind,
                ToolKind::Read | ToolKind::Search | ToolKind::Think | ToolKind::Fetch
            ),
        }
    }

    fn from_mode_id(mode_id: &SessionModeId) -> Option<Self> {
        match mode_id.0.as_ref() {
            SESSION_MODE_ASK_ID => Some(Self::Ask),
            SESSION_MODE_ACCEPT_EDITS_ID => Some(Self::AcceptEdits),
            SESSION_MODE_YOLO_ID => Some(Self::Yolo),
            _ => None,
        }
    }
}

pub(crate) async fn request_tool_permission(
    state: &Arc<Mutex<AdapterState>>,
    context: &ToolContext,
    call: &DeepSeekToolCall,
    kind: ToolKind,
    requester: &dyn PermissionRequester,
) -> Result<PermissionDecision, agent_client_protocol::Error> {
    let posture = {
        let guard = state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let session = guard.sessions.get(&context.session_id).ok_or_else(|| {
            agent_client_protocol::Error::invalid_params()
                .data(format!("unknown session id: {}", context.session_id.0))
        })?;

        if session.permission_allow_always.contains(call.name()) {
            return Ok(PermissionDecision::AllowAlways);
        }

        session.mode
    };

    if posture.allows_without_prompt(kind) {
        return Ok(PermissionDecision::AllowByMode);
    }

    let request = RequestPermissionRequest::new(
        context.session_id.clone(),
        ToolCallUpdate::new(
            call.id().to_string(),
            ToolCallUpdateFields::new()
                .kind(kind)
                .status(ToolCallStatus::Pending)
                .title(call.name().to_string())
                .raw_input(tool_raw_input(call)),
        ),
        permission_options(),
    );

    let response = requester.request_permission(request).await?;
    let decision = match response.outcome {
        RequestPermissionOutcome::Cancelled => PermissionDecision::Cancelled,
        RequestPermissionOutcome::Selected(selected) => match selected.option_id.0.as_ref() {
            PERMISSION_ALLOW_ONCE_OPTION_ID => PermissionDecision::AllowOnce,
            PERMISSION_ALLOW_ALWAYS_OPTION_ID => PermissionDecision::AllowAlways,
            PERMISSION_REJECT_ONCE_OPTION_ID => PermissionDecision::RejectOnce,
            PERMISSION_REJECT_ALWAYS_OPTION_ID => PermissionDecision::RejectAlways,
            other => {
                return Err(agent_client_protocol::Error::invalid_params()
                    .data(format!("unknown permission option selected: {other}")));
            }
        },
        _ => {
            return Err(agent_client_protocol::Error::invalid_params()
                .data("unsupported permission outcome variant"));
        }
    };

    if decision == PermissionDecision::AllowAlways {
        let mut guard = state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let session = guard.sessions.get_mut(&context.session_id).ok_or_else(|| {
            agent_client_protocol::Error::invalid_params()
                .data(format!("unknown session id: {}", context.session_id.0))
        })?;
        session
            .permission_allow_always
            .insert(call.name().to_string());
    }

    Ok(decision)
}

fn permission_options() -> Vec<PermissionOption> {
    vec![
        PermissionOption::new(
            PERMISSION_ALLOW_ONCE_OPTION_ID,
            "Allow once",
            PermissionOptionKind::AllowOnce,
        ),
        PermissionOption::new(
            PERMISSION_ALLOW_ALWAYS_OPTION_ID,
            "Allow always",
            PermissionOptionKind::AllowAlways,
        ),
        PermissionOption::new(
            PERMISSION_REJECT_ONCE_OPTION_ID,
            "Reject once",
            PermissionOptionKind::RejectOnce,
        ),
        PermissionOption::new(
            PERMISSION_REJECT_ALWAYS_OPTION_ID,
            "Reject always",
            PermissionOptionKind::RejectAlways,
        ),
    ]
}

#[derive(Debug, Deserialize)]
struct ReadFileArguments {
    path: PathBuf,
    line: Option<u32>,
    limit: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ListDirArguments {
    path: PathBuf,
}

#[derive(Debug, Deserialize)]
struct GlobArguments {
    pattern: String,
}

#[derive(Debug, Deserialize)]
struct GrepArguments {
    pattern: String,
}

const TOOL_OUTPUT_LIMIT: usize = 200;
const TOOL_OUTPUT_LIMIT_U32: u32 = 200;

/// Registry for tools the model can call during a turn.
trait ToolRegistry: Send + Sync {
    /// Return tool definitions to advertise to the model.
    fn definitions(&self) -> Vec<ToolDefinition>;

    /// Execute a complete model-requested tool call.
    fn execute<'a>(
        &'a self,
        call: &'a DeepSeekToolCall,
        context: &'a ToolContext,
        connection: Option<&'a dyn ReadTextFileRequester>,
    ) -> BoxFuture<'a, ToolExecution>;
}

#[cfg(test)]
#[derive(Debug)]
struct EmptyToolRegistry;

#[cfg(test)]
impl ToolRegistry for EmptyToolRegistry {
    fn definitions(&self) -> Vec<ToolDefinition> {
        Vec::new()
    }

    fn execute<'a>(
        &'a self,
        call: &'a DeepSeekToolCall,
        _context: &'a ToolContext,
        _connection: Option<&'a dyn ReadTextFileRequester>,
    ) -> BoxFuture<'a, ToolExecution> {
        Box::pin(async move { ToolExecution::failed(format!("unknown tool: {}", call.name())) })
    }
}

#[derive(Debug)]
struct ReadOnlyToolRegistry;

impl ToolRegistry for ReadOnlyToolRegistry {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![
            read_file_tool_definition(),
            list_dir_tool_definition(),
            glob_tool_definition(),
            grep_tool_definition(),
        ]
    }

    fn execute<'a>(
        &'a self,
        call: &'a DeepSeekToolCall,
        context: &'a ToolContext,
        connection: Option<&'a dyn ReadTextFileRequester>,
    ) -> BoxFuture<'a, ToolExecution> {
        Box::pin(async move {
            match call.name() {
                "read_file" => read_file_tool_execution(call, context, connection).await,
                "list_dir" => list_dir_tool_execution(call, context),
                "glob" => glob_tool_execution(call, context),
                "grep" => grep_tool_execution(call, context),
                _ => ToolExecution::failed(format!("unknown tool: {}", call.name())),
            }
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ToolExecution {
    content: String,
    raw_output: Value,
    success: bool,
}

impl ToolExecution {
    #[cfg(test)]
    fn completed(content: impl Into<String>, raw_output: Value) -> Self {
        Self {
            content: content.into(),
            raw_output,
            success: true,
        }
    }

    fn failed(message: impl Into<String>) -> Self {
        let message = message.into();
        Self {
            content: message.clone(),
            raw_output: serde_json::json!({ "error": message }),
            success: false,
        }
    }

    fn content_for_model(&self) -> &str {
        &self.content
    }

    fn status(&self) -> ToolCallStatus {
        if self.success {
            ToolCallStatus::Completed
        } else {
            ToolCallStatus::Failed
        }
    }
}

fn read_file_tool_definition() -> ToolDefinition {
    ToolDefinition::new(
        "read_file",
        "Read a text file, using the client's file system when available.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "line": { "type": "integer", "minimum": 1 },
                "limit": { "type": "integer", "minimum": 1 },
            },
            "required": ["path"],
            "additionalProperties": false,
        }),
    )
}

fn list_dir_tool_definition() -> ToolDefinition {
    ToolDefinition::new(
        "list_dir",
        "List entries in a directory.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
            },
            "required": ["path"],
            "additionalProperties": false,
        }),
    )
}

fn glob_tool_definition() -> ToolDefinition {
    ToolDefinition::new(
        "glob",
        "Find paths matching a glob pattern.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string" },
            },
            "required": ["pattern"],
            "additionalProperties": false,
        }),
    )
}

fn grep_tool_definition() -> ToolDefinition {
    ToolDefinition::new(
        "grep",
        "Search files for a regular expression.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string" },
            },
            "required": ["pattern"],
            "additionalProperties": false,
        }),
    )
}

async fn read_file_tool_execution(
    call: &DeepSeekToolCall,
    context: &ToolContext,
    connection: Option<&dyn ReadTextFileRequester>,
) -> ToolExecution {
    let parsed_arguments = match serde_json::from_str::<ReadFileArguments>(call.arguments()) {
        Ok(arguments) => arguments,
        Err(error) => {
            return ToolExecution::failed(format!("invalid read_file arguments: {error}"));
        }
    };

    let resolved_path = resolve_tool_path(context, &parsed_arguments.path);
    let start_line = parsed_arguments.line.unwrap_or(1);
    let requested_limit = parsed_arguments.limit.unwrap_or(TOOL_OUTPUT_LIMIT_U32);
    let limit = requested_limit.min(TOOL_OUTPUT_LIMIT_U32);

    if start_line == 0 {
        return ToolExecution::failed("read_file line must be at least 1");
    }

    if requested_limit == 0 {
        return ToolExecution::failed("read_file limit must be at least 1");
    }

    let file_result = if context
        .client_capabilities
        .as_ref()
        .is_some_and(|capabilities| capabilities.fs.read_text_file)
    {
        match connection {
            Some(connection) => {
                read_file_from_client(
                    connection,
                    &context.session_id,
                    &resolved_path,
                    start_line,
                    limit,
                )
                .await
            }
            None => Err("read_file needs a client connection for fs/read_text_file".to_owned()),
        }
    } else {
        read_file_from_local(&resolved_path, start_line, limit)
    };

    match file_result {
        Ok(file_slice) => ToolExecution {
            content: file_slice,
            raw_output: serde_json::json!({
                "path": resolved_path,
                "line": start_line,
                "limit": limit,
                "source": if context
                    .client_capabilities
                    .as_ref()
                    .is_some_and(|capabilities| capabilities.fs.read_text_file)
                {
                    "client"
                } else {
                    "local"
                },
            }),
            success: true,
        },
        Err(error) => ToolExecution::failed(error),
    }
}

fn list_dir_tool_execution(call: &DeepSeekToolCall, context: &ToolContext) -> ToolExecution {
    let parsed_arguments = match serde_json::from_str::<ListDirArguments>(call.arguments()) {
        Ok(arguments) => arguments,
        Err(error) => {
            return ToolExecution::failed(format!("invalid list_dir arguments: {error}"));
        }
    };

    let resolved_path = resolve_tool_path(context, &parsed_arguments.path);
    let entries = match collect_directory_entries(&resolved_path) {
        Ok(entries) => entries,
        Err(error) => return ToolExecution::failed(error),
    };

    let truncated = entries.len() > TOOL_OUTPUT_LIMIT;
    let entries = entries
        .into_iter()
        .take(TOOL_OUTPUT_LIMIT)
        .collect::<Vec<_>>();
    let output_text = render_tool_lines(&entries, truncated, "entries", TOOL_OUTPUT_LIMIT);

    ToolExecution {
        content: output_text,
        raw_output: serde_json::json!({
            "path": resolved_path,
            "entries": entries,
            "truncated": truncated,
        }),
        success: true,
    }
}

fn glob_tool_execution(call: &DeepSeekToolCall, context: &ToolContext) -> ToolExecution {
    let parsed_arguments = match serde_json::from_str::<GlobArguments>(call.arguments()) {
        Ok(arguments) => arguments,
        Err(error) => return ToolExecution::failed(format!("invalid glob arguments: {error}")),
    };

    let matcher = match Glob::new(&parsed_arguments.pattern) {
        Ok(glob) => {
            let mut builder = GlobSetBuilder::new();
            builder.add(glob);
            match builder.build() {
                Ok(set) => set,
                Err(error) => {
                    return ToolExecution::failed(format!("invalid glob pattern: {error}"));
                }
            }
        }
        Err(error) => return ToolExecution::failed(format!("invalid glob pattern: {error}")),
    };

    let root_gitignore = build_root_gitignore(&context.cwd);
    let mut glob_paths = Vec::new();
    let walker = WalkBuilder::new(&context.cwd)
        .hidden(false)
        .parents(true)
        .ignore(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .build();
    for entry in walker {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => return ToolExecution::failed(error.to_string()),
        };

        if !entry
            .file_type()
            .is_some_and(|file_type| file_type.is_file())
        {
            continue;
        }

        if is_hidden_path(entry.path()) {
            continue;
        }

        let path = entry.path();
        let relative_path = path.strip_prefix(&context.cwd).unwrap_or(path);
        if root_gitignore.as_ref().is_some_and(|matcher| {
            matcher
                .matched_path_or_any_parents(relative_path, false)
                .is_ignore()
        }) {
            continue;
        }
        if matcher.is_match(relative_path) || matcher.is_match(path) {
            glob_paths.push(relative_path.display().to_string());
        }
    }

    glob_paths.sort_unstable();
    let truncated = glob_paths.len() > TOOL_OUTPUT_LIMIT;
    let entries = glob_paths
        .into_iter()
        .take(TOOL_OUTPUT_LIMIT)
        .collect::<Vec<_>>();
    let output_text = render_tool_lines(&entries, truncated, "matches", TOOL_OUTPUT_LIMIT);

    ToolExecution {
        content: output_text,
        raw_output: serde_json::json!({
            "pattern": parsed_arguments.pattern,
            "matches": entries,
            "truncated": truncated,
        }),
        success: true,
    }
}

fn grep_tool_execution(call: &DeepSeekToolCall, context: &ToolContext) -> ToolExecution {
    let parsed_arguments = match serde_json::from_str::<GrepArguments>(call.arguments()) {
        Ok(arguments) => arguments,
        Err(error) => return ToolExecution::failed(format!("invalid grep arguments: {error}")),
    };

    let matcher = match RegexMatcher::new_line_matcher(&parsed_arguments.pattern) {
        Ok(matcher) => matcher,
        Err(error) => return ToolExecution::failed(format!("invalid grep regex: {error}")),
    };

    let root_gitignore = build_root_gitignore(&context.cwd);
    let (mut grep_hits, truncated) =
        match collect_grep_matches(&context.cwd, root_gitignore.as_ref(), &matcher) {
            Ok(result) => result,
            Err(error) => return ToolExecution::failed(error),
        };

    grep_hits.sort_unstable_by(|left, right| {
        left.path
            .cmp(&right.path)
            .then(left.line.cmp(&right.line))
            .then(left.text.cmp(&right.text))
    });
    let lines = grep_hits
        .into_iter()
        .take(TOOL_OUTPUT_LIMIT)
        .map(|entry| format!("{}:{}:{}", entry.path, entry.line, entry.text))
        .collect::<Vec<_>>();
    let output_text = render_tool_lines(&lines, truncated, "matches", TOOL_OUTPUT_LIMIT);

    ToolExecution {
        content: output_text,
        raw_output: serde_json::json!({
            "pattern": parsed_arguments.pattern,
            "matches": lines,
            "truncated": truncated,
        }),
        success: true,
    }
}

fn collect_grep_matches(
    root: &Path,
    root_gitignore: Option<&ignore::gitignore::Gitignore>,
    matcher: &RegexMatcher,
) -> Result<(Vec<GrepMatch>, bool), String> {
    let mut searcher = SearcherBuilder::new()
        .binary_detection(BinaryDetection::quit(b'\x00'))
        .line_number(true)
        .build();
    let mut grep_hits = Vec::<GrepMatch>::new();
    let mut truncated = false;

    let walker = WalkBuilder::new(root)
        .hidden(false)
        .parents(true)
        .ignore(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .build();
    for entry in walker {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => return Err(error.to_string()),
        };

        if !entry
            .file_type()
            .is_some_and(|file_type| file_type.is_file())
        {
            continue;
        }

        if is_hidden_path(entry.path()) {
            continue;
        }

        let path = entry.path().to_path_buf();
        let relative_path = path.strip_prefix(root).unwrap_or(&path);
        if root_gitignore.as_ref().is_some_and(|matcher| {
            matcher
                .matched_path_or_any_parents(relative_path, false)
                .is_ignore()
        }) {
            continue;
        }

        let search_result = searcher.search_path(
            matcher,
            &path,
            UTF8(|line_number, line| {
                if grep_hits.len() >= TOOL_OUTPUT_LIMIT {
                    truncated = true;
                    return Ok(false);
                }

                grep_hits.push(GrepMatch {
                    path: relative_path.display().to_string(),
                    line: line_number,
                    text: line.to_string(),
                });
                if grep_hits.len() >= TOOL_OUTPUT_LIMIT {
                    truncated = true;
                    Ok(false)
                } else {
                    Ok(true)
                }
            }),
        );
        if let Err(error) = search_result {
            return Err(format!("failed to grep {}: {error}", path.display()));
        }

        if truncated {
            break;
        }
    }

    Ok((grep_hits, truncated))
}

async fn read_file_from_client(
    connection: &dyn ReadTextFileRequester,
    session_id: &SessionId,
    path: &Path,
    line: u32,
    limit: u32,
) -> Result<String, String> {
    let response = connection
        .read_text_file(
            ReadTextFileRequest::new(session_id.clone(), path.to_path_buf())
                .line(line)
                .limit(limit),
        )
        .await
        .map_err(|error| error.to_string())?;

    Ok(response.content)
}

fn read_file_from_local(path: &Path, line: u32, limit: u32) -> Result<String, String> {
    let text = fs::read_to_string(path).map_err(|error| error.to_string())?;
    let lines: Vec<&str> = text.lines().collect();

    let start_index = usize::try_from(line.saturating_sub(1))
        .map_err(|error| format!("line number is too large: {error}"))?;
    let max_lines =
        usize::try_from(limit).map_err(|error| format!("line limit is too large: {error}"))?;

    let content = lines
        .iter()
        .skip(start_index)
        .take(max_lines)
        .copied()
        .collect::<Vec<&str>>()
        .join("\n");

    Ok(content)
}

fn resolve_tool_path(context: &ToolContext, path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }

    let candidate = context.cwd.join(path);
    if candidate.exists() {
        return candidate;
    }

    for directory in &context.additional_directories {
        let alternate = directory.join(path);
        if alternate.exists() {
            return alternate;
        }
    }

    candidate
}

fn collect_directory_entries(path: &Path) -> Result<Vec<String>, String> {
    let mut entries = fs::read_dir(path)
        .map_err(|error| format!("failed to read directory {}: {error}", path.display()))?
        .map(|entry| {
            entry.map_err(|error| {
                format!(
                    "failed to read directory entry in {}: {error}",
                    path.display()
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    entries.sort_unstable_by(|left, right| {
        left.file_name()
            .to_string_lossy()
            .cmp(&right.file_name().to_string_lossy())
    });

    Ok(entries
        .into_iter()
        .map(|entry| {
            let display = entry.file_name().to_string_lossy().into_owned();
            match entry.file_type() {
                Ok(file_type) if file_type.is_dir() => format!("{display}/"),
                _ => display,
            }
        })
        .collect())
}

fn is_hidden_path(path: &Path) -> bool {
    path.components()
        .any(|component| component.as_os_str().to_string_lossy().starts_with('.'))
}

fn build_root_gitignore(root: &Path) -> Option<ignore::gitignore::Gitignore> {
    let gitignore_path = root.join(".gitignore");
    if !gitignore_path.is_file() {
        return None;
    }

    let mut builder = GitignoreBuilder::new(root);
    if builder.add(&gitignore_path).is_some() {
        return None;
    }

    builder.build().ok()
}

fn render_tool_lines(lines: &[String], truncated: bool, label: &str, limit: usize) -> String {
    let mut output = lines.join("\n");

    if truncated {
        if !output.is_empty() {
            output.push('\n');
        }
        let _ = write!(output, "... truncated after {limit} {label}");
    }

    output
}

#[derive(Debug, Clone)]
struct GrepMatch {
    path: String,
    line: u64,
    text: String,
}

#[derive(Debug, Default)]
struct PendingToolCalls {
    calls: Vec<PendingToolCall>,
}

impl PendingToolCalls {
    fn push(&mut self, delta: &ToolCallDelta) {
        let index = delta.index();
        while self.calls.len() <= index {
            self.calls.push(PendingToolCall::default());
        }

        if let Some(call) = self.calls.get_mut(index) {
            if let Some(id) = delta.id() {
                call.id = Some(id.to_string());
            }
            if let Some(name) = delta.name() {
                call.name = Some(name.to_string());
            }
            if let Some(arguments) = delta.arguments() {
                call.arguments.push_str(arguments);
            }
        }
    }

    fn finish(self) -> Result<Vec<DeepSeekToolCall>, agent_client_protocol::Error> {
        self.calls
            .into_iter()
            .enumerate()
            .map(|(index, call)| call.finish(index))
            .collect()
    }
}

#[derive(Debug, Default)]
struct PendingToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl PendingToolCall {
    fn finish(self, index: usize) -> Result<DeepSeekToolCall, agent_client_protocol::Error> {
        let id = self.id.ok_or_else(|| {
            agent_client_protocol::Error::invalid_params()
                .data(format!("tool call delta {index} is missing an id"))
        })?;
        let name = self.name.ok_or_else(|| {
            agent_client_protocol::Error::invalid_params().data(format!(
                "tool call delta {index} is missing a function name"
            ))
        })?;

        Ok(DeepSeekToolCall::new(id, name, self.arguments))
    }
}

fn report_tool_call(
    session_id: &SessionId,
    notify: &mut impl FnMut(SessionNotification) -> Result<(), agent_client_protocol::Error>,
    call: &DeepSeekToolCall,
) -> Result<(), agent_client_protocol::Error> {
    notify(session_notification(
        session_id.clone(),
        SessionUpdate::ToolCall(
            AcpToolCall::new(call.id().to_string(), call.name().to_string())
                .kind(ToolKind::Read)
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
    notify(session_notification(
        session_id.clone(),
        SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
            call.id().to_string(),
            ToolCallUpdateFields::new()
                .status(result.status())
                .content(vec![ToolCallContent::from(result.content.clone())])
                .raw_output(result.raw_output.clone()),
        )),
    ))
}

fn tool_raw_input(call: &DeepSeekToolCall) -> Value {
    serde_json::from_str(call.arguments())
        .unwrap_or_else(|_| Value::String(call.arguments().to_string()))
}

fn build_initialize_response(protocol_version: ProtocolVersion) -> InitializeResponse {
    InitializeResponse::new(protocol_version).agent_capabilities(
        AgentCapabilities::new()
            .load_session(false)
            .prompt_capabilities(PromptCapabilities::new().embedded_context(true).image(true))
            .auth(AgentAuthCapabilities::new()),
    )
}

fn record_client_capabilities(
    state: &Arc<Mutex<AdapterState>>,
    client_capabilities: ClientCapabilities,
) -> Result<(), agent_client_protocol::Error> {
    let mut guard = state
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    guard.client_capabilities = Some(client_capabilities);
    Ok(())
}

fn handle_cancel_notification(
    state: &Arc<Mutex<AdapterState>>,
    notification: &CancelNotification,
) -> Result<(), agent_client_protocol::Error> {
    let guard = state
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    if let Some(active_turn) = guard
        .sessions
        .get(&notification.session_id)
        .and_then(|session| session.active_turn.as_ref())
    {
        active_turn.cancel();
    }

    Ok(())
}

fn clear_active_turn(
    state: &Arc<Mutex<AdapterState>>,
    session_id: &SessionId,
) -> Result<(), agent_client_protocol::Error> {
    let mut guard = state
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    if let Some(session) = guard.sessions.get_mut(session_id) {
        session.active_turn = None;
    }

    Ok(())
}

fn validate_session_paths(request: &NewSessionRequest) -> Result<(), agent_client_protocol::Error> {
    if !request.cwd.is_absolute() {
        return Err(agent_client_protocol::Error::invalid_params()
            .data("session cwd must be an absolute path"));
    }

    if request
        .additional_directories
        .iter()
        .any(|path| !path.is_absolute())
    {
        return Err(agent_client_protocol::Error::invalid_params()
            .data("additional session directories must be absolute paths"));
    }

    Ok(())
}

fn default_session_modes() -> SessionModeState {
    SessionModeState::new(
        PermissionPosture::Ask.mode_id(),
        vec![
            SessionMode::new(PermissionPosture::Ask.mode_id(), "Ask"),
            SessionMode::new(PermissionPosture::AcceptEdits.mode_id(), "Accept edits"),
            SessionMode::new(PermissionPosture::Yolo.mode_id(), "Yolo"),
        ],
    )
}

fn text_from_prompt(prompt: &[ContentBlock]) -> Result<String, agent_client_protocol::Error> {
    let mut text = String::new();

    for block in prompt {
        match block {
            ContentBlock::Text(content) => text.push_str(&content.text),
            _ => {
                return Err(agent_client_protocol::Error::invalid_params()
                    .data("only text prompt blocks are supported"));
            }
        }
    }

    if text.trim().is_empty() {
        return Err(agent_client_protocol::Error::invalid_params()
            .data("prompt must include non-empty text"));
    }

    Ok(text)
}

fn session_notification(
    session_id: agent_client_protocol::schema::SessionId,
    update: SessionUpdate,
) -> SessionNotification {
    SessionNotification::new(session_id, update)
}

fn stop_reason_from_finish(reason: &FinishReason) -> StopReason {
    match reason {
        FinishReason::EndTurn | FinishReason::ToolCalls | FinishReason::Other(_) => {
            StopReason::EndTurn
        }
        FinishReason::MaxTokens => StopReason::MaxTokens,
        FinishReason::Refusal => StopReason::Refusal,
    }
}

#[derive(Debug, Default)]
struct AdapterState {
    client_capabilities: Option<ClientCapabilities>,
    sessions: HashMap<agent_client_protocol::schema::SessionId, SessionRecord>,
}

#[derive(Debug, Default)]
struct SessionRecord {
    cwd: PathBuf,
    additional_directories: Vec<PathBuf>,
    history: Vec<ChatMessage>,
    active_turn: Option<CancellationToken>,
    mode: PermissionPosture,
    permission_allow_always: HashSet<String>,
}

#[cfg(test)]
mod tests {
    use super::{
        AdapterState, Backend, Cli, Command, DevSmokeResult, EmptyToolRegistry, MAX_TURN_REQUESTS,
        MockLlmClient, PendingToolCalls, PermissionDecision, PermissionPosture,
        PermissionRequester, ReadOnlyToolRegistry, ReadTextFileRequester, ToolContext,
        ToolExecution, ToolRegistry, build_dev_agent, build_initialize_response,
        exercise_permission_gate_smoke, glob_tool_execution, grep_tool_execution,
        handle_authenticate_request, handle_cancel_notification, handle_initialize_request,
        handle_new_session_request, handle_prompt_request, handle_set_session_mode_request,
        list_dir_tool_execution, llm_client_for_backend, print_dev_smoke_result,
        read_file_tool_execution, request_tool_permission, run_smoke_flow, serve_with_transport,
    };
    use agent_client_protocol::schema::McpServer;
    use agent_client_protocol::{Agent, Channel, Client};
    use deepseek_acp_adapter::deepseek::{
        ChatMessage, ChatRequest, DeepSeekError, FinishReason, LlmClient, StreamEvent,
        ToolCall as DeepSeekToolCall, ToolCallDelta, ToolDefinition,
    };
    use futures_util::future::BoxFuture;
    use futures_util::stream::{self, BoxStream};
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};
    use uuid::Uuid;

    use agent_client_protocol::schema::{
        CancelNotification, ClientCapabilities, ContentBlock, FileSystemCapabilities, ImageContent,
        InitializeRequest, NewSessionRequest, PermissionOptionKind, PromptRequest, ProtocolVersion,
        ReadTextFileRequest, ReadTextFileResponse, RequestPermissionOutcome,
        RequestPermissionRequest, RequestPermissionResponse, SelectedPermissionOutcome,
        SessionModeId, SessionNotification, SessionUpdate, SetSessionModeRequest, StopReason,
        ToolKind,
    };
    use clap::Parser;
    use futures_util::StreamExt;
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
        fn definitions(&self) -> Vec<ToolDefinition> {
            self.definitions.clone()
        }

        fn execute<'a>(
            &'a self,
            call: &'a DeepSeekToolCall,
            _context: &'a ToolContext,
            _connection: Option<&'a dyn ReadTextFileRequester>,
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

    struct FakePermissionRequester {
        requests: Arc<Mutex<Vec<RequestPermissionRequest>>>,
        responses: Mutex<VecDeque<RequestPermissionResponse>>,
    }

    impl FakePermissionRequester {
        fn new(responses: Vec<RequestPermissionResponse>) -> Self {
            Self {
                requests: Arc::new(Mutex::new(Vec::new())),
                responses: Mutex::new(VecDeque::from(responses)),
            }
        }

        fn requests(&self) -> Arc<Mutex<Vec<RequestPermissionRequest>>> {
            Arc::clone(&self.requests)
        }
    }

    impl PermissionRequester for FakePermissionRequester {
        fn request_permission(
            &self,
            request: RequestPermissionRequest,
        ) -> BoxFuture<'_, Result<RequestPermissionResponse, agent_client_protocol::Error>>
        {
            self.requests
                .lock()
                .map(|mut requests| requests.push(request))
                .ok();

            let response = self
                .responses
                .lock()
                .map_err(|error| {
                    agent_client_protocol::Error::internal_error().data(error.to_string())
                })
                .and_then(|mut responses| {
                    responses.pop_front().ok_or_else(|| {
                        agent_client_protocol::Error::internal_error()
                            .data("fake permission requester was exhausted")
                    })
                });

            Box::pin(async move { response })
        }
    }

    type PermissionModeFixture = (
        Arc<Mutex<AdapterState>>,
        agent_client_protocol::schema::SessionId,
        ToolContext,
        DeepSeekToolCall,
        DeepSeekToolCall,
    );

    fn permission_mode_fixture() -> Result<PermissionModeFixture, agent_client_protocol::Error> {
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let session = handle_new_session_request(&state, &NewSessionRequest::new("/tmp"))?;
        let context = ToolContext {
            session_id: session.session_id.clone(),
            cwd: std::path::PathBuf::from("/tmp"),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let edit_call = DeepSeekToolCall::new(
            "call-edit",
            "write_file",
            serde_json::json!({ "path": "file.txt" }).to_string(),
        );
        let shell_call = DeepSeekToolCall::new(
            "call-shell",
            "run_command",
            serde_json::json!({ "command": "echo hi" }).to_string(),
        );

        Ok((state, session.session_id, context, edit_call, shell_call))
    }

    #[test_log::test]
    fn parses_serve_subcommand() {
        let parsed = Cli::try_parse_from(["deepseek-acp-adapter", "serve"]);

        assert!(matches!(
            parsed,
            Ok(Cli {
                command: Command::Serve {
                    backend: Backend::Real
                }
            })
        ));
    }

    #[test_log::test]
    fn parses_dev_subcommand() {
        let parsed = Cli::try_parse_from([
            "deepseek-acp-adapter",
            "dev",
            "--backend",
            "mock",
            "--prompt",
            "smoke",
        ]);

        assert!(matches!(
            parsed,
            Ok(Cli {
                command: Command::Dev {
                    backend: Backend::Mock,
                    prompt,
                }
            }) if prompt == "smoke"
        ));
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

    #[test_log::test]
    fn build_initialize_response_advertises_expected_caps() {
        let response = build_initialize_response(ProtocolVersion::LATEST);

        assert_eq!(response.protocol_version, ProtocolVersion::LATEST);
        assert!(!response.agent_capabilities.load_session);
        assert!(response.agent_capabilities.prompt_capabilities.image);
        assert!(
            response
                .agent_capabilities
                .prompt_capabilities
                .embedded_context
        );
        assert!(response.auth_methods.is_empty());
    }

    #[test_log::test]
    fn initialize_handshake_records_client_capabilities() -> Result<(), agent_client_protocol::Error>
    {
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let request = InitializeRequest::new(ProtocolVersion::LATEST).client_capabilities(
            ClientCapabilities::new()
                .fs(FileSystemCapabilities::new()
                    .read_text_file(true)
                    .write_text_file(false))
                .terminal(true),
        );

        let response = handle_initialize_request(&state, request)?;

        assert_eq!(response.protocol_version, ProtocolVersion::LATEST);
        let guard = state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        assert_eq!(
            guard.client_capabilities.clone(),
            Some(
                ClientCapabilities::new()
                    .fs(FileSystemCapabilities::new()
                        .read_text_file(true)
                        .write_text_file(false),)
                    .terminal(true),
            )
        );

        Ok(())
    }

    #[test_log::test]
    fn authenticate_request_returns_empty_response() {
        let response = handle_authenticate_request();

        assert!(response.meta.is_none());
    }

    #[test_log::test]
    fn new_session_returns_id_and_mode() -> Result<(), agent_client_protocol::Error> {
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let response = handle_new_session_request(&state, &NewSessionRequest::new("/tmp"))?;

        assert!(response.session_id.0.starts_with("session-"));
        let modes = response
            .modes
            .ok_or_else(agent_client_protocol::Error::internal_error)?;
        assert_eq!(modes.current_mode_id.0.as_ref(), "ask");
        assert_eq!(modes.available_modes.len(), 3);
        assert!(
            modes
                .available_modes
                .iter()
                .any(|mode| mode.id.0.as_ref() == "ask")
        );
        assert!(
            modes
                .available_modes
                .iter()
                .any(|mode| mode.id.0.as_ref() == "accept-edits")
        );
        assert!(
            modes
                .available_modes
                .iter()
                .any(|mode| mode.id.0.as_ref() == "yolo")
        );

        let guard = state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        assert!(guard.sessions.contains_key(&response.session_id));

        Ok(())
    }

    #[test_log::test]
    fn set_mode_updates_session_state() -> Result<(), agent_client_protocol::Error> {
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let session = handle_new_session_request(&state, &NewSessionRequest::new("/tmp"))?;

        let response = handle_set_session_mode_request(
            &state,
            &SetSessionModeRequest::new(session.session_id.clone(), "accept-edits"),
        )?;

        assert!(response.meta.is_none());
        let guard = state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let stored = guard.sessions.get(&session.session_id).ok_or_else(|| {
            agent_client_protocol::Error::internal_error().data("missing stored session")
        })?;
        assert_eq!(stored.mode, PermissionPosture::AcceptEdits);

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn permission_request_prompts_and_caches_allow_always()
    -> Result<(), agent_client_protocol::Error> {
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let session = handle_new_session_request(&state, &NewSessionRequest::new("/tmp"))?;
        let context = ToolContext {
            session_id: session.session_id.clone(),
            cwd: std::path::PathBuf::from("/tmp"),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let call = DeepSeekToolCall::new(
            "call-1",
            "write_file",
            serde_json::json!({ "path": "file.txt" }).to_string(),
        );
        let requester = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
            RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                super::PERMISSION_ALLOW_ALWAYS_OPTION_ID,
            )),
        )]);

        let decision =
            request_tool_permission(&state, &context, &call, ToolKind::Edit, &requester).await?;

        assert_eq!(decision, PermissionDecision::AllowAlways);
        let requests = requester.requests();
        {
            let request_guard = requests
                .lock()
                .map_err(agent_client_protocol::Error::into_internal_error)?;
            assert_eq!(request_guard.len(), 1);
            let request = &request_guard[0];
            assert_eq!(request.session_id, session.session_id);
            assert_eq!(request.options.len(), 4);
            assert_eq!(
                request
                    .options
                    .iter()
                    .map(|option| option.kind)
                    .collect::<Vec<_>>(),
                vec![
                    PermissionOptionKind::AllowOnce,
                    PermissionOptionKind::AllowAlways,
                    PermissionOptionKind::RejectOnce,
                    PermissionOptionKind::RejectAlways,
                ]
            );
            assert_eq!(
                request.tool_call.fields.raw_input,
                Some(serde_json::json!({ "path": "file.txt" }))
            );
        }

        let second_requester = FakePermissionRequester::new(Vec::new());
        let second_decision =
            request_tool_permission(&state, &context, &call, ToolKind::Edit, &second_requester)
                .await?;

        assert_eq!(second_decision, PermissionDecision::AllowAlways);
        let second_requests = second_requester.requests();
        let second_guard = second_requests
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        assert!(second_guard.is_empty());

        let guard = state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let stored = guard.sessions.get(&session.session_id).ok_or_else(|| {
            agent_client_protocol::Error::internal_error().data("missing stored session")
        })?;
        assert!(stored.permission_allow_always.contains("write_file"));

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn permission_request_rejects_without_caching() -> Result<(), agent_client_protocol::Error>
    {
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let session = handle_new_session_request(&state, &NewSessionRequest::new("/tmp"))?;
        let context = ToolContext {
            session_id: session.session_id.clone(),
            cwd: std::path::PathBuf::from("/tmp"),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let call = DeepSeekToolCall::new(
            "call-2",
            "run_command",
            serde_json::json!({ "command": "echo hi" }).to_string(),
        );
        let requester = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
            RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                super::PERMISSION_REJECT_ONCE_OPTION_ID,
            )),
        )]);

        let decision =
            request_tool_permission(&state, &context, &call, ToolKind::Execute, &requester).await?;

        assert_eq!(decision, PermissionDecision::RejectOnce);
        let requests = requester.requests();
        let request_guard = requests
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        assert_eq!(request_guard.len(), 1);
        drop(request_guard);

        let guard = state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let stored = guard.sessions.get(&session.session_id).ok_or_else(|| {
            agent_client_protocol::Error::internal_error().data("missing stored session")
        })?;
        assert!(!stored.permission_allow_always.contains("run_command"));

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn permission_posture_ask_prompts_all_mutations()
    -> Result<(), agent_client_protocol::Error> {
        let (state, _session_id, context, edit_call, shell_call) = permission_mode_fixture()?;
        let requester = FakePermissionRequester::new(vec![
            RequestPermissionResponse::new(RequestPermissionOutcome::Selected(
                SelectedPermissionOutcome::new(super::PERMISSION_ALLOW_ONCE_OPTION_ID),
            )),
            RequestPermissionResponse::new(RequestPermissionOutcome::Selected(
                SelectedPermissionOutcome::new(super::PERMISSION_ALLOW_ONCE_OPTION_ID),
            )),
        ]);

        assert_eq!(
            request_tool_permission(&state, &context, &edit_call, ToolKind::Edit, &requester)
                .await?,
            PermissionDecision::AllowOnce
        );
        assert_eq!(
            request_tool_permission(&state, &context, &shell_call, ToolKind::Execute, &requester)
                .await?,
            PermissionDecision::AllowOnce
        );
        assert_eq!(
            requester
                .requests()
                .lock()
                .map_err(agent_client_protocol::Error::into_internal_error)?
                .len(),
            2
        );

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn permission_posture_accept_edits_skips_edit_prompts()
    -> Result<(), agent_client_protocol::Error> {
        let (state, session_id, context, edit_call, shell_call) = permission_mode_fixture()?;
        handle_set_session_mode_request(
            &state,
            &SetSessionModeRequest::new(session_id.clone(), "accept-edits"),
        )?;
        let requester = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
            RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                super::PERMISSION_ALLOW_ONCE_OPTION_ID,
            )),
        )]);

        assert_eq!(
            request_tool_permission(&state, &context, &edit_call, ToolKind::Edit, &requester)
                .await?,
            PermissionDecision::AllowByMode
        );
        assert_eq!(
            request_tool_permission(&state, &context, &shell_call, ToolKind::Execute, &requester)
                .await?,
            PermissionDecision::AllowOnce
        );
        assert_eq!(
            requester
                .requests()
                .lock()
                .map_err(agent_client_protocol::Error::into_internal_error)?
                .len(),
            1
        );

        let guard = state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let stored = guard.sessions.get(&session_id).ok_or_else(|| {
            agent_client_protocol::Error::internal_error().data("missing stored session")
        })?;
        assert_eq!(stored.mode, PermissionPosture::AcceptEdits);

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn permission_posture_yolo_auto_allows_all_mutations()
    -> Result<(), agent_client_protocol::Error> {
        let (state, session_id, context, edit_call, shell_call) = permission_mode_fixture()?;
        handle_set_session_mode_request(&state, &SetSessionModeRequest::new(session_id, "yolo"))?;
        let requester = FakePermissionRequester::new(Vec::new());

        assert_eq!(
            request_tool_permission(&state, &context, &edit_call, ToolKind::Edit, &requester)
                .await?,
            PermissionDecision::AllowByMode
        );
        assert_eq!(
            request_tool_permission(&state, &context, &shell_call, ToolKind::Execute, &requester)
                .await?,
            PermissionDecision::AllowByMode
        );
        assert!(
            requester
                .requests()
                .lock()
                .map_err(agent_client_protocol::Error::into_internal_error)?
                .is_empty()
        );

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn prompt_streams_updates_and_stores_history() -> Result<(), agent_client_protocol::Error>
    {
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let session = handle_new_session_request(&state, &NewSessionRequest::new("/tmp"))?;
        let client = FakeLlmClient::new(vec![
            Ok(StreamEvent::Thought("thinking".to_string())),
            Ok(StreamEvent::Message("hello".to_string())),
            Ok(StreamEvent::Message(" world".to_string())),
            Ok(StreamEvent::Finished(FinishReason::EndTurn)),
        ]);
        let requests = client.requests();
        let mut notifications = Vec::new();

        let response = handle_prompt_request(
            &state,
            &client,
            &EmptyToolRegistry,
            None,
            PromptRequest::new(session.session_id.clone(), vec![ContentBlock::from("hi")]),
            |notification| {
                notifications.push(notification);
                Ok(())
            },
        )
        .await?;

        assert_eq!(response.stop_reason, StopReason::EndTurn);
        assert_eq!(notifications.len(), 3);
        assert!(matches!(
            notifications[0].update,
            SessionUpdate::AgentThoughtChunk(_)
        ));
        assert!(matches!(
            notifications[1].update,
            SessionUpdate::AgentMessageChunk(_)
        ));
        assert!(matches!(
            notifications[2].update,
            SessionUpdate::AgentMessageChunk(_)
        ));

        let request_guard = requests
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        assert_eq!(request_guard.len(), 1);
        assert_eq!(request_guard[0].messages()[0].content(), "hi");
        drop(request_guard);

        let state_guard = state
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
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let session = handle_new_session_request(&state, &NewSessionRequest::new("/tmp"))?;
        let session_id = session.session_id.clone();
        let client = Arc::new(FakeLlmClient::with_steps(vec![
            FakeStreamStep::Event(Ok(StreamEvent::Message("partial".to_string()))),
            FakeStreamStep::WaitForCancel,
        ]));
        let (notification_tx, mut notification_rx) =
            tokio::sync::mpsc::unbounded_channel::<SessionNotification>();

        let prompt_state = Arc::clone(&state);
        let prompt_session_id = session_id.clone();
        let prompt_client = Arc::clone(&client);
        let prompt_task = tokio::spawn(async move {
            handle_prompt_request(
                &prompt_state,
                prompt_client.as_ref(),
                &EmptyToolRegistry,
                None,
                PromptRequest::new(prompt_session_id, vec![ContentBlock::from("cancel me")]),
                |notification| {
                    notification_tx
                        .send(notification)
                        .map_err(agent_client_protocol::Error::into_internal_error)?;
                    Ok(())
                },
            )
            .await
        });

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

        handle_cancel_notification(&state, &CancelNotification::new(session_id.clone()))?;
        let response = prompt_task
            .await
            .map_err(agent_client_protocol::Error::into_internal_error)??;

        assert_eq!(response.stop_reason, StopReason::Cancelled);
        let guard = state
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
            super::stream_model_turn(
                &client,
                &messages,
                &tool_definitions,
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
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let session = handle_new_session_request(&state, &NewSessionRequest::new("/tmp"))?;
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
        let registry = FakeToolRegistry::new();
        let tool_calls = registry.calls();
        let mut notifications = Vec::new();

        let response = handle_prompt_request(
            &state,
            &client,
            &registry,
            None,
            PromptRequest::new(
                session.session_id.clone(),
                vec![ContentBlock::from("use tool")],
            ),
            |notification| {
                notifications.push(notification);
                Ok(())
            },
        )
        .await?;

        assert_eq!(response.stop_reason, StopReason::EndTurn);
        assert!(matches!(
            notifications[0].update,
            SessionUpdate::ToolCall(_)
        ));
        assert!(matches!(
            notifications[1].update,
            SessionUpdate::ToolCallUpdate(_)
        ));
        assert!(matches!(
            notifications[2].update,
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
    async fn prompt_tool_loop_stops_at_turn_cap() -> Result<(), agent_client_protocol::Error> {
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let session = handle_new_session_request(&state, &NewSessionRequest::new("/tmp"))?;
        let streams = (0..MAX_TURN_REQUESTS)
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
            .collect();
        let client = FakeLlmClient::with_streams(streams);
        let requests = client.requests();
        let registry = FakeToolRegistry::new();

        let response = handle_prompt_request(
            &state,
            &client,
            &registry,
            None,
            PromptRequest::new(session.session_id, vec![ContentBlock::from("loop")]),
            |_| Ok(()),
        )
        .await?;

        assert_eq!(response.stop_reason, StopReason::MaxTurnRequests);
        let request_guard = requests
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        assert_eq!(request_guard.len(), MAX_TURN_REQUESTS);

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn prompt_replays_history_on_next_turn() -> Result<(), agent_client_protocol::Error> {
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let session = handle_new_session_request(&state, &NewSessionRequest::new("/tmp"))?;
        let first_client = FakeLlmClient::new(vec![
            Ok(StreamEvent::Message("first answer".to_string())),
            Ok(StreamEvent::Finished(FinishReason::EndTurn)),
        ]);
        handle_prompt_request(
            &state,
            &first_client,
            &EmptyToolRegistry,
            None,
            PromptRequest::new(
                session.session_id.clone(),
                vec![ContentBlock::from("first")],
            ),
            |_| Ok(()),
        )
        .await?;

        let second_client =
            FakeLlmClient::new(vec![Ok(StreamEvent::Finished(FinishReason::MaxTokens))]);
        let second_requests = second_client.requests();
        let response = handle_prompt_request(
            &state,
            &second_client,
            &EmptyToolRegistry,
            None,
            PromptRequest::new(session.session_id, vec![ContentBlock::from("second")]),
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
    async fn read_file_tool_uses_local_fallback() -> Result<(), agent_client_protocol::Error> {
        let temp_root =
            std::env::temp_dir().join(format!("deepseek-acp-adapter-local-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let file_path = temp_root.join("sample.txt");
        std::fs::write(&file_path, "alpha\nbeta\ngamma\ndelta\n")
            .map_err(agent_client_protocol::Error::into_internal_error)?;

        let context = ToolContext {
            session_id: agent_client_protocol::schema::SessionId::new("session-local"),
            cwd: temp_root.clone(),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let call = DeepSeekToolCall::new(
            "call-local",
            "read_file",
            serde_json::json!({
                "path": "sample.txt",
                "line": 2,
                "limit": 2,
            })
            .to_string(),
        );

        let result = read_file_tool_execution(&call, &context, None).await;

        assert!(result.success);
        assert_eq!(result.content, "beta\ngamma");
        assert_eq!(result.raw_output["source"], "local");
        assert_eq!(result.raw_output["line"], 2);
        assert_eq!(result.raw_output["limit"], 2);
        assert_eq!(result.raw_output["path"], serde_json::json!(file_path));

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn read_file_tool_defaults_line_and_limit() -> Result<(), agent_client_protocol::Error> {
        let temp_root =
            std::env::temp_dir().join(format!("deepseek-acp-adapter-defaults-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let file_path = temp_root.join("sample.txt");
        std::fs::write(&file_path, "alpha\nbeta\ngamma\n")
            .map_err(agent_client_protocol::Error::into_internal_error)?;

        let context = ToolContext {
            session_id: agent_client_protocol::schema::SessionId::new("session-defaults"),
            cwd: temp_root.clone(),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let call = DeepSeekToolCall::new(
            "call-defaults",
            "read_file",
            serde_json::json!({
                "path": "sample.txt",
            })
            .to_string(),
        );

        let result = read_file_tool_execution(&call, &context, None).await;

        assert!(result.success);
        assert_eq!(result.content, "alpha\nbeta\ngamma");
        assert_eq!(result.raw_output["source"], "local");
        assert_eq!(result.raw_output["line"], 1);
        assert_eq!(result.raw_output["limit"], 200);
        assert_eq!(result.raw_output["path"], serde_json::json!(file_path));

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn read_file_tool_routes_to_client_fs() -> Result<(), agent_client_protocol::Error> {
        let temp_root =
            std::env::temp_dir().join(format!("deepseek-acp-adapter-client-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;

        let observed_request = Arc::new(Mutex::new(None::<ReadTextFileRequest>));
        let observed_request_for_server = Arc::clone(&observed_request);
        let (client_transport, server_transport) = Channel::duplex();

        let server = tokio::spawn(async move {
            Agent
                .builder()
                .on_receive_request(
                    async move |request: ReadTextFileRequest, responder, _cx| {
                        let mut guard = observed_request_for_server
                            .lock()
                            .map_err(agent_client_protocol::Error::into_internal_error)?;
                        *guard = Some(request.clone());
                        responder.respond(ReadTextFileResponse::new(
                            "buffered line two\nbuffered line three",
                        ))
                    },
                    agent_client_protocol::on_receive_request!(),
                )
                .connect_to(server_transport)
                .await
        });

        let context = ToolContext {
            session_id: agent_client_protocol::schema::SessionId::new("session-client"),
            cwd: temp_root.clone(),
            additional_directories: Vec::new(),
            client_capabilities: Some(
                ClientCapabilities::new().fs(FileSystemCapabilities::new()
                    .read_text_file(true)
                    .write_text_file(false)),
            ),
        };
        let call = DeepSeekToolCall::new(
            "call-client",
            "read_file",
            serde_json::json!({
                "path": "buffer.txt",
                "line": 2,
                "limit": 2,
            })
            .to_string(),
        );

        let result = Client
            .builder()
            .connect_with(client_transport, move |connection| async move {
                let result = read_file_tool_execution(
                    &call,
                    &context,
                    Some(&connection as &dyn ReadTextFileRequester),
                )
                .await;
                Ok(result)
            })
            .await?;

        assert!(result.success);
        assert_eq!(result.content, "buffered line two\nbuffered line three");
        assert_eq!(result.raw_output["source"], "client");
        assert_eq!(result.raw_output["line"], 2);
        assert_eq!(result.raw_output["limit"], 2);
        assert_eq!(
            result.raw_output["path"],
            serde_json::json!(temp_root.join("buffer.txt"))
        );

        let request_guard = observed_request
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let request = request_guard.as_ref().ok_or_else(|| {
            agent_client_protocol::Error::internal_error().data("missing read_text_file request")
        })?;
        assert_eq!(request.session_id.0.as_ref(), "session-client");
        assert_eq!(request.path, temp_root.join("buffer.txt"));
        assert_eq!(request.line, Some(2));
        assert_eq!(request.limit, Some(2));
        drop(request_guard);

        server.abort();

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn local_tools_list_dir_and_glob() -> Result<(), agent_client_protocol::Error> {
        let temp_root = std::env::temp_dir().join(format!(
            "deepseek-acp-adapter-local-tools-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(temp_root.join("src"))
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        std::fs::create_dir_all(temp_root.join("ignored"))
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        std::fs::write(temp_root.join("README.md"), "read me")
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        std::fs::write(temp_root.join("src/lib.rs"), "pub fn lib() {}")
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        std::fs::write(temp_root.join("src/main.rs"), "fn main() {}")
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        std::fs::write(temp_root.join("ignored/secret.rs"), "fn secret() {}")
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        std::fs::write(temp_root.join(".gitignore"), "ignored/\n")
            .map_err(agent_client_protocol::Error::into_internal_error)?;

        let context = ToolContext {
            session_id: agent_client_protocol::schema::SessionId::new("session-local-tools"),
            cwd: temp_root.clone(),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let registry = ReadOnlyToolRegistry;

        let list_result = registry
            .execute(
                &DeepSeekToolCall::new(
                    "call-list",
                    "list_dir",
                    serde_json::json!({ "path": "." }).to_string(),
                ),
                &context,
                None,
            )
            .await;
        assert!(list_result.content.contains("README.md"));
        assert!(list_result.content.contains("src/"));
        assert_eq!(
            list_result.raw_output["truncated"],
            serde_json::json!(false)
        );

        let glob_result = registry
            .execute(
                &DeepSeekToolCall::new(
                    "call-glob",
                    "glob",
                    serde_json::json!({ "pattern": "**/*.rs" }).to_string(),
                ),
                &context,
                None,
            )
            .await;
        assert!(glob_result.content.contains("src/lib.rs"));
        assert!(glob_result.content.contains("src/main.rs"));
        assert!(!glob_result.content.contains("ignored/secret.rs"));
        assert_eq!(
            glob_result.raw_output["truncated"],
            serde_json::json!(false)
        );

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn local_tools_grep_respects_gitignore_and_truncates()
    -> Result<(), agent_client_protocol::Error> {
        let temp_root =
            std::env::temp_dir().join(format!("deepseek-acp-adapter-grep-{}", Uuid::new_v4()));
        std::fs::create_dir_all(temp_root.join("ignored"))
            .map_err(agent_client_protocol::Error::into_internal_error)?;

        let visible = (1..=201).map(|_| "needle").collect::<Vec<_>>().join("\n");
        std::fs::write(temp_root.join("visible.rs"), format!("{visible}\n"))
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        std::fs::write(temp_root.join("ignored/secret.rs"), "needle\n")
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        std::fs::write(temp_root.join(".gitignore"), "ignored/\n")
            .map_err(agent_client_protocol::Error::into_internal_error)?;

        let context = ToolContext {
            session_id: agent_client_protocol::schema::SessionId::new("session-grep"),
            cwd: temp_root.clone(),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let registry = ReadOnlyToolRegistry;

        let result = registry
            .execute(
                &DeepSeekToolCall::new(
                    "call-grep",
                    "grep",
                    serde_json::json!({ "pattern": "needle" }).to_string(),
                ),
                &context,
                None,
            )
            .await;

        assert!(result.content.contains("visible.rs:1:needle"));
        assert!(result.content.contains("visible.rs:200:needle"));
        assert!(result.content.contains("... truncated after 200 matches"));
        assert!(!result.content.contains("ignored/secret.rs"));
        assert_eq!(result.raw_output["truncated"], serde_json::json!(true));

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn dev_smoke_flow_runs_initialize_new_and_prompt()
    -> Result<(), agent_client_protocol::Error> {
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let llm_client: Arc<dyn LlmClient> = Arc::new(MockLlmClient);
        let tool_registry: Arc<dyn ToolRegistry> = Arc::new(EmptyToolRegistry);
        let (client_transport, server_transport) = Channel::duplex();
        let server_state = Arc::clone(&state);
        let server_client = Arc::clone(&llm_client);
        let server_tools = Arc::clone(&tool_registry);

        let server = tokio::spawn(async move {
            serve_with_transport(server_transport, server_state, server_client, server_tools).await
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
    fn permission_posture_helpers_cover_all_branches() {
        assert_eq!(PermissionPosture::Ask.mode_id().0.as_ref(), "ask");
        assert_eq!(
            PermissionPosture::AcceptEdits.mode_id().0.as_ref(),
            "accept-edits"
        );
        assert_eq!(PermissionPosture::Yolo.mode_id().0.as_ref(), "yolo");
        assert_eq!(
            PermissionPosture::from_mode_id(&SessionModeId::new("ask")),
            Some(PermissionPosture::Ask)
        );
        assert_eq!(
            PermissionPosture::from_mode_id(&SessionModeId::new("accept-edits")),
            Some(PermissionPosture::AcceptEdits)
        );
        assert_eq!(
            PermissionPosture::from_mode_id(&SessionModeId::new("yolo")),
            Some(PermissionPosture::Yolo)
        );
        assert_eq!(
            PermissionPosture::from_mode_id(&SessionModeId::new("bogus")),
            None
        );
        assert!(!PermissionPosture::Ask.allows_without_prompt(ToolKind::Edit));
        assert!(PermissionPosture::AcceptEdits.allows_without_prompt(ToolKind::Edit));
        assert!(!PermissionPosture::AcceptEdits.allows_without_prompt(ToolKind::Execute));
        assert!(PermissionPosture::Yolo.allows_without_prompt(ToolKind::Execute));
        assert!(!PermissionPosture::Yolo.allows_without_prompt(ToolKind::Read));
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
                ChatRequest::new(vec![deepseek_acp_adapter::deepseek::ChatMessage::user(
                    "hello",
                )]),
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
        assert!(super::init_tracing().is_err());
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
    async fn request_permission_handles_unknown_session_and_cancelled()
    -> Result<(), agent_client_protocol::Error> {
        let missing_state = Arc::new(Mutex::new(AdapterState::default()));
        let missing_context = ToolContext {
            session_id: agent_client_protocol::schema::SessionId::new("missing-session"),
            cwd: std::path::PathBuf::from("/tmp"),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let missing_call = DeepSeekToolCall::new(
            "missing-call",
            "write_file",
            serde_json::json!({ "path": "file.txt" }).to_string(),
        );
        let missing_requester = FakePermissionRequester::new(Vec::new());

        let Err(error) = request_tool_permission(
            &missing_state,
            &missing_context,
            &missing_call,
            ToolKind::Edit,
            &missing_requester,
        )
        .await
        else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("expected missing session id to fail"));
        };
        assert!(error.to_string().contains("unknown session id"));

        let state = Arc::new(Mutex::new(AdapterState::default()));
        let session = handle_new_session_request(&state, &NewSessionRequest::new("/tmp"))?;
        let context = ToolContext {
            session_id: session.session_id.clone(),
            cwd: std::path::PathBuf::from("/tmp"),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let call = DeepSeekToolCall::new(
            "cancelled-call",
            "run_command",
            serde_json::json!({ "command": "echo hi" }).to_string(),
        );
        let requester = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
            RequestPermissionOutcome::Cancelled,
        )]);

        assert_eq!(
            request_tool_permission(&state, &context, &call, ToolKind::Execute, &requester).await?,
            PermissionDecision::Cancelled
        );

        Ok(())
    }

    #[test_log::test]
    fn helper_validation_and_prompt_error_branches() -> Result<(), agent_client_protocol::Error> {
        assert_eq!(
            super::text_from_prompt(&[ContentBlock::from("hello"), ContentBlock::from(" world")])?,
            "hello world"
        );

        let image_prompt = vec![ContentBlock::Image(ImageContent::new(
            "aGVsbG8=",
            "image/png",
        ))];
        let Err(error) = super::text_from_prompt(&image_prompt) else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("expected image prompt to fail"));
        };
        assert!(
            error
                .to_string()
                .contains("only text prompt blocks are supported")
        );

        let Err(error) = super::text_from_prompt(&[]) else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("expected empty prompt to fail"));
        };
        assert!(
            error
                .to_string()
                .contains("prompt must include non-empty text")
        );

        let relative_request = NewSessionRequest::new("relative");
        let Err(error) = super::validate_session_paths(&relative_request) else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("expected relative cwd to fail"));
        };
        assert!(
            error
                .to_string()
                .contains("session cwd must be an absolute path")
        );

        let relative_additional = NewSessionRequest::new("/tmp")
            .additional_directories(vec![std::path::PathBuf::from("relative")]);
        let Err(error) = super::validate_session_paths(&relative_additional) else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("expected relative additional directory to fail"));
        };
        assert!(
            error
                .to_string()
                .contains("additional session directories must be absolute paths")
        );

        Ok(())
    }

    #[test_log::test]
    fn helper_raw_input_and_finish_reason_cover_branches() {
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
            super::stop_reason_from_finish(&FinishReason::EndTurn),
            StopReason::EndTurn
        );
        assert_eq!(
            super::stop_reason_from_finish(&FinishReason::ToolCalls),
            StopReason::EndTurn
        );
        assert_eq!(
            super::stop_reason_from_finish(&FinishReason::Other("rate_limit".to_string())),
            StopReason::EndTurn
        );
        assert_eq!(
            super::stop_reason_from_finish(&FinishReason::MaxTokens),
            StopReason::MaxTokens
        );
        assert_eq!(
            super::stop_reason_from_finish(&FinishReason::Refusal),
            StopReason::Refusal
        );
    }

    #[test_log::test]
    fn helper_path_functions_cover_error_branches() -> Result<(), agent_client_protocol::Error> {
        let temp_root =
            std::env::temp_dir().join(format!("deepseek-acp-adapter-path-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        assert!(super::build_root_gitignore(&temp_root).is_none());
        assert!(super::is_hidden_path(std::path::Path::new(".gitignore")));
        assert!(!super::is_hidden_path(std::path::Path::new("src/lib.rs")));

        let alternate_directory = temp_root.join("alternate");
        std::fs::create_dir_all(&alternate_directory)
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        std::fs::write(alternate_directory.join("found.txt"), "found")
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let context = ToolContext {
            session_id: agent_client_protocol::schema::SessionId::new("session-paths"),
            cwd: temp_root.clone(),
            additional_directories: vec![alternate_directory.clone()],
            client_capabilities: None,
        };
        assert_eq!(
            super::resolve_tool_path(&context, std::path::Path::new("found.txt")),
            alternate_directory.join("found.txt")
        );
        assert_eq!(
            super::resolve_tool_path(&context, std::path::Path::new("/abs/path")),
            std::path::PathBuf::from("/abs/path")
        );
        assert_eq!(
            super::resolve_tool_path(&context, std::path::Path::new("missing.txt")),
            temp_root.join("missing.txt")
        );
        assert!(super::collect_directory_entries(&temp_root.join("missing-dir")).is_err());

        Ok(())
    }

    #[test_log::test]
    fn pending_tool_calls_require_complete_metadata() -> Result<(), agent_client_protocol::Error> {
        let mut missing_id = PendingToolCalls::default();
        missing_id.push(&ToolCallDelta::new(
            1,
            None,
            Some("echo".to_string()),
            Some("{}".to_string()),
        ));
        let Err(error) = missing_id.finish() else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("expected missing tool call id to fail"));
        };
        assert!(error.to_string().contains("missing an id"));

        let mut missing_name = PendingToolCalls::default();
        missing_name.push(&ToolCallDelta::new(
            0,
            Some("call-1".to_string()),
            None,
            Some("{}".to_string()),
        ));
        let Err(error) = missing_name.finish() else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("expected missing tool call name to fail"));
        };
        assert!(error.to_string().contains("missing a function name"));

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn prompt_request_rejects_active_turn() -> Result<(), agent_client_protocol::Error> {
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let session = handle_new_session_request(&state, &NewSessionRequest::new("/tmp"))?;
        {
            let mut guard = state
                .lock()
                .map_err(agent_client_protocol::Error::into_internal_error)?;
            let record = guard.sessions.get_mut(&session.session_id).ok_or_else(|| {
                agent_client_protocol::Error::internal_error().data("missing stored session")
            })?;
            record.active_turn = Some(CancellationToken::new());
        }

        let Err(error) = handle_prompt_request(
            &state,
            &MockLlmClient,
            &EmptyToolRegistry,
            None,
            PromptRequest::new(session.session_id, vec![ContentBlock::from("hi")]),
            |_| Ok(()),
        )
        .await
        else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("expected active turn to reject prompt"));
        };
        assert!(error.to_string().contains("already has an active turn"));

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn read_file_tool_error_paths_report_failures() -> Result<(), agent_client_protocol::Error>
    {
        let temp_root =
            std::env::temp_dir().join(format!("deepseek-acp-adapter-tools-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        std::fs::write(temp_root.join("visible.txt"), "one\ntwo\nthree")
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let context = ToolContext {
            session_id: agent_client_protocol::schema::SessionId::new("session-tools"),
            cwd: temp_root.clone(),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };

        let invalid_read = read_file_tool_execution(
            &DeepSeekToolCall::new("invalid-read", "read_file", "not json"),
            &context,
            None,
        )
        .await;
        assert!(!invalid_read.success);
        assert!(invalid_read.content.contains("invalid read_file arguments"));

        let zero_line_read = read_file_tool_execution(
            &DeepSeekToolCall::new(
                "zero-line",
                "read_file",
                serde_json::json!({
                    "path": "visible.txt",
                    "line": 0,
                })
                .to_string(),
            ),
            &context,
            None,
        )
        .await;
        assert!(!zero_line_read.success);
        assert!(zero_line_read.content.contains("line must be at least 1"));

        let zero_limit_read = read_file_tool_execution(
            &DeepSeekToolCall::new(
                "zero-limit",
                "read_file",
                serde_json::json!({
                    "path": "visible.txt",
                    "limit": 0,
                })
                .to_string(),
            ),
            &context,
            None,
        )
        .await;
        assert!(!zero_limit_read.success);
        assert!(zero_limit_read.content.contains("limit must be at least 1"));

        let missing_file_read = read_file_tool_execution(
            &DeepSeekToolCall::new(
                "missing-file",
                "read_file",
                serde_json::json!({ "path": "missing.txt" }).to_string(),
            ),
            &context,
            None,
        )
        .await;
        assert!(!missing_file_read.success);
        assert!(!missing_file_read.content.is_empty());

        let client_context = ToolContext {
            session_id: agent_client_protocol::schema::SessionId::new("session-client-tools"),
            cwd: temp_root,
            additional_directories: Vec::new(),
            client_capabilities: Some(
                ClientCapabilities::new().fs(FileSystemCapabilities::new().read_text_file(true)),
            ),
        };
        let missing_connection_read = read_file_tool_execution(
            &DeepSeekToolCall::new(
                "missing-connection",
                "read_file",
                serde_json::json!({ "path": "visible.txt" }).to_string(),
            ),
            &client_context,
            None,
        )
        .await;
        assert!(!missing_connection_read.success);
        assert!(
            missing_connection_read
                .content
                .contains("needs a client connection")
        );

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn local_tool_error_paths_report_failures() -> Result<(), agent_client_protocol::Error> {
        let temp_root =
            std::env::temp_dir().join(format!("deepseek-acp-adapter-tools-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let context = ToolContext {
            session_id: agent_client_protocol::schema::SessionId::new("session-tools"),
            cwd: temp_root.clone(),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };

        let invalid_list = list_dir_tool_execution(
            &DeepSeekToolCall::new("invalid-list", "list_dir", "not json"),
            &context,
        );
        assert!(!invalid_list.success);
        assert!(invalid_list.content.contains("invalid list_dir arguments"));

        let missing_directory = list_dir_tool_execution(
            &DeepSeekToolCall::new(
                "missing-directory",
                "list_dir",
                serde_json::json!({ "path": "missing-dir" }).to_string(),
            ),
            &context,
        );
        assert!(!missing_directory.success);
        assert!(!missing_directory.content.is_empty());

        let invalid_glob = glob_tool_execution(
            &DeepSeekToolCall::new("invalid-glob", "glob", "not json"),
            &context,
        );
        assert!(!invalid_glob.success);
        assert!(invalid_glob.content.contains("invalid glob arguments"));

        let invalid_glob_pattern = glob_tool_execution(
            &DeepSeekToolCall::new(
                "invalid-glob-pattern",
                "glob",
                serde_json::json!({ "pattern": "[" }).to_string(),
            ),
            &context,
        );
        assert!(!invalid_glob_pattern.success);
        assert!(
            invalid_glob_pattern
                .content
                .contains("invalid glob pattern")
        );

        let invalid_grep = grep_tool_execution(
            &DeepSeekToolCall::new("invalid-grep", "grep", "not json"),
            &context,
        );
        assert!(!invalid_grep.success);
        assert!(invalid_grep.content.contains("invalid grep arguments"));

        let invalid_grep_regex = grep_tool_execution(
            &DeepSeekToolCall::new(
                "invalid-grep-regex",
                "grep",
                serde_json::json!({ "pattern": "(" }).to_string(),
            ),
            &context,
        );
        assert!(!invalid_grep_regex.success);
        assert!(invalid_grep_regex.content.contains("invalid grep regex"));

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn serve_with_transport_handles_authenticate_and_mode_updates()
    -> Result<(), agent_client_protocol::Error> {
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let llm_client: Arc<dyn LlmClient> = Arc::new(MockLlmClient);
        let tool_registry: Arc<dyn ToolRegistry> = Arc::new(EmptyToolRegistry);
        let (client_transport, server_transport) = Channel::duplex();
        let server_state = Arc::clone(&state);
        let server_client = Arc::clone(&llm_client);
        let server_tools = Arc::clone(&tool_registry);

        let server = tokio::spawn(async move {
            serve_with_transport(server_transport, server_state, server_client, server_tools).await
        });

        Client
            .builder()
            .connect_with(client_transport, async move |cx| {
                let initialize_response = cx
                    .send_request(InitializeRequest::new(ProtocolVersion::LATEST))
                    .block_task()
                    .await?;
                assert!(!initialize_response.agent_capabilities.load_session);

                let authenticate_response = cx
                    .send_request(agent_client_protocol::schema::AuthenticateRequest::new(
                        "none",
                    ))
                    .block_task()
                    .await?;
                assert!(authenticate_response.meta.is_none());

                let new_session_response = cx
                    .send_request(NewSessionRequest::new("/tmp"))
                    .block_task()
                    .await?;
                let set_mode_response = cx
                    .send_request(SetSessionModeRequest::new(
                        new_session_response.session_id.clone(),
                        "yolo",
                    ))
                    .block_task()
                    .await?;
                assert!(set_mode_response.meta.is_none());

                Ok(())
            })
            .await?;

        server.abort();

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn registry_and_tool_execution_helpers_cover_error_branches()
    -> Result<(), agent_client_protocol::Error> {
        let context = ToolContext {
            session_id: agent_client_protocol::schema::SessionId::new("session-registry"),
            cwd: std::path::PathBuf::from("/tmp"),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };

        let empty_result = EmptyToolRegistry
            .execute(
                &DeepSeekToolCall::new("empty", "anything", "{}"),
                &context,
                None,
            )
            .await;
        assert!(!empty_result.success);
        assert!(empty_result.content.contains("unknown tool: anything"));

        let read_only_result = ReadOnlyToolRegistry
            .execute(
                &DeepSeekToolCall::new("read-only", "bogus", "{}"),
                &context,
                None,
            )
            .await;
        assert!(!read_only_result.success);
        assert!(read_only_result.content.contains("unknown tool: bogus"));

        let failed = ToolExecution::failed("boom");
        assert!(!failed.success);
        assert_eq!(
            failed.status(),
            agent_client_protocol::schema::ToolCallStatus::Failed
        );
        assert_eq!(failed.content_for_model(), "boom");

        let succeeded = ToolExecution::completed("ok", serde_json::json!({ "value": 1 }));
        assert_eq!(
            succeeded.status(),
            agent_client_protocol::schema::ToolCallStatus::Completed
        );
        assert_eq!(succeeded.content_for_model(), "ok");

        Ok(())
    }

    #[test_log::test]
    fn handle_set_session_mode_request_rejects_invalid_inputs()
    -> Result<(), agent_client_protocol::Error> {
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let session = handle_new_session_request(&state, &NewSessionRequest::new("/tmp"))?;

        let Err(error) = handle_set_session_mode_request(
            &state,
            &SetSessionModeRequest::new(session.session_id.clone(), "bogus"),
        ) else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("expected invalid mode id to fail"));
        };
        assert!(error.to_string().contains("unsupported session mode"));

        let Err(error) = handle_set_session_mode_request(
            &state,
            &SetSessionModeRequest::new(
                agent_client_protocol::schema::SessionId::new("missing"),
                "ask",
            ),
        ) else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("expected missing session id to fail"));
        };
        assert!(error.to_string().contains("unknown session id"));

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn handle_prompt_request_rejects_unknown_session()
    -> Result<(), agent_client_protocol::Error> {
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let Err(error) = handle_prompt_request(
            &state,
            &MockLlmClient,
            &EmptyToolRegistry,
            None,
            PromptRequest::new(
                agent_client_protocol::schema::SessionId::new("missing"),
                vec![ContentBlock::from("hi")],
            ),
            |_| Ok(()),
        )
        .await
        else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("expected missing session id to fail"));
        };
        assert!(error.to_string().contains("unknown session id"));

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn request_permission_rejects_unknown_option() -> Result<(), agent_client_protocol::Error>
    {
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let session = handle_new_session_request(&state, &NewSessionRequest::new("/tmp"))?;
        let context = ToolContext {
            session_id: session.session_id.clone(),
            cwd: std::path::PathBuf::from("/tmp"),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let call = DeepSeekToolCall::new(
            "unknown-option-call",
            "write_file",
            serde_json::json!({ "path": "file.txt" }).to_string(),
        );
        let requester = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
            RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new("bogus")),
        )]);

        let Err(error) =
            request_tool_permission(&state, &context, &call, ToolKind::Edit, &requester).await
        else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("expected unknown permission option to fail"));
        };
        assert!(
            error
                .to_string()
                .contains("unknown permission option selected")
        );

        Ok(())
    }
}
