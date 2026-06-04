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
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::{error::Error, process::ExitCode};

use agent_client_protocol::schema::{
    AgentAuthCapabilities, AgentCapabilities, AuthenticateRequest, AuthenticateResponse,
    AvailableCommand, AvailableCommandInput, AvailableCommandsUpdate, CancelNotification,
    ClientCapabilities, ContentBlock, ContentChunk, CreateTerminalRequest, CreateTerminalResponse,
    Implementation, InitializeRequest, InitializeResponse, McpServer, McpServerStdio,
    NewSessionRequest, NewSessionResponse, PermissionOption, PermissionOptionKind, Plan, PlanEntry,
    PlanEntryPriority, PlanEntryStatus, PromptCapabilities, PromptRequest, PromptResponse,
    ProtocolVersion, ReadTextFileRequest, ReadTextFileResponse, ReleaseTerminalRequest,
    ReleaseTerminalResponse, RequestPermissionOutcome, RequestPermissionRequest,
    RequestPermissionResponse, SelectedPermissionOutcome, SessionConfigOption,
    SessionConfigOptionCategory, SessionConfigOptionValue, SessionConfigSelectOption,
    SessionConfigValueId, SessionId, SessionMode, SessionModeId, SessionModeState,
    SessionNotification, SessionUpdate, SetSessionConfigOptionRequest,
    SetSessionConfigOptionResponse, SetSessionModeRequest, SetSessionModeResponse, StopReason,
    TerminalOutputRequest, TerminalOutputResponse, ToolCall as AcpToolCall, ToolCallContent,
    ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields, ToolKind, UnstructuredCommandInput,
    WaitForTerminalExitRequest, WaitForTerminalExitResponse, WriteTextFileRequest,
    WriteTextFileResponse,
};
use agent_client_protocol::util::MatchDispatch;
use agent_client_protocol::{AcpAgent, Agent, Client, ConnectTo, SessionMessage, Stdio};
use clap::{Parser, Subcommand, ValueEnum};
use deepseek_acp_adapter::deepseek::{
    ChatMessage, ChatRequest, DeepSeekClient, DeepSeekConfig, DeepSeekError, FinishReason,
    LlmClient, StreamEvent, ToolCall as DeepSeekToolCall, ToolCallDelta, ToolDefinition,
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
use rmcp::model::{CallToolRequestParams, Content as McpContent, JsonObject, Tool as McpTool};
use rmcp::service::RunningService;
use rmcp::transport::{ConfigureCommandExt, TokioChildProcess};
use rmcp::{Peer, RoleClient, ServiceExt};
use serde::Deserialize;
use serde_json::Value;
use tokio::process::Command as TokioCommand;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

type AdapterResult<T> = Result<T, Box<dyn Error + Send + Sync + 'static>>;
const MAX_TURN_REQUESTS: usize = 100;
const PERMISSION_ALLOW_ONCE_OPTION_ID: &str = "allow_once";
const PERMISSION_ALLOW_ALWAYS_OPTION_ID: &str = "allow_always";
const PERMISSION_REJECT_ONCE_OPTION_ID: &str = "reject_once";
const PERMISSION_REJECT_ALWAYS_OPTION_ID: &str = "reject_always";
const SESSION_MODE_ASK_ID: &str = "ask";
const SESSION_MODE_ACCEPT_EDITS_ID: &str = "accept-edits";
const SESSION_MODE_YOLO_ID: &str = "yolo";
const SESSION_CONFIG_MODE_ID: &str = "mode";
const SESSION_CONFIG_MODEL_ID: &str = "model";
const SESSION_CONFIG_REASONING_EFFORT_ID: &str = "reasoning_effort";
const DEEPSEEK_V4_FLASH_MODEL_ID: &str = "deepseek-v4-flash";
const DEEPSEEK_V4_PRO_MODEL_ID: &str = "deepseek-v4-pro";
const REASONING_EFFORT_HIGH_ID: &str = "high";
const REASONING_EFFORT_MAX_ID: &str = "max";
const MCP_TOOL_PREFIX: &str = "mcp";
const ADAPTER_NAME: &str = env!("CARGO_PKG_NAME");
const ADAPTER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Returns the list of available slash commands for the `DeepSeek` adapter.
///
/// These commands are advertised to the client via `AvailableCommandsUpdate`
/// after session creation, letting users invoke common workflows.
#[must_use]
fn adapter_available_commands() -> Vec<AvailableCommand> {
    vec![
        AvailableCommand::new("explain", "Explain selected code or a concept in detail").input(
            AvailableCommandInput::Unstructured(UnstructuredCommandInput::new(
                "The code or concept to explain",
            )),
        ),
        AvailableCommand::new("fix", "Identify and fix issues in the selected code").input(
            AvailableCommandInput::Unstructured(UnstructuredCommandInput::new(
                "The code with issues to fix",
            )),
        ),
        AvailableCommand::new("test", "Generate tests for the selected code").input(
            AvailableCommandInput::Unstructured(UnstructuredCommandInput::new(
                "The code to generate tests for",
            )),
        ),
        AvailableCommand::new(
            "search",
            "Search the codebase for relevant code or documentation",
        )
        .input(AvailableCommandInput::Unstructured(
            UnstructuredCommandInput::new("The search query or keywords"),
        )),
        AvailableCommand::new("clear", "Clear the conversation history and start fresh"),
    ]
}

/// Build a `Plan` from a user prompt by splitting it into logical steps.
///
/// If the prompt contains multiple sentences, each becomes a plan entry.
/// Otherwise a single entry captures the entire request.
#[must_use]
fn plan_from_prompt(prompt: &str) -> Plan {
    let entries: Vec<PlanEntry> = if prompt.contains('.') || prompt.contains('\n') {
        prompt
            .split(['.', '\n'])
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| {
                PlanEntry::new(
                    s.to_string(),
                    PlanEntryPriority::Medium,
                    PlanEntryStatus::Pending,
                )
            })
            .collect()
    } else {
        vec![PlanEntry::new(
            prompt.to_string(),
            PlanEntryPriority::High,
            PlanEntryStatus::InProgress,
        )]
    };

    Plan::new(entries)
}

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
    let tool_registry = Arc::new(AdapterToolRegistry);
    let state = Arc::new(Mutex::new(AdapterState::new(initial_model_from_env())));
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

trait ReadTextFileRequester: Send + Sync {
    fn read_text_file(
        &self,
        request: ReadTextFileRequest,
    ) -> BoxFuture<'_, Result<ReadTextFileResponse, agent_client_protocol::Error>>;
}

trait WriteTextFileRequester: Send + Sync {
    fn write_text_file(
        &self,
        request: WriteTextFileRequest,
    ) -> BoxFuture<'_, Result<WriteTextFileResponse, agent_client_protocol::Error>>;
}

/// Trait for creating a terminal via ACP client `terminal/create`.
trait CreateTerminalRequester: Send + Sync {
    /// Create a terminal and execute a command.
    fn create_terminal(
        &self,
        request: CreateTerminalRequest,
    ) -> BoxFuture<'_, Result<CreateTerminalResponse, agent_client_protocol::Error>>;
}

/// Trait for getting terminal output via ACP client `terminal/output`.
trait TerminalOutputRequester: Send + Sync {
    /// Get the current output and status of a terminal.
    fn terminal_output(
        &self,
        request: TerminalOutputRequest,
    ) -> BoxFuture<'_, Result<TerminalOutputResponse, agent_client_protocol::Error>>;
}

/// Trait for waiting for terminal exit via ACP client `terminal/wait_for_exit`.
trait WaitForTerminalExitRequester: Send + Sync {
    /// Wait for a terminal command to exit.
    fn wait_for_terminal_exit(
        &self,
        request: WaitForTerminalExitRequest,
    ) -> BoxFuture<'_, Result<WaitForTerminalExitResponse, agent_client_protocol::Error>>;
}

/// Trait for releasing a terminal via ACP client `terminal/release`.
trait ReleaseTerminalRequester: Send + Sync {
    /// Release a terminal and free its resources.
    fn release_terminal(
        &self,
        request: ReleaseTerminalRequest,
    ) -> BoxFuture<'_, Result<ReleaseTerminalResponse, agent_client_protocol::Error>>;
}

/// Combined trait for all terminal operations.
trait TerminalRequester:
    CreateTerminalRequester
    + TerminalOutputRequester
    + WaitForTerminalExitRequester
    + ReleaseTerminalRequester
{
}

impl<T> TerminalRequester for T where
    T: CreateTerminalRequester
        + TerminalOutputRequester
        + WaitForTerminalExitRequester
        + ReleaseTerminalRequester
        + ?Sized
{
}

impl CreateTerminalRequester for agent_client_protocol::ConnectionTo<Agent> {
    fn create_terminal(
        &self,
        request: CreateTerminalRequest,
    ) -> BoxFuture<'_, Result<CreateTerminalResponse, agent_client_protocol::Error>> {
        Box::pin(async move { self.send_request(request).block_task().await })
    }
}

impl CreateTerminalRequester for agent_client_protocol::ConnectionTo<Client> {
    fn create_terminal(
        &self,
        request: CreateTerminalRequest,
    ) -> BoxFuture<'_, Result<CreateTerminalResponse, agent_client_protocol::Error>> {
        Box::pin(async move { self.send_request(request).block_task().await })
    }
}

impl TerminalOutputRequester for agent_client_protocol::ConnectionTo<Agent> {
    fn terminal_output(
        &self,
        request: TerminalOutputRequest,
    ) -> BoxFuture<'_, Result<TerminalOutputResponse, agent_client_protocol::Error>> {
        Box::pin(async move { self.send_request(request).block_task().await })
    }
}

impl TerminalOutputRequester for agent_client_protocol::ConnectionTo<Client> {
    fn terminal_output(
        &self,
        request: TerminalOutputRequest,
    ) -> BoxFuture<'_, Result<TerminalOutputResponse, agent_client_protocol::Error>> {
        Box::pin(async move { self.send_request(request).block_task().await })
    }
}

impl WaitForTerminalExitRequester for agent_client_protocol::ConnectionTo<Agent> {
    fn wait_for_terminal_exit(
        &self,
        request: WaitForTerminalExitRequest,
    ) -> BoxFuture<'_, Result<WaitForTerminalExitResponse, agent_client_protocol::Error>> {
        Box::pin(async move { self.send_request(request).block_task().await })
    }
}

impl WaitForTerminalExitRequester for agent_client_protocol::ConnectionTo<Client> {
    fn wait_for_terminal_exit(
        &self,
        request: WaitForTerminalExitRequest,
    ) -> BoxFuture<'_, Result<WaitForTerminalExitResponse, agent_client_protocol::Error>> {
        Box::pin(async move { self.send_request(request).block_task().await })
    }
}

impl ReleaseTerminalRequester for agent_client_protocol::ConnectionTo<Agent> {
    fn release_terminal(
        &self,
        request: ReleaseTerminalRequest,
    ) -> BoxFuture<'_, Result<ReleaseTerminalResponse, agent_client_protocol::Error>> {
        Box::pin(async move { self.send_request(request).block_task().await })
    }
}

impl ReleaseTerminalRequester for agent_client_protocol::ConnectionTo<Client> {
    fn release_terminal(
        &self,
        request: ReleaseTerminalRequest,
    ) -> BoxFuture<'_, Result<ReleaseTerminalResponse, agent_client_protocol::Error>> {
        Box::pin(async move { self.send_request(request).block_task().await })
    }
}

trait ToolCallRequester:
    ReadTextFileRequester + WriteTextFileRequester + PermissionRequester + TerminalRequester
{
}

impl<T> ToolCallRequester for T where
    T: ReadTextFileRequester
        + WriteTextFileRequester
        + PermissionRequester
        + TerminalRequester
        + ?Sized
{
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

impl WriteTextFileRequester for agent_client_protocol::ConnectionTo<Agent> {
    fn write_text_file(
        &self,
        request: WriteTextFileRequest,
    ) -> BoxFuture<'_, Result<WriteTextFileResponse, agent_client_protocol::Error>> {
        Box::pin(async move { self.send_request(request).block_task().await })
    }
}

impl WriteTextFileRequester for agent_client_protocol::ConnectionTo<Client> {
    fn write_text_file(
        &self,
        request: WriteTextFileRequest,
    ) -> BoxFuture<'_, Result<WriteTextFileResponse, agent_client_protocol::Error>> {
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

async fn serve_with_transport(
    transport: impl ConnectTo<Agent> + 'static,
    state: Arc<Mutex<AdapterState>>,
    llm_client: Arc<dyn LlmClient>,
    tool_registry: Arc<dyn ToolRegistry>,
) -> Result<(), agent_client_protocol::Error> {
    let initialize_state = Arc::clone(&state);
    let new_session_state = Arc::clone(&state);
    let set_mode_state = Arc::clone(&state);
    let set_config_state = Arc::clone(&state);
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
            async move |request: NewSessionRequest, responder, cx| {
                let session_state = Arc::clone(&new_session_state);
                let connection = cx.clone();

                cx.spawn(async move {
                    let response =
                        handle_new_session_request_connected(&session_state, &request).await?;
                    let session_id = response.session_id.clone();

                    // Advertise available slash commands after session creation.
                    let commands = adapter_available_commands();
                    if !commands.is_empty() {
                        connection.send_notification(session_notification(
                            session_id,
                            SessionUpdate::AvailableCommandsUpdate(AvailableCommandsUpdate::new(
                                commands,
                            )),
                        ))?;
                    }

                    responder.respond(response)?;
                    Ok(())
                })?;

                Ok(())
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
            async move |request: SetSessionConfigOptionRequest, responder, _cx| {
                responder.respond(handle_set_session_config_option_request(
                    &set_config_state,
                    &request,
                )?)
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
                        Some(&connection as &dyn ToolCallRequester),
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
    if !request.mcp_servers.is_empty() {
        return Err(agent_client_protocol::Error::invalid_params()
            .data("MCP servers require the async session setup path"));
    }
    insert_session_record(state, request, Vec::new())
}

async fn handle_new_session_request_connected(
    state: &Arc<Mutex<AdapterState>>,
    request: &NewSessionRequest,
) -> Result<NewSessionResponse, agent_client_protocol::Error> {
    validate_session_paths(request)?;
    let mcp_sessions = connect_mcp_sessions(&request.mcp_servers).await?;
    insert_session_record(state, request, mcp_sessions)
}

fn insert_session_record(
    state: &Arc<Mutex<AdapterState>>,
    request: &NewSessionRequest,
    mcp_sessions: Vec<McpSession>,
) -> Result<NewSessionResponse, agent_client_protocol::Error> {
    validate_session_paths(request)?;
    let session_id = format!("session-{}", Uuid::new_v4());
    let mut guard = state
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let default_model = guard.default_model.clone();
    guard.sessions.insert(
        session_id.clone().into(),
        SessionRecord {
            cwd: request.cwd.clone(),
            additional_directories: request.additional_directories.clone(),
            history: Vec::new(),
            active_turn: None,
            mode: PermissionPosture::Ask,
            model: default_model,
            reasoning_effort: ReasoningEffort::High,
            permission_allow_always: HashSet::new(),
            mcp_sessions,
        },
    );

    let session = guard
        .sessions
        .get(&SessionId::new(session_id.clone()))
        .ok_or_else(|| {
            agent_client_protocol::Error::internal_error().data("failed to create session")
        })?;

    Ok(NewSessionResponse::new(session_id)
        .modes(default_session_modes())
        .config_options(session_config_options(session)))
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

fn handle_set_session_config_option_request(
    state: &Arc<Mutex<AdapterState>>,
    request: &SetSessionConfigOptionRequest,
) -> Result<SetSessionConfigOptionResponse, agent_client_protocol::Error> {
    let mut guard = state
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let session = guard.sessions.get_mut(&request.session_id).ok_or_else(|| {
        agent_client_protocol::Error::invalid_params()
            .data(format!("unknown session id: {}", request.session_id.0))
    })?;
    let value = config_value_id(&request.value)?;

    match request.config_id.0.as_ref() {
        SESSION_CONFIG_MODE_ID => {
            let mode_id = SessionModeId::new(value.0.clone());
            let Some(mode) = PermissionPosture::from_mode_id(&mode_id) else {
                return Err(agent_client_protocol::Error::invalid_params()
                    .data(format!("unsupported session mode: {}", value.0)));
            };
            session.mode = mode;
        }
        SESSION_CONFIG_MODEL_ID => {
            let model = value.0.as_ref();
            validate_session_model(session, model)?;
            session.model = model.to_string();
        }
        SESSION_CONFIG_REASONING_EFFORT_ID => {
            let Some(effort) = ReasoningEffort::from_value_id(value) else {
                return Err(agent_client_protocol::Error::invalid_params()
                    .data(format!("unsupported reasoning effort: {}", value.0)));
            };
            session.reasoning_effort = effort;
        }
        _ => {
            return Err(agent_client_protocol::Error::invalid_params().data(format!(
                "unsupported session config option: {}",
                request.config_id.0
            )));
        }
    }

    Ok(SetSessionConfigOptionResponse::new(session_config_options(
        session,
    )))
}

fn config_value_id(
    value: &SessionConfigOptionValue,
) -> Result<&SessionConfigValueId, agent_client_protocol::Error> {
    value.as_value_id().ok_or_else(|| {
        agent_client_protocol::Error::invalid_params()
            .data("session config option requires a selectable value id")
    })
}

async fn handle_prompt_request(
    state: &Arc<Mutex<AdapterState>>,
    llm_client: &dyn LlmClient,
    tool_registry: &dyn ToolRegistry,
    connection: Option<&dyn ToolCallRequester>,
    request: PromptRequest,
    mut notify: impl FnMut(SessionNotification) -> Result<(), agent_client_protocol::Error>,
) -> Result<PromptResponse, agent_client_protocol::Error> {
    let user_text = text_from_prompt(&request.prompt)?;
    let user_message = ChatMessage::user(user_text.clone());
    let session_id = request.session_id.clone();
    let cancellation_token = CancellationToken::new();
    let (messages, tool_context, model, reasoning_effort) = {
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
            session.model.clone(),
            session.reasoning_effort,
        )
    };

    // Emit a plan derived from the user's prompt so the client can see
    // the intended execution strategy before streaming begins.
    let plan = plan_from_prompt(&user_text);
    if !plan.entries.is_empty() {
        notify(session_notification(
            session_id.clone(),
            SessionUpdate::Plan(plan),
        ))?;
    }

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
        ModelRequestSettings {
            model: &model,
            reasoning_effort,
        },
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
    connection: Option<&'a dyn ToolCallRequester>,
    tool_context: ToolContext,
    request: PromptRequest,
    cancellation_token: CancellationToken,
}

#[derive(Debug, Clone, Copy)]
struct ModelRequestSettings<'a> {
    model: &'a str,
    reasoning_effort: ReasoningEffort,
}

async fn run_prompt_turn(
    env: PromptTurnEnvironment<'_>,
    mut messages: Vec<ChatMessage>,
    model_settings: ModelRequestSettings<'_>,
    notify: &mut impl FnMut(SessionNotification) -> Result<(), agent_client_protocol::Error>,
) -> Result<PromptResponse, agent_client_protocol::Error> {
    let tool_definitions = env
        .tool_registry
        .definitions(&env.tool_context, env.state)?;

    let mut stop_reason = StopReason::MaxTurnRequests;

    for _ in 0..MAX_TURN_REQUESTS {
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
                .execute(tool_call, &env.tool_context, env.state, env.connection)
                .await;
            report_tool_result(&env.request.session_id, notify, tool_call, &tool_result)?;
            messages.push(ChatMessage::tool_result(
                tool_call.id(),
                tool_result.content_for_model(),
            ));
        }
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum ReasoningEffort {
    #[default]
    High,
    Max,
}

impl ReasoningEffort {
    const fn id(self) -> &'static str {
        match self {
            Self::High => REASONING_EFFORT_HIGH_ID,
            Self::Max => REASONING_EFFORT_MAX_ID,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::High => "High",
            Self::Max => "Max",
        }
    }

    const fn description(self) -> &'static str {
        match self {
            Self::High => "Default DeepSeek thinking effort.",
            Self::Max => "Maximum DeepSeek thinking effort for complex agent work.",
        }
    }

    fn from_value_id(value: &SessionConfigValueId) -> Option<Self> {
        match value.0.as_ref() {
            REASONING_EFFORT_HIGH_ID => Some(Self::High),
            REASONING_EFFORT_MAX_ID => Some(Self::Max),
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

#[derive(Debug, Deserialize)]
struct WriteFileArguments {
    path: PathBuf,
    content: String,
}

#[derive(Debug, Deserialize)]
struct EditFileArguments {
    path: PathBuf,
    old_text: String,
    new_text: String,
}

#[derive(Debug, Deserialize)]
struct RunCommandArguments {
    command: String,
}

const TOOL_OUTPUT_LIMIT: usize = 200;
const TOOL_OUTPUT_LIMIT_U32: u32 = 200;
const COMMAND_OUTPUT_LIMIT: usize = 20_000;

/// Registry for tools the model can call during a turn.
trait ToolRegistry: Send + Sync {
    /// Return tool definitions to advertise to the model.
    fn definitions(
        &self,
        context: &ToolContext,
        state: &Arc<Mutex<AdapterState>>,
    ) -> Result<Vec<ToolDefinition>, agent_client_protocol::Error>;

    /// Return the ACP kind used when displaying and gating a tool call.
    fn kind(&self, name: &str) -> ToolKind;

    /// Execute a complete model-requested tool call.
    fn execute<'a>(
        &'a self,
        call: &'a DeepSeekToolCall,
        context: &'a ToolContext,
        state: &'a Arc<Mutex<AdapterState>>,
        connection: Option<&'a dyn ToolCallRequester>,
    ) -> BoxFuture<'a, ToolExecution>;
}

#[cfg(test)]
#[derive(Debug)]
struct EmptyToolRegistry;

#[cfg(test)]
impl ToolRegistry for EmptyToolRegistry {
    fn definitions(
        &self,
        _context: &ToolContext,
        _state: &Arc<Mutex<AdapterState>>,
    ) -> Result<Vec<ToolDefinition>, agent_client_protocol::Error> {
        Ok(Vec::new())
    }

    fn kind(&self, _name: &str) -> ToolKind {
        ToolKind::Other
    }

    fn execute<'a>(
        &'a self,
        call: &'a DeepSeekToolCall,
        _context: &'a ToolContext,
        _state: &'a Arc<Mutex<AdapterState>>,
        _connection: Option<&'a dyn ToolCallRequester>,
    ) -> BoxFuture<'a, ToolExecution> {
        Box::pin(async move { ToolExecution::failed(format!("unknown tool: {}", call.name())) })
    }
}

#[derive(Debug)]
struct AdapterToolRegistry;

impl ToolRegistry for AdapterToolRegistry {
    fn definitions(
        &self,
        context: &ToolContext,
        state: &Arc<Mutex<AdapterState>>,
    ) -> Result<Vec<ToolDefinition>, agent_client_protocol::Error> {
        let mut definitions = vec![
            read_file_tool_definition(),
            list_dir_tool_definition(),
            glob_tool_definition(),
            grep_tool_definition(),
            write_file_tool_definition(),
            edit_file_tool_definition(),
            run_command_tool_definition(),
        ];
        definitions.extend(session_mcp_tool_definitions(state, &context.session_id)?);
        Ok(definitions)
    }

    fn kind(&self, name: &str) -> ToolKind {
        match name {
            "read_file" | "list_dir" => ToolKind::Read,
            "glob" | "grep" => ToolKind::Search,
            "write_file" | "edit_file" => ToolKind::Edit,
            "run_command" => ToolKind::Execute,
            name if name.starts_with(MCP_TOOL_PREFIX) => ToolKind::Other,
            _ => ToolKind::Other,
        }
    }

    fn execute<'a>(
        &'a self,
        call: &'a DeepSeekToolCall,
        context: &'a ToolContext,
        state: &'a Arc<Mutex<AdapterState>>,
        connection: Option<&'a dyn ToolCallRequester>,
    ) -> BoxFuture<'a, ToolExecution> {
        Box::pin(async move {
            match call.name() {
                "read_file" => {
                    read_file_tool_execution(
                        call,
                        context,
                        connection.map(|requester| requester as &dyn ReadTextFileRequester),
                    )
                    .await
                }
                "list_dir" => list_dir_tool_execution(call, context),
                "glob" => glob_tool_execution(call, context),
                "grep" => grep_tool_execution(call, context),
                "write_file" => {
                    write_file_tool_execution(
                        state,
                        call,
                        context,
                        connection.map(|requester| requester as &dyn WriteTextFileRequester),
                        connection.map(|requester| requester as &dyn PermissionRequester),
                    )
                    .await
                }
                "edit_file" => {
                    edit_file_tool_execution(
                        state,
                        call,
                        context,
                        connection.map(|requester| requester as &dyn ReadTextFileRequester),
                        connection.map(|requester| requester as &dyn WriteTextFileRequester),
                        connection.map(|requester| requester as &dyn PermissionRequester),
                    )
                    .await
                }
                "run_command" => {
                    run_command_tool_execution(
                        state,
                        call,
                        context,
                        connection.map(|requester| requester as &dyn PermissionRequester),
                        connection.map(|requester| requester as &dyn TerminalRequester),
                    )
                    .await
                }
                name if name.starts_with(MCP_TOOL_PREFIX) => {
                    mcp_tool_execution(state, call, context).await
                }
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

#[derive(Debug, Clone)]
struct McpToolTarget {
    server_name: String,
    original_name: String,
    peer: Peer<RoleClient>,
}

fn session_mcp_tool_definitions(
    state: &Arc<Mutex<AdapterState>>,
    session_id: &SessionId,
) -> Result<Vec<ToolDefinition>, agent_client_protocol::Error> {
    let guard = state
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let session = guard.sessions.get(session_id).ok_or_else(|| {
        agent_client_protocol::Error::invalid_params()
            .data(format!("unknown session id: {}", session_id.0))
    })?;

    Ok(session
        .mcp_sessions
        .iter()
        .flat_map(|session| session.tools.iter().map(|tool| tool.definition.clone()))
        .collect())
}

async fn mcp_tool_execution(
    state: &Arc<Mutex<AdapterState>>,
    call: &DeepSeekToolCall,
    context: &ToolContext,
) -> ToolExecution {
    let target = match find_mcp_tool_target(state, &context.session_id, call.name()) {
        Ok(Some(target)) => target,
        Ok(None) => return ToolExecution::failed(format!("unknown MCP tool: {}", call.name())),
        Err(error) => return ToolExecution::failed(error.to_string()),
    };

    let arguments = match mcp_call_arguments(call) {
        Ok(arguments) => arguments,
        Err(error) => return ToolExecution::failed(error),
    };

    let result = target
        .peer
        .call_tool(
            CallToolRequestParams::new(target.original_name.clone()).with_arguments(arguments),
        )
        .await;

    match result {
        Ok(result) => {
            let model_output = mcp_tool_result_text(&result.content);
            let raw_output = serde_json::to_value(&result).unwrap_or_else(|error| {
                serde_json::json!({
                    "error": format!("failed to serialize MCP tool result: {error}")
                })
            });
            ToolExecution {
                content: model_output,
                raw_output,
                success: !result.is_error.unwrap_or(false),
            }
        }
        Err(error) => ToolExecution::failed(format!(
            "MCP tool '{}' on server '{}' failed: {error}",
            target.original_name, target.server_name
        )),
    }
}

fn find_mcp_tool_target(
    state: &Arc<Mutex<AdapterState>>,
    session_id: &SessionId,
    exposed_name: &str,
) -> Result<Option<McpToolTarget>, agent_client_protocol::Error> {
    let guard = state
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let session = guard.sessions.get(session_id).ok_or_else(|| {
        agent_client_protocol::Error::invalid_params()
            .data(format!("unknown session id: {}", session_id.0))
    })?;

    Ok(session.mcp_sessions.iter().find_map(|mcp_session| {
        mcp_session.tools.iter().find_map(|tool| {
            if tool.exposed_name == exposed_name {
                Some(McpToolTarget {
                    server_name: mcp_session.name.clone(),
                    original_name: tool.original_name.clone(),
                    peer: mcp_session.peer.clone(),
                })
            } else {
                None
            }
        })
    }))
}

fn mcp_call_arguments(call: &DeepSeekToolCall) -> Result<JsonObject, String> {
    match serde_json::from_str::<Value>(call.arguments()) {
        Ok(Value::Object(arguments)) => Ok(arguments),
        Ok(_) => Err(format!(
            "MCP tool '{}' arguments must be a JSON object",
            call.name()
        )),
        Err(error) => Err(format!(
            "invalid MCP tool '{}' arguments: {error}",
            call.name()
        )),
    }
}

fn mcp_tool_result_text(content: &[McpContent]) -> String {
    let parts = content
        .iter()
        .map(|content| {
            content.raw.as_text().map_or_else(
                || {
                    serde_json::to_string(&content.raw)
                        .unwrap_or_else(|error| format!("failed to serialize MCP content: {error}"))
                },
                |text| text.text.clone(),
            )
        })
        .collect::<Vec<_>>();

    if parts.is_empty() {
        String::new()
    } else {
        parts.join("\n")
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

fn write_file_tool_definition() -> ToolDefinition {
    ToolDefinition::new(
        "write_file",
        "Write UTF-8 text to a file, creating or replacing the file.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "content": { "type": "string" },
            },
            "required": ["path", "content"],
            "additionalProperties": false,
        }),
    )
}

fn edit_file_tool_definition() -> ToolDefinition {
    ToolDefinition::new(
        "edit_file",
        "Replace one exact UTF-8 text span in an existing file.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "old_text": { "type": "string" },
                "new_text": { "type": "string" },
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": false,
        }),
    )
}

fn run_command_tool_definition() -> ToolDefinition {
    ToolDefinition::new(
        "run_command",
        "Run a shell command in the session working directory.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": { "type": "string" },
            },
            "required": ["command"],
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
                if local_file_is_non_utf8(&resolved_path) {
                    return ToolExecution::failed(non_utf8_file_message(&resolved_path));
                }

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

async fn write_file_tool_execution(
    state: &Arc<Mutex<AdapterState>>,
    call: &DeepSeekToolCall,
    context: &ToolContext,
    write_connection: Option<&dyn WriteTextFileRequester>,
    permission_requester: Option<&dyn PermissionRequester>,
) -> ToolExecution {
    let parsed_arguments = match serde_json::from_str::<WriteFileArguments>(call.arguments()) {
        Ok(arguments) => arguments,
        Err(error) => {
            return ToolExecution::failed(format!("invalid write_file arguments: {error}"));
        }
    };

    if let Err(error) =
        require_tool_permission(state, context, call, ToolKind::Edit, permission_requester).await
    {
        return ToolExecution::failed(error);
    }

    let resolved_path = resolve_tool_path(context, &parsed_arguments.path);
    let use_client_write = context
        .client_capabilities
        .as_ref()
        .is_some_and(|capabilities| capabilities.fs.write_text_file);
    let write_result = if use_client_write {
        match write_connection {
            Some(connection) => {
                write_file_to_client(
                    connection,
                    &context.session_id,
                    &resolved_path,
                    &parsed_arguments.content,
                )
                .await
            }
            None => Err("write_file needs a client connection for fs/write_text_file".to_owned()),
        }
    } else {
        write_file_to_local(&resolved_path, &parsed_arguments.content)
    };

    match write_result {
        Ok(()) => {
            let byte_count = parsed_arguments.content.len();
            ToolExecution {
                content: format!("wrote {byte_count} bytes to {}", resolved_path.display()),
                raw_output: serde_json::json!({
                    "path": resolved_path,
                    "bytes": byte_count,
                    "source": if use_client_write { "client" } else { "local" },
                }),
                success: true,
            }
        }
        Err(error) => ToolExecution::failed(error),
    }
}

async fn edit_file_tool_execution(
    state: &Arc<Mutex<AdapterState>>,
    call: &DeepSeekToolCall,
    context: &ToolContext,
    read_connection: Option<&dyn ReadTextFileRequester>,
    write_connection: Option<&dyn WriteTextFileRequester>,
    permission_requester: Option<&dyn PermissionRequester>,
) -> ToolExecution {
    let parsed_arguments = match serde_json::from_str::<EditFileArguments>(call.arguments()) {
        Ok(arguments) => arguments,
        Err(error) => {
            return ToolExecution::failed(format!("invalid edit_file arguments: {error}"));
        }
    };

    if parsed_arguments.old_text.is_empty() {
        return ToolExecution::failed("edit_file old_text must not be empty");
    }

    let resolved_path = resolve_tool_path(context, &parsed_arguments.path);
    let use_client_read = context
        .client_capabilities
        .as_ref()
        .is_some_and(|capabilities| capabilities.fs.read_text_file);
    let original_result = if use_client_read {
        match read_connection {
            Some(connection) => {
                read_full_file_from_client(connection, &context.session_id, &resolved_path).await
            }
            None => Err("edit_file needs a client connection for fs/read_text_file".to_owned()),
        }
    } else {
        fs::read_to_string(&resolved_path).map_err(|error| {
            format!(
                "failed to read {} before editing: {error}",
                resolved_path.display()
            )
        })
    };

    let original = match original_result {
        Ok(file_text) => file_text,
        Err(error) => return ToolExecution::failed(error),
    };

    let matches = original.matches(&parsed_arguments.old_text).count();
    if matches == 0 {
        return ToolExecution::failed(format!(
            "edit_file could not find old_text in {}",
            resolved_path.display()
        ));
    }
    if matches > 1 {
        return ToolExecution::failed(format!(
            "edit_file found old_text {matches} times in {}; provide a unique span",
            resolved_path.display()
        ));
    }

    if let Err(error) =
        require_tool_permission(state, context, call, ToolKind::Edit, permission_requester).await
    {
        return ToolExecution::failed(error);
    }

    let updated = original.replacen(&parsed_arguments.old_text, &parsed_arguments.new_text, 1);
    let use_client_write = context
        .client_capabilities
        .as_ref()
        .is_some_and(|capabilities| capabilities.fs.write_text_file);
    let write_result = if use_client_write {
        match write_connection {
            Some(connection) => {
                write_file_to_client(connection, &context.session_id, &resolved_path, &updated)
                    .await
            }
            None => Err("edit_file needs a client connection for fs/write_text_file".to_owned()),
        }
    } else {
        write_file_to_local(&resolved_path, &updated)
    };

    match write_result {
        Ok(()) => ToolExecution {
            content: format!("edited {}", resolved_path.display()),
            raw_output: serde_json::json!({
                "path": resolved_path,
                "replacements": 1,
                "read_source": if use_client_read { "client" } else { "local" },
                "write_source": if use_client_write { "client" } else { "local" },
            }),
            success: true,
        },
        Err(error) => ToolExecution::failed(error),
    }
}

async fn run_command_tool_execution(
    state: &Arc<Mutex<AdapterState>>,
    call: &DeepSeekToolCall,
    context: &ToolContext,
    permission_requester: Option<&dyn PermissionRequester>,
    terminal_connection: Option<&dyn TerminalRequester>,
) -> ToolExecution {
    let parsed_arguments = match serde_json::from_str::<RunCommandArguments>(call.arguments()) {
        Ok(arguments) => arguments,
        Err(error) => {
            return ToolExecution::failed(format!("invalid run_command arguments: {error}"));
        }
    };

    if parsed_arguments.command.trim().is_empty() {
        return ToolExecution::failed("run_command command must not be empty");
    }

    if let Err(error) = require_tool_permission(
        state,
        context,
        call,
        ToolKind::Execute,
        permission_requester,
    )
    .await
    {
        return ToolExecution::failed(error);
    }

    // Route to client terminal methods when the client advertises terminal support.
    if context
        .client_capabilities
        .as_ref()
        .is_some_and(|capabilities| capabilities.terminal)
    {
        return run_command_via_terminal(
            &context.session_id,
            &context.cwd,
            &parsed_arguments.command,
            terminal_connection,
        )
        .await;
    }

    // Fall back to local process execution.
    let cwd = context.cwd.clone();
    let command = parsed_arguments.command;
    let output = match tokio::task::spawn_blocking(move || {
        std::process::Command::new("sh")
            .arg("-lc")
            .arg(&command)
            .current_dir(cwd)
            .output()
    })
    .await
    {
        Ok(Ok(output)) => output,
        Ok(Err(error)) => return ToolExecution::failed(format!("failed to run command: {error}")),
        Err(error) => return ToolExecution::failed(format!("run_command task failed: {error}")),
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined_output = render_command_output(&stdout, &stderr, output.status.code());
    let (display_output, truncated) = truncate_tool_output(&combined_output, COMMAND_OUTPUT_LIMIT);

    ToolExecution {
        content: display_output,
        raw_output: serde_json::json!({
            "exit_code": output.status.code(),
            "success": output.status.success(),
            "stdout": stdout,
            "stderr": stderr,
            "truncated": truncated,
        }),
        success: output.status.success(),
    }
}

/// Run a command through ACP client terminal methods.
///
/// Creates a terminal via `terminal/create`, waits for exit via
/// `terminal/wait_for_exit`, reads output via `terminal/output`, and
/// releases via `terminal/release`.
async fn run_command_via_terminal(
    session_id: &SessionId,
    cwd: &Path,
    command: &str,
    connection: Option<&dyn TerminalRequester>,
) -> ToolExecution {
    let Some(terminal_requester) = connection else {
        return ToolExecution::failed("terminal support advertised but no connection available");
    };

    // Create terminal and execute the command.
    let create_request = CreateTerminalRequest::new(session_id.clone(), command)
        .cwd(Some(cwd.to_path_buf()))
        .output_byte_limit(Some(COMMAND_OUTPUT_LIMIT as u64));
    let create_response = match terminal_requester.create_terminal(create_request).await {
        Ok(response) => response,
        Err(error) => {
            return ToolExecution::failed(format!("terminal/create failed: {error}"));
        }
    };
    let terminal_id = create_response.terminal_id;

    // Wait for the command to finish.
    let wait_request = WaitForTerminalExitRequest::new(session_id.clone(), terminal_id.clone());
    let wait_response = match terminal_requester
        .wait_for_terminal_exit(wait_request)
        .await
    {
        Ok(response) => response,
        Err(error) => {
            // Try to release on error before returning.
            let _ = terminal_requester
                .release_terminal(ReleaseTerminalRequest::new(
                    session_id.clone(),
                    terminal_id.clone(),
                ))
                .await;
            return ToolExecution::failed(format!("terminal/wait_for_exit failed: {error}"));
        }
    };

    // Read terminal output.
    let output_request = TerminalOutputRequest::new(session_id.clone(), terminal_id.clone());
    let output_response = match terminal_requester.terminal_output(output_request).await {
        Ok(response) => response,
        Err(error) => {
            let _ = terminal_requester
                .release_terminal(ReleaseTerminalRequest::new(
                    session_id.clone(),
                    terminal_id.clone(),
                ))
                .await;
            return ToolExecution::failed(format!("terminal/output failed: {error}"));
        }
    };

    // Release the terminal.
    if let Err(error) = terminal_requester
        .release_terminal(ReleaseTerminalRequest::new(
            session_id.clone(),
            terminal_id.clone(),
        ))
        .await
    {
        return ToolExecution::failed(format!("terminal/release failed: {error}"));
    }

    let exit_code = wait_response.exit_status.exit_code;
    let success = exit_code == Some(0);
    let exit_code_i32 = exit_code.and_then(|code| i32::try_from(code).ok());
    let combined_output = render_command_output(&output_response.output, "", exit_code_i32);
    let (display_output, truncated) = truncate_tool_output(&combined_output, COMMAND_OUTPUT_LIMIT);

    ToolExecution {
        content: display_output,
        raw_output: serde_json::json!({
            "exit_code": exit_code,
            "success": success,
            "stdout": output_response.output,
            "stderr": "",
            "truncated": truncated || output_response.truncated,
        }),
        success,
    }
}

async fn require_tool_permission(
    state: &Arc<Mutex<AdapterState>>,
    context: &ToolContext,
    call: &DeepSeekToolCall,
    kind: ToolKind,
    requester: Option<&dyn PermissionRequester>,
) -> Result<(), String> {
    let requester = requester.ok_or_else(|| {
        format!(
            "{} requires a client connection that can request permissions",
            call.name()
        )
    })?;

    match request_tool_permission(state, context, call, kind, requester).await {
        Ok(
            PermissionDecision::AllowOnce
            | PermissionDecision::AllowAlways
            | PermissionDecision::AllowByMode,
        ) => Ok(()),
        Ok(PermissionDecision::RejectOnce | PermissionDecision::RejectAlways) => {
            Err(format!("{} was rejected by permission policy", call.name()))
        }
        Ok(PermissionDecision::Cancelled) => {
            Err(format!("{} permission request was cancelled", call.name()))
        }
        Err(error) => Err(format!(
            "failed to request permission for {}: {error}",
            call.name()
        )),
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
        .map_err(|error| read_file_client_error(path, &error.to_string()))?;

    Ok(response.content)
}

async fn read_full_file_from_client(
    connection: &dyn ReadTextFileRequester,
    session_id: &SessionId,
    path: &Path,
) -> Result<String, String> {
    let response = connection
        .read_text_file(ReadTextFileRequest::new(
            session_id.clone(),
            path.to_path_buf(),
        ))
        .await
        .map_err(|error| read_file_client_error(path, &error.to_string()))?;

    Ok(response.content)
}

async fn write_file_to_client(
    connection: &dyn WriteTextFileRequester,
    session_id: &SessionId,
    path: &Path,
    content: &str,
) -> Result<(), String> {
    connection
        .write_text_file(WriteTextFileRequest::new(
            session_id.clone(),
            path.to_path_buf(),
            content.to_owned(),
        ))
        .await
        .map_err(|error| {
            format!(
                "failed to write {} through client fs/write_text_file: {error}",
                path.display()
            )
        })?;

    Ok(())
}

fn write_file_to_local(path: &Path, content: &str) -> Result<(), String> {
    fs::write(path, content.as_bytes())
        .map_err(|error| format!("failed to write {}: {error}", path.display()))
}

fn read_file_from_local(path: &Path, line: u32, limit: u32) -> Result<String, String> {
    let text = fs::read_to_string(path).map_err(|error| read_file_local_error(path, &error))?;
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

fn local_file_is_non_utf8(path: &Path) -> bool {
    fs::read_to_string(path).is_err_and(|error| error.kind() == ErrorKind::InvalidData)
}

fn read_file_local_error(path: &Path, error: &std::io::Error) -> String {
    if error.kind() == ErrorKind::InvalidData {
        return non_utf8_file_message(path);
    }

    format!("failed to read {}: {error}", path.display())
}

fn read_file_client_error(path: &Path, message: &str) -> String {
    if is_utf8_error_message(message) {
        return non_utf8_file_message(path);
    }

    format!(
        "failed to read {} through client fs/read_text_file: {message}",
        path.display()
    )
}

fn non_utf8_file_message(path: &Path) -> String {
    format!(
        "read_file only supports UTF-8 text files; {} appears to be binary or non-UTF-8",
        path.display()
    )
}

fn is_utf8_error_message(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("valid utf-8")
        || lower.contains("invalid utf-8")
        || lower.contains("non-utf-8")
        || lower.contains("utf8")
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

fn render_command_output(stdout: &str, stderr: &str, exit_code: Option<i32>) -> String {
    let mut output = String::new();

    if !stdout.is_empty() {
        output.push_str("stdout:\n");
        output.push_str(stdout);
        if !stdout.ends_with('\n') {
            output.push('\n');
        }
    }

    if !stderr.is_empty() {
        if !output.is_empty() {
            output.push('\n');
        }
        output.push_str("stderr:\n");
        output.push_str(stderr);
        if !stderr.ends_with('\n') {
            output.push('\n');
        }
    }

    if output.is_empty() {
        let status = exit_code.map_or_else(|| "signal".to_string(), |code| code.to_string());
        let _ = write!(output, "command exited with status {status}");
    }

    output
}

fn truncate_tool_output(output: &str, limit: usize) -> (String, bool) {
    let truncated = output.chars().count() > limit;
    if !truncated {
        return (output.to_string(), false);
    }

    let mut content = output.chars().take(limit).collect::<String>();
    let _ = write!(content, "\n... truncated after {limit} characters");
    (content, true)
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

fn build_initialize_response(_protocol_version: ProtocolVersion) -> InitializeResponse {
    InitializeResponse::new(ProtocolVersion::LATEST)
        .agent_capabilities(
            AgentCapabilities::new()
                .load_session(false)
                .prompt_capabilities(PromptCapabilities::new())
                .auth(AgentAuthCapabilities::new()),
        )
        .agent_info(Implementation::new(ADAPTER_NAME, ADAPTER_VERSION))
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

async fn connect_mcp_sessions(
    servers: &[McpServer],
) -> Result<Vec<McpSession>, agent_client_protocol::Error> {
    let mut sessions = Vec::new();

    for server in servers {
        match server {
            McpServer::Stdio(stdio) => sessions.push(connect_mcp_stdio_session(stdio).await?),
            _ => {
                return Err(agent_client_protocol::Error::invalid_params()
                    .data("only stdio MCP servers are supported"));
            }
        }
    }

    Ok(sessions)
}

async fn connect_mcp_stdio_session(
    server: &McpServerStdio,
) -> Result<McpSession, agent_client_protocol::Error> {
    if !server.command.is_absolute() {
        return Err(agent_client_protocol::Error::invalid_params().data(format!(
            "MCP server '{}' command must be absolute",
            server.name
        )));
    }

    let command = TokioCommand::new(&server.command).configure(|command| {
        command.args(&server.args);
        for variable in &server.env {
            command.env(&variable.name, &variable.value);
        }
    });
    let transport = TokioChildProcess::new(command).map_err(|error| {
        agent_client_protocol::Error::invalid_params().data(format!(
            "failed to start MCP server '{}': {error}",
            server.name
        ))
    })?;
    let service = ().serve(transport).await.map_err(|error| {
        agent_client_protocol::Error::invalid_params().data(format!(
            "failed to initialize MCP server '{}': {error}",
            server.name
        ))
    })?;
    let peer = service.peer().clone();
    let tools = peer.list_all_tools().await.map_err(|error| {
        agent_client_protocol::Error::invalid_params().data(format!(
            "failed to list MCP tools for server '{}': {error}",
            server.name
        ))
    })?;
    let mappings = mcp_tool_mappings(&server.name, tools);

    Ok(McpSession {
        name: server.name.clone(),
        tools: mappings,
        peer,
        _service: service,
    })
}

fn mcp_tool_mappings(server_name: &str, tools: Vec<McpTool>) -> Vec<McpToolMapping> {
    tools
        .into_iter()
        .map(|tool| {
            let original_name = tool.name.to_string();
            let exposed_name = mcp_tool_name(server_name, &original_name);
            let description = tool.description.map_or_else(
                || format!("MCP tool '{original_name}' from server '{server_name}'"),
                |description| description.to_string(),
            );
            let definition = ToolDefinition::new(
                exposed_name.clone(),
                description,
                Value::Object(tool.input_schema.as_ref().clone()),
            );

            McpToolMapping {
                exposed_name,
                original_name,
                definition,
            }
        })
        .collect()
}

fn mcp_tool_name(server_name: &str, tool_name: &str) -> String {
    format!(
        "{MCP_TOOL_PREFIX}__{}__{}",
        sanitize_tool_name_part(server_name),
        sanitize_tool_name_part(tool_name)
    )
}

fn sanitize_tool_name_part(value: &str) -> String {
    let mut sanitized = String::new();
    for character in value.chars() {
        if character.is_ascii_alphanumeric() {
            sanitized.push(character.to_ascii_lowercase());
        } else {
            sanitized.push('_');
        }
    }

    let trimmed = sanitized.trim_matches('_');
    if trimmed.is_empty() {
        "unnamed".to_string()
    } else {
        trimmed.to_string()
    }
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

fn session_config_options(session: &SessionRecord) -> Vec<SessionConfigOption> {
    vec![
        SessionConfigOption::select(
            SESSION_CONFIG_MODE_ID,
            "Approval Preset",
            session.mode.mode_id().0,
            session_mode_select_options(),
        )
        .category(SessionConfigOptionCategory::Mode)
        .description("Choose how the adapter requests permission for tools."),
        SessionConfigOption::select(
            SESSION_CONFIG_MODEL_ID,
            "Model",
            session.model.clone(),
            model_select_options(session.model.as_str()),
        )
        .category(SessionConfigOptionCategory::Model)
        .description("Choose which DeepSeek model the adapter should use."),
        SessionConfigOption::select(
            SESSION_CONFIG_REASONING_EFFORT_ID,
            "Reasoning Effort",
            session.reasoning_effort.id(),
            reasoning_effort_select_options(),
        )
        .category(SessionConfigOptionCategory::ThoughtLevel)
        .description("Choose how much DeepSeek thinking effort to request."),
    ]
}

fn session_mode_select_options() -> Vec<SessionConfigSelectOption> {
    default_session_modes()
        .available_modes
        .into_iter()
        .map(|mode| SessionConfigSelectOption::new(mode.id.0, mode.name))
        .collect()
}

fn model_select_options(current_model: &str) -> Vec<SessionConfigSelectOption> {
    let mut options = Vec::new();
    if !is_known_model(current_model) {
        options.push(
            SessionConfigSelectOption::new(current_model.to_string(), current_model.to_string())
                .description("Current model from DEEPSEEK_MODEL."),
        );
    }

    options.extend([
        SessionConfigSelectOption::new(DEEPSEEK_V4_PRO_MODEL_ID, "DeepSeek V4 Pro")
            .description("DeepSeek V4 Pro thinking model."),
        SessionConfigSelectOption::new(DEEPSEEK_V4_FLASH_MODEL_ID, "DeepSeek V4 Flash")
            .description("DeepSeek V4 Flash model."),
    ]);

    options
}

fn reasoning_effort_select_options() -> Vec<SessionConfigSelectOption> {
    [ReasoningEffort::High, ReasoningEffort::Max]
        .into_iter()
        .map(|effort| {
            SessionConfigSelectOption::new(effort.id(), effort.name())
                .description(effort.description())
        })
        .collect()
}

fn validate_session_model(
    session: &SessionRecord,
    model: &str,
) -> Result<(), agent_client_protocol::Error> {
    if is_known_model(model) || model == session.model {
        return Ok(());
    }

    Err(agent_client_protocol::Error::invalid_params()
        .data(format!("unsupported DeepSeek model: {model}")))
}

fn is_known_model(model: &str) -> bool {
    matches!(model, DEEPSEEK_V4_PRO_MODEL_ID | DEEPSEEK_V4_FLASH_MODEL_ID)
}

fn initial_model_from_env() -> String {
    std::env::var("DEEPSEEK_MODEL")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DeepSeekConfig::DEFAULT_MODEL.to_string())
}

fn text_from_prompt(prompt: &[ContentBlock]) -> Result<String, agent_client_protocol::Error> {
    let mut text = String::new();

    for block in prompt {
        match block {
            ContentBlock::Text(content) => text.push_str(&content.text),
            ContentBlock::ResourceLink(link) => text.push_str(&resource_link_prompt_text(link)),
            _ => {
                return Err(agent_client_protocol::Error::invalid_params()
                    .data("only text and resource link prompt blocks are supported"));
            }
        }
    }

    if text.trim().is_empty() {
        return Err(agent_client_protocol::Error::invalid_params()
            .data("prompt must include non-empty text"));
    }

    Ok(text)
}

fn resource_link_prompt_text(link: &agent_client_protocol::schema::ResourceLink) -> String {
    let display_name = link.title.as_deref().unwrap_or(link.name.as_str());
    let mut rendered = String::new();
    rendered.push_str("[resource] ");
    rendered.push_str(display_name);
    rendered.push_str(" <");
    rendered.push_str(&link.uri);
    rendered.push('>');

    if let Some(description) = &link.description {
        rendered.push_str(" - ");
        rendered.push_str(description);
    }

    rendered
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

#[derive(Debug)]
struct AdapterState {
    default_model: String,
    client_capabilities: Option<ClientCapabilities>,
    sessions: HashMap<agent_client_protocol::schema::SessionId, SessionRecord>,
}

#[derive(Debug)]
struct McpSession {
    name: String,
    tools: Vec<McpToolMapping>,
    peer: Peer<RoleClient>,
    _service: RunningService<RoleClient, ()>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct McpToolMapping {
    exposed_name: String,
    original_name: String,
    definition: ToolDefinition,
}

impl AdapterState {
    fn new(default_model: impl Into<String>) -> Self {
        Self {
            default_model: default_model.into(),
            client_capabilities: None,
            sessions: HashMap::new(),
        }
    }
}

impl Default for AdapterState {
    fn default() -> Self {
        Self::new(DeepSeekConfig::DEFAULT_MODEL)
    }
}

#[derive(Debug, Default)]
struct SessionRecord {
    cwd: PathBuf,
    additional_directories: Vec<PathBuf>,
    history: Vec<ChatMessage>,
    active_turn: Option<CancellationToken>,
    mode: PermissionPosture,
    model: String,
    reasoning_effort: ReasoningEffort,
    permission_allow_always: HashSet<String>,
    mcp_sessions: Vec<McpSession>,
}

#[cfg(test)]
mod tests {
    use super::{
        AdapterState, AdapterToolRegistry, Backend, Cli, Command, DevSmokeResult,
        EmptyToolRegistry, MAX_TURN_REQUESTS, McpSession, MockLlmClient, ModelRequestSettings,
        PendingToolCalls, PermissionDecision, PermissionPosture, PermissionRequester,
        ReadTextFileRequester, ReasoningEffort, SESSION_CONFIG_MODEL_ID,
        SESSION_CONFIG_REASONING_EFFORT_ID, ToolCallRequester, ToolContext, ToolExecution,
        ToolRegistry, WriteTextFileRequester, build_dev_agent, build_initialize_response,
        connect_mcp_sessions, edit_file_tool_execution, exercise_permission_gate_smoke,
        glob_tool_execution, grep_tool_execution, handle_authenticate_request,
        handle_cancel_notification, handle_initialize_request, handle_new_session_request,
        handle_prompt_request, handle_set_session_config_option_request,
        handle_set_session_mode_request, list_dir_tool_execution, llm_client_for_backend,
        mcp_tool_mappings, print_dev_smoke_result, read_file_tool_execution,
        request_tool_permission, run_command_tool_execution, run_smoke_flow, serve_with_transport,
        write_file_tool_execution,
    };
    use agent_client_protocol::schema::{McpServer, McpServerStdio};
    use agent_client_protocol::{Agent, Channel, Client};
    use deepseek_acp_adapter::deepseek::{
        ChatMessage, ChatRequest, DeepSeekError, FinishReason, LlmClient, StreamEvent,
        ToolCall as DeepSeekToolCall, ToolCallDelta, ToolDefinition,
    };
    use futures_util::future::BoxFuture;
    use futures_util::stream::{self, BoxStream};
    use rmcp::model::{
        CallToolRequestParams, CallToolResult, Content as McpContent, ListToolsResult,
        PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool as McpTool,
    };
    use rmcp::service::{RequestContext, RoleServer};
    use rmcp::{ServerHandler, ServiceExt};
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};
    use uuid::Uuid;

    use agent_client_protocol::schema::{
        CancelNotification, ClientCapabilities, ContentBlock, FileSystemCapabilities, ImageContent,
        Implementation, InitializeRequest, NewSessionRequest, PermissionOptionKind, PromptRequest,
        ProtocolVersion, ReadTextFileRequest, ReadTextFileResponse, RequestPermissionOutcome,
        RequestPermissionRequest, RequestPermissionResponse, ResourceLink,
        SelectedPermissionOutcome, SessionConfigKind, SessionConfigOption,
        SessionConfigOptionCategory, SessionModeId, SessionNotification, SessionUpdate,
        SetSessionConfigOptionRequest, SetSessionModeRequest, StopReason, ToolKind,
        WriteTextFileRequest, WriteTextFileResponse,
    };
    use clap::Parser;
    use futures_util::StreamExt;
    use serde_json::Value;
    use std::path::PathBuf;
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
            _state: &Arc<Mutex<AdapterState>>,
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
            _state: &'a Arc<Mutex<AdapterState>>,
            _connection: Option<&'a dyn ToolCallRequester>,
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

    #[derive(Debug, Clone)]
    struct EchoMcpServer;

    impl ServerHandler for EchoMcpServer {
        fn get_info(&self) -> ServerInfo {
            ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
        }

        async fn call_tool(
            &self,
            request: CallToolRequestParams,
            _context: RequestContext<RoleServer>,
        ) -> Result<CallToolResult, rmcp::ErrorData> {
            let message = request
                .arguments
                .as_ref()
                .and_then(|arguments| arguments.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("");
            Ok(CallToolResult::success(vec![McpContent::text(format!(
                "echo: {message}"
            ))]))
        }

        async fn list_tools(
            &self,
            _request: Option<PaginatedRequestParams>,
            _context: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, rmcp::ErrorData> {
            Ok(ListToolsResult {
                tools: vec![McpTool::new(
                    "echo",
                    "Echo a provided message",
                    rmcp::model::object(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "message": { "type": "string" }
                        },
                        "required": ["message"]
                    })),
                )],
                ..Default::default()
            })
        }
    }

    async fn connected_echo_mcp_session() -> Result<McpSession, agent_client_protocol::Error> {
        let (server_transport, client_transport) = tokio::io::duplex(4096);
        let server_task = tokio::spawn(async move {
            let running = EchoMcpServer
                .serve(server_transport)
                .await
                .map_err(|error| error.to_string())?;
            running.waiting().await.map_err(|error| error.to_string())?;
            Ok::<(), String>(())
        });
        drop(server_task);

        let service = ().serve(client_transport).await.map_err(|error| {
            agent_client_protocol::Error::internal_error()
                .data(format!("failed to initialize test MCP client: {error}"))
        })?;
        let peer = service.peer().clone();
        let tools = peer.list_all_tools().await.map_err(|error| {
            agent_client_protocol::Error::internal_error()
                .data(format!("failed to list test MCP tools: {error}"))
        })?;

        Ok(McpSession {
            name: "Echo Server".to_string(),
            tools: mcp_tool_mappings("Echo Server", tools),
            peer,
            _service: service,
        })
    }

    struct CountingReadTextFileRequester {
        calls: Arc<Mutex<usize>>,
    }

    impl CountingReadTextFileRequester {
        fn new() -> Self {
            Self {
                calls: Arc::new(Mutex::new(0)),
            }
        }

        fn calls(&self) -> Arc<Mutex<usize>> {
            Arc::clone(&self.calls)
        }
    }

    impl ReadTextFileRequester for CountingReadTextFileRequester {
        fn read_text_file(
            &self,
            _request: ReadTextFileRequest,
        ) -> BoxFuture<'_, Result<ReadTextFileResponse, agent_client_protocol::Error>> {
            Box::pin(async move {
                let mut guard = self
                    .calls
                    .lock()
                    .map_err(agent_client_protocol::Error::into_internal_error)?;
                *guard += 1;
                Ok(ReadTextFileResponse::new("client content"))
            })
        }
    }

    struct Utf8FailingReadTextFileRequester;

    impl ReadTextFileRequester for Utf8FailingReadTextFileRequester {
        fn read_text_file(
            &self,
            _request: ReadTextFileRequest,
        ) -> BoxFuture<'_, Result<ReadTextFileResponse, agent_client_protocol::Error>> {
            Box::pin(async move {
                Err(agent_client_protocol::Error::internal_error()
                    .data("stream did not contain valid UTF-8"))
            })
        }
    }

    struct RecordingWriteTextFileRequester {
        requests: Arc<Mutex<Vec<WriteTextFileRequest>>>,
    }

    impl RecordingWriteTextFileRequester {
        fn new() -> Self {
            Self {
                requests: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn requests(&self) -> Arc<Mutex<Vec<WriteTextFileRequest>>> {
            Arc::clone(&self.requests)
        }
    }

    impl WriteTextFileRequester for RecordingWriteTextFileRequester {
        fn write_text_file(
            &self,
            request: WriteTextFileRequest,
        ) -> BoxFuture<'_, Result<WriteTextFileResponse, agent_client_protocol::Error>> {
            self.requests
                .lock()
                .map(|mut requests| requests.push(request))
                .ok();

            Box::pin(async move { Ok(WriteTextFileResponse::new()) })
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

    fn select_current_value(
        options: &[SessionConfigOption],
        id: &str,
    ) -> Result<String, agent_client_protocol::Error> {
        let option = options
            .iter()
            .find(|option| option.id.0.as_ref() == id)
            .ok_or_else(|| {
                agent_client_protocol::Error::internal_error()
                    .data(format!("missing config option {id}"))
            })?;

        let SessionConfigKind::Select(select) = &option.kind else {
            return Err(agent_client_protocol::Error::internal_error()
                .data(format!("config option {id} is not a select")));
        };

        Ok(select.current_value.0.to_string())
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
        assert_eq!(
            response.agent_info,
            Some(Implementation::new(
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION"),
            ))
        );
        assert!(!response.agent_capabilities.load_session);
        assert!(!response.agent_capabilities.prompt_capabilities.image);
        assert!(!response.agent_capabilities.prompt_capabilities.audio);
        assert!(
            !response
                .agent_capabilities
                .prompt_capabilities
                .embedded_context
        );
        assert!(response.auth_methods.is_empty());
    }

    #[test_log::test]
    fn build_initialize_response_uses_latest_supported_protocol_version()
    -> Result<(), agent_client_protocol::Error> {
        let unsupported_protocol_version = serde_json::from_str::<ProtocolVersion>("2")
            .map_err(agent_client_protocol::Error::into_internal_error)?;

        let response = build_initialize_response(unsupported_protocol_version);

        assert_eq!(response.protocol_version, ProtocolVersion::LATEST);
        Ok(())
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
    fn new_session_advertises_model_and_reasoning_config_options()
    -> Result<(), agent_client_protocol::Error> {
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let response = handle_new_session_request(&state, &NewSessionRequest::new("/tmp"))?;
        let options = response
            .config_options
            .ok_or_else(agent_client_protocol::Error::internal_error)?;

        let model = options
            .iter()
            .find(|option| option.id.0.as_ref() == SESSION_CONFIG_MODEL_ID)
            .ok_or_else(|| {
                agent_client_protocol::Error::internal_error().data("missing model option")
            })?;
        assert_eq!(model.category, Some(SessionConfigOptionCategory::Model));
        assert_eq!(
            select_current_value(&options, SESSION_CONFIG_MODEL_ID)?,
            "deepseek-v4-pro"
        );

        let reasoning = options
            .iter()
            .find(|option| option.id.0.as_ref() == SESSION_CONFIG_REASONING_EFFORT_ID)
            .ok_or_else(|| {
                agent_client_protocol::Error::internal_error()
                    .data("missing reasoning effort option")
            })?;
        assert_eq!(
            reasoning.category,
            Some(SessionConfigOptionCategory::ThoughtLevel)
        );
        assert_eq!(
            select_current_value(&options, SESSION_CONFIG_REASONING_EFFORT_ID)?,
            "high"
        );

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn mcp_stdio_launch_failure_returns_invalid_params() {
        let result = connect_mcp_sessions(&[McpServer::Stdio(McpServerStdio::new(
            "broken",
            "/definitely/not/a/real/mcp-server",
        ))])
        .await;

        assert!(result.is_err());
        let error_text = result
            .err()
            .map_or_else(String::new, |error| format!("{error:?}"));
        assert!(error_text.contains("failed to start MCP server 'broken'"));
    }

    #[test_log::test]
    fn mcp_tool_mappings_prefix_and_preserve_schema() {
        let mappings = mcp_tool_mappings(
            "Test Server",
            vec![McpTool::new(
                "Read File",
                "Read through MCP",
                rmcp::model::object(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" }
                    }
                })),
            )],
        );

        assert_eq!(mappings.len(), 1);
        let mapping = &mappings[0];
        assert_eq!(mapping.exposed_name, "mcp__test_server__read_file");
        assert_eq!(mapping.original_name, "Read File");
        assert_eq!(mapping.definition.name(), "mcp__test_server__read_file");
        assert_eq!(mapping.definition.description(), "Read through MCP");
        assert_eq!(
            mapping.definition.parameters()["properties"]["path"]["type"],
            "string"
        );
    }

    #[test_log::test(tokio::test)]
    async fn adapter_registry_exposes_and_executes_session_mcp_tools()
    -> Result<(), agent_client_protocol::Error> {
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let response = handle_new_session_request(&state, &NewSessionRequest::new("/tmp"))?;
        let mcp_session = connected_echo_mcp_session().await?;
        {
            let mut guard = state
                .lock()
                .map_err(agent_client_protocol::Error::into_internal_error)?;
            let session = guard
                .sessions
                .get_mut(&response.session_id)
                .ok_or_else(|| {
                    agent_client_protocol::Error::internal_error().data("missing session")
                })?;
            session.mcp_sessions.push(mcp_session);
        }

        let context = ToolContext {
            session_id: response.session_id.clone(),
            cwd: PathBuf::from("/tmp"),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let registry = AdapterToolRegistry;
        let definitions = registry.definitions(&context, &state)?;
        assert!(
            definitions
                .iter()
                .any(|definition| definition.name() == "mcp__echo_server__echo")
        );

        let result = registry
            .execute(
                &DeepSeekToolCall::new(
                    "call-mcp",
                    "mcp__echo_server__echo",
                    serde_json::json!({ "message": "hello" }).to_string(),
                ),
                &context,
                &state,
                None,
            )
            .await;

        assert!(result.success);
        assert_eq!(result.content, "echo: hello");

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

    #[test_log::test]
    fn set_config_option_updates_session_model_and_reasoning()
    -> Result<(), agent_client_protocol::Error> {
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let session = handle_new_session_request(&state, &NewSessionRequest::new("/tmp"))?;

        let model_response = handle_set_session_config_option_request(
            &state,
            &SetSessionConfigOptionRequest::new(
                session.session_id.clone(),
                SESSION_CONFIG_MODEL_ID,
                "deepseek-v4-flash",
            ),
        )?;
        assert_eq!(
            select_current_value(&model_response.config_options, SESSION_CONFIG_MODEL_ID)?,
            "deepseek-v4-flash"
        );

        let reasoning_response = handle_set_session_config_option_request(
            &state,
            &SetSessionConfigOptionRequest::new(
                session.session_id.clone(),
                SESSION_CONFIG_REASONING_EFFORT_ID,
                "max",
            ),
        )?;
        assert_eq!(
            select_current_value(
                &reasoning_response.config_options,
                SESSION_CONFIG_REASONING_EFFORT_ID,
            )?,
            "max"
        );

        let guard = state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let stored = guard.sessions.get(&session.session_id).ok_or_else(|| {
            agent_client_protocol::Error::internal_error().data("missing stored session")
        })?;
        assert_eq!(stored.model, "deepseek-v4-flash");
        assert_eq!(stored.reasoning_effort, ReasoningEffort::Max);

        Ok(())
    }

    #[test_log::test]
    fn set_config_option_rejects_unknown_option() -> Result<(), agent_client_protocol::Error> {
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let session = handle_new_session_request(&state, &NewSessionRequest::new("/tmp"))?;

        let Err(error) = handle_set_session_config_option_request(
            &state,
            &SetSessionConfigOptionRequest::new(session.session_id, "unknown", "value"),
        ) else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("expected unknown config option to fail"));
        };

        assert_eq!(error.code, agent_client_protocol::ErrorCode::InvalidParams);

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn prompt_uses_updated_session_model_and_reasoning()
    -> Result<(), agent_client_protocol::Error> {
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let session = handle_new_session_request(&state, &NewSessionRequest::new("/tmp"))?;
        handle_set_session_config_option_request(
            &state,
            &SetSessionConfigOptionRequest::new(
                session.session_id.clone(),
                SESSION_CONFIG_MODEL_ID,
                "deepseek-v4-flash",
            ),
        )?;
        handle_set_session_config_option_request(
            &state,
            &SetSessionConfigOptionRequest::new(
                session.session_id.clone(),
                SESSION_CONFIG_REASONING_EFFORT_ID,
                "max",
            ),
        )?;

        let client = FakeLlmClient::new(vec![Ok(StreamEvent::Finished(FinishReason::EndTurn))]);
        let requests = client.requests();

        let response = handle_prompt_request(
            &state,
            &client,
            &EmptyToolRegistry,
            None,
            PromptRequest::new(session.session_id, vec![ContentBlock::from("hi")]),
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

        // First notification is the plan; skip it.
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
        assert!(matches!(notifications[0].update, SessionUpdate::Plan(_)));
        assert!(matches!(
            notifications[1].update,
            SessionUpdate::ToolCall(_)
        ));
        assert!(matches!(
            notifications[2].update,
            SessionUpdate::ToolCallUpdate(_)
        ));
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
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let session = handle_new_session_request(&state, &NewSessionRequest::new("/tmp"))?;
        let mut streams = (0..MAX_TURN_REQUESTS)
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
            &state,
            &client,
            &registry,
            None,
            PromptRequest::new(session.session_id.clone(), vec![ContentBlock::from("loop")]),
            |_| Ok(()),
        )
        .await?;

        assert_eq!(response.stop_reason, StopReason::MaxTurnRequests);
        let request_guard = requests
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        assert_eq!(request_guard.len(), MAX_TURN_REQUESTS);
        drop(request_guard);

        let guard = state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let record = guard.sessions.get(&session.session_id).ok_or_else(|| {
            agent_client_protocol::Error::internal_error().data("missing session")
        })?;
        assert_eq!(record.history.len(), 1 + (MAX_TURN_REQUESTS * 2));

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
    async fn read_file_tool_rejects_local_non_utf8_before_client_fs()
    -> Result<(), agent_client_protocol::Error> {
        let temp_root =
            std::env::temp_dir().join(format!("deepseek-acp-adapter-non-utf8-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let file_path = temp_root.join("artifact.bin");
        std::fs::write(&file_path, [0xff, 0xfe, 0xfd])
            .map_err(agent_client_protocol::Error::into_internal_error)?;

        let context = ToolContext {
            session_id: agent_client_protocol::schema::SessionId::new("session-non-utf8"),
            cwd: temp_root.clone(),
            additional_directories: Vec::new(),
            client_capabilities: Some(
                ClientCapabilities::new().fs(FileSystemCapabilities::new().read_text_file(true)),
            ),
        };
        let call = DeepSeekToolCall::new(
            "call-non-utf8",
            "read_file",
            serde_json::json!({ "path": "artifact.bin" }).to_string(),
        );
        let requester = CountingReadTextFileRequester::new();
        let calls = requester.calls();

        let result = read_file_tool_execution(
            &call,
            &context,
            Some(&requester as &dyn ReadTextFileRequester),
        )
        .await;

        assert!(!result.success);
        assert!(result.content.contains("only supports UTF-8 text files"));
        assert!(result.content.contains(&file_path.display().to_string()));
        assert_eq!(
            *calls
                .lock()
                .map_err(agent_client_protocol::Error::into_internal_error)?,
            0
        );

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn read_file_tool_sanitizes_client_non_utf8_error()
    -> Result<(), agent_client_protocol::Error> {
        let temp_root = std::env::temp_dir().join(format!(
            "deepseek-acp-adapter-client-utf8-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;

        let context = ToolContext {
            session_id: agent_client_protocol::schema::SessionId::new("session-client-utf8"),
            cwd: temp_root,
            additional_directories: Vec::new(),
            client_capabilities: Some(
                ClientCapabilities::new().fs(FileSystemCapabilities::new().read_text_file(true)),
            ),
        };
        let call = DeepSeekToolCall::new(
            "call-client-utf8",
            "read_file",
            serde_json::json!({ "path": "client-only.bin" }).to_string(),
        );

        let result = read_file_tool_execution(
            &call,
            &context,
            Some(&Utf8FailingReadTextFileRequester as &dyn ReadTextFileRequester),
        )
        .await;

        assert!(!result.success);
        assert!(result.content.contains("only supports UTF-8 text files"));
        assert!(!result.content.contains("Internal error"));
        assert!(
            !result
                .content
                .contains("stream did not contain valid UTF-8")
        );

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn write_file_tool_routes_to_client_fs_write() -> Result<(), agent_client_protocol::Error>
    {
        let temp_root = std::env::temp_dir().join(format!(
            "deepseek-acp-adapter-write-client-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let session = handle_new_session_request(&state, &NewSessionRequest::new(&temp_root))?;
        let context = ToolContext {
            session_id: session.session_id.clone(),
            cwd: temp_root.clone(),
            additional_directories: Vec::new(),
            client_capabilities: Some(
                ClientCapabilities::new().fs(FileSystemCapabilities::new().write_text_file(true)),
            ),
        };

        let permission_requester =
            FakePermissionRequester::new(vec![RequestPermissionResponse::new(
                RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                    super::PERMISSION_ALLOW_ONCE_OPTION_ID,
                )),
            )]);
        let write_requester = RecordingWriteTextFileRequester::new();
        let requests = write_requester.requests();
        let call = DeepSeekToolCall::new(
            "write-client-call",
            "write_file",
            serde_json::json!({
                "path": "note.txt",
                "content": "alpha beta gamma",
            })
            .to_string(),
        );

        let result = write_file_tool_execution(
            &state,
            &call,
            &context,
            Some(&write_requester as &dyn WriteTextFileRequester),
            Some(&permission_requester),
        )
        .await;

        assert!(result.success);
        assert_eq!(result.raw_output["source"], "client");
        assert!(!temp_root.join("note.txt").exists());

        let requests_guard = requests
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let request = requests_guard.first().ok_or_else(|| {
            agent_client_protocol::Error::internal_error().data("missing write_text_file request")
        })?;
        assert_eq!(request.session_id, session.session_id);
        assert_eq!(request.path, temp_root.join("note.txt"));
        assert_eq!(request.content, "alpha beta gamma");

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn edit_file_tool_routes_to_client_fs_read_and_write()
    -> Result<(), agent_client_protocol::Error> {
        let temp_root = std::env::temp_dir().join(format!(
            "deepseek-acp-adapter-edit-client-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let session = handle_new_session_request(&state, &NewSessionRequest::new(&temp_root))?;
        let context = ToolContext {
            session_id: session.session_id.clone(),
            cwd: temp_root.clone(),
            additional_directories: Vec::new(),
            client_capabilities: Some(
                ClientCapabilities::new().fs(FileSystemCapabilities::new()
                    .read_text_file(true)
                    .write_text_file(true)),
            ),
        };

        let permission_requester =
            FakePermissionRequester::new(vec![RequestPermissionResponse::new(
                RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                    super::PERMISSION_ALLOW_ONCE_OPTION_ID,
                )),
            )]);
        let read_requester = CountingReadTextFileRequester::new();
        let read_calls = read_requester.calls();
        let write_requester = RecordingWriteTextFileRequester::new();
        let write_requests = write_requester.requests();
        let call = DeepSeekToolCall::new(
            "edit-client-call",
            "edit_file",
            serde_json::json!({
                "path": "note.txt",
                "old_text": "content",
                "new_text": "buffer",
            })
            .to_string(),
        );

        let result = edit_file_tool_execution(
            &state,
            &call,
            &context,
            Some(&read_requester as &dyn ReadTextFileRequester),
            Some(&write_requester as &dyn WriteTextFileRequester),
            Some(&permission_requester),
        )
        .await;

        assert!(result.success);
        assert_eq!(result.raw_output["read_source"], "client");
        assert_eq!(result.raw_output["write_source"], "client");
        assert!(!temp_root.join("note.txt").exists());
        assert_eq!(
            *read_calls
                .lock()
                .map_err(agent_client_protocol::Error::into_internal_error)?,
            1
        );

        let write_requests_guard = write_requests
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let request = write_requests_guard.first().ok_or_else(|| {
            agent_client_protocol::Error::internal_error().data("missing write_text_file request")
        })?;
        assert_eq!(request.session_id, session.session_id);
        assert_eq!(request.path, temp_root.join("note.txt"));
        assert_eq!(request.content, "client buffer");

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn write_and_edit_file_tools_modify_local_files_after_permission()
    -> Result<(), agent_client_protocol::Error> {
        let temp_root =
            std::env::temp_dir().join(format!("deepseek-acp-adapter-write-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let session = handle_new_session_request(&state, &NewSessionRequest::new(&temp_root))?;
        let context = ToolContext {
            session_id: session.session_id.clone(),
            cwd: temp_root.clone(),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };

        let write_requester = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
            RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                super::PERMISSION_ALLOW_ONCE_OPTION_ID,
            )),
        )]);
        let write_call = DeepSeekToolCall::new(
            "write-call",
            "write_file",
            serde_json::json!({
                "path": "note.txt",
                "content": "alpha beta gamma",
            })
            .to_string(),
        );

        let write_result =
            write_file_tool_execution(&state, &write_call, &context, None, Some(&write_requester))
                .await;

        assert!(write_result.success);
        assert_eq!(write_result.raw_output["source"], "local");
        assert_eq!(
            std::fs::read_to_string(temp_root.join("note.txt"))
                .map_err(agent_client_protocol::Error::into_internal_error)?,
            "alpha beta gamma"
        );

        let edit_requester = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
            RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                super::PERMISSION_ALLOW_ONCE_OPTION_ID,
            )),
        )]);
        let edit_call = DeepSeekToolCall::new(
            "edit-call",
            "edit_file",
            serde_json::json!({
                "path": "note.txt",
                "old_text": "beta",
                "new_text": "delta",
            })
            .to_string(),
        );

        let edit_result = edit_file_tool_execution(
            &state,
            &edit_call,
            &context,
            None,
            None,
            Some(&edit_requester),
        )
        .await;

        assert!(edit_result.success);
        assert_eq!(edit_result.raw_output["read_source"], "local");
        assert_eq!(edit_result.raw_output["write_source"], "local");
        assert_eq!(
            std::fs::read_to_string(temp_root.join("note.txt"))
                .map_err(agent_client_protocol::Error::into_internal_error)?,
            "alpha delta gamma"
        );

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn run_command_tool_executes_in_session_cwd_after_permission()
    -> Result<(), agent_client_protocol::Error> {
        let temp_root =
            std::env::temp_dir().join(format!("deepseek-acp-adapter-command-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let session = handle_new_session_request(&state, &NewSessionRequest::new(&temp_root))?;
        let context = ToolContext {
            session_id: session.session_id.clone(),
            cwd: temp_root,
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let requester = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
            RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                super::PERMISSION_ALLOW_ONCE_OPTION_ID,
            )),
        )]);
        let call = DeepSeekToolCall::new(
            "command-call",
            "run_command",
            serde_json::json!({ "command": "printf shell-ok" }).to_string(),
        );

        let result =
            run_command_tool_execution(&state, &call, &context, Some(&requester), None).await;

        assert!(result.success);
        assert!(result.content.contains("stdout:"));
        assert!(result.content.contains("shell-ok"));
        assert_eq!(result.raw_output["exit_code"], serde_json::json!(0));

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
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let registry = AdapterToolRegistry;

        let list_result = registry
            .execute(
                &DeepSeekToolCall::new(
                    "call-list",
                    "list_dir",
                    serde_json::json!({ "path": "." }).to_string(),
                ),
                &context,
                &state,
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
                &state,
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
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let registry = AdapterToolRegistry;

        let result = registry
            .execute(
                &DeepSeekToolCall::new(
                    "call-grep",
                    "grep",
                    serde_json::json!({ "pattern": "needle" }).to_string(),
                ),
                &context,
                &state,
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

        let resource_link_prompt = vec![ContentBlock::ResourceLink(ResourceLink::new(
            "docs",
            "file:///docs/reference.md",
        ))];
        assert_eq!(
            super::text_from_prompt(&resource_link_prompt)?,
            "[resource] docs <file:///docs/reference.md>"
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
                .contains("only text and resource link prompt blocks are supported")
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
        let state = Arc::new(Mutex::new(AdapterState::default()));

        let empty_result = EmptyToolRegistry
            .execute(
                &DeepSeekToolCall::new("empty", "anything", "{}"),
                &context,
                &state,
                None,
            )
            .await;
        assert!(!empty_result.success);
        assert!(empty_result.content.contains("unknown tool: anything"));

        let read_only_result = AdapterToolRegistry
            .execute(
                &DeepSeekToolCall::new("read-only", "bogus", "{}"),
                &context,
                &state,
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
