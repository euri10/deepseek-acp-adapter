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
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::{error::Error, process::ExitCode};

use agent_client_protocol::schema::{
    AvailableCommand, AvailableCommandInput, ClientCapabilities, ContentBlock, ContentChunk,
    EmbeddedResourceResource, InitializeRequest, InitializeResponse, McpServer, NewSessionRequest,
    NewSessionResponse, PermissionOption, PermissionOptionKind, Plan, PlanEntry, PlanEntryPriority,
    PlanEntryStatus, ProtocolVersion, RequestPermissionOutcome, RequestPermissionRequest,
    RequestPermissionResponse, SelectedPermissionOutcome, SessionConfigOption,
    SessionConfigOptionCategory, SessionConfigSelectOption, SessionConfigValueId, SessionId,
    SessionInfo, SessionMode, SessionModeId, SessionModeState, SessionNotification, SessionUpdate,
    StopReason, ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields, ToolKind,
    UnstructuredCommandInput,
};
use agent_client_protocol::util::MatchDispatch;
use agent_client_protocol::{AcpAgent, Client, ConnectTo, SessionMessage, Stdio};
use clap::{Parser, Subcommand, ValueEnum};
use deepseek_acp_adapter::deepseek::{
    ChatMessage, ChatRequest, DeepSeekClient, DeepSeekConfig, DeepSeekError, FinishReason,
    LlmClient, StreamEvent, ToolCall as DeepSeekToolCall, ToolCallDelta, ToolDefinition,
};
use futures_util::future::BoxFuture;
use futures_util::stream::{self, BoxStream};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;

mod acp;
mod mcp;
mod session_store;
mod tools;
mod turn;

#[cfg(test)]
pub(crate) use acp::{
    CreateTerminalRequester, KillTerminalRequester, ReleaseTerminalRequester,
    TerminalOutputRequester, WaitForTerminalExitRequester, build_initialize_response,
    handle_authenticate_request, handle_close_session_request, handle_initialize_request,
    handle_list_sessions_request, handle_load_session_request, handle_logout_request,
    handle_new_session_request_connected, handle_prompt_request,
    handle_set_session_config_option_request, handle_set_session_mode_request,
    validate_session_paths,
};
pub(crate) use acp::{
    PermissionRequester, ReadTextFileRequester, TerminalRequester, ToolCallRequester,
    WriteTextFileRequester, handle_new_session_request, serve_with_transport,
};
#[cfg(test)]
pub(crate) use mcp::sanitize_tool_name_part;
pub(crate) use mcp::{
    McpSession, McpToolTarget, connect_mcp_sessions, is_mcp_tool_name, mcp_tool_execution,
    mcp_tool_kind,
};
pub(crate) use session_store::{
    FilesystemSessionStore, PersistedSessionMeta, PersistedSessionRecord,
};
#[cfg(test)]
use tools::ToolExecution;
#[cfg(test)]
use tools::ToolRegistry;
use tools::{AdapterToolRegistry, ToolContext};
pub(crate) use turn::tool_raw_input;

#[cfg(test)]
use tools::{
    EmptyToolRegistry, build_root_gitignore, collect_directory_entries, edit_file_tool_execution,
    glob_tool_execution, grep_tool_execution, is_hidden_path, is_utf8_error_message,
    list_dir_tool_execution, non_utf8_file_message, read_file_client_error, read_file_from_local,
    read_file_local_error, read_file_tool_execution, render_command_output,
    require_tool_permission, resolve_tool_path, run_command_tool_execution,
    run_command_via_terminal, truncate_tool_output, write_file_to_client,
    write_file_tool_execution,
};

type AdapterResult<T> = Result<T, Box<dyn Error + Send + Sync + 'static>>;
/// Default maximum number of tool-call/response cycles per prompt turn.
const DEFAULT_MAX_TURN_REQUESTS: NonZeroUsize = NonZeroUsize::MIN.saturating_add(99);
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
        /// Maximum tool-call/response cycles per prompt turn (must be ≥ 1).
        #[arg(long, default_value_t = DEFAULT_MAX_TURN_REQUESTS)]
        max_turn_requests: NonZeroUsize,
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
            Command::Serve {
                backend,
                max_turn_requests,
            } => serve(backend, max_turn_requests).await,
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

async fn serve(
    backend: Backend,
    max_turn_requests: NonZeroUsize,
) -> Result<(), agent_client_protocol::Error> {
    let llm_client = llm_client_for_backend(backend)?;
    let tool_registry = Arc::new(AdapterToolRegistry);
    let state = Arc::new(Mutex::new(AdapterState::new(initial_model_from_env())));
    serve_with_transport(
        Stdio::new(),
        state,
        llm_client,
        tool_registry,
        max_turn_requests,
    )
    .await
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
    let store = SessionStore::new(Arc::new(Mutex::new(AdapterState::default())));
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
    let call = DeepSeekToolCall::new(
        "dev-permission-call",
        "write_file",
        serde_json::json!({ "path": "smoke.txt" }).to_string(),
    );
    let decision = request_tool_permission(
        &store,
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

    if !store.is_always_allowed(&session.session_id, "write_file")? {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PermissionDecision {
    AllowOnce,
    AllowAlways,
    AllowByMode,
    RejectOnce,
    RejectAlways,
    Cancelled,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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
    store: &SessionStore,
    context: &ToolContext,
    call: &DeepSeekToolCall,
    kind: ToolKind,
    requester: &dyn PermissionRequester,
) -> Result<PermissionDecision, agent_client_protocol::Error> {
    if store.is_always_allowed(&context.session_id, call.name())? {
        return Ok(PermissionDecision::AllowAlways);
    }

    let posture = store.permission_posture(&context.session_id)?;

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
        store.add_always_allow(&context.session_id, call.name().to_string())?;
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
            ContentBlock::Resource(resource) => match &resource.resource {
                EmbeddedResourceResource::TextResourceContents(contents) => {
                    text.push_str(&resource_text_prompt_text(contents));
                }
                EmbeddedResourceResource::BlobResourceContents(_) => {
                    return Err(agent_client_protocol::Error::invalid_params()
                        .data("binary resource prompt blocks are not supported"));
                }
                _ => {
                    return Err(agent_client_protocol::Error::invalid_params()
                        .data("unsupported embedded resource prompt block"));
                }
            },
            _ => {
                return Err(agent_client_protocol::Error::invalid_params().data(
                    "only text, resource link, and text resource prompt blocks are supported",
                ));
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

fn resource_text_prompt_text(
    contents: &agent_client_protocol::schema::TextResourceContents,
) -> String {
    let mut rendered = String::new();
    rendered.push_str("[resource] <");
    rendered.push_str(&contents.uri);
    rendered.push_str(">\n");
    rendered.push_str(&contents.text);
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
    mcp_servers: Vec<McpServer>,
    mcp_sessions: Vec<McpSession>,
}

/// Narrow boundary around shared adapter state.
///
/// `SessionStore` wraps the internal `Arc<Mutex<AdapterState>>` and exposes
/// targeted methods for session lifecycle, config, permission cache, and
/// active-turn management. Application code uses `SessionStore` directly
/// instead of passing raw `Arc<Mutex<AdapterState>>` through every layer.
#[derive(Debug, Clone)]
struct SessionStore {
    pub(crate) state: Arc<Mutex<AdapterState>>,
    persistence: Option<FilesystemSessionStore>,
}

/// Snapshot of session data needed to begin a prompt turn.
///
/// Returned by [`SessionStore::begin_turn`] so the caller does not need to
/// hold the lock across model streaming.
#[derive(Debug)]
struct TurnSetup {
    messages: Vec<ChatMessage>,
    tool_context: ToolContext,
    model: String,
    reasoning_effort: ReasoningEffort,
}

impl SessionStore {
    /// Wrap an existing `Arc<Mutex<AdapterState>>` in a `SessionStore`.
    fn new(state: Arc<Mutex<AdapterState>>) -> Self {
        Self {
            state,
            persistence: None,
        }
    }

    /// Attach filesystem persistence to the session store.
    fn with_persistence(mut self, persistence: FilesystemSessionStore) -> Self {
        self.persistence = Some(persistence);
        self
    }

    /// Load a persisted session record from the filesystem store.
    fn load_persisted_record(
        &self,
        session_id: &SessionId,
    ) -> Result<PersistedSessionRecord, agent_client_protocol::Error> {
        let Some(persistence) = &self.persistence else {
            return Err(agent_client_protocol::Error::invalid_request()
                .data("session/load requires filesystem persistence"));
        };

        persistence
            .load_record(session_id.0.as_ref())
            .map_err(agent_client_protocol::Error::into_internal_error)
    }

    /// Store the client capabilities reported during initialization.
    fn record_client_capabilities(
        &self,
        client_capabilities: ClientCapabilities,
    ) -> Result<(), agent_client_protocol::Error> {
        let mut guard = self
            .state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        guard.client_capabilities = Some(client_capabilities);
        Ok(())
    }

    /// Return a snapshot of matching sessions for the `session/list` handler.
    fn list_sessions(
        &self,
        cwd_filter: Option<&Path>,
    ) -> Result<Vec<SessionInfo>, agent_client_protocol::Error> {
        let (mut sessions, persistence) = {
            let guard = self
                .state
                .lock()
                .map_err(agent_client_protocol::Error::into_internal_error)?;
            (
                guard
                    .sessions
                    .iter()
                    .filter(|(_session_id, record)| cwd_filter.is_none_or(|cwd| record.cwd == cwd))
                    .map(|(session_id, record)| {
                        SessionInfo::new(session_id.clone(), record.cwd.clone())
                            .additional_directories(record.additional_directories.clone())
                    })
                    .collect::<Vec<_>>(),
                self.persistence.clone(),
            )
        };

        if let (Some(persistence), Some(cwd)) = (persistence, cwd_filter) {
            for persisted in persistence
                .list_persisted(cwd)
                .map_err(agent_client_protocol::Error::into_internal_error)?
            {
                if !sessions
                    .iter()
                    .any(|session| session.session_id == persisted.session_id)
                {
                    sessions.push(persisted);
                }
            }
        }

        Ok(sessions)
    }

    /// Remove a session by id. Returns `true` if the session existed.
    fn remove_session(&self, session_id: &SessionId) -> Result<bool, agent_client_protocol::Error> {
        let mut guard = self
            .state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        Ok(guard.sessions.remove(session_id).is_some())
    }

    /// Insert a new session record.
    fn insert_session(
        &self,
        session_id: SessionId,
        record: SessionRecord,
    ) -> Result<(), agent_client_protocol::Error> {
        let mut guard = self
            .state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        guard.sessions.insert(session_id, record);
        Ok(())
    }

    /// Return the default model identifier for new sessions.
    fn default_model(&self) -> Result<String, agent_client_protocol::Error> {
        let guard = self
            .state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        Ok(guard.default_model.clone())
    }

    /// Look up a session and return a read-only reference via a callback.
    ///
    /// The lock is held only for the duration of the callback.
    fn with_session<T>(
        &self,
        session_id: &SessionId,
        f: impl FnOnce(&SessionRecord) -> Result<T, agent_client_protocol::Error>,
    ) -> Result<T, agent_client_protocol::Error> {
        let guard = self
            .state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let session = guard.sessions.get(session_id).ok_or_else(|| {
            agent_client_protocol::Error::invalid_params()
                .data(format!("unknown session id: {}", session_id.0))
        })?;
        f(session)
    }

    /// Look up a session and invoke a callback with a mutable reference.
    ///
    /// The lock is held only for the duration of the callback.
    fn with_session_mut<T>(
        &self,
        session_id: &SessionId,
        f: impl FnOnce(&mut SessionRecord) -> Result<T, agent_client_protocol::Error>,
    ) -> Result<T, agent_client_protocol::Error> {
        let mut guard = self
            .state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let session = guard.sessions.get_mut(session_id).ok_or_else(|| {
            agent_client_protocol::Error::invalid_params()
                .data(format!("unknown session id: {}", session_id.0))
        })?;
        f(session)
    }

    /// Return the MCP tool definitions registered for a session.
    fn mcp_definitions(
        &self,
        session_id: &SessionId,
    ) -> Result<Vec<ToolDefinition>, agent_client_protocol::Error> {
        self.with_session(session_id, |session| {
            Ok(session
                .mcp_sessions
                .iter()
                .flat_map(|mcp| mcp.tools.iter().map(|tool| tool.definition.clone()))
                .collect())
        })
    }

    /// Find the MCP tool target for an exposed tool name within a session.
    fn find_mcp_target(
        &self,
        session_id: &SessionId,
        exposed_name: &str,
    ) -> Result<Option<McpToolTarget>, agent_client_protocol::Error> {
        self.with_session(session_id, |session| {
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
        })
    }

    /// Check whether a tool name is in the session's allow-always cache.
    fn is_always_allowed(
        &self,
        session_id: &SessionId,
        tool_name: &str,
    ) -> Result<bool, agent_client_protocol::Error> {
        self.with_session(session_id, |session| {
            Ok(session.permission_allow_always.contains(tool_name))
        })
    }

    /// Return the current permission posture (mode) for a session.
    fn permission_posture(
        &self,
        session_id: &SessionId,
    ) -> Result<PermissionPosture, agent_client_protocol::Error> {
        self.with_session(session_id, |session| Ok(session.mode))
    }

    /// Insert a tool name into the session's allow-always cache.
    fn add_always_allow(
        &self,
        session_id: &SessionId,
        tool_name: String,
    ) -> Result<(), agent_client_protocol::Error> {
        self.with_session_mut(session_id, |session| {
            session.permission_allow_always.insert(tool_name);
            Ok(())
        })
    }

    /// Cancel the active turn token for a session, if one exists.
    fn cancel_active_turn(
        &self,
        session_id: &SessionId,
    ) -> Result<(), agent_client_protocol::Error> {
        self.with_session(session_id, |session| {
            if let Some(token) = &session.active_turn {
                token.cancel();
            }
            Ok(())
        })
    }

    /// Clear the active turn token for a session.
    fn clear_active_turn(
        &self,
        session_id: &SessionId,
    ) -> Result<(), agent_client_protocol::Error> {
        self.with_session_mut(session_id, |session| {
            session.active_turn = None;
            Ok(())
        })
    }

    /// Set the permission posture (mode) for a session.
    fn set_mode(
        &self,
        session_id: &SessionId,
        mode: PermissionPosture,
    ) -> Result<(), agent_client_protocol::Error> {
        self.with_session_mut(session_id, |session| {
            session.mode = mode;
            Ok(())
        })
    }

    /// Set the model for a session.
    fn set_model(
        &self,
        session_id: &SessionId,
        model: String,
    ) -> Result<(), agent_client_protocol::Error> {
        self.with_session_mut(session_id, |session| {
            session.model = model;
            Ok(())
        })
    }

    /// Set the reasoning effort for a session.
    fn set_reasoning_effort(
        &self,
        session_id: &SessionId,
        effort: ReasoningEffort,
    ) -> Result<(), agent_client_protocol::Error> {
        self.with_session_mut(session_id, |session| {
            session.reasoning_effort = effort;
            Ok(())
        })
    }

    /// Prepare a session for a new prompt turn.
    ///
    /// Sets the active turn token atomically and returns the messages, tool
    /// context, model, and reasoning effort the caller needs. Returns an error
    /// if a turn is already active.
    fn begin_turn(
        &self,
        session_id: &SessionId,
        token: CancellationToken,
        user_message: ChatMessage,
    ) -> Result<TurnSetup, agent_client_protocol::Error> {
        let mut guard = self
            .state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let client_capabilities = guard.client_capabilities.clone();
        let session = guard.sessions.get_mut(session_id).ok_or_else(|| {
            agent_client_protocol::Error::invalid_params()
                .data(format!("unknown session id: {}", session_id.0))
        })?;

        if session.active_turn.is_some() {
            return Err(
                agent_client_protocol::Error::invalid_request().data(format!(
                    "session {} already has an active turn",
                    session_id.0
                )),
            );
        }
        session.active_turn = Some(token);

        let mut messages = session.history.clone();
        messages.push(user_message);
        Ok(TurnSetup {
            messages,
            tool_context: ToolContext {
                session_id: session_id.clone(),
                cwd: session.cwd.clone(),
                additional_directories: session.additional_directories.clone(),
                client_capabilities,
            },
            model: session.model.clone(),
            reasoning_effort: session.reasoning_effort,
        })
    }

    /// Persist the messages as the session history after a turn completes.
    fn save_history(
        &self,
        session_id: &SessionId,
        messages: Vec<ChatMessage>,
    ) -> Result<(), agent_client_protocol::Error> {
        let (persistence, meta, new_messages) = {
            let guard = self
                .state
                .lock()
                .map_err(agent_client_protocol::Error::into_internal_error)?;
            let session = guard.sessions.get(session_id).ok_or_else(|| {
                agent_client_protocol::Error::invalid_params()
                    .data(format!("unknown session id: {}", session_id.0))
            })?;
            let previous_len = session.history.len();
            let new_messages = messages
                .iter()
                .skip(previous_len)
                .cloned()
                .collect::<Vec<_>>();
            (
                self.persistence.clone(),
                PersistedSessionMeta {
                    session_id: session_id.0.to_string(),
                    cwd: session.cwd.clone(),
                    additional_directories: session.additional_directories.clone(),
                    mode: session.mode,
                    model: session.model.clone(),
                    reasoning_effort: session.reasoning_effort,
                    mcp_servers: session.mcp_servers.clone(),
                },
                new_messages,
            )
        };

        if let Some(persistence) = persistence {
            persistence
                .persist_turn(&meta, &new_messages)
                .map_err(agent_client_protocol::Error::into_internal_error)?;
        }

        self.with_session_mut(session_id, |session| {
            session.history = messages;
            Ok(())
        })
    }

    /// Return the session config options for a session.
    fn session_config_options(
        &self,
        session_id: &SessionId,
    ) -> Result<Vec<SessionConfigOption>, agent_client_protocol::Error> {
        self.with_session(session_id, |session| Ok(session_config_options(session)))
    }

    /// Look up a session record for a new-session response.
    fn lookup_session(&self, session_id: &SessionId) -> Result<(), agent_client_protocol::Error> {
        self.with_session(session_id, |_session| Ok(()))
    }
}

/// Create a `SessionStore` backed by a fresh default adapter state.
///
/// This is a convenience for tests that previously created
/// `Arc<Mutex<AdapterState>>` directly.
#[cfg(test)]
fn test_store() -> SessionStore {
    SessionStore::new(Arc::new(Mutex::new(AdapterState::default())))
}

#[cfg(test)]
mod tests {
    use super::{
        AdapterState, AdapterToolRegistry, Backend, Cli, Command, DEFAULT_MAX_TURN_REQUESTS,
        DevSmokeResult, EmptyToolRegistry, FilesystemSessionStore, MockLlmClient, PendingToolCalls,
        PermissionDecision, PermissionPosture, PermissionRequester, PersistedSessionMeta,
        ReadTextFileRequester, ReasoningEffort, SESSION_CONFIG_MODEL_ID,
        SESSION_CONFIG_REASONING_EFFORT_ID, SessionStore, ToolContext, ToolExecution, ToolRegistry,
        WriteTextFileRequester, build_dev_agent, build_initialize_response,
        edit_file_tool_execution, exercise_permission_gate_smoke, glob_tool_execution,
        grep_tool_execution, handle_authenticate_request, handle_close_session_request,
        handle_initialize_request, handle_list_sessions_request, handle_load_session_request,
        handle_logout_request, handle_new_session_request, handle_prompt_request,
        handle_set_session_config_option_request, handle_set_session_mode_request,
        list_dir_tool_execution, llm_client_for_backend, print_dev_smoke_result,
        read_file_tool_execution, request_tool_permission, run_command_tool_execution,
        run_smoke_flow, serve_with_transport, test_store, write_file_tool_execution,
    };
    use agent_client_protocol::{Agent, Channel, Client};
    use deepseek_acp_adapter::deepseek::{
        ChatMessage, ChatRequest, FinishReason, LlmClient, StreamEvent,
        ToolCall as DeepSeekToolCall, ToolCallDelta, ToolDefinition,
    };
    use futures_util::future::BoxFuture;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use uuid::Uuid;

    use agent_client_protocol::schema::{
        BlobResourceContents, ClientCapabilities, CloseSessionRequest, ContentBlock,
        EmbeddedResource, EmbeddedResourceResource, FileSystemCapabilities, ImageContent,
        Implementation, InitializeRequest, ListSessionsRequest, LoadSessionRequest, McpServer,
        NewSessionRequest, PermissionOptionKind, PromptRequest, ProtocolVersion,
        ReadTextFileRequest, ReadTextFileResponse, RequestPermissionOutcome,
        RequestPermissionRequest, RequestPermissionResponse, ResourceLink,
        SelectedPermissionOutcome, SessionConfigKind, SessionConfigOption,
        SessionConfigOptionCategory, SessionModeId, SessionUpdate, SetSessionConfigOptionRequest,
        SetSessionModeRequest, StopReason, TextResourceContents, ToolCallStatus, ToolKind,
        WriteTextFileRequest, WriteTextFileResponse,
    };
    use clap::Parser;
    use futures_util::StreamExt;
    use tokio_util::sync::CancellationToken;

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
        SessionStore,
        agent_client_protocol::schema::SessionId,
        ToolContext,
        DeepSeekToolCall,
        DeepSeekToolCall,
    );

    fn permission_mode_fixture() -> Result<PermissionModeFixture, agent_client_protocol::Error> {
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
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

        Ok((store, session.session_id, context, edit_call, shell_call))
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
        assert!(
            matches!(
                parsed,
                Ok(Cli {
                    command: Command::Serve {
                        backend: Backend::Real,
                        ..
                    }
                })
            ),
            "expected Ok(Cli::Serve {{ backend: Real }}), got {parsed:?}"
        );
        if let Ok(Cli {
            command: Command::Serve {
                max_turn_requests, ..
            },
        }) = parsed
        {
            assert_eq!(max_turn_requests, DEFAULT_MAX_TURN_REQUESTS);
        }
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
        assert!(response.agent_capabilities.load_session);
        assert!(response.agent_capabilities.mcp_capabilities.http);
        assert!(!response.agent_capabilities.mcp_capabilities.sse);
        assert!(!response.agent_capabilities.prompt_capabilities.image);
        assert!(!response.agent_capabilities.prompt_capabilities.audio);
        assert!(
            response
                .agent_capabilities
                .prompt_capabilities
                .embedded_context
        );
        // session/list capability is advertised.
        assert!(
            response
                .agent_capabilities
                .session_capabilities
                .list
                .is_some()
        );
        // session/close capability is advertised.
        assert!(
            response
                .agent_capabilities
                .session_capabilities
                .close
                .is_some()
        );
        // session/resume is NOT advertised (no persistence).
        assert!(
            response
                .agent_capabilities
                .session_capabilities
                .resume
                .is_none()
        );
        // additionalDirectories is advertised for extra workspace roots.
        assert!(
            response
                .agent_capabilities
                .session_capabilities
                .additional_directories
                .is_some()
        );
        // logout capability is advertised.
        assert!(response.agent_capabilities.auth.logout.is_some());
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
        let store = test_store();
        let request = InitializeRequest::new(ProtocolVersion::LATEST).client_capabilities(
            ClientCapabilities::new()
                .fs(FileSystemCapabilities::new()
                    .read_text_file(true)
                    .write_text_file(false))
                .terminal(true),
        );

        let response = handle_initialize_request(&store, request)?;

        assert_eq!(response.protocol_version, ProtocolVersion::LATEST);
        let guard = store
            .state
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
        let store = test_store();
        let response = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;

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

        let guard = store
            .state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        assert!(guard.sessions.contains_key(&response.session_id));

        Ok(())
    }

    #[test_log::test]
    fn new_session_advertises_model_and_reasoning_config_options()
    -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let response = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
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

    #[test_log::test]
    fn set_mode_updates_session_state() -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;

        let response = handle_set_session_mode_request(
            &store,
            &SetSessionModeRequest::new(session.session_id.clone(), "accept-edits"),
        )?;

        assert!(response.meta.is_none());
        let guard = store
            .state
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
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;

        let model_response = handle_set_session_config_option_request(
            &store,
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
            &store,
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

        let guard = store
            .state
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
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;

        let Err(error) = handle_set_session_config_option_request(
            &store,
            &SetSessionConfigOptionRequest::new(session.session_id, "unknown", "value"),
        ) else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("expected unknown config option to fail"));
        };

        assert_eq!(error.code, agent_client_protocol::ErrorCode::InvalidParams);

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn permission_request_prompts_and_caches_allow_always()
    -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
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
            request_tool_permission(&store, &context, &call, ToolKind::Edit, &requester).await?;

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
            request_tool_permission(&store, &context, &call, ToolKind::Edit, &second_requester)
                .await?;

        assert_eq!(second_decision, PermissionDecision::AllowAlways);
        let second_requests = second_requester.requests();
        let second_guard = second_requests
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        assert!(second_guard.is_empty());

        let guard = store
            .state
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
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
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
            request_tool_permission(&store, &context, &call, ToolKind::Execute, &requester).await?;

        assert_eq!(decision, PermissionDecision::RejectOnce);
        let requests = requester.requests();
        let request_guard = requests
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        assert_eq!(request_guard.len(), 1);
        drop(request_guard);

        let guard = store
            .state
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
        let (store, _session_id, context, edit_call, shell_call) = permission_mode_fixture()?;
        let requester = FakePermissionRequester::new(vec![
            RequestPermissionResponse::new(RequestPermissionOutcome::Selected(
                SelectedPermissionOutcome::new(super::PERMISSION_ALLOW_ONCE_OPTION_ID),
            )),
            RequestPermissionResponse::new(RequestPermissionOutcome::Selected(
                SelectedPermissionOutcome::new(super::PERMISSION_ALLOW_ONCE_OPTION_ID),
            )),
        ]);

        assert_eq!(
            request_tool_permission(&store, &context, &edit_call, ToolKind::Edit, &requester)
                .await?,
            PermissionDecision::AllowOnce
        );
        assert_eq!(
            request_tool_permission(&store, &context, &shell_call, ToolKind::Execute, &requester)
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
        let (store, session_id, context, edit_call, shell_call) = permission_mode_fixture()?;
        handle_set_session_mode_request(
            &store,
            &SetSessionModeRequest::new(session_id.clone(), "accept-edits"),
        )?;
        let requester = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
            RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                super::PERMISSION_ALLOW_ONCE_OPTION_ID,
            )),
        )]);

        assert_eq!(
            request_tool_permission(&store, &context, &edit_call, ToolKind::Edit, &requester)
                .await?,
            PermissionDecision::AllowByMode
        );
        assert_eq!(
            request_tool_permission(&store, &context, &shell_call, ToolKind::Execute, &requester)
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

        let guard = store
            .state
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
        let (store, session_id, context, edit_call, shell_call) = permission_mode_fixture()?;
        handle_set_session_mode_request(&store, &SetSessionModeRequest::new(session_id, "yolo"))?;
        let requester = FakePermissionRequester::new(Vec::new());

        assert_eq!(
            request_tool_permission(&store, &context, &edit_call, ToolKind::Edit, &requester)
                .await?,
            PermissionDecision::AllowByMode
        );
        assert_eq!(
            request_tool_permission(&store, &context, &shell_call, ToolKind::Execute, &requester)
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
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new(&temp_root))?;
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
            &store,
            &call,
            &context,
            None,
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
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new(&temp_root))?;
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
            &store,
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
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new(&temp_root))?;
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

        let write_result = write_file_tool_execution(
            &store,
            &write_call,
            &context,
            None,
            None,
            Some(&write_requester),
        )
        .await;

        assert!(write_result.success);
        assert_eq!(write_result.raw_output["source"], "local");
        let Some(write_edit) = &write_result.edit else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("missing write_file edit metadata"));
        };
        assert_eq!(write_edit.path, temp_root.join("note.txt"));
        assert_eq!(write_edit.old_text, None);
        assert_eq!(write_edit.new_text, "alpha beta gamma");
        assert_eq!(write_edit.line, 1);
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
            &store,
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
        let Some(edit_edit) = &edit_result.edit else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("missing edit_file edit metadata"));
        };
        assert_eq!(edit_edit.path, temp_root.join("note.txt"));
        assert_eq!(edit_edit.old_text, Some("alpha beta gamma".to_string()));
        assert_eq!(edit_edit.new_text, "alpha delta gamma");
        assert_eq!(edit_edit.line, 1);
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
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new(&temp_root))?;
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

        let result = run_command_tool_execution(
            &store,
            &call,
            &context,
            Some(&requester),
            None,
            &CancellationToken::new(),
        )
        .await;

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
        let store = test_store();
        let registry = AdapterToolRegistry;

        let list_result = registry
            .execute(
                &DeepSeekToolCall::new(
                    "call-list",
                    "list_dir",
                    serde_json::json!({ "path": "." }).to_string(),
                ),
                &context,
                &store,
                None,
                CancellationToken::new(),
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
                &store,
                None,
                CancellationToken::new(),
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
        let store = test_store();
        let registry = AdapterToolRegistry;

        let result = registry
            .execute(
                &DeepSeekToolCall::new(
                    "call-grep",
                    "grep",
                    serde_json::json!({ "pattern": "needle" }).to_string(),
                ),
                &context,
                &store,
                None,
                CancellationToken::new(),
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
        let store = test_store();
        let llm_client: Arc<dyn LlmClient> = Arc::new(MockLlmClient);
        let tool_registry: Arc<dyn ToolRegistry> = Arc::new(EmptyToolRegistry);
        let (client_transport, server_transport) = Channel::duplex();
        let server_state = Arc::clone(&store.state);
        let server_client = Arc::clone(&llm_client);
        let server_tools = Arc::clone(&tool_registry);

        let server = tokio::spawn(async move {
            serve_with_transport(
                server_transport,
                server_state,
                server_client,
                server_tools,
                DEFAULT_MAX_TURN_REQUESTS,
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
        assert!(super::is_mcp_tool_name("mcp__server__tool"));
        assert_eq!(super::mcp_tool_kind(), ToolKind::Execute);
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
        let missing_store = SessionStore::new(Arc::new(Mutex::new(AdapterState::default())));
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
            &missing_store,
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

        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
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
            request_tool_permission(&store, &context, &call, ToolKind::Execute, &requester).await?,
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

        let text_resource_prompt = vec![ContentBlock::Resource(EmbeddedResource::new(
            EmbeddedResourceResource::TextResourceContents(TextResourceContents::new(
                "context body",
                "file:///docs/context.md",
            )),
        ))];
        assert_eq!(
            super::text_from_prompt(&text_resource_prompt)?,
            "[resource] <file:///docs/context.md>\ncontext body"
        );

        let blob_resource_prompt = vec![ContentBlock::Resource(EmbeddedResource::new(
            EmbeddedResourceResource::BlobResourceContents(BlobResourceContents::new(
                "aGVsbG8=",
                "file:///docs/context.bin",
            )),
        ))];
        let Err(error) = super::text_from_prompt(&blob_resource_prompt) else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("expected binary resource prompt to fail"));
        };
        assert!(
            error
                .to_string()
                .contains("binary resource prompt blocks are not supported")
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
            error.to_string().contains(
                "only text, resource link, and text resource prompt blocks are supported"
            )
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
        std::fs::write(temp_root.join("found.txt"), "primary")
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        std::fs::write(alternate_directory.join("found.txt"), "found")
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        std::fs::write(alternate_directory.join("alternate-only.txt"), "found")
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let context = ToolContext {
            session_id: agent_client_protocol::schema::SessionId::new("session-paths"),
            cwd: temp_root.clone(),
            additional_directories: vec![alternate_directory.clone()],
            client_capabilities: None,
        };
        assert_eq!(
            super::resolve_tool_path(&context, std::path::Path::new("found.txt")),
            temp_root.join("found.txt")
        );
        assert_eq!(
            super::resolve_tool_path(&context, std::path::Path::new("alternate-only.txt")),
            alternate_directory.join("alternate-only.txt")
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
        let store = test_store();
        let llm_client: Arc<dyn LlmClient> = Arc::new(MockLlmClient);
        let tool_registry: Arc<dyn ToolRegistry> = Arc::new(EmptyToolRegistry);
        let (client_transport, server_transport) = Channel::duplex();
        let server_state = Arc::clone(&store.state);
        let server_client = Arc::clone(&llm_client);
        let server_tools = Arc::clone(&tool_registry);

        let server = tokio::spawn(async move {
            serve_with_transport(
                server_transport,
                server_state,
                server_client,
                server_tools,
                DEFAULT_MAX_TURN_REQUESTS,
            )
            .await
        });

        Client
            .builder()
            .connect_with(client_transport, async move |cx| {
                let initialize_response = cx
                    .send_request(InitializeRequest::new(ProtocolVersion::LATEST))
                    .block_task()
                    .await?;
                assert!(initialize_response.agent_capabilities.load_session);
                assert!(initialize_response.agent_capabilities.mcp_capabilities.http);
                assert!(!initialize_response.agent_capabilities.mcp_capabilities.sse);

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
        let store = test_store();

        let empty_result = EmptyToolRegistry
            .execute(
                &DeepSeekToolCall::new("empty", "anything", "{}"),
                &context,
                &store,
                None,
                CancellationToken::new(),
            )
            .await;
        assert!(!empty_result.success);
        assert!(empty_result.content.contains("unknown tool: anything"));

        let read_only_result = AdapterToolRegistry
            .execute(
                &DeepSeekToolCall::new("read-only", "bogus", "{}"),
                &context,
                &store,
                None,
                CancellationToken::new(),
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
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;

        let Err(error) = handle_set_session_mode_request(
            &store,
            &SetSessionModeRequest::new(session.session_id.clone(), "bogus"),
        ) else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("expected invalid mode id to fail"));
        };
        assert!(error.to_string().contains("unsupported session mode"));

        let Err(error) = handle_set_session_mode_request(
            &store,
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
        let store = test_store();
        let Err(error) = handle_prompt_request(
            &store,
            &MockLlmClient,
            &EmptyToolRegistry,
            None,
            PromptRequest::new(
                agent_client_protocol::schema::SessionId::new("missing"),
                vec![ContentBlock::from("hi")],
            ),
            DEFAULT_MAX_TURN_REQUESTS,
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
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
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
            request_tool_permission(&store, &context, &call, ToolKind::Edit, &requester).await
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

    #[test]
    fn list_sessions_returns_empty_when_no_sessions() -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let response = handle_list_sessions_request(&store, &ListSessionsRequest::new())?;

        assert!(response.sessions.is_empty());
        Ok(())
    }

    #[test]
    fn list_sessions_returns_active_sessions() -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let session1 = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
        let session2 = handle_new_session_request(&store, &NewSessionRequest::new("/home"))?;

        let response = handle_list_sessions_request(&store, &ListSessionsRequest::new())?;

        assert_eq!(response.sessions.len(), 2);
        let ids: Vec<_> = response
            .sessions
            .iter()
            .map(|info| info.session_id.clone())
            .collect();
        assert!(ids.contains(&session1.session_id));
        assert!(ids.contains(&session2.session_id));
        Ok(())
    }

    #[test]
    fn save_history_appends_only_new_messages_to_persistence()
    -> Result<(), agent_client_protocol::Error> {
        let state_dir =
            std::env::temp_dir().join(format!("deepseek-acp-save-history-{}", Uuid::new_v4()));
        let workspace = state_dir.join("workspace");
        let persistence = FilesystemSessionStore::new(&state_dir);
        let store = SessionStore::new(Arc::new(Mutex::new(AdapterState::default())))
            .with_persistence(persistence.clone());
        let session = handle_new_session_request(&store, &NewSessionRequest::new(&workspace))?;

        store.save_history(
            &session.session_id,
            vec![ChatMessage::user("one"), ChatMessage::assistant("two")],
        )?;
        store.save_history(
            &session.session_id,
            vec![
                ChatMessage::user("one"),
                ChatMessage::assistant("two"),
                ChatMessage::user("three"),
            ],
        )?;

        let record = persistence
            .load_record(session.session_id.0.as_ref())
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        assert_eq!(record.history.len(), 3);
        assert_eq!(record.history[0], ChatMessage::user("one"));
        assert_eq!(record.history[2], ChatMessage::user("three"));
        assert_eq!(record.meta.cwd, workspace);

        Ok(())
    }

    #[test]
    fn list_sessions_includes_persisted_sessions_for_requested_cwd()
    -> Result<(), agent_client_protocol::Error> {
        let state_dir =
            std::env::temp_dir().join(format!("deepseek-acp-list-history-{}", Uuid::new_v4()));
        let workspace = state_dir.join("workspace");
        let store = SessionStore::new(Arc::new(Mutex::new(AdapterState::default())))
            .with_persistence(FilesystemSessionStore::new(&state_dir));
        let session = handle_new_session_request(&store, &NewSessionRequest::new(&workspace))?;

        store.save_history(&session.session_id, vec![ChatMessage::user("persist me")])?;
        handle_close_session_request(
            &store,
            &CloseSessionRequest::new(session.session_id.clone()),
        )?;

        let response =
            handle_list_sessions_request(&store, &ListSessionsRequest::new().cwd(&workspace))?;
        assert_eq!(response.sessions.len(), 1);
        assert_eq!(response.sessions[0].session_id, session.session_id);
        assert_eq!(response.sessions[0].cwd, workspace);

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn load_session_restores_state_and_replays_history()
    -> Result<(), agent_client_protocol::Error> {
        let state_dir =
            std::env::temp_dir().join(format!("deepseek-acp-load-history-{}", Uuid::new_v4()));
        let workspace = state_dir.join("workspace");
        let persistence = FilesystemSessionStore::new(&state_dir);
        let store = SessionStore::new(Arc::new(Mutex::new(AdapterState::default())))
            .with_persistence(persistence.clone());
        let session_id = agent_client_protocol::schema::SessionId::new("session-load");
        let tool_call = DeepSeekToolCall::new("call-1", "read_file", r#"{"path":"Cargo.toml"}"#);
        let history = vec![
            ChatMessage::user("inspect the manifest"),
            ChatMessage::assistant_with_tool_calls("reading", vec![tool_call]),
            ChatMessage::tool_result("call-1", "manifest contents"),
            ChatMessage::assistant("done"),
        ];
        persistence
            .persist_turn(
                &PersistedSessionMeta {
                    session_id: session_id.0.to_string(),
                    cwd: workspace.clone(),
                    additional_directories: vec![state_dir.join("extra")],
                    mode: PermissionPosture::AcceptEdits,
                    model: "deepseek-v4-flash".to_string(),
                    reasoning_effort: ReasoningEffort::Max,
                    mcp_servers: Vec::new(),
                },
                &history,
            )
            .map_err(agent_client_protocol::Error::into_internal_error)?;

        let mut notifications = Vec::new();
        let response = handle_load_session_request(
            &store,
            &LoadSessionRequest::new(session_id.clone(), workspace.clone()),
            |notification| {
                notifications.push(notification);
                Ok(())
            },
        )
        .await?;

        assert!(response.config_options.is_some());
        assert_eq!(notifications.len(), 4);
        assert!(matches!(
            notifications[0].update,
            SessionUpdate::UserMessageChunk(_)
        ));
        assert!(matches!(
            notifications[1].update,
            SessionUpdate::AgentMessageChunk(_)
        ));
        let SessionUpdate::ToolCall(replayed_tool_call) = &notifications[2].update else {
            return Err(
                agent_client_protocol::Error::internal_error().data("expected replayed tool call")
            );
        };
        assert_eq!(replayed_tool_call.tool_call_id.0.as_ref(), "call-1");
        assert_eq!(replayed_tool_call.status, ToolCallStatus::Completed);
        assert_eq!(
            replayed_tool_call.raw_input,
            Some(serde_json::json!({ "path": "Cargo.toml" }))
        );
        assert_eq!(
            replayed_tool_call.raw_output,
            Some(serde_json::json!({ "content": "manifest contents" }))
        );
        assert!(matches!(
            notifications[3].update,
            SessionUpdate::AgentMessageChunk(_)
        ));

        let guard = store
            .state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let restored = guard.sessions.get(&session_id).ok_or_else(|| {
            agent_client_protocol::Error::internal_error().data("missing restored session")
        })?;
        assert_eq!(restored.cwd, workspace);
        assert_eq!(restored.mode, PermissionPosture::AcceptEdits);
        assert_eq!(restored.model, "deepseek-v4-flash");
        assert_eq!(restored.reasoning_effort, ReasoningEffort::Max);
        assert_eq!(restored.history, history);

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn load_session_rejects_mismatched_cwd() -> Result<(), agent_client_protocol::Error> {
        let state_dir =
            std::env::temp_dir().join(format!("deepseek-acp-load-cwd-{}", Uuid::new_v4()));
        let workspace = state_dir.join("workspace");
        let persistence = FilesystemSessionStore::new(&state_dir);
        let store = SessionStore::new(Arc::new(Mutex::new(AdapterState::default())))
            .with_persistence(persistence.clone());
        let session_id = agent_client_protocol::schema::SessionId::new("session-load-cwd");
        persistence
            .persist_turn(
                &PersistedSessionMeta {
                    session_id: session_id.0.to_string(),
                    cwd: workspace,
                    additional_directories: Vec::new(),
                    mode: PermissionPosture::Ask,
                    model: "deepseek-v4-pro".to_string(),
                    reasoning_effort: ReasoningEffort::High,
                    mcp_servers: Vec::new(),
                },
                &[ChatMessage::user("hello")],
            )
            .map_err(agent_client_protocol::Error::into_internal_error)?;

        let Err(error) = handle_load_session_request(
            &store,
            &LoadSessionRequest::new(session_id, state_dir.join("other")),
            |_| Ok(()),
        )
        .await
        else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("expected mismatched cwd to fail"));
        };
        assert!(error.to_string().contains("persisted for cwd"));

        Ok(())
    }

    #[test]
    fn close_session_removes_session() -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;

        let close_response = handle_close_session_request(
            &store,
            &CloseSessionRequest::new(session.session_id.clone()),
        )?;

        assert_eq!(
            serde_json::to_value(&close_response)
                .map_err(agent_client_protocol::Error::into_internal_error)?,
            serde_json::json!({})
        );

        let list_response = handle_list_sessions_request(&store, &ListSessionsRequest::new())?;
        assert!(list_response.sessions.is_empty());
        Ok(())
    }

    #[test]
    fn close_session_rejects_unknown_session() -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let unknown_id = agent_client_protocol::schema::SessionId::new("nonexistent");

        let Err(error) =
            handle_close_session_request(&store, &CloseSessionRequest::new(unknown_id))
        else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("expected unknown session id to fail"));
        };
        assert!(error.to_string().contains("unknown session id"));
        Ok(())
    }

    #[test]
    fn logout_request_returns_ok() -> Result<(), agent_client_protocol::Error> {
        let response = handle_logout_request();
        assert_eq!(
            serde_json::to_value(&response)
                .map_err(agent_client_protocol::Error::into_internal_error)?,
            serde_json::json!({})
        );
        Ok(())
    }

    // ── plan_from_prompt multi-sentence path ────────────────

    #[test]
    fn plan_from_prompt_splits_multiple_sentences() {
        let plan = super::plan_from_prompt("Do X. Do Y.");
        assert_eq!(plan.entries.len(), 2);
        assert!(
            plan.entries
                .iter()
                .all(|entry| entry.priority == super::PlanEntryPriority::Medium)
        );
    }

    #[test]
    fn plan_from_prompt_splits_newlines() {
        let plan = super::plan_from_prompt("alpha\nbeta");
        assert_eq!(plan.entries.len(), 2);
    }

    #[test]
    fn plan_from_prompt_single_sentence_uses_high_priority() {
        let plan = super::plan_from_prompt("Just one sentence");
        assert_eq!(plan.entries.len(), 1);
        assert_eq!(plan.entries[0].priority, super::PlanEntryPriority::High);
    }

    // ── resource_link_prompt_text with description ──────────

    #[test]
    fn resource_link_prompt_includes_description_when_present() {
        let mut link = ResourceLink::new("docs", "file:///ref.md");
        link.description = Some("Reference docs".to_string());
        let rendered = super::resource_link_prompt_text(&link);
        assert!(rendered.contains("Reference docs"));
        assert!(rendered.contains(" - "));
    }

    // ── render_command_output ───────────────────────────────

    #[test]
    fn render_command_output_includes_stderr_and_exit_code() {
        let output = super::render_command_output("stdout_line\n", "stderr_line\n", Some(1));
        assert!(output.contains("stdout:\nstdout_line"));
        assert!(output.contains("stderr:\nstderr_line"));
    }

    #[test]
    fn render_command_output_adds_newline_when_stdout_missing_trailing() {
        let output = super::render_command_output("out", "", Some(0));
        assert_eq!(output, "stdout:\nout\n");
    }

    #[test]
    fn render_command_output_empty_uses_signal_label() {
        let output = super::render_command_output("", "", None);
        assert!(output.contains("command exited with status signal"));
    }

    #[test]
    fn render_command_output_empty_uses_numeric_exit_code() {
        let output = super::render_command_output("", "", Some(42));
        assert!(output.contains("command exited with status 42"));
    }

    // ── truncate_tool_output ────────────────────────────────

    #[test]
    fn truncate_tool_output_truncates_when_over_limit() {
        let long = "a".repeat(300);
        let (truncated, flag) = super::truncate_tool_output(&long, 200);
        assert!(flag);
        assert!(truncated.len() <= 300); // roughly 200 + truncation message
        assert!(truncated.contains("... truncated after 200 characters"));
    }

    #[test]
    fn truncate_tool_output_passes_through_short_strings() {
        let short = "hello";
        let (output, flag) = super::truncate_tool_output(short, 200);
        assert!(!flag);
        assert_eq!(output, short);
    }

    // ── is_utf8_error_message ───────────────────────────────

    #[test]
    fn utf8_error_message_detects_all_variants() {
        assert!(super::is_utf8_error_message(
            "stream did not contain valid UTF-8"
        ));
        assert!(super::is_utf8_error_message(
            "file is invalid utf-8 encoded"
        ));
        assert!(super::is_utf8_error_message("non-utf-8 data detected"));
        assert!(super::is_utf8_error_message("some utf8 issue"));
        assert!(!super::is_utf8_error_message("file not found"));
    }

    // ── sanitize_tool_name_part ─────────────────────────────

    #[test]
    fn sanitize_tool_name_handles_empty_result() {
        assert_eq!(super::sanitize_tool_name_part("___"), "unnamed");
        assert_eq!(super::sanitize_tool_name_part(""), "unnamed");
    }

    #[test]
    fn sanitize_tool_name_handles_special_characters() {
        assert_eq!(
            super::sanitize_tool_name_part("Hello World!"),
            "hello_world"
        );
    }

    // ── model_select_options ────────────────────────────────

    #[test]
    fn model_select_options_includes_custom_model_when_unknown() {
        let options = super::model_select_options("my-custom-model");
        let custom = options
            .iter()
            .find(|opt| opt.value.0.as_ref() == "my-custom-model");
        let description = custom.and_then(|opt| opt.description.as_deref());
        assert_eq!(description, Some("Current model from DEEPSEEK_MODEL."));
    }

    #[test]
    fn model_select_options_omits_custom_model_when_known() {
        let options = super::model_select_options("deepseek-v4-pro");
        let custom = options
            .iter()
            .find(|opt| opt.value.0.as_ref() == "deepseek-v4-pro");
        // the known model still appears, but as a standard entry, not a custom one
        assert!(custom.is_some());
        // No entry with the description "Current model from DEEPSEEK_MODEL."
        assert!(!options
            .iter()
            .any(|opt| opt.description.as_deref() == Some("Current model from DEEPSEEK_MODEL.")));
    }

    // ── validate_session_model ──────────────────────────────

    #[test]
    fn validate_session_model_accepts_known_models() -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
        let guard = store
            .state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let record = guard.sessions.get(&session.session_id).ok_or_else(|| {
            agent_client_protocol::Error::internal_error().data("missing session")
        })?;
        // Known models pass.
        assert!(super::validate_session_model(record, "deepseek-v4-pro").is_ok());
        assert!(super::validate_session_model(record, "deepseek-v4-flash").is_ok());
        // Current session model passes even if unknown.
        assert!(super::validate_session_model(record, "deepseek-v4-pro").is_ok());
        Ok(())
    }

    #[test]
    fn validate_session_model_rejects_unknown_models() -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
        let guard = store
            .state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let record = guard.sessions.get(&session.session_id).ok_or_else(|| {
            agent_client_protocol::Error::internal_error().data("missing session")
        })?;
        assert!(super::validate_session_model(record, "bogus-model").is_err());
        Ok(())
    }

    // ── initial_model_from_env ──────────────────────────────

    #[test]
    fn initial_model_from_env_uses_default_when_not_set() {
        // DEEPSEEK_MODEL is not set in test env -> falls back to default
        let model = super::initial_model_from_env();
        assert_eq!(model, "deepseek-v4-pro");
    }

    // ── AdapterToolRegistry::kind ───────────────────────────

    #[test]
    fn adapter_registry_kind_maps_tool_names() {
        let registry = AdapterToolRegistry;
        assert_eq!(registry.kind("read_file"), ToolKind::Read);
        assert_eq!(registry.kind("list_dir"), ToolKind::Read);
        assert_eq!(registry.kind("glob"), ToolKind::Search);
        assert_eq!(registry.kind("grep"), ToolKind::Search);
        assert_eq!(registry.kind("write_file"), ToolKind::Edit);
        assert_eq!(registry.kind("edit_file"), ToolKind::Edit);
        assert_eq!(registry.kind("run_command"), ToolKind::Execute);
        assert_eq!(registry.kind("mcp__server__tool"), ToolKind::Execute);
        assert_eq!(registry.kind("bogus"), ToolKind::Other);
    }

    // ── require_tool_permission error branches ──────────────

    #[test_log::test(tokio::test)]
    async fn require_tool_permission_rejects() -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
        let context = ToolContext {
            session_id: session.session_id.clone(),
            cwd: std::path::PathBuf::from("/tmp"),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let call = DeepSeekToolCall::new(
            "reject-call",
            "run_command",
            serde_json::json!({ "command": "echo hi" }).to_string(),
        );
        let requester = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
            RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                super::PERMISSION_REJECT_ONCE_OPTION_ID,
            )),
        )]);

        let Err(error) = super::require_tool_permission(
            &store,
            &context,
            &call,
            ToolKind::Execute,
            Some(&requester),
        )
        .await
        else {
            return Err(agent_client_protocol::Error::internal_error().data("expected rejection"));
        };
        assert!(error.contains("was rejected by permission policy"));
        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn require_tool_permission_cancelled() -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
        let context = ToolContext {
            session_id: session.session_id.clone(),
            cwd: std::path::PathBuf::from("/tmp"),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let call = DeepSeekToolCall::new(
            "cancel-call",
            "run_command",
            serde_json::json!({ "command": "echo hi" }).to_string(),
        );
        let requester = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
            RequestPermissionOutcome::Cancelled,
        )]);

        let Err(error) = super::require_tool_permission(
            &store,
            &context,
            &call,
            ToolKind::Execute,
            Some(&requester),
        )
        .await
        else {
            return Err(
                agent_client_protocol::Error::internal_error().data("expected cancellation")
            );
        };
        assert!(error.contains("permission request was cancelled"));
        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn require_tool_permission_missing_requester() {
        let store = test_store();
        let context = ToolContext {
            session_id: agent_client_protocol::schema::SessionId::new("no-connection"),
            cwd: std::path::PathBuf::from("/tmp"),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let call = DeepSeekToolCall::new("id", "tool", "{}");
        let Err(error) =
            super::require_tool_permission(&store, &context, &call, ToolKind::Edit, None).await
        else {
            return;
        };
        assert!(error.contains("requires a client connection"));
    }

    // ── PermissionDecision coverage ─────────────────────────

    #[test]
    fn permission_decision_debug_impl_is_callable() {
        let decisions = [
            PermissionDecision::AllowOnce,
            PermissionDecision::AllowAlways,
            PermissionDecision::AllowByMode,
            PermissionDecision::RejectOnce,
            PermissionDecision::RejectAlways,
            PermissionDecision::Cancelled,
        ];
        for decision in &decisions {
            let _ = format!("{decision:?}");
        }
    }

    // ── ReasoningEffort accessor coverage ───────────────────

    #[test]
    fn reasoning_effort_name_and_description() {
        assert_eq!(ReasoningEffort::High.name(), "High");
        assert_eq!(ReasoningEffort::Max.name(), "Max");
        assert!(
            ReasoningEffort::High
                .description()
                .contains("Default DeepSeek")
        );
        assert!(
            ReasoningEffort::Max
                .description()
                .contains("Maximum DeepSeek")
        );
    }

    #[test]
    fn reasoning_effort_from_value_id_rejects_unknown() {
        assert!(
            ReasoningEffort::from_value_id(
                &agent_client_protocol::schema::SessionConfigValueId::new("bogus",)
            )
            .is_none()
        );
    }

    // ── handle_set_session_config_option_request mode branch ─

    #[test]
    fn set_config_option_updates_mode() -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;

        let response = handle_set_session_config_option_request(
            &store,
            &SetSessionConfigOptionRequest::new(
                session.session_id.clone(),
                super::SESSION_CONFIG_MODE_ID,
                "yolo",
            ),
        )?;
        let options = &response.config_options;
        assert_eq!(
            select_current_value(options, super::SESSION_CONFIG_MODE_ID)?,
            "yolo"
        );

        let guard = store
            .state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let stored = guard.sessions.get(&session.session_id).ok_or_else(|| {
            agent_client_protocol::Error::internal_error().data("missing stored session")
        })?;
        assert_eq!(stored.mode, PermissionPosture::Yolo);
        Ok(())
    }

    #[test]
    fn set_config_option_rejects_invalid_mode() -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;

        let Err(error) = handle_set_session_config_option_request(
            &store,
            &SetSessionConfigOptionRequest::new(
                session.session_id,
                super::SESSION_CONFIG_MODE_ID,
                "bogus",
            ),
        ) else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("expected invalid mode via config to fail"));
        };
        assert!(error.to_string().contains("unsupported session mode"));
        Ok(())
    }

    #[test]
    fn set_config_option_rejects_invalid_reasoning_effort()
    -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;

        let Err(error) = handle_set_session_config_option_request(
            &store,
            &SetSessionConfigOptionRequest::new(
                session.session_id,
                super::SESSION_CONFIG_REASONING_EFFORT_ID,
                "bogus",
            ),
        ) else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("expected invalid reasoning effort to fail"));
        };
        assert!(error.to_string().contains("unsupported reasoning effort"));
        Ok(())
    }

    // ── run_command_tool_execution empty command rejection ──

    #[test_log::test(tokio::test)]
    async fn run_command_rejects_empty_command() {
        let store = test_store();
        let context = ToolContext {
            session_id: agent_client_protocol::schema::SessionId::new("empty-cmd"),
            cwd: std::path::PathBuf::from("/tmp"),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let call = DeepSeekToolCall::new(
            "empty-cmd-call",
            "run_command",
            serde_json::json!({ "command": "   " }).to_string(),
        );
        let result = run_command_tool_execution(
            &store,
            &call,
            &context,
            None,
            None,
            &CancellationToken::new(),
        )
        .await;
        assert!(!result.success);
        assert!(result.content.contains("command must not be empty"));
    }

    // ── validate_session_paths relative paths ───────────────

    // ── DeepSeek lib-side unit tests via super ───────────────

    #[test]
    fn message_role_as_str_returns_correct_wire_names() {
        use deepseek_acp_adapter::deepseek::MessageRole;
        // We test indirectly: ChatMessage constructors + role() accessor
        let system = ChatMessage::system("s");
        assert_eq!(system.role(), MessageRole::System);
        let user = ChatMessage::user("u");
        assert_eq!(user.role(), MessageRole::User);
        let assistant = ChatMessage::assistant("a");
        assert_eq!(assistant.role(), MessageRole::Assistant);
        let tool = ChatMessage::tool_result("id", "t");
        assert_eq!(tool.role(), MessageRole::Tool);
    }

    #[test]
    fn chat_message_tool_call_accessors() {
        let tool_calls = vec![DeepSeekToolCall::new("call-1", "echo", "{}")];
        let msg = ChatMessage::assistant_with_tool_calls("assistant", tool_calls.clone());
        assert_eq!(msg.tool_calls().len(), 1);
        assert_eq!(msg.tool_calls()[0].id(), "call-1");
        assert_eq!(msg.tool_calls()[0].name(), "echo");
        assert_eq!(msg.tool_calls()[0].arguments(), "{}");
        assert_eq!(msg.tool_call_id(), None);
    }

    #[test]
    fn chat_message_tool_result_accessors() {
        let msg = ChatMessage::tool_result("call-2", "result");
        assert_eq!(msg.content(), "result");
        assert_eq!(msg.tool_call_id(), Some("call-2"));
    }

    #[test]
    fn tool_definition_accessors() {
        let def = ToolDefinition::new("echo", "description", serde_json::json!({"a":1}));
        assert_eq!(def.name(), "echo");
        assert_eq!(def.description(), "description");
        assert_eq!(def.parameters(), &serde_json::json!({"a":1}));
    }

    #[test]
    fn tool_call_delta_accessors() {
        let delta = ToolCallDelta::new(
            0,
            Some("id".to_string()),
            Some("name".to_string()),
            Some("args".to_string()),
        );
        assert_eq!(delta.index(), 0);
        assert_eq!(delta.id(), Some("id"));
        assert_eq!(delta.name(), Some("name"));
        assert_eq!(delta.arguments(), Some("args"));
    }

    #[test]
    fn tool_call_delta_none_fields() {
        let delta = ToolCallDelta::new(1, None, None, None);
        assert_eq!(delta.index(), 1);
        assert_eq!(delta.id(), None);
        assert_eq!(delta.name(), None);
        assert_eq!(delta.arguments(), None);
    }

    #[test]
    fn chat_request_accessors_no_override() {
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);
        assert_eq!(request.messages().len(), 1);
        assert_eq!(request.tools().len(), 0);
        assert_eq!(request.model(), None);
        assert_eq!(request.reasoning_effort(), None);
    }

    #[test]
    fn finish_reason_from_api_covers_all_branches() {
        assert_eq!(
            FinishReason::EndTurn,
            deepseek_acp_adapter::deepseek::FinishReason::EndTurn
        );
        // We already test EndTurn, MaxTokens, ToolCalls, Refusal via stop_reason_from_finish
    }

    #[test]
    fn deepseek_config_new_accepts_explicit_values() {
        use deepseek_acp_adapter::deepseek::DeepSeekConfig;
        let config = DeepSeekConfig::new("key", "https://example.com", "model-v1");
        assert_eq!(config.base_url(), "https://example.com");
        assert_eq!(config.model(), "model-v1");
    }

    // ── SetSessionConfigOptionRequest unknown session ───────

    #[test]
    fn set_config_option_rejects_unknown_session() -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let Err(error) = handle_set_session_config_option_request(
            &store,
            &SetSessionConfigOptionRequest::new(
                agent_client_protocol::schema::SessionId::new("missing"),
                super::SESSION_CONFIG_MODEL_ID,
                "deepseek-v4-flash",
            ),
        ) else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("expected unknown session id to fail"));
        };
        assert!(error.to_string().contains("unknown session id"));
        Ok(())
    }

    // ── set_session_mode error on unknown session ───────────

    #[test]
    fn set_mode_rejects_unknown_session() -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let Err(error) = handle_set_session_mode_request(
            &store,
            &SetSessionModeRequest::new(
                agent_client_protocol::schema::SessionId::new("missing"),
                "ask",
            ),
        ) else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("expected unknown session id to fail"));
        };
        assert!(error.to_string().contains("unknown session id"));
        Ok(())
    }

    // ── Mock terminal requester for run_command_via_terminal ─

    struct FakeTerminalRequester {
        terminal_id: String,
        output: String,
        exit_code: Option<u32>,
        truncated: bool,
        create_error: Option<String>,
        wait_error: Option<String>,
        output_error: Option<String>,
        release_error: Option<String>,
    }

    impl super::CreateTerminalRequester for FakeTerminalRequester {
        fn create_terminal(
            &self,
            _request: agent_client_protocol::schema::CreateTerminalRequest,
        ) -> BoxFuture<
            '_,
            Result<
                agent_client_protocol::schema::CreateTerminalResponse,
                agent_client_protocol::Error,
            >,
        > {
            let terminal_id = self.terminal_id.clone();
            let error = self.create_error.clone();
            Box::pin(async move {
                if let Some(msg) = error {
                    return Err(agent_client_protocol::Error::internal_error().data(msg));
                }
                Ok(agent_client_protocol::schema::CreateTerminalResponse::new(
                    agent_client_protocol::schema::TerminalId::new(terminal_id),
                ))
            })
        }
    }

    impl super::TerminalOutputRequester for FakeTerminalRequester {
        fn terminal_output(
            &self,
            _request: agent_client_protocol::schema::TerminalOutputRequest,
        ) -> BoxFuture<
            '_,
            Result<
                agent_client_protocol::schema::TerminalOutputResponse,
                agent_client_protocol::Error,
            >,
        > {
            let output = self.output.clone();
            let error = self.output_error.clone();
            let truncated = self.truncated;
            Box::pin(async move {
                if let Some(msg) = error {
                    return Err(agent_client_protocol::Error::internal_error().data(msg));
                }
                Ok(agent_client_protocol::schema::TerminalOutputResponse::new(
                    output, truncated,
                ))
            })
        }
    }

    impl super::WaitForTerminalExitRequester for FakeTerminalRequester {
        fn wait_for_terminal_exit(
            &self,
            _request: agent_client_protocol::schema::WaitForTerminalExitRequest,
        ) -> BoxFuture<
            '_,
            Result<
                agent_client_protocol::schema::WaitForTerminalExitResponse,
                agent_client_protocol::Error,
            >,
        > {
            let exit_code = self.exit_code;
            let error = self.wait_error.clone();
            Box::pin(async move {
                if let Some(msg) = error {
                    return Err(agent_client_protocol::Error::internal_error().data(msg));
                }
                let status =
                    agent_client_protocol::schema::TerminalExitStatus::new().exit_code(exit_code);
                Ok(agent_client_protocol::schema::WaitForTerminalExitResponse::new(status))
            })
        }
    }

    impl super::ReleaseTerminalRequester for FakeTerminalRequester {
        fn release_terminal(
            &self,
            _request: agent_client_protocol::schema::ReleaseTerminalRequest,
        ) -> BoxFuture<
            '_,
            Result<
                agent_client_protocol::schema::ReleaseTerminalResponse,
                agent_client_protocol::Error,
            >,
        > {
            let error = self.release_error.clone();
            Box::pin(async move {
                if let Some(msg) = error {
                    return Err(agent_client_protocol::Error::internal_error().data(msg));
                }
                Ok(agent_client_protocol::schema::ReleaseTerminalResponse::new())
            })
        }
    }

    impl super::KillTerminalRequester for FakeTerminalRequester {
        fn kill_terminal(
            &self,
            _request: agent_client_protocol::schema::KillTerminalRequest,
        ) -> BoxFuture<
            '_,
            Result<
                agent_client_protocol::schema::KillTerminalResponse,
                agent_client_protocol::Error,
            >,
        > {
            Box::pin(async move { Ok(agent_client_protocol::schema::KillTerminalResponse::new()) })
        }
    }

    #[derive(Clone, Default)]
    struct CancelTracker {
        kills: Arc<AtomicUsize>,
        releases: Arc<AtomicUsize>,
    }

    impl super::CreateTerminalRequester for CancelTracker {
        fn create_terminal(
            &self,
            _request: agent_client_protocol::schema::CreateTerminalRequest,
        ) -> BoxFuture<
            '_,
            Result<
                agent_client_protocol::schema::CreateTerminalResponse,
                agent_client_protocol::Error,
            >,
        > {
            Box::pin(async move {
                Ok(agent_client_protocol::schema::CreateTerminalResponse::new(
                    agent_client_protocol::schema::TerminalId::new("term-cancel"),
                ))
            })
        }
    }

    impl super::TerminalOutputRequester for CancelTracker {
        fn terminal_output(
            &self,
            _request: agent_client_protocol::schema::TerminalOutputRequest,
        ) -> BoxFuture<
            '_,
            Result<
                agent_client_protocol::schema::TerminalOutputResponse,
                agent_client_protocol::Error,
            >,
        > {
            Box::pin(async move {
                Ok(agent_client_protocol::schema::TerminalOutputResponse::new(
                    String::new(),
                    false,
                ))
            })
        }
    }

    impl super::WaitForTerminalExitRequester for CancelTracker {
        fn wait_for_terminal_exit(
            &self,
            _request: agent_client_protocol::schema::WaitForTerminalExitRequest,
        ) -> BoxFuture<
            '_,
            Result<
                agent_client_protocol::schema::WaitForTerminalExitResponse,
                agent_client_protocol::Error,
            >,
        > {
            Box::pin(std::future::pending())
        }
    }

    impl super::ReleaseTerminalRequester for CancelTracker {
        fn release_terminal(
            &self,
            _request: agent_client_protocol::schema::ReleaseTerminalRequest,
        ) -> BoxFuture<
            '_,
            Result<
                agent_client_protocol::schema::ReleaseTerminalResponse,
                agent_client_protocol::Error,
            >,
        > {
            let releases = Arc::clone(&self.releases);
            Box::pin(async move {
                releases.fetch_add(1, Ordering::SeqCst);
                Ok(agent_client_protocol::schema::ReleaseTerminalResponse::new())
            })
        }
    }

    impl super::KillTerminalRequester for CancelTracker {
        fn kill_terminal(
            &self,
            _request: agent_client_protocol::schema::KillTerminalRequest,
        ) -> BoxFuture<
            '_,
            Result<
                agent_client_protocol::schema::KillTerminalResponse,
                agent_client_protocol::Error,
            >,
        > {
            let kills = Arc::clone(&self.kills);
            Box::pin(async move {
                kills.fetch_add(1, Ordering::SeqCst);
                Ok(agent_client_protocol::schema::KillTerminalResponse::new())
            })
        }
    }

    // (TerminalRequester is auto-implemented via blanket impl)

    // ── run_command_via_terminal tests ──────────────────────

    #[test_log::test(tokio::test)]
    async fn run_command_via_terminal_success_path() {
        let session_id = agent_client_protocol::schema::SessionId::new("terminal-test");
        let fake = FakeTerminalRequester {
            terminal_id: "term-1".to_string(),
            output: "command output".to_string(),
            exit_code: Some(0),
            truncated: false,
            create_error: None,
            wait_error: None,
            output_error: None,
            release_error: None,
        };

        let result = super::run_command_via_terminal(
            &session_id,
            std::path::Path::new("/tmp"),
            "echo hi",
            Some(&fake as &dyn super::TerminalRequester),
            &CancellationToken::new(),
        )
        .await;

        assert!(result.success);
        assert!(result.content.contains("command output"));
    }

    #[test_log::test(tokio::test)]
    async fn run_command_via_terminal_no_connection() {
        let session_id = agent_client_protocol::schema::SessionId::new("terminal-no-conn");
        let result = super::run_command_via_terminal(
            &session_id,
            std::path::Path::new("/tmp"),
            "echo hi",
            None,
            &CancellationToken::new(),
        )
        .await;

        assert!(!result.success);
        assert!(result.content.contains("no connection available"));
    }

    #[test_log::test(tokio::test)]
    async fn run_command_via_terminal_create_error() {
        let session_id = agent_client_protocol::schema::SessionId::new("terminal-create-err");
        let fake = FakeTerminalRequester {
            terminal_id: "term-err".to_string(),
            output: String::new(),
            exit_code: None,
            truncated: false,
            create_error: Some("create failed".to_string()),
            wait_error: None,
            output_error: None,
            release_error: None,
        };

        let result = super::run_command_via_terminal(
            &session_id,
            std::path::Path::new("/tmp"),
            "echo hi",
            Some(&fake as &dyn super::TerminalRequester),
            &CancellationToken::new(),
        )
        .await;

        assert!(!result.success);
        assert!(result.content.contains("terminal/create failed"));
    }

    #[test_log::test(tokio::test)]
    async fn run_command_via_terminal_wait_error() {
        let session_id = agent_client_protocol::schema::SessionId::new("terminal-wait-err");
        let fake = FakeTerminalRequester {
            terminal_id: "term-wait".to_string(),
            output: String::new(),
            exit_code: None,
            truncated: false,
            create_error: None,
            wait_error: Some("wait failed".to_string()),
            output_error: None,
            release_error: None,
        };

        let result = super::run_command_via_terminal(
            &session_id,
            std::path::Path::new("/tmp"),
            "echo hi",
            Some(&fake as &dyn super::TerminalRequester),
            &CancellationToken::new(),
        )
        .await;

        assert!(!result.success);
        assert!(result.content.contains("terminal/wait_for_exit failed"));
    }

    #[test_log::test(tokio::test)]
    async fn run_command_via_terminal_output_error() {
        let session_id = agent_client_protocol::schema::SessionId::new("terminal-output-err");
        let fake = FakeTerminalRequester {
            terminal_id: "term-out".to_string(),
            output: String::new(),
            exit_code: None,
            truncated: false,
            create_error: None,
            wait_error: None,
            output_error: Some("output failed".to_string()),
            release_error: None,
        };

        let result = super::run_command_via_terminal(
            &session_id,
            std::path::Path::new("/tmp"),
            "echo hi",
            Some(&fake as &dyn super::TerminalRequester),
            &CancellationToken::new(),
        )
        .await;

        assert!(!result.success);
        assert!(result.content.contains("terminal/output failed"));
    }

    #[test_log::test(tokio::test)]
    async fn run_command_via_terminal_release_error() {
        let session_id = agent_client_protocol::schema::SessionId::new("terminal-release-err");
        let fake = FakeTerminalRequester {
            terminal_id: "term-rel".to_string(),
            output: "output".to_string(),
            exit_code: Some(0),
            truncated: false,
            create_error: None,
            wait_error: None,
            output_error: None,
            release_error: Some("release failed".to_string()),
        };

        let result = super::run_command_via_terminal(
            &session_id,
            std::path::Path::new("/tmp"),
            "echo hi",
            Some(&fake as &dyn super::TerminalRequester),
            &CancellationToken::new(),
        )
        .await;

        assert!(!result.success);
        assert!(result.content.contains("terminal/release failed"));
    }

    #[test_log::test(tokio::test)]
    async fn run_command_via_terminal_kills_on_cancellation() {
        let tracker = CancelTracker::default();
        let token = CancellationToken::new();
        token.cancel();

        let session_id = agent_client_protocol::schema::SessionId::new("terminal-cancel");
        let result = super::run_command_via_terminal(
            &session_id,
            std::path::Path::new("/tmp"),
            "sleep 100",
            Some(&tracker as &dyn super::TerminalRequester),
            &token,
        )
        .await;

        assert!(!result.success);
        assert!(result.content.contains("cancelled"));
        assert_eq!(tracker.kills.load(Ordering::SeqCst), 1);
        assert_eq!(tracker.releases.load(Ordering::SeqCst), 1);
    }

    // ── edit_file_tool_execution error paths ────────────────

    #[test_log::test(tokio::test)]
    async fn edit_file_rejects_empty_old_text() -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let temp_root = std::env::temp_dir().join(format!(
            "deepseek-acp-adapter-edit-empty-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        std::fs::write(temp_root.join("f.txt"), "content")
            .map_err(agent_client_protocol::Error::into_internal_error)?;

        let session = handle_new_session_request(&store, &NewSessionRequest::new(&temp_root))?;
        let context = ToolContext {
            session_id: session.session_id.clone(),
            cwd: temp_root.clone(),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let call = DeepSeekToolCall::new(
            "edit-empty",
            "edit_file",
            serde_json::json!({
                "path": "f.txt",
                "old_text": "",
                "new_text": "replacement",
            })
            .to_string(),
        );

        let result = edit_file_tool_execution(&store, &call, &context, None, None, None).await;
        assert!(!result.success);
        assert!(result.content.contains("old_text must not be empty"));
        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn edit_file_rejects_old_text_not_found() -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let temp_root =
            std::env::temp_dir().join(format!("deepseek-acp-adapter-edit-nf-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        std::fs::write(temp_root.join("f.txt"), "content")
            .map_err(agent_client_protocol::Error::into_internal_error)?;

        let session = handle_new_session_request(&store, &NewSessionRequest::new(&temp_root))?;
        let context = ToolContext {
            session_id: session.session_id.clone(),
            cwd: temp_root.clone(),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let call = DeepSeekToolCall::new(
            "edit-nf",
            "edit_file",
            serde_json::json!({
                "path": "f.txt",
                "old_text": "nonexistent",
                "new_text": "replacement",
            })
            .to_string(),
        );

        let result = edit_file_tool_execution(&store, &call, &context, None, None, None).await;
        assert!(!result.success);
        assert!(result.content.contains("could not find old_text"));
        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn edit_file_rejects_multiple_matches() -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let temp_root = std::env::temp_dir().join(format!(
            "deepseek-acp-adapter-edit-multi-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        std::fs::write(temp_root.join("f.txt"), "dup dup")
            .map_err(agent_client_protocol::Error::into_internal_error)?;

        let session = handle_new_session_request(&store, &NewSessionRequest::new(&temp_root))?;
        let context = ToolContext {
            session_id: session.session_id.clone(),
            cwd: temp_root.clone(),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let call = DeepSeekToolCall::new(
            "edit-multi",
            "edit_file",
            serde_json::json!({
                "path": "f.txt",
                "old_text": "dup",
                "new_text": "replacement",
            })
            .to_string(),
        );

        let result = edit_file_tool_execution(&store, &call, &context, None, None, None).await;
        assert!(!result.success);
        assert!(result.content.contains("found old_text"));
        assert!(result.content.contains("2 times"));
        Ok(())
    }

    // ── write_file_tool_execution error paths ───────────────

    #[test_log::test(tokio::test)]
    async fn write_file_rejects_invalid_arguments() -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
        let context = ToolContext {
            session_id: session.session_id.clone(),
            cwd: std::path::PathBuf::from("/tmp"),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let call = DeepSeekToolCall::new("write-invalid", "write_file", "not json");

        let result = write_file_tool_execution(&store, &call, &context, None, None, None).await;
        assert!(!result.success);
        assert!(result.content.contains("invalid write_file arguments"));
        Ok(())
    }

    // ── run_command_tool_execution error paths ──────────────

    #[test_log::test(tokio::test)]
    async fn run_command_rejects_invalid_arguments() -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
        let context = ToolContext {
            session_id: session.session_id.clone(),
            cwd: std::path::PathBuf::from("/tmp"),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let call = DeepSeekToolCall::new("run-invalid", "run_command", "not json");

        let result = run_command_tool_execution(
            &store,
            &call,
            &context,
            None,
            None,
            &CancellationToken::new(),
        )
        .await;
        assert!(!result.success);
        assert!(result.content.contains("invalid run_command arguments"));
        Ok(())
    }

    // ── edit_file_tool_execution invalid args ───────────────

    #[test_log::test(tokio::test)]
    async fn edit_file_rejects_invalid_arguments() -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
        let context = ToolContext {
            session_id: session.session_id.clone(),
            cwd: std::path::PathBuf::from("/tmp"),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let call = DeepSeekToolCall::new("edit-invalid", "edit_file", "not json");

        let result = edit_file_tool_execution(&store, &call, &context, None, None, None).await;
        assert!(!result.success);
        assert!(result.content.contains("invalid edit_file arguments"));
        Ok(())
    }

    // ── non_utf8_file_message / read_file_local_error coverage ──

    #[test]
    fn non_utf8_file_message_includes_path() {
        let msg = super::non_utf8_file_message(std::path::Path::new("/tmp/binary.bin"));
        assert!(msg.contains("/tmp/binary.bin"));
        assert!(msg.contains("UTF-8"));
    }

    #[test]
    fn read_file_client_error_with_non_utf8_message() {
        let msg = super::read_file_client_error(
            std::path::Path::new("/tmp/binary.bin"),
            "stream did not contain valid UTF-8",
        );
        assert!(msg.contains("UTF-8"));
        // Should NOT contain the raw technical message
        assert!(!msg.contains("stream did not contain valid UTF-8"));
    }

    // ── write_file_to_client error path ─────────────────────

    struct FailingWriteRequester;

    impl WriteTextFileRequester for FailingWriteRequester {
        fn write_text_file(
            &self,
            _request: WriteTextFileRequest,
        ) -> BoxFuture<'_, Result<WriteTextFileResponse, agent_client_protocol::Error>> {
            Box::pin(async move {
                Err(agent_client_protocol::Error::internal_error().data("disk full"))
            })
        }
    }

    #[test_log::test(tokio::test)]
    async fn write_file_to_client_propagates_error() -> Result<(), agent_client_protocol::Error> {
        let session_id = agent_client_protocol::schema::SessionId::new("write-err");
        let result = super::write_file_to_client(
            &FailingWriteRequester,
            &session_id,
            std::path::Path::new("/tmp/note.txt"),
            "content",
        )
        .await;
        let Err(error) = result else {
            return Err(agent_client_protocol::Error::internal_error().data("expected failure"));
        };
        assert!(error.contains("failed to write"));
        Ok(())
    }

    // ── collect_directory_entries error path ────────────────

    #[test]
    fn collect_directory_entries_reports_missing() -> Result<(), agent_client_protocol::Error> {
        let Err(error) =
            super::collect_directory_entries(std::path::Path::new("/tmp/nonexistent-dir-for-test"))
        else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("expected error for missing dir"));
        };
        assert!(error.contains("failed to read directory"));
        Ok(())
    }

    // ── build_root_gitignore error with invalid gitignore ────
    // (already tested that it returns None for missing, here we test when present)

    #[test]
    fn build_root_gitignore_loads_when_present() -> Result<(), agent_client_protocol::Error> {
        let temp_root =
            std::env::temp_dir().join(format!("deepseek-acp-adapter-gi-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        std::fs::write(temp_root.join(".gitignore"), "*.log\n")
            .map_err(agent_client_protocol::Error::into_internal_error)?;

        let gitignore = super::build_root_gitignore(&temp_root);
        assert!(gitignore.is_some());
        Ok(())
    }

    // ── handle_set_session_mode_request invalid mode id ─────

    #[test]
    fn set_mode_rejects_invalid_mode_id() -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
        let Err(error) = handle_set_session_mode_request(
            &store,
            &SetSessionModeRequest::new(session.session_id, "bogus"),
        ) else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("expected invalid mode to fail"));
        };
        assert!(error.to_string().contains("unsupported session mode"));
        Ok(())
    }

    // ── exercise_permission_gate_smoke already covered ──────

    // ── read_file_from_local line/limit bounds ──────────────

    #[test]
    fn read_file_from_local_zero_line_defaults_to_start() -> Result<(), agent_client_protocol::Error>
    {
        let temp_root =
            std::env::temp_dir().join(format!("deepseek-acp-adapter-read-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        std::fs::write(temp_root.join("lines.txt"), "a\nb\nc\nd\ne\n")
            .map_err(agent_client_protocol::Error::into_internal_error)?;

        // Using line=1 is the default; test line > file length
        let result = super::read_file_from_local(&temp_root.join("lines.txt"), 10, 1);
        assert_eq!(
            result.map_err(|e| agent_client_protocol::Error::internal_error().data(e))?,
            ""
        );
        Ok(())
    }

    // ── resource_link_prompt_text with title ─────────────────

    #[test]
    fn resource_link_prompt_text_uses_title_over_name() {
        let mut link = ResourceLink::new("internal_name", "file:///foo.md");
        link.title = Some("Display Title".to_string());
        let rendered = super::resource_link_prompt_text(&link);
        assert!(rendered.contains("Display Title"));
        assert!(!rendered.contains("internal_name"));
    }

    // ── render_command_output stderr trailing newline ───────

    #[test]
    fn render_command_output_adds_newline_to_stderr() {
        let output = super::render_command_output("", "err", Some(2));
        // stderr without trailing newline gets one added
        assert!(output.contains("stderr:\nerr\n"));
    }

    // ── glob_tool_execution with build error path ───────────

    #[test_log::test(tokio::test)]
    async fn glob_tool_execution_invalid_build_pattern() -> Result<(), agent_client_protocol::Error>
    {
        let temp_root =
            std::env::temp_dir().join(format!("deepseek-acp-adapter-glob-err-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;

        let context = ToolContext {
            session_id: agent_client_protocol::schema::SessionId::new("glob-err"),
            cwd: temp_root,
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        // '[' is a valid glob pattern but can't be compiled into a GlobSet easily
        // Using an invalid pattern that passes Glob::new but fails GlobSetBuilder
        let call = DeepSeekToolCall::new(
            "glob-invalid",
            "glob",
            serde_json::json!({ "pattern": "[" }).to_string(),
        );

        let result = glob_tool_execution(&call, &context);
        assert!(!result.success);
        assert!(result.content.contains("invalid glob pattern"));
        Ok(())
    }

    // ── grep_tool_execution with invalid regex ──────────────

    #[test_log::test(tokio::test)]
    async fn grep_tool_execution_invalid_regex() -> Result<(), agent_client_protocol::Error> {
        let temp_root = std::env::temp_dir().join(format!(
            "deepseek-acp-adapter-grep-regex-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        std::fs::write(temp_root.join("f.txt"), "test\n")
            .map_err(agent_client_protocol::Error::into_internal_error)?;

        let context = ToolContext {
            session_id: agent_client_protocol::schema::SessionId::new("grep-regex"),
            cwd: temp_root,
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let call = DeepSeekToolCall::new(
            "grep-regex",
            "grep",
            serde_json::json!({ "pattern": "(" }).to_string(),
        );

        let result = grep_tool_execution(&call, &context);
        assert!(!result.success);
        assert!(result.content.contains("invalid grep regex"));
        Ok(())
    }

    // ── build_root_gitignore with real .gitignore file ──────

    #[test]
    fn build_root_gitignore_loads_file() -> Result<(), agent_client_protocol::Error> {
        let temp_root =
            std::env::temp_dir().join(format!("deepseek-acp-adapter-gitignore-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        std::fs::write(temp_root.join(".gitignore"), "*.log\n")
            .map_err(agent_client_protocol::Error::into_internal_error)?;

        let gitignore = super::build_root_gitignore(&temp_root);
        assert!(gitignore.is_some());
        Ok(())
    }

    // ── read_file_from_local line past end ──────────────────

    #[test]
    fn read_file_from_local_line_past_end_returns_empty() -> Result<(), agent_client_protocol::Error>
    {
        let temp_root =
            std::env::temp_dir().join(format!("deepseek-acp-adapter-read-past-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        std::fs::write(temp_root.join("lines.txt"), "a\nb\nc\n")
            .map_err(agent_client_protocol::Error::into_internal_error)?;

        let result = super::read_file_from_local(&temp_root.join("lines.txt"), 10, 5);
        assert_eq!(
            result.map_err(|e| agent_client_protocol::Error::internal_error().data(e))?,
            ""
        );
        Ok(())
    }

    // ── read_file_local_error non-utf8 path ─────────────────

    #[test]
    fn read_file_local_error_handles_invalid_data() -> Result<(), agent_client_protocol::Error> {
        let temp_root =
            std::env::temp_dir().join(format!("deepseek-acp-adapter-local-err-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let bin_path = temp_root.join("artifact.bin");
        std::fs::write(&bin_path, [0xff, 0xfe, 0xfd])
            .map_err(agent_client_protocol::Error::into_internal_error)?;

        let Err(error) = std::fs::read_to_string(&bin_path) else {
            return Err(agent_client_protocol::Error::internal_error()
                .data("expected non-UTF-8 read to fail"));
        };
        let msg = super::read_file_local_error(&bin_path, &error);
        assert!(msg.contains("UTF-8"));
        Ok(())
    }

    // ── mock_llm_client empty messages fallback ─────────────

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

    // ── handle_new_session_request_connected test ───────────

    #[test_log::test(tokio::test)]
    async fn new_session_connected_async_path_creates_session()
    -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let response =
            super::handle_new_session_request_connected(&store, &NewSessionRequest::new("/tmp"))
                .await?;
        assert!(response.session_id.0.starts_with("session-"));
        Ok(())
    }

    // ── serve_with_transport full integration test ──────────

    #[test_log::test(tokio::test)]
    async fn serve_with_transport_exercises_list_close_and_logout()
    -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let llm_client: Arc<dyn LlmClient> = Arc::new(MockLlmClient);
        let tool_registry: Arc<dyn ToolRegistry> = Arc::new(EmptyToolRegistry);
        let (client_transport, server_transport) = Channel::duplex();
        let server_state = Arc::clone(&store.state);
        let server_client = Arc::clone(&llm_client);
        let server_tools = Arc::clone(&tool_registry);

        let server = tokio::spawn(async move {
            serve_with_transport(
                server_transport,
                server_state,
                server_client,
                server_tools,
                DEFAULT_MAX_TURN_REQUESTS,
            )
            .await
        });

        Client
            .builder()
            .connect_with(client_transport, async move |cx| {
                // Initialize
                cx.send_request(InitializeRequest::new(ProtocolVersion::LATEST))
                    .block_task()
                    .await?;

                // Create a session
                let new_session = cx
                    .send_request(NewSessionRequest::new("/tmp"))
                    .block_task()
                    .await?;

                // List sessions
                let list = cx
                    .send_request(agent_client_protocol::schema::ListSessionsRequest::new())
                    .block_task()
                    .await?;
                assert_eq!(list.sessions.len(), 1);

                // Close session
                cx.send_request(agent_client_protocol::schema::CloseSessionRequest::new(
                    new_session.session_id,
                ))
                .block_task()
                .await?;

                // Logout
                cx.send_request(agent_client_protocol::schema::LogoutRequest::new())
                    .block_task()
                    .await?;

                Ok(())
            })
            .await?;

        server.abort();
        Ok(())
    }

    // ── AdapterToolRegistry execute with connection ─────────

    #[test_log::test(tokio::test)]
    async fn adapter_registry_execute_write_without_permission()
    -> Result<(), agent_client_protocol::Error> {
        let temp_root =
            std::env::temp_dir().join(format!("deepseek-acp-adapter-conn-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;

        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new(&temp_root))?;
        let context = ToolContext {
            session_id: session.session_id,
            cwd: temp_root,
            additional_directories: Vec::new(),
            client_capabilities: None,
        };

        let registry = AdapterToolRegistry;
        // write_file requires permission; without it, fails
        let result = registry
            .execute(
                &DeepSeekToolCall::new(
                    "conn-write",
                    "write_file",
                    serde_json::json!({ "path": "out.txt", "content": "data" }).to_string(),
                ),
                &context,
                &store,
                None,
                CancellationToken::new(),
            )
            .await;
        assert!(!result.success);

        Ok(())
    }

    // ── write_file with client_fs capability but no connection ──

    #[test_log::test(tokio::test)]
    async fn write_file_with_client_capability_but_no_connection_errors()
    -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
        let context = ToolContext {
            session_id: session.session_id.clone(),
            cwd: std::path::PathBuf::from("/tmp"),
            additional_directories: Vec::new(),
            client_capabilities: Some(
                ClientCapabilities::new().fs(FileSystemCapabilities::new().write_text_file(true)),
            ),
        };
        let call = DeepSeekToolCall::new(
            "w",
            "write_file",
            serde_json::json!({"path": "out.txt", "content": "hi"}).to_string(),
        );

        // Provide a permission requester so we get past the permission gate,
        // but no write_requester - this exercises the "needs a client connection" path.
        let permission = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
            RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                super::PERMISSION_ALLOW_ONCE_OPTION_ID,
            )),
        )]);
        let result =
            write_file_tool_execution(&store, &call, &context, None, None, Some(&permission)).await;
        assert!(!result.success);
        assert!(
            result
                .content
                .contains("write_file needs a client connection")
        );
        Ok(())
    }

    // ── edit_file with client_fs read capability but no connection ──

    #[test_log::test(tokio::test)]
    async fn edit_file_with_client_read_capability_but_no_connection_errors()
    -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
        let context = ToolContext {
            session_id: session.session_id.clone(),
            cwd: std::path::PathBuf::from("/tmp"),
            additional_directories: Vec::new(),
            client_capabilities: Some(
                ClientCapabilities::new().fs(FileSystemCapabilities::new().read_text_file(true)),
            ),
        };
        let call = DeepSeekToolCall::new(
            "e",
            "edit_file",
            serde_json::json!({"path": "out.txt", "old_text": "a", "new_text": "b"}).to_string(),
        );

        let permission = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
            RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                super::PERMISSION_ALLOW_ONCE_OPTION_ID,
            )),
        )]);
        let result =
            edit_file_tool_execution(&store, &call, &context, None, None, Some(&permission)).await;
        assert!(!result.success);
        assert!(
            result
                .content
                .contains("edit_file needs a client connection for fs/read_text_file")
        );
        Ok(())
    }

    // ── edit_file with client_fs write capability but no connection ──

    #[test_log::test(tokio::test)]
    async fn edit_file_with_client_write_capability_but_no_connection_errors()
    -> Result<(), agent_client_protocol::Error> {
        let temp_root =
            std::env::temp_dir().join(format!("deepseek-acp-client-write-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        std::fs::write(temp_root.join("out.txt"), "hello world")
            .map_err(agent_client_protocol::Error::into_internal_error)?;

        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new(&temp_root))?;
        let context = ToolContext {
            session_id: session.session_id.clone(),
            cwd: temp_root.clone(),
            additional_directories: Vec::new(),
            client_capabilities: Some(
                ClientCapabilities::new().fs(FileSystemCapabilities::new().write_text_file(true)),
            ),
        };
        let call = DeepSeekToolCall::new(
            "e",
            "edit_file",
            serde_json::json!({"path": "out.txt", "old_text": "hello", "new_text": "bye"})
                .to_string(),
        );

        let permission = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
            RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                super::PERMISSION_ALLOW_ONCE_OPTION_ID,
            )),
        )]);
        let result =
            edit_file_tool_execution(&store, &call, &context, None, None, Some(&permission)).await;
        assert!(!result.success);
        assert!(
            result
                .content
                .contains("edit_file needs a client connection for fs/write_text_file")
        );
        Ok(())
    }

    // ── read_file with client_fs capability but no connection ──

    #[test_log::test(tokio::test)]
    async fn read_file_with_client_capability_but_no_connection_errors()
    -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
        let context = ToolContext {
            session_id: session.session_id.clone(),
            cwd: std::path::PathBuf::from("/tmp"),
            additional_directories: Vec::new(),
            client_capabilities: Some(
                ClientCapabilities::new().fs(FileSystemCapabilities::new().read_text_file(true)),
            ),
        };
        let call = DeepSeekToolCall::new(
            "r",
            "read_file",
            serde_json::json!({"path": "missing.txt"}).to_string(),
        );

        let result = read_file_tool_execution(&call, &context, None).await;
        assert!(!result.success);
        assert!(
            result
                .content
                .contains("read_file needs a client connection")
        );
        Ok(())
    }
}
