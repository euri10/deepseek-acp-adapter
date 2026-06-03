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

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::{error::Error, process::ExitCode};

use agent_client_protocol::schema::{
    AgentAuthCapabilities, AgentCapabilities, AuthenticateRequest, AuthenticateResponse,
    CancelNotification, ClientCapabilities, ContentBlock, ContentChunk, InitializeRequest,
    InitializeResponse, NewSessionRequest, NewSessionResponse, PromptCapabilities, PromptRequest,
    PromptResponse, ProtocolVersion, ReadTextFileRequest, ReadTextFileResponse,
    RequestPermissionOutcome, RequestPermissionRequest, RequestPermissionResponse,
    SelectedPermissionOutcome, SessionId, SessionMode, SessionModeState, SessionNotification,
    SessionUpdate, StopReason, ToolCall as AcpToolCall, ToolCallContent, ToolCallStatus,
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
use serde::Deserialize;
use serde_json::Value;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

type AdapterResult<T> = Result<T, Box<dyn Error + Send + Sync + 'static>>;
const MAX_TURN_REQUESTS: usize = 25;

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

async fn serve_with_transport(
    transport: impl ConnectTo<Agent> + 'static,
    state: Arc<Mutex<AdapterState>>,
    llm_client: Arc<dyn LlmClient>,
    tool_registry: Arc<dyn ToolRegistry>,
) -> Result<(), agent_client_protocol::Error> {
    let initialize_state = Arc::clone(&state);
    let new_session_state = Arc::clone(&state);
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
        },
    );

    Ok(NewSessionResponse::new(session_id).modes(default_session_modes()))
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

#[derive(Debug, Deserialize)]
struct ReadFileArguments {
    path: PathBuf,
    line: Option<u32>,
    limit: Option<u32>,
}

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
        vec![read_file_tool_definition()]
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
    let requested_limit = parsed_arguments.limit.unwrap_or(200);
    let limit = requested_limit.min(200);

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
    SessionModeState::new("chat", vec![SessionMode::new("chat", "Chat")])
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
}

#[cfg(test)]
mod tests {
    use super::{
        AdapterState, Backend, Cli, Command, EmptyToolRegistry, MAX_TURN_REQUESTS, MockLlmClient,
        ReadTextFileRequester, ToolContext, ToolExecution, ToolRegistry, build_dev_agent,
        build_initialize_response, handle_authenticate_request, handle_cancel_notification,
        handle_initialize_request, handle_new_session_request, handle_prompt_request,
        read_file_tool_execution, run_smoke_flow, serve_with_transport,
    };
    use agent_client_protocol::schema::McpServer;
    use agent_client_protocol::{Agent, Channel, Client};
    use deepseek_acp_adapter::deepseek::{
        ChatRequest, DeepSeekError, FinishReason, LlmClient, StreamEvent,
        ToolCall as DeepSeekToolCall, ToolCallDelta, ToolDefinition,
    };
    use futures_util::future::BoxFuture;
    use futures_util::stream::{self, BoxStream};
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};
    use uuid::Uuid;

    use agent_client_protocol::schema::{
        CancelNotification, ClientCapabilities, ContentBlock, FileSystemCapabilities,
        InitializeRequest, NewSessionRequest, PromptRequest, ProtocolVersion, ReadTextFileRequest,
        ReadTextFileResponse, SessionNotification, SessionUpdate, StopReason,
    };
    use clap::Parser;
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
        assert_eq!(modes.current_mode_id.0.as_ref(), "chat");
        assert_eq!(modes.available_modes.len(), 1);

        let guard = state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        assert!(guard.sessions.contains_key(&response.session_id));

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
}
