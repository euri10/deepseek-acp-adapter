//! ACP-facing transport registration and request handling.

use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::{
    AgentAuthCapabilities, AgentCapabilities, AuthenticateRequest, AuthenticateResponse,
    AvailableCommandsUpdate, CancelNotification, CloseSessionRequest, CloseSessionResponse,
    ConfigOptionUpdate, ContentBlock, ContentChunk, CreateTerminalRequest, CreateTerminalResponse,
    CurrentModeUpdate, Implementation, InitializeRequest, InitializeResponse, KillTerminalRequest,
    KillTerminalResponse, ListSessionsRequest, ListSessionsResponse, LoadSessionRequest,
    LoadSessionResponse, LogoutCapabilities, LogoutRequest, LogoutResponse, McpCapabilities,
    NewSessionRequest, NewSessionResponse, PromptRequest, PromptResponse, ProtocolVersion,
    ReadTextFileRequest, ReadTextFileResponse, ReleaseTerminalRequest, ReleaseTerminalResponse,
    RequestPermissionRequest, RequestPermissionResponse, ResumeSessionRequest,
    ResumeSessionResponse, SessionAdditionalDirectoriesCapabilities, SessionCapabilities,
    SessionCloseCapabilities, SessionConfigOptionValue, SessionConfigValueId, SessionId,
    SessionListCapabilities, SessionNotification, SessionResumeCapabilities, SessionUpdate,
    SetSessionConfigOptionRequest, SetSessionConfigOptionResponse, SetSessionModeRequest,
    SetSessionModeResponse, TerminalOutputRequest, TerminalOutputResponse, ToolCall as AcpToolCall,
    ToolCallContent, ToolCallStatus, WaitForTerminalExitRequest, WaitForTerminalExitResponse,
    WriteTextFileRequest, WriteTextFileResponse,
};
use agent_client_protocol::{Agent, Client, ConnectTo};
use deepseek_acp_adapter::deepseek::{
    ChatMessage, LlmClient, MessageRole, ToolCall as DeepSeekToolCall,
};
use futures_util::future::BoxFuture;
use uuid::Uuid;

use crate::tools::ToolRegistry;
use crate::{
    ADAPTER_NAME, ADAPTER_VERSION, AdapterState, FilesystemSessionStore, McpSession,
    PermissionPosture, ReasoningEffort, SESSION_CONFIG_MODE_ID, SESSION_CONFIG_MODEL_ID,
    SESSION_CONFIG_REASONING_EFFORT_ID, SessionRecord, SessionStore, adapter_available_commands,
    connect_mcp_sessions, default_session_modes, session_notification, tool_raw_input,
    validate_session_model,
};

type AcpRequestFuture<'a, T> = BoxFuture<'a, Result<T, agent_client_protocol::Error>>;

pub(crate) trait ReadTextFileRequester: Send + Sync {
    fn read_text_file(
        &self,
        request: ReadTextFileRequest,
    ) -> AcpRequestFuture<'_, ReadTextFileResponse>;
}

pub(crate) trait WriteTextFileRequester: Send + Sync {
    fn write_text_file(
        &self,
        request: WriteTextFileRequest,
    ) -> AcpRequestFuture<'_, WriteTextFileResponse>;
}

/// Trait for creating a terminal via ACP client `terminal/create`.
pub(crate) trait CreateTerminalRequester: Send + Sync {
    /// Create a terminal and execute a command.
    fn create_terminal(
        &self,
        request: CreateTerminalRequest,
    ) -> AcpRequestFuture<'_, CreateTerminalResponse>;
}

/// Trait for getting terminal output via ACP client `terminal/output`.
pub(crate) trait TerminalOutputRequester: Send + Sync {
    /// Get the current output and status of a terminal.
    fn terminal_output(
        &self,
        request: TerminalOutputRequest,
    ) -> AcpRequestFuture<'_, TerminalOutputResponse>;
}

/// Trait for waiting for terminal exit via ACP client `terminal/wait_for_exit`.
pub(crate) trait WaitForTerminalExitRequester: Send + Sync {
    /// Wait for a terminal command to exit.
    fn wait_for_terminal_exit(
        &self,
        request: WaitForTerminalExitRequest,
    ) -> AcpRequestFuture<'_, WaitForTerminalExitResponse>;
}

/// Trait for releasing a terminal via ACP client `terminal/release`.
pub(crate) trait ReleaseTerminalRequester: Send + Sync {
    /// Release a terminal and free its resources.
    fn release_terminal(
        &self,
        request: ReleaseTerminalRequest,
    ) -> AcpRequestFuture<'_, ReleaseTerminalResponse>;
}

/// Trait for killing a terminal command via ACP client `terminal/kill`.
pub(crate) trait KillTerminalRequester: Send + Sync {
    /// Kill a terminal's running command without releasing the terminal.
    fn kill_terminal(
        &self,
        request: KillTerminalRequest,
    ) -> AcpRequestFuture<'_, KillTerminalResponse>;
}

/// Combined trait for all terminal operations.
pub(crate) trait TerminalRequester:
    CreateTerminalRequester
    + TerminalOutputRequester
    + WaitForTerminalExitRequester
    + ReleaseTerminalRequester
    + KillTerminalRequester
{
}

impl<T> TerminalRequester for T where
    T: CreateTerminalRequester
        + TerminalOutputRequester
        + WaitForTerminalExitRequester
        + ReleaseTerminalRequester
        + KillTerminalRequester
        + ?Sized
{
}

impl CreateTerminalRequester for agent_client_protocol::ConnectionTo<Client> {
    fn create_terminal(
        &self,
        request: CreateTerminalRequest,
    ) -> AcpRequestFuture<'_, CreateTerminalResponse> {
        Box::pin(self.send_request(request).block_task())
    }
}

impl TerminalOutputRequester for agent_client_protocol::ConnectionTo<Client> {
    fn terminal_output(
        &self,
        request: TerminalOutputRequest,
    ) -> AcpRequestFuture<'_, TerminalOutputResponse> {
        Box::pin(self.send_request(request).block_task())
    }
}

impl WaitForTerminalExitRequester for agent_client_protocol::ConnectionTo<Client> {
    fn wait_for_terminal_exit(
        &self,
        request: WaitForTerminalExitRequest,
    ) -> AcpRequestFuture<'_, WaitForTerminalExitResponse> {
        Box::pin(self.send_request(request).block_task())
    }
}

impl ReleaseTerminalRequester for agent_client_protocol::ConnectionTo<Client> {
    fn release_terminal(
        &self,
        request: ReleaseTerminalRequest,
    ) -> AcpRequestFuture<'_, ReleaseTerminalResponse> {
        Box::pin(self.send_request(request).block_task())
    }
}

impl KillTerminalRequester for agent_client_protocol::ConnectionTo<Client> {
    fn kill_terminal(
        &self,
        request: KillTerminalRequest,
    ) -> AcpRequestFuture<'_, KillTerminalResponse> {
        Box::pin(self.send_request(request).block_task())
    }
}

pub(crate) trait ToolCallRequester:
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
    ) -> AcpRequestFuture<'_, RequestPermissionResponse>;
}

// Production always speaks as the agent, so the tool layer is handed a
// `ConnectionTo<Client>` (see `serve_with_transport`). The `ConnectionTo<Agent>`
// direction exists only as a test seam: `read_file_tool_execution` is covered
// against a real client-backed connection (whose handle is a
// `ConnectionTo<Agent>`). Only `read_text_file` is needed for that seam, so the
// other requester traits are intentionally implemented for `ConnectionTo<Client>`
// alone.
impl ReadTextFileRequester for agent_client_protocol::ConnectionTo<Agent> {
    fn read_text_file(
        &self,
        request: ReadTextFileRequest,
    ) -> AcpRequestFuture<'_, ReadTextFileResponse> {
        Box::pin(self.send_request(request).block_task())
    }
}

impl ReadTextFileRequester for agent_client_protocol::ConnectionTo<Client> {
    fn read_text_file(
        &self,
        request: ReadTextFileRequest,
    ) -> AcpRequestFuture<'_, ReadTextFileResponse> {
        Box::pin(self.send_request(request).block_task())
    }
}

/// Recover from a known ACP-client interop quirk where a successful
/// `fs/write_text_file` is reported with a JSON-null result payload.
///
/// Some ACP clients answer `fs/write_text_file` with a literal `null` result.
/// Deserializing that `null` into [`WriteTextFileResponse`] fails with a
/// [`ParseError`](agent_client_protocol::ErrorCode::ParseError) whose `data`
/// records `{"json": null, "phase": "deserialization"}`. That specific failure is
/// equivalent to an empty success response, so it is mapped back to
/// `Ok(WriteTextFileResponse::new())`. Every other error is propagated unchanged.
fn recover_null_write_response(
    result: Result<WriteTextFileResponse, agent_client_protocol::Error>,
) -> Result<WriteTextFileResponse, agent_client_protocol::Error> {
    result.or_else(|err| {
        let is_null_payload_deser_failure = err.code
            == agent_client_protocol::ErrorCode::ParseError
            && err.data.as_ref().is_some_and(|d| {
                d.get("json").is_some_and(serde_json::Value::is_null)
                    && d.get("phase").and_then(serde_json::Value::as_str) == Some("deserialization")
            });
        if is_null_payload_deser_failure {
            Ok(WriteTextFileResponse::new())
        } else {
            Err(err)
        }
    })
}

impl WriteTextFileRequester for agent_client_protocol::ConnectionTo<Client> {
    fn write_text_file(
        &self,
        request: WriteTextFileRequest,
    ) -> AcpRequestFuture<'_, WriteTextFileResponse> {
        Box::pin(async move {
            recover_null_write_response(self.send_request(request).block_task().await)
        })
    }
}

impl PermissionRequester for agent_client_protocol::ConnectionTo<Client> {
    fn request_permission(
        &self,
        request: RequestPermissionRequest,
    ) -> AcpRequestFuture<'_, RequestPermissionResponse> {
        Box::pin(self.send_request(request).block_task())
    }
}

/// Run the ACP stdio server with the given transport, state, LLM client, and tool
/// registry.
///
/// The builder pattern with many request handler registrations unavoidably spans
/// many lines. Each handler is factored into a separate function for testability.
#[allow(clippy::too_many_lines)]
pub(crate) async fn serve_with_transport(
    transport: impl ConnectTo<Agent> + 'static,
    state: Arc<Mutex<AdapterState>>,
    llm_client: Arc<dyn LlmClient>,
    tool_registry: Arc<dyn ToolRegistry>,
    max_turn_requests: NonZeroUsize,
) -> Result<(), agent_client_protocol::Error> {
    let persistence = FilesystemSessionStore::from_default_state_dir()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let store = SessionStore::new(state).with_persistence(persistence);
    let initialize_store = store.clone();
    let new_session_store = store.clone();
    let load_session_store = store.clone();
    let resume_session_store = store.clone();
    let set_mode_store = store.clone();
    let set_config_store = store.clone();
    let prompt_store = store.clone();
    let prompt_client = Arc::clone(&llm_client);
    let prompt_tools = Arc::clone(&tool_registry);
    let cancel_store = store.clone();
    let list_sessions_store = store.clone();
    let close_session_store = store.clone();

    Agent
        .builder()
        .name("deepseek-acp-adapter")
        .on_receive_request(
            async move |request: InitializeRequest, responder, _cx| {
                responder.respond(handle_initialize_request(&initialize_store, request)?)
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
                let session_store = new_session_store.clone();
                let connection = cx.clone();

                cx.spawn(async move {
                    let response =
                        handle_new_session_request_connected(&session_store, &request).await?;
                    let session_id = response.session_id.clone();

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
            async move |request: LoadSessionRequest, responder, cx| {
                let session_store = load_session_store.clone();
                let connection = cx.clone();

                cx.spawn(async move {
                    let result =
                        handle_load_session_request(&session_store, &request, |notification| {
                            connection.send_notification(notification)
                        })
                        .await;

                    responder.respond_with_result(result)?;
                    Ok(())
                })?;

                Ok(())
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: ResumeSessionRequest, responder, cx| {
                let session_store = resume_session_store.clone();

                cx.spawn(async move {
                    let result = handle_resume_session_request(&session_store, &request).await;

                    responder.respond_with_result(result)?;
                    Ok(())
                })?;

                Ok(())
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: SetSessionModeRequest, responder, cx| {
                let connection = cx.clone();
                responder.respond(handle_set_session_mode_request_notifying(
                    &set_mode_store,
                    &request,
                    |notification| connection.send_notification(notification),
                )?)
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: SetSessionConfigOptionRequest, responder, cx| {
                let connection = cx.clone();
                responder.respond(handle_set_session_config_option_request_notifying(
                    &set_config_store,
                    &request,
                    |notification| connection.send_notification(notification),
                )?)
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: PromptRequest, responder, cx| {
                let store = prompt_store.clone();
                let client = Arc::clone(&prompt_client);
                let tools = Arc::clone(&prompt_tools);
                let connection = cx.clone();

                cx.spawn(async move {
                    let result = handle_prompt_request(
                        &store,
                        client.as_ref(),
                        tools.as_ref(),
                        Some(&connection as &dyn ToolCallRequester),
                        request,
                        max_turn_requests,
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
        .on_receive_request(
            async move |request: ListSessionsRequest, responder, _cx| {
                responder.respond(handle_list_sessions_request(
                    &list_sessions_store,
                    &request,
                )?)
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: CloseSessionRequest, responder, _cx| {
                responder.respond(handle_close_session_request(
                    &close_session_store,
                    &request,
                )?)
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |_request: LogoutRequest, responder, _cx| {
                responder.respond(handle_logout_request())
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_notification(
            async move |notification: CancelNotification, _cx| {
                cancel_store.cancel_active_turn(&notification.session_id)
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_to(transport)
        .await
}

pub(crate) fn handle_initialize_request(
    store: &SessionStore,
    request: InitializeRequest,
) -> Result<InitializeResponse, agent_client_protocol::Error> {
    store.record_client_capabilities(request.client_capabilities)?;
    Ok(build_initialize_response(request.protocol_version))
}

pub(crate) fn handle_authenticate_request() -> AuthenticateResponse {
    AuthenticateResponse::new()
}

pub(crate) fn handle_list_sessions_request(
    store: &SessionStore,
    request: &ListSessionsRequest,
) -> Result<ListSessionsResponse, agent_client_protocol::Error> {
    let sessions = store.list_sessions(request.cwd.as_deref())?;
    Ok(ListSessionsResponse::new(sessions))
}

pub(crate) fn handle_close_session_request(
    store: &SessionStore,
    request: &CloseSessionRequest,
) -> Result<CloseSessionResponse, agent_client_protocol::Error> {
    let existed = store.remove_session(&request.session_id)?;
    if !existed {
        return Err(agent_client_protocol::Error::invalid_params()
            .data(format!("unknown session id: {}", request.session_id.0)));
    }

    Ok(CloseSessionResponse::new())
}

/// Handle a `logout` request.
///
/// This adapter has no persistent auth state, so logout is a no-op.
pub(crate) fn handle_logout_request() -> LogoutResponse {
    LogoutResponse::new()
}

pub(crate) fn handle_new_session_request(
    store: &SessionStore,
    request: &NewSessionRequest,
) -> Result<NewSessionResponse, agent_client_protocol::Error> {
    if !request.mcp_servers.is_empty() {
        return Err(agent_client_protocol::Error::invalid_params()
            .data("MCP servers require the async session setup path"));
    }
    insert_session_record(store, request, Vec::new())
}

pub(crate) async fn handle_new_session_request_connected(
    store: &SessionStore,
    request: &NewSessionRequest,
) -> Result<NewSessionResponse, agent_client_protocol::Error> {
    validate_session_paths(request)?;
    let mcp_sessions = connect_mcp_sessions(&request.mcp_servers).await?;
    insert_session_record(store, request, mcp_sessions)
}

pub(crate) async fn handle_load_session_request(
    store: &SessionStore,
    request: &LoadSessionRequest,
    mut notify: impl FnMut(SessionNotification) -> Result<(), agent_client_protocol::Error>,
) -> Result<LoadSessionResponse, agent_client_protocol::Error> {
    validate_load_session_paths(request)?;
    let (session_id, history) =
        restore_persisted_session(store, &request.session_id, &request.cwd).await?;
    replay_session_history(&session_id, &history, &mut notify)?;

    Ok(LoadSessionResponse::new()
        .modes(default_session_modes())
        .config_options(store.session_config_options(&session_id)?))
}

pub(crate) async fn handle_resume_session_request(
    store: &SessionStore,
    request: &ResumeSessionRequest,
) -> Result<ResumeSessionResponse, agent_client_protocol::Error> {
    validate_resume_session_paths(request)?;
    let (session_id, _) =
        restore_persisted_session(store, &request.session_id, &request.cwd).await?;

    Ok(ResumeSessionResponse::new()
        .modes(default_session_modes())
        .config_options(store.session_config_options(&session_id)?))
}

async fn restore_persisted_session(
    store: &SessionStore,
    requested_session_id: &SessionId,
    cwd: &std::path::Path,
) -> Result<(SessionId, Vec<ChatMessage>), agent_client_protocol::Error> {
    let persisted = store.load_persisted_record(requested_session_id)?;
    if persisted.meta.cwd != cwd {
        return Err(agent_client_protocol::Error::invalid_params().data(format!(
            "session {} was persisted for cwd {}, not {}",
            requested_session_id.0,
            persisted.meta.cwd.display(),
            cwd.display()
        )));
    }

    let mcp_sessions = connect_mcp_sessions(&persisted.meta.mcp_servers).await?;
    let session_id = SessionId::new(persisted.meta.session_id.clone());
    if session_id != *requested_session_id {
        return Err(agent_client_protocol::Error::invalid_params().data(format!(
            "persisted session id {} does not match requested session id {}",
            session_id.0, requested_session_id.0
        )));
    }
    let history = persisted.history;
    store.insert_session(
        session_id.clone(),
        SessionRecord {
            cwd: persisted.meta.cwd,
            additional_directories: persisted.meta.additional_directories,
            history: history.clone(),
            active_turn: None,
            mode: persisted.meta.mode,
            model: persisted.meta.model,
            reasoning_effort: persisted.meta.reasoning_effort,
            permission_allow_always: HashSet::new(),
            mcp_servers: persisted.meta.mcp_servers,
            mcp_sessions,
        },
    )?;

    Ok((session_id, history))
}

fn insert_session_record(
    store: &SessionStore,
    request: &NewSessionRequest,
    mcp_sessions: Vec<McpSession>,
) -> Result<NewSessionResponse, agent_client_protocol::Error> {
    validate_session_paths(request)?;
    let session_id = format!("session-{}", Uuid::new_v4());
    let default_model = store.default_model()?;
    let sid: SessionId = session_id.clone().into();
    store.insert_session(
        sid.clone(),
        SessionRecord {
            cwd: request.cwd.clone(),
            additional_directories: request.additional_directories.clone(),
            history: Vec::new(),
            active_turn: None,
            mode: PermissionPosture::Ask,
            model: default_model,
            reasoning_effort: ReasoningEffort::High,
            permission_allow_always: HashSet::new(),
            mcp_servers: request.mcp_servers.clone(),
            mcp_sessions,
        },
    )?;

    store.lookup_session(&sid)?;

    Ok(NewSessionResponse::new(session_id)
        .modes(default_session_modes())
        .config_options(store.session_config_options(&sid)?))
}

fn replay_session_history(
    session_id: &SessionId,
    history: &[ChatMessage],
    notify: &mut impl FnMut(SessionNotification) -> Result<(), agent_client_protocol::Error>,
) -> Result<(), agent_client_protocol::Error> {
    for message in history {
        match message.role() {
            MessageRole::User => notify(session_notification(
                session_id.clone(),
                SessionUpdate::UserMessageChunk(ContentChunk::new(ContentBlock::from(
                    message.content().to_string(),
                ))),
            ))?,
            MessageRole::Assistant => {
                replay_assistant_message(session_id, history, message, notify)?;
            }
            MessageRole::System | MessageRole::Tool => {}
        }
    }
    Ok(())
}

fn replay_assistant_message(
    session_id: &SessionId,
    history: &[ChatMessage],
    message: &ChatMessage,
    notify: &mut impl FnMut(SessionNotification) -> Result<(), agent_client_protocol::Error>,
) -> Result<(), agent_client_protocol::Error> {
    if !message.content().is_empty() {
        notify(session_notification(
            session_id.clone(),
            SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::from(
                message.content().to_string(),
            ))),
        ))?;
    }

    for tool_call in message.tool_calls() {
        notify(session_notification(
            session_id.clone(),
            SessionUpdate::ToolCall(replayed_tool_call(tool_call, history)),
        ))?;
    }

    Ok(())
}

fn replayed_tool_call(tool_call: &DeepSeekToolCall, history: &[ChatMessage]) -> AcpToolCall {
    let output = tool_result_content(tool_call.id(), history).unwrap_or_default();
    AcpToolCall::new(tool_call.id().to_string(), tool_call.name().to_string())
        .status(ToolCallStatus::Completed)
        .raw_input(tool_raw_input(tool_call))
        .raw_output(serde_json::json!({ "content": output }))
        .content(vec![ToolCallContent::from(output)])
}

fn tool_result_content(tool_call_id: &str, history: &[ChatMessage]) -> Option<String> {
    history.iter().find_map(|message| {
        if message.role() == MessageRole::Tool && message.tool_call_id() == Some(tool_call_id) {
            Some(message.content().to_string())
        } else {
            None
        }
    })
}

#[cfg(test)]
pub(crate) fn handle_set_session_mode_request(
    store: &SessionStore,
    request: &SetSessionModeRequest,
) -> Result<SetSessionModeResponse, agent_client_protocol::Error> {
    handle_set_session_mode_request_notifying(store, request, |_| Ok(()))
}

pub(crate) fn handle_set_session_mode_request_notifying(
    store: &SessionStore,
    request: &SetSessionModeRequest,
    mut notify: impl FnMut(SessionNotification) -> Result<(), agent_client_protocol::Error>,
) -> Result<SetSessionModeResponse, agent_client_protocol::Error> {
    let Some(mode) = PermissionPosture::from_mode_id(&request.mode_id) else {
        return Err(agent_client_protocol::Error::invalid_params()
            .data(format!("unsupported session mode: {}", request.mode_id.0)));
    };

    store.set_mode(&request.session_id, mode)?;
    notify(session_notification(
        request.session_id.clone(),
        SessionUpdate::CurrentModeUpdate(CurrentModeUpdate::new(request.mode_id.clone())),
    ))?;
    Ok(SetSessionModeResponse::new())
}

#[cfg(test)]
pub(crate) fn handle_set_session_config_option_request(
    store: &SessionStore,
    request: &SetSessionConfigOptionRequest,
) -> Result<SetSessionConfigOptionResponse, agent_client_protocol::Error> {
    handle_set_session_config_option_request_notifying(store, request, |_| Ok(()))
}

pub(crate) fn handle_set_session_config_option_request_notifying(
    store: &SessionStore,
    request: &SetSessionConfigOptionRequest,
    mut notify: impl FnMut(SessionNotification) -> Result<(), agent_client_protocol::Error>,
) -> Result<SetSessionConfigOptionResponse, agent_client_protocol::Error> {
    let value = config_value_id(&request.value)?;

    match request.config_id.0.as_ref() {
        SESSION_CONFIG_MODE_ID => {
            let mode_id = agent_client_protocol::schema::SessionModeId::new(value.0.clone());
            let Some(mode) = PermissionPosture::from_mode_id(&mode_id) else {
                return Err(agent_client_protocol::Error::invalid_params()
                    .data(format!("unsupported session mode: {}", value.0)));
            };
            store.set_mode(&request.session_id, mode)?;
        }
        SESSION_CONFIG_MODEL_ID => {
            let model = value.0.as_ref();
            store.with_session(&request.session_id, |session| {
                validate_session_model(session, model)?;
                Ok(())
            })?;
            store.set_model(&request.session_id, model.to_string())?;
        }
        SESSION_CONFIG_REASONING_EFFORT_ID => {
            let Some(effort) = ReasoningEffort::from_value_id(value) else {
                return Err(agent_client_protocol::Error::invalid_params()
                    .data(format!("unsupported reasoning effort: {}", value.0)));
            };
            store.set_reasoning_effort(&request.session_id, effort)?;
        }
        _ => {
            return Err(agent_client_protocol::Error::invalid_params().data(format!(
                "unsupported session config option: {}",
                request.config_id.0
            )));
        }
    }

    let config_options = store.session_config_options(&request.session_id)?;
    notify(session_notification(
        request.session_id.clone(),
        SessionUpdate::ConfigOptionUpdate(ConfigOptionUpdate::new(config_options.clone())),
    ))?;
    Ok(SetSessionConfigOptionResponse::new(config_options))
}

pub(crate) fn config_value_id(
    value: &SessionConfigOptionValue,
) -> Result<&SessionConfigValueId, agent_client_protocol::Error> {
    value.as_value_id().ok_or_else(|| {
        agent_client_protocol::Error::invalid_params()
            .data("session config option requires a selectable value id")
    })
}

pub(crate) async fn handle_prompt_request(
    store: &SessionStore,
    llm_client: &dyn LlmClient,
    tool_registry: &dyn ToolRegistry,
    connection: Option<&dyn ToolCallRequester>,
    request: PromptRequest,
    max_turn_requests: NonZeroUsize,
    notify: impl FnMut(SessionNotification) -> Result<(), agent_client_protocol::Error>,
) -> Result<PromptResponse, agent_client_protocol::Error> {
    crate::turn::handle_prompt_request(
        store,
        llm_client,
        tool_registry,
        connection,
        request,
        max_turn_requests,
        notify,
    )
    .await
}

pub(crate) fn build_initialize_response(_protocol_version: ProtocolVersion) -> InitializeResponse {
    InitializeResponse::new(ProtocolVersion::LATEST)
        .agent_capabilities(
            AgentCapabilities::new()
                .load_session(true)
                .prompt_capabilities(
                    agent_client_protocol::schema::PromptCapabilities::new().embedded_context(true),
                )
                .mcp_capabilities(McpCapabilities::new().http(true))
                .session_capabilities(
                    SessionCapabilities::new()
                        .additional_directories(SessionAdditionalDirectoriesCapabilities::new())
                        .list(SessionListCapabilities::new())
                        .resume(SessionResumeCapabilities::new())
                        .close(SessionCloseCapabilities::new()),
                )
                .auth(AgentAuthCapabilities::new().logout(LogoutCapabilities::new())),
        )
        .agent_info(Implementation::new(ADAPTER_NAME, ADAPTER_VERSION))
}

pub(crate) fn validate_session_paths(
    request: &NewSessionRequest,
) -> Result<(), agent_client_protocol::Error> {
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

pub(crate) fn validate_load_session_paths(
    request: &LoadSessionRequest,
) -> Result<(), agent_client_protocol::Error> {
    if !request.cwd.is_absolute() {
        return Err(agent_client_protocol::Error::invalid_params()
            .data(format!("cwd must be absolute: {}", request.cwd.display())));
    }

    for path in &request.additional_directories {
        if !path.is_absolute() {
            return Err(agent_client_protocol::Error::invalid_params().data(format!(
                "additional directory must be absolute: {}",
                path.display()
            )));
        }
    }

    Ok(())
}

pub(crate) fn validate_resume_session_paths(
    request: &ResumeSessionRequest,
) -> Result<(), agent_client_protocol::Error> {
    if !request.cwd.is_absolute() {
        return Err(agent_client_protocol::Error::invalid_params()
            .data("session cwd must be an absolute path"));
    }
    for directory in &request.additional_directories {
        if !directory.is_absolute() {
            return Err(agent_client_protocol::Error::invalid_params()
                .data("additional directories must be absolute paths"));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests;
