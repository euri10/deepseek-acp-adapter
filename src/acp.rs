//! ACP-facing transport registration and request handling.

use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::{
    AgentAuthCapabilities, AgentCapabilities, AuthenticateRequest, AuthenticateResponse,
    AvailableCommandsUpdate, CancelNotification, CloseSessionRequest, CloseSessionResponse,
    CreateTerminalRequest, CreateTerminalResponse, Implementation, InitializeRequest,
    InitializeResponse, ListSessionsRequest, ListSessionsResponse, LogoutCapabilities,
    LogoutRequest, LogoutResponse, NewSessionRequest, NewSessionResponse, PromptRequest,
    PromptResponse, ProtocolVersion, ReadTextFileRequest, ReadTextFileResponse,
    ReleaseTerminalRequest, ReleaseTerminalResponse, RequestPermissionRequest,
    RequestPermissionResponse, SessionAdditionalDirectoriesCapabilities, SessionCapabilities,
    SessionCloseCapabilities, SessionConfigOptionValue, SessionConfigValueId, SessionId,
    SessionListCapabilities, SessionNotification, SessionUpdate, SetSessionConfigOptionRequest,
    SetSessionConfigOptionResponse, SetSessionModeRequest, SetSessionModeResponse,
    TerminalOutputRequest, TerminalOutputResponse, WaitForTerminalExitRequest,
    WaitForTerminalExitResponse, WriteTextFileRequest, WriteTextFileResponse,
};
use agent_client_protocol::{Agent, Client, ConnectTo};
use deepseek_acp_adapter::deepseek::LlmClient;
use futures_util::future::BoxFuture;
use uuid::Uuid;

use crate::tools::ToolRegistry;
use crate::{
    ADAPTER_NAME, ADAPTER_VERSION, AdapterState, McpSession, PermissionPosture, ReasoningEffort,
    SESSION_CONFIG_MODE_ID, SESSION_CONFIG_MODEL_ID, SESSION_CONFIG_REASONING_EFFORT_ID,
    SessionRecord, SessionStore, adapter_available_commands, connect_mcp_sessions,
    default_session_modes, session_notification, validate_session_model,
};

pub(crate) trait ReadTextFileRequester: Send + Sync {
    fn read_text_file(
        &self,
        request: ReadTextFileRequest,
    ) -> BoxFuture<'_, Result<ReadTextFileResponse, agent_client_protocol::Error>>;
}

pub(crate) trait WriteTextFileRequester: Send + Sync {
    fn write_text_file(
        &self,
        request: WriteTextFileRequest,
    ) -> BoxFuture<'_, Result<WriteTextFileResponse, agent_client_protocol::Error>>;
}

/// Trait for creating a terminal via ACP client `terminal/create`.
pub(crate) trait CreateTerminalRequester: Send + Sync {
    /// Create a terminal and execute a command.
    fn create_terminal(
        &self,
        request: CreateTerminalRequest,
    ) -> BoxFuture<'_, Result<CreateTerminalResponse, agent_client_protocol::Error>>;
}

/// Trait for getting terminal output via ACP client `terminal/output`.
pub(crate) trait TerminalOutputRequester: Send + Sync {
    /// Get the current output and status of a terminal.
    fn terminal_output(
        &self,
        request: TerminalOutputRequest,
    ) -> BoxFuture<'_, Result<TerminalOutputResponse, agent_client_protocol::Error>>;
}

/// Trait for waiting for terminal exit via ACP client `terminal/wait_for_exit`.
pub(crate) trait WaitForTerminalExitRequester: Send + Sync {
    /// Wait for a terminal command to exit.
    fn wait_for_terminal_exit(
        &self,
        request: WaitForTerminalExitRequest,
    ) -> BoxFuture<'_, Result<WaitForTerminalExitResponse, agent_client_protocol::Error>>;
}

/// Trait for releasing a terminal via ACP client `terminal/release`.
pub(crate) trait ReleaseTerminalRequester: Send + Sync {
    /// Release a terminal and free its resources.
    fn release_terminal(
        &self,
        request: ReleaseTerminalRequest,
    ) -> BoxFuture<'_, Result<ReleaseTerminalResponse, agent_client_protocol::Error>>;
}

/// Combined trait for all terminal operations.
pub(crate) trait TerminalRequester:
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
        Box::pin(async move {
            self.send_request(request)
                .block_task()
                .await
                .or_else(|err| {
                    let is_null_payload_deser_failure = err.code
                        == agent_client_protocol::ErrorCode::ParseError
                        && err.data.as_ref().is_some_and(|d| {
                            d.get("json").is_some_and(serde_json::Value::is_null)
                                && d.get("phase").and_then(serde_json::Value::as_str)
                                    == Some("deserialization")
                        });
                    if is_null_payload_deser_failure {
                        Ok(WriteTextFileResponse::new())
                    } else {
                        Err(err)
                    }
                })
        })
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
    let store = SessionStore::new(state);
    let initialize_store = store.clone();
    let new_session_store = store.clone();
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
            async move |request: SetSessionModeRequest, responder, _cx| {
                responder.respond(handle_set_session_mode_request(&set_mode_store, &request)?)
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: SetSessionConfigOptionRequest, responder, _cx| {
                responder.respond(handle_set_session_config_option_request(
                    &set_config_store,
                    &request,
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
    _request: &ListSessionsRequest,
) -> Result<ListSessionsResponse, agent_client_protocol::Error> {
    let sessions = store.list_sessions()?;
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
            mcp_sessions,
        },
    )?;

    store.lookup_session(&sid)?;

    Ok(NewSessionResponse::new(session_id)
        .modes(default_session_modes())
        .config_options(store.session_config_options(&sid)?))
}

pub(crate) fn handle_set_session_mode_request(
    store: &SessionStore,
    request: &SetSessionModeRequest,
) -> Result<SetSessionModeResponse, agent_client_protocol::Error> {
    let Some(mode) = PermissionPosture::from_mode_id(&request.mode_id) else {
        return Err(agent_client_protocol::Error::invalid_params()
            .data(format!("unsupported session mode: {}", request.mode_id.0)));
    };

    store.set_mode(&request.session_id, mode)?;
    Ok(SetSessionModeResponse::new())
}

pub(crate) fn handle_set_session_config_option_request(
    store: &SessionStore,
    request: &SetSessionConfigOptionRequest,
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

    Ok(SetSessionConfigOptionResponse::new(
        store.session_config_options(&request.session_id)?,
    ))
}

fn config_value_id(
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
                .load_session(false)
                .prompt_capabilities(agent_client_protocol::schema::PromptCapabilities::new())
                .session_capabilities(
                    SessionCapabilities::new()
                        .additional_directories(SessionAdditionalDirectoriesCapabilities::new())
                        .list(SessionListCapabilities::new())
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
