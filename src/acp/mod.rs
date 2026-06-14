//! ACP-facing transport registration and request handling.

use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::{
    AgentAuthCapabilities, AgentCapabilities, AuthenticateRequest, AuthenticateResponse,
    AvailableCommandsUpdate, CancelNotification, CloseSessionRequest, CloseSessionResponse,
    ConfigOptionUpdate, ContentBlock, ContentChunk, CurrentModeUpdate, DeleteSessionRequest,
    DeleteSessionResponse, Implementation, InitializeRequest, InitializeResponse,
    ListSessionsRequest, ListSessionsResponse, LoadSessionRequest, LoadSessionResponse,
    LogoutCapabilities, LogoutRequest, LogoutResponse, McpCapabilities, MessageId,
    NewSessionRequest, NewSessionResponse, PromptRequest, PromptResponse, ProtocolVersion,
    ResumeSessionRequest, ResumeSessionResponse, SessionAdditionalDirectoriesCapabilities,
    SessionCapabilities, SessionCloseCapabilities, SessionConfigOptionValue, SessionConfigValueId,
    SessionDeleteCapabilities, SessionId, SessionListCapabilities, SessionNotification,
    SessionResumeCapabilities, SessionUpdate, SetSessionConfigOptionRequest,
    SetSessionConfigOptionResponse, SetSessionModeRequest, SetSessionModeResponse,
    ToolCall as AcpToolCall, ToolCallContent, ToolCallStatus,
};
use agent_client_protocol::{Agent, ConnectTo};
use deepseek_acp_adapter::deepseek::{
    ChatMessage, LlmClient, MessageRole, ToolCall as DeepSeekToolCall,
};
use uuid::Uuid;

use crate::tools::ToolRegistry;
use crate::{
    ADAPTER_NAME, ADAPTER_VERSION, AdapterState, FilesystemSessionStore, McpSession,
    PermissionPosture, ReasoningEffort, SESSION_CONFIG_MODE_ID, SESSION_CONFIG_MODEL_ID,
    SESSION_CONFIG_REASONING_EFFORT_ID, SessionRecord, SessionStore, adapter_available_commands,
    connect_mcp_sessions, default_session_modes, session_notification, tool_raw_input,
    validate_session_model,
};

pub(crate) mod requesters;
pub(crate) use requesters::*;

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
    serve_with_transport_and_state_dir(
        transport,
        state,
        llm_client,
        tool_registry,
        max_turn_requests,
        None,
    )
    .await
}

pub(crate) async fn serve_with_transport_and_state_dir(
    transport: impl ConnectTo<Agent> + 'static,
    state: Arc<Mutex<AdapterState>>,
    llm_client: Arc<dyn LlmClient>,
    tool_registry: Arc<dyn ToolRegistry>,
    max_turn_requests: NonZeroUsize,
    state_dir: Option<std::path::PathBuf>,
) -> Result<(), agent_client_protocol::Error> {
    let persistence = match state_dir {
        Some(dir) => FilesystemSessionStore::new(dir),
        None => FilesystemSessionStore::from_default_state_dir()
            .map_err(agent_client_protocol::Error::into_internal_error)?,
    };
    serve_with_transport_impl(
        transport,
        state,
        llm_client,
        tool_registry,
        max_turn_requests,
        persistence,
    )
    .await
}

// Handler registration is inherently repetitive and linear — splitting would require
// threading a builder through a chain of partial-application functions, adding more
// complexity than the current setup.
#[allow(clippy::too_many_lines)]
async fn serve_with_transport_impl(
    transport: impl ConnectTo<Agent> + 'static,
    state: Arc<Mutex<AdapterState>>,
    llm_client: Arc<dyn LlmClient>,
    tool_registry: Arc<dyn ToolRegistry>,
    max_turn_requests: NonZeroUsize,
    persistence: FilesystemSessionStore,
) -> Result<(), agent_client_protocol::Error> {
    // Handler registration is inherently repetitive: each request type requires its own
    // .on_receive_request() call with transport/state setup. Extracting individual handlers
    // would require passing the builder through a chain of functions or storing it in a
    // struct, adding more complexity and boilerplate than the current linear setup.
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
    let delete_session_store = store.clone();

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

                let session_id = request.session_id.clone();
                tracing::debug!(session_id = %session_id, "received session/prompt request");

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

                    tracing::debug!(
                        session_id = %session_id,
                        result_is_ok = result.is_ok(),
                        "session/prompt request completed"
                    );
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
            async move |request: DeleteSessionRequest, responder, _cx| {
                responder.respond(handle_delete_session_request(
                    &delete_session_store,
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
                cancel_store.cancel_active_turn(&notification.session_id)?;
                Ok(())
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
    tracing::debug!(
        session_count = sessions.len(),
        cwd = ?request.cwd.as_deref(),
        "returning sessions"
    );
    for session in &sessions {
        tracing::debug!(
            session_id = %session.session_id.0,
            cwd = %session.cwd.display(),
            "session returned"
        );
    }
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

pub(crate) fn handle_delete_session_request(
    store: &SessionStore,
    request: &DeleteSessionRequest,
) -> Result<DeleteSessionResponse, agent_client_protocol::Error> {
    let existed = store.delete_session(&request.session_id)?;
    if !existed {
        return Err(agent_client_protocol::Error::invalid_params()
            .data(format!("unknown session id: {}", request.session_id.0)));
    }

    Ok(DeleteSessionResponse::new())
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

    let mut response = LoadSessionResponse::new()
        .modes(default_session_modes())
        .config_options(store.session_config_options(&session_id)?);
    if let Some(meta) = store.session_meta(&session_id) {
        response = response.meta(meta);
    }
    Ok(response)
}

pub(crate) async fn handle_resume_session_request(
    store: &SessionStore,
    request: &ResumeSessionRequest,
) -> Result<ResumeSessionResponse, agent_client_protocol::Error> {
    validate_resume_session_paths(request)?;
    let (session_id, _) =
        restore_persisted_session(store, &request.session_id, &request.cwd).await?;

    let mut response = ResumeSessionResponse::new()
        .modes(default_session_modes())
        .config_options(store.session_config_options(&session_id)?);
    if let Some(meta) = store.session_meta(&session_id) {
        response = response.meta(meta);
    }
    Ok(response)
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

    // Backward compat: old persisted sessions may not have title/updated_at.
    let title = persisted
        .meta
        .title
        .clone()
        .filter(|t| !t.is_empty())
        .unwrap_or_else(|| crate::derive_session_title(&history));
    let updated_at = persisted
        .meta
        .updated_at
        .clone()
        .unwrap_or_else(crate::iso_timestamp_now);

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
            title,
            updated_at,
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
    let now = crate::iso_timestamp_now();
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
            title: String::new(),
            updated_at: now,
        },
    )?;

    store.lookup_session(&sid)?;

    let mut response = NewSessionResponse::new(session_id)
        .modes(default_session_modes())
        .config_options(store.session_config_options(&sid)?);
    if let Some(meta) = store.session_meta(&sid) {
        response = response.meta(meta);
    }
    Ok(response)
}

fn replay_session_history(
    session_id: &SessionId,
    history: &[ChatMessage],
    notify: &mut impl FnMut(SessionNotification) -> Result<(), agent_client_protocol::Error>,
) -> Result<(), agent_client_protocol::Error> {
    for message in history {
        match message.role() {
            MessageRole::User => {
                let message_id: MessageId = Uuid::new_v4().to_string().into();
                notify(session_notification(
                    session_id.clone(),
                    SessionUpdate::UserMessageChunk(
                        ContentChunk::new(ContentBlock::from(message.content().to_string()))
                            .message_id(message_id),
                    ),
                ))?;
            }
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
        let message_id: MessageId = Uuid::new_v4().to_string().into();
        notify(session_notification(
            session_id.clone(),
            SessionUpdate::AgentMessageChunk(
                ContentChunk::new(ContentBlock::from(message.content().to_string()))
                    .message_id(message_id),
            ),
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
    AcpToolCall::new(
        tool_call.id().to_string(),
        crate::turn::tool_call_title(tool_call),
    )
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
    .map_err(Into::into)
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
                        .delete(SessionDeleteCapabilities::new())
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
