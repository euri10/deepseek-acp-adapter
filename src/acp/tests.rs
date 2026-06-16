#![allow(clippy::indexing_slicing)]
use super::{
    build_initialize_response, config_value_id, handle_authenticate_request,
    handle_close_session_request, handle_delete_session_request, handle_initialize_request,
    handle_list_sessions_request, handle_load_session_request, handle_logout_request,
    handle_new_session_request, handle_new_session_request_connected, handle_prompt_request,
    handle_resume_session_request, handle_set_session_config_option_request,
    handle_set_session_config_option_request_notifying, handle_set_session_mode_request,
    handle_set_session_mode_request_notifying, replay_assistant_message, replay_session_history,
    replayed_tool_call, restore_persisted_session, serve_with_transport_and_state_dir,
    tool_result_content, validate_load_session_paths, validate_resume_session_paths,
    validate_session_paths,
};
use crate::dev::MockLlmClient;
use crate::session::{
    AdapterState, DEFAULT_MAX_TURN_REQUESTS, PERMISSION_ALLOW_ALWAYS_OPTION_ID,
    PERMISSION_ALLOW_ONCE_OPTION_ID, PERMISSION_REJECT_ONCE_OPTION_ID, PermissionDecision,
    PermissionPosture, ReasoningEffort, SESSION_CONFIG_MODE_ID, SESSION_CONFIG_MODEL_ID,
    SESSION_CONFIG_REASONING_EFFORT_ID, SessionRecord, SessionStore, initial_model_from_env,
    model_select_options, request_tool_permission, validate_session_model,
};
use crate::session_store::{FilesystemSessionStore, PersistedSessionMeta};
use crate::test_utils::*;
use crate::tools::{
    AdapterToolRegistry, EmptyToolRegistry, ToolContext, ToolRegistry, require_tool_permission,
};
use agent_client_protocol::schema::{
    ClientCapabilities, CloseSessionRequest, ContentBlock, DeleteSessionRequest,
    FileSystemCapabilities, Implementation, InitializeRequest, ListSessionsRequest,
    LoadSessionRequest, NewSessionRequest, PermissionOptionKind, PromptRequest, ProtocolVersion,
    RequestPermissionOutcome, RequestPermissionResponse, ResumeSessionRequest,
    SelectedPermissionOutcome, SessionConfigOptionCategory, SessionConfigOptionValue,
    SessionUpdate, SetSessionConfigOptionRequest, SetSessionModeRequest, ToolCallContent,
    ToolCallStatus, ToolKind,
};
use agent_client_protocol::{Channel, Client};
use deepseek_acp_adapter::deepseek::{ChatMessage, LlmClient};
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

fn test_store() -> SessionStore {
    SessionStore::new(Arc::new(Mutex::new(AdapterState::default())))
}

fn select_current_value(
    options: &[agent_client_protocol::schema::SessionConfigOption],
    id: &str,
) -> Result<String, agent_client_protocol::Error> {
    crate::test_utils::select_current_value(options, id)
}

#[test]
fn new_session_with_mcp_servers_rejected_synchronously() -> Result<(), agent_client_protocol::Error>
{
    let store = test_store();
    let request = NewSessionRequest::new("/tmp").mcp_servers(vec![
        agent_client_protocol::schema::McpServer::Stdio(
            agent_client_protocol::schema::McpServerStdio::new("test", "/usr/bin/true"),
        ),
    ]);
    let Err(error) = handle_new_session_request(&store, &request) else {
        return Err(agent_client_protocol::Error::internal_error()
            .data("expected MCP session request to be rejected"));
    };
    assert!(
        error
            .to_string()
            .contains("MCP servers require the async session setup path")
    );
    Ok(())
}

#[test]
fn validate_session_paths_rejects_relative_cwd() -> Result<(), agent_client_protocol::Error> {
    let request = NewSessionRequest::new("relative/path");
    let Err(error) = validate_session_paths(&request) else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected relative cwd to fail")
        );
    };
    assert!(
        error
            .to_string()
            .contains("session cwd must be an absolute path")
    );
    Ok(())
}

#[test]
fn validate_session_paths_rejects_relative_additional_directory()
-> Result<(), agent_client_protocol::Error> {
    let request = NewSessionRequest::new("/tmp")
        .additional_directories(vec![std::path::PathBuf::from("not-absolute")]);
    let Err(error) = validate_session_paths(&request) else {
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

#[test_log::test(tokio::test)]
async fn prompt_request_rejects_active_turn() -> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
    {
        let mut guard = store
            .state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let record = guard.sessions.get_mut(&session.session_id).ok_or_else(|| {
            agent_client_protocol::Error::internal_error().data("missing stored session")
        })?;
        record.active_turn = Some(CancellationToken::new());
    }

    let Err(error) = handle_prompt_request(
        &store,
        &MockLlmClient,
        &EmptyToolRegistry,
        None,
        PromptRequest::new(session.session_id, vec![ContentBlock::from("hi")]),
        DEFAULT_MAX_TURN_REQUESTS,
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
    assert!(
        response
            .agent_capabilities
            .session_capabilities
            .list
            .is_some()
    );
    assert!(
        response
            .agent_capabilities
            .session_capabilities
            .close
            .is_some()
    );
    assert!(
        response
            .agent_capabilities
            .session_capabilities
            .delete
            .is_some()
    );
    assert!(
        response
            .agent_capabilities
            .session_capabilities
            .resume
            .is_some()
    );
    assert!(
        response
            .agent_capabilities
            .session_capabilities
            .additional_directories
            .is_some()
    );
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
fn initialize_handshake_records_client_capabilities() -> Result<(), agent_client_protocol::Error> {
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
            agent_client_protocol::Error::internal_error().data("missing reasoning effort option")
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
fn set_mode_emits_current_mode_update() -> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
    let mut notifications = Vec::new();

    handle_set_session_mode_request_notifying(
        &store,
        &SetSessionModeRequest::new(session.session_id.clone(), "yolo"),
        |notification| {
            notifications.push(notification);
            Ok(())
        },
    )?;

    assert_eq!(notifications.len(), 1);
    assert_eq!(notifications[0].session_id, session.session_id);
    let SessionUpdate::CurrentModeUpdate(update) = &notifications[0].update else {
        return Err(agent_client_protocol::Error::internal_error()
            .data("expected current mode update notification"));
    };
    assert_eq!(update.current_mode_id.0.as_ref(), "yolo");

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

#[test_log::test]
fn set_config_option_emits_config_option_update() -> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
    let mut notifications = Vec::new();

    let response = handle_set_session_config_option_request_notifying(
        &store,
        &SetSessionConfigOptionRequest::new(
            session.session_id.clone(),
            SESSION_CONFIG_REASONING_EFFORT_ID,
            "max",
        ),
        |notification| {
            notifications.push(notification);
            Ok(())
        },
    )?;

    assert_eq!(notifications.len(), 1);
    assert_eq!(notifications[0].session_id, session.session_id);
    let SessionUpdate::ConfigOptionUpdate(update) = &notifications[0].update else {
        return Err(agent_client_protocol::Error::internal_error()
            .data("expected config option update notification"));
    };
    assert_eq!(
        select_current_value(&update.config_options, SESSION_CONFIG_REASONING_EFFORT_ID,)?,
        "max"
    );
    assert_eq!(update.config_options, response.config_options);

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
    let call = deepseek_acp_adapter::deepseek::ToolCall::new(
        "call-1",
        "write_file",
        serde_json::json!({ "path": "file.txt" }).to_string(),
    );
    let requester = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
        RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
            PERMISSION_ALLOW_ALWAYS_OPTION_ID,
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
        request_tool_permission(&store, &context, &call, ToolKind::Edit, &second_requester).await?;

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
async fn permission_request_rejects_without_caching() -> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
    let context = ToolContext {
        session_id: session.session_id.clone(),
        cwd: std::path::PathBuf::from("/tmp"),
        additional_directories: Vec::new(),
        client_capabilities: None,
    };
    let call = deepseek_acp_adapter::deepseek::ToolCall::new(
        "call-2",
        "run_command",
        serde_json::json!({ "command": "echo hi" }).to_string(),
    );
    let requester = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
        RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
            PERMISSION_REJECT_ONCE_OPTION_ID,
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
async fn permission_posture_ask_prompts_all_mutations() -> Result<(), agent_client_protocol::Error>
{
    let (store, _session_id, context, edit_call, shell_call) = permission_mode_fixture()?;
    let requester = FakePermissionRequester::new(vec![
        RequestPermissionResponse::new(RequestPermissionOutcome::Selected(
            SelectedPermissionOutcome::new(PERMISSION_ALLOW_ONCE_OPTION_ID),
        )),
        RequestPermissionResponse::new(RequestPermissionOutcome::Selected(
            SelectedPermissionOutcome::new(PERMISSION_ALLOW_ONCE_OPTION_ID),
        )),
    ]);

    assert_eq!(
        request_tool_permission(&store, &context, &edit_call, ToolKind::Edit, &requester).await?,
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
            PERMISSION_ALLOW_ONCE_OPTION_ID,
        )),
    )]);

    assert_eq!(
        request_tool_permission(&store, &context, &edit_call, ToolKind::Edit, &requester).await?,
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
        request_tool_permission(&store, &context, &edit_call, ToolKind::Edit, &requester).await?,
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
async fn serve_with_transport_handles_authenticate_and_mode_updates()
-> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let llm_client: Arc<dyn LlmClient> = Arc::new(MockLlmClient);
    let tool_registry: Arc<dyn ToolRegistry> = Arc::new(EmptyToolRegistry);
    let (client_transport, server_transport) = Channel::duplex();
    let server_state = Arc::clone(&store.state);
    let server_client = Arc::clone(&llm_client);
    let server_tools = Arc::clone(&tool_registry);

    let state_dir = std::env::temp_dir().join(format!("deepseek-acp-auth-test-{}", Uuid::new_v4()));

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

    Agent
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

#[test_log::test]
fn handle_set_session_mode_request_rejects_invalid_inputs()
-> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;

    let Err(error) = handle_set_session_mode_request(
        &store,
        &SetSessionModeRequest::new(session.session_id.clone(), "bogus"),
    ) else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected invalid mode id to fail")
        );
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
async fn handle_prompt_request_rejects_unknown_session() -> Result<(), agent_client_protocol::Error>
{
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
async fn request_permission_rejects_unknown_option() -> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
    let context = ToolContext {
        session_id: session.session_id.clone(),
        cwd: std::path::PathBuf::from("/tmp"),
        additional_directories: Vec::new(),
        client_capabilities: None,
    };
    let call = deepseek_acp_adapter::deepseek::ToolCall::new(
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
fn model_select_options_includes_custom_model_when_unknown() {
    let options = model_select_options("my-custom-model");
    let custom = options
        .iter()
        .find(|opt| opt.value.0.as_ref() == "my-custom-model");
    let description = custom.and_then(|opt| opt.description.as_deref());
    assert_eq!(description, Some("Current model from DEEPSEEK_MODEL."));
}

#[test]
fn model_select_options_omits_custom_model_when_known() {
    let options = model_select_options("deepseek-v4-pro");
    let custom = options
        .iter()
        .find(|opt| opt.value.0.as_ref() == "deepseek-v4-pro");
    assert!(custom.is_some());
    assert!(
        !options
            .iter()
            .any(|opt| opt.description.as_deref() == Some("Current model from DEEPSEEK_MODEL."))
    );
}

#[test]
fn validate_session_model_accepts_known_models() -> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
    let guard = store
        .state
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let record = guard
        .sessions
        .get(&session.session_id)
        .ok_or_else(|| agent_client_protocol::Error::internal_error().data("missing session"))?;
    assert!(validate_session_model(record, "deepseek-v4-pro").is_ok());
    assert!(validate_session_model(record, "deepseek-v4-flash").is_ok());
    assert!(validate_session_model(record, "deepseek-v4-pro").is_ok());
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
    let record = guard
        .sessions
        .get(&session.session_id)
        .ok_or_else(|| agent_client_protocol::Error::internal_error().data("missing session"))?;
    assert!(validate_session_model(record, "bogus-model").is_err());
    Ok(())
}

#[test]
fn initial_model_from_env_uses_default_when_not_set() {
    let model = initial_model_from_env();
    assert_eq!(model, "deepseek-v4-pro");
}

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
    let call = deepseek_acp_adapter::deepseek::ToolCall::new(
        "reject-call",
        "run_command",
        serde_json::json!({ "command": "echo hi" }).to_string(),
    );
    let requester = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
        RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
            PERMISSION_REJECT_ONCE_OPTION_ID,
        )),
    )]);

    let Err(error) =
        require_tool_permission(&store, &context, &call, ToolKind::Execute, Some(&requester)).await
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
    let call = deepseek_acp_adapter::deepseek::ToolCall::new(
        "cancel-call",
        "run_command",
        serde_json::json!({ "command": "echo hi" }).to_string(),
    );
    let requester = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
        RequestPermissionOutcome::Cancelled,
    )]);

    let Err(error) =
        require_tool_permission(&store, &context, &call, ToolKind::Execute, Some(&requester)).await
    else {
        return Err(agent_client_protocol::Error::internal_error().data("expected cancellation"));
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
    let call = deepseek_acp_adapter::deepseek::ToolCall::new("id", "tool", "{}");
    let Err(error) = require_tool_permission(&store, &context, &call, ToolKind::Edit, None).await
    else {
        return;
    };
    assert!(error.contains("requires a client connection"));
}

#[test_log::test(tokio::test)]
async fn request_permission_handles_unknown_session_and_cancelled()
-> Result<(), agent_client_protocol::Error> {
    let missing_store = test_store();
    let missing_context = ToolContext {
        session_id: agent_client_protocol::schema::SessionId::new("missing-session"),
        cwd: std::path::PathBuf::from("/tmp"),
        additional_directories: Vec::new(),
        client_capabilities: None,
    };
    let missing_call = deepseek_acp_adapter::deepseek::ToolCall::new(
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
    let call = deepseek_acp_adapter::deepseek::ToolCall::new(
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

#[test]
fn set_config_option_updates_mode() -> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;

    let response = handle_set_session_config_option_request(
        &store,
        &SetSessionConfigOptionRequest::new(
            session.session_id.clone(),
            SESSION_CONFIG_MODE_ID,
            "yolo",
        ),
    )?;
    let options = &response.config_options;
    assert_eq!(
        select_current_value(options, SESSION_CONFIG_MODE_ID)?,
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
        &SetSessionConfigOptionRequest::new(session.session_id, SESSION_CONFIG_MODE_ID, "bogus"),
    ) else {
        return Err(agent_client_protocol::Error::internal_error()
            .data("expected invalid mode via config to fail"));
    };
    assert!(error.to_string().contains("unsupported session mode"));
    Ok(())
}

#[test]
fn set_config_option_rejects_invalid_reasoning_effort() -> Result<(), agent_client_protocol::Error>
{
    let store = test_store();
    let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;

    let Err(error) = handle_set_session_config_option_request(
        &store,
        &SetSessionConfigOptionRequest::new(
            session.session_id,
            SESSION_CONFIG_REASONING_EFFORT_ID,
            "bogus",
        ),
    ) else {
        return Err(agent_client_protocol::Error::internal_error()
            .data("expected invalid reasoning effort to fail"));
    };
    assert!(error.to_string().contains("unsupported reasoning effort"));
    Ok(())
}

#[test]
fn set_config_option_rejects_unknown_session() -> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let Err(error) = handle_set_session_config_option_request(
        &store,
        &SetSessionConfigOptionRequest::new(
            agent_client_protocol::schema::SessionId::new("missing"),
            SESSION_CONFIG_MODEL_ID,
            "deepseek-v4-flash",
        ),
    ) else {
        return Err(agent_client_protocol::Error::internal_error()
            .data("expected unknown session id to fail"));
    };
    assert!(error.to_string().contains("unknown session id"));
    Ok(())
}

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

#[test]
fn set_mode_rejects_invalid_mode_id() -> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
    let Err(error) = handle_set_session_mode_request(
        &store,
        &SetSessionModeRequest::new(session.session_id, "bogus"),
    ) else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected invalid mode to fail")
        );
    };
    assert!(error.to_string().contains("unsupported session mode"));
    Ok(())
}

#[test_log::test(tokio::test)]
async fn new_session_connected_async_path_creates_session()
-> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let response =
        handle_new_session_request_connected(&store, &NewSessionRequest::new("/tmp")).await?;
    assert!(response.session_id.0.starts_with("session-"));
    Ok(())
}

#[test_log::test(tokio::test)]
async fn serve_with_transport_exercises_list_close_and_logout()
-> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let state_dir = std::env::temp_dir().join(format!("deepseek-acp-list-test-{}", Uuid::new_v4()));
    let llm_client: Arc<dyn LlmClient> = Arc::new(MockLlmClient);
    let tool_registry: Arc<dyn ToolRegistry> = Arc::new(EmptyToolRegistry);
    let (client_transport, server_transport) = Channel::duplex();
    let server_state = Arc::clone(&store.state);
    let server_client = Arc::clone(&llm_client);
    let server_tools = Arc::clone(&tool_registry);

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

    Agent
        .builder()
        .connect_with(client_transport, async move |cx| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::LATEST))
                .block_task()
                .await?;

            let new_session = cx
                .send_request(NewSessionRequest::new("/tmp"))
                .block_task()
                .await?;

            let list = cx
                .send_request(agent_client_protocol::schema::ListSessionsRequest::new())
                .block_task()
                .await?;
            assert_eq!(list.sessions.len(), 1);

            cx.send_request(agent_client_protocol::schema::CloseSessionRequest::new(
                new_session.session_id,
            ))
            .block_task()
            .await?;

            cx.send_request(agent_client_protocol::schema::LogoutRequest::new())
                .block_task()
                .await?;

            Ok(())
        })
        .await?;

    server.abort();
    Ok(())
}

#[test_log::test(tokio::test)]
async fn serve_with_transport_drives_new_session_config_prompt_and_cancel()
-> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let llm_client: Arc<dyn LlmClient> = Arc::new(MockLlmClient);
    let tool_registry: Arc<dyn ToolRegistry> = Arc::new(EmptyToolRegistry);
    let (client_transport, server_transport) = Channel::duplex();
    let server_state = Arc::clone(&store.state);
    let server_client = Arc::clone(&llm_client);
    let server_tools = Arc::clone(&tool_registry);
    let state_dir =
        std::env::temp_dir().join(format!("deepseek-acp-config-test-{}", Uuid::new_v4()));

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

    Agent
        .builder()
        .connect_with(client_transport, async move |cx| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::LATEST))
                .block_task()
                .await?;

            // new_session handler: the async/spawned path that also emits the
            // available-commands notification.
            let new_session = cx
                .send_request(NewSessionRequest::new("/tmp"))
                .block_task()
                .await?;
            let session_id = new_session.session_id.clone();

            // set_session_config_option handler.
            let config_response = cx
                .send_request(SetSessionConfigOptionRequest::new(
                    session_id.clone(),
                    SESSION_CONFIG_MODEL_ID,
                    "deepseek-v4-flash",
                ))
                .block_task()
                .await?;
            assert_eq!(
                select_current_value(&config_response.config_options, SESSION_CONFIG_MODEL_ID)?,
                "deepseek-v4-flash"
            );

            // prompt handler (MockLlmClient returns a canned EndTurn reply).
            let prompt_response = cx
                .send_request(PromptRequest::new(
                    session_id.clone(),
                    vec![ContentBlock::from("hello")],
                ))
                .block_task()
                .await?;
            assert_eq!(
                prompt_response.stop_reason,
                agent_client_protocol::schema::StopReason::EndTurn
            );

            // cancel notification handler (no active turn -> a no-op).
            cx.send_notification(agent_client_protocol::schema::CancelNotification::new(
                session_id,
            ))?;

            // A trailing round-trip request guarantees in-order delivery has
            // processed the cancel notification above before the server stops.
            cx.send_request(agent_client_protocol::schema::ListSessionsRequest::new())
                .block_task()
                .await?;

            Ok(())
        })
        .await?;

    server.abort();
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
        &[ChatMessage::user("one"), ChatMessage::assistant("two")],
    )?;
    store.save_history(
        &session.session_id,
        &[
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

    store.save_history(&session.session_id, &[ChatMessage::user("persist me")])?;
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

#[test]
fn list_sessions_merges_active_and_persisted_sessions_for_requested_cwd()
-> Result<(), agent_client_protocol::Error> {
    let state_dir =
        std::env::temp_dir().join(format!("deepseek-acp-list-merge-{}", Uuid::new_v4()));
    let workspace = state_dir.join("workspace");
    let persistence = FilesystemSessionStore::new(&state_dir);
    let store = SessionStore::new(Arc::new(Mutex::new(AdapterState::default())))
        .with_persistence(persistence.clone());
    let active = handle_new_session_request(&store, &NewSessionRequest::new(&workspace))?;
    store.save_history(&active.session_id, &[ChatMessage::user("active")])?;

    let persisted_id = agent_client_protocol::schema::SessionId::new("session-persisted-list");
    persistence
        .persist_turn(
            &PersistedSessionMeta {
                session_id: persisted_id.0.to_string(),
                cwd: workspace.clone(),
                additional_directories: vec![state_dir.join("extra")],
                mode: PermissionPosture::Ask,
                model: "deepseek-v4-pro".to_string(),
                reasoning_effort: ReasoningEffort::High,
                mcp_servers: Vec::new(),
                title: None,
                updated_at: None,
            },
            &[ChatMessage::user("persisted")],
        )
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    let response =
        handle_list_sessions_request(&store, &ListSessionsRequest::new().cwd(&workspace))?;
    let ids = response
        .sessions
        .iter()
        .map(|session| session.session_id.clone())
        .collect::<Vec<_>>();

    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&active.session_id));
    assert!(ids.contains(&persisted_id));
    assert_eq!(
        ids.iter()
            .filter(|session_id| **session_id == active.session_id)
            .count(),
        1
    );
    let persisted = response
        .sessions
        .iter()
        .find(|session| session.session_id == persisted_id)
        .ok_or_else(|| {
            agent_client_protocol::Error::internal_error()
                .data("missing persisted session from list response")
        })?;
    assert_eq!(
        persisted.additional_directories,
        vec![state_dir.join("extra")]
    );

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
    let tool_call = deepseek_acp_adapter::deepseek::ToolCall::new(
        "call-1",
        "read_file",
        r#"{"path":"Cargo.toml"}"#,
    );
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
                title: None,
                updated_at: None,
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

    assert!(response.modes.is_some());
    assert!(response.config_options.is_some());
    assert_eq!(notifications.len(), 4);
    let SessionUpdate::UserMessageChunk(user_chunk) = &notifications[0].update else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected user message chunk")
        );
    };
    let SessionUpdate::AgentMessageChunk(first_assistant_chunk) = &notifications[1].update else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected assistant message chunk")
        );
    };
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
    let SessionUpdate::AgentMessageChunk(second_assistant_chunk) = &notifications[3].update else {
        return Err(agent_client_protocol::Error::internal_error()
            .data("expected final assistant message chunk"));
    };
    assert!(user_chunk.message_id.is_some());
    assert!(first_assistant_chunk.message_id.is_some());
    assert!(second_assistant_chunk.message_id.is_some());
    assert_ne!(user_chunk.message_id, first_assistant_chunk.message_id);
    assert_ne!(
        first_assistant_chunk.message_id,
        second_assistant_chunk.message_id
    );

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
async fn resume_session_restores_state_without_replay() -> Result<(), agent_client_protocol::Error>
{
    let state_dir =
        std::env::temp_dir().join(format!("deepseek-acp-resume-history-{}", Uuid::new_v4()));
    let workspace = state_dir.join("workspace");
    let persistence = FilesystemSessionStore::new(&state_dir);
    let store = SessionStore::new(Arc::new(Mutex::new(AdapterState::default())))
        .with_persistence(persistence.clone());
    let session_id = agent_client_protocol::schema::SessionId::new("session-resume");
    let history = vec![
        ChatMessage::user("restore me"),
        ChatMessage::assistant("restored"),
    ];
    persistence
        .persist_turn(
            &PersistedSessionMeta {
                session_id: session_id.0.to_string(),
                cwd: workspace.clone(),
                additional_directories: vec![state_dir.join("extra")],
                mode: PermissionPosture::Yolo,
                model: "deepseek-v4-flash".to_string(),
                reasoning_effort: ReasoningEffort::Max,
                mcp_servers: Vec::new(),
                title: None,
                updated_at: None,
            },
            &history,
        )
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    let response = handle_resume_session_request(
        &store,
        &ResumeSessionRequest::new(session_id.clone(), workspace.clone()),
    )
    .await?;

    assert!(response.modes.is_some());
    assert!(response.config_options.is_some());
    let guard = store
        .state
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let restored = guard.sessions.get(&session_id).ok_or_else(|| {
        agent_client_protocol::Error::internal_error().data("missing resumed session")
    })?;
    assert_eq!(restored.cwd, workspace);
    assert_eq!(restored.mode, PermissionPosture::Yolo);
    assert_eq!(restored.model, "deepseek-v4-flash");
    assert_eq!(restored.reasoning_effort, ReasoningEffort::Max);
    assert_eq!(restored.history, history);

    Ok(())
}

#[test_log::test(tokio::test)]
async fn resume_session_rejects_relative_additional_directory()
-> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let request = ResumeSessionRequest::new("session-resume-invalid", "/tmp")
        .additional_directories(vec![std::path::PathBuf::from("relative")]);

    let Err(error) = handle_resume_session_request(&store, &request).await else {
        return Err(agent_client_protocol::Error::internal_error()
            .data("expected relative additional directory to fail"));
    };
    assert!(
        error
            .to_string()
            .contains("additional directories must be absolute paths")
    );

    Ok(())
}

#[test_log::test(tokio::test)]
async fn load_session_rejects_mismatched_cwd() -> Result<(), agent_client_protocol::Error> {
    let state_dir = std::env::temp_dir().join(format!("deepseek-acp-load-cwd-{}", Uuid::new_v4()));
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
                title: None,
                updated_at: None,
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
        return Err(
            agent_client_protocol::Error::internal_error().data("expected mismatched cwd to fail")
        );
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

    let Err(error) = handle_close_session_request(&store, &CloseSessionRequest::new(unknown_id))
    else {
        return Err(agent_client_protocol::Error::internal_error()
            .data("expected unknown session id to fail"));
    };
    assert!(error.to_string().contains("unknown session id"));
    Ok(())
}

#[test]
fn delete_session_removes_memory_and_persistence() -> Result<(), agent_client_protocol::Error> {
    let state_dir =
        std::env::temp_dir().join(format!("deepseek-acp-delete-session-{}", Uuid::new_v4()));
    let workspace = state_dir.join("workspace");
    let persistence = FilesystemSessionStore::new(&state_dir);
    let store = SessionStore::new(Arc::new(Mutex::new(AdapterState::default())))
        .with_persistence(persistence.clone());
    let session_id = agent_client_protocol::schema::SessionId::new("session-delete");
    let active_turn = CancellationToken::new();
    store.insert_session(
        session_id.clone(),
        SessionRecord {
            cwd: workspace.clone(),
            additional_directories: Vec::new(),
            history: Vec::new(),
            active_turn: Some(active_turn.clone()),
            mode: PermissionPosture::Ask,
            model: "deepseek-v4-pro".to_string(),
            reasoning_effort: ReasoningEffort::High,
            permission_allow_always: std::collections::HashSet::new(),
            mcp_servers: Vec::new(),
            mcp_sessions: Vec::new(),
            title: "temporary title".to_string(),
            updated_at: "2026-06-14T00:00:00Z".to_string(),
        },
    )?;
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
                title: Some("temporary title".to_string()),
                updated_at: Some("2026-06-14T00:00:00Z".to_string()),
            },
            &[ChatMessage::user("hello")],
        )
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    let response =
        handle_delete_session_request(&store, &DeleteSessionRequest::new(session_id.clone()))?;
    assert_eq!(
        serde_json::to_value(&response)
            .map_err(agent_client_protocol::Error::into_internal_error)?,
        serde_json::json!({})
    );
    assert!(active_turn.is_cancelled());

    let guard = store
        .state
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    assert!(!guard.sessions.contains_key(&session_id));
    drop(guard);

    assert!(persistence.load_record(session_id.0.as_ref()).is_err());

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

// ── Delegation impl coverage ────────────────────────────────────
//
// The following tests exercise the production `ConnectionTo<Client>`
// requester delegation impls for terminal, read/write text file, and
// permission requesters. Each test sets up a duplex channel where the
// agent role (whose connection is `ConnectionTo<Client>`, exactly as in
// `serve_with_transport`) drives requests *through the requester trait
// wrappers* while a client role responds, so the wrappers'
// `self.send_request(request).block_task()` bodies are genuinely covered.
use super::{
    CreateTerminalRequester, KillTerminalRequester, PermissionRequester, ReadTextFileRequester,
    ReleaseTerminalRequester, TerminalOutputRequester, WaitForTerminalExitRequester,
    WriteTextFileRequester, recover_null_write_response,
};
use agent_client_protocol::Agent;
use agent_client_protocol::schema::{
    CreateTerminalRequest, CreateTerminalResponse, KillTerminalRequest, KillTerminalResponse,
    ReadTextFileRequest, ReadTextFileResponse, ReleaseTerminalRequest, ReleaseTerminalResponse,
    RequestPermissionRequest, TerminalExitStatus, TerminalId, TerminalOutputRequest,
    TerminalOutputResponse, WaitForTerminalExitRequest, WaitForTerminalExitResponse,
    WriteTextFileRequest, WriteTextFileResponse,
};

#[allow(clippy::too_many_lines)]
// Covers all 5 terminal wrapper methods in one client-driven flow; splitting would duplicate the server setup.
#[test_log::test(tokio::test)]
async fn connection_to_client_terminal_requester_delegation()
-> Result<(), agent_client_protocol::Error> {
    let (client_transport, server_transport) = Channel::duplex();

    let server = tokio::spawn(async move {
        Client
            .builder()
            .on_receive_request(
                async move |_request: CreateTerminalRequest, responder, _cx| {
                    responder.respond(CreateTerminalResponse::new(TerminalId::new("term-1")))
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                async move |_request: TerminalOutputRequest, responder, _cx| {
                    responder.respond(TerminalOutputResponse::new(
                        "terminal output content",
                        false,
                    ))
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                async move |_request: WaitForTerminalExitRequest, responder, _cx| {
                    let status = TerminalExitStatus::new().exit_code(Some(0));
                    responder.respond(WaitForTerminalExitResponse::new(status))
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                async move |_request: ReleaseTerminalRequest, responder, _cx| {
                    responder.respond(ReleaseTerminalResponse::new())
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                async move |_request: KillTerminalRequest, responder, _cx| {
                    responder.respond(KillTerminalResponse::new())
                },
                agent_client_protocol::on_receive_request!(),
            )
            .connect_to(server_transport)
            .await
    });

    // The agent role's connection handle is a `ConnectionTo<Client>` — the
    // same concrete type `serve_with_transport` hands the tool layer — so
    // driving requests through the requester wrappers covers the production
    // delegation impls (not the raw library `send_request`).
    Agent
        .builder()
        .connect_with(client_transport, async move |cx| {
            let create_response = cx
                .create_terminal(CreateTerminalRequest::new(
                    agent_client_protocol::schema::SessionId::new("session-terminal"),
                    "echo hi",
                ))
                .await?;
            assert_eq!(create_response.terminal_id.0.as_ref(), "term-1");

            let output_response = cx
                .terminal_output(TerminalOutputRequest::new(
                    agent_client_protocol::schema::SessionId::new("session-terminal"),
                    TerminalId::new("term-1"),
                ))
                .await?;
            assert_eq!(output_response.output, "terminal output content");

            let wait_response = cx
                .wait_for_terminal_exit(WaitForTerminalExitRequest::new(
                    agent_client_protocol::schema::SessionId::new("session-terminal"),
                    TerminalId::new("term-1"),
                ))
                .await?;
            assert_eq!(wait_response.exit_status.exit_code, Some(0));

            let release_response = cx
                .release_terminal(ReleaseTerminalRequest::new(
                    agent_client_protocol::schema::SessionId::new("session-terminal"),
                    TerminalId::new("term-1"),
                ))
                .await?;
            assert!(release_response.meta.is_none());

            let kill_response = cx
                .kill_terminal(KillTerminalRequest::new(
                    agent_client_protocol::schema::SessionId::new("session-terminal"),
                    TerminalId::new("term-1"),
                ))
                .await?;
            assert!(kill_response.meta.is_none());

            Ok(())
        })
        .await?;

    server.abort();
    Ok(())
}

#[test_log::test(tokio::test)]
async fn connection_to_client_read_write_requester_delegation()
-> Result<(), agent_client_protocol::Error> {
    let (client_transport, server_transport) = Channel::duplex();
    let observed_write = Arc::new(Mutex::new(None::<(std::path::PathBuf, String)>));
    let server_write = Arc::clone(&observed_write);

    let server = tokio::spawn(async move {
        Client
            .builder()
            .on_receive_request(
                async move |_request: ReadTextFileRequest, responder, _cx| {
                    responder.respond(ReadTextFileResponse::new("server read this content"))
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                async move |request: WriteTextFileRequest, responder, _cx| {
                    {
                        let mut guard = server_write
                            .lock()
                            .map_err(agent_client_protocol::Error::into_internal_error)?;
                        *guard = Some((request.path.clone(), request.content.clone()));
                    }
                    responder.respond(WriteTextFileResponse::new())
                },
                agent_client_protocol::on_receive_request!(),
            )
            .connect_to(server_transport)
            .await
    });

    Agent
        .builder()
        .connect_with(client_transport, async move |cx| {
            let read_response = cx
                .read_text_file(ReadTextFileRequest::new(
                    agent_client_protocol::schema::SessionId::new("session-read-write"),
                    std::path::PathBuf::from("/tmp/file.txt"),
                ))
                .await?;
            assert_eq!(read_response.content, "server read this content");

            cx.write_text_file(WriteTextFileRequest::new(
                agent_client_protocol::schema::SessionId::new("session-read-write"),
                std::path::PathBuf::from("/tmp/written.txt"),
                "client data",
            ))
            .await?;

            Ok(())
        })
        .await?;

    server.abort();

    {
        let guard = observed_write
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let (path, content) = guard.as_ref().ok_or_else(|| {
            agent_client_protocol::Error::internal_error()
                .data("server did not receive write request")
        })?;
        assert_eq!(path, &std::path::PathBuf::from("/tmp/written.txt"));
        assert_eq!(content, "client data");
    }

    Ok(())
}

#[test_log::test(tokio::test)]
async fn connection_to_client_permission_requester_delegation()
-> Result<(), agent_client_protocol::Error> {
    let (client_transport, server_transport) = Channel::duplex();

    let server = tokio::spawn(async move {
        Client
            .builder()
            .on_receive_request(
                async move |request: RequestPermissionRequest, responder, _cx| {
                    let outcome = request.options.first().map_or(
                        RequestPermissionOutcome::Cancelled,
                        |option| {
                            RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                                option.option_id.clone(),
                            ))
                        },
                    );
                    responder.respond(RequestPermissionResponse::new(outcome))
                },
                agent_client_protocol::on_receive_request!(),
            )
            .connect_to(server_transport)
            .await
    });

    let response = Agent
        .builder()
        .connect_with(client_transport, async move |cx| {
            cx.request_permission(RequestPermissionRequest::new(
                agent_client_protocol::schema::SessionId::new("session-permission"),
                agent_client_protocol::schema::ToolCallUpdate::new(
                    "call-permission",
                    agent_client_protocol::schema::ToolCallUpdateFields::new()
                        .kind(agent_client_protocol::schema::ToolKind::Execute)
                        .status(agent_client_protocol::schema::ToolCallStatus::Pending)
                        .title("run_command")
                        .raw_input(serde_json::json!({ "command": "echo hi" })),
                ),
                vec![agent_client_protocol::schema::PermissionOption::new(
                    "allow-once",
                    "Allow once",
                    agent_client_protocol::schema::PermissionOptionKind::AllowOnce,
                )],
            ))
            .await
        })
        .await?;

    let RequestPermissionOutcome::Selected(selected) = &response.outcome else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected selected outcome")
        );
    };
    assert_eq!(selected.option_id.0.as_ref(), "allow-once");

    server.abort();
    Ok(())
}

// ── recover_null_write_response ─────────────────────────

#[test]
fn recover_null_write_response_passes_through_success() {
    let result = recover_null_write_response(Ok(WriteTextFileResponse::new()));
    assert!(result.is_ok());
}

#[test]
fn recover_null_write_response_recovers_null_payload_parse_error() {
    // A `null` write result reported as a deserialization ParseError is the
    // known client interop quirk and must be treated as an empty success.
    let err = agent_client_protocol::Error::parse_error()
        .data(serde_json::json!({ "json": null, "phase": "deserialization" }));
    let result = recover_null_write_response(Err(err));
    assert!(result.is_ok());
}

#[test]
fn recover_null_write_response_propagates_unrelated_parse_error()
-> Result<(), agent_client_protocol::Error> {
    // ParseError without the null-payload signature must not be swallowed.
    let err = agent_client_protocol::Error::parse_error()
        .data(serde_json::json!({ "json": "oops", "phase": "deserialization" }));
    let Err(returned) = recover_null_write_response(Err(err)) else {
        return Err(agent_client_protocol::Error::internal_error()
            .data("expected unrelated parse error to propagate"));
    };
    assert_eq!(returned.code, agent_client_protocol::ErrorCode::ParseError);
    Ok(())
}

#[test]
fn recover_null_write_response_propagates_non_parse_error()
-> Result<(), agent_client_protocol::Error> {
    // A different error code (here invalid_params) is always propagated.
    let err = agent_client_protocol::Error::invalid_params().data("disk full");
    let Err(returned) = recover_null_write_response(Err(err)) else {
        return Err(agent_client_protocol::Error::internal_error()
            .data("expected non-parse error to propagate"));
    };
    assert!(returned.to_string().contains("disk full"));
    Ok(())
}

// ── Path-validation edge cases ─────────────────────────

#[test]
fn validate_load_session_paths_rejects_relative_cwd() -> Result<(), agent_client_protocol::Error> {
    let request = LoadSessionRequest::new("s1", std::path::PathBuf::from("relative"));
    let Err(error) = validate_load_session_paths(&request) else {
        return Err(agent_client_protocol::Error::internal_error().data("expected rejection"));
    };
    assert!(error.to_string().contains("cwd must be absolute"));
    Ok(())
}

#[test]
fn validate_load_session_paths_rejects_relative_additional()
-> Result<(), agent_client_protocol::Error> {
    let request = LoadSessionRequest::new("s1", std::path::PathBuf::from("/tmp"))
        .additional_directories(vec![std::path::PathBuf::from("relative")]);
    let Err(error) = validate_load_session_paths(&request) else {
        return Err(agent_client_protocol::Error::internal_error().data("expected rejection"));
    };
    assert!(
        error
            .to_string()
            .contains("additional directory must be absolute")
    );
    Ok(())
}

#[test]
fn validate_resume_session_paths_rejects_relative_cwd() -> Result<(), agent_client_protocol::Error>
{
    let request = ResumeSessionRequest::new("s1", std::path::PathBuf::from("relative"));
    let Err(error) = validate_resume_session_paths(&request) else {
        return Err(agent_client_protocol::Error::internal_error().data("expected rejection"));
    };
    assert!(
        error
            .to_string()
            .contains("session cwd must be an absolute path")
    );
    Ok(())
}

#[test]
fn validate_resume_session_paths_rejects_relative_additional()
-> Result<(), agent_client_protocol::Error> {
    let request = ResumeSessionRequest::new("s1", std::path::PathBuf::from("/tmp"))
        .additional_directories(vec![std::path::PathBuf::from("relative")]);
    let Err(error) = validate_resume_session_paths(&request) else {
        return Err(agent_client_protocol::Error::internal_error().data("expected rejection"));
    };
    assert!(
        error
            .to_string()
            .contains("additional directories must be absolute paths")
    );
    Ok(())
}

#[test]
fn validate_load_session_paths_accepts_absolute_additional()
-> Result<(), agent_client_protocol::Error> {
    // Exercises the loop-continue path: an absolute additional directory is
    // accepted and validation returns `Ok`.
    let request = LoadSessionRequest::new("s1", std::path::PathBuf::from("/workspace"))
        .additional_directories(vec![
            std::path::PathBuf::from("/abs/one"),
            std::path::PathBuf::from("/abs/two"),
        ]);
    validate_load_session_paths(&request)
}

#[test]
fn validate_resume_session_paths_accepts_absolute_additional()
-> Result<(), agent_client_protocol::Error> {
    let request = ResumeSessionRequest::new("s1", std::path::PathBuf::from("/workspace"))
        .additional_directories(vec![
            std::path::PathBuf::from("/abs/one"),
            std::path::PathBuf::from("/abs/two"),
        ]);
    validate_resume_session_paths(&request)
}

// ── config_value_id ─────────────────────────────────────

#[test]
fn config_value_id_extracts_value_id() -> Result<(), agent_client_protocol::Error> {
    let value = SessionConfigOptionValue::value_id("model-1");
    let result = config_value_id(&value)?;
    assert_eq!(result.0.as_ref(), "model-1");
    Ok(())
}

#[test]
fn config_value_id_rejects_boolean_value() -> Result<(), agent_client_protocol::Error> {
    let value = SessionConfigOptionValue::boolean(true);
    let Err(error) = config_value_id(&value) else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected boolean rejection")
        );
    };
    assert!(
        error
            .to_string()
            .contains("session config option requires a selectable value id")
    );
    Ok(())
}

// ── tool_result_content ─────────────────────────────────

#[test]
fn tool_result_content_finds_matching_tool_result() {
    let history = vec![
        ChatMessage::user("run tool"),
        ChatMessage::assistant_with_tool_calls(
            "calling",
            vec![deepseek_acp_adapter::deepseek::ToolCall::new(
                "call-1", "echo", "{}",
            )],
        ),
        ChatMessage::tool_result("call-1", "tool output"),
        ChatMessage::assistant("done"),
    ];

    assert_eq!(
        tool_result_content("call-1", &history),
        Some("tool output".to_string())
    );
}

#[test]
fn tool_result_content_returns_none_when_not_found() {
    let history = vec![ChatMessage::user("hi"), ChatMessage::assistant("hello")];

    assert_eq!(tool_result_content("missing-call", &history), None);
}

#[test]
fn tool_result_content_returns_none_for_empty_history() {
    let history: Vec<ChatMessage> = Vec::new();
    assert_eq!(tool_result_content("any-call", &history), None);
}

// ── replayed_tool_call ──────────────────────────────────

#[test]
fn replayed_tool_call_formats_acp_tool_call_with_output() -> Result<(), agent_client_protocol::Error>
{
    let history = vec![
        ChatMessage::user("run echo"),
        ChatMessage::assistant_with_tool_calls(
            "using echo",
            vec![deepseek_acp_adapter::deepseek::ToolCall::new(
                "call-1",
                "echo",
                r#"{"message":"hi"}"#,
            )],
        ),
        ChatMessage::tool_result("call-1", "echo: hi"),
    ];

    let tool_call =
        deepseek_acp_adapter::deepseek::ToolCall::new("call-1", "echo", r#"{"message":"hi"}"#);
    let replayed = replayed_tool_call(&tool_call, &history);

    assert_eq!(replayed.tool_call_id.0.as_ref(), "call-1");
    assert_eq!(replayed.title, "echo");
    assert_eq!(replayed.status, ToolCallStatus::Completed);
    assert_eq!(
        replayed.raw_input,
        Some(serde_json::json!({ "message": "hi" }))
    );
    assert_eq!(
        replayed.raw_output,
        Some(serde_json::json!({ "content": "echo: hi" }))
    );
    assert_eq!(replayed.content.len(), 1);
    let ToolCallContent::Content(content) = &replayed.content[0] else {
        return Err(agent_client_protocol::Error::internal_error()
            .data("expected Content tool call content"));
    };
    let ContentBlock::Text(text) = &content.content else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected Text content block")
        );
    };
    assert_eq!(text.text, "echo: hi");
    Ok(())
}

#[test]
fn replayed_tool_call_defaults_output_when_tool_result_missing()
-> Result<(), agent_client_protocol::Error> {
    let tool_call =
        deepseek_acp_adapter::deepseek::ToolCall::new("call-2", "echo", r#"{"message":"hi"}"#);
    let history: Vec<ChatMessage> = Vec::new();
    let replayed = replayed_tool_call(&tool_call, &history);

    assert_eq!(replayed.tool_call_id.0.as_ref(), "call-2");
    assert_eq!(replayed.status, ToolCallStatus::Completed);
    assert_eq!(
        replayed.raw_output,
        Some(serde_json::json!({ "content": "" }))
    );
    assert_eq!(replayed.content.len(), 1);
    let ToolCallContent::Content(content) = &replayed.content[0] else {
        return Err(agent_client_protocol::Error::internal_error()
            .data("expected Content tool call content"));
    };
    let ContentBlock::Text(text) = &content.content else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected Text content block")
        );
    };
    assert_eq!(text.text, "");
    Ok(())
}

// ── replay_session_history ──────────────────────────────

#[test]
fn replay_session_history_emits_user_and_assistant_notifications()
-> Result<(), agent_client_protocol::Error> {
    let session_id = agent_client_protocol::schema::SessionId::new("replay-test");
    let history = vec![ChatMessage::user("hello"), ChatMessage::assistant("world")];
    let mut notifications = Vec::new();

    replay_session_history(&session_id, &history, &mut |notification| {
        notifications.push(notification);
        Ok(())
    })?;

    assert_eq!(notifications.len(), 2);
    let SessionUpdate::UserMessageChunk(user_chunk) = &notifications[0].update else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected user message chunk")
        );
    };
    let SessionUpdate::AgentMessageChunk(assistant_chunk) = &notifications[1].update else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected assistant message chunk")
        );
    };
    assert!(user_chunk.message_id.is_some());
    assert!(assistant_chunk.message_id.is_some());
    assert_ne!(user_chunk.message_id, assistant_chunk.message_id);
    Ok(())
}

#[test]
fn replay_session_history_skips_system_and_tool_messages()
-> Result<(), agent_client_protocol::Error> {
    let session_id = agent_client_protocol::schema::SessionId::new("replay-skip");
    let history = vec![
        ChatMessage::system("system prompt"),
        ChatMessage::user("user prompt"),
        ChatMessage::tool_result("call-1", "result"),
    ];
    let mut notifications = Vec::new();

    replay_session_history(&session_id, &history, &mut |notification| {
        notifications.push(notification);
        Ok(())
    })?;

    assert_eq!(notifications.len(), 1);
    let SessionUpdate::UserMessageChunk(user_chunk) = &notifications[0].update else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected user message chunk")
        );
    };
    assert!(user_chunk.message_id.is_some());
    Ok(())
}

#[test]
fn replay_session_history_replays_tool_calls_for_assistant()
-> Result<(), agent_client_protocol::Error> {
    let session_id = agent_client_protocol::schema::SessionId::new("replay-tools");
    let tool_call =
        deepseek_acp_adapter::deepseek::ToolCall::new("call-1", "echo", r#"{"message":"hi"}"#);
    let history = vec![
        ChatMessage::user("use echo"),
        ChatMessage::assistant_with_tool_calls("using tool", vec![tool_call]),
        ChatMessage::tool_result("call-1", "echo: hi"),
    ];
    let mut notifications = Vec::new();

    replay_session_history(&session_id, &history, &mut |notification| {
        notifications.push(notification);
        Ok(())
    })?;

    assert_eq!(notifications.len(), 3);
    let SessionUpdate::UserMessageChunk(user_chunk) = &notifications[0].update else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected user message chunk")
        );
    };
    let SessionUpdate::AgentMessageChunk(assistant_chunk) = &notifications[1].update else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected assistant message chunk")
        );
    };
    assert!(matches!(
        notifications[2].update,
        SessionUpdate::ToolCall(_)
    ));
    assert!(user_chunk.message_id.is_some());
    assert!(assistant_chunk.message_id.is_some());
    assert_ne!(user_chunk.message_id, assistant_chunk.message_id);
    Ok(())
}

#[test]
fn replay_session_history_uses_stable_message_ids() -> Result<(), agent_client_protocol::Error> {
    let session_id = agent_client_protocol::schema::SessionId::new("replay-stable");
    let history = vec![
        ChatMessage::user("same user text"),
        ChatMessage::assistant("same assistant text"),
    ];
    let mut first_replay = Vec::new();
    let mut second_replay = Vec::new();

    replay_session_history(&session_id, &history, &mut |notification| {
        first_replay.push(notification);
        Ok(())
    })?;
    replay_session_history(&session_id, &history, &mut |notification| {
        second_replay.push(notification);
        Ok(())
    })?;

    let SessionUpdate::UserMessageChunk(first_user) = &first_replay[0].update else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected user message chunk")
        );
    };
    let SessionUpdate::AgentMessageChunk(first_assistant) = &first_replay[1].update else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected assistant message chunk")
        );
    };
    let SessionUpdate::UserMessageChunk(second_user) = &second_replay[0].update else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected user message chunk")
        );
    };
    let SessionUpdate::AgentMessageChunk(second_assistant) = &second_replay[1].update else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected assistant message chunk")
        );
    };

    assert_eq!(first_user.message_id, second_user.message_id);
    assert_eq!(first_assistant.message_id, second_assistant.message_id);
    assert_ne!(first_user.message_id, first_assistant.message_id);
    let first_user_id = first_user.message_id.as_ref().ok_or_else(|| {
        agent_client_protocol::Error::internal_error().data("missing user message id")
    })?;
    let first_assistant_id = first_assistant.message_id.as_ref().ok_or_else(|| {
        agent_client_protocol::Error::internal_error().data("missing assistant message id")
    })?;
    Uuid::parse_str(first_user_id.0.as_ref())
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    Uuid::parse_str(first_assistant_id.0.as_ref())
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    Ok(())
}

// ── replay_assistant_message ────────────────────────────

#[test]
fn replay_assistant_message_emits_content_and_tool_calls()
-> Result<(), agent_client_protocol::Error> {
    let session_id = agent_client_protocol::schema::SessionId::new("replay-asst");
    let history: Vec<ChatMessage> = Vec::new();
    let tool_calls = vec![deepseek_acp_adapter::deepseek::ToolCall::new(
        "call-1", "echo", "{}",
    )];
    let message = ChatMessage::assistant_with_tool_calls("assistant text", tool_calls);
    let mut notifications = Vec::new();

    replay_assistant_message(&session_id, &history, 0, &message, &mut |notification| {
        notifications.push(notification);
        Ok(())
    })?;

    assert_eq!(notifications.len(), 2);
    let SessionUpdate::AgentMessageChunk(chunk) = &notifications[0].update else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected assistant message chunk")
        );
    };
    assert!(chunk.message_id.is_some());
    assert!(matches!(
        notifications[1].update,
        SessionUpdate::ToolCall(_)
    ));
    Ok(())
}

#[test]
fn replay_assistant_message_skips_empty_content() -> Result<(), agent_client_protocol::Error> {
    let session_id = agent_client_protocol::schema::SessionId::new("replay-empty");
    let history: Vec<ChatMessage> = Vec::new();
    let message = ChatMessage::assistant_with_tool_calls("", vec![]);
    let mut notifications = Vec::new();

    replay_assistant_message(&session_id, &history, 0, &message, &mut |notification| {
        notifications.push(notification);
        Ok(())
    })?;

    assert!(notifications.is_empty());
    Ok(())
}

#[test]
fn replay_assistant_message_content_only_no_tool_calls() -> Result<(), agent_client_protocol::Error>
{
    let session_id = agent_client_protocol::schema::SessionId::new("replay-content");
    let history: Vec<ChatMessage> = Vec::new();
    let message = ChatMessage::assistant("just text");
    let mut notifications = Vec::new();

    replay_assistant_message(&session_id, &history, 0, &message, &mut |notification| {
        notifications.push(notification);
        Ok(())
    })?;

    assert_eq!(notifications.len(), 1);
    let SessionUpdate::AgentMessageChunk(chunk) = &notifications[0].update else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected assistant message chunk")
        );
    };
    assert!(chunk.message_id.is_some());
    Ok(())
}

// ── restore_persisted_session: id mismatch ──────────────

#[test_log::test(tokio::test)]
async fn restore_persisted_session_rejects_mismatched_id()
-> Result<(), agent_client_protocol::Error> {
    let state_dir =
        std::env::temp_dir().join(format!("deepseek-acp-restore-id-{}", Uuid::new_v4()));
    let workspace = state_dir.join("workspace");
    let persistence = FilesystemSessionStore::new(&state_dir);
    let store = SessionStore::new(Arc::new(Mutex::new(AdapterState::default())))
        .with_persistence(persistence.clone());

    // Persist a session normally with matching meta.session_id and
    // directory name, so load_record succeeds.
    let session_id = agent_client_protocol::schema::SessionId::new("session-restore-id");
    persistence
        .persist_turn(
            &PersistedSessionMeta {
                session_id: session_id.0.to_string(),
                cwd: workspace.clone(),
                additional_directories: Vec::new(),
                mode: PermissionPosture::Ask,
                model: "deepseek-v4-pro".to_string(),
                reasoning_effort: ReasoningEffort::High,
                mcp_servers: Vec::new(),
                title: None,
                updated_at: None,
            },
            &[ChatMessage::user("hello")],
        )
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    // Overwrite meta.json with a mismatched session_id so the
    // id-comparison guard in restore_persisted_session fires.
    let meta_dir = state_dir.join("sessions").join("session-restore-id");
    let mismatched_meta = PersistedSessionMeta {
        session_id: "session-restore-id-mismatched".to_string(),
        cwd: workspace.clone(),
        additional_directories: Vec::new(),
        mode: PermissionPosture::Ask,
        model: "deepseek-v4-pro".to_string(),
        reasoning_effort: ReasoningEffort::High,
        mcp_servers: Vec::new(),
        title: None,
        updated_at: None,
    };
    let meta_json = serde_json::to_string(&mismatched_meta)
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    std::fs::write(meta_dir.join("meta.json"), meta_json.as_bytes())
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    let Err(error) = restore_persisted_session(&store, &session_id, &workspace).await else {
        return Err(agent_client_protocol::Error::internal_error()
            .data("expected mismatched persisted session id to fail"));
    };

    assert!(
        error
            .to_string()
            .contains("does not match requested session id")
    );
    Ok(())
}

// ── _meta field: historyJsonlPath ───────────────────────

#[test]
fn new_session_without_persistence_omits_meta() -> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let response = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
    assert!(response.meta.is_none());
    Ok(())
}

#[test]
fn new_session_with_persistence_includes_history_jsonl_path_in_meta()
-> Result<(), agent_client_protocol::Error> {
    let state_dir = std::env::temp_dir().join(format!("deepseek-acp-new-meta-{}", Uuid::new_v4()));
    let store = SessionStore::new(Arc::new(Mutex::new(AdapterState::default())))
        .with_persistence(FilesystemSessionStore::new(&state_dir));
    let response = handle_new_session_request(&store, &NewSessionRequest::new(&state_dir))?;

    let meta = response
        .meta
        .ok_or_else(|| agent_client_protocol::Error::internal_error().data("missing _meta"))?;
    let path = meta
        .get("historyJsonlPath")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            agent_client_protocol::Error::internal_error()
                .data("missing or non-string historyJsonlPath in _meta")
        })?;
    assert!(
        path.ends_with("/history.jsonl"),
        "expected path to end with /history.jsonl, got: {path}"
    );
    assert!(
        path.contains(&state_dir.to_string_lossy().to_string()),
        "expected path to contain state dir, got: {path}"
    );
    Ok(())
}

#[test_log::test(tokio::test)]
async fn load_session_response_includes_history_jsonl_path_in_meta()
-> Result<(), agent_client_protocol::Error> {
    let state_dir = std::env::temp_dir().join(format!("deepseek-acp-load-meta-{}", Uuid::new_v4()));
    let workspace = state_dir.join("workspace");
    let persistence = FilesystemSessionStore::new(&state_dir);
    let store = SessionStore::new(Arc::new(Mutex::new(AdapterState::default())))
        .with_persistence(persistence.clone());
    let session_id = agent_client_protocol::schema::SessionId::new("session-load-meta");
    persistence
        .persist_turn(
            &PersistedSessionMeta {
                session_id: session_id.0.to_string(),
                cwd: workspace.clone(),
                additional_directories: Vec::new(),
                mode: PermissionPosture::Ask,
                model: "deepseek-v4-pro".to_string(),
                reasoning_effort: ReasoningEffort::High,
                mcp_servers: Vec::new(),
                title: None,
                updated_at: None,
            },
            &[ChatMessage::user("hello")],
        )
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    let response = handle_load_session_request(
        &store,
        &LoadSessionRequest::new(session_id.clone(), workspace.clone()),
        |_| Ok(()),
    )
    .await?;

    let meta = response
        .meta
        .ok_or_else(|| agent_client_protocol::Error::internal_error().data("missing _meta"))?;
    let path = meta
        .get("historyJsonlPath")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            agent_client_protocol::Error::internal_error()
                .data("missing or non-string historyJsonlPath in _meta")
        })?;
    assert!(
        path.ends_with("/history.jsonl"),
        "expected path to end with /history.jsonl, got: {path}"
    );
    assert!(
        path.contains(session_id.0.as_ref()),
        "expected path to contain session id, got: {path}"
    );
    Ok(())
}

#[test_log::test(tokio::test)]
async fn resume_session_response_includes_history_jsonl_path_in_meta()
-> Result<(), agent_client_protocol::Error> {
    let state_dir =
        std::env::temp_dir().join(format!("deepseek-acp-resume-meta-{}", Uuid::new_v4()));
    let workspace = state_dir.join("workspace");
    let persistence = FilesystemSessionStore::new(&state_dir);
    let store = SessionStore::new(Arc::new(Mutex::new(AdapterState::default())))
        .with_persistence(persistence.clone());
    let session_id = agent_client_protocol::schema::SessionId::new("session-resume-meta");
    persistence
        .persist_turn(
            &PersistedSessionMeta {
                session_id: session_id.0.to_string(),
                cwd: workspace.clone(),
                additional_directories: Vec::new(),
                mode: PermissionPosture::Ask,
                model: "deepseek-v4-pro".to_string(),
                reasoning_effort: ReasoningEffort::High,
                mcp_servers: Vec::new(),
                title: None,
                updated_at: None,
            },
            &[ChatMessage::user("hello")],
        )
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    let response = handle_resume_session_request(
        &store,
        &ResumeSessionRequest::new(session_id.clone(), workspace.clone()),
    )
    .await?;

    let meta = response
        .meta
        .ok_or_else(|| agent_client_protocol::Error::internal_error().data("missing _meta"))?;
    let path = meta
        .get("historyJsonlPath")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            agent_client_protocol::Error::internal_error()
                .data("missing or non-string historyJsonlPath in _meta")
        })?;
    assert!(
        path.ends_with("/history.jsonl"),
        "expected path to end with /history.jsonl, got: {path}"
    );
    assert!(
        path.contains(session_id.0.as_ref()),
        "expected path to contain session id, got: {path}"
    );
    Ok(())
}

#[test]
fn list_sessions_with_persistence_includes_history_jsonl_path_in_meta()
-> Result<(), agent_client_protocol::Error> {
    let state_dir = std::env::temp_dir().join(format!("deepseek-acp-list-meta-{}", Uuid::new_v4()));
    let workspace = state_dir.join("workspace");
    let store = SessionStore::new(Arc::new(Mutex::new(AdapterState::default())))
        .with_persistence(FilesystemSessionStore::new(&state_dir));
    let session = handle_new_session_request(&store, &NewSessionRequest::new(&workspace))?;

    let response =
        handle_list_sessions_request(&store, &ListSessionsRequest::new().cwd(&workspace))?;
    assert_eq!(response.sessions.len(), 1);
    let info = &response.sessions[0];

    let meta = info.meta.as_ref().ok_or_else(|| {
        agent_client_protocol::Error::internal_error().data("missing _meta on SessionInfo")
    })?;
    let path = meta
        .get("historyJsonlPath")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            agent_client_protocol::Error::internal_error()
                .data("missing or non-string historyJsonlPath in _meta")
        })?;
    assert!(
        path.ends_with("/history.jsonl"),
        "expected path to end with /history.jsonl, got: {path}"
    );
    assert!(
        path.contains(session.session_id.0.as_ref()),
        "expected path to contain session id, got: {path}"
    );
    Ok(())
}
