//! Session state, permission model, and in-memory session store.
//!
//! This module owns the domain types that were previously scattered through
//! `main.rs`.  Extracting them here fixes the dependency direction: protocol
//! handlers in [`crate::acp`] now depend on `crate::session` instead of the
//! binary entrypoint.

use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::{
    ClientCapabilities, McpServer, PermissionOption, PermissionOptionKind,
    RequestPermissionOutcome, RequestPermissionRequest, SessionConfigOption,
    SessionConfigOptionCategory, SessionConfigSelectOption, SessionConfigValueId, SessionId,
    SessionInfo, SessionMode, SessionModeId, SessionModeState, ToolCallStatus, ToolCallUpdate,
    ToolCallUpdateFields, ToolKind,
};
use deepseek_acp_adapter::deepseek::{
    ChatMessage, DeepSeekConfig, ToolCall as DeepSeekToolCall, ToolCallDelta, ToolDefinition,
};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::acp::PermissionRequester;
use crate::mcp::{McpSession, McpToolTarget};
use crate::session_store::{FilesystemSessionStore, PersistedSessionMeta, PersistedSessionRecord};
use crate::tools::ToolContext;
use crate::turn::tool_raw_input;

/// Default maximum number of tool-call/response cycles per prompt turn.
pub(crate) const DEFAULT_MAX_TURN_REQUESTS: NonZeroUsize = NonZeroUsize::MIN.saturating_add(99);
pub(crate) const PERMISSION_ALLOW_ONCE_OPTION_ID: &str = "allow_once";
pub(crate) const PERMISSION_ALLOW_ALWAYS_OPTION_ID: &str = "allow_always";
pub(crate) const PERMISSION_REJECT_ONCE_OPTION_ID: &str = "reject_once";
pub(crate) const PERMISSION_REJECT_ALWAYS_OPTION_ID: &str = "reject_always";
pub(crate) const SESSION_MODE_ASK_ID: &str = "ask";
pub(crate) const SESSION_MODE_ACCEPT_EDITS_ID: &str = "accept-edits";
pub(crate) const SESSION_MODE_YOLO_ID: &str = "yolo";
pub(crate) const SESSION_CONFIG_MODE_ID: &str = "mode";
pub(crate) const SESSION_CONFIG_MODEL_ID: &str = "model";
pub(crate) const SESSION_CONFIG_REASONING_EFFORT_ID: &str = "reasoning_effort";
pub(crate) const DEEPSEEK_V4_FLASH_MODEL_ID: &str = "deepseek-v4-flash";
pub(crate) const DEEPSEEK_V4_PRO_MODEL_ID: &str = "deepseek-v4-pro";
pub(crate) const REASONING_EFFORT_HIGH_ID: &str = "high";
pub(crate) const REASONING_EFFORT_MAX_ID: &str = "max";

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
pub(crate) enum PermissionPosture {
    #[default]
    Ask,
    AcceptEdits,
    Yolo,
}

impl PermissionPosture {
    pub(crate) fn mode_id(self) -> SessionModeId {
        match self {
            Self::Ask => SessionModeId::new(SESSION_MODE_ASK_ID),
            Self::AcceptEdits => SessionModeId::new(SESSION_MODE_ACCEPT_EDITS_ID),
            Self::Yolo => SessionModeId::new(SESSION_MODE_YOLO_ID),
        }
    }

    pub(crate) const fn allows_without_prompt(self, kind: ToolKind) -> bool {
        match self {
            Self::Ask => false,
            Self::AcceptEdits => matches!(kind, ToolKind::Edit),
            Self::Yolo => !matches!(
                kind,
                ToolKind::Read | ToolKind::Search | ToolKind::Think | ToolKind::Fetch
            ),
        }
    }

    pub(crate) fn from_mode_id(mode_id: &SessionModeId) -> Option<Self> {
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
pub(crate) enum ReasoningEffort {
    #[default]
    High,
    Max,
}

impl ReasoningEffort {
    pub(crate) const fn id(self) -> &'static str {
        match self {
            Self::High => REASONING_EFFORT_HIGH_ID,
            Self::Max => REASONING_EFFORT_MAX_ID,
        }
    }

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::High => "High",
            Self::Max => "Max",
        }
    }

    pub(crate) const fn description(self) -> &'static str {
        match self {
            Self::High => "Default DeepSeek thinking effort.",
            Self::Max => "Maximum DeepSeek thinking effort for complex agent work.",
        }
    }

    pub(crate) fn from_value_id(value: &SessionConfigValueId) -> Option<Self> {
        match value.0.as_ref() {
            REASONING_EFFORT_HIGH_ID => Some(Self::High),
            REASONING_EFFORT_MAX_ID => Some(Self::Max),
            _ => None,
        }
    }
}

/// Ask the client (or fall back to posture) whether a tool call is allowed.
///
/// # Errors
///
/// Returns an ACP error when the session is unknown, the permission request
/// cannot be sent, or the client returns an unrecognized outcome.
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
pub(crate) struct PendingToolCalls {
    calls: Vec<PendingToolCall>,
}

impl PendingToolCalls {
    pub(crate) fn push(&mut self, delta: &ToolCallDelta) {
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

    pub(crate) fn finish(self) -> Result<Vec<DeepSeekToolCall>, agent_client_protocol::Error> {
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

pub(crate) fn default_session_modes() -> SessionModeState {
    SessionModeState::new(
        PermissionPosture::Ask.mode_id(),
        vec![
            SessionMode::new(PermissionPosture::Ask.mode_id(), "Ask"),
            SessionMode::new(PermissionPosture::AcceptEdits.mode_id(), "Accept edits"),
            SessionMode::new(PermissionPosture::Yolo.mode_id(), "Yolo"),
        ],
    )
}

pub(crate) fn session_config_options(session: &SessionRecord) -> Vec<SessionConfigOption> {
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

pub(crate) fn model_select_options(current_model: &str) -> Vec<SessionConfigSelectOption> {
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

pub(crate) fn validate_session_model(
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

pub(crate) fn initial_model_from_env() -> String {
    std::env::var("DEEPSEEK_MODEL")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DeepSeekConfig::DEFAULT_MODEL.to_string())
}

#[derive(Debug)]
pub(crate) struct AdapterState {
    pub(crate) default_model: String,
    pub(crate) client_capabilities: Option<ClientCapabilities>,
    pub(crate) sessions: HashMap<SessionId, SessionRecord>,
}

impl AdapterState {
    pub(crate) fn new(default_model: impl Into<String>) -> Self {
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
pub(crate) struct SessionRecord {
    pub(crate) cwd: PathBuf,
    pub(crate) additional_directories: Vec<PathBuf>,
    pub(crate) history: Vec<ChatMessage>,
    pub(crate) active_turn: Option<CancellationToken>,
    pub(crate) mode: PermissionPosture,
    pub(crate) model: String,
    pub(crate) reasoning_effort: ReasoningEffort,
    pub(crate) permission_allow_always: HashSet<String>,
    pub(crate) mcp_servers: Vec<McpServer>,
    pub(crate) mcp_sessions: Vec<McpSession>,
}

/// Narrow boundary around shared adapter state.
///
/// `SessionStore` wraps the internal `Arc<Mutex<AdapterState>>` and exposes
/// targeted methods for session lifecycle, config, permission cache, and
/// active-turn management. Application code uses `SessionStore` directly
/// instead of passing raw `Arc<Mutex<AdapterState>>` through every layer.
#[derive(Debug, Clone)]
pub(crate) struct SessionStore {
    pub(crate) state: Arc<Mutex<AdapterState>>,
    persistence: Option<FilesystemSessionStore>,
}

/// Snapshot of session data needed to begin a prompt turn.
///
/// Returned by [`SessionStore::begin_turn`] so the caller does not need to
/// hold the lock across model streaming.
#[derive(Debug)]
pub(crate) struct TurnSetup {
    pub(crate) messages: Vec<ChatMessage>,
    pub(crate) tool_context: ToolContext,
    pub(crate) model: String,
    pub(crate) reasoning_effort: ReasoningEffort,
}

impl SessionStore {
    /// Wrap an existing `Arc<Mutex<AdapterState>>` in a `SessionStore`.
    pub(crate) fn new(state: Arc<Mutex<AdapterState>>) -> Self {
        Self {
            state,
            persistence: None,
        }
    }

    /// Attach filesystem persistence to the session store.
    pub(crate) fn with_persistence(mut self, persistence: FilesystemSessionStore) -> Self {
        self.persistence = Some(persistence);
        self
    }

    /// Load a persisted session record from the filesystem store.
    pub(crate) fn load_persisted_record(
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
    pub(crate) fn record_client_capabilities(
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
    pub(crate) fn list_sessions(
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
    pub(crate) fn remove_session(
        &self,
        session_id: &SessionId,
    ) -> Result<bool, agent_client_protocol::Error> {
        let mut guard = self
            .state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        Ok(guard.sessions.remove(session_id).is_some())
    }

    /// Insert a new session record.
    pub(crate) fn insert_session(
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
    pub(crate) fn default_model(&self) -> Result<String, agent_client_protocol::Error> {
        let guard = self
            .state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        Ok(guard.default_model.clone())
    }

    /// Look up a session and return a read-only reference via a callback.
    ///
    /// The lock is held only for the duration of the callback.
    pub(crate) fn with_session<T>(
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
    pub(crate) fn with_session_mut<T>(
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
    pub(crate) fn mcp_definitions(
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
    pub(crate) fn find_mcp_target(
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
    pub(crate) fn is_always_allowed(
        &self,
        session_id: &SessionId,
        tool_name: &str,
    ) -> Result<bool, agent_client_protocol::Error> {
        self.with_session(session_id, |session| {
            Ok(session.permission_allow_always.contains(tool_name))
        })
    }

    /// Return the current permission posture (mode) for a session.
    pub(crate) fn permission_posture(
        &self,
        session_id: &SessionId,
    ) -> Result<PermissionPosture, agent_client_protocol::Error> {
        self.with_session(session_id, |session| Ok(session.mode))
    }

    /// Insert a tool name into the session's allow-always cache.
    pub(crate) fn add_always_allow(
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
    pub(crate) fn cancel_active_turn(
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
    pub(crate) fn clear_active_turn(
        &self,
        session_id: &SessionId,
    ) -> Result<(), agent_client_protocol::Error> {
        self.with_session_mut(session_id, |session| {
            session.active_turn = None;
            Ok(())
        })
    }

    /// Set the permission posture (mode) for a session.
    pub(crate) fn set_mode(
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
    pub(crate) fn set_model(
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
    pub(crate) fn set_reasoning_effort(
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
    pub(crate) fn begin_turn(
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
    pub(crate) fn save_history(
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
    pub(crate) fn session_config_options(
        &self,
        session_id: &SessionId,
    ) -> Result<Vec<SessionConfigOption>, agent_client_protocol::Error> {
        self.with_session(session_id, |session| Ok(session_config_options(session)))
    }

    /// Look up a session record for a new-session response.
    pub(crate) fn lookup_session(
        &self,
        session_id: &SessionId,
    ) -> Result<(), agent_client_protocol::Error> {
        self.with_session(session_id, |_session| Ok(()))
    }
}

#[cfg(test)]
mod tests;
