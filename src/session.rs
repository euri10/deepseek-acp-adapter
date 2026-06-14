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
use deepseek_acp_adapter::deepseek::MessageRole;
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
use deepseek_acp_adapter::error::AdapterError;
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
/// Returns an [`AdapterError`] when the session is unknown, the permission request
/// cannot be sent, or the client returns an unrecognized outcome.
pub(crate) async fn request_tool_permission(
    store: &SessionStore,
    context: &ToolContext,
    call: &DeepSeekToolCall,
    kind: ToolKind,
    requester: &dyn PermissionRequester,
) -> Result<PermissionDecision, AdapterError> {
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
                .title(crate::turn::tool_call_title(call))
                .raw_input(tool_raw_input(call)),
        ),
        permission_options(),
    );

    let response = requester
        .request_permission(request)
        .await
        .map_err(|e| AdapterError::Internal(e.to_string()))?;
    let decision = match response.outcome {
        RequestPermissionOutcome::Cancelled => PermissionDecision::Cancelled,
        RequestPermissionOutcome::Selected(selected) => match selected.option_id.0.as_ref() {
            PERMISSION_ALLOW_ONCE_OPTION_ID => PermissionDecision::AllowOnce,
            PERMISSION_ALLOW_ALWAYS_OPTION_ID => PermissionDecision::AllowAlways,
            PERMISSION_REJECT_ONCE_OPTION_ID => PermissionDecision::RejectOnce,
            PERMISSION_REJECT_ALWAYS_OPTION_ID => PermissionDecision::RejectAlways,
            other => {
                return Err(AdapterError::InvalidParams(format!(
                    "unknown permission option selected: {other}"
                )));
            }
        },
        _ => {
            return Err(AdapterError::InvalidParams(
                "unsupported permission outcome variant".to_string(),
            ));
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

    pub(crate) fn finish(self) -> Result<Vec<DeepSeekToolCall>, AdapterError> {
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
    fn finish(self, index: usize) -> Result<DeepSeekToolCall, AdapterError> {
        let id = self.id.ok_or_else(|| {
            AdapterError::InvalidParams(format!("tool call delta {index} is missing an id"))
        })?;
        let name = self.name.ok_or_else(|| {
            AdapterError::InvalidParams(format!(
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
) -> Result<(), AdapterError> {
    if is_known_model(model) || model == session.model {
        return Ok(());
    }

    Err(AdapterError::InvalidParams(format!(
        "unsupported DeepSeek model: {model}"
    )))
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

/// Produce an ISO 8601 UTC timestamp string from the system clock.
///
/// Uses public-domain calendar arithmetic (Hinnant) to avoid pulling in
/// `chrono` or `time` as a direct dependency.  The format is
/// `YYYY-MM-DDTHH:MM:SSZ` with second precision.
pub(crate) fn iso_timestamp_now() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map_or_else(
            || "1970-01-01T00:00:00Z".to_string(),
            |dur| {
                let secs = dur.as_secs();
                let days = secs / 86400;
                let seconds_today = secs % 86400;

                let hours = seconds_today / 3600;
                let minutes = (seconds_today % 3600) / 60;
                let seconds = seconds_today % 60;

                let (year, month, day) = unix_days_to_ymd(days);

                format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
            },
        )
}

fn unix_days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    days += 719_468; // Adjust to proleptic Gregorian calendar
    let era = days / 146_097;
    let day_of_era = days % 146_097;

    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;

    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);

    let month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month + 2) / 5 + 1;

    let month = if month < 10 { month + 3 } else { month - 9 };
    let year = if month <= 2 { year + 1 } else { year };

    (year, month, day)
}

/// Derive a human-readable session title from the message history.
///
/// Returns the first user message's content truncated to 80 characters, or
/// `"New session"` if no user message is found.
pub(crate) fn derive_session_title(history: &[ChatMessage]) -> String {
    const MAX_TITLE_LEN: usize = 80;

    history
        .iter()
        .find(|msg| msg.role() == MessageRole::User)
        .map_or_else(
            || "New session".to_string(),
            |msg| {
                let text: String = msg
                    .content()
                    .chars()
                    .map(|c| if c == '\n' { ' ' } else { c })
                    .collect();
                let trimmed = text.trim();
                if trimmed.chars().count() <= MAX_TITLE_LEN {
                    trimmed.to_string()
                } else {
                    format!(
                        "{}…",
                        trimmed.chars().take(MAX_TITLE_LEN).collect::<String>()
                    )
                }
            },
        )
}

#[derive(Debug)]
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
    /// Human-readable session title, derived from the first user message.
    pub(crate) title: String,
    /// ISO 8601 timestamp of the last activity.
    pub(crate) updated_at: String,
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
    /// Human-readable session title after the turn begins.
    pub(crate) title: String,
    /// Whether this turn derived the title for the first time.
    pub(crate) title_changed: bool,
    /// ISO 8601 timestamp of the session's latest activity.
    pub(crate) updated_at: String,
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
    ) -> Result<PersistedSessionRecord, AdapterError> {
        let Some(persistence) = &self.persistence else {
            return Err(AdapterError::InvalidRequest(
                "session/load requires filesystem persistence".to_string(),
            ));
        };

        persistence
            .load_record(session_id.0.as_ref())
            .map_err(|e| AdapterError::Internal(e.to_string()))
    }

    /// Return the absolute path to the session's `history.jsonl` file, if
    /// filesystem persistence is configured.
    pub(crate) fn history_jsonl_path(&self, session_id: &SessionId) -> Option<PathBuf> {
        self.persistence
            .as_ref()
            .and_then(|p| p.history_jsonl_path(session_id.0.as_ref()).ok())
    }

    /// Build a `serde_json::Map` for the `_meta` field containing the history
    /// JSONL path, if persistence is available.
    pub(crate) fn session_meta(
        &self,
        session_id: &SessionId,
    ) -> Option<serde_json::Map<String, serde_json::Value>> {
        let path = self.history_jsonl_path(session_id)?;
        let mut meta = serde_json::Map::new();
        meta.insert(
            "historyJsonlPath".to_string(),
            serde_json::Value::String(path.to_string_lossy().to_string()),
        );
        Some(meta)
    }

    /// Store the client capabilities reported during initialization.
    pub(crate) fn record_client_capabilities(
        &self,
        client_capabilities: ClientCapabilities,
    ) -> Result<(), AdapterError> {
        let mut guard = self
            .state
            .lock()
            .map_err(|e| AdapterError::Internal(e.to_string()))?;
        guard.client_capabilities = Some(client_capabilities);
        Ok(())
    }

    /// Return a snapshot of matching sessions for the `session/list` handler.
    pub(crate) fn list_sessions(
        &self,
        cwd_filter: Option<&Path>,
    ) -> Result<Vec<SessionInfo>, AdapterError> {
        let (mut sessions, persistence) = {
            let guard = self
                .state
                .lock()
                .map_err(|e| AdapterError::Internal(e.to_string()))?;
            (
                guard
                    .sessions
                    .iter()
                    .filter(|(_session_id, record)| cwd_filter.is_none_or(|cwd| record.cwd == cwd))
                    .map(|(session_id, record)| {
                        let mut info = SessionInfo::new(session_id.clone(), record.cwd.clone())
                            .additional_directories(record.additional_directories.clone());
                        if !record.title.is_empty() {
                            info = info.title(record.title.clone());
                        }
                        info = info.updated_at(record.updated_at.clone());
                        info
                    })
                    .collect::<Vec<_>>(),
                self.persistence.clone(),
            )
        };

        if let Some(persistence) = persistence {
            let persisted_list = persistence
                .list_persisted()
                .map_err(|e| AdapterError::Internal(e.to_string()))?;
            for persisted in persisted_list {
                // Always include ALL persisted sessions, regardless of cwd_filter.
                // Users need to see all saved sessions to resume them, even from different directories.
                // The cwd validation (if needed) happens during the actual resume call.
                if !sessions
                    .iter()
                    .any(|session| session.session_id == persisted.session_id)
                {
                    sessions.push(persisted);
                }
            }
        }

        // Attach _meta with the history JSONL path for every session that has
        // a filesystem persistence store configured.
        if self.persistence.is_some() {
            for session in &mut sessions {
                if let Some(meta) = self.session_meta(&session.session_id) {
                    // `meta` is a public field on SessionInfo; direct assignment
                    // is allowed despite `#[non_exhaustive]` (which only restricts
                    // struct literal construction and exhaustive matching).
                    session.meta = Some(meta);
                }
            }
        }

        // Sort most-recently-updated first so that the /resume picker surfaces
        // recent sessions at the top regardless of filesystem iteration order.
        // Sessions without an `updated_at` timestamp sort to the end.
        sessions.sort_by(|a, b| {
            let a_ts = a.updated_at.as_deref().unwrap_or("");
            let b_ts = b.updated_at.as_deref().unwrap_or("");
            b_ts.cmp(a_ts)
        });

        Ok(sessions)
    }

    /// Remove a session by id. Returns `true` if the session existed.
    pub(crate) fn remove_session(&self, session_id: &SessionId) -> Result<bool, AdapterError> {
        let mut guard = self
            .state
            .lock()
            .map_err(|e| AdapterError::Internal(e.to_string()))?;
        Ok(guard.sessions.remove(session_id).is_some())
    }

    /// Remove a session from memory and persistent storage.
    pub(crate) fn delete_session(&self, session_id: &SessionId) -> Result<bool, AdapterError> {
        let persistence = self.persistence.clone();
        let deleted_from_memory = {
            let mut guard = self
                .state
                .lock()
                .map_err(|e| AdapterError::Internal(e.to_string()))?;
            if let Some(session) = guard.sessions.get(session_id)
                && let Some(token) = &session.active_turn
            {
                token.cancel();
            }
            guard.sessions.remove(session_id).is_some()
        };

        let deleted_from_persistence = if let Some(persistence) = persistence {
            persistence
                .delete_session(session_id.0.as_ref())
                .map_err(|e| AdapterError::Internal(e.to_string()))?
        } else {
            false
        };

        Ok(deleted_from_memory || deleted_from_persistence)
    }

    /// Insert a new session record.
    pub(crate) fn insert_session(
        &self,
        session_id: SessionId,
        record: SessionRecord,
    ) -> Result<(), AdapterError> {
        let mut guard = self
            .state
            .lock()
            .map_err(|e| AdapterError::Internal(e.to_string()))?;
        guard.sessions.insert(session_id, record);
        Ok(())
    }

    /// Return the default model identifier for new sessions.
    pub(crate) fn default_model(&self) -> Result<String, AdapterError> {
        let guard = self
            .state
            .lock()
            .map_err(|e| AdapterError::Internal(e.to_string()))?;
        Ok(guard.default_model.clone())
    }

    /// Look up a session and return a read-only reference via a callback.
    ///
    /// The lock is held only for the duration of the callback.
    pub(crate) fn with_session<T>(
        &self,
        session_id: &SessionId,
        f: impl FnOnce(&SessionRecord) -> Result<T, AdapterError>,
    ) -> Result<T, AdapterError> {
        let guard = self
            .state
            .lock()
            .map_err(|e| AdapterError::Internal(e.to_string()))?;
        let session = guard.sessions.get(session_id).ok_or_else(|| {
            AdapterError::InvalidParams(format!("unknown session id: {}", session_id.0))
        })?;
        f(session)
    }

    /// Look up a session and invoke a callback with a mutable reference.
    ///
    /// The lock is held only for the duration of the callback.
    pub(crate) fn with_session_mut<T>(
        &self,
        session_id: &SessionId,
        f: impl FnOnce(&mut SessionRecord) -> Result<T, AdapterError>,
    ) -> Result<T, AdapterError> {
        let mut guard = self
            .state
            .lock()
            .map_err(|e| AdapterError::Internal(e.to_string()))?;
        let session = guard.sessions.get_mut(session_id).ok_or_else(|| {
            AdapterError::InvalidParams(format!("unknown session id: {}", session_id.0))
        })?;
        f(session)
    }

    /// Return the MCP tool definitions registered for a session.
    pub(crate) fn mcp_definitions(
        &self,
        session_id: &SessionId,
    ) -> Result<Vec<ToolDefinition>, AdapterError> {
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
    ) -> Result<Option<McpToolTarget>, AdapterError> {
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
    ) -> Result<bool, AdapterError> {
        self.with_session(session_id, |session| {
            Ok(session.permission_allow_always.contains(tool_name))
        })
    }

    /// Return the current permission posture (mode) for a session.
    pub(crate) fn permission_posture(
        &self,
        session_id: &SessionId,
    ) -> Result<PermissionPosture, AdapterError> {
        self.with_session(session_id, |session| Ok(session.mode))
    }

    /// Insert a tool name into the session's allow-always cache.
    pub(crate) fn add_always_allow(
        &self,
        session_id: &SessionId,
        tool_name: String,
    ) -> Result<(), AdapterError> {
        self.with_session_mut(session_id, |session| {
            session.permission_allow_always.insert(tool_name);
            Ok(())
        })
    }

    /// Cancel the active turn token for a session, if one exists.
    pub(crate) fn cancel_active_turn(&self, session_id: &SessionId) -> Result<(), AdapterError> {
        self.with_session(session_id, |session| {
            if let Some(token) = &session.active_turn {
                token.cancel();
            }
            Ok(())
        })
    }

    /// Clear the active turn token for a session.
    pub(crate) fn clear_active_turn(&self, session_id: &SessionId) -> Result<(), AdapterError> {
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
    ) -> Result<(), AdapterError> {
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
    ) -> Result<(), AdapterError> {
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
    ) -> Result<(), AdapterError> {
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
    ) -> Result<TurnSetup, AdapterError> {
        let mut guard = self
            .state
            .lock()
            .map_err(|e| AdapterError::Internal(e.to_string()))?;
        let client_capabilities = guard.client_capabilities.clone();
        let session = guard.sessions.get_mut(session_id).ok_or_else(|| {
            AdapterError::InvalidParams(format!("unknown session id: {}", session_id.0))
        })?;

        if session.active_turn.is_some() {
            return Err(AdapterError::InvalidRequest(format!(
                "session {} already has an active turn",
                session_id.0
            )));
        }
        session.active_turn = Some(token);

        // Bump the last-activity timestamp on every prompt turn.
        session.updated_at = iso_timestamp_now();

        // Derive title from the first user message if not yet set.
        // We check the history + this turn's user message together, because
        // on the very first turn the history is empty and the current prompt
        // is the only user message available.
        let title_changed = session.title.is_empty();
        if title_changed {
            let mut candidate_messages = session.history.clone();
            candidate_messages.push(user_message.clone());
            session.title = derive_session_title(&candidate_messages);
        }

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
            title: session.title.clone(),
            title_changed,
            updated_at: session.updated_at.clone(),
        })
    }

    /// Persist the messages as the session history after a turn completes.
    pub(crate) fn save_history(
        &self,
        session_id: &SessionId,
        messages: &[ChatMessage],
    ) -> Result<(), AdapterError> {
        let (persistence, meta, new_messages) = {
            let guard = self
                .state
                .lock()
                .map_err(|e| AdapterError::Internal(e.to_string()))?;
            let session = guard.sessions.get(session_id).ok_or_else(|| {
                AdapterError::InvalidParams(format!("unknown session id: {}", session_id.0))
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
                    title: Some(session.title.clone()),
                    updated_at: Some(session.updated_at.clone()),
                },
                new_messages,
            )
        };

        if let Some(persistence) = persistence {
            persistence
                .persist_turn(&meta, &new_messages)
                .map_err(|e| AdapterError::Internal(e.to_string()))?;
        }

        self.with_session_mut(session_id, |session| {
            session.history = messages.to_vec();
            Ok(())
        })
    }

    /// Return the session config options for a session.
    pub(crate) fn session_config_options(
        &self,
        session_id: &SessionId,
    ) -> Result<Vec<SessionConfigOption>, AdapterError> {
        self.with_session(session_id, |session| Ok(session_config_options(session)))
    }

    /// Look up a session record for a new-session response.
    pub(crate) fn lookup_session(&self, session_id: &SessionId) -> Result<(), AdapterError> {
        self.with_session(session_id, |_session| Ok(()))
    }
}

#[cfg(test)]
mod tests;
