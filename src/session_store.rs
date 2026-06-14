//! Filesystem-backed session persistence.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use agent_client_protocol::schema::{McpServer, SessionId, SessionInfo};
use deepseek_acp_adapter::deepseek::ChatMessage;
use deepseek_acp_adapter::error::SessionPersistenceError;
use serde::{Deserialize, Serialize};

use crate::{PermissionPosture, ReasoningEffort};

const SESSIONS_DIR: &str = "sessions";
const META_FILE: &str = "meta.json";
const HISTORY_FILE: &str = "history.jsonl";
const APPLICATION_STATE_DIR: &str = "deepseek-acp-adapter";

/// Filesystem-backed persistence for ACP session metadata and chat history.
#[derive(Debug, Clone)]
pub(crate) struct FilesystemSessionStore {
    state_dir: PathBuf,
}

/// Persisted metadata for one ACP session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PersistedSessionMeta {
    /// ACP session id.
    pub(crate) session_id: String,
    /// Session working directory.
    pub(crate) cwd: PathBuf,
    /// Additional directories available to the session.
    pub(crate) additional_directories: Vec<PathBuf>,
    /// Permission mode active for the session.
    pub(crate) mode: PermissionPosture,
    /// Model selected for the session.
    pub(crate) model: String,
    /// `DeepSeek` reasoning effort selected for the session.
    pub(crate) reasoning_effort: ReasoningEffort,
    /// MCP servers originally attached to the session.
    pub(crate) mcp_servers: Vec<McpServer>,
    /// Human-readable session title (absent in sessions created before this field was added).
    pub(crate) title: Option<String>,
    /// ISO 8601 timestamp of last activity (absent in sessions created before this field was added).
    pub(crate) updated_at: Option<String>,
}

/// Persisted session metadata plus replayable chat history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PersistedSessionRecord {
    /// Metadata loaded from `meta.json`.
    pub(crate) meta: PersistedSessionMeta,
    /// Chat messages loaded from `history.jsonl`.
    pub(crate) history: Vec<ChatMessage>,
}

impl FilesystemSessionStore {
    /// Create a store rooted at `state_dir`.
    pub(crate) fn new(state_dir: impl Into<PathBuf>) -> Self {
        Self {
            state_dir: state_dir.into(),
        }
    }

    /// Create a store rooted under `XDG_STATE_HOME` or `$HOME/.local/state`.
    pub(crate) fn from_default_state_dir() -> Result<Self, SessionPersistenceError> {
        Ok(Self::new(default_state_dir()?))
    }

    /// Append a completed turn's new messages and refresh session metadata.
    pub(crate) fn persist_turn(
        &self,
        meta: &PersistedSessionMeta,
        messages: &[ChatMessage],
    ) -> Result<(), SessionPersistenceError> {
        let session_dir = self.session_dir(&meta.session_id)?;
        fs::create_dir_all(&session_dir)?;
        Self::write_meta(&session_dir, meta)?;
        Self::append_history(&session_dir, messages)?;
        Ok(())
    }

    /// Load one persisted session record by id.
    pub(crate) fn load_record(
        &self,
        session_id: &str,
    ) -> Result<PersistedSessionRecord, SessionPersistenceError> {
        let session_dir = self.session_dir(session_id)?;
        let meta = Self::read_meta(&session_dir)?;
        let history = Self::read_history(&session_dir)?;
        Ok(PersistedSessionRecord { meta, history })
    }

    /// Delete a persisted session directory, including metadata and history.
    ///
    /// Returns `true` when a session directory existed and was removed.
    pub(crate) fn delete_session(&self, session_id: &str) -> Result<bool, SessionPersistenceError> {
        let session_dir = self.session_dir(session_id)?;
        if !session_dir.exists() {
            return Ok(false);
        }

        fs::remove_dir_all(session_dir)?;
        Ok(true)
    }

    /// List all persisted sessions regardless of working directory.
    pub(crate) fn list_persisted(&self) -> Result<Vec<SessionInfo>, SessionPersistenceError> {
        let sessions_dir = self.state_dir.join(SESSIONS_DIR);
        if !sessions_dir.exists() {
            return Ok(Vec::new());
        }

        let mut sessions = Vec::new();
        for entry in fs::read_dir(sessions_dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            if let Ok(meta) = Self::read_meta(&entry.path()) {
                let mut info = SessionInfo::new(SessionId::new(meta.session_id), meta.cwd)
                    .additional_directories(meta.additional_directories);
                if let Some(title) = &meta.title {
                    info = info.title(title.clone());
                }
                if let Some(updated_at) = &meta.updated_at {
                    info = info.updated_at(updated_at.clone());
                }
                sessions.push(info);
            }
        }
        Ok(sessions)
    }

    /// Return the absolute path to the session's `history.jsonl` file.
    ///
    /// # Errors
    ///
    /// Returns [`SessionPersistenceError::InvalidSessionId`] if the session id
    /// contains path separators or other invalid characters.
    pub(crate) fn history_jsonl_path(
        &self,
        session_id: &str,
    ) -> Result<PathBuf, SessionPersistenceError> {
        Ok(self.session_dir(session_id)?.join(HISTORY_FILE))
    }

    fn session_dir(&self, session_id: &str) -> Result<PathBuf, SessionPersistenceError> {
        validate_session_id(session_id)?;
        Ok(self.state_dir.join(SESSIONS_DIR).join(session_id))
    }

    fn write_meta(
        session_dir: &Path,
        meta: &PersistedSessionMeta,
    ) -> Result<(), SessionPersistenceError> {
        let tmp_path = session_dir.join("meta.json.tmp");
        let mut file = File::create(&tmp_path)?;
        serde_json::to_writer_pretty(&mut file, meta)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        fs::rename(tmp_path, session_dir.join(META_FILE))?;
        Ok(())
    }

    fn append_history(
        session_dir: &Path,
        messages: &[ChatMessage],
    ) -> Result<(), SessionPersistenceError> {
        if messages.is_empty() {
            return Ok(());
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(session_dir.join(HISTORY_FILE))?;
        for message in messages {
            serde_json::to_writer(&mut file, message)?;
            file.write_all(b"\n")?;
        }
        file.flush()?;
        Ok(())
    }

    fn read_meta(session_dir: &Path) -> Result<PersistedSessionMeta, SessionPersistenceError> {
        let file = File::open(session_dir.join(META_FILE))?;
        Ok(serde_json::from_reader(file)?)
    }

    fn read_history(session_dir: &Path) -> Result<Vec<ChatMessage>, SessionPersistenceError> {
        let path = session_dir.join(HISTORY_FILE);
        if !path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut messages = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            messages.push(serde_json::from_str(&line)?);
        }
        Ok(messages)
    }
}

fn default_state_dir() -> Result<PathBuf, SessionPersistenceError> {
    if let Some(path) = std::env::var_os("XDG_STATE_HOME") {
        return Ok(PathBuf::from(path).join(APPLICATION_STATE_DIR));
    }

    let Some(home) = std::env::var_os("HOME") else {
        return Err(SessionPersistenceError::StateDir(
            "neither XDG_STATE_HOME nor HOME is set".to_string(),
        ));
    };

    Ok(PathBuf::from(home)
        .join(".local")
        .join("state")
        .join(APPLICATION_STATE_DIR))
}

fn validate_session_id(session_id: &str) -> Result<(), SessionPersistenceError> {
    let valid = !session_id.is_empty()
        && session_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'));

    if valid {
        Ok(())
    } else {
        Err(SessionPersistenceError::InvalidSessionId(
            session_id.to_string(),
        ))
    }
}

#[cfg(test)]
mod tests;
