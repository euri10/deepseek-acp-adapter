//! Shared test utilities for the `DeepSeek` `ACP` adapter.
//!
//! This module provides fakes and helpers that are used by unit tests across
//! multiple source files.  It is only compiled in `#[cfg(test)]` mode.

#![cfg(test)]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::{
    CreateTerminalRequest, CreateTerminalResponse, KillTerminalRequest, KillTerminalResponse,
    ReadTextFileRequest, ReadTextFileResponse, ReleaseTerminalRequest, ReleaseTerminalResponse,
    RequestPermissionRequest, RequestPermissionResponse, SessionConfigKind, SessionConfigOption,
    TerminalExitStatus, TerminalId, TerminalOutputRequest, TerminalOutputResponse,
    WaitForTerminalExitRequest, WaitForTerminalExitResponse, WriteTextFileRequest,
    WriteTextFileResponse,
};
use deepseek_acp_adapter::deepseek::ToolCall as DeepSeekToolCall;
use futures_util::future::BoxFuture;

use crate::acp::{
    CreateTerminalRequester, KillTerminalRequester, PermissionRequester, ReadTextFileRequester,
    ReleaseTerminalRequester, TerminalOutputRequester, WaitForTerminalExitRequester,
    WriteTextFileRequester,
};
use crate::session::SessionStore;
use crate::tools::ToolContext;

// ── CountingReadTextFileRequester ───────────────────────────

pub(crate) struct CountingReadTextFileRequester {
    calls: Arc<Mutex<usize>>,
}

impl CountingReadTextFileRequester {
    pub(crate) fn new() -> Self {
        Self {
            calls: Arc::new(Mutex::new(0)),
        }
    }

    pub(crate) fn calls(&self) -> Arc<Mutex<usize>> {
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

// ── Utf8FailingReadTextFileRequester ────────────────────────

pub(crate) struct Utf8FailingReadTextFileRequester;

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

// ── RecordingWriteTextFileRequester ─────────────────────────

pub(crate) struct RecordingWriteTextFileRequester {
    requests: Arc<Mutex<Vec<WriteTextFileRequest>>>,
}

impl RecordingWriteTextFileRequester {
    pub(crate) fn new() -> Self {
        Self {
            requests: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub(crate) fn requests(&self) -> Arc<Mutex<Vec<WriteTextFileRequest>>> {
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

// ── FakePermissionRequester ─────────────────────────────────

pub(crate) struct FakePermissionRequester {
    requests: Arc<Mutex<Vec<RequestPermissionRequest>>>,
    responses: Mutex<VecDeque<RequestPermissionResponse>>,
}

impl FakePermissionRequester {
    pub(crate) fn new(responses: Vec<RequestPermissionResponse>) -> Self {
        Self {
            requests: Arc::new(Mutex::new(Vec::new())),
            responses: Mutex::new(VecDeque::from(responses)),
        }
    }

    pub(crate) fn requests(&self) -> Arc<Mutex<Vec<RequestPermissionRequest>>> {
        Arc::clone(&self.requests)
    }
}

impl PermissionRequester for FakePermissionRequester {
    fn request_permission(
        &self,
        request: RequestPermissionRequest,
    ) -> BoxFuture<'_, Result<RequestPermissionResponse, agent_client_protocol::Error>> {
        self.requests
            .lock()
            .map(|mut requests| requests.push(request))
            .ok();

        let response = self
            .responses
            .lock()
            .map_err(|error| agent_client_protocol::Error::internal_error().data(error.to_string()))
            .and_then(|mut responses| {
                responses.pop_front().ok_or_else(|| {
                    agent_client_protocol::Error::internal_error()
                        .data("fake permission requester was exhausted")
                })
            });

        Box::pin(async move { response })
    }
}

// ── PermissionModeFixture & permission_mode_fixture ─────────

pub(crate) type PermissionModeFixture = (
    SessionStore,
    agent_client_protocol::schema::SessionId,
    ToolContext,
    DeepSeekToolCall,
    DeepSeekToolCall,
);

pub(crate) fn permission_mode_fixture()
-> Result<PermissionModeFixture, agent_client_protocol::Error> {
    use crate::test_store;
    let store = test_store();
    let session = crate::acp::handle_new_session_request(
        &store,
        &agent_client_protocol::schema::NewSessionRequest::new("/tmp"),
    )?;
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

    Ok((
        store.clone(),
        session.session_id,
        context,
        edit_call,
        shell_call,
    ))
}

// ── select_current_value ────────────────────────────────────

pub(crate) fn select_current_value(
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

// ── FakeTerminalRequester ───────────────────────────────────

pub(crate) struct FakeTerminalRequester {
    pub(crate) terminal_id: String,
    pub(crate) output: String,
    pub(crate) exit_code: Option<u32>,
    pub(crate) truncated: bool,
    pub(crate) create_error: Option<String>,
    pub(crate) wait_error: Option<String>,
    pub(crate) output_error: Option<String>,
    pub(crate) release_error: Option<String>,
}

impl CreateTerminalRequester for FakeTerminalRequester {
    fn create_terminal(
        &self,
        _request: CreateTerminalRequest,
    ) -> BoxFuture<'_, Result<CreateTerminalResponse, agent_client_protocol::Error>> {
        let terminal_id = self.terminal_id.clone();
        let error = self.create_error.clone();
        Box::pin(async move {
            if let Some(msg) = error {
                return Err(agent_client_protocol::Error::internal_error().data(msg));
            }
            Ok(CreateTerminalResponse::new(TerminalId::new(terminal_id)))
        })
    }
}

impl TerminalOutputRequester for FakeTerminalRequester {
    fn terminal_output(
        &self,
        _request: TerminalOutputRequest,
    ) -> BoxFuture<'_, Result<TerminalOutputResponse, agent_client_protocol::Error>> {
        let output = self.output.clone();
        let error = self.output_error.clone();
        let truncated = self.truncated;
        Box::pin(async move {
            if let Some(msg) = error {
                return Err(agent_client_protocol::Error::internal_error().data(msg));
            }
            Ok(TerminalOutputResponse::new(output, truncated))
        })
    }
}

impl WaitForTerminalExitRequester for FakeTerminalRequester {
    fn wait_for_terminal_exit(
        &self,
        _request: WaitForTerminalExitRequest,
    ) -> BoxFuture<'_, Result<WaitForTerminalExitResponse, agent_client_protocol::Error>> {
        let exit_code = self.exit_code;
        let error = self.wait_error.clone();
        Box::pin(async move {
            if let Some(msg) = error {
                return Err(agent_client_protocol::Error::internal_error().data(msg));
            }
            let status = TerminalExitStatus::new().exit_code(exit_code);
            Ok(WaitForTerminalExitResponse::new(status))
        })
    }
}

impl ReleaseTerminalRequester for FakeTerminalRequester {
    fn release_terminal(
        &self,
        _request: ReleaseTerminalRequest,
    ) -> BoxFuture<'_, Result<ReleaseTerminalResponse, agent_client_protocol::Error>> {
        let error = self.release_error.clone();
        Box::pin(async move {
            if let Some(msg) = error {
                return Err(agent_client_protocol::Error::internal_error().data(msg));
            }
            Ok(ReleaseTerminalResponse::new())
        })
    }
}

impl KillTerminalRequester for FakeTerminalRequester {
    fn kill_terminal(
        &self,
        _request: KillTerminalRequest,
    ) -> BoxFuture<'_, Result<KillTerminalResponse, agent_client_protocol::Error>> {
        Box::pin(async move { Ok(KillTerminalResponse::new()) })
    }
}

// ── CancelTracker ───────────────────────────────────────────

#[derive(Clone, Default)]
pub(crate) struct CancelTracker {
    pub(crate) kills: Arc<AtomicUsize>,
    pub(crate) releases: Arc<AtomicUsize>,
}

impl CreateTerminalRequester for CancelTracker {
    fn create_terminal(
        &self,
        _request: CreateTerminalRequest,
    ) -> BoxFuture<'_, Result<CreateTerminalResponse, agent_client_protocol::Error>> {
        Box::pin(async move { Ok(CreateTerminalResponse::new(TerminalId::new("term-cancel"))) })
    }
}

impl TerminalOutputRequester for CancelTracker {
    fn terminal_output(
        &self,
        _request: TerminalOutputRequest,
    ) -> BoxFuture<'_, Result<TerminalOutputResponse, agent_client_protocol::Error>> {
        Box::pin(async move { Ok(TerminalOutputResponse::new(String::new(), false)) })
    }
}

impl WaitForTerminalExitRequester for CancelTracker {
    fn wait_for_terminal_exit(
        &self,
        _request: WaitForTerminalExitRequest,
    ) -> BoxFuture<'_, Result<WaitForTerminalExitResponse, agent_client_protocol::Error>> {
        Box::pin(std::future::pending())
    }
}

impl ReleaseTerminalRequester for CancelTracker {
    fn release_terminal(
        &self,
        _request: ReleaseTerminalRequest,
    ) -> BoxFuture<'_, Result<ReleaseTerminalResponse, agent_client_protocol::Error>> {
        let releases = Arc::clone(&self.releases);
        Box::pin(async move {
            releases.fetch_add(1, Ordering::SeqCst);
            Ok(ReleaseTerminalResponse::new())
        })
    }
}

impl KillTerminalRequester for CancelTracker {
    fn kill_terminal(
        &self,
        _request: KillTerminalRequest,
    ) -> BoxFuture<'_, Result<KillTerminalResponse, agent_client_protocol::Error>> {
        let kills = Arc::clone(&self.kills);
        Box::pin(async move {
            kills.fetch_add(1, Ordering::SeqCst);
            Ok(KillTerminalResponse::new())
        })
    }
}

// ── FailingWriteRequester ───────────────────────────────────

pub(crate) struct FailingWriteRequester;

impl WriteTextFileRequester for FailingWriteRequester {
    fn write_text_file(
        &self,
        _request: WriteTextFileRequest,
    ) -> BoxFuture<'_, Result<WriteTextFileResponse, agent_client_protocol::Error>> {
        Box::pin(
            async move { Err(agent_client_protocol::Error::internal_error().data("disk full")) },
        )
    }
}
