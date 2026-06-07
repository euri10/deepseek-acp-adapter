//! ACP requester traits and their production implementations.
//!
//! These traits abstract over ACP client connections so that tool execution and
//! permission logic can be unit-tested without a real transport.

use agent_client_protocol::schema::{
    CreateTerminalRequest, CreateTerminalResponse, KillTerminalRequest, KillTerminalResponse,
    ReadTextFileRequest, ReadTextFileResponse, ReleaseTerminalRequest, ReleaseTerminalResponse,
    RequestPermissionRequest, RequestPermissionResponse, TerminalOutputRequest,
    TerminalOutputResponse, WaitForTerminalExitRequest, WaitForTerminalExitResponse,
    WriteTextFileRequest, WriteTextFileResponse,
};
use agent_client_protocol::{Agent, Client};
use futures_util::future::BoxFuture;

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

pub(crate) fn recover_null_write_response(
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
