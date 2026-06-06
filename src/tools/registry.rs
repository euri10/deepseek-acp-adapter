//! Tool registry, context types, and execution results.

#[allow(unused_imports)]
use std::path::PathBuf;

use agent_client_protocol::schema::{SessionId, ToolCallStatus, ToolKind};
use deepseek_acp_adapter::deepseek::{ToolCall as DeepSeekToolCall, ToolDefinition};
use futures_util::future::BoxFuture;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::execution::{
    edit_file_tool_definition, edit_file_tool_execution, glob_tool_definition, glob_tool_execution,
    grep_tool_definition, grep_tool_execution, list_dir_tool_definition, list_dir_tool_execution,
    read_file_tool_definition, read_file_tool_execution, run_command_tool_definition,
    run_command_tool_execution, write_file_tool_definition, write_file_tool_execution,
};
use crate::{
    PermissionRequester, ReadTextFileRequester, SessionStore, TerminalRequester, ToolCallRequester,
    WriteTextFileRequester, is_mcp_tool_name, mcp_tool_execution, mcp_tool_kind,
};

#[derive(Debug, Clone)]
pub(crate) struct ToolContext {
    pub(crate) session_id: SessionId,
    pub(crate) cwd: PathBuf,
    pub(crate) additional_directories: Vec<PathBuf>,
    pub(crate) client_capabilities: Option<agent_client_protocol::schema::ClientCapabilities>,
}

/// Registry for tools the model can call during a turn.
pub(crate) trait ToolRegistry: Send + Sync {
    /// Return tool definitions to advertise to the model.
    fn definitions(
        &self,
        context: &ToolContext,
        store: &crate::SessionStore,
    ) -> Result<Vec<ToolDefinition>, agent_client_protocol::Error>;

    /// Return the ACP kind used when displaying and gating a tool call.
    fn kind(&self, name: &str) -> ToolKind;

    /// Execute a complete model-requested tool call.
    ///
    /// The `cancellation_token` is cancelled when the turn is cancelled (via
    /// `session/cancel`); long-running tools (e.g. terminal commands) should race
    /// their work against it and abort promptly.
    fn execute<'a>(
        &'a self,
        call: &'a DeepSeekToolCall,
        context: &'a ToolContext,
        store: &'a crate::SessionStore,
        connection: Option<&'a dyn crate::ToolCallRequester>,
        cancellation_token: CancellationToken,
    ) -> BoxFuture<'a, ToolExecution>;
}

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct EmptyToolRegistry;

impl ToolRegistry for EmptyToolRegistry {
    fn definitions(
        &self,
        _context: &ToolContext,
        _store: &crate::SessionStore,
    ) -> Result<Vec<ToolDefinition>, agent_client_protocol::Error> {
        Ok(Vec::new())
    }

    fn kind(&self, _name: &str) -> ToolKind {
        ToolKind::Other
    }

    fn execute<'a>(
        &'a self,
        call: &'a DeepSeekToolCall,
        _context: &'a ToolContext,
        _store: &'a crate::SessionStore,
        _connection: Option<&'a dyn crate::ToolCallRequester>,
        _cancellation_token: CancellationToken,
    ) -> BoxFuture<'a, ToolExecution> {
        Box::pin(async move { ToolExecution::failed(format!("unknown tool: {}", call.name())) })
    }
}

#[derive(Debug)]
pub(crate) struct AdapterToolRegistry;

impl ToolRegistry for AdapterToolRegistry {
    fn definitions(
        &self,
        context: &ToolContext,
        store: &crate::SessionStore,
    ) -> Result<Vec<ToolDefinition>, agent_client_protocol::Error> {
        let mut definitions = vec![
            read_file_tool_definition(),
            list_dir_tool_definition(),
            glob_tool_definition(),
            grep_tool_definition(),
            write_file_tool_definition(),
            edit_file_tool_definition(),
            run_command_tool_definition(),
        ];
        definitions.extend(store.mcp_definitions(&context.session_id)?);
        Ok(definitions)
    }

    fn kind(&self, name: &str) -> ToolKind {
        match name {
            "read_file" | "list_dir" => ToolKind::Read,
            "glob" | "grep" => ToolKind::Search,
            "write_file" | "edit_file" => ToolKind::Edit,
            "run_command" => ToolKind::Execute,
            name if crate::is_mcp_tool_name(name) => crate::mcp_tool_kind(),
            _ => ToolKind::Other,
        }
    }

    fn execute<'a>(
        &'a self,
        call: &'a DeepSeekToolCall,
        context: &'a ToolContext,
        store: &'a crate::SessionStore,
        connection: Option<&'a dyn crate::ToolCallRequester>,
        cancellation_token: CancellationToken,
    ) -> BoxFuture<'a, ToolExecution> {
        Box::pin(async move {
            match call.name() {
                "read_file" => {
                    read_file_tool_execution(
                        call,
                        context,
                        connection.map(|requester| requester as &dyn crate::ReadTextFileRequester),
                    )
                    .await
                }
                "list_dir" => list_dir_tool_execution(call, context),
                "glob" => glob_tool_execution(call, context),
                "grep" => grep_tool_execution(call, context),
                "write_file" => {
                    write_file_tool_execution(
                        store,
                        call,
                        context,
                        connection.map(|requester| requester as &dyn crate::ReadTextFileRequester),
                        connection.map(|requester| requester as &dyn crate::WriteTextFileRequester),
                        connection.map(|requester| requester as &dyn crate::PermissionRequester),
                    )
                    .await
                }
                "edit_file" => {
                    edit_file_tool_execution(
                        store,
                        call,
                        context,
                        connection.map(|requester| requester as &dyn crate::ReadTextFileRequester),
                        connection.map(|requester| requester as &dyn crate::WriteTextFileRequester),
                        connection.map(|requester| requester as &dyn crate::PermissionRequester),
                    )
                    .await
                }
                "run_command" => {
                    run_command_tool_execution(
                        store,
                        call,
                        context,
                        connection.map(|requester| requester as &dyn crate::PermissionRequester),
                        connection.map(|requester| requester as &dyn crate::TerminalRequester),
                        &cancellation_token,
                    )
                    .await
                }
                name if crate::is_mcp_tool_name(name) => {
                    crate::mcp_tool_execution(store, call, context).await
                }
                _ => ToolExecution::failed(format!("unknown tool: {}", call.name())),
            }
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ToolExecution {
    pub(crate) content: String,
    pub(crate) raw_output: Value,
    pub(crate) success: bool,
    pub(crate) edit: Option<ToolEdit>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ToolEdit {
    pub(crate) path: PathBuf,
    pub(crate) old_text: Option<String>,
    pub(crate) new_text: String,
    pub(crate) line: u32,
}

impl ToolExecution {
    #[allow(dead_code)]
    pub(crate) fn completed(content: impl Into<String>, raw_output: Value) -> Self {
        Self {
            content: content.into(),
            raw_output,
            success: true,
            edit: None,
        }
    }

    pub(crate) fn failed(message: impl Into<String>) -> Self {
        let message = message.into();
        Self {
            content: message.clone(),
            raw_output: serde_json::json!({ "error": message }),
            success: false,
            edit: None,
        }
    }

    pub(crate) fn content_for_model(&self) -> &str {
        &self.content
    }

    pub(crate) fn status(&self) -> ToolCallStatus {
        if self.success {
            ToolCallStatus::Completed
        } else {
            ToolCallStatus::Failed
        }
    }
}
