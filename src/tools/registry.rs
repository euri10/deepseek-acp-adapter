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
    // Only used in test code currently; suppress dead_code until a non-test
    // caller materializes.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acp::handle_new_session_request;
    use crate::test_store;
    use agent_client_protocol::schema::NewSessionRequest;
    use deepseek_acp_adapter::deepseek::ToolCall as DeepSeekToolCall;
    use tokio_util::sync::CancellationToken;

    fn registry_context(cwd: std::path::PathBuf) -> ToolContext {
        ToolContext {
            session_id: agent_client_protocol::schema::SessionId::new("session-registry-test"),
            cwd,
            additional_directories: Vec::new(),
            client_capabilities: None,
        }
    }

    #[test]
    fn empty_registry_definitions_returns_empty() -> Result<(), agent_client_protocol::Error> {
        let registry = EmptyToolRegistry;
        let context = registry_context(std::path::PathBuf::from("/tmp"));
        let store = test_store();
        let definitions = registry.definitions(&context, &store)?;
        assert!(definitions.is_empty());
        Ok(())
    }

    #[test]
    fn empty_registry_kind_returns_other() {
        let registry = EmptyToolRegistry;
        assert_eq!(registry.kind("anything"), ToolKind::Other);
    }

    #[test_log::test(tokio::test)]
    async fn empty_registry_execute_returns_failed() {
        let registry = EmptyToolRegistry;
        let context = registry_context(std::path::PathBuf::from("/tmp"));
        let store = test_store();
        let call = DeepSeekToolCall::new("empty-call", "test_tool", "{}");
        let result = registry
            .execute(&call, &context, &store, None, CancellationToken::new())
            .await;
        assert!(!result.success);
        assert!(result.content.contains("unknown tool: test_tool"));
    }

    #[test_log::test(tokio::test)]
    async fn adapter_registry_execute_read_file_local() -> Result<(), agent_client_protocol::Error>
    {
        let temp_root =
            std::env::temp_dir().join(format!("deepseek-acp-reg-read-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        std::fs::write(temp_root.join("sample.txt"), "alpha\nbeta\ngamma\n")
            .map_err(agent_client_protocol::Error::into_internal_error)?;

        let registry = AdapterToolRegistry;
        let context = registry_context(temp_root.clone());
        let store = test_store();
        let call = DeepSeekToolCall::new(
            "reg-read",
            "read_file",
            serde_json::json!({"path": "sample.txt"}).to_string(),
        );
        let result = registry
            .execute(&call, &context, &store, None, CancellationToken::new())
            .await;
        assert!(result.success);
        assert_eq!(result.content, "alpha\nbeta\ngamma");
        assert_eq!(result.raw_output["source"], "local");
        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn adapter_registry_execute_write_file_local_no_permission()
    -> Result<(), agent_client_protocol::Error> {
        let temp_root =
            std::env::temp_dir().join(format!("deepseek-acp-reg-write-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;

        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new(&temp_root))?;

        let registry = AdapterToolRegistry;
        let context = ToolContext {
            session_id: session.session_id.clone(),
            cwd: temp_root.clone(),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let call = DeepSeekToolCall::new(
            "reg-write",
            "write_file",
            serde_json::json!({"path": "out.txt", "content": "hello world"}).to_string(),
        );
        let result = registry
            .execute(&call, &context, &store, None, CancellationToken::new())
            .await;
        // write_file requires permission which is denied without a requester
        assert!(!result.success);
        assert!(result.content.contains("requires a client connection"));
        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn adapter_registry_execute_edit_file_local_no_permission()
    -> Result<(), agent_client_protocol::Error> {
        let temp_root =
            std::env::temp_dir().join(format!("deepseek-acp-reg-edit-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        std::fs::write(temp_root.join("source.txt"), "original content\n")
            .map_err(agent_client_protocol::Error::into_internal_error)?;

        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new(&temp_root))?;

        let registry = AdapterToolRegistry;
        let context = ToolContext {
            session_id: session.session_id.clone(),
            cwd: temp_root.clone(),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let call = DeepSeekToolCall::new(
            "reg-edit",
            "edit_file",
            serde_json::json!({
                "path": "source.txt",
                "old_text": "original",
                "new_text": "modified"
            })
            .to_string(),
        );
        let result = registry
            .execute(&call, &context, &store, None, CancellationToken::new())
            .await;
        // edit_file requires permission which is denied without a requester
        assert!(!result.success);
        assert!(result.content.contains("requires a client connection"));
        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn adapter_registry_execute_run_command_local_no_permission()
    -> Result<(), agent_client_protocol::Error> {
        let temp_root =
            std::env::temp_dir().join(format!("deepseek-acp-reg-cmd-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root)
            .map_err(agent_client_protocol::Error::into_internal_error)?;

        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new(&temp_root))?;

        let registry = AdapterToolRegistry;
        let context = ToolContext {
            session_id: session.session_id.clone(),
            cwd: temp_root.clone(),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let call = DeepSeekToolCall::new(
            "reg-cmd",
            "run_command",
            serde_json::json!({"command": "echo hello"}).to_string(),
        );
        let result = registry
            .execute(&call, &context, &store, None, CancellationToken::new())
            .await;
        // run_command requires permission which is denied without a requester
        assert!(!result.success);
        assert!(result.content.contains("requires a client connection"));
        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn adapter_registry_execute_bogus_tool() {
        let registry = AdapterToolRegistry;
        let context = registry_context(std::path::PathBuf::from("/tmp"));
        let store = test_store();
        let call = DeepSeekToolCall::new("bogus-call", "no_such_tool", "{}");
        let result = registry
            .execute(&call, &context, &store, None, CancellationToken::new())
            .await;
        assert!(!result.success);
        assert!(result.content.contains("unknown tool: no_such_tool"));
    }

    #[test]
    fn tool_execution_completed_constructs_correctly() {
        let exec = ToolExecution::completed("done", serde_json::json!({"ok": true}));
        assert!(exec.success);
        assert_eq!(exec.content, "done");
        assert_eq!(exec.raw_output, serde_json::json!({"ok": true}));
        assert!(exec.edit.is_none());
        assert_eq!(exec.status(), ToolCallStatus::Completed);
        assert_eq!(exec.content_for_model(), "done");
    }

    #[test]
    fn tool_execution_failed_constructs_correctly() {
        let exec = ToolExecution::failed("error message");
        assert!(!exec.success);
        assert_eq!(exec.content, "error message");
        assert_eq!(
            exec.raw_output,
            serde_json::json!({"error": "error message"})
        );
        assert!(exec.edit.is_none());
        assert_eq!(exec.status(), ToolCallStatus::Failed);
        assert_eq!(exec.content_for_model(), "error message");
    }

    #[test]
    fn tool_execution_status_returns_completed_when_success() {
        let exec = ToolExecution {
            content: String::new(),
            raw_output: serde_json::Value::Null,
            success: true,
            edit: None,
        };
        assert_eq!(exec.status(), ToolCallStatus::Completed);
    }

    #[test]
    fn tool_execution_status_returns_failed_when_not_success() {
        let exec = ToolExecution {
            content: String::new(),
            raw_output: serde_json::Value::Null,
            success: false,
            edit: None,
        };
        assert_eq!(exec.status(), ToolCallStatus::Failed);
    }

    #[test]
    fn tool_execution_content_for_model_returns_content_ref() {
        let exec = ToolExecution {
            content: "the response".to_string(),
            raw_output: serde_json::Value::Null,
            success: true,
            edit: None,
        };
        assert_eq!(exec.content_for_model(), "the response");
    }
}
