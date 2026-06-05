//! Built-in tool execution for the `DeepSeek` `ACP` adapter.

use std::fmt::Write as _;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use agent_client_protocol::schema::{
    CreateTerminalRequest, KillTerminalRequest, ReadTextFileRequest, ReleaseTerminalRequest,
    SessionId, TerminalOutputRequest, ToolCallStatus, ToolKind, WaitForTerminalExitRequest,
    WriteTextFileRequest,
};
use deepseek_acp_adapter::deepseek::{ToolCall as DeepSeekToolCall, ToolDefinition};
use futures_util::future::BoxFuture;
use globset::{Glob, GlobSetBuilder};
use grep::regex::RegexMatcher;
use grep::searcher::sinks::UTF8;
use grep::searcher::{BinaryDetection, SearcherBuilder};
use ignore::WalkBuilder;
use ignore::gitignore::GitignoreBuilder;
use serde::Deserialize;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::request_tool_permission;

const TOOL_OUTPUT_LIMIT: usize = 200;
const TOOL_OUTPUT_LIMIT_U32: u32 = 200;
const COMMAND_OUTPUT_LIMIT: usize = 20_000;

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
        store: &super::SessionStore,
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
        store: &'a super::SessionStore,
        connection: Option<&'a dyn super::ToolCallRequester>,
        cancellation_token: CancellationToken,
    ) -> BoxFuture<'a, ToolExecution>;
}

#[cfg(test)]
#[derive(Debug)]
pub(crate) struct EmptyToolRegistry;

#[cfg(test)]
impl ToolRegistry for EmptyToolRegistry {
    fn definitions(
        &self,
        _context: &ToolContext,
        _store: &super::SessionStore,
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
        _store: &'a super::SessionStore,
        _connection: Option<&'a dyn super::ToolCallRequester>,
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
        store: &super::SessionStore,
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
            name if super::is_mcp_tool_name(name) => super::mcp_tool_kind(),
            _ => ToolKind::Other,
        }
    }

    fn execute<'a>(
        &'a self,
        call: &'a DeepSeekToolCall,
        context: &'a ToolContext,
        store: &'a super::SessionStore,
        connection: Option<&'a dyn super::ToolCallRequester>,
        cancellation_token: CancellationToken,
    ) -> BoxFuture<'a, ToolExecution> {
        Box::pin(async move {
            match call.name() {
                "read_file" => {
                    read_file_tool_execution(
                        call,
                        context,
                        connection.map(|requester| requester as &dyn super::ReadTextFileRequester),
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
                        connection.map(|requester| requester as &dyn super::ReadTextFileRequester),
                        connection.map(|requester| requester as &dyn super::WriteTextFileRequester),
                        connection.map(|requester| requester as &dyn super::PermissionRequester),
                    )
                    .await
                }
                "edit_file" => {
                    edit_file_tool_execution(
                        store,
                        call,
                        context,
                        connection.map(|requester| requester as &dyn super::ReadTextFileRequester),
                        connection.map(|requester| requester as &dyn super::WriteTextFileRequester),
                        connection.map(|requester| requester as &dyn super::PermissionRequester),
                    )
                    .await
                }
                "run_command" => {
                    run_command_tool_execution(
                        store,
                        call,
                        context,
                        connection.map(|requester| requester as &dyn super::PermissionRequester),
                        connection.map(|requester| requester as &dyn super::TerminalRequester),
                        &cancellation_token,
                    )
                    .await
                }
                name if super::is_mcp_tool_name(name) => {
                    super::mcp_tool_execution(store, call, context).await
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
    #[cfg(test)]
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

#[derive(Debug, Deserialize)]
struct ReadFileArguments {
    path: PathBuf,
    line: Option<u32>,
    limit: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ListDirArguments {
    path: PathBuf,
}

#[derive(Debug, Deserialize)]
struct GlobArguments {
    pattern: String,
}

#[derive(Debug, Deserialize)]
struct GrepArguments {
    pattern: String,
}

#[derive(Debug, Deserialize)]
struct WriteFileArguments {
    path: PathBuf,
    content: String,
}

#[derive(Debug, Deserialize)]
struct EditFileArguments {
    path: PathBuf,
    old_text: String,
    new_text: String,
}

#[derive(Debug, Deserialize)]
struct RunCommandArguments {
    command: String,
}

pub(crate) fn read_file_tool_definition() -> ToolDefinition {
    ToolDefinition::new(
        "read_file",
        "Read a text file, using the client's file system when available.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "line": { "type": "integer", "minimum": 1 },
                "limit": { "type": "integer", "minimum": 1 },
            },
            "required": ["path"],
            "additionalProperties": false,
        }),
    )
}

pub(crate) fn list_dir_tool_definition() -> ToolDefinition {
    ToolDefinition::new(
        "list_dir",
        "List entries in a directory.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
            },
            "required": ["path"],
            "additionalProperties": false,
        }),
    )
}

pub(crate) fn glob_tool_definition() -> ToolDefinition {
    ToolDefinition::new(
        "glob",
        "Find paths matching a glob pattern.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string" },
            },
            "required": ["pattern"],
            "additionalProperties": false,
        }),
    )
}

pub(crate) fn grep_tool_definition() -> ToolDefinition {
    ToolDefinition::new(
        "grep",
        "Search files for a regular expression.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string" },
            },
            "required": ["pattern"],
            "additionalProperties": false,
        }),
    )
}

pub(crate) fn write_file_tool_definition() -> ToolDefinition {
    ToolDefinition::new(
        "write_file",
        "Write UTF-8 text to a file, creating or replacing the file.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "content": { "type": "string" },
            },
            "required": ["path", "content"],
            "additionalProperties": false,
        }),
    )
}

pub(crate) fn edit_file_tool_definition() -> ToolDefinition {
    ToolDefinition::new(
        "edit_file",
        "Replace one exact UTF-8 text span in an existing file.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "old_text": { "type": "string" },
                "new_text": { "type": "string" },
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": false,
        }),
    )
}

pub(crate) fn run_command_tool_definition() -> ToolDefinition {
    ToolDefinition::new(
        "run_command",
        "Run a shell command in the session working directory.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": { "type": "string" },
            },
            "required": ["command"],
            "additionalProperties": false,
        }),
    )
}

pub(crate) async fn read_file_tool_execution(
    call: &DeepSeekToolCall,
    context: &ToolContext,
    connection: Option<&dyn super::ReadTextFileRequester>,
) -> ToolExecution {
    let parsed_arguments = match serde_json::from_str::<ReadFileArguments>(call.arguments()) {
        Ok(arguments) => arguments,
        Err(error) => {
            return ToolExecution::failed(format!("invalid read_file arguments: {error}"));
        }
    };

    let resolved_path = resolve_tool_path(context, &parsed_arguments.path);
    let start_line = parsed_arguments.line.unwrap_or(1);
    let requested_limit = parsed_arguments.limit.unwrap_or(TOOL_OUTPUT_LIMIT_U32);
    let limit = requested_limit.min(TOOL_OUTPUT_LIMIT_U32);

    if start_line == 0 {
        return ToolExecution::failed("read_file line must be at least 1");
    }

    if requested_limit == 0 {
        return ToolExecution::failed("read_file limit must be at least 1");
    }

    let file_result = if context
        .client_capabilities
        .as_ref()
        .is_some_and(|capabilities| capabilities.fs.read_text_file)
    {
        match connection {
            Some(connection) => {
                if local_file_is_non_utf8(&resolved_path) {
                    return ToolExecution::failed(non_utf8_file_message(&resolved_path));
                }

                read_file_from_client(
                    connection,
                    &context.session_id,
                    &resolved_path,
                    start_line,
                    limit,
                )
                .await
            }
            None => Err("read_file needs a client connection for fs/read_text_file".to_owned()),
        }
    } else {
        read_file_from_local(&resolved_path, start_line, limit)
    };

    match file_result {
        Ok(file_slice) => ToolExecution {
            content: file_slice,
            raw_output: serde_json::json!({
                "path": resolved_path,
                "line": start_line,
                "limit": limit,
                "source": if context
                    .client_capabilities
                    .as_ref()
                    .is_some_and(|capabilities| capabilities.fs.read_text_file)
                {
                    "client"
                } else {
                    "local"
                },
            }),
            success: true,
            edit: None,
        },
        Err(error) => ToolExecution::failed(error),
    }
}

pub(crate) async fn write_file_tool_execution(
    store: &super::SessionStore,
    call: &DeepSeekToolCall,
    context: &ToolContext,
    read_connection: Option<&dyn super::ReadTextFileRequester>,
    write_connection: Option<&dyn super::WriteTextFileRequester>,
    permission_requester: Option<&dyn super::PermissionRequester>,
) -> ToolExecution {
    let parsed_arguments = match serde_json::from_str::<WriteFileArguments>(call.arguments()) {
        Ok(arguments) => arguments,
        Err(error) => {
            return ToolExecution::failed(format!("invalid write_file arguments: {error}"));
        }
    };

    if let Err(error) =
        require_tool_permission(store, context, call, ToolKind::Edit, permission_requester).await
    {
        return ToolExecution::failed(error);
    }

    let resolved_path = resolve_tool_path(context, &parsed_arguments.path);
    let use_client_write = context
        .client_capabilities
        .as_ref()
        .is_some_and(|capabilities| capabilities.fs.write_text_file);
    let old_text = match read_existing_text(
        context,
        &resolved_path,
        read_connection,
        use_client_write,
    )
    .await
    {
        Ok(text) => text,
        Err(error) => return ToolExecution::failed(error),
    };
    let write_result = if use_client_write {
        match write_connection {
            Some(connection) => {
                write_file_to_client(
                    connection,
                    &context.session_id,
                    &resolved_path,
                    &parsed_arguments.content,
                )
                .await
            }
            None => Err("write_file needs a client connection for fs/write_text_file".to_owned()),
        }
    } else {
        write_file_to_local(&resolved_path, &parsed_arguments.content)
    };

    match write_result {
        Ok(()) => {
            let byte_count = parsed_arguments.content.len();
            ToolExecution {
                content: format!("wrote {byte_count} bytes to {}", resolved_path.display()),
                raw_output: serde_json::json!({
                    "path": resolved_path,
                    "bytes": byte_count,
                    "source": if use_client_write { "client" } else { "local" },
                }),
                success: true,
                edit: Some(ToolEdit {
                    path: resolved_path,
                    old_text,
                    new_text: parsed_arguments.content,
                    line: 1,
                }),
            }
        }
        Err(error) => ToolExecution::failed(error),
    }
}

pub(crate) async fn edit_file_tool_execution(
    store: &super::SessionStore,
    call: &DeepSeekToolCall,
    context: &ToolContext,
    read_connection: Option<&dyn super::ReadTextFileRequester>,
    write_connection: Option<&dyn super::WriteTextFileRequester>,
    permission_requester: Option<&dyn super::PermissionRequester>,
) -> ToolExecution {
    let parsed_arguments = match serde_json::from_str::<EditFileArguments>(call.arguments()) {
        Ok(arguments) => arguments,
        Err(error) => {
            return ToolExecution::failed(format!("invalid edit_file arguments: {error}"));
        }
    };

    if parsed_arguments.old_text.is_empty() {
        return ToolExecution::failed("edit_file old_text must not be empty");
    }

    let resolved_path = resolve_tool_path(context, &parsed_arguments.path);
    let use_client_read = context
        .client_capabilities
        .as_ref()
        .is_some_and(|capabilities| capabilities.fs.read_text_file);
    let original_result = if use_client_read {
        match read_connection {
            Some(connection) => {
                read_full_file_from_client(connection, &context.session_id, &resolved_path).await
            }
            None => Err("edit_file needs a client connection for fs/read_text_file".to_owned()),
        }
    } else {
        fs::read_to_string(&resolved_path).map_err(|error| {
            format!(
                "failed to read {} before editing: {error}",
                resolved_path.display()
            )
        })
    };

    let original = match original_result {
        Ok(file_text) => file_text,
        Err(error) => return ToolExecution::failed(error),
    };

    let matches = original.matches(&parsed_arguments.old_text).count();
    if matches == 0 {
        return ToolExecution::failed(format!(
            "edit_file could not find old_text in {}",
            resolved_path.display()
        ));
    }
    if matches > 1 {
        return ToolExecution::failed(format!(
            "edit_file found old_text {matches} times in {}; provide a unique span",
            resolved_path.display()
        ));
    }

    if let Err(error) =
        require_tool_permission(store, context, call, ToolKind::Edit, permission_requester).await
    {
        return ToolExecution::failed(error);
    }

    let edit_line = match original.find(&parsed_arguments.old_text) {
        Some(offset) => line_number_for_offset(&original, offset),
        None => 1,
    };
    let updated = original.replacen(&parsed_arguments.old_text, &parsed_arguments.new_text, 1);
    let use_client_write = context
        .client_capabilities
        .as_ref()
        .is_some_and(|capabilities| capabilities.fs.write_text_file);
    let write_result = if use_client_write {
        match write_connection {
            Some(connection) => {
                write_file_to_client(connection, &context.session_id, &resolved_path, &updated)
                    .await
            }
            None => Err("edit_file needs a client connection for fs/write_text_file".to_owned()),
        }
    } else {
        write_file_to_local(&resolved_path, &updated)
    };

    match write_result {
        Ok(()) => ToolExecution {
            content: format!("edited {}", resolved_path.display()),
            raw_output: serde_json::json!({
                "path": resolved_path,
                "replacements": 1,
                "read_source": if use_client_read { "client" } else { "local" },
                "write_source": if use_client_write { "client" } else { "local" },
            }),
            success: true,
            edit: Some(ToolEdit {
                path: resolved_path,
                old_text: Some(original),
                new_text: updated,
                line: edit_line,
            }),
        },
        Err(error) => ToolExecution::failed(error),
    }
}

pub(crate) async fn run_command_tool_execution(
    store: &super::SessionStore,
    call: &DeepSeekToolCall,
    context: &ToolContext,
    permission_requester: Option<&dyn super::PermissionRequester>,
    terminal_connection: Option<&dyn super::TerminalRequester>,
    cancellation_token: &CancellationToken,
) -> ToolExecution {
    let parsed_arguments = match serde_json::from_str::<RunCommandArguments>(call.arguments()) {
        Ok(arguments) => arguments,
        Err(error) => {
            return ToolExecution::failed(format!("invalid run_command arguments: {error}"));
        }
    };

    if parsed_arguments.command.trim().is_empty() {
        return ToolExecution::failed("run_command command must not be empty");
    }

    if let Err(error) = require_tool_permission(
        store,
        context,
        call,
        ToolKind::Execute,
        permission_requester,
    )
    .await
    {
        return ToolExecution::failed(error);
    }

    if context
        .client_capabilities
        .as_ref()
        .is_some_and(|capabilities| capabilities.terminal)
    {
        return run_command_via_terminal(
            &context.session_id,
            &context.cwd,
            &parsed_arguments.command,
            terminal_connection,
            cancellation_token,
        )
        .await;
    }

    let cwd = context.cwd.clone();
    let command = parsed_arguments.command;
    let output = match tokio::task::spawn_blocking(move || {
        std::process::Command::new("sh")
            .arg("-lc")
            .arg(&command)
            .current_dir(cwd)
            .output()
    })
    .await
    {
        Ok(Ok(output)) => output,
        Ok(Err(error)) => return ToolExecution::failed(format!("failed to run command: {error}")),
        Err(error) => return ToolExecution::failed(format!("run_command task failed: {error}")),
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined_output = render_command_output(&stdout, &stderr, output.status.code());
    let (display_output, truncated) = truncate_tool_output(&combined_output, COMMAND_OUTPUT_LIMIT);

    ToolExecution {
        content: display_output,
        raw_output: serde_json::json!({
            "exit_code": output.status.code(),
            "success": output.status.success(),
            "stdout": stdout,
            "stderr": stderr,
            "truncated": truncated,
        }),
        success: output.status.success(),
        edit: None,
    }
}

pub(crate) async fn run_command_via_terminal(
    session_id: &SessionId,
    cwd: &Path,
    command: &str,
    connection: Option<&dyn super::TerminalRequester>,
    cancellation_token: &CancellationToken,
) -> ToolExecution {
    let Some(terminal_requester) = connection else {
        return ToolExecution::failed("terminal support advertised but no connection available");
    };

    let create_request = CreateTerminalRequest::new(session_id.clone(), command)
        .cwd(Some(cwd.to_path_buf()))
        .output_byte_limit(Some(COMMAND_OUTPUT_LIMIT as u64));
    let create_response = match terminal_requester.create_terminal(create_request).await {
        Ok(response) => response,
        Err(error) => {
            return ToolExecution::failed(format!("terminal/create failed: {error}"));
        }
    };
    let terminal_id = create_response.terminal_id;

    let wait_request = WaitForTerminalExitRequest::new(session_id.clone(), terminal_id.clone());
    let wait_response = tokio::select! {
        // Turn cancelled while the command is running: kill it, then release the
        // terminal so the client frees its resources.
        () = cancellation_token.cancelled() => {
            let _ = terminal_requester
                .kill_terminal(KillTerminalRequest::new(
                    session_id.clone(),
                    terminal_id.clone(),
                ))
                .await;
            let _ = terminal_requester
                .release_terminal(ReleaseTerminalRequest::new(
                    session_id.clone(),
                    terminal_id.clone(),
                ))
                .await;
            return ToolExecution::failed("run_command cancelled");
        }
        result = terminal_requester.wait_for_terminal_exit(wait_request) => match result {
            Ok(response) => response,
            Err(error) => {
                let _ = terminal_requester
                    .release_terminal(ReleaseTerminalRequest::new(
                        session_id.clone(),
                        terminal_id.clone(),
                    ))
                    .await;
                return ToolExecution::failed(format!("terminal/wait_for_exit failed: {error}"));
            }
        },
    };

    let output_request = TerminalOutputRequest::new(session_id.clone(), terminal_id.clone());
    let output_response = match terminal_requester.terminal_output(output_request).await {
        Ok(response) => response,
        Err(error) => {
            let _ = terminal_requester
                .release_terminal(ReleaseTerminalRequest::new(
                    session_id.clone(),
                    terminal_id.clone(),
                ))
                .await;
            return ToolExecution::failed(format!("terminal/output failed: {error}"));
        }
    };

    if let Err(error) = terminal_requester
        .release_terminal(ReleaseTerminalRequest::new(
            session_id.clone(),
            terminal_id.clone(),
        ))
        .await
    {
        return ToolExecution::failed(format!("terminal/release failed: {error}"));
    }

    let exit_code = wait_response.exit_status.exit_code;
    let success = exit_code == Some(0);
    let exit_code_i32 = exit_code.and_then(|code| i32::try_from(code).ok());
    let combined_output = render_command_output(&output_response.output, "", exit_code_i32);
    let (display_output, truncated) = truncate_tool_output(&combined_output, COMMAND_OUTPUT_LIMIT);

    ToolExecution {
        content: display_output,
        raw_output: serde_json::json!({
            "exit_code": exit_code,
            "success": success,
            "stdout": output_response.output,
            "stderr": "",
            "truncated": truncated || output_response.truncated,
        }),
        success,
        edit: None,
    }
}

pub(crate) fn list_dir_tool_execution(
    call: &DeepSeekToolCall,
    context: &ToolContext,
) -> ToolExecution {
    let parsed_arguments = match serde_json::from_str::<ListDirArguments>(call.arguments()) {
        Ok(arguments) => arguments,
        Err(error) => {
            return ToolExecution::failed(format!("invalid list_dir arguments: {error}"));
        }
    };

    let resolved_path = resolve_tool_path(context, &parsed_arguments.path);
    let entries = match collect_directory_entries(&resolved_path) {
        Ok(entries) => entries,
        Err(error) => return ToolExecution::failed(error),
    };

    let truncated = entries.len() > TOOL_OUTPUT_LIMIT;
    let entries = entries
        .into_iter()
        .take(TOOL_OUTPUT_LIMIT)
        .collect::<Vec<_>>();
    let output_text = render_tool_lines(&entries, truncated, "entries", TOOL_OUTPUT_LIMIT);

    ToolExecution {
        content: output_text,
        raw_output: serde_json::json!({
            "path": resolved_path,
            "entries": entries,
            "truncated": truncated,
        }),
        success: true,
        edit: None,
    }
}

pub(crate) fn glob_tool_execution(call: &DeepSeekToolCall, context: &ToolContext) -> ToolExecution {
    let parsed_arguments = match serde_json::from_str::<GlobArguments>(call.arguments()) {
        Ok(arguments) => arguments,
        Err(error) => return ToolExecution::failed(format!("invalid glob arguments: {error}")),
    };

    let matcher = match Glob::new(&parsed_arguments.pattern) {
        Ok(glob) => {
            let mut builder = GlobSetBuilder::new();
            builder.add(glob);
            match builder.build() {
                Ok(set) => set,
                Err(error) => {
                    return ToolExecution::failed(format!("invalid glob pattern: {error}"));
                }
            }
        }
        Err(error) => return ToolExecution::failed(format!("invalid glob pattern: {error}")),
    };

    let root_gitignore = build_root_gitignore(&context.cwd);
    let mut glob_paths = Vec::new();
    let walker = WalkBuilder::new(&context.cwd)
        .hidden(false)
        .parents(true)
        .ignore(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .build();
    for entry in walker {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => return ToolExecution::failed(error.to_string()),
        };

        if !entry
            .file_type()
            .is_some_and(|file_type| file_type.is_file())
        {
            continue;
        }

        if is_hidden_path(entry.path()) {
            continue;
        }

        let path = entry.path();
        let relative_path = path.strip_prefix(&context.cwd).unwrap_or(path);
        if root_gitignore.as_ref().is_some_and(|matcher| {
            matcher
                .matched_path_or_any_parents(relative_path, false)
                .is_ignore()
        }) {
            continue;
        }
        if matcher.is_match(relative_path) || matcher.is_match(path) {
            glob_paths.push(relative_path.display().to_string());
        }
    }

    glob_paths.sort_unstable();
    let truncated = glob_paths.len() > TOOL_OUTPUT_LIMIT;
    let entries = glob_paths
        .into_iter()
        .take(TOOL_OUTPUT_LIMIT)
        .collect::<Vec<_>>();
    let output_text = render_tool_lines(&entries, truncated, "matches", TOOL_OUTPUT_LIMIT);

    ToolExecution {
        content: output_text,
        raw_output: serde_json::json!({
            "pattern": parsed_arguments.pattern,
            "matches": entries,
            "truncated": truncated,
        }),
        success: true,
        edit: None,
    }
}

pub(crate) fn grep_tool_execution(call: &DeepSeekToolCall, context: &ToolContext) -> ToolExecution {
    let parsed_arguments = match serde_json::from_str::<GrepArguments>(call.arguments()) {
        Ok(arguments) => arguments,
        Err(error) => return ToolExecution::failed(format!("invalid grep arguments: {error}")),
    };

    let matcher = match RegexMatcher::new_line_matcher(&parsed_arguments.pattern) {
        Ok(matcher) => matcher,
        Err(error) => return ToolExecution::failed(format!("invalid grep regex: {error}")),
    };

    let root_gitignore = build_root_gitignore(&context.cwd);
    let (mut grep_hits, truncated) =
        match collect_grep_matches(&context.cwd, root_gitignore.as_ref(), &matcher) {
            Ok(result) => result,
            Err(error) => return ToolExecution::failed(error),
        };

    grep_hits.sort_unstable_by(|left, right| {
        left.path
            .cmp(&right.path)
            .then(left.line.cmp(&right.line))
            .then(left.text.cmp(&right.text))
    });
    let lines = grep_hits
        .into_iter()
        .take(TOOL_OUTPUT_LIMIT)
        .map(|entry| format!("{}:{}:{}", entry.path, entry.line, entry.text))
        .collect::<Vec<_>>();
    let output_text = render_tool_lines(&lines, truncated, "matches", TOOL_OUTPUT_LIMIT);

    ToolExecution {
        content: output_text,
        raw_output: serde_json::json!({
            "pattern": parsed_arguments.pattern,
            "matches": lines,
            "truncated": truncated,
        }),
        success: true,
        edit: None,
    }
}

pub(crate) async fn require_tool_permission(
    store: &super::SessionStore,
    context: &ToolContext,
    call: &DeepSeekToolCall,
    kind: ToolKind,
    requester: Option<&dyn super::PermissionRequester>,
) -> Result<(), String> {
    let requester = requester.ok_or_else(|| {
        format!(
            "{} requires a client connection that can request permissions",
            call.name()
        )
    })?;

    match request_tool_permission(store, context, call, kind, requester).await {
        Ok(
            super::PermissionDecision::AllowOnce
            | super::PermissionDecision::AllowAlways
            | super::PermissionDecision::AllowByMode,
        ) => Ok(()),
        Ok(super::PermissionDecision::RejectOnce | super::PermissionDecision::RejectAlways) => {
            Err(format!("{} was rejected by permission policy", call.name()))
        }
        Ok(super::PermissionDecision::Cancelled) => {
            Err(format!("{} permission request was cancelled", call.name()))
        }
        Err(error) => Err(format!(
            "failed to request permission for {}: {error}",
            call.name()
        )),
    }
}

async fn read_file_from_client<'a>(
    connection: &'a dyn super::ReadTextFileRequester,
    session_id: &'a SessionId,
    path: &'a Path,
    line: u32,
    limit: u32,
) -> Result<String, String> {
    let response = connection
        .read_text_file(
            ReadTextFileRequest::new(session_id.clone(), path.to_path_buf())
                .line(line)
                .limit(limit),
        )
        .await
        .map_err(|error| read_file_client_error(path, &error.to_string()))?;

    Ok(response.content)
}

async fn read_full_file_from_client<'a>(
    connection: &'a dyn super::ReadTextFileRequester,
    session_id: &'a SessionId,
    path: &'a Path,
) -> Result<String, String> {
    let response = connection
        .read_text_file(ReadTextFileRequest::new(
            session_id.clone(),
            path.to_path_buf(),
        ))
        .await
        .map_err(|error| read_file_client_error(path, &error.to_string()))?;

    Ok(response.content)
}

async fn read_existing_text(
    context: &ToolContext,
    path: &Path,
    read_connection: Option<&dyn super::ReadTextFileRequester>,
    use_client_write: bool,
) -> Result<Option<String>, String> {
    if use_client_write {
        let can_client_read = context
            .client_capabilities
            .as_ref()
            .is_some_and(|capabilities| capabilities.fs.read_text_file);
        if !can_client_read {
            return Ok(None);
        }
        let Some(connection) = read_connection else {
            return Ok(None);
        };
        return read_full_file_from_client(connection, &context.session_id, path)
            .await
            .map(Some);
    }

    match fs::read_to_string(path) {
        Ok(text) => Ok(Some(text)),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
        Err(error) => Err(read_file_local_error(path, &error)),
    }
}

pub(crate) async fn write_file_to_client(
    connection: &dyn super::WriteTextFileRequester,
    session_id: &SessionId,
    path: &Path,
    content: &str,
) -> Result<(), String> {
    connection
        .write_text_file(WriteTextFileRequest::new(
            session_id.clone(),
            path.to_path_buf(),
            content.to_owned(),
        ))
        .await
        .map_err(|error| {
            format!(
                "failed to write {} through client fs/write_text_file: {error}",
                path.display()
            )
        })?;

    Ok(())
}

fn write_file_to_local(path: &Path, content: &str) -> Result<(), String> {
    fs::write(path, content.as_bytes())
        .map_err(|error| format!("failed to write {}: {error}", path.display()))
}

fn line_number_for_offset(text: &str, offset: usize) -> u32 {
    let Some(prefix) = text.get(..offset) else {
        return 1;
    };
    let line = prefix
        .bytes()
        .filter(|byte| *byte == b'\n')
        .count()
        .saturating_add(1);

    u32::try_from(line).unwrap_or(u32::MAX)
}

pub(crate) fn read_file_from_local(path: &Path, line: u32, limit: u32) -> Result<String, String> {
    let text = fs::read_to_string(path).map_err(|error| read_file_local_error(path, &error))?;
    let lines: Vec<&str> = text.lines().collect();

    let start_index = usize::try_from(line.saturating_sub(1))
        .map_err(|error| format!("line number is too large: {error}"))?;
    let max_lines =
        usize::try_from(limit).map_err(|error| format!("line limit is too large: {error}"))?;

    let content = lines
        .iter()
        .skip(start_index)
        .take(max_lines)
        .copied()
        .collect::<Vec<&str>>()
        .join("\n");

    Ok(content)
}

pub(crate) fn local_file_is_non_utf8(path: &Path) -> bool {
    fs::read_to_string(path).is_err_and(|error| error.kind() == ErrorKind::InvalidData)
}

pub(crate) fn read_file_local_error(path: &Path, error: &std::io::Error) -> String {
    if error.kind() == ErrorKind::InvalidData {
        return non_utf8_file_message(path);
    }

    format!("failed to read {}: {error}", path.display())
}

pub(crate) fn read_file_client_error(path: &Path, message: &str) -> String {
    if is_utf8_error_message(message) {
        return non_utf8_file_message(path);
    }

    format!(
        "failed to read {} through client fs/read_text_file: {message}",
        path.display()
    )
}

pub(crate) fn non_utf8_file_message(path: &Path) -> String {
    format!(
        "read_file only supports UTF-8 text files; {} appears to be binary or non-UTF-8",
        path.display()
    )
}

pub(crate) fn is_utf8_error_message(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("valid utf-8")
        || lower.contains("invalid utf-8")
        || lower.contains("non-utf-8")
        || lower.contains("utf8")
}

pub(crate) fn resolve_tool_path(context: &ToolContext, path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }

    let candidate = context.cwd.join(path);
    if candidate.exists() {
        return candidate;
    }

    for directory in &context.additional_directories {
        let alternate = directory.join(path);
        if alternate.exists() {
            return alternate;
        }
    }

    candidate
}

pub(crate) fn collect_directory_entries(path: &Path) -> Result<Vec<String>, String> {
    let mut entries = fs::read_dir(path)
        .map_err(|error| format!("failed to read directory {}: {error}", path.display()))?
        .map(|entry| {
            entry.map_err(|error| {
                format!(
                    "failed to read directory entry in {}: {error}",
                    path.display()
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    entries.sort_unstable_by(|left, right| {
        left.file_name()
            .to_string_lossy()
            .cmp(&right.file_name().to_string_lossy())
    });

    Ok(entries
        .into_iter()
        .map(|entry| {
            let display = entry.file_name().to_string_lossy().into_owned();
            match entry.file_type() {
                Ok(file_type) if file_type.is_dir() => format!("{display}/"),
                _ => display,
            }
        })
        .collect())
}

pub(crate) fn is_hidden_path(path: &Path) -> bool {
    path.components()
        .any(|component| component.as_os_str().to_string_lossy().starts_with('.'))
}

pub(crate) fn build_root_gitignore(root: &Path) -> Option<ignore::gitignore::Gitignore> {
    let gitignore_path = root.join(".gitignore");
    if !gitignore_path.is_file() {
        return None;
    }

    let mut builder = GitignoreBuilder::new(root);
    if builder.add(&gitignore_path).is_some() {
        return None;
    }

    builder.build().ok()
}

pub(crate) fn render_tool_lines(
    lines: &[String],
    truncated: bool,
    label: &str,
    limit: usize,
) -> String {
    let mut output = lines.join("\n");

    if truncated {
        if !output.is_empty() {
            output.push('\n');
        }
        let _ = write!(output, "... truncated after {limit} {label}");
    }

    output
}

pub(crate) fn render_command_output(stdout: &str, stderr: &str, exit_code: Option<i32>) -> String {
    let mut output = String::new();

    if !stdout.is_empty() {
        output.push_str("stdout:\n");
        output.push_str(stdout);
        if !stdout.ends_with('\n') {
            output.push('\n');
        }
    }

    if !stderr.is_empty() {
        if !output.is_empty() {
            output.push('\n');
        }
        output.push_str("stderr:\n");
        output.push_str(stderr);
        if !stderr.ends_with('\n') {
            output.push('\n');
        }
    }

    if output.is_empty() {
        let status = exit_code.map_or_else(|| "signal".to_string(), |code| code.to_string());
        let _ = write!(output, "command exited with status {status}");
    }

    output
}

pub(crate) fn truncate_tool_output(output: &str, limit: usize) -> (String, bool) {
    let truncated = output.chars().count() > limit;
    if !truncated {
        return (output.to_string(), false);
    }

    let mut content = output.chars().take(limit).collect::<String>();
    let _ = write!(content, "\n... truncated after {limit} characters");
    (content, true)
}

#[derive(Debug, Clone)]
struct GrepMatch {
    path: String,
    line: u64,
    text: String,
}

fn collect_grep_matches(
    root: &Path,
    root_gitignore: Option<&ignore::gitignore::Gitignore>,
    matcher: &RegexMatcher,
) -> Result<(Vec<GrepMatch>, bool), String> {
    let mut searcher = SearcherBuilder::new()
        .binary_detection(BinaryDetection::quit(b'\x00'))
        .line_number(true)
        .build();
    let mut grep_hits = Vec::<GrepMatch>::new();
    let mut truncated = false;

    let walker = WalkBuilder::new(root)
        .hidden(false)
        .parents(true)
        .ignore(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .build();
    for entry in walker {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => return Err(error.to_string()),
        };

        if !entry
            .file_type()
            .is_some_and(|file_type| file_type.is_file())
        {
            continue;
        }

        if is_hidden_path(entry.path()) {
            continue;
        }

        let path = entry.path().to_path_buf();
        let relative_path = path.strip_prefix(root).unwrap_or(&path);
        if root_gitignore.as_ref().is_some_and(|matcher| {
            matcher
                .matched_path_or_any_parents(relative_path, false)
                .is_ignore()
        }) {
            continue;
        }

        let search_result = searcher.search_path(
            matcher,
            &path,
            UTF8(|line_number, line| {
                if grep_hits.len() >= TOOL_OUTPUT_LIMIT {
                    truncated = true;
                    return Ok(false);
                }

                grep_hits.push(GrepMatch {
                    path: relative_path.display().to_string(),
                    line: line_number,
                    text: line.to_string(),
                });
                if grep_hits.len() >= TOOL_OUTPUT_LIMIT {
                    truncated = true;
                    Ok(false)
                } else {
                    Ok(true)
                }
            }),
        );
        if let Err(error) = search_result {
            return Err(format!("failed to grep {}: {error}", path.display()));
        }

        if truncated {
            break;
        }
    }

    Ok((grep_hits, truncated))
}
