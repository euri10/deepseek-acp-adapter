//! Built-in tool execution for the `DeepSeek` `ACP` adapter.
#![allow(unused_imports)]
//!
//! Modules:
//! - `registry`: `ToolContext`, `ToolRegistry`, registry impls, `ToolExecution`
//! - `execution`: tool definitions, execution functions, helpers, tests

mod execution;
#[allow(unused_imports)]
mod registry;

pub(crate) use execution::{
    build_root_gitignore, collect_directory_entries, edit_file_tool_definition,
    edit_file_tool_execution, glob_tool_definition, glob_tool_execution, grep_tool_definition,
    grep_tool_execution, is_hidden_path, is_utf8_error_message, list_dir_tool_definition,
    list_dir_tool_execution, non_utf8_file_message, read_file_client_error, read_file_from_local,
    read_file_local_error, read_file_tool_definition, read_file_tool_execution,
    render_command_output, render_tool_lines, require_tool_permission, resolve_tool_path,
    run_command_tool_definition, run_command_tool_execution, run_command_via_terminal,
    truncate_tool_output, write_file_to_client, write_file_tool_definition,
    write_file_tool_execution,
};
pub(crate) use registry::{
    AdapterToolRegistry, EmptyToolRegistry, ToolContext, ToolEdit, ToolExecution, ToolRegistry,
};
