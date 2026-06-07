//! Built-in tool execution for the `DeepSeek` `ACP` adapter.
//!
//! Modules:
//! - `registry`: `ToolContext`, `ToolRegistry`, registry impls, `ToolExecution`
//! - `execution`: tool definitions, execution functions, helpers, tests

mod execution;
mod registry;

#[cfg(test)]
pub(crate) use execution::require_tool_permission;
pub(crate) use registry::{AdapterToolRegistry, ToolContext, ToolExecution, ToolRegistry};
#[cfg(test)]
pub(crate) use registry::{EmptyToolRegistry, ToolEdit};
