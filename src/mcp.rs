//! MCP session startup, tool mapping, and invocation helpers.

use std::collections::HashMap;

use agent_client_protocol::schema::{
    HttpHeader, McpServer, McpServerHttp, McpServerStdio, ToolKind,
};
use deepseek_acp_adapter::deepseek::{ToolCall as DeepSeekToolCall, ToolDefinition};
use http::{HeaderName, HeaderValue};
use rmcp::model::{CallToolRequestParams, Content as McpContent, JsonObject, Tool as McpTool};
use rmcp::service::RunningService;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::{ConfigureCommandExt, StreamableHttpClientTransport, TokioChildProcess};
use rmcp::{Peer, RoleClient, ServiceExt};
use serde_json::Value;
use tokio::process::Command as TokioCommand;

use crate::SessionStore;
use crate::tools::{ToolContext, ToolExecution};

/// Prefix used for model-visible MCP tool names.
pub(crate) const MCP_TOOL_PREFIX: &str = "mcp";
const MCP_TOOL_NAME_PREFIX: &str = "mcp__";

/// Permission kind used for all MCP tools.
///
/// MCP servers are external executors with unknown side effects, so they are
/// treated like command execution for approval decisions.
pub(crate) const MCP_TOOL_KIND: ToolKind = ToolKind::Execute;

#[derive(Debug)]
pub(crate) struct McpSession {
    pub(crate) name: String,
    pub(crate) tools: Vec<McpToolMapping>,
    pub(crate) peer: Peer<RoleClient>,
    pub(crate) _service: RunningService<RoleClient, ()>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct McpToolMapping {
    pub(crate) exposed_name: String,
    pub(crate) original_name: String,
    pub(crate) definition: ToolDefinition,
}

#[derive(Debug, Clone)]
pub(crate) struct McpToolTarget {
    pub(crate) server_name: String,
    pub(crate) original_name: String,
    pub(crate) peer: Peer<RoleClient>,
}

/// Return whether a tool name belongs to an MCP-backed tool.
#[must_use]
pub(crate) fn is_mcp_tool_name(name: &str) -> bool {
    name.starts_with(MCP_TOOL_NAME_PREFIX)
}

/// Return the explicit ACP permission kind used for MCP tools.
#[must_use]
pub(crate) const fn mcp_tool_kind() -> ToolKind {
    MCP_TOOL_KIND
}

/// Execute an MCP tool call for the given session.
#[must_use]
pub(crate) async fn mcp_tool_execution(
    store: &SessionStore,
    call: &DeepSeekToolCall,
    context: &ToolContext,
) -> ToolExecution {
    let target = match store.find_mcp_target(&context.session_id, call.name()) {
        Ok(Some(target)) => target,
        Ok(None) => return ToolExecution::failed(format!("unknown MCP tool: {}", call.name())),
        Err(error) => return ToolExecution::failed(error.to_string()),
    };

    let arguments = match mcp_call_arguments(call) {
        Ok(arguments) => arguments,
        Err(error) => return ToolExecution::failed(error),
    };

    let result = target
        .peer
        .call_tool(
            CallToolRequestParams::new(target.original_name.clone()).with_arguments(arguments),
        )
        .await;

    match result {
        Ok(result) => {
            let model_output = mcp_tool_result_text(&result.content);
            let raw_output = serde_json::to_value(&result).unwrap_or_else(|error| {
                serde_json::json!({
                    "error": format!("failed to serialize MCP tool result: {error}")
                })
            });
            ToolExecution {
                content: model_output,
                raw_output,
                success: !result.is_error.unwrap_or(false),
                edit: None,
            }
        }
        Err(error) => ToolExecution::failed(format!(
            "MCP tool '{}' on server '{}' failed: {error}",
            target.original_name, target.server_name
        )),
    }
}

/// Parse MCP tool arguments from the model-emitted JSON payload.
pub(crate) fn mcp_call_arguments(call: &DeepSeekToolCall) -> Result<JsonObject, String> {
    match serde_json::from_str::<Value>(call.arguments()) {
        Ok(Value::Object(arguments)) => Ok(arguments),
        Ok(_) => Err(format!(
            "MCP tool '{}' arguments must be a JSON object",
            call.name()
        )),
        Err(error) => Err(format!(
            "invalid MCP tool '{}' arguments: {error}",
            call.name()
        )),
    }
}

/// Render MCP result content into the plain text fed back to the model.
#[must_use]
pub(crate) fn mcp_tool_result_text(content: &[McpContent]) -> String {
    let parts = content
        .iter()
        .map(|content| {
            content.raw.as_text().map_or_else(
                || {
                    serde_json::to_string(&content.raw)
                        .unwrap_or_else(|error| format!("failed to serialize MCP content: {error}"))
                },
                |text| text.text.clone(),
            )
        })
        .collect::<Vec<_>>();

    if parts.is_empty() {
        String::new()
    } else {
        parts.join("\n")
    }
}

/// Connect all requested MCP servers for a new ACP session.
///
/// # Errors
///
/// Returns an ACP error when a server uses a non-stdio transport, the command
/// is invalid, the process cannot be started, or the server cannot be queried
/// for tools.
pub(crate) async fn connect_mcp_sessions(
    servers: &[McpServer],
) -> Result<Vec<McpSession>, agent_client_protocol::Error> {
    let mut sessions = Vec::new();

    for server in servers {
        match server {
            McpServer::Stdio(stdio) => sessions.push(connect_mcp_stdio_session(stdio).await?),
            McpServer::Http(http) => sessions.push(connect_mcp_http_session(http).await?),
            McpServer::Sse(_) => {
                return Err(agent_client_protocol::Error::invalid_params()
                    .data("SSE MCP servers are not supported"));
            }
            _ => {
                return Err(agent_client_protocol::Error::invalid_params()
                    .data("unsupported MCP server transport"));
            }
        }
    }

    Ok(sessions)
}

/// Connect a single stdio MCP server and collect its advertised tools.
///
/// # Errors
///
/// Returns an ACP error when the command path is not absolute, the process
/// fails to start, initialization fails, or tool discovery fails.
pub(crate) async fn connect_mcp_stdio_session(
    server: &McpServerStdio,
) -> Result<McpSession, agent_client_protocol::Error> {
    if !server.command.is_absolute() {
        return Err(agent_client_protocol::Error::invalid_params().data(format!(
            "MCP server '{}' command must be absolute",
            server.name
        )));
    }

    let command = TokioCommand::new(&server.command).configure(|command| {
        command.args(&server.args);
        for variable in &server.env {
            command.env(&variable.name, &variable.value);
        }
    });
    let transport = TokioChildProcess::new(command).map_err(|error| {
        agent_client_protocol::Error::invalid_params().data(format!(
            "failed to start MCP server '{}': {error}",
            server.name
        ))
    })?;
    let service = ().serve(transport).await.map_err(|error| {
        agent_client_protocol::Error::invalid_params().data(format!(
            "failed to initialize MCP server '{}': {error}",
            server.name
        ))
    })?;
    mcp_session_from_service(&server.name, service).await
}

/// Connect a single streamable HTTP MCP server and collect its advertised tools.
///
/// # Errors
///
/// Returns an ACP error when headers are invalid, initialization fails, or tool
/// discovery fails.
pub(crate) async fn connect_mcp_http_session(
    server: &McpServerHttp,
) -> Result<McpSession, agent_client_protocol::Error> {
    let custom_headers = mcp_http_headers(&server.headers, &server.name)?;
    let config = StreamableHttpClientTransportConfig::with_uri(server.url.clone())
        .custom_headers(custom_headers);
    let transport = StreamableHttpClientTransport::from_config(config);
    let service = ().serve(transport).await.map_err(|error| {
        agent_client_protocol::Error::invalid_params().data(format!(
            "failed to initialize MCP server '{}': {error}",
            server.name
        ))
    })?;

    mcp_session_from_service(&server.name, service).await
}

async fn mcp_session_from_service(
    server_name: &str,
    service: RunningService<RoleClient, ()>,
) -> Result<McpSession, agent_client_protocol::Error> {
    let peer = service.peer().clone();
    let tools = peer.list_all_tools().await.map_err(|error| {
        agent_client_protocol::Error::invalid_params().data(format!(
            "failed to list MCP tools for server '{server_name}': {error}",
        ))
    })?;
    let mappings = mcp_tool_mappings(server_name, tools);

    Ok(McpSession {
        name: server_name.to_string(),
        tools: mappings,
        peer,
        _service: service,
    })
}

fn mcp_http_headers(
    headers: &[HttpHeader],
    server_name: &str,
) -> Result<HashMap<HeaderName, HeaderValue>, agent_client_protocol::Error> {
    let mut parsed = HashMap::with_capacity(headers.len());
    for header in headers {
        let name = HeaderName::from_bytes(header.name.as_bytes()).map_err(|error| {
            agent_client_protocol::Error::invalid_params().data(format!(
                "invalid HTTP header name '{}' for MCP server '{server_name}': {error}",
                header.name
            ))
        })?;
        let value = HeaderValue::from_str(&header.value).map_err(|error| {
            agent_client_protocol::Error::invalid_params().data(format!(
                "invalid HTTP header value for '{}' on MCP server '{server_name}': {error}",
                header.name
            ))
        })?;
        parsed.insert(name, value);
    }
    Ok(parsed)
}

/// Map MCP server tool metadata into model-visible tool definitions.
#[must_use]
pub(crate) fn mcp_tool_mappings(server_name: &str, tools: Vec<McpTool>) -> Vec<McpToolMapping> {
    tools
        .into_iter()
        .map(|tool| {
            let original_name = tool.name.to_string();
            let exposed_name = mcp_tool_name(server_name, &original_name);
            let description = tool.description.map_or_else(
                || format!("MCP tool '{original_name}' from server '{server_name}'"),
                |description| description.to_string(),
            );
            let definition = ToolDefinition::new(
                exposed_name.clone(),
                description,
                Value::Object(tool.input_schema.as_ref().clone()),
            );

            McpToolMapping {
                exposed_name,
                original_name,
                definition,
            }
        })
        .collect()
}

fn mcp_tool_name(server_name: &str, tool_name: &str) -> String {
    format!(
        "{MCP_TOOL_PREFIX}__{}__{}",
        sanitize_tool_name_part(server_name),
        sanitize_tool_name_part(tool_name)
    )
}

pub(crate) fn sanitize_tool_name_part(value: &str) -> String {
    let mut sanitized = String::new();
    for character in value.chars() {
        if character.is_ascii_alphanumeric() {
            sanitized.push(character.to_ascii_lowercase());
        } else {
            sanitized.push('_');
        }
    }

    let trimmed = sanitized.trim_matches('_');
    if trimmed.is_empty() {
        "unnamed".to_string()
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests;
