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
mod tests {
    use super::{
        McpSession, connect_mcp_sessions, connect_mcp_stdio_session, mcp_call_arguments,
        mcp_http_headers, mcp_tool_execution, mcp_tool_mappings, mcp_tool_result_text,
    };
    use crate::tools::{AdapterToolRegistry, ToolContext, ToolRegistry};
    use crate::{
        PermissionDecision, PermissionRequester, handle_new_session_request,
        handle_set_session_mode_request, request_tool_permission, test_store,
    };
    use agent_client_protocol::schema::{
        McpServer, McpServerStdio, NewSessionRequest, RequestPermissionOutcome,
        RequestPermissionRequest, RequestPermissionResponse, SelectedPermissionOutcome,
        SetSessionModeRequest, ToolKind,
    };
    use deepseek_acp_adapter::deepseek::ToolCall as DeepSeekToolCall;
    use futures_util::future::BoxFuture;
    use rmcp::model::{
        CallToolRequestParams, CallToolResult, Content as McpContent, ListToolsResult,
        PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool as McpTool,
    };
    use rmcp::service::{RequestContext, RoleServer};
    use rmcp::{ServerHandler, ServiceExt};
    use serde_json::Value;
    use std::collections::VecDeque;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    #[derive(Debug, Clone)]
    struct EchoMcpServer;

    impl ServerHandler for EchoMcpServer {
        fn get_info(&self) -> ServerInfo {
            ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
        }

        async fn call_tool(
            &self,
            request: CallToolRequestParams,
            _context: RequestContext<RoleServer>,
        ) -> Result<CallToolResult, rmcp::ErrorData> {
            let message = request
                .arguments
                .as_ref()
                .and_then(|arguments| arguments.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("");
            Ok(CallToolResult::success(vec![McpContent::text(format!(
                "echo: {message}"
            ))]))
        }

        async fn list_tools(
            &self,
            _request: Option<PaginatedRequestParams>,
            _context: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, rmcp::ErrorData> {
            Ok(ListToolsResult {
                tools: vec![McpTool::new(
                    "echo",
                    "Echo a provided message",
                    rmcp::model::object(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "message": { "type": "string" }
                        },
                        "required": ["message"]
                    })),
                )],
                ..Default::default()
            })
        }
    }

    async fn connected_echo_mcp_session() -> Result<McpSession, agent_client_protocol::Error> {
        let (server_transport, client_transport) = tokio::io::duplex(4096);
        let server_task = tokio::spawn(async move {
            let running = EchoMcpServer
                .serve(server_transport)
                .await
                .map_err(|error| error.to_string())?;
            running.waiting().await.map_err(|error| error.to_string())?;
            Ok::<(), String>(())
        });
        drop(server_task);

        let service = ().serve(client_transport).await.map_err(|error| {
            agent_client_protocol::Error::internal_error()
                .data(format!("failed to initialize test MCP client: {error}"))
        })?;
        let peer = service.peer().clone();
        let tools = peer.list_all_tools().await.map_err(|error| {
            agent_client_protocol::Error::internal_error()
                .data(format!("failed to list test MCP tools: {error}"))
        })?;

        Ok(McpSession {
            name: "Echo Server".to_string(),
            tools: mcp_tool_mappings("Echo Server", tools),
            peer,
            _service: service,
        })
    }

    struct FakePermissionRequester {
        requests: Arc<Mutex<Vec<RequestPermissionRequest>>>,
        responses: Mutex<VecDeque<RequestPermissionResponse>>,
    }

    impl FakePermissionRequester {
        fn new(responses: Vec<RequestPermissionResponse>) -> Self {
            Self {
                requests: Arc::new(Mutex::new(Vec::new())),
                responses: Mutex::new(VecDeque::from(responses)),
            }
        }

        fn requests(&self) -> Arc<Mutex<Vec<RequestPermissionRequest>>> {
            Arc::clone(&self.requests)
        }
    }

    impl PermissionRequester for FakePermissionRequester {
        fn request_permission(
            &self,
            request: RequestPermissionRequest,
        ) -> BoxFuture<'_, Result<RequestPermissionResponse, agent_client_protocol::Error>>
        {
            self.requests
                .lock()
                .map(|mut requests| requests.push(request))
                .ok();

            let response = self
                .responses
                .lock()
                .map_err(|error| {
                    agent_client_protocol::Error::internal_error().data(error.to_string())
                })
                .and_then(|mut responses| {
                    responses.pop_front().ok_or_else(|| {
                        agent_client_protocol::Error::internal_error()
                            .data("fake permission requester was exhausted")
                    })
                });

            Box::pin(async move { response })
        }
    }

    #[test_log::test(tokio::test)]
    async fn mcp_stdio_launch_failure_returns_invalid_params() {
        let result = connect_mcp_sessions(&[McpServer::Stdio(McpServerStdio::new(
            "broken",
            "/definitely/not/a/real/mcp-server",
        ))])
        .await;

        assert!(result.is_err());
        let error_text = result
            .err()
            .map_or_else(String::new, |error| format!("{error:?}"));
        assert!(error_text.contains("failed to start MCP server 'broken'"));
    }

    #[test_log::test]
    fn mcp_tool_mappings_prefix_and_preserve_schema() {
        let mappings = mcp_tool_mappings(
            "Test Server",
            vec![McpTool::new(
                "Read File",
                "Read through MCP",
                rmcp::model::object(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" }
                    }
                })),
            )],
        );

        assert_eq!(mappings.len(), 1);
        let mapping = &mappings[0];
        assert_eq!(mapping.exposed_name, "mcp__test_server__read_file");
        assert_eq!(mapping.original_name, "Read File");
        assert_eq!(mapping.definition.name(), "mcp__test_server__read_file");
        assert_eq!(mapping.definition.description(), "Read through MCP");
        assert_eq!(
            mapping.definition.parameters()["properties"]["path"]["type"],
            "string"
        );
    }

    #[test_log::test(tokio::test)]
    async fn adapter_registry_exposes_and_executes_session_mcp_tools()
    -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let response = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
        let mcp_session = connected_echo_mcp_session().await?;
        {
            let mut guard = store
                .state
                .lock()
                .map_err(agent_client_protocol::Error::into_internal_error)?;
            let session = guard
                .sessions
                .get_mut(&response.session_id)
                .ok_or_else(|| {
                    agent_client_protocol::Error::internal_error().data("missing session")
                })?;
            session.mcp_sessions.push(mcp_session);
        }

        let context = ToolContext {
            session_id: response.session_id.clone(),
            cwd: PathBuf::from("/tmp"),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let registry = AdapterToolRegistry;
        let definitions = registry.definitions(&context, &store)?;
        assert!(
            definitions
                .iter()
                .any(|definition| definition.name() == "mcp__echo_server__echo")
        );

        let result = registry
            .execute(
                &DeepSeekToolCall::new(
                    "call-mcp",
                    "mcp__echo_server__echo",
                    serde_json::json!({ "message": "hello" }).to_string(),
                ),
                &context,
                &store,
                None,
                tokio_util::sync::CancellationToken::new(),
            )
            .await;

        assert!(result.success);
        assert_eq!(result.content, "echo: hello");

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn mcp_tools_use_explicit_execute_permission_kind()
    -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
        handle_set_session_mode_request(
            &store,
            &SetSessionModeRequest::new(session.session_id.clone(), "accept-edits"),
        )?;
        let context = ToolContext {
            session_id: session.session_id,
            cwd: PathBuf::from("/tmp"),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let call = DeepSeekToolCall::new(
            "call-mcp-permission",
            "mcp__server__tool",
            serde_json::json!({ "message": "hello" }).to_string(),
        );
        let requester = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
            RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                crate::PERMISSION_ALLOW_ONCE_OPTION_ID,
            )),
        )]);

        let decision =
            request_tool_permission(&store, &context, &call, super::mcp_tool_kind(), &requester)
                .await?;

        assert_eq!(decision, PermissionDecision::AllowOnce);
        let requests = requester.requests();
        let request_guard = requests
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        assert_eq!(request_guard.len(), 1);
        assert_eq!(
            request_guard[0].tool_call.fields.kind,
            Some(ToolKind::Execute)
        );

        Ok(())
    }

    #[test]
    fn mcp_call_arguments_rejects_non_object_json() -> Result<(), agent_client_protocol::Error> {
        let call = DeepSeekToolCall::new("mcp-args", "mcp__server__tool", "[1,2,3]");
        let Err(error) = mcp_call_arguments(&call) else {
            return Err(
                agent_client_protocol::Error::internal_error().data("expected object rejection")
            );
        };
        assert!(error.contains("arguments must be a JSON object"));
        Ok(())
    }

    #[test]
    fn mcp_call_arguments_rejects_invalid_json() -> Result<(), agent_client_protocol::Error> {
        let call = DeepSeekToolCall::new("mcp-args", "mcp__server__tool", "{oops");
        let Err(error) = mcp_call_arguments(&call) else {
            return Err(
                agent_client_protocol::Error::internal_error().data("expected JSON rejection")
            );
        };
        assert!(error.contains("invalid MCP tool"));
        Ok(())
    }

    #[test]
    fn mcp_tool_result_text_returns_empty_for_no_content() {
        let result: &[McpContent] = &[];
        assert_eq!(mcp_tool_result_text(result), "");
    }

    #[test_log::test(tokio::test)]
    async fn connect_mcp_sessions_rejects_sse() {
        let result = connect_mcp_sessions(&[McpServer::Sse(
            agent_client_protocol::schema::McpServerSse::new("events", "http://localhost/sse"),
        )])
        .await;
        let Err(error) = result else {
            return;
        };
        assert!(
            error
                .to_string()
                .contains("SSE MCP servers are not supported")
        );
    }

    #[test]
    fn mcp_http_headers_parses_custom_headers() -> Result<(), agent_client_protocol::Error> {
        let headers = [agent_client_protocol::schema::HttpHeader::new(
            "X-Client-Trace",
            "trace-id",
        )];

        let parsed = mcp_http_headers(&headers, "remote")?;

        assert_eq!(parsed.len(), 1);
        let header_name = http::HeaderName::from_static("x-client-trace");
        assert_eq!(
            parsed
                .get(&header_name)
                .and_then(|value| value.to_str().ok()),
            Some("trace-id")
        );
        Ok(())
    }

    #[test]
    fn mcp_http_headers_rejects_invalid_header_name() -> Result<(), agent_client_protocol::Error> {
        let headers = [agent_client_protocol::schema::HttpHeader::new(
            "bad header",
            "secret",
        )];

        let Err(error) = mcp_http_headers(&headers, "remote") else {
            return Err(
                agent_client_protocol::Error::internal_error().data("expected header rejection")
            );
        };

        let error = error.to_string();
        assert!(error.contains("invalid HTTP header name"));
        assert!(!error.contains("secret"));
        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn mcp_tool_execution_unknown_tool() -> Result<(), agent_client_protocol::Error> {
        let store = test_store();
        let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
        let context = ToolContext {
            session_id: session.session_id.clone(),
            cwd: PathBuf::from("/tmp"),
            additional_directories: Vec::new(),
            client_capabilities: None,
        };
        let call = DeepSeekToolCall::new("mcp-unknown", "mcp__nonexistent__tool", "{}");
        let result = mcp_tool_execution(&store, &call, &context).await;
        assert!(!result.success);
        assert!(result.content.contains("unknown MCP tool"));
        Ok(())
    }

    #[test]
    fn mcp_tool_result_text_serializes_non_text_content() {
        let content = vec![McpContent::text("text part")];
        assert_eq!(mcp_tool_result_text(&content), "text part");
    }

    #[test_log::test(tokio::test)]
    async fn connect_mcp_sessions_with_empty_list_returns_ok() {
        let result = connect_mcp_sessions(&[]).await;
        assert!(result.is_ok());
        let sessions: Vec<McpSession> = result.unwrap_or_default();
        assert!(sessions.is_empty());
    }

    #[test_log::test(tokio::test)]
    async fn mcp_stdio_session_rejects_relative_command() {
        let stdio = McpServerStdio::new("rel", "relative/path");
        let result = connect_mcp_stdio_session(&stdio).await;
        let Err(error) = result else {
            return;
        };
        assert!(error.to_string().contains("command must be absolute"));
    }

    #[test]
    fn is_mcp_tool_name_matches_prefixed_only() {
        assert!(super::is_mcp_tool_name("mcp__server__tool"));
        assert!(!super::is_mcp_tool_name("mcp_server_tool"));
        assert!(!super::is_mcp_tool_name(""));
        assert!(!super::is_mcp_tool_name("read_file"));
    }

    #[test]
    fn mcp_tool_kind_is_execute() {
        assert_eq!(super::mcp_tool_kind(), ToolKind::Execute);
    }

    #[test]
    fn mcp_tool_result_text_serializes_image_content() {
        let content = vec![McpContent::image("base64data", "image/png")];
        let result = mcp_tool_result_text(&content);
        assert!(!result.is_empty());
        assert!(result.contains("image"));
    }

    #[test]
    fn mcp_tool_result_text_concatenates_multiple_parts() {
        let content = vec![McpContent::text("first"), McpContent::text("second")];
        assert_eq!(mcp_tool_result_text(&content), "first\nsecond");
    }

    #[test]
    fn sanitize_tool_name_handles_empty_result() {
        assert_eq!(super::sanitize_tool_name_part("___"), "unnamed");
        assert_eq!(super::sanitize_tool_name_part(""), "unnamed");
    }

    #[test]
    fn sanitize_tool_name_handles_special_characters() {
        assert_eq!(
            super::sanitize_tool_name_part("Hello World!"),
            "hello_world"
        );
    }
}
