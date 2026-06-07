#![forbid(unsafe_code)]
#![deny(
    warnings,
    missing_docs,
    clippy::all,
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::unimplemented
)]

//! Harnessless stdio MCP server fixture for child-process tests.

use std::error::Error;

use rmcp::model::{
    CallToolRequestParams, CallToolResult, Content as McpContent, ListToolsResult,
    PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool as McpTool,
};
use rmcp::service::{RequestContext, RoleServer};
use rmcp::transport::stdio;
use rmcp::{ServerHandler, ServiceExt};
use serde_json::Value;

const RUN_FIXTURE_ENV: &str = "DEEPSEEK_ACP_ADAPTER_RUN_MCP_FIXTURE";

#[derive(Debug, Clone)]
struct StdioFixtureServer {
    launch_arg: String,
    env_token: String,
    list_tools_error: bool,
}

impl ServerHandler for StdioFixtureServer {
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
            "stdio echo: {message}; arg: {}; env: {}",
            self.launch_arg, self.env_token
        ))]))
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, rmcp::ErrorData> {
        if self.list_tools_error {
            return Err(rmcp::ErrorData::internal_error(
                "simulated list tools failure",
                None,
            ));
        }

        Ok(ListToolsResult {
            tools: vec![McpTool::new(
                "stdio_echo",
                "Echoes a message and launch metadata",
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync + 'static>> {
    if std::env::var_os(RUN_FIXTURE_ENV).is_none() {
        return Ok(());
    }

    let server = StdioFixtureServer {
        launch_arg: std::env::args()
            .nth(1)
            .unwrap_or_else(|| "missing-arg".to_string()),
        env_token: std::env::var("MCP_FIXTURE_TOKEN").unwrap_or_else(|_| "missing-env".to_string()),
        list_tools_error: std::env::var("MCP_FIXTURE_MODE").is_ok_and(|mode| mode == "list_error"),
    };
    let service = server.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}
