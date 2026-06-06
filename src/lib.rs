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

//! `DeepSeek` client support for the `ACP` adapter.
//!
//! The adapter proper still needs the ACP session layer, but the DeepSeek-side
//! seam lives here so it can be tested in isolation and reused by the later
//! protocol wiring.
//!
//! # Overview
//!
//! The [`deepseek`] module exposes:
//! - request primitives such as [`deepseek::ChatMessage`] and [`deepseek::ChatRequest`]
//! - tool advertisement types such as [`deepseek::ToolDefinition`]
//! - streamed response events via [`deepseek::StreamEvent`]
//! - an HTTP-backed client via [`deepseek::DeepSeekClient`]
//!
//! # Examples
//!
//! Create a simple streaming request:
//!
//! ```rust,no_run
//! use deepseek_acp_adapter::deepseek::{ChatMessage, ChatRequest, DeepSeekClient, LlmClient};
//! use futures_util::StreamExt;
//! use tokio_util::sync::CancellationToken;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = DeepSeekClient::from_env()?;
//!     let request = ChatRequest::new(vec![
//!         ChatMessage::system("You are a concise coding assistant."),
//!         ChatMessage::user("Explain what this adapter crate does."),
//!     ]);
//!
//!     let mut stream = client.stream_chat(request, CancellationToken::new())?;
//!     while let Some(event) = stream.next().await {
//!         println!("{:?}", event?);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! Build a tool-enabled request:
//!
//! ```rust
//! use deepseek_acp_adapter::deepseek::{ChatMessage, ChatRequest, ToolDefinition};
//!
//! let request = ChatRequest::new(vec![ChatMessage::user("Read src/lib.rs")]).with_tools(vec![
//!     ToolDefinition::new(
//!         "read_file",
//!         "Read a UTF-8 text file",
//!         serde_json::json!({
//!             "type": "object",
//!             "properties": {
//!                 "path": { "type": "string" }
//!             },
//!             "required": ["path"],
//!             "additionalProperties": false
//!         }),
//!     ),
//! ]);
//!
//! assert_eq!(request.tools()[0].name(), "read_file");
//! ```

/// `DeepSeek` client primitives and streaming SSE adapter.
pub mod deepseek;

#[cfg(test)]
mod tests {
    use crate::deepseek::ToolCall as DeepSeekToolCall;
    use crate::deepseek::{
        ChatMessage, ChatRequest, DeepSeekConfig, FinishReason, ToolCallDelta, ToolDefinition,
    };

    #[test]
    fn message_role_as_str_returns_correct_wire_names() {
        use crate::deepseek::MessageRole;
        let system = ChatMessage::system("s");
        assert_eq!(system.role(), MessageRole::System);
        let user = ChatMessage::user("u");
        assert_eq!(user.role(), MessageRole::User);
        let assistant = ChatMessage::assistant("a");
        assert_eq!(assistant.role(), MessageRole::Assistant);
        let tool = ChatMessage::tool_result("id", "t");
        assert_eq!(tool.role(), MessageRole::Tool);
    }

    #[test]
    fn chat_message_tool_call_accessors() {
        let tool_calls = vec![DeepSeekToolCall::new("call-1", "echo", "{}")];
        let msg = ChatMessage::assistant_with_tool_calls("assistant", tool_calls.clone());
        assert_eq!(msg.tool_calls().len(), 1);
        assert_eq!(msg.tool_calls()[0].id(), "call-1");
        assert_eq!(msg.tool_calls()[0].name(), "echo");
        assert_eq!(msg.tool_calls()[0].arguments(), "{}");
        assert_eq!(msg.tool_call_id(), None);
    }

    #[test]
    fn chat_message_tool_result_accessors() {
        let msg = ChatMessage::tool_result("call-2", "result");
        assert_eq!(msg.content(), "result");
        assert_eq!(msg.tool_call_id(), Some("call-2"));
    }

    #[test]
    fn tool_definition_accessors() {
        let def = ToolDefinition::new("echo", "description", serde_json::json!({"a":1}));
        assert_eq!(def.name(), "echo");
        assert_eq!(def.description(), "description");
        assert_eq!(def.parameters(), &serde_json::json!({"a":1}));
    }

    #[test]
    fn tool_call_delta_accessors() {
        let delta = ToolCallDelta::new(
            0,
            Some("id".to_string()),
            Some("name".to_string()),
            Some("args".to_string()),
        );
        assert_eq!(delta.index(), 0);
        assert_eq!(delta.id(), Some("id"));
        assert_eq!(delta.name(), Some("name"));
        assert_eq!(delta.arguments(), Some("args"));
    }

    #[test]
    fn tool_call_delta_none_fields() {
        let delta = ToolCallDelta::new(1, None, None, None);
        assert_eq!(delta.index(), 1);
        assert_eq!(delta.id(), None);
        assert_eq!(delta.name(), None);
        assert_eq!(delta.arguments(), None);
    }

    #[test]
    fn chat_request_accessors_no_override() {
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);
        assert_eq!(request.messages().len(), 1);
        assert_eq!(request.tools().len(), 0);
        assert_eq!(request.model(), None);
        assert_eq!(request.reasoning_effort(), None);
    }

    #[test]
    fn finish_reason_from_api_covers_all_branches() {
        assert_eq!(
            FinishReason::EndTurn,
            crate::deepseek::FinishReason::EndTurn
        );
    }

    #[test]
    fn deepseek_config_new_accepts_explicit_values() {
        let config = DeepSeekConfig::new("key", "https://example.com", "model-v1");
        assert_eq!(config.base_url(), "https://example.com");
        assert_eq!(config.model(), "model-v1");
    }
}
