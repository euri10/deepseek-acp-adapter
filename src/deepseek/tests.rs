use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use futures_util::StreamExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use super::client::{DeepSeekClient, LlmClient};
use super::config::{DeepSeekConfig, Environment};
use super::stream::{is_retryable_transport_error, parse_chat_completion_chunk};
use super::{
    ChatMessage, ChatRequest, DeepSeekError, FinishReason, MessageRole, StreamEvent, ToolCall,
    ToolCallDelta, ToolDefinition,
};

struct FakeEnvironment {
    values: BTreeMap<&'static str, &'static str>,
}

impl Environment for FakeEnvironment {
    fn var(&self, key: &str) -> Option<String> {
        self.values.get(key).map(|value| (*value).to_string())
    }
}

async fn spawn_sse_server(
    response_body: String,
    captured_request_body: Arc<Mutex<Option<String>>>,
) -> Result<(String, tokio::task::JoinHandle<Result<(), String>>), DeepSeekError> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|error| DeepSeekError::InvalidResponse(error.to_string()))?;
    let address = listener
        .local_addr()
        .map_err(|error| DeepSeekError::InvalidResponse(error.to_string()))?;

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.map_err(|error| error.to_string())?;
        let mut request = Vec::new();
        let mut buffer = [0_u8; 4096];

        let header_end = loop {
            let read = socket
                .read(&mut buffer)
                .await
                .map_err(|error| error.to_string())?;
            if read == 0 {
                return Err("connection closed before request completed".to_string());
            }
            request.extend_from_slice(&buffer[..read]);
            if let Some(index) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                break index + 4;
            }
        };

        let headers =
            String::from_utf8(request[..header_end].to_vec()).map_err(|error| error.to_string())?;
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                if name.eq_ignore_ascii_case("content-length") {
                    value.trim().parse::<usize>().ok()
                } else {
                    None
                }
            })
            .ok_or_else(|| "missing Content-Length header".to_string())?;

        let mut body = request[header_end..].to_vec();
        while body.len() < content_length {
            let read = socket
                .read(&mut buffer)
                .await
                .map_err(|error| error.to_string())?;
            if read == 0 {
                break;
            }
            body.extend_from_slice(&buffer[..read]);
        }
        body.truncate(content_length);

        let body_text = String::from_utf8(body).map_err(|error| error.to_string())?;
        {
            let mut guard = captured_request_body
                .lock()
                .map_err(|error| error.to_string())?;
            *guard = Some(body_text);
        }

        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            response_body.len(),
            response_body
        );
        socket
            .write_all(response.as_bytes())
            .await
            .map_err(|error| error.to_string())?;
        socket.shutdown().await.map_err(|error| error.to_string())?;
        Ok(())
    });

    Ok((format!("http://{address}"), server))
}

#[test_log::test]
fn config_uses_defaults_and_requires_key() -> Result<(), DeepSeekError> {
    let environment = FakeEnvironment {
        values: BTreeMap::from([("DEEPSEEK_API_KEY", "secret")]),
    };

    let config = DeepSeekConfig::from_environment(&environment)?;

    assert_eq!(config.base_url(), DeepSeekConfig::DEFAULT_BASE_URL);
    assert_eq!(config.model(), DeepSeekConfig::DEFAULT_MODEL);

    let missing_key = FakeEnvironment {
        values: BTreeMap::new(),
    };

    let Err(error) = DeepSeekConfig::from_environment(&missing_key) else {
        return Err(DeepSeekError::InvalidResponse(
            "expected missing API key to fail".to_string(),
        ));
    };

    assert!(matches!(error, DeepSeekError::MissingApiKey));
    assert_eq!(error.to_string(), "DEEPSEEK_API_KEY is not set");

    Ok(())
}

#[test_log::test]
fn config_trims_values_and_defaults_blank_entries() -> Result<(), DeepSeekError> {
    let environment = FakeEnvironment {
        values: BTreeMap::from([
            ("DEEPSEEK_API_KEY", "  secret-token  "),
            ("DEEPSEEK_BASE_URL", "   "),
            ("DEEPSEEK_MODEL", "  custom-model  "),
        ]),
    };

    let config = DeepSeekConfig::from_environment(&environment)?;

    assert_eq!(config.base_url(), DeepSeekConfig::DEFAULT_BASE_URL);
    assert_eq!(config.model(), "custom-model");

    Ok(())
}

#[test_log::test]
fn parses_reasoning_and_text_chunks_in_order() -> Result<(), DeepSeekError> {
    let fixture = r#"
    {
      "choices": [
        {
          "delta": {
            "reasoning_content": "thinking",
            "content": "answer"
          },
          "finish_reason": "stop"
        }
      ]
    }
    "#;

    let updates = parse_chat_completion_chunk(fixture)?;

    assert_eq!(
        updates,
        vec![
            StreamEvent::Thought("thinking".to_string()),
            StreamEvent::Message("answer".to_string()),
            StreamEvent::Finished(FinishReason::EndTurn),
        ]
    );

    Ok(())
}

#[test_log::test]
fn parses_empty_chunks_and_unknown_finish_reasons() -> Result<(), DeepSeekError> {
    let fixture = r#"
    {
      "choices": [
        {
          "delta": {
            "reasoning_content": "",
            "content": ""
          },
          "finish_reason": "rate_limit"
        }
      ]
    }
    "#;

    let updates = parse_chat_completion_chunk(fixture)?;

    assert_eq!(
        updates,
        vec![StreamEvent::Finished(FinishReason::Other(
            "rate_limit".to_string()
        ))]
    );

    Ok(())
}

#[test_log::test]
fn parses_tool_call_deltas() -> Result<(), DeepSeekError> {
    let fixture = r#"
    {
      "choices": [
        {
          "delta": {
            "tool_calls": [
              {
                "index": 0,
                "id": "call-1",
                "function": {
                  "name": "read_file",
                  "arguments": "{\"path\":\"Cargo.toml\"}"
                }
              }
            ]
          },
          "finish_reason": "tool_calls"
        }
      ]
    }
    "#;

    let updates = parse_chat_completion_chunk(fixture)?;

    let StreamEvent::ToolCallDelta(delta) = &updates[0] else {
        return Err(DeepSeekError::InvalidResponse(
            "expected tool call delta".to_string(),
        ));
    };
    assert_eq!(delta.index(), 0);
    assert_eq!(delta.id(), Some("call-1"));
    assert_eq!(delta.name(), Some("read_file"));
    assert_eq!(delta.arguments(), Some(r#"{"path":"Cargo.toml"}"#));
    assert_eq!(updates[1], StreamEvent::Finished(FinishReason::ToolCalls));

    Ok(())
}

#[test_log::test]
fn rejects_chunks_without_choices() -> Result<(), DeepSeekError> {
    let fixture = r#"{ "choices": [] }"#;

    let Err(error) = parse_chat_completion_chunk(fixture) else {
        return Err(DeepSeekError::InvalidResponse(
            "expected empty choice list to fail".to_string(),
        ));
    };

    assert!(matches!(error, DeepSeekError::InvalidResponse(_)));
    assert_eq!(
        error.to_string(),
        "invalid DeepSeek response: chat completion chunk did not include any choices"
    );

    Ok(())
}

#[test_log::test]
fn message_role_round_trips_to_wire_variant() {
    let message = ChatMessage::assistant("hi");

    assert_eq!(message.role(), MessageRole::Assistant);
    assert_eq!(message.content(), "hi");
}

#[test_log::test]
fn client_rejects_empty_api_key() -> Result<(), DeepSeekError> {
    let client = DeepSeekClient::new(DeepSeekConfig::new(
        "",
        "https://api.deepseek.com",
        "deepseek-v4-pro",
    ));
    let request = ChatRequest::new(vec![ChatMessage::user("hello")]);

    let Err(error) = client.stream_chat(request, CancellationToken::new()) else {
        return Err(DeepSeekError::InvalidResponse(
            "expected empty API key to be rejected".to_string(),
        ));
    };

    assert!(matches!(error, DeepSeekError::MissingApiKey));
    Ok(())
}

#[test_log::test(tokio::test)]
async fn deepseek_client_stream_chat_serializes_request_and_parses_events()
-> Result<(), DeepSeekError> {
    let captured_request_body = Arc::new(Mutex::new(None::<String>));
    let response_body = concat!(
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking\",\"content\":\"answer\",\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"function\":{\"name\":\"echo\",\"arguments\":\"{\\\"value\\\":1}\"}}]},\"finish_reason\":\"content_filter\"}]}\n\n",
        "data: [DONE]\n\n"
    )
    .to_string();
    let (base_url, server) =
        spawn_sse_server(response_body, Arc::clone(&captured_request_body)).await?;
    let expected_base_url = base_url.clone();
    let client = DeepSeekClient::new(DeepSeekConfig::new("secret", base_url, "mock-model"));
    let config = client.config();
    assert_eq!(config.base_url(), expected_base_url);
    assert_eq!(config.model(), "mock-model");

    let tool_definition = ToolDefinition::new(
        "echo",
        "Echo a value",
        serde_json::json!({
            "type": "object",
            "properties": {
                "value": { "type": "integer" }
            }
        }),
    );
    let request = ChatRequest::new(vec![
        ChatMessage::system("system prompt"),
        ChatMessage::user("hello"),
        ChatMessage::assistant_with_tool_calls(
            "assistant",
            vec![ToolCall::new("call-1", "echo", r#"{"value":1}"#)],
        ),
        ChatMessage::tool_result("call-1", "tool output"),
    ])
    .with_tools(vec![tool_definition.clone()])
    .with_model("request-model")
    .with_reasoning_effort("max");

    assert_eq!(request.messages().len(), 4);
    assert_eq!(request.tools().len(), 1);
    assert_eq!(request.model(), Some("request-model"));
    assert_eq!(request.reasoning_effort(), Some("max"));

    let mut stream = client.stream_chat(request, CancellationToken::new())?;
    let mut events = Vec::new();
    while let Some(item) = stream.next().await {
        events.push(item?);
    }

    assert_eq!(
        events,
        vec![
            StreamEvent::Thought("thinking".to_string()),
            StreamEvent::Message("answer".to_string()),
            StreamEvent::ToolCallDelta(ToolCallDelta::new(
                0,
                Some("call-1".to_string()),
                Some("echo".to_string()),
                Some(r#"{"value":1}"#.to_string()),
            )),
            StreamEvent::Finished(FinishReason::Refusal),
        ]
    );

    server
        .await
        .map_err(|error| DeepSeekError::InvalidResponse(error.to_string()))?
        .map_err(DeepSeekError::InvalidResponse)?;

    let request_guard = captured_request_body
        .lock()
        .map_err(|error| DeepSeekError::InvalidResponse(error.to_string()))?;
    let request_body = request_guard
        .as_ref()
        .ok_or_else(|| DeepSeekError::InvalidResponse("missing request body".to_string()))?;
    let request_json: serde_json::Value = serde_json::from_str(request_body)?;

    assert_eq!(request_json["model"], "request-model");
    assert_eq!(request_json["reasoning_effort"], "max");
    assert_eq!(request_json["stream"], serde_json::json!(true));
    assert_eq!(request_json["messages"][0]["role"], "system");
    assert_eq!(request_json["messages"][1]["role"], "user");
    assert_eq!(
        request_json["messages"][2]["tool_calls"][0]["function"]["name"],
        "echo"
    );
    assert_eq!(request_json["messages"][3]["role"], "tool");
    assert_eq!(request_json["messages"][3]["tool_call_id"], "call-1");
    assert_eq!(request_json["tools"][0]["type"], "function");
    assert_eq!(request_json["tools"][0]["function"]["name"], "echo");

    Ok(())
}

#[test_log::test(tokio::test)]
async fn deepseek_client_reports_stream_end_without_finish_reason() -> Result<(), DeepSeekError> {
    let captured_request_body = Arc::new(Mutex::new(None::<String>));
    let response_body = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n",
        "data: [DONE]\n\n"
    )
    .to_string();
    let (base_url, server) =
        spawn_sse_server(response_body, Arc::clone(&captured_request_body)).await?;
    let client = DeepSeekClient::new(DeepSeekConfig::new("secret", base_url, "mock-model"));
    let mut stream = client.stream_chat(
        ChatRequest::new(vec![ChatMessage::user("hello")]),
        CancellationToken::new(),
    )?;

    let first_event = stream.next().await.ok_or_else(|| {
        DeepSeekError::InvalidResponse("expected message event before stream error".to_string())
    })??;
    assert_eq!(first_event, StreamEvent::Message("partial".to_string()));

    let Err(error) = stream
        .next()
        .await
        .ok_or_else(|| DeepSeekError::InvalidResponse("expected stream error".to_string()))?
    else {
        return Err(DeepSeekError::InvalidResponse(
            "expected missing finish reason to fail".to_string(),
        ));
    };
    assert!(matches!(error, DeepSeekError::InvalidResponse(_)));
    assert_eq!(
        error.to_string(),
        "invalid DeepSeek response: stream ended before a finish reason was received"
    );

    server
        .await
        .map_err(|error| DeepSeekError::InvalidResponse(error.to_string()))?
        .map_err(DeepSeekError::InvalidResponse)?;

    let request_guard = captured_request_body
        .lock()
        .map_err(|error| DeepSeekError::InvalidResponse(error.to_string()))?;
    assert!(request_guard.as_ref().is_some());

    Ok(())
}

#[test_log::test]
fn deepseek_error_from_event_source_error_uses_transport_variant() {
    let error = DeepSeekError::from(reqwest_eventsource::Error::StreamEnded);

    assert!(matches!(error, DeepSeekError::Transport(_)));
    assert_eq!(
        error.to_string(),
        "`DeepSeek` SSE transport error: Stream ended"
    );
}

#[test_log::test]
fn is_retryable_transport_error_returns_false_for_non_transport_errors() {
    assert!(!is_retryable_transport_error(
        &reqwest_eventsource::Error::StreamEnded
    ));
    let Err(utf8_error) = String::from_utf8(vec![0xFF_u8]) else {
        return;
    };
    assert!(!is_retryable_transport_error(
        &reqwest_eventsource::Error::Utf8(utf8_error)
    ));
}

#[test_log::test(tokio::test)]
async fn retries_stream_on_connection_drop_before_events() -> Result<(), DeepSeekError> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|e| DeepSeekError::InvalidResponse(e.to_string()))?;
    let addr = listener
        .local_addr()
        .map_err(|e| DeepSeekError::InvalidResponse(e.to_string()))?;

    let response_body = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n"
    )
    .to_string();

    let server = tokio::spawn(async move {
        let _ = listener.accept().await.map_err(|e| e.to_string())?;

        let (mut socket, _) = listener.accept().await.map_err(|e| e.to_string())?;
        let mut buf = [0_u8; 4096];
        let mut received = Vec::new();
        loop {
            let n = socket.read(&mut buf).await.map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            received.extend_from_slice(&buf[..n]);
            if received.windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
        }
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            response_body.len(),
            response_body
        );
        socket
            .write_all(response.as_bytes())
            .await
            .map_err(|e| e.to_string())?;
        socket.shutdown().await.map_err(|e| e.to_string())?;
        Ok::<(), String>(())
    });

    let client = DeepSeekClient::new(DeepSeekConfig::new(
        "secret",
        format!("http://{addr}"),
        "mock-model",
    ));
    let mut stream = client.stream_chat(
        ChatRequest::new(vec![ChatMessage::user("hello")]),
        CancellationToken::new(),
    )?;

    let event = stream
        .next()
        .await
        .ok_or_else(|| DeepSeekError::InvalidResponse("expected message event".to_string()))??;
    assert_eq!(event, StreamEvent::Message("hello".to_string()));

    let event = stream
        .next()
        .await
        .ok_or_else(|| DeepSeekError::InvalidResponse("expected finish event".to_string()))??;
    assert_eq!(event, StreamEvent::Finished(FinishReason::EndTurn));

    server
        .await
        .map_err(|e| DeepSeekError::InvalidResponse(e.to_string()))?
        .map_err(DeepSeekError::InvalidResponse)?;

    Ok(())
}

#[test_log::test]
fn deepseek_config_rejects_blank_api_key_from_environment() {
    let environment = FakeEnvironment {
        values: BTreeMap::from([("DEEPSEEK_API_KEY", "   ")]),
    };

    assert!(matches!(
        DeepSeekConfig::from_environment(&environment),
        Err(DeepSeekError::MissingApiKey)
    ));
}

#[test_log::test]
fn finish_reason_length_maps_to_max_tokens() {
    assert_eq!(FinishReason::from_api("length"), FinishReason::MaxTokens);
}

#[test_log::test]
fn finish_reason_tool_calls_maps_to_tool_calls() {
    assert_eq!(
        FinishReason::from_api("tool_calls"),
        FinishReason::ToolCalls
    );
}

#[test_log::test]
fn finish_reason_content_filter_maps_to_refusal() {
    assert_eq!(
        FinishReason::from_api("content_filter"),
        FinishReason::Refusal
    );
}

#[test_log::test]
fn finish_reason_stop_maps_to_end_turn() {
    assert_eq!(FinishReason::from_api("stop"), FinishReason::EndTurn);
}

#[test_log::test]
fn finish_reason_unknown_maps_to_other() {
    assert!(matches!(
        FinishReason::from_api("some_unknown_reason"),
        FinishReason::Other(_)
    ));
}

#[test_log::test]
fn parse_chunk_with_invalid_json_fails() {
    assert!(matches!(
        parse_chat_completion_chunk("not json"),
        Err(DeepSeekError::Json(_))
    ));
}

#[test_log::test]
fn chat_request_model_and_reasoning_effort_accessors() {
    let request = ChatRequest::new(vec![ChatMessage::user("hello")])
        .with_model("custom-model")
        .with_reasoning_effort("medium");

    assert_eq!(request.model(), Some("custom-model"));
    assert_eq!(request.reasoning_effort(), Some("medium"));
}

#[test_log::test]
fn chat_request_tool_call_id_accessor() {
    let result = ChatMessage::tool_result("call-42", "done");

    assert_eq!(result.tool_call_id(), Some("call-42"));
    assert_eq!(result.role(), MessageRole::Tool);
    assert_eq!(result.content(), "done");
}

#[test_log::test]
fn parse_chunk_with_tool_call_no_function() -> Result<(), DeepSeekError> {
    let fixture = r#"
    {
      "choices": [
        {
          "delta": {
            "tool_calls": [
              {
                "index": 1,
                "id": "call-2"
              }
            ]
          },
          "finish_reason": "tool_calls"
        }
      ]
    }
    "#;

    let updates = parse_chat_completion_chunk(fixture)?;
    let StreamEvent::ToolCallDelta(delta) = &updates[0] else {
        return Err(DeepSeekError::InvalidResponse(
            "expected tool call delta".to_string(),
        ));
    };
    assert_eq!(delta.index(), 1);
    assert_eq!(delta.id(), Some("call-2"));
    assert_eq!(delta.name(), None);
    assert_eq!(delta.arguments(), None);
    assert_eq!(updates[1], StreamEvent::Finished(FinishReason::ToolCalls));
    Ok(())
}

#[test_log::test]
fn parse_chunk_with_no_tool_calls_and_empty_delta() -> Result<(), DeepSeekError> {
    let fixture = r#"
    {
      "choices": [
        {
          "delta": {},
          "finish_reason": "stop"
        }
      ]
    }
    "#;

    let updates = parse_chat_completion_chunk(fixture)?;
    assert_eq!(updates, vec![StreamEvent::Finished(FinishReason::EndTurn)]);
    Ok(())
}

#[test_log::test]
fn tool_call_accessors_expose_fields() {
    let tool_call = ToolCall::new("call-1", "read_file", r#"{"path":"src/lib.rs"}"#);

    assert_eq!(tool_call.id(), "call-1");
    assert_eq!(tool_call.name(), "read_file");
    assert_eq!(tool_call.arguments(), r#"{"path":"src/lib.rs"}"#);
}

#[test_log::test]
fn tool_call_delta_constructs_and_accessors() {
    let delta = ToolCallDelta::new(
        0,
        Some("call-1".to_string()),
        Some("read_file".to_string()),
        Some(r#"{"path":"file"}"#.to_string()),
    );

    assert_eq!(delta.index(), 0);
    assert_eq!(delta.id(), Some("call-1"));
    assert_eq!(delta.name(), Some("read_file"));
    assert_eq!(delta.arguments(), Some(r#"{"path":"file"}"#));
}

#[test_log::test]
fn tool_call_delta_with_empty_fields() {
    let delta = ToolCallDelta::new(0, None, None, None);

    assert_eq!(delta.index(), 0);
    assert_eq!(delta.id(), None);
    assert_eq!(delta.name(), None);
    assert_eq!(delta.arguments(), None);
}

#[test_log::test]
fn chat_message_content_and_tool_calls_accessors() {
    let tool_calls = vec![ToolCall::new("c1", "n", "{}")];
    let assistant = ChatMessage::assistant_with_tool_calls("text", tool_calls);

    assert!(!assistant.tool_calls().is_empty());
    assert_eq!(assistant.tool_calls()[0].name(), "n");
    assert_eq!(assistant.tool_call_id(), None);
}

#[test_log::test]
fn chat_message_role_system_and_user() {
    assert_eq!(ChatMessage::system("s").role(), MessageRole::System);
    assert_eq!(ChatMessage::user("u").role(), MessageRole::User);
}

#[test_log::test]
fn chat_request_empty_by_default() {
    let request = ChatRequest::new(vec![]);

    assert!(request.messages().is_empty());
    assert!(request.tools().is_empty());
    assert_eq!(request.model(), None);
    assert_eq!(request.reasoning_effort(), None);
}

#[test_log::test]
fn tool_definition_accessors_expose_fields() {
    let definition = ToolDefinition::new(
        "echo",
        "Echo a value",
        serde_json::json!({
            "type": "object",
            "properties": {
                "value": { "type": "integer" }
            }
        }),
    );

    assert_eq!(definition.name(), "echo");
    assert_eq!(definition.description(), "Echo a value");
    assert_eq!(
        definition.parameters(),
        &serde_json::json!({
            "type": "object",
            "properties": {
                "value": { "type": "integer" }
            }
        })
    );
}

#[test_log::test(tokio::test)]
async fn deepseek_client_reports_parse_errors_from_sse_payloads() -> Result<(), DeepSeekError> {
    let captured_request_body = Arc::new(Mutex::new(None::<String>));
    let response_body = "data: not-json\n\n".to_string();
    let (base_url, server) =
        spawn_sse_server(response_body, Arc::clone(&captured_request_body)).await?;
    let client = DeepSeekClient::new(DeepSeekConfig::new("secret", base_url, "mock-model"));
    let mut stream = client.stream_chat(
        ChatRequest::new(vec![ChatMessage::user("hello")]),
        CancellationToken::new(),
    )?;

    let Err(error) = stream
        .next()
        .await
        .ok_or_else(|| DeepSeekError::InvalidResponse("expected parse error".to_string()))?
    else {
        return Err(DeepSeekError::InvalidResponse(
            "expected invalid JSON to fail".to_string(),
        ));
    };
    assert!(matches!(error, DeepSeekError::Json(_)));

    server
        .await
        .map_err(|error| DeepSeekError::InvalidResponse(error.to_string()))?
        .map_err(DeepSeekError::InvalidResponse)?;

    let request_guard = captured_request_body
        .lock()
        .map_err(|error| DeepSeekError::InvalidResponse(error.to_string()))?;
    assert!(request_guard.as_ref().is_some());

    Ok(())
}
