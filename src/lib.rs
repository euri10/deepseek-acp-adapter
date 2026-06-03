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

/// `DeepSeek` client primitives and streaming SSE adapter.
pub mod deepseek {
    use std::env;

    use futures_util::{
        StreamExt,
        stream::{self, BoxStream},
    };
    use reqwest::Client as HttpClient;
    use reqwest_eventsource::{Event, EventSource};
    use serde::{Deserialize, Serialize};
    use thiserror::Error;
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    /// Errors returned by `DeepSeek` configuration, request setup, or SSE parsing.
    #[derive(Debug, Error)]
    pub enum DeepSeekError {
        /// The `DEEPSEEK_API_KEY` environment variable was not set or was empty.
        #[error("DEEPSEEK_API_KEY is not set")]
        MissingApiKey,
        /// The request could not be cloned for SSE streaming.
        #[error("failed to clone DeepSeek streaming request: {0}")]
        RequestClone(#[from] reqwest_eventsource::CannotCloneRequestError),
        /// The SSE transport failed while streaming events.
        #[error("`DeepSeek` SSE transport error: {0}")]
        Transport(Box<reqwest_eventsource::Error>),
        /// The model returned a chunk that could not be decoded.
        #[error("invalid DeepSeek response: {0}")]
        InvalidResponse(String),
        /// The model returned malformed JSON.
        #[error("failed to parse DeepSeek response: {0}")]
        Json(#[from] serde_json::Error),
    }

    impl From<reqwest_eventsource::Error> for DeepSeekError {
        fn from(error: reqwest_eventsource::Error) -> Self {
            Self::Transport(Box::new(error))
        }
    }

    /// Conversation role encoded in a chat-completions request.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
    #[serde(rename_all = "lowercase")]
    pub enum MessageRole {
        /// System instruction message.
        System,
        /// User input message.
        User,
        /// Assistant continuation message.
        Assistant,
    }

    impl MessageRole {
        fn as_str(self) -> &'static str {
            match self {
                Self::System => "system",
                Self::User => "user",
                Self::Assistant => "assistant",
            }
        }
    }

    /// A single chat message passed to `DeepSeek`.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ChatMessage {
        role: MessageRole,
        content: String,
    }

    impl ChatMessage {
        /// Create a system message.
        #[must_use]
        pub fn system(content: impl Into<String>) -> Self {
            Self {
                role: MessageRole::System,
                content: content.into(),
            }
        }

        /// Create a user message.
        #[must_use]
        pub fn user(content: impl Into<String>) -> Self {
            Self {
                role: MessageRole::User,
                content: content.into(),
            }
        }

        /// Create an assistant message.
        #[must_use]
        pub fn assistant(content: impl Into<String>) -> Self {
            Self {
                role: MessageRole::Assistant,
                content: content.into(),
            }
        }

        /// Return the message role.
        #[must_use]
        pub fn role(&self) -> MessageRole {
            self.role
        }

        /// Return the message content.
        #[must_use]
        pub fn content(&self) -> &str {
            &self.content
        }
    }

    #[derive(Debug, Serialize)]
    struct WireMessage {
        role: String,
        content: String,
    }

    impl From<&ChatMessage> for WireMessage {
        fn from(message: &ChatMessage) -> Self {
            Self {
                role: message.role.as_str().to_string(),
                content: message.content.clone(),
            }
        }
    }

    /// A chat-completions request that can be streamed from `DeepSeek`.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ChatRequest {
        messages: Vec<ChatMessage>,
    }

    impl ChatRequest {
        /// Create a new request from a list of chat messages.
        #[must_use]
        pub fn new(messages: Vec<ChatMessage>) -> Self {
            Self { messages }
        }

        fn into_messages(self) -> Vec<ChatMessage> {
            self.messages
        }

        /// Return the request messages.
        #[must_use]
        pub fn messages(&self) -> &[ChatMessage] {
            &self.messages
        }
    }

    /// A normalized update emitted while streaming a `DeepSeek` response.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum StreamEvent {
        /// A chunk of model reasoning.
        Thought(String),
        /// A chunk of user-facing assistant text.
        Message(String),
        /// The model reported a terminal finish reason.
        Finished(FinishReason),
    }

    /// Terminal finish reasons returned by `DeepSeek`.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum FinishReason {
        /// The turn ended normally.
        EndTurn,
        /// The model hit the token limit.
        MaxTokens,
        /// The model produced a tool call.
        ToolCalls,
        /// The model refused to continue.
        Refusal,
        /// Any other provider-specific finish reason.
        Other(String),
    }

    impl FinishReason {
        fn from_api(value: &str) -> Self {
            match value {
                "stop" => Self::EndTurn,
                "length" => Self::MaxTokens,
                "tool_calls" => Self::ToolCalls,
                "content_filter" => Self::Refusal,
                other => Self::Other(other.to_string()),
            }
        }
    }

    /// Configuration for the `DeepSeek` client.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct DeepSeekConfig {
        api_key: String,
        base_url: String,
        model: String,
    }

    impl DeepSeekConfig {
        /// Default `DeepSeek` OpenAI-compatible base URL.
        pub const DEFAULT_BASE_URL: &str = "https://api.deepseek.com";
        /// Default model used by the adapter.
        pub const DEFAULT_MODEL: &str = "deepseek-v4-pro";

        /// Create a config from explicit values.
        #[must_use]
        pub fn new(
            api_key: impl Into<String>,
            base_url: impl Into<String>,
            model: impl Into<String>,
        ) -> Self {
            Self {
                api_key: api_key.into(),
                base_url: base_url.into(),
                model: model.into(),
            }
        }

        /// Load config from `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, and `DEEPSEEK_MODEL`.
        ///
        /// # Errors
        ///
        /// Returns `MissingApiKey` when the API key is absent or empty.
        pub fn from_env() -> Result<Self, DeepSeekError> {
            Self::from_environment(&SystemEnvironment)
        }

        fn from_environment(env: &impl Environment) -> Result<Self, DeepSeekError> {
            let api_key = env
                .var("DEEPSEEK_API_KEY")
                .ok_or(DeepSeekError::MissingApiKey)?;

            let api_key = api_key.trim().to_string();
            if api_key.is_empty() {
                return Err(DeepSeekError::MissingApiKey);
            }

            let base_url = env
                .var("DEEPSEEK_BASE_URL")
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| Self::DEFAULT_BASE_URL.to_string());

            let model = env
                .var("DEEPSEEK_MODEL")
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| Self::DEFAULT_MODEL.to_string());

            Ok(Self {
                api_key,
                base_url,
                model,
            })
        }

        /// Return the configured base URL.
        #[must_use]
        pub fn base_url(&self) -> &str {
            &self.base_url
        }

        /// Return the configured model name.
        #[must_use]
        pub fn model(&self) -> &str {
            &self.model
        }
    }

    /// A `DeepSeek` chat-completions client.
    #[derive(Debug, Clone)]
    pub struct DeepSeekClient {
        http: HttpClient,
        config: DeepSeekConfig,
    }

    impl DeepSeekClient {
        /// Build a client from explicit configuration.
        #[must_use]
        pub fn new(config: DeepSeekConfig) -> Self {
            Self {
                http: HttpClient::new(),
                config,
            }
        }

        /// Build a client from process environment.
        ///
        /// # Errors
        ///
        /// Returns `MissingApiKey` when the required key is absent or empty.
        pub fn from_env() -> Result<Self, DeepSeekError> {
            Ok(Self::new(DeepSeekConfig::from_env()?))
        }

        /// Return the client configuration.
        #[must_use]
        pub fn config(&self) -> &DeepSeekConfig {
            &self.config
        }
    }

    /// A client abstraction for streaming chat-completions turns.
    pub trait LlmClient: Send + Sync {
        /// Stream a turn and yield normalized reasoning, text, and terminal events.
        ///
        /// The stream should stop promptly when `cancellation_token` is cancelled.
        ///
        /// # Errors
        ///
        /// Returns an error if the request cannot be constructed or the transport fails.
        fn stream_chat(
            &self,
            request: ChatRequest,
            cancellation_token: CancellationToken,
        ) -> Result<BoxStream<'static, Result<StreamEvent, DeepSeekError>>, DeepSeekError>;
    }

    impl LlmClient for DeepSeekClient {
        fn stream_chat(
            &self,
            request: ChatRequest,
            cancellation_token: CancellationToken,
        ) -> Result<BoxStream<'static, Result<StreamEvent, DeepSeekError>>, DeepSeekError> {
            if self.config.api_key.trim().is_empty() {
                return Err(DeepSeekError::MissingApiKey);
            }

            let body = ChatCompletionRequest {
                model: self.config.model.clone(),
                messages: request
                    .into_messages()
                    .into_iter()
                    .map(|message| WireMessage::from(&message))
                    .collect(),
                stream: true,
            };

            let request = self
                .http
                .post(format!(
                    "{}/chat/completions",
                    self.config.base_url.trim_end_matches('/')
                ))
                .bearer_auth(&self.config.api_key)
                .json(&body);

            let mut event_source = EventSource::new(request)?;
            let (tx, rx) = mpsc::unbounded_channel::<Result<StreamEvent, DeepSeekError>>();

            tokio::spawn(async move {
                let mut saw_finish = false;

                loop {
                    let event = tokio::select! {
                        () = cancellation_token.cancelled() => return,
                        event = event_source.next() => event,
                    };

                    let Some(event) = event else {
                        break;
                    };

                    match event {
                        Ok(Event::Open) => {}
                        Ok(Event::Message(message)) => {
                            if message.data.trim() == "[DONE]" {
                                break;
                            }

                            match parse_chat_completion_chunk(&message.data) {
                                Ok(updates) => {
                                    for update in updates {
                                        if matches!(update, StreamEvent::Finished(_)) {
                                            saw_finish = true;
                                        }

                                        if tx.send(Ok(update)).is_err() {
                                            return;
                                        }
                                    }
                                }
                                Err(error) => {
                                    let _ = tx.send(Err(error));
                                    return;
                                }
                            }
                        }
                        Err(error) => {
                            let _ = tx.send(Err(error.into()));
                            return;
                        }
                    }
                }

                if !saw_finish && !cancellation_token.is_cancelled() {
                    let _ = tx.send(Err(DeepSeekError::InvalidResponse(
                        "stream ended before a finish reason was received".to_string(),
                    )));
                }
            });

            Ok(stream::unfold(rx, |mut rx| async move {
                rx.recv().await.map(|item| (item, rx))
            })
            .boxed())
        }
    }

    trait Environment {
        fn var(&self, key: &str) -> Option<String>;
    }

    struct SystemEnvironment;

    impl Environment for SystemEnvironment {
        fn var(&self, key: &str) -> Option<String> {
            env::var_os(key).and_then(|value| value.into_string().ok())
        }
    }

    #[derive(Debug, Serialize)]
    struct ChatCompletionRequest {
        model: String,
        messages: Vec<WireMessage>,
        stream: bool,
    }

    #[derive(Debug, Deserialize)]
    struct ChatCompletionChunk {
        choices: Vec<ChatChoice>,
    }

    #[derive(Debug, Deserialize)]
    struct ChatChoice {
        delta: ChatDelta,
        #[serde(default)]
        finish_reason: Option<String>,
    }

    #[derive(Debug, Default, Deserialize)]
    struct ChatDelta {
        #[serde(default)]
        reasoning_content: Option<String>,
        #[serde(default)]
        content: Option<String>,
    }

    fn parse_chat_completion_chunk(payload: &str) -> Result<Vec<StreamEvent>, DeepSeekError> {
        let chunk: ChatCompletionChunk = serde_json::from_str(payload)?;
        let Some(choice) = chunk.choices.into_iter().next() else {
            return Err(DeepSeekError::InvalidResponse(
                "chat completion chunk did not include any choices".to_string(),
            ));
        };

        let mut updates = Vec::new();

        if let Some(reasoning) = choice
            .delta
            .reasoning_content
            .filter(|value| !value.is_empty())
        {
            updates.push(StreamEvent::Thought(reasoning));
        }

        if let Some(content) = choice.delta.content.filter(|value| !value.is_empty()) {
            updates.push(StreamEvent::Message(content));
        }

        if let Some(finish_reason) = choice.finish_reason {
            updates.push(StreamEvent::Finished(FinishReason::from_api(
                &finish_reason,
            )));
        }

        Ok(updates)
    }

    #[cfg(test)]
    mod tests {
        use super::{
            ChatMessage, ChatRequest, DeepSeekClient, DeepSeekConfig, DeepSeekError, Environment,
            FinishReason, LlmClient, MessageRole, StreamEvent, parse_chat_completion_chunk,
        };

        use std::collections::BTreeMap;
        use tokio_util::sync::CancellationToken;

        struct FakeEnvironment {
            values: BTreeMap<&'static str, &'static str>,
        }

        impl Environment for FakeEnvironment {
            fn var(&self, key: &str) -> Option<String> {
                self.values.get(key).map(|value| (*value).to_string())
            }
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
    }
}
