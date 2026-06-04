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
        /// Tool result message.
        Tool,
    }

    impl MessageRole {
        fn as_str(self) -> &'static str {
            match self {
                Self::System => "system",
                Self::User => "user",
                Self::Assistant => "assistant",
                Self::Tool => "tool",
            }
        }
    }

    /// A single chat message passed to `DeepSeek`.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ChatMessage {
        role: MessageRole,
        content: String,
        tool_calls: Vec<ToolCall>,
        tool_call_id: Option<String>,
    }

    impl ChatMessage {
        /// Create a system message.
        #[must_use]
        pub fn system(content: impl Into<String>) -> Self {
            Self {
                role: MessageRole::System,
                content: content.into(),
                tool_calls: Vec::new(),
                tool_call_id: None,
            }
        }

        /// Create a user message.
        #[must_use]
        pub fn user(content: impl Into<String>) -> Self {
            Self {
                role: MessageRole::User,
                content: content.into(),
                tool_calls: Vec::new(),
                tool_call_id: None,
            }
        }

        /// Create an assistant message.
        #[must_use]
        pub fn assistant(content: impl Into<String>) -> Self {
            Self {
                role: MessageRole::Assistant,
                content: content.into(),
                tool_calls: Vec::new(),
                tool_call_id: None,
            }
        }

        /// Create an assistant message that requested tool calls.
        #[must_use]
        pub fn assistant_with_tool_calls(
            content: impl Into<String>,
            tool_calls: Vec<ToolCall>,
        ) -> Self {
            Self {
                role: MessageRole::Assistant,
                content: content.into(),
                tool_calls,
                tool_call_id: None,
            }
        }

        /// Create a tool result message.
        #[must_use]
        pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
            Self {
                role: MessageRole::Tool,
                content: content.into(),
                tool_calls: Vec::new(),
                tool_call_id: Some(tool_call_id.into()),
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

        /// Return assistant tool calls attached to this message.
        #[must_use]
        pub fn tool_calls(&self) -> &[ToolCall] {
            &self.tool_calls
        }

        /// Return the tool call id for a tool result message.
        #[must_use]
        pub fn tool_call_id(&self) -> Option<&str> {
            self.tool_call_id.as_deref()
        }
    }

    #[derive(Debug, Serialize)]
    struct WireMessage {
        role: String,
        content: String,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<WireToolCall>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_call_id: Option<String>,
    }

    impl From<&ChatMessage> for WireMessage {
        fn from(message: &ChatMessage) -> Self {
            Self {
                role: message.role.as_str().to_string(),
                content: message.content.clone(),
                tool_calls: message.tool_calls.iter().map(WireToolCall::from).collect(),
                tool_call_id: message.tool_call_id.clone(),
            }
        }
    }

    /// A callable function advertised to `DeepSeek`.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ToolDefinition {
        name: String,
        description: String,
        parameters: serde_json::Value,
    }

    impl ToolDefinition {
        /// Create a tool definition.
        #[must_use]
        pub fn new(
            name: impl Into<String>,
            description: impl Into<String>,
            parameters: serde_json::Value,
        ) -> Self {
            Self {
                name: name.into(),
                description: description.into(),
                parameters,
            }
        }

        /// Return the function name.
        #[must_use]
        pub fn name(&self) -> &str {
            &self.name
        }

        /// Return the function description.
        #[must_use]
        pub fn description(&self) -> &str {
            &self.description
        }

        /// Return the JSON schema parameters.
        #[must_use]
        pub fn parameters(&self) -> &serde_json::Value {
            &self.parameters
        }
    }

    /// A complete tool call requested by the model.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ToolCall {
        id: String,
        name: String,
        arguments: String,
    }

    impl ToolCall {
        /// Create a complete tool call.
        #[must_use]
        pub fn new(
            id: impl Into<String>,
            name: impl Into<String>,
            arguments: impl Into<String>,
        ) -> Self {
            Self {
                id: id.into(),
                name: name.into(),
                arguments: arguments.into(),
            }
        }

        /// Return the provider tool-call id.
        #[must_use]
        pub fn id(&self) -> &str {
            &self.id
        }

        /// Return the function name.
        #[must_use]
        pub fn name(&self) -> &str {
            &self.name
        }

        /// Return the raw JSON argument string.
        #[must_use]
        pub fn arguments(&self) -> &str {
            &self.arguments
        }
    }

    #[derive(Debug, Serialize)]
    struct WireToolDefinition {
        r#type: &'static str,
        function: WireToolFunctionDefinition,
    }

    impl From<&ToolDefinition> for WireToolDefinition {
        fn from(definition: &ToolDefinition) -> Self {
            Self {
                r#type: "function",
                function: WireToolFunctionDefinition {
                    name: definition.name.clone(),
                    description: definition.description.clone(),
                    parameters: definition.parameters.clone(),
                },
            }
        }
    }

    #[derive(Debug, Serialize)]
    struct WireToolFunctionDefinition {
        name: String,
        description: String,
        parameters: serde_json::Value,
    }

    #[derive(Debug, Serialize)]
    struct WireToolCall {
        id: String,
        r#type: &'static str,
        function: WireToolCallFunction,
    }

    impl From<&ToolCall> for WireToolCall {
        fn from(call: &ToolCall) -> Self {
            Self {
                id: call.id.clone(),
                r#type: "function",
                function: WireToolCallFunction {
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                },
            }
        }
    }

    #[derive(Debug, Serialize)]
    struct WireToolCallFunction {
        name: String,
        arguments: String,
    }

    /// A chat-completions request that can be streamed from `DeepSeek`.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ChatRequest {
        messages: Vec<ChatMessage>,
        tools: Vec<ToolDefinition>,
        model: Option<String>,
        reasoning_effort: Option<String>,
    }

    impl ChatRequest {
        /// Create a new request from a list of chat messages.
        #[must_use]
        pub fn new(messages: Vec<ChatMessage>) -> Self {
            Self {
                messages,
                tools: Vec::new(),
                model: None,
                reasoning_effort: None,
            }
        }

        /// Attach tool definitions to the request.
        #[must_use]
        pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
            self.tools = tools;
            self
        }

        /// Override the configured model for this request.
        #[must_use]
        pub fn with_model(mut self, model: impl Into<String>) -> Self {
            self.model = Some(model.into());
            self
        }

        /// Set the model reasoning effort for this request.
        #[must_use]
        pub fn with_reasoning_effort(mut self, reasoning_effort: impl Into<String>) -> Self {
            self.reasoning_effort = Some(reasoning_effort.into());
            self
        }

        fn into_parts(
            self,
        ) -> (
            Vec<ChatMessage>,
            Vec<ToolDefinition>,
            Option<String>,
            Option<String>,
        ) {
            (self.messages, self.tools, self.model, self.reasoning_effort)
        }

        /// Return the request messages.
        #[must_use]
        pub fn messages(&self) -> &[ChatMessage] {
            &self.messages
        }

        /// Return request tool definitions.
        #[must_use]
        pub fn tools(&self) -> &[ToolDefinition] {
            &self.tools
        }

        /// Return the request model override.
        #[must_use]
        pub fn model(&self) -> Option<&str> {
            self.model.as_deref()
        }

        /// Return the request reasoning effort.
        #[must_use]
        pub fn reasoning_effort(&self) -> Option<&str> {
            self.reasoning_effort.as_deref()
        }
    }

    /// A normalized update emitted while streaming a `DeepSeek` response.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum StreamEvent {
        /// A chunk of model reasoning.
        Thought(String),
        /// A chunk of user-facing assistant text.
        Message(String),
        /// A streamed tool-call delta.
        ToolCallDelta(ToolCallDelta),
        /// The model reported a terminal finish reason.
        Finished(FinishReason),
    }

    /// A partial streamed tool call.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments: Option<String>,
    }

    impl ToolCallDelta {
        /// Create a streamed tool-call delta.
        #[must_use]
        pub fn new(
            index: usize,
            id: Option<String>,
            name: Option<String>,
            arguments: Option<String>,
        ) -> Self {
            Self {
                index,
                id,
                name,
                arguments,
            }
        }

        /// Return the streamed tool-call index.
        #[must_use]
        pub fn index(&self) -> usize {
            self.index
        }

        /// Return the provider id delta, if present.
        #[must_use]
        pub fn id(&self) -> Option<&str> {
            self.id.as_deref()
        }

        /// Return the function name delta, if present.
        #[must_use]
        pub fn name(&self) -> Option<&str> {
            self.name.as_deref()
        }

        /// Return the argument delta, if present.
        #[must_use]
        pub fn arguments(&self) -> Option<&str> {
            self.arguments.as_deref()
        }
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

    enum StreamAttemptOutcome {
        Complete,
        Cancelled,
        ShouldRetry,
    }

    async fn run_stream_attempt(
        mut event_source: EventSource,
        tx: &mpsc::UnboundedSender<Result<StreamEvent, DeepSeekError>>,
        cancellation_token: &CancellationToken,
        attempt: u32,
        max_retries: u32,
    ) -> StreamAttemptOutcome {
        let mut saw_finish = false;
        let mut events_sent: u32 = 0;

        loop {
            let event = tokio::select! {
                () = cancellation_token.cancelled() => return StreamAttemptOutcome::Cancelled,
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
                                events_sent += 1;
                                if tx.send(Ok(update)).is_err() {
                                    return StreamAttemptOutcome::Cancelled;
                                }
                            }
                        }
                        Err(error) => {
                            let _ = tx.send(Err(error));
                            return StreamAttemptOutcome::Cancelled;
                        }
                    }
                }
                Err(error) => {
                    if events_sent == 0
                        && attempt < max_retries
                        && is_retryable_transport_error(&error)
                    {
                        return StreamAttemptOutcome::ShouldRetry;
                    }
                    let _ = tx.send(Err(error.into()));
                    return StreamAttemptOutcome::Cancelled;
                }
            }
        }

        if !saw_finish && !cancellation_token.is_cancelled() {
            let _ = tx.send(Err(DeepSeekError::InvalidResponse(
                "stream ended before a finish reason was received".to_string(),
            )));
        }
        StreamAttemptOutcome::Complete
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

            let (messages, tools, model, reasoning_effort) = request.into_parts();
            let body = ChatCompletionRequest {
                model: model.unwrap_or_else(|| self.config.model.clone()),
                messages: messages
                    .into_iter()
                    .map(|message| WireMessage::from(&message))
                    .collect(),
                tools: tools.iter().map(WireToolDefinition::from).collect(),
                stream: true,
                reasoning_effort,
            };

            let http = self.http.clone();
            let url = format!(
                "{}/chat/completions",
                self.config.base_url.trim_end_matches('/')
            );
            let api_key = self.config.api_key.clone();

            let (tx, rx) = mpsc::unbounded_channel::<Result<StreamEvent, DeepSeekError>>();

            tokio::spawn(async move {
                const MAX_RETRIES: u32 = 3;

                for attempt in 0..=MAX_RETRIES {
                    if attempt > 0 {
                        let delay_ms = 100u64 * (1u64 << (attempt - 1));
                        tracing::warn!(
                            attempt,
                            delay_ms,
                            "retrying DeepSeek SSE stream after retryable transport error"
                        );
                        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                    }

                    let request = http.post(&url).bearer_auth(&api_key).json(&body);

                    let event_source = match EventSource::new(request) {
                        Ok(es) => es,
                        Err(error) => {
                            let _ = tx.send(Err(DeepSeekError::from(error)));
                            return;
                        }
                    };

                    if !matches!(
                        run_stream_attempt(
                            event_source,
                            &tx,
                            &cancellation_token,
                            attempt,
                            MAX_RETRIES
                        )
                        .await,
                        StreamAttemptOutcome::ShouldRetry
                    ) {
                        return;
                    }
                }
            });

            Ok(stream::unfold(rx, |mut rx| async move {
                rx.recv().await.map(|item| (item, rx))
            })
            .boxed())
        }
    }

    /// Returns true when a transport error is safe to retry before any events are emitted.
    ///
    /// Only `Transport`-variant errors carrying network-level conditions qualify — these
    /// arise from stale pooled connections, server-side shutdowns, and TCP disconnects.
    /// Parse errors, status errors, and redirect loops are not retryable.
    ///
    /// Two broad categories are accepted:
    /// - `is_connect()` / `is_request()`: server dropped the connection before any HTTP
    ///   response arrived (e.g. `hyper::Error(IncompleteMessage)` on a stale pool reuse).
    /// - Body/decode errors whose source chain contains a retryable IO kind
    ///   (`BrokenPipe`, `ConnectionReset`, `UnexpectedEof`, `ConnectionAborted`): TCP dropped
    ///   mid-stream, which is safe to restart only while `events_sent == 0` (guarded by
    ///   the caller).
    fn is_retryable_transport_error(error: &reqwest_eventsource::Error) -> bool {
        use std::error::Error as StdError;

        let reqwest_eventsource::Error::Transport(ref reqwest_error) = *error else {
            return false;
        };

        // Connection and request-level failures mean no HTTP response was received at
        // all — always safe to retry (no risk of duplicate events).
        if reqwest_error.is_connect() || reqwest_error.is_request() {
            return true;
        }

        // Body or decode errors can arise from a mid-stream TCP disconnection.  Walk
        // the source chain for a retryable IO condition.
        let mut current: Option<&(dyn StdError + 'static)> =
            (reqwest_error as &dyn StdError).source();
        while let Some(err) = current {
            if let Some(io) = err.downcast_ref::<std::io::Error>() {
                return matches!(
                    io.kind(),
                    std::io::ErrorKind::BrokenPipe
                        | std::io::ErrorKind::ConnectionReset
                        | std::io::ErrorKind::UnexpectedEof
                        | std::io::ErrorKind::ConnectionAborted
                );
            }
            current = err.source();
        }
        false
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
        #[serde(skip_serializing_if = "Vec::is_empty")]
        tools: Vec<WireToolDefinition>,
        stream: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_effort: Option<String>,
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
        #[serde(default)]
        tool_calls: Vec<ChatToolCallDelta>,
    }

    #[derive(Debug, Deserialize)]
    struct ChatToolCallDelta {
        index: usize,
        #[serde(default)]
        id: Option<String>,
        function: Option<ChatToolCallFunctionDelta>,
    }

    #[derive(Debug, Default, Deserialize)]
    struct ChatToolCallFunctionDelta {
        #[serde(default)]
        name: Option<String>,
        #[serde(default)]
        arguments: Option<String>,
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

        for tool_call in choice.delta.tool_calls {
            updates.push(StreamEvent::ToolCallDelta(ToolCallDelta::new(
                tool_call.index,
                tool_call.id,
                tool_call
                    .function
                    .as_ref()
                    .and_then(|function| function.name.clone()),
                tool_call.function.and_then(|function| function.arguments),
            )));
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
            FinishReason, LlmClient, MessageRole, StreamEvent, ToolCall, ToolCallDelta,
            ToolDefinition, is_retryable_transport_error, parse_chat_completion_chunk,
        };

        use futures_util::StreamExt;
        use std::collections::BTreeMap;
        use std::sync::{Arc, Mutex};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;
        use tokio_util::sync::CancellationToken;

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
                    if let Some(index) = request.windows(4).position(|window| window == b"\r\n\r\n")
                    {
                        break index + 4;
                    }
                };

                let headers = String::from_utf8(request[..header_end].to_vec())
                    .map_err(|error| error.to_string())?;
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
            let request_body = request_guard.as_ref().ok_or_else(|| {
                DeepSeekError::InvalidResponse("missing request body".to_string())
            })?;
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
        async fn deepseek_client_reports_stream_end_without_finish_reason()
        -> Result<(), DeepSeekError> {
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
                DeepSeekError::InvalidResponse(
                    "expected message event before stream error".to_string(),
                )
            })??;
            assert_eq!(first_event, StreamEvent::Message("partial".to_string()));

            let Err(error) = stream.next().await.ok_or_else(|| {
                DeepSeekError::InvalidResponse("expected stream error".to_string())
            })?
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
            // 0xFF is not valid UTF-8 so from_utf8 always returns Err here.
            let Err(utf8_error) = String::from_utf8(vec![0xFF_u8]) else {
                return; // unreachable: 0xFF is invalid UTF-8
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
                // First connection: accept then immediately close to simulate a stale
                // pooled connection that the server has already shut down.
                let _ = listener.accept().await.map_err(|e| e.to_string())?;

                // Second connection: serve a valid SSE response.
                let (mut socket, _) = listener.accept().await.map_err(|e| e.to_string())?;
                let mut buf = [0u8; 4096];
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

            let event = stream.next().await.ok_or_else(|| {
                DeepSeekError::InvalidResponse("expected message event".to_string())
            })??;
            assert_eq!(event, StreamEvent::Message("hello".to_string()));

            let event = stream.next().await.ok_or_else(|| {
                DeepSeekError::InvalidResponse("expected finish event".to_string())
            })??;
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
        async fn deepseek_client_reports_parse_errors_from_sse_payloads()
        -> Result<(), DeepSeekError> {
            let captured_request_body = Arc::new(Mutex::new(None::<String>));
            let response_body = "data: not-json\n\n".to_string();
            let (base_url, server) =
                spawn_sse_server(response_body, Arc::clone(&captured_request_body)).await?;
            let client = DeepSeekClient::new(DeepSeekConfig::new("secret", base_url, "mock-model"));
            let mut stream = client.stream_chat(
                ChatRequest::new(vec![ChatMessage::user("hello")]),
                CancellationToken::new(),
            )?;

            let Err(error) = stream.next().await.ok_or_else(|| {
                DeepSeekError::InvalidResponse("expected parse error".to_string())
            })?
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
    }
}
