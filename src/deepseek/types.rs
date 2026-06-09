use serde::{Deserialize, Serialize};

/// Conversation role encoded in a chat-completions request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

/// A single chat message passed to `DeepSeek`.
///
/// Use constructor helpers such as [`ChatMessage::system`] and
/// [`ChatMessage::tool_result`] to keep role-specific fields consistent.
///
/// # Examples
///
/// ```rust
/// use deepseek_acp_adapter::deepseek::{ChatMessage, MessageRole, ToolCall};
///
/// let assistant = ChatMessage::assistant_with_tool_calls(
///     "I need to inspect a file first.",
///     vec![ToolCall::new("call-1", "read_file", r#"{"path":"src/lib.rs"}"#)],
/// );
///
/// assert_eq!(assistant.role(), MessageRole::Assistant);
/// assert_eq!(assistant.tool_calls()[0].name(), "read_file");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    pub const fn role(&self) -> MessageRole {
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
pub(crate) struct WireMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<WireToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl From<&ChatMessage> for WireMessage {
    fn from(message: &ChatMessage) -> Self {
        // The OpenAI-compatible API expects `content` to be null or omitted
        // when the message has `tool_calls` and no textual content.
        // Sending `"content": ""` can cause 400 Bad Request on some providers.
        let content = if message.role == MessageRole::Assistant && message.content.is_empty() {
            None
        } else {
            Some(message.content.clone())
        };
        Self {
            role: message.role.as_str().to_string(),
            content,
            tool_calls: message.tool_calls.iter().map(WireToolCall::from).collect(),
            tool_call_id: message.tool_call_id.clone(),
        }
    }
}

/// A callable function advertised to `DeepSeek`.
///
/// The `parameters` value should be a JSON Schema object describing the
/// expected argument shape.
///
/// # Examples
///
/// ```rust
/// use deepseek_acp_adapter::deepseek::ToolDefinition;
///
/// let tool = ToolDefinition::new(
///     "grep",
///     "Search files for a regular expression",
///     serde_json::json!({
///         "type": "object",
///         "properties": {
///             "pattern": { "type": "string" }
///         },
///         "required": ["pattern"],
///         "additionalProperties": false
///     }),
/// );
///
/// assert_eq!(tool.name(), "grep");
/// ```
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
///
/// Instances of this type usually come from accumulated streamed
/// [`ToolCallDelta`] fragments once the model has finished emitting a tool
/// request.
///
/// # Examples
///
/// ```rust
/// use deepseek_acp_adapter::deepseek::ToolCall;
///
/// let call = ToolCall::new("call-1", "read_file", r#"{"path":"Cargo.toml"}"#);
///
/// assert_eq!(call.id(), "call-1");
/// assert_eq!(call.arguments(), r#"{"path":"Cargo.toml"}"#);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
pub(crate) struct WireToolDefinition {
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
///
/// Requests are immutable builder values: start with [`ChatRequest::new`]
/// and then add optional tools, model overrides, or reasoning effort.
///
/// # Examples
///
/// ```rust
/// use deepseek_acp_adapter::deepseek::{ChatMessage, ChatRequest};
///
/// let request = ChatRequest::new(vec![ChatMessage::user("Summarize the diff")])
///     .with_model("deepseek-v4-flash")
///     .with_reasoning_effort("high");
///
/// assert_eq!(request.model(), Some("deepseek-v4-flash"));
/// assert_eq!(request.reasoning_effort(), Some("high"));
/// ```
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

    pub(crate) fn into_parts(
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
///
/// This flattens provider wire chunks into events the adapter can consume
/// incrementally without exposing the raw SSE schema.
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
///
/// The provider may emit tool call metadata and JSON arguments across many
/// chunks. Callers typically buffer deltas by [`ToolCallDelta::index`]
/// until a full tool call can be reconstructed.
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
    pub const fn new(
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
    pub const fn index(&self) -> usize {
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
///
/// These are normalized from provider-specific strings and can be mapped to
/// the adapter's higher-level stop reasons.
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
    pub(crate) fn from_api(value: &str) -> Self {
        match value {
            "stop" => Self::EndTurn,
            "length" => Self::MaxTokens,
            "tool_calls" => Self::ToolCalls,
            "content_filter" => Self::Refusal,
            other => Self::Other(other.to_string()),
        }
    }
}
