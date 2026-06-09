use futures_util::{
    StreamExt,
    stream::{self, BoxStream},
};
use reqwest::Client as HttpClient;
use reqwest_eventsource::EventSource;
use serde::Serialize;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::config::SystemEnvironment;
use super::stream::{StreamAttemptOutcome, run_stream_attempt};
use super::types::{ChatRequest, WireMessage, WireToolDefinition};
use super::{DeepSeekConfig, DeepSeekError, StreamEvent};

/// A `DeepSeek` chat-completions client.
///
/// The client implements [`LlmClient`] and streams normalized
/// [`StreamEvent`] values from `DeepSeek`'s OpenAI-compatible chat endpoint.
///
/// # Examples
///
/// ```rust
/// use deepseek_acp_adapter::deepseek::{DeepSeekClient, DeepSeekConfig};
///
/// let client = DeepSeekClient::new(DeepSeekConfig::new(
///     "test-key",
///     "https://api.deepseek.com",
///     "deepseek-v4-pro",
/// ));
///
/// assert_eq!(client.config().model(), "deepseek-v4-pro");
/// ```
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
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use deepseek_acp_adapter::deepseek::DeepSeekClient;
    ///
    /// let client = DeepSeekClient::from_env()?;
    /// assert!(!client.config().base_url().is_empty());
    /// # Ok::<(), deepseek_acp_adapter::deepseek::DeepSeekError>(())
    /// ```
    pub fn from_env() -> Result<Self, DeepSeekError> {
        Ok(Self::new(DeepSeekConfig::from_environment(
            &SystemEnvironment,
        )?))
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
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use deepseek_acp_adapter::deepseek::{ChatMessage, ChatRequest, DeepSeekClient, LlmClient};
    /// use futures_util::StreamExt;
    /// use tokio_util::sync::CancellationToken;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = DeepSeekClient::from_env()?;
    ///     let request = ChatRequest::new(vec![ChatMessage::user("Say hello")]);
    ///     let mut stream = client.stream_chat(request, CancellationToken::new())?;
    ///
    ///     while let Some(event) = stream.next().await {
    ///         let _ = event?;
    ///     }
    ///
    ///     Ok(())
    /// }
    /// ```
    fn stream_chat(
        &self,
        request: ChatRequest,
        cancellation_token: CancellationToken,
    ) -> Result<BoxStream<'static, Result<StreamEvent, DeepSeekError>>, DeepSeekError>;
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

impl LlmClient for DeepSeekClient {
    fn stream_chat(
        &self,
        request: ChatRequest,
        cancellation_token: CancellationToken,
    ) -> Result<BoxStream<'static, Result<StreamEvent, DeepSeekError>>, DeepSeekError> {
        if self.config.api_key().trim().is_empty() {
            return Err(DeepSeekError::MissingApiKey);
        }

        let (messages, tools, model, reasoning_effort) = request.into_parts();
        let body = ChatCompletionRequest {
            model: model.unwrap_or_else(|| self.config.model().to_string()),
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
            self.config.base_url().trim_end_matches('/')
        );
        let api_key = self.config.api_key().to_string();

        let (tx, rx) = mpsc::unbounded_channel::<Result<StreamEvent, DeepSeekError>>();

        tokio::spawn(async move {
            const MAX_RETRIES: u32 = 3;

            for attempt in 0..=MAX_RETRIES {
                if attempt > 0 {
                    let delay_ms = 100_u64 * (1_u64 << (attempt - 1));
                    tracing::warn!(
                        attempt,
                        delay_ms,
                        "retrying DeepSeek SSE stream after retryable transport error"
                    );
                    tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                }

                let request = http.post(&url).bearer_auth(&api_key).json(&body);

                tracing::debug!(
                    url = %url,
                    model = %body.model,
                    message_count = body.messages.len(),
                    tool_count = body.tools.len(),
                    stream = body.stream,
                    reasoning_effort = ?body.reasoning_effort,
                    "sending chat completion request to DeepSeek"
                );

                if tracing::enabled!(tracing::Level::TRACE)
                    && let Ok(request_json) = serde_json::to_string(&body)
                {
                    tracing::trace!(request_body = %request_json, "DeepSeek request body");
                }

                let event_source = match EventSource::new(request) {
                    Ok(es) => {
                        tracing::debug!("successfully created SSE event source");
                        es
                    }
                    Err(error) => {
                        tracing::error!(error = ?error, "failed to create SSE event source");
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
