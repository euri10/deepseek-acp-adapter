//! Command-line entrypoint for the `DeepSeek` `ACP` adapter.

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
// `#[must_use]` on every internal binary helper is noise at this stage.
#![allow(clippy::must_use_candidate)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::{error::Error, process::ExitCode};

use agent_client_protocol::schema::{
    AgentAuthCapabilities, AgentCapabilities, AuthenticateRequest, AuthenticateResponse,
    CancelNotification, ClientCapabilities, ContentBlock, ContentChunk, InitializeRequest,
    InitializeResponse, NewSessionRequest, NewSessionResponse, PromptCapabilities, PromptRequest,
    PromptResponse, ProtocolVersion, SessionId, SessionMode, SessionModeState, SessionNotification,
    SessionUpdate, StopReason,
};
use agent_client_protocol::{Agent, ConnectTo, Stdio};
use clap::{Parser, Subcommand};
use deepseek_acp_adapter::deepseek::{
    ChatMessage, ChatRequest, DeepSeekClient, FinishReason, LlmClient, StreamEvent,
};
use futures_util::StreamExt;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

type AdapterResult<T> = Result<T, Box<dyn Error + Send + Sync + 'static>>;

#[derive(Debug, Parser)]
#[command(
    name = "deepseek-acp-adapter",
    version,
    about = "ACP stdio adapter for DeepSeek-backed coding sessions"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, PartialEq, Eq, Subcommand)]
enum Command {
    /// Run the ACP server over standard input and output.
    Serve,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> AdapterResult<()> {
    init_tracing()?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    runtime.block_on(async {
        match Cli::parse().command {
            Command::Serve => serve().await,
        }
    })?;

    Ok(())
}

fn init_tracing() -> AdapterResult<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .try_init()?;
    Ok(())
}

async fn serve() -> Result<(), agent_client_protocol::Error> {
    let llm_client = Arc::new(
        DeepSeekClient::from_env().map_err(agent_client_protocol::Error::into_internal_error)?,
    );
    let state = Arc::new(Mutex::new(AdapterState::default()));
    serve_with_transport(Stdio::new(), state, llm_client).await
}

async fn serve_with_transport(
    transport: impl ConnectTo<Agent>,
    state: Arc<Mutex<AdapterState>>,
    llm_client: Arc<dyn LlmClient>,
) -> Result<(), agent_client_protocol::Error> {
    let initialize_state = Arc::clone(&state);
    let new_session_state = Arc::clone(&state);
    let prompt_state = Arc::clone(&state);
    let prompt_client = Arc::clone(&llm_client);
    let cancel_state = Arc::clone(&state);

    Agent
        .builder()
        .name("deepseek-acp-adapter")
        .on_receive_request(
            async move |request: InitializeRequest, responder, _cx| {
                responder.respond(handle_initialize_request(&initialize_state, request)?)
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |_request: AuthenticateRequest, responder, _cx| {
                responder.respond(handle_authenticate_request())
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: NewSessionRequest, responder, _cx| {
                responder.respond(handle_new_session_request(&new_session_state, &request)?)
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: PromptRequest, responder, cx| {
                let state = Arc::clone(&prompt_state);
                let client = Arc::clone(&prompt_client);
                let connection = cx.clone();

                cx.spawn(async move {
                    let result =
                        handle_prompt_request(&state, client.as_ref(), request, |notification| {
                            connection.send_notification(notification)
                        })
                        .await;

                    responder.respond_with_result(result)?;
                    Ok(())
                })?;

                Ok(())
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_notification(
            async move |notification: CancelNotification, _cx| {
                handle_cancel_notification(&cancel_state, &notification)
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_to(transport)
        .await
}

fn handle_initialize_request(
    state: &Arc<Mutex<AdapterState>>,
    request: InitializeRequest,
) -> Result<InitializeResponse, agent_client_protocol::Error> {
    record_client_capabilities(state, request.client_capabilities)?;
    Ok(build_initialize_response(request.protocol_version))
}

fn handle_authenticate_request() -> AuthenticateResponse {
    AuthenticateResponse::new()
}

fn handle_new_session_request(
    state: &Arc<Mutex<AdapterState>>,
    request: &NewSessionRequest,
) -> Result<NewSessionResponse, agent_client_protocol::Error> {
    validate_session_paths(request)?;

    let session_id = format!("session-{}", Uuid::new_v4());
    let mut guard = state
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    guard
        .sessions
        .insert(session_id.clone().into(), SessionRecord::default());

    Ok(NewSessionResponse::new(session_id).modes(default_session_modes()))
}

async fn handle_prompt_request(
    state: &Arc<Mutex<AdapterState>>,
    llm_client: &dyn LlmClient,
    request: PromptRequest,
    mut notify: impl FnMut(SessionNotification) -> Result<(), agent_client_protocol::Error>,
) -> Result<PromptResponse, agent_client_protocol::Error> {
    let user_text = text_from_prompt(&request.prompt)?;
    let user_message = ChatMessage::user(user_text.clone());
    let session_id = request.session_id.clone();
    let cancellation_token = CancellationToken::new();
    let messages = {
        let mut guard = state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let session = guard.sessions.get_mut(&request.session_id).ok_or_else(|| {
            agent_client_protocol::Error::invalid_params()
                .data(format!("unknown session id: {}", request.session_id.0))
        })?;
        if session.active_turn.is_some() {
            return Err(
                agent_client_protocol::Error::invalid_request().data(format!(
                    "session {} already has an active turn",
                    request.session_id.0
                )),
            );
        }
        session.active_turn = Some(cancellation_token.clone());

        let mut messages = session.history.clone();
        messages.push(user_message.clone());
        messages
    };

    let result = run_prompt_turn(
        state,
        llm_client,
        request,
        user_message,
        messages,
        cancellation_token.clone(),
        &mut notify,
    )
    .await;
    clear_active_turn(state, &session_id)?;
    result
}

async fn run_prompt_turn(
    state: &Arc<Mutex<AdapterState>>,
    llm_client: &dyn LlmClient,
    request: PromptRequest,
    user_message: ChatMessage,
    messages: Vec<ChatMessage>,
    cancellation_token: CancellationToken,
    notify: &mut impl FnMut(SessionNotification) -> Result<(), agent_client_protocol::Error>,
) -> Result<PromptResponse, agent_client_protocol::Error> {
    let mut stream = llm_client
        .stream_chat(ChatRequest::new(messages), cancellation_token.clone())
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let mut assistant_text = String::new();
    let mut stop_reason = StopReason::EndTurn;

    loop {
        let event = tokio::select! {
            () = cancellation_token.cancelled() => {
                stop_reason = StopReason::Cancelled;
                break;
            }
            event = stream.next() => event,
        };

        let Some(event) = event else {
            if cancellation_token.is_cancelled() {
                stop_reason = StopReason::Cancelled;
            }
            break;
        };

        match event.map_err(agent_client_protocol::Error::into_internal_error)? {
            StreamEvent::Thought(chunk) => notify(session_notification(
                request.session_id.clone(),
                SessionUpdate::AgentThoughtChunk(ContentChunk::new(chunk.into())),
            ))?,
            StreamEvent::Message(chunk) => {
                assistant_text.push_str(&chunk);
                notify(session_notification(
                    request.session_id.clone(),
                    SessionUpdate::AgentMessageChunk(ContentChunk::new(chunk.into())),
                ))?;
            }
            StreamEvent::Finished(reason) => {
                stop_reason = stop_reason_from_finish(&reason);
            }
        }
    }

    if stop_reason != StopReason::Cancelled {
        let mut guard = state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let session = guard.sessions.get_mut(&request.session_id).ok_or_else(|| {
            agent_client_protocol::Error::invalid_params()
                .data(format!("unknown session id: {}", request.session_id.0))
        })?;
        session.history.push(user_message);
        session.history.push(ChatMessage::assistant(assistant_text));
    }

    Ok(PromptResponse::new(stop_reason))
}

fn build_initialize_response(protocol_version: ProtocolVersion) -> InitializeResponse {
    InitializeResponse::new(protocol_version).agent_capabilities(
        AgentCapabilities::new()
            .load_session(false)
            .prompt_capabilities(PromptCapabilities::new().embedded_context(true).image(true))
            .auth(AgentAuthCapabilities::new()),
    )
}

fn record_client_capabilities(
    state: &Arc<Mutex<AdapterState>>,
    client_capabilities: ClientCapabilities,
) -> Result<(), agent_client_protocol::Error> {
    let mut guard = state
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    guard.client_capabilities = Some(client_capabilities);
    Ok(())
}

fn handle_cancel_notification(
    state: &Arc<Mutex<AdapterState>>,
    notification: &CancelNotification,
) -> Result<(), agent_client_protocol::Error> {
    let guard = state
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    if let Some(active_turn) = guard
        .sessions
        .get(&notification.session_id)
        .and_then(|session| session.active_turn.as_ref())
    {
        active_turn.cancel();
    }

    Ok(())
}

fn clear_active_turn(
    state: &Arc<Mutex<AdapterState>>,
    session_id: &SessionId,
) -> Result<(), agent_client_protocol::Error> {
    let mut guard = state
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    if let Some(session) = guard.sessions.get_mut(session_id) {
        session.active_turn = None;
    }

    Ok(())
}

fn validate_session_paths(request: &NewSessionRequest) -> Result<(), agent_client_protocol::Error> {
    if !request.cwd.is_absolute() {
        return Err(agent_client_protocol::Error::invalid_params()
            .data("session cwd must be an absolute path"));
    }

    if request
        .additional_directories
        .iter()
        .any(|path| !path.is_absolute())
    {
        return Err(agent_client_protocol::Error::invalid_params()
            .data("additional session directories must be absolute paths"));
    }

    Ok(())
}

fn default_session_modes() -> SessionModeState {
    SessionModeState::new("chat", vec![SessionMode::new("chat", "Chat")])
}

fn text_from_prompt(prompt: &[ContentBlock]) -> Result<String, agent_client_protocol::Error> {
    let mut text = String::new();

    for block in prompt {
        match block {
            ContentBlock::Text(content) => text.push_str(&content.text),
            _ => {
                return Err(agent_client_protocol::Error::invalid_params()
                    .data("only text prompt blocks are supported"));
            }
        }
    }

    if text.trim().is_empty() {
        return Err(agent_client_protocol::Error::invalid_params()
            .data("prompt must include non-empty text"));
    }

    Ok(text)
}

fn session_notification(
    session_id: agent_client_protocol::schema::SessionId,
    update: SessionUpdate,
) -> SessionNotification {
    SessionNotification::new(session_id, update)
}

fn stop_reason_from_finish(reason: &FinishReason) -> StopReason {
    match reason {
        FinishReason::EndTurn | FinishReason::ToolCalls | FinishReason::Other(_) => {
            StopReason::EndTurn
        }
        FinishReason::MaxTokens => StopReason::MaxTokens,
        FinishReason::Refusal => StopReason::Refusal,
    }
}

#[derive(Debug, Default)]
struct AdapterState {
    client_capabilities: Option<ClientCapabilities>,
    sessions: HashMap<agent_client_protocol::schema::SessionId, SessionRecord>,
}

#[derive(Debug, Default)]
struct SessionRecord {
    history: Vec<ChatMessage>,
    active_turn: Option<CancellationToken>,
}

#[cfg(test)]
mod tests {
    use super::{
        AdapterState, Cli, Command, build_initialize_response, handle_authenticate_request,
        handle_cancel_notification, handle_initialize_request, handle_new_session_request,
        handle_prompt_request,
    };
    use deepseek_acp_adapter::deepseek::{
        ChatRequest, DeepSeekError, FinishReason, LlmClient, StreamEvent,
    };
    use futures_util::stream::{self, BoxStream};
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    use agent_client_protocol::schema::{
        CancelNotification, ClientCapabilities, ContentBlock, FileSystemCapabilities,
        InitializeRequest, NewSessionRequest, PromptRequest, ProtocolVersion, SessionNotification,
        SessionUpdate, StopReason,
    };
    use clap::Parser;
    use tokio_util::sync::CancellationToken;

    struct FakeLlmClient {
        requests: Arc<Mutex<Vec<ChatRequest>>>,
        steps: Mutex<Option<Vec<FakeStreamStep>>>,
    }

    impl FakeLlmClient {
        fn new(events: Vec<Result<StreamEvent, DeepSeekError>>) -> Self {
            Self::with_steps(events.into_iter().map(FakeStreamStep::Event).collect())
        }

        fn with_steps(steps: Vec<FakeStreamStep>) -> Self {
            Self {
                requests: Arc::new(Mutex::new(Vec::new())),
                steps: Mutex::new(Some(steps)),
            }
        }

        fn requests(&self) -> Arc<Mutex<Vec<ChatRequest>>> {
            Arc::clone(&self.requests)
        }
    }

    impl LlmClient for FakeLlmClient {
        fn stream_chat(
            &self,
            request: ChatRequest,
            cancellation_token: CancellationToken,
        ) -> Result<BoxStream<'static, Result<StreamEvent, DeepSeekError>>, DeepSeekError> {
            self.requests
                .lock()
                .map_err(|error| DeepSeekError::InvalidResponse(error.to_string()))?
                .push(request);
            let steps = self
                .steps
                .lock()
                .map_err(|error| DeepSeekError::InvalidResponse(error.to_string()))?
                .take()
                .ok_or_else(|| {
                    DeepSeekError::InvalidResponse(
                        "fake client stream was requested more than once".to_string(),
                    )
                })?;

            Ok(Box::pin(stream::unfold(
                (VecDeque::from(steps), cancellation_token),
                |(mut steps, cancellation_token)| async move {
                    let step = steps.pop_front()?;
                    match step {
                        FakeStreamStep::Event(event) => Some((event, (steps, cancellation_token))),
                        FakeStreamStep::WaitForCancel => {
                            cancellation_token.cancelled().await;
                            None
                        }
                    }
                },
            )))
        }
    }

    enum FakeStreamStep {
        Event(Result<StreamEvent, DeepSeekError>),
        WaitForCancel,
    }

    #[test_log::test]
    fn parses_serve_subcommand() {
        let parsed = Cli::try_parse_from(["deepseek-acp-adapter", "serve"]);

        assert!(matches!(
            parsed,
            Ok(Cli {
                command: Command::Serve
            })
        ));
    }

    #[test_log::test]
    fn build_initialize_response_advertises_expected_caps() {
        let response = build_initialize_response(ProtocolVersion::LATEST);

        assert_eq!(response.protocol_version, ProtocolVersion::LATEST);
        assert!(!response.agent_capabilities.load_session);
        assert!(response.agent_capabilities.prompt_capabilities.image);
        assert!(
            response
                .agent_capabilities
                .prompt_capabilities
                .embedded_context
        );
        assert!(response.auth_methods.is_empty());
    }

    #[test_log::test]
    fn initialize_handshake_records_client_capabilities() -> Result<(), agent_client_protocol::Error>
    {
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let request = InitializeRequest::new(ProtocolVersion::LATEST).client_capabilities(
            ClientCapabilities::new()
                .fs(FileSystemCapabilities::new()
                    .read_text_file(true)
                    .write_text_file(false))
                .terminal(true),
        );

        let response = handle_initialize_request(&state, request)?;

        assert_eq!(response.protocol_version, ProtocolVersion::LATEST);
        let guard = state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        assert_eq!(
            guard.client_capabilities.clone(),
            Some(
                ClientCapabilities::new()
                    .fs(FileSystemCapabilities::new()
                        .read_text_file(true)
                        .write_text_file(false),)
                    .terminal(true),
            )
        );

        Ok(())
    }

    #[test_log::test]
    fn authenticate_request_returns_empty_response() {
        let response = handle_authenticate_request();

        assert!(response.meta.is_none());
    }

    #[test_log::test]
    fn new_session_returns_id_and_mode() -> Result<(), agent_client_protocol::Error> {
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let response = handle_new_session_request(&state, &NewSessionRequest::new("/tmp"))?;

        assert!(response.session_id.0.starts_with("session-"));
        let modes = response
            .modes
            .ok_or_else(agent_client_protocol::Error::internal_error)?;
        assert_eq!(modes.current_mode_id.0.as_ref(), "chat");
        assert_eq!(modes.available_modes.len(), 1);

        let guard = state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        assert!(guard.sessions.contains_key(&response.session_id));

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn prompt_streams_updates_and_stores_history() -> Result<(), agent_client_protocol::Error>
    {
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let session = handle_new_session_request(&state, &NewSessionRequest::new("/tmp"))?;
        let client = FakeLlmClient::new(vec![
            Ok(StreamEvent::Thought("thinking".to_string())),
            Ok(StreamEvent::Message("hello".to_string())),
            Ok(StreamEvent::Message(" world".to_string())),
            Ok(StreamEvent::Finished(FinishReason::EndTurn)),
        ]);
        let requests = client.requests();
        let mut notifications = Vec::new();

        let response = handle_prompt_request(
            &state,
            &client,
            PromptRequest::new(session.session_id.clone(), vec![ContentBlock::from("hi")]),
            |notification| {
                notifications.push(notification);
                Ok(())
            },
        )
        .await?;

        assert_eq!(response.stop_reason, StopReason::EndTurn);
        assert_eq!(notifications.len(), 3);
        assert!(matches!(
            notifications[0].update,
            SessionUpdate::AgentThoughtChunk(_)
        ));
        assert!(matches!(
            notifications[1].update,
            SessionUpdate::AgentMessageChunk(_)
        ));
        assert!(matches!(
            notifications[2].update,
            SessionUpdate::AgentMessageChunk(_)
        ));

        let request_guard = requests
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        assert_eq!(request_guard.len(), 1);
        assert_eq!(request_guard[0].messages()[0].content(), "hi");
        drop(request_guard);

        let state_guard = state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let stored = state_guard
            .sessions
            .get(&session.session_id)
            .ok_or_else(|| {
                agent_client_protocol::Error::internal_error().data("missing stored session")
            })?;
        assert_eq!(stored.history.len(), 2);
        assert_eq!(stored.history[0].content(), "hi");
        assert_eq!(stored.history[1].content(), "hello world");

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn cancel_notification_stops_active_prompt() -> Result<(), agent_client_protocol::Error> {
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let session = handle_new_session_request(&state, &NewSessionRequest::new("/tmp"))?;
        let session_id = session.session_id.clone();
        let client = Arc::new(FakeLlmClient::with_steps(vec![
            FakeStreamStep::Event(Ok(StreamEvent::Message("partial".to_string()))),
            FakeStreamStep::WaitForCancel,
        ]));
        let (notification_tx, mut notification_rx) =
            tokio::sync::mpsc::unbounded_channel::<SessionNotification>();

        let prompt_state = Arc::clone(&state);
        let prompt_session_id = session_id.clone();
        let prompt_client = Arc::clone(&client);
        let prompt_task = tokio::spawn(async move {
            handle_prompt_request(
                &prompt_state,
                prompt_client.as_ref(),
                PromptRequest::new(prompt_session_id, vec![ContentBlock::from("cancel me")]),
                |notification| {
                    notification_tx
                        .send(notification)
                        .map_err(agent_client_protocol::Error::into_internal_error)?;
                    Ok(())
                },
            )
            .await
        });

        let notification = notification_rx
            .recv()
            .await
            .ok_or_else(|| agent_client_protocol::Error::internal_error().data("missing update"))?;
        let SessionUpdate::AgentMessageChunk(chunk) = notification.update else {
            return Err(
                agent_client_protocol::Error::internal_error().data("expected agent message chunk")
            );
        };
        let ContentBlock::Text(text) = chunk.content else {
            return Err(agent_client_protocol::Error::internal_error().data("expected text chunk"));
        };
        assert_eq!(text.text, "partial");

        handle_cancel_notification(&state, &CancelNotification::new(session_id.clone()))?;
        let response = prompt_task
            .await
            .map_err(agent_client_protocol::Error::into_internal_error)??;

        assert_eq!(response.stop_reason, StopReason::Cancelled);
        let guard = state
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let session = guard.sessions.get(&session_id).ok_or_else(|| {
            agent_client_protocol::Error::internal_error().data("missing session")
        })?;
        assert!(session.active_turn.is_none());
        assert!(session.history.is_empty());

        Ok(())
    }

    #[test_log::test(tokio::test)]
    async fn prompt_replays_history_on_next_turn() -> Result<(), agent_client_protocol::Error> {
        let state = Arc::new(Mutex::new(AdapterState::default()));
        let session = handle_new_session_request(&state, &NewSessionRequest::new("/tmp"))?;
        let first_client = FakeLlmClient::new(vec![
            Ok(StreamEvent::Message("first answer".to_string())),
            Ok(StreamEvent::Finished(FinishReason::EndTurn)),
        ]);
        handle_prompt_request(
            &state,
            &first_client,
            PromptRequest::new(
                session.session_id.clone(),
                vec![ContentBlock::from("first")],
            ),
            |_| Ok(()),
        )
        .await?;

        let second_client =
            FakeLlmClient::new(vec![Ok(StreamEvent::Finished(FinishReason::MaxTokens))]);
        let second_requests = second_client.requests();
        let response = handle_prompt_request(
            &state,
            &second_client,
            PromptRequest::new(session.session_id, vec![ContentBlock::from("second")]),
            |_| Ok(()),
        )
        .await?;

        assert_eq!(response.stop_reason, StopReason::MaxTokens);
        let request_guard = second_requests
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?;
        let messages = request_guard[0].messages();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].content(), "first");
        assert_eq!(messages[1].content(), "first answer");
        assert_eq!(messages[2].content(), "second");

        Ok(())
    }
}
