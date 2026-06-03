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

use std::sync::{Arc, Mutex};
use std::{error::Error, process::ExitCode};

use agent_client_protocol::schema::{
    AgentAuthCapabilities, AgentCapabilities, AuthenticateRequest, AuthenticateResponse,
    ClientCapabilities, InitializeRequest, InitializeResponse, PromptCapabilities, ProtocolVersion,
};
use agent_client_protocol::{Agent, ConnectTo, Stdio};
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

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
    let state = Arc::new(Mutex::new(AdapterState::default()));
    serve_with_transport(Stdio::new(), state).await
}

async fn serve_with_transport(
    transport: impl ConnectTo<Agent>,
    state: Arc<Mutex<AdapterState>>,
) -> Result<(), agent_client_protocol::Error> {
    let initialize_state = Arc::clone(&state);

    Agent
        .builder()
        .name("deepseek-acp-adapter")
        .on_receive_request(
            async move |request: InitializeRequest, responder, _cx| {
                record_client_capabilities(&initialize_state, request.client_capabilities)?;
                responder.respond(build_initialize_response(request.protocol_version))
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |_request: AuthenticateRequest, responder, _cx| {
                responder.respond(AuthenticateResponse::new())
            },
            agent_client_protocol::on_receive_request!(),
        )
        .connect_to(transport)
        .await
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

#[derive(Debug, Default)]
struct AdapterState {
    client_capabilities: Option<ClientCapabilities>,
}

#[cfg(test)]
mod tests {
    use super::{
        AdapterState, Cli, Command, build_initialize_response, record_client_capabilities,
    };
    use std::sync::{Arc, Mutex};

    use agent_client_protocol::schema::{
        ClientCapabilities, FileSystemCapabilities, InitializeRequest, ProtocolVersion,
    };
    use clap::Parser;

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

        record_client_capabilities(&state, request.client_capabilities)?;

        let response = build_initialize_response(request.protocol_version);

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
}
