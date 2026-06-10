# DeepSeek ACP Adapter

`deepseek-acp-adapter` is a headless ACP server that exposes DeepSeek as an agent to ACP-capable editors.

> [!WARNING]
> This is alpha software. Expect breaking changes, incomplete ACP coverage, and rough edges while the adapter is still being shaped.

## Installation

```bash
cargo install deepseek-acp-adapter
```

## Debugging

For debugging prefer the included [`acp-debug.sh`](acp-debug.sh) wrapper instead of invoking the adapter binary directly. It keeps normal stdio behavior intact for ACP while appending the adapter's streams to `.local/state/codecompanion-acp`, you'll find `20260610-080836-deepseek-acp-adapter-stderr.log` for stderr and `20260610-080836-codex-acp-stdout-jsonrpc.log` for the jsonrpc messages.


## Architecture

The adapter bridges two independent channels:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                         deepseek-acp-adapter                                           │
│                                                                                        │
│  Editor ──ACP/stdio──▶ ┌─────────────────┐  ┌─────────────────┐                        │
│  (Zed,      JSON-RPC   │  acp.rs         │  │  deepseek/*     │                        │
│   Neovim,   frames  ◀──│  ACP transport  │  │  HTTPS + SSE    │──▶ DeepSeek API        │
│   ...)                 │  + request      │  │  client, types, │  │  api.deepseek.com   │
│                        │  handlers       │  │  stream parser  │  │ /chat/completions   │
│                        └─────────┬───────┘  └────────┬────────┘                        │
│                                  │                   │                                 │
│                           ┌──────▼───────────────────▼──────────────────┐              │
│                           │ · turn.rs           · tools.rs     · mcp.rs │              │
│                           │ · Session state     · tool loop    · MCP    │              │
│                           │ · Permission gating · cancellation          │              │
│                           └───────────────────┬─────────────────────────┘              │
│                                               │                                        │
│                                   ┌───────────▼───────────┐                            │
│                                   │   session_store.rs    │                            │
│                                   │   JSONL persistence   │                            │
│                                   └───────────────────────┘                            │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

**Left side** — the adapter speaks the [Agent Client Protocol](https://agentclientprotocol.com) (ACP) over stdio as JSON-RPC 2.0 frames. The `agent-client-protocol` crate handles the wire protocol; [`acp.rs`](src/acp.rs) registers request handlers and translates between ACP schema types and the adapter's internal types.

**Right side** — the adapter speaks HTTPS + Server-Sent Events to DeepSeek's OpenAI-compatible `/chat/completions` endpoint via a thin client owned by this crate in [`src/deepseek/`](src/deepseek/). A [`LlmClient`](src/deepseek/client.rs) trait provides the mock seam for testing without a live API key.

**Middle** — the adapter is the translator *and* the agent harness. [`turn.rs`](src/turn.rs) orchestrates the prompt→tool-call→execute→feed-back loop. [`tools.rs`](src/tools.rs) registers built-in tools (read/write/edit files, glob, grep, shell commands) and routes execution to the right backend. [`mcp.rs`](src/mcp.rs) connects to external MCP servers and exposes their tools through the same loop. [`session_store.rs`](src/session_store.rs) provides optional filesystem persistence so sessions survive process restarts.

### Module Map

**Binary Modules** (adapter runtime):

| Module | Responsibility |
|--------|---------------|
| [`acp/`](src/acp/) | ACP transport registration, request handler dispatch, response builders, permission requesters |
| [`session.rs`](src/session.rs) | Session state, permission model, in-memory session store, session lifecycle |
| [`turn.rs`](src/turn.rs) | Prompt-turn orchestration: LLM streaming, tool-call accumulation, loop control, cancellation |
| [`tools/`](src/tools/) | Built-in tool execution with two submodules: |
| [`registry.rs`](src/tools/registry.rs) | `ToolRegistry` trait, `ToolContext`, `AdapterToolRegistry` impl, tool metadata |
| [`execution/`](src/tools/execution) | Tool definitions, argument parsing, execution (read/write/edit/grep/glob/command), output truncation |
| [`mcp.rs`](src/mcp.rs) | MCP server connection (stdio + HTTP streamable), tool-name mapping, invocation, result rendering |
| [`session_store.rs`](src/session_store.rs) | Filesystem-backed session metadata and JSONL chat-history persistence |
| [`dev.rs`](src/dev.rs) | Development utilities, smoke tests, CLI testing backends |
| [`error.rs`](src/error.rs) | Unified domain error type (adapter crate root) |

**Library Modules** (`deepseek` - reusable client):

| Module | Responsibility |
|--------|---------------|
| [`deepseek/types.rs`](src/deepseek/types.rs) | Chat message, request, tool definition, and stream-event types (public facade) |
| [`deepseek/client.rs`](src/deepseek/client.rs) | HTTP client with SSE retry, `LlmClient` trait, `DeepSeekClient` impl |
| [`deepseek/stream.rs`](src/deepseek/stream.rs) | SSE event parsing, tool-call delta reassembly, finish-reason mapping |
| [`deepseek/config.rs`](src/deepseek/config.rs) | Environment-driven config (`DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`) |
| [`deepseek/error.rs`](src/deepseek/error.rs) | Typed error enum (config, HTTP, SSE, JSON, transport) |

### Design Principles

- **Translation boundary**: ACP and HTTP types stay at their respective edges. Business logic in the adapter core (`turn`, `tools`, `session_store`) depends only on the adapter's own types — not on `agent-client-protocol` schema types or raw HTTP types.
- **Testable seams**: The `LlmClient` trait lets prompt-turn tests run against canned SSE fixtures without a network. The `ToolRegistry` trait lets tool-loop tests inject fake tools. ACP handler tests use in-memory fake client connections.
- **Single async runtime**: Tokio multi-thread throughout. No lock is held across `.await`. No mixing of async runtimes.
- **No unsafe code**: `#![forbid(unsafe_code)]` at every crate root.

## Requirements

- Rust stable
- `DEEPSEEK_API_KEY`
- Optional: `DEEPSEEK_BASE_URL`
- Optional: `DEEPSEEK_MODEL`

If `DEEPSEEK_BASE_URL` is unset, the adapter uses `https://api.deepseek.com`.
If `DEEPSEEK_MODEL` is unset, the adapter uses `deepseek-v4-pro`.


## Editor Setup

### CodeCompanion

CodeCompanion uses ACP adapters for chat interactions. Extend the adapter config with this server and select it for chat.

```lua
require("codecompanion").setup({
  adapters = {
    acp = {
      deepseek_acp = function()
        return require("codecompanion.adapters").extend("deepseek_acp", {
          commands = {
            default = {
              "deepseek-acp-adapter",
              "serve",
            },
          },
          env = {
            DEEPSEEK_API_KEY = "your-api-key",
            DEEPSEEK_BASE_URL = "https://api.deepseek.com",
            DEEPSEEK_MODEL = "deepseek-v4-pro",
          },
        })
      end,
    },
  },
  interactions = {
    chat = {
      adapter = "deepseek_acp",
    },
  },
})
```

### Zed

Zed can run any ACP-capable agent as an external agent. Put the adapter command and its environment in `settings.json` under `agent_servers`.

```json
{
  "agent_servers": {
    "DeepSeek ACP": {
      "type": "custom",
      "command": "deepseek-acp-adapter",
      "args": ["serve"],
      "env": {
        "DEEPSEEK_API_KEY": "your-api-key",
        "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
        "DEEPSEEK_MODEL": "deepseek-v4-pro"
      }
    }
  }
}
```

If Zed is launched from a GUI app launcher, it may not inherit your shell environment. Set the adapter env vars in Zed's agent server config instead of relying on your terminal session.

> [!WARNING]
> I don't have Zed so this is totally untested

## Supported Modes

- `ask`
- `accept-edits`
- `yolo`

`session/set_mode` switches posture live during a session. In `accept-edits`, edit actions auto-approve while shell actions still prompt. In `yolo`, mutating tools auto-approve.

## Supported Tools

- `read_file`
- `write_file`
- `edit_file`
- `run_command`

Tool calls are permission-gated and surfaced through ACP so the editor can show native diffs and command output.

For sessions that advertise `additionalDirectories`, relative file paths resolve against the
session `cwd` first and then each additional directory in order. Absolute paths are passed
through unchanged, and `run_command` runs as a regular shell command rooted at `cwd` rather
than a filesystem sandbox.

## ACP Protocol Coverage

| Feature | Status |
|---------|--------|
| `initialize` | ✅ Full |
| `authenticate` | ✅ No-op (no auth required) |
| `session/new` | ✅ Full (async path with MCP startup) |
| `session/list` | ✅ Full |
| `session/close` | ✅ Full |
| `session/load` | ✅ Full (restores persisted state and replays history) |
| `session/resume` | ✅ Full (restores persisted state without replay) |
| `session/prompt` | ✅ Full (text-only, tool loop, cancellation, plan/thought streaming) |
| `session/cancel` | ✅ Full |
| `session/set_mode` | ✅ Full |
| `session/set_config_option` | ✅ Full |
| `session/request_permission` | ✅ Full |
| `agent_plan` / `current_mode_update` / `config_option_update` / `available_commands_update` | ✅ Emitted |
| `session_info_update` | ❌ Not emitted |
| `logout` | ✅ No-op |
| `fs/read_text_file` | ✅ Client fs or local fallback |
| `fs/write_text_file` | ✅ Client fs or local fallback |
| `terminal/*` | ✅ Used for `run_command` when the client advertises terminal support |
| MCP tools (stdio) | ✅ Full |
| MCP tools (streamable HTTP) | ✅ Full |
| MCP tools (SSE) | ❌ Not supported |

## Current Limitations

- No TUI
- No `session_info_update` notifications
- No MCP SSE transport
- No auto model router
- No `apply_patch`-style edits in v0.1

## Library API

The crate also exposes a reusable `deepseek` module for request construction and
streaming response handling. Generate the API docs locally with:

```bash
cargo doc --no-deps
```

Typical library entry points:

- `deepseek::ChatMessage` for system, user, assistant, and tool-result messages
- `deepseek::ChatRequest` for model/tool request construction
- `deepseek::ToolDefinition` for JSON-schema tool advertisement
- `deepseek::StreamEvent` for normalized streamed output
- `deepseek::DeepSeekClient` for HTTP-backed streaming requests

Minimal streaming example:

```rust,no_run
use deepseek_acp_adapter::deepseek::{ChatMessage, ChatRequest, DeepSeekClient, LlmClient};
use futures_util::StreamExt;
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = DeepSeekClient::from_env()?;
    let request = ChatRequest::new(vec![ChatMessage::user("Summarize this repository")]);
    let mut stream = client.stream_chat(request, CancellationToken::new())?;

    while let Some(event) = stream.next().await {
        println!("{:?}", event?);
    }

    Ok(())
}
```


