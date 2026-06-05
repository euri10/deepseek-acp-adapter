# DeepSeek ACP Adapter

`deepseek-acp-adapter` is a headless ACP server that exposes DeepSeek as an agent to ACP-capable editors.

## Requirements

- Rust stable
- `DEEPSEEK_API_KEY`
- Optional: `DEEPSEEK_BASE_URL`
- Optional: `DEEPSEEK_MODEL`

If `DEEPSEEK_BASE_URL` is unset, the adapter uses `https://api.deepseek.com`.
If `DEEPSEEK_MODEL` is unset, the adapter uses `deepseek-v4-pro`.

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

## Run It

```bash
cargo run -- serve
```

For local smoke tests, the binary also has a hidden dev mode:

```bash
cargo run -- dev
```

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
              "cargo",
              "run",
              "--",
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
      "command": "cargo",
      "args": ["run", "--", "serve"],
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
| `session/load` | ❌ Not implemented (adapter is stateless) |
| `session/resume` | ❌ Not implemented (no session persistence) |
| `session/prompt` | ✅ Full (text-only, tool loop, cancellation) |
| `session/cancel` | ✅ Full |
| `session/set_mode` | ✅ Full |
| `session/set_config_option` | ✅ Full |
| `session/request_permission` | ✅ Full |
| `session_info_update` | ❌ Not emitted |
| `logout` | ✅ No-op |
| `fs/read_text_file` | ✅ Client fs or local fallback |
| `fs/write_text_file` | ✅ Client fs or local fallback |
| `terminal/*` | ❌ Not implemented (uses local shell for `run_command`) |
| MCP tools (stdio) | ✅ Full |
| MCP tools (HTTP/SSE) | ❌ Not supported |

## Current Limitations

- No TUI
- No `session/load` or `session/resume` (adapter is stateless between connections)
- No terminal client methods (`run_command` uses the local shell)
- No `agent_plan` / `config_option_update` / `session_info_update` notifications
- No non-stdio MCP transports
- No auto model router
- No `apply_patch`-style edits in v0.1
