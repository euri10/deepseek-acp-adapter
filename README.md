# DeepSeek ACP Adapter

`deepseek-acp-adapter` is a headless ACP server that exposes DeepSeek as an agent to ACP-capable editors.

## Requirements

- Rust stable
- `DEEPSEEK_API_KEY`
- Optional: `DEEPSEEK_BASE_URL`
- Optional: `DEEPSEEK_MODEL`

If `DEEPSEEK_BASE_URL` is unset, the adapter uses `https://api.deepseek.com`.
If `DEEPSEEK_MODEL` is unset, the adapter uses `deepseek-v4-pro`.

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

## Current Limitations

- No TUI
- No MCP client support
- No `loadSession`
- No auto model router
- No `apply_patch`-style edits in v0.1

