#!/usr/bin/env bash
set -u

LOG_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/codecompanion-acp"
mkdir -p "$LOG_DIR"

ACP_BINARY="$1"

TS="$(date +%Y%m%d-%H%M%S)"
STDERR_LOG="$LOG_DIR/$TS-$ACP_BINARY-stderr.log"
STDOUT_LOG="$LOG_DIR/$TS-$ACP_BINARY-stdout-jsonrpc.log"

echo "cwd: $(pwd)" >> "$STDERR_LOG"
echo "argv: $*" >> "$STDERR_LOG"
echo "env PATH: $PATH" >> "$STDERR_LOG"

# stdout is ACP JSON-RPC, so tee it without adding anything.
# stderr is safe to redirect to a log file.
exec "$@" 2>>"$STDERR_LOG" | tee -a "$STDOUT_LOG"
