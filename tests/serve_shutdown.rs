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

//! Integration test for the serve shutdown path.
//!
//! Regression guard for dangling `deepseek-acp-adapter serve` processes: when
//! the ACP client closes stdin, the adapter must exit instead of hanging
//! forever. The test spawns the real binary, closes its stdin, and asserts the
//! process exits within a short timeout.

use std::error::Error;
use std::process::Stdio;
use std::time::Duration;

use tokio::process::Command;

/// Closing the child's stdin must make `serve` exit promptly.
///
/// Uses the `mock` backend so no `DEEPSEEK_API_KEY` is required to reach the
/// serve loop. If the dangling-process hang ever regresses, the `timeout`
/// elapses and the test fails instead of blocking the suite.
///
/// # Errors
///
/// Returns an error if the binary cannot be spawned, the wait operation fails, or the timeout elapses.
///
/// # Panics
///
/// Panics if the child exits with a non-zero status.
#[test_log::test(tokio::test)]
async fn serve_exits_when_stdin_closes() -> Result<(), Box<dyn Error>> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_deepseek-acp-adapter"))
        .args(["serve", "--backend", "mock"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()?;

    // Drop the write end of the child's stdin → EOF, which must trigger shutdown.
    drop(child.stdin.take());

    let status = tokio::time::timeout(Duration::from_secs(5), child.wait()).await??;
    assert!(status.success(), "serve exited unsuccessfully: {status:?}");
    Ok(())
}
