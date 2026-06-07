## 0. DEVELOPMENT PHASE

This project has no external users and is in active early development. **Breaking changes are acceptable.** Do not create backward-compatibility shims, deprecation wrappers, or migration paths unless explicitly asked. Fix the code directly. We want to do things the right way with no accumulated tech debt.

---

## 1. TOOLING & ENVIRONMENT (MANDATORY)

### 1.1 Rust Version

- Stable toolchain only
- MSRV declared in Cargo.toml
- Edition must be specified

### 1.2 Formatting (REQUIRED)

```bash
cargo fmt --all
cargo fmt --all -- --check
```

CI must enforce formatting checks.

### 1.3 Linting (REQUIRED)

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

At crate root:

```rust
#![forbid(unsafe_code)]
#![deny(
    warnings,
    clippy::all,
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::unimplemented,
)]
```

No blanket `#[allow(...)]`. Any lint suppression must include explicit justification.

> **Forbidden blanket suppressions** — never add at crate or module level:
> - `#[allow(clippy::missing_errors_doc)]`
> - `#[allow(clippy::missing_panics_doc)]`
>
> Fix the underlying issue: add `# Errors` / `# Panics` doc sections to every public
> function that can fail or panic.

> **Permitted narrow suppressions** — allowed at crate root only with a justifying comment:
> - `#[allow(clippy::must_use_candidate)]` — when `#[must_use]` on the full API surface is impractical
> - `#[allow(clippy::cast_possible_truncation)]` / `cast_sign_loss` / `cast_lossless` / `cast_possible_wrap` — only in crates with intentional chip/pot integer arithmetic
> - `#[allow(clippy::wildcard_imports)]` — only for crate-internal re-export modules
> - `#[allow(clippy::too_many_lines)]` / `#[allow(clippy::doc_markdown)]` — only with a comment explaining why refactoring is not practical

> **Preferred**: configure workspace-level lints in root `Cargo.toml` so suppressions are
> declared once:
>
> ```toml
> [workspace.lints.clippy]
> must_use_candidate = "allow"       # See §1.3 justification above
> cast_possible_truncation = "allow" # Intentional in chip/pot arithmetic
> ```
>
> Crates then opt in with `lints.workspace = true` in their `[package]` table.

### 1.4 Property-Based & Benchmark Testing

- **Property-based tests** (`proptest`) — put in `#[cfg(test)]` modules alongside unit tests.
- **Benchmarks** (`criterion`) — live in `benches/` under the relevant crate.

### 1.5 Testing (REQUIRED)

The full test procedure — **in order** — is:

```bash
echo "yeah"

# 2. Establish a baseline BEFORE touching any code.
git stash
cargo test -q 2>&1 | grep -E "^test result|FAILED"
git stash pop

# 3. Run the full suite with your changes applied.
cargo test -q 2>&1 | grep -E "^test result|FAILED"

# 4. Every 'test result' line must say 'ok'.
#    Any FAILED line that was NOT in the baseline must be fixed before closing the task.

# 5. Formatting and linting must also pass.
cargo fmt --all && cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

For coverage reporting, use `cargo llvm-cov`. Do not use `cargo tarpaulin`.

#### Test logging convention

New tests that benefit from tracing output should use `test-log` as the default
test attribute. `test-log` is declared in `[workspace.dependencies]`; pick it up
in any crate via `[dev-dependencies] test-log = { workspace = true }`. The
workspace configuration enables only the `trace` feature, so events flow through
`tracing-subscriber` (not `env_logger`).

```rust
#[test_log::test]
fn synchronous_test() { /* tracing events become visible on failure */ }

#[test_log::test(tokio::test)]
async fn async_test() { /* same, for async */ }
```

Use `tracing-test` only per-test when a test must assert on log output (e.g. a
silent recovery path with no other observable side effect). Do not blanket-add
either crate to existing tests; adopt them as new tests are written.

---

## 2. PROJECT STRUCTURE

TO WRITE LATER

Rules:

- Business logic must not depend on I/O.
- External dependencies must not leak into domain layer.
- Public APIs must be documented.
- Avoid circular module dependencies.
- Keep modules cohesive and small.

---

## 3. CODING STANDARDS

### 3.1 Ownership & Borrowing

Accept `&str` not `&String`, `&[T]` not `&Vec<T>` in function signatures. Avoid unnecessary `.clone()`.

### 3.2 Global State

`static mut` is forbidden. Use `Arc<T>`, `Mutex<T>`, `RwLock<T>`, `OnceLock`, or dependency injection. No hidden global mutable state.

### 3.3 Async

- Use a single async runtime consistently.
- Do not hold locks across `.await`.

---

## 4. DOCUMENTATION STANDARDS

Run `cargo doc --no-deps` — undocumented public APIs are not allowed.

---

## 5. DANGEROUS RUST USAGE (STRICT POLICY)

### 5.1 Unsafe Code

Forbidden by default (`#![forbid(unsafe_code)]`). Allowed only when encapsulated in a safe
abstraction, fully documented with invariants, reviewed by maintainers, and covered by tests.

```rust
// SAFETY:
// - ptr is non-null
// - properly aligned
// - valid for reads
// - no aliasing
unsafe {
    std::ptr::read(ptr)
}
```

### 5.2 Prohibited Without Formal Justification

- `std::mem::transmute`, `std::mem::zeroed`, `std::mem::uninitialized`
- Raw pointer arithmetic
- Manual `Drop` manipulation
- Self-referential structs, incorrect `Pin` usage
- `static mut`
- Manual `Send` / `Sync` impl (requires a `// SAFETY:` comment)
- FFI without safe wrapper
- Artificial lifetime extension

### 5.3 Undefined Behavior Risk Areas

Prefer `slice.get(index)` over `slice[index]`. Avoid: dangling pointer dereferences,
invalid enum discriminants, aliasing rule violations, invalid UTF-8 assumptions, data races,
double frees, use-after-free.

### 5.4 Concurrency Hazards

- Define lock ordering to prevent deadlocks.
- Avoid nested `Arc<Mutex<T>>`.
- Never block inside async or hold locks across await points.
- Never implement `Send`/`Sync` without formal reasoning.

---

---

## 6. SECURITY & DEPENDENCIES

- Never log secrets. Validate all external input. Use constant-time comparison for secrets.
- Do not expose internal errors directly to users. Sanitize deserialized input.
- Keep the dependency tree minimal. Prefer well-maintained crates. No duplicate functionality.
- Audit before adding a new dependency: `cargo audit`.

---

## 7. ABSOLUTE RULES

Priority: **Safety → Correctness → Clarity → Performance**

Never:
- `unwrap()` / `expect()` / `panic!()` / `todo!()` / `unimplemented!()`
- Undocumented `unsafe`
- `#[allow(warnings)]` or any unsupported blanket lint suppression
- Silent error swallowing
- Hidden panics in library code
- Thread-unsafe global state
- Mixing async runtimes
- `git reset --hard`, `git clean -fd`, `rm -rf`, or any command that can delete or overwrite code without explicit user instruction — if uncertain, stop and ask

Before closing any task, verify:

- [ ] `cargo fmt --all` passes
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` passes
- [ ] All tests pass with no new failures vs. the baseline
- [ ] Public APIs documented
- [ ] No new `unwrap` / `expect` / `panic`
- [ ] Any lint suppression has a justifying comment

---

## 8. CODE EDITING DISCIPLINE

### No File Proliferation

Revise existing files in place. Never create versioned copies (`main_v2.rs`, `handler_improved.rs`, `handler_old.rs`). New files are only for genuinely new functionality — the bar is high.

### No Script-Based Changes

Never run a shell script that rewrites code files via regex or sed pipelines. Make changes manually; for many structurally identical changes, use `ast-grep`. Brittle text-based transforms create more problems than they solve.

### Search Tool Selection

Use **`ast-grep`** when structure matters — renaming APIs, finding all `.unwrap()` calls, enforcing patterns, applying safe rewrites across the codebase:

```bash
# Find all unwrap() calls
ast-grep run -l Rust -p '$EXPR.unwrap()'

# Find functions returning a specific type
ast-grep run -l Rust -p 'fn $NAME($$$) -> Result<$T, $E> { $$$}'
```

Use **`ripgrep`** when text is enough — hunting literals, TODOs, config values, quick recon:

```bash
rg -n 'unwrap\(' -t rust
```

Combine them: `rg` to shortlist files, `ast-grep` to match or rewrite precisely:

```bash
rg -l -t rust 'unwrap\(' | xargs ast-grep run -l Rust -p '$X.unwrap()'
```

### Consult Docs for Unfamiliar Libraries

If you are not confident about a third-party crate's current API, look up its documentation before writing code. Do not guess at method signatures or feature flags.

---

If it compiles, it is not necessarily correct.
If it passes tests, it is not necessarily safe.
If it uses unsafe, it must prove correctness.

<!-- bv-agent-instructions-v2 -->

---

## Beads Workflow Integration

This project uses [beads_rust](https://github.com/Dicklesworthstone/beads_rust) (`br`) for issue tracking and [beads_viewer](https://github.com/Dicklesworthstone/beads_viewer) (`bv`) for graph-aware triage. Issues are stored in `.beads/` and tracked in git.

### Using bv as an AI sidecar

bv is a graph-aware triage engine for Beads projects (.beads/beads.jsonl). Instead of parsing JSONL or hallucinating graph traversal, use robot flags for deterministic, dependency-aware outputs with precomputed metrics (PageRank, betweenness, critical path, cycles, HITS, eigenvector, k-core).

**Scope boundary:** bv handles *what to work on* (triage, priority, planning). `br` handles creating, modifying, and closing beads.

**CRITICAL: Use ONLY --robot-* flags. Bare bv launches an interactive TUI that blocks your session.**

#### The Workflow: Start With Triage

**`bv --robot-triage` is your single entry point.** It returns everything you need in one call:
- `quick_ref`: at-a-glance counts + top 3 picks
- `recommendations`: ranked actionable items with scores, reasons, unblock info
- `quick_wins`: low-effort high-impact items
- `blockers_to_clear`: items that unblock the most downstream work
- `project_health`: status/type/priority distributions, graph metrics
- `commands`: copy-paste shell commands for next steps

```bash
bv --robot-triage        # THE MEGA-COMMAND: start here
bv --robot-next          # Minimal: just the single top pick + claim command

# Token-optimized output (TOON) for lower LLM context usage:
bv --robot-triage --format toon
```

Before claiming, verify current state with `br show <id> --json` or `br ready --json`. `recommendations` can include graph-important blocked or assigned work; only `quick_ref.top_picks` and non-empty `claim_command` fields represent claimable work.

#### Other bv Commands

| Command | Returns |
|---------|---------|
| `--robot-plan` | Parallel execution tracks with unblocks lists |
| `--robot-priority` | Priority misalignment detection with confidence |
| `--robot-insights` | Full metrics: PageRank, betweenness, HITS, eigenvector, critical path, cycles, k-core |
| `--robot-alerts` | Stale issues, blocking cascades, priority mismatches |
| `--robot-suggest` | Hygiene: duplicates, missing deps, label suggestions, cycle breaks |
| `--robot-diff --diff-since <ref>` | Changes since ref: new/closed/modified issues |
| `--robot-graph [--graph-format=json\|dot\|mermaid]` | Dependency graph export |

#### Scoping & Filtering

```bash
bv --robot-plan --label backend              # Scope to label's subgraph
bv --robot-insights --as-of HEAD~30          # Historical point-in-time
bv --recipe actionable --robot-plan          # Pre-filter: ready to work (no blockers)
bv --recipe high-impact --robot-triage       # Pre-filter: top PageRank scores
```

### br Commands for Issue Management

```bash
br ready              # Show issues ready to work (no blockers)
br list --status=open # All open issues
br show <id>          # Full issue details with dependencies
br create --title="..." --type=task --priority=2
br update <id> --status=in_progress
br close <id> --reason="Completed"
br close <id1> <id2>  # Close multiple issues at once
br sync --flush-only  # Export DB to JSONL
```

### Workflow Pattern

1. **Triage**: Run `bv --robot-triage` to find the highest-impact actionable work
2. **Claim**: Use `br update <id> --status=in_progress`
3. **Work**: Implement the task
4. **Complete**: Use `br close <id>`
5. **Sync**: Always run `br sync --flush-only` at session end

### Key Concepts

- **Dependencies**: Issues can block other issues. `br ready` shows only unblocked work.
- **Priority**: P0=critical, P1=high, P2=medium, P3=low, P4=backlog (use numbers 0-4, not words)
- **Types**: task, bug, feature, epic, chore, docs, question
- **Blocking**: `br dep add <issue> <depends-on>` to add dependencies

### Session Protocol

```bash
git status              # Check what changed
git add <files>         # Stage code changes
br sync --flush-only    # Export beads changes to JSONL
git commit -m "..."     # Commit everything
git push                # Push to remote
```

<!-- end-bv-agent-instructions -->
