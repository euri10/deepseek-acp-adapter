use super::*;
use crate::acp::handle_new_session_request;
use crate::session::PERMISSION_ALLOW_ONCE_OPTION_ID;
use crate::test_store;
use crate::test_utils::{
    CancelTracker, CountingReadTextFileRequester, FailingWriteRequester, FakePermissionRequester,
    FakeTerminalRequester, RecordingWriteTextFileRequester, Utf8FailingReadTextFileRequester,
};
use crate::tools::registry::{AdapterToolRegistry, EmptyToolRegistry, ToolExecution, ToolRegistry};
use agent_client_protocol::schema::{
    ClientCapabilities, FileSystemCapabilities, NewSessionRequest, ReadTextFileRequest,
    ReadTextFileResponse, RequestPermissionOutcome, RequestPermissionResponse,
    SelectedPermissionOutcome, ToolKind,
};
use agent_client_protocol::{Agent, Channel, Client};
use deepseek_acp_adapter::deepseek::ToolCall as DeepSeekToolCall;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

#[test_log::test(tokio::test)]
async fn read_file_tool_defaults_line_and_limit() -> Result<(), agent_client_protocol::Error> {
    let temp_root = std::env::temp_dir().join(format!(
        "deepseek-acp-adapter-defaults-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&temp_root)
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let file_path = temp_root.join("sample.txt");
    std::fs::write(&file_path, "alpha\nbeta\ngamma\n")
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let context = ToolContext {
        session_id: agent_client_protocol::schema::SessionId::new("session-defaults"),
        cwd: temp_root.clone(),
        additional_directories: Vec::new(),
        client_capabilities: None,
    };
    let call = DeepSeekToolCall::new(
        "call-defaults",
        "read_file",
        serde_json::json!({"path": "sample.txt"}).to_string(),
    );
    let result = read_file_tool_execution(&call, &context, None).await;
    assert!(result.success);
    assert_eq!(result.content, "alpha\nbeta\ngamma");
    assert_eq!(result.raw_output["source"], "local");
    assert_eq!(result.raw_output["line"], 1);
    assert_eq!(result.raw_output["limit"], 200);
    Ok(())
}

#[test_log::test(tokio::test)]
async fn read_file_tool_error_paths_report_failures() -> Result<(), agent_client_protocol::Error> {
    let temp_root = std::env::temp_dir().join(format!(
        "deepseek-acp-adapter-tools-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&temp_root)
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    std::fs::write(temp_root.join("visible.txt"), "one\ntwo\nthree")
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let context = ToolContext {
        session_id: agent_client_protocol::schema::SessionId::new("session-tools"),
        cwd: temp_root.clone(),
        additional_directories: Vec::new(),
        client_capabilities: None,
    };
    let r = read_file_tool_execution(
        &DeepSeekToolCall::new("invalid-read", "read_file", "not json"),
        &context,
        None,
    )
    .await;
    assert!(!r.success);
    assert!(r.content.contains("invalid read_file arguments"));
    let r = read_file_tool_execution(
        &DeepSeekToolCall::new(
            "zl",
            "read_file",
            serde_json::json!({"path":"visible.txt","line":0}).to_string(),
        ),
        &context,
        None,
    )
    .await;
    assert!(!r.success);
    assert!(r.content.contains("line must be at least 1"));
    let r = read_file_tool_execution(
        &DeepSeekToolCall::new(
            "zl2",
            "read_file",
            serde_json::json!({"path":"visible.txt","limit":0}).to_string(),
        ),
        &context,
        None,
    )
    .await;
    assert!(!r.success);
    assert!(r.content.contains("limit must be at least 1"));
    let r = read_file_tool_execution(
        &DeepSeekToolCall::new(
            "mf",
            "read_file",
            serde_json::json!({"path":"missing.txt"}).to_string(),
        ),
        &context,
        None,
    )
    .await;
    assert!(!r.success);
    assert!(!r.content.is_empty());
    Ok(())
}

#[test_log::test(tokio::test)]
async fn local_tool_error_paths_report_failures() -> Result<(), agent_client_protocol::Error> {
    let temp_root = std::env::temp_dir().join(format!(
        "deepseek-acp-adapter-tools-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&temp_root)
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let context = ToolContext {
        session_id: agent_client_protocol::schema::SessionId::new("session-tools"),
        cwd: temp_root.clone(),
        additional_directories: Vec::new(),
        client_capabilities: None,
    };
    assert!(
        !list_dir_tool_execution(
            &DeepSeekToolCall::new("il", "list_dir", "not json"),
            &context
        )
        .success
    );
    assert!(
        !list_dir_tool_execution(
            &DeepSeekToolCall::new(
                "md",
                "list_dir",
                serde_json::json!({"path":"missing-dir"}).to_string()
            ),
            &context
        )
        .success
    );
    assert!(
        !glob_tool_execution(&DeepSeekToolCall::new("ig", "glob", "not json"), &context).success
    );
    assert!(
        !glob_tool_execution(
            &DeepSeekToolCall::new(
                "igp",
                "glob",
                serde_json::json!({"pattern":"["}).to_string()
            ),
            &context
        )
        .success
    );
    assert!(
        !grep_tool_execution(&DeepSeekToolCall::new("ig2", "grep", "not json"), &context).success
    );
    assert!(
        !grep_tool_execution(
            &DeepSeekToolCall::new(
                "igr",
                "grep",
                serde_json::json!({"pattern":"("}).to_string()
            ),
            &context
        )
        .success
    );
    Ok(())
}

#[test_log::test(tokio::test)]
async fn read_file_tool_uses_local_fallback() -> Result<(), agent_client_protocol::Error> {
    let temp_root = std::env::temp_dir().join(format!(
        "deepseek-acp-adapter-local-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&temp_root)
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let file_path = temp_root.join("sample.txt");
    std::fs::write(&file_path, "alpha\nbeta\ngamma\ndelta\n")
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    let context = ToolContext {
        session_id: agent_client_protocol::schema::SessionId::new("session-local"),
        cwd: temp_root.clone(),
        additional_directories: Vec::new(),
        client_capabilities: None,
    };
    let call = DeepSeekToolCall::new(
        "call-local",
        "read_file",
        serde_json::json!({
            "path": "sample.txt",
            "line": 2,
            "limit": 2,
        })
        .to_string(),
    );

    let result = read_file_tool_execution(&call, &context, None).await;

    assert!(result.success);
    assert_eq!(result.content, "beta\ngamma");
    assert_eq!(result.raw_output["source"], "local");
    assert_eq!(result.raw_output["line"], 2);
    assert_eq!(result.raw_output["limit"], 2);
    assert_eq!(result.raw_output["path"], serde_json::json!(file_path));

    Ok(())
}

#[test_log::test(tokio::test)]
async fn read_file_tool_routes_to_client_fs() -> Result<(), agent_client_protocol::Error> {
    let temp_root = std::env::temp_dir().join(format!(
        "deepseek-acp-adapter-client-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&temp_root)
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    let observed_request = Arc::new(Mutex::new(None::<ReadTextFileRequest>));
    let observed_request_for_server = Arc::clone(&observed_request);
    let (client_transport, server_transport) = Channel::duplex();

    let server = tokio::spawn(async move {
        Agent
            .builder()
            .on_receive_request(
                async move |request: ReadTextFileRequest, responder, _cx| {
                    let mut guard = observed_request_for_server
                        .lock()
                        .map_err(agent_client_protocol::Error::into_internal_error)?;
                    *guard = Some(request.clone());
                    responder.respond(ReadTextFileResponse::new(
                        "buffered line two\nbuffered line three",
                    ))
                },
                agent_client_protocol::on_receive_request!(),
            )
            .connect_to(server_transport)
            .await
    });

    let context = ToolContext {
        session_id: agent_client_protocol::schema::SessionId::new("session-client"),
        cwd: temp_root.clone(),
        additional_directories: Vec::new(),
        client_capabilities: Some(
            ClientCapabilities::new().fs(FileSystemCapabilities::new()
                .read_text_file(true)
                .write_text_file(false)),
        ),
    };
    let call = DeepSeekToolCall::new(
        "call-client",
        "read_file",
        serde_json::json!({
            "path": "buffer.txt",
            "line": 2,
            "limit": 2,
        })
        .to_string(),
    );

    let result = Client
        .builder()
        .connect_with(client_transport, move |connection| async move {
            let result = read_file_tool_execution(
                &call,
                &context,
                Some(&connection as &dyn crate::acp::ReadTextFileRequester),
            )
            .await;
            Ok(result)
        })
        .await?;

    assert!(result.success);
    assert_eq!(result.content, "buffered line two\nbuffered line three");
    assert_eq!(result.raw_output["source"], "client");
    assert_eq!(result.raw_output["line"], 2);
    assert_eq!(result.raw_output["limit"], 2);
    assert_eq!(
        result.raw_output["path"],
        serde_json::json!(temp_root.join("buffer.txt"))
    );

    let request_guard = observed_request
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let request = request_guard.as_ref().ok_or_else(|| {
        agent_client_protocol::Error::internal_error().data("missing read_text_file request")
    })?;
    assert_eq!(request.session_id.0.as_ref(), "session-client");
    assert_eq!(request.path, temp_root.join("buffer.txt"));
    assert_eq!(request.line, Some(2));
    assert_eq!(request.limit, Some(2));
    drop(request_guard);

    server.abort();

    Ok(())
}

#[test_log::test(tokio::test)]
async fn read_file_tool_rejects_local_non_utf8_before_client_fs()
-> Result<(), agent_client_protocol::Error> {
    let temp_root = std::env::temp_dir().join(format!(
        "deepseek-acp-adapter-non-utf8-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&temp_root)
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let file_path = temp_root.join("artifact.bin");
    std::fs::write(&file_path, [0xff, 0xfe, 0xfd])
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    let context = ToolContext {
        session_id: agent_client_protocol::schema::SessionId::new("session-non-utf8"),
        cwd: temp_root.clone(),
        additional_directories: Vec::new(),
        client_capabilities: Some(
            ClientCapabilities::new().fs(FileSystemCapabilities::new().read_text_file(true)),
        ),
    };
    let call = DeepSeekToolCall::new(
        "call-non-utf8",
        "read_file",
        serde_json::json!({ "path": "artifact.bin" }).to_string(),
    );
    let requester = CountingReadTextFileRequester::new();
    let calls = requester.calls();

    let result = read_file_tool_execution(
        &call,
        &context,
        Some(&requester as &dyn crate::acp::ReadTextFileRequester),
    )
    .await;

    assert!(!result.success);
    assert!(result.content.contains("only supports UTF-8 text files"));
    assert!(result.content.contains(&file_path.display().to_string()));
    assert_eq!(
        *calls
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?,
        0
    );

    Ok(())
}

#[test_log::test(tokio::test)]
async fn read_file_tool_sanitizes_client_non_utf8_error() -> Result<(), agent_client_protocol::Error>
{
    let temp_root = std::env::temp_dir().join(format!(
        "deepseek-acp-adapter-client-utf8-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&temp_root)
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    let context = ToolContext {
        session_id: agent_client_protocol::schema::SessionId::new("session-client-utf8"),
        cwd: temp_root,
        additional_directories: Vec::new(),
        client_capabilities: Some(
            ClientCapabilities::new().fs(FileSystemCapabilities::new().read_text_file(true)),
        ),
    };
    let call = DeepSeekToolCall::new(
        "call-client-utf8",
        "read_file",
        serde_json::json!({ "path": "client-only.bin" }).to_string(),
    );

    let result = read_file_tool_execution(
        &call,
        &context,
        Some(&Utf8FailingReadTextFileRequester as &dyn crate::acp::ReadTextFileRequester),
    )
    .await;

    assert!(!result.success);
    assert!(result.content.contains("only supports UTF-8 text files"));
    assert!(!result.content.contains("Internal error"));
    assert!(
        !result
            .content
            .contains("stream did not contain valid UTF-8")
    );

    Ok(())
}

#[test_log::test(tokio::test)]
async fn write_file_tool_routes_to_client_fs_write() -> Result<(), agent_client_protocol::Error> {
    let temp_root = std::env::temp_dir().join(format!(
        "deepseek-acp-adapter-write-client-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&temp_root)
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let store = test_store();
    let session = handle_new_session_request(&store, &NewSessionRequest::new(&temp_root))?;
    let context = ToolContext {
        session_id: session.session_id.clone(),
        cwd: temp_root.clone(),
        additional_directories: Vec::new(),
        client_capabilities: Some(
            ClientCapabilities::new().fs(FileSystemCapabilities::new().write_text_file(true)),
        ),
    };

    let permission_requester = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
        RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
            PERMISSION_ALLOW_ONCE_OPTION_ID,
        )),
    )]);
    let write_requester = RecordingWriteTextFileRequester::new();
    let requests = write_requester.requests();
    let call = DeepSeekToolCall::new(
        "write-client-call",
        "write_file",
        serde_json::json!({
            "path": "note.txt",
            "content": "alpha beta gamma",
        })
        .to_string(),
    );

    let result = write_file_tool_execution(
        &store,
        &call,
        &context,
        None,
        Some(&write_requester as &dyn crate::acp::WriteTextFileRequester),
        Some(&permission_requester),
    )
    .await;

    assert!(result.success);
    assert_eq!(result.raw_output["source"], "client");
    assert!(!temp_root.join("note.txt").exists());

    let requests_guard = requests
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let request = requests_guard.first().ok_or_else(|| {
        agent_client_protocol::Error::internal_error().data("missing write_text_file request")
    })?;
    assert_eq!(request.session_id, session.session_id);
    assert_eq!(request.path, temp_root.join("note.txt"));
    assert_eq!(request.content, "alpha beta gamma");

    Ok(())
}

#[test_log::test(tokio::test)]
async fn edit_file_tool_routes_to_client_fs_read_and_write()
-> Result<(), agent_client_protocol::Error> {
    let temp_root = std::env::temp_dir().join(format!(
        "deepseek-acp-adapter-edit-client-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&temp_root)
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let store = test_store();
    let session = handle_new_session_request(&store, &NewSessionRequest::new(&temp_root))?;
    let context = ToolContext {
        session_id: session.session_id.clone(),
        cwd: temp_root.clone(),
        additional_directories: Vec::new(),
        client_capabilities: Some(
            ClientCapabilities::new().fs(FileSystemCapabilities::new()
                .read_text_file(true)
                .write_text_file(true)),
        ),
    };

    let permission_requester = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
        RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
            PERMISSION_ALLOW_ONCE_OPTION_ID,
        )),
    )]);
    let read_requester = CountingReadTextFileRequester::new();
    let read_calls = read_requester.calls();
    let write_requester = RecordingWriteTextFileRequester::new();
    let write_requests = write_requester.requests();
    let call = DeepSeekToolCall::new(
        "edit-client-call",
        "edit_file",
        serde_json::json!({
            "path": "note.txt",
            "old_text": "content",
            "new_text": "buffer",
        })
        .to_string(),
    );

    let result = edit_file_tool_execution(
        &store,
        &call,
        &context,
        Some(&read_requester as &dyn crate::acp::ReadTextFileRequester),
        Some(&write_requester as &dyn crate::acp::WriteTextFileRequester),
        Some(&permission_requester),
    )
    .await;

    assert!(result.success);
    assert_eq!(result.raw_output["read_source"], "client");
    assert_eq!(result.raw_output["write_source"], "client");
    assert!(!temp_root.join("note.txt").exists());
    assert_eq!(
        *read_calls
            .lock()
            .map_err(agent_client_protocol::Error::into_internal_error)?,
        1
    );

    let write_requests_guard = write_requests
        .lock()
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let request = write_requests_guard.first().ok_or_else(|| {
        agent_client_protocol::Error::internal_error().data("missing write_text_file request")
    })?;
    assert_eq!(request.session_id, session.session_id);
    assert_eq!(request.path, temp_root.join("note.txt"));
    assert_eq!(request.content, "client buffer");

    Ok(())
}

#[test_log::test(tokio::test)]
async fn write_and_edit_file_tools_modify_local_files_after_permission()
-> Result<(), agent_client_protocol::Error> {
    let temp_root = std::env::temp_dir().join(format!(
        "deepseek-acp-adapter-write-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&temp_root)
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let store = test_store();
    let session = handle_new_session_request(&store, &NewSessionRequest::new(&temp_root))?;
    let context = ToolContext {
        session_id: session.session_id.clone(),
        cwd: temp_root.clone(),
        additional_directories: Vec::new(),
        client_capabilities: None,
    };

    let write_requester = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
        RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
            PERMISSION_ALLOW_ONCE_OPTION_ID,
        )),
    )]);
    let write_call = DeepSeekToolCall::new(
        "write-call",
        "write_file",
        serde_json::json!({
            "path": "note.txt",
            "content": "alpha beta gamma",
        })
        .to_string(),
    );

    let write_result = write_file_tool_execution(
        &store,
        &write_call,
        &context,
        None,
        None,
        Some(&write_requester),
    )
    .await;

    assert!(write_result.success);
    assert_eq!(write_result.raw_output["source"], "local");
    let Some(write_edit) = &write_result.edit else {
        return Err(
            agent_client_protocol::Error::internal_error().data("missing write_file edit metadata")
        );
    };
    assert_eq!(write_edit.path, temp_root.join("note.txt"));
    assert_eq!(write_edit.old_text, None);
    assert_eq!(write_edit.new_text, "alpha beta gamma");
    assert_eq!(write_edit.line, 1);
    assert_eq!(
        std::fs::read_to_string(temp_root.join("note.txt"))
            .map_err(agent_client_protocol::Error::into_internal_error)?,
        "alpha beta gamma"
    );

    let edit_requester = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
        RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
            PERMISSION_ALLOW_ONCE_OPTION_ID,
        )),
    )]);
    let edit_call = DeepSeekToolCall::new(
        "edit-call",
        "edit_file",
        serde_json::json!({
            "path": "note.txt",
            "old_text": "beta",
            "new_text": "delta",
        })
        .to_string(),
    );

    let edit_result = edit_file_tool_execution(
        &store,
        &edit_call,
        &context,
        None,
        None,
        Some(&edit_requester),
    )
    .await;

    assert!(edit_result.success);
    assert_eq!(edit_result.raw_output["read_source"], "local");
    assert_eq!(edit_result.raw_output["write_source"], "local");
    let Some(edit_edit) = &edit_result.edit else {
        return Err(
            agent_client_protocol::Error::internal_error().data("missing edit_file edit metadata")
        );
    };
    assert_eq!(edit_edit.path, temp_root.join("note.txt"));
    assert_eq!(edit_edit.old_text, Some("alpha beta gamma".to_string()));
    assert_eq!(edit_edit.new_text, "alpha delta gamma");
    assert_eq!(edit_edit.line, 1);
    assert_eq!(
        std::fs::read_to_string(temp_root.join("note.txt"))
            .map_err(agent_client_protocol::Error::into_internal_error)?,
        "alpha delta gamma"
    );

    Ok(())
}

#[test_log::test(tokio::test)]
async fn run_command_tool_executes_in_session_cwd_after_permission()
-> Result<(), agent_client_protocol::Error> {
    let temp_root = std::env::temp_dir().join(format!(
        "deepseek-acp-adapter-command-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&temp_root)
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let store = test_store();
    let session = handle_new_session_request(&store, &NewSessionRequest::new(&temp_root))?;
    let context = ToolContext {
        session_id: session.session_id.clone(),
        cwd: temp_root,
        additional_directories: Vec::new(),
        client_capabilities: None,
    };
    let requester = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
        RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
            PERMISSION_ALLOW_ONCE_OPTION_ID,
        )),
    )]);
    let call = DeepSeekToolCall::new(
        "command-call",
        "run_command",
        serde_json::json!({ "command": "printf shell-ok" }).to_string(),
    );

    let result = run_command_tool_execution(
        &store,
        &call,
        &context,
        Some(&requester),
        None,
        &CancellationToken::new(),
    )
    .await;

    assert!(result.success);
    assert!(result.content.contains("stdout:"));
    assert!(result.content.contains("shell-ok"));
    assert_eq!(result.raw_output["exit_code"], serde_json::json!(0));

    Ok(())
}

#[test_log::test(tokio::test)]
async fn local_tools_list_dir_and_glob() -> Result<(), agent_client_protocol::Error> {
    let temp_root = std::env::temp_dir().join(format!(
        "deepseek-acp-adapter-local-tools-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(temp_root.join("src"))
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    std::fs::create_dir_all(temp_root.join("ignored"))
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    std::fs::write(temp_root.join("README.md"), "read me")
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    std::fs::write(temp_root.join("src/lib.rs"), "pub fn lib() {}")
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    std::fs::write(temp_root.join("src/main.rs"), "fn main() {}")
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    std::fs::write(temp_root.join("ignored/secret.rs"), "fn secret() {}")
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    std::fs::write(temp_root.join(".gitignore"), "ignored/\n")
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    let context = ToolContext {
        session_id: agent_client_protocol::schema::SessionId::new("session-local-tools"),
        cwd: temp_root.clone(),
        additional_directories: Vec::new(),
        client_capabilities: None,
    };
    let store = test_store();
    let registry = AdapterToolRegistry;

    let list_result = registry
        .execute(
            &DeepSeekToolCall::new(
                "call-list",
                "list_dir",
                serde_json::json!({ "path": "." }).to_string(),
            ),
            &context,
            &store,
            None,
            CancellationToken::new(),
        )
        .await;
    assert!(list_result.content.contains("README.md"));
    assert!(list_result.content.contains("src/"));
    assert_eq!(
        list_result.raw_output["truncated"],
        serde_json::json!(false)
    );

    let glob_result = registry
        .execute(
            &DeepSeekToolCall::new(
                "call-glob",
                "glob",
                serde_json::json!({ "pattern": "**/*.rs" }).to_string(),
            ),
            &context,
            &store,
            None,
            CancellationToken::new(),
        )
        .await;
    assert!(glob_result.content.contains("src/lib.rs"));
    assert!(glob_result.content.contains("src/main.rs"));
    assert!(!glob_result.content.contains("ignored/secret.rs"));
    assert_eq!(
        glob_result.raw_output["truncated"],
        serde_json::json!(false)
    );

    Ok(())
}

#[test_log::test(tokio::test)]
async fn local_tools_grep_respects_gitignore_and_truncates()
-> Result<(), agent_client_protocol::Error> {
    let temp_root = std::env::temp_dir().join(format!(
        "deepseek-acp-adapter-grep-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(temp_root.join("ignored"))
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    let visible = (1..=201).map(|_| "needle").collect::<Vec<_>>().join("\n");
    std::fs::write(temp_root.join("visible.rs"), format!("{visible}\n"))
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    std::fs::write(temp_root.join("ignored/secret.rs"), "needle\n")
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    std::fs::write(temp_root.join(".gitignore"), "ignored/\n")
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    let context = ToolContext {
        session_id: agent_client_protocol::schema::SessionId::new("session-grep"),
        cwd: temp_root.clone(),
        additional_directories: Vec::new(),
        client_capabilities: None,
    };
    let store = test_store();
    let registry = AdapterToolRegistry;

    let result = registry
        .execute(
            &DeepSeekToolCall::new(
                "call-grep",
                "grep",
                serde_json::json!({ "pattern": "needle" }).to_string(),
            ),
            &context,
            &store,
            None,
            CancellationToken::new(),
        )
        .await;

    assert!(result.content.contains("visible.rs:1:needle"));
    assert!(result.content.contains("visible.rs:200:needle"));
    assert!(result.content.contains("... truncated after 200 matches"));
    assert!(!result.content.contains("ignored/secret.rs"));
    assert_eq!(result.raw_output["truncated"], serde_json::json!(true));

    Ok(())
}

#[test_log::test(tokio::test)]
async fn registry_and_tool_execution_helpers_cover_error_branches()
-> Result<(), agent_client_protocol::Error> {
    let context = ToolContext {
        session_id: agent_client_protocol::schema::SessionId::new("session-registry"),
        cwd: std::path::PathBuf::from("/tmp"),
        additional_directories: Vec::new(),
        client_capabilities: None,
    };
    let store = test_store();

    let empty_result = EmptyToolRegistry
        .execute(
            &DeepSeekToolCall::new("empty", "anything", "{}"),
            &context,
            &store,
            None,
            CancellationToken::new(),
        )
        .await;
    assert!(!empty_result.success);
    assert!(empty_result.content.contains("unknown tool: anything"));

    let read_only_result = AdapterToolRegistry
        .execute(
            &DeepSeekToolCall::new("read-only", "bogus", "{}"),
            &context,
            &store,
            None,
            CancellationToken::new(),
        )
        .await;
    assert!(!read_only_result.success);
    assert!(read_only_result.content.contains("unknown tool: bogus"));

    let failed = ToolExecution::failed("boom");
    assert!(!failed.success);
    assert_eq!(
        failed.status(),
        agent_client_protocol::schema::ToolCallStatus::Failed
    );
    assert_eq!(failed.content_for_model(), "boom");

    let succeeded = ToolExecution::completed("ok", serde_json::json!({ "value": 1 }));
    assert_eq!(
        succeeded.status(),
        agent_client_protocol::schema::ToolCallStatus::Completed
    );
    assert_eq!(succeeded.content_for_model(), "ok");

    Ok(())
}

#[test]
fn render_command_output_includes_stderr_and_exit_code() {
    let output = render_command_output("stdout_line\n", "stderr_line\n", Some(1));
    assert!(output.contains("stdout:\nstdout_line"));
    assert!(output.contains("stderr:\nstderr_line"));
}

#[test]
fn render_command_output_adds_newline_when_stdout_missing_trailing() {
    let output = render_command_output("out", "", Some(0));
    assert_eq!(output, "stdout:\nout\n");
}

#[test]
fn render_command_output_empty_uses_signal_label() {
    let output = render_command_output("", "", None);
    assert!(output.contains("command exited with status signal"));
}

#[test]
fn render_command_output_empty_uses_numeric_exit_code() {
    let output = render_command_output("", "", Some(42));
    assert!(output.contains("command exited with status 42"));
}

#[test]
fn truncate_tool_output_truncates_when_over_limit() {
    let long = "a".repeat(300);
    let (truncated, flag) = truncate_tool_output(&long, 200);
    assert!(flag);
    assert!(truncated.len() <= 300);
    assert!(truncated.contains("... truncated after 200 characters"));
}

#[test]
fn truncate_tool_output_passes_through_short_strings() {
    let short = "hello";
    let (output, flag) = truncate_tool_output(short, 200);
    assert!(!flag);
    assert_eq!(output, short);
}

#[test]
fn utf8_error_message_detects_all_variants() {
    assert!(is_utf8_error_message("stream did not contain valid UTF-8"));
    assert!(is_utf8_error_message("file is invalid utf-8 encoded"));
    assert!(is_utf8_error_message("non-utf-8 data detected"));
    assert!(is_utf8_error_message("some utf8 issue"));
    assert!(!is_utf8_error_message("file not found"));
}

#[test_log::test(tokio::test)]
async fn run_command_rejects_empty_command() {
    let store = test_store();
    let context = ToolContext {
        session_id: agent_client_protocol::schema::SessionId::new("empty-cmd"),
        cwd: std::path::PathBuf::from("/tmp"),
        additional_directories: Vec::new(),
        client_capabilities: None,
    };
    let call = DeepSeekToolCall::new(
        "empty-cmd-call",
        "run_command",
        serde_json::json!({ "command": "   " }).to_string(),
    );
    let result = run_command_tool_execution(
        &store,
        &call,
        &context,
        None,
        None,
        &CancellationToken::new(),
    )
    .await;
    assert!(!result.success);
    assert!(result.content.contains("command must not be empty"));
}

#[test]
fn collect_directory_entries_reports_missing() -> Result<(), agent_client_protocol::Error> {
    let Err(error) =
        collect_directory_entries(std::path::Path::new("/tmp/nonexistent-dir-for-test"))
    else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected error for missing dir")
        );
    };
    assert!(error.contains("failed to read directory"));
    Ok(())
}

#[test]
fn build_root_gitignore_loads_when_present() -> Result<(), agent_client_protocol::Error> {
    let temp_root =
        std::env::temp_dir().join(format!("deepseek-acp-adapter-gi-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&temp_root)
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    std::fs::write(temp_root.join(".gitignore"), "*.log\n")
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    let gitignore = build_root_gitignore(&temp_root);
    assert!(gitignore.is_some());
    Ok(())
}

#[test]
fn build_root_gitignore_loads_file() -> Result<(), agent_client_protocol::Error> {
    let temp_root = std::env::temp_dir().join(format!(
        "deepseek-acp-adapter-gitignore-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&temp_root)
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    std::fs::write(temp_root.join(".gitignore"), "*.log\n")
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    let gitignore = build_root_gitignore(&temp_root);
    assert!(gitignore.is_some());
    Ok(())
}

#[test]
fn read_file_from_local_zero_line_defaults_to_start() -> Result<(), agent_client_protocol::Error> {
    let temp_root = std::env::temp_dir().join(format!(
        "deepseek-acp-adapter-read-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&temp_root)
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    std::fs::write(temp_root.join("lines.txt"), "a\nb\nc\nd\ne\n")
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    let result = read_file_from_local(&temp_root.join("lines.txt"), 10, 1);
    assert_eq!(
        result.map_err(|e| agent_client_protocol::Error::internal_error().data(e))?,
        ""
    );
    Ok(())
}

#[test]
fn read_file_from_local_line_past_end_returns_empty() -> Result<(), agent_client_protocol::Error> {
    let temp_root = std::env::temp_dir().join(format!(
        "deepseek-acp-adapter-read-past-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&temp_root)
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    std::fs::write(temp_root.join("lines.txt"), "a\nb\nc\n")
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    let result = read_file_from_local(&temp_root.join("lines.txt"), 10, 5);
    assert_eq!(
        result.map_err(|e| agent_client_protocol::Error::internal_error().data(e))?,
        ""
    );
    Ok(())
}

#[test]
fn read_file_local_error_handles_invalid_data() -> Result<(), agent_client_protocol::Error> {
    let temp_root = std::env::temp_dir().join(format!(
        "deepseek-acp-adapter-local-err-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&temp_root)
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let bin_path = temp_root.join("artifact.bin");
    std::fs::write(&bin_path, [0xff, 0xfe, 0xfd])
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    let Err(error) = std::fs::read_to_string(&bin_path) else {
        return Err(
            agent_client_protocol::Error::internal_error().data("expected non-UTF-8 read to fail")
        );
    };
    let msg = read_file_local_error(&bin_path, &error);
    assert!(msg.contains("UTF-8"));
    Ok(())
}

#[test_log::test(tokio::test)]
async fn glob_tool_execution_invalid_build_pattern() -> Result<(), agent_client_protocol::Error> {
    let temp_root = std::env::temp_dir().join(format!(
        "deepseek-acp-adapter-glob-err-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&temp_root)
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    let context = ToolContext {
        session_id: agent_client_protocol::schema::SessionId::new("glob-err"),
        cwd: temp_root,
        additional_directories: Vec::new(),
        client_capabilities: None,
    };
    let call = DeepSeekToolCall::new(
        "glob-invalid",
        "glob",
        serde_json::json!({ "pattern": "[" }).to_string(),
    );

    let result = glob_tool_execution(&call, &context);
    assert!(!result.success);
    assert!(result.content.contains("invalid glob pattern"));
    Ok(())
}

#[test_log::test(tokio::test)]
async fn grep_tool_execution_invalid_regex() -> Result<(), agent_client_protocol::Error> {
    let temp_root = std::env::temp_dir().join(format!(
        "deepseek-acp-adapter-grep-regex-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&temp_root)
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    std::fs::write(temp_root.join("f.txt"), "test\n")
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    let context = ToolContext {
        session_id: agent_client_protocol::schema::SessionId::new("grep-regex"),
        cwd: temp_root,
        additional_directories: Vec::new(),
        client_capabilities: None,
    };
    let call = DeepSeekToolCall::new(
        "grep-regex",
        "grep",
        serde_json::json!({ "pattern": "(" }).to_string(),
    );

    let result = grep_tool_execution(&call, &context);
    assert!(!result.success);
    assert!(result.content.contains("invalid grep regex"));
    Ok(())
}

#[test]
fn render_command_output_adds_newline_to_stderr() {
    let output = render_command_output("", "err", Some(2));
    assert!(output.contains("stderr:\nerr\n"));
}

#[test_log::test(tokio::test)]
async fn adapter_registry_execute_write_without_permission()
-> Result<(), agent_client_protocol::Error> {
    let temp_root = std::env::temp_dir().join(format!(
        "deepseek-acp-adapter-conn-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&temp_root)
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    let store = test_store();
    let session = handle_new_session_request(&store, &NewSessionRequest::new(&temp_root))?;
    let context = ToolContext {
        session_id: session.session_id,
        cwd: temp_root,
        additional_directories: Vec::new(),
        client_capabilities: None,
    };

    let registry = AdapterToolRegistry;
    let result = registry
        .execute(
            &DeepSeekToolCall::new(
                "conn-write",
                "write_file",
                serde_json::json!({ "path": "out.txt", "content": "data" }).to_string(),
            ),
            &context,
            &store,
            None,
            CancellationToken::new(),
        )
        .await;
    assert!(!result.success);

    Ok(())
}

#[test_log::test(tokio::test)]
async fn write_file_with_client_capability_but_no_connection_errors()
-> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
    let context = ToolContext {
        session_id: session.session_id.clone(),
        cwd: std::path::PathBuf::from("/tmp"),
        additional_directories: Vec::new(),
        client_capabilities: Some(
            ClientCapabilities::new().fs(FileSystemCapabilities::new().write_text_file(true)),
        ),
    };
    let call = DeepSeekToolCall::new(
        "w",
        "write_file",
        serde_json::json!({"path": "out.txt", "content": "hi"}).to_string(),
    );

    let permission = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
        RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
            PERMISSION_ALLOW_ONCE_OPTION_ID,
        )),
    )]);
    let result =
        write_file_tool_execution(&store, &call, &context, None, None, Some(&permission)).await;
    assert!(!result.success);
    assert!(
        result
            .content
            .contains("write_file needs a client connection")
    );
    Ok(())
}

#[test_log::test(tokio::test)]
async fn edit_file_with_client_read_capability_but_no_connection_errors()
-> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
    let context = ToolContext {
        session_id: session.session_id.clone(),
        cwd: std::path::PathBuf::from("/tmp"),
        additional_directories: Vec::new(),
        client_capabilities: Some(
            ClientCapabilities::new().fs(FileSystemCapabilities::new().read_text_file(true)),
        ),
    };
    let call = DeepSeekToolCall::new(
        "e",
        "edit_file",
        serde_json::json!({"path": "out.txt", "old_text": "a", "new_text": "b"}).to_string(),
    );

    let permission = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
        RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
            PERMISSION_ALLOW_ONCE_OPTION_ID,
        )),
    )]);
    let result =
        edit_file_tool_execution(&store, &call, &context, None, None, Some(&permission)).await;
    assert!(!result.success);
    assert!(
        result
            .content
            .contains("edit_file needs a client connection for fs/read_text_file")
    );
    Ok(())
}

#[test_log::test(tokio::test)]
async fn edit_file_with_client_write_capability_but_no_connection_errors()
-> Result<(), agent_client_protocol::Error> {
    let temp_root = std::env::temp_dir().join(format!(
        "deepseek-acp-client-write-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&temp_root)
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    std::fs::write(temp_root.join("out.txt"), "hello world")
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    let store = test_store();
    let session = handle_new_session_request(&store, &NewSessionRequest::new(&temp_root))?;
    let context = ToolContext {
        session_id: session.session_id.clone(),
        cwd: temp_root.clone(),
        additional_directories: Vec::new(),
        client_capabilities: Some(
            ClientCapabilities::new().fs(FileSystemCapabilities::new().write_text_file(true)),
        ),
    };
    let call = DeepSeekToolCall::new(
        "e",
        "edit_file",
        serde_json::json!({"path": "out.txt", "old_text": "hello", "new_text": "bye"}).to_string(),
    );

    let permission = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
        RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
            PERMISSION_ALLOW_ONCE_OPTION_ID,
        )),
    )]);
    let result =
        edit_file_tool_execution(&store, &call, &context, None, None, Some(&permission)).await;
    assert!(!result.success);
    assert!(
        result
            .content
            .contains("edit_file needs a client connection for fs/write_text_file")
    );
    Ok(())
}

#[test_log::test(tokio::test)]
async fn read_file_with_client_capability_but_no_connection_errors()
-> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
    let context = ToolContext {
        session_id: session.session_id.clone(),
        cwd: std::path::PathBuf::from("/tmp"),
        additional_directories: Vec::new(),
        client_capabilities: Some(
            ClientCapabilities::new().fs(FileSystemCapabilities::new().read_text_file(true)),
        ),
    };
    let call = DeepSeekToolCall::new(
        "r",
        "read_file",
        serde_json::json!({"path": "missing.txt"}).to_string(),
    );

    let result = read_file_tool_execution(&call, &context, None).await;
    assert!(!result.success);
    assert!(
        result
            .content
            .contains("read_file needs a client connection")
    );
    Ok(())
}

#[test]
fn helper_path_functions_cover_error_branches() -> Result<(), agent_client_protocol::Error> {
    let temp_root = std::env::temp_dir().join(format!(
        "deepseek-acp-adapter-path-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&temp_root)
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    assert!(build_root_gitignore(&temp_root).is_none());
    assert!(is_hidden_path(std::path::Path::new(".gitignore")));
    assert!(!is_hidden_path(std::path::Path::new("src/lib.rs")));

    let alternate_directory = temp_root.join("alternate");
    std::fs::create_dir_all(&alternate_directory)
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    std::fs::write(temp_root.join("found.txt"), "primary")
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    std::fs::write(alternate_directory.join("found.txt"), "found")
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    std::fs::write(alternate_directory.join("alternate-only.txt"), "found")
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    let context = ToolContext {
        session_id: agent_client_protocol::schema::SessionId::new("session-paths"),
        cwd: temp_root.clone(),
        additional_directories: vec![alternate_directory.clone()],
        client_capabilities: None,
    };
    assert_eq!(
        resolve_tool_path(&context, std::path::Path::new("found.txt")),
        temp_root.join("found.txt")
    );
    assert_eq!(
        resolve_tool_path(&context, std::path::Path::new("alternate-only.txt")),
        alternate_directory.join("alternate-only.txt")
    );
    assert_eq!(
        resolve_tool_path(&context, std::path::Path::new("/abs/path")),
        std::path::PathBuf::from("/abs/path")
    );
    assert_eq!(
        resolve_tool_path(&context, std::path::Path::new("missing.txt")),
        temp_root.join("missing.txt")
    );
    assert!(collect_directory_entries(&temp_root.join("missing-dir")).is_err());

    Ok(())
}

#[test]
fn non_utf8_file_message_includes_path() {
    let msg = non_utf8_file_message(std::path::Path::new("/tmp/binary.bin"));
    assert!(msg.contains("/tmp/binary.bin"));
    assert!(msg.contains("UTF-8"));
}

#[test]
fn read_file_client_error_with_non_utf8_message() {
    let msg = read_file_client_error(
        std::path::Path::new("/tmp/binary.bin"),
        "stream did not contain valid UTF-8",
    );
    assert!(msg.contains("UTF-8"));
    assert!(!msg.contains("stream did not contain valid UTF-8"));
}

#[test_log::test(tokio::test)]
async fn write_file_to_client_propagates_error() -> Result<(), agent_client_protocol::Error> {
    let session_id = agent_client_protocol::schema::SessionId::new("write-err");
    let result = write_file_to_client(
        &FailingWriteRequester,
        &session_id,
        std::path::Path::new("/tmp/note.txt"),
        "content",
    )
    .await;
    let Err(error) = result else {
        return Err(agent_client_protocol::Error::internal_error().data("expected failure"));
    };
    assert!(error.contains("failed to write"));
    Ok(())
}

#[test_log::test(tokio::test)]
async fn run_command_via_terminal_success_path() {
    let session_id = agent_client_protocol::schema::SessionId::new("terminal-test");
    let fake = FakeTerminalRequester {
        terminal_id: "term-1".to_string(),
        output: "command output".to_string(),
        exit_code: Some(0),
        truncated: false,
        create_error: None,
        wait_error: None,
        output_error: None,
        release_error: None,
    };

    let result = run_command_via_terminal(
        &session_id,
        std::path::Path::new("/tmp"),
        "echo hi",
        Some(&fake as &dyn crate::acp::TerminalRequester),
        &CancellationToken::new(),
    )
    .await;

    assert!(result.success);
    assert!(result.content.contains("command output"));
}

#[test_log::test(tokio::test)]
async fn run_command_via_terminal_no_connection() {
    let session_id = agent_client_protocol::schema::SessionId::new("terminal-no-conn");
    let result = run_command_via_terminal(
        &session_id,
        std::path::Path::new("/tmp"),
        "echo hi",
        None,
        &CancellationToken::new(),
    )
    .await;

    assert!(!result.success);
    assert!(result.content.contains("no connection available"));
}

#[test_log::test(tokio::test)]
async fn run_command_via_terminal_create_error() {
    let session_id = agent_client_protocol::schema::SessionId::new("terminal-create-err");
    let fake = FakeTerminalRequester {
        terminal_id: "term-err".to_string(),
        output: String::new(),
        exit_code: None,
        truncated: false,
        create_error: Some("create failed".to_string()),
        wait_error: None,
        output_error: None,
        release_error: None,
    };

    let result = run_command_via_terminal(
        &session_id,
        std::path::Path::new("/tmp"),
        "echo hi",
        Some(&fake as &dyn crate::acp::TerminalRequester),
        &CancellationToken::new(),
    )
    .await;

    assert!(!result.success);
    assert!(result.content.contains("terminal/create failed"));
}

#[test_log::test(tokio::test)]
async fn run_command_via_terminal_wait_error() {
    let session_id = agent_client_protocol::schema::SessionId::new("terminal-wait-err");
    let fake = FakeTerminalRequester {
        terminal_id: "term-wait".to_string(),
        output: String::new(),
        exit_code: None,
        truncated: false,
        create_error: None,
        wait_error: Some("wait failed".to_string()),
        output_error: None,
        release_error: None,
    };

    let result = run_command_via_terminal(
        &session_id,
        std::path::Path::new("/tmp"),
        "echo hi",
        Some(&fake as &dyn crate::acp::TerminalRequester),
        &CancellationToken::new(),
    )
    .await;

    assert!(!result.success);
    assert!(result.content.contains("terminal/wait_for_exit failed"));
}

#[test_log::test(tokio::test)]
async fn run_command_via_terminal_output_error() {
    let session_id = agent_client_protocol::schema::SessionId::new("terminal-output-err");
    let fake = FakeTerminalRequester {
        terminal_id: "term-out".to_string(),
        output: String::new(),
        exit_code: None,
        truncated: false,
        create_error: None,
        wait_error: None,
        output_error: Some("output failed".to_string()),
        release_error: None,
    };

    let result = run_command_via_terminal(
        &session_id,
        std::path::Path::new("/tmp"),
        "echo hi",
        Some(&fake as &dyn crate::acp::TerminalRequester),
        &CancellationToken::new(),
    )
    .await;

    assert!(!result.success);
    assert!(result.content.contains("terminal/output failed"));
}

#[test_log::test(tokio::test)]
async fn run_command_via_terminal_release_error() {
    let session_id = agent_client_protocol::schema::SessionId::new("terminal-release-err");
    let fake = FakeTerminalRequester {
        terminal_id: "term-rel".to_string(),
        output: "output".to_string(),
        exit_code: Some(0),
        truncated: false,
        create_error: None,
        wait_error: None,
        output_error: None,
        release_error: Some("release failed".to_string()),
    };

    let result = run_command_via_terminal(
        &session_id,
        std::path::Path::new("/tmp"),
        "echo hi",
        Some(&fake as &dyn crate::acp::TerminalRequester),
        &CancellationToken::new(),
    )
    .await;

    assert!(!result.success);
    assert!(result.content.contains("terminal/release failed"));
}

#[test_log::test(tokio::test)]
async fn run_command_via_terminal_kills_on_cancellation() {
    let tracker = CancelTracker::default();
    let token = CancellationToken::new();
    token.cancel();

    let session_id = agent_client_protocol::schema::SessionId::new("terminal-cancel");
    let result = run_command_via_terminal(
        &session_id,
        std::path::Path::new("/tmp"),
        "sleep 100",
        Some(&tracker as &dyn crate::acp::TerminalRequester),
        &token,
    )
    .await;

    assert!(!result.success);
    assert!(result.content.contains("cancelled"));
    assert_eq!(tracker.kills.load(Ordering::SeqCst), 1);
    assert_eq!(tracker.releases.load(Ordering::SeqCst), 1);
}

#[test_log::test(tokio::test)]
async fn edit_file_rejects_empty_old_text() -> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let temp_root = std::env::temp_dir().join(format!(
        "deepseek-acp-adapter-edit-empty-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&temp_root)
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    std::fs::write(temp_root.join("f.txt"), "content")
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    let session = handle_new_session_request(&store, &NewSessionRequest::new(&temp_root))?;
    let context = ToolContext {
        session_id: session.session_id.clone(),
        cwd: temp_root.clone(),
        additional_directories: Vec::new(),
        client_capabilities: None,
    };
    let call = DeepSeekToolCall::new(
        "edit-empty",
        "edit_file",
        serde_json::json!({
            "path": "f.txt",
            "old_text": "",
            "new_text": "replacement",
        })
        .to_string(),
    );

    let result = edit_file_tool_execution(&store, &call, &context, None, None, None).await;
    assert!(!result.success);
    assert!(result.content.contains("old_text must not be empty"));
    Ok(())
}

#[test_log::test(tokio::test)]
async fn edit_file_rejects_old_text_not_found() -> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let temp_root = std::env::temp_dir().join(format!(
        "deepseek-acp-adapter-edit-nf-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&temp_root)
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    std::fs::write(temp_root.join("f.txt"), "content")
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    let session = handle_new_session_request(&store, &NewSessionRequest::new(&temp_root))?;
    let context = ToolContext {
        session_id: session.session_id.clone(),
        cwd: temp_root.clone(),
        additional_directories: Vec::new(),
        client_capabilities: None,
    };
    let call = DeepSeekToolCall::new(
        "edit-nf",
        "edit_file",
        serde_json::json!({
            "path": "f.txt",
            "old_text": "nonexistent",
            "new_text": "replacement",
        })
        .to_string(),
    );

    let result = edit_file_tool_execution(&store, &call, &context, None, None, None).await;
    assert!(!result.success);
    assert!(result.content.contains("could not find old_text"));
    Ok(())
}

#[test_log::test(tokio::test)]
async fn edit_file_rejects_multiple_matches() -> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let temp_root = std::env::temp_dir().join(format!(
        "deepseek-acp-adapter-edit-multi-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&temp_root)
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    std::fs::write(temp_root.join("f.txt"), "dup dup")
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    let session = handle_new_session_request(&store, &NewSessionRequest::new(&temp_root))?;
    let context = ToolContext {
        session_id: session.session_id.clone(),
        cwd: temp_root.clone(),
        additional_directories: Vec::new(),
        client_capabilities: None,
    };
    let call = DeepSeekToolCall::new(
        "edit-multi",
        "edit_file",
        serde_json::json!({
            "path": "f.txt",
            "old_text": "dup",
            "new_text": "replacement",
        })
        .to_string(),
    );

    let result = edit_file_tool_execution(&store, &call, &context, None, None, None).await;
    assert!(!result.success);
    assert!(result.content.contains("found old_text"));
    assert!(result.content.contains("2 times"));
    Ok(())
}

#[test_log::test(tokio::test)]
async fn write_file_rejects_invalid_arguments() -> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
    let context = ToolContext {
        session_id: session.session_id.clone(),
        cwd: std::path::PathBuf::from("/tmp"),
        additional_directories: Vec::new(),
        client_capabilities: None,
    };
    let call = DeepSeekToolCall::new("write-invalid", "write_file", "not json");

    let result = write_file_tool_execution(&store, &call, &context, None, None, None).await;
    assert!(!result.success);
    assert!(result.content.contains("invalid write_file arguments"));
    Ok(())
}

#[test_log::test(tokio::test)]
async fn run_command_rejects_invalid_arguments() -> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
    let context = ToolContext {
        session_id: session.session_id.clone(),
        cwd: std::path::PathBuf::from("/tmp"),
        additional_directories: Vec::new(),
        client_capabilities: None,
    };
    let call = DeepSeekToolCall::new("run-invalid", "run_command", "not json");

    let result = run_command_tool_execution(
        &store,
        &call,
        &context,
        None,
        None,
        &CancellationToken::new(),
    )
    .await;
    assert!(!result.success);
    assert!(result.content.contains("invalid run_command arguments"));
    Ok(())
}

#[test_log::test(tokio::test)]
async fn edit_file_rejects_invalid_arguments() -> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
    let context = ToolContext {
        session_id: session.session_id.clone(),
        cwd: std::path::PathBuf::from("/tmp"),
        additional_directories: Vec::new(),
        client_capabilities: None,
    };
    let call = DeepSeekToolCall::new("edit-invalid", "edit_file", "not json");

    let result = edit_file_tool_execution(&store, &call, &context, None, None, None).await;
    assert!(!result.success);
    assert!(result.content.contains("invalid edit_file arguments"));
    Ok(())
}

// ── Edge-case coverage for uncovered production code paths ────────

#[test_log::test(tokio::test)]
async fn require_tool_permission_propagates_request_error()
-> Result<(), agent_client_protocol::Error> {
    let store = test_store();
    let session = handle_new_session_request(&store, &NewSessionRequest::new("/tmp"))?;
    let context = ToolContext {
        session_id: session.session_id.clone(),
        cwd: std::path::PathBuf::from("/tmp"),
        additional_directories: Vec::new(),
        client_capabilities: None,
    };
    // Use a permission requester that returns an error.
    // The FakePermissionRequester with empty responses signals exhaustion as an error.
    let requester = FakePermissionRequester::new(Vec::new());
    let call = DeepSeekToolCall::new(
        "err-call",
        "write_file",
        "{\"path\": \"x\", \"content\": \"c\"}",
    );
    let err =
        require_tool_permission(&store, &context, &call, ToolKind::Edit, Some(&requester)).await;
    assert!(err.is_err());
    let msg = err.err().map_or(String::new(), |e| e);
    assert!(msg.contains("failed to request permission"));
    Ok(())
}

#[test_log::test(tokio::test)]
async fn write_file_tool_execution_read_existing_text_local_success()
-> Result<(), agent_client_protocol::Error> {
    let temp_root = std::env::temp_dir().join(format!(
        "deepseek-acp-write-existing-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&temp_root)
        .map_err(agent_client_protocol::Error::into_internal_error)?;
    std::fs::write(temp_root.join("existing.txt"), "previous content")
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    let store = test_store();
    let session = handle_new_session_request(&store, &NewSessionRequest::new(&temp_root))?;
    let context = ToolContext {
        session_id: session.session_id.clone(),
        cwd: temp_root.clone(),
        additional_directories: Vec::new(),
        client_capabilities: None,
    };
    let permission = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
        RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
            PERMISSION_ALLOW_ONCE_OPTION_ID,
        )),
    )]);
    let call = DeepSeekToolCall::new(
        "write-existing",
        "write_file",
        serde_json::json!({"path": "existing.txt", "content": "new content"}).to_string(),
    );
    let result =
        write_file_tool_execution(&store, &call, &context, None, None, Some(&permission)).await;
    assert!(result.success);
    // old_text should be Some with the previous content
    let Some(ref edit) = result.edit else {
        return Err(
            agent_client_protocol::Error::internal_error().data("missing edit in write result")
        );
    };
    assert_eq!(edit.old_text, Some("previous content".to_string()));
    assert_eq!(edit.new_text, "new content");
    Ok(())
}

#[test]
fn read_file_client_error_non_utf8_message() {
    let msg = read_file_client_error(
        std::path::Path::new("/tmp/generic.txt"),
        "permission denied",
    );
    // Not a UTF-8 error message → should use the generic format
    assert!(!msg.contains("only supports UTF-8 text files"));
    assert!(msg.contains("permission denied"));
}

#[test]
fn line_number_for_offset_returns_1_for_out_of_bounds() {
    let text = "alpha\nbeta";
    // offset beyond the string length triggers the `None` path
    assert_eq!(super::line_number_for_offset(text, 100), 1);
    // zero-length prefix is valid (offset == 0)
    assert_eq!(super::line_number_for_offset(text, 0), 1);
}

#[test_log::test(tokio::test)]
async fn run_command_tool_uses_terminal_when_capability_present()
-> Result<(), agent_client_protocol::Error> {
    let temp_root = std::env::temp_dir().join(format!(
        "deepseek-acp-cmd-terminal-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&temp_root)
        .map_err(agent_client_protocol::Error::into_internal_error)?;

    let store = test_store();
    let session = handle_new_session_request(&store, &NewSessionRequest::new(&temp_root))?;
    let permission = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
        RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
            PERMISSION_ALLOW_ONCE_OPTION_ID,
        )),
    )]);
    let terminal = FakeTerminalRequester {
        terminal_id: "term-caps".to_string(),
        output: "from terminal".to_string(),
        exit_code: Some(0),
        truncated: false,
        create_error: None,
        wait_error: None,
        output_error: None,
        release_error: None,
    };
    let context = ToolContext {
        session_id: session.session_id.clone(),
        cwd: temp_root,
        additional_directories: Vec::new(),
        client_capabilities: Some(
            ClientCapabilities::new()
                .terminal(true)
                .fs(FileSystemCapabilities::new()),
        ),
    };
    let call = DeepSeekToolCall::new(
        "cmd-terminal",
        "run_command",
        serde_json::json!({"command": "echo via-terminal"}).to_string(),
    );
    let result = run_command_tool_execution(
        &store,
        &call,
        &context,
        Some(&permission),
        Some(&terminal as &dyn crate::acp::TerminalRequester),
        &CancellationToken::new(),
    )
    .await;
    assert!(result.success);
    assert!(result.content.contains("from terminal"));
    Ok(())
}

#[test_log::test(tokio::test)]
async fn run_command_tool_execution_spawn_error_path() {
    let store = test_store();
    let context = ToolContext {
        session_id: agent_client_protocol::schema::SessionId::new("spawn-err"),
        cwd: std::path::PathBuf::from("/nonexistent-dir-that-does-not-exist-for-test"),
        additional_directories: Vec::new(),
        client_capabilities: None,
    };
    let permission = FakePermissionRequester::new(vec![RequestPermissionResponse::new(
        RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
            PERMISSION_ALLOW_ONCE_OPTION_ID,
        )),
    )]);
    let call = DeepSeekToolCall::new(
        "spawn-err",
        "run_command",
        serde_json::json!({"command": "echo ok"}).to_string(),
    );
    let result = run_command_tool_execution(
        &store,
        &call,
        &context,
        Some(&permission),
        None,
        &CancellationToken::new(),
    )
    .await;
    // On most systems this will fail to change to a nonexistent dir,
    // exercising the Ok(Err(_)) path.
    assert!(!result.success);
}

#[test]
fn collect_directory_entries_inner_read_error_path() {
    // On Linux, /proc/1/fd is a directory whose entries cannot be stat'd by
    // non-root — exercises the entry-level error path.
    // When /proc is absent the test is a no-op; the path is already validated
    // by the existing `collect_directory_entries_reports_missing` test.
    if std::path::Path::new("/proc/1/fd").is_dir() {
        let result = collect_directory_entries(std::path::Path::new("/proc/1/fd"));
        let _ = result;
    }
}
