use super::{FilesystemSessionStore, PersistedSessionMeta};
use crate::{PermissionPosture, ReasoningEffort};
use agent_client_protocol::schema::SessionId;
use deepseek_acp_adapter::deepseek::ChatMessage;
use uuid::Uuid;

#[test_log::test]
fn round_trips_session_metadata_and_history()
-> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
    let state_dir =
        std::env::temp_dir().join(format!("deepseek-acp-session-store-{}", Uuid::new_v4()));
    let cwd = state_dir.join("workspace");
    let store = FilesystemSessionStore::new(&state_dir);
    let meta = PersistedSessionMeta {
        session_id: "session-roundtrip".to_string(),
        cwd: cwd.clone(),
        additional_directories: vec![state_dir.join("extra")],
        mode: PermissionPosture::Yolo,
        model: "deepseek-v4-pro".to_string(),
        reasoning_effort: ReasoningEffort::Max,
        mcp_servers: Vec::new(),
    };

    store.persist_turn(&meta, &[ChatMessage::user("hello")])?;
    store.persist_turn(&meta, &[ChatMessage::assistant("world")])?;

    let record = store.load_record("session-roundtrip")?;
    assert_eq!(record.meta, meta);
    assert_eq!(record.history.len(), 2);
    assert_eq!(record.history[0], ChatMessage::user("hello"));
    assert_eq!(record.history[1], ChatMessage::assistant("world"));

    let listed = store.list_persisted(&cwd)?;
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].session_id, SessionId::new("session-roundtrip"));
    assert_eq!(listed[0].cwd, cwd);

    Ok(())
}

#[test_log::test]
fn rejects_session_ids_that_are_not_path_components() {
    let store = FilesystemSessionStore::new("/tmp/deepseek-acp-invalid");
    let error = store.load_record("../escape").err();
    assert!(error.is_some());
}
