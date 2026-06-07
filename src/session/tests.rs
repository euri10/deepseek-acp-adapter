use super::{PendingToolCalls, PermissionDecision, PermissionPosture, ReasoningEffort};
use agent_client_protocol::schema::SessionModeId;

#[test]
fn permission_decision_debug_impl_is_callable() {
    let decisions = [
        PermissionDecision::AllowOnce,
        PermissionDecision::AllowAlways,
        PermissionDecision::AllowByMode,
        PermissionDecision::RejectOnce,
        PermissionDecision::RejectAlways,
        PermissionDecision::Cancelled,
    ];
    for decision in &decisions {
        let _ = format!("{decision:?}");
    }
}

#[test]
fn reasoning_effort_name_and_description() {
    assert_eq!(ReasoningEffort::High.name(), "High");
    assert_eq!(ReasoningEffort::Max.name(), "Max");
    assert!(
        ReasoningEffort::High
            .description()
            .contains("Default DeepSeek")
    );
    assert!(
        ReasoningEffort::Max
            .description()
            .contains("Maximum DeepSeek")
    );
}

#[test]
fn reasoning_effort_from_value_id_rejects_unknown() {
    assert!(
        ReasoningEffort::from_value_id(&agent_client_protocol::schema::SessionConfigValueId::new(
            "bogus",
        ))
        .is_none()
    );
}

#[test]
fn pending_tool_calls_require_complete_metadata() -> Result<(), agent_client_protocol::Error> {
    use deepseek_acp_adapter::deepseek::ToolCallDelta;

    let mut missing_id = PendingToolCalls::default();
    missing_id.push(&ToolCallDelta::new(
        1,
        None,
        Some("echo".to_string()),
        Some("{}".to_string()),
    ));
    let Err(error) = missing_id.finish() else {
        return Err(agent_client_protocol::Error::internal_error()
            .data("expected missing tool call id to fail"));
    };
    assert!(error.to_string().contains("missing an id"));

    let mut missing_name = PendingToolCalls::default();
    missing_name.push(&ToolCallDelta::new(
        0,
        Some("call-1".to_string()),
        None,
        Some("{}".to_string()),
    ));
    let Err(error) = missing_name.finish() else {
        return Err(agent_client_protocol::Error::internal_error()
            .data("expected missing tool call name to fail"));
    };
    assert!(error.to_string().contains("missing a function name"));

    Ok(())
}

#[test]
fn permission_posture_helpers_cover_all_branches() {
    use crate::mcp::{is_mcp_tool_name, mcp_tool_kind};
    use agent_client_protocol::schema::ToolKind;

    assert_eq!(PermissionPosture::Ask.mode_id().0.as_ref(), "ask");
    assert_eq!(
        PermissionPosture::AcceptEdits.mode_id().0.as_ref(),
        "accept-edits"
    );
    assert_eq!(PermissionPosture::Yolo.mode_id().0.as_ref(), "yolo");
    assert_eq!(
        PermissionPosture::from_mode_id(&SessionModeId::new("ask")),
        Some(PermissionPosture::Ask)
    );
    assert_eq!(
        PermissionPosture::from_mode_id(&SessionModeId::new("accept-edits")),
        Some(PermissionPosture::AcceptEdits)
    );
    assert_eq!(
        PermissionPosture::from_mode_id(&SessionModeId::new("yolo")),
        Some(PermissionPosture::Yolo)
    );
    assert_eq!(
        PermissionPosture::from_mode_id(&SessionModeId::new("bogus")),
        None
    );
    assert!(!PermissionPosture::Ask.allows_without_prompt(ToolKind::Edit));
    assert!(PermissionPosture::AcceptEdits.allows_without_prompt(ToolKind::Edit));
    assert!(!PermissionPosture::AcceptEdits.allows_without_prompt(ToolKind::Execute));
    assert!(PermissionPosture::Yolo.allows_without_prompt(ToolKind::Execute));
    assert!(!PermissionPosture::Yolo.allows_without_prompt(ToolKind::Read));
    assert!(is_mcp_tool_name("mcp__server__tool"));
    assert_eq!(mcp_tool_kind(), ToolKind::Execute);
}
