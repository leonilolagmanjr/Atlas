import unittest

from models import ExecutionContext, TaskStatus
from tools import (
    ExecutionMode,
    PermissionEngine,
    PermissionLevel,
    RiskLevel,
    Tool,
    ToolMetadata,
    ToolRegistry,
    ToolResult,
)


class ExampleTool(Tool):
    metadata = ToolMetadata(
        name="example.read",
        description="Read an example value.",
        category="test",
        permission_level=PermissionLevel.READ_ONLY,
        risk_level=RiskLevel.READ_ONLY,
    )

    def execute(self, parameters):
        self.validate(parameters)
        return ToolResult(success=True, status="completed", output=parameters)


class RuntimeContractTests(unittest.TestCase):
    def test_context_snapshot_contains_task_state(self):
        context = ExecutionContext(user_input="inspect the system")
        context.goal = "Inspect system state"
        context.status = TaskStatus.RUNNING
        context.tool_calls.append({"tool": "example.read", "status": "completed"})

        snapshot = context.to_dict()

        self.assertEqual(snapshot["user_input"], "inspect the system")
        self.assertEqual(snapshot["status"], "RUNNING")
        self.assertEqual(snapshot["tool_calls"][0]["tool"], "example.read")

    def test_registry_rejects_duplicate_names(self):
        registry = ToolRegistry()
        registry.register(ExampleTool())

        with self.assertRaises(ValueError):
            registry.register(ExampleTool())

    def test_tool_validation_rejects_non_mapping_input(self):
        with self.assertRaises(TypeError):
            ExampleTool().validate([])

    def test_safe_mode_allows_read_only_tools(self):
        decision = PermissionEngine(mode=ExecutionMode.SAFE).evaluate(ExampleTool().metadata)

        self.assertTrue(decision.allowed)
        self.assertFalse(decision.requires_confirmation)

    def test_confirm_mode_requires_consequential_approval(self):
        metadata = ToolMetadata(
            name="example.write",
            description="Write an example value.",
            category="test",
            permission_level=PermissionLevel.MEDIUM_RISK,
            risk_level=RiskLevel.MEDIUM,
        )

        decision = PermissionEngine(mode=ExecutionMode.CONFIRM).evaluate(metadata)

        self.assertFalse(decision.allowed)
        self.assertTrue(decision.requires_confirmation)

    def test_critical_operations_are_denied_by_default(self):
        metadata = ToolMetadata(
            name="example.critical",
            description="Critical test operation.",
            category="test",
            permission_level=PermissionLevel.CRITICAL,
            risk_level=RiskLevel.CRITICAL,
        )

        decision = PermissionEngine(mode=ExecutionMode.AUTONOMOUS).evaluate(metadata)

        self.assertFalse(decision.allowed)
        self.assertFalse(decision.requires_confirmation)


if __name__ == "__main__":
    unittest.main()
