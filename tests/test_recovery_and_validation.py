"""Execution, validation, failure-classification, and recovery tests."""

from __future__ import annotations

import unittest

from executor import classify_failure
from models import (
    ExecutionContext,
    ExecutionPlan,
    ExecutionStep,
    StructuredIntent,
    TaskStatus,
)
from reasoning.recovery import RecoveryManager


class FailureClassificationTests(unittest.TestCase):
    def _context_with_error(self, message: str) -> ExecutionContext:
        context = ExecutionContext(user_input="test")
        context.errors.append(message)
        return context

    def test_missing_target_is_not_recoverable(self):
        result = classify_failure(self._context_with_error("Could not resolve a trusted executable for: foo"))
        self.assertEqual(result["category"], "missing_target")
        self.assertFalse(result["recoverable"])

    def test_invalid_arguments_is_recoverable(self):
        result = classify_failure(self._context_with_error("parameter 'query' must be a string"))
        self.assertEqual(result["category"], "invalid_arguments")
        self.assertTrue(result["recoverable"])

    def test_permission_failure_is_not_recoverable(self):
        result = classify_failure(self._context_with_error("user confirmation required"))
        self.assertEqual(result["category"], "permission")
        self.assertFalse(result["recoverable"])

    def test_timeout_is_recoverable(self):
        result = classify_failure(self._context_with_error("request timed out"))
        self.assertEqual(result["category"], "timeout")
        self.assertTrue(result["recoverable"])

    def test_gui_focus_failure_is_not_recoverable(self):
        result = classify_failure(self._context_with_error("Application window could not be focused"))
        self.assertEqual(result["category"], "gui_unavailable")
        self.assertFalse(result["recoverable"])


class RecoveryTests(unittest.TestCase):
    def _failed_step(self, *, tool: str = "web.search", parameters=None) -> ExecutionStep:
        return ExecutionStep(
            id="step_1",
            name="Search",
            action="invoke_tool",
            description="",
            metadata={"tool": tool, "parameters": parameters or {"query": "MrBeast videos"}},
        )

    def test_recovery_corrects_arguments_for_recoverable_failure(self):
        def fake_ask(**_kwargs):
            return '{"recoverable": true, "reason": "query should not include videos", "corrected_arguments": {"query": "MrBeast"}}'

        context = ExecutionContext(
            user_input="search youtube for mrbeast videos",
            structured_intent=StructuredIntent(intent="search", query="MrBeast"),
        )
        context.errors.append("parameter rejected: query")
        context.metadata["failure_classification"] = classify_failure(context)
        step = self._failed_step()

        recovered = RecoveryManager(ask=fake_ask).attempt_recovery(context, step)

        self.assertTrue(recovered)
        self.assertEqual(step.metadata["parameters"]["query"], "MrBeast")
        self.assertEqual(step.metadata["recovery_attempts"], 1)

    def test_recovery_refuses_when_failure_is_permanent(self):
        def fake_ask(**_kwargs):
            return '{"recoverable": true, "corrected_arguments": {"query": "MrBeast"}}'

        context = ExecutionContext(user_input="x")
        context.errors.append("Could not resolve a trusted executable for: nope")
        context.metadata["failure_classification"] = classify_failure(context)
        step = self._failed_step()

        self.assertFalse(RecoveryManager(ask=fake_ask).attempt_recovery(context, step))

    def test_recovery_stops_on_identical_arguments(self):
        def fake_ask(**_kwargs):
            return '{"recoverable": true, "corrected_arguments": {"query": "MrBeast videos"}}'

        context = ExecutionContext(user_input="x")
        context.errors.append("invalid parameter")
        context.metadata["failure_classification"] = classify_failure(context)
        step = self._failed_step(parameters={"query": "MrBeast videos"})

        self.assertFalse(RecoveryManager(ask=fake_ask).attempt_recovery(context, step))

    def test_recovery_respects_retry_limit(self):
        def fake_ask(**_kwargs):
            return '{"recoverable": true, "corrected_arguments": {"query": "new"}}'

        context = ExecutionContext(user_input="x")
        context.errors.append("invalid parameter")
        context.metadata["failure_classification"] = classify_failure(context)
        step = self._failed_step()
        step.metadata["recovery_attempts"] = 99

        self.assertFalse(RecoveryManager(ask=fake_ask).attempt_recovery(context, step))

    def test_recovery_never_targets_non_catalog_capability(self):
        context = ExecutionContext(user_input="x")
        context.errors.append("invalid parameter")
        context.metadata["failure_classification"] = classify_failure(context)
        step = self._failed_step(tool="powershell.execute")

        self.assertFalse(RecoveryManager(ask=lambda **_: "{}").attempt_recovery(context, step))


class PlanValidationTests(unittest.TestCase):
    def _validate(self, plan):
        from brain import _validate_plan

        return _validate_plan(plan)

    def test_valid_tool_plan_passes(self):
        plan = ExecutionPlan(
            user_question="x",
            steps=[
                ExecutionStep(
                    id="s1", name="n", action="invoke_tool", description="",
                    metadata={"tool": "web.search", "parameters": {"query": "x"}},
                )
            ],
        )
        self.assertTrue(self._validate(plan)["valid"])

    def test_unknown_capability_is_rejected(self):
        plan = ExecutionPlan(
            user_question="x",
            steps=[
                ExecutionStep(
                    id="s1", name="n", action="invoke_tool", description="",
                    metadata={"tool": "shell.exec", "parameters": {}},
                )
            ],
        )
        result = self._validate(plan)
        self.assertFalse(result["valid"])
        self.assertTrue(any("unavailable capability" in error for error in result["errors"]))

    def test_missing_tool_name_is_rejected(self):
        plan = ExecutionPlan(
            user_question="x",
            steps=[ExecutionStep(id="s1", name="n", action="invoke_tool", description="", metadata={})],
        )
        self.assertFalse(self._validate(plan)["valid"])

    def test_unknown_action_is_rejected(self):
        plan = ExecutionPlan(
            user_question="x",
            steps=[ExecutionStep(id="s1", name="n", action="run_shell", description="")],
        )
        self.assertFalse(self._validate(plan)["valid"])


if __name__ == "__main__":
    unittest.main()



class ApprovalSignatureTests(unittest.TestCase):
    # An approval grant is bound to the exact arguments the user approved. A
    # bounded recovery that rewrites a consequential step's arguments produces a
    # different signature, so the old grant no longer authorizes the new effect.

    def test_signature_is_stable_for_identical_arguments(self):
        from executor import approval_signature

        self.assertEqual(
            approval_signature("filesystem.write", {"path": "a.txt", "text": "hi"}),
            approval_signature("filesystem.write", {"text": "hi", "path": "a.txt"}),
        )

    def test_signature_changes_when_arguments_change(self):
        from executor import approval_signature

        self.assertNotEqual(
            approval_signature("filesystem.write", {"path": "a.txt"}),
            approval_signature("filesystem.write", {"path": "b.txt"}),
        )

    def test_executor_requires_reapproval_after_argument_change(self):
        from executor import Executor, approval_signature
        from tools import ExecutionMode, PermissionEngine, ToolRegistry, ToolRouter
        from tools.base import PermissionLevel, RiskLevel, Tool, ToolMetadata, ToolResult

        calls: list[dict] = []

        class ConfirmingWrite(Tool):
            metadata = ToolMetadata(
                name="filesystem.write",
                description="filesystem.write",
                category="computer.filesystem",
                permission_level=PermissionLevel.MEDIUM_RISK,
                risk_level=RiskLevel.MEDIUM,
            )

            def execute(self, parameters):
                calls.append(dict(parameters))
                return ToolResult(success=True, status="completed", output={"path": parameters.get("path")})

        registry = ToolRegistry()
        registry.register(ConfirmingWrite())
        router = ToolRouter(registry=registry, permission_engine=PermissionEngine(mode=ExecutionMode.CONFIRM))
        executor = Executor(
            vector_store=type("V", (), {"search": lambda *a, **k: []})(),
            system_prompt="s",
            retrieval_template="{context}{question}",
            tool_router=router,
        )

        plan = ExecutionPlan(
            user_question="save",
            steps=[
                ExecutionStep(
                    id="s1",
                    name="write",
                    action="invoke_tool",
                    description="write",
                    metadata={"tool": "filesystem.write", "parameters": {"path": "approved.txt", "text": "hi"}},
                )
            ],
        )
        context = ExecutionContext(user_input="save")
        # The user approved the step with path "approved.txt".
        context.metadata["approved_tools"] = {"filesystem.write"}
        context.metadata["approved_signatures"] = {
            approval_signature("filesystem.write", {"path": "approved.txt", "text": "hi"})
        }
        # A recovery rewrote the destination to something else.
        plan.steps[0].metadata["parameters"] = {"path": "different.txt", "text": "hi"}

        executor.execute(plan, context)

        # The altered write was not authorized, so the tool never ran.
        self.assertEqual(calls, [])
        self.assertEqual(context.status, TaskStatus.WAITING_FOR_CONFIRMATION)

    def test_executor_allows_unchanged_approved_arguments(self):
        from executor import Executor, approval_signature
        from tools import ExecutionMode, PermissionEngine, ToolRegistry, ToolRouter
        from tools.base import PermissionLevel, RiskLevel, Tool, ToolMetadata, ToolResult

        calls: list[dict] = []

        class ConfirmingWrite(Tool):
            metadata = ToolMetadata(
                name="filesystem.write",
                description="filesystem.write",
                category="computer.filesystem",
                permission_level=PermissionLevel.MEDIUM_RISK,
                risk_level=RiskLevel.MEDIUM,
            )

            def execute(self, parameters):
                calls.append(dict(parameters))
                return ToolResult(success=True, status="completed", output={"path": parameters.get("path")})

        registry = ToolRegistry()
        registry.register(ConfirmingWrite())
        router = ToolRouter(registry=registry, permission_engine=PermissionEngine(mode=ExecutionMode.CONFIRM))
        executor = Executor(
            vector_store=type("V", (), {"search": lambda *a, **k: []})(),
            system_prompt="s",
            retrieval_template="{context}{question}",
            tool_router=router,
        )
        plan = ExecutionPlan(
            user_question="save",
            steps=[
                ExecutionStep(
                    id="s1",
                    name="write",
                    action="invoke_tool",
                    description="write",
                    metadata={"tool": "filesystem.write", "parameters": {"path": "approved.txt", "text": "hi"}},
                )
            ],
        )
        context = ExecutionContext(user_input="save")
        context.metadata["approved_tools"] = {"filesystem.write"}
        context.metadata["approved_signatures"] = {
            approval_signature("filesystem.write", {"path": "approved.txt", "text": "hi"})
        }

        executor.execute(plan, context)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["path"], "approved.txt")
