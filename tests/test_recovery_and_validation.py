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

    def test_ui_verification_failure_is_recoverable_and_distinct(self):
        result = classify_failure(
            self._context_with_error(
                "applications.write_text did not produce the expected effect: "
                "the text was not present in Edit after writing it."
            )
        )
        self.assertEqual(result["category"], "ui_verification_failed")
        self.assertTrue(result["recoverable"])

    def test_ui_target_missing_is_recoverable(self):
        result = classify_failure(
            self._context_with_error("Notepad window could not be found after 5s")
        )
        self.assertEqual(result["category"], "ui_target_missing")
        self.assertTrue(result["recoverable"])

    def test_verification_failure_is_not_misread_as_execution_failure(self):
        # The two failure kinds must classify differently even though both
        # reach the executor as errors.
        verification = classify_failure(
            self._context_with_error(
                "applications.write_text did not produce the expected effect: absent"
            )
        )
        execution = classify_failure(
            self._context_with_error("SendMessageW failed: access denied")
        )
        self.assertEqual(verification["category"], "ui_verification_failed")
        self.assertNotEqual(execution["category"], "ui_verification_failed")


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


class UiRecoveryTests(unittest.TestCase):
    """Deterministic (no-model) recovery for computer-interaction failures."""

    def _step(self, parameters: dict) -> ExecutionStep:
        return ExecutionStep(
            id="write_step",
            name="Write text",
            action="invoke_tool",
            description="",
            metadata={"tool": "applications.write_text", "parameters": parameters},
        )

    def _context(self, message: str) -> ExecutionContext:
        context = ExecutionContext(user_input="x")
        context.errors.append(message)
        context.metadata["failure_classification"] = classify_failure(context)
        return context

    def test_verification_failure_switches_delivery_without_the_model(self):
        def refusing_ask(**_kwargs):
            raise AssertionError("deterministic recovery must not call the model")

        context = self._context(
            "applications.write_text did not produce the expected effect: absent"
        )
        step = self._step({"application": "Notepad", "text": "hi", "delivery": "auto"})

        self.assertTrue(RecoveryManager(ask=refusing_ask).attempt_recovery(context, step))
        self.assertEqual(step.metadata["parameters"]["delivery"], "paste")
        self.assertEqual(step.metadata["recovery_strategy"], "deterministic")

    def test_delivery_fallback_chain_is_bounded(self):
        context = self._context(
            "applications.write_text did not produce the expected effect: absent"
        )
        step = self._step({"application": "Notepad", "text": "hi", "delivery": "paste"})

        self.assertTrue(
            RecoveryManager(ask=lambda **_: "{}").attempt_recovery(context, step)
        )
        self.assertEqual(step.metadata["parameters"]["delivery"], "direct")

    def test_target_missing_doubles_the_wait_bounded(self):
        context = self._context("Notepad window could not be found after 5s")
        step = self._step({"application": "Notepad", "text": "hi", "wait_seconds": 3})

        self.assertTrue(
            RecoveryManager(ask=lambda **_: "{}").attempt_recovery(context, step)
        )
        self.assertEqual(step.metadata["parameters"]["wait_seconds"], 6)

    def test_target_missing_never_waits_longer_than_the_cap(self):
        context = self._context("Notepad window could not be found after 20s")
        step = self._step({"application": "Notepad", "text": "hi", "wait_seconds": 15})

        # At the cap there is no deterministic fix and no model is available,
        # so recovery must stop instead of stalling longer.
        self.assertFalse(RecoveryManager(ask=None).attempt_recovery(context, step))
        self.assertEqual(step.metadata["parameters"]["wait_seconds"], 15)

    def test_deterministic_ui_recovery_never_touches_other_tools(self):
        context = self._context(
            "web.search did not produce the expected effect: absent"
        )
        step = ExecutionStep(
            id="search_step",
            name="Search",
            action="invoke_tool",
            description="",
            metadata={"tool": "web.search", "parameters": {"query": "cars"}},
        )

        # No model available: the deterministic UI corrections do not apply.
        self.assertFalse(RecoveryManager(ask=None).attempt_recovery(context, step))
        self.assertEqual(step.metadata["parameters"], {"query": "cars"})

    def test_recovery_records_history_entry(self):
        context = self._context(
            "applications.write_text did not produce the expected effect: absent"
        )
        step = self._step({"application": "Notepad", "text": "hi", "delivery": "auto"})

        self.assertTrue(
            RecoveryManager(ask=lambda **_: "{}").attempt_recovery(context, step)
        )
        history = context.metadata["recovery_history"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["step_id"], "write_step")
        self.assertEqual(history[0]["reason"], "ui_verification_failed")


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
