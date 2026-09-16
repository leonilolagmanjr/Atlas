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
