"""Integration tests for vision tool contracts, permissions, verification, recovery.

These drive the vision tools through the real ToolRegistry / ToolRouter /
PermissionEngine / Executor / TaskVerifier / RecoveryManager with fake
perception and input backends, so no GUI session is required.
"""

from __future__ import annotations

import unittest

from computer.interaction import ComputerClickTool, ComputerTypeTool
from computer.vision.ocr import NullOcrProvider
from computer.vision.perception import PerceptionEngine
from computer.vision.providers import NullVisionProvider
from computer.vision_tools import VisionFindTool, VisionObserveTool
from computer.runtime import register_read_only_tools
from executor import Executor, classify_failure
from models import ExecutionContext, ExecutionPlan, ExecutionStep
from reasoning.recovery import RecoveryManager
from reasoning.verifier import TaskVerifier
from tools import ExecutionMode, PermissionEngine, ToolRegistry, ToolRouter
from tools.base import PermissionLevel, RiskLevel, Tool, ToolMetadata, ToolResult

from tests.test_vision_perception import (
    FakeCapture,
    FakeInputBackend,
    FakeWindowsProvider,
    _window_payload,
)


def _engine(*, windows=None, capture=None):
    return PerceptionEngine(
        ocr=NullOcrProvider(),
        vision=NullVisionProvider(),
        windows_provider=windows or FakeWindowsProvider(_window_payload()),
        capture=capture or FakeCapture(),
        max_image_size=0,
    )


class VisionToolTests(unittest.TestCase):
    def test_vision_observe_returns_structured_state(self):
        tool = VisionObserveTool(engine=_engine())
        result = tool.execute({})
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.status, "observed")
        self.assertIn("elements", result.output)
        self.assertIn("observation_id", result.output)

    def test_vision_find_locates_label(self):
        tool = VisionFindTool(engine=_engine())
        result = tool.execute({"query": "Continue"})
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.status, "found")
        self.assertEqual(result.output["candidates"][0]["name"], "Continue")

    def test_vision_find_reports_not_found_honestly(self):
        tool = VisionFindTool(engine=_engine())
        result = tool.execute({"query": "Nonexistent Widget"})
        self.assertTrue(result.success)
        self.assertEqual(result.status, "not_found")
        self.assertEqual(result.output["candidates"], [])

    def test_vision_find_requires_query(self):
        tool = VisionFindTool(engine=_engine())
        result = tool.execute({})
        self.assertFalse(result.success)


class InteractionToolTests(unittest.TestCase):
    def test_click_validates_and_emits_input(self):
        backend = FakeInputBackend()
        tool = ComputerClickTool(engine=_engine(), input_backend=backend)
        result = tool.execute({"description": "Continue"})
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.status, "clicked")
        self.assertEqual(backend.events[0][0], "click")
        self.assertEqual(result.output["target"]["name"], "Continue")

    def test_click_rejects_unknown_target(self):
        backend = FakeInputBackend()
        tool = ComputerClickTool(engine=_engine(), input_backend=backend)
        result = tool.execute({"description": "Nonexistent"})
        self.assertFalse(result.success)
        self.assertEqual(backend.events, [])

    def test_click_rejects_coordinates_outside_element(self):
        backend = FakeInputBackend()
        tool = ComputerClickTool(engine=_engine(), input_backend=backend)
        result = tool.execute({"element_id": "uia_101_3", "x": 0, "y": 0})
        self.assertFalse(result.success)
        self.assertEqual(backend.events, [])

    def test_type_enters_text_after_focusing(self):
        backend = FakeInputBackend()
        tool = ComputerTypeTool(engine=_engine(), input_backend=backend)
        result = tool.execute({"description": "Continue", "text": "hello"})
        self.assertTrue(result.success, result.error)
        kinds = [event[0] for event in backend.events]
        self.assertIn("click", kinds)
        self.assertIn("type", kinds)
        self.assertEqual(result.output["characters"], 5)

    def test_type_without_text_fails(self):
        tool = ComputerTypeTool(engine=_engine(), input_backend=FakeInputBackend())
        result = tool.execute({"description": "Continue"})
        self.assertFalse(result.success)


class VisionRegistrationTests(unittest.TestCase):
    def test_vision_and_interaction_tools_are_registered(self):
        registry = ToolRegistry()
        register_read_only_tools(registry)
        names = set(registry.names())
        for expected in (
            "computer.vision_observe",
            "computer.find",
            "computer.click",
            "computer.double_click",
            "computer.right_click",
            "computer.move",
            "computer.drag",
            "computer.type",
            "computer.keypress",
            "computer.scroll",
            "computer.focus",
        ):
            self.assertIn(expected, names)

    def test_observation_tools_are_plannable_and_read_only(self):
        from tools.capabilities import CapabilityRegistry

        capabilities = CapabilityRegistry()
        observe = capabilities.get("computer.vision_observe")
        find = capabilities.get("computer.find")
        self.assertEqual(observe.risk_level, "read_only")
        self.assertFalse(observe.requires_confirmation)
        self.assertEqual(find.risk_level, "read_only")

    def test_interaction_tools_require_confirmation(self):
        from tools.capabilities import CapabilityRegistry

        capabilities = CapabilityRegistry()
        for name in ("computer.click", "computer.type", "computer.keypress"):
            capability = capabilities.get(name)
            self.assertTrue(capability.requires_confirmation, name)


class VisionPermissionTests(unittest.TestCase):
    def _router(self, mode: ExecutionMode) -> ToolRouter:
        registry = ToolRegistry()
        register_read_only_tools(registry)
        return ToolRouter(registry=registry, permission_engine=PermissionEngine(mode=mode))

    def test_observe_is_permitted_without_confirmation(self):
        router = self._router(ExecutionMode.CONFIRM)
        result = router.execute("computer.vision_observe", {})
        # It may succeed or fail on this platform, but never be gated.
        self.assertNotEqual(result.status, "confirmation_required")

    def test_click_requires_confirmation_in_confirm_mode(self):
        router = self._router(ExecutionMode.CONFIRM)
        result = router.execute("computer.click", {"description": "Continue"})
        self.assertEqual(result.status, "confirmation_required")

    def test_click_denied_in_safe_mode(self):
        router = self._router(ExecutionMode.SAFE)
        result = router.execute("computer.click", {"description": "Continue"})
        self.assertEqual(result.status, "denied")


class VisionVerifierTests(unittest.TestCase):
    def setUp(self):
        self.verifier = TaskVerifier()

    def test_visual_action_with_target_is_unverified_pending_observation(self):
        outcome = self.verifier.verify(
            "computer.click",
            {"status": "clicked", "target": {"name": "Continue", "source": "uia"}},
            success=True,
        )
        self.assertFalse(outcome.verified)
        self.assertEqual(outcome.status, "unverified")
        self.assertIn("Continue", outcome.detail)

    def test_confirmed_visual_action_is_verified(self):
        outcome = self.verifier.verify(
            "computer.focus",
            {"status": "focused", "verified": True, "target": {"name": "Notepad"}},
            success=True,
        )
        self.assertTrue(outcome.verified)

    def test_find_with_candidates_is_verified(self):
        outcome = self.verifier.verify(
            "computer.find", {"candidates": [{"name": "Continue"}]}, success=True
        )
        self.assertTrue(outcome.verified)

    def test_find_without_candidates_is_unverified(self):
        outcome = self.verifier.verify(
            "computer.find", {"candidates": [], "summary": "none"}, success=True
        )
        self.assertFalse(outcome.verified)


class ClosedLoopExecutorTests(unittest.TestCase):
    def test_visual_action_is_re_observed_by_the_executor(self):
        """OBSERVE -> ACT -> OBSERVE: a visual action triggers a follow-up observation."""

        observed: list[tuple] = []

        class RecordingObserver:
            def observe_application(self, application=None, *, wait_seconds=None, include_controls=True):
                observed.append((application, wait_seconds))
                return {
                    "status": "observed",
                    "fidelity": "controls",
                    "summary": "Observed Chrome.",
                    "window": {"title": "YouTube - Chrome", "application": "chrome.exe"},
                    "text": "",
                }

        class FakeClick(Tool):
            metadata = ToolMetadata(
                name="computer.click",
                description="click",
                category="computer.interaction",
                permission_level=PermissionLevel.MEDIUM_RISK,
                risk_level=RiskLevel.MEDIUM,
                verifiable=True,
            )

            def execute(self, parameters):
                return ToolResult(
                    success=True,
                    status="clicked",
                    output={"status": "clicked", "target": {"name": parameters.get("description")}},
                )

        registry = ToolRegistry()
        registry.register(FakeClick())
        router = ToolRouter(
            registry=registry,
            permission_engine=PermissionEngine(mode=ExecutionMode.AUTONOMOUS),
        )
        executor = Executor(
            vector_store=object(),
            system_prompt="",
            retrieval_template="",
            tool_router=router,
            observer=RecordingObserver(),
        )
        plan = ExecutionPlan(
            user_question="click Continue",
            steps=[
                ExecutionStep(
                    id="s1",
                    name="Click",
                    action="invoke_tool",
                    description="click",
                    metadata={"tool": "computer.click", "parameters": {"description": "Continue"}},
                )
            ],
        )
        context = ExecutionContext(user_input="click Continue")

        executor.execute(plan, context)

        # The executor observed the UI after the visual action (OBSERVE again).
        self.assertEqual(len(observed), 1)
        self.assertIn("last_ui_observation", context.metadata)

    def test_computer_observe_routes_to_visual_state_when_requested(self):
        from computer.perception import _wants_visual_state
        self.assertTrue(_wants_visual_state({"full_screen": True}))
        self.assertTrue(_wants_visual_state({"include_ocr": True}))
        self.assertFalse(_wants_visual_state({}))
        # A plain observe request stays on the window-only path.
        self.assertFalse(_wants_visual_state({"application": "Notepad"}))

class VisionRecoveryTests(unittest.TestCase):
    def _visual_step(self, parameters):
        return ExecutionStep(
            id="click_step",
            name="Click",
            action="invoke_tool",
            description="",
            metadata={"tool": "computer.click", "parameters": parameters},
        )

    def _context(self, message):
        context = ExecutionContext(user_input="x")
        context.errors.append(message)
        context.metadata["failure_classification"] = classify_failure(context)
        return context

    def test_target_missing_doubles_the_wait(self):
        context = self._context("Chrome window could not be found after 5s")
        step = self._visual_step({"description": "Continue", "wait_seconds": 2})
        self.assertTrue(RecoveryManager(ask=None).attempt_recovery(context, step))
        self.assertEqual(step.metadata["parameters"]["wait_seconds"], 4)
        self.assertEqual(step.metadata["recovery_strategy"], "deterministic")

    def test_verification_failure_enables_vlm_then_stops(self):
        context = self._context(
            "computer.click did not produce the expected effect: absent"
        )
        step = self._visual_step({"description": "Continue"})
        manager = RecoveryManager(ask=None)
        self.assertTrue(manager.attempt_recovery(context, step))
        self.assertTrue(step.metadata["parameters"]["allow_vlm"])
        # A second attempt with the model already on widens the age once.
        self.assertTrue(manager.attempt_recovery(context, step))
        self.assertIn("max_age_seconds", step.metadata["parameters"])
        # A third attempt has no new deterministic fix and must stop.
        self.assertFalse(manager.attempt_recovery(context, step))

    def test_visual_recovery_never_touches_other_tools(self):
        context = self._context("web.search did not produce the expected effect: absent")
        step = ExecutionStep(
            id="s",
            name="n",
            action="invoke_tool",
            description="",
            metadata={"tool": "web.search", "parameters": {"query": "cars"}},
        )
        self.assertFalse(RecoveryManager(ask=None).attempt_recovery(context, step))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
