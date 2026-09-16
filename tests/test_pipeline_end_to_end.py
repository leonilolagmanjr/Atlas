"""End-to-end pipeline tests using fake LLM and fake tools.

These exercise the real Brain/Planner/Executor wiring without touching the
network, the model, or the filesystem.
"""

from __future__ import annotations

import unittest

from brain import Brain
from executor import Executor
from models import StructuredIntent, TaskStatus
from planner import Planner
from reasoning.interpreter import SemanticInterpreter
from tools import ExecutionMode, PermissionEngine, ToolRegistry, ToolRouter
from tools.base import Tool, ToolMetadata, ToolResult


class FakeVectorStore:
    def search(self, *_args, **_kwargs):
        return []


class RecordingTool(Tool):
    """A catalog capability that records the arguments it received."""

    def __init__(self, name: str, output: dict) -> None:
        self.metadata = ToolMetadata(name=name, description=name, category="test")
        self._output = output
        self.calls: list[dict] = []

    def execute(self, parameters):
        self.calls.append(dict(parameters))
        return ToolResult(success=True, status="completed", output=self._output)


class ConfirmingTool(Tool):
    # A medium-risk tool that records calls and can require confirmation.
    def __init__(self, name: str, output: dict) -> None:
        from tools.base import PermissionLevel, RiskLevel
        self.metadata = ToolMetadata(
            name=name,
            description=name,
            category="computer.applications",
            permission_level=PermissionLevel.MEDIUM_RISK,
            risk_level=RiskLevel.MEDIUM,
        )
        self._output = output
        self.calls: list[dict] = []

    def execute(self, parameters):
        self.calls.append(dict(parameters))
        return ToolResult(success=True, status="written", output=self._output)


def build_brain(ask, tools: list[Tool], *, mode: ExecutionMode = ExecutionMode.AUTONOMOUS) -> Brain:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    router = ToolRouter(
        registry=registry,
        permission_engine=PermissionEngine(mode=mode),
    )
    return Brain(
        vector_store=FakeVectorStore(),
        system_prompt="sys",
        retrieval_template="{conversation_history}{context}{question}",
        planner=Planner(registry=router, ask=ask),
        executor=Executor(
            vector_store=FakeVectorStore(),
            system_prompt="sys",
            retrieval_template="{conversation_history}{context}{question}",
            tool_router=router,
        ),
        tool_router=router,
        interpreter=SemanticInterpreter(ask=ask),
        llm_ask=ask,
    )


class MultiStepExecutionTests(unittest.TestCase):
    def test_content_then_write_resolves_generated_text(self):
        generator = RecordingTool("content.generate", {"text": "cars poem body"})
        writer = RecordingTool(
            "applications.write_text",
            {"pid": 1, "application": "Notepad", "characters": 14},
        )

        def fake_ask(*, system_prompt, user_prompt):
            if "content generator" in system_prompt:
                return "cars poem body"
            return '{"intent":"write_content","action":"create","content_type":"poem","topic":"cars","destination":"Notepad","confidence":0.97}'

        brain = build_brain(fake_ask, [generator, writer])
        out = brain.process("create a poem about cars in notepad")
        ctx = brain.last_context

        self.assertEqual(ctx.status, TaskStatus.COMPLETED)
        # Generated content reached the writer as a resolved value.
        self.assertEqual(writer.calls[0]["application"].casefold(), "notepad")
        self.assertEqual(writer.calls[0]["text"], "cars poem body")
        self.assertEqual(generator.calls[0]["topic"], "cars")
        # The final response reports the write that completed the task.
        self.assertIn("Notepad", out or "")

    def test_search_plan_runs_single_search_step(self):
        search = RecordingTool("web.search", {"query": "mrbeast", "results": []})

        def fake_ask(*, system_prompt, user_prompt):
            return "{}"

        brain = build_brain(fake_ask, [search])
        brain.process("search youtube for mrbeast videos")
        ctx = brain.last_context

        self.assertEqual(ctx.selected_tool, "web.search")
        self.assertEqual(len(search.calls), 1)
        self.assertEqual(search.calls[0]["query"], "mrbeast")
        self.assertEqual(search.calls[0]["site"], "youtube")

    def test_approval_resumes_without_regenerating_content(self):
        generator = RecordingTool("content.generate", {"text": "original poem"})
        writer = ConfirmingTool(
            "applications.write_text",
            {"pid": 1, "application": "Notepad", "characters": 13},
        )

        def fake_ask(*, system_prompt, user_prompt):
            if "content generator" in system_prompt:
                return "original poem"
            return '{"intent":"write_content","action":"create","content_type":"poem","topic":"wind","destination":"Notepad","confidence":0.97}'

        brain = build_brain(fake_ask, [generator, writer], mode=ExecutionMode.CONFIRM)
        first = brain.process("write a poem about the wind in notepad")
        ctx = brain.last_context
        # The write step must pause for confirmation, not fail.
        self.assertEqual(ctx.status, TaskStatus.WAITING_FOR_CONFIRMATION)
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(writer.calls, [])

        # Approving must run the write exactly once and NOT regenerate content.
        result = brain.approve_pending()
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(len(writer.calls), 1)
        self.assertEqual(writer.calls[0]["text"], "original poem")
        self.assertEqual(brain.last_context.status, TaskStatus.COMPLETED)
        self.assertIn("Notepad", result)

    def test_approval_does_not_loop(self):
        generator = RecordingTool("content.generate", {"text": "poem body"})
        writer = ConfirmingTool("applications.write_text", {"pid": 1, "application": "Notepad"})

        def fake_ask(*, system_prompt, user_prompt):
            if "content generator" in system_prompt:
                return "poem body"
            return '{"intent":"write_content","action":"create","content_type":"poem","destination":"Notepad","confidence":0.97}'

        brain = build_brain(fake_ask, [generator, writer], mode=ExecutionMode.CONFIRM)
        brain.process("write a poem in notepad")
        brain.approve_pending()

        # The plan must not still be waiting after a single approval.
        self.assertNotEqual(brain.last_context.status, TaskStatus.WAITING_FOR_CONFIRMATION)

    def test_ambiguous_request_asks_without_executing(self):
        writer = RecordingTool("applications.write_text", {"pid": 1})

        brain = build_brain(lambda **_: "{}", [writer])
        out = brain.process("open it and write something about cars")

        self.assertEqual(brain.last_context.status, TaskStatus.UNCERTAIN)
        self.assertEqual(writer.calls, [])
        self.assertIsNotNone(out)


class ContextualFollowUpTests(unittest.TestCase):
    def test_follow_up_reuses_destination_from_prior_turn(self):
        generator = RecordingTool("content.generate", {"text": "body"})
        writer = RecordingTool("applications.write_text", {"pid": 1, "application": "Notepad"})
        launcher = RecordingTool("applications.launch_named", {"pid": 1, "application": "notepad"})

        brain = build_brain(lambda **_: "{}", [generator, writer, launcher])
        brain.process("open notepad")
        brain.process("write a poem")

        # The second turn must still reach Notepad (destination inherited).
        self.assertTrue(writer.calls)
        self.assertEqual(writer.calls[-1]["application"].casefold(), "notepad")


if __name__ == "__main__":
    unittest.main()
