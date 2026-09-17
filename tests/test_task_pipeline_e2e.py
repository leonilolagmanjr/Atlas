"""End-to-end tests for the Task pipeline through the live Brain wiring.

These exercise the real Brain -> interpreter -> validator -> task planner ->
executor chain using a fake model and fake tools, so they run without a network,
a real Qwen model, or filesystem access.
"""

from __future__ import annotations

import unittest

from brain import Brain
from executor import Executor, classify_failure
from models import TaskStatus
from planner import Planner
from reasoning.task_interpreter import SemanticTaskInterpreter
from reasoning.verifier import TaskVerifier
from tools import ExecutionMode, PermissionEngine, ToolRegistry, ToolRouter
from tools.base import Tool, ToolMetadata, ToolResult


class FakeVectorStore:
    def search(self, *_args, **_kwargs):
        return []


class RecordingTool(Tool):
    def __init__(self, name: str, output: dict) -> None:
        self.metadata = ToolMetadata(name=name, description=name, category="test")
        self._output = output
        self.calls: list[dict] = []

    def execute(self, parameters):
        self.calls.append(dict(parameters))
        return ToolResult(success=True, status="completed", output=self._output)


class FormattingTool(RecordingTool):
    def __init__(self, output: dict) -> None:
        super().__init__("content.format", output)


class ConfirmingTool(RecordingTool):
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
        self.calls = []


class ConfirmingFormatTool(FormattingTool):
    def __init__(self, output: dict) -> None:
        from tools.base import PermissionLevel, RiskLevel
        super().__init__(output)
        self.metadata = ToolMetadata(
            name="content.format",
            description="content.format",
            category="content.formatting",
            permission_level=PermissionLevel.MEDIUM_RISK,
            risk_level=RiskLevel.MEDIUM,
        )


def build_brain(ask, tools, *, mode: ExecutionMode = ExecutionMode.AUTONOMOUS) -> Brain:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    router = ToolRouter(registry=registry, permission_engine=PermissionEngine(mode=mode))
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
        interpreter=SemanticTaskInterpreter(ask=ask),
        llm_ask=ask,
    )


class DeterministicPipelineTests(unittest.TestCase):
    def test_about_cars_reaches_generator_topic_and_writer(self):
        generator = RecordingTool("content.generate", {"text": "cars poem"})
        formatter = FormattingTool({"text": "cars poem"})
        writer = RecordingTool("applications.write_text", {"pid": 1, "application": "Notepad", "characters": 9})

        brain = build_brain(lambda **_: "{}", [generator, formatter, writer])
        brain.process("Create a poem in Notepad about cars.")

        ctx = brain.last_context
        self.assertEqual(ctx.status, TaskStatus.COMPLETED)
        self.assertEqual(generator.calls[0]["topic"], "cars")
        self.assertEqual(writer.calls[0]["application"], "Notepad")
        self.assertEqual(writer.calls[0]["text"], "cars poem")

    def test_plan_has_no_leaked_modifier_in_application(self):
        generator = RecordingTool("content.generate", {"text": "x"})
        formatter = FormattingTool({"text": "x"})
        writer = RecordingTool("applications.write_text", {"pid": 1, "application": "Notepad", "characters": 1})

        brain = build_brain(lambda **_: "{}", [generator, formatter, writer])
        brain.process("Create a poem in Notepad about cars.")

        tools = [s.metadata.get("tool") for s in brain.last_context.execution_plan.steps]
        self.assertEqual(tools, ["content.generate", "content.format", "applications.write_text"])

    def test_multi_step_plan_still_pauses_for_confirmation(self):
        generator = RecordingTool("content.generate", {"text": "x"})
        formatter = ConfirmingFormatTool({"text": "x"})
        writer = ConfirmingTool("applications.write_text", {"pid": 1, "application": "Notepad", "characters": 1})

        brain = build_brain(lambda **_: "{}", [generator, formatter, writer], mode=ExecutionMode.CONFIRM)
        brain.process("Create a poem in Notepad about cars.")
        ctx = brain.last_context
        self.assertEqual(ctx.status, TaskStatus.WAITING_FOR_CONFIRMATION)
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(writer.calls, [])

        brain.approve_pending()
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(len(writer.calls), 1)
        self.assertEqual(brain.last_context.status, TaskStatus.COMPLETED)

    def test_informational_request_is_answered_by_reasoning_engine(self):
        brain = build_brain(lambda **_: "Quantum computing uses qubits.", [])
        # A knowledge question is answered by the reasoning layer, never as a
        # tool action, and never with "not in knowledge base".
        out = brain.process("What is quantum computing?")
        ctx = brain.last_context
        self.assertEqual(ctx.metadata["task"]["task_type"], "informational")
        self.assertEqual(ctx.status, TaskStatus.COMPLETED)
        self.assertIsNone(ctx.execution_plan)
        self.assertIn("qubits", out)
        self.assertNotIn("knowledge base", out)

    def test_ambiguous_request_asks_without_executing(self):
        writer = RecordingTool("applications.write_text", {"pid": 1})
        brain = build_brain(lambda **_: "{}", [writer])

        out = brain.process("Open it and write something about cars.")

        self.assertEqual(brain.last_context.status, TaskStatus.UNCERTAIN)
        self.assertEqual(writer.calls, [])
        self.assertIsNotNone(out)


class LLMInterpretedPipelineTests(unittest.TestCase):
    """The model supplies the structured task; Atlas still validates it."""

    def test_model_task_is_validated_and_executed(self):
        generator = RecordingTool("content.generate", {"text": "launch window body"})
        formatter = FormattingTool({"text": "launch window body"})
        writer = RecordingTool("applications.write_text", {"pid": 1, "application": "Notepad", "characters": 3})

        def fake_ask(*, system_prompt, user_prompt):
            if "content generator" in system_prompt:
                return "launch window body"
            # The model returns a structured task capturing a modifier the
            # deterministic pass does not know about ("uplifting" style).
            return (
                '{"task_type":"content_creation","goal":"create_content","confidence":0.95,'
                '"actions":['
                '{"action_id":"a1","capability":"content.generate",'
                '"parameters":{"content_type":"poem","topic":"the wind","style":"uplifting"},'
                '"produces":"generated_text"},'
                '{"action_id":"a2","capability":"content.format",'
                '"parameters":{"content":"$generated_text","destination":"Notepad"},'
                '"depends_on":["a1"],"produces":"formatted_text"},'
                '{"action_id":"a3","capability":"applications.write_text",'
                '"parameters":{"application":"Notepad","text":"$formatted_text"},"depends_on":["a2"]}'
                ']}'
            )

        brain = build_brain(fake_ask, [generator, formatter, writer])
        # A deliberately vague request stays below the deterministic confidence
        # threshold, so the model is consulted (Phase 15: use Qwen for genuine
        # ambiguity, not for mechanical steps).
        brain.process("compose something uplifting for me")

        ctx = brain.last_context
        self.assertEqual(ctx.metadata["task"]["source"], "llm")
        self.assertEqual(generator.calls[0]["topic"], "the wind")
        self.assertEqual(writer.calls[0]["text"], "launch window body")

    def test_model_cannot_invent_a_capability(self):
        def fake_ask(*, system_prompt, user_prompt):
            return (
                '{"task_type":"computer_action","execution_required":true,"actions":'
                '[{"capability":"windows.delete_everything","parameters":{}}]}'
            )

        brain = build_brain(fake_ask, [])
        out = brain.process("delete everything on my computer")

        ctx = brain.last_context
        self.assertEqual(ctx.status, TaskStatus.FAILED)
        # The invented capability was rejected by validation, not executed.
        self.assertIn("unknown capability", " ".join(ctx.metadata["task_validation"]["errors"]))
        self.assertIsNotNone(out)

    def test_malformed_model_output_falls_back_to_deterministic(self):
        generator = RecordingTool("content.generate", {"text": "body"})
        formatter = FormattingTool({"text": "body"})
        writer = RecordingTool("applications.write_text", {"pid": 1, "application": "Notepad", "characters": 4})

        def fake_ask(*, system_prompt, user_prompt):
            if "content generator" in system_prompt:
                return "body"
            return "I cannot help with that."  # not JSON

        brain = build_brain(fake_ask, [generator, formatter, writer])
        brain.process("Create a poem in Notepad about cars.")

        ctx = brain.last_context
        # The deterministic fallback still produced a working plan.
        self.assertEqual(ctx.status, TaskStatus.COMPLETED)
        self.assertEqual(writer.calls[0]["application"], "Notepad")


class VerificationIntegrationTests(unittest.TestCase):
    def test_read_only_web_search_is_served_by_the_reasoning_engine(self):
        # A pure web-search request is answered by the reasoning engine: it runs
        # the planned web.search once and returns an attributed answer, rather
        # than an unprocessed result dump from the executor.
        search = RecordingTool("web.search", {"query": "cars", "results": [{"url": "u"}]})
        brain = build_brain(lambda **_: "A grounded web answer.", [search])
        out = brain.process("Search the web for cars.")

        ctx = brain.last_context
        self.assertEqual(len(search.calls), 1)
        self.assertEqual(search.calls[0]["query"], "cars")
        self.assertEqual(ctx.selected_tool, "web.search")
        self.assertIn("web", ctx.metadata["reasoning_answer"]["provenance"])
        self.assertTrue(out.strip())

    def test_empty_search_still_produces_an_honest_answer(self):
        search = RecordingTool("web.search", {"query": "cars", "results": []})
        brain = build_brain(lambda **_: "A grounded web answer.", [search])
        brain.process("Search the web for cars.")

        ctx = brain.last_context
        self.assertEqual(len(search.calls), 1)
        # No usable results: the engine must not claim a grounded web answer.
        self.assertNotEqual(
            ctx.metadata["reasoning_answer"]["mode"], "web_research"
        )

    def test_verifier_never_claims_unverifiable_success(self):
        outcome = TaskVerifier().verify("mystery.capability", {"ok": True}, True)
        self.assertFalse(outcome.verified)
        self.assertEqual(outcome.status, "unverified")


class ReplanningGuardTests(unittest.TestCase):
    def test_failure_classification_blocks_permanent_failures(self):
        from models import ExecutionContext

        context = ExecutionContext(user_input="x")
        context.errors.append("Could not resolve a trusted executable for: nope")
        result = classify_failure(context)
        self.assertFalse(result["recoverable"])


if __name__ == "__main__":
    unittest.main()
