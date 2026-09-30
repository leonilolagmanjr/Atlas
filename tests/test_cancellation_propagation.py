"""Milestone 6 (real cancellation): frontend -> runtime -> Brain -> executor.

Cancellation is only meaningful if the work actually stops. These tests use the
real ``Brain`` and the real ``Executor`` with fake tools, and assert two
separate things:

* a cancel request that arrives mid-plan stops the plan: later steps are marked
  SKIPPED and their tools are never called, and the recorded state is
  ``CANCELLED``;
* a turn that finished *before* the cancel arrived is reported as completed,
  because reporting it as cancelled would claim work was stopped when it was
  not.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from brain import Brain
from executor import Executor
from memory.conversation_runtime import (
    EVENT_CANCELLED,
    CancellationToken,
    ConversationRuntime,
)
from memory.context_manager import ContextManager
from memory.memory_manager import MemoryManager
from memory.storage import ConversationStore
from models import PlanStatus, StepStatus, TaskStatus
from planner import Planner
from tools import ExecutionMode, PermissionEngine, ToolRegistry, ToolRouter
from tools.base import Tool, ToolMetadata, ToolResult


class _NullIndex:
    available = False

    def search(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return []

    def index_message(self, **_kwargs: Any) -> bool:
        return False

    def forget_conversation(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _NullEmbedder:
    def index(self, *_args: Any, **_kwargs: Any) -> bool:
        return False

    def similarities(self, *_args: Any, **_kwargs: Any) -> dict[str, float]:
        return {}

    def delete(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _FakeVectorStore:
    def search(self, *_args, **_kwargs):
        return []


class RecordingTool(Tool):
    """A capability that records its calls and can trip a cancel token."""

    def __init__(self, name: str, output: dict, token: CancellationToken | None = None) -> None:
        self.metadata = ToolMetadata(name=name, description=name, category="test")
        self._output = output
        self._token = token
        self.calls: list[dict] = []

    def execute(self, parameters):
        self.calls.append(dict(parameters))
        if self._token is not None:
            # The user pressed Stop while this tool was running.
            self._token.cancel("stop pressed during execution")
        return ToolResult(success=True, status="completed", output=self._output)


def build_brain(ask, tools, *, executor: Executor | None = None) -> Brain:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    router = ToolRouter(registry=registry, permission_engine=PermissionEngine(mode=ExecutionMode.AUTONOMOUS))
    return Brain(
        vector_store=_FakeVectorStore(),
        system_prompt="sys",
        retrieval_template="{conversation_history}{context}{question}",
        planner=Planner(registry=router, ask=ask),
        executor=executor or Executor(
            vector_store=_FakeVectorStore(),
            system_prompt="sys",
            retrieval_template="{conversation_history}{context}{question}",
            tool_router=router,
        ),
        tool_router=router,
        llm_ask=ask,
    )


def _poem_task(*, system_prompt: str) -> str:
    if "content generator" in system_prompt:
        return "poem body"
    return (
        '{"intent":"write_content","action":"create","content_type":"poem",'
        '"topic":"cars","destination":"Notepad","confidence":0.97}'
    )


class ExecutorCancellationTests(unittest.TestCase):
    def test_cancel_before_the_first_step_runs_nothing(self) -> None:
        from models import ExecutionContext, ExecutionPlan, ExecutionStep

        router = ToolRouter(
            registry=ToolRegistry(),
            permission_engine=PermissionEngine(mode=ExecutionMode.AUTONOMOUS),
        )
        executor = Executor(
            vector_store=_FakeVectorStore(),
            system_prompt="sys",
            retrieval_template="{conversation_history}{context}{question}",
            tool_router=router,
        )
        plan = ExecutionPlan(
            user_question="write it",
            steps=[
                ExecutionStep(
                    id="s1",
                    name="write",
                    action="invoke_tool",
                    description="write a file",
                    metadata={"tool": "filesystem.write", "parameters": {"path": "a.txt"}},
                )
            ],
        )
        context = ExecutionContext(user_input="write it")
        executor.execute(plan, context, cancel=lambda: True)

        self.assertEqual(context.status, TaskStatus.CANCELLED)
        self.assertEqual(plan.status, PlanStatus.SKIPPED)
        self.assertEqual(plan.steps[0].status, StepStatus.SKIPPED)
        self.assertTrue(context.metadata.get("cancelled"))
        self.assertEqual(context.tool_calls, [])
        self.assertIn("ancel", context.final_response or "")

    def test_cancelling_mid_plan_skips_later_steps(self) -> None:
        token = CancellationToken()
        generator = RecordingTool("content.generate", {"text": "poem body"}, token=token)
        formatter = RecordingTool("content.format", {"text": "poem body"}, token=token)
        writer = RecordingTool("applications.write_text", {"pid": 1, "application": "Notepad", "characters": 9}, token=token)

        brain = build_brain(lambda **kwargs: _poem_task(**kwargs), [generator, formatter, writer])
        out = brain.process("create a poem about cars in notepad", cancel=token.is_cancelled)
        context = brain.last_context

        self.assertEqual(context.status, TaskStatus.CANCELLED)
        self.assertIn("ancel", out or "")
        # The tool that ran before the cancel is recorded; nothing after it ran.
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(formatter.calls, [])
        self.assertEqual(writer.calls, [])
        statuses = [step.status for step in context.execution_plan.steps]
        self.assertIn(StepStatus.SKIPPED, statuses)
        # Completion verification must not turn a cancelled plan into a failure.
        self.assertNotEqual(context.status, TaskStatus.FAILED)


class BrainCancellationTests(unittest.TestCase):
    def test_cancel_before_planning_stops_the_request(self) -> None:
        search = RecordingTool("web.search", {"query": "docker", "results": []})
        brain = build_brain(lambda **_kwargs: "{}", [search])
        out = brain.process("search the web for docker", cancel=lambda: True)
        self.assertEqual(brain.last_context.status, TaskStatus.CANCELLED)
        self.assertEqual(search.calls, [])
        self.assertIn("ancel", out)

    def test_cancel_token_is_not_inherited_by_a_later_approval(self) -> None:
        token = CancellationToken()
        token.cancel()
        brain = build_brain(lambda **_kwargs: "{}", [RecordingTool("web.search", {"results": []})])
        brain.process("search the web for docker", cancel=token.is_cancelled)
        # The token belongs to that one request; nothing stale survives it.
        self.assertIsNone(brain._cancel)  # noqa: SLF001


class RuntimeCancellationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.token = CancellationToken()
        self.generator = RecordingTool("content.generate", {"text": "poem body"}, token=self.token)
        self.formatter = RecordingTool("content.format", {"text": "poem body"}, token=self.token)
        self.writer = RecordingTool("applications.write_text", {"pid": 1, "application": "Notepad", "characters": 9}, token=self.token)
        self.brain = build_brain(
            lambda **kwargs: _poem_task(**kwargs), [self.generator, self.formatter, self.writer]
        )
        manager = MemoryManager(ask=None, index=_NullIndex())
        manager._store = ConversationStore(root_folder=self.root)  # noqa: SLF001
        manager._sessions._store = manager._store  # noqa: SLF001
        manager._sessions._active_path = self.root / "active_session.json"  # noqa: SLF001
        from memory.user_memory import UserMemoryStore

        self.runtime = ConversationRuntime(
            memory_manager=manager,
            brain=self.brain,
            context_manager=ContextManager(memory_manager=manager, index=_NullIndex()),
            user_memory=UserMemoryStore(path=self.root / "user_memory.jsonl", embedder=_NullEmbedder()),
            ask=lambda **_kwargs: "an answer",
        )

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_a_stopped_turn_is_recorded_cancelled_and_stops_working(self) -> None:
        events: list[tuple[str, dict]] = []
        result = self.runtime.handle_message(
            "create a poem about cars in notepad",
            events=lambda event: events.append((event.type, event.data)),
            token=self.token,
        )

        self.assertEqual(result.execution_state, "cancelled")
        self.assertEqual(self.formatter.calls, [])
        self.assertEqual(self.writer.calls, [])
        self.assertIn(("cancelled", {"conversation_id": result.conversation_id}), events)

        stored = self.runtime.get_messages(result.conversation_id)
        assistant = [m for m in stored if m["role"] == "assistant"]
        self.assertEqual(assistant[-1]["execution_state"], "cancelled")

    def test_a_turn_that_finished_first_is_reported_completed(self) -> None:
        """A cancel that arrives after the work finished must not rewrite history.

        The stop request lands during the *last* tool call. The plan does finish,
        so the turn is reported as completed rather than as a stopped one.
        """

        token = CancellationToken()
        tools = [
            RecordingTool("content.generate", {"text": "poem body"}),
            RecordingTool("content.format", {"text": "poem body"}),
            RecordingTool("applications.write_text", {"pid": 1, "application": "Notepad", "characters": 9}, token=token),
        ]
        runtime = ConversationRuntime(
            memory_manager=self.runtime._memory,  # noqa: SLF001
            brain=build_brain(lambda **kwargs: _poem_task(**kwargs), tools),
            context_manager=self.runtime.context_manager,
            user_memory=self.runtime.user_memory,
            ask=lambda **_kwargs: "an answer",
        )
        result = runtime.handle_message("create a poem about cars in notepad", token=token)
        self.assertTrue(token.is_cancelled())
        self.assertEqual(result.execution_state, "completed")
        self.assertEqual(len(tools[-1].calls), 1)

    def test_a_brain_without_a_cancel_parameter_still_works(self) -> None:
        class LegacyBrain:
            def __init__(self) -> None:
                self.requests: list[str] = []
                self.last_context = None

            def process(self, user_input: str) -> str:  # no cancel parameter
                self.requests.append(user_input)
                return f"delegated: {user_input}"

        legacy = LegacyBrain()
        runtime = ConversationRuntime(
            memory_manager=self.runtime._memory,  # noqa: SLF001
            brain=legacy,
            context_manager=self.runtime.context_manager,
            user_memory=self.runtime.user_memory,
            ask=lambda **_kwargs: "an answer",
        )
        result = runtime.handle_message("open notepad", token=CancellationToken())
        self.assertEqual(legacy.requests, ["open notepad"])
        self.assertEqual(result.text, "delegated: open notepad")


class LiveExecutionEventTests(unittest.TestCase):
    """A delegated turn must stream real activity, not one result at the end."""

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        manager = MemoryManager(ask=None, index=_NullIndex())
        manager._store = ConversationStore(root_folder=self.root)  # noqa: SLF001
        manager._sessions._store = manager._store  # noqa: SLF001
        manager._sessions._active_path = self.root / "active_session.json"  # noqa: SLF001
        self.tools = [
            RecordingTool("content.generate", {"text": "poem body"}),
            RecordingTool("content.format", {"text": "poem body"}),
            RecordingTool(
                "applications.write_text",
                {"pid": 1, "application": "Notepad", "characters": 9},
            ),
        ]
        from memory.user_memory import UserMemoryStore

        self.runtime = ConversationRuntime(
            memory_manager=manager,
            brain=build_brain(lambda **kwargs: _poem_task(**kwargs), self.tools),
            context_manager=ContextManager(memory_manager=manager, index=_NullIndex()),
            user_memory=UserMemoryStore(path=self.root / "user_memory.jsonl", embedder=_NullEmbedder()),
            ask=lambda **_kwargs: "an answer",
        )

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_tool_events_arrive_while_the_plan_is_running(self) -> None:
        events: list[tuple[str, dict]] = []
        self.runtime.handle_message(
            "create a poem about cars in notepad",
            events=lambda event: events.append((event.type, event.data)),
        )
        types = [event_type for event_type, _ in events]
        self.assertIn("tool_started", types)
        self.assertIn("tool_completed", types)
        started = [data for name, data in events if name == "tool_started" and data.get("tool")]
        completed = [data for name, data in events if name == "tool_completed" and data.get("tool")]
        # Every real invocation is reported, and no synthetic duplicate is added
        # after the fact for a turn that already streamed its activity.
        self.assertEqual(
            sorted(data["tool"] for data in started),
            sorted(tool.metadata.name for tool in self.tools),
        )
        self.assertEqual(len(completed), len(self.tools))
        # Tool activity is published while the turn runs, before the answer is
        # reported as finished.
        completed_index = types.index("assistant_completed")
        self.assertTrue(all(index < completed_index for index, name in enumerate(types) if name == "tool_completed"))

    def test_no_tool_values_are_published_in_events(self) -> None:
        events: list[tuple[str, dict]] = []
        self.runtime.handle_message(
            "create a poem about cars in notepad",
            events=lambda event: events.append((event.type, event.data)),
        )
        published = " ".join(str(data) for _, data in events)
        # Parameter *names* are reported so a client can show the action; the
        # values (generated text, paths) are not published in the event stream.
        self.assertIn("parameters", published)
        self.assertNotIn("poem body", published)

    def test_verification_is_reported_when_the_executor_verifies(self) -> None:
        events: list[tuple[str, dict]] = []
        self.runtime.handle_message(
            "create a poem about cars in notepad",
            events=lambda event: events.append((event.type, event.data)),
        )
        verifications = [data for name, data in events if name == "verification"]
        self.assertTrue(verifications)
        self.assertTrue(all("tool" in data and "status" in data for data in verifications))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

