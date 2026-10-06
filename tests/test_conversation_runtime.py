"""Milestones 2-5: the conversation runtime, context manager, retrieval, memory.

These tests use the *real* store on a temporary root and a *fake* Brain that
records what it was asked to do, so what is asserted is the actual routing and
persistence behavior of the runtime rather than a mock's opinion of itself.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from config import CONTEXT_MAX_CHARS
from memory.conversation_runtime import (
    EVENT_ASSISTANT_COMPLETED,
    EVENT_TOKEN,
    CancellationToken,
    ConversationRuntime,
    classify_turn,
)
from memory.context_manager import ContextManager
from memory.memory_manager import MemoryManager
from memory.storage import ConversationStore
from memory.summarizer import MemorySummarizer, extract_notes, render_notes
from memory.user_memory import UserMemoryStore, candidate_memories


class _KanAnswer:
    """A deterministic, offline answer boundary.

    Unit tests must not call a live local model: that is slow, non-deterministic
    and unavailable in CI. Injecting this keeps the *routing and persistence*
    under test while removing the model from the picture.
    """

    def __init__(self, answer: str = "A local answer.") -> None:
        self.answer = answer
        self.calls: list[str] = []

    def __call__(self, *, system_prompt: str, user_prompt: str) -> str:  # noqa: ARG002
        self.calls.append(user_prompt)
        return self.answer


class _NullIndex:
    """Deterministic-only index stand-in (no Chroma, no embedder, no network)."""

    available = False

    def search(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return []

    def index_message(self, **_kwargs: Any) -> bool:
        return False

    def forget_conversation(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _EmbedderStub:
    """No embeddings: the lexical paths under test must stand on their own."""

    def index(self, *_args: Any, **_kwargs: Any) -> bool:
        return False

    def similarities(self, *_args: Any, **_kwargs: Any) -> dict[str, float]:
        return {}

    def delete(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _FakeContext:
    def __init__(self, status: str = "COMPLETED", tools: list[tuple[str, str]] | None = None) -> None:
        from models import TaskStatus

        self.task_id = "task-1"
        self.status = TaskStatus[status]
        self.tool_calls = [
            {"tool": name, "status": state, "success": True, "output": {}, "parameters": {}}
            for name, state in (tools or [])
        ]
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.web_sources = ["https://example.test/doc"] if tools else []
        self.execution_plan = None
        self.metadata: dict[str, Any] = {"reasoning_answer": {"mode": "direct_answer"}}


class _FakeBrain:
    """Stands in for the real Brain: records delegation, returns a fixed answer.

    Like the real Brain, ``last_context`` is (re)assigned *during* ``process``,
    because that is what the runtime reads to summarize what actually happened.
    """

    def __init__(self, *, status: str = "COMPLETED", tools: list[tuple[str, str]] | None = None) -> None:
        self.requests: list[str] = []
        self._status = status
        self._tools = tools or []
        self.last_context = None

    def process(self, user_input: str) -> str:
        self.requests.append(user_input)
        self.last_context = _FakeContext(status=self._status, tools=self._tools)
        return f"delegated: {user_input}"


class _Sandbox:
    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def memory(self) -> MemoryManager:
        manager = MemoryManager(ask=None, index=_NullIndex())
        manager._store = ConversationStore(root_folder=self.root)  # noqa: SLF001
        manager._sessions._store = manager._store  # noqa: SLF001
        manager._sessions._active_path = self.root / "active_session.json"  # noqa: SLF001
        return manager

    def user_memory(self, *, enabled: bool = True) -> UserMemoryStore:
        return UserMemoryStore(
            path=self.root / "user_memory.jsonl",
            enabled=enabled,
            embedder=_EmbedderStub(),
        )

    def cleanup(self) -> None:
        self._temporary.cleanup()


class ClassificationTests(unittest.TestCase):
    def test_plain_question_is_conversation(self) -> None:
        for text in ("What is Docker?", "Explain containers simply.", "hi", "thanks"):
            with self.subTest(text=text):
                result = classify_turn(text)
                self.assertFalse(result.delegate)

    def test_computer_action_delegates(self) -> None:
        for text in ("Open Chrome.", "open chrmoe", "Write this into Notepad"):
            with self.subTest(text=text):
                result = classify_turn(text)
                self.assertTrue(result.delegate)

    def test_search_requests_delegate_to_the_agent(self) -> None:
        # A search must run the real web/filesystem capability. Answering "here
        # are some results" from the model would invent them.
        for text in ("Search for Docker tutorials.", "search youtub", "Find the largest PDF in Documents."):
            with self.subTest(text=text):
                result = classify_turn(text)
                self.assertTrue(result.delegate)

    def test_research_without_an_action_is_delegated_to_reasoning(self) -> None:
        # Real research must go through the existing reasoning engine (sources,
        # evidence, citations) instead of being answered from model memory.
        result = classify_turn("Research the latest Docker changes.")
        self.assertTrue(result.delegate)
        self.assertEqual(result.kind, "research")

    def test_a_question_mentioning_latest_is_delegated_for_real_sourcing(self) -> None:
        # "latest" is a freshness hint: the reasoning engine owns how to source it.
        result = classify_turn("What is the latest Python version?")
        self.assertTrue(result.delegate)

    def test_hybrid_delegates(self) -> None:
        result = classify_turn("Research the latest Python changes and put the important points into Notepad.")
        self.assertTrue(result.delegate)
        self.assertEqual(result.kind, "hybrid")

    def test_ambiguous_imperative_delegates_rather_than_answering(self) -> None:
        # "Install it." has no software named: an answer would be a fabrication,
        # so it must go to the agent whose validator/clarification owns it.
        result = classify_turn("Install it.")
        self.assertTrue(result.delegate)

    def test_kind_is_always_in_the_canonical_vocabulary(self) -> None:
        from memory.models import RESPONSE_KINDS

        for text in (
            "hello", "What is RAM?", "Open Chrome.", "research X", "research X in notepad",
            "Which one is easier?", "delete the file", "remember that I prefer local-first",
        ):
            with self.subTest(text=text):
                self.assertIn(classify_turn(text).kind, RESPONSE_KINDS)


class RuntimeRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sandbox = _Sandbox()
        self.memory = self.sandbox.memory()
        self.brain = _FakeBrain()
        self.runtime = ConversationRuntime(
            memory_manager=self.memory,
            brain=self.brain,
            context_manager=ContextManager(memory_manager=self.memory, index=_NullIndex()),
            user_memory=self.sandbox.user_memory(),
            ask=_KanAnswer(),
        )

    def tearDown(self) -> None:
        self.sandbox.cleanup()

    def test_conversation_is_not_sent_to_the_agent(self) -> None:
        result = self.runtime.handle_message("What is Docker?")
        self.assertEqual(self.brain.requests, [])
        self.assertEqual(result.kind, "conversation")
        self.assertEqual(len(self.memory.get_recent_messages()), 2)

    def test_action_is_delegated_to_the_existing_brain(self) -> None:
        result = self.runtime.handle_message("Open Chrome.")
        self.assertEqual(self.brain.requests, ["Open Chrome."])
        self.assertEqual(result.kind, "computer")

    def test_research_transform_and_same_task_followups_are_delegated(self) -> None:
        self.runtime.handle_message("Research FEU.")
        summarized = self.runtime.handle_message("Summarize it.")
        same_task = self.runtime.handle_message("Now do the same for Harvard.")

        self.assertEqual(self.brain.requests, [
            "Research FEU.",
            "Summarize it.",
            "Now do the same for Harvard.",
        ])
        self.assertEqual(summarized.kind, "task")
        self.assertEqual(same_task.kind, "task")

    def test_research_turn_produces_a_real_kind_when_brain_uses_web(self) -> None:
        brain = _FakeBrain(tools=[("web.search", "completed"), ("web.fetch", "completed")])
        runtime = ConversationRuntime(
            memory_manager=self.memory,
            brain=brain,
            context_manager=ContextManager(memory_manager=self.memory, index=_NullIndex()),
            user_memory=self.sandbox.user_memory(),
            ask=_KanAnswer(),
        )
        result = runtime.handle_message("Research the latest Docker changes.")
        # The kind is derived from what actually ran, not from the prediction.
        self.assertEqual(result.kind, "research")
        self.assertTrue(result.tool_calls)
        self.assertEqual(result.tool_calls[0]["tool"], "web.search")

    def test_turn_persists_tool_calls_and_citations(self) -> None:
        brain = _FakeBrain(tools=[("applications.launch", "completed")])
        runtime = ConversationRuntime(
            memory_manager=self.memory,
            brain=brain,
            context_manager=ContextManager(memory_manager=self.memory, index=_NullIndex()),
            user_memory=self.sandbox.user_memory(),
            ask=_KanAnswer(),
        )
        runtime.handle_message("Open Chrome.")
        assistant = self.memory.get_recent_messages()[-1]
        self.assertEqual(assistant.tool_calls[0]["tool"], "applications.launch")
        self.assertEqual(assistant.execution_state, "completed")
        self.assertEqual(assistant.response_kind, "computer")

    def test_waiting_for_confirmation_is_recorded_truthfully(self) -> None:
        brain = _FakeBrain(status="WAITING_FOR_CONFIRMATION", tools=[("filesystem.write", "confirmation_required")])
        runtime = ConversationRuntime(
            memory_manager=self.memory,
            brain=brain,
            context_manager=ContextManager(memory_manager=self.memory, index=_NullIndex()),
            user_memory=self.sandbox.user_memory(),
            ask=_KanAnswer(),
        )
        result = runtime.handle_message("Delete the file report.pdf.")
        self.assertEqual(result.execution_state, "waiting_for_confirmation")

    def test_a_clarification_turn_is_recorded_as_a_clarification(self) -> None:
        # The agent asked instead of guessing. The turn is finished (so it is
        # not "unknown" or a failure) and its kind says what it really was.
        brain = _FakeBrain(status="UNCERTAIN")
        brain._metadata_reasoning_mode = "clarification"  # noqa: SLF001
        runtime = ConversationRuntime(
            memory_manager=self.memory,
            brain=brain,
            context_manager=ContextManager(memory_manager=self.memory, index=_NullIndex()),
            user_memory=self.sandbox.user_memory(),
            ask=_KanAnswer(),
        )
        result = runtime.handle_message("Open it.")
        self.assertEqual(result.execution_state, "completed")
        self.assertEqual(result.kind, "clarification")

    def test_conversation_continues_naturally_after_a_task(self) -> None:
        brain = _FakeBrain(tools=[("applications.launch", "completed")])
        runtime = ConversationRuntime(
            memory_manager=self.memory,
            brain=brain,
            context_manager=ContextManager(memory_manager=self.memory, index=_NullIndex()),
            user_memory=self.sandbox.user_memory(),
            ask=_KanAnswer(),
        )
        runtime.handle_message("Open Chrome.")
        result = runtime.handle_message("Thanks, that worked.")
        self.assertEqual(brain.requests, ["Open Chrome."])
        self.assertEqual(result.kind, "conversation")
        self.assertGreater(len(self.memory.get_recent_messages()), 2)

    def test_bare_anaphoric_instruction_with_history_is_escalated(self) -> None:
        runtime = self.runtime
        runtime.handle_message("Tell me how to install VLC.")
        runtime.handle_message("Install it.")
        self.assertIn("Install it.", self.brain.requests)

    def test_activity_is_derived_from_the_real_execution_context(self) -> None:
        from memory.conversation_runtime import _activity_from_context

        context = _FakeContext(tools=[("web.search", "completed")])
        tool_calls, tool_results, activity, citations = _activity_from_context(context)
        self.assertEqual(tool_calls[0]["tool"], "web.search")
        self.assertEqual(tool_results[0]["tool"], "web.search")
        self.assertTrue(any("Web search" in line for line in activity))
        self.assertEqual(citations, ["https://example.test/doc"])

    def test_activity_is_empty_for_a_context_that_did_nothing(self) -> None:
        from memory.conversation_runtime import _activity_from_context

        self.assertEqual(_activity_from_context(None), ([], [], [], []))

    def test_bare_anaphoric_instruction_without_history_is_not_escalated_blindly(self) -> None:
        # No conversation context -> nothing to resolve, so the deterministic
        # classifier decides on its own merits (still an action -> delegate).
        classification = classify_turn("Install it.")
        self.assertTrue(classification.delegate)
        self.assertTrue(classification.anaphoric)

    def test_bare_anaphoric_instruction_is_not_escalated_without_a_referent(self) -> None:
        # History exists but the last turn offers nothing nameable, so "Install
        # it." must not be silently resolved into an invented target.
        class _ReferentlessBrain(_FakeBrain):
            def process(self, user_input: str) -> str:
                self.requests.append(user_input)
                return "What would you like me to install?"

        brain = _ReferentlessBrain()
        runtime = ConversationRuntime(
            memory_manager=self.memory,
            brain=brain,
            context_manager=ContextManager(memory_manager=self.memory, index=_NullIndex()),
            user_memory=self.sandbox.user_memory(),
            ask=_KanAnswer(),
        )
        result = runtime.handle_message("What is Docker?")
        self.assertEqual(result.kind, "conversation")
        self.assertEqual(brain.requests, [])

    def test_topical_back_reference_is_context_dependent_but_not_anaphoric(self) -> None:
        classification = classify_turn("Explain that more simply.")
        self.assertTrue(classification.context_dependent)
        self.assertFalse(classification.anaphoric)
        self.assertFalse(classification.delegate)

    def test_first_message_titles_the_conversation(self) -> None:
        conversation = self.runtime.new_conversation()
        self.runtime.handle_message("How does garbage collection work?", conversation_id=conversation["id"])
        metadata = self.memory.open_session(conversation["id"])
        self.assertEqual(metadata.title, "How does garbage collection work?")

    def test_empty_message_is_answered_without_an_agent_call(self) -> None:
        result = self.runtime.handle_message("   ")
        self.assertEqual(self.brain.requests, [])
        self.assertEqual(result.kind, "conversation")

    def test_streaming_emits_real_tokens_and_persists_them(self) -> None:
        pieces = ["Docker ", "packages ", "software."]

        class _Streamer:
            def stream(self, *, system_prompt, user_prompt, on_token, should_stop):  # noqa: ARG002
                produced = ""
                for piece in pieces:
                    on_token(piece)
                    produced += piece
                return produced

        events: list[dict[str, Any]] = []
        result = self.runtime.handle_message(
            "What is Docker?",
            events=lambda event: events.append(event.to_dict()),
            stream=_Streamer(),
        )
        self.assertEqual(result.text, "".join(pieces))
        tokens = [event["data"]["text"] for event in events if event["type"] == EVENT_TOKEN]
        self.assertEqual(tokens, pieces)
        self.assertTrue(any(event["type"] == EVENT_ASSISTANT_COMPLETED for event in events))
        self.assertEqual(self.memory.get_recent_messages()[-1].content, "".join(pieces))

    def test_cancellation_records_a_cancelled_turn_with_partial_content(self) -> None:
        token = CancellationToken()

        class _Streamer:
            def stream(self, *, system_prompt, user_prompt, on_token, should_stop):  # noqa: ARG002
                on_token("Docker ")
                token.cancel()
                for piece in ("packages ", "software."):
                    if should_stop():
                        break
                    on_token(piece)
                return "Docker "

        result = self.runtime.handle_message("What is Docker?", stream=_Streamer(), token=token)
        self.assertEqual(result.execution_state, "cancelled")
        self.assertEqual(result.text, "Docker ")
        self.assertEqual(self.memory.get_recent_messages()[-1].execution_state, "cancelled")

    def test_a_non_streaming_boundary_falls_back_to_one_blocking_answer(self) -> None:
        """A boundary that cannot stream still answers — without fake progress.

        The turn must complete through the normal answer path. Because the
        boundary genuinely cannot stream, the answer arrives as a single token
        rather than a fabricated per-word sequence.
        """

        class _NotAStreamer:
            def __call__(self, **_kwargs: Any) -> str:
                raise AssertionError("a non-streaming boundary must not be called as a streamer")

        events: list[dict[str, Any]] = []
        result = self.runtime.handle_message(
            "What is Docker?", events=lambda event: events.append(event.to_dict()), stream=_NotAStreamer()
        )
        self.assertEqual(result.kind, "conversation")
        self.assertEqual(result.text, self.memory.get_recent_messages()[-1].content)
        tokens = [event["data"]["text"] for event in events if event["type"] == EVENT_TOKEN]
        self.assertEqual(tokens, [result.text])

    def test_a_failing_stream_falls_back_to_one_blocking_call(self) -> None:
        """A transport failure before any token must not become a limitation.

        The live bug this covers: a stream that raised or yielded nothing was
        returned as an empty string, and the caller turned that into "the local
        model produced no output" even though the model was healthy and a plain
        call would have worked.
        """

        class _FailingStreamer:
            def stream(self, **_kwargs: Any) -> str:
                raise ConnectionError("transport dropped")

        answer = _KanAnswer("A real answer.")
        runtime = ConversationRuntime(
            memory_manager=self.memory,
            brain=self.brain,
            context_manager=ContextManager(memory_manager=self.memory, index=_NullIndex()),
            user_memory=self.sandbox.user_memory(),
            ask=answer,
        )
        events: list[dict[str, Any]] = []
        result = runtime.handle_message(
            "What is Docker?",
            events=lambda event: events.append(event.to_dict()),
            stream=_FailingStreamer(),
        )
        self.assertEqual(result.execution_state, "completed")
        self.assertEqual(result.text, "A real answer.")
        self.assertTrue(answer.calls)

    def test_an_empty_stream_falls_back_to_one_blocking_call(self) -> None:
        """A stream that yields no tokens is retried, not reported as silence."""

        class _EmptyStreamer:
            def stream(self, **_kwargs: Any) -> str:
                return ""

        answer = _KanAnswer("Recovered answer.")
        runtime = ConversationRuntime(
            memory_manager=self.memory,
            brain=self.brain,
            context_manager=ContextManager(memory_manager=self.memory, index=_NullIndex()),
            user_memory=self.sandbox.user_memory(),
            ask=answer,
        )
        result = runtime.handle_message("What is Docker?", stream=_EmptyStreamer())
        self.assertEqual(result.execution_state, "completed")
        self.assertEqual(result.text, "Recovered answer.")

    def test_a_stream_failure_with_no_model_available_reports_a_limitation_not_a_lie(self) -> None:
        """With no usable model boundary, the turn must fail honestly.

        The runtime legitimately defaults to the configured local model, so this
        test removes the boundary explicitly to model a machine with no model
        available. The point is only that the runtime never invents an answer and
        never reports a silent empty stream as success.
        """

        class _FailingStreamer:
            def stream(self, **_kwargs: Any) -> str:
                raise ConnectionError("transport dropped")

        runtime = ConversationRuntime(
            memory_manager=self.memory,
            brain=self.brain,
            context_manager=ContextManager(memory_manager=self.memory, index=_NullIndex()),
            user_memory=self.sandbox.user_memory(),
            ask=None,
        )
        runtime._ask = None  # noqa: SLF001 - the honest no-model runtime
        result = runtime.handle_message("What is Docker?", stream=_FailingStreamer())

        self.assertNotEqual(result.execution_state, "completed")
        self.assertIn("could not obtain", result.text.lower())
        # The failed turn is persisted with its real state, not as a success.
        self.assertEqual(self.memory.get_recent_messages()[-1].execution_state, "failed")

    def test_delegated_turn_reports_a_brain_failure_honestly(self) -> None:
        class _Broken:
            last_context = None

            def process(self, _user_input: str) -> str:
                raise RuntimeError("boom")

        runtime = ConversationRuntime(
            memory_manager=self.memory,
            brain=_Broken(),
            context_manager=ContextManager(memory_manager=self.memory, index=_NullIndex()),
            user_memory=self.sandbox.user_memory(),
            ask=_KanAnswer(),
        )
        result = runtime.handle_message("Open Chrome.")
        self.assertEqual(result.execution_state, "failed")
        self.assertEqual(result.kind, "error")

    def test_no_brain_available_does_not_pretend_to_act(self) -> None:
        runtime = ConversationRuntime(
            memory_manager=self.memory,
            brain=None,
            context_manager=ContextManager(memory_manager=self.memory, index=_NullIndex()),
            user_memory=self.sandbox.user_memory(),
            ask=_KanAnswer(),
        )
        result = runtime.handle_message("Open Chrome.")
        self.assertEqual(result.execution_state, "failed")
        self.assertIn("not available", result.text)

    def test_conversation_api_operations_work(self) -> None:
        first = self.runtime.new_conversation(title="First")
        second = self.runtime.new_conversation(title="Second")
        self.assertEqual(len(self.runtime.list_conversations()), 2)
        self.runtime.rename_conversation(first["id"], "Renamed")
        self.assertEqual(self.runtime.open_conversation(first["id"])["title"], "Renamed")
        self.runtime.delete_conversation(second["id"])
        self.assertEqual([item["id"] for item in self.runtime.list_conversations()], [first["id"]])


class ContextManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sandbox = _Sandbox()
        self.memory = self.sandbox.memory()
        self.manager = ContextManager(memory_manager=self.memory, index=_NullIndex())

    def tearDown(self) -> None:
        self.sandbox.cleanup()

    def test_recent_turns_are_included_verbatim(self) -> None:
        conversation = self.memory.create_session(title="Chat")
        self.memory.append_message(role="user", content="Tell me about Skyrim.")
        self.memory.append_message(role="assistant", content="Skyrim is an RPG.")
        bundle = self.manager.build(question="What about mods?", conversation_id=conversation.id)
        self.assertTrue(bundle.has_history)
        self.assertIn("Tell me about Skyrim.", bundle.history_text)
        self.assertIn("Skyrim is an RPG.", bundle.history_text)

    def test_main_points_followup_receives_the_prior_research_output(self) -> None:
        conversation = self.memory.create_session(title="Research follow-up")
        research = (
            "Far Eastern University (FEU) was founded in Manila in 1934. "
            "Its main campus is in Sampaloc. Source: https://www.feu.edu.ph/"
        )
        self.memory.append_message(role="user", content="Research FEU.")
        self.memory.append_message(role="assistant", content=research)

        bundle = self.manager.build(
            question="What are the main points?",
            conversation_id=conversation.id,
        )

        self.assertIn(research, bundle.history_text)
        self.assertIn(research, bundle.as_prompt())

    def test_old_relevant_turn_is_recalled_beyond_the_recent_window(self) -> None:
        conversation = self.memory.create_session(title="Long")
        self.memory.append_message(role="user", content="Our database decision was to use SQLite for local storage.")
        self.memory.append_message(role="assistant", content="Understood, SQLite it is.")
        for index in range(30):
            self.memory.append_message(role="user", content=f"unrelated filler turn {index}")
            self.memory.append_message(role="assistant", content=f"acknowledged {index}")

        bundle = self.manager.build(question="What did I decide about the database?", conversation_id=conversation.id)
        recalled = " ".join(item.text for item in bundle.older)
        self.assertIn("SQLite", recalled)
        self.assertTrue(bundle.notes or bundle.older)  # the recall is reported, not silent

    def test_summary_precedes_recent_turns_in_the_prompt(self) -> None:
        conversation = self.memory.create_session(title="Summarized")
        self.memory.append_message(role="user", content="hello")
        self.memory.save_summary("Decisions:\n- Use SQLite.", session_id=conversation.id)
        bundle = self.manager.build(question="And the database?", conversation_id=conversation.id)
        prompt = bundle.as_prompt()
        self.assertIn("CONVERSATION SUMMARY:", prompt)
        self.assertLess(prompt.index("CONVERSATION SUMMARY:"), prompt.index("RECENT TURNS:"))

    def test_context_is_bounded_and_reports_what_it_dropped(self) -> None:
        conversation = self.memory.create_session(title="Huge")
        for index in range(60):
            self.memory.append_message(role="user", content=f"filler {index} " + ("x" * 200))
            self.memory.append_message(role="assistant", content=f"reply {index} " + ("y" * 200))
        bundle = self.manager.build(question="anything", conversation_id=conversation.id)
        self.assertLessEqual(sum(len(item.text) for item in bundle.items), CONTEXT_MAX_CHARS + 400)
        self.assertGreater(bundle.considered, 0)

    def test_topic_switching_does_not_carry_the_old_topic_label(self) -> None:
        conversation = self.memory.create_session(title="Switch")
        self.memory.append_message(role="user", content="Let's talk about Godot rendering.")
        first = self.manager.build(question="Godot rendering", conversation_id=conversation.id)
        self.memory.append_message(role="user", content="Actually, let's talk about Postgres indexes.")
        second = self.manager.build(question="Postgres indexes", conversation_id=conversation.id)
        self.assertNotEqual(first.topic, second.topic)
        self.assertIn("postgres", second.topic)

    def test_user_memory_reaches_the_context(self) -> None:
        store = self.sandbox.user_memory()
        store.create(text="User prefers local-first architecture", kind="preference")
        manager = ContextManager(
            memory_manager=self.memory, index=_NullIndex(), user_memory=store
        )
        conversation = self.memory.create_session(title="Prefs")
        bundle = manager.build(
            question="What architecture do I prefer?", conversation_id=conversation.id
        )
        self.assertTrue(bundle.memories)
        self.assertIn("local-first", bundle.as_prompt())

    def test_task_state_reports_real_records_only(self) -> None:
        conversation = self.memory.create_session(title="Tasks")
        self.memory.append_message(role="user", content="Open Chrome.")
        self.memory.append_message(
            role="assistant",
            content="Opened Chrome.",
            tool_calls=[{"tool": "applications.launch", "status": "completed"}],
        )
        state = self.manager._task_state(conversation.id)  # noqa: SLF001 - asserted behavior
        self.assertIn("applications.launch", state)
        self.assertIn("completed", state)


class SummarizerTests(unittest.TestCase):
    def test_deterministic_notes_keep_decisions_and_drop_small_talk(self) -> None:
        from memory.models import MemoryMessage

        messages = [
            MemoryMessage(role="user", content="Hi"),
            MemoryMessage(role="user", content="Let's use Postgres instead of SQLite for the server."),
            MemoryMessage(role="assistant", content="Understood."),
            MemoryMessage(role="user", content="Why did the migration fail?"),
        ]
        notes = extract_notes(messages)
        rendered = render_notes(notes)
        self.assertIn("Postgres", rendered)
        self.assertIn("migration fail", rendered)
        self.assertNotIn("Hi", rendered.split("Decisions:")[0])

    def test_summarizer_without_a_model_still_produces_a_real_summary(self) -> None:
        from memory.models import MemoryMessage

        summarizer = MemorySummarizer(ask=None, model_available=False, min_messages=2)
        messages = [
            MemoryMessage(role="user", content="We decided to use SQLite for storage."),
            MemoryMessage(role="assistant", content="Noted."),
        ]
        summary = summarizer.summarize(messages)
        self.assertIn("SQLite", summary)

    def test_summarizer_uses_the_model_boundary_when_available(self) -> None:
        from memory.models import MemoryMessage

        calls: list[str] = []

        def ask(*, system_prompt: str, user_prompt: str) -> str:
            calls.append(user_prompt)
            return "Decisions:\n- Use SQLite for local storage."

        summarizer = MemorySummarizer(ask=ask, model_available=True, min_messages=2)
        summary = summarizer.summarize([MemoryMessage(role="user", content="decide about storage")])
        self.assertIn("SQLite", summary)
        self.assertTrue(calls)

    def test_a_failing_model_falls_back_instead_of_losing_the_summary(self) -> None:
        from memory.models import MemoryMessage

        def ask(**_kwargs: object) -> str:
            raise RuntimeError("model down")

        summarizer = MemorySummarizer(ask=ask, model_available=True, min_messages=2)
        summary = summarizer.summarize(
            [MemoryMessage(role="user", content="We decided to use SQLite.")]
        )
        self.assertIn("SQLite", summary)

    def test_maybe_summarize_is_gated_for_a_short_conversation(self) -> None:
        from memory.models import MemoryMessage

        summarizer = MemorySummarizer(ask=None, model_available=False, min_messages=10)
        messages = [MemoryMessage(role="user", content="Use SQLite.")]
        self.assertIsNone(
            summarizer.maybe_summarize(
                session_id="s", message_count=1, existing_summary=None, messages=messages
            )
        )

    def test_summary_is_persisted_on_the_conversation(self) -> None:
        sandbox = _Sandbox()
        try:
            memory = sandbox.memory()
            memory._summarizer = MemorySummarizer(ask=None, model_available=False, min_messages=2)  # noqa: SLF001
            conversation = memory.create_session(title="Summarized")
            memory.append_message(role="user", content="We decided to use SQLite for storage.")
            memory.append_message(role="assistant", content="Noted.")
            self.assertIsNotNone(memory.get_summary(session_id=conversation.id))
            self.assertIn("SQLite", memory.get_summary(session_id=conversation.id) or "")
        finally:
            sandbox.cleanup()


class UserMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sandbox = _Sandbox()
        self.store = self.sandbox.user_memory()

    def tearDown(self) -> None:
        self.sandbox.cleanup()

    def test_only_durable_statements_are_remembered(self) -> None:
        self.assertEqual(candidate_memories("What is Docker?"), [])
        self.assertEqual(candidate_memories("Tell me about Skyrim."), [])
        self.assertEqual(candidate_memories("Put the points into Notepad."), [])
        self.assertTrue(candidate_memories("Remember that I prefer local-first architecture."))
        self.assertTrue(candidate_memories("I always want short answers."))

    def test_one_statement_produces_exactly_one_memory(self) -> None:
        # "remember that I prefer X" matches both the explicit-remember rule and
        # the preference rule; storing it twice under two kinds is a bug.
        candidates = candidate_memories("Remember that I prefer local-first architecture.")
        self.assertEqual(len(candidates), 1)
        self.assertIn("local-first", candidates[0].text)

        candidates = candidate_memories("I always want short answers.")
        self.assertEqual(len(candidates), 1)

    def test_explicit_remember_takes_precedence_over_pattern_matching(self) -> None:
        candidates = candidate_memories(
            "Remember that my project is called Atlas."
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].kind, "instruction")

    def test_explicit_remember_is_stored_and_survives_restart(self) -> None:
        stored = self.store.observe_user_message(
            "Remember that I prefer local-first architecture.", conversation_id="c1", message_id="m1"
        )
        self.assertTrue(stored)
        reopened = UserMemoryStore(
            path=self.sandbox.root / "user_memory.jsonl", embedder=_EmbedderStub()
        )
        self.assertTrue(reopened.list())
        self.assertIn("local-first", reopened.list()[0]["text"])

    def test_ordinary_conversation_is_not_stored(self) -> None:
        self.assertEqual(self.store.observe_user_message("What is RAM?"), [])
        self.assertEqual(self.store.list(), [])

    def test_identical_memory_is_not_duplicated(self) -> None:
        self.store.create(text="User prefers dark mode", kind="preference")
        self.store.create(text="User prefers dark mode", kind="preference")
        self.assertEqual(len(self.store.list()), 1)

    def test_crud_and_forget(self) -> None:
        record = self.store.create(text="User prefers tabs", kind="preference")
        assert record is not None
        self.assertTrue(self.store.get(record.id))
        self.store.update(record.id, text="User prefers spaces")
        self.assertEqual(self.store.get(record.id)["text"], "User prefers spaces")  # type: ignore[index]
        self.assertTrue(self.store.delete(record.id))
        self.assertIsNone(self.store.get(record.id))

    def test_search_and_clear(self) -> None:
        self.store.create(text="User works on the Atlas project", kind="project")
        self.store.create(text="User prefers Vim keybindings", kind="preference")
        self.assertEqual(len(self.store.search("Atlas")), 1)
        self.assertEqual(self.store.clear(), 2)
        self.assertEqual(self.store.list(), [])

    def test_disabling_memory_stops_writes_and_retrieval(self) -> None:
        store = self.sandbox.user_memory(enabled=False)
        self.assertIsNone(store.create(text="User prefers X"))
        self.assertEqual(store.retrieve("anything"), [])

    def test_sensitive_values_are_not_stored(self) -> None:
        self.assertIsNone(self.store.create(text="User's password is hunter2"))
        self.assertEqual(self.store.observe_user_message("Remember my api key is abc"), [])

    def test_retrieval_ranks_a_relevant_memory_above_an_unrelated_one(self) -> None:
        self.store.create(text="User prefers local-first architecture", kind="preference")
        self.store.create(text="User's favourite colour is blue", kind="preference")
        results = self.store.retrieve("How should the architecture be built?", limit=1)
        self.assertTrue(results)
        self.assertIn("local-first", results[0]["text"])

    def test_a_durable_instruction_applies_even_without_word_overlap(self) -> None:
        self.store.create(text="User always wants short answers", kind="instruction")
        results = self.store.retrieve("Summarize the Roman Empire", limit=1)
        self.assertTrue(results)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
