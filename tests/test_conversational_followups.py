"""Broad multi-turn conversational reasoning tests (context-aware follow-ups).

These tests are written from the *user's* perspective: each is a small
conversation, not a unit test of one component. They assert the property the
feature exists for —

    a follow-up is understood in the context of the conversation, before Atlas
    decides whether to answer, transform, or act

— for many different kinds of conversation, without any of them being special
cased. The vocabulary in the prompts is deliberately varied so a passing suite
cannot be an artefact of matching one example phrase.

Fresh-context discipline
------------------------
Turn 1 is served, persisted, and its turn-local execution state is *discarded*:
the runtime is rebuilt from the same on-disk store and active session before
turn 2. A test therefore cannot pass merely because an executor or context
object stayed alive in memory; the follow-up must be reconstructed from
persisted messages and tool results, exactly as it would be across a real turn
boundary (or an app restart).

Offline and deterministic
-------------------------
No live model is called. ``ask`` is a recording double so a test can inspect the
*grounded prompt* the model would have received, and the ``Brain`` double records
what it was asked to execute and with what conversation grounding. Capabilities
come from the real registry, so routing is exercised for real.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from memory.context_manager import ContextManager
from memory.conversation_runtime import (
    ConversationRuntime,
)
from memory.memory_manager import MemoryManager
from memory.storage import ConversationStore


class _NullIndex:
    available = False

    def search(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return []

    def index_message(self, **_kwargs: Any) -> bool:
        return False

    def forget_conversation(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _EmbedderStub:
    def index(self, *_args: Any, **_kwargs: Any) -> bool:
        return False

    def similarities(self, *_args: Any, **_kwargs: Any) -> dict[str, float]:
        return {}

    def delete(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _RecordingAsk:
    """A deterministic model boundary that records the prompt it was given."""

    def __init__(self, answer: str = "A grounded answer.") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def __call__(self, *, system_prompt: str, user_prompt: str) -> str:  # noqa: ARG002
        self.prompts.append(user_prompt)
        return self.answer


class _RecordingBrain:
    """A Brain double that records the input and the grounding it received."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.last_context = None

    def process(
        self,
        user_input: str,
        cancel: Any = None,  # noqa: ARG002
        on_event: Any = None,  # noqa: ARG002
        conversation_context: str = "",
    ) -> str:
        self.calls.append({"input": user_input, "context": conversation_context})
        return f"delegated: {user_input}"


class _Conversation:
    """One persisted conversation with rebuildable, turn-local runtime state.

    ``send`` builds a *fresh* runtime from the persisted store for every turn, so
    every turn after the first is served by an object that has no turn-local state
    from the previous turn. That is what makes these tests about crossing turn
    boundaries rather than about object identity.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self.memory = MemoryManager(ask=None, index=_NullIndex())
        self.memory._store = ConversationStore(root_folder=root)  # noqa: SLF001
        self.memory._sessions._store = self.memory._store  # noqa: SLF001
        self.memory._sessions._active_path = root / "active_session.json"  # noqa: SLF001
        self.ask = _RecordingAsk()
        self.brain = _RecordingBrain()
        self.memory.create_session(title="conversation")

    def _runtime(self) -> ConversationRuntime:
        return ConversationRuntime(
            memory_manager=self.memory,
            brain=self.brain,
            context_manager=ContextManager(memory_manager=self.memory, index=_NullIndex()),
            ask=self.ask,
        )

    def send(self, text: str):
        self.ask.prompts.clear()
        self.brain.calls.clear()
        return self._runtime().handle_message(text)

    def send_all(self, *messages: str):
        """Serve several turns, keeping the record of every delegated turn."""

        self.brain.calls.clear()
        results = []
        for message in messages:
            self.ask.prompts.clear()
            results.append(self._runtime().handle_message(message))
        return results

    @property
    def last_model_prompt(self) -> str:
        return self.ask.prompts[-1] if self.ask.prompts else ""

    @property
    def last_brain_call(self) -> dict[str, Any] | None:
        return self.brain.calls[-1] if self.brain.calls else None


class _Sandbox:
    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def conversation(self) -> _Conversation:
        return _Conversation(self.root)

    def cleanup(self) -> None:
        self._temporary.cleanup()


class ContextualFollowUpTests(unittest.TestCase):
    """The whole point: a follow-up is grounded in what was actually discussed."""

    def setUp(self) -> None:
        self.sandbox = _Sandbox()

    def tearDown(self) -> None:
        self.sandbox.cleanup()

    def test_contextual_question_is_answered_with_the_prior_turn(self) -> None:
        # Explain X -> "Why?"  The answer must see X, not start from nothing.
        chat = self.sandbox.conversation()
        chat.send("Explain Retrieval Augmented Generation and how chunking fits in.")
        result = chat.send("Why does chunking matter?")

        self.assertFalse(chat.brain.calls, "a contextual question is not a task")
        prompt = chat.last_model_prompt
        self.assertIn("Retrieval Augmented Generation", prompt)
        self.assertIn("Why does chunking matter?", prompt)
        self.assertEqual(result.kind, "conversation")

    def test_contextual_reasoning_question_uses_the_discussion(self) -> None:
        chat = self.sandbox.conversation()
        chat.send("We could store everything in SQLite for the desktop app.")
        chat.send("Is that actually a good idea?")

        self.assertFalse(chat.brain.calls)
        self.assertIn("SQLite", chat.last_model_prompt)

    def test_implicit_attribute_question_is_grounded(self) -> None:
        # "What about performance?" names no subject at all.
        chat = self.sandbox.conversation()
        chat.send("Describe how Kubernetes schedules containers across nodes.")
        chat.send("What about performance?")

        self.assertFalse(chat.brain.calls)
        self.assertIn("Kubernetes", chat.last_model_prompt)

    def test_implicit_context_without_an_explicit_reference(self) -> None:
        chat = self.sandbox.conversation()
        chat.send("Research the current state of nuclear fusion.")
        chat.send("What are the biggest obstacles?")

        prompt = chat.last_model_prompt
        self.assertIn("nuclear fusion", prompt)
        self.assertIn("biggest obstacles", prompt)

    def test_selection_over_multiple_previous_results_is_grounded(self) -> None:
        chat = self.sandbox.conversation()
        chat.send("Find the best laptops under 50000 pesos with strong CPUs.")
        chat.send("Which one would you choose for local AI?")

        prompt = chat.last_model_prompt
        self.assertIn("laptops", prompt)
        self.assertIn("Which one would you choose for local AI?", prompt)

    def test_missing_context_asks_instead_of_hallucinating(self) -> None:
        # No prior turn at all: a bare reference must not be answered from nothing.
        chat = self.sandbox.conversation()
        result = chat.send("Can you improve that?")

        self.assertFalse(chat.brain.calls)
        # Either a clarification, or a conversational turn whose prompt contains no
        # fabricated referent. What must NOT happen is a confident grounded answer.
        if result.kind != "clarification":
            self.assertNotIn("previous", chat.last_model_prompt.casefold())


class ContextualExecutionTests(unittest.TestCase):
    """Contextual messages that *are* executable must reach the agent, grounded."""

    def setUp(self) -> None:
        self.sandbox = _Sandbox()

    def tearDown(self) -> None:
        self.sandbox.cleanup()

    def test_transform_previous_content_becomes_an_action_with_context(self) -> None:
        chat = self.sandbox.conversation()
        chat.send("Write a professional introduction about the Atlas project.")
        result = chat.send("Make it sound less corporate.")

        call = chat.last_brain_call
        self.assertIsNotNone(call, "a transformation of prior content must reach the agent")
        self.assertTrue(call["context"], "the agent must receive the conversation grounding")
        self.assertIn("Atlas", call["context"])
        self.assertIn(result.kind, {"computer", "task", "hybrid"})

    def test_deliver_previous_output_to_an_application(self) -> None:
        chat = self.sandbox.conversation()
        chat.send("Summarize the key points of the Atacama Desert climate.")
        chat.send("Put that in Notepad.")

        call = chat.last_brain_call
        self.assertIsNotNone(call)
        self.assertTrue(call["context"])

    def test_computer_workflow_continues_across_turns(self) -> None:
        chat = self.sandbox.conversation()
        chat.send_all("Open Chrome.", "Go to GitHub.", "Search for Atlas AI.")

        inputs = [call["input"] for call in chat.brain.calls]
        # Every step of the workflow reaches the agent (none is swallowed as chat).
        self.assertEqual(inputs, ["Open Chrome.", "Go to GitHub.", "Search for Atlas AI."])
        # The follow-ups carry the grounding of the turns that came before them.
        self.assertTrue(chat.brain.calls[1]["context"])
        self.assertIn("Chrome", chat.brain.calls[1]["context"])
        self.assertTrue(chat.brain.calls[2]["context"])

    def test_same_task_for_a_new_subject_still_reaches_the_agent(self) -> None:
        chat = self.sandbox.conversation()
        chat.send("Summarize the film Arrival into Notepad.")
        chat.send("Now do the same for Interstellar.")

        call = chat.last_brain_call
        self.assertIsNotNone(call)
        self.assertTrue(call["context"])


class LongConversationTests(unittest.TestCase):
    """Relevance must survive a long conversation, not just the last turn."""

    def setUp(self) -> None:
        self.sandbox = _Sandbox()

    def tearDown(self) -> None:
        self.sandbox.cleanup()

    def test_follow_up_after_many_unrelated_turns_still_finds_the_topic(self) -> None:
        chat = self.sandbox.conversation()
        chat.send("Our persistence decision is to use SQLite for the desktop app.")
        for index in range(12):
            chat.send(f"Unrelated small talk number {index}.")
        chat.send("Remind me why we picked that database engine?")

        prompt = chat.last_model_prompt
        self.assertIn("SQLite", prompt)


class ModelIndependenceTests(unittest.TestCase):
    """Grounding is provided by the architecture, not by a model's memory."""

    def setUp(self) -> None:
        self.sandbox = _Sandbox()

    def tearDown(self) -> None:
        self.sandbox.cleanup()

    def test_grounding_is_in_the_prompt_not_left_to_the_model(self) -> None:
        chat = self.sandbox.conversation()
        chat.send("The deployment target is a single Windows 11 workstation.")
        chat.send("Would that still work with a smaller GPU?")

        prompt = chat.last_model_prompt
        # The grounding is *supplied*: the earlier turn is in the prompt verbatim,
        # so a weaker local model does not have to remember it.
        self.assertIn("Windows 11", prompt)


class ClassificationGeneralityTests(unittest.TestCase):
    """Contextual detection must be structural, not a list of example phrases."""

    def test_it_is_not_a_phrase_list(self) -> None:
        # These all continue a conversation without containing any of the words
        # from the examples in the specification.
        from reasoning.reference_resolver import ReferenceResolver

        resolver = ReferenceResolver()
        history = (
            "User: We are designing the telemetry pipeline.\n"
            "Assistant: The telemetry pipeline batches events before shipping."
        )
        for text in (
            "What about throughput?",
            "Could that become a bottleneck?",
            "Is that safe under load?",
            "How so?",
            "Which trade-off matters more?",
        ):
            with self.subTest(text=text):
                resolved = resolver.resolve(text, history=history)
                self.assertTrue(resolved.contextual, resolved.to_dict())

    def test_a_self_contained_request_is_not_marked_contextual(self) -> None:
        from reasoning.reference_resolver import ReferenceResolver

        resolver = ReferenceResolver()
        history = "User: Tell me about wine regions.\nAssistant: Wine regions differ by climate."
        resolved = resolver.resolve("Write a poem about motorcycles.", history=history)
        self.assertFalse(resolved.contextual)


if __name__ == "__main__":
    unittest.main()
