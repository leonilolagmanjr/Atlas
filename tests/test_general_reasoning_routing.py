"""Regression tests for general-purpose routing (spec sections 6, 9, 10, 11, 19, 25).

These lock in the fixes that stop ordinary/general requests from falling through
to a web or "not in knowledge base" route when the semantic intent is a local
file, a self query, or a model-knowledge question.

Everything is offline: the deterministic interpreter runs with the LLM disabled,
the model is a canned stub, and tool execution is a recording fake.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from computer.runtime import register_read_only_tools  # noqa: E402
from config import COMPUTER_ROOT  # noqa: E402
from reasoning.answer_generator import AnswerGenerator  # noqa: E402
from reasoning.query_router import QueryRouter  # noqa: E402
from reasoning.reasoning_engine import ReasoningEngine  # noqa: E402
from reasoning.reasoning_models import ResponseMode, SourceType  # noqa: E402
from reasoning.self_introspection import SelfIntrospection  # noqa: E402
from reasoning.source_selector import SourceSelector  # noqa: E402
from reasoning.task_interpreter import SemanticTaskInterpreter  # noqa: E402
from tools.base import ToolResult  # noqa: E402
from tools.capabilities import CapabilityRegistry  # noqa: E402
from tools.registry import ToolRegistry  # noqa: E402


class FakeAsk:
    def __init__(self, answer: str = "A general-knowledge answer.") -> None:
        self.answer = answer
        self.calls = 0

    def __call__(self, *, system_prompt: str = "", user_prompt: str = "", **_kw) -> str:
        self.calls += 1
        return self.answer


def _capabilities() -> CapabilityRegistry:
    registry = ToolRegistry()
    register_read_only_tools(registry, root=COMPUTER_ROOT, ask=None)
    return CapabilityRegistry(registry)


def _interpreter() -> SemanticTaskInterpreter:
    caps = _capabilities()
    return SemanticTaskInterpreter(ask=None, capabilities=caps, capability_catalog=caps.render_catalog())


def _route(prompt: str):
    interpreter = _interpreter()
    caps = _capabilities()
    task = interpreter.interpret(prompt)
    signals = QueryRouter().route(
        prompt, task=task, self_introspection=SelfIntrospection(caps, model_name="test-model")
    )
    plan = SourceSelector(caps).select(signals, task_type=task.task_type)
    return task, signals, plan


class FileRoutingRegressionTests(unittest.TestCase):
    """Personal-document lookups must be local file requests, never web search."""

    def test_find_my_resume_is_a_file_lookup(self):
        task, signals, plan = _route("Find my resume.")
        self.assertEqual([a.capability for a in task.actions], ["filesystem.search"])
        self.assertNotIn("web", [s.value for s in plan.sources])
        self.assertIn("files", [s.value for s in plan.sources])
        self.assertIsNotNone(signals.file_intent)
        self.assertEqual(signals.file_intent.subject, "resume")

    def test_search_my_documents_for_phrase_is_content_search(self):
        task, signals, plan = _route("Search my documents for 'climate change'.")
        self.assertEqual(
            [a.capability for a in task.actions], ["filesystem.search_content"]
        )
        self.assertEqual(task.actions[0].parameters["query"], "climate change")
        self.assertNotIn("web", [s.value for s in plan.sources])

    def test_explicit_web_platform_overrides_file_hint(self):
        # A named web platform still wins over a weak personal-document hint.
        _, signals, plan = _route("Search youtube for my resume videos.")
        self.assertIn("web", [s.value for s in plan.sources])

    def test_plain_web_search_still_routes_to_web(self):
        task, _, plan = _route("Find the latest NVIDIA GPU.")
        self.assertIn("web", [s.value for s in plan.sources])


class CreateFileRegressionTests(unittest.TestCase):
    def test_create_text_file_produces_a_write_action(self):
        task, _, plan = _route("Create a text file.")
        self.assertEqual([a.capability for a in task.actions], ["filesystem.write"])
        self.assertEqual(task.actions[0].parameters["path"], "atlas_created_txt.txt")
        self.assertTrue(plan.requires_action)

    def test_create_file_with_name_uses_that_name(self):
        task = _interpreter().interpret("Create a text file named notes.txt")
        capabilities = [a.capability for a in task.actions]
        # A named file request still writes the file; generating content first is
        # an acceptable (and helpful) prefix step.
        self.assertIn("filesystem.write", capabilities)
        write = task.actions[-1]
        self.assertEqual(write.capability, "filesystem.write")
        self.assertEqual(write.parameters["path"], "notes.txt")


class HybridRegressionTests(unittest.TestCase):
    def test_search_then_write_composes_a_write_action(self):
        task, _, plan = _route(
            "Search the web for the latest Python version and write it into Notepad."
        )
        capabilities = [a.capability for a in task.actions]
        self.assertEqual(capabilities, ["web.search", "applications.write_text"])
        # The write consumes the search output, and the query excludes the clause.
        write = task.actions[-1]
        self.assertEqual(write.parameters["text"], "$search_results")
        self.assertNotIn("write", task.actions[0].parameters["query"])
        self.assertTrue(plan.requires_action)

    def test_plain_search_is_not_a_write(self):
        task = _interpreter().interpret("Search YouTube for popular videos.")
        self.assertEqual([a.capability for a in task.actions], ["web.search"])


class NegativeRoutingTests(unittest.TestCase):
    """Semantic intent must dominate keyword matching (spec section 19)."""

    def test_poem_about_cars_is_not_a_video_search(self):
        task, signals, plan = _route("Create a poem in Notepad about cars.")
        capabilities = [a.capability for a in task.actions]
        self.assertEqual(capabilities, ["content.generate", "applications.write_text"])
        self.assertNotIn("web.search", capabilities)
        self.assertIn("computer", [s.value for s in plan.sources])
        self.assertEqual(task.entities.get("topic"), "cars")

    def test_plain_question_does_not_reach_for_web(self):
        _, signals, plan = _route("Explain recursion.")
        self.assertFalse(signals.time_sensitive)
        self.assertFalse(signals.explicit_web_request)
        self.assertEqual([s.value for s in plan.sources], ["knowledge", "model"])

    def test_current_question_reaches_for_web(self):
        _, signals, plan = _route("What is the latest Python version?")
        self.assertTrue(signals.time_sensitive)
        self.assertEqual(plan.sources[0].value, "web")

        def test_self_query_routes_to_self(self):
            _, signals, plan = _route("What can you do?")
            self.assertTrue(signals.is_self_query)
            self.assertEqual([s.value for s in plan.sources], ["self"])

class AmbiguousRequestTests(unittest.TestCase):
    # Requests with no resolvable target must ask, not guess (spec section 25).

    def test_find_that_file_asks_for_clarification(self):
        task, signals, plan = _route("Find that file.")
        self.assertFalse([a for a in task.actions if a.capability == "web.search"])
        self.assertTrue(signals.ambiguous_reference)
        self.assertEqual(plan.sources, ())

    def test_bare_search_verb_with_no_target_is_not_a_web_action(self):
        task = _interpreter().interpret("Search it.")
        self.assertEqual([a.capability for a in task.actions], [])

    def test_concrete_search_still_produces_a_web_action(self):
        task = _interpreter().interpret("Search for the latest AI news.")
        self.assertEqual([a.capability for a in task.actions], ["web.search"])


class ReasoningDelegationTests(unittest.TestCase):
    # Read-only research is served here; mutations delegate (spec sections 10/22).

    def test_pure_web_search_is_served_by_the_engine(self):
        task = _interpreter().interpret("Search the web for the latest Python version.")
        self.assertFalse(ReasoningEngine._must_delegate(task))

    def test_file_lookup_actions_are_served_by_the_engine(self):
        task = _interpreter().interpret("Find my resume.")
        self.assertFalse(ReasoningEngine._must_delegate(task))

    def test_file_mutation_delegates(self):
        task = _interpreter().interpret("Create a text file.")
        self.assertTrue(ReasoningEngine._must_delegate(task))

    def test_app_launch_delegates(self):
        task = _interpreter().interpret("Open Notepad.")
        self.assertTrue(ReasoningEngine._must_delegate(task))

    def test_content_creation_delegates(self):
        task = _interpreter().interpret("Create a poem in Notepad about cars.")
        self.assertTrue(ReasoningEngine._must_delegate(task))


class ExecutorWebResultRenderingTests(unittest.TestCase):
    # A search-then-write hybrid must hand usable text to the write step.

    def test_named_web_output_is_rendered_as_text(self):
        from executor import _render_web_results

        rendered = _render_web_results(
            {
                "query": "python",
                "results": [
                    {"title": "Python 3.13", "url": "https://python.org", "snippet": "Newest."}
                ],
            }
        )
        self.assertIn("Python 3.13", rendered)
        self.assertIn("https://python.org", rendered)

    def test_empty_web_results_render_honestly(self):
        from executor import _render_web_results

        rendered = _render_web_results({"query": "obscure", "results": []})
        self.assertIn("No web results", rendered)


if __name__ == "__main__":
    unittest.main()
