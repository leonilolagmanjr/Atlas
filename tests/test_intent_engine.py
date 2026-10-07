"""Intent Engine 2.0 — context-aware goal and intent understanding tests.

These lock in the behaviour of the understanding layer that sits between the
semantic interpreter and the deterministic planner/executor:

    USER LANGUAGE -> INTENT -> DESIRED OUTCOME -> STRUCTURED TASK -> PLAN

The suite is deliberately offline and deterministic: the interpreter runs with
the LLM disabled (so the deterministic reading is exercised), capabilities come
from a real registry, and no tool ever executes. This is the regression fence
for the classes of request Atlas previously misread — answer-vs-research,
topic-vs-destination, creation-vs-retrieval, transformation-vs-new-task, and
follow-up reference resolution.

Sections map to the Intent Engine 2.0 specification:

* 25 — the required positive/normalisation cases
* 28 — the real-world cases Atlas previously struggled with
* also covers negative behaviour (no spurious web, no spurious app, no lost
  context, malformed output never reaching the executor).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from computer.runtime import register_read_only_tools  # noqa: E402
from config import COMPUTER_ROOT  # noqa: E402
from models_task import Task  # noqa: E402
from reasoning.correction_memory import CorrectionMemory  # noqa: E402
from reasoning.intent_engine import IntentEngine  # noqa: E402
from reasoning.reference_resolver import ReferenceResolver  # noqa: E402
from reasoning.task_interpreter import SemanticTaskInterpreter  # noqa: E402
from reasoning.task_planner import TaskPlanner  # noqa: E402
from reasoning.task_validator import TaskValidator  # noqa: E402
from tools.capabilities import CapabilityRegistry  # noqa: E402
from tools.registry import ToolRegistry  # noqa: E402


def _capabilities() -> CapabilityRegistry:
    registry = ToolRegistry()
    register_read_only_tools(registry, root=COMPUTER_ROOT, ask=None)
    return CapabilityRegistry(registry)


def _interpreter() -> SemanticTaskInterpreter:
    caps = _capabilities()
    return SemanticTaskInterpreter(
        ask=None, capabilities=caps, capability_catalog=caps.render_catalog()
    )


def _understand(
    text: str,
    *,
    prior_task: Task | None = None,
    history: str = "",
) -> Task:
    """Interpret and understand a request exactly as Brain does, minus execution."""

    task = _interpreter().interpret(text, context=prior_task, history=history)
    engine = IntentEngine(capabilities=_capabilities())
    return engine.understand(task, prior_task=prior_task, history=history)


class DirectAnswerTests(unittest.TestCase):
    """Information want, not an action: the model answers directly."""

    def test_what_is_python_is_an_answer(self):
        task = _understand("What is Python?")
        self.assertEqual(task.goal, "answer")
        self.assertFalse(task.needs_web)
        self.assertFalse(task.needs_application)
        self.assertFalse(task.needs_files)
        self.assertEqual(task.actions, [])
        self.assertFalse(task.execution_required)


class ResearchTests(unittest.TestCase):
    """Current-information requests need web evidence, not a model answer."""

    def test_latest_rtx_gpus_needs_web(self):
        task = _understand("Find the latest information about RTX GPUs.")
        self.assertEqual(task.goal, "research")
        self.assertTrue(task.needs_web)
        self.assertFalse(task.needs_application)
        self.assertIn("web.search", [a.capability for a in task.actions])


class CreationTests(unittest.TestCase):
    """Creative content is generated, never researched."""

    def test_poem_about_cars_generates_without_web(self):
        task = _understand("Write a poem about cars.")
        self.assertEqual(task.goal, "create")
        self.assertEqual(task.object_type, "content")
        self.assertFalse(task.needs_web)
        self.assertFalse(task.needs_application)

    def test_poem_about_cars_in_notepad_is_delivery(self):
        task = _understand("Write a poem about cars in Notepad.")
        self.assertEqual(task.goal, "create_and_deliver")
        self.assertFalse(task.needs_web, "a poem must never trigger a web search")
        self.assertTrue(task.needs_application)
        caps = [a.capability for a in task.actions]
        self.assertIn("content.generate", caps)
        self.assertIn("applications.write_text", caps)
        # Notepad is the destination, never the research topic.
        self.assertNotIn("web.search", caps)
        self.assertNotIn("web.research", caps)


class ResearchAndTransformationTests(unittest.TestCase):
    """Retrieve -> transform -> (optionally) deliver, without losing a stage."""

    def test_search_and_summarize_no_destination(self):
        task = _understand("Search how to write a resume and summarize the important points.")
        self.assertEqual(task.goal, "research")
        self.assertTrue(task.needs_web)
        self.assertIn("summarize", task.transformations or [task.goal])
        caps = [a.capability for a in task.actions]
        self.assertTrue(any(c.startswith("web.") for c in caps))
        self.assertIn("content.generate", caps)

    def test_search_summarize_and_notepad(self):
        task = _understand(
            "Search how to write a resume and put the important points into Notepad."
        )
        self.assertEqual(task.goal, "research_and_deliver")
        self.assertTrue(task.needs_web)
        self.assertTrue(task.needs_application)
        caps = [a.capability for a in task.actions]
        self.assertTrue(any(c.startswith("web.") for c in caps))
        self.assertIn("applications.write_text", caps)


class FollowUpAndReferenceTests(unittest.TestCase):
    """Follow-ups must resolve against the previous task, not start fresh."""

    def setUp(self):
        self.first = _understand("Search for the latest NVIDIA GPUs.")

    def test_make_it_shorter_transforms_previous_output(self):
        task = _understand(
            "Make it shorter.",
            prior_task=self.first,
            history="User: Search for the latest NVIDIA GPUs.\nAssistant: GPU list...",
        )
        self.assertIn(task.goal, {"transform", "summarize", "modify"})
        self.assertIn("shorten", task.transformations)
        self.assertFalse(task.needs_web, "shortening a prior answer is not new research")
        self.assertIn("previous_output", task.context_references)

    def test_put_that_in_notepad_delivers_previous_output(self):
        task = _understand(
            "Put that in Notepad.",
            prior_task=self.first,
            history="Assistant: GPU list...",
        )
        self.assertTrue(task.needs_application)
        self.assertIn("previous_output", task.context_references)
        self.assertIn(
            "applications.write_text", [a.capability for a in task.actions]
        )

    def test_do_the_same_for_interstellar_inherits_and_replaces_subject(self):
        prior = _understand("Summarize the Avatar movie in Notepad.")
        task = _understand(
            "Do the same thing for Interstellar.",
            prior_task=prior,
            history="Assistant: Avatar summary...",
        )
        # The reading keeps a reference to the previous task structure.
        self.assertTrue(task.context_references)
        self.assertTrue(
            any("previous_task" in ref for ref in task.context_references),
            task.context_references,
        )
        # And it inherits the delivery goal rather than becoming a plain answer.
        self.assertEqual(task.goal, prior.goal)


class ApplicationActionTests(unittest.TestCase):
    """A named application with an open verb is an action, not research."""

    def test_open_notepad_launches_the_app(self):
        task = _understand("Open Notepad.")
        self.assertIn(task.goal, {"execute", "organize"})
        self.assertTrue(task.needs_application)
        self.assertFalse(task.needs_web)
        caps = [a.capability for a in task.actions]
        self.assertTrue(any(c.startswith("applications.") for c in caps))


class FileTaskTests(unittest.TestCase):
    """A personal document is a local file subject, never a web query."""

    def test_summarize_my_resume_is_local(self):
        task = _understand("Summarize my resume.")
        self.assertFalse(task.needs_web, "my resume is local, not web research")


class AmbiguityTests(unittest.TestCase):
    """Missing topic/source/destination must surface, not be guessed."""

    def test_get_me_some_videos_is_ambiguous(self):
        task = _understand("Get me some videos.")
        self.assertLess(task.confidence, 0.75)

    def test_get_me_mrbeast_videos_is_research(self):
        # A named entity + video object resolves the ambiguity to a video search.
        task = _understand("Find MrBeast videos.")
        self.assertIn(task.goal, {"research", "find"})
        self.assertEqual(task.object_type, "video")
        self.assertTrue(task.needs_web)

    def test_which_one_without_context_is_unresolved(self):
        task = _understand("Which one should I use?")
        self.assertTrue(
            task.ambiguities or task.context_references,
            "a selection reference with no context must be surfaced",
        )


class MultiStepTests(unittest.TestCase):
    """Three-part request keeps every operation and the destination."""

    def test_find_three_compare_and_put_in_notepad(self):
        task = _understand(
            "Find three good resume formats, compare them, and put the differences in Notepad."
        )
        self.assertTrue(task.needs_web)
        self.assertTrue(task.needs_application)
        caps = [a.capability for a in task.actions]
        self.assertTrue(any(c.startswith("web.") for c in caps))
        self.assertIn("applications.write_text", caps)


class NegativeBehaviourTests(unittest.TestCase):
    """The mistakes a small local model makes must stay repaired."""

    def test_creative_task_does_not_search_web(self):
        task = _understand("Write a poem about cars in Notepad.")
        self.assertFalse(task.needs_web)
        self.assertNotIn("web.search", task.required_capabilities)

    def test_notepad_is_a_destination_not_a_research_topic(self):
        task = _understand("Write a poem about cars in Notepad.")
        self.assertNotEqual(str(task.entities.get("topic", "")).casefold(), "notepad")

    def test_information_request_does_not_open_notepad(self):
        task = _understand("What is Python?")
        self.assertFalse(task.needs_application)
        self.assertEqual(task.actions, [])

    def test_transformation_of_previous_is_not_new_research(self):
        first = _understand("Search for the latest NVIDIA GPUs.")
        task = _understand(
            "Make it shorter.", prior_task=first, history="Assistant: GPU list..."
        )
        self.assertFalse(task.needs_web)
        self.assertFalse(task.needs_files)

    def test_required_capabilities_all_exist_in_registry(self):
        """Atlas — not the model — confirms tools exist; nothing invented."""
        caps = _capabilities()
        engine = IntentEngine(capabilities=caps)
        for prompt in [
            "Write a poem about cars in Notepad.",
            "Search how to write a resume and put the important points into Notepad.",
            "Open Notepad.",
            "Find MrBeast videos.",
        ]:
            task = _interpreter().interpret(prompt)
            engine.understand(task)
            report = engine.validate_capabilities(task)
            self.assertEqual(
                report["missing"],
                [],
                f"invented capability for {prompt!r}: {report['missing']}",
            )

    def test_validated_task_plans_without_error(self):
        """Every action a delivery task proposes must validate and plan cleanly."""
        caps = _capabilities()
        validator = TaskValidator(caps)
        planner = TaskPlanner(capabilities=caps)
        task = _understand("Write a poem about cars in Notepad.")
        result = validator.validate(task)
        self.assertTrue(result.valid, result.errors)
        decision = planner.create_plan(result.actions and task or task, user_question=task.original_prompt)
        self.assertTrue(decision.plan.steps)

    def test_malformed_model_output_never_reaches_executor(self):
        """Task.from_mapping must sanitise untrusted output instead of raising."""
        malformed = {
            "goal": "create",
            "actions": [
                {"capability": "", "parameters": "not-a-dict"},  # dropped: no capability
                {"capability": "totally.made.up", "parameters": {}},  # kept, fails validation
                "not-a-dict",
            ],
            "operations": ["nonsense", "generate"],
            "object_type": 12345,
            "confidence": "not-a-number",
        }
        task = Task.from_mapping(malformed, prompt="do something")
        # Empty/malformed entries are dropped; the invented capability survives
        # as data but is rejected by the registry-backed validator.
        self.assertEqual([a.capability for a in task.actions], ["totally.made.up"])
        self.assertEqual(task.operations, ["generate"])
        self.assertEqual(task.object_type, "unknown")
        self.assertEqual(task.confidence, 0.0)
        result = TaskValidator(_capabilities()).validate(task)
        self.assertFalse(result.valid)


class ReferenceResolverUnitTests(unittest.TestCase):
    """The resolver is deterministic data; test it directly too."""

    def setUp(self):
        self.resolver = ReferenceResolver()

    def test_pronoun_resolves_to_previous_output(self):
        ctx = self.resolver.resolve("Make it shorter.", prior_task=Task(goal="research"))
        self.assertTrue(ctx.mutates_output)
        self.assertEqual(ctx.transformation, "shorten")

    def test_same_task_pattern_sets_replace_subject(self):
        ctx = self.resolver.resolve(
            "Do the same thing for Interstellar.", prior_task=Task(goal="create_and_deliver")
        )
        self.assertTrue(ctx.inherit)
        self.assertTrue(ctx.replace_subject)
        self.assertEqual(ctx.new_subject, "interstellar")

    def test_unresolvable_reference_is_surfaced(self):
        ctx = self.resolver.resolve("Put that in Notepad.")  # no prior task/history
        self.assertTrue(ctx.has_reference)
        self.assertTrue(ctx.unresolved)


class CorrectionMemoryTests(unittest.TestCase):
    """Structured lessons override the reading for a matching class of phrasing."""

    def test_high_confidence_lesson_overrides_reading(self):
        memory = CorrectionMemory()
        memory._lessons = []  # deterministic: no on-disk lessons

        def _record(lesson):  # noqa: ANN001
            memory._lessons.append(lesson)
            return True

        from reasoning.correction_memory import Lesson

        _record(
            Lesson(
                type="task_pattern",
                pattern=r"summarize .* in notepad",
                correct_behavior=["research", "summarize", "write to notepad"],
                fields={"needs_web": True, "needs_application": True},
                confidence=0.95,
            )
        )
        engine = IntentEngine(capabilities=_capabilities(), correction_memory=memory)
        task = _interpreter().interpret("Summarize the Avatar movie in Notepad.")
        engine.understand(task)
        self.assertTrue(task.needs_web)
        self.assertTrue(task.needs_application)

    def test_low_confidence_lesson_is_ignored(self):
        memory = CorrectionMemory()
        from reasoning.correction_memory import Lesson

        memory._lessons = [
            Lesson(
                type="task_pattern",
                pattern="anything",
                fields={"goal": "execute"},
                confidence=0.5,  # below _MIN_CONFIDENCE
            )
        ]
        engine = IntentEngine(capabilities=_capabilities(), correction_memory=memory)
        task = _interpreter().interpret("What is Python?")
        engine.understand(task)
        self.assertEqual(task.goal, "answer")


if __name__ == "__main__":
    unittest.main()
