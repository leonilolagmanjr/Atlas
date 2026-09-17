"""Semantic task-understanding tests for the Task IR pipeline.

Covers the overhaul requirements directly:

* The full natural-language matrix from the specification.
* The critical regression: "Create a poem in Notepad" vs
  "Create a poem in Notepad about cars" must resolve to the same task family
  with the topic preserved (and must NOT leak into the application name).
* Generalization to unseen combinations (no per-phrase rules).
* Task IR construction robust against malformed model output.
* Validation against the capability registry.
* Dynamic planning with dependency-ordered variable references.
* Verification and bounded replanning.
"""

from __future__ import annotations

import unittest

from models_task import Task, TaskAction
from reasoning.task_interpreter import SemanticTaskInterpreter
from reasoning.task_planner import TaskPlanner
from reasoning.task_validator import TaskValidator
from reasoning.verifier import TaskVerifier
from tools.capabilities import CapabilityRegistry


def interpret(text: str) -> Task:
    # Deterministic interpreter: no network, no model.
    return SemanticTaskInterpreter(enabled=False).interpret(text)


def capabilities_for(task: Task) -> list[str]:
    return [action.capability for action in task.actions]


def params_for(task: Task, capability: str) -> dict:
    for action in task.actions:
        if action.capability == capability:
            return action.parameters
    raise AssertionError(f"{capability} not in task actions")


class NotepadRegressionTests(unittest.TestCase):
    """The central example: adding "about cars" must not break the command."""

    def test_plain_poem_and_topic_poem_share_task_family(self):
        plain = interpret("Create a poem in Notepad.")
        topic = interpret("Create a poem in Notepad about cars.")

        for task in (plain, topic):
            self.assertEqual(task.task_type, "content_creation")
            self.assertEqual(capabilities_for(task), ["content.generate", "applications.write_text"])

    def test_about_cars_becomes_the_topic_not_the_application(self):
        task = interpret("Create a poem in Notepad about cars.")

        # The destination is exactly Notepad; "about cars" becomes the topic.
        self.assertEqual(task.entities["application"], "Notepad")
        self.assertEqual(task.entities["topic"], "cars")
        self.assertEqual(params_for(task, "content.generate")["topic"], "cars")
        self.assertEqual(params_for(task, "applications.write_text")["application"], "Notepad")

    def test_application_name_never_absorbs_the_modifier(self):
        # Regression: the application name must never include "about ...".
        for text in [
            "Create a poem in Notepad about cars.",
            "Write a poem about cars in Notepad.",
            "Open Notepad and write a poem about motorcycles.",
        ]:
            with self.subTest(text=text):
                task = interpret(text)
                self.assertNotIn("about", task.entities.get("application", "").casefold())

    def test_write_step_consumes_generated_content_by_reference(self):
        task = interpret("Create a poem in Notepad about cars.")
        self.assertEqual(params_for(task, "applications.write_text")["text"], "$generated_text")


class NaturalLanguageVariationTests(unittest.TestCase):
    """Many phrasings resolve to essentially the same semantic task."""

    SAME_TASK_CASES = [
        "Create a poem in Notepad about cars.",
        "Write me a poem about cars in Notepad.",
        "Open Notepad and write a poem about cars.",
        "Can you put a poem about automobiles into Notepad?",
        "Make something poetic about cars and put it in Notepad.",
    ]

    def test_variations_produce_a_content_plus_write_task(self):
        for text in self.SAME_TASK_CASES:
            with self.subTest(text=text):
                task = interpret(text)
                self.assertEqual(
                    sorted(capabilities_for(task)),
                    ["applications.write_text", "content.generate"],
                )
                self.assertEqual(task.entities.get("content_type"), "poem")


class GeneralizationTests(unittest.TestCase):
    """Unseen combinations must work without a new rule."""

    def test_unseen_content_and_topic_combination(self):
        task = interpret("Write a short story about space exploration in Notepad.")
        self.assertEqual(params_for(task, "content.generate")["content_type"], "story")
        self.assertEqual(params_for(task, "content.generate")["topic"], "space exploration")
        self.assertEqual(params_for(task, "applications.write_text")["application"], "Notepad")

    def test_builtin_app_banana_must_not_be_special_cased(self):
        # A content type Atlas has never heard of is still captured as text.
        task = interpret("Compose a haiku about the ocean in Notepad.")
        self.assertEqual(params_for(task, "content.generate")["content_type"], "haiku")
        self.assertEqual(params_for(task, "content.generate")["topic"], "the ocean")

    def test_tone_and_length_modifiers_are_preserved(self):
        task = interpret("Create a poem in Notepad about motorcycles using a funny tone.")
        content = params_for(task, "content.generate")
        self.assertEqual(content["topic"], "motorcycles")
        self.assertEqual(content["tone"], "funny")


class SearchTaskTests(unittest.TestCase):
    def test_open_chrome_and_search_youtube(self):
        task = interpret("Open Chrome and search YouTube for popular car videos.")
        caps = capabilities_for(task)
        self.assertIn("web.search", caps)
        self.assertIn("applications.launch_named", caps)
        # The browser is opened before the search runs.
        self.assertEqual(caps.index("applications.launch_named"), 0)
        search_params = params_for(task, "web.search")
        self.assertEqual(search_params["site"], "youtube")
        self.assertEqual(search_params["sort"], "popular")

    def test_search_the_web_for_latest_python_release(self):
        task = interpret("Search the web for the latest Python release.")
        self.assertIn("web.search", capabilities_for(task))
        query = params_for(task, "web.search")["query"]
        self.assertIn("python", query.casefold())


class WebArtifactHybridTests(unittest.TestCase):
    """A web artifact request must plan a research step for the artifact itself."""

    def test_bare_script_phrase_plans_artifact_research(self):
        task = interpret("search web for bee movie script and copy to notepad")
        self.assertEqual(capabilities_for(task), ["web.research", "applications.write_text"])
        research = params_for(task, "web.research")
        self.assertTrue(research["must_be_artifact"])
        self.assertEqual(research["goal"], "retrieve_document")
        # The plan carries the interpreter's content noun; the retrieval layer
        # normalizes it to the canonical type (web_task -> "movie_script").
        self.assertEqual(research["content_type"], "script")
        self.assertEqual(params_for(task, "applications.write_text")["text"], "$web_content")

    def test_question_about_the_script_is_not_an_artifact_research(self):
        task = interpret("how long is the bee movie script")
        self.assertNotIn("web.research", capabilities_for(task))


class ReconcileBackfillTests(unittest.TestCase):
    """The deterministic pass backfills the LLM's under-specified actions."""

    def test_web_research_action_inherits_artifact_parameters(self):
        interpreter = SemanticTaskInterpreter(enabled=False)
        heuristic = interpreter.interpret("search the web for bee movie script and copy to notepad")
        llm = Task(
            task_type="computer_action",
            goal="perform_action",
            original_prompt="search the web for bee movie script and copy to notepad",
            actions=[
                TaskAction(
                    action_id="a1", capability="web.research",
                    parameters={"query": "bee movie script"},
                ),
                TaskAction(
                    action_id="a2", capability="applications.write_text",
                    parameters={"application": "notepad", "text": "$web_content"},
                    depends_on=["a1"],
                ),
            ],
        )
        merged = interpreter._reconcile(llm, heuristic)
        research = params_for(merged, "web.research")
        self.assertTrue(research["must_be_artifact"])
        self.assertEqual(research["goal"], "retrieve_document")
        # Backfilled from the deterministic pass, which uses the interpreter's
        # content noun ("script"); web_task normalizes it to "movie_script".
        self.assertEqual(research["content_type"], "script")
        # The model's write step is preserved and still consumable.
        self.assertEqual(params_for(merged, "applications.write_text")["text"], "$web_content")


class FileTaskTests(unittest.TestCase):
    def test_find_largest_pdf_and_move_to_documents(self):
        task = interpret("Find the largest PDF in Downloads and move it to my Documents folder.")
        caps = capabilities_for(task)
        self.assertIn("filesystem.search", caps)
        self.assertIn("filesystem.move", caps)
        search_params = params_for(task, "filesystem.search")
        self.assertEqual(search_params["pattern"], "*.pdf")
        self.assertEqual(search_params["select"], "largest")
        move_params = params_for(task, "filesystem.move")
        # Source is a reference to the search output, not a literal guess.
        self.assertEqual(move_params["source"], "$largest_match")
        self.assertEqual(move_params["destination"], "Documents")

    def test_create_folder_on_desktop(self):
        task = interpret("Create a folder called Projects on my desktop.")
        self.assertEqual(capabilities_for(task), ["filesystem.create_folder"])
        path = params_for(task, "filesystem.create_folder")["path"]
        self.assertIn("Desktop", path)
        self.assertIn("Projects", path)


class InformationalTaskTests(unittest.TestCase):
    """Knowledge workflows must not be broken by the task pipeline."""

    CASES = [
        "What is quantum computing?",
        "Explain quantum computing in simple terms.",
        "Compare Windows and Linux.",
        "Summarize this document.",
    ]

    def test_informational_requests_need_no_execution(self):
        for text in self.CASES:
            with self.subTest(text=text):
                task = interpret(text)
                self.assertEqual(task.task_type, "informational")
                self.assertFalse(task.execution_required)
                self.assertEqual(task.actions, [])


class AmbiguityTests(unittest.TestCase):
    def test_unresolved_pronoun_target_requests_clarification(self):
        task = interpret("Open it and write something about cars.")
        self.assertTrue(task.needs_clarification)
        self.assertIsNotNone(task.clarification_question)

    def test_clear_request_does_not_need_clarification(self):
        self.assertFalse(interpret("Create a poem in Notepad about cars.").needs_clarification)

    def test_move_it_after_a_search_is_not_ambiguous(self):
        task = interpret("Find the largest PDF in Downloads and move it to Documents.")
        self.assertFalse(task.needs_clarification)


class FollowUpContextTests(unittest.TestCase):
    def test_follow_up_inherits_prior_destination(self):
        interpreter = SemanticTaskInterpreter(enabled=False)
        first = interpreter.interpret("Open Notepad.")
        follow_up = interpreter.interpret("Write a poem about cars.", context=first)

        self.assertIn("applications.write_text", capabilities_for(follow_up))
        self.assertEqual(params_for(follow_up, "applications.write_text")["application"], "Notepad")


class TaskConstructionRobustnessTests(unittest.TestCase):
    """Malformed model output must never produce an invalid Task."""

    def test_from_mapping_ignores_unknown_keys_and_bad_types(self):
        task = Task.from_mapping(
            {
                "task_type": "not_a_real_type",
                "goal": 123,
                "actions": "not-a-list",
                "confidence": 9,
                "entities": "nope",
            },
            prompt="x",
        )
        self.assertEqual(task.task_type, "unknown")
        self.assertEqual(task.actions, [])
        self.assertEqual(task.confidence, 1.0)
        self.assertEqual(task.entities, {})

    def test_action_without_capability_is_dropped(self):
        task = Task.from_mapping(
            {"actions": [{"action_id": "a", "parameters": {}}, {"capability": "web.search", "parameters": {"query": "x"}}]},
            prompt="x",
        )
        self.assertEqual(len(task.actions), 1)
        self.assertEqual(task.actions[0].capability, "web.search")

    def test_arguments_key_is_accepted_as_parameters(self):
        task = Task.from_mapping(
            {"actions": [{"capability": "web.search", "arguments": {"query": "cars"}}]},
            prompt="x",
        )
        self.assertEqual(task.actions[0].parameters["query"], "cars")

    def test_references_are_discovered(self):
        task = interpret("Create a poem in Notepad about cars.")
        self.assertIn("generated_text", task.references())


class TaskValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = TaskValidator(CapabilityRegistry(None))

    def test_valid_multistep_task_passes(self):
        task = interpret("Create a poem in Notepad about cars.")
        result = self.validator.validate(task)
        self.assertTrue(result.valid, result.errors)
        self.assertTrue(result.requires_confirmation)

    def test_unknown_capability_is_rejected(self):
        task = Task(
            task_type="computer_action",
            execution_required=True,
            actions=[TaskAction(action_id="a", capability="shell.exec", parameters={})],
        )
        result = self.validator.validate(task)
        self.assertFalse(result.valid)
        self.assertTrue(any("unknown capability" in e for e in result.errors))

    def test_missing_required_parameter_is_rejected(self):
        task = Task(
            task_type="computer_action",
            execution_required=True,
            actions=[TaskAction(action_id="a", capability="web.search", parameters={})],
        )
        result = self.validator.validate(task)
        self.assertFalse(result.valid)
        self.assertTrue(any("required" in e for e in result.errors))

    def test_wrong_parameter_type_is_rejected(self):
        task = Task(
            task_type="computer_action",
            execution_required=True,
            actions=[
                TaskAction(
                    action_id="a",
                    capability="web.search",
                    parameters={"query": "x", "max_results": {"bad": 1}},
                )
            ],
        )
        self.assertFalse(self.validator.validate(task).valid)

    def test_dangling_reference_is_rejected(self):
        task = Task(
            task_type="computer_action",
            execution_required=True,
            actions=[
                TaskAction(
                    action_id="w",
                    capability="applications.write_text",
                    parameters={"application": "Notepad", "text": "$nobody_produces_this"},
                )
            ],
        )
        result = self.validator.validate(task)
        self.assertFalse(result.valid)
        self.assertTrue(any("references" in e for e in result.errors))


class TaskPlanningTests(unittest.TestCase):
    def test_plan_orders_dependencies_and_keeps_references(self):
        task = interpret("Create a poem in Notepad about cars.")
        decision = TaskPlanner(capabilities=CapabilityRegistry(None)).create_plan(
            task, user_question="x"
        )
        tools = [step.metadata["tool"] for step in decision.plan.steps]
        self.assertEqual(tools, ["content.generate", "applications.write_text"])
        self.assertEqual(decision.plan.steps[1].metadata["parameters"]["text"], "$generated_text")
        self.assertEqual(decision.plan.steps[0].metadata["produces"], "generated_text")

    def test_ordering_is_respected_when_actions_are_out_of_order(self):
        task = Task(
            task_type="computer_action",
            actions=[
                TaskAction(
                    action_id="write",
                    capability="applications.write_text",
                    parameters={"application": "Notepad", "text": "$generated_text"},
                    depends_on=["gen"],
                ),
                TaskAction(
                    action_id="gen",
                    capability="content.generate",
                    parameters={"content_type": "poem"},
                    produces="generated_text",
                ),
            ],
        )
        decision = TaskPlanner(capabilities=CapabilityRegistry(None)).create_plan(task, user_question="x")
        tools = [step.metadata["tool"] for step in decision.plan.steps]
        self.assertEqual(tools[0], "content.generate")
        self.assertEqual(tools[1], "applications.write_text")


class VerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.verifier = TaskVerifier()

    def test_launch_is_verified_by_pid(self):
        outcome = self.verifier.verify("applications.launch_named", {"pid": 10}, True)
        self.assertTrue(outcome.verified)

    def test_write_text_is_verified_by_character_count(self):
        outcome = self.verifier.verify("applications.write_text", {"characters": 5}, True)
        self.assertTrue(outcome.verified)

    def test_empty_search_is_unverified(self):
        outcome = self.verifier.verify("web.search", {"results": []}, True)
        self.assertFalse(outcome.verified)
        self.assertEqual(outcome.status, "unverified")

    def test_unknown_capability_is_honestly_unverified(self):
        outcome = self.verifier.verify("something.else", {"ok": True}, True)
        self.assertFalse(outcome.verified)

    def test_failure_is_reported_as_failed(self):
        outcome = self.verifier.verify("web.search", None, False)
        self.assertEqual(outcome.status, "failed")


if __name__ == "__main__":
    unittest.main()
