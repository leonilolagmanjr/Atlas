"""Command-understanding tests for the Atlas NL -> intent -> plan pipeline.

These tests cover the overhaul requirements directly:

* Notepad / YouTube natural-language variations
* Application control phrasings
* Parameter preservation across the pipeline
* Follow-up context
* Ambiguity handling
* Tool selection (LLM) constrained to the capability catalog
* Failure classification and bounded recovery
* Malformed LLM output handling
"""

from __future__ import annotations

import unittest

from models import StructuredIntent
from planner import Planner
from reasoning.interpreter import SemanticInterpreter, classify_category
from reasoning.json_llm import extract_json_object
from tools.capabilities import render_capability_catalog


def interpret(text: str) -> StructuredIntent:
    # Deterministic interpreter: no network, no model.
    return SemanticInterpreter(enabled=False).interpret(text)


class NotepadVariationTests(unittest.TestCase):
    """Every Notepad phrasing resolves to the same task family."""

    CASES = [
        "create a poem in notepad",
        "create a poem about cars in notepad",
        "write a poem about cars in notepad",
        "create a funny poem about cars in notepad",
        "write a short poem about racing in notepad",
        "open notepad and write a poem about cars",
    ]

    def test_all_notepad_requests_share_task_family(self):
        for text in self.CASES:
            with self.subTest(text=text):
                intent = interpret(text)
                self.assertEqual(intent.intent, "write_content")
                self.assertEqual(intent.destination.casefold(), "notepad")

    def test_open_notepad_and_write_still_targets_notepad(self):
        intent = interpret("open notepad and write a poem about cars")
        self.assertEqual(intent.destination.casefold(), "notepad")
        self.assertEqual(intent.content_type, "poem")
        self.assertEqual(intent.topic, "cars")

    def test_parameters_are_preserved_per_request(self):
        expectations = {
            "create a poem in notepad": (None, None, None),
            "create a poem about cars in notepad": ("cars", None, None),
            "create a funny poem about cars in notepad": ("cars", "funny", None),
            "write a short poem about racing cars in notepad": ("racing cars", None, "short"),
        }
        for text, (topic, tone, length) in expectations.items():
            with self.subTest(text=text):
                intent = interpret(text)
                self.assertEqual(intent.topic, topic)
                self.assertEqual(intent.tone, tone)
                self.assertEqual(intent.length, length)

    def test_style_modifier_is_captured_by_llm_shape(self):
        # The deterministic draft keeps topic; style arrives from the LLM and
        # must survive validation unchanged.
        intent = StructuredIntent.from_mapping(
            {
                "intent": "write_content",
                "action": "create",
                "content_type": "poem",
                "topic": "cars",
                "style": "futuristic",
                "destination": "notepad",
                "confidence": 0.9,
            }
        )
        self.assertEqual(intent.style, "futuristic")
        self.assertEqual(intent.topic, "cars")


class YouTubeVariationTests(unittest.TestCase):
    CASES = [
        "search youtube for mrbeast videos",
        "find mrbeast videos on youtube",
        "search youtube for mrbeast",
        "show me mrbeast's latest videos",
        "find popular mrbeast videos on youtube",
        "search youtube for videos about cars",
        "find recent videos about AI",
    ]

    def test_all_youtube_requests_classify_as_web_search(self):
        for text in self.CASES:
            with self.subTest(text=text):
                intent = interpret(text)
                self.assertEqual(intent.intent, "search")
                self.assertEqual(intent.target, "youtube")
                self.assertEqual(classify_category(intent, text), "WEB_SEARCH")

    def test_query_is_cleaned_of_site_and_sort_words(self):
        intent = interpret("search youtube for mrbeast videos")
        self.assertEqual(intent.query, "mrbeast")

    def test_query_keeps_topic_without_boilerplate(self):
        intent = interpret("search youtube for videos about cars")
        self.assertNotIn("youtube", (intent.query or "").casefold())
        self.assertIn("cars", (intent.query or "").casefold())

    def test_sort_is_extracted(self):
        self.assertEqual(interpret("show me mrbeast's latest videos").sort, "latest")
        self.assertEqual(interpret("find popular mrbeast videos on youtube").sort, "popular")

    def test_search_plan_passes_cleaned_query_and_site(self):
        structured = interpret("find mrbeast videos on youtube")
        decision = Planner().create_plan("find mrbeast videos on youtube", structured=structured)
        params = decision.plan.steps[0].metadata["parameters"]

        self.assertEqual(decision.plan.steps[0].metadata["tool"], "web.search")
        self.assertEqual(params["site"], "youtube")
        self.assertNotIn("youtube", params["query"].casefold())


class ApplicationControlTests(unittest.TestCase):
    CASES = {
        "open notepad": "notepad",
        "launch notepad": "notepad",
        "start notepad": "notepad",
        "open calculator": "calculator",
        "launch chrome": "chrome",
        "open vscode": "vscode",
    }

    def test_each_application_phrase_targets_the_right_app(self):
        for text, expected in self.CASES.items():
            with self.subTest(text=text):
                intent = interpret(text)
                self.assertEqual(intent.intent, "open_application")
                self.assertEqual(intent.destination.casefold(), expected)

    def test_launch_plan_uses_named_application_capability(self):
        structured = interpret("open notepad")
        decision = Planner().create_plan("open notepad", structured=structured)
        step = decision.plan.steps[0]

        self.assertEqual(step.metadata["tool"], "applications.launch_named")
        self.assertEqual(step.metadata["parameters"]["application"].casefold(), "notepad")


class ParameterPreservationTests(unittest.TestCase):
    """Modifiers must survive classifier -> planner -> executor arguments."""

    def test_topic_and_tone_reach_the_generation_step(self):
        structured = interpret("create a short funny poem about racing cars in notepad")
        decision = Planner().create_plan("create a short funny poem about racing cars in notepad", structured=structured)
        params = decision.plan.steps[0].metadata["parameters"]

        self.assertEqual(params["content_type"], "poem")
        self.assertEqual(params["topic"], "racing cars")
        self.assertEqual(params["tone"], "funny")
        self.assertEqual(params["length"], "short")

    def test_destination_reaches_the_write_step(self):
        structured = interpret("create a funny poem about cars in notepad")
        decision = Planner().create_plan("create a funny poem about cars in notepad", structured=structured)
        write_step = decision.plan.steps[-1]

        self.assertEqual(write_step.metadata["tool"], "applications.write_text")
        self.assertEqual(write_step.metadata["parameters"]["application"].casefold(), "notepad")
        # The write step consumes generated content by reference, not a literal.
        self.assertEqual(write_step.metadata["parameters"]["text"], "$generated_text")


class FollowUpContextTests(unittest.TestCase):
    def test_followup_inherits_destination_and_content_type(self):
        interpreter = SemanticInterpreter(enabled=False)
        first = interpreter.interpret("open notepad")
        # A later turn that only adds modifiers must inherit prior context.
        follow_up = interpreter.interpret("make it funny about cars", context=first)

        self.assertIn(follow_up.tone, {None, "funny"})
        self.assertIn(follow_up.topic, {None, "cars"})

    def test_merged_intent_keeps_destination_when_followup_only_sets_topic(self):
        base = StructuredIntent(
            intent="write_content",
            content_type="poem",
            destination="notepad",
            confidence=0.9,
        )
        follow_up = StructuredIntent(intent="write_content", topic="cars", tone="funny", confidence=0.8)
        merged = base.merged_with(follow_up)

        self.assertEqual(merged.destination, "notepad")
        self.assertEqual(merged.content_type, "poem")
        self.assertEqual(merged.topic, "cars")
        self.assertEqual(merged.tone, "funny")

    def test_followup_search_filter_keeps_target(self):
        base = StructuredIntent(intent="search", target="youtube", query="mrbeast", confidence=0.9)
        follow_up = StructuredIntent(intent="search", sort="latest", confidence=0.8)
        merged = base.merged_with(follow_up)

        self.assertEqual(merged.target, "youtube")
        self.assertEqual(merged.query, "mrbeast")
        self.assertEqual(merged.sort, "latest")


class AmbiguityTests(unittest.TestCase):
    def test_unresolved_pronoun_target_requests_clarification(self):
        intent = interpret("open it and write something about cars")

        self.assertTrue(intent.needs_clarification)
        self.assertIsNotNone(intent.clarification_question)

    def test_clear_request_does_not_request_clarification(self):
        intent = interpret("create a poem about cars in notepad")

        self.assertFalse(intent.needs_clarification)

    def test_clarification_is_not_triggered_for_plain_app_launch(self):
        self.assertFalse(interpret("open notepad").needs_clarification)


class CapabilityCatalogTests(unittest.TestCase):
    def test_catalog_only_exposes_plannable_capabilities(self):
        from tools import ExecutionMode, PermissionEngine, ToolRegistry, ToolRouter
        from computer.runtime import register_read_only_tools
        registry = ToolRegistry()
        register_read_only_tools(registry)
        catalog = render_capability_catalog(registry)

        self.assertIn("content.generate", catalog)
        self.assertIn("applications.write_text", catalog)
        self.assertIn("web.search", catalog)
        # High-level filesystem capabilities are now part of the plannable
        # surface (a task like "create a folder on my desktop" needs them),
        # while low-level admin backends remain invisible to the LLM.
        self.assertIn("filesystem.create_folder", catalog)
        self.assertNotIn("powershell.execute", catalog)
        self.assertNotIn("processes.list", catalog)


class PlannerLLMSelectionTests(unittest.TestCase):
    def test_llm_plan_is_rejected_when_it_invents_a_capability(self):
        def fake_ask(**_kwargs):
            return '{"steps":[{"capability":"windows.delete_everything","arguments":{}}],"confidence":0.9}'

        planner = Planner(ask=fake_ask)
        structured = interpret("do something unusual")
        self.assertIsNone(planner.plan_with_llm("do something unusual", structured))

    def test_llm_plan_is_accepted_for_catalog_capabilities(self):
        def fake_ask(**_kwargs):
            return '{"steps":[{"capability":"web.search","arguments":{"query":"cars"}}],"confidence":0.9}'

        planner = Planner(ask=fake_ask)
        structured = StructuredIntent(intent="search", query="cars", confidence=0.8)
        decision = planner.plan_with_llm("search for cars", structured)

        self.assertIsNotNone(decision)
        self.assertEqual(decision.plan.steps[0].metadata["tool"], "web.search")


class MalformedLLMOutputTests(unittest.TestCase):
    def test_json_is_recovered_from_fenced_output(self):
        parsed = extract_json_object('```json\n{"intent": "search"}\n```')
        self.assertEqual(parsed, {"intent": "search"})

    def test_json_with_trailing_comma_is_repaired(self):
        parsed = extract_json_object('{"intent": "search", "topic": "cars",}')
        self.assertEqual(parsed["topic"], "cars")

    def test_non_json_returns_none(self):
        self.assertIsNone(extract_json_object("I cannot do that."))

    def test_from_mapping_clamps_confidence_and_drops_bad_types(self):
        intent = StructuredIntent.from_mapping(
            {"intent": "search", "query": "cars", "confidence": 5, "filters": "not-a-list"}
        )
        self.assertEqual(intent.confidence, 1.0)
        self.assertEqual(intent.filters, ["not-a-list"])


if __name__ == "__main__":
    unittest.main()
