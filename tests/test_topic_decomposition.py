"""Regression tests for semantic topic decomposition and query normalization.

These tests cover the failure this work fixes directly:

    "research about sykrim and write it in notepad"

Atlas must understand the subject ("Skyrim"/"sykrim") and must NOT send the
literal instruction phrase ("about sykrim") as the research query.

They test the *intermediate representation* — the topic, the constructed query,
the transformation, and the destination — not only whether the final application
action occurs, because the failure originated in interpretation, not execution.
"""

from __future__ import annotations

import unittest

from models_task import Task
from reasoning.task_interpreter import SemanticTaskInterpreter
from reasoning.topic_extraction import extract_topic, normalize_query


def interpret(text: str) -> Task:
    # Deterministic interpreter: no network, no model.
    return SemanticTaskInterpreter(enabled=False).interpret(text)


def web_query(task: Task) -> str:
    for action in task.actions:
        if action.capability in ("web.research", "web.search"):
            return str(action.parameters.get("query") or "")
    return ""


def web_target(task: Task) -> str:
    for action in task.actions:
        if action.capability == "web.research":
            return str(action.parameters.get("target") or "")
    return ""


def capability_list(task: Task) -> list[str]:
    return [action.capability for action in task.actions]


class TopicExtractionUnitTests(unittest.TestCase):
    """The semantic extractor itself, independent of the interpreter."""

    def test_connector_is_stripped_from_the_subject(self):
        for text in ("about Skyrim", "on Skyrim", "regarding Skyrim", "concerning Skyrim"):
            with self.subTest(text=text):
                reading = extract_topic(text)
                self.assertEqual(reading.subject.casefold(), "skyrim")
                self.assertEqual(reading.connector, text.split()[0].casefold())

    def test_framing_head_noun_is_stripped(self):
        reading = extract_topic("information on Skyrim")
        self.assertEqual(reading.subject.casefold(), "skyrim")
        self.assertEqual(reading.stripped_head, "information")

    def test_inner_about_is_preserved(self):
        # "the second about may be semantically meaningful" (spec section 7).
        reading = extract_topic("the concept of a story about Skyrim")
        self.assertIn("about skyrim", reading.subject.casefold())
        self.assertIn("concept", reading.subject.casefold())

    def test_content_noun_is_separated_from_the_subject(self):
        reading = extract_topic("find videos about Skyrim", content_noun="video")
        self.assertEqual(reading.subject.casefold(), "skyrim")
        self.assertEqual(reading.content_noun, "video")

    def test_compound_subject_is_kept_whole(self):
        for text, expected in (
            ("the history of Skyrim", "history of skyrim"),
            ("how Skyrim's leveling system works", "how skyrim's leveling system works"),
            ("popular builds for Skyrim", "popular builds for skyrim"),
        ):
            with self.subTest(text=text):
                self.assertEqual(extract_topic(text).subject.casefold(), expected)

    def test_query_never_begins_with_a_connector_or_verb(self):
        for text in (
            "research about Skyrim",
            "find information about Skyrim",
            "look up information on Skyrim",
            "search the web for cars",
        ):
            with self.subTest(text=text):
                subject = extract_topic(text).subject.casefold()
                self.assertFalse(subject.startswith(("about ", "on ", "information ", "research ", "find ")))

    def test_confident_typo_correction_only_with_a_known_entity(self):
        # With no known entity, a typo is preserved (never silently invented).
        preserved = extract_topic("about sykrim")
        self.assertEqual(preserved.subject.casefold(), "sykrim")
        # With a known entity one edit away, a capitalized typo is corrected.
        corrected = extract_topic("about Skryim mods", known_entities=["Skyrim"])
        self.assertIn("Skyrim", corrected.normalized)

    def test_common_words_are_never_corrected(self):
        # "mode" must not become "code": a real word is not a typo.
        reading = extract_topic("Skyrim survival mode", known_entities=["code"])
        self.assertIn("mode", reading.normalized.casefold())

    def test_normalize_query_reports_change(self):
        cleaned, changed = normalize_query("about Skyrim")
        self.assertTrue(changed)
        self.assertEqual(cleaned.casefold(), "skyrim")
        _, unchanged = normalize_query("Skyrim")
        self.assertFalse(unchanged)


class ResearchRequestRegressionTests(unittest.TestCase):
    """The research matrix from the specification (section 14)."""

    CASES = [
        ("research about Skyrim", "skyrim"),
        ("research on Skyrim", "skyrim"),
        ("research regarding Skyrim", "skyrim"),
        ("research concerning Skyrim", "skyrim"),
        ("find information about Skyrim", "skyrim"),
        ("look up information on Skyrim", "skyrim"),
        ("tell me about Skyrim", "skyrim"),
        ("research the history of Skyrim", "history of skyrim"),
        ("research about the best Skyrim mods", "best skyrim mods"),
    ]

    def test_topic_is_extracted_without_instruction_leakage(self):
        for text, expected in self.CASES:
            with self.subTest(text=text):
                task = interpret(text)
                self.assertEqual(
                    str(task.entities.get("topic") or "").casefold(), expected
                )

    def test_research_query_does_not_contain_a_connector(self):
        for text, _ in self.CASES:
            with self.subTest(text=text):
                task = interpret(text)
                query = web_query(task)
                if query:
                    self.assertFalse(
                        query.casefold().startswith(("about ", "on ", "information ", "regarding "))
                    )


class CompoundTaskRegressionTests(unittest.TestCase):
    """Research + transformation + destination must be three separate slots."""

    def test_research_and_write_to_notepad(self):
        task = interpret("research about Skyrim and write it in Notepad")
        self.assertEqual(capability_list(task), [
            "web.research", "content.format", "applications.write_text",
        ])
        self.assertEqual(str(task.entities.get("topic") or "").casefold(), "skyrim")
        self.assertEqual(task.entities.get("application"), "Notepad")
        # The critical assertion: the connector must not contaminate the query.
        self.assertEqual(web_query(task).casefold(), "skyrim")

    def test_research_history_and_write_key_points(self):
        task = interpret(
            "research about the history of Skyrim and write the key points in Notepad"
        )
        self.assertEqual(str(task.entities.get("topic") or "").casefold(), "history of skyrim")
        self.assertEqual(task.entities.get("application"), "Notepad")
        # A transformation step reshapes the retrieved content before delivery.
        self.assertIn("content.generate", capability_list(task))

    def test_research_summarize_and_put_in_notepad(self):
        task = interpret(
            "research about Skyrim, summarize the important information, and put it in Notepad"
        )
        caps = capability_list(task)
        self.assertIn("web.research", caps)
        self.assertIn("content.generate", caps)
        self.assertIn("applications.write_text", caps)
        self.assertEqual(str(task.entities.get("topic") or "").casefold(), "skyrim")
        self.assertEqual(task.entities.get("application"), "Notepad")

    def test_research_how_leveling_works_summarize_and_write(self):
        task = interpret(
            "research how Skyrim's leveling system works, summarize it, and write it in Notepad"
        )
        caps = capability_list(task)
        self.assertIn("web.research", caps)
        self.assertIn("content.generate", caps)
        self.assertIn("applications.write_text", caps)
        topic = str(task.entities.get("topic") or "").casefold()
        self.assertIn("leveling system", topic)
        self.assertEqual(web_query(task).casefold(), topic)

    def test_research_and_save_important_findings(self):
        task = interpret("research Skyrim and save the important findings to Notepad")
        caps = capability_list(task)
        self.assertIn("web.research", caps)
        self.assertIn("applications.write_text", caps)
        self.assertEqual(str(task.entities.get("topic") or "").casefold(), "skyrim")
        self.assertEqual(task.entities.get("application"), "Notepad")

    def test_research_how_to_make_a_resume_keeps_the_whole_subject(self):
        task = interpret("research how to make a resume and write the key points in Notepad")
        topic = str(task.entities.get("topic") or "").casefold()
        self.assertEqual(topic, "how to make a resume")
        self.assertIn("web.research", capability_list(task))


class NonResearchRegressionTests(unittest.TestCase):
    """The same slot separation applies to other actions (spec section 9)."""

    def test_write_a_poem_about_cars_does_not_research(self):
        task = interpret("write a poem about cars in Notepad")
        caps = capability_list(task)
        self.assertEqual(task.entities.get("content_type"), "poem")
        self.assertEqual(str(task.entities.get("topic") or "").casefold(), "cars")
        self.assertEqual(task.entities.get("application"), "Notepad")
        # No unnecessary web research for a creative task.
        self.assertNotIn("web.research", caps)
        self.assertNotIn("web.search", caps)

    def test_find_videos_about_skyrim_on_youtube(self):
        task = interpret("find videos about Skyrim on YouTube")
        self.assertEqual(task.entities.get("site"), "youtube")
        search = next(
            (a for a in task.actions if a.capability in ("web.search", "web.research")), None
        )
        self.assertIsNotNone(search)
        self.assertEqual(str(search.parameters.get("query") or "").casefold(), "skyrim")

    def test_write_an_explanation_about_photosynthesis(self):
        task = interpret("open Notepad and write a short explanation about photosynthesis")
        topic = str(task.entities.get("topic") or "").casefold()
        self.assertEqual(topic, "photosynthesis")
        self.assertEqual(task.entities.get("application"), "Notepad")

    def test_find_information_and_summarize_without_destination(self):
        task = interpret("find information about climate change and summarize it")
        topic = str(task.entities.get("topic") or "").casefold()
        self.assertEqual(topic, "climate change")


class TopicReadingDiagnosticsTests(unittest.TestCase):
    """The structured reading must be inspectable on the task (spec section 15)."""

    def test_task_exposes_the_topic_reading(self):
        task = interpret("research about Skyrim and write it in Notepad")
        self.assertIn("topic_reading", task.to_dict())
        reading = task.topic_reading
        self.assertEqual(reading.get("raw_topic", "").casefold(), "about skyrim")
        self.assertEqual(reading.get("connector"), "about")
        self.assertEqual(reading.get("subject", "").casefold(), "skyrim")
        self.assertEqual(task.research_query.casefold(), "skyrim")

    def test_normalization_is_recorded_when_it_happens(self):
        reading = extract_topic("about Skyrim", known_entities=["Skyrim"])
        # No correction needed here, so nothing is mislabeled as normalized.
        self.assertEqual(reading.normalized, reading.subject)

    def test_execution_trace_records_the_query(self):
        task = interpret("research about Skyrim")
        traces = [entry for entry in task.execution_trace if entry.get("stage") == "interpretation"]
        self.assertTrue(traces)
        self.assertEqual(str(traces[-1]["details"].get("research_query") or "").casefold(), "skyrim")


class QueryNormalizationRecoveryTests(unittest.TestCase):
    """Bounded recovery when a malformed query returns nothing (spec section 12)."""

    def _engine(self):
        # _normalized_retry_query uses no instance state; construct the engine
        # with lightweight fakes so the recovery logic is exercised in isolation.
        from computer.runtime import register_read_only_tools
        from config import COMPUTER_ROOT
        from reasoning.answer_generator import AnswerGenerator
        from reasoning.reasoning_engine import ReasoningEngine
        from reasoning.self_introspection import SelfIntrospection
        from tools.capabilities import CapabilityRegistry
        from tools.registry import ToolRegistry

        ask = lambda *a, **k: ""
        registry = ToolRegistry()
        register_read_only_tools(registry, root=COMPUTER_ROOT, ask=ask)
        capabilities = CapabilityRegistry(registry)
        return ReasoningEngine(
            self_introspection=SelfIntrospection(capabilities, model_name="test"),
            answer_generator=AnswerGenerator(ask=ask),
            tool_runner=lambda name, params: None,
            capabilities=capabilities,
        )

    def _evidence(self, last_query: str):
        from models_task import EvidenceState
        state = EvidenceState(target="sykrim", goal="find_information")
        state.last_query = last_query
        return state

    def test_recovery_cleans_a_connector_leaked_query(self):
        engine = self._engine()
        task = interpret("research about Skyrim")
        state = self._evidence("about sykrim")
        retry = engine._normalized_retry_query(task, state)
        self.assertIsNotNone(retry)
        # The retry must not carry the "about" connector.
        self.assertFalse(retry.casefold().startswith("about"))

    def test_recovery_prefers_the_task_topic_when_it_differs(self):
        engine = self._engine()
        task = interpret("research how Skyrim's leveling system works")
        state = self._evidence("about skyrim")
        retry = engine._normalized_retry_query(task, state)
        self.assertIsNotNone(retry)
        self.assertIn("leveling system", retry.casefold())

    def test_recovery_is_none_when_the_query_is_already_clean(self):
        engine = self._engine()
        task = interpret("research about Skyrim")
        state = self._evidence("skyrim")
        # The query equals the task's normalized topic, so there is no better
        # query to try: recovery returns None and does not loop.
        self.assertIsNone(engine._normalized_retry_query(task, state))


if __name__ == "__main__":
    unittest.main()
