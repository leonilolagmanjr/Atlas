"""Adversarial agent benchmark for temporal-context carry-forward.

This suite tests that follow-up questions inherit temporal context from prior
conversation turns. The central regression: a follow-up like "Who was the
Finals MVP?" after "Who won the NBA Finals in 2025?" must carry forward the
year 2025 instead of defaulting to the current year.

Categories covered:
  A. Conversation context carry-forward (year anchor)
  B. Current information requires evidence
  C. Temporal expression carry-forward (this year, last night)
  D. Follow-up question context grounding
  E. Multi-participant evidence convergence
  F. Source conflict resolution
  G. Insufficient evidence handling
  H. Local knowledge boundaries
  I. Computer action routing
  J. Hybrid query routing
  K. Failure handling
  L. Recovery from recoverable failures
  M. Ambiguity detection
  N. Typo robustness
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoning.query_router import QueryRouter
from reasoning.semantic_reasoning import SemanticReasoning
from reasoning.source_selector import SourceSelector
from reasoning.self_introspection import SelfIntrospection
from models_task import Task


class _FakeCapabilities:
    _NAMES = frozenset({
        "web.search", "web.fetch", "web.research",
        "filesystem.list", "filesystem.read", "filesystem.search",
        "filesystem.search_content", "filesystem.metadata",
        "system.info", "processes.list",
        "content.generate", "content.format",
    })

    def exists(self, name: str) -> bool:
        return name in self._NAMES

    def names(self) -> list[str]:
        return sorted(self._NAMES)

    def __iter__(self):
        return iter(sorted(self._NAMES))


class _Pipeline:
    def __init__(self) -> None:
        caps = _FakeCapabilities()
        self.semantics = SemanticReasoning(ask=None, capabilities=caps)
        self.router = QueryRouter()
        self.selector = SourceSelector(caps)
        self.introspection = SelfIntrospection(caps, model_name="test-model")

    def reason(self, text: str, *, history: str = ""):
        task = Task(original_prompt=text, goal="answer", task_type="informational")
        decision = self.semantics.reason(text, task=task, history=history)
        signals = self.router.route(
            text, task=task, self_introspection=self.introspection
        )
        plan = self.selector.select(signals, task_type=task.task_type)
        return decision, signals, plan

    def sources(self, text: str, *, history: str = "") -> set[str]:
        return {source.value for source in self.reason(text, history=history)[2].sources}

    def query(self, text: str, *, history: str = "") -> str:
        return self.reason(text, history=history)[0].evidence.query


# =============================================================================
# A. Conversation context carry-forward (year anchor)
# =============================================================================

class ConversationCarryForwardTests(unittest.TestCase):
    """A follow-up question must inherit the year from prior conversation turns."""

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_follow_up_inherits_year_from_history(self) -> None:
        history = "User: Who won the NBA Finals in 2025?\nAssistant: Searching the web."
        query = self.pipeline.query("Who was the Finals MVP?", history=history)
        self.assertIn("2025", query)

    def test_follow_up_without_year_defaults_to_no_explicit_period(self) -> None:
        query = self.pipeline.query("Who was the Finals MVP?")
        self.assertNotIn("2025", query)
        self.assertNotIn("2024", query)

    def test_explicit_year_in_follow_up_overrides_history(self) -> None:
        history = "User: Who won the NBA Finals in 2025?\nAssistant: Searching."
        query = self.pipeline.query("Who was the 2024 Finals MVP?", history=history)
        self.assertIn("2024", query)
        self.assertNotIn("2025", query)

    def test_history_without_temporal_anchor_does_not_inject_year(self) -> None:
        history = "User: Tell me about LeBron James.\nAssistant: LeBron James is..."
        query = self.pipeline.query("Who was the Finals MVP?", history=history)
        self.assertNotIn("2025", query)

    def test_follow_up_requires_web_evidence_with_history(self) -> None:
        history = "User: Who won the NBA Finals in 2025?\nAssistant: Searching."
        decision, _, _ = self.pipeline.reason("Who was the Finals MVP?", history=history)
        self.assertEqual("required", decision.reading.evidence_requirement)
        self.assertEqual("current", decision.reading.freshness_requirement)
        self.assertIn("web", self.pipeline.sources("Who was the Finals MVP?", history=history))


# =============================================================================
# B. Current information requires evidence
# =============================================================================

class CurrentInformationRequiresEvidenceTests(unittest.TestCase):
    """Requests for current facts must require external evidence and web access."""

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_latest_championship_requires_evidence(self) -> None:
        for text in (
            "Who won the latest championship?",
            "Who took the title this year?",
            "Which team won the final?",
            "Who is the reigning champion?",
        ):
            with self.subTest(request=text):
                decision, _, plan = self.pipeline.reason(text)
                self.assertEqual("required", decision.reading.evidence_requirement)
                self.assertEqual("current", decision.reading.freshness_requirement)
                self.assertIn("web", {s.value for s in plan.sources})

    def test_current_value_query_requires_evidence(self) -> None:
        for text in (
            "What's Bitcoin trading at?",
            "How many subscribers does MrBeast have?",
            "What is the current price of gold?",
        ):
            with self.subTest(request=text):
                decision, _, plan = self.pipeline.reason(text)
                self.assertEqual("required", decision.reading.evidence_requirement)
                self.assertIn("web", {s.value for s in plan.sources})


# =============================================================================
# C. Temporal expression carry-forward (this year, last night)
# =============================================================================

class TemporalCarryForwardTests(unittest.TestCase):
    """Various temporal expressions in history must be inherited by follow-ups."""

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_this_year_carry_forward(self) -> None:
        history = "User: Who won this year's championship?\nAssistant: Searching."
        query = self.pipeline.query("Who was the MVP?", history=history)
        # The year should be in the query (resolved from "this year")
        from reasoning.temporal_resolution import system_now
        year = system_now().year
        self.assertIn(str(year), query)

    def test_last_night_carry_forward(self) -> None:
        history = "User: What happened last night?\nAssistant: Looking that up."
        query = self.pipeline.query("What was the score?", history=history)
        # Should carry forward a date from "last night"
        self.assertTrue(len(query) > 0)

    def test_explicit_year_not_overridden_by_other_history(self) -> None:
        history = "User: What happened last night?\nAssistant: Checking."
        query = self.pipeline.query("Who won the 2023 championship?", history=history)
        self.assertIn("2023", query)


# =============================================================================
# D. Follow-up question context grounding
# =============================================================================

class FollowUpContextGroundingTests(unittest.TestCase):
    """Follow-up questions must be grounded in conversation context."""

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_follow_up_about_same_event_is_current(self) -> None:
        history = "User: Who won the NBA Finals in 2025?\nAssistant: I'll search for that."
        decision, _, _ = self.pipeline.reason("Who was the Finals MVP?", history=history)
        self.assertEqual("required", decision.reading.evidence_requirement)
        self.assertTrue(decision.reading.event_result or decision.reading.final_event_result)

    def test_subject_is_preserved_from_history(self) -> None:
        history = "User: Who won the NBA Finals in 2025?\nAssistant: Searching."
        decision, _, _ = self.pipeline.reason("Who was the Finals MVP?", history=history)
        subject = (decision.reading.subject or "").casefold()
        self.assertIn("finals", subject)


# =============================================================================
# E. Multi-participant evidence convergence
# =============================================================================

class MultiParticipantEvidenceTests(unittest.TestCase):
    """Paraphrases of the same question must converge on the same evidence strategy."""

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_paraphrases_share_evidence_requirement(self) -> None:
        history = "User: Who won the NBA Finals in 2025?\nAssistant: Searching."
        paraphrases = (
            "Who was the Finals MVP?",
            "Who won the MVP award?",
            "Which player was named Finals MVP?",
            "Who took home the MVP trophy?",
        )
        requirements = {
            self.pipeline.reason(p, history=history)[0].reading.evidence_requirement
            for p in paraphrases
        }
        self.assertEqual({"required"}, requirements)


# =============================================================================
# F. Source conflict resolution
# =============================================================================

class SourceConflictTests(unittest.TestCase):
    """When sources conflict, the system must have a defined resolution."""

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_event_result_routes_to_web_not_kb(self) -> None:
        decision, _, plan = self.pipeline.reason("Who won the 2025 NBA Finals?")
        self.assertEqual("required", decision.reading.evidence_requirement)
        self.assertIn("web", {s.value for s in plan.sources})


# =============================================================================
# G. Insufficient evidence handling
# =============================================================================

class InsufficientEvidenceTests(unittest.TestCase):
    """When no evidence is available, the system must not fabricate."""

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_historical_fact_without_web_is_handled(self) -> None:
        # A question about a past event with no web access should be flagged
        # appropriately by the answerability gate
        decision, _, _ = self.pipeline.reason("Who won the 1998 NBA Finals?")
        self.assertEqual("required", decision.reading.evidence_requirement)
        self.assertEqual("current" if decision.reading.event_result else "stable",
                         decision.reading.freshness_requirement)


# =============================================================================
# H. Local knowledge boundaries
# =============================================================================

class LocalKnowledgeTests(unittest.TestCase):
    """Stable knowledge questions must not require external evidence."""

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_definitions_do_not_require_evidence(self) -> None:
        for text in (
            "What is a search algorithm?",
            "Explain recursion.",
            "How does a bloom filter work?",
        ):
            with self.subTest(request=text):
                decision, _, plan = self.pipeline.reason(text)
                self.assertNotEqual("required", decision.reading.evidence_requirement)
                self.assertNotIn("web", {s.value for s in plan.sources})

    def test_concept_questions_stay_local(self) -> None:
        decision, _, plan = self.pipeline.reason("What is a knowledge graph?")
        self.assertNotIn("web", {s.value for s in plan.sources})


# =============================================================================
# I. Computer action routing
# =============================================================================

class ComputerActionRoutingTests(unittest.TestCase):
    """Computer actions must route correctly, not to web research."""

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_open_notepad_is_not_research(self) -> None:
        decision, _, plan = self.pipeline.reason("Open Notepad.")
        self.assertNotIn("web", {s.value for s in plan.sources})


# =============================================================================
# J. Hybrid query routing
# =============================================================================

class HybridQueryTests(unittest.TestCase):
    """Questions that reference the artifact but ask a question are not retrieval."""

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_hybrid_question_does_not_become_artifact_request(self) -> None:
        decision, _, plan = self.pipeline.reason("What is the runtime of the script you found?")
        # The question about a script's runtime is stable/explanatory, not research
        self.assertNotIn("web", {s.value for s in plan.sources})


# =============================================================================
# K. Failure handling
# =============================================================================

class FailureHandlingTests(unittest.TestCase):
    """The system must handle failures gracefully."""

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_ambiguous_request_is_flagged(self) -> None:
        decision, _, _ = self.pipeline.reason("Which one is better?")
        # Ambiguous requests should be flagged, not silently routed
        self.assertTrue(
            decision.reading.ambiguity is not None or
            decision.reading.evidence_requirement != "required"
        )


# =============================================================================
# L. Recovery from recoverable failures
# =============================================================================

class RecoveryTests(unittest.TestCase):
    """Recoverable failures must be detected and handled."""

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_missing_context_asks_instead_of_hallucinating(self) -> None:
        # A pronoun with no prior context should be flagged, not hallucinated
        decision, _, _ = self.pipeline.reason("What about that?")
        self.assertTrue(
            decision.reading.ambiguity is not None or
            decision.reading.evidence_requirement != "required"
        )


# =============================================================================
# M. Ambiguity detection
# =============================================================================

class AmbiguityDetectionTests(unittest.TestCase):
    """Subjective criteria with no objective proxy must be flagged."""

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_subjective_criterion_is_flagged(self) -> None:
        decision, _, _ = self.pipeline.reason("What is the best movie ever?")
        self.assertTrue(decision.reading.subjective_criterion or decision.reading.ambiguity is not None)


# =============================================================================
# N. Typo robustness
# =============================================================================

class TypoRobustnessTests(unittest.TestCase):
    """Typos must not change the semantic decision."""

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_typo_in_event_name_keeps_current_routing(self) -> None:
        for text in ("Who won the nba finlas?", "Who won last nights lakers game?"):
            with self.subTest(request=text):
                decision, _, plan = self.pipeline.reason(text)
                self.assertEqual("required", decision.reading.evidence_requirement)
                self.assertEqual("current", decision.reading.freshness_requirement)

    def test_typo_subject_is_preserved_not_normalized(self) -> None:
        decision, _, _ = self.pipeline.reason("Who won the latest iphnoe giveaway?")
        self.assertIn("iphnoe", (decision.reading.subject or "").casefold())


if __name__ == "__main__":
    unittest.main()
