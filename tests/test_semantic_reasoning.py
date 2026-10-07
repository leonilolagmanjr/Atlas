"""Reliability evaluation suite for Atlas's semantic reasoning layer.

This is the *evaluation* suite the spec asks for (sections 11 and 12). It does
not only check that a final answer "looks good". It measures the intermediate
properties that make the answer trustworthy:

* **Semantic equivalence** - five differently worded requests meaning the same
  thing must produce an equivalent semantic reading and execution strategy.
* **Entity resolution** - the subject is extracted from the paraphrase.
* **Evidence-requirement accuracy** - current/ranking requests require evidence;
  stable knowledge does not.
* **Capability-selection accuracy** - the tools chosen match the goal's needs.
* **Answerability** - a ranking with no evidence must not be answered.
* **Ambiguity handling** - a subjective criterion with no proxy is flagged.

Everything is offline: the deterministic interpreter runs with the LLM disabled,
the model is absent, and the tool registry is the read-only test registry.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from computer.runtime import register_read_only_tools  # noqa: E402
from config import COMPUTER_ROOT  # noqa: E402
from reasoning.intent_engine import IntentEngine  # noqa: E402
from reasoning.query_router import QueryRouter  # noqa: E402
from reasoning.semantic_reasoning import SemanticReasoning  # noqa: E402
from reasoning.self_introspection import SelfIntrospection  # noqa: E402
from reasoning.source_selector import SourceSelector  # noqa: E402
from reasoning.task_interpreter import SemanticTaskInterpreter  # noqa: E402
from tools.capabilities import CapabilityRegistry  # noqa: E402
from tools.registry import ToolRegistry  # noqa: E402


def _capabilities() -> CapabilityRegistry:
    registry = ToolRegistry()
    register_read_only_tools(registry, root=COMPUTER_ROOT, ask=None)
    return CapabilityRegistry(registry)


class _Pipeline:
    """A deterministic, offline instance of the semantic reasoning pipeline."""

    def __init__(self) -> None:
        self.caps = _capabilities()
        self.interpreter = SemanticTaskInterpreter(
            ask=None, capabilities=self.caps, capability_catalog=self.caps.render_catalog()
        )
        self.intent_engine = IntentEngine(capabilities=self.caps)
        self.semantics = SemanticReasoning(ask=None, capabilities=self.caps)
        self.router = QueryRouter()
        self.selector = SourceSelector(self.caps)
        self.introspection = SelfIntrospection(self.caps, model_name="test-model")

    def reason(self, text: str, *, task=None, prior_task=None):
        """Return (task, semantic_decision, signals, source_plan).

        ``task``/``prior_task`` allow a follow-up to be evaluated against the
        previous turn, exactly as the Brain does.
        """

        if task is None:
            task = self.interpreter.interpret(text, context=prior_task)
            task = self.intent_engine.understand(task, prior_task=prior_task)
        decision = self.semantics.reason(text, task=task, prior_task=prior_task)
        signals = self.router.route(
            text, task=task, prior_task=prior_task, self_introspection=self.introspection
        )
        plan = self.selector.select(signals, task_type=task.task_type)
        return task, decision, signals, plan

    def sources(self, text: str) -> list[str]:
        return [source.value for source in self.reason(text)[3].sources]


# -- 1. Semantic equivalence: entity information ------------------------------

class EntityInformationEquivalenceTests(unittest.TestCase):
    """`Who is MrBeast?` and its paraphrases must converge on one reading."""

    PARAPHRASES = [
        "Who is MrBeast?",
        "Tell me about MrBeast.",
        "Search MrBeast.",
        "Give me information about MrBeast.",
        "Look up MrBeast.",
        "What do you know about MrBeast?",
    ]

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_all_paraphrases_resolve_the_same_subject(self):
        for text in self.PARAPHRASES:
            with self.subTest(text=text):
                _, decision, _, _ = self.pipeline.reason(text)
                self.assertEqual(
                    decision.reading.subject.casefold(), "mrbeast",
                    f"subject not resolved for: {text}",
                )

    def test_all_paraphrases_agree_on_the_evidence_strategy(self):
        # The whole point of the Core Problem: the *strategy* must be equivalent
        # even though the wordings differ.
        strategies = set()
        for text in self.PARAPHRASES:
            _, decision, _, _ = self.pipeline.reason(text)
            strategies.add(decision.evidence.resolution_strategy)
        self.assertEqual(
            len(strategies), 1,
            f"semantically equivalent requests produced different strategies: {strategies}",
        )

    def test_all_paraphrases_consult_web_evidence(self):
        # "Who is MrBeast?" and "Search MrBeast." must reach the same source
        # strategy: web evidence, not model memory alone.
        for text in self.PARAPHRASES:
            with self.subTest(text=text):
                sources = self.pipeline.sources(text)
                self.assertIn("web", sources, f"did not consult web evidence for: {text}")

    def test_alias_resolves_to_the_same_evidence_treatment(self):
        # A different name for the same entity ("Jimmy Donaldson") must be treated
        # the same as the canonical name: a preferred-evidence external entity.
        _, decision, _, _ = self.pipeline.reason("Who is Jimmy Donaldson?")
        self.assertNotEqual(decision.reading.subject, "")
        self.assertIn(decision.reading.evidence_requirement, {"preferred", "required"})
        self.assertIn("web", self.pipeline.sources("Who is Jimmy Donaldson?"))

    def test_known_for_paraphrase_is_the_same_information_goal(self):
        _, decision, _, _ = self.pipeline.reason("What is MrBeast known for?")
        self.assertIn("mrbeast", decision.reading.subject.casefold())
        self.assertIn(decision.reading.operation, {"explain", "identify", "retrieve"})


# -- 2. Rankings and comparison ----------------------------------------------

class RankingEquivalenceTests(unittest.TestCase):
    """Ranking paraphrases must converge on ranking + current + required evidence."""

    PARAPHRASES = [
        "Who is the most famous Minecraft YouTuber?",
        "Who is the biggest Minecraft YouTuber?",
        "Which Minecraft YouTuber has the most subscribers?",
        "Who is #1 in Minecraft on YouTube?",
        "Which Minecraft creator has the largest audience?",
        "Who's the biggest Minecraft YouTuber right now?",
        "Who has the biggest Minecraft channel?",
    ]

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_all_rankings_are_comparative(self):
        for text in self.PARAPHRASES:
            with self.subTest(text=text):
                task, decision, _, _ = self.pipeline.reason(text)
                self.assertTrue(decision.reading.comparative, f"not comparative: {text}")
                self.assertTrue(task.comparative)

    def test_all_rankings_require_current_external_evidence(self):
        for text in self.PARAPHRASES:
            with self.subTest(text=text):
                _, decision, _, _ = self.pipeline.reason(text)
                self.assertEqual(decision.reading.freshness_requirement, "current")
                self.assertEqual(decision.reading.evidence_requirement, "required")
                self.assertTrue(decision.requires_retrieval)

    def test_all_rankings_select_a_research_capability(self):
        for text in self.PARAPHRASES:
            with self.subTest(text=text):
                _, decision, _, _ = self.pipeline.reason(text)
                self.assertTrue(
                    {"web.research", "web.search"} & set(decision.capabilities.capabilities),
                    f"no research capability selected for: {text}",
                )

    def test_all_rankings_route_to_web(self):
        for text in self.PARAPHRASES:
            with self.subTest(text=text):
                self.assertIn("web", self.pipeline.sources(text))

    def test_famous_gets_an_objective_proxy(self):
        # "Most famous" is subjective; for a YouTube question, subscriber count is
        # a reasonable objective proxy. The reading must name it.
        _, decision, _, _ = self.pipeline.reason("Who is the most famous Minecraft YouTuber?")
        self.assertEqual(decision.reading.criterion, "most famous")
        self.assertEqual(decision.reading.criterion_proxy, "subscriber count")
        self.assertEqual(
            decision.evidence.resolution_strategy, "use_subscriber_count_as_proxy"
        )

    def test_ranking_query_uses_subject_and_criterion(self):
        _, decision, _, _ = self.pipeline.reason("Who is the most famous Minecraft YouTuber?")
        query = decision.evidence.query.casefold()
        self.assertIn("minecraft", query)


# -- 3. Stable factual knowledge ---------------------------------------------

class StableKnowledgeTests(unittest.TestCase):
    """Stable facts must NOT reach for the web merely because they are questions."""

    STABLE = [
        "What is Minecraft?",
        "What is photosynthesis?",
        "What is HTTP?",
    ]

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_stable_questions_do_not_require_evidence(self):
        for text in self.STABLE:
            with self.subTest(text=text):
                _, decision, _, _ = self.pipeline.reason(text)
                self.assertEqual(decision.reading.evidence_requirement, "unnecessary")

    def test_stable_questions_are_answerable_without_retrieval(self):
        for text in self.STABLE:
            with self.subTest(text=text):
                _, decision, _, _ = self.pipeline.reason(text)
                verdict = self.pipeline.semantics.evaluate_answerability(decision, evidence_count=0)
                self.assertTrue(verdict.can_answer)

    def test_stable_questions_are_not_routed_to_the_web_by_the_semantic_layer(self):
        for text in self.STABLE:
            with self.subTest(text=text):
                features = self.pipeline.reason(text)[2].features
                self.assertNotIn("semantic_evidence_required", features)


# -- 4. Current information --------------------------------------------------

class CurrentInformationTests(unittest.TestCase):
    CURRENT = [
        "Who is the current president of the Philippines?",
        "What is the current Minecraft version?",
        "Who currently has the most YouTube subscribers?",
    ]

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_current_questions_require_evidence(self):
        for text in self.CURRENT:
            with self.subTest(text=text):
                _, decision, _, _ = self.pipeline.reason(text)
                self.assertEqual(decision.reading.evidence_requirement, "required")

    def test_current_questions_route_to_the_web(self):
        for text in self.CURRENT:
            with self.subTest(text=text):
                self.assertIn("web", self.pipeline.sources(text))

    def test_current_question_without_evidence_cannot_be_answered(self):
        _, decision, _, _ = self.pipeline.reason(
            "What is the current Minecraft version?"
        )
        verdict = self.pipeline.semantics.evaluate_answerability(
            decision, evidence_count=0, retrieved_ok=True
        )
        self.assertFalse(verdict.can_answer)
        self.assertTrue(verdict.must_disclose_uncertainty)


# -- 5. Answerability gate ---------------------------------------------------

class AnswerabilityGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_ranking_without_evidence_is_not_answered(self):
        _, decision, _, _ = self.pipeline.reason(
            "Who is the most famous Minecraft YouTuber?"
        )
        verdict = self.pipeline.semantics.evaluate_answerability(decision, evidence_count=0)
        self.assertFalse(verdict.can_answer)
        self.assertEqual(verdict.status, "insufficient")

    def test_ranking_with_evidence_is_answerable(self):
        _, decision, _, _ = self.pipeline.reason(
            "Who is the most famous Minecraft YouTuber?"
        )
        verdict = self.pipeline.semantics.evaluate_answerability(decision, evidence_count=3)
        self.assertTrue(verdict.can_answer)
        self.assertEqual(verdict.status, "sufficient")

    def test_preferred_entity_answer_discloses_missing_evidence(self):
        _, decision, _, _ = self.pipeline.reason("Who is MrBeast?")
        verdict = self.pipeline.semantics.evaluate_answerability(decision, evidence_count=0)
        self.assertTrue(verdict.can_answer)
        self.assertTrue(verdict.must_disclose_uncertainty)

    def test_every_status_maps_to_an_explicit_action(self):
        from reasoning.evidence_policy import STATUS_ACTIONS

        for status, action in STATUS_ACTIONS.items():
            with self.subTest(status=status):
                self.assertTrue(action)


# -- 6. Ambiguity ------------------------------------------------------------

class AmbiguityTests(unittest.TestCase):
    AMBIGUOUS = [
        "What is the best Minecraft server?",
        "Who is the best football player?",
    ]

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_subjective_criterion_is_flagged(self):
        for text in self.AMBIGUOUS:
            with self.subTest(text=text):
                _, decision, _, _ = self.pipeline.reason(text)
                self.assertIn(decision.reading.ambiguity, {"medium", "high"})
                self.assertTrue(decision.evidence.subjective_without_proxy)

    def test_subjective_criterion_asks_rather_than_invents(self):
        for text in self.AMBIGUOUS:
            with self.subTest(text=text):
                _, decision, _, _ = self.pipeline.reason(text)
                verdict = self.pipeline.semantics.evaluate_answerability(
                    decision, evidence_count=0
                )
                # With no evidence, the only honest move is to ask which measure
                # to use - not to invent an objective answer.
                self.assertFalse(verdict.can_answer)
                self.assertIsNotNone(verdict.clarification_question)

    def test_subjective_criterion_with_evidence_answers_but_discloses(self):
        _, decision, _, _ = self.pipeline.reason("What is the best Minecraft server?")
        verdict = self.pipeline.semantics.evaluate_answerability(decision, evidence_count=3)
        self.assertTrue(verdict.can_answer)
        self.assertTrue(verdict.must_disclose_uncertainty)


# -- 7. Contextual follow-ups ------------------------------------------------

class FollowUpResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_pronoun_follow_up_resolves_with_prior_context(self):
        first = "Who is MrBeast?"
        task = self.pipeline.interpreter.interpret(first)
        task = self.pipeline.intent_engine.understand(task)
        self.pipeline.semantics.reason(first, task=task)

        follow = "How many subscribers does he have?"
        task2, decision, _, _ = self.pipeline.reason(follow, prior_task=task)
        # The follow-up is grounded in the conversation, and the semantic layer
        # records that dependency instead of inventing a fresh subject.
        self.assertTrue(
            decision.reading.contextual or bool(task2.context_references),
            "follow-up was not grounded in the conversation",
        )

    def test_put_that_in_notepad_resolves_previous_output(self):
        first = "Who is MrBeast?"
        task = self.pipeline.interpreter.interpret(first)
        task = self.pipeline.intent_engine.understand(task)
        self.pipeline.semantics.reason(first, task=task)

        follow = "Put that in Notepad."
        task2, decision, _, _ = self.pipeline.reason(follow, prior_task=task)
        # The delivery is recognised as a follow-up action, not a new subject.
        self.assertIn(
            decision.reading.operation,
            {"act", "create", "transform", "retrieve", "identify"},
        )


# -- 8. Tools-as-capabilities (not intents) ----------------------------------

class ToolsAsCapabilitiesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_unavailable_capability_is_dropped_not_invented(self):
        # A model may suggest a capability the runtime lacks; the planner must
        # drop it rather than plan against a tool Atlas cannot run.
        _, decision, _, _ = self.pipeline.reason("Who is the most famous Minecraft YouTuber?")
        reports = decision.capabilities.to_dict()
        for capability in reports["selected_capabilities"]:
            self.assertTrue(self.pipeline.caps.exists(capability), capability)

    def test_multi_capability_request_selects_multiple_capabilities(self):
        # A single request may legitimately need several capabilities. A
        # "research and deliver to Notepad" request is exactly that.
        task, _, _, _ = self.pipeline.reason(
            "Find the most popular Minecraft YouTuber and put the answer in Notepad."
        )
        capabilities = set(a.capability for a in task.actions)
        self.assertTrue(
            {"web.research", "web.search"} & capabilities,
            "the research half of the request selected no research capability",
        )
        self.assertTrue(
            {"applications.write_text", "filesystem.write"} & capabilities,
            "the delivery half of the request selected no write capability",
        )

    def test_no_capability_is_selected_from_a_keyword_alone(self):
        # A generation request that happens to contain the word "search" as a
        # topic must not be turned into a web search by a keyword rule.
        _, decision, _, _ = self.pipeline.reason("Write a poem about search engines.")
        self.assertNotIn("web.search", decision.capabilities.capabilities)


# -- 9. Reasoning trace ------------------------------------------------------

class ReasoningTraceTests(unittest.TestCase):
    REQUIRED_KEYS = {
        "goal", "subject", "operation", "criteria", "criterion_proxy", "freshness",
        "evidence_required", "selected_capabilities", "ambiguity",
        "resolution_strategy", "comparative",
    }

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_trace_has_the_documented_shape(self):
        _, decision, _, _ = self.pipeline.reason(
            "Who is the most famous Minecraft YouTuber?"
        )
        trace = decision.trace()
        missing = self.REQUIRED_KEYS - set(trace)
        self.assertFalse(missing, f"trace is missing keys: {missing}")

    def test_trace_explains_the_ranking_decision(self):
        _, decision, _, _ = self.pipeline.reason(
            "Who is the most famous Minecraft YouTuber?"
        )
        trace = decision.trace()
        self.assertEqual(trace["operation"], "rank")
        self.assertTrue(trace["comparative"])
        self.assertEqual(trace["freshness"], "current")
        self.assertTrue(trace["evidence_required"])
        self.assertEqual(trace["resolution_strategy"], "use_subscriber_count_as_proxy")
        self.assertIn("web.research", trace["selected_capabilities"])

    def test_trace_never_contains_raw_reasoning_prose(self):
        # The trace is structured diagnostics, not hidden chain-of-thought: every
        # value is a short scalar, list, or short string.
        _, decision, _, _ = self.pipeline.reason("Who is MrBeast?")
        for key, value in decision.trace().items():
            with self.subTest(key=key):
                if isinstance(value, str):
                    self.assertLess(len(value), 400)
                elif isinstance(value, list):
                    self.assertTrue(all(isinstance(item, (str, bool, int, float)) for item in value))


# -- 10. Backward compatibility ---------------------------------------------

class BackwardCompatibilityTests(unittest.TestCase):
    """The semantic layer must not disturb the existing deterministic behaviour."""

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_file_lookup_stays_local(self):
        sources = self.pipeline.sources("Find my resume.")
        self.assertIn("files", sources)
        self.assertNotIn("web", sources)

    def test_content_creation_is_unchanged(self):
        task, _, _, _ = self.pipeline.reason("Write a poem about cars.")
        self.assertEqual([a.capability for a in task.actions], ["content.generate"])

    def test_self_description_stays_self(self):
        _, _, signals, plan = self.pipeline.reason("What can you do?")
        self.assertTrue(signals.is_self_query)
        self.assertEqual([s.value for s in plan.sources], ["self"])

    def test_existing_intent_goals_still_populate_the_task(self):
        # The internal execution categories (INTENT_GOALS) survive untouched.
        task, _, _, _ = self.pipeline.reason("Who is the most famous Minecraft YouTuber?")
        self.assertNotEqual(task.goal, "")
        self.assertTrue(task.object_type)


if __name__ == "__main__":
    unittest.main()
