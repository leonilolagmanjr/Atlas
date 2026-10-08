"""Adversarial generalization benchmark for Atlas request understanding.

This suite exists to test the *architecture*, not the implementation's examples.
Every request here is phrased differently from the wording used while the
semantic layer was implemented, and many name entities and domains that appear
nowhere in Atlas's routing logic. The claims under test are semantic:

* a request for the factual outcome of an event is recognized as such whatever
  the *words* used ("won", "took the title", "was crowned", "ended up winning");
* a request about the current world requires external evidence, while a request
  for a definition does not;
* an unseen competition, sport, or organisation needs no new rule;
* six paraphrases of one question converge on the same routing and evidence
  requirement.

The tests assert *properties* (evidence requirement, freshness, whether web is
consulted, whether a subject was resolved). They never assert an exact query
string, because that would encode one phrasing as the answer rather than testing
that the understanding generalizes.

Nothing here executes a tool: the pipeline is deterministic and offline.
"""

from __future__ import annotations

import unittest

from reasoning.query_router import QueryRouter
from reasoning.semantic_reasoning import SemanticReasoning
from reasoning.source_selector import SourceSelector
from reasoning.self_introspection import SelfIntrospection


class _FakeCapabilities:
    """A registry that reports every research capability as available."""

    _NAMES = frozenset(
        {
            "web.search", "web.fetch", "web.research",
            "filesystem.list", "filesystem.read", "filesystem.search",
            "filesystem.search_content", "filesystem.metadata",
            "system.info", "processes.list",
            "content.generate", "content.format",
        }
    )

    def exists(self, name: str) -> bool:
        return name in self._NAMES

    def names(self) -> list[str]:
        return sorted(self._NAMES)

    def __iter__(self):
        return iter(sorted(self._NAMES))


class _Pipeline:
    """The deterministic understanding -> routing pipeline, offline."""

    def __init__(self) -> None:
        caps = _FakeCapabilities()
        self.semantics = SemanticReasoning(ask=None, capabilities=caps)
        self.router = QueryRouter()
        self.selector = SourceSelector(caps)
        self.introspection = SelfIntrospection(caps, model_name="test-model")

    def route(self, text: str):
        from models_task import Task

        task = Task(original_prompt=text, goal="answer", task_type="informational")
        decision = self.semantics.reason(text, task=task)
        signals = self.router.route(
            text, task=task, self_introspection=self.introspection
        )
        plan = self.selector.select(signals, task_type=task.task_type)
        return decision, signals, plan

    def sources(self, text: str) -> set[str]:
        return {source.value for source in self.route(text)[2].sources}


#: Requests whose answer depends on the current state of the world. Every one of
#: these must require external evidence and must consult the web.
CURRENT_AND_TEMPORAL: tuple[str, ...] = (
    "Who won the latest championship?",
    "Who took the title this year?",
    "Who won yesterday's game?",
    "What happened last night?",
    "What happened three days ago?",
    "Who won two consecutive championships?",
    "Which team won the final?",
    "Who ended up winning?",
    "The championship went to whom?",
    "Did the Lakers win last night?",
    "Tell me who took the championship this year.",
    "I need to know who won yesterday.",
    "Tell me who took the championship.",
    "Who is the reigning champion of the Norwegian chess league?",
    "Who won the latest Quidditch World Cup?",
    "Who took the crown at the 2026 Baku Masters?",
    "Who won the nba finlas?",
    "Who won last nights lakers game?",
    "Who won the most recent title on the professional darts tour?",
)

#: Requests whose answer is a live *value* rather than an event outcome.
VALUE_REQUESTS: tuple[str, ...] = (
    "What's Bitcoin trading at?",
    "What is Bitcoin going for right now?",
    "Can you check what Bitcoin is at?",
    "How many subscribers does MrBeast have?",
    "What is the current price of gold?",
)

#: Concepts. These are answerable from knowledge; they must not be routed to the
#: web merely because a technical word appears.
CONCEPTUAL: tuple[str, ...] = (
    "What is a search algorithm?",
    "Explain recursion.",
    'What does "current" mean in physics?',
    "What is a knowledge graph?",
    "How does a bloom filter work?",
    "Why did the 2024 Lakers win?",
)

#: Requests that name an external entity and ask *about* it. These may consult
#: the web; what matters is that the subject is resolved and evidence is not
#: classified as unnecessary.
ABOUT_AN_ENTITY: tuple[str, ...] = (
    "Who is MrBeast?",
    "Tell me about Skyrim.",
    "Search MrBeast.",
    "Look up MrBeast.",
    "What do you know about MrBeast?",
)

#: Six paraphrases of one question. They must converge on one evidence
#: requirement and one freshness requirement.
PARAPHRASES: tuple[str, ...] = (
    "Who won the championship?",
    "Who took the title?",
    "Which team won the final?",
    "Who was crowned champion?",
    "Who ended up winning the championship?",
    "Who won the most recent championship?",
)


class CurrentInformationRequiresEvidenceTests(unittest.TestCase):
    """A current/event request must demand external evidence."""

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_every_current_request_requires_evidence(self) -> None:
        failures: list[str] = []
        for text in CURRENT_AND_TEMPORAL:
            decision, _, _ = self.pipeline.route(text)
            if decision.reading.evidence_requirement != "required":
                failures.append(
                    "%s -> %s" % (text, decision.reading.evidence_requirement)
                )
        self.assertEqual([], failures, "not 'required':\n" + "\n".join(failures))

    def test_every_current_request_is_fresh(self) -> None:
        failures: list[str] = []
        for text in CURRENT_AND_TEMPORAL:
            decision, _, _ = self.pipeline.route(text)
            if decision.reading.freshness_requirement != "current":
                failures.append(
                    "%s -> %s" % (text, decision.reading.freshness_requirement)
                )
        self.assertEqual([], failures, "not 'current':\n" + "\n".join(failures))

    def test_every_current_request_consults_the_web(self) -> None:
        failures: list[str] = []
        for text in CURRENT_AND_TEMPORAL:
            if "web" not in self.pipeline.sources(text):
                failures.append(text)
        self.assertEqual([], failures, "did not consult the web:\n" + "\n".join(failures))


class ConceptualStaysLocalTests(unittest.TestCase):
    """A concept is answered from knowledge, not researched."""

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_concepts_do_not_require_evidence(self) -> None:
        failures: list[str] = []
        for text in CONCEPTUAL:
            decision, _, _ = self.pipeline.route(text)
            if decision.reading.evidence_requirement == "required":
                failures.append(
                    "%s -> %s" % (text, decision.reading.evidence_requirement)
                )
        self.assertEqual([], failures, "wrongly 'required':\n" + "\n".join(failures))

    def test_concepts_do_not_become_web_lookups(self) -> None:
        failures = [t for t in CONCEPTUAL if "web" in self.pipeline.sources(t)]
        self.assertEqual([], failures, "routed to web:\n" + "\n".join(failures))


class ValueRequestTests(unittest.TestCase):
    """A request for a live value must require external evidence."""

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_values_require_evidence(self) -> None:
        failures: list[str] = []
        for text in VALUE_REQUESTS:
            decision, _, _ = self.pipeline.route(text)
            if decision.reading.evidence_requirement != "required":
                failures.append(
                    "%s -> %s" % (text, decision.reading.evidence_requirement)
                )
        self.assertEqual([], failures, "not 'required':\n" + "\n".join(failures))


class UnseenDomainTests(unittest.TestCase):
    """Semantic properties must outrank known entities.

    The domains below appear nowhere in Atlas's routing vocabulary. If routing
    depended on recognising "NBA" or "World Cup", these would fail.
    """

    UNSEEN = (
        "Who won the latest Summoner's Rift Invitational?",
        "Who took the title at the Bodrum Regatta?",
        "Who is the champion of the Outer Hebrides shinty league?",
        "Who won the 2026 Tbilisi open?",
        "Which competitor won last night's Ulaanbaatar marathon?",
    )

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_unseen_events_are_current_and_required(self) -> None:
        for text in self.UNSEEN:
            with self.subTest(request=text):
                decision, _, plan = self.pipeline.route(text)
                self.assertEqual("required", decision.reading.evidence_requirement)
                self.assertEqual("current", decision.reading.freshness_requirement)
                self.assertIn("web", {source.value for source in plan.sources})

    def test_unseen_event_name_survives_into_the_subject(self) -> None:
        """The entity the user named must not be discarded by the reader."""
        decision, _, _ = self.pipeline.route(
            "Who won the latest Summoner's Rift Invitational?"
        )
        subject = (decision.reading.subject or "").casefold()
        self.assertIn("summoner", subject)


class ParaphraseConvergenceTests(unittest.TestCase):
    """Differently worded forms of one question must reach one decision."""

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_paraphrases_share_one_evidence_requirement(self) -> None:
        requirements = {
            self.pipeline.route(text)[0].reading.evidence_requirement
            for text in PARAPHRASES
        }
        self.assertEqual({"required"}, requirements)

    def test_paraphrases_share_one_freshness_requirement(self) -> None:
        freshness = {
            self.pipeline.route(text)[0].reading.freshness_requirement
            for text in PARAPHRASES
        }
        self.assertEqual({"current"}, freshness)

    def test_paraphrases_all_consult_the_web(self) -> None:
        for text in PARAPHRASES:
            with self.subTest(request=text):
                self.assertIn("web", self.pipeline.sources(text))


class TypoRobustnessTests(unittest.TestCase):
    """A misspelling must not change the decision, and the entity is preserved."""

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_typos_keep_the_decision(self) -> None:
        for text in ("Who won the nba finlas?", "Who won last nights lakers game?"):
            with self.subTest(request=text):
                decision, _, _ = self.pipeline.route(text)
                self.assertEqual("required", decision.reading.evidence_requirement)
                self.assertEqual("current", decision.reading.freshness_requirement)

    def test_misspelled_subject_is_not_normalized_away(self) -> None:
        """Atlas must not invent a correction for an entity it cannot verify."""
        decision, _, _ = self.pipeline.route("Who won the latest iphnoe giveaway?")
        self.assertIn("iphnoe", (decision.reading.subject or "").casefold())


class SelfQueryPurityTests(unittest.TestCase):
    """A question about Atlas's own abilities stays a self query."""

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_self_queries_do_not_become_entity_lookups(self) -> None:
        for text in ("What can you do?", "Who are you?", "What are your capabilities?"):
            with self.subTest(request=text):
                self.assertNotIn("web", self.pipeline.sources(text))

    def test_naming_an_entity_beside_you_is_an_external_request(self) -> None:
        """A message that merely contains "you" but names an entity is external."""
        decision, _, _ = self.pipeline.route("What do you know about MrBeast?")
        self.assertNotEqual("unnecessary", decision.reading.evidence_requirement)


class ExplanationIsNotLookupTests(unittest.TestCase):
    """Asking why something happened is reasoning, not retrieval."""

    def setUp(self) -> None:
        self.pipeline = _Pipeline()

    def test_explanations_are_not_researched(self) -> None:
        for text in (
            "Why did the 2024 Lakers win?",
            "How does a search algorithm work?",
            "Explain why the printer is slow.",
        ):
            with self.subTest(request=text):
                self.assertNotIn("web", self.pipeline.sources(text))


if __name__ == "__main__":
    unittest.main()
