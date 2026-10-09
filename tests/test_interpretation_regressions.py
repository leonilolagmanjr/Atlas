"""Regressions for the behavioral-evaluation failures (interpretation/routing).

These lock in the fixes for the four simulated behavioral cases that failed,
*and* the neighboring behavior each fix must not break. They are written from the
user's perspective and assert the decision (which capability/source is chosen,
whether a clarification is asked), not prompt wording, so a passing suite cannot
be an artefact of matching the four original sentences.

Root causes covered:

1. **Local precedence.** A local-machine request (a folder listing, the screen)
   or one the interpreter already resolved to a local capability must never be
   "upgraded" into a public web search, even when the semantic layer scored it as
   needing evidence. ``CapabilityPlanner`` used to check "current information"
   before "local", so a local listing ran ``web.search``.
2. **Self-contained destination.** "search the web for X and put it in Notepad"
   is not a back-reference: ``it`` refers to the same sentence's result, so the
   request must not be blocked as ambiguous prior output.
3. **Blocking ambiguity.** An ambiguity the engine cannot settle from context and
   that the interpreter built no plan for must produce a clarification question
   ("Find that file." with no prior turn) rather than an ungrounded search. A
   *conversational* reference must NOT: the conversation source is the resolution.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from computer.runtime import register_read_only_tools  # noqa: E402
from config import COMPUTER_ROOT  # noqa: E402
from reasoning.capability_planner import CapabilityPlanner  # noqa: E402
from reasoning.evidence_policy import EvidenceDecision  # noqa: E402
from reasoning.intent_engine import IntentEngine  # noqa: E402
from reasoning.reference_resolver import ReferenceResolver  # noqa: E402
from reasoning.semantic_request import SemanticRequest  # noqa: E402
from reasoning.task_interpreter import SemanticTaskInterpreter  # noqa: E402
from tools.capabilities import CapabilityRegistry  # noqa: E402
from tools.registry import ToolRegistry  # noqa: E402


def _capabilities() -> CapabilityRegistry:
    registry = ToolRegistry()
    register_read_only_tools(registry, root=COMPUTER_ROOT, ask=None)
    return CapabilityRegistry(registry)


class _Interp:
    """Deterministic interpret + intent reading, as the Brain uses them."""

    def __init__(self) -> None:
        self.caps = _capabilities()
        self.interpreter = SemanticTaskInterpreter(
            ask=None, capabilities=self.caps, capability_catalog=self.caps.render_catalog()
        )
        self.intent = IntentEngine(capabilities=self.caps)

    def task(self, text: str, *, prior_task=None, history: str = ""):
        task = self.interpreter.interpret(text, context=prior_task)
        return self.intent.understand(task, prior_task=prior_task, history=history)


# ---------------------------------------------------------------------------
# 1. Local precedence: a local request is never sent to the public web
# ---------------------------------------------------------------------------


class LocalPrecedenceTests(unittest.TestCase):
    """A local-machine request must select a local capability, never web.search."""

    def setUp(self) -> None:
        self.caps = _capabilities()
        self.interp = _Interp()

    def test_folder_listing_selects_a_local_capability(self):
        capabilities = self.interp.task("List the files in the tests folder.")
        self.assertIn("filesystem.list", [a.capability for a in capabilities.actions])

    def test_folder_listing_is_not_upgraded_to_web_search(self):
        # Even with the semantic layer demanding retrieval, a local capability is
        # what gets planned: web.search is never added to a local listing.
        for text in (
            "List the files in the tests folder.",
            "Show me the files in my downloads folder.",
            "What files are in the Documents folder?",
            "List the files in this project.",
        ):
            with self.subTest(text=text):
                task = self.interp.task(text)
                planner = CapabilityPlanner(self.caps)
                plan = planner.plan(
                    SemanticRequest(subject="folder", local=True, freshness_requirement="current"),
                    EvidenceDecision(requirement="required", freshness="current"),
                    task=task,
                )
                self.assertNotIn("web.search", plan.capabilities)
                self.assertNotIn("web.research", plan.capabilities)

    def test_screen_request_selects_a_screen_observation_capability(self):
        # "my screen" is local; the capabilities it selects must be observation
        # capabilities, never a public web search.
        task = self.interp.task("What is on my screen right now?")
        planner = CapabilityPlanner(self.caps)
        plan = planner.plan(
            SemanticRequest(subject="screen", local=True, freshness_requirement="current"),
            EvidenceDecision(requirement="required", freshness="current"),
            task=task,
        )
        self.assertFalse(
            {"web.search", "web.research"} & set(plan.capabilities),
            f"a screen request selected a web capability: {plan.capabilities}",
        )
        self.assertTrue(
            {"computer.observe", "computer.vision_observe", "computer.find", "system.info"}
            & set(plan.capabilities),
            f"a screen request selected no observation capability: {plan.capabilities}",
        )

    def test_screen_paraphrases_are_all_local(self):
        for text in (
            "What is on my screen right now?",
            "What is showing on my screen?",
            "Show me the open windows.",
        ):
            with self.subTest(text=text):
                task = self.interp.task(text)
                planner = CapabilityPlanner(self.caps)
                plan = planner.plan(
                    SemanticRequest(subject="screen", local=True, freshness_requirement="current"),
                    EvidenceDecision(requirement="required", freshness="current"),
                    task=task,
                )
                self.assertNotIn("web.search", plan.capabilities)

    def test_a_genuine_hybrid_still_uses_the_web(self):
        # The counterexample: a request that really needs external facts must
        # still get a retrieval capability. "search the web ... and write it"
        # planned a web capability, so the planner must keep it.
        task = self.interp.task("Search the web for the Bee Movie script and put it in Notepad.")
        planned = {a.capability for a in task.actions}
        self.assertTrue(
            {"web.search", "web.research"} & planned,
            f"the hybrid request planned no retrieval capability: {planned}",
        )

    def test_external_entity_question_still_uses_the_web(self):
        # A non-local information request is unaffected by the local precedence.
        for text in (
            "Who is MrBeast?",
            "What is the latest Docker release?",
            "Research the newest Python version.",
        ):
            with self.subTest(text=text):
                task = self.interp.task(text)
                planner = CapabilityPlanner(self.caps)
                plan = planner.plan(
                    SemanticRequest(subject="topic", local=False, freshness_requirement="current"),
                    EvidenceDecision(requirement="required", freshness="current"),
                    task=task,
                )
                self.assertTrue(
                    {"web.search", "web.research"} & set(plan.capabilities),
                    f"an external request selected no web capability: {plan.capabilities}",
                )


# ---------------------------------------------------------------------------
# 2. A self-contained request is not a back-reference
# ---------------------------------------------------------------------------


class SelfContainedDestinationTests(unittest.TestCase):
    """A destination phrase inside a self-contained request is not a reference."""

    def setUp(self) -> None:
        self.resolver = ReferenceResolver()

    def test_self_contained_search_and_deliver_is_not_a_reference(self):
        for text in (
            "Search the web for the Bee Movie script and put it in Notepad.",
            "Find the Skyrim script and write it in Notepad.",
            "Research the latest AI news and save it to a file.",
            "Look up the recipe and put it in Word.",
        ):
            with self.subTest(text=text):
                resolved = self.resolver.resolve(text, prior_task=None, history="")
                self.assertNotEqual(
                    resolved.target,
                    "previous_output",
                    f"a self-contained request was treated as a back-reference: {text}",
                )

    def test_destination_only_followup_is_still_a_reference(self):
        # The behavior that must be preserved: "put that in Notepad" with no
        # producing clause really is a reference to earlier output.
        for text in (
            "Put that in Notepad.",
            "Put it in Notepad.",
            "Save it to a file.",
            "Write that into Word.",
        ):
            with self.subTest(text=text):
                resolved = self.resolver.resolve(
                    text, prior_task=None, history="user: research something"
                )
                self.assertEqual(resolved.target, "previous_output")
                self.assertTrue(resolved.has_reference)

    def test_deliver_first_then_search_is_still_a_reference(self):
        # The producing verb comes *after* the destination phrase, so the pronoun
        # still refers to earlier output: the guard must be order-aware.
        resolved = self.resolver.resolve("Put that in Notepad, then search for more.", history="x")
        self.assertEqual(resolved.target, "previous_output")

    def test_continuation_phrases_are_unaffected(self):
        # Unrelated reference behavior must not regress.
        for text in ("Continue.", "Tell me more.", "Elaborate on that."):
            with self.subTest(text=text):
                resolved = self.resolver.resolve(text, prior_task=None, history="x")
                self.assertTrue(resolved.has_reference)


# ---------------------------------------------------------------------------
# 3. A blocking ambiguity asks; a conversational one does not
# ---------------------------------------------------------------------------


class BlockingAmbiguityTests(unittest.TestCase):
    """An unresolvable reference asks for clarification, from no context."""

    def setUp(self) -> None:
        self.interp = _Interp()

    def test_unresolved_file_reference_asks_instead_of_searching(self):
        for text in (
            "Find that file.",
            "Open that file.",
            "Find that document.",
        ):
            with self.subTest(text=text):
                task = self.interp.task(text, prior_task=None, history="")
                self.assertTrue(
                    task.needs_clarification,
                    f"an unresolvable reference did not ask for clarification: {text}",
                )
                self.assertTrue(task.clarification_question)
                self.assertFalse(task.actions, f"a clarification still planned actions: {task.actions}")

    def test_conversational_reference_does_not_ask(self):
        # The counterexample the first fix attempt broke: "what do you know
        # about X" is grounded in the conversation, not an unresolved reference.
        for text in (
            "What do you know about MrBeast?",
            "Who is MrBeast?",
            "Tell me about Docker.",
        ):
            with self.subTest(text=text):
                task = self.interp.task(text, prior_task=None, history="")
                self.assertFalse(
                    task.needs_clarification,
                    f"a conversational request became a clarification: {text}",
                )

    def test_resolved_followup_does_not_ask(self):
        # With a prior task present, "put that in Notepad" resolves and proceeds.
        prior = self.interp.task("Research the Bee Movie script.")
        task = self.interp.task(
            "Put that in Notepad.",
            prior_task=prior,
            history="user: Research the Bee Movie script.\nassistant: <script>",
        )
        self.assertFalse(task.needs_clarification)
        self.assertTrue(task.actions, "a resolved follow-up planned no action")


if __name__ == "__main__":
    unittest.main()
