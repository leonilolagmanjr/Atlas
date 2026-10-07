"""Tests for the evidence gate (spec section: evidence gate).

Validates that the gate blocks fabricated answers when evidence is required
but insufficient, and that it allows model fallback with disclosure when
evidence is merely preferred.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models_task import Task  # noqa: E402
from reasoning.evidence_manager import EvidenceManager  # noqa: E402
from reasoning.evidence_gate import EvidenceGate  # noqa: E402
from reasoning.reasoning_models import ResponseMode, SourceType  # noqa: E402
from tools.base import ToolResult  # noqa: E402


def task_for(
    question: str,
    *,
    evidence_requirement: str = "unnecessary",
    comparative: bool = False,
    criterion: str = "",
    criterion_proxy: str = "",
    current_information_required: bool = False,
    semantic_reading: dict | None = None,
) -> Task:
    """Build a Task with semantic-layer fields populated."""

    task = Task(goal=question, original_prompt=question)
    task.evidence_requirement = evidence_requirement
    task.comparative = comparative
    task.criterion = criterion
    task.criterion_proxy = criterion_proxy
    task.current_information_required = current_information_required
    if semantic_reading is not None:
        task.semantic_reading = semantic_reading
    elif evidence_requirement != "unnecessary" or comparative:
        task.semantic_reading = {
            "goal": question,
            "subject": question,
            "evidence_requirement": evidence_requirement,
            "comparative": comparative,
            "criterion": criterion,
            "criterion_proxy": criterion_proxy,
            "freshness_requirement": "current" if current_information_required else "any",
        }
    return task


def _empty_web_runner():
    """A tool runner that returns no web results (simulates retrieval failure)."""

    class EmptyWebRunner:
        def __init__(self):
            self.calls = []
            self.web_calls = []

        def __call__(self, name, parameters):
            self.calls.append((name, dict(parameters)))
            if name.startswith("web."):
                self.web_calls.append(name)
            if name == "web.search":
                return ToolResult(
                    success=True, status="completed", output={"provider": "test", "results": []}
                )
            if name == "web.fetch":
                return None
            return ToolResult.failure(f"no fake output for {name}", recoverable=True)

    return EmptyWebRunner()


def _recording_runner():
    """A tool runner that returns one web search result per call."""

    class RecordingRunner:
        def __init__(self):
            self.calls = []
            self.web_calls = []

        def __call__(self, name, parameters):
            self.calls.append((name, dict(parameters)))
            if name.startswith("web."):
                self.web_calls.append(name)
            if name == "web.search":
                return ToolResult(
                    success=True,
                    status="completed",
                    output={
                        "provider": "test",
                        "results": [
                            {"title": "Page", "url": "https://example.com", "snippet": "content"}
                        ],
                    },
                )
            if name == "web.fetch":
                return None
            return ToolResult.failure(f"no fake output for {name}", recoverable=True)

    return RecordingRunner()


class FakeAsk:
    """Canned model stub that records how often it was consulted."""

    def __init__(self, answer: str = "A model answer.") -> None:
        self.answer = answer
        self.calls = 0

    def __call__(self, *, system_prompt: str = "", user_prompt: str = "", **_kw) -> str:
        self.calls += 1
        return self.answer


def _build_engine(ask, tool_runner, retrieval=None):
    """Build a ReasoningEngine like the test suite helper does."""
    from computer.runtime import register_read_only_tools
    from config import COMPUTER_ROOT
    from reasoning.answer_generator import AnswerGenerator
    from reasoning.query_router import QueryRouter  # noqa: F401
    from reasoning.reasoning_engine import ReasoningEngine
    from reasoning.self_introspection import SelfIntrospection
    from tools.capabilities import CapabilityRegistry
    from tools.registry import ToolRegistry
    from knowledge_search import retrieve

    registry = ToolRegistry()
    register_read_only_tools(registry, root=COMPUTER_ROOT, ask=ask)
    capabilities = CapabilityRegistry(registry)
    kwargs = {}
    if retrieval is not None:
        kwargs["retrieve"] = retrieval
    return ReasoningEngine(
        self_introspection=SelfIntrospection(capabilities, model_name="test-model"),
        answer_generator=AnswerGenerator(ask=ask),
        tool_runner=tool_runner,
        capabilities=capabilities,
        **kwargs,
    )


class EvidenceGateUnitTests(unittest.TestCase):
    """Direct tests of EvidenceGate.evaluate() logic."""

    def test_evidence_required_with_no_evidence_blocks_answer(self):
        gate = EvidenceGate()
        task = task_for("Who is the current president?", evidence_requirement="required")
        evidence = EvidenceManager()
        result = gate.evaluate(task=task, evidence_manager=evidence)
        self.assertFalse(result.can_answer)
        self.assertEqual(result.status, "insufficient")

    def test_evidence_required_with_evidence_allows_answer(self):
        gate = EvidenceGate()
        task = task_for("Who is the current president?", evidence_requirement="required")
        evidence = EvidenceManager()
        evidence.add_web_results(
            {"results": [{"title": "President", "url": "https://example.com", "snippet": "Joe Biden"}], "provider": "test"}
        )
        result = gate.evaluate(task=task, evidence_manager=evidence)
        self.assertTrue(result.can_answer)
        self.assertEqual(result.status, "sufficient")

    def test_evidence_preferred_with_no_evidence_allows_answer_with_disclosure(self):
        gate = EvidenceGate()
        task = task_for("Who is Elon Musk?", evidence_requirement="preferred")
        evidence = EvidenceManager()
        result = gate.evaluate(task=task, evidence_manager=evidence)
        self.assertTrue(result.can_answer)
        self.assertTrue(result.must_disclose)
        self.assertEqual(result.status, "sufficient")

    def test_evidence_unnecessary_with_no_evidence_allows_answer(self):
        gate = EvidenceGate()
        task = task_for("What is Python?", evidence_requirement="unnecessary")
        evidence = EvidenceManager()
        result = gate.evaluate(task=task, evidence_manager=evidence)
        self.assertTrue(result.can_answer)
        self.assertFalse(result.must_disclose)

    def test_comparative_subjective_with_insufficient_evidence_blocks(self):
        gate = EvidenceGate()
        task = task_for(
            "Who is the most famous Minecraft YouTuber?",
            evidence_requirement="required",
            comparative=True,
            criterion="most famous",
        )
        evidence = EvidenceManager()
        evidence.add_web_results(
            {"results": [{"title": "A", "url": "https://a.com", "snippet": "a"}], "provider": "test"}
        )
        result = gate.evaluate(task=task, evidence_manager=evidence)
        self.assertFalse(result.can_answer)
        self.assertEqual(result.status, "ambiguous")

    def test_comparative_with_proxy_and_sufficient_evidence_allows(self):
        gate = EvidenceGate()
        task = task_for(
            "Who is the most famous Minecraft YouTuber?",
            evidence_requirement="required",
            comparative=True,
            criterion="most famous",
            criterion_proxy="subscriber count",
        )
        evidence = EvidenceManager()
        evidence.add_web_results(
            {"results": [
                {"title": "A", "url": "https://a.com", "snippet": "a"},
                {"title": "B", "url": "https://b.com", "snippet": "b"},
            ], "provider": "test"}
        )
        result = gate.evaluate(task=task, evidence_manager=evidence)
        self.assertTrue(result.can_answer)
        self.assertEqual(result.status, "sufficient")


class ReasoningEngineGateTests(unittest.TestCase):
    """Integration tests: the gate is wired into the reasoning engine."""

    def test_required_evidence_with_no_retrieval_does_not_fabricate(self):
        ask = FakeAsk("I would invent a name here.")
        runner = _empty_web_runner()
        engine = _build_engine(ask, runner)
        task = task_for(
            "Who is the current Mrs. America 2024?",
            evidence_requirement="required",
            current_information_required=True,
        )
        answer = engine.handle_request(question=task.goal, task=task)
        self.assertIsNotNone(answer)
        self.assertNotIn("I would invent", answer.text)
        self.assertIn(answer.mode, {ResponseMode.CLARIFICATION, ResponseMode.LIMITATION})
        self.assertEqual(ask.calls, 0)

    def test_required_evidence_with_retrieved_web_answer_proceeds(self):
        from tests.test_reasoning_behavior import make_retrieve

        ask = FakeAsk("The latest Python is 3.14.")
        runner = _recording_runner()
        engine = _build_engine(ask, runner, retrieval=make_retrieve())
        task = task_for(
            "What is the latest Python version?",
            evidence_requirement="required",
            current_information_required=True,
        )
        answer = engine.handle_request(question=task.goal, task=task)
        self.assertIsNotNone(answer)
        self.assertIn(SourceType.WEB, answer.provenance)

    def test_comparative_required_evidence_blocks_without_web(self):
        ask = FakeAsk("The best one is X.")
        runner = _empty_web_runner()
        engine = _build_engine(ask, runner)
        task = task_for(
            "Best programming language",
            evidence_requirement="required",
            comparative=True,
            criterion="best",
        )
        answer = engine.handle_request(question=task.goal, task=task)
        self.assertIsNotNone(answer)
        self.assertNotIn(SourceType.MODEL, answer.provenance)
        self.assertIn(answer.mode, {ResponseMode.CLARIFICATION, ResponseMode.LIMITATION})

    def test_preferred_evidence_with_no_retrieval_falls_back_to_model(self):
        ask = FakeAsk("Known entity from model knowledge.")
        runner = _empty_web_runner()
        engine = _build_engine(ask, runner)
        task = task_for(
            "Who is Elon Musk?",
            evidence_requirement="preferred",
        )
        answer = engine.handle_request(question=task.goal, task=task)
        self.assertIsNotNone(answer)
        self.assertIn(SourceType.MODEL, answer.provenance)


class EvidenceManagerGateTests(unittest.TestCase):
    """Tests for the usability_count and has_usable fixes."""

    def test_usable_count_counts_non_model_items(self):
        evidence = EvidenceManager()
        evidence.add_model_note("some note", question="test")
        self.assertEqual(evidence.usable_count(), 0)
        evidence.add_web_results(
            {"results": [{"title": "Page", "url": "https://example.com", "snippet": "content"}], "provider": "test"}
        )
        self.assertEqual(evidence.usable_count(), 1)

    def test_has_usable_excludes_model_notes(self):
        evidence = EvidenceManager()
        evidence.add_model_note("just a note", question="test")
        self.assertFalse(evidence.has_usable())
        evidence.add_web_results(
            {"results": [{"title": "Page", "url": "https://example.com", "snippet": "content"}], "provider": "test"}
        )
        self.assertTrue(evidence.has_usable())

    def test_has_usable_dead_code_removed(self):
        """has_usable must not have dead code after the return statement."""
        import inspect
        from reasoning.evidence_manager import EvidenceManager as EM

        source = inspect.getsource(EM.has_usable)
        lines = source.split("\n")
        # The method should have exactly one return statement
        returns = [line.strip() for line in lines if line.strip().startswith("return")]
        self.assertEqual(len(returns), 1)


if __name__ == "__main__":
    unittest.main()
