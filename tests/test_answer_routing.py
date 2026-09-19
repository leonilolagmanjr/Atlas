"""Answer-generator routing tests: informational vs artifact web research.

An informational web-research answer must go through the evidence-grounded
model prompt (which names the question and cites sources), while an artifact
retrieval keeps the deterministic as-is document synthesis. The two paths were
previously conflated because ``task.evidence_state`` is always present.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models_task import EvidenceSource, EvidenceState, Task  # noqa: E402
from reasoning.answer_generator import AnswerGenerator  # noqa: E402
from reasoning.evidence_manager import EvidenceManager  # noqa: E402
from reasoning.reasoning_models import ResponseMode  # noqa: E402


class _RecordingAsk:
    def __init__(self, reply: str = "Grounded model answer.") -> None:
        self.reply = reply
        self.calls = 0
        self.last_prompt = ""

    def __call__(self, *, system_prompt="", user_prompt="", **_kw) -> str:
        self.calls += 1
        self.last_prompt = user_prompt
        return self.reply


def _evidence_with_web() -> EvidenceManager:
    evidence = EvidenceManager()
    evidence.add_web_results(
        {"results": [{"title": "RTX 5090 Review", "url": "https://example.com/a", "snippet": "fast"}]}
    )
    evidence.add_web_page({"url": "https://example.com/a", "title": "RTX 5090 Review", "text": "Benchmarks show a large gain."})
    return evidence


def _state(must_be_artifact: bool) -> EvidenceState:
    state = EvidenceState(target="rtx 5090", goal="find_information", must_be_artifact=must_be_artifact)
    state.add_source(
        EvidenceSource(
            source_id="s1",
            url="https://example.com/a",
            title="RTX 5090 Review",
            content_type="article",
            content="Benchmarks show a large gain over the previous generation.",
            relevance_score=0.9,
            quality_score=0.8,
            completeness=0.7,
        )
    )
    return state


class WebAnswerRoutingTests(unittest.TestCase):
    def test_informational_query_uses_the_model_grounded_prompt(self):
        ask = _RecordingAsk()
        generator = AnswerGenerator(ask=ask)
        task = Task(goal="find_information", original_prompt="search for rtx 5090 benchmarks")
        task.evidence_state = _state(must_be_artifact=False)

        answer = generator.grounded(
            "search for rtx 5090 benchmarks",
            _evidence_with_web(),
            mode=ResponseMode.WEB_RESEARCH,
            task=task,
        )

        # The model was consulted, and the prompt carried the user's question.
        self.assertEqual(ask.calls, 1)
        self.assertIn("rtx 5090 benchmarks", ask.last_prompt)
        self.assertTrue(answer.used_model)

    def test_artifact_request_keeps_deterministic_synthesis(self):
        ask = _RecordingAsk()
        generator = AnswerGenerator(ask=ask)
        task = Task(goal="retrieve_document", original_prompt="get the bee movie script")
        task.evidence_state = _state(must_be_artifact=True)

        answer = generator.grounded(
            "get the bee movie script",
            _evidence_with_web(),
            mode=ResponseMode.WEB_RESEARCH,
            task=task,
        )

        # The artifact path is deterministic and does not spend a model call.
        self.assertEqual(ask.calls, 0)
        self.assertFalse(answer.used_model)


if __name__ == "__main__":
    unittest.main()
