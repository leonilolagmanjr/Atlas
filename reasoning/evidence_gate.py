"""Evidence gate: the deterministic answerability checkpoint.

The gate runs *after* evidence is gathered but *before* a response is generated.
It asks a single question the spec requires between retrieval and generation:
does Atlas have enough to answer the *actual* request without fabricating?

The gate is the boundary between "I looked" and "I can say so". When it says
the request cannot be answered (``can_answer=False``), the engine must not emit a
direct/generated answer -- it surfaces a limitation or asks for clarification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from reasoning.evidence_policy import (
    AMBIGUOUS,
    CONFLICTING,
    INSUFFICIENT,
    SUFFICIENT,
    UNKNOWN,
    Answerability,
    EvidenceAssessment,
    EvidenceDecision,
    EvidencePolicy,
    STATUS_ACTIONS,
)
from reasoning.semantic_request import (
    EVIDENCE_PREFERRED,
    EVIDENCE_REQUIRED,
    EVIDENCE_UNNECESSARY,
    FRESHNESS_CURRENT,
    SemanticRequest,
)

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Outcome of running the evidence gate for one request."""

    answerability: Answerability
    usable_count: int
    evidence_conflicting: bool
    retrieved_ok: bool
    decision: EvidenceDecision | None = None
    reading: SemanticRequest | None = None
    assessment: EvidenceAssessment | None = None

    @property
    def can_answer(self) -> bool:
        return self.answerability.can_answer

    @property
    def must_disclose(self) -> bool:
        return self.answerability.must_disclose_uncertainty

    @property
    def status(self) -> str:
        return self.answerability.status

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "action": self.answerability.action,
            "can_answer": self.can_answer,
            "must_disclose": self.must_disclose,
            "usable_count": self.usable_count,
            "evidence_conflicting": self.evidence_conflicting,
            "retrieved_ok": self.retrieved_ok,
            "assessment": self.assessment.to_dict() if self.assessment is not None else None,
            "reasons": list(self.answerability.reasons),
        }


class EvidenceGate:
    """Check whether an evidence collection satisfies the policy for one request.

    The gate never retrieves evidence itself -- it consumes whatever the
    :class:`EvidenceManager` already collected and the semantic *reading* that
    describes the request.
    """

    def __init__(self, policy: EvidencePolicy | None = None) -> None:
        self._policy = policy or EvidencePolicy()

    def evaluate(
        self,
        *,
        task: Any,
        evidence_manager: Any,
    ) -> GateResult:
        """Run the answerability gate against collected evidence.

        Parameters
        ----------
        task
            The :class:`Task` carrying the semantic reading on ``semantic_reading``
            (a dict) and ``evidence_requirement`` / ``comparative`` attributes.
        evidence_manager
            An :class:`EvidenceManager` with evidence already gathered.
        """

        reading = _reconstruct_reading(task)
        decision = self._policy.decide(reading, question=getattr(task, "goal", "") or "")
        usable = evidence_manager.usable_count() if hasattr(evidence_manager, "usable_count") else 0
        conflict = bool(getattr(task, "evidence_conflicting", False) or getattr(evidence_manager, "evidence_conflicting", False))
        retrieved_ok = bool(getattr(task, "retrieval_failed", False) is False)
        # If any retrieved (non-model) evidence was collected, retrieval itself
        # was attempted and did not outright fail, even if it came up short.
        if usable > 0:
            retrieved_ok = True

        assessment = self._policy.assess(
            reading,
            evidence_count=usable,
            evidence_conflicting=conflict,
            retrieved_ok=retrieved_ok,
            evidence_items=getattr(evidence_manager, "ranked", lambda: [])(),
        )
        answerability = self._policy.evaluate(
            reading,
            decision,
            evidence_count=usable,
            evidence_conflicting=conflict,
            retrieved_ok=retrieved_ok,
        )

        if not answerability.can_answer:
            logger.info(
                "evidence gate blocked answer: status=%s reason=%s",
                answerability.status,
                answerability.reasons,
            )

        return GateResult(
            answerability=answerability,
            usable_count=usable,
            evidence_conflicting=conflict,
            retrieved_ok=retrieved_ok,
            decision=decision,
            reading=reading,
            assessment=assessment,
        )

    def can_answer_from(
        self,
        *,
        task: Any,
        evidence_manager: Any,
    ) -> bool:
        """Convenience: True when the gate says Atlas may proceed to a response."""

        return self.evaluate(task=task, evidence_manager=evidence_manager).can_answer


def _reconstruct_reading(task: Any) -> SemanticRequest:
    """Rebuild a :class:`SemanticRequest` from task IR fields.

    The semantic layer writes the reading onto the task as a dict
    (``semantic_reading``) and the evidence decision's key fields onto flat
    attributes. The reasoning engine must reconstruct a reading object so the
    deterministic policy can re-evaluate answerability after retrieval.
    """

    raw = getattr(task, "semantic_reading", None) or {}
    if isinstance(raw, dict) and raw:
        return SemanticRequest.from_mapping(raw)
    # Fallbacks from flat task attributes (keeps the gate usable when only the
    # IR-level fields were set, e.g. by older interpreters).
    return SemanticRequest(
        goal=getattr(task, "goal", "") or "",
        evidence_requirement=getattr(task, "evidence_requirement", EVIDENCE_UNNECESSARY),
        comparative=getattr(task, "comparative", False),
        criterion=getattr(task, "criterion", ""),
        criterion_proxy=getattr(task, "criterion_proxy", ""),
        freshness_requirement=(
            FRESHNESS_CURRENT if getattr(task, "current_information_required", False)
            else "any"
        ),
    )
