"""The semantic reasoning layer: understanding -> requirements -> capabilities.

This is the single component the Brain calls, in place of ad-hoc intent routing,
to transform a user request into an explicit, open-ended goal representation and
the capabilities needed to satisfy it:

    USER MESSAGE
      -> SEMANTIC UNDERSTANDING      (SemanticUnderstanding)
      -> GOAL REPRESENTATION         (SemanticRequest)
      -> REQUIREMENTS / REASONING    (EvidencePolicy.decide)
      -> CAPABILITY SELECTION        (CapabilityPlanner.plan)

It never plans or executes. It writes its reading and its validated capability
requirements back onto the existing :class:`~models_task.Task` IR, so the
interpreter, validator, planner, executor, verification, and recovery layers are
all reused unchanged.

Two architectural invariants are enforced here:

* **Understanding precedes routing.** This layer runs *before*
  :class:`~reasoning.task_validator.TaskValidator` and the planner, and it may
  change the task's evidence requirement and capabilities - but it can never
  reach into the executor or the permission layer.
* **No keyword intent routing.** It reads the :class:`SemanticRequest` structure
  (operation, comparative, criterion, freshness, evidence requirement), never
  ``if "search" in message``.

The emitted reasoning trace is structured diagnostics, never hidden
chain-of-thought, and it is designed to let a developer tell apart: did Atlas
misunderstand, choose the wrong capability, receive bad evidence, fail to
evaluate evidence, or hallucinate at generation time?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from models_task import Task
from reasoning.capability_planner import CapabilityPlan, CapabilityPlanner, requirements_to_actions
from reasoning.evidence_policy import (
    Answerability,
    EvidenceDecision,
    EvidencePolicy,
)
from reasoning.semantic_request import SemanticRequest
from reasoning.semantic_understanding import SemanticUnderstanding

logger = logging.getLogger(__name__)

#: Read-only research capabilities that mean "Atlas consulted external evidence".
_EVIDENCE_CAPABILITIES = frozenset(
    {
        "web.search", "web.research", "web.fetch",
        "filesystem.search", "filesystem.read", "filesystem.list",
        "filesystem.search_content", "filesystem.metadata",
        "system.info", "processes.list",
    }
)


@dataclass
class SemanticDecision:
    """Everything the semantic layer decided for one request."""

    reading: SemanticRequest = field(default_factory=SemanticRequest)
    evidence: EvidenceDecision = field(default_factory=EvidenceDecision)
    capabilities: CapabilityPlan = field(default_factory=CapabilityPlan)
    #: True when the reading added a capability the interpreter had not planned.
    _added_capabilities: list[str] = field(default_factory=list)

    @property
    def added_capabilities(self) -> list[str]:
        return self._added_capabilities

    @property
    def requires_retrieval(self) -> bool:
        """True when the evidence decision says Atlas must retrieve first."""

        return self.evidence.requires_retrieval

    def to_dict(self) -> dict[str, Any]:
        return {
            "reading": self.reading.to_dict(),
            "evidence": self.evidence.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "added_capabilities": list(self._added_capabilities),
        }

    def trace(self) -> dict[str, Any]:
        """Return the compact, structured diagnostic trace for developers.

        This is the shape the spec asks for: it exposes the goal, subject,
        operation, criterion, freshness, evidence requirement, selected
        capabilities, ambiguity, resolution strategy, and evidence status - and
        nothing that resembles private chain-of-thought.
        """

        reading = self.reading
        return {
            "goal": reading.goal,
            "subject": reading.subject,
            "operation": reading.operation,
            "criteria": reading.criterion,
            "criterion_proxy": reading.criterion_proxy,
            "freshness": reading.freshness_requirement,
            "evidence_required": reading.evidence_requirement != "unnecessary",
            "evidence_requirement": reading.evidence_requirement,
            "selected_capabilities": self.capabilities.capabilities,
            "unavailable_capabilities": self.capabilities.unavailable,
            "ambiguity": reading.ambiguity,
            "resolution_strategy": self.evidence.resolution_strategy,
            "comparative": reading.comparative,
            "candidate_set": reading.candidate_set,
            "contextual": reading.contextual,
            "local": reading.local,
            "confidence": round(float(reading.confidence), 3),
            "notes": list(reading.notes),
        }


class SemanticReasoning:
    """Run semantic understanding, evidence reasoning, and capability selection.

    ``ask`` is the optional model caller; when it is ``None`` the layer is fully
    deterministic and offline. ``capabilities`` is the live capability registry
    used to validate every proposed capability.
    """

    def __init__(self, *, ask: Any = None, capabilities: Any = None) -> None:
        self._understanding = SemanticUnderstanding(ask=ask)
        self._evidence = EvidencePolicy()
        self._capabilities = CapabilityPlanner(capabilities)

    # -- public API -------------------------------------------------------------

    def reason(
        self,
        text: str,
        *,
        task: Task,
        prior_task: Task | None = None,
        history: str = "",
    ) -> SemanticDecision:
        """Understand ``text`` and augment ``task`` with the resulting decision."""

        reading = self._understanding.understand(
            text, task=task, history=history, prior_task=prior_task
        )
        # The temporal authority resolves the request's time window once. The
        # evidence policy composes the retrieval query from it as meaning
        # (subject + relation + period), so query generation and freshness read the
        # same resolution instead of each re-deriving time from the sentence.
        temporal_period = ""
        try:
            from reasoning.temporal_resolution import TemporalResolver

            temporal_period = TemporalResolver().resolve_from_structure(
                text,
                final_event_result=reading.final_event_result,
                freshness=reading.freshness_requirement,
                superlative=reading.comparative,
            ).resolved_period
        except Exception:  # noqa: BLE001 - resolution is best-effort, never fatal
            logger.exception("Temporal resolution failed during semantic reasoning")
        evidence = self._evidence.decide(reading, question=text, temporal_period=temporal_period)
        capabilities = self._capabilities.plan(reading, evidence, task=task)

        # Reconcile: the evidence decision is authoritative, so the reading and
        # the task agree on what is required. A stronger requirement from the
        # semantic pass (e.g. a ranking that must be evidenced) upgrades the
        # reading the model may have under-called.
        if _strength(evidence.requirement) > _strength(reading.evidence_requirement):
            reading.evidence_requirement = evidence.requirement

        decision = SemanticDecision(reading=reading, evidence=evidence, capabilities=capabilities)
        decision._added_capabilities = self._apply(task, decision)
        return decision

    def evaluate_answerability(
        self,
        decision: SemanticDecision,
        *,
        evidence_count: int = 0,
        evidence_conflicting: bool = False,
        retrieved_ok: bool = True,
    ) -> Answerability:
        """Run the answerability gate for an already-computed decision."""

        return self._evidence.evaluate(
            decision.reading,
            decision.evidence,
            evidence_count=evidence_count,
            evidence_conflicting=evidence_conflicting,
            retrieved_ok=retrieved_ok,
        )

    # -- write-back -------------------------------------------------------------

    def _apply(self, task: Task, decision: SemanticDecision) -> list[str]:
        """Write the semantic decision back onto the Task IR. Returns additions."""

        reading = decision.reading
        task.semantic_reading = reading.to_dict()
        task.evidence_requirement = reading.evidence_requirement
        task.comparative = reading.comparative
        task.criterion = reading.criterion
        task.criterion_proxy = reading.criterion_proxy
        task.semantic_capabilities = list(decision.capabilities.capabilities)

        # Refresh the freshness flag so the router/reasoning engine consult one
        # decision rather than re-deriving it from keyword lists.
        if reading.freshness_requirement == "current":
            task.current_information_required = True
        if reading.local:
            task.entities.setdefault("local_scope", True)
        if reading.criterion and reading.criterion_proxy:
            task.entities.setdefault("criterion", reading.criterion)
            task.entities.setdefault("criterion_proxy", reading.criterion_proxy)

        before = {action.capability for action in task.actions}
        task.actions = requirements_to_actions(
            decision.capabilities, task_actions=list(task.actions), existing_capabilities=before
        )
        added = [action.capability for action in task.actions if action.capability not in before]
        if added:
            reading.notes.append(
                "semantic layer added capability requirement(s): " + ", ".join(added)
            )
        return added

    # -- answerability helpers --------------------------------------------------

    @staticmethod
    def evidence_capabilities(task: Task) -> list[str]:
        """Return the evidence-gathering capabilities a task's actions will use."""

        return [
            action.capability
            for action in task.actions
            if action.capability in _EVIDENCE_CAPABILITIES
        ]


def _strength(requirement: str) -> int:
    return {"unnecessary": 0, "preferred": 1, "required": 2}.get(requirement, 0)
