"""Capability planning: SemanticRequest + EvidenceDecision -> capability requirements.

Tools are **capabilities available to Atlas**, not intents. This module answers
exactly one question:

    Given what the user is trying to accomplish, which capabilities are useful?

It does not ask "which intent corresponds to this tool?". A single request may
select several capabilities, and the selection emerges from the *requirements* of
the goal - for example a ranking request needs candidate discovery, evidence, and
comparison, which maps onto a research/search capability.

Design rules:

* **The model proposes, Atlas disposes.** A model may name candidate
  capabilities; every one is validated against the live
  :class:`~tools.capabilities.CapabilityRegistry` before it becomes a
  requirement. A capability Atlas does not have is dropped, never invented.
* **Deterministic fallback is structural.** When no model capability list is
  available, requirements are inferred from the reading's operation, evidence
  requirement, and the interpreter's already-planned actions - not from a
  keyword lookup.
* **Nothing here executes.** The output is a list of capability *requirements*
  for the task validator and planner; permissions, schemas, timeouts,
  cancellation, verification, and recovery remain the executor's job.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from models_task import Task, TaskAction
from reasoning.evidence_policy import EvidenceDecision
from reasoning.semantic_request import (
    EVIDENCE_REQUIRED,
    FRESHNESS_CURRENT,
    OP_ACT,
    OP_COMPARE,
    OP_CREATE,
    OP_EXPLAIN,
    OP_IDENTIFY,
    OP_RANK,
    OP_RETRIEVE,
    OP_TRANSFORM,
    OP_UNKNOWN,
    SemanticRequest,
)

logger = logging.getLogger(__name__)


#: The capability that satisfies "go find out" for each domain. These are *sets of
#: candidate capabilities per need*, resolved against the live registry, not
#: intents: a need may be satisfied by more than one capability and Atlas picks
#: the strongest available one.
_RESEARCH_CANDIDATES: tuple[str, ...] = ("web.research", "web.search")
_WEB_SEARCH_CANDIDATES: tuple[str, ...] = ("web.search", "web.research")
_LOCAL_KNOWLEDGE_CAPABILITY = "knowledge_search"
_FILE_SEARCH_CANDIDATES: tuple[str, ...] = (
    "filesystem.search", "filesystem.search_content", "filesystem.list",
)

#: Capabilities that mutate local state. Used only to *describe* a requirement's
#: risk; the validator re-derives the real confirmation flags from the registry.
_MUTATING_PREFIXES: tuple[str, ...] = (
    "filesystem.write", "filesystem.create_folder", "filesystem.move",
    "filesystem.copy", "applications.write_text", "applications.launch",
    "powershell.execute",
)


@dataclass
class CapabilityRequirement:
    """One capability Atlas judged useful, with why and with what arguments."""

    capability: str
    #: The need this capability satisfies ("external_evidence", "comparison", ...).
    need: str
    #: Semantic parameters the requirement proposes (validated downstream).
    parameters: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "need": self.need,
            "parameters": dict(self.parameters),
            "reason": self.reason,
        }


@dataclass
class CapabilityPlan:
    """The selected capabilities for one request, in the order they should run."""

    requirements: list[CapabilityRequirement] = field(default_factory=list)
    #: Capabilities that were dropped because the runtime does not have them.
    unavailable: list[str] = field(default_factory=list)
    #: Capabilities the model proposed that Atlas confirmed.
    model_proposed: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def capabilities(self) -> list[str]:
        return [requirement.capability for requirement in self.requirements]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_capabilities": self.capabilities,
            "requirements": [requirement.to_dict() for requirement in self.requirements],
            "unavailable": list(self.unavailable),
            "model_proposed": list(self.model_proposed),
            "reasons": list(self.reasons),
        }


class CapabilityPlanner:
    """Turn a semantic reading + evidence decision into capability requirements.

    ``capabilities`` is the live :class:`~tools.capabilities.CapabilityRegistry`.
    When a requirement names a capability the runtime does not have, the planner
    substitutes the strongest available alternative in the same family, and
    records the substitution - it never plans against a tool Atlas cannot run.
    """

    def __init__(self, capabilities: Any = None) -> None:
        self._capabilities = capabilities

    # -- public API -------------------------------------------------------------

    def plan(
        self,
        reading: SemanticRequest,
        decision: EvidenceDecision,
        *,
        task: Task | None = None,
        model_capabilities: list[str] | None = None,
    ) -> CapabilityPlan:
        """Return the ordered capability requirements for ``reading``."""

        plan = CapabilityPlan()

        # 1. Capabilities the *model* proposed, validated against the registry.
        for name in model_capabilities or []:
            capability = str(name or "").strip()
            if not capability:
                continue
            if self._available(capability):
                plan.model_proposed.append(capability)
            else:
                plan.unavailable.append(capability)
        if plan.unavailable:
            plan.reasons.append(
                "dropped capabilities the runtime does not have: "
                + ", ".join(sorted(set(plan.unavailable)))
            )

        # 2. Structural requirements derived from the reading.
        structural = self._structural_requirements(reading, decision, task=task)
        plan.requirements.extend(structural)

        # 3. Model-proposed capabilities that are genuinely useful are appended
        #    (after the structural ones) so an explicit, validated suggestion is
        #    honoured without displacing the requirement the goal itself implies.
        for capability in plan.model_proposed:
            if capability not in plan.capabilities:
                plan.requirements.append(
                    CapabilityRequirement(
                        capability=capability,
                        need="model_suggested",
                        parameters=_parameters_for(capability, reading, decision),
                        reason="the interpreter suggested this capability",
                    )
                )

        plan.reasons.extend(decision.reasons)
        return plan

    # -- structural inference ---------------------------------------------------

    def _structural_requirements(
        self,
        reading: SemanticRequest,
        decision: EvidenceDecision,
        *,
        task: Task | None,
    ) -> list[CapabilityRequirement]:
        """Infer requirements from the goal's shape, never from keywords."""

        requirements: list[CapabilityRequirement] = []
        needs_external = decision.requirement == EVIDENCE_REQUIRED or (
            decision.requirement == "preferred" and decision.freshness == FRESHNESS_CURRENT
        )

        # A ranking/comparison: discover candidates, gather evidence, compare.
        if decision.needs_comparison:
            capability = self._best_available(_RESEARCH_CANDIDATES)
            if capability:
                requirements.append(
                    CapabilityRequirement(
                        capability=capability,
                        need="candidate_discovery_and_evidence",
                        parameters=_parameters_for(capability, reading, decision),
                        reason=(
                            "a ranking requires candidate discovery, evidence, and "
                            "comparison; the candidates must be retrieved before they "
                            "can be compared"
                        ),
                    )
                )
            return requirements

        # A current-information question: retrieve before answering.
        if reading.freshness_requirement == FRESHNESS_CURRENT or decision.requirement == EVIDENCE_REQUIRED:
            capability = self._best_available(_WEB_SEARCH_CANDIDATES)
            if capability:
                requirements.append(
                    CapabilityRequirement(
                        capability=capability,
                        need="current_information",
                        parameters=_parameters_for(capability, reading, decision),
                        reason="the answer may depend on current/public information",
                    )
                )
            return requirements

        # A local-machine request: observe, never search the web.
        if reading.local:
            if self._available("system.info"):
                requirements.append(
                    CapabilityRequirement(
                        capability="system.info",
                        need="local_observation",
                        parameters={},
                        reason="the request is about this machine",
                    )
                )
            return requirements

        # An action the interpreter already planned: keep its actions and only add
        # an evidence step when the action genuinely needs facts from outside.
        if reading.operation == OP_ACT or reading.operation == OP_CREATE:
            if decision.requirement == EVIDENCE_REQUIRED and task is not None:
                planned = {action.capability for action in task.actions}
                if not (planned & set(_WEB_SEARCH_CANDIDATES) | set(_RESEARCH_CANDIDATES)):
                    capability = self._best_available(_WEB_SEARCH_CANDIDATES)
                    if capability:
                        requirements.append(
                            CapabilityRequirement(
                                capability=capability,
                                need="facts_for_action",
                                parameters=_parameters_for(capability, reading, decision),
                                reason="the action needs current facts to complete",
                            )
                        )
            return requirements

        # A transformation of content Atlas already has: no new retrieval.
        if reading.operation == OP_TRANSFORM:
            return requirements

        # A retrieval/explanation about an external entity: prefer grounded
        # evidence when a research capability exists; otherwise fall back to the
        # model, which the answerability gate marks as unverified.
        if reading.operation in {OP_EXPLAIN, OP_IDENTIFY, OP_RETRIEVE, OP_UNKNOWN}:
            if decision.requirement == EVIDENCE_REQUIRED:
                capability = self._best_available(_WEB_SEARCH_CANDIDATES)
                if capability:
                    requirements.append(
                        CapabilityRequirement(
                            capability=capability,
                            need="external_evidence",
                            parameters=_parameters_for(capability, reading, decision),
                            reason="external evidence is required to answer reliably",
                        )
                    )
            # `preferred` evidence is *recorded* but does not force a retrieval
            # step: the reasoning engine consults web evidence before the model
            # when it is available, and answers from the model when it is not.
            return requirements

        return requirements

    # -- helpers ----------------------------------------------------------------

    def _best_available(self, candidates: tuple[str, ...]) -> str | None:
        for name in candidates:
            if self._available(name):
                return name
        return None

    def _available(self, capability: str) -> bool:
        if self._capabilities is None:
            return True
        exists = getattr(self._capabilities, "exists", None)
        if callable(exists):
            try:
                return bool(exists(capability))
            except Exception:  # pragma: no cover - a registry probe must not fail
                return False
        return True


def _parameters_for(
    capability: str,
    reading: SemanticRequest,
    decision: EvidenceDecision,
) -> dict[str, Any]:
    """Build validated-shape parameters for a research/search capability."""

    query = (decision.query or reading.subject or "").strip()
    parameters: dict[str, Any] = {}
    if capability in {"web.search", "web.research", "knowledge_search", "filesystem.search_content"}:
        if query:
            parameters["query"] = query
    if capability == "web.research":
        # A ranking wants the artifact of a ranking page, not a document title.
        parameters["goal"] = "find_information"
        parameters["must_be_artifact"] = False
        if reading.criterion:
            parameters["target"] = query or reading.criterion
    if capability == "filesystem.search" and query:
        parameters["pattern"] = f"*{query}*"
    return parameters


def requirements_to_actions(
    plan: CapabilityPlan,
    *,
    task_actions: list[TaskAction],
    existing_capabilities: set[str] | None = None,
) -> list[TaskAction]:
    """Turn capability requirements into task actions, without duplicating actions.

    The interpreter's own actions are authoritative for anything it already
    planned (a destination, a content type, an application). This only *adds* a
    retrieval step the semantic reading requires and the interpreter's plan lacks,
    which is exactly the case the Core Problem describes: a request that clearly
    needs external information but produced no retrieval action.

    Idempotent: calling it twice with the same plan adds nothing the second time.
    """

    actions = list(task_actions)
    existing = set(existing_capabilities or (action.capability for action in task_actions))

    appended = 0
    for requirement in plan.requirements:
        capability = requirement.capability
        if capability in existing:
            continue
        # Never add a read-only research step when the plan is purely generative:
        # the content.generate action is the user's goal and adding a search would
        # change its meaning.
        if capability in {"web.search", "web.research"} and "content.generate" in existing:
            # Unless the requirement is explicitly to gather facts for the plan.
            if requirement.need not in {"facts_for_action", "candidate_discovery_and_evidence",
                                        "current_information", "external_evidence"}:
                continue
        appended += 1
        actions.append(
            TaskAction(
                action_id=f"semantic_{appended}",
                capability=capability,
                parameters=dict(requirement.parameters),
                description=requirement.reason or f"{capability} for the request",
                produces="web_content" if capability == "web.research" else "search_results",
                expected_output="retrieved evidence",
                risk_level="read_only",
            )
        )
        existing.add(capability)

    return actions
