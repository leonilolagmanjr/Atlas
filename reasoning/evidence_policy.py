"""Evidence policy: does Atlas need external evidence, and is it sufficient?

This module answers two questions the spec puts between *understanding* and
*capability selection*:

1. **Evidence requirement.** Given a :class:`~reasoning.semantic_request.
   SemanticRequest`, must Atlas retrieve before it may answer? A ranking request
   or a current-information request cannot be answered from model memory without
   the risk of fabrication, so the policy marks it *required*; stable textbook
   knowledge is *unnecessary*; an information request about a real external
   entity is *preferred*.
2. **Answerability gate.** After retrieval, do we have enough to answer the
   *actual* request? The gate returns one of SUFFICIENT / INSUFFICIENT /
   AMBIGUOUS / CONFLICTING / UNKNOWN, with the action each status implies.
   Response generation must never fabricate simply because the pipeline finished.

The decision is semantic (it reads the reading, not a keyword list) combined with
deterministic policy (the thresholds and the status -> action mapping are fixed,
inspectable constants, never a model's judgement).

Nothing here executes tools or mutates state.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from reasoning.semantic_request import (
    AMBIGUITY_HIGH,
    EVIDENCE_PREFERRED,
    EVIDENCE_REQUIRED,
    EVIDENCE_UNNECESSARY,
    FRESHNESS_CURRENT,
    FRESHNESS_STABLE,
    SemanticRequest,
)

logger = logging.getLogger(__name__)

# -- answerability statuses ---------------------------------------------------

SUFFICIENT = "sufficient"
INSUFFICIENT = "insufficient"
AMBIGUOUS = "ambiguous"
CONFLICTING = "conflicting"
UNKNOWN = "unknown"

#: What each status implies for the next stage (spec section 5).
STATUS_ACTIONS: dict[str, str] = {
    SUFFICIENT: "proceed_to_response",
    INSUFFICIENT: "retrieve_more_or_use_another_capability_or_ask",
    AMBIGUOUS: "infer_when_safe_else_ask",
    CONFLICTING: "cross_check_prefer_authoritative_or_acknowledge",
    UNKNOWN: "surface_explicitly_never_fabricate",
}

#: Deterministic evidence-count policy (never a model's opinion).
MIN_SOURCES_FOR_RANKING = 2
MIN_SOURCES_FOR_CURRENT = 1
MIN_SOURCES_FOR_PREFERRED = 1


@dataclass
class EvidenceDecision:
    """The evidence requirement + answerability verdict for one request."""

    requirement: str = EVIDENCE_UNNECESSARY
    freshness: str = "any"
    #: The query Atlas should retrieve with (subject + criterion, not the raw prompt).
    query: str = ""
    #: True when a ranking/comparison must be resolved by comparing candidates.
    needs_comparison: bool = False
    #: The user's criterion and the objective proxy used to settle it, if any.
    criterion: str = ""
    criterion_proxy: str = ""
    #: True when a subjective criterion could not be reduced to a measurement.
    subjective_without_proxy: bool = False
    #: The candidate set the comparison ranges over.
    candidate_set: str = ""
    #: The explicit resolution strategy for the reasoning trace.
    resolution_strategy: str = ""
    #: Structured semantic requirement; used by downstream evidence assessment.
    information_requirement: InformationRequirement | None = None
    #: Short, inspectable reasons (never chain-of-thought).
    reasons: list[str] = field(default_factory=list)

    @property
    def requires_retrieval(self) -> bool:
        return self.requirement in {EVIDENCE_REQUIRED, EVIDENCE_PREFERRED}

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement": self.requirement,
            "freshness": self.freshness,
            "query": self.query,
            "needs_comparison": self.needs_comparison,
            "criterion": self.criterion,
            "criterion_proxy": self.criterion_proxy,
            "subjective_without_proxy": self.subjective_without_proxy,
            "candidate_set": self.candidate_set,
            "resolution_strategy": self.resolution_strategy,
            "information_requirement": None if self.information_requirement is None else {
                "entity": self.information_requirement.entity,
                "property": self.information_requirement.property,
                "state": self.information_requirement.state,
                "criterion": self.information_requirement.criterion,
                "temporal_scope": self.information_requirement.temporal_scope,
            },
            "reasons": list(self.reasons),
        }


@dataclass
class Answerability:
    """The gate verdict: may Atlas answer, or must it do something else first?"""

    status: str = UNKNOWN
    action: str = STATUS_ACTIONS[UNKNOWN]
    #: A question to ask the user when the only correct move is clarification.
    clarification_question: str | None = None
    #: True when Atlas may proceed to generate a response.
    can_answer: bool = False
    #: True when the response must disclose that evidence was not available.
    must_disclose_uncertainty: bool = False
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "action": self.action,
            "clarification_question": self.clarification_question,
            "can_answer": self.can_answer,
            "must_disclose_uncertainty": self.must_disclose_uncertainty,
            "reasons": list(self.reasons),
        }


@dataclass
class InformationRequirement:
    """Structured description of the fact the user asks for.

    Priority 1: this is the shared semantic contract that generalizes across
    paraphrases like 'latest Python version' and 'who is leading the NBA finals'.
    """

    entity: str = ""
    property: str = ""
    state: str = "final"
    criterion: dict[str, Any] = field(default_factory=lambda: {"kind": "objective", "proxy": "", "operator": ""})
    temporal_scope: str = "stable"

    def requires_final_state(self) -> bool:
        return self.state in {"final", "live", "historical"} and self.temporal_scope != "stable"


@dataclass
class EvidenceAssessment:
    """Outcome of evaluating retrieved evidence against a fact requirement."""

    status: str = SUFFICIENT
    kind: str = ""
    retry: str = ""
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "kind": self.kind,
            "retry": self.retry,
            "reasons": list(self.reasons),
        }


class EvidencePolicy:
    """Derive the evidence decision and evaluate answerability."""

    # -- evidence requirement ---------------------------------------------------

    def decide(self, reading: SemanticRequest, *, question: str = "") -> EvidenceDecision:
        """Return the evidence decision for a semantic reading."""

        decision = EvidenceDecision(
            requirement=reading.evidence_requirement,
            freshness=reading.freshness_requirement,
            criterion=reading.criterion,
            criterion_proxy=reading.criterion_proxy,
            candidate_set=reading.candidate_set,
            needs_comparison=reading.comparative or reading.ranking,
            information_requirement=InformationRequirement(
                entity=reading.subject,
                property=reading.criterion or reading.goal,
                state="final" if reading.final_event_result else ("live" if reading.current_knowledge else "stable"),
                criterion={
                    "kind": "subjective" if reading.subjective_criterion else "objective" if reading.objective_criterion else "temporal",
                    "proxy": reading.criterion_proxy,
                    "operator": "latest" if reading.latest_request else "current" if reading.current_knowledge else "",
                },
                temporal_scope="realtime" if reading.current_knowledge else ("historical" if reading.historical_knowledge else "stable"),
            ),
        )
        if reading.current_knowledge or reading.latest_request or reading.most_recent_request or reading.ranking or reading.dynamic_quantity:
            decision.requirement = EVIDENCE_REQUIRED
            decision.freshness = FRESHNESS_CURRENT
        elif reading.stable_knowledge:
            decision.requirement = EVIDENCE_UNNECESSARY
            decision.freshness = FRESHNESS_STABLE
        decision.query = _retrieval_query(reading, question=question)

        # A subjective criterion with no measurable proxy: Atlas must reason about
        # the ambiguity rather than present a generation as an objective answer.
        if reading.comparative and reading.criterion and not reading.criterion_proxy:
            if _is_subjective(reading.criterion):
                decision.subjective_without_proxy = True
                decision.reasons.append(
                    f"criterion '{reading.criterion}' is subjective with no objective proxy"
                )

        if reading.comparative:
            decision.resolution_strategy = (
                f"use_{_slug(reading.criterion_proxy)}_as_proxy"
                if reading.criterion_proxy
                else "ask_user_to_disambiguate_criterion"
            )
            decision.reasons.append(
                "ranking/comparison requires candidate discovery, evidence, and comparison"
            )
        elif reading.freshness_requirement == FRESHNESS_CURRENT:
            decision.resolution_strategy = "retrieve_current_information"
            decision.reasons.append("the answer depends on current information")
        elif decision.requirement == EVIDENCE_PREFERRED:
            decision.resolution_strategy = "ground_in_retrieved_evidence"
            decision.reasons.append("external entity; evidence improves reliability")
        else:
            decision.resolution_strategy = "answer_from_model_knowledge"

        return decision

    # -- answerability gate -----------------------------------------------------

    def assess(
        self,
        reading: SemanticRequest,
        *,
        evidence_count: int = 0,
        evidence_conflicting: bool = False,
        retrieved_ok: bool = True,
        evidence_items: Iterable[Any] | None = None,
    ) -> EvidenceAssessment:
        """Classify retrieved evidence as sufficient / insufficient / conflicting.

        Priority 1: the policy must distinguish live-state and final-state facts from
        stable knowledge, and must refuse the common 'intermediate result' failure mode.
        """

        assessment = EvidenceAssessment(status=SUFFICIENT, reasons=[])
        if reading.comparative or reading.ranking:
            if evidence_count < MIN_SOURCES_FOR_RANKING:
                return EvidenceAssessment(
                    status=INSUFFICIENT,
                    kind="intermediate_state" if reading.final_event_result else "wrong_granularity",
                    retry="re-query_with_completed_result_or_ranked_source",
                    reasons=["ranking requires a completed result or a ranked candidate set"],
                )
            if evidence_conflicting:
                return EvidenceAssessment(
                    status="conflicting",
                    kind="conflicting",
                    retry="cross_check authoritative sources",
                    reasons=["retrieved sources disagree"],
                )
            return assessment

        if reading.subjective_criterion and not reading.criterion_proxy:
            return EvidenceAssessment(
                status=AMBIGUOUS,
                kind="ambiguous_source",
                retry="ask_for_proxy_or_disclose_criterion",
                reasons=["subjective criterion without a measurable objective proxy"],
            )

        if reading.freshness_requirement == FRESHNESS_CURRENT:
            if evidence_count < MIN_SOURCES_FOR_CURRENT:
                return EvidenceAssessment(
                    status=INSUFFICIENT,
                    kind="stale" if not retrieved_ok else "intermediate_state",
                    retry="query_with_freshness_qualifier",
                    reasons=["current fact requires fresh evidence"],
                )
            return assessment

        if reading.latest_request or reading.most_recent_request:
            if evidence_count < MIN_SOURCES_FOR_CURRENT:
                return EvidenceAssessment(
                    status=INSUFFICIENT,
                    kind="stale",
                    retry="query_for_latest_available_final_state",
                    reasons=["latest/most-recent queries need fresh evidence"],
                )
            return assessment

        if evidence_count == 0 and reading.evidence_requirement in {EVIDENCE_REQUIRED, EVIDENCE_PREFERRED}:
            return EvidenceAssessment(
                status=INSUFFICIENT,
                kind="ambiguous_source",
                retry="retrieve_or_ask_for_clarification",
                reasons=["no usable evidence was retrieved"],
            )

        if evidence_items is not None:
            for item in evidence_items:
                if not hasattr(item, "metadata"):
                    continue
                if item.metadata.get("state") == "intermediate":
                    return EvidenceAssessment(
                        status=INSUFFICIENT,
                        kind="intermediate_state",
                        retry="re-query_for_completed_result",
                        reasons=["retrieved item describes an intermediate state rather than the final result"],
                    )
                if item.metadata.get("freshness") == "stale":
                    return EvidenceAssessment(
                        status=INSUFFICIENT,
                        kind="stale",
                        retry="query_with_freshness_qualifier",
                        reasons=["retrieved evidence is stale"],
                    )

        return assessment

    def evaluate(
        self,
        reading: SemanticRequest,
        decision: EvidenceDecision,
        *,
        evidence_count: int = 0,
        evidence_conflicting: bool = False,
        retrieved_ok: bool = True,
    ) -> Answerability:
        """Return whether Atlas has enough to answer the *actual* request.

        ``evidence_count`` is the number of usable retrieved sources;
        ``evidence_conflicting`` is set by the retrieval layer when sources
        disagree; ``retrieved_ok`` is False when retrieval was attempted and
        failed outright.
        """

        # A subjective criterion with no objective proxy: the honest move is to
        # surface the ambiguity (and answer with an explicit criterion when the
        # user's intent is inferable from context) rather than invent an answer.
        if decision.subjective_without_proxy:
            if evidence_count >= MIN_SOURCES_FOR_RANKING:
                return Answerability(
                    status=AMBIGUOUS,
                    action=STATUS_ACTIONS[AMBIGUOUS],
                    can_answer=True,
                    must_disclose_uncertainty=True,
                    reasons=[
                        f"'{decision.criterion}' has no objective measure; "
                        "answer with the criterion stated explicitly"
                    ],
                )
            return Answerability(
                status=AMBIGUOUS,
                action=STATUS_ACTIONS[AMBIGUOUS],
                clarification_question=(
                    f"'{decision.criterion}' is subjective. Which measure should I use "
                    "- for example a specific metric or the most recent ranking you trust?"
                ),
                can_answer=False,
                must_disclose_uncertainty=True,
                reasons=["subjective criterion and no evidence to ground a proxy"],
            )

        if decision.needs_comparison:
            if evidence_count >= MIN_SOURCES_FOR_RANKING:
                return Answerability(
                    status=SUFFICIENT,
                    action=STATUS_ACTIONS[SUFFICIENT],
                    can_answer=True,
                    reasons=[
                        f"{evidence_count} candidate source(s) retrieved for comparison"
                    ],
                )
            return Answerability(
                status=INSUFFICIENT,
                action=STATUS_ACTIONS[INSUFFICIENT],
                can_answer=False,
                must_disclose_uncertainty=True,
                reasons=[
                    "a ranking was requested but too few candidates/sources were "
                    "retrieved to compare"
                ],
            )

        if decision.requirement == EVIDENCE_REQUIRED:
            if evidence_count >= MIN_SOURCES_FOR_CURRENT:
                if evidence_conflicting:
                    return Answerability(
                        status=CONFLICTING,
                        action=STATUS_ACTIONS[CONFLICTING],
                        can_answer=True,
                        must_disclose_uncertainty=True,
                        reasons=["retrieved sources disagree; prefer authoritative/recent"],
                    )
                return Answerability(
                    status=SUFFICIENT,
                    action=STATUS_ACTIONS[SUFFICIENT],
                    can_answer=True,
                    reasons=[f"{evidence_count} source(s) retrieved for a current answer"],
                )
            return Answerability(
                status=INSUFFICIENT if retrieved_ok else UNKNOWN,
                action=STATUS_ACTIONS[INSUFFICIENT if retrieved_ok else UNKNOWN],
                can_answer=False,
                must_disclose_uncertainty=True,
                reasons=[
                    "current/required evidence was requested but no usable source "
                    "was retrieved; Atlas must not state a current fact"
                ],
            )

        if decision.requirement == EVIDENCE_PREFERRED:
            if evidence_count >= MIN_SOURCES_FOR_PREFERRED:
                return Answerability(
                    status=SUFFICIENT,
                    action=STATUS_ACTIONS[SUFFICIENT],
                    can_answer=True,
                    reasons=[f"{evidence_count} source(s) retrieved; answer is grounded"],
                )
            return Answerability(
                status=SUFFICIENT,
                action=STATUS_ACTIONS[SUFFICIENT],
                can_answer=True,
                must_disclose_uncertainty=True,
                reasons=[
                    "no external evidence was retrieved; answering from model "
                    "knowledge with that limitation disclosed"
                ],
            )

        # Evidence unnecessary: stable knowledge, a local observation, an action,
        # or a transformation of content Atlas already has.
        if reading.ambiguity == AMBIGUITY_HIGH:
            return Answerability(
                status=AMBIGUOUS,
                action=STATUS_ACTIONS[AMBIGUOUS],
                can_answer=False,
                clarification_question=(
                    "I want to make sure I answer the right thing - can you tell me "
                    "what you'd like me to look at?"
                ),
                reasons=["the subject could not be resolved from the request or context"],
            )
        return Answerability(
            status=SUFFICIENT,
            action=STATUS_ACTIONS[SUFFICIENT],
            can_answer=True,
            reasons=["answerable without external evidence"],
        )


# -- helpers ------------------------------------------------------------------


#: Criteria that are inherently subjective (no measurement settles them). Kept in
#: sync with :mod:`reasoning.semantic_analysis`; defined here too so the policy is
#: usable without importing the analyser.
_SUBJECTIVE_MARKERS: tuple[str, ...] = (
    "best", "greatest", "favorite", "favourite", "coolest", "funniest",
    "most fun", "most beautiful", "most important", "most influential",
    "most famous", "most popular", "most talented", "smartest", "nicest",
    "most enjoyable", "must-see",
)

#: Descriptive filler that must not leak into a retrieval query.
_QUERY_NOISE: tuple[str, ...] = (
    "search", "find", "look up", "google", "tell me", "about", "information",
    "info", "details", "please", "give me", "show me", "what is", "who is",
    "what are", "which", "the", "a", "an", "for", "on", "regarding",
)


def _is_subjective(criterion: str) -> bool:
    lowered = (criterion or "").casefold()
    return any(marker in lowered for marker in _SUBJECTIVE_MARKERS)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (value or "").casefold()).strip("_")
    return slug or "criterion"


def _retrieval_query(reading: SemanticRequest, *, question: str) -> str:
    """Build the query Atlas should retrieve with.

    For a ranking, the query combines the candidate set with the criterion (and
    the proxy when one exists), because "most famous Minecraft YouTuber" is
    answered by searching for a ranking, not by searching the raw sentence. For
    everything else, the subject is the query - never the raw instruction, so
    instruction language cannot leak into a tool call.
    """

    subject = (reading.subject or "").strip()
    if reading.comparative:
        parts: list[str] = []
        if subject:
            parts.append(subject)
        elif reading.candidate_set:
            parts.append(reading.candidate_set)
        if reading.criterion_proxy:
            parts.append(reading.criterion_proxy)
        elif reading.criterion:
            parts.append(reading.criterion)
        if parts:
            return " ".join(dict.fromkeys(parts)).strip()
    if subject:
        return subject
    return _strip_noise(question)


def _strip_noise(text: str) -> str:
    value = " " + " ".join((text or "").casefold().split()) + " "
    for noise in _QUERY_NOISE:
        value = value.replace(f" {noise} ", " ")
    return value.strip()


def merge_evidence_requirements(values: Iterable[str]) -> str:
    """Combine evidence requirements, taking the strongest (required > preferred)."""

    order = {EVIDENCE_REQUIRED: 3, EVIDENCE_PREFERRED: 2, EVIDENCE_UNNECESSARY: 1}
    best = EVIDENCE_UNNECESSARY
    for value in values:
        if order.get(value, 0) > order.get(best, 0):
            best = value
    return best
