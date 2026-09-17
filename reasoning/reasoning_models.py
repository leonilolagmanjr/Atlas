"""Structured reasoning vocabulary and decisions for Atlas.

This module is the boundary between *understanding* (the Task IR) and *doing*
(the deterministic planner and executor). The reasoning engine answers one
question:

    "What does this request actually need, and where should Atlas get it?"

Design rules:

* A decision is data, not prose. It names sources, capabilities, and a response
  mode so deterministic code can validate it and execute it.
* The decision *extends* the Task IR (:class:`models_task.Task`) instead of
  competing with it: the engine writes the decision back onto the task.
* Nothing in this module executes anything. The model proposes; Atlas validates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SourceType(str, Enum):
    """Where an answer or an action can come from.

    These are deliberately distinct concepts:

    * ``SELF`` - Atlas's own capability registry and runtime state.
    * ``CONVERSATION`` - what was said earlier in this session.
    * ``MEMORY`` - durable/session memory recall.
    * ``KNOWLEDGE`` - the local indexed document knowledge base.
    * ``MODEL`` - the configured language model's own knowledge.
    * ``FILES`` - user-approved local files.
    * ``WEB`` - public internet research (always untrusted evidence).
    * ``COMPUTER`` - computer-control actions (launch, write, move, ...).
    * ``SYSTEM`` - local system state (OS, disk, processes, services).
    """

    SELF = "self"
    CONVERSATION = "conversation"
    MEMORY = "memory"
    KNOWLEDGE = "knowledge"
    MODEL = "model"
    FILES = "files"
    WEB = "web"
    COMPUTER = "computer"
    SYSTEM = "system"


class RequestType(str, Enum):
    """High-level shape of the request after semantic reasoning."""

    QUESTION = "question"
    SELF_QUERY = "self_query"
    MEMORY_QUERY = "memory_query"
    ACTION = "action"
    HYBRID = "hybrid"
    CLARIFICATION = "clarification"


class ResponseMode(str, Enum):
    """How the final response should be produced and attributed."""

    #: Answer from the model's own knowledge (no local grounding required).
    DIRECT_ANSWER = "direct_answer"
    #: Answer grounded in retrieved local documents.
    GROUNDED_ANSWER = "grounded_answer"
    #: Answer grounded in retrieved, attributed web evidence.
    WEB_RESEARCH = "web_research"
    #: Answer grounded in local file evidence.
    FILE_LOOKUP = "file_lookup"
    #: Answer grounded in local system observations.
    SYSTEM_DIAGNOSIS = "system_diagnosis"
    #: Answer generated from the live capability registry (no LLM needed).
    SELF_DESCRIPTION = "self_description"
    #: Answer generated from conversation/memory recall.
    MEMORY_RECALL = "memory_recall"
    #: Report of an executed action and its verification.
    ACTION_REPORT = "action_report"
    #: Atlas genuinely cannot obtain the information and says so honestly.
    LIMITATION = "limitation"
    #: Atlas needs the user to disambiguate before it can proceed.
    CLARIFICATION = "clarification"


class ConfidenceLevel(str, Enum):
    """Deterministic confidence buckets with an explicit meaning each."""

    #: A retrieved source (or Atlas's own registry) directly supports the answer.
    HIGH = "high"
    #: Answerable, but only from model knowledge or weak (snippet-level) evidence.
    MEDIUM = "medium"
    #: Cannot be answered yet: more retrieval or clarification is required.
    LOW = "low"
    #: Consequential/irreversible action: never proceed without validation.
    CRITICAL = "critical"


class ReasoningStage(str, Enum):
    """Inspectable reasoning phases exposed to logs and the UI.

    These stages describe *what Atlas is doing*, never hidden chain-of-thought.
    """

    UNDERSTANDING = "UNDERSTANDING"
    SOURCE_SELECTION = "SOURCE_SELECTION"
    RETRIEVING = "RETRIEVING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    EVALUATING = "EVALUATING"
    ANSWERING = "ANSWERING"


#: Read-only sources: safe for the reasoning loop to consult without approval.
OBSERVATION_SOURCES: frozenset[SourceType] = frozenset(
    {
        SourceType.SELF,
        SourceType.CONVERSATION,
        SourceType.MEMORY,
        SourceType.KNOWLEDGE,
        SourceType.MODEL,
        SourceType.FILES,
        SourceType.WEB,
        SourceType.SYSTEM,
    }
)

#: Sources that imply a mutating action and therefore a confirmation boundary.
ACTION_SOURCES: frozenset[SourceType] = frozenset({SourceType.COMPUTER})


@dataclass(frozen=True)
class SourcePlan:
    """An ordered, inspectable plan of sources for a single request."""

    sources: tuple[SourceType, ...] = ()
    reason: str = ""
    #: Capabilities the engine intends to use, best first.
    capabilities: tuple[str, ...] = ()
    #: True when the answer depends on information that changes over time.
    current_information_required: bool = False
    #: True when at least one selected capability mutates local state.
    requires_action: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources": [source.value for source in self.sources],
            "reason": self.reason,
            "capabilities": list(self.capabilities),
            "current_information_required": self.current_information_required,
            "requires_action": self.requires_action,
        }


@dataclass
class ReasoningDecision:
    """The structured decision the reasoning engine produces for one request.

    This is the machine-readable answer to "what does the user want, which
    sources are required, how should the result be verified and reported?".

    It intentionally carries the whole reasoning decision surface and is written
    back onto the Task IR, so the planner and executor consume one object rather
    than two competing representations.
    """

    goal: str = "unknown"
    request_type: RequestType = RequestType.QUESTION
    requires_web: bool = False
    requires_files: bool = False
    requires_memory: bool = False
    requires_knowledge: bool = False
    requires_computer: bool = False
    requires_system: bool = False
    requires_self_introspection: bool = False
    requires_model_knowledge: bool = False
    requires_clarification: bool = False
    current_information_required: bool = False
    sources: list[SourceType] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    reason: str = ""
    response_mode: ResponseMode = ResponseMode.DIRECT_ANSWER
    #: Deterministic 0..1 answerability score (never a model-invented number).
    confidence: float = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.LOW
    needs_clarification: bool = False
    clarification_question: str | None = None
    #: True when the local model was consulted for this decision.
    llm_used: bool = False
    #: True when the request needs multi-step execution.
    multi_step: bool = False
    #: True when verification of the executed action is required.
    requires_verification: bool = False
    source_plan: SourcePlan = field(default_factory=SourcePlan)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "request_type": self.request_type.value,
            "requires_web": self.requires_web,
            "requires_files": self.requires_files,
            "requires_memory": self.requires_memory,
            "requires_knowledge": self.requires_knowledge,
            "requires_computer": self.requires_computer,
            "requires_system": self.requires_system,
            "requires_self_introspection": self.requires_self_introspection,
            "requires_model_knowledge": self.requires_model_knowledge,
            "requires_clarification": self.requires_clarification,
            "current_information_required": self.current_information_required,
            "sources": [source.value for source in self.sources],
            "actions": list(self.actions),
            "reason": self.reason,
            "response_mode": self.response_mode.value,
            "confidence": round(self.confidence, 4),
            "confidence_level": self.confidence_level.value,
            "needs_clarification": self.needs_clarification,
            "clarification_question": self.clarification_question,
            "llm_used": self.llm_used,
            "multi_step": self.multi_step,
            "requires_verification": self.requires_verification,
            "source_plan": self.source_plan.to_dict(),
        }

    def confidence_for(
        self,
        *,
        has_evidence: bool,
        evidence_from_retrieval: bool,
    ) -> ConfidenceLevel:
        """Return the deterministic confidence bucket for the current evidence.

        Policy (explicit, never a magic number):

        * any mutating action -> ``CRITICAL`` (validate/confirm, never assume)
        * retrieved or self-described evidence -> ``HIGH``
        * model knowledge only -> ``MEDIUM``
        * nothing retrieved -> ``LOW``
        """

        if self.requires_computer or self.source_plan.requires_action:
            return ConfidenceLevel.CRITICAL
        if has_evidence and evidence_from_retrieval:
            return ConfidenceLevel.HIGH
        if self.response_mode in {ResponseMode.SELF_DESCRIPTION, ResponseMode.MEMORY_RECALL}:
            return ConfidenceLevel.HIGH
        if has_evidence or self.requires_model_knowledge:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW
@dataclass(frozen=True)
class ReasoningStep:
    """One inspectable stage record in a reasoning trace."""

    stage: ReasoningStage
    detail: str
    iteration: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "detail": self.detail,
            "iteration": self.iteration,
            "metadata": dict(self.metadata),
        }


@dataclass
class ReasoningTrace:
    """Ordered, structured reasoning state for a single request.

    Exposed through ``ExecutionContext.metadata["reasoning"]`` so logs and the
    UI can show *what Atlas did* without exposing private chain-of-thought.
    """

    steps: list[ReasoningStep] = field(default_factory=list)
    iteration: int = 0

    def record(self, stage: ReasoningStage, detail: str, **metadata: Any) -> ReasoningStep:
        step = ReasoningStep(stage=stage, detail=detail, iteration=self.iteration, metadata=metadata)
        self.steps.append(step)
        return step

    def to_list(self) -> list[dict[str, Any]]:
        return [step.to_dict() for step in self.steps]

    def summary(self) -> str:
        return " -> ".join(step.stage.value for step in self.steps)


@dataclass
class ReasoningOutcome:
    """The result of running the reasoning loop for one request."""

    answer: str = ""
    response_mode: ResponseMode = ResponseMode.LIMITATION
    sources_used: list[SourceType] = field(default_factory=list)
    confidence_level: ConfidenceLevel = ConfidenceLevel.LOW
    iteration: int = 0
    evidence_count: int = 0
    verified: bool | None = None
    pending_confirmation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "response_mode": self.response_mode.value,
            "sources_used": [source.value for source in self.sources_used],
            "confidence_level": self.confidence_level.value,
            "iteration": self.iteration,
            "evidence_count": self.evidence_count,
            "verified": self.verified,
            "pending_confirmation": self.pending_confirmation,
        }
