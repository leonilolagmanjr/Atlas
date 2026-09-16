"""Shared Atlas execution models.

These dataclasses describe the request state and execution plan passed between
Brain, Planner, Executor, retrieval, and LLM components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from vector_store import SearchHit


class StepStatus(str, Enum):
    """Lifecycle state for a single execution step."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class PlanStatus(str, Enum):
    """Lifecycle state for an execution plan."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    WAITING_FOR_CONFIRMATION = "WAITING_FOR_CONFIRMATION"


class TaskStatus(str, Enum):
    """Lifecycle state for the complete user task."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNCERTAIN = "UNCERTAIN"
    WAITING_FOR_CONFIRMATION = "WAITING_FOR_CONFIRMATION"


@dataclass
class Evidence:
    """Evidence gathered for answering a request.

    This is intentionally richer for V3.2, but remains backward compatible
    with the existing pipeline (context/chunks/sources/metadata).
    """

    # Backward-compatible fields used by current code
    context: str = ""
    chunks: list[SearchHit] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # First-class placeholders for future citation/evidence engine upgrades
    document: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    chunk_ids: list[str] = field(default_factory=list)
    confidence: Optional[float] = None
    retrieval_method: Optional[str] = None


@dataclass
class RetrievalResult:
    """Shared retrieval output for execution context consumers."""

    context: str
    best_distance: Optional[float]
    retrieved_chunks: list[SearchHit] = field(default_factory=list)
    expanded_queries: list[str] = field(default_factory=list)
    diagnostics: Optional[Any] = None


#: Canonical free-text modifier fields carried by a StructuredIntent.
#: Keeping this list in one place guarantees that parameter preservation,
#: merging, and serialization all agree on the same schema.
INTENT_TEXT_FIELDS: tuple[str, ...] = (
    "action",
    "target",
    "content_type",
    "topic",
    "query",
    "destination",
    "style",
    "tone",
    "length",
    "sort",
)


@dataclass
class StructuredIntent:
    """Canonical, compositional representation of a user request.

    The semantic interpreter (Qwen) proposes these fields; deterministic code
    consumes and validates them. Nothing here is executable: executors only
    ever act on validated capabilities, never on raw model text.
    """

    intent: str = "unknown"
    action: Optional[str] = None
    target: Optional[str] = None
    content_type: Optional[str] = None
    topic: Optional[str] = None
    query: Optional[str] = None
    destination: Optional[str] = None
    style: Optional[str] = None
    tone: Optional[str] = None
    length: Optional[str] = None
    sort: Optional[str] = None
    filters: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    entities: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    source: str = "deterministic"

    def parameters(self) -> dict[str, Any]:
        """Return the non-empty modifiers as a flat mapping."""

        values: dict[str, Any] = {}
        for name in INTENT_TEXT_FIELDS:
            value = getattr(self, name)
            if value:
                values[name] = value
        if self.filters:
            values["filters"] = list(self.filters)
        if self.constraints:
            values["constraints"] = list(self.constraints)
        if self.entities:
            values["entities"] = dict(self.entities)
        return values

    def merged_with(self, other: "StructuredIntent") -> "StructuredIntent":
        """Overlay a follow-up intent onto this one without losing context.

        Fields explicitly present on ``other`` win; unrelated fields inherit
        from ``self`` so a follow-up like "make it about cars" keeps the
        previously established destination and content type.
        """

        merged = StructuredIntent(**self.to_dict())
        # An explicit follow-up intent label wins; otherwise inherit the base.
        if other.intent and other.intent not in {"unknown", ""}:
            merged.intent = other.intent
        for name in INTENT_TEXT_FIELDS:
            value = getattr(other, name)
            if value:
                setattr(merged, name, value)
        merged.filters = list(dict.fromkeys([*self.filters, *other.filters]))
        merged.constraints = list(dict.fromkeys([*self.constraints, *other.constraints]))
        merged.entities = {**self.entities, **other.entities}
        merged.options = {**self.options, **other.options}
        merged.confidence = min(self.confidence or 1.0, other.confidence)
        merged.needs_clarification = other.needs_clarification
        merged.clarification_question = other.clarification_question
        merged.source = other.source or self.source
        return merged

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            **{name: getattr(self, name) for name in INTENT_TEXT_FIELDS},
            "filters": list(self.filters),
            "constraints": list(self.constraints),
            "entities": dict(self.entities),
            "options": dict(self.options),
            "confidence": self.confidence,
            "needs_clarification": self.needs_clarification,
            "clarification_question": self.clarification_question,
            "source": self.source,
        }

    @classmethod
    def from_mapping(cls, data: dict[str, Any], *, source: str = "llm") -> "StructuredIntent":
        """Build a validated intent from untrusted model output.

        Unknown keys are ignored, wrong types are coerced or dropped, and the
        confidence is clamped. Malformed model output never reaches an executor.
        """

        def _text(value: Any) -> Optional[str]:
            if value is None:
                return None
            if isinstance(value, (str, int, float)):
                text = str(value).strip()
                if text and text.casefold() not in {"null", "none", "n/a", ""}:
                    return text
            return None

        def _list(value: Any) -> list[str]:
            if isinstance(value, str):
                text = value.strip()
                return [text] if text else []
            if isinstance(value, (list, tuple)):
                return [str(item).strip() for item in value if str(item).strip()]
            return []

        def _confidence(value: Any) -> float:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return 0.5
            return max(0.0, min(1.0, number))

        entities = data.get("entities")
        options = data.get("options")
        return cls(
            intent=_text(data.get("intent")) or "unknown",
            **{name: _text(data.get(name)) for name in INTENT_TEXT_FIELDS},
            filters=_list(data.get("filters")),
            constraints=_list(data.get("constraints")),
            entities=entities if isinstance(entities, dict) else {},
            options=options if isinstance(options, dict) else {},
            confidence=_confidence(data.get("confidence")),
            needs_clarification=bool(data.get("needs_clarification")),
            clarification_question=_text(data.get("clarification_question")),
            source=source,
        )


@dataclass
class ExecutionStep:
    """One executable unit in an Atlas plan."""

    id: str
    name: str
    action: str
    description: str
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """Ordered plan for fulfilling a user request."""

    user_question: str
    steps: list[ExecutionStep]
    plan_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: PlanStatus = PlanStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlannerDecision:
    """Planner output wrapper reserved for future strategy metadata."""

    plan: ExecutionPlan
    strategy: str = "deterministic"
    confidence: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionContext:
    """Central state for a single Atlas request execution."""

    user_input: str
    task_id: str = field(default_factory=lambda: str(uuid4()))
    normalized_input: Optional[str] = None
    goal: Optional[str] = None
    intent: Optional[str] = None
    intent_category: Optional[str] = None
    structured_intent: Optional[StructuredIntent] = None
    status: TaskStatus = TaskStatus.PENDING
    execution_plan: Optional[ExecutionPlan] = None
    current_step: Optional[str] = None
    completed_steps: list[str] = field(default_factory=list)
    pending_steps: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    memory_references: list[str] = field(default_factory=list)
    web_sources: list[str] = field(default_factory=list)
    permissions: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    verification_results: list[dict[str, Any]] = field(default_factory=list)
    evidence: Evidence = field(default_factory=Evidence)
    retrieval_result: Optional[RetrievalResult] = None
    llm_response: Optional[str] = None
    final_response: Optional[str] = None
    selected_tool: Optional[str] = None
    confidence: Optional[float] = None
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def request(self) -> str:
        """Expose the task request using the runtime terminology."""

        return self.user_input

    @property
    def final_result(self) -> Optional[str]:
        """Expose the final response using the runtime terminology."""

        return self.final_response

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible snapshot for logs and checkpoints."""

        return _json_safe(self)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _json_safe(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
