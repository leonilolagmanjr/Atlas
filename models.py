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

