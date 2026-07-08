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


@dataclass
class Evidence:
    """Evidence gathered for answering a request."""

    context: str = ""
    chunks: list[SearchHit] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


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
    normalized_input: Optional[str] = None
    intent: Optional[str] = None
    execution_plan: Optional[ExecutionPlan] = None
    evidence: Evidence = field(default_factory=Evidence)
    retrieval_result: Optional[RetrievalResult] = None
    llm_response: Optional[str] = None
    final_response: Optional[str] = None
    selected_tool: Optional[str] = None
    confidence: Optional[float] = None
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

