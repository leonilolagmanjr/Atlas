"""Local HTTP API for the Atlas frontend.

This adapter exposes existing runtime capabilities; it does not reimplement
planning, retrieval, memory, or tool execution in the web layer.
"""

from __future__ import annotations

import logging
import threading
import time
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import PureWindowsPath
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from brain import Brain
from computer.applications import InstalledApplicationsTool
from computer.runtime import register_read_only_tools
from computer.system import SystemInfoTool
from config import COMPUTER_ROOT, ENABLE_CONVERSATION_STREAMING, EXECUTION_MODE, OLLAMA_MODEL, TASK_STORE_FILE
from experience.models import FAILURE_CATEGORIES, FAILURE_CATEGORY_LABELS
from experience.service import ExperienceService, FeedbackRequest
from indexer import index_knowledge_base
from llm import ask
from memory.conversation_runtime import CancellationToken, ConversationRuntime
from memory.memory_manager import MemoryManager
from memory.user_memory import MEMORY_KINDS, UserMemoryStore
from models import TaskStatus
from reasoning.reasoning_models import ReasoningStage, ResponseMode, SourceType
from task_store import TaskStore
from tools import ExecutionMode, PermissionEngine, ToolRegistry, ToolRouter
from tools.discovery import ToolDiscovery
from tools.knowledge import ToolKnowledgeStore, load_json
from vector_store import VectorStore

logger = logging.getLogger(__name__)


class TaskRequest(BaseModel):
    request: str = Field(min_length=1, max_length=4000)


class MessagePayload(BaseModel):
    """One conversational turn submitted by the UI."""

    message: str = Field(min_length=1, max_length=8000)
    conversation_id: str | None = Field(default=None, max_length=128)
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    #: Request the SSE endpoint's live token events. Ignored by POST /messages.
    stream: bool = False


class ConversationRenamePayload(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class MemoryPayload(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    kind: str = Field(default="preference", max_length=32)

    @model_validator(mode="after")
    def validate_kind(self) -> "MemoryPayload":
        if self.kind not in MEMORY_KINDS:
            raise ValueError("unrecognized memory kind")
        return self


class MemoryUpdatePayload(BaseModel):
    text: str | None = Field(default=None, max_length=2000)
    kind: str | None = Field(default=None, max_length=32)

    @model_validator(mode="after")
    def validate_kind(self) -> "MemoryUpdatePayload":
        if self.kind is not None and self.kind not in MEMORY_KINDS:
            raise ValueError("unrecognized memory kind")
        return self


class MemoryTogglePayload(BaseModel):
    enabled: bool


class FeedbackPayload(BaseModel):
    """Explicit Success/Failed feedback for one task record.

    Only ``outcome`` is required: the user is never forced to justify a click.
    ``failure_category`` is validated against the canonical vocabulary so the
    frontend cannot invent a category that pattern analysis does not understand.
    """

    outcome: str = Field(min_length=1, max_length=32)
    reason: str = Field(default="", max_length=500)
    failure_category: str = Field(default="", max_length=64)
    correction: str = Field(default="", max_length=2000)
    expected_behavior: str = Field(default="", max_length=1000)
    response_quality: str = Field(default="", max_length=32)

    @model_validator(mode="after")
    def validate_vocabulary(self) -> "FeedbackPayload":
        normalized = self.outcome.strip().casefold()
        if normalized not in {"success", "failure"}:
            raise ValueError("outcome must be 'success' or 'failure'")
        object.__setattr__(self, "outcome", normalized)
        category = self.failure_category.strip().casefold()
        if category and category not in FAILURE_CATEGORIES:
            raise ValueError("unrecognized failure_category")
        object.__setattr__(self, "failure_category", category)
        return self


def _bounded_text(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join("".join(character for character in value[:limit] if character.isprintable() or character.isspace()).split())


def _source_names(value: Any) -> list[str]:
    allowed = {source.value for source in SourceType}
    if not isinstance(value, (list, tuple)):
        return []
    return list(dict.fromkeys(item for item in value[:32] if isinstance(item, str) and item in allowed))


def _safe_citation(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 2048 or any(not character.isprintable() for character in value):
        return ""
    value = value.strip()
    try:
        parsed = urlsplit(value)
        if parsed.scheme in {"https", "http"}:
            if not parsed.hostname:
                return ""
            hostname = parsed.hostname
            if ":" in hostname:
                hostname = f"[{hostname}]"
            port = f":{parsed.port}" if parsed.port else ""
            return urlunsplit((parsed.scheme, hostname + port, parsed.path, "", ""))[:512]
        if parsed.scheme and not PureWindowsPath(value).drive:
            return ""
    except ValueError:
        return ""
    return _bounded_text(PureWindowsPath(value).name, 160)


def _sanitize_observability(value: dict[str, Any]) -> dict[str, Any]:
    stages = {stage.value for stage in ReasoningStage}
    reasoning = []
    raw_steps = value.get("reasoning")
    if isinstance(raw_steps, (list, tuple)):
        for step in raw_steps[:64]:
            if not isinstance(step, dict) or not isinstance(step.get("stage"), str) or step["stage"] not in stages:
                continue
            iteration = step.get("iteration", 0)
            reasoning.append({
                "stage": step["stage"],
                "detail": _bounded_text(step.get("detail"), 240),
                "iteration": min(max(iteration, 0), 100) if type(iteration) is int else 0,
            })
    citations = value.get("citations")
    safe_citations = []
    if isinstance(citations, (list, tuple)):
        for citation in citations[:16]:
            safe = _safe_citation(citation)
            if safe and safe not in safe_citations:
                safe_citations.append(safe)
    mode = value.get("response_mode")
    evidence = value.get("evidence")
    safe_evidence = None
    if isinstance(evidence, dict):
        count = evidence.get("count", 0)
        safe_evidence = {
            "count": min(max(count, 0), 1000000) if type(count) is int else 0,
            "sources": _source_names(evidence.get("sources")),
        }
    return {
        "reasoning": reasoning,
        "provenance": _source_names(value.get("provenance")),
        "citations": safe_citations,
        "response_mode": mode if isinstance(mode, str) and mode in {item.value for item in ResponseMode} else None,
        "evidence": safe_evidence,
    }


def _reasoning_snapshot(context: Any) -> dict[str, Any]:
    metadata = getattr(context, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    task = metadata.get("task")
    task = task if isinstance(task, dict) else {}
    task_context = task.get("context")
    task_context = task_context if isinstance(task_context, dict) else {}
    answer = metadata.get("reasoning_answer")
    answer = answer if isinstance(answer, dict) else {}
    answer_metadata = answer.get("metadata")
    answer_metadata = answer_metadata if isinstance(answer_metadata, dict) else {}
    answer_evidence = answer_metadata.get("evidence")
    answer_evidence = answer_evidence if isinstance(answer_evidence, dict) else {}
    return _sanitize_observability({
        "reasoning": metadata.get("reasoning") or task_context.get("reasoning") or answer_metadata.get("reasoning"),
        "provenance": answer.get("provenance") or task_context.get("provenance"),
        "citations": answer.get("citations") or task_context.get("citations"),
        "response_mode": answer.get("mode") or task.get("response_mode") or task_context.get("response_mode"),
        "evidence": answer_evidence or task_context.get("evidence"),
    })


class TaskRecord(BaseModel):
    id: str
    request: str
    status: str
    response: str | None = None
    created_at: float
    updated_at: float
    task_id: str | None = None
    plan: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    web_sources: list[str] = Field(default_factory=list)
    web_results: list[dict[str, Any]] = Field(default_factory=list)
    reasoning: list[dict[str, Any]] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    response_mode: str | None = None
    evidence: dict[str, Any] | None = None
    # -- human feedback / experience linkage -------------------------------
    #: Whether this task should show Success/Failed controls at all. Casual
    #: conversation stays False so feedback UI never appears for a plain answer.
    feedback_available: bool = False
    #: "success" | "failure" | "unknown"; persisted so the answer survives restart.
    feedback_outcome: str = "unknown"
    feedback_category: str = ""
    feedback_category_label: str = ""
    feedback_reason: str = ""
    feedback_correction: str = ""
    #: The experience this feedback was attached to, for traceability.
    experience_id: str = ""

    @model_validator(mode="before")
    @classmethod
    def sanitize_observability(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {**value, **_sanitize_observability(value)}
        return value


@dataclass
class AtlasService:
    # Serialize every task through a single FIFO worker.
    # A single worker guarantees that Atlas never runs two tasks (or two plan
    # steps) at the same time, so computer-control actions cannot overlap.
    # Tasks submitted while another is running wait in an explicit queue.

    brain: Brain | None = None
    registry: ToolRegistry | None = None
    #: The conversation runtime ABOVE the Brain. It owns conversation state and
    #: turn routing; it never plans or executes on its own.
    runtime: ConversationRuntime | None = None
    #: Long-term user memory, exposed for the memory-management API.
    user_memory: UserMemoryStore | None = None
    #: Experience loop: feedback capture, durable experience records, retrieval.
    experience: ExperienceService = field(default_factory=ExperienceService)
    _tasks: dict[str, TaskRecord] = field(default_factory=dict)
    _task_store: TaskStore = field(default_factory=lambda: TaskStore(path=TASK_STORE_FILE))
    _executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=1))
    _lock: threading.RLock = field(default_factory=threading.RLock)
    # FIFO of record ids waiting to run, plus the id currently running.
    _queue: list[str] = field(default_factory=list)
    _running_id: str | None = None
    # Records currently being approved/denied, to reject double submissions.
    _resolving: set[str] = field(default_factory=set)
    # Live conversation turns (turn_id -> CancellationToken). Used for real
    # cancellation of an in-flight conversational turn.
    _turn_tokens: dict[str, CancellationToken] = field(default_factory=dict)

    def __post_init__(self) -> None:
        persisted = self._task_store.load()
        for task_id, payload in persisted.items():
            try:
                record = TaskRecord(**payload)
            except Exception:
                logger.warning("Skipping invalid persisted task: %s", task_id)
                continue
            if record.status in {
                TaskStatus.PENDING.value,
                TaskStatus.RUNNING.value,
                TaskStatus.WAITING_FOR_CONFIRMATION.value,
            }:
                record.status = TaskStatus.FAILED.value
                record.response = "Task interrupted when the Atlas API stopped."
                record.errors.append("The task could not be resumed after an API restart.")
                record.updated_at = time.time()
            self._tasks[record.id] = record
        self._persist()

    def ensure_runtime(self) -> None:
        if self.brain is not None:
            return
        with self._lock:
            if self.brain is not None:
                return
            vector_store = VectorStore()
            index_knowledge_base(vector_store=vector_store)
            memory_manager = MemoryManager()
            registry = ToolRegistry()
            register_read_only_tools(registry, root=COMPUTER_ROOT, ask=ask)
            router = ToolRouter(
                registry=registry,
                permission_engine=PermissionEngine(mode=ExecutionMode(EXECUTION_MODE.lower())),
            )
            self.registry = registry
            self.brain = Brain(
                vector_store=vector_store,
                system_prompt=_read_prompt("system.txt"),
                retrieval_template=_read_prompt("retrieval.txt"),
                memory_manager=memory_manager,
                tool_router=router,
                llm_ask=ask,
                experience_service=self.experience,
            )
            # The conversation runtime sits above the Brain and shares the SAME
            # memory manager, so a conversation and a delegated task see one
            # history rather than two divergent ones.
            self.user_memory = UserMemoryStore()
            self.runtime = ConversationRuntime(
                memory_manager=memory_manager,
                brain=self.brain,
                user_memory=self.user_memory,
                model_name=OLLAMA_MODEL,
                # Attachment reading reuses the registry's own file tools, so an
                # attachment cannot bypass the root check, size bounds or
                # permission engine.
                tool_runner=router.execute,
            )

    # -- conversational API ------------------------------------------------------

    def conversations(self) -> list[dict[str, Any]]:
        self.ensure_runtime()
        assert self.runtime is not None
        return [
            {**item, "message_count": item.get("message_count", 0)}
            for item in self.runtime.list_conversations()
        ]

    def new_conversation(self, title: str | None = None) -> dict[str, Any]:
        self.ensure_runtime()
        assert self.runtime is not None
        return self.runtime.new_conversation(title=title)

    def conversation_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        self.ensure_runtime()
        assert self.runtime is not None
        return self.runtime.get_messages(conversation_id)

    def search_conversations(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        self.ensure_runtime()
        assert self.runtime is not None
        return self.runtime.search(query, limit=min(max(limit, 1), 100))

    def rename_conversation(self, conversation_id: str, title: str) -> dict[str, Any]:
        self.ensure_runtime()
        assert self.runtime is not None
        self.runtime.rename_conversation(conversation_id, title)
        return self.runtime.open_conversation(conversation_id)

    def delete_conversation(self, conversation_id: str) -> None:
        self.ensure_runtime()
        assert self.runtime is not None
        self.runtime.delete_conversation(conversation_id)

    def send_message(
        self,
        *,
        message: str,
        conversation_id: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        events: Any | None = None,
        stream_model: bool = False,
        turn_id: str | None = None,
        stream: Any | None = None,
        token: CancellationToken | None = None,
    ) -> dict[str, Any]:
        """Serve one conversational turn and return the real turn result.

        ``stream`` and ``token`` are injectable so a caller (the SSE endpoint, or
        a test) can supply the real streaming boundary and the real cancellation
        token. When ``stream_model`` is set without an injected boundary, the
        configured provider's streaming call is used.
        """

        self.ensure_runtime()
        assert self.runtime is not None
        if token is None:
            token = CancellationToken()
        if turn_id:
            with self._lock:
                self._turn_tokens[turn_id] = token
        try:
            result = self.runtime.handle_message(
                message,
                conversation_id=conversation_id,
                attachments=attachments,
                events=events,
                token=token,
                stream=stream if stream is not None else (_streaming_model() if stream_model else None),
            )
            return result.to_dict()
        finally:
            if turn_id:
                with self._lock:
                    self._turn_tokens.pop(turn_id, None)

    def cancel_turn(self, turn_id: str) -> bool:
        """Cancel an in-flight conversational turn (real cooperative token)."""

        with self._lock:
            token = self._turn_tokens.get(turn_id)
        if token is None:
            return False
        token.cancel()
        return True

    # -- long-term user memory ---------------------------------------------------

    def memory_records(self) -> list[dict[str, Any]]:
        self.ensure_runtime()
        assert self.user_memory is not None
        return self.user_memory.list()

    def memory_status(self) -> dict[str, Any]:
        self.ensure_runtime()
        assert self.user_memory is not None
        return {
            "enabled": self.user_memory.enabled,
            "count": len(self.user_memory.list()),
            "kinds": sorted(MEMORY_KINDS),
        }

    def set_memory_enabled(self, enabled: bool) -> dict[str, Any]:
        self.ensure_runtime()
        assert self.user_memory is not None
        self.user_memory.set_enabled(enabled)
        return self.memory_status()

    def search_memory(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        self.ensure_runtime()
        assert self.user_memory is not None
        return self.user_memory.search(query, limit=min(max(limit, 1), 100))

    def update_memory(self, memory_id: str, *, text: str | None = None, kind: str | None = None) -> dict[str, Any]:
        self.ensure_runtime()
        assert self.user_memory is not None
        updated = self.user_memory.update(memory_id, text=text, kind=kind)
        if updated is None:
            raise HTTPException(status_code=404, detail="Memory not found")
        return updated

    def delete_memory(self, memory_id: str) -> None:
        self.ensure_runtime()
        assert self.user_memory is not None
        if not self.user_memory.delete(memory_id):
            raise HTTPException(status_code=404, detail="Memory not found")

    def clear_memory(self) -> dict[str, Any]:
        self.ensure_runtime()
        assert self.user_memory is not None
        removed = self.user_memory.clear()
        return {"removed": removed, **self.memory_status()}

    def submit(self, request: str) -> TaskRecord:
        self.ensure_runtime()
        now = time.time()
        record = TaskRecord(
            id=str(uuid4()),
            request=request,
            status=TaskStatus.PENDING.value,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._tasks[record.id] = record
            self._queue.append(record.id)
            self._persist()
        # Hand off to the single worker; it drains the FIFO one task at a time.
        self._executor.submit(self._drain_queue)
        return record
    def _drain_queue(self) -> None:
        # Run queued tasks one at a time, in submission order.
        while True:
            with self._lock:
                # Only one drain loop may hold the running slot.
                if self._running_id is not None or not self._queue:
                    return
                record_id = self._queue.pop(0)
                self._running_id = record_id
            try:
                self._run(record_id)
            finally:
                with self._lock:
                    self._running_id = None
    def _run(self, record_id: str) -> None:
        with self._lock:
            record = self._tasks.get(record_id)
            if record is None:
                return
            record.status = TaskStatus.RUNNING.value
            record.updated_at = time.time()
            self._persist()
        try:
            assert self.brain is not None
            response = self.brain.process(record.request)
            context = self.brain.last_context
            with self._lock:
                record.response = response
                record.status = context.status.value if context is not None else TaskStatus.COMPLETED.value
                record.task_id = context.task_id if context is not None else None
                record.plan = _plan_snapshot(context)
                record.tool_calls = list(context.tool_calls) if context is not None else []
                record.errors = list(context.errors) if context is not None else []
                record.warnings = list(context.warnings) if context is not None else []
                record.web_sources = list(context.web_sources) if context is not None else []
                record.web_results = list(getattr(context, "metadata", {}).get("web_results", [])) if context is not None else []
                self._capture_reasoning(record, context)
                self._capture_feedback_state(record, context)
                record.updated_at = time.time()
                self._persist()
        except Exception as exc:
            logger.exception("API task failed")
            with self._lock:
                record.status = TaskStatus.FAILED.value
                record.response = "Atlas could not complete this task."
                record.errors = [str(exc)]
                record.updated_at = time.time()
                self._persist()

    def approve(self, record_id: str) -> TaskRecord:
        return self._resolve(record_id, approve=True)

    def deny(self, record_id: str) -> TaskRecord:
        return self._resolve(record_id, approve=False)

    def _resolve(self, record_id: str, *, approve: bool) -> TaskRecord:
        # Approve or deny once. A second in-flight request is rejected.
        # This prevents duplicate authorization: the resolution lock admits one
        # approve/deny per record at a time, and the status flips before the
        # (possibly slow) resume begins.
        self.ensure_runtime()
        with self._lock:
            record = self._tasks.get(record_id)
            if record is None:
                raise HTTPException(status_code=404, detail="Task not found")
            if record_id in self._resolving:
                raise HTTPException(status_code=409, detail="This action is already being processed")
            if record.status != TaskStatus.WAITING_FOR_CONFIRMATION or not record.task_id:
                raise HTTPException(status_code=409, detail="Task is not awaiting approval")
            # Claim the record and mark it busy immediately so a second click
            # cannot pass the guard above while the resume is running.
            self._resolving.add(record_id)
            record.status = TaskStatus.RUNNING.value
            record.updated_at = time.time()
            self._persist()
        try:
            assert self.brain is not None
            if approve:
                record.response = self.brain.approve_pending(record.task_id)
            else:
                record.response = self.brain.deny_pending(record.task_id)
            context = self.brain.last_context
            fallback = TaskStatus.COMPLETED.value if approve else TaskStatus.CANCELLED.value
            with self._lock:
                record.status = context.status.value if context is not None else fallback
                record.plan = _plan_snapshot(context)
                record.tool_calls = list(context.tool_calls) if context is not None else []
                record.errors = list(context.errors) if context is not None else []
                record.warnings = list(context.warnings) if context is not None else []
                record.web_sources = list(context.web_sources) if context is not None else []
                record.web_results = list(getattr(context, "metadata", {}).get("web_results", [])) if context is not None else []
                self._capture_reasoning(record, context)
                self._capture_feedback_state(record, context)
                record.updated_at = time.time()
                self._persist()
            return record
        finally:
            with self._lock:
                self._resolving.discard(record_id)
    def queue_snapshot(self) -> dict[str, Any]:
        # Return the running task id and the FIFO of pending task ids.
        with self._lock:
            return {"running": self._running_id, "pending": list(self._queue)}

    def _get(self, record_id: str) -> TaskRecord:
        with self._lock:
            record = self._tasks.get(record_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return record

    def _capture_reasoning(self, record: TaskRecord, context: Any) -> None:
        if context is None or context.task_id != record.task_id:
            return
        snapshot = _reasoning_snapshot(context)
        for key, value in snapshot.items():
            setattr(record, key, value)
        if record.reasoning:
            logger.info(
                "API reasoning snapshot: stages=%s mode=%s provenance=%s evidence_count=%s",
                ",".join(step["stage"] for step in record.reasoning),
                record.response_mode,
                ",".join(record.provenance),
                record.evidence["count"] if record.evidence else 0,
            )

    def _persist(self) -> None:
        self._task_store.save(self._tasks)

    # -- human feedback / experience loop -------------------------------------

    def _capture_feedback_state(self, record: TaskRecord, context: Any) -> None:
        """Decide whether this task shows feedback controls, and record why.

        The decision lives in the experience service so the rule (feedback for
        tool execution/research/application/multi-step/deliverables, not for
        casual conversation) has exactly one home. Nothing here calls a model.
        """

        if context is None:
            record.feedback_available = False
            return
        try:
            record.feedback_available = self.experience.should_offer_feedback(context)
            metadata = getattr(context, "metadata", {}) or {}
            recorded = bool(metadata.get("experience_recorded"))
            experience_id = str(metadata.get("experience_id") or "")
            if recorded:
                record.experience_id = experience_id
            elif record.feedback_available:
                # The task is eligible for feedback but the Brain did not record
                # it (for example the loop was disabled). Record it now so the
                # user's answer still has something durable to attach to.
                outcome = self.experience.record_task_outcome(context, record_id=record.id)
                record.experience_id = outcome.experience_id
        except Exception:  # noqa: BLE001 - feedback plumbing never breaks a task
            logger.exception("Could not evaluate feedback availability")
            record.feedback_available = False

    def submit_feedback(self, record_id: str, payload: FeedbackPayload) -> TaskRecord:
        """Attach explicit Success/Failed feedback to one task record.

        Feedback is changeable: submitting again updates the same experience
        rather than creating a duplicate, and the click itself is persisted first
        so the answer survives even if promotion is skipped.
        """

        self.ensure_runtime()
        with self._lock:
            record = self._tasks.get(record_id)
            if record is None:
                raise HTTPException(status_code=404, detail="Task not found")
            if record.status in {TaskStatus.PENDING.value, TaskStatus.RUNNING.value}:
                raise HTTPException(status_code=409, detail="Task is still running")
        request = FeedbackRequest.from_payload(
            record_id,
            {
                "outcome": payload.outcome,
                "reason": payload.reason,
                "failure_category": payload.failure_category,
                "correction": payload.correction,
                "expected_behavior": payload.expected_behavior,
                "response_quality": payload.response_quality,
            },
        )
        result = self.experience.apply_feedback(request)
        if not result.ok:
            raise HTTPException(status_code=409, detail=result.detail or "Feedback was not accepted")
        with self._lock:
            record.feedback_outcome = result.feedback
            record.feedback_category = result.failure_category
            record.feedback_category_label = result.failure_category_label or FAILURE_CATEGORY_LABELS.get(result.failure_category, "")
            record.feedback_reason = payload.reason
            record.feedback_correction = result.user_correction
            if result.experience_id:
                record.experience_id = result.experience_id
            record.updated_at = time.time()
            self._persist()
            return record

    def experience_status(self) -> dict[str, Any]:
        """Expose bounded experience-loop counts (no task content)."""

        status = self.experience.status()
        status["analysis"] = self.experience.analysis_report()
        return status


def _read_prompt(name: str) -> str:
    return (COMPUTER_ROOT / "prompts" / name).read_text(encoding="utf-8")


def _sse(event: str, data: dict[str, Any]) -> str:
    """Encode one Server-Sent Event.

    The payload is JSON on a single ``data:`` line, so the client never has to
    parse a multi-line frame and a partial write cannot corrupt a frame.
    """

    import json as _json

    return f"event: {event}\ndata: {_json.dumps(data, ensure_ascii=False)}\n\n"


def _streaming_model():
    """Return the real streaming model boundary, or ``None`` when unavailable.

    Streaming is only offered when the configured provider can actually stream.
    Returning ``None`` is honest: the turn is still served, just without live
    tokens, rather than showing progress that did not come from the model.
    """

    try:
        from llm import STREAMING_BOUNDARY

        if not STREAMING_BOUNDARY.supports_streaming():
            return None
        return STREAMING_BOUNDARY
    except Exception:  # noqa: BLE001 - no streaming is a supported state
        logger.exception("Streaming model unavailable; serving without token events")
        return None


def _plan_snapshot(context: Any) -> dict[str, Any] | None:
    if context is None or context.execution_plan is None:
        return None
    plan = context.execution_plan
    return {
        "id": plan.plan_id,
        "status": plan.status.value,
        "steps": [
            {
                "id": step.id,
                "name": step.name,
                "action": step.action,
                "description": step.description,
                "status": step.status.value,
                "metadata": {
                    key: value
                    for key, value in step.metadata.items()
                    if key not in {"parameters"}
                } | _safe_plan_parameters(step),
            }
            for step in plan.steps
        ],
    }


def _safe_plan_parameters(step: Any) -> dict[str, Any]:
    """Expose non-sensitive planning details needed by the control room."""

    if step.metadata.get("tool") != "powershell.execute":
        return {}
    parameters = step.metadata.get("parameters") or {}
    command = parameters.get("command") if isinstance(parameters, dict) else None
    return {"command": command} if isinstance(command, str) else {}


service = AtlasService()
app = FastAPI(title="Atlas Local API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "online", "model": OLLAMA_MODEL, "local": True, "api": "ready"}


@app.get("/api/system")
def system_info() -> dict[str, Any]:
    result = SystemInfoTool().execute({})
    if not result.success:
        raise HTTPException(status_code=503, detail=result.error)
    return result.output


@app.get("/api/tools")
def tools() -> dict[str, Any]:
    service.ensure_runtime()
    assert service.registry is not None
    return {
        "tools": [
            {
                "name": metadata.name,
                "description": metadata.description,
                "category": metadata.category,
                "permission_level": metadata.permission_level.value,
                "risk_level": metadata.risk_level.value,
            }
            for metadata in service.registry.list_metadata()
        ]
    }


@app.get("/api/tool-knowledge")
def tool_knowledge(query: str = "", limit: int = 50) -> dict[str, Any]:
    """Expose the data-driven tool catalog used by planner discovery."""

    records = load_json(COMPUTER_ROOT / "tools" / "powershell_commands.json")
    store = ToolKnowledgeStore(records)
    bounded_limit = min(max(limit, 1), 100)
    selected = store.search(query, tool_type="powershell", limit=bounded_limit) if query.strip() else store.list_records()[:bounded_limit]
    return {"tools": [record.to_dict() for record in selected]}


@app.get("/api/tool-discovery")
def tool_discovery(query: str, limit: int = 5) -> dict[str, Any]:
    """Return structured candidates without executing a command."""

    records = load_json(COMPUTER_ROOT / "tools" / "powershell_commands.json")
    candidates = ToolDiscovery(ToolKnowledgeStore(records)).discover(
        query,
        tool_type="powershell",
        limit=min(max(limit, 1), 20),
    )
    return {"candidates": [candidate.__dict__ for candidate in candidates]}


@app.get("/api/applications")
def applications() -> dict[str, Any]:
    result = InstalledApplicationsTool().execute({})
    if not result.success:
        raise HTTPException(status_code=503, detail=result.error)
    return result.output


@app.get("/api/tasks")
def tasks() -> dict[str, Any]:
    with service._lock:
        return {"tasks": list(service._tasks.values())}

@app.get("/api/queue")
def queue() -> dict[str, Any]:
    # Expose the single-worker task queue so the UI can show what is waiting.
    return service.queue_snapshot()


@app.post("/api/tasks", status_code=202)
def create_task(payload: TaskRequest) -> TaskRecord:
    return service.submit(payload.request.strip())


@app.get("/api/tasks/{record_id}")
def task(record_id: str) -> TaskRecord:
    return service._get(record_id)


@app.post("/api/tasks/{record_id}/approve")
def approve_task(record_id: str) -> TaskRecord:
    return service.approve(record_id)


@app.post("/api/tasks/{record_id}/feedback")
def submit_feedback(record_id: str, payload: FeedbackPayload) -> TaskRecord:
    """Record explicit Success/Failed feedback for a completed task.

    This is what turns a task into a learning signal: the answer is persisted
    durably, attached to the specific task result, and used to promote or correct
    the corresponding experience.
    """

    return service.submit_feedback(record_id, payload)


@app.get("/api/experience")
def experience_status() -> dict[str, Any]:
    """Expose bounded experience-loop counts and candidate (unapplied) findings."""

    return service.experience_status()


@app.post("/api/tasks/{record_id}/deny")
def deny_task(record_id: str) -> TaskRecord:
    return service.deny(record_id)


# ---------------------------------------------------------------------------
# Conversational API
#
# These endpoints expose the Conversation Runtime. They deliberately do NOT
# duplicate planning, retrieval or execution: a conversational turn is served by
# the runtime, which delegates anything consequential to the same Brain the task
# endpoints already use.
# ---------------------------------------------------------------------------


@app.get("/api/conversations")
def conversations() -> dict[str, Any]:
    return {"conversations": service.conversations()}


@app.post("/api/conversations", status_code=201)
def create_conversation(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    title = None
    if isinstance(payload, dict):
        raw = payload.get("title")
        if isinstance(raw, str) and raw.strip():
            title = raw.strip()[:200]
    return service.new_conversation(title=title)


@app.get("/api/conversations/search")
def search_conversations(query: str, limit: int = 20) -> dict[str, Any]:
    if not query.strip():
        return {"results": []}
    return {"results": service.search_conversations(query.strip(), limit=limit)}


@app.get("/api/conversations/{conversation_id}/messages")
def conversation_messages(conversation_id: str) -> dict[str, Any]:
    return {"messages": service.conversation_messages(conversation_id)}


@app.patch("/api/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, payload: ConversationRenamePayload) -> dict[str, Any]:
    return service.rename_conversation(conversation_id, payload.title.strip())


@app.delete("/api/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str) -> None:
    service.delete_conversation(conversation_id)


@app.post("/api/messages", status_code=202)
def send_message(payload: MessagePayload) -> dict[str, Any]:
    """Serve one conversational turn without streaming.

    Streaming clients use ``POST /api/messages/stream`` and receive the same
    turn as real Server-Sent Events.
    """

    return service.send_message(
        message=payload.message.strip(),
        conversation_id=payload.conversation_id,
        attachments=payload.attachments,
    )


@app.post("/api/messages/stream")
def stream_message(payload: MessagePayload) -> StreamingResponse:
    """Serve one conversational turn as a real Server-Sent Event stream.

    Every event corresponds to something that actually happened: a streamed
    token from the model, a real tool invocation, a real execution state. The
    final ``assistant_completed``/``cancelled``/``error`` event carries the
    persisted turn, so a reconnecting client never has to guess what occurred.
    """

    if not ENABLE_CONVERSATION_STREAMING:
        raise HTTPException(status_code=503, detail="Conversation streaming is disabled")

    turn_id = str(uuid4())
    token = CancellationToken()
    # Register before the generator starts so an immediate cancel can find it.
    service._turn_tokens[turn_id] = token  # noqa: SLF001 - same service instance

    def events():
        events_queue: "Queue[Any]" = Queue()
        finished = object()
        outcome: dict[str, Any] = {}

        def sink(event: Any) -> None:
            events_queue.put(event.to_dict())

        def run_turn() -> None:
            try:
                outcome["result"] = service.send_message(
                    message=payload.message.strip(),
                    conversation_id=payload.conversation_id,
                    attachments=payload.attachments,
                    events=sink,
                    stream_model=True,
                    turn_id=turn_id,
                    token=token,
                )
            except Exception as exc:  # noqa: BLE001 - reported to the client honestly
                logger.exception("Conversational turn failed")
                outcome["error"] = exc
            finally:
                events_queue.put(finished)

        worker = threading.Thread(target=run_turn, name=f"atlas-turn-{turn_id[:8]}", daemon=True)
        worker.start()

        yield _sse("open", {"turn_id": turn_id})
        try:
            while True:
                event = events_queue.get()
                if event is finished:
                    break
                # Forwarded the moment it was produced: a token or a tool call
                # reaches the client while Atlas is still working.
                yield _sse(event.get("type", "event"), event.get("data", {}))
        finally:
            with service._lock:  # noqa: SLF001 - same service instance
                service._turn_tokens.pop(turn_id, None)  # noqa: SLF001
            # A client that disconnects mid-turn stops reading. The token is
            # raised so the worker stops at its next safe checkpoint instead of
            # finishing work nobody is waiting for.
            token.cancel("client disconnected")

        if "error" in outcome:
            yield _sse("error", {"detail": str(outcome["error"]), "turn_id": turn_id})
            return
        result = outcome.get("result", {})
        state = result.get("execution_state")
        terminal = {
            "completed": "assistant_completed",
            "cancelled": "cancelled",
            "failed": "error",
        }.get(str(state), "assistant_completed")
        yield _sse(terminal, {"turn_id": turn_id, **result})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/turns/{turn_id}/cancel")
def cancel_turn(turn_id: str) -> dict[str, Any]:
    """Cancel an in-flight conversational turn.

    This raises a real cooperative token that the streaming answer loop, the
    reasoning engine, the Brain (after interpretation and after planning), and
    the executor (before every step) all consult; the remaining steps are
    skipped and the turn is persisted as ``cancelled``. A turn that had already
    finished is still reported as completed. An in-flight blocking call is not
    force-killed, which is stated honestly rather than claimed.
    """

    cancelled = service.cancel_turn(turn_id)
    if not cancelled:
        raise HTTPException(status_code=404, detail="No such running turn")
    return {"turn_id": turn_id, "cancelled": True}


# ---------------------------------------------------------------------------
# Long-term memory API (view / search / edit / delete / clear / disable)
# ---------------------------------------------------------------------------


@app.get("/api/memory")
def memory_records(query: str = "", limit: int = 50) -> dict[str, Any]:
    if query.strip():
        return {"memories": service.search_memory(query.strip(), limit=limit), **service.memory_status()}
    return {"memories": service.memory_records(), **service.memory_status()}


@app.post("/api/memory", status_code=201)
def create_memory(payload: MemoryPayload) -> dict[str, Any]:
    service.ensure_runtime()
    assert service.user_memory is not None
    record = service.user_memory.create(text=payload.text, kind=payload.kind, source="user_explicit")
    if record is None:
        raise HTTPException(status_code=400, detail="That memory was not stored")
    return record.to_dict()


@app.get("/api/memory/status")
def memory_status() -> dict[str, Any]:
    return service.memory_status()


@app.post("/api/memory/enabled")
def set_memory_enabled(payload: MemoryTogglePayload) -> dict[str, Any]:
    return service.set_memory_enabled(payload.enabled)


@app.patch("/api/memory/{memory_id}")
def update_memory(memory_id: str, payload: MemoryUpdatePayload) -> dict[str, Any]:
    return service.update_memory(memory_id, text=payload.text, kind=payload.kind)


@app.delete("/api/memory/{memory_id}", status_code=204)
def delete_memory(memory_id: str) -> None:
    service.delete_memory(memory_id)


@app.post("/api/memory/clear")
def clear_memory() -> dict[str, Any]:
    return service.clear_memory()
