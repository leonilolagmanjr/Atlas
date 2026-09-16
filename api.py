"""Local HTTP API for the Atlas frontend.

This adapter exposes existing runtime capabilities; it does not reimplement
planning, retrieval, memory, or tool execution in the web layer.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from brain import Brain
from computer.applications import InstalledApplicationsTool
from computer.runtime import register_read_only_tools
from computer.system import SystemInfoTool
from config import COMPUTER_ROOT, EXECUTION_MODE, OLLAMA_MODEL, TASK_STORE_FILE
from indexer import index_knowledge_base
from memory.memory_manager import MemoryManager
from models import TaskStatus
from task_store import TaskStore
from tools import ExecutionMode, PermissionEngine, ToolRegistry, ToolRouter
from tools.discovery import ToolDiscovery
from tools.knowledge import ToolKnowledgeStore, load_json
from vector_store import VectorStore

logger = logging.getLogger(__name__)


class TaskRequest(BaseModel):
    request: str = Field(min_length=1, max_length=4000)


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


@dataclass
class AtlasService:
    brain: Brain | None = None
    registry: ToolRegistry | None = None
    _tasks: dict[str, TaskRecord] = field(default_factory=dict)
    _task_store: TaskStore = field(default_factory=lambda: TaskStore(path=TASK_STORE_FILE))
    _executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=1))
    _lock: threading.RLock = field(default_factory=threading.RLock)

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
            register_read_only_tools(registry, root=COMPUTER_ROOT)
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
            )

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
            self._persist()
        self._executor.submit(self._run, record.id)
        return record

    def _run(self, record_id: str) -> None:
        with self._lock:
            record = self._tasks[record_id]
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
        self.ensure_runtime()
        record = self._get(record_id)
        if record.status != TaskStatus.WAITING_FOR_CONFIRMATION or not record.task_id:
            raise HTTPException(status_code=409, detail="Task is not awaiting approval")
        assert self.brain is not None
        record.response = self.brain.approve_pending(record.task_id)
        context = self.brain.last_context
        record.status = context.status.value if context is not None else TaskStatus.COMPLETED.value
        record.plan = _plan_snapshot(context)
        record.tool_calls = list(context.tool_calls) if context is not None else []
        record.errors = list(context.errors) if context is not None else []
        record.warnings = list(context.warnings) if context is not None else []
        record.updated_at = time.time()
        with self._lock:
            self._persist()
        return record

    def deny(self, record_id: str) -> TaskRecord:
        self.ensure_runtime()
        record = self._get(record_id)
        if record.status != TaskStatus.WAITING_FOR_CONFIRMATION or not record.task_id:
            raise HTTPException(status_code=409, detail="Task is not awaiting approval")
        assert self.brain is not None
        record.response = self.brain.deny_pending(record.task_id)
        context = self.brain.last_context
        record.status = context.status.value if context is not None else TaskStatus.CANCELLED.value
        record.plan = _plan_snapshot(context)
        record.errors = list(context.errors) if context is not None else []
        record.warnings = list(context.warnings) if context is not None else []
        record.updated_at = time.time()
        with self._lock:
            self._persist()
        return record

    def _get(self, record_id: str) -> TaskRecord:
        with self._lock:
            record = self._tasks.get(record_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return record

    def _persist(self) -> None:
        self._task_store.save(self._tasks)


def _read_prompt(name: str) -> str:
    return (COMPUTER_ROOT / "prompts" / name).read_text(encoding="utf-8")


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


@app.post("/api/tasks", status_code=202)
def create_task(payload: TaskRequest) -> TaskRecord:
    return service.submit(payload.request.strip())


@app.get("/api/tasks/{record_id}")
def task(record_id: str) -> TaskRecord:
    return service._get(record_id)


@app.post("/api/tasks/{record_id}/approve")
def approve_task(record_id: str) -> TaskRecord:
    return service.approve(record_id)


@app.post("/api/tasks/{record_id}/deny")
def deny_task(record_id: str) -> TaskRecord:
    return service.deny(record_id)
