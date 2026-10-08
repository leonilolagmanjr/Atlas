"""Memory models.

These are independent of Atlas execution models so the memory subsystem can
evolve without coupling to Brain/Executor internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def system_now() -> datetime:
    """Return the current local datetime with timezone information.

    This is the authoritative system clock for Atlas.  It prefers the host's
    local timezone so relative temporal expressions (``today``, ``last night``)
    resolve correctly for the user's locale.  Fallback is UTC.
    """
    utc = utcnow()
    try:
        return utc.astimezone()
    except Exception:
        return utc.replace(tzinfo=timezone.utc)


#: Execution states a persisted assistant turn can carry. These are the
#: *actual* recorded states, not display conveniences: a cancelled turn stays
#: ``cancelled`` even after a restart.
EXECUTION_STATES: frozenset[str] = frozenset(
    {"completed", "failed", "cancelled", "waiting_for_confirmation", "running", "unknown"}
)

#: How a turn was served. Conversation is not a task; a message records which
#: path actually produced it so the UI can show real activity rather than a
#: fabricated one.
RESPONSE_KINDS: frozenset[str] = frozenset(
    {
        "conversation",   # direct conversational answer, no task built
        "retrieval",      # local knowledge retrieval
        "research",       # web research
        "task",           # multi-step validated task
        "computer",       # computer/application action
        "hybrid",         # retrieval/research + action
        "clarification",  # Atlas asked instead of guessing
        "memory",         # answered from conversation memory and/or user memory
        "system",         # local machine inspection
        "error",
    }
)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return [item for item in value]
    if isinstance(value, tuple):
        return list(value)
    return []


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


@dataclass
class ConversationalClaim:
    """An assistant factual claim with verification status.

    Priority 3: this reuses the existing conversation record rather than creating a
    second memory system.
    """

    text: str = ""
    turn_id: int = 0
    claim_status: str = "unverified"
    source_evidence: Any | None = None
    fresh_until: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "turn_id": self.turn_id,
            "claim_status": self.claim_status,
            "source_evidence": self.source_evidence.to_dict() if hasattr(self.source_evidence, "to_dict") else self.source_evidence,
            "fresh_until": self.fresh_until.isoformat() if self.fresh_until else None,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ConversationalClaim":
        fresh_until = data.get("fresh_until")
        return ConversationalClaim(
            text=str(data.get("text") or ""),
            turn_id=int(data.get("turn_id") or 0),
            claim_status=str(data.get("claim_status") or "unverified"),
            source_evidence=data.get("source_evidence"),
            fresh_until=datetime.fromisoformat(fresh_until) if fresh_until else None,
        )


@dataclass
class MemoryMessage:
    """A single conversational turn message.

    A message is not a task result. One assistant message may carry tool calls,
    citations, an execution state, and a conversation-level response kind, so a
    whole conversation (questions, research, tasks, computer actions, follow-ups)
    is reconstructable from the stored history exactly as it happened.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    role: str = "user"  # "user" | "assistant"
    content: str = ""
    timestamp: datetime = field(default_factory=utcnow)
    turn_number: int = 0
    #: Uploaded/referenced inputs attached to this turn (name, kind, path, ...).
    attachments: list[dict[str, Any]] = field(default_factory=list)
    #: Tool invocations performed while producing this message.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    #: Results/observations returned by those tools.
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    #: Sources cited in the answer.
    citations: list[str] = field(default_factory=list)
    #: Temporal claims emitted during the assistant turn.
    claims: list[ConversationalClaim] = field(default_factory=list)
    #: completed | failed | cancelled | waiting_for_confirmation | running | unknown
    execution_state: str = "completed"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def response_kind(self) -> str:
        kind = str(self.metadata.get("response_kind") or "")
        return kind if kind in RESPONSE_KINDS else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "turn_number": self.turn_number,
            "attachments": self.attachments,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "citations": self.citations,
            "claims": [claim.to_dict() for claim in self.claims],
            "execution_state": self.execution_state,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "MemoryMessage":
        ts = d.get("timestamp")
        timestamp = utcnow() if not ts else datetime.fromisoformat(ts)
        state = str(d.get("execution_state") or "completed")
        claims = [
            ConversationalClaim.from_dict(item) if isinstance(item, dict) else ConversationalClaim(str(item))
            for item in _as_list(d.get("claims"))
        ]
        return MemoryMessage(
            id=d.get("id") or str(uuid4()),
            role=d.get("role") or "user",
            content=d.get("content") or "",
            timestamp=timestamp,
            turn_number=int(d.get("turn_number") or 0),
            attachments=_as_list(d.get("attachments")),
            tool_calls=_as_list(d.get("tool_calls")),
            tool_results=_as_list(d.get("tool_results")),
            citations=[str(item) for item in _as_list(d.get("citations")) if item],
            claims=claims,
            execution_state=state if state in EXECUTION_STATES else "unknown",
            metadata=_as_dict(d.get("metadata")),
        )


@dataclass
class ConversationSessionMetadata:
    id: str
    title: str
    created_at: datetime = field(default_factory=utcnow)
    last_modified: datetime = field(default_factory=utcnow)
    message_count: int = 0
    summary: str | None = None
    archived: bool = False
    #: Model that produced the assistant turns in this conversation.
    model: str = ""
    #: Free-form conversation metadata (topic, tags, source counts, ...).
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.last_modified = utcnow()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "message_count": self.message_count,
            "summary": self.summary,
            "archived": self.archived,
            "model": self.model,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ConversationSessionMetadata":
        created_at = datetime.fromisoformat(d["created_at"]) if d.get("created_at") else utcnow()
        last_modified = (
            datetime.fromisoformat(d["last_modified"]) if d.get("last_modified") else utcnow()
        )
        return ConversationSessionMetadata(
            id=d["id"],
            title=d.get("title") or "Untitled",
            created_at=created_at,
            last_modified=last_modified,
            message_count=int(d.get("message_count") or 0),
            summary=d.get("summary"),
            archived=bool(d.get("archived") or False),
            model=str(d.get("model") or ""),
            metadata=_as_dict(d.get("metadata")),
        )

