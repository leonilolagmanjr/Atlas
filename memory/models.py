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


@dataclass
class MemoryMessage:
    """A single conversational turn message."""

    id: str = field(default_factory=lambda: str(uuid4()))
    role: str = "user"  # "user" | "assistant"
    content: str = ""
    timestamp: datetime = field(default_factory=utcnow)
    turn_number: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "turn_number": self.turn_number,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "MemoryMessage":
        ts = d.get("timestamp")
        timestamp = utcnow() if not ts else datetime.fromisoformat(ts)
        return MemoryMessage(
            id=d.get("id") or str(uuid4()),
            role=d.get("role") or "user",
            content=d.get("content") or "",
            timestamp=timestamp,
            turn_number=int(d.get("turn_number") or 0),
            metadata=d.get("metadata") or {},
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
        )

