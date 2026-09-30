"""Repository contracts for Atlas's authoritative structured state.

These interfaces are the *only* persistence vocabulary the rest of Atlas sees.
The Brain, the Conversation Runtime, the experience loop, the planner and the API
all speak in terms of conversations, tasks, memories and experiences; none of
them knows whether those live in SQLite, in a file, or somewhere else.

Each contract is intentionally grouped by domain rather than one interface per
table. Feedback belongs to the experience loop; the active-conversation pointer
belongs to the conversation store.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ConversationRepository(Protocol):
    """Durable conversations, their messages, and the active-conversation pointer."""

    def list_conversation_ids(self) -> list[str]: ...

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        """Return stored metadata, or ``None`` when the conversation is unknown."""

    def create_conversation(self, conversation_id: str, title: str) -> dict[str, Any]:
        """Insert a new conversation and return its stored metadata."""

    def save_conversation(self, conversation_id: str, metadata: dict[str, Any]) -> None:
        """Replace the stored metadata for an existing conversation."""

    def delete_conversation(self, conversation_id: str) -> None: ...

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        """Return every message of a conversation in transcript order."""

    def save_messages(self, conversation_id: str, messages: list[dict[str, Any]]) -> None:
        """Replace the conversation's transcript, preserving message ids."""

    def update_message(
        self, conversation_id: str, message_id: str, changes: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Apply field changes to one message; ``None`` when it does not exist."""

    def get_active_conversation_id(self) -> str | None: ...

    def set_active_conversation_id(self, conversation_id: str) -> None: ...

    def clear_active_conversation_id(self) -> None: ...


@runtime_checkable
class TaskRepository(Protocol):
    """Durable API task snapshots.

    Task *execution* stays with the Brain and Executor. This repository only
    persists the API-visible snapshot so task history survives a restart.
    """

    def list_tasks(self) -> list[dict[str, Any]]:
        """Return every stored task, most recently updated first."""

    def get_task(self, task_id: str) -> dict[str, Any] | None: ...

    def save_tasks(self, records: dict[str, dict[str, Any]]) -> None:
        """Persist the given tasks and drop any task no longer present."""

    def clear(self) -> None: ...


@runtime_checkable
class UserMemoryRepository(Protocol):
    """Durable long-term user memory."""

    def list_memories(self) -> list[dict[str, Any]]:
        """Return every stored memory in insertion order."""

    def get_memory(self, memory_id: str) -> dict[str, Any] | None: ...

    def add_memory(self, memory: dict[str, Any]) -> dict[str, Any]:
        """Append one memory and return it as stored."""

    def update_memory(self, memory_id: str, changes: dict[str, Any]) -> dict[str, Any] | None: ...

    def delete_memory(self, memory_id: str) -> bool: ...

    def clear_memories(self) -> int:
        """Remove every memory, returning how many were removed."""


@runtime_checkable
class ExperienceRepository(Protocol):
    """Durable experience records."""

    def list_experiences(self) -> list[dict[str, Any]]:
        """Return every stored experience in insertion order."""

    def get_experience(self, experience_id: str) -> dict[str, Any] | None: ...

    def add_experience(self, experience: dict[str, Any]) -> bool: ...

    def replace_experience(self, experience: dict[str, Any]) -> bool:
        """Overwrite an existing experience in place."""

    def delete_experiences_below_ordinal(self, keep_from: int) -> int:
        """Bound the store by dropping the oldest records; returns how many."""

    def clear_experiences(self) -> None: ...


@runtime_checkable
class FeedbackRepository(Protocol):
    """Durable user feedback clicks, recorded before any experience is updated."""

    def list_feedback(self) -> list[dict[str, Any]]:
        """Return every feedback event in the order it was recorded."""

    def record_feedback(self, event: dict[str, Any]) -> bool: ...

    def latest_feedback(self, record_id: str) -> dict[str, Any] | None:
        """Return the most recent feedback for a task record, if any."""

    def clear_feedback(self) -> None: ...


@runtime_checkable
class IndexStateRepository(Protocol):
    """Structured metadata about which documents are indexed.

    The documents themselves stay on the filesystem. Chroma owns the vectors.
    This repository owns the "is this document current?" question.
    """

    def get_hashes(self) -> dict[str, str]:
        """Return the stored document id -> content hash mapping."""

    def set_hash(self, doc_id: str, content_hash: str, *, path: str = "") -> None: ...

    def remove(self, doc_id: str) -> None: ...

    def describe(self) -> list[dict[str, Any]]:
        """Return per-document metadata (used for diagnostics)."""


@runtime_checkable
class CorrectionMemoryRepository(Protocol):
    """Durable structured interpretation lessons."""

    def list_lessons(self) -> list[dict[str, Any]]: ...

    def add_lesson(self, lesson: dict[str, Any]) -> bool:
        """Append one lesson. False when an identical lesson already exists."""

    def increment_hits(self, lesson_type: str, pattern: str) -> None: ...
