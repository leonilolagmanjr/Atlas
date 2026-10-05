"""Persistence for conversation sessions.

Conversations used to be a tree of files::

    memory/sessions/<id>/metadata.json
    memory/sessions/<id>/messages.json
    memory/sessions/<id>/summary.txt

They are now rows in the authoritative ``atlas.db`` (``conversations`` and
``messages`` tables). This class is a thin, model-aware façade over
:class:`persistence.repositories.SqliteConversationRepository`, so the rest of
the memory subsystem keeps its existing vocabulary while the storage moved.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from memory.models import ConversationSessionMetadata, MemoryMessage
from persistence.context import database_for_legacy_path, default_persistence
from persistence.database import AtlasDatabase
from persistence.repositories import SqliteConversationRepository

logger = logging.getLogger(__name__)


class ConversationStore:
    """Durable conversations and transcripts, backed by ``atlas.db``."""

    def __init__(
        self,
        *,
        root_folder: Path | str | None = None,
        repository: SqliteConversationRepository | None = None,
        database: AtlasDatabase | None = None,
    ) -> None:
        self._owned_database = database
        self._owns_database = False
        if repository is not None:
            self._repository = repository
        elif database is not None:
            self._repository = SqliteConversationRepository(database)
        elif root_folder is not None:
            self._owned_database = database_for_legacy_path(root_folder)
            self._owns_database = True
            self._repository = SqliteConversationRepository(self._owned_database)
        else:
            context = default_persistence()
            self._owned_database = context.database
            self._repository = context.conversations

    @property
    def repository(self) -> SqliteConversationRepository:
        return self._repository

    def close(self) -> None:
        """Release this store's SQLite connection when it owns the database."""

        if self._owns_database and self._owned_database is not None:
            self._owned_database.close()

    # -- conversations ---------------------------------------------------------

    def list_sessions(self) -> list[str]:
        return self._repository.list_conversation_ids()

    def load_metadata(self, session_id: str) -> ConversationSessionMetadata:
        stored = self._repository.get_conversation(session_id)
        if stored is None:
            raise FileNotFoundError(f"No stored conversation {session_id!r}")
        return ConversationSessionMetadata.from_dict(stored)

    def save_metadata(self, session_id: str, metadata: ConversationSessionMetadata) -> None:
        self._repository.save_conversation(session_id, metadata.to_dict())

    # -- messages --------------------------------------------------------------

    def load_messages(self, session_id: str) -> list[MemoryMessage]:
        return [MemoryMessage.from_dict(row) for row in self._repository.list_messages(session_id)]

    def save_messages(self, session_id: str, messages: list[MemoryMessage]) -> None:
        self._repository.save_messages(session_id, [message.to_dict() for message in messages])

    def update_message(
        self, session_id: str, message_id: str, **changes: Any
    ) -> MemoryMessage | None:
        """Apply field changes to one stored message (used for live states).

        Fields not listed here are ignored rather than silently injected, so an
        arbitrary caller cannot corrupt the stored shape.
        """

        allowed = {
            "role", "content", "attachments", "tool_calls", "tool_results",
            "citations", "execution_state", "metadata",
        }
        payload = {
            key: value
            for key, value in changes.items()
            if key in allowed and value is not None
        }
        updated = self._repository.update_message(session_id, message_id, payload)
        return MemoryMessage.from_dict(updated) if updated is not None else None

    # -- summary ---------------------------------------------------------------

    def load_summary(self, session_id: str) -> str | None:
        stored = self._repository.get_conversation(session_id)
        return stored.get("summary") if stored is not None else None

    def save_summary(self, session_id: str, summary: str | None) -> None:
        if summary is None:
            return
        metadata = self.load_metadata(session_id)
        metadata.summary = summary
        metadata.touch()
        self.save_metadata(session_id, metadata)

    def delete_session(self, session_id: str) -> None:
        self._repository.delete_conversation(session_id)
