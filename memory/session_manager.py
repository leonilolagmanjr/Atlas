"""Session manager.

Tracks session metadata and active session pointer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from config import MAX_SESSIONS
from memory.models import ConversationSessionMetadata
from memory.storage import ConversationStore

logger = logging.getLogger(__name__)


@dataclass
class ActiveSession:
    session_id: str


class SessionManager:
    def __init__(
        self,
        *,
        store: ConversationStore,
        root_folder: Path | None = None,
        auto_save: bool = True,
    ) -> None:
        self._store = store
        # ``root_folder`` is retained for backward compatibility but is no longer
        # used: the active-conversation pointer now lives in atlas.db alongside
        # the conversations it points at.
        self._root_folder = root_folder
        self._auto_save = auto_save

    def get_active_session_id(self) -> str | None:
        return self._store.repository.get_active_conversation_id()

    def create_session(self, *, title: str | None = None) -> ConversationSessionMetadata:
        # Best-effort session cap
        existing = self._store.list_sessions()
        if MAX_SESSIONS and len(existing) >= MAX_SESSIONS:
            # We do not implement LRU yet; just proceed to create.
            logger.warning("MAX_SESSIONS reached; creating additional session")

        session_id = str(uuid4())
        metadata = ConversationSessionMetadata(
            id=session_id,
            title=title or "Untitled",
        )
        self._store.save_metadata(session_id, metadata)
        self.switch_active_session(session_id)
        logger.info("Memory session created: session_id=%s title=%s", session_id, metadata.title)
        return metadata

    def load_session(self, session_id: str) -> ConversationSessionMetadata:
        metadata = self._store.load_metadata(session_id)
        logger.info("Memory session loaded: session_id=%s title=%s", session_id, metadata.title)
        return metadata

    def switch_active_session(self, session_id: str) -> None:
        self._store.repository.set_active_conversation_id(session_id)
        logger.info("Memory session switched: session_id=%s", session_id)

    def rename_session(self, session_id: str, *, new_title: str) -> None:
        metadata = self._store.load_metadata(session_id)
        metadata.title = new_title
        metadata.touch()
        self._store.save_metadata(session_id, metadata)
        logger.info("Memory session renamed: session_id=%s title=%s", session_id, new_title)

    def archive_session(self, session_id: str) -> None:
        metadata = self._store.load_metadata(session_id)
        metadata.archived = True
        metadata.touch()
        self._store.save_metadata(session_id, metadata)
        logger.info("Memory session archived: session_id=%s", session_id)

    def delete_session(self, session_id: str) -> None:
        self._store.delete_session(session_id)
        if self.get_active_session_id() == session_id:
            # The pointer no longer refers to an existing conversation.
            try:
                self._store.repository.clear_active_conversation_id()
            except Exception:
                logger.exception("Failed clearing active session pointer")
        logger.info("Memory session deleted: session_id=%s", session_id)

    def list_sessions(self) -> list[ConversationSessionMetadata]:
        out: list[ConversationSessionMetadata] = []
        for session_id in self._store.list_sessions():
            try:
                out.append(self._store.load_metadata(session_id))
            except Exception:
                logger.exception("Failed loading metadata for session_id=%s", session_id)
        out.sort(key=lambda m: m.last_modified, reverse=True)
        return out

