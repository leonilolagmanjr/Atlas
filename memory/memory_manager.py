"""MemoryManager — first-class conversational memory façade."""

from __future__ import annotations

import logging
from typing import Optional

from config import AUTO_SAVE, MAX_RETAINED_MESSAGES, MEMORY_FOLDER, AUTO_SUMMARIZE_THRESHOLD
from memory.context_builder import ContextBuilder
from memory.models import ConversationSessionMetadata, MemoryMessage
from memory.session_manager import SessionManager
from memory.storage import ConversationStore
from memory.summarizer import MemorySummarizer

logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self) -> None:
        self._store = ConversationStore(root_folder=MEMORY_FOLDER)
        self._sessions = SessionManager(store=self._store, root_folder=MEMORY_FOLDER, auto_save=AUTO_SAVE)
        self._context_builder = ContextBuilder()
        self._summarizer = MemorySummarizer()

    # ---- Session APIs (future frontend entry points) ----
    def create_session(self, *, title: str | None = None) -> ConversationSessionMetadata:
        return self._sessions.create_session(title=title)

    def list_sessions(self) -> list[ConversationSessionMetadata]:
        return self._sessions.list_sessions()

    def open_session(self, session_id: str) -> ConversationSessionMetadata:
        # Ensure exists, then switch active
        meta = self._sessions.load_session(session_id)
        self._sessions.switch_active_session(session_id)
        return meta

    def delete_session(self, session_id: str) -> None:
        self._sessions.delete_session(session_id)

    def rename_session(self, session_id: str, *, new_title: str) -> None:
        self._sessions.rename_session(session_id, new_title=new_title)

    def archive_session(self, session_id: str) -> None:
        self._sessions.archive_session(session_id)

    def get_active_session_id(self) -> str | None:
        return self._sessions.get_active_session_id()

    def get_active_metadata(self) -> ConversationSessionMetadata | None:
        sid = self.get_active_session_id()
        if not sid:
            return None
        return self._sessions.load_session(sid)

    # ---- Message APIs ----
    def _get_or_create_active_session(self) -> ConversationSessionMetadata:
        active = self.get_active_session_id()
        if active:
            return self._sessions.load_session(active)
        return self.create_session(title="Untitled")

    def append_message(
        self,
        *,
        role: str,
        content: str,
        turn_number: int | None = None,
        metadata: Optional[dict] = None,
    ) -> MemoryMessage:
        metadata = metadata or {}
        session = self._get_or_create_active_session()
        messages = self._store.load_messages(session.id)
        turn = turn_number if turn_number is not None else (messages[-1].turn_number + 1 if messages else 1)

        msg = MemoryMessage(
            role=role,
            content=content,
            turn_number=int(turn),
            metadata=metadata,
        )
        messages.append(msg)

        # Update session metadata
        session.message_count = len(messages)
        session.touch()
        self._store.save_messages(session.id, messages)
        # Summarizer placeholder interface
        if AUTO_SUMMARIZE_THRESHOLD and session.message_count >= AUTO_SUMMARIZE_THRESHOLD:
            session.summary = self._summarizer.maybe_summarize(
                session_id=session.id,
                message_count=session.message_count,
                existing_summary=session.summary,
            )
        self._store.save_metadata(session.id, session)

        if session.summary is not None:
            self._store.save_summary(session.id, session.summary)

        logger.info(
            "Memory message appended: session_id=%s role=%s turn=%d total_messages=%d",
            session.id,
            role,
            msg.turn_number,
            session.message_count,
        )
        return msg

    def get_recent_messages(self, *, session_id: str | None = None) -> list[MemoryMessage]:
        sid = session_id or self.get_active_session_id()
        if not sid:
            return []
        return self._store.load_messages(sid)

    def build_conversation_history_for_prompt(self, *, session_id: str | None = None) -> str:
        sid = session_id or self.get_active_session_id()
        if not sid:
            return ""
        messages = self._store.load_messages(sid)
        return self._context_builder.build_prompt_context(messages=messages)

    # ---- Export/Import (future) ----
    # For now, provide architecture hooks.

