"""MemoryManager — first-class conversational memory façade."""

from __future__ import annotations

import logging
from typing import Optional

from config import AUTO_SAVE
from memory.context_builder import ContextBuilder
from memory.models import (
    EXECUTION_STATES,
    ConversationSessionMetadata,
    MemoryMessage,
)
from memory.session_manager import SessionManager
from memory.storage import ConversationStore
from memory.summarizer import MemorySummarizer

logger = logging.getLogger(__name__)


def _excerpt(content: str, needle: str, *, window: int = 160) -> str:
    """Return a bounded excerpt of ``content`` centred on ``needle``."""

    lowered = content.casefold()
    index = lowered.find(needle)
    if index < 0:
        return content[:window]
    start = max(0, index - window // 2)
    return ("..." if start else "") + content[start:start + window].strip()


class MemoryManager:
    def __init__(self, *, ask=None, index=None) -> None:
        from config import ENABLE_LLM_INTERPRETATION

        self._store = ConversationStore()
        self._sessions = SessionManager(store=self._store, auto_save=AUTO_SAVE)
        self._context_builder = ContextBuilder()
        # Real summarization: the model boundary is injectable and optional.
        if ask is None and ENABLE_LLM_INTERPRETATION:
            try:
                from llm import ask as default_ask

                ask = default_ask
            except Exception:  # noqa: BLE001 - no model is a supported state
                ask = None
        self._summarizer = MemorySummarizer(ask=ask, model_available=ask is not None)
        # Conversation retrieval index (separate collection, optional embeddings).
        if index is None:
            from memory.conversation_index import ConversationIndex

            index = ConversationIndex()
        self._index = index

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
        # A deleted conversation must not remain recallable from the index.
        try:
            self._index.forget_conversation(session_id)
        except Exception:  # noqa: BLE001
            logger.exception("Could not remove conversation from the index")

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
        attachments: Optional[list[dict]] = None,
        tool_calls: Optional[list[dict]] = None,
        tool_results: Optional[list[dict]] = None,
        citations: Optional[list[str]] = None,
        execution_state: str = "completed",
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
            attachments=list(attachments or []),
            tool_calls=list(tool_calls or []),
            tool_results=list(tool_results or []),
            citations=[str(item) for item in (citations or []) if item],
            execution_state=execution_state if execution_state in EXECUTION_STATES else "unknown",
        )
        messages.append(msg)

        # Update session metadata
        session.message_count = len(messages)
        session.touch()
        self._store.save_messages(session.id, messages)
        # Real summarization, gated so a short chat pays nothing for it.
        updated_summary = self._summarizer.maybe_summarize(
            session_id=session.id,
            message_count=session.message_count,
            existing_summary=session.summary,
            messages=messages,
        )
        if updated_summary != session.summary:
            session.summary = updated_summary
        self._store.save_metadata(session.id, session)

        if session.summary is not None:
            self._store.save_summary(session.id, session.summary)

        # Keep the conversation retrieval index current (best-effort: the index
        # is optional and its absence never affects the stored conversation).
        try:
            self._index.index_message(conversation_id=session.id, message=msg)
        except Exception:  # noqa: BLE001
            logger.exception("Conversation indexing failed")

        logger.info(
            "Memory message appended: session_id=%s role=%s turn=%d total_messages=%d",
            session.id,
            role,
            msg.turn_number,
            session.message_count,
        )
        return msg

    def update_message(self, message_id: str, **changes) -> MemoryMessage | None:
        """Update one stored message in the active session.

        Used for genuine state transitions (for example a streamed assistant
        turn that is still running when the user cancels it), so what is stored
        is what actually happened.
        """

        session_id = self.get_active_session_id()
        if not session_id:
            return None
        updated = self._store.update_message(session_id, message_id, **changes)
        if updated is not None and updated.content.strip():
            try:
                self._index.index_message(conversation_id=session_id, message=updated)
            except Exception:  # noqa: BLE001
                logger.exception("Conversation re-indexing failed")
        return updated

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

    def get_summary(self, *, session_id: str | None = None) -> str | None:
        sid = session_id or self.get_active_session_id()
        if not sid:
            return None
        return self._store.load_summary(sid)

    def save_summary(self, summary: str | None, *, session_id: str | None = None) -> None:
        sid = session_id or self.get_active_session_id()
        if not sid or summary is None:
            return
        meta = self._store.load_metadata(sid)
        meta.summary = summary
        meta.touch()
        self._store.save_metadata(sid, meta)
        self._store.save_summary(sid, summary)

    # ---- Conversation search (deterministic, no model, no network) ----

    def search_conversations(
        self, query: str, *, limit: int = 20, session_id: str | None = None
    ) -> list[dict]:
        """Deterministically search stored conversations for a query string.

        This is the *lexical* fallback and the exact-phrase path. Semantic
        retrieval over past conversations lives in ``memory.conversation_index``
        and uses the existing embedding infrastructure; this method never calls
        a model, so search keeps working with no model available.
        """

        needle = " ".join((query or "").casefold().split())
        if not needle:
            return []
        results: list[dict] = []
        session_ids = [session_id] if session_id else self._store.list_sessions()
        for sid in session_ids:
            try:
                messages = self._store.load_messages(sid)
            except Exception:
                logger.exception("Failed loading messages for session_id=%s", sid)
                continue
            try:
                meta = self._store.load_metadata(sid)
                title = meta.title
            except Exception:
                title = "Untitled"
            summary = (self._store.load_summary(sid) or "")
            if needle in summary.casefold():
                results.append({
                    "conversation_id": sid,
                    "title": title,
                    "message_id": "",
                    "role": "summary",
                    "excerpt": summary[:400],
                    "timestamp": None,
                    "score": 1.0,
                    "match": "summary",
                })
            for message in messages:
                haystack = message.content.casefold()
                if needle not in haystack:
                    continue
                results.append({
                    "conversation_id": sid,
                    "title": title,
                    "message_id": message.id,
                    "role": message.role,
                    "excerpt": _excerpt(message.content, needle),
                    "timestamp": message.timestamp.isoformat(),
                    "score": 0.8,
                    "match": "message",
                })
                if len(results) >= limit:
                    return results[:limit]
        return results[:limit]

    def list_messages(
        self, session_id: str, *, limit: int | None = None
    ) -> list[MemoryMessage]:
        """Return a conversation's messages, optionally only the most recent."""

        messages = self._store.load_messages(session_id)
        if limit is not None and limit >= 0:
            return messages[-limit:] if limit else []
        return messages

    # ---- Export/Import (future) ----
    # For now, provide architecture hooks.

