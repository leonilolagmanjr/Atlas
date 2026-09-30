"""Filesystem persistence for conversation sessions.

Storage layout:

memory/
  sessions/
    <session-id>/
      metadata.json
      messages.json
      summary.txt

We intentionally keep this separate from knowledge/vector storage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


from memory.models import ConversationSessionMetadata, MemoryMessage

logger = logging.getLogger(__name__)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


class ConversationStore:
    def __init__(self, *, root_folder: Path) -> None:
        self._root_folder = root_folder
        self._sessions_root = self._root_folder / "sessions"
        _ensure_dir(self._sessions_root)

    def _session_folder(self, session_id: str) -> Path:
        return self._sessions_root / session_id

    def _metadata_path(self, session_id: str) -> Path:
        return self._session_folder(session_id) / "metadata.json"

    def _messages_path(self, session_id: str) -> Path:
        return self._session_folder(session_id) / "messages.json"

    def _summary_path(self, session_id: str) -> Path:
        return self._session_folder(session_id) / "summary.txt"

    def list_sessions(self) -> list[str]:
        if not self._sessions_root.exists():
            return []
        return [p.name for p in self._sessions_root.iterdir() if p.is_dir()]

    def load_metadata(self, session_id: str) -> ConversationSessionMetadata:
        meta_path = self._metadata_path(session_id)
        with meta_path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        return ConversationSessionMetadata.from_dict(d)

    def save_metadata(self, session_id: str, metadata: ConversationSessionMetadata) -> None:
        folder = self._session_folder(session_id)
        _ensure_dir(folder)
        meta_path = self._metadata_path(session_id)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2)

    def load_messages(self, session_id: str) -> list[MemoryMessage]:
        messages_path = self._messages_path(session_id)
        if not messages_path.exists():
            return []
        with messages_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        msgs = [MemoryMessage.from_dict(m) for m in (data or [])]
        return msgs

    def save_messages(self, session_id: str, messages: list[MemoryMessage]) -> None:
        folder = self._session_folder(session_id)
        _ensure_dir(folder)
        messages_path = self._messages_path(session_id)
        # Write atomically: a message history is the conversation, so a crash
        # mid-write must never truncate it. Atomic replace keeps the previous
        # file intact until the new one is complete.
        temporary = messages_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as f:
            json.dump([m.to_dict() for m in messages], f, indent=2)
        temporary.replace(messages_path)

    def update_message(
        self, session_id: str, message_id: str, **changes: Any
    ) -> MemoryMessage | None:
        """Apply field changes to one stored message (used for live states).

        Returns the updated message, or ``None`` when it does not exist. Fields
        not listed here are ignored rather than silently injected, so an
        arbitrary caller cannot corrupt the stored shape.
        """

        allowed = {
            "role", "content", "attachments", "tool_calls", "tool_results",
            "citations", "execution_state", "metadata",
        }
        messages = self.load_messages(session_id)
        for index, message in enumerate(messages):
            if message.id != message_id:
                continue
            for key, value in changes.items():
                if key in allowed and value is not None:
                    setattr(message, key, value)
            messages[index] = message
            self.save_messages(session_id, messages)
            return message
        return None

    def load_summary(self, session_id: str) -> str | None:
        p = self._summary_path(session_id)
        if not p.exists():
            return None
        return p.read_text(encoding="utf-8")

    def save_summary(self, session_id: str, summary: str | None) -> None:
        p = self._summary_path(session_id)
        if summary is None:
            # Keep old summary if it exists; caller decides.
            return
        _ensure_dir(p.parent)
        p.write_text(summary, encoding="utf-8")

    def delete_session(self, session_id: str) -> None:
        folder = self._session_folder(session_id)
        if not folder.exists():
            return
        for child in folder.rglob("*"):
            if child.is_file():
                child.unlink(missing_ok=True)
        # finally delete folders
        for child in sorted(folder.rglob("*"), reverse=True):
            if child.is_dir():
                child.rmdir()
        folder.rmdir()

