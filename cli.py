"""Atlas CLI.

This CLI is backend-only: it delegates business logic to Atlas subsystems.

Commands:
- /new [title]
- /list
- /open <session_id>
- /delete <session_id>
- /rename <session_id> <new_title>
- /history
- /export <session_id>
- /import <path>
- /clear
- /help

For normal chat, just enter a question (stored into the active session).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from memory.memory_manager import MemoryManager
from memory.models import ConversationSessionMetadata

logger = logging.getLogger(__name__)


def _parse_title(raw: str | None) -> str | None:
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def _session_to_export_dict(meta: ConversationSessionMetadata, messages: list[dict]) -> dict:
    return {
        "type": "atlas_memory_export_v1",
        "session": meta.to_dict(),
        "messages": messages,
    }


class CLI:
    def __init__(self, *, memory_manager: MemoryManager) -> None:
        self._memory = memory_manager

    def _ensure_active(self) -> None:
        if not self._memory.get_active_session_id():
            self._memory.create_session(title="Untitled")

    def help_text(self) -> str:
        return (
            "Commands:\n"
            "  /new [title]                 Create a new session and switch active\n"
            "  /list                         List sessions\n"
            "  /open <session_id>          Open/switch session\n"
            "  /delete <session_id>       Delete session\n"
            "  /rename <session_id> <t>   Rename session\n"
            "  /history                     Show recent history for active session\n"
            "  /export <session_id>        Export session to JSON file\n"
            "  /import <path>              Import session JSON and create it\n"
            "  /clear                       Clear active session messages (keeps session)\n"
            "  /help                         Show this help\n"
            "\n"
            "Any other input is treated as a user question.\n"
        )

    def handle_command(self, raw: str) -> Optional[str]:
        raw = raw.strip()
        if not raw:
            return None
        if not raw.startswith("/"):
            return None

        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        if cmd == "/help":
            return self.help_text()

        if cmd == "/new":
            title = _parse_title(rest) if rest else None
            meta = self._memory.create_session(title=title)
            return f"Created session: {meta.id} ({meta.title})"

        if cmd == "/list":
            sessions = self._memory.list_sessions()
            if not sessions:
                return "No sessions found."
            lines = ["Sessions (most recent first):"]
            for s in sessions[:50]:
                lines.append(f"- {s.id} | {s.title} | messages={s.message_count} | archived={s.archived}")
            return "\n".join(lines)

        if cmd == "/open":
            session_id = rest.strip()
            if not session_id:
                return "Usage: /open <session_id>"
            meta = self._memory.open_session(session_id)
            return f"Opened session: {meta.id} ({meta.title})"

        if cmd == "/delete":
            session_id = rest.strip()
            if not session_id:
                return "Usage: /delete <session_id>"
            self._memory.delete_session(session_id)
            return f"Deleted session: {session_id}"

        if cmd == "/rename":
            # /rename <id> <new_title>
            pieces = rest.split(maxsplit=1)
            if len(pieces) != 2:
                return "Usage: /rename <session_id> <new_title>"
            session_id, new_title = pieces[0].strip(), pieces[1].strip()
            if not session_id or not new_title:
                return "Usage: /rename <session_id> <new_title>"
            self._memory.rename_session(session_id, new_title=new_title)
            return f"Renamed session: {session_id} -> {new_title}"

        if cmd == "/history":
            self._ensure_active()
            history = self._memory.build_conversation_history_for_prompt()
            return history or "(no messages yet)"

        if cmd == "/export":
            # /export <session_id> [path]
            pieces = rest.split(maxsplit=1)
            if not pieces or not pieces[0].strip():
                return "Usage: /export <session_id> [path]"
            session_id = pieces[0].strip()
            out_path = Path(pieces[1].strip()) if len(pieces) == 2 and pieces[1].strip() else None

            meta = self._memory.open_session(session_id)
            messages = self._memory.get_recent_messages(session_id=session_id)
            d = _session_to_export_dict(meta, [m.to_dict() for m in messages])

            if out_path is None:
                out_path = Path.cwd() / f"memory_export_{session_id}.json"

            out_path.write_text(json.dumps(d, indent=2), encoding="utf-8")
            return f"Exported session {session_id} to {out_path}"

        if cmd == "/import":
            path_str = rest.strip()
            if not path_str:
                return "Usage: /import <path>"
            path = Path(path_str)
            if not path.exists():
                return f"Import file not found: {path}"

            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("type") != "atlas_memory_export_v1":
                return "Unsupported export format."

            session_meta = payload.get("session") or {}
            session_id = session_meta.get("id")
            title = session_meta.get("title") or "Imported"

            meta = self._memory.create_session(title=title)
            # Note: we do not reuse the original session_id to avoid collisions.
            messages = payload.get("messages") or []
            for m in messages:
                role = m.get("role")
                content = m.get("content")
                if role and content is not None:
                    self._memory.append_message(role=role, content=content)

            self._memory.open_session(meta.id)
            return f"Imported session '{title}' as new session: {meta.id}"

        if cmd == "/clear":
            # Simplest backend-only clear: delete and recreate active session.
            self._ensure_active()
            sid = self._memory.get_active_session_id()
            if not sid:
                return "No active session."
            title = (self._memory.get_active_metadata().title if self._memory.get_active_metadata() else None) or "Untitled"
            self._memory.delete_session(sid)
            self._memory.create_session(title=title)
            return "Cleared active session (new empty session created)."

        return f"Unknown command: {cmd}. Use /help"


