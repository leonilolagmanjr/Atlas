"""Conversation context builder."""

from __future__ import annotations

import logging
from typing import List

from config import MAX_RETAINED_MESSAGES
from memory.models import MemoryMessage

logger = logging.getLogger(__name__)


class ContextBuilder:
    def build_prompt_context(self, *, messages: list[MemoryMessage], include_summary: bool = True) -> str:
        """Build a prompt-ready conversation history string.

        Initial implementation is deterministic and based on a message window.
        """

        if not messages:
            return ""

        recent = messages[-MAX_RETAINED_MESSAGES:] if MAX_RETAINED_MESSAGES else messages

        parts: list[str] = []
        for m in recent:
            role = m.role.strip().lower()
            if role not in {"user", "assistant"}:
                role = "user"
            prefix = "User" if role == "user" else "Assistant"
            parts.append(f"{prefix}: {m.content}")

        return "\n".join(parts)

