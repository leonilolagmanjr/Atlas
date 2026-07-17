"""Conversation summarization.

This module is intentionally a placeholder foundation.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class MemorySummarizer:
    def maybe_summarize(self, *, session_id: str, message_count: int, existing_summary: str | None) -> str | None:
        """Return updated summary (or existing summary if unchanged).

        Placeholder: does not perform real summarization yet.
        """

        # Future: when message_count exceeds threshold, summarize older messages.
        # For now, keep any existing summary.
        return existing_summary

