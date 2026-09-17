"""Content formatting capability: normalize, structure, and render content for destinations."""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from content_formatting import (
    ContentFormatter,
    ContentType,
    DestinationType,
    format_content,
    format_content_with_repair,
)
from tools.base import PermissionLevel, RiskLevel, Tool, ToolMetadata, ToolResult

logger = logging.getLogger(__name__)


class ContentFormatTool(Tool):
    metadata = ToolMetadata(
        name="content.format",
        description=(
            "Normalize, structure, and render raw content for a specific destination. "
            "Detects content type (script, article, transcript, code, list, etc.), "
            "reconstructs paragraphs, and formats for Notepad, Markdown, code editors, etc."
        ),
        category="content.formatting",
        input_schema={
            "content": {"type": "string", "description": "Raw content to format"},
            "destination": {"type": "string", "description": "Target application (notepad, word, markdown, code_editor, terminal)"},
            "title": {"type": "string", "description": "Optional title for the content"},
            "source": {"type": "string", "description": "Source identifier (URL, filename, etc.)"},
            "url": {"type": "string", "description": "Source URL if from web"},
            "repair": {"type": "boolean", "description": "Enable automatic formatting repair on verification failure"},
        },
        output_schema={
            "text": {"type": "string"},
            "content_type": {"type": "string"},
            "metadata": {"type": "object"},
        },
        permission_level=PermissionLevel.READ_ONLY,
        risk_level=RiskLevel.READ_ONLY,
        required_parameters=("content",),
        verifiable=True,
        produces=("formatted_text",),
    )

    def __init__(self, *, repair_enabled: bool = True) -> None:
        self._formatter = ContentFormatter()
        self._repair_enabled = repair_enabled

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            content = str(parameters.get("content") or "")
            if not content.strip():
                return ToolResult.failure("Content cannot be empty")

            destination = str(parameters.get("destination") or "notepad").strip()
            title = str(parameters.get("title") or "").strip()
            source = str(parameters.get("source") or "").strip()
            url = str(parameters.get("url") or "").strip()
            repair = bool(parameters.get("repair", self._repair_enabled))

            if repair:
                formatted, metadata = self._formatter.format_with_repair(
                    content, destination, title, source, url
                )
            else:
                formatted, metadata = self._formatter.format(
                    content, destination, title, source, url
                )

            return ToolResult(
                success=True,
                status="completed",
                output={
                    "text": formatted,
                    "content_type": metadata["content_type"],
                    "metadata": metadata,
                },
                metadata={"verification_passed": metadata["verification_passed"]},
            )
        except (KeyError, TypeError, ValueError, OSError) as exc:
            logger.exception("Content formatting failed")
            return ToolResult.failure(f"Content formatting failed: {exc}", recoverable=True)


import re