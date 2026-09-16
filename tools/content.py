"""Capability for generating requested content (poems, stories, essays...).

This tool owns the *generation* of text. Writing that text to an application is
a separate, permission-gated capability (``applications.write_text``). Keeping
them separate lets the planner compose multi-step tasks without hard-coding a
single "poem in notepad" command.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from reasoning.prompts import CONTENT_SYSTEM
from tools.base import PermissionLevel, RiskLevel, Tool, ToolMetadata, ToolResult

logger = logging.getLogger(__name__)


class ContentGenerationTool(Tool):
    metadata = ToolMetadata(
        name="content.generate",
        description=(
            "Generate creative or informational text from a requested content type, "
            "topic, tone, style, and length."
        ),
        category="content.generation",
        input_schema={
            "content_type": {"type": "string", "description": "poem, story, essay, ..."},
            "topic": {"type": "string", "description": "subject the content is about"},
            "tone": {"type": "string", "description": "e.g. funny, serious"},
            "style": {"type": "string", "description": "e.g. futuristic, formal"},
            "length": {"type": "string", "description": "short or long"},
            "instructions": {"type": "string", "description": "extra free-form guidance"},
        },
        output_schema={"text": {"type": "string"}},
        permission_level=PermissionLevel.READ_ONLY,
        risk_level=RiskLevel.READ_ONLY,
    )

    def __init__(self, *, ask: Optional[Callable[..., str]] = None) -> None:
        self._ask = ask

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        self.validate(parameters)
        if self._ask is None:
            return ToolResult.failure(
                "Content generation requires the local language model.", recoverable=False
            )

        content_type = str(parameters.get("content_type") or "text").strip()
        topic = str(parameters.get("topic") or "").strip()
        tone = str(parameters.get("tone") or "").strip()
        style = str(parameters.get("style") or "").strip()
        length = str(parameters.get("length") or "").strip()

        request = self._build_request(
            content_type=content_type, topic=topic, tone=tone, style=style, length=length
        )
        # content.generate must return prose, so a JSON-looking answer is not
        # expected; call the model directly for the free-text body instead.
        try:
            text = self._ask(system_prompt=CONTENT_SYSTEM, user_prompt=request)
        except Exception as exc:  # noqa: BLE001 - surface as recoverable failure
            logger.exception("Content generation failed")
            return ToolResult.failure(f"Content generation failed: {exc}", recoverable=True)

        text = (text or "").strip()
        if not text:
            return ToolResult.failure("The language model returned no content.", recoverable=True)
        return ToolResult(
            success=True,
            status="generated",
            output={
                "text": text,
                "content_type": content_type,
                "topic": topic or None,
                "tone": tone or None,
                "style": style or None,
                "length": length or None,
            },
        )

    @staticmethod
    def _build_request(
        *,
        content_type: str,
        topic: str,
        tone: str,
        style: str,
        length: str,
    ) -> str:
        parts = [f"Write a {content_type}."]
        if topic:
            parts.append(f"Topic: {topic}.")
        if tone:
            parts.append(f"Tone: {tone}.")
        if style:
            parts.append(f"Style: {style}.")
        if length:
            parts.append(f"Length: {length}.")
        parts.append("Return only the content body.")
        return " ".join(parts)
