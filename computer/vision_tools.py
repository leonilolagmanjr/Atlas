"""Tool contracts for Atlas's local vision ("Eyes") perception layer.

Exposes the layered perception engine through the existing tool framework:

* ``computer.observe``  capture and analyze the current desktop state
* ``computer.find``     find a visual/UI element by text, type, or description

Both are read-only observations. They never execute an action and never grant
permission; interaction tools live in :mod:`computer.interaction` and stay
permission-gated. Screen text is returned as untrusted data, never as
instructions.
"""

from __future__ import annotations

import logging
from typing import Any

from config import (
    VISION_CONFIDENCE_THRESHOLD,
    VISION_ENABLED,
    VISION_MODEL,
    VISION_OCR_LANGUAGES,
    VISION_PROVIDER,
    VISION_TIMEOUT,
)

from computer.vision.models import Bounds, VisualState
from computer.vision.perception import (
    ObserveRequest,
    PerceptionEngine,
    build_perception_engine,
)
from tools.base import PermissionLevel, RiskLevel, Tool, ToolMetadata, ToolResult

logger = logging.getLogger(__name__)


def build_vision_engine() -> PerceptionEngine:
    """Construct the configured perception engine, degrading each layer."""

    return build_perception_engine(
        vision_enabled=VISION_ENABLED,
        provider=VISION_PROVIDER,
        model=VISION_MODEL,
        timeout=VISION_TIMEOUT,
        ocr_languages=VISION_OCR_LANGUAGES,
    )


def _region_from_parameters(parameters: dict[str, Any]) -> Bounds | None:
    region = parameters.get("region")
    if region is None:
        return None
    try:
        return Bounds.from_rect(region)
    except (TypeError, ValueError):
        return None


class VisionObserveTool(Tool):
    """Read-only, layered observation of the current desktop state."""

    metadata = ToolMetadata(
        name="computer.vision_observe",
        description=(
            "Capture and analyze the current screen into a structured visual state "
            "(windows, UI elements, text) using Windows UI Automation, local OCR, "
            "image processing, and optionally a local vision model."
        ),
        category="computer.perception",
        input_schema={
            "application": {"type": "string", "description": "optional application/window to observe"},
            "full_screen": {"type": "boolean", "description": "observe the whole screen instead of the active window"},
            "region": {
                "type": "object",
                "description": "optional region {left, top, right, bottom} or {left, top, width, height}",
            },
            "include_uia": {"type": "boolean", "description": "read the Windows UI Automation tree"},
            "include_ocr": {"type": "boolean", "description": "read visible text with local OCR"},
            "allow_vlm": {"type": "boolean", "description": "consult the local vision model (only when needed)"},
            "vlm_question": {"type": "string", "description": "structured question for the vision model"},
        },
        output_schema={
            "observation_id": {"type": "string"},
            "screen": {"type": "object"},
            "active_window": {"type": "object"},
            "elements": {"type": "array"},
            "text": {"type": "array"},
            "sources": {"type": "array"},
        },
        permission_level=PermissionLevel.READ_ONLY,
        risk_level=RiskLevel.READ_ONLY,
        verifiable=True,
    )

    def __init__(self, *, engine: PerceptionEngine | None = None) -> None:
        self._engine = engine

    def _resolved_engine(self) -> PerceptionEngine:
        if self._engine is None:
            self._engine = build_vision_engine()
        return self._engine

    def observe(self, parameters: dict[str, Any]) -> VisualState:
        """Observe the screen and return the typed visual state."""

        request = ObserveRequest(
            application=str(parameters.get("application") or "").strip(),
            region=_region_from_parameters(parameters),
            include_ocr=bool(parameters.get("include_ocr", True)),
            include_uia=bool(parameters.get("include_uia", True)),
            allow_vlm=bool(parameters.get("allow_vlm", False)),
            vlm_question=str(parameters.get("vlm_question") or ""),
            full_screen=bool(parameters.get("full_screen", False)),
        )
        return self._resolved_engine().observe(request)

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            if not VISION_ENABLED:
                return ToolResult.failure(
                    "Local vision is disabled (VISION_ENABLED is false).", recoverable=False
                )
            state = self.observe(parameters)
            output = state.to_dict()
            output["status"] = "observed"
            output["summary"] = _summarize(state)
            return ToolResult(success=True, status="observed", output=output)
        except (OSError, ValueError) as exc:
            return ToolResult.failure(str(exc), recoverable=True)


class VisionFindTool(Tool):
    """Read-only element search over the current visual state."""

    metadata = ToolMetadata(
        name="computer.find",
        description=(
            "Find a visible UI element by its label, a semantic description, or its "
            "type (button, text field, link, ...) using the layered perception "
            "engine. Returns structured candidates with bounds and confidence."
        ),
        category="computer.perception",
        input_schema={
            "query": {"type": "string", "description": "visible text/label or description to find"},
            "element_type": {"type": "string", "description": "button, text_field, link, checkbox, ..."},
            "application": {"type": "string", "description": "optional application/window to search"},
            "region": {"type": "object", "description": "optional region to restrict the search"},
            "max_results": {"type": "integer"},
            "allow_vlm": {"type": "boolean", "description": "let the local vision model help locate the element"},
        },
        output_schema={
            "candidates": {"type": "array"},
            "count": {"type": "integer"},
            "observation_id": {"type": "string"},
        },
        permission_level=PermissionLevel.READ_ONLY,
        risk_level=RiskLevel.READ_ONLY,
        required_parameters=("query",),
        verifiable=True,
    )

    def __init__(self, *, engine: PerceptionEngine | None = None) -> None:
        self._engine = engine

    def _resolved_engine(self) -> PerceptionEngine:
        if self._engine is None:
            self._engine = build_vision_engine()
        return self._engine

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            if not VISION_ENABLED:
                return ToolResult.failure(
                    "Local vision is disabled (VISION_ENABLED is false).", recoverable=False
                )
            query = str(parameters.get("query") or "").strip()
            if not query:
                return ToolResult.failure("A query is required to find an element.")
            request = ObserveRequest(
                application=str(parameters.get("application") or "").strip(),
                region=_region_from_parameters(parameters),
                include_ocr=True,
                include_uia=True,
                allow_vlm=bool(parameters.get("allow_vlm", False)),
                vlm_question=f"Locate the element described as: {query}",
            )
            engine = self._resolved_engine()
            state = engine.observe(request)
            wanted_type = str(parameters.get("element_type") or "").strip()
            candidates = state.find_by_name(query)
            if wanted_type:
                from computer.vision.models import ElementType

                try:
                    element_type = ElementType(wanted_type.casefold())
                except ValueError:
                    element_type = None
                if element_type is not None:
                    typed = [element for element in state.elements if element.type == element_type]
                    typed.sort(key=lambda item: item.confidence, reverse=True)
                    candidates = typed or candidates
            if not candidates:
                candidates = engine.find(query, request)
            max_results = _bounded_int(parameters.get("max_results"), default=10, low=1, high=50)
            threshold = min(VISION_CONFIDENCE_THRESHOLD, 0.3)
            bounded = [
                element.to_dict()
                for element in candidates[:max_results]
                if element.confidence >= threshold
            ]
            return ToolResult(
                success=True,
                status="found" if bounded else "not_found",
                output={
                    "status": "found" if bounded else "not_found",
                    "query": query,
                    "candidates": bounded,
                    "count": len(bounded),
                    "observation_id": state.observation_id,
                    "summary": (
                        f"Found {len(bounded)} candidate(s) for '{query}'."
                        if bounded
                        else f"No visible element matches '{query}' in the current observation."
                    ),
                },
            )
        except (OSError, ValueError) as exc:
            return ToolResult.failure(str(exc), recoverable=True)


def _bounded_int(value: Any, *, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _summarize(state: VisualState) -> str:
    application = state.detected_application or "the desktop"
    parts = [f"Observed {application}"]
    if state.active_window and state.active_window.title:
        parts.append(f"window '{state.active_window.title}'")
    parts.append(f"{len(state.elements)} element(s)")
    parts.append(f"{len(state.text)} text run(s)")
    if state.sources:
        parts.append("via " + ", ".join(state.sources))
    summary = ", ".join(parts) + "."
    if state.notes:
        summary += " + " + " ".join(state.notes)
    return summary
