"""Permission-gated visual interaction tools for Atlas.

These tools are the ACTION half of the closed loop and the only vision
capabilities that change the machine, so each is permission-gated through the
existing :class:`~tools.permissions.PermissionEngine` and validated through
:mod:`computer.vision.actions` before any OS input happens.

A tool accepts an operation described against an *observation*: an element id,
a visible description, or (for a deliberate, verified case) explicit
coordinates. It never invents coordinates from a model string. After acting it
records what it targeted so the verifier and a following OBSERVE can confirm the
effect, rather than trusting the input call.

The perception engine and input backend are injectable; the default constructs
real ones, and tests pass fakes so no GUI is required.
"""

from __future__ import annotations

import logging
from typing import Any

from config import (
    VISION_ENABLED,
    VISION_MODEL,
    VISION_OBSERVATION_MAX_AGE_SECONDS,
    VISION_OCR_LANGUAGES,
    VISION_PROVIDER,
    VISION_TIMEOUT,
)

from computer.vision.actions import VisualTarget, validate_visual_target
from computer.vision.models import VisualState
from computer.vision.perception import PerceptionEngine, build_perception_engine
from tools.base import PermissionLevel, RiskLevel, Tool, ToolMetadata, ToolResult

logger = logging.getLogger(__name__)

#: Capabilities in this family whose effect is worth an immediate re-observe.
VISUAL_ACTION_CAPABILITIES: frozenset[str] = frozenset(
    {
        "computer.click",
        "computer.double_click",
        "computer.right_click",
        "computer.drag",
        "computer.type",
        "computer.keypress",
        "computer.scroll",
        "computer.move",
        "computer.focus",
    }
)


def _bounded_int(value: Any, *, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


class _VisualToolBase(Tool):
    """Shared plumbing: perceive, validate a target, act, report."""

    def __init__(
        self,
        *,
        engine: PerceptionEngine | None = None,
        input_backend: Any | None = None,
    ) -> None:
        self._engine = engine
        self._input = input_backend

    def _resolved_engine(self) -> PerceptionEngine:
        if self._engine is None:
            self._engine = build_perception_engine(
                vision_enabled=VISION_ENABLED,
                provider=VISION_PROVIDER,
                model=VISION_MODEL,
                timeout=VISION_TIMEOUT,
                ocr_languages=VISION_OCR_LANGUAGES,
            )
        return self._engine

    def _resolved_input(self) -> Any:
        if self._input is None:
            from computer.vision.input import default_input_backend

            self._input = default_input_backend()
        return self._input

    def _observe_and_validate(
        self,
        parameters: dict[str, Any],
        *,
        extra_question: str = "",
    ) -> tuple[bool, str, VisualState | None, VisualTarget | None]:
        """Observe the screen and validate the requested target against it."""

        engine = self._resolved_engine()
        from computer.vision.perception import ObserveRequest

        request = ObserveRequest(
            application=str(parameters.get("application") or "").strip(),
            include_ocr=bool(parameters.get("include_ocr", True)),
            include_uia=True,
            allow_vlm=bool(parameters.get("allow_vlm", False)),
            vlm_question=extra_question,
        )
        state = engine.observe(request)
        validated = validate_visual_target(
            state=state,
            element_id=str(parameters.get("element_id") or ""),
            description=str(parameters.get("description") or parameters.get("target") or ""),
            x=parameters.get("x"),
            y=parameters.get("y"),
            observation_id=str(parameters.get("observation_id") or ""),
            max_age_seconds=float(
                parameters.get("max_age_seconds", VISION_OBSERVATION_MAX_AGE_SECONDS)
            ),
            expected_application=str(parameters.get("application") or "").strip(),
        )
        if not validated.ok or validated.target is None:
            return False, validated.reason, state, None
        return True, validated.reason, state, validated.target

    def _result(self, status: str, output: dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, status=status, output=output)

    def _failure(self, error: str, *, recoverable: bool = True) -> ToolResult:
        return ToolResult.failure(error, recoverable=recoverable)


_TARGET_SCHEMA: dict[str, Any] = {
    "element_id": {
        "type": "string",
        "description": "id of an element from the latest observation (preferred)",
    },
    "description": {
        "type": "string",
        "description": "visible label/description of the target element, e.g. 'Continue'",
    },
    "x": {"type": "integer", "description": "explicit screen x (validated against the observation)"},
    "y": {"type": "integer", "description": "explicit screen y (validated against the observation)"},
    "observation_id": {"type": "string", "description": "the observation this target came from"},
    "application": {"type": "string", "description": "expected application/window"},
    "allow_vlm": {"type": "boolean", "description": "allow a local VLM to help locate the target"},
}


class ComputerClickTool(_VisualToolBase):
    """Click a validated on-screen element."""

    metadata = ToolMetadata(
        name="computer.click",
        description=(
            "Click a UI element located in the current observation (by element id "
            "or visible label). Coordinates are validated against the observation."
        ),
        category="computer.interaction",
        input_schema={**_TARGET_SCHEMA, "button": {"type": "string", "description": "left, right or middle"}},
        output_schema={
            "status": {"type": "string"},
            "target": {"type": "object"},
            "observation_id": {"type": "string"},
        },
        permission_level=PermissionLevel.MEDIUM_RISK,
        risk_level=RiskLevel.MEDIUM,
        verifiable=True,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        return self._click(parameters, count=1, button=str(parameters.get("button") or "left"))

    def _click(self, parameters: dict[str, Any], *, count: int, button: str) -> ToolResult:
        try:
            self.validate(parameters)
            ok, reason, state, target = self._observe_and_validate(parameters)
            if not ok or target is None or state is None:
                return self._failure(reason)
            self._resolved_input().click(target.x, target.y, button=button, count=count)
            return self._result(
                "clicked",
                {
                    "status": "clicked",
                    "target": target.to_dict(),
                    "observation_id": state.observation_id,
                    "button": button,
                    "clicks": count,
                    "verified": False,
                },
            )
        except (OSError, ValueError) as exc:
            return self._failure(str(exc), recoverable=False)


class ComputerDoubleClickTool(ComputerClickTool):
    """Double-click a validated on-screen element."""

    metadata = ToolMetadata(
        name="computer.double_click",
        description="Double-click a UI element located in the current observation.",
        category="computer.interaction",
        input_schema=dict(_TARGET_SCHEMA),
        output_schema={"status": {"type": "string"}, "target": {"type": "object"}},
        permission_level=PermissionLevel.MEDIUM_RISK,
        risk_level=RiskLevel.MEDIUM,
        verifiable=True,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        return self._click(parameters, count=2, button="left")


class ComputerRightClickTool(ComputerClickTool):
    """Right-click a validated on-screen element."""

    metadata = ToolMetadata(
        name="computer.right_click",
        description="Right-click a UI element located in the current observation.",
        category="computer.interaction",
        input_schema=dict(_TARGET_SCHEMA),
        output_schema={"status": {"type": "string"}, "target": {"type": "object"}},
        permission_level=PermissionLevel.MEDIUM_RISK,
        risk_level=RiskLevel.MEDIUM,
        verifiable=True,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        return self._click(parameters, count=1, button="right")


class ComputerMoveTool(_VisualToolBase):
    """Move the pointer to a validated on-screen element."""

    metadata = ToolMetadata(
        name="computer.move",
        description="Move the mouse pointer to a UI element located in the current observation.",
        category="computer.interaction",
        input_schema=dict(_TARGET_SCHEMA),
        output_schema={"status": {"type": "string"}, "target": {"type": "object"}},
        permission_level=PermissionLevel.LOW_RISK,
        risk_level=RiskLevel.LOW,
        verifiable=True,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            ok, reason, state, target = self._observe_and_validate(parameters)
            if not ok or target is None or state is None:
                return self._failure(reason)
            self._resolved_input().move(target.x, target.y)
            return self._result(
                "moved",
                {"status": "moved", "target": target.to_dict(), "observation_id": state.observation_id},
            )
        except (OSError, ValueError) as exc:
            return self._failure(str(exc), recoverable=False)


class ComputerDragTool(_VisualToolBase):
    """Drag from one validated element to another."""

    metadata = ToolMetadata(
        name="computer.drag",
        description=(
            "Drag from a source element to a destination element, both located in "
            "the current observation."
        ),
        category="computer.interaction",
        input_schema={
            "from_element_id": {"type": "string"},
            "from_description": {"type": "string"},
            "to_element_id": {"type": "string"},
            "to_description": {"type": "string"},
            "application": {"type": "string"},
        },
        output_schema={"status": {"type": "string"}, "from": {"type": "object"}, "to": {"type": "object"}},
        permission_level=PermissionLevel.MEDIUM_RISK,
        risk_level=RiskLevel.MEDIUM,
        verifiable=True,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            engine = self._resolved_engine()
            from computer.vision.perception import ObserveRequest

            state = engine.observe(
                ObserveRequest(application=str(parameters.get("application") or "").strip())
            )
            start = validate_visual_target(
                state=state,
                element_id=str(parameters.get("from_element_id") or ""),
                description=str(parameters.get("from_description") or ""),
            )
            end = validate_visual_target(
                state=state,
                element_id=str(parameters.get("to_element_id") or ""),
                description=str(parameters.get("to_description") or ""),
            )
            if not start.ok or start.target is None:
                return self._failure(f"Drag source: {start.reason}")
            if not end.ok or end.target is None:
                return self._failure(f"Drag destination: {end.reason}")
            self._resolved_input().drag(
                start.target.x, start.target.y, end.target.x, end.target.y
            )
            return self._result(
                "dragged",
                {
                    "status": "dragged",
                    "from": start.target.to_dict(),
                    "to": end.target.to_dict(),
                    "observation_id": state.observation_id,
                },
            )
        except (OSError, ValueError) as exc:
            return self._failure(str(exc), recoverable=False)


class ComputerTypeTool(_VisualToolBase):
    """Type text into a validated element (focusing it first if needed)."""

    metadata = ToolMetadata(
        name="computer.type",
        description=(
            "Type text into a UI element located in the current observation. The "
            "element is focused by clicking it first so the text lands in the "
            "intended control."
        ),
        category="computer.interaction",
        input_schema={
            **_TARGET_SCHEMA,
            "text": {"type": "string", "description": "exact text to type"},
        },
        output_schema={"status": {"type": "string"}, "target": {"type": "object"}, "characters": {"type": "integer"}},
        permission_level=PermissionLevel.HIGH_RISK,
        risk_level=RiskLevel.HIGH,
        required_parameters=("text",),
        verifiable=True,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            text = str(parameters.get("text") or "")
            if not text:
                return self._failure("No text was provided to type.", recoverable=False)
            ok, reason, state, target = self._observe_and_validate(parameters)
            if not ok or target is None or state is None:
                return self._failure(reason)
            backend = self._resolved_input()
            backend.click(target.x, target.y)
            backend.type_text(text)
            return self._result(
                "typed",
                {
                    "status": "typed",
                    "target": target.to_dict(),
                    "characters": len(text),
                    "observation_id": state.observation_id,
                    "verified": False,
                },
            )
        except (OSError, ValueError) as exc:
            return self._failure(str(exc), recoverable=False)


class ComputerKeypressTool(_VisualToolBase):
    """Send a named key (Enter, Tab, Escape, ...) to the active window."""

    metadata = ToolMetadata(
        name="computer.keypress",
        description="Send a named key press (enter, tab, escape, up, down, ...).",
        category="computer.interaction",
        input_schema={
            "key": {"type": "string", "description": "enter, tab, escape, up, down, ..."},
            "application": {"type": "string"},
        },
        output_schema={"status": {"type": "string"}, "key": {"type": "string"}},
        permission_level=PermissionLevel.MEDIUM_RISK,
        risk_level=RiskLevel.MEDIUM,
        required_parameters=("key",),
        verifiable=True,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            key = str(parameters.get("key") or "").strip().casefold()
            if not key:
                return self._failure("No key was provided.", recoverable=False)
            backend = self._resolved_input()
            try:
                backend.keypress(key)
            except ValueError as exc:
                return self._failure(str(exc), recoverable=False)
            return self._result("pressed", {"status": "pressed", "key": key, "verified": False})
        except OSError as exc:
            return self._failure(str(exc), recoverable=False)


class ComputerScrollTool(_VisualToolBase):
    """Scroll the active window (optionally after targeting an element)."""

    metadata = ToolMetadata(
        name="computer.scroll",
        description="Scroll the view up or down, optionally after moving to an element.",
        category="computer.interaction",
        input_schema={
            "direction": {"type": "string", "description": "up or down"},
            "amount": {"type": "integer", "description": "number of wheel notches (1-20)"},
            "element_id": {"type": "string"},
            "description": {"type": "string"},
            "application": {"type": "string"},
        },
        output_schema={"status": {"type": "string"}, "direction": {"type": "string"}},
        permission_level=PermissionLevel.LOW_RISK,
        risk_level=RiskLevel.LOW,
        verifiable=True,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            direction = str(parameters.get("direction") or "down").strip().casefold()
            notches = _bounded_int(parameters.get("amount"), default=3, low=1, high=20)
            backend = self._resolved_input()
            # If an element is named, position the wheel over it so the scroll
            # lands in the right pane; otherwise scroll at the current position.
            if parameters.get("element_id") or parameters.get("description"):
                ok, reason, _state, target = self._observe_and_validate(parameters)
                if not ok or target is None:
                    return self._failure(reason)
                backend.move(target.x, target.y)
            delta = notches * 120 if direction == "up" else -notches * 120
            backend.scroll(delta)
            return self._result(
                "scrolled",
                {"status": "scrolled", "direction": direction, "notches": notches, "verified": False},
            )
        except (OSError, ValueError) as exc:
            return self._failure(str(exc), recoverable=False)


class ComputerFocusTool(_VisualToolBase):
    """Bring an observed window to the foreground."""

    metadata = ToolMetadata(
        name="computer.focus",
        description="Focus (bring to the foreground) an observed application window.",
        category="computer.interaction",
        input_schema={"application": {"type": "string", "description": "application or window title"}},
        output_schema={"status": {"type": "string"}, "application": {"type": "string"}},
        permission_level=PermissionLevel.LOW_RISK,
        risk_level=RiskLevel.LOW,
        verifiable=True,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            application = str(parameters.get("application") or "").strip()
            if not application:
                return self._failure("An application or window title is required.", recoverable=False)
            engine = self._resolved_engine()
            from computer.vision.perception import ObserveRequest

            state = engine.observe(ObserveRequest(application=application, include_ocr=False))
            if state.active_window is None:
                return self._failure(f"No window matching '{application}' is currently open.")
            from computer.vision.input import focus_window

            focused = focus_window(state.active_window.hwnd)
            return self._result(
                "focused" if focused else "focus_requested",
                {
                    "status": "focused" if focused else "focus_requested",
                    "application": state.active_window.title or application,
                    "hwnd": state.active_window.hwnd,
                    "verified": bool(focused),
                },
            )
        except (OSError, ValueError) as exc:
            return self._failure(str(exc), recoverable=False)
