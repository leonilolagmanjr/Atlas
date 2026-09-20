"""Visual action validation and coordination.

This is the safety layer between a model-proposed action and the OS. It exists
because a model must never be able to issue raw screen coordinates that Atlas
executes blindly. Every visual action is validated against the observation that
produced the target:

1. The action must reference an observation (``observation_id``) or an element
   id from one, or be an explicit, permission-gated coordinate with no
   observation.
2. The referenced observation must still be valid (not stale).
3. The target must still be present and its bounds must contain the click point.
4. The active window/application must match the observation's, when declared.
5. Significant UI change since the observation means re-observe first.

Nothing here grants permission: validation is separate from the permission
system, which remains the executor's gate for consequential actions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from config import VISION_OBSERVATION_MAX_AGE_SECONDS

from computer.vision.models import Bounds, VisualElement, VisualState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisualTarget:
    """A validated click/drag/typing target resolved from an observation."""

    x: int
    y: int
    bounds: Bounds
    observation_id: str
    source: str
    name: str = ""
    element_id: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "bounds": self.bounds.to_dict(),
            "observation_id": self.observation_id,
            "source": self.source,
            "name": self.name,
            "element_id": self.element_id,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ValidationResult:
    """The outcome of validating an action against an observation."""

    ok: bool
    reason: str
    target: Optional[VisualTarget] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "target": self.target.to_dict() if self.target else None,
        }


def _reject(reason: str) -> ValidationResult:
    return ValidationResult(ok=False, reason=reason)


def validate_visual_target(
    *,
    state: Optional[VisualState],
    element_id: str = "",
    description: str = "",
    x: Optional[int] = None,
    y: Optional[int] = None,
    observation_id: str = "",
    require_fresh: bool = True,
    max_age_seconds: float = VISION_OBSERVATION_MAX_AGE_SECONDS,
    expected_application: str = "",
) -> ValidationResult:
    """Validate a proposed visual target against an observation.

    Returns a :class:`ValidationResult`; a rejected result must not be acted on.
    The rules are deliberately strict: a stale observation, an element that is
    gone, or a point outside the element's bounds is rejected rather than
    guessed at.
    """

    if state is None:
        if x is None or y is None:
            return _reject("No observation is available and no coordinates were provided.")
        # A coordinate-only action with no observation is allowed structurally
        # (permission is a separate gate), but is flagged as unverified.
        return ValidationResult(
            ok=True,
            reason="No observation; acting on explicit coordinates without visual verification.",
            target=VisualTarget(
                x=int(x),
                y=int(y),
                bounds=Bounds(int(x), int(y), int(x), int(y)),
                observation_id="",
                source="explicit",
            ),
        )

    if require_fresh and state.is_stale(max_age_seconds=max_age_seconds):
        return _reject(
            f"The observation is stale ({state.age_seconds:.1f}s old); re-observe before acting."
        )

    if expected_application:
        detected = (state.detected_application or "").casefold()
        if detected and expected_application.casefold() not in detected:
            return _reject(
                f"The active application is '{state.detected_application}', not "
                f"'{expected_application}'; re-observe the intended window."
            )

    element: Optional[VisualElement] = None
    if element_id:
        element = state.element(element_id)
        if element is None:
            return _reject(f"Element '{element_id}' is not present in the current observation.")
    elif description:
        matches = state.find_by_name(description)
        if not matches:
            return _reject(f"No visible element matches '{description}' in the current observation.")
        element = matches[0]

    if element is not None:
        if not element.visible:
            return _reject(f"Element '{element.name or element.id}' is not visible.")
        if not element.enabled:
            return _reject(f"Element '{element.name or element.id}' is disabled.")
        point = element.bounds.center
        if x is not None and y is not None:
            if not element.bounds.contains_point(int(x), int(y), tolerance=2):
                return _reject(
                    f"Coordinates ({x}, {y}) fall outside the bounds of "
                    f"'{element.name or element.id}'."
                )
            point = (int(x), int(y))
        return ValidationResult(
            ok=True,
            reason=f"Targeted '{element.name or element.id}' ({element.source.value}).",
            target=VisualTarget(
                x=point[0],
                y=point[1],
                bounds=element.bounds,
                observation_id=element.observation_id or state.observation_id,
                source=element.source.value,
                name=element.name,
                element_id=element.id,
                confidence=element.confidence,
            ),
        )

    if x is None or y is None:
        return _reject("Provide an element id, a description, or explicit coordinates.")

    if not _within_screen(state, int(x), int(y)):
        return _reject(f"Coordinates ({x}, {y}) are outside the observed screen bounds.")

    return ValidationResult(
        ok=True,
        reason="Acting on explicit coordinates within the observed screen.",
        target=VisualTarget(
            x=int(x),
            y=int(y),
            bounds=Bounds(int(x), int(y), int(x), int(y)),
            observation_id=observation_id or state.observation_id,
            source="explicit",
        ),
    )


def _within_screen(state: VisualState, x: int, y: int) -> bool:
    if state.screen.width <= 0 or state.screen.height <= 0:
        return True
    return 0 <= x <= state.screen.width and 0 <= y <= state.screen.height


def coerce_bounds(value: Any) -> Optional[Bounds]:
    """Coerce an externally supplied bounds value into typed :class:`Bounds`."""

    if value is None:
        return None
    if isinstance(value, Bounds):
        return value
    try:
        bounds = Bounds.from_rect(value)
    except (TypeError, ValueError):
        return None
    if bounds.width < 0 or bounds.height < 0:
        return None
    return bounds
