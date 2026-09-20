"""Typed internal representation of what Atlas can see on screen.

These dataclasses are the structured visual state consumed by reasoning,
planning, action validation, and verification. They are deliberately plain
data (JSON-serializable via :meth:`VisualState.to_dict`) so they cross process
and test boundaries without a language model in the loop.

Design rules encoded here:

* Every observable fact records its ``source`` so a consumer can tell a Windows
  UI Automation fact from an OCR text run from a VLM guess.
* A ``VisualElement`` always carries a confidence and its observation id, so an
  action can be validated against the observation that produced it and stale
  observations can be rejected.
* Nothing here is an instruction. Screen text is untrusted data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


class ObservationSource(str, Enum):
    """Where an observed fact came from.

    Ordered from most to least reliable. ``UIA`` publishes accessibility data
    Windows already maintains; ``OCR`` reads pixels of text; ``IMAGE`` is
    deterministic pixel/geometry analysis; ``VLM`` is a model proposal.
    """

    UIA = "uia"
    OCR = "ocr"
    IMAGE = "image"
    VLM = "vlm"
    WIN32 = "win32"


class ElementType(str, Enum):
    """Coarse, action-relevant classification of a visual element."""

    BUTTON = "button"
    TEXT_FIELD = "text_field"
    LINK = "link"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    DROPDOWN = "dropdown"
    MENU_ITEM = "menu_item"
    TAB = "tab"
    IMAGE = "image"
    TEXT = "text"
    WINDOW = "window"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Bounds:
    """An axis-aligned rectangle in absolute screen pixels.

    Uses half-open semantics (right/bottom exclusive) with top-left origin, the
    Windows screen convention. ``width``/``height``/``center`` are derived.
    """

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def center(self) -> tuple[int, int]:
        return (self.left + self.width // 2, self.top + self.height // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def contains_point(self, x: int, y: int, *, tolerance: int = 0) -> bool:
        """Return True when (x, y) lies inside the rectangle (+/- tolerance)."""

        return (
            self.left - tolerance <= x <= self.right + tolerance
            and self.top - tolerance <= y <= self.bottom + tolerance
        )

    def contains(self, other: "Bounds", *, tolerance: int = 0) -> bool:
        """Return True when ``other`` is wholly inside this rectangle."""

        return (
            other.left >= self.left - tolerance
            and other.top >= self.top - tolerance
            and other.right <= self.right + tolerance
            and other.bottom <= self.bottom + tolerance
        )

    def intersection(self, other: "Bounds") -> Optional["Bounds"]:
        left = max(self.left, other.left)
        top = max(self.top, other.top)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        if right <= left or bottom <= top:
            return None
        return Bounds(left, top, right, bottom)

    def to_dict(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_rect(cls, rect: Any) -> "Bounds":
        """Build bounds from a 4-tuple/list ``(left, top, right, bottom)``."""

        if isinstance(rect, (list, tuple)) and len(rect) == 4:
            left, top, right, bottom = (int(value or 0) for value in rect)
            return cls(left, top, right, bottom)
        if isinstance(rect, dict):
            if all(key in rect for key in ("left", "top", "right", "bottom")):
                return cls(
                    int(rect["left"]),
                    int(rect["top"]),
                    int(rect["right"]),
                    int(rect["bottom"]),
                )
            left = int(rect.get("left", rect.get("x", 0)) or 0)
            top = int(rect.get("top", rect.get("y", 0)) or 0)
            width = int(rect.get("width", 0) or 0)
            height = int(rect.get("height", 0) or 0)
            return cls(left, top, left + width, top + height)
        return cls(0, 0, 0, 0)


@dataclass(frozen=True)
class TextObservation:
    """A run of visible text located on screen.

    Either comes from OCR (with a ``confidence`` in [0, 1]) or from UI
    Automation's accessible name/value (``confidence`` typically 1.0).
    """

    text: str
    bounds: Bounds
    confidence: float = 1.0
    source: ObservationSource = ObservationSource.OCR

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "bounds": self.bounds.to_dict(),
            "confidence": round(float(self.confidence), 4),
            "source": self.source.value,
        }


@dataclass
class VisualElement:
    """One interactable or readable element on screen.

    ``id`` is a stable, per-observation identifier an action references so that
    coordinates are never trusted raw: the action points at an element, and the
    executor validates it against the observation that produced it.
    """

    id: str
    type: ElementType
    name: str = ""
    bounds: Bounds = field(default_factory=lambda: Bounds(0, 0, 0, 0))
    confidence: float = 1.0
    source: ObservationSource = ObservationSource.UIA
    enabled: bool = True
    visible: bool = True
    focused: bool = False
    selected: bool = False
    value: str = ""
    #: Observation this element belongs to (for staleness checks).
    observation_id: str = ""
    #: Owning window handle / application, when known.
    window_hwnd: int = 0
    application: str = ""
    #: Automation/class identifier for diagnostics and dedup.
    role: str = ""
    #: Free-form relationships ("parent=element_1", "inside=window_x").
    relationships: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "bounds": self.bounds.to_dict(),
            "confidence": round(float(self.confidence), 4),
            "source": self.source.value,
            "enabled": self.enabled,
            "visible": self.visible,
            "focused": self.focused,
            "selected": self.selected,
            "value": self.value,
            "observation_id": self.observation_id,
            "window_hwnd": self.window_hwnd,
            "application": self.application,
            "role": self.role,
            "relationships": dict(self.relationships),
        }


@dataclass(frozen=True)
class WindowState:
    """A visible top-level window with its geometry and identity."""

    hwnd: int
    title: str
    application: str = ""
    class_name: str = ""
    pid: int = 0
    bounds: Bounds = field(default_factory=lambda: Bounds(0, 0, 0, 0))
    focused: bool = False
    minimized: bool = False
    visible: bool = True
    element_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hwnd": self.hwnd,
            "title": self.title,
            "application": self.application,
            "class_name": self.class_name,
            "pid": self.pid,
            "bounds": self.bounds.to_dict(),
            "focused": self.focused,
            "minimized": self.minimized,
            "visible": self.visible,
            "element_count": self.element_count,
        }


@dataclass(frozen=True)
class ScreenInfo:
    """Overall screen geometry and monitor layout."""

    width: int
    height: int
    monitors: tuple[dict[str, Any], ...] = ()
    scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "monitors": list(self.monitors),
            "scale": self.scale,
        }


@dataclass
class VisualState:
    """A complete structured observation of the desktop at one instant.

    This is the object reasoning/planning/verification consume. Its
    ``observation_id`` and ``timestamp`` are what make coordinate safety
    possible: an action references an element id and an observation id, and the
    executor rejects the action when the observation is stale or the element is
    no longer where it was.
    """

    observation_id: str = field(default_factory=lambda: f"obs_{uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)
    screen: ScreenInfo = field(default_factory=lambda: ScreenInfo(0, 0))
    active_window: Optional[WindowState] = None
    windows: list[WindowState] = field(default_factory=list)
    elements: list[VisualElement] = field(default_factory=list)
    text: list[TextObservation] = field(default_factory=list)
    #: Which perception layers actually contributed (for honest reporting).
    sources: list[str] = field(default_factory=list)
    #: Optional VLM interpretation, never an instruction.
    vlm_interpretation: Optional[dict[str, Any]] = None
    detected_application: str = ""
    screenshot_metadata: dict[str, Any] = field(default_factory=dict)
    region: Optional[Bounds] = None
    notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # -- derived accessors ---------------------------------------------------

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.timestamp)

    def is_stale(self, *, max_age_seconds: float) -> bool:
        return self.age_seconds > max(0.0, max_age_seconds)

    def element(self, element_id: str) -> Optional[VisualElement]:
        for element in self.elements:
            if element.id == element_id:
                return element
        return None

    def elements_in(self, bounds: Bounds) -> list[VisualElement]:
        found: list[VisualElement] = []
        for element in self.elements:
            if bounds.intersection(element.bounds) is not None:
                found.append(element)
        return found

    def text_in(self, bounds: Bounds) -> list[TextObservation]:
        found: list[TextObservation] = []
        for run in self.text:
            if bounds.intersection(run.bounds) is not None:
                found.append(run)
        return found

    def full_text(self) -> str:
        """All visible text joined in reading order (top-to-bottom, left-to-right)."""

        ordered = sorted(self.text, key=lambda run: (run.bounds.top, run.bounds.left))
        return "\n".join(run.text for run in ordered if run.text)

    def find_by_name(self, query: str, *, element_type: ElementType | None = None) -> list[VisualElement]:
        """Return elements whose name fuzzy-matches ``query``, best first."""

        needle = str(query or "").strip().casefold()
        if not needle:
            return []
        scored: list[tuple[int, VisualElement]] = []
        for element in self.elements:
            if element_type is not None and element.type != element_type:
                continue
            name = element.name.casefold()
            value = element.value.casefold()
            score = 0
            if name == needle:
                score = 100
            elif name.startswith(needle):
                score = 80
            elif needle in name:
                score = 60
            elif value == needle:
                score = 50
            elif needle in value:
                score = 30
            if score:
                # Prefer higher confidence and, among equals, the center.
                scored.append((score, element))
        scored.sort(key=lambda item: (item[0], item[1].confidence), reverse=True)
        return [element for _, element in scored]

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "timestamp": self.timestamp,
            "screen": self.screen.to_dict(),
            "active_window": self.active_window.to_dict() if self.active_window else None,
            "windows": [window.to_dict() for window in self.windows],
            "elements": [element.to_dict() for element in self.elements],
            "text": [run.to_dict() for run in self.text],
            "sources": list(self.sources),
            "vlm_interpretation": self.vlm_interpretation,
            "detected_application": self.detected_application,
            "screenshot_metadata": dict(self.screenshot_metadata),
            "region": self.region.to_dict() if self.region else None,
            "notes": list(self.notes),
            "errors": list(self.errors),
            "age_seconds": round(self.age_seconds, 3),
            "timestamp_iso": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
        }
