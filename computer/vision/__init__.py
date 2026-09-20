"""Local computer-vision perception layer for Atlas ("Eyes").

This package implements the perception half of the computer-interaction loop::

    SCREEN -> PERCEPTION -> STRUCTURED VISUAL STATE -> REASONING -> ACTION
           -> OBSERVATION -> VERIFICATION

It is layered so that the cheapest, most reliable source is tried first and a
language model is only consulted when deterministic perception is insufficient::

    Layer 1  Windows UI Automation  (computer.vision.uia)
    Layer 2  Local OCR              (computer.vision.ocr)
    Layer 3  Image processing       (computer.vision.imaging)
    Layer 4  Local VLM              (computer.vision.providers)

The layers compose into a single typed :class:`~computer.vision.models.VisualState`
by :mod:`computer.vision.perception`. Everything here is optional and degrades
gracefully: with no VLM, UIA + OCR still work; with no OCR, UIA still works;
with neither, existing Atlas functionality is unchanged.

Screen content is untrusted external data. Nothing in this package executes
actions: it only observes. Consequential actions remain permission-gated through
the existing tool/permission/executor pipeline.
"""

from __future__ import annotations

from computer.vision.models import (
    Bounds,
    ElementType,
    ObservationSource,
    ScreenInfo,
    TextObservation,
    VisualElement,
    VisualState,
    WindowState,
)

__all__ = [
    "Bounds",
    "ElementType",
    "ObservationSource",
    "ScreenInfo",
    "TextObservation",
    "VisualElement",
    "VisualState",
    "WindowState",
]
