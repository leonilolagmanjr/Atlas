"""Layered perception orchestrator for Atlas.

This is where the perception layers compose into one :class:`VisualState`::

    Layer 1  Windows UI Automation   (computer.vision.uia)
    Layer 2  Local OCR               (computer.vision.ocr)
    Layer 3  Image processing        (computer.vision.imaging)
    Layer 4  Local VLM (optional)    (computer.vision.providers)

The strategy is *cheapest-first* and *model-last*:

1. Capture only the region that is needed (active window / explicit region /
   full screen) and downscale it, so OCR and VLM cost stays low.
2. Read the UI Automation tree. Where UIA exposes controls, those become typed
   elements; OCR is then only run for regions UIA did not cover, or when the
   caller explicitly asks for text.
3. Fall back to OCR for text UIA cannot expose.
4. Consult the VLM only when requested and required, and never for every call.

A :class:`PerceptionEngine` owns the injectable providers so tests drive it with
fakes and no GUI. It caches the last observation and can detect change so the
executor can decide whether a re-observe is warranted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from config import (
    VISION_CONFIDENCE_THRESHOLD,
    VISION_MAX_IMAGE_SIZE,
    VISION_MAX_VLM_CALLS,
    VISION_PREFER_REGION_OCR,
    VISION_TIMEOUT,
)

from computer.vision import imaging
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
from computer.vision.ocr import OcrProvider, build_ocr_provider
from computer.vision.providers import (
    NullVisionProvider,
    VisionProvider,
    build_vision_provider,
    proposed_elements,
)
from computer.vision.uia import (
    UiaWindow,
    parse_uia_tree,
    read_uia_windows,
    uia_elements_to_visual,
)

logger = logging.getLogger(__name__)


@dataclass
class ObserveRequest:
    """What the caller wants observed. Everything has a safe default."""

    application: str = ""
    window_hwnd: int = 0
    region: Optional[Bounds] = None
    include_ocr: bool = True
    include_uia: bool = True
    allow_vlm: bool = False
    vlm_question: str = ""
    full_screen: bool = False
    max_elements: int = 200


class PerceptionEngine:
    """Compose capture, UIA, OCR, image processing, and the VLM into a VisualState.

    All providers are injectable. With the defaults, a machine that has neither
    tesseract nor a local vision model still produces UIA + image-processing
    observations; a machine with none of them still produces a window-only
    observation. Nothing here executes an action.
    """

    def __init__(
        self,
        *,
        ocr: OcrProvider | None = None,
        vision: VisionProvider | None = None,
        windows_provider: Any | None = None,
        capture: Any | None = None,
        max_vlm_calls: int = VISION_MAX_VLM_CALLS,
        max_image_size: int = VISION_MAX_IMAGE_SIZE,
    ) -> None:
        self._ocr = ocr if ocr is not None else build_ocr_provider()
        self._vision = vision if vision is not None else NullVisionProvider()
        # Injectable seams: tests supply fakes; production uses the real readers.
        self._windows_provider = windows_provider or _default_windows_provider
        self._capture = capture or imaging.capture_region
        self._max_vlm_calls = max(0, int(max_vlm_calls))
        self._max_image_size = max(0, int(max_image_size))
        self._vlm_calls = 0
        self._last_state: VisualState | None = None
        self._last_screenshot: Any | None = None

    # -- configuration -------------------------------------------------------

    @property
    def ocr_available(self) -> bool:
        return self._ocr.available()

    @property
    def vision_available(self) -> bool:
        return self._vision.available()

    @property
    def last_state(self) -> VisualState | None:
        return self._last_state

    # -- observation ---------------------------------------------------------

    def observe(self, request: ObserveRequest | None = None) -> VisualState:
        """Capture and analyze the screen, returning a structured VisualState."""

        request = request or ObserveRequest()
        self._vlm_calls = 0
        state = VisualState()
        state.region = request.region

        windows = self._windows_provider()
        uia_windows = parse_uia_tree(windows) if windows else []
        active_window = self._select_active_window(uia_windows, request)
        if active_window is not None:
            state.active_window = _window_state(active_window)
            state.detected_application = active_window.application or active_window.title

        state.windows = [_window_state(window) for window in uia_windows]
        state.screen = self._screen_info()

        target_bounds = self._target_bounds(active_window, request)
        screenshot = self._capture_screenshot(target_bounds, request, state)
        if screenshot is not None:
            state.screenshot_metadata = screenshot.to_metadata()
            state.screenshot_metadata["change_ratio"] = self._change_ratio(screenshot)

        if request.include_uia:
            self._collect_uia(state, active_window, request)

        if request.include_ocr:
            self._collect_ocr(state, screenshot, active_window, request)

        self._collect_image_elements(state, screenshot, active_window)

        if request.allow_vlm and screenshot is not None:
            self._enrich_with_vlm(state, screenshot, request)

        state.text.sort(key=lambda run: (run.bounds.top, run.bounds.left))
        self._last_state = state
        return state

    def find(self, query: str, request: ObserveRequest | None = None) -> list[VisualElement]:
        """Observe and return elements matching a text/semantic query, best first."""

        state = self.observe(request)
        matches = state.find_by_name(query)
        if matches:
            return matches
        # A query that names a type ("the search box") should also match by
        # element type, not only by visible label.
        wanted_type = _type_from_query(query)
        if wanted_type is not None:
            typed = [element for element in state.elements if element.type == wanted_type]
            typed.sort(key=lambda item: item.confidence, reverse=True)
            return typed
        return []

    # -- layer helpers -------------------------------------------------------

    def _select_active_window(self, windows: list[UiaWindow], request: ObserveRequest) -> Optional[UiaWindow]:
        if request.window_hwnd:
            for window in windows:
                if window.hwnd == int(request.window_hwnd):
                    return window
        if request.application:
            needle = request.application.strip().casefold()
            scored: list[tuple[int, UiaWindow]] = []
            for window in windows:
                title = window.title.casefold()
                application = window.application.casefold()
                score = 0
                if title == needle:
                    score = 100
                elif needle and needle in title:
                    score = 60
                elif needle and needle in application:
                    score = 40
                if score:
                    scored.append((score + (10 if window.focused else 0), window))
            if scored:
                scored.sort(key=lambda item: item[0], reverse=True)
                return scored[0][1]
            return None
        for window in windows:
            if window.focused:
                return window
        return windows[0] if windows else None

    def _screen_info(self) -> ScreenInfo:
        bounds = imaging.virtual_screen_bounds()
        return ScreenInfo(
            width=bounds.width,
            height=bounds.height,
            monitors=imaging.monitors(),
        )

    def _target_bounds(self, active_window: Optional[UiaWindow], request: ObserveRequest) -> Optional[Bounds]:
        if request.region is not None:
            return request.region
        if request.full_screen:
            return None
        if active_window is not None and active_window.bounds.width > 0:
            return active_window.bounds
        return None

    def _capture_screenshot(self, bounds: Optional[Bounds], request: ObserveRequest, state: VisualState):
        if not imaging.screen_capture_available():
            state.notes.append("Screen capture is only available on Windows.")
            return None
        try:
            screenshot = self._capture(bounds, max_size=self._max_image_size if request.include_ocr else 0)
        except Exception as exc:  # noqa: BLE001 - capture is best-effort
            state.errors.append(f"screen_capture_failed: {exc!r}")
            return None
        # Report the captured region, which may be a specific window/region.
        return screenshot

    def _collect_uia(
        self,
        state: VisualState,
        active_window: Optional[UiaWindow],
        request: ObserveRequest,
    ) -> None:
        if active_window is None:
            return
        visual = uia_elements_to_visual(
            active_window,
            observation_id=state.observation_id,
            max_elements=request.max_elements,
            min_confidence=min(0.3, VISION_CONFIDENCE_THRESHOLD),
        )
        state.elements.extend(visual)
        state.sources.append("uia")
        for element in visual:
            if element.name:
                state.text.append(
                    TextObservation(
                        text=element.name,
                        bounds=element.bounds,
                        confidence=element.confidence,
                        source=ObservationSource.UIA,
                    )
                )

    def _collect_ocr(
        self,
        state: VisualState,
        screenshot: Any,
        active_window: Optional[UiaWindow],
        request: ObserveRequest,
    ) -> None:
        if screenshot is None:
            return
        if not self._ocr.available():
            state.notes.append("Local OCR is not installed; text may be incomplete.")
            return
        try:
            runs = self._ocr.recognize(screenshot.data)
        except Exception as exc:  # noqa: BLE001 - OCR is best-effort
            state.errors.append(f"ocr_failed: {exc!r}")
            return
        # Region preference: when a window is known, keep OCR text inside it so
        # the caller does not repeatedly process the entire screen.
        region = self._target_bounds(active_window, request) if VISION_PREFER_REGION_OCR else None
        kept: list[TextObservation] = []
        for run in runs:
            if region is not None and region.intersection(run.bounds) is None:
                continue
            kept.append(run)
        state.text.extend(kept)
        if kept:
            state.sources.append("ocr")
        # OCR text can also become a targetable element when it looks actionable.
        state.elements.extend(self._ocr_elements(kept, state.observation_id, active_window))

    def _ocr_elements(
        self,
        runs: list[TextObservation],
        observation_id: str,
        active_window: Optional[UiaWindow],
    ) -> list[VisualElement]:
        elements: list[VisualElement] = []
        for index, run in enumerate(runs):
            label = run.text.strip()
            if not label:
                continue
            element_type = _guess_type_from_label(label)
            elements.append(
                VisualElement(
                    id=f"ocr_{index}",
                    type=element_type,
                    name=label,
                    bounds=run.bounds,
                    confidence=run.confidence,
                    source=ObservationSource.OCR,
                    observation_id=observation_id,
                    window_hwnd=active_window.hwnd if active_window else 0,
                    application=active_window.application if active_window else "",
                )
            )
        return elements

    def _collect_image_elements(self, state: VisualState, screenshot: Any, active_window: Optional[UiaWindow]) -> None:
        if screenshot is None or not imaging.imaging_available():
            return
        # Deterministic color-region localization is a coarse affordance hint.
        regions = imaging.find_color_regions(
            screenshot,
            lower=(0, 90, 190),
            upper=(60, 150, 255),
            min_area=400,
            max_regions=10,
        )
        for index, bounds in enumerate(regions):
            state.elements.append(
                VisualElement(
                    id=f"image_{index}",
                    type=ElementType.UNKNOWN,
                    name="",
                    bounds=bounds,
                    confidence=0.35,
                    source=ObservationSource.IMAGE,
                    observation_id=state.observation_id,
                    window_hwnd=active_window.hwnd if active_window else 0,
                )
            )
        if regions:
            state.sources.append("image")

    def _enrich_with_vlm(self, state: VisualState, screenshot: Any, request: ObserveRequest) -> None:
        if not self._vision.available():
            state.notes.append("No local vision model is configured; VLM enrichment was skipped.")
            return
        if self._vlm_calls >= self._max_vlm_calls:
            return
        task = "locate_element" if request.vlm_question else "describe_screen"
        context = {
            "application": state.detected_application,
            "window_title": state.active_window.title if state.active_window else "",
            "known_text": state.full_text()[:500],
        }
        try:
            result = self._vision.query(
                image=screenshot.data,
                task=task,
                question=request.vlm_question,
                context=context,
                timeout=VISION_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001 - VLM is optional
            state.errors.append(f"vlm_failed: {exc!r}")
            return
        self._vlm_calls += 1
        state.vlm_interpretation = result
        state.sources.append("vlm")
        for label, element_type, bounds, confidence in proposed_elements(result):
            if confidence < VISION_CONFIDENCE_THRESHOLD:
                continue
            state.elements.append(
                VisualElement(
                    id=f"vlm_{len(state.elements)}",
                    type=_coerce_element_type(element_type),
                    name=label,
                    bounds=bounds,
                    confidence=confidence,
                    source=ObservationSource.VLM,
                    observation_id=state.observation_id,
                    window_hwnd=state.active_window.hwnd if state.active_window else 0,
                )
            )

    # -- change detection ----------------------------------------------------

    def _change_ratio(self, screenshot: Any) -> float:
        """Coarse change ratio vs. the previous capture (0.0 = first capture)."""

        previous = self._last_screenshot
        self._last_screenshot = screenshot
        if previous is None:
            return 0.0
        try:
            return imaging.image_difference(previous, screenshot)
        except Exception:  # noqa: BLE001 - change detection is best-effort
            return 0.0


def _window_state(window: UiaWindow) -> WindowState:
    return WindowState(
        hwnd=window.hwnd,
        title=window.title,
        application=window.application,
        class_name=window.class_name,
        pid=window.pid,
        bounds=window.bounds,
        focused=window.focused,
        minimized=window.minimized,
        element_count=len(window.elements),
    )


_ACTION_WORD_HINTS: tuple[str, ...] = (
    "button",
    "submit",
    "search",
    "continue",
    "next",
    "back",
    "cancel",
    "ok",
    "close",
    "sign in",
    "log in",
    "login",
    "download",
    "open",
    "save",
    "allow",
    "accept",
    "agree",
    "menu",
)


def _guess_type_from_label(label: str) -> ElementType:
    """Heuristically type an OCR text run from how it reads.

    OCR gives text, not widgets, so this classifies obvious affordances as
    buttons/links/fields while leaving plain prose as text. It is a hint, not a
    fact, and carries OCR confidence accordingly.
    """

    lowered = label.strip().casefold()
    if not lowered:
        return ElementType.TEXT
    if lowered in {"search", "search…", "search...", "type here", "enter text", "search or enter address"}:
        return ElementType.TEXT_FIELD
    if any(lowered == hint or lowered.startswith(f"{hint} ") for hint in _ACTION_WORD_HINTS):
        return ElementType.BUTTON
    if lowered.startswith("http://") or lowered.startswith("https://") or lowered.startswith("www."):
        return ElementType.LINK
    return ElementType.TEXT


_TYPE_QUERY_HINTS: dict[ElementType, tuple[str, ...]] = {
    ElementType.TEXT_FIELD: ("search box", "search field", "text field", "textbox", "input", "address bar", "url bar", "text box", "field"),
    ElementType.BUTTON: ("button", "btn"),
    ElementType.LINK: ("link", "hyperlink"),
    ElementType.CHECKBOX: ("checkbox", "check box", "tick box"),
    ElementType.RADIO: ("radio", "radio button", "option"),
    ElementType.DROPDOWN: ("dropdown", "drop down", "combo", "select", "menu"),
    ElementType.TAB: ("tab",),
}


def _type_from_query(query: str) -> Optional[ElementType]:
    lowered = str(query or "").casefold()
    for element_type, hints in _TYPE_QUERY_HINTS.items():
        if any(hint in lowered for hint in hints):
            return element_type
    return None


def _coerce_element_type(value: str) -> ElementType:
    try:
        return ElementType(str(value).strip().casefold())
    except ValueError:
        return ElementType.UNKNOWN


def _default_windows_provider() -> list[dict[str, Any]]:
    """Default UIA source: read the real tree, or synthesize a Win32 fallback.

    When UI Automation is unavailable, the existing Win32 window inventory from
    :mod:`computer.perception` still provides the window layer, so Atlas always
    at least knows what windows are open.
    """

    windows = read_uia_windows()
    if windows:
        return windows
    try:
        from computer.perception import list_windows

        fallback: list[dict[str, Any]] = []
        for window in list_windows():
            fallback.append(
                {
                    "hwnd": window.hwnd,
                    "title": window.title,
                    "class_name": window.class_name,
                    "pid": window.pid,
                    "application": window.executable,
                    "bounds": list(window.rect),
                    "focused": window.focused,
                    "minimized": window.minimized,
                    "elements": [],
                }
            )
        return fallback
    except Exception:  # noqa: BLE001 - non-Windows or unreadable
        return []


def build_perception_engine(
    *,
    vision_enabled: bool,
    provider: str,
    model: str,
    timeout: float,
    ocr_languages: str = "eng",
    max_vlm_calls: int = VISION_MAX_VLM_CALLS,
    max_image_size: int = VISION_MAX_IMAGE_SIZE,
) -> PerceptionEngine:
    """Construct the engine from configuration, degrading each layer honestly."""

    ocr = build_ocr_provider(name="tesseract", languages=ocr_languages, timeout=min(timeout, 30.0))
    if not vision_enabled:
        vision: VisionProvider = NullVisionProvider()
    else:
        vision = build_vision_provider(provider=provider, model=model, timeout=timeout)
    return PerceptionEngine(
        ocr=ocr,
        vision=vision,
        max_vlm_calls=max_vlm_calls,
        max_image_size=max_image_size,
    )
