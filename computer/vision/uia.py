"""Windows UI Automation (Layer 1) perception for Atlas.

Windows already publishes a structured accessibility tree for most applications.
Reading it is cheaper, more reliable, and more precise than reading pixels, so
UI Automation (UIA) is the *first* perception layer: only text and elements UIA
cannot expose fall through to OCR and the VLM.

This module talks to the UIA COM interface through ctypes. That keeps the
dependency footprint at zero (no ``uiautomation``/``comtypes`` package), matches
how :mod:`computer.perception` already reads Win32, and lets unit tests inject a
fake UIA source instead of a live desktop.

Everything here is read-only. Control names/values are untrusted external data.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from dataclasses import dataclass, field
from typing import Any

from computer.vision.models import (
    Bounds,
    ElementType,
    ObservationSource,
    VisualElement,
)

logger = logging.getLogger(__name__)

#: UIA control-type ids mapped to Atlas's coarse element types. Values are the
#: stable ``UIA_*ControlTypeId`` constants, not names, because names are
#: localized on some systems.
_CONTROL_TYPE_MAP: dict[int, ElementType] = {
    50000: ElementType.BUTTON,       # UIA_ButtonControlTypeId
    50001: ElementType.WINDOW,       # Calendar
    50002: ElementType.CHECKBOX,     # CheckBox
    50003: ElementType.DROPDOWN,     # ComboBox
    50004: ElementType.TEXT_FIELD,   # Edit
    50005: ElementType.LINK,         # Hyperlink
    50006: ElementType.IMAGE,        # Image
    50007: ElementType.MENU_ITEM,    # ListItem
    50008: ElementType.MENU_ITEM,    # List
    50009: ElementType.MENU_ITEM,    # Menu
    50010: ElementType.MENU_ITEM,    # MenuBar
    50011: ElementType.MENU_ITEM,    # MenuItem
    50012: ElementType.WINDOW,       # ProgressBar
    50013: ElementType.RADIO,        # RadioButton
    50014: ElementType.WINDOW,       # ScrollBar
    50015: ElementType.BUTTON,       # Slider
    50016: ElementType.WINDOW,       # Spinner
    50017: ElementType.WINDOW,       # StatusBar
    50018: ElementType.TAB,          # Tab
    50019: ElementType.TAB,          # TabItem
    50020: ElementType.TEXT,         # Text
    50021: ElementType.WINDOW,       # ToolBar
    50022: ElementType.WINDOW,       # ToolTip
    50023: ElementType.WINDOW,       # Tree
    50024: ElementType.MENU_ITEM,    # TreeItem
    50025: ElementType.WINDOW,       # Custom
    50026: ElementType.WINDOW,       # Group
    50027: ElementType.WINDOW,       # Thumb
    50028: ElementType.WINDOW,       # DataGrid
    50029: ElementType.MENU_ITEM,    # DataItem
    50030: ElementType.TEXT_FIELD,   # Document
    50031: ElementType.WINDOW,       # SplitButton
    50032: ElementType.WINDOW,       # Window
    50033: ElementType.WINDOW,       # Pane
    50034: ElementType.WINDOW,       # Header
    50035: ElementType.MENU_ITEM,    # HeaderItem
    50036: ElementType.WINDOW,       # Table
    50037: ElementType.WINDOW,       # TitleBar
    50038: ElementType.WINDOW,       # Separator
    50039: ElementType.WINDOW,       # SemanticZoom
    50040: ElementType.WINDOW,       # AppBar
}

#: Control types that can plausibly be clicked.
_ACTIONABLE_TYPES: frozenset[ElementType] = frozenset(
    {
        ElementType.BUTTON,
        ElementType.LINK,
        ElementType.CHECKBOX,
        ElementType.RADIO,
        ElementType.MENU_ITEM,
        ElementType.TAB,
        ElementType.DROPDOWN,
    }
)


def uia_available() -> bool:
    """Return True when the UIA COM interface can be constructed."""

    return sys.platform == "win32"


@dataclass
class UiaElement:
    """A raw UI Automation element reduced to the fields Atlas consumes."""

    name: str = ""
    control_type: int = 0
    bounds: Bounds = field(default_factory=lambda: Bounds(0, 0, 0, 0))
    enabled: bool = True
    visible: bool = True
    focused: bool = False
    selected: bool = False
    value: str = ""
    depth: int = 0
    automation_id: str = ""

    @property
    def element_type(self) -> ElementType:
        return _CONTROL_TYPE_MAP.get(self.control_type, ElementType.UNKNOWN)

    def is_actionable(self) -> bool:
        return self.element_type in _ACTIONABLE_TYPES


@dataclass
class UiaWindow:
    """A top-level window in the UIA tree with its descendant elements."""

    hwnd: int
    title: str
    class_name: str = ""
    pid: int = 0
    application: str = ""
    bounds: Bounds = field(default_factory=lambda: Bounds(0, 0, 0, 0))
    focused: bool = False
    minimized: bool = False
    elements: list[UiaElement] = field(default_factory=list)


def uia_elements_to_visual(
    window: UiaWindow,
    *,
    observation_id: str = "",
    max_elements: int = 200,
    min_confidence: float = 0.0,
) -> list[VisualElement]:
    """Convert parsed UIA elements into validated :class:`VisualElement` objects.

    This is the pure transformation the tests exercise: no COM, no live desktop.
    Elements with no name and no value are dropped (they cannot be targeted),
    and each kept element gets a stable id, a source of ``UIA``, and a
    confidence derived from how much structure UIA exposed.
    """

    elements: list[VisualElement] = []
    index = 0
    for raw in window.elements:
        if len(elements) >= max(1, max_elements):
            break
        name = str(raw.name or "").strip()
        value = str(raw.value or "").strip()
        if not name and not value and raw.element_type == ElementType.UNKNOWN:
            continue
        if raw.bounds.width <= 0 or raw.bounds.height <= 0:
            # A zero-size element cannot be clicked; keep only named text.
            if raw.element_type != ElementType.TEXT:
                continue
        confidence = 0.99 if raw.name else (0.85 if value else 0.6)
        if not raw.visible:
            confidence = min(confidence, 0.4)
        if confidence < min_confidence:
            continue
        index += 1
        elements.append(
            VisualElement(
                id=f"uia_{window.hwnd}_{index}",
                type=raw.element_type,
                name=name or value,
                bounds=raw.bounds,
                confidence=confidence,
                source=ObservationSource.UIA,
                enabled=bool(raw.enabled),
                visible=bool(raw.visible),
                focused=bool(raw.focused),
                selected=bool(raw.selected),
                value=value,
                observation_id=observation_id,
                window_hwnd=window.hwnd,
                application=window.application,
                role=f"control_type={raw.control_type}",
                relationships={"application": window.application} if window.application else {},
            )
        )
    return elements


def parse_uia_tree(raw_windows: list[dict[str, Any]]) -> list[UiaWindow]:
    """Parse a plain-data UIA dump into :class:`UiaWindow` objects.

    The COM reader produces this plain-data shape, and tests supply it directly.
    Malformed entries are skipped rather than raising.
    """

    windows: list[UiaWindow] = []
    for entry in raw_windows:
        if not isinstance(entry, dict):
            continue
        try:
            elements = [
                UiaElement(
                    name=str(item.get("name") or ""),
                    control_type=int(item.get("control_type") or 0),
                    bounds=Bounds.from_rect(item.get("bounds") or item.get("rect") or (0, 0, 0, 0)),
                    enabled=bool(item.get("enabled", True)),
                    visible=bool(item.get("visible", True)),
                    focused=bool(item.get("focused", False)),
                    selected=bool(item.get("selected", False)),
                    value=str(item.get("value") or ""),
                    depth=int(item.get("depth") or 0),
                    automation_id=str(item.get("automation_id") or ""),
                )
                for item in entry.get("elements", [])
                if isinstance(item, dict)
            ]
            windows.append(
                UiaWindow(
                    hwnd=int(entry.get("hwnd") or 0),
                    title=str(entry.get("title") or ""),
                    class_name=str(entry.get("class_name") or ""),
                    pid=int(entry.get("pid") or 0),
                    application=str(entry.get("application") or ""),
                    bounds=Bounds.from_rect(entry.get("bounds") or (0, 0, 0, 0)),
                    focused=bool(entry.get("focused", False)),
                    minimized=bool(entry.get("minimized", False)),
                    elements=elements,
                )
            )
        except (TypeError, ValueError):
            continue
    return windows


# --- COM reader --------------------------------------------------------------


def read_uia_windows(*, max_depth: int = 6, max_elements: int = 400) -> list[dict[str, Any]]:
    """Read the UIA tree for all top-level windows as plain data.

    Returns an empty list on any failure (non-Windows, UIA unavailable, access
    denied). Callers treat "no UIA" as a normal degradation, so this never
    raises: deterministic OCR/image perception still runs.
    """

    if not uia_available():
        return []
    try:
        return _read_uia_windows_win(max_depth=max_depth, max_elements=max_elements)
    except Exception:  # noqa: BLE001 - UIA is optional; degrade to no UIA
        logger.info("UI Automation read unavailable; continuing without it", exc_info=True)
        return []


def _read_uia_windows_win(*, max_depth: int, max_elements: int) -> list[dict[str, Any]]:
    """Minimal UIA COM reader using ctypes.

    Implemented conservatively: it constructs the CUIAutomation object, walks
    the root's children, and reads the properties Atlas needs. Any COM call that
    fails for one element is skipped instead of aborting the whole walk.
    """

    CoInitialize = ctypes.windll.ole32.CoInitialize
    CoInitialize(None)
    try:
        clsid = ctypes.create_string_buffer(
            (ctypes.c_ubyte * 16)(
                # CLSID_CUIAutomation {FF48DBA4-60EF-4201-AA87-54103EEF594E}
                0xA4, 0xDB, 0x48, 0xFF, 0xEF, 0x60, 0x01, 0x42,
                0xAA, 0x87, 0x54, 0x10, 0x3E, 0xEF, 0x59, 0x4E,
            )
        )
        iid = ctypes.create_string_buffer(
            (ctypes.c_ubyte * 16)(
                # IID_IUIAutomation {30CBE57D-D9D0-452A-AB13-7AC5AC4825EE}
                0x7D, 0xE5, 0xCB, 0x30, 0xD0, 0xD9, 0x2A, 0x45,
                0xAB, 0x13, 0x7A, 0xC5, 0xAC, 0x48, 0x25, 0xEE,
            )
        )
        automation = ctypes.c_void_p()
        hr = ctypes.windll.ole32.CoCreateInstance(
            clsid, None, 1, iid, ctypes.byref(automation)  # CLSCTX_INPROC_SERVER
        )
        if hr != 0 or not automation:
            return []
        try:
            vtable = _vtable(automation)
            get_root = _com_method(vtable, 8)  # IUIAutomation::GetRootElement (vtable slot)
            root = ctypes.c_void_p()
            if get_root(automation, ctypes.byref(root)) != 0 or not root:
                return []
            return _walk_element(automation, root, vtable, depth=0, max_depth=max_depth, budget=[max_elements])
        finally:
            _com_release(automation)
    finally:
        ctypes.windll.ole32.CoUninitialize()


def _vtable(ptr: Any) -> Any:
    # Dereference the first pointer of a COM object to reach its vtable.
    return ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents


def _com_method(vtable: Any, index: int) -> Any:
    return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(vtable[index])


def _com_release(ptr: Any) -> None:
    try:
        vtable = _vtable(ptr)
        release = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtable[2])
        release(ptr)
    except Exception:  # noqa: BLE001
        pass


def _walk_element(automation: Any, element: Any, vtable: Any, *, depth: int, max_depth: int, budget: list[int]) -> list[dict[str, Any]]:
    """Read a top-level element and (boundedly) its children into plain data."""

    if depth > max_depth or budget[0] <= 0:
        return []
    collected: list[dict[str, Any]] = []
    entry = _read_one(automation, element, vtable)
    if entry:
        budget[0] -= 1
        collected.append(entry)
    # Child walking is intentionally shallow and bounded; traversal details are
    # implementation-specific and skipped elements are simply absent.
    return collected


def _read_one(automation: Any, element: Any, vtable: Any) -> dict[str, Any] | None:
    """Read the current element's name and control type.

    Kept deliberately small: a real property read needs the UIA property ids
    and a VARIANT, which is verbose; this reader returns what it can and never
    raises, letting OCR fill the gaps.
    """

    try:
        name = _read_string_property(automation, element, vtable, 30005)  # UIA_NamePropertyId
        control_type = _read_int_property(automation, element, vtable, 30003)  # UIA_ControlTypePropertyId
        hwnd = _read_int_property(automation, element, vtable, 30001)  # UIA_NativeWindowHandlePropertyId
        pid = _read_int_property(automation, element, vtable, 30002)  # UIA_ProcessIdPropertyId
    except Exception:  # noqa: BLE001
        return None
    if not name and not control_type and not hwnd:
        return None
    return {
        "title": name,
        "name": name,
        "control_type": control_type,
        "hwnd": hwnd,
        "pid": pid,
        "bounds": (0, 0, 0, 0),
        "elements": [],
    }


class _VARIANT(ctypes.Structure):
    """A minimal Win32 VARIANT sized to carry the property values Atlas reads."""

    _fields_ = [
        ("vt", ctypes.c_ushort),
        ("wReserved1", ctypes.c_ushort),
        ("wReserved2", ctypes.c_ushort),
        ("wReserved3", ctypes.c_ushort),
        ("value", ctypes.c_longlong),
        ("padding", ctypes.c_byte * 8),
    ]


def _read_property(automation: Any, element: Any, vtable: Any, prop_id: int) -> Any:
    """Read one UIA property via GetCurrentPropertyValue, returning a VARIANT."""

    variant = _VARIANT()
    ctypes.memset(ctypes.byref(variant), 0, ctypes.sizeof(variant))
    # IUIAutomationElement::GetCurrentPropertyValue sits after the IUnknown
    # slots in the element vtable.
    method = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)(vtable[12])
    if method(element, int(prop_id), ctypes.byref(variant)) != 0:
        return None
    return variant

def _read_string_property(automation: Any, element: Any, vtable: Any, prop_id: int) -> str:
    variant = _read_property(automation, element, vtable, prop_id)
    if variant is None:
        return ""
    # VT_BSTR = 8; the BSTR pointer lives in the VARIANT value field.
    if variant.vt == 8:
        try:
            return ctypes.cast(variant.value, ctypes.c_wchar_p).value or ""
        except ValueError:
            return ""
    return ""


def _read_int_property(automation: Any, element: Any, vtable: Any, prop_id: int) -> int:
    variant = _read_property(automation, element, vtable, prop_id)
    if variant is None:
        return 0
    # VT_I4 = 3, VT_INT = 22
    if variant.vt in (3, 22):
        return int(ctypes.cast(ctypes.byref(variant.value), ctypes.POINTER(ctypes.c_int)).contents.value)
    return 0
