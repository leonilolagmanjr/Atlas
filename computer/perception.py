"""Read-only Windows UI perception for Atlas.

Atlas can only act on a computer it can observe. This module is the observation
half of the computer-interaction loop that the executor drives:

    OBSERVE -> PLAN -> ACT -> OBSERVE -> VERIFY -> RECOVER

It reports *structured* UI facts (windows, controls, focused control, readable
text) because Windows already publishes that structure. It does not screenshot
and it does not reason about pixels. A screenshot/OCR provider would widen
coverage for applications that hide their structure, but it would implement the
same observation contract, so nothing outside this module needs to change.

Everything here is read-only and dependency-free (ctypes + user32), which is why
perception never requires confirmation.

Two honesty rules are built into the report:

* ``fidelity`` says how much structure was actually available
  (``controls`` > ``window_only`` > ``unavailable``).
* ``status`` says whether the observation itself is usable
  (``observed`` / ``target_missing`` / ``unsupported``).

Control text is read through ``SendMessageTimeoutW`` with a short timeout so a
hung application cannot freeze Atlas.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from tools.base import PermissionLevel, RiskLevel, Tool, ToolMetadata, ToolResult

#: Where an observation came from. Recorded so a future provider (UI Automation,
#: browser DOM, OCR) can be added without changing consumers.
PERCEPTION_SOURCE = "win32_ui"

#: Control class names that accept text. Covers classic Win32 edit controls and
#: the modern rich-edit controls used by Windows 11 apps (Notepad, WordPad).
TEXT_CONTROL_CLASSES: frozenset[str] = frozenset(
    {
        "edit",
        "richedit",
        "richedit20a",
        "richedit20w",
        "richedit50w",
        "richeditd2dpt",
        "textbox",
        "scintilla",
    }
)

#: How long a control may take to answer WM_GETTEXTLENGTH/WM_GETTEXT before the
#: read is abandoned. Prevents a busy application from blocking the executor.
_MESSAGE_TIMEOUT_MS = 3000
_SMTO_ABORTIFHUNG = 0x0002
_WM_GETTEXT = 0x000D
_WM_GETTEXTLENGTH = 0x000E
_MAX_TEXT_LIMIT = 200_000


def perception_available() -> bool:
    """Return True when this platform can expose structured UI state."""

    if sys.platform != "win32":
        return False
    try:
        return bool(ctypes.windll.user32)
    except (AttributeError, OSError):  # pragma: no cover - non-Windows only
        return False


def user32() -> Any:
    """Return the Windows user32 library, or raise when unavailable."""

    if not perception_available():
        raise OSError("UI observation requires Windows")
    return ctypes.windll.user32


def _hwnd_value(handle: Any) -> int:
    """Normalize a ctypes HWND (or a plain int, as tests use) to an int."""

    if handle is None:
        return 0
    value = getattr(handle, "value", handle)
    return int(value or 0)


@dataclass(frozen=True)
class WindowInfo:
    """A top-level window Atlas can see."""

    hwnd: int
    title: str
    class_name: str
    pid: int
    executable: str
    rect: tuple[int, int, int, int] = (0, 0, 0, 0)
    visible: bool = True
    minimized: bool = False
    focused: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "hwnd": self.hwnd,
            "title": self.title,
            "class_name": self.class_name,
            "pid": self.pid,
            "executable": self.executable,
            "rect": list(self.rect),
            "visible": self.visible,
            "minimized": self.minimized,
            "focused": self.focused,
            "application": self.executable or self.class_name or self.title,
        }


@dataclass(frozen=True)
class ControlInfo:
    """A control (widget) inside a window."""

    hwnd: int
    class_name: str
    name: str
    text_length: int
    editable: bool
    enabled: bool = True
    visible: bool = True
    focused: bool = False
    rect: tuple[int, int, int, int] = (0, 0, 0, 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hwnd": self.hwnd,
            "class_name": self.class_name,
            "name": self.name,
            "text_length": self.text_length,
            "editable": self.editable,
            "enabled": self.enabled,
            "visible": self.visible,
            "focused": self.focused,
            "rect": list(self.rect),
        }


@dataclass(frozen=True)
class UiObservation:
    """What Atlas could see of one application window."""

    available: bool
    status: str
    fidelity: str
    summary: str
    source: str = PERCEPTION_SOURCE
    window: dict[str, Any] | None = None
    controls: list[dict[str, Any]] = field(default_factory=list)
    focused_control: dict[str, Any] | None = None
    text: str = ""
    text_truncated: bool = False
    controls_inspected: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "status": self.status,
            "fidelity": self.fidelity,
            "source": self.source,
            "summary": self.summary,
            "window": self.window,
            "controls": self.controls,
            "focused_control": self.focused_control,
            "text": self.text,
            "text_truncated": self.text_truncated,
            "controls_inspected": self.controls_inspected,
            "error": self.error,
        }


def unavailable(summary: str, *, error: str | None = None) -> UiObservation:
    """Build the honest report used when nothing can be observed."""

    return UiObservation(
        available=False,
        status="unsupported",
        fidelity="unavailable",
        summary=summary,
        error=error,
    )


def target_missing(name: str) -> UiObservation:
    """Build the report used when the requested application has no window."""

    label = str(name or "").strip() or "the requested application"
    return UiObservation(
        available=True,
        status="target_missing",
        fidelity="unavailable",
        summary=f"No open window matches {label}.",
    )


# -- low-level reads -------------------------------------------------------------


def send_message(user32_lib: Any, handle: Any, message: int, wparam: Any, lparam: Any) -> int:
    """Send a window message with a bounded timeout when the OS supports it."""

    send_timeout = getattr(user32_lib, "SendMessageTimeoutW", None)
    if send_timeout is not None:
        result = ctypes.c_size_t(0)
        try:
            delivered = send_timeout(
                handle,
                message,
                wparam,
                lparam,
                _SMTO_ABORTIFHUNG,
                _MESSAGE_TIMEOUT_MS,
                ctypes.byref(result),
            )
        except Exception:  # noqa: BLE001 - fall back to a blocking send
            delivered = 0
        if not delivered:
            return 0
        return int(result.value)
    try:
        return int(user32_lib.SendMessageW(handle, message, wparam, lparam))
    except Exception:  # noqa: BLE001 - an unreadable control reports no text
        return 0


def read_control_text(user32_lib: Any, handle: Any, *, limit: int = _MAX_TEXT_LIMIT) -> str:
    """Read a control's current text, bounded and non-blocking."""

    length = send_message(user32_lib, handle, _WM_GETTEXTLENGTH, 0, 0)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(min(int(length), max(1, limit)) + 1)
    send_message(user32_lib, handle, _WM_GETTEXT, len(buffer), buffer)
    return buffer.value


def window_text(user32_lib: Any, handle: Any) -> str:
    """Read a window's title text."""

    length = int(user32_lib.GetWindowTextLengthW(handle))
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32_lib.GetWindowTextW(handle, buffer, len(buffer))
    return buffer.value


def class_name_of(user32_lib: Any, handle: Any) -> str:
    """Read a window or control's class name."""

    buffer = ctypes.create_unicode_buffer(256)
    user32_lib.GetClassNameW(handle, buffer, len(buffer))
    return buffer.value


def process_image_name(pid: int) -> str:
    """Return the executable file name for a process id, or an empty string."""

    if not perception_available() or not pid:
        return ""
    kernel32 = ctypes.windll.kernel32
    process = kernel32.OpenProcess(0x1000, False, int(pid))
    if not process:
        return ""
    try:
        buffer = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(len(buffer))
        if kernel32.QueryFullProcessImageNameW(process, 0, buffer, ctypes.byref(size)):
            return buffer.value.rsplit("\\", 1)[-1]
        return ""
    finally:
        kernel32.CloseHandle(process)


# -- window inventory -----------------------------------------------------------


def _rect_of(user32_lib: Any, handle: Any) -> tuple[int, int, int, int]:
    rect = wintypes.RECT()
    try:
        user32_lib.GetWindowRect(handle, ctypes.byref(rect))
    except Exception:  # noqa: BLE001 - geometry is optional metadata
        return (0, 0, 0, 0)
    return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))


def window_info(
    user32_lib: Any,
    handle: Any,
    *,
    foreground: int = 0,
    image_name: Callable[[int], str] | None = None,
) -> WindowInfo:
    """Describe one top-level window."""

    pid = wintypes.DWORD()
    try:
        user32_lib.GetWindowThreadProcessId(handle, ctypes.byref(pid))
    except Exception:  # noqa: BLE001 - a dead window has no process
        pid = wintypes.DWORD(0)
    process_id = int(pid.value)
    reader = image_name or process_image_name
    hwnd = _hwnd_value(handle)
    return WindowInfo(
        hwnd=hwnd,
        title=window_text(user32_lib, handle),
        class_name=class_name_of(user32_lib, handle),
        pid=process_id,
        executable=reader(process_id) if process_id else "",
        rect=_rect_of(user32_lib, handle),
        visible=bool(user32_lib.IsWindowVisible(handle)),
        minimized=bool(user32_lib.IsIconic(handle)),
        focused=bool(foreground) and hwnd == int(foreground),
    )


def list_windows(
    *,
    user32_lib: Any | None = None,
    include_hidden: bool = False,
    titled_only: bool = False,
    max_windows: int = 200,
    image_name: Callable[[int], str] | None = None,
) -> list[WindowInfo]:
    """Enumerate the top-level windows Windows reports for this session."""

    user32_lib = user32_lib or user32()
    foreground = _hwnd_value(user32_lib.GetForegroundWindow())
    found: list[WindowInfo] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def collect(handle: wintypes.HWND, _param: wintypes.LPARAM) -> bool:
        if len(found) >= max_windows:
            return False
        info = window_info(user32_lib, handle, foreground=foreground, image_name=image_name)
        if not include_hidden and (not info.visible or not info.title):
            return True
        if titled_only and not info.title:
            return True
        found.append(info)
        return True

    user32_lib.EnumWindows(collect, 0)
    return found


def _match_score(window: WindowInfo, query: str) -> int:
    """Score how well a window answers an application-name query."""

    if not query:
        return 0
    title = window.title.casefold()
    executable = window.executable.casefold()
    class_name = window.class_name.casefold()
    stem = executable.rsplit(".", 1)[0]
    if title == query:
        return 1000
    if title.startswith(query):
        return 800
    if query in title:
        return 600
    if executable == f"{query}.exe" or stem == query:
        return 500
    if class_name == query:
        return 400
    tokens = [token for token in query.replace(".", " ").replace("-", " ").split() if token]
    if tokens and all(token in title for token in tokens):
        return 300
    if tokens and all(token in class_name for token in tokens):
        return 200
    return 0


def matches_application(window: WindowInfo, query: str) -> bool:
    """Return True when ``window`` plausibly belongs to the named application."""

    return _match_score(window, str(query or "").strip().casefold()) > 0


def find_window(
    application: str | None = None,
    *,
    user32_lib: Any | None = None,
    timeout: float = 0.0,
    poll_interval: float = 0.25,
    include_hidden: bool = False,
    image_name: Callable[[int], str] | None = None,
) -> WindowInfo | None:
    """Find the window for an application, waiting up to ``timeout`` seconds.

    With no ``application`` the focused window is returned, which is what "look
    at what is in front of me" means.
    """

    user32_lib = user32_lib or user32()
    query = str(application or "").strip().casefold()
    deadline = time.time() + max(0.0, timeout)
    while True:
        windows = list_windows(
            user32_lib=user32_lib,
            include_hidden=include_hidden,
            image_name=image_name,
        )
        if not query:
            return next((window for window in windows if window.focused), None)
        best: WindowInfo | None = None
        best_score = 0
        for window in windows:
            score = _match_score(window, query)
            if score > best_score:
                best, best_score = window, score
        if best is not None:
            return best
        if time.time() >= deadline:
            return None
        time.sleep(poll_interval)


# -- control inventory ----------------------------------------------------------


def control_info(user32_lib: Any, handle: Any, *, name_limit: int = 200) -> ControlInfo:
    """Describe one control inside a window."""

    class_name = class_name_of(user32_lib, handle)
    text = read_control_text(user32_lib, handle, limit=name_limit)
    return ControlInfo(
        hwnd=_hwnd_value(handle),
        class_name=class_name,
        name=text.strip(),
        text_length=len(text),
        editable=class_name.casefold() in TEXT_CONTROL_CLASSES,
        enabled=bool(user32_lib.IsWindowEnabled(handle)),
        visible=bool(user32_lib.IsWindowVisible(handle)),
        rect=_rect_of(user32_lib, handle),
    )


def window_controls(
    window: Any,
    *,
    user32_lib: Any | None = None,
    max_controls: int = 40,
    name_limit: int = 200,
) -> list[ControlInfo]:
    """Enumerate the descendant controls of a window (Windows enumerates all depths)."""

    user32_lib = user32_lib or user32()
    found: list[ControlInfo] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def collect(handle: wintypes.HWND, _param: wintypes.LPARAM) -> bool:
        if len(found) >= max(1, max_controls):
            return False
        try:
            found.append(control_info(user32_lib, handle, name_limit=name_limit))
        except Exception:  # noqa: BLE001 - skip controls that cannot be read
            return True
        return True

    try:
        user32_lib.EnumChildWindows(window, collect, 0)
    except Exception:  # noqa: BLE001 - a window without controls is valid
        return found
    return found


def find_text_control(user32_lib: Any, window: Any) -> Any:
    """Find an editable descendant control, falling back to the window itself.

    Modern applications nest the edit control several levels deep, so the whole
    subtree is searched. Returning the window is deliberate: messages then still
    reach the application even when no classic text control exists.
    """

    found = [0]

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def find_child(handle: wintypes.HWND, _param: wintypes.LPARAM) -> bool:
        if class_name_of(user32_lib, handle).casefold() in TEXT_CONTROL_CLASSES:
            found[0] = _hwnd_value(handle)
            return False
        return True

    try:
        user32_lib.EnumChildWindows(window, find_child, 0)
    except Exception:  # noqa: BLE001 - fall back to the parent window
        return window
    return found[0] or window


class _GuiThreadInfo(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hwndActive", wintypes.HWND),
        ("hwndFocus", wintypes.HWND),
        ("hwndCapture", wintypes.HWND),
        ("hwndMenuOwner", wintypes.HWND),
        ("hwndMoveSize", wintypes.HWND),
        ("hwndCaret", wintypes.HWND),
        ("rcCaret", wintypes.RECT),
    ]


def focused_control_handle(user32_lib: Any, window: Any) -> int:
    """Return the control that currently has keyboard focus inside a window."""

    get_thread_info = getattr(user32_lib, "GetGUIThreadInfo", None)
    if get_thread_info is None:
        return 0
    try:
        thread_id = int(user32_lib.GetWindowThreadProcessId(window, None))
        if not thread_id:
            return 0
        info = _GuiThreadInfo()
        info.cbSize = ctypes.sizeof(_GuiThreadInfo)
        if not get_thread_info(thread_id, ctypes.byref(info)):
            return 0
        return _hwnd_value(info.hwndFocus)
    except Exception:  # noqa: BLE001 - focus is optional metadata
        return 0


def find_target_window(
    user32_lib: Any,
    pid: int,
    executable_name: str,
    *,
    timeout: float = 5.0,
    image_name: Callable[[int], str] | None = None,
) -> Any:
    """Find the window of a just-launched process.

    Packaged apps (Windows 11 Notepad) hand off to another process, so the
    launched pid is only one of three signals: pid first, then process image
    name, then window class name. The window can take a moment to appear, so
    discovery is retried until the timeout. Returns 0 when nothing matched.
    """

    stem = str(executable_name).rsplit(".", 1)[0].casefold()
    image = str(executable_name).casefold()
    deadline = time.time() + max(0.0, timeout)
    while True:
        windows = list_windows(
            user32_lib=user32_lib, titled_only=True, image_name=image_name
        )
        by_pid = next((window for window in windows if window.pid == int(pid)), None)
        if by_pid is not None:
            return by_pid.hwnd
        by_image = next(
            (window for window in windows if window.executable.casefold() == image), None
        )
        if by_image is not None:
            return by_image.hwnd
        by_class = next(
            (window for window in windows if window.class_name.casefold() == stem), None
        )
        if by_class is not None:
            return by_class.hwnd
        if time.time() >= deadline:
            return 0
        time.sleep(0.25)


# -- observations ---------------------------------------------------------------


def _window_label(window: WindowInfo) -> str:
    name = window.title.strip() or window.class_name.strip() or "Untitled window"
    process = window.executable.strip() or "unknown process"
    position = "focused" if window.focused else "not focused"
    minimized = ", minimized" if window.minimized else ""
    return f"{name} ({process}, pid {window.pid}, {position}{minimized})"


def describe_window(
    window: WindowInfo,
    *,
    user32_lib: Any | None = None,
    max_controls: int = 40,
    text_limit: int = 4000,
    include_controls: bool = True,
) -> UiObservation:
    """Observe one window: its controls, its readable text, and where focus is."""

    user32_lib = user32_lib or user32()
    handle = window.hwnd
    controls_inspected = bool(include_controls)
    controls = (
        window_controls(handle, user32_lib=user32_lib, max_controls=max_controls)
        if controls_inspected
        else []
    )
    editable = next((control for control in controls if control.editable), None)
    text = ""
    text_truncated = False
    if editable is not None:
        full_text = read_control_text(user32_lib, editable.hwnd, limit=max(1, text_limit) + 1)
        text_truncated = len(full_text) > text_limit
        text = full_text[:text_limit]

    focused_handle = focused_control_handle(user32_lib, handle)
    if not focused_handle:
        focused_handle = _hwnd_value(find_text_control(user32_lib, handle))
    focused = next(
        (control for control in controls if control.hwnd == focused_handle), None
    )
    if focused is not None:
        focused_control: dict[str, Any] | None = focused.to_dict()
    elif focused_handle and focused_handle != handle:
        focused_control = {
            "hwnd": focused_handle,
            "class_name": class_name_of(user32_lib, focused_handle),
        }
    else:
        focused_control = None

    if not controls_inspected:
        fidelity = "window_only"
        summary = f"{_window_label(window)}; controls were not inspected for this request."
    elif editable is not None:
        fidelity = "controls"
        summary = (
            f"{_window_label(window)}; {len(controls)} control(s); text entry control "
            f"{editable.class_name} holds {len(text)} character(s)."
        )
        if text_truncated:
            summary += f" Text was truncated to {text_limit} characters."
    else:
        fidelity = "window_only"
        summary = (
            f"{_window_label(window)}; {len(controls)} control(s) but no editable "
            "text control could be read."
        )

    return UiObservation(
        available=True,
        status="observed",
        fidelity=fidelity,
        summary=summary,
        window=window.to_dict(),
        controls=[control.to_dict() for control in controls],
        focused_control=focused_control,
        text=text,
        text_truncated=text_truncated,
        controls_inspected=controls_inspected,
    )


def observe(
    application: str | None = None,
    *,
    user32_lib: Any | None = None,
    wait_seconds: float = 0.0,
    max_controls: int = 40,
    text_limit: int = 4000,
    include_controls: bool = True,
) -> UiObservation:
    """Observe the current or named application.

    Never raises for an unobservable target: the result says what was available,
    which is what lets the executor verify, recover, or report honestly.
    """

    if user32_lib is None:
        if not perception_available():
            return unavailable(
                "UI observation requires a Windows desktop session.",
                error="perception_unavailable",
            )
        user32_lib = user32()

    query = str(application or "").strip()
    window = (
        find_window(query, user32_lib=user32_lib, timeout=max(0.0, wait_seconds))
        if query
        else find_window(None, user32_lib=user32_lib)
    )
    if window is None:
        if query:
            return target_missing(query)
        return unavailable("No focused window could be observed.", error="no_focused_window")
    return describe_window(
        window,
        user32_lib=user32_lib,
        max_controls=max_controls,
        text_limit=text_limit,
        include_controls=include_controls,
    )


# -- tools ----------------------------------------------------------------------


def _bounded_int(value: Any, *, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


class ComputerObserver:
    """The executor's observation seam for post-action UI state.

    Wraps :func:`observe` so the executor can ask "what does the application look
    like now?" without knowing how the state was obtained. It is read-only, so it
    never needs permission, and it never raises for an unobservable target.
    """

    def __init__(
        self,
        *,
        user32_lib: Any | None = None,
        default_wait_seconds: float = 3.0,
        max_controls: int = 40,
        text_limit: int = 4000,
    ) -> None:
        self._user32 = user32_lib
        self._default_wait_seconds = max(0.0, default_wait_seconds)
        self._max_controls = max_controls
        self._text_limit = text_limit

    def observe_application(
        self,
        application: str | None = None,
        *,
        wait_seconds: float | None = None,
        include_controls: bool = True,
    ) -> dict[str, Any]:
        """Observe an application (or the active window) and return plain data."""

        timeout = self._default_wait_seconds if wait_seconds is None else max(0.0, wait_seconds)
        return observe(
            application,
            user32_lib=self._user32,
            wait_seconds=timeout,
            max_controls=self._max_controls,
            text_limit=self._text_limit,
            include_controls=include_controls,
        ).to_dict()


class OpenWindowsTool(Tool):
    """Read-only window inventory: which applications are on the desktop."""

    metadata = ToolMetadata(
        name="computer.windows",
        description=(
            "List open desktop windows (title, class, process, focus state) "
            "without changing anything."
        ),
        category="computer.perception",
        input_schema={
            "application": {
                "type": "string",
                "description": "optional application/window name filter, e.g. Notepad",
            },
            "include_hidden": {"type": "boolean", "description": "include untitled windows"},
            "max_results": {"type": "integer"},
        },
        output_schema={
            "windows": {"type": "array"},
            "count": {"type": "integer"},
            "source": {"type": "string"},
        },
        permission_level=PermissionLevel.READ_ONLY,
        risk_level=RiskLevel.READ_ONLY,
        verifiable=True,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            if not perception_available():
                return ToolResult.failure("Window inspection requires Windows")
            max_results = _bounded_int(
                parameters.get("max_results"), default=50, low=1, high=200
            )
            windows = list_windows(
                include_hidden=bool(parameters.get("include_hidden")),
                max_windows=max_results,
            )
            application = str(parameters.get("application") or "").strip()
            if application:
                windows = [
                    window for window in windows if matches_application(window, application)
                ]
            return ToolResult(
                success=True,
                status="observed",
                output={
                    "windows": [window.to_dict() for window in windows],
                    "count": len(windows),
                    "source": PERCEPTION_SOURCE,
                },
            )
        except (OSError, ValueError) as exc:
            return ToolResult.failure(str(exc), recoverable=True)


class ComputerObserveTool(Tool):
    """Read-only UI observation: window, controls, focus, and readable text."""

    metadata = ToolMetadata(
        name="computer.observe",
        description=(
            "Observe the current or named application's UI state: window, "
            "controls, focused control, and readable text."
        ),
        category="computer.perception",
        input_schema={
            "application": {
                "type": "string",
                "description": "optional application/window name; defaults to the active window",
            },
            "wait_seconds": {
                "type": "integer",
                "description": "bounded seconds to wait for the window (0-15)",
            },
            "max_controls": {"type": "integer"},
            "text_limit": {"type": "integer"},
            "include_controls": {"type": "boolean"},
        },
        output_schema={
            "status": {"type": "string"},
            "fidelity": {"type": "string"},
            "window": {"type": "object"},
            "controls": {"type": "array"},
            "text": {"type": "string"},
            "summary": {"type": "string"},
        },
        permission_level=PermissionLevel.READ_ONLY,
        risk_level=RiskLevel.READ_ONLY,
        verifiable=True,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            observation = observe(
                str(parameters.get("application") or "").strip() or None,
                wait_seconds=_bounded_int(
                    parameters.get("wait_seconds"), default=0, low=0, high=15
                ),
                max_controls=_bounded_int(
                    parameters.get("max_controls"), default=40, low=1, high=200
                ),
                text_limit=_bounded_int(
                    parameters.get("text_limit"), default=4000, low=0, high=_MAX_TEXT_LIMIT
                ),
                include_controls=bool(parameters.get("include_controls", True)),
            )
            return ToolResult(
                success=True,
                status=observation.status,
                output=observation.to_dict(),
            )
        except (OSError, ValueError) as exc:
            return ToolResult.failure(str(exc), recoverable=True)
