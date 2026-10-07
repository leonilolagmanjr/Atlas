"""Low-level Win32 mouse/keyboard input for Atlas.

This module is the ACTION half of the visual interaction loop. It performs the
actual OS input (``SendInput``) and is the only place in the vision subsystem
that changes anything on the machine. Everything above it (tools, coordinator)
is responsible for permission, coordinate validation, and verification; this
layer just does what it is told, precisely.

Keeping the OS input here means the validation logic and the tools can be unit
tested with a fake input backend and no GUI.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_MOUSEEVENTF_RIGHTDOWN = 0x0008
_MOUSEEVENTF_RIGHTUP = 0x0010
_MOUSEEVENTF_MIDDLEDOWN = 0x0020
_MOUSEEVENTF_MIDDLEUP = 0x0040
_MOUSEEVENTF_WHEEL = 0x0800
_MOUSEEVENTF_MOVE = 0x0001
_MOUSEEVENTF_ABSOLUTE = 0x8000

_INPUT_MOUSE = 0
_INPUT_KEYBOARD = 1
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_UNICODE = 0x0004

#: Virtual key codes for the named keys Atlas accepts.
KEY_CODES: dict[str, int] = {
    "enter": 0x0D,
    "return": 0x0D,
    "tab": 0x09,
    "escape": 0x1B,
    "esc": 0x1B,
    "space": 0x20,
    "backspace": 0x08,
    "delete": 0x2E,
    "up": 0x26,
    "down": 0x28,
    "left": 0x25,
    "right": 0x27,
    "home": 0x24,
    "end": 0x23,
    "pageup": 0x21,
    "pagedown": 0x22,
    "f1": 0x70,
    "f2": 0x71,
    "f3": 0x72,
    "f4": 0x73,
    "f5": 0x74,
    "f6": 0x75,
}


def input_available() -> bool:
    """Return True when this platform can synthesize input."""

    return sys.platform == "win32"


@dataclass
class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


@dataclass
class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("union", _INPUT_UNION)]


def _send_mouse(flags: int, *, dx: int = 0, dy: int = 0, data: int = 0) -> None:
    user32 = ctypes.windll.user32
    item = _INPUT(
        type=_INPUT_MOUSE,
        union=_INPUT_UNION(mi=_MOUSEINPUT(dx, dy, data, flags, 0, None)),
    )
    user32.SendInput(1, ctypes.byref(item), ctypes.sizeof(_INPUT))


def _send_key(vk: int, *, keyup: bool = False, unicode_scan: int | None = None) -> None:
    user32 = ctypes.windll.user32
    flags = _KEYEVENTF_KEYUP if keyup else 0
    scan = 0
    if unicode_scan is not None:
        flags |= _KEYEVENTF_UNICODE
        scan = unicode_scan
    item = _INPUT(
        type=_INPUT_KEYBOARD,
        union=_INPUT_UNION(ki=_KEYBDINPUT(vk if unicode_scan is None else 0, scan, flags, 0, None)),
    )
    user32.SendInput(1, ctypes.byref(item), ctypes.sizeof(_INPUT))


class InputBackend:
    """The injectable OS input boundary. The default drives the real desktop."""

    def move(self, x: int, y: int) -> None:
        user32 = ctypes.windll.user32
        # Absolute coordinates must be normalized to the 0..65535 space SendInput
        # expects for the virtual desktop, which is not 1:1 with pixels.
        screen_w = max(1, int(user32.GetSystemMetrics(0)))
        screen_h = max(1, int(user32.GetSystemMetrics(1)))
        norm_x = int((x * 65535) / screen_w)
        norm_y = int((y * 65535) / screen_h)
        _send_mouse(_MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE, dx=norm_x, dy=norm_y)

    def click(self, x: int, y: int, *, button: str = "left", count: int = 1) -> None:
        down, up = _button_flags(button)
        for _ in range(max(1, count)):
            self.move(x, y)
            time.sleep(0.01)
            _send_mouse(down)
            time.sleep(0.02)
            _send_mouse(up)
            time.sleep(0.02)

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, *, button: str = "left") -> None:
        down, up = _button_flags(button)
        self.move(start_x, start_y)
        time.sleep(0.05)
        _send_mouse(down)
        time.sleep(0.05)
        self.move(end_x, end_y)
        time.sleep(0.05)
        _send_mouse(up)

    def scroll(self, amount: int) -> None:
        # A wheel notch is 120 units; positive scrolls up.
        _send_mouse(_MOUSEEVENTF_WHEEL, data=int(amount))

    def keypress(self, key: str) -> None:
        code = KEY_CODES.get(str(key).strip().casefold())
        if code is None:
            raise ValueError(f"Unsupported key: {key}")
        _send_key(code)
        _send_key(code, keyup=True)

    def type_text(self, text: str) -> None:
        for character in text:
            _send_key(0, unicode_scan=ord(character))
            _send_key(0, keyup=True, unicode_scan=ord(character))


def _button_flags(button: str) -> tuple[int, int]:
    name = str(button or "left").strip().casefold()
    if name == "right":
        return _MOUSEEVENTF_RIGHTDOWN, _MOUSEEVENTF_RIGHTUP
    if name == "middle":
        return _MOUSEEVENTF_MIDDLEDOWN, _MOUSEEVENTF_MIDDLEUP
    return _MOUSEEVENTF_LEFTDOWN, _MOUSEEVENTF_LEFTUP


def default_input_backend() -> InputBackend:
    return InputBackend()


def get_cursor_position() -> tuple[int, int]:
    """Return the current cursor position (0, 0 when unavailable)."""

    if not input_available():
        return (0, 0)
    try:
        point = ctypes.wintypes.POINT()  # type: ignore[attr-defined]
        if ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
            return (int(point.x), int(point.y))
    except Exception:  # noqa: BLE001
        pass
    return (0, 0)


def focus_window(hwnd: int) -> bool:
    """Bring a window to the foreground, working around foreground locks."""

    if not input_available() or not hwnd:
        return False
    try:
        user32 = ctypes.windll.user32
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.SetForegroundWindow(hwnd)
        return int(user32.GetForegroundWindow()) == int(hwnd)
    except Exception:  # noqa: BLE001
        return False
