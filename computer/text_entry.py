"""Permission-gated text entry into a resolved Windows application window.

Entry is one half of the computer-interaction loop; the other half is observing
whether the text actually landed. This module therefore does not report success
just because a message was sent: after delivery it reads the target control back
through :mod:`computer.perception` and reports one of

* ``confirmed``   - the text is readable in the control (verified);
* ``failed``      - the control is readable and the text is *not* there, which
                    the executor records as a verification failure (distinct
                    from an execution failure) so bounded recovery can retry;
* ``unconfirmed`` - the control cannot be read back, so Atlas says so instead of
                    claiming verification.
"""

from __future__ import annotations

import ctypes
import base64
import subprocess
import time
from typing import Any

from computer.launch import resolve_application_name
from computer.perception import TEXT_CONTROL_CLASSES as _TEXT_CONTROL_CLASSES
from computer.perception import class_name_of as _class_name
from computer.perception import find_target_window as _find_target_window
from computer.perception import find_text_control as _find_text_control
from computer.perception import read_control_text as _control_text
from computer.perception import send_message as _send_message
from computer.perception import window_text as _window_text
from tools.base import PermissionLevel, RiskLevel, Tool, ToolMetadata, ToolResult

#: Delivery strategies, in the order the automatic path tries them.
#: ``direct`` writes through a window message (focus-free and most reliable);
#: ``paste`` asks the control to paste from the clipboard; ``auto`` tries direct,
#: then paste, then a focus-based clipboard paste for legacy controls.
DELIVERY_MODES: tuple[str, ...] = ("auto", "direct", "paste")
_WM_SETTEXT = 0x000C
_WM_PASTE = 0x0302


def _bounded_seconds(value: Any, *, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _text_present(current: str, submitted: str) -> bool:
    """Return True when ``submitted`` is readable inside a control's text."""

    needle = submitted.strip()
    haystack = current.strip()
    if not needle or not haystack:
        return False
    if needle in haystack:
        return True
    head, tail = needle[:200], needle[-200:]
    return head in haystack and tail in haystack


def _set_text_directly(user32: Any, control: Any, text: str) -> bool:
    """Write text into a control through WM_SETTEXT, with no focus or clipboard.

    This is the most reliable path and works even when the application is
    minimized or unfocused. Existing document text is preserved by prepending it
    to the new text instead of clobbering an open document.
    """

    try:
        existing = _control_text(user32, control)
        combined = f"{existing}\n{text}" if existing.strip() else text
        result = user32.SendMessageW(control, _WM_SETTEXT, 0, ctypes.c_wchar_p(combined))
    except Exception:  # noqa: BLE001
        return False
    # SendMessageW returns nonzero on success for WM_SETTEXT.
    return bool(result)


class ApplicationTextEntryTool(Tool):
    metadata = ToolMetadata(
        name="applications.write_text",
        description=(
            "Open a trusted Windows application, enter supplied text, and read "
            "the result back to verify it was written."
        ),
        category="computer.applications",
        input_schema={
            "application": {"type": "string", "description": "target application name, e.g. Notepad"},
            "text": {"type": "string", "description": "exact text to enter"},
            "delivery": {
                "type": "string",
                "description": "auto (default), direct (window message) or paste",
            },
            "wait_seconds": {
                "type": "integer",
                "description": "bounded seconds to wait for the application window (1-15)",
            },
        },
        output_schema={
            "pid": {"type": "integer"},
            "application": {"type": "string"},
            "characters": {"type": "integer"},
            "delivery": {"type": "string"},
            "observed": {"type": "boolean"},
            "observed_characters": {"type": "integer"},
            "verification": {"type": "string"},
            "target": {"type": "object"},
        },
        permission_level=PermissionLevel.MEDIUM_RISK,
        risk_level=RiskLevel.MEDIUM,
        verifiable=True,
    )

    def __init__(self, *, startup_wait_seconds: float = 1.5) -> None:
        self._startup_wait_seconds = startup_wait_seconds


    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            application = str(parameters["application"]).strip()
            text = str(parameters["text"])
            if not application:
                return ToolResult.failure("Application name cannot be empty")
            if not text.strip():
                return ToolResult.failure("Text cannot be empty")
            if len(text) > 100_000:
                return ToolResult.failure("Text exceeds the 100,000 character limit")
            delivery = str(parameters.get("delivery") or "auto").strip().casefold()
            if delivery not in DELIVERY_MODES:
                return ToolResult.failure(
                    f"delivery must be one of: {', '.join(DELIVERY_MODES)}"
                )
            window_timeout = _bounded_seconds(
                parameters.get("wait_seconds"), default=5, low=1, high=15
            )

            executable = resolve_application_name(application)
            if executable is None:
                return ToolResult.failure(
                    f"Could not resolve a trusted executable for: {application}",
                    recoverable=True,
                )

            process = subprocess.Popen(
                [str(executable)],
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(self._startup_wait_seconds)
            observation = self._type_into_process(
                process.pid,
                text,
                executable.name,
                delivery=delivery,
                window_timeout=window_timeout,
            )
            if not isinstance(observation, dict):
                observation = {}
            observed = observation.get("observed")
            if observed is True:
                verification = "confirmed"
            elif observed is False:
                verification = "failed"
            else:
                verification = "unconfirmed"
            return ToolResult(
                success=True,
                status="written" if verification != "failed" else "verification_failed",
                output={
                    "pid": process.pid,
                    "application": application,
                    "executable": str(executable),
                    "characters": len(text),
                    "delivery": observation.get("delivery"),
                    "observed": observed,
                    "observed_characters": observation.get("observed_characters"),
                    "verification": verification,
                    "target": observation.get("target"),
                },
            )
        except (KeyError, OSError, ValueError, subprocess.SubprocessError) as exc:
            return ToolResult.failure(str(exc), recoverable=True)

    @classmethod
    def _type_into_process(
        cls,
        pid: int,
        text: str,
        executable_name: str,
        *,
        delivery: str = "auto",
        window_timeout: float = 5.0,
    ) -> dict[str, Any]:
        """Deliver ``text`` and observe the result. Returns an observation dict."""

        if not hasattr(ctypes, "windll"):
            raise OSError("Direct Windows text entry requires Windows")

        user32 = ctypes.windll.user32
        window = _find_target_window(user32, pid, executable_name, timeout=window_timeout)
        if not window:
            raise OSError(
                f"Application window for '{executable_name}' could not be found "
                f"within {window_timeout:.0f}s"
            )

        control = _find_text_control(user32, window)
        target = {
            "window_title": _window_text(user32, window),
            "window_class": _class_name(user32, window),
            "control_class": _class_name(user32, control),
        }

        method = cls._deliver(user32, window, control, text, delivery=delivery)
        if method is None:
            raise OSError(
                f"Text could not be delivered to {executable_name} "
                f"using the '{delivery}' delivery path"
            )

        # Allow the control time to process the text before reading back.
        time.sleep(0.5)

        observed, observed_characters = cls._confirm_text(user32, control, text, method=method)
        return {
            "delivery": method,
            "observed": observed,
            "observed_characters": observed_characters,
            "target": target,
        }

    @classmethod
    def _deliver(
        cls,
        user32: Any,
        window: Any,
        control: Any,
        text: str,
        *,
        delivery: str,
    ) -> str | None:
        """Try the requested delivery paths in order; return the method used."""

        if delivery in {"auto", "direct"} and _set_text_directly(user32, control, text):
            return "wm_settext"
        if delivery in {"auto", "paste"} and _paste_into_control(user32, control, text):
            return "wm_paste"
        if delivery == "auto" and _paste_via_focus(user32, window, text):
            return "clipboard_paste"
        return None

    @staticmethod
    def _paste_via_focus(user32: Any, window: Any, text: str) -> bool:
        # Last resort for legacy controls that ignore window messages: focus the
        # window and paste from the clipboard.
        try:
            user32.ShowWindow(window, 9)
            user32.BringWindowToTop(window)
            if not _focus_window(user32, window):
                return False
            time.sleep(0.2)
            _set_clipboard(text)
            paste_script = (
                "$shell=New-Object -ComObject WScript.Shell; "
                "Start-Sleep -Milliseconds 150; $shell.SendKeys('^v')"
            )
            completed = subprocess.run(
                ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", paste_script],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return completed.returncode == 0

    @staticmethod
    def _confirm_text(
        user32: Any,
        control: Any,
        text: str,
        *,
        method: str,
    ) -> tuple[bool | None, int | None]:
        """Read the control back and decide whether the text actually landed."""

        deadline = time.time() + (4.0 if method == "clipboard_paste" else 3.0)
        current = ""
        while True:
            current = _control_text(user32, control)
            if _text_present(current, text):
                return True, len(current)
            if time.time() >= deadline:
                break
            time.sleep(0.2)
        # An unreadable control (no known text class) leaves the result unknown.
        class_name = _class_name(user32, control).casefold()
        if class_name not in _TEXT_CONTROL_CLASSES:
            return None, None
        # Known text control (including RichEditD2DPT) but the expected text is not
        # present: verification failure.
        return False, len(current)


def _paste_into_control(user32: Any, control: Any, text: str) -> bool:
    # Put the text on the clipboard, then ask the control to paste itself via
    # WM_PASTE. Unlike SendKeys, this is addressed to the control directly, so it
    # does not depend on which window is currently focused. Whether the paste
    # worked is decided by the read-back in _confirm_text, not here.
    try:
        _set_clipboard(text)
        _send_message(user32, control, _WM_PASTE, 0, 0)
    except Exception:  # noqa: BLE001
        return False
    return True


def _focus_window(user32: Any, window: Any, *, attempts: int = 5) -> bool:
    # Windows restricts foreground stealing, so a single SetForegroundWindow can
    # fail even for a freshly launched window. Retry with the thread-input attach
    # trick and settle time between attempts before giving up.
    kernel32 = ctypes.windll.kernel32
    current_thread = kernel32.GetCurrentThreadId()
    for _ in range(attempts):
        foreground = user32.GetForegroundWindow()
        foreground_thread = user32.GetWindowThreadProcessId(foreground, None) if foreground else 0
        target_thread = user32.GetWindowThreadProcessId(window, None)
        attached = False
        if target_thread and foreground_thread and foreground_thread != target_thread:
            attached = bool(user32.AttachThreadInput(current_thread, target_thread, True))
        try:
            focused = bool(user32.SetForegroundWindow(window)) or user32.GetForegroundWindow() == window
            user32.SetFocus(window)
        finally:
            if attached:
                user32.AttachThreadInput(current_thread, target_thread, False)
        if focused:
            return True
        user32.ShowWindow(window, 9)
        user32.BringWindowToTop(window)
        time.sleep(0.25)
    return user32.GetForegroundWindow() == window


def _set_clipboard(text: str) -> None:
    encoded = base64.b64encode(text.encode("utf-16le")).decode("ascii")
    script = (
        "$ErrorActionPreference='Stop'; "
        f"$text=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String('{encoded}')); "
        "Set-Clipboard -Value $text"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        raise OSError(completed.stderr.strip() or "Could not set the Windows clipboard")