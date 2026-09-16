"""Permission-gated text entry into a resolved Windows application window."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import base64
import subprocess
import time
from typing import Any

from computer.launch import resolve_application_name
from tools.base import PermissionLevel, RiskLevel, Tool, ToolMetadata, ToolResult


class ApplicationTextEntryTool(Tool):
    metadata = ToolMetadata(
        name="applications.write_text",
        description="Open a trusted Windows application and enter supplied text into it.",
        category="computer.applications",
        input_schema={
            "application": {"type": "string", "description": "target application name, e.g. Notepad"},
            "text": {"type": "string", "description": "exact text to enter"},
        },
        output_schema={"pid": {"type": "integer"}, "application": {"type": "string"}, "characters": {"type": "integer"}},
        permission_level=PermissionLevel.MEDIUM_RISK,
        risk_level=RiskLevel.MEDIUM,
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

            executable = resolve_application_name(application)
            if executable is None:
                return ToolResult.failure(f"Could not resolve a trusted executable for: {application}", recoverable=True)

            process = subprocess.Popen(
                [str(executable)],
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(self._startup_wait_seconds)
            self._type_into_process(process.pid, text, executable.name)
            return ToolResult(
                success=True,
                status="written",
                output={
                    "pid": process.pid,
                    "application": application,
                    "executable": str(executable),
                    "characters": len(text),
                },
            )
        except (KeyError, OSError, ValueError, subprocess.SubprocessError) as exc:
            return ToolResult.failure(str(exc), recoverable=True)

    @staticmethod
    def _type_into_process(pid: int, text: str, executable_name: str) -> None:
        if not hasattr(ctypes, "windll"):
            raise OSError("Direct Windows text entry requires Windows")

        user32 = ctypes.windll.user32
        window = _find_target_window(user32, pid, executable_name)
        if not window:
            raise OSError("Application window could not be found")

        # Windows foreground-locking makes SetForegroundWindow unreliable, and
        # SendKeys targets whatever happens to be focused. So text is delivered
        # directly to the target control's message queue, which needs no focus.
        control = _find_text_control(user32, window)
        if _set_text_directly(user32, control, text):
            return
        if _paste_into_control(user32, control, text):
            return
        # Last resort: focus the window and paste via the clipboard.
        user32.ShowWindow(window, 9)
        user32.BringWindowToTop(window)
        if not _focus_window(user32, window):
            raise OSError("Application window could not be focused")
        time.sleep(0.2)
        _set_clipboard(text)
        paste_script = "$shell=New-Object -ComObject WScript.Shell; Start-Sleep -Milliseconds 150; $shell.SendKeys('^v')"
        completed = subprocess.run(
            ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", paste_script],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            shell=False,
        )
        if completed.returncode != 0:
            raise OSError(completed.stderr.strip() or "Could not paste text into the focused application")


def _find_target_window(user32: Any, pid: int, executable_name: str, *, timeout: float = 5.0) -> Any:
    # Match a window by the launched pid first, then by process image name, then
    # by window class name. Packaged apps (Windows 11 Notepad) hand off to a
    # different process, so image/class matching is required, and the window can
    # take a moment to appear, so discovery is retried until the timeout.
    stem = executable_name.rsplit(".", 1)[0].casefold()
    deadline = time.time() + timeout
    while True:
        window = wintypes.HWND()
        fallback_window = wintypes.HWND()
        class_match = wintypes.HWND()

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def find_window(hwnd: wintypes.HWND, _: wintypes.LPARAM) -> bool:
            process_id = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
            if not user32.IsWindowVisible(hwnd) or not user32.GetWindowTextLengthW(hwnd):
                return True
            if process_id.value == pid:
                window.value = hwnd
                return False
            if _process_image_name(process_id.value).casefold() == executable_name.casefold():
                fallback_window.value = hwnd
            elif _class_name(user32, hwnd).casefold() == stem:
                class_match.value = hwnd
            return True
        user32.EnumWindows(find_window, 0)
        found = window.value or fallback_window.value or class_match.value
        if found:
            return found
        if time.time() >= deadline:
            return None
        time.sleep(0.25)


#: Control class names that accept text. Covers classic Win32 edit controls and
#: the modern rich-edit controls used by Windows 11 apps (Notepad, WordPad).
_TEXT_CONTROL_CLASSES = {
    "edit",
    "richedit",
    "richedit20a",
    "richedit20w",
    "richedit50w",
    "richeditd2dpt",
    "textbox",
    "scintilla",
}

def _find_text_control(user32: Any, window: Any) -> Any:
    # Find an editable descendant of the target window. Modern apps nest the edit
    # control several levels deep, so the whole subtree is searched. If none is
    # found, the top-level window is returned so messages still reach it.
    control = wintypes.HWND()

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def find_child(hwnd: wintypes.HWND, _: wintypes.LPARAM) -> bool:
        if _class_name(user32, hwnd).casefold() in _TEXT_CONTROL_CLASSES:
            control.value = hwnd
            return False
        return True
    try:
        user32.EnumChildWindows(window, find_child, 0)
    except Exception:  # noqa: BLE001 - fall back to the parent window
        return window
    return control.value or window


def _class_name(user32: Any, hwnd: Any) -> str:
    buffer = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buffer, len(buffer))
    return buffer.value


def _set_text_directly(user32: Any, control: Any, text: str) -> bool:
    # WM_SETTEXT replaces a control's contents without any focus or clipboard.
    # This is the most reliable path and works even when the app is minimized or
    # unfocused. To avoid clobbering an already-open document, any existing text
    # is preserved by prepending the control's current contents.
    WM_SETTEXT = 0x000C
    try:
        existing = _control_text(user32, control)
        combined = f"{existing}\n{text}" if existing.strip() else text
        result = user32.SendMessageW(control, WM_SETTEXT, 0, ctypes.c_wchar_p(combined))
    except Exception:  # noqa: BLE001
        return False
    # SendMessageW returns nonzero on success for WM_SETTEXT.
    return bool(result)


def _control_text(user32: Any, control: Any, *, limit: int = 200_000) -> str:
    WM_GETTEXTLENGTH = 0x000E
    WM_GETTEXT = 0x000D
    try:
        length = int(user32.SendMessageW(control, WM_GETTEXTLENGTH, 0, 0))
        if length <= 0:
            return ""
        buffer = ctypes.create_unicode_buffer(min(length, limit) + 1)
        user32.SendMessageW(control, WM_GETTEXT, len(buffer), buffer)
        return buffer.value
    except Exception:  # noqa: BLE001
        return ""


def _paste_into_control(user32: Any, control: Any, text: str) -> bool:
    # Put the text on the clipboard, then ask the control to paste itself via
    # WM_PASTE. Unlike SendKeys, this is addressed to the control directly, so it
    # does not depend on which window is currently focused.
    WM_PASTE = 0x0302
    try:
        _set_clipboard(text)
        user32.SendMessageW(control, WM_PASTE, 0, 0)
    except Exception:  # noqa: BLE001
        return False
    return _control_contains(user32, control, text)


def _control_contains(user32: Any, control: Any, text: str) -> bool:
    WM_GETTEXTLENGTH = 0x000E
    try:
        length = int(user32.SendMessageW(control, WM_GETTEXTLENGTH, 0, 0))
    except Exception:  # noqa: BLE001
        return False
    return length >= len(text.strip())


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


def _process_image_name(pid: int) -> str:
    kernel32 = ctypes.windll.kernel32
    process = kernel32.OpenProcess(0x1000, False, pid)
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
