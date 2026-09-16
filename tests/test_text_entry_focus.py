"""Regression tests for focus-independent Windows text entry.

These cover the concrete bug where Notepad text entry failed whenever Notepad
was not the foreground window:

* text is delivered via window messages (WM_SETTEXT/WM_PASTE), not SendKeys,
  so it does not depend on which window is focused;
* the modern Windows 11 Notepad edit control (``RichEditD2DPT``) is recognized;
* the target window is found by class name as well as by process image name,
  because packaged apps launch a stub process that hands off to another.
"""

from __future__ import annotations

import ctypes
import unittest

from computer.text_entry import (
    _TEXT_CONTROL_CLASSES,
    _find_text_control,
    _set_text_directly,
)


class _FakeUser32:
    """Minimal fake of the user32 calls used by the text-entry helpers."""

    def __init__(self, children: list[tuple[int, str]], *, existing_text: str = "") -> None:
        self._children = children
        self._existing = existing_text
        self.set_text_calls: list[tuple[int, str]] = []

    def GetClassNameW(self, hwnd, buffer, _length):
        for handle, name in self._children:
            if handle == hwnd:
                buffer.value = name
                return len(name)
        buffer.value = ""
        return 0

    def EnumChildWindows(self, _window, callback, _param):
        for handle, _name in self._children:
            if not callback(handle, 0):
                return

    def SendMessageW(self, hwnd, message, _wparam, lparam):
        if message == 0x000E:  # WM_GETTEXTLENGTH
            return len(self._existing)
        if message == 0x000D:  # WM_GETTEXT
            lparam.value = self._existing
            return len(self._existing)
        if message == 0x000C:  # WM_SETTEXT
            self.set_text_calls.append((hwnd, str(lparam.value)))
            return 1
        return 0


class TextControlDetectionTests(unittest.TestCase):
    def test_modern_notepad_richedit_is_detected(self):
        # Windows 11 Notepad nests RichEditD2DPT under NotepadTextBox.
        user32 = _FakeUser32([(10, "NotepadTextBox"), (20, "RichEditD2DPT")])

        control = _find_text_control(user32, 1)

        self.assertEqual(control, 20)

    def test_classic_edit_control_is_detected(self):
        user32 = _FakeUser32([(30, "Edit")])

        self.assertEqual(_find_text_control(user32, 1), 30)

    def test_falls_back_to_window_when_no_control_found(self):
        user32 = _FakeUser32([(40, "SomeChrome")])

        self.assertEqual(_find_text_control(user32, 1), 1)

    def test_known_classes_include_rich_edit_d2d(self):
        self.assertIn("richeditd2dpt", _TEXT_CONTROL_CLASSES)


class DirectTextDeliveryTests(unittest.TestCase):
    def test_text_is_delivered_without_focus(self):
        # No focus/foreground calls exist on the fake; success proves delivery
        # does not depend on the window being focused.
        user32 = _FakeUser32([], existing_text="")

        delivered = _set_text_directly(user32, 99, "hello atlas")

        self.assertTrue(delivered)
        self.assertEqual(user32.set_text_calls, [(99, "hello atlas")])

    def test_existing_content_is_preserved(self):
        user32 = _FakeUser32([], existing_text="previous content")

        _set_text_directly(user32, 99, "new line")

        self.assertEqual(user32.set_text_calls, [(99, "previous content\nnew line")])

    def test_returns_false_when_settext_fails(self):
        class FailingUser32(_FakeUser32):
            def SendMessageW(self, hwnd, message, wparam, lparam):
                if message == 0x000C:
                    return 0
                return super().SendMessageW(hwnd, message, wparam, lparam)

        self.assertFalse(_set_text_directly(FailingUser32([]), 99, "text"))


if __name__ == "__main__":
    unittest.main()
