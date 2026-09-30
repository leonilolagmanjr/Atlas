"""Milestone 7.2: natural language, typos, and application names.

People type "open chrmoe" and "open vsocde". Atlas must resolve the intended
application through the *existing* alias/registry resolution rather than a
growing list of misspellings, and it must refuse to guess when two applications
are equally plausible. These tests use the real resolver and a real system
executable, so they assert the behavior the runtime will actually get.
"""

from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from computer.launch import (
    _APPLICATION_ALIASES,
    _confident_application_correction,
    known_application_names,
    resolve_application_name,
)
from memory.conversation_runtime import classify_turn

WINDOWS = sys.platform == "win32"


class ClassificationStillDelegatesTests(unittest.TestCase):
    """A typo must never turn an action into a chat answer."""

    def test_mistyped_actions_are_still_actions(self) -> None:
        for text in ("open chrmoe", "open vsocde", "search youtub", "open notepda"):
            with self.subTest(text=text):
                result = classify_turn(text)
                self.assertTrue(result.delegate, f"{text!r} must reach the agent")


class ConfidentCorrectionTests(unittest.TestCase):
    def test_near_misses_are_corrected_to_a_known_application(self) -> None:
        cases = {
            "chrmoe": "chrome",
            "vsocde": "vscode",
            "notepda": "notepad",
        }
        for mistyped, expected in cases.items():
            with self.subTest(mistyped=mistyped):
                self.assertEqual(_confident_application_correction(mistyped), expected)

    def test_a_correctly_spelled_name_is_never_corrected(self) -> None:
        for name in known_application_names():
            with self.subTest(name=name):
                self.assertEqual(_confident_application_correction(name), "")

    def test_an_unrelated_word_is_not_corrected(self) -> None:
        # "sykrim" is a topic, not an application. Guessing an application from
        # it would open the wrong program.
        for word in ("sykrim", "zzzzqqqq", "qqq"):
            with self.subTest(word=word):
                self.assertEqual(_confident_application_correction(word), "")

    def test_an_ambiguous_typo_is_refused_rather_than_guessed(self) -> None:
        aliases = {
            "calc": ("calc.exe",),
            "calx": ("calx.exe",),
        }
        with patch.dict(_APPLICATION_ALIASES, aliases, clear=True):
            # "calcx" is exactly one edit from both "calc" and "calx": with no
            # clear winner nothing is corrected, so Atlas reports the unknown
            # application instead of opening the wrong one.
            self.assertEqual(_confident_application_correction("calcx"), "")


@unittest.skipUnless(WINDOWS, "application resolution is a Windows capability")
class ResolutionFallbackTests(unittest.TestCase):
    def test_a_mistyped_name_resolves_through_the_correction(self) -> None:
        # Map the friendly name onto a binary that really exists on this
        # machine, so the fallback path is exercised end to end without needing
        # a specific third-party application to be installed. "calculator" is
        # used because no `calculator.exe` exists, so the alias (not a
        # same-named binary) is what resolution has to reach.
        with patch.dict(_APPLICATION_ALIASES, {"calculator": ("calc.exe",)}, clear=True):
            resolved = resolve_application_name("calculater")
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.name.casefold(), "calc.exe")

    def test_a_typo_of_something_not_installed_still_fails(self) -> None:
        with patch.dict(_APPLICATION_ALIASES, {"zzzznotinstalled": ("zzzznotinstalled.exe",)}, clear=True):
            self.assertIsNone(resolve_application_name("zzzznotinstalled"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
