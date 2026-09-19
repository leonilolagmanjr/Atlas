"""Post-action verification contracts (reasoning.verifier.TaskVerifier)."""

from __future__ import annotations

import unittest

from reasoning.verifier import TaskVerifier


class TaskVerifierTests(unittest.TestCase):
    def setUp(self):
        self.verifier = TaskVerifier()

    def test_failed_execution_is_reported_failed(self):
        outcome = self.verifier.verify("web.search", {}, success=False)
        self.assertEqual(outcome.status, "failed")
        self.assertFalse(outcome.verified)

    def test_unknown_capability_is_unverified_not_assumed_ok(self):
        outcome = self.verifier.verify("unknown.capability", {"ok": True}, success=True)
        self.assertEqual(outcome.status, "unverified")
        self.assertFalse(outcome.verified)

    def test_write_text_observed_false_is_a_verification_failure(self):
        # The action executed, but the text is provably not in the control.
        outcome = self.verifier.verify(
            "applications.write_text",
            {
                "application": "Notepad",
                "characters": 120,
                "observed": False,
                "target": {"control_class": "Edit", "window_title": "Untitled - Notepad"},
            },
            success=True,
        )
        self.assertEqual(outcome.status, "failed")
        self.assertFalse(outcome.verified)
        self.assertIn("not present", outcome.detail)

    def test_write_text_observed_true_is_verified(self):
        outcome = self.verifier.verify(
            "applications.write_text",
            {"application": "Notepad", "characters": 120, "observed": True, "observed_characters": 120},
            success=True,
        )
        self.assertEqual(outcome.status, "verified")
        self.assertTrue(outcome.verified)

    def test_write_text_without_read_back_is_honest_about_it(self):
        outcome = self.verifier.verify(
            "applications.write_text",
            {"application": "Notepad", "characters": 120},
            success=True,
        )
        # Sent is not proven: the detail must say the control could not be read.
        self.assertIn("could not be read back", outcome.detail)

    def test_observe_window_status_is_verified(self):
        outcome = self.verifier.verify(
            "computer.observe",
            {"status": "observed", "summary": "Notepad window is visible."},
            success=True,
        )
        self.assertEqual(outcome.status, "verified")
        self.assertTrue(outcome.verified)

    def test_observe_target_missing_is_not_verified(self):
        outcome = self.verifier.verify(
            "computer.observe",
            {"status": "target_missing", "summary": "No matching window was found."},
            success=True,
        )
        self.assertEqual(outcome.status, "unverified")
        self.assertFalse(outcome.verified)

    def test_observe_unsupported_is_not_verified(self):
        outcome = self.verifier.verify(
            "computer.observe", {"status": "unsupported"}, success=True
        )
        self.assertEqual(outcome.status, "unverified")
        self.assertFalse(outcome.verified)

    def test_window_list_with_windows_is_verified(self):
        outcome = self.verifier.verify(
            "computer.windows",
            {"windows": [{"title": "Untitled - Notepad"}], "count": 1},
            success=True,
        )
        self.assertEqual(outcome.status, "verified")
        self.assertTrue(outcome.verified)

    def test_empty_window_list_is_unverified(self):
        outcome = self.verifier.verify(
            "computer.windows", {"windows": [], "count": 0}, success=True
        )
        self.assertEqual(outcome.status, "unverified")
        self.assertFalse(outcome.verified)

    def test_launch_without_pid_is_unverified(self):
        outcome = self.verifier.verify("applications.launch_named", {}, success=True)
        self.assertEqual(outcome.status, "unverified")

    def test_launch_with_pid_is_verified(self):
        outcome = self.verifier.verify(
            "applications.launch_named", {"pid": 4242}, success=True
        )
        self.assertEqual(outcome.status, "verified")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
