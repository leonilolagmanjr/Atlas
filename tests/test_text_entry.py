import unittest
from pathlib import Path
from unittest.mock import patch

from computer.perception import (
    _control_class_rank,
    find_text_control,
    find_target_window,
    list_titled_window_handles,
)
from computer.text_entry import ApplicationTextEntryTool
from intent_classifier import IntentClassifier
from planner import Planner


class _FakeUser32:
    """Minimal fake of the user32 calls used by delivery and read-back."""

    def __init__(self, *, control_class: str = "Edit", text: str = "") -> None:
        self._control_class = control_class
        self._text = text
        self.paste_calls: list[int] = []

    def GetClassNameW(self, _hwnd, buffer, _length):
        buffer.value = self._control_class
        return len(self._control_class)

    def SendMessageTimeoutW(self, _hwnd, message, _wparam, lparam, _flags, _timeout, result):
        value = self.SendMessageW(_hwnd, message, _wparam, lparam)
        result._obj.value = value
        return 1

    def SendMessageW(self, _hwnd, message, _wparam, lparam):
        if message == 0x000E:  # WM_GETTEXTLENGTH
            return len(self._text)
        if message == 0x000D:  # WM_GETTEXT
            lparam.value = self._text
            return len(self._text)
        if message == 0x0302:  # WM_PASTE
            self.paste_calls.append(_hwnd)
            return 1
        return 0


class TextEntryTests(unittest.TestCase):
    def test_poem_request_is_write_application_intent(self):
        result = IntentClassifier().classify("Create a short poem in Notepad")

        self.assertEqual(result.intent, "WRITE_APPLICATION")

    def test_poem_request_plans_notepad_text_entry(self):
        # The structured path generates content first, then writes it to the
        # destination. The legacy path (no structured intent) still plans a
        # single write step against the named application.
        decision = Planner().create_plan(
            "Create a short poem in Notepad",
            intent="WRITE_APPLICATION",
        )
        step = decision.plan.steps[0]

        self.assertEqual(decision.strategy, "deterministic_application_text_entry")
        self.assertEqual(step.metadata["tool"], "applications.write_text")
        self.assertEqual(step.metadata["parameters"]["application"], "Notepad")

    def test_structured_poem_request_generates_then_writes(self):
        from models import StructuredIntent
        structured = StructuredIntent(
            intent="write_content",
            action="create",
            content_type="poem",
            destination="Notepad",
            topic="cars",
            confidence=0.95,
        )
        decision = Planner().create_plan("create a poem about cars in notepad", structured=structured)

        self.assertEqual(decision.strategy, "deterministic_application_text_entry")
        tools = [step.metadata.get("tool") for step in decision.plan.steps]
        self.assertEqual(tools, ["content.generate", "applications.write_text"])
        self.assertEqual(
            decision.plan.steps[0].metadata["parameters"]["topic"], "cars"
        )
        self.assertEqual(
            decision.plan.steps[1].metadata["parameters"]["application"], "Notepad"
        )

    @patch("computer.text_entry.resolve_application_name")
    @patch("computer.text_entry.subprocess.Popen")
    @patch("computer.text_entry.time.sleep")
    @patch("computer.text_entry.ApplicationTextEntryTool._type_into_process")
    def test_text_entry_launches_and_types_without_shell(self, type_text, sleep, popen, resolve):
        resolve.return_value = Path("C:/Windows/System32/notepad.exe")
        popen.return_value.pid = 321
        type_text.return_value = {
            "delivery": "wm_settext",
            "observed": True,
            "observed_characters": 10,
            "target": {"control_class": "RichEditD2DPT"},
        }

        result = ApplicationTextEntryTool(startup_wait_seconds=0).execute(
            {"application": "Notepad", "text": "Atlas poem"}
        )

        self.assertTrue(result.success)
        self.assertEqual(result.output["characters"], 10)
        self.assertEqual(result.output["verification"], "confirmed")
        self.assertTrue(result.output["observed"])
        self.assertEqual(result.output["delivery"], "wm_settext")
        self.assertFalse(popen.call_args.kwargs["shell"])
        # The launch must be preceded by a snapshot of pre-existing windows, and
        # that snapshot must reach the target search: a packaged app (Windows 11
        # Notepad) redirects the launch into a running instance, so the launched
        # pid owns no window and the wrong document would otherwise be chosen.
        called = type_text.call_args
        self.assertEqual(called.args[:3], (321, "Atlas poem", "notepad.exe"))
        self.assertEqual(called.kwargs["delivery"], "auto")
        self.assertEqual(called.kwargs["window_timeout"], 5)
        self.assertIn("exclude", called.kwargs)
        self.assertIsInstance(called.kwargs["exclude"], frozenset)

    @patch("computer.text_entry.resolve_application_name")
    @patch("computer.text_entry.subprocess.Popen")
    @patch("computer.text_entry.time.sleep")
    @patch("computer.text_entry.ApplicationTextEntryTool._type_into_process")
    def test_absent_text_is_reported_as_a_verification_failure(
        self, type_text, sleep, popen, resolve
    ):
        # Delivery reported success but Atlas could not find the text when it read
        # the control back. That is a verification failure, not an execution
        # failure: the tool still returns success so the executor can tell the two
        # apart and retry with bounded recovery.
        resolve.return_value = Path("C:/Windows/System32/notepad.exe")
        popen.return_value.pid = 321
        type_text.return_value = {
            "delivery": "wm_settext",
            "observed": False,
            "observed_characters": 0,
            "target": {"control_class": "RichEditD2DPT"},
        }

        result = ApplicationTextEntryTool(startup_wait_seconds=0).execute(
            {"application": "Notepad", "text": "Atlas poem"}
        )

        self.assertTrue(result.success)
        self.assertEqual(result.status, "verification_failed")
        self.assertEqual(result.output["verification"], "failed")
        self.assertFalse(result.output["observed"])

    @patch("computer.text_entry.resolve_application_name")
    @patch("computer.text_entry.subprocess.Popen")
    @patch("computer.text_entry.time.sleep")
    @patch("computer.text_entry.ApplicationTextEntryTool._type_into_process")
    def test_unreadable_control_reports_unconfirmed_not_verified(
        self, type_text, sleep, popen, resolve
    ):
        resolve.return_value = Path("C:/Windows/System32/notepad.exe")
        popen.return_value.pid = 321
        type_text.return_value = {
            "delivery": "wm_settext",
            "observed": None,
            "observed_characters": None,
            "target": {"control_class": "Chrome_RenderWidgetHostHWND"},
        }

        result = ApplicationTextEntryTool(startup_wait_seconds=0).execute(
            {"application": "Edge", "text": "Atlas poem"}
        )

        self.assertTrue(result.success)
        self.assertEqual(result.status, "written")
        self.assertEqual(result.output["verification"], "unconfirmed")
        self.assertIsNone(result.output["observed"])

    def test_unknown_delivery_mode_is_rejected_before_launching(self):
        result = ApplicationTextEntryTool().execute(
            {"application": "Notepad", "text": "hello", "delivery": "telepathy"}
        )

        self.assertFalse(result.success)
        self.assertIn("delivery must be one of", result.error)


class TextEntryObservationTests(unittest.TestCase):
    """The read-back is what decides verified / failed / unconfirmed."""

    def test_read_back_confirms_text_that_is_present(self):
        user32 = _FakeUser32(control_class="RichEditD2DPT", text="Atlas poem")

        observed, characters = ApplicationTextEntryTool._confirm_text(
            user32, 7, "Atlas poem", method="wm_settext"
        )

        self.assertTrue(observed)
        self.assertEqual(characters, len("Atlas poem"))

    def test_read_back_reports_failure_when_the_control_is_empty(self):
        user32 = _FakeUser32(control_class="RichEditD2DPT", text="")

        observed, characters = ApplicationTextEntryTool._confirm_text(
            user32, 7, "Atlas poem", method="wm_settext"
        )

        self.assertFalse(observed)
        self.assertEqual(characters, 0)

    def test_read_back_reports_unknown_for_a_non_text_control(self):
        # A control Atlas cannot read must not be reported as a failure.
        user32 = _FakeUser32(control_class="Chrome_RenderWidgetHostHWND", text="")

        observed, characters = ApplicationTextEntryTool._confirm_text(
            user32, 7, "Atlas poem", method="wm_settext"
        )

        self.assertIsNone(observed)
        self.assertIsNone(characters)

    def test_existing_content_does_not_hide_the_new_text(self):
        user32 = _FakeUser32(control_class="Edit", text="old line\nAtlas poem")

        observed, _ = ApplicationTextEntryTool._confirm_text(
            user32, 7, "Atlas poem", method="wm_settext"
        )

        self.assertTrue(observed)

    def test_crlf_conversion_does_not_look_like_a_missing_write(self):
        # Windows controls store newlines as CRLF. A byte-exact comparison would
        # report a false verification failure for text that plainly landed, which
        # is the regression that made a successful Notepad write raise
        # "did not produce the expected effect".
        submitted = "POEM ABOUT CARS\n\nSteel horses resting under amber light."
        stored = submitted.replace("\n", "\r\n")
        user32 = _FakeUser32(control_class="RichEditD2DPT", text=stored)

        observed, characters = ApplicationTextEntryTool._confirm_text(
            user32, 7, submitted, method="wm_settext"
        )

        self.assertTrue(observed)
        self.assertEqual(characters, len(stored))

    def test_absent_text_is_still_a_failure_after_normalization(self):
        # Normalizing line endings must not turn a genuine miss into a success.
        user32 = _FakeUser32(control_class="RichEditD2DPT", text="something else entirely\r\n")

        observed, _ = ApplicationTextEntryTool._confirm_text(
            user32, 7, "POEM ABOUT CARS\n\nSteel horses.", method="wm_settext"
        )

        self.assertFalse(observed)

    def test_direct_delivery_does_not_fall_through_to_paste_when_forced(self):
        # With delivery="direct" Atlas uses only the window message; if that is
        # rejected it reports an execution failure instead of silently trying a
        # different mechanism the caller did not ask for.
        user32 = _FakeUser32(control_class="Edit", text="")
        user32.SendMessageW = lambda *args: 0

        delivered = ApplicationTextEntryTool._deliver(
            user32, 1, 7, "Atlas poem", delivery="direct"
        )

        self.assertIsNone(delivered)
        self.assertEqual(user32.paste_calls, [])


if __name__ == "__main__":
    unittest.main()


class TargetWindowSelectionTests(unittest.TestCase):
    """The target must be the window the launch created, not a stale one."""

    def test_packaged_app_redirect_excludes_preexisting_windows(self):
        # Windows 11 Notepad redirects a fresh launch into the running instance,
        # so the launched pid owns no window. With two documents already open the
        # old fallback ("first window with this image") picked the wrong one.
        from computer.perception import WindowInfo

        stale = WindowInfo(
            hwnd=100, title="Untitled - Notepad", class_name="Notepad",
            pid=999, executable="notepad.exe", focused=False, minimized=False,
        )
        fresh = WindowInfo(
            hwnd=200, title="Untitled - Notepad", class_name="Notepad",
            pid=555, executable="notepad.exe", focused=True, minimized=False,
        )
        with patch("computer.perception.list_windows", return_value=[fresh, stale]):
            chosen = find_target_window(
                _NoopUser32(), 321, "notepad.exe", timeout=0.0, exclude=frozenset({100})
            )
        self.assertEqual(chosen, 200)

    def test_launched_pid_wins_when_it_owns_a_window(self):
        from computer.perception import WindowInfo

        other = WindowInfo(
            hwnd=100, title="other - Notepad", class_name="Notepad",
            pid=999, executable="notepad.exe", focused=False, minimized=False,
        )
        owned = WindowInfo(
            hwnd=200, title="owned - Notepad", class_name="Notepad",
            pid=321, executable="notepad.exe", focused=False, minimized=False,
        )
        with patch("computer.perception.list_windows", return_value=[other, owned]):
            chosen = find_target_window(_NoopUser32(), 321, "notepad.exe", timeout=0.0)
        self.assertEqual(chosen, 200)

    def test_no_match_returns_zero_so_the_caller_reports_honestly(self):
        with patch("computer.perception.list_windows", return_value=[]):
            self.assertEqual(
                find_target_window(_NoopUser32(), 321, "notepad.exe", timeout=0.0), 0
            )

    def test_window_snapshot_is_a_frozenset_of_handles(self):
        from computer.perception import WindowInfo

        window = WindowInfo(
            hwnd=42, title="t", class_name="Notepad", pid=1,
            executable="notepad.exe", focused=False, minimized=False,
        )
        with patch("computer.perception.list_windows", return_value=[window]):
            snapshot = list_titled_window_handles(user32_lib=_NoopUser32())
        self.assertEqual(snapshot, frozenset({42}))

    def test_snapshot_failure_degrades_to_empty_not_an_exception(self):
        with patch("computer.perception.list_windows", side_effect=RuntimeError("boom")):
            self.assertEqual(list_titled_window_handles(user32_lib=_NoopUser32()), frozenset())


class TextControlSelectionTests(unittest.TestCase):
    """A container control must never be chosen over the real text surface."""

    def test_rich_edit_is_preferred_over_a_notepad_container(self):
        # Windows 11 Notepad nests NotepadTextBox around RichEditD2DPT. Writing
        # to the container sets the window title, so the read-back legitimately
        # found nothing at the document - the original bug.
        children = {10: "NotepadTextBox", 20: "RichEditD2DPT"}

        class FakeUser32:
            def GetClassNameW(self, hwnd, buffer, _length):
                buffer.value = children[hwnd]
                return len(buffer.value)

            def EnumChildWindows(self, _parent, callback, _param):
                for hwnd in children:
                    if not callback(hwnd, 0):
                        break

        self.assertEqual(find_text_control(FakeUser32(), 1), 20)

    def test_class_ranking_orders_surfaces_before_wrappers(self):
        self.assertLess(_control_class_rank("RichEditD2DPT"), _control_class_rank("Edit"))
        self.assertLess(_control_class_rank("Edit"), _control_class_rank("NotepadTextBox"))

    def test_window_is_returned_when_no_control_matches(self):
        class FakeUser32:
            def GetClassNameW(self, _hwnd, buffer, _length):
                buffer.value = "Static"
                return 6

            def EnumChildWindows(self, _parent, callback, _param):
                callback(5, 0)

        self.assertEqual(find_text_control(FakeUser32(), 77), 77)


class _NoopUser32:
    """Placeholder user32 for tests whose list_windows is patched out."""

    def EnumChildWindows(self, *_args, **_kwargs):
        return 0
