import unittest
from pathlib import Path
from unittest.mock import patch

from computer.text_entry import ApplicationTextEntryTool
from intent_classifier import IntentClassifier
from planner import Planner


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

        result = ApplicationTextEntryTool(startup_wait_seconds=0).execute(
            {"application": "Notepad", "text": "Atlas poem"}
        )

        self.assertTrue(result.success)
        self.assertEqual(result.output["characters"], 10)
        self.assertFalse(popen.call_args.kwargs["shell"])
        type_text.assert_called_once_with(321, "Atlas poem", "notepad.exe")


if __name__ == "__main__":
    unittest.main()
