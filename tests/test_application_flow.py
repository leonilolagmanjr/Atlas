import unittest

from cli import CLI
from intent_classifier import IntentClassifier
from planner import Planner


class ApplicationFlowTests(unittest.TestCase):
    def test_explicit_executable_is_application_intent(self):
        result = IntentClassifier().classify(r"Open C:\Windows\System32\notepad.exe")

        self.assertEqual(result.intent, "APPLICATION")

    def test_application_plan_contains_tool_parameters(self):
        decision = Planner().create_plan(
            r"Launch C:\Windows\System32\notepad.exe",
            intent="APPLICATION",
        )
        step = decision.plan.steps[0]

        self.assertEqual(step.action, "invoke_tool")
        self.assertEqual(step.metadata["tool"], "applications.launch")
        self.assertEqual(
            step.metadata["parameters"]["executable"],
            r"C:\Windows\System32\notepad.exe",
        )

    def test_cli_approval_commands_use_callbacks(self):
        class MemoryStub:
            pass

        cli = CLI(
            memory_manager=MemoryStub(),
            approve_callback=lambda: "approved",
            deny_callback=lambda: "denied",
        )

        self.assertEqual(cli.handle_command("/approve"), "approved")
        self.assertEqual(cli.handle_command("/deny"), "denied")


if __name__ == "__main__":
    unittest.main()
