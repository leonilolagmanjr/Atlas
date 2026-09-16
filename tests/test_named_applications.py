import unittest
from pathlib import Path
from unittest.mock import patch

from computer.launch import NamedApplicationLaunchTool, resolve_application_name
from intent_classifier import IntentClassifier
from planner import Planner


class NamedApplicationTests(unittest.TestCase):
    def test_discord_request_is_application_intent(self):
        result = IntentClassifier().classify("Open Discord")

        self.assertEqual(result.intent, "APPLICATION")

    def test_arbitrary_named_application_request_is_application_intent(self):
        result = IntentClassifier().classify("Launch Spotify")

        self.assertEqual(result.intent, "APPLICATION")

    def test_discord_request_uses_named_launch_tool(self):
        decision = Planner().create_plan("Open Discord", intent="APPLICATION")
        step = decision.plan.steps[0]

        self.assertEqual(decision.strategy, "deterministic_named_application_launch")
        self.assertEqual(step.metadata["tool"], "applications.launch_named")
        self.assertEqual(step.metadata["parameters"]["application"], "Discord")

    @patch("computer.launch.resolve_application_name")
    @patch("computer.launch.subprocess.Popen")
    def test_named_launch_uses_non_shell_process(self, popen, resolve):
        executable = Path("C:/Apps/Discord/Discord.exe")
        resolve.return_value = executable
        popen.return_value.pid = 123

        result = NamedApplicationLaunchTool().execute({"application": "discord"})

        self.assertTrue(result.success)
        self.assertEqual(result.output["pid"], 123)
        self.assertFalse(popen.call_args.kwargs["shell"])

    @unittest.skipUnless(__import__("sys").platform == "win32", "Windows application registry is required")
    def test_installed_discord_resolves_when_present(self):
        resolved = resolve_application_name("Discord")
        self.assertTrue(resolved is None or resolved.name.casefold() == "discord.exe")


if __name__ == "__main__":
    unittest.main()
