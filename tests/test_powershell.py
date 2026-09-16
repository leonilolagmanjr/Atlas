import unittest
from pathlib import Path
from unittest.mock import patch

from computer.powershell import PowerShellTool, PowerShellValidator
from tools.knowledge import ToolKnowledgeStore, load_json


class PowerShellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        records = load_json(Path("tools/powershell_commands.json"))
        cls.knowledge = ToolKnowledgeStore(records)

    def test_documented_read_only_command_is_safe(self):
        result = PowerShellValidator(self.knowledge).validate("Get-Process | Select-Object -First 3")

        self.assertTrue(result.valid)
        self.assertEqual(result.risk_level, "safe")

    def test_dangerous_commands_are_blocked_before_subprocess(self):
        result = PowerShellValidator(self.knowledge).validate("Remove-Item C:\\Temp -Recurse -Force")

        self.assertFalse(result.valid)
        self.assertIn("deletion", result.reason)

    def test_unknown_command_is_not_executable(self):
        result = PowerShellValidator(self.knowledge).validate("Some-UnknownCommand")

        self.assertFalse(result.valid)
        self.assertIn("trusted PowerShell knowledge base", result.reason)

    def test_encoded_commands_are_blocked(self):
        result = PowerShellValidator(self.knowledge).validate("Get-Process -EncodedCommand abc")

        self.assertFalse(result.valid)
        self.assertIn("encoded", result.reason)

    @patch("computer.powershell.shutil.which", return_value="powershell.exe")
    @patch("computer.powershell.subprocess.run")
    def test_execution_returns_structured_result(self, run, _which):
        run.return_value.returncode = 0
        run.return_value.stdout = "Name  Id\nAtlas 42\n"
        run.return_value.stderr = ""

        result = PowerShellTool(knowledge=self.knowledge).execute({"command": "Get-Process"})

        self.assertTrue(result.success)
        self.assertEqual(result.output["exit_code"], 0)
        self.assertIn("Atlas", result.output["stdout"])
        self.assertFalse(run.call_args.kwargs["shell"])


if __name__ == "__main__":
    unittest.main()
