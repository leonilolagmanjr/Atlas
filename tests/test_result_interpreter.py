import unittest

from tools.base import ToolResult
from tools.result_interpreter import interpret_tool_result


class ResultInterpreterTests(unittest.TestCase):
    def test_process_json_becomes_readable_summary(self):
        result = ToolResult(
            success=True,
            status="completed",
            output={
                "stdout": '[{"Name":"Atlas","Id":42,"WorkingSet64":2097152}]',
            },
        )

        summary = interpret_tool_result("powershell.execute", result)

        self.assertIn("Top processes by memory usage", summary)
        self.assertIn("2.0 MB", summary)

    def test_pipeline_requires_known_formatter(self):
        from pathlib import Path
        from computer.powershell import PowerShellValidator
        from tools.knowledge import ToolKnowledgeStore, load_json

        knowledge = ToolKnowledgeStore(load_json(Path("tools/powershell_commands.json")))
        validation = PowerShellValidator(knowledge).validate("Get-Process | Unknown-Formatter")

        self.assertFalse(validation.valid)
        self.assertIn("Unknown-Formatter", validation.reason)

    def test_command_chaining_is_rejected(self):
        from pathlib import Path
        from computer.powershell import PowerShellValidator
        from tools.knowledge import ToolKnowledgeStore, load_json

        knowledge = ToolKnowledgeStore(load_json(Path("tools/powershell_commands.json")))
        validation = PowerShellValidator(knowledge).validate("Get-Process; Get-Service")

        self.assertFalse(validation.valid)
        self.assertIn("chaining", validation.reason)


if __name__ == "__main__":
    unittest.main()
