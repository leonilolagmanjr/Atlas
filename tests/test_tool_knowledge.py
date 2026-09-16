import unittest
from pathlib import Path

from tools.discovery import ToolDiscovery
from tools.knowledge import ToolKnowledgeStore, load_json


class ToolKnowledgeTests(unittest.TestCase):
    def setUp(self):
        records = load_json(Path("tools/powershell_commands.json"))
        self.discovery = ToolDiscovery(ToolKnowledgeStore(records))

    def test_json_records_are_searchable_by_capability(self):
        candidates = self.discovery.discover("inspect memory usage", tool_type="powershell")

        self.assertTrue(candidates)
        self.assertEqual(candidates[0].tool, "Get-Process")
        self.assertTrue(candidates[0].read_only)

    def test_unknown_capability_has_no_candidates(self):
        self.assertEqual(self.discovery.discover("control satellites", tool_type="powershell"), [])


if __name__ == "__main__":
    unittest.main()
