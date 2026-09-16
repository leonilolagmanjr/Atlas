import unittest

from intent_classifier import IntentClassifier
from planner import Planner


class PowerShellPlanningTests(unittest.TestCase):
    def test_memory_request_selects_computer_intent(self):
        result = IntentClassifier().classify("Show me what applications are using the most RAM")

        self.assertEqual(result.intent, "COMPUTER")

    def test_memory_request_uses_documented_powershell_tool(self):
        decision = Planner().create_plan(
            "Show me what applications are using the most RAM",
            intent="COMPUTER",
        )
        step = decision.plan.steps[0]

        self.assertEqual(decision.strategy, "knowledge_backed_powershell_selection")
        self.assertEqual(step.metadata["tool"], "powershell.execute")
        self.assertTrue(step.metadata["parameters"]["command"].startswith("Get-Process"))
        self.assertIn("Get-Process", step.metadata["candidates"])

    def test_destructive_request_does_not_get_a_powershell_plan(self):
        decision = Planner().create_plan("Delete this directory", intent="UNKNOWN")

        self.assertNotEqual(decision.strategy, "knowledge_backed_powershell_selection")


if __name__ == "__main__":
    unittest.main()
