"""Temporary probe: current routing + whether any perception capability exists."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import COMPUTER_ROOT
from computer.runtime import register_read_only_tools
from reasoning.task_interpreter import SemanticTaskInterpreter
from tools.capabilities import CapabilityRegistry
from tools.registry import ToolRegistry

PROMPTS = [
    "What are the most popular YouTube videos about cars?",
    "Write a poem about cars in Notepad.",
    "Search YouTube for popular car videos and put the titles into Notepad.",
    "Create a presentation about climate change.",
    "Open Notepad and type hello.",
]

registry = ToolRegistry()
register_read_only_tools(registry, root=COMPUTER_ROOT, ask=lambda **k: "")
caps = CapabilityRegistry(registry)
interp = SemanticTaskInterpreter(ask=None, capabilities=caps)

print("REGISTERED TOOLS:")
for name in registry.names():
    print("  ", name)

for prompt in PROMPTS:
    task = interp.interpret(prompt)
    print("=" * 72)
    print("PROMPT:", prompt)
    print("  task_type=", task.task_type, "request_type=", task.request_type)
    print("  goal=", task.goal, "conf=", round(task.confidence, 3))
    print("  sources=", task.sources)
    for action in task.actions:
        print("   action:", action.capability, action.parameters, "produces=", action.produces)
