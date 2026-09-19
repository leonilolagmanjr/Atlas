"""Temporary probe 4: broader realistic request families."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoning.task_interpreter import SemanticTaskInterpreter
from tools.capabilities import CapabilityRegistry
from tools.registry import ToolRegistry
from computer.runtime import register_read_only_tools
from config import COMPUTER_ROOT

PROMPTS = [
    "write a letter to my boss asking for time off and save it as letter.txt",
    "create a summary of my report.pdf and put it in notepad",
    "search for flights to Tokyo and save the cheapest ones to a file",
    "find the largest file in my Downloads and delete it",
    "open chrome and search for weather",
    "make a numbered list of 3 productivity tips",
    "download a picture of a cat",
    "install vscode",
    "send an email to john",
    "write python code to reverse a string and save it as reverse.py",
    "copy my resume to the desktop",
    "what's the capital of Japan",
    "summarize this article that I read earlier",
    "search youtube for lofi music and save the links to a file",
]

registry = ToolRegistry()
register_read_only_tools(registry, root=COMPUTER_ROOT, ask=lambda **k: "")
caps = CapabilityRegistry(registry)
interp = SemanticTaskInterpreter(ask=None, capabilities=caps)

for p in PROMPTS:
    t = interp.interpret(p)
    print("=" * 70)
    print(f"PROMPT: {p}")
    print(f"  type={t.task_type} req={t.request_type} clarify={t.needs_clarification} conf={t.confidence}")
    print(f"  entities={t.entities}")
    for a in t.actions:
        print(f"    {a.capability} {a.parameters}")
