"""One-shot compact ground truth for the failing section-25 behavior tests.

Run:  python scripts/_probe_s25.py
Every printed region is an edit anchor for the fix pass. No writes here.
"""
from __future__ import annotations

import dataclasses
import inspect
import re


def grab(path, pattern, before=1, after=5, label=None, occurrence=1):
    src = open(path, encoding="utf-8").read().splitlines()
    hits = [i for i, line in enumerate(src) if re.search(pattern, line)]
    if not hits:
        print(f"### {label or pattern}: NOT FOUND in {path}")
        print()
        return
    hit = hits[min(occurrence, len(hits)) - 1]
    lo, hi = max(0, hit - before), min(len(src), hit + after)
    print(f"### {label or pattern}  [{path} L{hit + 1}; {len(hits)} hits]")
    for i in range(lo, hi):
        print(f"{i + 1}: {src[i]}")
    print()


def list_defs(path):
    src = open(path, encoding="utf-8").read().splitlines()
    print(f"### defs in {path}")
    for i, line in enumerate(src):
        if re.match(r"\s*(def |class )\s", line):
            print(f"{i + 1}: {line.strip()}")
    print()


print("=" * 20, "ENGINE", "=" * 20)
list_defs("reasoning/reasoning_engine.py")
grab("reasoning/reasoning_engine.py", r"def handle_request", 0, 40, "handle_request")
grab("reasoning/reasoning_engine.py", r"retrieved_chunks", 6, 14, "knowledge gather key")

print("=" * 20, "ROUTER", "=" * 20)
list_defs("reasoning/query_router.py")
grab("reasoning/query_router.py", r"explicit_web_request", 4, 10, "web signal")
grab("reasoning/query_router.py", r"QuerySignals\(", 0, 16, "QuerySignals ctor")

print("=" * 20, "TESTS", "=" * 20)
src = open("tests/test_reasoning_behavior.py", encoding="utf-8").read().splitlines()
print("### test names")
for i, line in enumerate(src):
    if re.match(r"\s*def test_", line):
        print(f"{i + 1}: {line.strip()}")
print()
grab("tests/test_reasoning_behavior.py", r"retrieved_chunks|\"chunks\"|'chunks'", 5, 12, "test retrieval fake")
grab("tests/test_reasoning_behavior.py", r"def _?fake|class Fake", 0, 16, "fake classes")
grab("tests/test_reasoning_behavior.py", r"Open Notepad", 4, 14, "notepad test")
grab("tests/test_reasoning_behavior.py", r"Search the web|signals", 3, 10, "web-signal test")

print("=" * 20, "REAL SHAPES", "=" * 20)
print("### deterministic interpretation")
try:
    from computer.runtime import register_read_only_tools
    from reasoning.task_interpreter import SemanticTaskInterpreter
    from tools.capabilities import CapabilityRegistry, render_capability_catalog
    from tools.registry import ToolRegistry

    registry = ToolRegistry()
    try:
        register_read_only_tools(registry)
    except Exception as exc:
        print("register_read_only_tools failed:", exc)
    caps = CapabilityRegistry(registry)
    interpreter = SemanticTaskInterpreter(
        ask=None, capabilities=caps, capability_catalog=render_capability_catalog(registry)
    )
    for prompt in ("Open Notepad.", "Search the web for the latest Python version."):
        task = interpreter.interpret(prompt)
        print(prompt, "->", {
            "task_type": getattr(task, "task_type", None),
            "requires_computer": getattr(task, "requires_computer", None),
            "needs_clarification": getattr(task, "needs_clarification", None),
            "actions": [getattr(a, "capability", None) for a in task.actions],
            "entities": getattr(task, "entities", None),
        })
except Exception:
    import traceback

    traceback.print_exc()
print()
print("### RetrievalResult shape")
try:
    from knowledge_search import RetrievalResult

    print("fields:", [f.name for f in dataclasses.fields(RetrievalResult)])
    if hasattr(RetrievalResult, "to_dict"):
        print(inspect.getsource(RetrievalResult.to_dict))
except Exception:
    import traceback

    traceback.print_exc()
