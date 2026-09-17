"""Offline smoke test for the reasoning engine (no network, no live model)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models_task import Task  # noqa: E402
from reasoning.answer_generator import AnswerGenerator  # noqa: E402
from reasoning.reasoning_engine import ReasoningEngine  # noqa: E402
from reasoning.self_introspection import SelfIntrospection  # noqa: E402
from tools.base import ToolResult  # noqa: E402
from tools.capabilities import CapabilityRegistry  # noqa: E402
from tools.registry import ToolRegistry  # noqa: E402
from tools.router import ToolRouter  # noqa: E402


class FakeAsk:
    def __call__(self, *, system_prompt="", user_prompt="", **kw):
        return "Python is a high-level programming language known for readability."


def offline_tool_runner(name, parameters):
    """Simulate deterministic tool execution without touching the machine."""
    print(f"    [tool call: {name} {dict(parameters)}]")
    return ToolResult.failure("offline smoke run", recoverable=True)


def main() -> int:
    registry = ToolRegistry()
    from computer.runtime import register_read_only_tools
    from config import COMPUTER_ROOT
    from tools.permissions import ExecutionMode, PermissionEngine

    register_read_only_tools(registry, root=COMPUTER_ROOT, ask=FakeAsk())
    capabilities = CapabilityRegistry(registry)
    engine = ReasoningEngine(
        self_introspection=SelfIntrospection(capabilities),
        answer_generator=AnswerGenerator(ask=FakeAsk()),
        tool_runner=offline_tool_runner,
        capabilities=capabilities,
        retrieve=lambda *_args: {"context": "", "best_distance": None, "sources": []},
    )
    prompts = [
        "What can you do?",
        "What is Python?",
        "What's the latest Python version?",
        "Open Notepad.",
        "What did I ask you earlier?",
        "What files are in my Downloads folder?",
        "Explain recursion.",
    ]
    failures = 0
    for q in prompts:
        try:
            out = engine.handle_request(
                question=q, task=Task(goal=q, original_prompt=q)
            )
            if out is None:
                print(f"{q!r:45} -> delegated to Task pipeline (None)")
                continue
            data = out.to_dict()
            print(
                f"{q!r:45} mode={data.get('mode')} "
                f"provenance={data.get('provenance')} "
                f"answer={out.text[:50]!r}"
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            import traceback

            traceback.print_exc()
            print(f"{q!r:45} FAIL {type(exc).__name__}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
