"""Scenarios: how a case is served — simulated (offline) or live.

A scenario owns the *environment* a case runs in:

* an isolated conversation store (a temp folder per run, so evaluation never
  touches the user's real conversations);
* the tool registry, permission engine and router (the real ones for live mode;
  a deterministic recording set for simulated mode where the case is about
  routing, not about the tool's real effect);
* the model boundary (a canned ``ask`` for simulated mode; the real provider for
  live mode);
* the dependency snapshot that decides BLOCKED vs FAIL.

The scenario exposes exactly two things to the harness: ``open_conversation()``
(returning an object with ``send(text) -> TurnObservation``) and
``dependency_status(case)`` / ``missing_dependency_for(case)``.

Crucially the scenario never plans, executes, or calls a tool directly. It only
wires the *existing* objects; the Conversation Runtime does the rest.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.cases import DependencyStatus, EvalCase
from evaluation.dependencies import probe_dependencies
from evaluation.harness import TurnObservation, _EventRecorder


class _PatchModuleConstants:
    """Temporarily override module-level constants, restoring them on exit.

    Atlas reads its configuration as module-level constants captured at import
    time (e.g. ``reasoning.task_interpreter.ENABLE_LLM_INTERPRETATION``). A
    simulated run needs those constants to hold deterministic values for the
    duration of one scenario without leaking that change into a live run, so the
    patch is applied and reverted inside a context manager.
    """

    def __init__(self, overrides: dict[Any, Any]) -> None:
        self._overrides = list(overrides.items())
        self._saved: list[tuple[Any, str, Any]] = []

    def __enter__(self) -> "_PatchModuleConstants":
        for module, values in self._overrides:
            for name, value in values.items():
                self._saved.append((module, name, getattr(module, name, None)))
                setattr(module, name, value)
        return self

    def __exit__(self, *_exc: object) -> None:
        for module, name, previous in reversed(self._saved):
            setattr(module, name, previous)
        self._saved.clear()


def _deterministic_interpretation() -> "_PatchModuleConstants":
    """Disable the optional LLM interpretation for a simulated scenario."""

    import config
    import reasoning.interpreter as legacy_interpreter
    import reasoning.recovery as recovery
    import reasoning.task_interpreter as task_interpreter

    overrides = {
        config: {"ENABLE_LLM_INTERPRETATION": False},
        task_interpreter: {"ENABLE_LLM_INTERPRETATION": False},
        legacy_interpreter: {"ENABLE_LLM_INTERPRETATION": False},
        recovery: {"ENABLE_LLM_INTERPRETATION": False},
    }
    return _PatchModuleConstants(overrides)


class _NullIndex:
    """Deterministic-only conversation index (no Chroma, no embedder, no network)."""

    available = False

    def search(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return []

    def index_message(self, **_kwargs: Any) -> bool:
        return False

    def forget_conversation(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _EmbedderStub:
    def index(self, *_args: Any, **_kwargs: Any) -> bool:
        return False

    def similarities(self, *_args: Any, **_kwargs: Any) -> dict[str, float]:
        return {}

    def delete(self, *_args: Any, **_kwargs: Any) -> None:
        return None


@dataclass
class _SimulatedAsk:
    """A deterministic model boundary for simulated cases.

    Simulated cases are about *routing*: which layer handled the message and
    which tools ran. The model's words are irrelevant to that decision, so this
    returns a fixed answer and records the prompts for the raw report. It never
    invents a tool call or an action.
    """

    answer: str = "This is a simulated answer produced without a live model."
    calls: list[str] = field(default_factory=list)

    def __call__(self, *, system_prompt: str = "", user_prompt: str = "", **_kw: Any) -> str:
        self.calls.append(user_prompt)
        return self.answer


class Conversation:
    """One evaluation conversation with rebuildable, turn-local runtime state.

    A *fresh* runtime is built from the persisted store for every turn, so a
    follow-up must be reconstructed from persisted state exactly as it would be
    across a real turn boundary (or an app restart). This is what makes the
    continuity cases about behavior rather than object identity.
    """

    def __init__(self, scenario: "Scenario", conversation_id: str) -> None:
        self._scenario = scenario
        self._conversation_id = conversation_id

    def send(self, text: str) -> TurnObservation:
        recorder = _EventRecorder()
        runtime = self._scenario.build_runtime()
        result = runtime.handle_message(
            text,
            conversation_id=self._conversation_id,
            events=recorder,
        )
        payload = result.to_dict()
        return TurnObservation(
            text=payload.get("text", ""),
            kind=payload.get("kind", ""),
            execution_state=payload.get("execution_state", ""),
            tool_calls=list(payload.get("tool_calls") or []),
            citations=list(payload.get("citations") or []),
            events=recorder.events,
            raw=payload,
        )


class Scenario:
    """Base scenario: isolated store, real registry wiring, dependency snapshot."""

    #: True when the scenario drives the real local model and real tools.
    live: bool = False

    def __init__(self, *, root: Path, dependencies: dict[str, Any] | None = None) -> None:
        self._root = root
        self._dependencies = dependencies or {}

    # -- lifecycle -------------------------------------------------------------

    def open_conversation(self) -> Conversation:
        memory = self._build_memory()
        metadata = memory.create_session(title="evaluation")
        return Conversation(self, metadata.id)

    def build_runtime(self) -> Any:
        raise NotImplementedError

    # -- dependency reporting --------------------------------------------------

    def dependency_status(self, case: EvalCase) -> dict[str, str]:
        return {
            name: self._state(name).value
            for name in case.requires
        }

    def missing_dependency_for(self, case: EvalCase) -> str:
        """Return a reason string when a required dependency is unavailable."""

        for name in case.requires:
            state = self._state(name)
            if state is DependencyStatus.UNAVAILABLE:
                return f"{name} unavailable: {self._reason(name)}"
            if state is DependencyStatus.DEGRADED:
                return f"{name} degraded: {self._reason(name)}"
        return ""

    def blocked_for(self, case: EvalCase) -> str:
        """Return a BLOCKED reason when the case cannot be served honestly.

        A case is blocked when it ``requires`` a dependency that is unavailable
        and it is configured to skip (the default). A case whose missing
        dependency *is its core target* (``skip_if_missing=False``) is served
        anyway so its honest-degradation behavior is evaluated.
        """

        for name in case.requires:
            if self._state(name) is DependencyStatus.UNAVAILABLE and case.skip_if_missing:
                return f"required dependency unavailable: {name} ({self._reason(name)})"
        return ""

    def _state(self, name: str) -> DependencyStatus:
        probe = self._dependencies.get(name)
        if probe is None:
            return DependencyStatus.NOT_REQUIRED
        return DependencyStatus(probe.status)

    def _reason(self, name: str) -> str:
        probe = self._dependencies.get(name)
        return getattr(probe, "reason", "") if probe else ""

    # -- shared builders -------------------------------------------------------

    def _build_memory(self) -> Any:
        """Build a MemoryManager backed by an isolated store under the run root."""

        from memory.memory_manager import MemoryManager
        from memory.storage import ConversationStore

        manager = MemoryManager(ask=None, index=_NullIndex())
        manager._store = ConversationStore(root_folder=self._root)  # noqa: SLF001
        manager._sessions._store = manager._store  # noqa: SLF001
        manager._sessions._active_path = self._root / "active_session.json"  # noqa: SLF001
        return manager

    def _build_user_memory(self) -> Any:
        from memory.user_memory import UserMemoryStore

        return UserMemoryStore(
            path=self._root / "user_memory.jsonl",
            enabled=False,
            embedder=_EmbedderStub(),
        )


class SimulatedScenario(Scenario):
    """Offline scenario: deterministic boundary, real routing, real registry.

    The tool registry, interpreter, Intent Engine, validator, planner and
    executor are the *real* ones, so the routing decision under test is genuine.
    Two things are made deterministic so the result is reproducible and never
    touches the network or the OS:

    * **LLM interpretation is disabled.** Atlas's interpreter is documented as
      "deterministic and conservative", and disabling the optional model
      consultation is a supported state (``enabled=False``). This makes the case
      about the deterministic routing layer rather than about whether a canned
      model happens to produce parseable JSON, which would fabricate the result.
    * **Tool effects are supplied as fixed data** (``DeterministicToolRouter``).
      The routing is real; the effect is not performed.

    A simulated result is therefore a statement about the deterministic routing
    layer only. It is reported separately from live results and never mixed.
    """

    live = False

    def __init__(self, *, root: Path, dependencies: dict[str, Any] | None = None) -> None:
        super().__init__(root=root, dependencies=dependencies)
        self._ask = _SimulatedAsk()
        self._brain = None
        self._interpretation_patch: _PatchModuleConstants | None = None
        #: Per-case simulated tool overrides, set by :meth:`open_conversation`.
        self._tool_overrides: dict[str, dict[str, Any]] = {}

    def set_case(self, case: EvalCase) -> None:
        """Apply a case's simulated tool overrides before it is served."""

        self._tool_overrides = dict(case.tool_overrides or {})
        self._brain = None

    def build_runtime(self) -> Any:
        from memory.context_manager import ContextManager
        from memory.conversation_runtime import ConversationRuntime

        memory = self._build_memory()
        with _deterministic_interpretation():
            brain = self._build_brain()
        return ConversationRuntime(
            memory_manager=memory,
            brain=brain,
            context_manager=ContextManager(memory_manager=memory, index=_NullIndex()),
            user_memory=self._build_user_memory(),
            ask=self._ask,
        )

    def open_conversation(self) -> Conversation:
        memory = self._build_memory()
        metadata = memory.create_session(title="evaluation")
        # Seed the active session so every rebuilt runtime sees the same one.
        self._active_conversation_id = metadata.id
        return Conversation(self, metadata.id)

    def _build_brain(self) -> Any:
        """Build a real Brain over the real registry, but never a real tool run.

        The Brain, planner, validator and executor are the *real* ones, so the
        routing decision under test is genuine. The tool router is wrapped so
        every tool executes against a deterministic fake: a simulated case must
        not touch the network or the filesystem, because a simulated run's result
        must be reproducible and must not mutate the machine.
        """

        if self._brain is not None:
            return self._brain

        from brain import Brain
        from executor import Executor
        from planner import Planner
        from tools import ExecutionMode, PermissionEngine, ToolRegistry, ToolRouter
        from computer.runtime import register_read_only_tools

        from config import COMPUTER_ROOT
        from evaluation.fake_tools import DeterministicToolRouter

        registry = ToolRegistry()
        register_read_only_tools(registry, root=COMPUTER_ROOT, ask=None)
        real_router = ToolRouter(
            registry=registry,
            permission_engine=PermissionEngine(mode=ExecutionMode.AUTONOMOUS),
        )
        router = DeterministicToolRouter(
            real_router,
            canned=self._tool_overrides.get("canned"),
            failing=self._tool_overrides.get("failing"),
        )

        brain = Brain(
            vector_store=_NoVectorStore(),
            system_prompt="sys",
            retrieval_template="{conversation_history}{context}{question}",
            planner=Planner(registry=router, ask=self._ask),
            executor=Executor(
                vector_store=_NoVectorStore(),
                system_prompt="sys",
                retrieval_template="{conversation_history}{context}{question}",
                tool_router=router,
            ),
            tool_router=router,
            llm_ask=self._ask,
        )
        self._brain = brain
        return brain


class LiveScenario(Scenario):
    """Live scenario: the real provider, the real registry, the real tools.

    Live cases are the only ones that call the local model and the network. Their
    results are reported separately from simulated results and are never mixed.
    """

    live = True

    def build_runtime(self) -> Any:
        from memory.context_manager import ContextManager
        from memory.conversation_runtime import ConversationRuntime
        from llm import ask

        memory = self._build_memory()
        brain = self._build_brain(memory)
        return ConversationRuntime(
            memory_manager=memory,
            brain=brain,
            context_manager=ContextManager(memory_manager=memory, index=_NullIndex()),
            user_memory=self._build_user_memory(),
            ask=ask,
        )

    def _build_brain(self, memory: Any) -> Any:
        if getattr(self, "_brain", None) is not None:
            return self._brain
        from brain import Brain
        from config import COMPUTER_ROOT
        from computer.runtime import register_read_only_tools
        from llm import ask
        from tools import ExecutionMode, PermissionEngine, ToolRegistry, ToolRouter

        registry = ToolRegistry()
        register_read_only_tools(registry, root=COMPUTER_ROOT, ask=ask)
        router = ToolRouter(
            registry=registry,
            permission_engine=PermissionEngine(mode=ExecutionMode.AUTONOMOUS),
        )
        self._brain = Brain(
            vector_store=_NoVectorStore(),
            system_prompt="sys",
            retrieval_template="{conversation_history}{context}{question}",
            memory_manager=memory,
            tool_router=router,
            llm_ask=ask,
        )
        return self._brain


class _NoVectorStore:
    """An empty vector store: no knowledge base is indexed during evaluation."""

    def search(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        return []


@dataclass
class RunEnvironment:
    """Owns the temporary root and the dependency snapshot for one run."""

    root: Path
    dependencies: dict[str, Any]

    @classmethod
    def create(cls, *, mode: str = "simulated") -> "RunEnvironment":
        temporary = tempfile.TemporaryDirectory(prefix="atlas-eval-")
        root = Path(temporary.name)
        env = cls(root=root, dependencies={})
        env._temporary = temporary  # type: ignore[attr-defined]
        env.dependencies = probe_dependencies(_required_dependencies())
        return env

    def scenario(self, *, live: bool) -> Scenario:
        if live:
            return LiveScenario(root=self.root, dependencies=self.dependencies)
        return SimulatedScenario(root=self.root, dependencies=self.dependencies)

    def cleanup(self) -> None:
        temporary = getattr(self, "_temporary", None)
        if temporary is not None:
            temporary.cleanup()


def _required_dependencies() -> list[str]:
    from evaluation.cases import all_cases

    names: list[str] = []
    for case in all_cases():
        for name in case.requires:
            if name not in names:
                names.append(name)
    return names
