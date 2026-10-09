"""Deterministic tool execution for simulated evaluation cases.

A simulated case must be reproducible and must not touch the network or the
filesystem. It still needs the *real routing* — the real registry, planner,
validator and executor decide which tool is selected — but the tool's own effect
is supplied here as fixed data.

``DeterministicToolRouter`` wraps the real :class:`tools.router.ToolRouter`. It
keeps the real tool's *metadata* (so capability gating, permission evaluation and
plan validation all behave exactly as in production) while replacing the
``execute`` result with a deterministic one. It can be configured to make a named
tool fail, which is how the honesty cases ("a failed tool must be reported as a
failure") are evaluated without breaking the machine.
"""

from __future__ import annotations

from typing import Any

from tools.base import ToolResult


#: Representative outputs for the read-only tools the evaluation exercises.
#: They are deliberately realistic (URLs, titles, snippets, file lists) so the
#: downstream evidence/synthesis logic has genuine data to work with and a
#: routing case measures routing rather than an empty tool result.
_DEFAULT_TOOL_OUTPUTS: dict[str, Any] = {
    "web.search": {
        "provider": "evaluation",
        "results": [
            {
                "title": "Evaluation reference page",
                "url": "https://example.test/evaluation",
                "snippet": "A deterministic reference page supplied by the evaluation harness.",
            }
        ],
    },
    "web.fetch": {
        "url": "https://example.test/evaluation",
        "title": "Evaluation reference page",
        "text": (
            "This is a deterministic reference page supplied by the evaluation "
            "harness. It contains enough text for retrieval and evidence checks. "
        ) * 4,
    },
    "web.research": {
        "provider": "evaluation",
        "sources": ["https://example.test/evaluation"],
        "content": (
            "Deterministic research content supplied by the evaluation harness. "
        ) * 4,
    },
    "filesystem.list": {
        "files": ["tests/test_example.py", "tests/test_other.py"],
        "matches": ["tests/test_example.py", "tests/test_other.py"],
        "count": 2,
    },
    "filesystem.search": {
        "matches": ["tests/test_example.py"],
        "count": 1,
    },
}


class DeterministicToolRouter:
    """Wrap a real tool router: real metadata, deterministic results.

    Parameters
    ----------
    inner:
        The real router whose registry/permission engine are used for validation
        and authorization. Its ``execute`` is never called for a tool whose name
        appears in :attr:`canned` or :attr:`failing`.
        canned:
        Optional mapping of ``tool name -> output`` (a ``dict``) returned as a
        successful :class:`ToolResult`.
    failing:
        Optional mapping of ``tool name -> error message`` returned as a failed
        :class:`ToolResult`. Its presence is how a failure is simulated honestly.
    """

    def __init__(
        self,
        inner: Any,
        *,
        canned: dict[str, Any] | None = None,
        failing: dict[str, str] | None = None,
        default_output: Any | None = None,
    ) -> None:
        self._inner = inner
        canned = dict(canned or {})
        # Fill in realistic defaults for the read tools the evaluation exercises.
        # A tool that returned an opaque payload would make retrieval look empty
        # and turn a routing case into a false failure; supplying representative
        # data is what lets the *routing* be measured on its own.
        for name, output in _DEFAULT_TOOL_OUTPUTS.items():
            canned.setdefault(name, output)
        self._canned = canned
        self._failing = dict(failing or {})
        self._default_output = default_output
        self.calls: list[tuple[str, dict[str, Any]]] = []

    # -- registry pass-through (validation stays real) -------------------------

    @property
    def _registry(self) -> Any:
        return getattr(self._inner, "_registry", None)

    def __getattr__(self, name: str) -> Any:
        # Any attribute the planner/validator needs but that is not overridden
        # (e.g. ``_registry``, ``_permission_engine``) comes from the real router.
        # ``__getattr__`` is only called for attributes normal lookup missed, but
        # guard against re-entrancy when ``_inner`` is not yet assigned.
        if name.startswith("__") or name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    # -- execution -------------------------------------------------------------

    def execute(self, name: str, parameters: dict[str, Any], *, approved: bool = False) -> ToolResult:
        self.calls.append((name, dict(parameters)))
        if name in self._failing:
            return ToolResult.failure(self._failing[name])
        if name in self._canned:
            return ToolResult(success=True, status="completed", output=self._canned[name])
        # A canned-but-unspecified tool succeeds with a neutral payload. This
        # keeps a multi-step plan moving without pretending a real effect.
        if self._default_output is not None:
            return ToolResult(success=True, status="completed", output=self._default_output)
        return ToolResult(success=True, status="completed", output={"ok": True, "tool": name})

    def executed_tools(self) -> list[str]:
        return [name for name, _ in self.calls]
