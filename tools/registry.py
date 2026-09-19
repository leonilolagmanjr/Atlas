"""Tool discovery and registration."""

from __future__ import annotations

from tools.base import Tool


class ToolRegistry:
    """Store tools by stable name and reject ambiguous registrations."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        name = tool.metadata.name.strip()
        if not name:
            raise ValueError("Tool name cannot be empty")
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def list_metadata(self) -> list[object]:
        return [tool.metadata for tool in self._tools.values()]

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())
