"""Permission-aware dispatch for registered tools."""

from __future__ import annotations

from typing import Any

from tools.base import ToolResult
from tools.permissions import PermissionEngine
from tools.registry import ToolRegistry


class ToolRouter:
    """Validate, authorize, and dispatch tool calls."""

    def __init__(self, *, registry: ToolRegistry, permission_engine: PermissionEngine) -> None:
        self._registry = registry
        self._permission_engine = permission_engine

    def execute(self, name: str, parameters: dict[str, Any], *, approved: bool = False) -> ToolResult:
        try:
            tool = self._registry.get(name)
        except KeyError as exc:
            return ToolResult.failure(str(exc))

        try:
            tool.validate(parameters)
        except (TypeError, ValueError) as exc:
            return ToolResult.failure(str(exc))

        decision = self._permission_engine.evaluate(tool.metadata)
        if decision.requires_confirmation and not approved:
            return ToolResult(
                success=False,
                status="confirmation_required",
                error=decision.reason,
                metadata={
                    "tool": name,
                    "risk_level": decision.risk_level.value,
                    "requires_confirmation": True,
                },
            )
        if not decision.allowed and not (approved and decision.requires_confirmation):
            return ToolResult(
                success=False,
                status="denied",
                error=decision.reason,
                metadata={"tool": name, "risk_level": decision.risk_level.value},
            )

        return tool.execute(parameters)
