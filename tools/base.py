"""Stable contracts for Atlas tools.

Tools are intentionally capability-neutral. Runtime integrations can implement
this contract without giving the planner direct access to subprocesses or APIs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PermissionLevel(str, Enum):
    READ_ONLY = "read_only"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    CRITICAL = "critical"


class RiskLevel(str, Enum):
    READ_ONLY = PermissionLevel.READ_ONLY.value
    LOW = PermissionLevel.LOW_RISK.value
    MEDIUM = PermissionLevel.MEDIUM_RISK.value
    HIGH = PermissionLevel.HIGH_RISK.value
    CRITICAL = PermissionLevel.CRITICAL.value


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    description: str
    category: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    permission_level: PermissionLevel = PermissionLevel.READ_ONLY
    risk_level: RiskLevel = RiskLevel.READ_ONLY


@dataclass
class ToolResult:
    success: bool
    status: str
    output: Any = None
    error: str | None = None
    recoverable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def failure(cls, error: str, *, recoverable: bool = False) -> "ToolResult":
        return cls(
            success=False,
            status="failed",
            error=error,
            recoverable=recoverable,
        )


class Tool(ABC):
    """A validated capability that can be discovered and invoked by Atlas."""

    metadata: ToolMetadata

    def validate(self, parameters: dict[str, Any]) -> None:
        if not isinstance(parameters, dict):
            raise TypeError("Tool parameters must be a dictionary")

    @abstractmethod
    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        """Execute validated parameters in the tool's own runtime boundary."""
        raise NotImplementedError
