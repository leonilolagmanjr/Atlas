"""Conservative permission decisions for planned tool calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from tools.base import PermissionLevel, RiskLevel, ToolMetadata


class ExecutionMode(str, Enum):
    SAFE = "safe"
    CONFIRM = "confirm"
    AUTONOMOUS = "autonomous"


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    requires_confirmation: bool
    reason: str
    risk_level: RiskLevel


@dataclass
class PermissionEngine:
    mode: ExecutionMode = ExecutionMode.CONFIRM
    overrides: dict[PermissionLevel, bool] = field(default_factory=dict)

    def evaluate(self, metadata: ToolMetadata) -> PermissionDecision:
        level = metadata.permission_level
        if level in self.overrides:
            allowed = self.overrides[level]
            return PermissionDecision(
                allowed=allowed,
                requires_confirmation=False,
                reason="explicit policy override",
                risk_level=metadata.risk_level,
            )

        if level == PermissionLevel.CRITICAL:
            return PermissionDecision(False, False, "critical operations are denied by default", metadata.risk_level)

        if self.mode == ExecutionMode.SAFE:
            allowed = level == PermissionLevel.READ_ONLY
            return PermissionDecision(
                allowed=allowed,
                requires_confirmation=False,
                reason="safe mode permits read-only tools only",
                risk_level=metadata.risk_level,
            )

        if level == PermissionLevel.READ_ONLY:
            return PermissionDecision(True, False, "read-only operation", metadata.risk_level)

        if self.mode == ExecutionMode.AUTONOMOUS and level in {
            PermissionLevel.LOW_RISK,
            PermissionLevel.MEDIUM_RISK,
        }:
            return PermissionDecision(True, False, "operation permitted by autonomous mode", metadata.risk_level)

        return PermissionDecision(
            allowed=False,
            requires_confirmation=True,
            reason="user confirmation required for consequential operation",
            risk_level=metadata.risk_level,
        )
