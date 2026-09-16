"""Validated, read-only PowerShell execution for documented tool commands."""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from tools.base import PermissionLevel, RiskLevel, Tool, ToolMetadata, ToolResult
from tools.knowledge import ToolKnowledgeStore


DANGEROUS_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(remove-item|del|erase|rd)\b", "file deletion"),
    (r"\b(stop-process|stop-service|restart-service|restart-computer|stop-computer)\b", "process or service control"),
    (r"\b(set-executionpolicy|set-service|new-service|sc\.exe)\b", "system configuration"),
    (r"\b(reg(?:\.exe)?|set-itemproperty|new-itemproperty)\b", "registry modification"),
    (r"\b(netsh|set-netfirewallrule|new-netfirewallrule)\b", "firewall modification"),
    (r"\b(register-scheduledtask|unregister-scheduledtask)\b", "scheduled task modification"),
    (r"\b(new-localuser|remove-localuser|add-localgroupmember)\b", "account or permission change"),
    (r"\b(invoke-webrequest|wget|curl|start-bitstransfer)\b", "external download"),
    (r"\b(invoke-expression|iex|&\s*['\"]|start-process)\b", "arbitrary or executable launch"),
    (r"-(encodedcommand|enc)\b", "encoded command"),
)


@dataclass(frozen=True)
class PowerShellValidation:
    valid: bool
    risk_level: str
    reason: str
    command_name: str | None = None


class PowerShellValidator:
    """Conservative pre-execution validator independent from the LLM."""

    def __init__(self, knowledge: ToolKnowledgeStore) -> None:
        self._knowledge = knowledge

    def validate(self, command: str) -> PowerShellValidation:
        normalized = command.strip()
        if not normalized:
            return PowerShellValidation(False, "critical", "PowerShell command cannot be empty")
        if len(normalized) > 8_000:
            return PowerShellValidation(False, "high", "PowerShell command exceeds the size limit")
        if "`" in normalized or "\x00" in normalized:
            return PowerShellValidation(False, "high", "PowerShell escaping or null bytes are not allowed")
        if re.search(r"[<>;\n\r]|\$\(.*\)|\b(script|function)\b", normalized, re.IGNORECASE):
            return PowerShellValidation(False, "high", "redirection, command chaining, substitution, and script blocks are not allowed")

        for pattern, reason in DANGEROUS_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                return PowerShellValidation(False, "high", f"blocked operation: {reason}")

        pipeline_parts = [part.strip() for part in normalized.split("|")]
        if any(not part for part in pipeline_parts):
            return PowerShellValidation(False, "high", "PowerShell pipeline contains an empty command")

        command_names: list[str] = []
        for pipeline_part in pipeline_parts:
            command_name = re.match(r"^([A-Za-z][A-Za-z0-9-]*)", pipeline_part)
            if command_name is None:
                return PowerShellValidation(False, "high", "Each pipeline stage must start with a named PowerShell command")
            name = command_name.group(1)
            command_names.append(name)
            try:
                record = self._knowledge.get(name, tool_type="powershell")
            except KeyError:
                return PowerShellValidation(False, "high", f"Command is not in the trusted PowerShell knowledge base: {name}")
            if record.destructive or not record.read_only or record.risk_level not in {"safe", "low"}:
                return PowerShellValidation(False, record.risk_level, f"Command is not approved for read-only execution: {name}")

        if not command_names:
            return PowerShellValidation(False, "high", "Command must start with a named PowerShell command")
        return PowerShellValidation(True, "safe", "documented read-only pipeline", command_names[0])


class PowerShellTool(Tool):
    metadata = ToolMetadata(
        name="powershell.execute",
        description="Execute one validated, documented read-only PowerShell command.",
        category="computer.powershell",
        input_schema={"command": {"type": "string"}, "timeout_seconds": {"type": "number"}},
        output_schema={
            "success": {"type": "boolean"},
            "exit_code": {"type": "integer"},
            "stdout": {"type": "string"},
            "stderr": {"type": "string"},
            "duration_ms": {"type": "integer"},
        },
        permission_level=PermissionLevel.READ_ONLY,
        risk_level=RiskLevel.READ_ONLY,
    )

    def __init__(self, *, knowledge: ToolKnowledgeStore, default_timeout: float = 15.0) -> None:
        self._validator = PowerShellValidator(knowledge)
        self._default_timeout = default_timeout

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            command = str(parameters["command"])
            validation = self._validator.validate(command)
            if not validation.valid:
                return ToolResult(
                    success=False,
                    status="validation_failed",
                    error=validation.reason,
                    metadata={"risk_level": validation.risk_level},
                )
            timeout = min(max(float(parameters.get("timeout_seconds", self._default_timeout)), 1.0), 60.0)
            executable = shutil.which("powershell.exe") or shutil.which("pwsh")
            if executable is None:
                return ToolResult.failure("PowerShell executable was not found", recoverable=True)
            started = time.perf_counter()
            try:
                completed = subprocess.run(
                    [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout,
                    check=False,
                    shell=False,
                )
            except subprocess.TimeoutExpired as exc:
                duration_ms = int((time.perf_counter() - started) * 1000)
                return ToolResult(
                    success=False,
                    status="timeout",
                    error=f"PowerShell command exceeded {timeout:g} seconds",
                    recoverable=True,
                    metadata={"duration_ms": duration_ms, "stdout": exc.stdout or "", "stderr": exc.stderr or ""},
                )
            duration_ms = int((time.perf_counter() - started) * 1000)
            output = {
                "success": completed.returncode == 0,
                "exit_code": completed.returncode,
                "stdout": completed.stdout[:100_000],
                "stderr": completed.stderr[:20_000],
                "duration_ms": duration_ms,
                "command": command,
                "command_name": validation.command_name,
            }
            return ToolResult(
                success=completed.returncode == 0,
                status="completed" if completed.returncode == 0 else "failed",
                output=output,
                error=completed.stderr.strip() or None,
                recoverable=completed.returncode != 0,
            )
        except (KeyError, TypeError, ValueError, OSError, subprocess.SubprocessError) as exc:
            return ToolResult.failure(str(exc), recoverable=True)
