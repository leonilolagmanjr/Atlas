"""Read-only Windows process inspection tools."""

from __future__ import annotations

import csv
import io
import subprocess
from typing import Any

from tools.base import PermissionLevel, RiskLevel, Tool, ToolMetadata, ToolResult


def _tasklist() -> list[dict[str, str]]:
    completed = subprocess.run(
        ["tasklist", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "tasklist failed")
    rows = csv.reader(io.StringIO(completed.stdout))
    return [
        {"image_name": row[0], "pid": row[1], "session_name": row[2], "memory": row[4]}
        for row in rows
        if len(row) >= 5
    ]


class ProcessListTool(Tool):
    metadata = ToolMetadata(
        name="processes.list",
        description="List currently running Windows processes.",
        category="computer.processes",
        output_schema={"processes": {"type": "array"}},
        permission_level=PermissionLevel.READ_ONLY,
        risk_level=RiskLevel.READ_ONLY,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            processes = _tasklist()
            return ToolResult(success=True, status="completed", output={"processes": processes})
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            return ToolResult.failure(str(exc), recoverable=True)


class ProcessInspectTool(Tool):
    metadata = ToolMetadata(
        name="processes.inspect",
        description="Check whether a named Windows process is running.",
        category="computer.processes",
        input_schema={"name": {"type": "string"}},
        output_schema={"found": {"type": "boolean"}, "processes": {"type": "array"}},
        permission_level=PermissionLevel.READ_ONLY,
        risk_level=RiskLevel.READ_ONLY,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            name = str(parameters["name"]).strip().lower()
            if not name:
                return ToolResult.failure("Process name cannot be empty")
            processes = [process for process in _tasklist() if process["image_name"].lower() == name]
            return ToolResult(
                success=True,
                status="completed",
                output={"found": bool(processes), "processes": processes},
            )
        except (KeyError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
            return ToolResult.failure(str(exc), recoverable=True)
