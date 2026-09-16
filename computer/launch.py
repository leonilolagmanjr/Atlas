"""Permission-gated application launch capability."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from tools.base import PermissionLevel, RiskLevel, Tool, ToolMetadata, ToolResult


class ApplicationLaunchTool(Tool):
    metadata = ToolMetadata(
        name="applications.launch",
        description="Launch one explicitly selected Windows executable.",
        category="computer.applications",
        input_schema={
            "executable": {"type": "string"},
            "arguments": {"type": "array", "items": {"type": "string"}},
        },
        output_schema={"pid": {"type": "integer"}, "executable": {"type": "string"}},
        permission_level=PermissionLevel.LOW_RISK,
        risk_level=RiskLevel.LOW,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            executable = Path(str(parameters["executable"])).expanduser().resolve()
            arguments = parameters.get("arguments", [])
            if not executable.is_file():
                return ToolResult.failure(f"Executable not found: {executable}")
            if executable.suffix.casefold() != ".exe":
                return ToolResult.failure("Application launch only accepts .exe files")
            if not isinstance(arguments, list) or not all(isinstance(item, str) for item in arguments):
                return ToolResult.failure("Application arguments must be a list of strings")

            process = subprocess.Popen(
                [str(executable), *arguments],
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return ToolResult(
                success=True,
                status="started",
                output={"pid": process.pid, "executable": str(executable)},
            )
        except (KeyError, OSError, ValueError, subprocess.SubprocessError) as exc:
            return ToolResult.failure(str(exc), recoverable=True)
