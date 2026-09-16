"""Read-only operating-system and disk information."""

from __future__ import annotations

import os
import platform
import shutil
from typing import Any

from tools.base import PermissionLevel, RiskLevel, Tool, ToolMetadata, ToolResult


class SystemInfoTool(Tool):
    metadata = ToolMetadata(
        name="system.info",
        description="Inspect operating-system, CPU, memory, and disk information.",
        category="computer.system",
        input_schema={"path": {"type": "string"}},
        output_schema={"system": {"type": "object"}},
        permission_level=PermissionLevel.READ_ONLY,
        risk_level=RiskLevel.READ_ONLY,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            path = parameters.get("path") or os.environ.get("SystemDrive", "C:\\")
            usage = shutil.disk_usage(path)
            return ToolResult(
                success=True,
                status="completed",
                output={
                    "system": {
                        "os": platform.system(),
                        "release": platform.release(),
                        "version": platform.version(),
                        "machine": platform.machine(),
                        "processor": platform.processor(),
                        "python": platform.python_version(),
                        "cpu_count": os.cpu_count(),
                        "disk": {
                            "path": path,
                            "total": usage.total,
                            "used": usage.used,
                            "free": usage.free,
                        },
                    }
                },
            )
        except (OSError, ValueError) as exc:
            return ToolResult.failure(str(exc), recoverable=True)
