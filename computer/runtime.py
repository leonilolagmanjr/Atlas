"""Bootstrap registration for safe local computer capabilities."""

from __future__ import annotations

from pathlib import Path

from computer.applications import InstalledApplicationSearchTool, InstalledApplicationsTool
from computer.filesystem import (
    FilesystemListTool,
    FilesystemMetadataTool,
    FilesystemReadTool,
    FilesystemSearchTool,
)
from computer.launch import ApplicationLaunchTool
from computer.processes import ProcessInspectTool, ProcessListTool
from computer.system import SystemInfoTool
from tools.registry import ToolRegistry


def register_read_only_tools(registry: ToolRegistry, *, root: Path | None = None) -> None:
    """Register the initial read-only computer tools into an existing registry."""

    filesystem_tools = [
        FilesystemListTool(root=root),
        FilesystemReadTool(root=root),
        FilesystemMetadataTool(root=root),
        FilesystemSearchTool(root=root),
    ]
    for tool in [
        *filesystem_tools,
        ProcessListTool(),
        ProcessInspectTool(),
        SystemInfoTool(),
        InstalledApplicationsTool(),
        InstalledApplicationSearchTool(),
        ApplicationLaunchTool(),
    ]:
        registry.register(tool)
