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
from computer.launch import ApplicationLaunchTool, NamedApplicationLaunchTool
from computer.powershell import PowerShellTool
from computer.processes import ProcessInspectTool, ProcessListTool
from computer.system import SystemInfoTool
from tools.knowledge import ToolKnowledgeStore, load_json
from tools.registry import ToolRegistry


def register_read_only_tools(registry: ToolRegistry, *, root: Path | None = None) -> None:
    """Register the initial read-only computer tools into an existing registry."""

    filesystem_tools = [
        FilesystemListTool(root=root),
        FilesystemReadTool(root=root),
        FilesystemMetadataTool(root=root),
        FilesystemSearchTool(root=root),
    ]
    knowledge = ToolKnowledgeStore(load_json(Path(__file__).parent.parent / "tools" / "powershell_commands.json"))
    for tool in [
        *filesystem_tools,
        ProcessListTool(),
        ProcessInspectTool(),
        SystemInfoTool(),
        InstalledApplicationsTool(),
        InstalledApplicationSearchTool(),
        ApplicationLaunchTool(),
        NamedApplicationLaunchTool(),
        PowerShellTool(knowledge=knowledge),
    ]:
        registry.register(tool)
