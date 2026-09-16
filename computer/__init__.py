"""Read-only local computer capabilities for Atlas."""

from computer.applications import InstalledApplicationSearchTool, InstalledApplicationsTool
from computer.launch import ApplicationLaunchTool
from computer.filesystem import (
    FilesystemListTool,
    FilesystemMetadataTool,
    FilesystemReadTool,
    FilesystemSearchTool,
)
from computer.processes import ProcessInspectTool, ProcessListTool
from computer.runtime import register_read_only_tools
from computer.system import SystemInfoTool

__all__ = [
    "InstalledApplicationSearchTool",
    "InstalledApplicationsTool",
    "ApplicationLaunchTool",
    "FilesystemListTool",
    "FilesystemMetadataTool",
    "FilesystemReadTool",
    "FilesystemSearchTool",
    "ProcessInspectTool",
    "ProcessListTool",
    "register_read_only_tools",
    "SystemInfoTool",
]
