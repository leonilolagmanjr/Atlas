"""Read-only local computer capabilities for Atlas."""

from computer.applications import InstalledApplicationSearchTool, InstalledApplicationsTool
from computer.launch import ApplicationLaunchTool
from computer.filesystem import (
    FilesystemListTool,
    FilesystemMetadataTool,
    FilesystemReadTool,
    FilesystemSearchContentTool,
    FilesystemSearchTool,
)
from computer.processes import ProcessInspectTool, ProcessListTool
from computer.perception import (
    ComputerObserveTool,
    ComputerObserver,
    OpenWindowsTool,
)
from computer.runtime import register_read_only_tools
from computer.system import SystemInfoTool
from computer.text_entry import ApplicationTextEntryTool

__all__ = [
    "ApplicationLaunchTool",
    "ApplicationTextEntryTool",
    "ComputerObserveTool",
    "ComputerObserver",
    "InstalledApplicationSearchTool",
    "InstalledApplicationsTool",
    "OpenWindowsTool",
    "register_read_only_tools",
    "SystemInfoTool",
    "FilesystemListTool",
    "FilesystemMetadataTool",
    "FilesystemReadTool",
    "FilesystemSearchTool",
    "FilesystemSearchContentTool",
    "ProcessInspectTool",
    "ProcessListTool",
]
