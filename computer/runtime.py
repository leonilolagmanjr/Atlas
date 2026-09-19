"""Bootstrap registration for safe local computer capabilities."""

from __future__ import annotations

from pathlib import Path

from config import ENABLE_COMPUTER_OBSERVATION

from computer.applications import InstalledApplicationSearchTool, InstalledApplicationsTool
from computer.filesystem import (
    FilesystemCopyTool,
    FilesystemCreateFolderTool,
    FilesystemListTool,
    FilesystemMetadataTool,
    FilesystemMoveTool,
    FilesystemReadTool,
    FilesystemSearchContentTool,
    FilesystemSearchTool,
    FilesystemWriteTool,
)
from computer.launch import ApplicationLaunchTool, NamedApplicationLaunchTool
from computer.perception import ComputerObserveTool, ComputerObserver, OpenWindowsTool
from computer.powershell import PowerShellTool
from computer.processes import ProcessInspectTool, ProcessListTool
from computer.system import SystemInfoTool
from computer.text_entry import ApplicationTextEntryTool
from tools.content import ContentGenerationTool
from tools.format import ContentFormatTool
from tools.knowledge import ToolKnowledgeStore, load_json
from tools.registry import ToolRegistry
from web import WebFetchTool, WebResearchTool, WebSearchTool


def build_computer_observer() -> ComputerObserver | None:
    """Create the read-only observer the executor uses for its closed loop.

    Returns None when ``ENABLE_COMPUTER_OBSERVATION`` is false, which leaves the
    executor with tool-reported evidence only.
    """

    if not ENABLE_COMPUTER_OBSERVATION:
        return None
    return ComputerObserver()


def register_read_only_tools(
    registry: ToolRegistry,
    *,
    root: Path | None = None,
    ask: object | None = None,
) -> None:
    """Register the initial read-only computer tools into an existing registry."""

    filesystem_tools = [
        FilesystemListTool(root=root),
        FilesystemReadTool(root=root),
        FilesystemMetadataTool(root=root),
        FilesystemSearchTool(root=root),
        FilesystemSearchContentTool(root=root),
        FilesystemWriteTool(root=root),
        FilesystemCreateFolderTool(root=root),
        FilesystemMoveTool(root=root),
        FilesystemCopyTool(root=root),
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
        ApplicationTextEntryTool(),
        OpenWindowsTool(),
        ComputerObserveTool(),
        ContentGenerationTool(ask=ask),
        ContentFormatTool(),
        PowerShellTool(knowledge=knowledge),
        WebSearchTool(),
        WebFetchTool(),
        WebResearchTool(ask=ask),
    ]:
        registry.register(tool)

    # Validator-facing capability labels that read-only graph tools expose.
    # Keep these aligned with the CapabilityRegistry labels below so the
    # deterministic interpreter and reasoning engine never misclassify an
    # observation-only capability as a mutating action.
    #
    # ``tools/capabilities.py`` is the authoritative source: it exposes a
    # ``CapabilityRegistry`` wired to the actual tool metadata and a
    # ``PLANNABLE_CAPABILITIES`` set. ``register_read_only_tools`` never
    # mutates that module namespace; it only confirms the observation tools
    # present in the actual registry so the validator/interpreter see the same
    # read-only classification the capability catalog already declares.
    observation_labels = ("computer.observe", "computer.windows")

    registry_labels = {metadata.name for metadata in registry.list_metadata()}
    missing = [label for label in observation_labels if label not in registry_labels]
    if missing:
        raise RuntimeError(
            "Read-only computer observation tools must be registered before the "
            "capability validator can classify them. Missing: " + ", ".join(missing)
        )
