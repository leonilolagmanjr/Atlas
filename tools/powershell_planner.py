"""Deterministic PowerShell task selection backed by tool knowledge."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tools.discovery import ToolCandidate, ToolDiscovery
from tools.knowledge import ToolKnowledgeStore, load_json


def _has_word(text: str, word: str) -> bool:
    """Return True when ``word`` appears as a whole word in ``text``.

    Substring matching is wrong here: "program" contains "ram" and
    "serviceable" contains "service", so a plain ``in`` test routes unrelated
    requests to the memory or services command.
    """

    return re.search(rf"\b{re.escape(word)}\b", text) is not None


@dataclass(frozen=True)
class PowerShellTaskPlan:
    command: str
    candidates: tuple[ToolCandidate, ...]
    capability: str


class PowerShellTaskPlanner:
    def __init__(self, store: ToolKnowledgeStore | None = None) -> None:
        if store is None:
            store = ToolKnowledgeStore(load_json(Path(__file__).with_name("powershell_commands.json")))
        self._discovery = ToolDiscovery(store)

    def create(self, request: str) -> PowerShellTaskPlan | None:
        normalized = request.casefold()
        if _has_word(normalized, "ram") or _has_word(normalized, "memory") or any(
            term in normalized for term in ("process", "application using")
        ):
            capability = "inspect memory usage"
            command = "Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 10 Name, Id, WorkingSet64 | ConvertTo-Json -Compress"
        elif _has_word(normalized, "service"):
            capability = "inspect services"
            command = "Get-Service | ConvertTo-Json -Compress"
        elif any(term in normalized for term in ("network connection", "tcp connection", "active connection")):
            capability = "inspect active network connections"
            command = "Get-NetTCPConnection | ConvertTo-Json -Compress"
        elif any(term in normalized for term in ("windows version", "computer information", "system information")):
            capability = "inspect Windows version"
            command = "Get-ComputerInfo | Select-Object WindowsProductName, WindowsVersion, OsBuildNumber | ConvertTo-Json -Compress"
        else:
            return None

        candidates = tuple(self._discovery.discover(capability, tool_type="powershell"))
        if not candidates:
            return None
        selected_name = candidates[0].tool
        if not command.casefold().startswith(selected_name.casefold()):
            return None
        return PowerShellTaskPlan(command=command, candidates=candidates, capability=capability)
