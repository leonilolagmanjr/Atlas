"""Permission-gated application launch capability."""

from __future__ import annotations

import subprocess
import os
import shutil
import sys
import re
from pathlib import Path
from typing import Any
from tools.base import PermissionLevel, RiskLevel, Tool, ToolMetadata, ToolResult
# Curated aliases mapping friendly names to real executables. This is a small
# interoperability table (not command recognition): users say "calculator" but
# the binary is `calc.exe`, "vscode" but the binary is `Code.exe`, and so on.
_APPLICATION_ALIASES: dict[str, tuple[str, ...]] = {
    "notepad": ("notepad.exe",),
    "calculator": ("calc.exe",),
    "calc": ("calc.exe",),
    "paint": ("mspaint.exe",),
    "explorer": ("explorer.exe",),
    "file explorer": ("explorer.exe",),
    "task manager": ("Taskmgr.exe",),
    "command prompt": ("cmd.exe",),
    "cmd": ("cmd.exe",),
    "terminal": ("wt.exe", "powershell.exe"),
    "powershell": ("powershell.exe",),
    "windows terminal": ("wt.exe",),
    "vscode": ("Code.exe",),
    "vs code": ("Code.exe",),
    "visual studio code": ("Code.exe",),
    "chrome": ("chrome.exe",),
    "google chrome": ("chrome.exe",),
    "edge": ("msedge.exe",),
    "microsoft edge": ("msedge.exe",),
    "firefox": ("firefox.exe",),
    "word": ("WINWORD.EXE",),
    "excel": ("EXCEL.EXE",),
    "powerpoint": ("POWERPNT.EXE",),
    "outlook": ("OUTLOOK.EXE",),
    "spotify": ("Spotify.exe",),
    "discord": ("Discord.exe",),
    "steam": ("steam.exe",),
    "vlc": ("vlc.exe",),
    "settings": ("SystemSettings.exe",),
}

def _windows_search_dirs() -> list[Path]:
    system_root = Path(os.environ.get("SystemRoot", "C:\\Windows"))
    return [system_root / "System32", system_root]

def _find_executable_below(root: Path, executable_name: str, *, max_depth: int = 3) -> list[Path]:
    # Shallow, case-insensitive search for an executable under a root.
    if not root.is_dir():
        return []
    matches: list[Path] = []
    target = executable_name.casefold()
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_file() and entry.name.casefold() == target:
                    matches.append(entry)
                elif entry.is_dir() and depth < max_depth:
                    stack.append((entry, depth + 1))
            except OSError:
                continue
    return matches

def _alias_executables(normalized: str) -> list[str]:
    # Return executable names for a friendly application name.
    aliases = list(_APPLICATION_ALIASES.get(normalized, ()))
    # Also match "notepad app", "the calculator", etc.
    for key, names in _APPLICATION_ALIASES.items():
        if key and key in normalized:
            aliases.extend(names)
    return aliases

def resolve_application_name(name: str) -> Path | None:
    """Resolve an installed application name to a concrete executable."""

    normalized = name.strip().casefold()
    candidates: list[Path] = []
    local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    program_files = Path(os.environ.get("ProgramFiles", "C:\\Program Files"))
    executable_name = normalized if normalized.endswith(".exe") else f"{normalized}.exe"

    # 1. Resolve through the alias table (friendly name -> real binary).
    for alias in _alias_executables(normalized):
        for directory in _windows_search_dirs():
            alias_candidate = directory / alias
            if alias_candidate.is_file():
                candidates.append(alias_candidate)
        located = shutil.which(alias)
        if located:
            candidates.append(Path(located))
        # Per-user installs (VS Code, etc.) live under %LOCALAPPDATA%\Programs.
        candidates.extend(_find_executable_below(local_app_data / "Programs", alias))

    # 2. Resolve directly by name via PATH and Windows system directories.
    executable = shutil.which(executable_name) or shutil.which(name)
    if executable:
        candidates.append(Path(executable))
    for directory in _windows_search_dirs():
        direct_candidate = directory / executable_name
        if direct_candidate.is_file():
            candidates.append(direct_candidate)

    candidates.extend([
        local_app_data / normalized / executable_name,
        program_files / normalized / executable_name,
        Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")) / normalized / executable_name,
    ])
    candidates.extend(_registry_application_executables(normalized))

    # Portable and per-user applications often install under a directory whose
    # name matches the request. Keep the scan shallow and file-type restricted.
    for root in (local_app_data, program_files):
        if not root.is_dir():
            continue
        for directory in root.iterdir():
            if normalized not in directory.name.casefold():
                continue
            try:
                candidates.extend(directory.glob("*.exe"))
                candidates.extend(directory.glob("*/*.exe"))
            except OSError:
                continue

    name_tokens = {token for token in re.findall(r"[a-z0-9]+", normalized) if len(token) >= 3}
    candidates.sort(
        key=lambda candidate: 0
        if candidate.stem.casefold() in name_tokens or candidate.stem.casefold() == normalized
        else 1
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.casefold() == ".exe":
            return candidate.resolve()
    return None


def _registry_application_executables(normalized_name: str) -> list[Path]:
    if sys.platform != "win32":
        return []
    try:
        import winreg
    except ImportError:
        return []

    locations = (
        (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_LOCAL_MACHINE, r"Software\Wow6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
    )
    candidates: list[Path] = []
    for root, path in locations:
        try:
            with winreg.OpenKey(root, path) as uninstall_key:
                for index in range(winreg.QueryInfoKey(uninstall_key)[0]):
                    try:
                        with winreg.OpenKey(uninstall_key, winreg.EnumKey(uninstall_key, index)) as app_key:
                            display_name = str(_registry_value(app_key, winreg, "DisplayName") or "").casefold()
                            install_location_value = str(_registry_value(app_key, winreg, "InstallLocation") or "").strip()
                            install_location = Path(install_location_value) if install_location_value else None
                            display_icon = str(_registry_value(app_key, winreg, "DisplayIcon") or "")
                            if normalized_name not in display_name and (install_location is None or normalized_name not in install_location.name.casefold()):
                                continue
                            icon_path = Path(re.split(r",\s*-?\d+$", display_icon.strip('"'), maxsplit=1)[0])
                            if icon_path.suffix.casefold() == ".exe":
                                candidates.append(icon_path)
                            if install_location is not None and install_location.is_dir():
                                candidates.extend(install_location.glob("*.exe"))
                                candidates.extend(install_location.glob("*/*.exe"))
                    except OSError:
                        continue
        except OSError:
            continue
    return candidates


def _registry_value(key: Any, winreg: Any, name: str) -> Any:
    try:
        return winreg.QueryValueEx(key, name)[0]
    except OSError:
        return None


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


class NamedApplicationLaunchTool(Tool):
    metadata = ToolMetadata(
        name="applications.launch_named",
        description="Resolve and launch a trusted installed Windows application by name.",
        category="computer.applications",
        input_schema={"application": {"type": "string"}},
        output_schema={"pid": {"type": "integer"}, "application": {"type": "string"}, "executable": {"type": "string"}},
        permission_level=PermissionLevel.LOW_RISK,
        risk_level=RiskLevel.LOW,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            application = str(parameters["application"]).strip()
            if not application:
                return ToolResult.failure("Application name cannot be empty")
            executable = resolve_application_name(application)
            if executable is None:
                return ToolResult.failure(f"Could not resolve a trusted executable for: {application}", recoverable=True)
            process = subprocess.Popen(
                [str(executable)],
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return ToolResult(
                success=True,
                status="started",
                output={"pid": process.pid, "application": application, "executable": str(executable)},
            )
        except (KeyError, OSError, ValueError, subprocess.SubprocessError) as exc:
            return ToolResult.failure(str(exc), recoverable=True)
