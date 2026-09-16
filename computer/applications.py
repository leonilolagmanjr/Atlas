"""Read-only Windows application inventory tools."""

from __future__ import annotations

import sys
from typing import Any

from tools.base import PermissionLevel, RiskLevel, Tool, ToolMetadata, ToolResult


_UNINSTALL_PATHS = (
    ("HKEY_LOCAL_MACHINE", r"Software\Microsoft\Windows\CurrentVersion\Uninstall"),
    ("HKEY_LOCAL_MACHINE", r"Software\Wow6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
    ("HKEY_CURRENT_USER", r"Software\Microsoft\Windows\CurrentVersion\Uninstall"),
)


def _read_installed_applications() -> list[dict[str, str]]:
    if sys.platform != "win32":
        raise RuntimeError("Installed application inspection requires Windows")

    import winreg

    roots = {
        "HKEY_LOCAL_MACHINE": winreg.HKEY_LOCAL_MACHINE,
        "HKEY_CURRENT_USER": winreg.HKEY_CURRENT_USER,
    }
    applications: dict[tuple[str, str], dict[str, str]] = {}
    for root_name, subkey_path in _UNINSTALL_PATHS:
        root = roots[root_name]
        try:
            with winreg.OpenKey(root, subkey_path) as uninstall_key:
                count = winreg.QueryInfoKey(uninstall_key)[0]
                for index in range(count):
                    try:
                        key_name = winreg.EnumKey(uninstall_key, index)
                        with winreg.OpenKey(uninstall_key, key_name) as app_key:
                            name = str(_value(app_key, winreg, "DisplayName") or "").strip()
                            if not name:
                                continue
                            version = str(_value(app_key, winreg, "DisplayVersion") or "").strip()
                            publisher = str(_value(app_key, winreg, "Publisher") or "").strip()
                            install_location = str(_value(app_key, winreg, "InstallLocation") or "").strip()
                            identity = (name.casefold(), version.casefold())
                            applications[identity] = {
                                "name": name,
                                "version": version,
                                "publisher": publisher,
                                "install_location": install_location,
                            }
                    except OSError:
                        continue
        except OSError:
            continue

    return sorted(applications.values(), key=lambda item: item["name"].casefold())


def _value(key: Any, winreg: Any, name: str) -> Any:
    try:
        return winreg.QueryValueEx(key, name)[0]
    except OSError:
        return None


class InstalledApplicationsTool(Tool):
    metadata = ToolMetadata(
        name="applications.list_installed",
        description="List installed Windows applications from registered uninstall entries.",
        category="computer.applications",
        output_schema={"applications": {"type": "array"}},
        permission_level=PermissionLevel.READ_ONLY,
        risk_level=RiskLevel.READ_ONLY,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            applications = _read_installed_applications()
            return ToolResult(
                success=True,
                status="completed",
                output={"applications": applications},
            )
        except (OSError, RuntimeError) as exc:
            return ToolResult.failure(str(exc), recoverable=True)


class InstalledApplicationSearchTool(Tool):
    metadata = ToolMetadata(
        name="applications.search_installed",
        description="Find installed Windows applications by name.",
        category="computer.applications",
        input_schema={"query": {"type": "string"}, "max_results": {"type": "integer"}},
        output_schema={"applications": {"type": "array"}},
        permission_level=PermissionLevel.READ_ONLY,
        risk_level=RiskLevel.READ_ONLY,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            query = str(parameters["query"]).strip().casefold()
            if not query:
                return ToolResult.failure("Application query cannot be empty")
            max_results = min(max(int(parameters.get("max_results", 25)), 1), 100)
            matches = [
                application
                for application in _read_installed_applications()
                if query in application["name"].casefold()
            ][:max_results]
            return ToolResult(
                success=True,
                status="completed",
                output={"applications": matches},
            )
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            return ToolResult.failure(str(exc), recoverable=True)
