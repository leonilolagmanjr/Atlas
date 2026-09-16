"""Bounded, read-only filesystem tools."""

from __future__ import annotations

import fnmatch
import shutil
from pathlib import Path
from typing import Any

from tools.base import PermissionLevel, RiskLevel, Tool, ToolMetadata, ToolResult


class _BoundedFilesystemTool(Tool):
    def __init__(self, *, root: Path | None = None) -> None:
        self._root = (root or Path.cwd()).expanduser().resolve()

    def _path(self, value: str) -> Path:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = self._root / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self._root)
        except ValueError as exc:
            raise ValueError(f"Path is outside the allowed root: {self._root}") from exc
        return resolved


class FilesystemListTool(_BoundedFilesystemTool):
    metadata = ToolMetadata(
        name="filesystem.list",
        description="List entries in an allowed directory.",
        category="computer.filesystem",
        input_schema={"path": {"type": "string"}},
        output_schema={"entries": {"type": "array"}},
        permission_level=PermissionLevel.READ_ONLY,
        risk_level=RiskLevel.READ_ONLY,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            directory = self._path(parameters.get("path", "."))
            if not directory.is_dir():
                return ToolResult.failure(f"Directory not found: {directory}")
            entries = [
                {
                    "name": entry.name,
                    "path": str(entry),
                    "kind": "directory" if entry.is_dir() else "file",
                }
                for entry in sorted(directory.iterdir(), key=lambda item: item.name.lower())
            ]
            return ToolResult(success=True, status="completed", output={"entries": entries})
        except (OSError, ValueError) as exc:
            return ToolResult.failure(str(exc), recoverable=True)


class FilesystemReadTool(_BoundedFilesystemTool):
    metadata = ToolMetadata(
        name="filesystem.read",
        description="Read a UTF-8 text file under the allowed root.",
        category="computer.filesystem",
        input_schema={"path": {"type": "string"}, "max_bytes": {"type": "integer"}},
        output_schema={"path": {"type": "string"}, "content": {"type": "string"}},
        permission_level=PermissionLevel.READ_ONLY,
        risk_level=RiskLevel.READ_ONLY,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            path = self._path(parameters["path"])
            max_bytes = min(int(parameters.get("max_bytes", 1_000_000)), 5_000_000)
            if not path.is_file():
                return ToolResult.failure(f"File not found: {path}")
            content = path.read_text(encoding="utf-8")
            truncated = len(content.encode("utf-8")) > max_bytes
            if truncated:
                content = content.encode("utf-8")[:max_bytes].decode("utf-8", errors="replace")
            return ToolResult(
                success=True,
                status="completed",
                output={"path": str(path), "content": content, "truncated": truncated},
            )
        except (KeyError, OSError, ValueError, UnicodeError) as exc:
            return ToolResult.failure(str(exc), recoverable=True)


class FilesystemMetadataTool(_BoundedFilesystemTool):
    metadata = ToolMetadata(
        name="filesystem.metadata",
        description="Inspect metadata for an allowed filesystem path.",
        category="computer.filesystem",
        input_schema={"path": {"type": "string"}},
        output_schema={"path": {"type": "string"}, "size": {"type": "integer"}},
        permission_level=PermissionLevel.READ_ONLY,
        risk_level=RiskLevel.READ_ONLY,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            path = self._path(parameters["path"])
            stats = path.stat()
            return ToolResult(
                success=True,
                status="completed",
                output={
                    "path": str(path),
                    "kind": "directory" if path.is_dir() else "file",
                    "size": stats.st_size,
                    "modified": stats.st_mtime,
                },
            )
        except (KeyError, OSError, ValueError) as exc:
            return ToolResult.failure(str(exc), recoverable=True)


class FilesystemSearchTool(_BoundedFilesystemTool):
    metadata = ToolMetadata(
        name="filesystem.search",
        description="Search filenames below the allowed root without modifying files.",
        category="computer.filesystem",
        input_schema={
            "pattern": {"type": "string", "description": "filename glob, e.g. *.pdf"},
            "path": {"type": "string", "description": "optional subfolder to search"},
            "max_results": {"type": "integer"},
        },
        output_schema={"matches": {"type": "array"}},
        permission_level=PermissionLevel.READ_ONLY,
        risk_level=RiskLevel.READ_ONLY,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            pattern = str(parameters["pattern"])
            max_results = min(max(int(parameters.get("max_results", 100)), 1), 500)
            # "search my Downloads folder" scopes the walk; otherwise the whole
            # allowed root is searched. The path is bounded by _path().
            base = self._path(str(parameters["path"])) if parameters.get("path") else self._root
            if not base.is_dir():
                return ToolResult.failure(f"Directory not found: {base}", recoverable=True)
            matches: list[str] = []
            for path in base.rglob("*"):
                if fnmatch.fnmatch(path.name, pattern):
                    matches.append(str(path))
                    if len(matches) >= max_results:
                        break
            return ToolResult(
                success=True,
                status="completed",
                output={"matches": matches, "truncated": len(matches) >= max_results},
            )
        except (KeyError, OSError, ValueError) as exc:
            return ToolResult.failure(str(exc), recoverable=True)
class FilesystemWriteTool(_BoundedFilesystemTool):
    """Permission-gated UTF-8 text write under the allowed root."""

    metadata = ToolMetadata(
        name="filesystem.write",
        description="Write UTF-8 text to a file under the allowed root.",
        category="computer.filesystem",
        input_schema={
            "path": {"type": "string", "description": "destination file path"},
            "text": {"type": "string", "description": "text to write"},
            "overwrite": {"type": "boolean", "description": "replace an existing file"},
        },
        output_schema={"path": {"type": "string"}, "bytes": {"type": "integer"}},
        permission_level=PermissionLevel.MEDIUM_RISK,
        risk_level=RiskLevel.MEDIUM,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            path = self._path(str(parameters["path"]))
            text = str(parameters["text"])
            overwrite = bool(parameters.get("overwrite", False))
            if path.exists() and not overwrite:
                return ToolResult.failure(
                    f"File already exists: {path}. Pass overwrite=true to replace it.",
                    recoverable=True,
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            return ToolResult(
                success=True,
                status="completed",
                output={"path": str(path), "bytes": len(text.encode("utf-8"))},
            )
        except (KeyError, OSError, ValueError) as exc:
            return ToolResult.failure(str(exc), recoverable=True)


class FilesystemCreateFolderTool(_BoundedFilesystemTool):
    """Permission-gated directory creation under the allowed root."""

    metadata = ToolMetadata(
        name="filesystem.create_folder",
        description="Create a directory under the allowed root.",
        category="computer.filesystem",
        input_schema={"path": {"type": "string", "description": "directory to create"}},
        output_schema={"path": {"type": "string"}, "created": {"type": "boolean"}},
        permission_level=PermissionLevel.MEDIUM_RISK,
        risk_level=RiskLevel.MEDIUM,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            path = self._path(str(parameters["path"]))
            existed = path.exists()
            path.mkdir(parents=True, exist_ok=True)
            return ToolResult(
                success=True,
                status="completed",
                output={"path": str(path), "created": not existed},
            )
        except (KeyError, OSError, ValueError) as exc:
            return ToolResult.failure(str(exc), recoverable=True)


class FilesystemMoveTool(_BoundedFilesystemTool):
    """Permission-gated move/rename under the allowed root."""

    metadata = ToolMetadata(
        name="filesystem.move",
        description="Move or rename a file or directory under the allowed root.",
        category="computer.filesystem",
        input_schema={
            "source": {"type": "string"},
            "destination": {"type": "string"},
            "overwrite": {"type": "boolean"},
        },
        output_schema={"source": {"type": "string"}, "destination": {"type": "string"}},
        permission_level=PermissionLevel.MEDIUM_RISK,
        risk_level=RiskLevel.MEDIUM,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            source = self._path(str(parameters["source"]))
            destination = self._path(str(parameters["destination"]))
            if not source.exists():
                return ToolResult.failure(f"Source not found: {source}", recoverable=True)
            if destination.exists() and not bool(parameters.get("overwrite", False)):
                return ToolResult.failure(
                    f"Destination already exists: {destination}", recoverable=True
                )
            # A destination directory means "move into it", preserving the name.
            if destination.is_dir():
                destination = destination / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
            return ToolResult(
                success=True,
                status="completed",
                output={"source": str(source), "destination": str(destination)},
            )
        except (KeyError, OSError, ValueError) as exc:
            return ToolResult.failure(str(exc), recoverable=True)


class FilesystemCopyTool(_BoundedFilesystemTool):
    """Permission-gated copy under the allowed root."""

    metadata = ToolMetadata(
        name="filesystem.copy",
        description="Copy a file or directory under the allowed root.",
        category="computer.filesystem",
        input_schema={
            "source": {"type": "string"},
            "destination": {"type": "string"},
            "overwrite": {"type": "boolean"},
        },
        output_schema={"source": {"type": "string"}, "destination": {"type": "string"}},
        permission_level=PermissionLevel.MEDIUM_RISK,
        risk_level=RiskLevel.MEDIUM,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            source = self._path(str(parameters["source"]))
            destination = self._path(str(parameters["destination"]))
            if not source.exists():
                return ToolResult.failure(f"Source not found: {source}", recoverable=True)
            if source.is_dir():
                return ToolResult.failure("Directory copy is not supported", recoverable=False)
            if destination.is_dir():
                destination = destination / source.name
            if destination.exists() and not bool(parameters.get("overwrite", False)):
                return ToolResult.failure(
                    f"Destination already exists: {destination}", recoverable=True
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            return ToolResult(
                success=True,
                status="completed",
                output={"source": str(source), "destination": str(destination)},
            )
        except (KeyError, OSError, ValueError, shutil.Error) as exc:
            return ToolResult.failure(str(exc), recoverable=True)
