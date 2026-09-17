"""Bounded, read-only filesystem tools."""

from __future__ import annotations

import codecs
import fnmatch
import io
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterator

import config

from tools.base import PermissionLevel, RiskLevel, Tool, ToolMetadata, ToolResult


class _BoundedFilesystemTool(Tool):
    _MAX_WALK_ENTRIES = 10_000
    _MAX_WALK_SECONDS = 5.0
    _MAX_PDF_BYTES = 10_000_000
    _MAX_PDF_PAGES = 20
    _EXCLUDED_NAMES = frozenset({
        ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
        ".kilo", ".vscode", ".ssh", ".aws", ".azure", ".gnupg", ".env",
        "database", "memory", "runtime", "logs", "sessions", "credentials", "secrets",
    })

    def __init__(self, *, root: Path | None = None) -> None:
        self._root = (root if root is not None else config.COMPUTER_ROOT).expanduser().resolve()

    @staticmethod
    def _limit(parameters: dict[str, Any], name: str, default: int, cap: int) -> int:
        value = parameters.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be a positive integer")
        if value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return min(value, cap)

    def _excluded(self, path: Path) -> bool:
        parts = path.relative_to(self._root).parts
        if any(part.casefold() in self._EXCLUDED_NAMES or part.casefold().startswith(".env.") for part in parts):
            return True
        if path.suffix.casefold() in {".key", ".pem", ".pfx", ".p12", ".sqlite", ".sqlite3", ".db"}:
            return True
        for private_path in (config.DATABASE_FOLDER, config.MEMORY_FOLDER, config.TASK_STORE_FILE, config.LOG_FILE):
            if path.is_relative_to(Path(private_path).resolve()):
                return True
        return False

    def _walk(self, base: Path, state: dict[str, Any], *, recursive: bool = True) -> Iterator[Path]:
        pending = [base]
        visited: set[Path] = set()
        while pending:
            if state["walked"] >= self._MAX_WALK_ENTRIES or time.monotonic() >= state["deadline"]:
                state["truncated"] = True
                return
            directory = pending.pop()
            try:
                directory = self._path(str(directory))
                if directory in visited or self._excluded(directory):
                    continue
                visited.add(directory)
                with os.scandir(directory) as entries:
                    for entry in entries:
                        if state["walked"] >= self._MAX_WALK_ENTRIES or time.monotonic() >= state["deadline"]:
                            state["truncated"] = True
                            return
                        state["walked"] += 1
                        candidate = Path(entry.path)
                        if self._excluded(candidate):
                            continue
                        try:
                            resolved = self._path(str(candidate))
                            if self._excluded(resolved):
                                continue
                            if resolved.is_dir():
                                if recursive and not entry.is_symlink() and not getattr(candidate, "is_junction", lambda: False)():
                                    pending.append(resolved)
                            elif not resolved.is_file():
                                continue
                        except (OSError, ValueError, RuntimeError):
                            continue
                        yield resolved
            except (OSError, ValueError, RuntimeError):
                state["truncated"] = True

    def _read_content(
        self, path: Path, max_bytes: int, *, deadline: float, max_input_bytes: int | None = None
    ) -> dict[str, Any]:
        path = self._path(str(path))
        if self._excluded(path):
            raise ValueError("File is in an excluded runtime path")
        if path.suffix.casefold() != ".pdf":
            with path.open("rb") as stream:
                data = stream.read(max_bytes + 1)
            truncated = len(data) > max_bytes
            decoder = codecs.getincrementaldecoder("utf-8")()
            content = decoder.decode(data[:max_bytes], final=not truncated)
            if "\x00" in content:
                raise ValueError("Binary files are not supported")
            return {"content": content, "truncated": truncated}
        input_limit = min(max_input_bytes or self._MAX_PDF_BYTES, self._MAX_PDF_BYTES)
        if path.stat().st_size > input_limit:
            raise ValueError(f"PDF exceeds the {input_limit}-byte input limit")
        from document_loader import PdfReader

        with path.open("rb") as stream:
            data = stream.read(input_limit + 1)
        if len(data) > input_limit:
            raise ValueError(f"PDF exceeds the {input_limit}-byte input limit")
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise ValueError("Encrypted PDFs are not supported")
            pages: list[str] = []
            remaining = max_bytes
            page_count = len(reader.pages)
            truncated = page_count > self._MAX_PDF_PAGES
            for index in range(min(page_count, self._MAX_PDF_PAGES)):
                if time.monotonic() >= deadline:
                    truncated = True
                    break
                text = reader.pages[index].extract_text() or ""
                encoded = (("\n" if pages else "") + text).encode("utf-8")
                pages.append(encoded[:remaining].decode("utf-8", errors="ignore"))
                remaining -= min(len(encoded), remaining)
                if remaining == 0:
                    truncated = truncated or len(encoded) > len(pages[-1].encode("utf-8")) or index + 1 < page_count
                    break
            return {"content": "".join(pages), "truncated": truncated, "pages_read": len(pages), "page_count": page_count}
        except Exception as exc:
            raise ValueError(f"Unable to extract PDF text: {exc}") from exc

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
            directory = self._path(str(parameters.get("path", ".")))
            if not directory.is_dir():
                return ToolResult.failure(f"Directory not found: {directory}")
            state: dict[str, Any] = {"walked": 0, "truncated": False, "deadline": time.monotonic() + self._MAX_WALK_SECONDS}
            entries = [
                {
                    "name": entry.name,
                    "path": str(entry),
                    "kind": "directory" if entry.is_dir() else "file",
                }
                for entry in self._walk(directory, state, recursive=False)
            ]
            return ToolResult(
                success=True,
                status="completed",
                output={"entries": sorted(entries, key=lambda item: item["name"].casefold()), "truncated": state["truncated"]},
            )
        except (OSError, ValueError) as exc:
            return ToolResult.failure(str(exc), recoverable=True)


class FilesystemReadTool(_BoundedFilesystemTool):
    metadata = ToolMetadata(
        name="filesystem.read",
        description="Read bounded UTF-8 text or PDF page text under the allowed root.",
        category="computer.filesystem",
        input_schema={"path": {"type": "string"}, "max_bytes": {"type": "integer"}},
        output_schema={"path": {"type": "string"}, "content": {"type": "string"}},
        permission_level=PermissionLevel.READ_ONLY,
        risk_level=RiskLevel.READ_ONLY,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            path = self._path(str(parameters["path"]))
            max_bytes = self._limit(parameters, "max_bytes", 1_000_000, 5_000_000)
            if not path.is_file():
                return ToolResult.failure(f"File not found: {path}")
            if self._excluded(path):
                return ToolResult.failure("File is in an excluded runtime path", recoverable=True)
            deadline = time.monotonic() + self._MAX_WALK_SECONDS
            extracted = self._read_content(path, max_bytes, deadline=deadline)
            content = extracted["content"]
            return ToolResult(
                success=True,
                status="completed",
                output={
                    "path": str(path),
                    "content": content,
                    "truncated": extracted["truncated"],
                    **{key: value for key, value in extracted.items() if key.startswith("page")},
                },
            )
        except (KeyError, OSError, ValueError, RuntimeError, UnicodeError) as exc:
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
            if not pattern or len(pattern) > 1_000:
                return ToolResult.failure("pattern must contain 1 to 1000 characters", recoverable=True)
            max_results = self._limit(parameters, "max_results", 100, 500)
            base = self._path(str(parameters["path"])) if parameters.get("path") else self._root
            if not base.is_dir():
                return ToolResult.failure(f"Directory not found: {base}", recoverable=True)
            matches: list[str] = []
            state: dict[str, Any] = {"walked": 0, "truncated": False, "deadline": time.monotonic() + self._MAX_WALK_SECONDS}
            for path in self._walk(base, state):
                if fnmatch.fnmatch(path.name.casefold(), pattern.casefold()):
                    matches.append(str(path))
                    if len(matches) >= max_results:
                        break
            truncated = state["truncated"] or len(matches) >= max_results
            return ToolResult(
                success=True,
                status="completed",
                output={"matches": matches, "truncated": truncated},
            )
        except (KeyError, OSError, ValueError) as exc:
            return ToolResult.failure(str(exc), recoverable=True)


class FilesystemSearchContentTool(_BoundedFilesystemTool):
    """Bounded full-text search across files below the allowed root.

    Read-only by construction: files are opened, never written. The scan is
    deliberately capped (file count, file size, result count) so a broad query
    cannot freeze the runtime or read unbounded binary data.
    """

    #: Hard runtime bounds; the reasoning layer may pass smaller values.
    _MAX_FILES_SCANNED = 2000
    _MAX_FILE_BYTES = 2_000_000
    _MAX_RESULTS_CAP = 50
    _EXCERPT_RADIUS = 80

    metadata = ToolMetadata(
        name="filesystem.search_content",
        description=(
            "Search the text contents of files below an allowed folder, returning "
            "matching file paths with a short excerpt and line number."
        ),
        category="computer.filesystem",
        input_schema={
            "query": {"type": "string", "description": "text to look for inside files"},
            "path": {"type": "string", "description": "optional subfolder to search"},
            "pattern": {"type": "string", "description": "optional filename glob, e.g. *.txt"},
            "max_results": {"type": "integer", "description": "bounded to 50"},
            "max_files": {"type": "integer", "description": "file count bounded to 2000"},
            "max_bytes": {"type": "integer", "description": "per-file input and text bytes bounded to 2000000"},
        },
        output_schema={"matches": {"type": "array"}, "scanned": {"type": "integer"}},
        permission_level=PermissionLevel.READ_ONLY,
        risk_level=RiskLevel.READ_ONLY,
        required_parameters=("query",),
        verifiable=True,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            query = str(parameters["query"]).strip()
            if not query:
                return ToolResult.failure("query must not be empty", recoverable=True)
            if len(query) > 1_000:
                return ToolResult.failure("query exceeds the 1000-character limit", recoverable=True)
            needle = query.casefold()

            max_results = self._limit(parameters, "max_results", 20, self._MAX_RESULTS_CAP)
            max_files = self._limit(
                parameters,
                "max_files",
                min(config.FILESYSTEM_CONTENT_MAX_FILES, self._MAX_FILES_SCANNED),
                self._MAX_FILES_SCANNED,
            )
            max_file_bytes = self._limit(
                parameters,
                "max_bytes",
                min(config.FILESYSTEM_CONTENT_MAX_BYTES, self._MAX_FILE_BYTES),
                self._MAX_FILE_BYTES,
            )
            pattern = str(parameters["pattern"]) if parameters.get("pattern") else None
            base = self._path(str(parameters["path"])) if parameters.get("path") else self._root
            if not base.is_dir():
                return ToolResult.failure(f"Directory not found: {base}", recoverable=True)

            matches: list[dict[str, Any]] = []
            scanned = 0
            state: dict[str, Any] = {"walked": 0, "truncated": False, "deadline": time.monotonic() + self._MAX_WALK_SECONDS}
            for path in self._walk(base, state):
                if not path.is_file():
                    continue
                if pattern and not fnmatch.fnmatch(path.name.casefold(), pattern.casefold()):
                    continue
                if scanned >= max_files:
                    state["truncated"] = True
                    break
                scanned += 1
                try:
                    if path.stat().st_size > max_file_bytes:
                        state["truncated"] = True
                        continue
                    extracted = self._read_content(
                        path, max_file_bytes, deadline=state["deadline"], max_input_bytes=max_file_bytes
                    )
                except (OSError, ValueError, RuntimeError):
                    state["truncated"] = True
                    continue
                state["truncated"] = state["truncated"] or extracted["truncated"]
                text = extracted["content"]
                hit = text.casefold().find(needle)
                if hit == -1:
                    continue
                matches.append(
                    {
                        "path": str(path),
                        "line": text.count("\n", 0, hit) + 1,
                        "excerpt": self._excerpt(text, hit, len(needle)),
                    }
                )
                if len(matches) >= max_results:
                    state["truncated"] = True
                    break

            return ToolResult(
                success=True,
                status="completed",
                output={
                    "matches": matches,
                    "scanned": scanned,
                    "truncated": state["truncated"],
                },
            )
        except (KeyError, OSError, ValueError) as exc:
            return ToolResult.failure(str(exc), recoverable=True)

    @classmethod
    def _excerpt(cls, text: str, start: int, length: int) -> str:
        lo = max(start - cls._EXCERPT_RADIUS, 0)
        hi = min(start + length + cls._EXCERPT_RADIUS, len(text))
        prefix = "…" if lo > 0 else ""
        suffix = "…" if hi < len(text) else ""
        return f"{prefix}{text[lo:hi].replace(chr(13), ' ').strip()}{suffix}"


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
            if destination.is_dir():
                destination = self._path(str(destination / source.name))
            if destination.exists() and not bool(parameters.get("overwrite", False)):
                return ToolResult.failure(
                    f"Destination already exists: {destination}", recoverable=True
                )
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
