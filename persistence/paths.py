"""Resolution of Atlas's application resources and mutable user data.

Atlas runs in two very different places:

* **development** — from a source checkout, where the repository directory is
  both the code and a convenient place to keep local state;
* **installed** — from ``Atlas.exe`` inside a read-only installation directory
  (for example ``C:\\Program Files``), where nothing next to the executable may
  be written and every mutable byte has to live elsewhere.

Rather than letting modules guess, two concepts are defined here:

``AtlasApplicationPaths``
    Read-only resources shipped with the application: prompts, bundled tool
    catalogs, frontend assets. Never written to.

``AtlasDataPaths``
    Everything mutable: the SQLite database, the Chroma directory, knowledge
    documents, artifacts, logs, models, configuration and exports.

The data root resolves in this order:

1. ``ATLAS_DATA_ROOT`` — explicit override, used by tests and by anyone who
   wants Atlas state somewhere specific.
2. ``%LOCALAPPDATA%\\Atlas`` when running frozen (a packaged Windows build).
3. The repository root when running from source (development convenience).

No username is ever hardcoded, and nothing outside this module should
reconstruct these paths from ``__file__``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

#: Directory name Atlas uses inside the user's local application data folder.
APP_DIR_NAME = "Atlas"

#: Environment variable that overrides the resolved data root.
DATA_ROOT_ENV = "ATLAS_DATA_ROOT"

#: File name of the authoritative SQLite database inside the data root.
DATABASE_FILE_NAME = "atlas.db"


def application_root() -> Path:
    """Return the directory holding Atlas's own, read-only resources.

    A frozen (packaged) build keeps its resources beside the executable; a
    source checkout keeps them in the repository root.
    """

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def local_app_data() -> Path:
    """Return the Windows per-user local application data folder.

    ``%LOCALAPPDATA%`` is read from the environment (never a literal path), and
    a documented fallback keeps the resolution working on a stripped-down
    environment or a non-Windows host.
    """

    value = os.environ.get("LOCALAPPDATA") or ""
    if value.strip():
        return Path(value)
    return Path.home() / "AppData" / "Local"


def is_frozen() -> bool:
    """True when Atlas is running from a packaged executable."""

    return bool(getattr(sys, "frozen", False))


@dataclass(frozen=True)
class AtlasApplicationPaths:
    """Read-only resources that ship with the application."""

    root: Path

    @property
    def prompts_dir(self) -> Path:
        return self.root / "prompts"

    @property
    def tools_dir(self) -> Path:
        return self.root / "tools"

    @property
    def frontend_dir(self) -> Path:
        return self.root / "frontend"

    @property
    def prompts_command_catalog(self) -> Path:
        """The bundled PowerShell command catalog used for planner discovery."""

        return self.tools_dir / "powershell_commands.json"


@dataclass(frozen=True)
class AtlasDataPaths:
    """Every mutable path Atlas owns, rooted at a single data directory."""

    root: Path

    # -- the single authoritative structured store ----------------------------
    @property
    def database_file(self) -> Path:
        """``atlas.db`` — the authoritative structured persistent state."""

        return self.root / DATABASE_FILE_NAME

    # -- semantic/vector retrieval index (kept deliberately separate) --------
    @property
    def chroma_dir(self) -> Path:
        """ChromaDB persistence directory. Never the same folder as SQLite."""

        return self.root / "chroma"

    # -- genuinely file-oriented data ----------------------------------------
    @property
    def knowledge_dir(self) -> Path:
        """PDF/documents Atlas indexes. The documents stay on disk."""

        return self.root / "knowledge"

    @property
    def artifacts_dir(self) -> Path:
        """Generated deliverables, screenshots and other produced files."""

        return self.root / "artifacts"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def config_dir(self) -> Path:
        """User-editable configuration. A file format, not a database."""

        return self.root / "config"

    @property
    def exports_dir(self) -> Path:
        """Optional JSON exports of conversations and other interchange files."""

        return self.root / "exports"

    @property
    def backups_dir(self) -> Path:
        """Where a copy of legacy data is kept before anything is touched."""

        return self.root / "backups"

    @property
    def log_file(self) -> Path:
        return self.logs_dir / "atlas.log"

    @property
    def settings_file(self) -> Path:
        """User-editable settings file (configuration, never Atlas state)."""

        return self.config_dir / "settings.json"

    def ensure(self) -> "AtlasDataPaths":
        """Create every directory Atlas expects to own."""

        for directory in (
            self.root,
            self.knowledge_dir,
            self.artifacts_dir,
            self.logs_dir,
            self.models_dir,
            self.config_dir,
            self.chroma_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self


def resolve_data_paths(*, override: Path | str | None = None) -> AtlasDataPaths:
    """Resolve the mutable data root for this process.

    See the module docstring for the resolution order. The result is *not*
    created here; call :meth:`AtlasDataPaths.ensure` for that, so path
    resolution stays side-effect free and easy to test.
    """

    if override is not None:
        root = Path(override).expanduser()
    else:
        from_env = os.environ.get(DATA_ROOT_ENV, "")
        if from_env.strip():
            root = Path(from_env).expanduser()
        elif is_frozen():
            root = local_app_data() / APP_DIR_NAME
        else:
            root = application_root()
    return AtlasDataPaths(root=root.resolve())


def resolve_application_paths(*, override: Path | str | None = None) -> AtlasApplicationPaths:
    """Resolve the read-only application resource root."""

    root = Path(override).expanduser() if override is not None else application_root()
    return AtlasApplicationPaths(root=root.resolve())
