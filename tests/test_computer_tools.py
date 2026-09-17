import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from computer.filesystem import (
    FilesystemListTool,
    FilesystemMetadataTool,
    FilesystemMoveTool,
    FilesystemReadTool,
    FilesystemSearchContentTool,
    FilesystemSearchTool,
)
from computer.applications import InstalledApplicationSearchTool
from computer.launch import ApplicationLaunchTool
from computer.system import SystemInfoTool
from computer.runtime import register_read_only_tools
from executor import Executor
from models import ExecutionContext, ExecutionPlan, ExecutionStep, PlanStatus, TaskStatus
from tools import ToolRegistry
from tools import ExecutionMode, PermissionEngine, ToolRouter


class ComputerToolTests(unittest.TestCase):
    def test_pdf_read_uses_document_loader_extraction(self):
        from document_loader import PdfReader, read_pdf_document
        from pypdf import PdfWriter
        from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "note.pdf"
            writer = PdfWriter()
            page = writer.add_blank_page(width=300, height=200)
            font = DictionaryObject({
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            })
            page[NameObject("/Resources")] = DictionaryObject({
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
            })
            stream = DecodedStreamObject()
            stream.set_data(b"BT /F1 12 Tf 20 100 Td (Atlas PDF evidence) Tj ET")
            page[NameObject("/Contents")] = stream
            writer.write(str(path))
            self.assertEqual(PdfReader(str(path)).pages[0].extract_text(), "Atlas PDF evidence")
            expected = read_pdf_document(path).full_text
            result = FilesystemReadTool(root=root).execute({"path": "note.pdf", "max_bytes": 100})
            self.assertTrue(result.success, result.error)
            self.assertEqual(result.output["content"], expected)
            self.assertFalse(result.output["truncated"])
            bounded = FilesystemReadTool(root=root).execute({"path": "note.pdf", "max_bytes": 5})
            self.assertEqual(bounded.output["content"], "Atlas")
            self.assertTrue(bounded.output["truncated"])
            search = FilesystemSearchContentTool(root=root).execute({"query": "PDF evidence", "pattern": "*.pdf"})
            self.assertTrue(search.success, search.error)
            self.assertEqual(len(search.output["matches"]), 1)
            self.assertIn("Atlas PDF evidence", search.output["matches"][0]["excerpt"])
            capped = FilesystemReadTool(root=root)
            capped._MAX_PDF_BYTES = 10
            with patch("document_loader.PdfReader") as reader:
                result = capped.execute({"path": "note.pdf"})
                self.assertFalse(result.success)
                reader.assert_not_called()
            writer.add_blank_page(width=300, height=200)
            writer.write(str(path))
            capped = FilesystemReadTool(root=root)
            capped._MAX_PDF_PAGES = 1
            result = capped.execute({"path": "note.pdf"})
            self.assertTrue(result.success, result.error)
            self.assertEqual(result.output["pages_read"], 1)
            self.assertEqual(result.output["page_count"], 2)
            self.assertTrue(result.output["truncated"])
            writer.encrypt("password")
            writer.write(str(path))
            self.assertFalse(capped.execute({"path": "note.pdf"}).success)
            path.write_bytes(b"invalid PDF")
            self.assertFalse(capped.execute({"path": "note.pdf"}).success)

    def test_read_bounds_stream_before_decoding(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "note.txt").write_bytes(b"x")
            stream = io.BytesIO("Atlas é evidence".encode("utf-8"))
            with patch.object(Path, "open", return_value=stream) as opened:
                result = FilesystemReadTool(root=root).execute({"path": "note.txt", "max_bytes": 7})
            self.assertTrue(result.success, result.error)
            self.assertEqual(result.output["content"], "Atlas ")
            self.assertTrue(result.output["truncated"])
            opened.assert_called_once_with("rb")

    def test_read_uses_only_bounded_stream_request(self):
        class GuardedStream(io.BytesIO):
            def read(self, size=-1):
                if size < 0 or size > 5:
                    raise AssertionError(f"Unbounded read: {size}")
                return super().read(size)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "note.txt").write_bytes(b"x")
            with patch.object(Path, "open", return_value=GuardedStream(b"abcdefgh")):
                result = FilesystemReadTool(root=root).execute({"path": "note.txt", "max_bytes": 4})
            self.assertTrue(result.success, result.error)
            self.assertEqual(result.output["content"], "abcd")

    def test_content_search_honors_file_byte_pattern_and_result_limits(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "one.txt").write_text("Atlas one", encoding="utf-8")
            (root / "two.txt").write_text("Atlas two", encoding="utf-8")
            (root / "large.txt").write_text("Atlas" * 100, encoding="utf-8")
            tool = FilesystemSearchContentTool(root=root)
            result = tool.execute({"query": "atlas", "path": ".", "pattern": "*.txt", "max_results": 1, "max_files": 1, "max_bytes": 30})
            self.assertTrue(result.success, result.error)
            self.assertLessEqual(result.output["scanned"], 1)
            self.assertLessEqual(len(result.output["matches"]), 1)
            self.assertTrue(result.output["truncated"])
            large = tool.execute({"query": "atlas", "pattern": "large.txt", "max_bytes": 30})
            self.assertTrue(large.success)
            self.assertEqual(large.output["matches"], [])
            self.assertTrue(large.output["truncated"])

    def test_file_search_excludes_runtime_and_dependency_defaults(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for name in (".git", ".venv", "node_modules", "runtime", "database", "memory"):
                (root / name).mkdir()
                (root / name / "secret.txt").write_text("private needle", encoding="utf-8")
            (root / ".env.local").write_text("private needle", encoding="utf-8")
            result = FilesystemSearchContentTool(root=root).execute({"query": "private needle"})
            self.assertTrue(result.success, result.error)
            self.assertEqual(result.output["matches"], [])
            self.assertEqual(result.output["scanned"], 0)
            listing = FilesystemListTool(root=root).execute({"path": "."})
            self.assertEqual(listing.output["entries"], [])
            read = FilesystemReadTool(root=root).execute({"path": ".env.local"})
            self.assertFalse(read.success)

    def test_search_walk_entry_and_time_caps_include_nonmatching_entries(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for index in range(10):
                (root / f"item-{index}").mkdir()
            for tool in (FilesystemSearchContentTool(root=root), FilesystemSearchTool(root=root)):
                tool._MAX_WALK_ENTRIES = 3
                result = tool.execute({"query": "needle", "pattern": "*.txt"})
                self.assertTrue(result.success, result.error)
                self.assertTrue(result.output["truncated"])
                self.assertEqual(result.output["matches"], [])
                tool._MAX_WALK_SECONDS = 0
                result = tool.execute({"query": "needle", "pattern": "*"})
                self.assertTrue(result.output["truncated"])
                self.assertEqual(result.output["matches"], [])

    def test_root_checks_resolved_search_candidates_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            parent = Path(temporary_directory)
            root = parent / "allowed"
            root.mkdir()
            outside = parent / "outside.txt"
            outside.write_text("outside needle", encoding="utf-8")
            try:
                (root / "linked.txt").symlink_to(outside)
                (root / "linked-dir").symlink_to(parent, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"Symlink creation unavailable: {exc}")
            for tool, arguments in (
                (FilesystemReadTool(root=root), {"path": "linked.txt"}),
                (FilesystemSearchContentTool(root=root), {"query": "needle", "path": "../"}),
            ):
                self.assertFalse(tool.execute(arguments).success)
            content = FilesystemSearchContentTool(root=root).execute({"query": "needle"})
            names = FilesystemSearchTool(root=root).execute({"pattern": "*"})
            listing = FilesystemListTool(root=root).execute({"path": "."})
            self.assertEqual(content.output["matches"], [])
            self.assertEqual(names.output["matches"], [])
            self.assertEqual(listing.output["entries"], [])

    def test_resolved_outside_candidate_is_skipped_without_symlink_privileges(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            parent = Path(temporary_directory).resolve()
            root = parent / "allowed"
            root.mkdir()
            candidate = root / "linked.txt"
            candidate.write_text("needle", encoding="utf-8")
            outside = parent / "outside.txt"
            outside.write_text("needle", encoding="utf-8")
            original = Path.resolve

            def resolve(path, *args, **kwargs):
                if path == candidate:
                    return outside
                return original(path, *args, **kwargs)

            with patch.object(Path, "resolve", resolve):
                result = FilesystemSearchContentTool(root=root).execute({"query": "needle"})
            self.assertTrue(result.success, result.error)
            self.assertEqual(result.output["matches"], [])
            self.assertEqual(result.output["scanned"], 0)

    def test_default_root_remains_configured_and_catalog_tracks_registry(self):
        from tools.capabilities import CapabilityRegistry

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            (root / "note.txt").write_text("Atlas", encoding="utf-8")
            with patch("config.COMPUTER_ROOT", root):
                tool = FilesystemReadTool()
            self.assertTrue(tool.execute({"path": "note.txt"}).success)
            self.assertFalse(tool.execute({"path": str(root.parent / "outside.txt")}).success)
            registry = ToolRegistry()
            capabilities = CapabilityRegistry(registry)
            self.assertEqual(capabilities.plannable(), [])
            registry.register(tool)
            self.assertEqual([item.name for item in capabilities.plannable()], ["filesystem.read"])
            self.assertFalse(capabilities.exists("filesystem.write"))

    def test_negative_and_invalid_bounded_arguments_fail(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "note.txt").write_text("Atlas", encoding="utf-8")
            for name in ("max_files", "max_bytes", "max_results"):
                for value in (-1, 0, True, "5", None):
                    with self.subTest(name=name, value=value):
                        result = FilesystemSearchContentTool(root=root).execute({"query": "Atlas", name: value})
                        self.assertFalse(result.success)
            self.assertFalse(FilesystemReadTool(root=root).execute({"path": "note.txt", "max_bytes": -1}).success)

    def test_move_into_existing_directory_normalizes_before_collision(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "target").mkdir()
            (root / "note.txt").write_text("Atlas", encoding="utf-8")
            tool = FilesystemMoveTool(root=root)
            result = tool.execute({"source": "note.txt", "destination": "target"})
            self.assertTrue(result.success, result.error)
            self.assertEqual(Path(result.output["destination"]), (root / "target" / "note.txt").resolve())
            (root / "note.txt").write_text("New", encoding="utf-8")
            collision = tool.execute({"source": "note.txt", "destination": "target"})
            self.assertFalse(collision.success)
            self.assertEqual((root / "target" / "note.txt").read_text(encoding="utf-8"), "Atlas")

    def test_runtime_registry_contracts_are_actual_and_root_scoped(self):
        from tools.capabilities import CapabilityRegistry, PLANNABLE_CAPABILITIES

        empty = CapabilityRegistry(ToolRegistry())
        self.assertEqual(empty.plannable(), [])
        self.assertFalse(empty.exists("filesystem.read"))
        self.assertEqual({item.name for item in CapabilityRegistry().plannable()}, PLANNABLE_CAPABILITIES)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "note.txt").write_text("Atlas", encoding="utf-8")
            registry = ToolRegistry()
            register_read_only_tools(registry, root=root)
            capabilities = CapabilityRegistry(registry)
            self.assertTrue(set(item.name for item in capabilities.plannable()).issubset(registry.names()))
            router = ToolRouter(registry=registry, permission_engine=PermissionEngine(mode=ExecutionMode.CONFIRM))
            for name, parameters in (
                ("filesystem.list", {"path": "."}),
                ("filesystem.search", {"pattern": "*.txt", "path": ".", "max_results": 1}),
                ("filesystem.read", {"path": "note.txt", "max_bytes": 100}),
                ("filesystem.search_content", {"query": "Atlas", "path": ".", "pattern": "*.txt", "max_results": 1, "max_files": 1, "max_bytes": 100}),
            ):
                result = router.execute(name, parameters)
                self.assertTrue(result.success, result.error)
                self.assertTrue(set(parameters).issubset(capabilities.get(name).parameters))
            self.assertFalse(router.execute("filesystem.read", {"path": "../outside.txt"}).success)

    def test_filesystem_tools_stay_inside_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "note.txt").write_text("Atlas test", encoding="utf-8")
            read_result = FilesystemReadTool(root=root).execute({"path": "note.txt"})
            outside_result = FilesystemReadTool(root=root).execute({"path": "../note.txt"})

            self.assertTrue(read_result.success)
            self.assertEqual(read_result.output["content"], "Atlas test")
            self.assertFalse(outside_result.success)

    def test_filesystem_search_is_bounded_and_read_only(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for index in range(3):
                (root / f"item-{index}.txt").write_text("x", encoding="utf-8")

            result = FilesystemSearchTool(root=root).execute({"pattern": "*.txt", "max_results": 2})

            self.assertTrue(result.success)
            self.assertEqual(len(result.output["matches"]), 2)
            self.assertTrue(result.output["truncated"])

    def test_filesystem_metadata_and_listing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "note.txt").write_text("Atlas", encoding="utf-8")

            listing = FilesystemListTool(root=root).execute({"path": "."})
            metadata = FilesystemMetadataTool(root=root).execute({"path": "note.txt"})

            self.assertTrue(listing.success)
            self.assertEqual(listing.output["entries"][0]["name"], "note.txt")
            self.assertTrue(metadata.success)
            self.assertEqual(metadata.output["kind"], "file")

    def test_system_tool_returns_structured_output(self):
        result = SystemInfoTool().execute({})

        self.assertIn(result.success, {True, False})
        if result.success:
            self.assertIn("system", result.output)

    def test_computer_tools_can_be_registered(self):
        registry = ToolRegistry()
        register_read_only_tools(registry)

        self.assertEqual(
            registry.names(),
            (
                "filesystem.list",
                "filesystem.read",
                "filesystem.metadata",
                "filesystem.search",
                "filesystem.search_content",
                # High-level, permission-gated filesystem mutations added for the
                # structured task pipeline (Phase 4 capability registry).
                "filesystem.write",
                "filesystem.create_folder",
                "filesystem.move",
                "filesystem.copy",
                "processes.list",
                "processes.inspect",
                "system.info",
                "applications.list_installed",
                "applications.search_installed",
                "applications.launch",
                "applications.launch_named",
                "applications.write_text",
                "content.generate",
                "powershell.execute",
                "web.search",
                "web.fetch",
            ),
        )

    def test_application_search_rejects_empty_query(self):
        result = InstalledApplicationSearchTool().execute({"query": "  "})

        self.assertFalse(result.success)
        self.assertIn("cannot be empty", result.error)

    def test_launch_requires_confirmation_in_default_mode(self):
        registry = ToolRegistry()
        registry.register(ApplicationLaunchTool())
        router = ToolRouter(
            registry=registry,
            permission_engine=PermissionEngine(mode=ExecutionMode.CONFIRM),
        )

        result = router.execute(
            "applications.launch",
            {"executable": "C:\\Windows\\System32\\notepad.exe"},
        )

        self.assertEqual(result.status, "confirmation_required")
        self.assertFalse(result.success)

    def test_autonomous_launch_uses_non_shell_process(self):
        registry = ToolRegistry()
        registry.register(ApplicationLaunchTool())
        router = ToolRouter(
            registry=registry,
            permission_engine=PermissionEngine(mode=ExecutionMode.AUTONOMOUS),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            executable = Path(temporary_directory) / "example.exe"
            executable.write_bytes(b"test")
            with patch("computer.launch.subprocess.Popen") as popen:
                popen.return_value.pid = 42
                result = router.execute("applications.launch", {"executable": str(executable)})

        self.assertTrue(result.success)
        self.assertEqual(result.output["pid"], 42)
        self.assertFalse(popen.call_args.kwargs["shell"])

    def test_executor_pauses_for_tool_confirmation(self):
        registry = ToolRegistry()
        registry.register(ApplicationLaunchTool())
        router = ToolRouter(
            registry=registry,
            permission_engine=PermissionEngine(mode=ExecutionMode.CONFIRM),
        )
        executor = Executor(
            vector_store=object(),
            system_prompt="",
            retrieval_template="",
            tool_router=router,
        )
        plan = ExecutionPlan(
            user_question="Open the application",
            steps=[
                ExecutionStep(
                    id="launch",
                    name="Launch application",
                    action="invoke_tool",
                    description="Launch selected executable",
                    metadata={
                        "tool": "applications.launch",
                        "parameters": {"executable": "C:\\Windows\\System32\\notepad.exe"},
                    },
                )
            ],
        )
        context = ExecutionContext(user_input=plan.user_question)

        executor.execute(plan, context)

        self.assertEqual(plan.status, PlanStatus.WAITING_FOR_CONFIRMATION)
        self.assertEqual(context.status, TaskStatus.WAITING_FOR_CONFIRMATION)
        self.assertEqual(plan.steps[0].status.value, "PENDING")
        self.assertEqual(context.tool_calls[0]["status"], "confirmation_required")


if __name__ == "__main__":
    unittest.main()
