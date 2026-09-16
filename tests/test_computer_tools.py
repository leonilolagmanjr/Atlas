import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from computer.filesystem import (
    FilesystemListTool,
    FilesystemMetadataTool,
    FilesystemReadTool,
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
                "processes.list",
                "processes.inspect",
                "system.info",
                "applications.list_installed",
                "applications.search_installed",
                "applications.launch",
                "applications.launch_named",
                "powershell.execute",
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
