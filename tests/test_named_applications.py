import unittest
from pathlib import Path
from unittest.mock import patch

from computer.launch import (
    NamedApplicationLaunchTool,
    _matches_registry_entry,
    _rank_registry_candidates,
    resolve_application_name,
)
from intent_classifier import IntentClassifier
from planner import Planner


class NamedApplicationTests(unittest.TestCase):
    def test_discord_request_is_application_intent(self):
        result = IntentClassifier().classify("Open Discord")

        self.assertEqual(result.intent, "APPLICATION")

    def test_arbitrary_named_application_request_is_application_intent(self):
        result = IntentClassifier().classify("Launch Spotify")

        self.assertEqual(result.intent, "APPLICATION")

    def test_discord_request_uses_named_launch_tool(self):
        decision = Planner().create_plan("Open Discord", intent="APPLICATION")
        step = decision.plan.steps[0]

        self.assertEqual(decision.strategy, "deterministic_named_application_launch")
        self.assertEqual(step.metadata["tool"], "applications.launch_named")
        self.assertEqual(step.metadata["parameters"]["application"], "Discord")

    @patch("computer.launch.resolve_application_name")
    @patch("computer.launch.subprocess.Popen")
    def test_named_launch_uses_non_shell_process(self, popen, resolve):
        executable = Path("C:/Apps/Discord/Discord.exe")
        resolve.return_value = executable
        popen.return_value.pid = 123

        result = NamedApplicationLaunchTool().execute({"application": "discord"})

        self.assertTrue(result.success)
        self.assertEqual(result.output["pid"], 123)
        self.assertFalse(popen.call_args.kwargs["shell"])

    @unittest.skipUnless(__import__("sys").platform == "win32", "Windows application registry is required")
    def test_installed_discord_resolves_when_present(self):
        resolved = resolve_application_name("Discord")
        self.assertTrue(resolved is None or resolved.name.casefold() == "discord.exe")

class _FakeWinreg:
    # Minimal winreg stub exposing exactly what the resolver touches.
    HKEY_CURRENT_USER = "HKCU"
    HKEY_LOCAL_MACHINE = "HKLM"

    def __init__(self, entry_key="RazerSynapse"):
        self._entry_key = entry_key
    def OpenKey(self, _root, _path):  # noqa: N802 - mirrors winreg API
        return type("Key", (), {"__enter__": lambda self: self, "__exit__": lambda *a: False})()

    def QueryInfoKey(self, _key):  # noqa: N802
        return (1, 0, 0)

    def EnumKey(self, _key, _index):  # noqa: N802
        return self._entry_key
    def QueryValueEx(self, _key, _name):  # noqa: N802
        raise OSError
class RegistryResolutionTests(unittest.TestCase):
    # Registry fallback must resolve apps whose uninstall entry has no
    # InstallLocation and a non-executable DisplayIcon (e.g. Razer Synapse,
    # whose DisplayIcon is a .ico under %ProgramData%).

    def test_registry_name_matches_by_token_not_just_full_string(self):
        self.assertTrue(_matches_registry_entry("razer synapse", "razer synapse 3", None))
        self.assertTrue(_matches_registry_entry("razer synapse", "razer synapse", None))
        self.assertFalse(_matches_registry_entry("razer synapse", "unrelated app", None))

    def test_rank_prefers_real_binary_over_uninstaller(self):
        ranked = _rank_registry_candidates(
            "razer synapse",
            [
                Path("C:/Program Files/Razer/Razer Synapse Setup/Uninstall-Synapse.exe"),
                Path("C:/Program Files/Razer/RazerAppEngine/RazerAppEngine.exe"),
            ],
        )
        self.assertEqual(ranked[0].name, "RazerAppEngine.exe")

    def test_displayicon_ico_install_dir_is_scanned(self):
        # Reproduces Razer Synapse: InstallLocation empty, DisplayIcon is a .ico
        # under the real install dir. The resolver must fall back to scanning
        # that directory (the parent of the icon) for executables.
        import computer.launch as launch
        record = {
            "DisplayName": "Razer Synapse",
            "InstallLocation": "",
            "DisplayIcon": str(Path("C:/Temp/Razer/Synapse/synapse.ico")),
        }
        real_exe = Path("C:/Temp/Razer/Synapse/RazerAppEngine.exe")

        def fake_registry_value(_key, _winreg, name):
            return record.get(name)

        def fake_glob(_self, pattern):
            return [real_exe] if pattern == "*.exe" else []

        with patch.object(launch, "sys") as mock_sys, patch.dict(
            "sys.modules", {"winreg": _FakeWinreg()}
        ), patch.object(launch, "_registry_value", side_effect=fake_registry_value), patch(
            "pathlib.Path.is_dir", return_value=True
        ), patch("pathlib.Path.is_file", return_value=True), patch(
            "pathlib.Path.glob", fake_glob
        ):
            mock_sys.platform = "win32"
            candidates = launch._registry_application_executables("razer synapse")

        self.assertIn(real_exe, candidates)


if __name__ == "__main__":
    unittest.main()
