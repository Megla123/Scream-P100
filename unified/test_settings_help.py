import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from config import CONTROL_DEFAULTS, DEFAULTS, SAMPLING_KEYS
from config_names import LEGACY_SERVER_FLAGS, NAMING_DEFAULTS
from settings_help import load_reference, markdown, reference_text, short_help


SOURCE = Path(__file__).parent


class SettingsReferenceTest(unittest.TestCase):
    def test_reference_covers_supported_names_and_generates_the_checked_in_document(self):
        entries = load_reference()
        for name in (*DEFAULTS, *(f"sampling.{field}" for field in SAMPLING_KEYS)):
            self.assertEqual(entries[name]["kind"], "setting", name)
        for name in (*CONTROL_DEFAULTS, *NAMING_DEFAULTS):
            self.assertEqual(entries[name]["kind"], "control", name)
        for flag in LEGACY_SERVER_FLAGS:
            name = "server_options." + flag[2:].replace("-", "_")
            self.assertIn(name, entries)
            self.assertIn(flag, reference_text(entries, name))
        self.assertEqual(markdown(entries), (SOURCE / "SETTINGS.md").read_text(encoding="utf-8"))
        synopsis = short_help(entries)
        for option in ("--config", "--environment-prefix", "--legacy-names", "--context", "--help-settings",
                       "--cookbook", "--recipe", "--import-recipe", "--export-recipe", "--parent-recipe"):
            self.assertIn(option, synopsis)

    def test_lookup_renders_canonical_paths_and_aliases(self):
        entries = load_reference()
        by_name = reference_text(entries, "context")
        for name in ("settings.context", "profiles.<profile>.settings.context", "--context", "SP100_CONTEXT",
                     "SCREAM_CONTEXT", "--ctx", "--max-context"):
            self.assertIn(name, by_name)
        for name in ("--context", "--ctx", "SP100_CONTEXT", "profiles.site.z97.settings.context"):
            self.assertEqual(reference_text(entries, name), by_name)
        nested = reference_text(entries, "strata_settings.vision.mmproj")
        self.assertIn("profiles.<profile>.settings.strata_settings.vision.mmproj", nested)
        self.assertIn("--set 'strata_settings.vision=OBJECT'", nested)
        self.assertIn("SP100_STRATA_SETTINGS (whole JSON object)", nested)
        self.assertIn("SP100_ENV__FNX_ROUTE_FILE", reference_text(entries, "env.fnx_route_file"))
        self.assertIn("SP100_SAMPLING__TOP_K", reference_text(entries, "settings.sampling.top_k"))
        self.assertIn("CUSTOM_CONTEXT", reference_text(entries, "CUSTOM_CONTEXT", prefix="CUSTOM"))
        self.assertIn("SP100_LEGACY_NAMES", reference_text(entries, "legacy_names", prefix="CUSTOM"))
        self.assertIn("settings.engine_options.parallel", reference_text(entries, "--slots"))
        self.assertIn("settings.sampling.temperature", reference_text(entries, "--temp"))
        self.assertIn("settings.server_options.gpu", reference_text(entries, "--gpu"))
        self.assertIn("settings.engine_options.example_flag", reference_text(entries, "SP100_ENGINE_OPTIONS__EXAMPLE_FLAG"))
        self.assertEqual(reference_text(entries, "--cookbook"), reference_text(entries, "config"))
        self.assertEqual(reference_text(entries, "--recipe"), reference_text(entries, "profile"))
        self.assertEqual(reference_text(entries, "--export-recipe"), reference_text(entries, "export_profile"))
        with self.assertRaisesRegex(ValueError, "launcher-owned"):
            reference_text(entries, "strata_settings.exe")
        with self.assertRaisesRegex(ValueError, "unknown settings reference name"):
            reference_text(entries, "bad_prefix.unknown")

    def test_invalid_reference_gets_a_clear_error(self):
        entries = load_reference()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings_reference.json"
            entries["context"]["related"] = ["no_such_setting"]
            path.write_text(json.dumps({"version": 1, "entries": entries}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "context.*unknown entries no_such_setting"):
                load_reference(path)
            del entries["context"]
            path.write_text(json.dumps({"version": 1, "entries": entries}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing context"):
                load_reference(path)

    def test_help_runs_before_config_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "nonexistent.json"
            result = subprocess.run([sys.executable, SOURCE / "scream.py", "--config", missing, "--help-settings=--ctx"],
                                    capture_output=True, text=True, check=True, timeout=10)
            self.assertIn("settings.context", result.stdout)
            self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
