from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


LAUNCHER = Path(__file__).with_name("scream.py").resolve()


class ProfileExchangeTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def invoke(self, *args, check=True, **environment):
        env = {key: value for key, value in os.environ.items() if not key.startswith(("SP100_", "SCREAM_"))}
        return subprocess.run([sys.executable, LAUNCHER, *args], env={**env, **environment},
                              capture_output=True, text=True, check=check, timeout=15)

    def put(self, name, data):
        path = self.root / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_export_file_settings_and_import_a_nested_delta(self):
        source = self.put("source.json", {
            "profile": "site", "environment_precedence": "environment", "settings": {"host": "0.0.0.0"},
            "profiles": {
                "default": {"settings": {"port": 8081}},
                "base": {"extends": "inkling-2-mini-strata", "settings": {"context": 32768,
                         "strata_settings": {"vision": {"model": "vocab.gguf", "mmproj": "old.gguf"}}}},
                "site": {"extends": "base", "settings": {"context": 65536,
                         "strata_settings": {"vision": {"mmproj": "new.gguf"}}, "env": {"FEATURE": None}}},
            },
        })
        exported = self.root / "site.json"
        self.invoke("--cookbook", source, "--export-recipe", exported, SP100_CONTEXT="8192")
        data = json.loads(exported.read_text())
        self.assertEqual(data, {"version": 1, "profile": "site", "settings": {
            "backend": "strata-inkling", "host": "0.0.0.0", "port": 8081, "context": 65536, "vision_residency": "auto",
            "strata_settings": {"vision": {"model": "vocab.gguf", "mmproj": "new.gguf"}}, "env": {"FEATURE": None},
        }})
        self.assertEqual(exported.stat().st_mode & 0o777, 0o600)

        target = self.put("destination.json", {"settings": {"host": "0.0.0.0"}, "profiles": {
            "default": {"settings": {"port": 8081}},
            "parent": {"extends": "inkling-2-mini-strata", "settings": {"context": 32768, "env": {"FEATURE": "on"},
                       "strata_settings": {"vision": {"model": "vocab.gguf", "mmproj": "old.gguf", "gpu": True}}}},
        }})
        self.invoke("--cookbook", target, "--recipe", "new-site", "--import-recipe", exported, "--parent-recipe", "parent")
        self.assertEqual(json.loads(target.read_text())["profiles"]["new-site"], {"extends": "parent", "settings": {
            "context": 65536, "strata_settings": {"vision": {"mmproj": "new.gguf"}}, "env": {"FEATURE": None},
        }})
        settings = json.loads(self.invoke("--cookbook", target, "--recipe", "new-site", "--print-settings").stdout)["settings"]
        self.assertEqual(settings["port"], 8081)
        self.assertEqual(settings["strata_settings"]["vision"], {"model": "vocab.gguf", "mmproj": "new.gguf", "gpu": True})
        self.assertIsNone(settings["env"]["FEATURE"])

    def test_equal_value_is_retained_when_its_layer_changes_precedence(self):
        target = self.put("cookbook.json", {"profiles": {"parent": {
            "extends": "inkling-2-mini-strata", "settings": {"sampling": {"temperature": 0.3, "top_k": 5}},
        }}})
        exported = self.put("sample.json", {"version": 1, "profile": "sample", "settings": {
            "sampling": {"temperature": 0.3, "top_k": 5},
            "strata_settings": {"sampling": {"temperature": 0.6}},
        }})
        self.invoke("--config", target, "--import-profile", exported, "--parent-profile", "parent")
        settings = json.loads(target.read_text())["profiles"]["sample"]["settings"]
        self.assertEqual(settings, {"sampling": {"temperature": 0.3}, "strata_settings": {"sampling": {"temperature": 0.6}}})
        applied = json.loads(self.invoke("--config", target, "--profile", "sample", "--print-settings").stdout)
        self.assertEqual(applied["settings"]["sampling"], {"temperature": 0.3, "top_k": 5})

    def test_invalid_exchange_leaves_existing_files_intact(self):
        target = self.put("cookbook.json", {"profiles": {"local": {"settings": {"port": 8888}}}})
        prior = target.read_bytes()
        exported = self.put("sample.json", {"version": 1, "profile": "local", "settings": {"port": 9000}})
        for args, message in (
            (("--import-profile", exported), "already exists"),
            (("--import-profile", exported, "--profile", "new", "--parent-profile", "missing"), "unknown profile"),
            (("--import-profile", exported, "--profile", "new", "--parent-profile", "new"), "its own parent"),
            (("--export-profile", target, "--profile", "local"), "File exists"),
        ):
            with self.subTest(args=args):
                result = self.invoke("--config", target, *args, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)
                self.assertEqual(target.read_bytes(), prior)

    def test_concurrent_imports_create_and_update_a_cookbook(self):
        target = self.root / "cookbook.json"
        exported = self.put("sample.json", {"version": 1, "profile": "sample", "settings": {"port": 9000}})
        with ThreadPoolExecutor(max_workers=2) as executor:
            jobs = [executor.submit(self.invoke, "--cookbook", target, "--import-recipe", exported, "--recipe", name)
                    for name in ("first", "second")]
            for job in jobs:
                job.result()
        profiles = json.loads(target.read_text())["profiles"]
        self.assertEqual(profiles, dict.fromkeys(("first", "second"), {"settings": {"port": 9000}}))

    def test_builtin_can_be_exported_to_stdout(self):
        cookbook = self.put("cookbook.json", {"profiles": {"default": {"settings": {"port": 9000}}}})
        output = self.invoke("--cookbook", cookbook, "--recipe", "inkling-2-mini-strata",
                             "--ignore-default-recipe", "--export-profile", "-")
        self.assertEqual(json.loads(output.stdout), {"version": 1, "profile": "inkling-2-mini-strata",
                                                   "settings": {"backend": "strata-inkling"}})
