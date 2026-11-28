from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from config import arguments, child_environment, json_value, resolve
from scream import (MEGLA_DRAFT, MEGLA_MODEL, engine_options, option_args, replace_options, strata_arguments,
                    strata_native_run, strata_run_config, strata_vision, strata_vision_residency)

LAUNCHER = Path(__file__).with_name("scream.py").absolute()
BUILT_WORKER = LAUNCHER.parents[1] / "build-unified/scream-p100"
VISION_GPU = "GPU-12345678-90ab-cdef-1234-567890abcdef"


class ProfilesTest(unittest.TestCase):
    def setUp(self):
        self.document = {
            "legacy_names": True,
            "settings": {"port": 8000, "env": {"FNX_X": "file", "FNX_Y": "file"}},
            "profiles": {
                "default": {"settings": {"host": "10.20.30.40", "port": 8001}},
                "common": {"extends": "qwen-flash-next-3.8", "settings": {
                    "port": 8002, "engine_options": {"--batch": 2},
                    "strata_settings": {"vision": {"gpu": True, "max_tokens": 400}},
                }},
                "small": {"extends": "common", "settings": {
                    "port": 8003, "engine_options": {"--batch": 1},
                    "strata_settings": {"vision": {"max_tokens": 200}},
                }},
            },
        }

    def test_defaults_and_profile_chain(self):
        settings, controls = resolve(self.document, {}, {})
        self.assertEqual(controls["profile"], "default")
        self.assertEqual(settings["port"], 8001)
        settings, _ = resolve(self.document, {"profile": "small"}, {})
        self.assertEqual((settings["port"], settings["host"], settings["backend"]),
                         (8003, "10.20.30.40", "strata-split"))
        self.assertEqual(settings["engine_options"], {"--batch": 1})
        self.assertEqual(settings["strata_settings"], {"vision": {"gpu": True, "max_tokens": 200}})
        settings, _ = resolve({}, {"profile": "inkling-2-mini-strata"}, {})
        self.assertEqual(settings["backend"], "strata-inkling")

    def test_vision_device_precedence(self):
        document = {"profiles": {"default": {"settings": {"vision_device": "cpu"}},
                                 "custom": {"extends": "inkling-2-mini", "settings": {"vision_device": VISION_GPU}}}}
        settings, _ = resolve(document, {"profile": "custom"}, {"SP100_VISION_DEVICE": "cpu"})
        self.assertEqual(settings["vision_device"], VISION_GPU)
        controls = {"profile": "custom", "environment_precedence": "environment"}
        settings, _ = resolve(document, controls, {"SP100_VISION_DEVICE": "cpu"})
        self.assertEqual(settings["vision_device"], "cpu")
        settings, _ = resolve(document, {**controls, "vision_device": VISION_GPU}, {"SP100_VISION_DEVICE": "cpu"})
        self.assertEqual(settings["vision_device"], VISION_GPU)
        settings, _ = resolve(document, {"ignore_default_profile": True}, {})
        self.assertIsNone(settings["vision_device"])

    def test_vision_residency_is_inherited_with_a_fallback(self):
        for profile in ("inkling-2-mini", "qwen-flash-next-3.8"):
            for ignore in (False, True):
                settings, _ = resolve({}, {"profile": profile, "ignore_default_profile": ignore}, {})
                self.assertEqual(settings["vision_residency"], "auto")

        document = {"legacy_names": True, "profiles": {"child": {
            "extends": "inkling-2-mini", "settings": {"vision_residency": "on-demand"},
        }}}
        cli = {"profile": "child"}
        environment = {"SP100_VISION_RESIDENCY": "resident"}
        settings, _ = resolve(document, cli, environment)
        self.assertEqual(settings["vision_residency"], "on-demand")
        settings, _ = resolve(document, {**cli, "environment_precedence": "environment"}, environment)
        self.assertEqual(settings["vision_residency"], "resident")
        settings, _ = resolve(document, {**cli, "vision_residency": "auto"}, environment)
        self.assertEqual(settings["vision_residency"], "auto")
        settings, _ = resolve(document, {"profile": "inkling-2-mini", "environment_precedence": "environment"},
                              {"INK2_VISION_RESIDENCY": "resident"})
        self.assertEqual(settings["vision_residency"], "resident")
        with self.assertRaisesRegex(ValueError, "vision_residency"):
            resolve({}, {"vision_residency": "always"}, {})

    def test_shadowed_scalar_environment_values_are_never_parsed(self):
        environ = {"SP100_PORT": "obsolete", "SP100_NUM_PASSES": "unset"}
        document = {"profile": "inkling-2-mini", "settings": {"port": 8010, "num_passes": 3}}
        settings, _ = resolve(document, {}, environ)
        self.assertEqual((settings["port"], settings["num_passes"]), (8010, 3))
        settings, _ = resolve(document, {"environment_precedence": "environment", "port": 8011, "num_passes": 2}, environ)
        self.assertEqual((settings["port"], settings["num_passes"]), (8011, 2))
        with self.assertRaises(ValueError):
            resolve(document, {"environment_precedence": "environment"}, environ)

    def test_file_prefix_and_bootstrap_prefix_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            path.write_text(json.dumps({"environment_prefix": "CUSTOM_TEST", "profile": "inkling-2-mini"}))
            environ = {"SP100_CONFIG": str(path), "SP100_PORT": "8901", "CUSTOM_TEST_PORT": "8902"}
            _, doc, cli, _ = arguments([], environ)
            settings, controls = resolve(doc, cli, environ)
            self.assertEqual((controls["environment_prefix"], settings["port"]), ("CUSTOM_TEST", 8902))
            custom = {"CUSTOM_TEST_CONFIG": str(path), "CUSTOM_TEST_PORT": "8903"}
            _, doc, cli, _ = arguments(["--environment-prefix", "CUSTOM_TEST"], custom)
            self.assertEqual(resolve(doc, cli, custom)[0]["port"], 8903)
            legacy = {"SCREAM_CONFIG": str(path)}
            self.assertEqual(arguments([], legacy)[1], {})
            self.assertEqual(arguments(["--legacy-names"], legacy)[1]["profile"], "inkling-2-mini")

    def test_legacy_conflicts_warn_without_values_and_cli_keeps_source_priority(self):
        environ = {"SP100_CONTEXT": "4096", "SCREAM_CONTEXT": "8192"}
        for order, expected_cli, expected_env in (("canonical", 16384, 4096), ("legacy", 32768, 8192)):
            with self.subTest(order=order), redirect_stderr(io.StringIO()) as stderr:
                flags = ["--legacy-names", "--legacy-name-precedence", order]
                _, doc, cli, _ = arguments([*flags, "--context", "16384", "--ctx", "32768"], environ)
                self.assertEqual(resolve(doc, cli, environ)[0]["context"], expected_cli)
                _, doc, cli, _ = arguments(flags, environ)
                self.assertEqual(resolve(doc, cli, environ)[0]["context"], expected_env)
                message = stderr.getvalue()
                for name in ("SP100_CONTEXT", "SCREAM_CONTEXT", "--context", "--ctx"):
                    self.assertIn(name, message)
                for value in ("4096", "8192", "16384", "32768"):
                    self.assertNotIn(value, message)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            arguments(["--ctx", "32768"], {})
        self.assertIsNone(resolve({}, {}, {"SCREAM_CONTEXT": "8192"})[0]["context"])

    def test_worker_option_names_and_aliases(self):
        environ = {"SP100_ENGINE_OPTIONS__PARALLEL": "3", "SP100_ENV__IK2_LOG_LEVEL": "verbose",
                   "IK2_LOG_LEVEL": "legacy", "STRATA_API_KEY": "required"}
        argv = ["--profile", "inkling-2-mini-strata", "--engine-options", '{"parallel":4}', "--batch", "5"]
        for order, expected_flag, expected_value, expected_env in (
                ("canonical", "--parallel", 4, "verbose"), ("legacy", "--parallel", ["5"], "legacy")):
            with self.subTest(order=order), redirect_stderr(io.StringIO()) as stderr:
                _, doc, cli, _ = arguments([*argv, "--legacy-names", "--legacy-name-precedence", order], environ)
                settings, controls = resolve(doc, cli, environ)
                self.assertEqual(settings["engine_options"], {expected_flag: expected_value})
                child = child_environment(settings, controls, environ, cli)
                self.assertEqual(child["IK2_LOG_LEVEL"], expected_env)
                for name in ("--batch", "SP100_ENV__IK2_LOG_LEVEL"):
                    self.assertIn(name, stderr.getvalue())
        doc = {"profile": "inkling-2-mini-strata", "settings": {"engine_options": {"parallel": 2},
                                                               "env": {"ik2_log_level": "file"}}}
        cli = {"environment_precedence": "environment"}
        settings, controls = resolve(doc, cli, environ)
        self.assertEqual(settings["engine_options"], {"--parallel": 3})
        child = child_environment(settings, controls, environ, cli)
        self.assertEqual(child["IK2_LOG_LEVEL"], "verbose")
        self.assertEqual(child["STRATA_API_KEY"], "required")
        settings, controls = resolve({}, {}, {"FNX_ARENA": "0"})
        self.assertNotIn("FNX_ARENA", child_environment(settings, controls, {"FNX_ARENA": "0"}, {}))

    def test_backend_aliases_follow_the_backend_and_translate_legacy_switches(self):
        environ = {"INK2_VISION_DEVICE": "cpu", "INK2_STRATA_SAMPLER": "1", "INK2_STREAM_TOOL_ARGUMENTS": "1"}
        settings, _ = resolve({"profile": "inkling-2-mini", "legacy_names": True}, {}, environ)
        self.assertEqual((settings["vision_device"], settings["sampler"], settings["tool_streaming"]),
                         ("cpu", "strata-inkling", "incremental"))
        settings, _ = resolve({"profile": "qwen-flash-next-3.8", "legacy_names": True}, {}, environ)
        self.assertEqual((settings["vision_device"], settings["sampler"], settings["tool_streaming"]),
                         (None, "native", "native"))
        cli = {"vision_device": None, "tool_streaming": "native"}
        settings, controls = resolve({"profile": "inkling-2-mini", "legacy_names": True}, cli, environ)
        self.assertEqual((settings["vision_device"], settings["tool_streaming"]), (None, "native"))
        self.assertFalse(environ.keys() & child_environment(settings, controls, environ, cli).keys())
        environ["INK2_SAMPLING_DEFAULTS"] = '{"repeat_penalty":1.1,"repeat_last_n":37}'
        settings, _ = resolve({"profile": "inkling-2-mini", "legacy_names": True}, {}, environ)
        self.assertEqual(settings["sampling"], {"repetition_penalty": 1.1, "penalty_last_n": 37})

    def test_backend_flag_alias_precedence_keeps_resets_and_reports_collisions(self):
        doc = {"profile": "inkling-2-mini-strata", "settings": {"engine_options": {"gpus": 2}}}
        settings, _ = resolve(doc, {"engine_options": {"devices": None}}, {})
        saved = {"args": ["--gpus", "2"]}
        strata_arguments(saved, engine_options(settings), "strata-inkling")
        self.assertEqual(saved["args"], [])
        base = ["--profile", "inkling-2-mini-strata", "--legacy-names", "--mode", "native"]
        with redirect_stderr(io.StringIO()) as warnings:
            _, doc, cli, _ = arguments([*base, "--engine-options", '{"parallel":2,"--slots":4,"repetition_guard":"on"}',
                                        "--no-anti-loop"], {})
            settings, _ = resolve(doc, cli, {})
            self.assertEqual(settings["engine_options"], {"--parallel": 2, "--repetition-guard": "on"})
            self.assertIn("--no-anti-loop", warnings.getvalue())
            self.assertIn("--slots", warnings.getvalue())
            _, doc, cli, _ = arguments([*base, "--legacy-name-precedence", "legacy", "--set", "sampling.temperature=0.2",
                                        "--temp", "0.8"], {})
            self.assertEqual(resolve(doc, cli, {})[0]["sampling"], {"temperature": 0.8})
        _, doc, cli, _ = arguments([*base, "--batch", "2", "--parallel", "4", "--batch", "6"], {})
        self.assertEqual(resolve(doc, cli, {})[0]["engine_options"], {"--parallel": ["6"]})
        env = {"SP100_ENGINE_OPTIONS": '{"parallel":2}', "SP100_ENGINE_OPTIONS__BATCH": "4"}
        self.assertEqual(resolve({"profile": "inkling-2-mini-strata"}, {}, env)[0]["engine_options"], {"--parallel": 4})

    def test_strata_user_args_preserve_cli_priority_over_lower_engine_options(self):
        for mode in ("serve", "native"):
            doc = {"profile": "inkling-2-mini-strata"}
            cli = {"mode": mode, "strata_settings": {"user_args": ["--slots", "4"]}}
            env = {"SP100_ENGINE_OPTIONS__PARALLEL": "2"}
            settings, _ = resolve(doc, cli, env)
            cfg = strata_run_config({}, settings["strata_settings"])
            strata_arguments(cfg, engine_options(settings), "strata-inkling")
            self.assertEqual(cfg["args"], ["--parallel", "4"])

    def test_flash_rope_flag_and_environment_value_use_source_precedence(self):
        doc = {"profile": "qwen-flash-next-3.8", "legacy_names": True,
               "settings": {"engine_options": {"--rope-scale": 3}}}
        environ = {"FNX_ROPE_SCALE": "2"}
        settings, controls = resolve(doc, {}, environ)
        self.assertEqual(settings["engine_options"]["--rope-scale"], 3)
        self.assertNotIn("FNX_ROPE_SCALE", child_environment(settings, controls, environ, {}))
        cli = {"environment_precedence": "environment"}
        settings, controls = resolve(doc, cli, environ)
        self.assertIsNone(settings["engine_options"]["--rope-scale"])
        self.assertEqual(child_environment(settings, controls, environ, cli)["FNX_ROPE_SCALE"], "2")
        cli["engine_options"] = {"--rope-scale": 4}
        settings, controls = resolve(doc, cli, environ)
        self.assertEqual(settings["engine_options"]["--rope-scale"], 4)
        self.assertNotIn("FNX_ROPE_SCALE", child_environment(settings, controls, environ, cli))

    def test_native_precedence_in_strata_saved_flags(self):
        doc = {"profile": "inkling-2-mini-strata", "legacy_names": True}
        saved = {"args": ["--gpus", "2"], "user_args": ["--ep-split", "0.25"], "args_written": ["--gpus", "2"]}
        environ = {"IK2_GPUS": "1", "IK2_EP_SPLIT": "0.5", "STRATA_INK_SPEC_BEAM": "3"}
        settings, controls = resolve(doc, {}, environ, run_config=saved)
        self.assertEqual(child_environment(settings, controls, environ, {}), {"STRATA_INK_SPEC_BEAM": "3"})
        cli = {"environment_precedence": "environment", "engine_options": {"--devices": "0", "--draft-beam": 2}}
        settings, controls = resolve(doc, cli, environ, run_config=saved)
        self.assertEqual(child_environment(settings, controls, environ, cli), {"IK2_EP_SPLIT": "0.5"})
        self.assertEqual(settings["engine_options"], {"--devices": "0", "--draft-beam": 2, "--ep-split": None})
        strata_arguments(saved, engine_options(settings), "strata-inkling")
        for key in ("args", "user_args", "args_written"):
            self.assertEqual(saved[key], ["--devices", "0", "--draft-beam", "2"])

    def test_strata_user_arg_bindings_follow_their_unified_source(self):
        saved = {"args": ["--gpus", "2", "--ep-split", "0.2"],
                 "args_written": ["--devices", "1,2", "--ep-split", "0.25"]}
        user_args = ["--devices", "0,2", "--ep-split", "0.4", "--cpu"]
        doc = {"profile": "inkling-2-mini-strata", "legacy_names": True,
               "settings": {"strata_settings": {"user_args": user_args}}}
        env = {"IK2_GPUS": "1", "IK2_EP_SPLIT": "0.3"}
        for overrides, env_wins in (({}, False), ({"environment_precedence": "environment"}, True),
                                   ({"environment_precedence": "environment", "strata_settings": {"user_args": user_args}}, False)):
            with self.subTest(overrides=overrides):
                settings, controls = resolve(doc, overrides, env, run_config=saved)
                configured = strata_run_config(saved, settings["strata_settings"])
                strata_arguments(configured, engine_options(settings), "strata-inkling")
                self.assertEqual(child_environment(settings, controls, env, overrides), env if env_wins else {})
                args = configured["args"]
                self.assertEqual(len(args), 0 if env_wins else 4)
                self.assertEqual(dict(zip(args[::2], args[1::2])), {} if env_wins else {"--devices": "0,2", "--ep-split": "0.4"})
                self.assertEqual(configured["user_args"], ["--cpu", *args])
                self.assertEqual(configured["args_written"], args)

    def test_repetition_guard_replaces_legacy_strata_option(self):
        doc = {"profile": "inkling-2-mini-strata", "settings": {"engine_options": {"--no-anti-loop": True}}}
        env = {"SP100_ENGINE_OPTIONS": '{"--repetition-guard":"on"}'}
        settings, _ = resolve(doc, {}, env)
        self.assertEqual(engine_options(settings), {"--repetition-guard": "off"})
        settings, _ = resolve(doc, {"environment_precedence": "environment"}, env)
        self.assertEqual(engine_options(settings), {"--repetition-guard": "on"})
        settings, _ = resolve(doc, {"engine_options": {"--repetition-guard": "on"}}, {})
        saved = {"args": ["--no-anti-loop", "--parallel", "2"], "user_args": ["--no-anti-loop"]}
        strata_arguments(saved, engine_options(settings), "strata-inkling")
        self.assertEqual(saved["args"], ["--parallel", "2", "--repetition-guard", "on"])
        self.assertEqual(saved["user_args"], ["--repetition-guard", "on"])

    def test_profile_environment_and_flags_resolve_in_ancestor_order(self):
        doc = {"profile": "child", "profiles": {
            "default": {"settings": {"engine_options": {"--draft-cost": 0.25, "--gpus": 2}}},
            "child": {"extends": "inkling-2-mini-strata",
                      "settings": {"env": {"STRATA_INK_SPEC_COST": "0.5"}, "engine_options": {"--devices": "0"}}},
        }}
        settings, _ = resolve(doc, {}, {})
        self.assertEqual(settings["engine_options"], {"--devices": "0", "--gpus": None, "--draft-cost": None})
        self.assertEqual(settings["env"], {"STRATA_INK_SPEC_COST": "0.5"})

    def test_inkling_can_remove_saved_gpu_selection_or_ignore_blank_environment(self):
        doc = {"profile": "inkling-2-mini-strata", "legacy_names": True}
        saved = {"args": ["--devices", "2"]}
        env = {"IK2_GPUS": "3"}
        cli = {"engine_options": {"--gpus": None}}
        settings, controls = resolve(doc, cli, env, run_config=saved)
        self.assertEqual(child_environment(settings, controls, env, cli)["IK2_GPUS"], "3")
        strata_arguments(saved, engine_options(settings), "strata-inkling")
        self.assertEqual(saved["args"], [])
        doc["settings"] = {"engine_options": {"--gpus": 2}}
        settings, _ = resolve(doc, {"environment_precedence": "environment"}, {"IK2_GPUS": ""})
        self.assertEqual(settings["engine_options"]["--gpus"], 2)

    def test_strata_inkling_flag_aliases_follow_source_order(self):
        doc = {"profile": "child", "profiles": {
            "default": {"settings": {"engine_options": {"--parallel": 2, "--kv": "auto"}}},
            "child": {"extends": "inkling-2-mini-strata", "settings": {"engine_options": {"--batch": 4, "--kv-dtype": "fp8"}}},
        }}
        env = {"SP100_ENGINE_OPTIONS": '{"--slots":8,"--kv":"k8v4"}'}
        for mode in ("native", "serve"):
            with self.subTest(mode=mode):
                settings, _ = resolve(doc, {"mode": mode}, env)
                self.assertEqual(settings["engine_options"], {"--parallel": 4, "--kv": "fp8"})
                controls = {"mode": mode, "environment_precedence": "environment"}
                settings, _ = resolve(doc, controls, env)
                self.assertEqual(settings["engine_options"], {"--parallel": 8, "--kv": "k8v4"})
                settings, _ = resolve(doc, {**controls, "engine_options": {"--parallel": 1, "--kv-dtype": None}}, env)
                config = {"args": ["--slots", "2", "--kv-dtype", "fp8", "--gpus", "2"], "user_args": ["--batch", "3"]}
                strata_arguments(config, engine_options(settings), "strata-inkling")
                self.assertEqual(config["args"], ["--gpus", "2", "--parallel", "1"])
                self.assertEqual(config["user_args"], ["--parallel", "1"])

    def test_strata_vision_device_only_scopes_reader(self):
        worker_env = {"CUDA_VISIBLE_DEVICES": "0,1"}
        cfg = {"backend": "cuda", "env": worker_env, "vision": {"exe": "strata-vision", "model": "text.gguf",
               "mmproj": "vision.gguf", "env": {"CUDA_VISIBLE_DEVICES": "0", "STRATA_VIT_PROF": "1"}}}
        strata_vision(cfg, VISION_GPU, "strata-split")
        self.assertEqual(cfg["env"], worker_env)
        self.assertEqual(cfg["vision"]["cuda_device"], VISION_GPU)
        self.assertEqual(cfg["vision"]["env"]["CUDA_VISIBLE_DEVICES"], VISION_GPU)
        self.assertEqual(cfg["vision"]["env"]["STRATA_VIT_PROF"], "1")
        self.assertTrue(cfg["vision"]["gpu"] and cfg["vision"]["require_gpu"])
        strata_vision(cfg, "cpu", "strata")
        self.assertIsNone(cfg["vision"]["cuda_device"])
        self.assertEqual(cfg["vision"]["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertFalse(cfg["vision"]["gpu"] or cfg["vision"]["require_gpu"])

    def test_strata_vision_lifecycle_uses_final_reader_placement(self):
        cfg = {"vision": {"gpu": True, "elastic": True, "elastic_idle_s": 4}}
        self.assertTrue(strata_vision_residency(cfg, "auto"))
        self.assertEqual(cfg["vision"], {"gpu": True, "elastic": False, "elastic_idle_s": 4, "scream_resident": True})
        self.assertTrue(strata_vision_residency(cfg, "on-demand"))
        self.assertTrue(cfg["vision"]["elastic"])
        self.assertFalse(cfg["vision"]["scream_resident"])
        self.assertEqual(cfg["vision"]["elastic_idle_s"], 4)
        cfg = {"vision": {"gpu": False, "elastic": False}}
        self.assertFalse(strata_vision_residency(cfg, "auto"))
        self.assertNotIn("scream_resident", cfg["vision"])
        self.assertTrue(strata_vision_residency(cfg, "resident"))
        self.assertTrue(cfg["vision"]["scream_resident"])
        cfg = {"args": ["--vision", "native.strata"]}
        self.assertFalse(strata_vision_residency(cfg, "auto"))

    def test_strata_inkling_accepts_http_gpu_projector(self):
        result = subprocess.run([sys.executable, LAUNCHER, "--profile", "inkling-2-mini-strata", "--vision-device", VISION_GPU,
                                 "--print-settings"], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["settings"]["vision_device"], VISION_GPU)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("reader", "text.gguf", "projector.gguf"):
                (root / name).touch()
            (root / "reader").chmod(0o700)
            worker_env = {"CUDA_VISIBLE_DEVICES": "0,1"}
            cfg = {"cwd": directory, "backend": "cpu", "env": worker_env,
                   "vision": {"exe": "reader", "model": "text.gguf", "mmproj": "projector.gguf"}}
            strata_vision(cfg, VISION_GPU, "strata-inkling")
            self.assertEqual(cfg["env"], worker_env)
            self.assertTrue(cfg["vision"]["inkling_embeddings"] and cfg["vision"]["require_gpu"])
            self.assertEqual(cfg["vision"]["env"]["CUDA_VISIBLE_DEVICES"], VISION_GPU)
            self.assertEqual(cfg["vision"]["mmproj"], str(root / "projector.gguf"))
        with self.assertRaisesRegex(ValueError, "vision.mmproj and vision.model"):
            strata_vision({}, VISION_GPU, "strata-inkling")
        cfg = {"args": ["--vision", "vision.strata"]}
        strata_vision(cfg, "cpu", "strata-inkling")
        self.assertEqual(cfg, {"args": ["--vision", "vision.strata"]})

    def test_ignore_implicit_default(self):
        self.document["ignore_default_profile"] = True
        settings, _ = resolve(self.document, {"profile": "small"}, {})
        self.assertEqual((settings["host"], settings["port"]), ("127.0.0.1", 8003))
        settings, _ = resolve(self.document, {"profile": "small", "ignore_default_profile": False}, {})
        self.assertEqual(settings["host"], "10.20.30.40")
        settings, _ = resolve(self.document, {"profile": "default"}, {})
        self.assertEqual(settings["host"], "10.20.30.40")
        self.document["profiles"]["explicit"] = {"extends": "default"}
        settings, _ = resolve(self.document, {"profile": "explicit"}, {})
        self.assertEqual((settings["host"], settings["port"]), ("10.20.30.40", 8001))

    def test_imported_joined_option_obeys_environment_precedence(self):
        doc = {"profile": "qwen-flash-next-3.8", "legacy_names": True}
        saved = {"args": ["--serve", "--rope-scale=2", "--batch=2"]}
        env = {"FNX_ROPE_SCALE": "3"}
        for order, expected_env, expected_args in (
            ("config", {}, ["--serve", "--rope-scale", "2", "--batch", "2"]),
            ("environment", env, ["--serve", "--batch", "2"]),
        ):
            with self.subTest(order=order):
                cli = {"environment_precedence": order}
                settings, controls = resolve(doc, cli, env, run_config=saved)
                cfg = dict(saved)
                strata_arguments(cfg, engine_options(settings), "strata-split")
                self.assertEqual(cfg["args"], expected_args)
                self.assertEqual(child_environment(settings, controls, env, cli), expected_env)

    def test_environment_order_and_cli_overrides(self):
        self.document["profile"] = "common"
        environ = {"SP100_PORT": "9000", "SP100_PROFILE": "small", "FNX_X": "exported", "FNX_Y": "exported",
                   "SP100_ENV": '{"FNX_X":"structured"}'}
        settings, controls = resolve(self.document, {}, environ)
        self.assertEqual((settings["port"], controls["profile"]), (8002, "common"))
        self.assertEqual(child_environment(settings, controls, environ, {})["FNX_X"], "file")
        cli = {"environment_precedence": "environment", "env": {"FNX_Y": None}, "port": 9080}
        settings, controls = resolve(self.document, cli, environ)
        self.assertEqual((settings["port"], controls["profile"]), (9080, "small"))
        env = child_environment(settings, controls, environ, cli)
        self.assertEqual(env["FNX_X"], "structured")
        self.assertNotIn("FNX_Y", env)

    def test_profile_errors(self):
        self.document["profiles"]["common"]["extends"] = "small"
        with self.assertRaisesRegex(ValueError, "cycle"):
            resolve(self.document, {}, {})
        self.document["profiles"]["common"]["extends"] = "does-not-exist"
        with self.assertRaisesRegex(ValueError, "unknown profile"):
            resolve(self.document, {}, {})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            json_value('{"port": 1, "port": 2}')

    def test_sampling_defaults_follow_profile_precedence(self):
        document = {
            "profile": "sampled",
            "profiles": {
                "default": {"settings": {"sampling": {"top_k": 32}}},
                "sampled": {"extends": "inkling-2-mini-megla-sampling",
                            "settings": {"sampling": {"temperature": 0.7}}},
            },
        }
        settings, _ = resolve(document, {"sampling": {"temperature": 0.5}},
                              {"SP100_SAMPLING": '{"temperature": 0.2, "min_p": 0.1}'})
        self.assertEqual(settings["sampler"], "strata-inkling")
        args = list(option_args(engine_options(settings)))
        self.assertEqual(args[:3], ["--sampler", "strata-inkling", "--sampling"])
        supplied = [float(part) for part in args[3].split()]
        self.assertEqual(supplied[:3], [0.5, 1.0, 32])
        self.assertAlmostEqual(supplied[3], 0.1)
        self.assertEqual(supplied[4:], [1.0, 0.0, 0.0, 64, 0])
        self.assertEqual(settings["sampling"], {"top_k": 32, "temperature": 0.5, "min_p": 0.1})
        with self.assertRaisesRegex(ValueError, "requires an Inkling execution path"):
            resolve(document, {"backend": "strata"}, {})

    def test_native_strata_inkling_receives_profile_sampling(self):
        settings, _ = resolve({}, {"profile": "inkling-2-mini-strata", "mode": "native",
                                  "sampling": {"temperature": 0.5, "top_k": 128, "seed": 81}}, {})
        self.assertEqual(list(option_args(engine_options(settings))),
                         ["--sampling", "0.5 1.0 128 0.0 1.0 0.0 0.0 64 81"])

    def test_native_flash_sampling_keeps_unset_values_sparse(self):
        for profile in ("qwen-flash-next-3.8", "qwen-flash-next-3.8-strata"):
            with self.subTest(profile=profile):
                document = {"profile": profile, "settings": {"sampling": {"temperature": 0.5, "seed": 0}}}
                settings, _ = resolve(document, {"mode": "native"}, {})
                self.assertEqual(engine_options(settings), {"--sampling": "0.5 - - - - - - - 0"})
                settings, _ = resolve(document, {"mode": "native", "sampling": {"seed": None}}, {})
                self.assertEqual(list(option_args(engine_options(settings))), ["--sampling", "0.5 - - - - - - - -"])
                settings, _ = resolve(document, {}, {})
                self.assertEqual(engine_options(settings), {"--sampling": "- - - - - - - - 0"})

    def test_native_flash_layer_sampler_flags_follow_profile_precedence(self):
        document = {"profile": "qwen-flash-next-3.8-strata", "settings": {
            "sampling": {"temperature": 0.5, "seed": 5}, "engine_options": {"--top-k": 0}}}
        cli = {"mode": "native", "engine_options": {"--greedy": True, "--seed": 0}}
        env = {"SP100_ENGINE_OPTIONS": '{"--top-k":32,"--temperature":0.25}'}
        settings, _ = resolve(document, cli, env)
        self.assertEqual(engine_options(settings), {"--seed": 0, "--sampling": "0.0 - 64 - - - - - 0"})
        settings, _ = resolve(document, {**cli, "environment_precedence": "environment"}, env)
        self.assertEqual(engine_options(settings), {"--seed": 0, "--sampling": "0.0 - 32 - - - - - 0"})
        document["settings"].pop("sampling")
        settings, _ = resolve(document, cli, {})
        self.assertEqual(engine_options(settings), {"--top-k": 0, "--greedy": True, "--seed": 0})

    def test_native_flash_layer_unrelated_sampling_fields_preserve_legacy_mode(self):
        for old_flags, expected_prefix, seed in (
            ({"--seed": 71}, ["--seed", "71"], "71"),
            ({"--temperature": 0.75}, ["--temperature", "0.75"], "-"),
            ({"--seed": 71, "--greedy": True}, ["--seed", "71", "--greedy"], "71"),
        ):
            with self.subTest(old_flags=old_flags):
                doc = {"profile": "qwen-flash-next-3.8-strata", "settings": {"engine_options": old_flags}}
                settings, _ = resolve(doc, {"mode": "native", "sampling": {"presence_penalty": 0.5}}, {})
                self.assertEqual(list(option_args(engine_options(settings))),
                                 [*expected_prefix, "--sampling", f"- - - - - - 0.5 - {seed}"])

    @unittest.skipUnless(BUILT_WORKER.is_file(), "build the unified native executable first")
    def test_native_flash_layer_user_args_sampling_uses_source_order(self):
        doc = {"profile": "qwen-flash-next-3.8-strata", "settings": {
            "strata_settings": {"user_args": ["--temperature=0.7"]}}}
        saved = {"sampling": {"temperature": 0.2}, "args": ["--help"]}
        env = {"SP100_SAMPLING": '{"temperature":0.4}'}
        for order, override, expected in (("config", {}, 0.7), ("environment", {}, 0.4),
                                           ("environment", {"strata_settings": {"user_args": ["--temperature=0.8"]}}, 0.8)):
            with self.subTest(order=order, override=override):
                cli = {"mode": "native", "environment_precedence": order, **override}
                settings, _ = resolve(doc, cli, env, run_config=saved)
                native, _ = strata_native_run(settings, BUILT_WORKER, engine_options(settings), saved)
                self.assertNotIn("--temperature", native)
                self.assertAlmostEqual(float(native[native.index("--sampling") + 1].split()[0]), expected)

    def test_strata_saved_and_nested_sampling_obey_config_precedence(self):
        doc = {"profile": "qwen-flash-next-3.8", "settings": {"strata_settings": {
            "sampling": {"temperature": 0.5, "repeat_last_n": 96}}}}
        saved = {"sampling": {"temperature": 0.125, "repeat_penalty": 1.25, "top_k": 24.0}}
        env = {"SP100_SAMPLING": '{"temperature":0.25,"top_k":12}'}
        for mode in ("serve", "native"):
            with self.subTest(mode=mode):
                settings, _ = resolve(doc, {"mode": mode}, env, run_config=saved)
                self.assertEqual(settings["sampling"], {"temperature": 0.5, "repetition_penalty": 1.25,
                                                        "top_k": 24, "penalty_last_n": 96})
                settings, _ = resolve(doc, {"mode": mode, "environment_precedence": "environment",
                                            "sampling": {"top_k": 32}}, env, run_config=saved)
                self.assertEqual(settings["sampling"], {"temperature": 0.25, "repetition_penalty": 1.25,
                                                        "top_k": 32, "penalty_last_n": 96})

    def test_strata_temperature_alias_respects_source_precedence(self):
        document = {"profile": "inkling-2-mini-strata", "settings": {
            "mode": "native", "engine_options": {"--temp": 0.25}}}
        environ = {"SP100_SAMPLING": '{"temperature":0.5}'}
        settings, _ = resolve(document, {}, environ)
        self.assertEqual(settings["sampling"]["temperature"], 0.25)
        self.assertNotIn("--temp", settings["engine_options"])
        settings, _ = resolve(document, {"environment_precedence": "environment"}, environ)
        self.assertEqual(settings["sampling"]["temperature"], 0.5)
        settings, _ = resolve(document, {"sampling": {"temperature": 0.75}}, environ)
        self.assertEqual(settings["sampling"]["temperature"], 0.75)
        document["settings"]["sampling"] = {"temperature": 0.25}
        settings, _ = resolve(document, {"engine_options": {"--temp": 0.0}}, environ)
        self.assertEqual(settings["sampling"]["temperature"], 0.0)

    def test_tool_streaming_follows_precedence_for_inkling(self):
        document = {"profile": "inkling-2-mini-strata", "settings": {"tool_streaming": "incremental"}}
        environ = {"SP100_TOOL_STREAMING": "buffered"}
        settings, _ = resolve(document, {}, environ)
        self.assertEqual(settings["tool_streaming"], "incremental")
        settings, _ = resolve(document, {"environment_precedence": "environment"}, environ)
        self.assertEqual(settings["tool_streaming"], "buffered")
        settings, _ = resolve(document, {"environment_precedence": "environment", "tool_streaming": "native"}, environ)
        self.assertEqual(settings["tool_streaming"], "native")
        with self.assertRaisesRegex(ValueError, "requires an Inkling execution path"):
            resolve(document, {"backend": "strata"}, {})

    def test_cli_replaces_nested_value_and_pass_count(self):
        result = subprocess.run(
            [sys.executable, LAUNCHER, "--profile", "inkling-2-mini", "--num-passes", "2",
             "--set", "engine_options.--mtp=null", "--set", 'engine_options.--draft-vocab="a file"',
             "--print-settings"], text=True, capture_output=True, check=True,
        )
        options = engine_options(json.loads(result.stdout)["settings"])
        self.assertEqual(list(option_args(options)),
                         ["--draft-vocab", "a file", "--num-passes", "2"])

    def test_flag_serialization_and_legacy_alias_replacement(self):
        args = ["--model", "old path", "--draft", "auto", "--slots", "2"]
        options = {"--native": "new path", "--draft": None, "--script": [["one"], ["two"]]}
        self.assertEqual(
            replace_options(args, options, ({"--model", "--native"},)),
            ["--slots", "2", "--native", "new path", "--script", "one", "--script", "two"],
        )


class LaunchTest(unittest.TestCase):
    @unittest.skipUnless(BUILT_WORKER.is_file(), "build the unified native executable first")
    def test_staged_runtime_can_be_moved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, moved = root / "first prefix", root / "moved prefix"
            subprocess.run(["cmake", "--install", BUILT_WORKER.parent, "--prefix", first, "--component", "scream-runtime"],
                           check=True, capture_output=True, text=True, timeout=30)
            first.rename(moved)
            worker = moved / "bin/scream-p100"
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith(("SCREAM_", "SP100_")) and key != "PYTHONPATH"}
            profiles = moved / "libexec/scream-p100/unified/builtin_profiles.json"
            source = LAUNCHER.with_name(profiles.name)
            self.assertEqual(profiles.read_bytes(), source.read_bytes())
            self.assertEqual(BUILT_WORKER.with_name("scream-builtin-profiles.json").read_bytes(), source.read_bytes())
            reference = profiles.with_name("settings_reference.json")
            self.assertEqual(reference.read_bytes(), LAUNCHER.with_name(reference.name).read_bytes())
            helped = subprocess.run([worker, "--help-settings", "vision_device"], cwd=root, env=env,
                                    check=True, capture_output=True, text=True, timeout=10)
            self.assertIn("SP100_VISION_DEVICE", helped.stdout)
            self.assertIn("settings.vision_device", helped.stdout)
            reference.rename(reference.with_suffix(".unused"))
            definitions = json.loads(profiles.read_text())
            definitions["packaging-fixture"] = {"extends": "inkling-2-mini-strata", "settings": {"port": 8124}}
            profiles.write_text(json.dumps(definitions))
            resolved = subprocess.run([worker, "--profile", "inkling-2-mini-strata", "--print-settings"],
                                      cwd=root, env=env, check=True, capture_output=True, text=True, timeout=10)
            self.assertEqual(json.loads(resolved.stdout)["settings"]["binary"], str(worker))
            resolved = subprocess.run([worker, "--profile", "packaging-fixture", "--print-settings"],
                                      cwd=root, env=env, check=True, capture_output=True, text=True, timeout=10)
            settings = json.loads(resolved.stdout)["settings"]
            self.assertEqual((settings["backend"], settings["port"]), ("strata-inkling", 8124))
            legacy = root / "strata.json"
            legacy.write_text(json.dumps({"args": ["--version"]}), encoding="utf-8")
            version = subprocess.run([worker, "--profile", "inkling-2-mini-strata", "--mode", "native", "--strata-config", legacy],
                                     cwd=root, env=env, check=True, capture_output=True, text=True, timeout=10)
            self.assertIn("strata-inkling", version.stdout)

    @unittest.skipUnless(BUILT_WORKER.is_file(), "build the unified native executable first")
    def test_executable_resolves_profiles_and_preserves_raw_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            alias = root / "engine"
            alias.symlink_to(BUILT_WORKER)
            config = root / "profiles.json"
            config.write_text(json.dumps({"profiles": {"default": {"settings": {
                "backend": "strata-inkling", "port": 8123,
            }}}}), encoding="utf-8")
            env = {key: value for key, value in os.environ.items() if not key.startswith("SCREAM_")}
            env.update(SP100_CONFIG=str(config), SP100_PORT="9000")
            resolved = subprocess.run([alias, "--print-settings"], cwd=root, env=env,
                                      check=True, capture_output=True, text=True, timeout=10)
            settings = json.loads(resolved.stdout)["settings"]
            self.assertEqual((settings["backend"], settings["port"], settings["binary"]),
                             ("strata-inkling", 8123, str(BUILT_WORKER.resolve())))
            configured = subprocess.run(
                [alias, "--configured", "--backend", "strata-inkling", "--mode", "native",
                 "--set", "engine_options.--version=true"],
                cwd=root, env=env, check=True, capture_output=True, text=True, timeout=10,
            )
            raw = subprocess.run([alias, "--backend", "strata-inkling", "--version"], cwd=root,
                                 check=True, capture_output=True, text=True, timeout=10)
            self.assertEqual(configured.stdout, raw.stdout)
            self.assertIn("strata-inkling", raw.stdout)

    def test_built_split_worker_receives_cli_rope_override(self):
        if not BUILT_WORKER.is_file():
            self.skipTest("the native unified worker has not been built")
        env = {**os.environ, "FNX_ROPE_SCALE": "2"}
        raw = subprocess.run([BUILT_WORKER, "--backend", "strata-split", "--rope-scale", "3"],
                             env=env, capture_output=True, text=True, timeout=10)
        self.assertIn("and FNX_ROPE_SCALE=2 disagree", raw.stderr)
        configured = subprocess.run([BUILT_WORKER, "--legacy-names", "--profile", "qwen-flash-next-3.8", "--mode", "native",
                                     "--environment-precedence", "environment", "--set", "engine_options.--rope-scale=3",
                                     "--set", "engine_options.--serve=true"],
                                    env=env, capture_output=True, text=True, timeout=10)
        self.assertIn("--native is required", configured.stderr)

    def test_built_split_worker_requires_mtp_only_with_model_drafting(self):
        if not BUILT_WORKER.is_file():
            self.skipTest("the native unified worker has not been built")
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "absent.gguf"
            prefix = [BUILT_WORKER, "--profile", "qwen-flash-next-3.8", "--mode", "native", "--model", model,
                      "--set", "engine_options.--serve=true"]
            disabled = subprocess.run([*prefix, "--set", "engine_options.--spec=0"], capture_output=True, text=True, timeout=10)
            enabled = subprocess.run([*prefix, "--set", "engine_options.--spec=1"], capture_output=True, text=True, timeout=10)
            self.assertIn(f"gguf: cannot open {model}", disabled.stdout + disabled.stderr)
            self.assertIn("--mtp is required when --spec is above 0", enabled.stderr)

    def test_staged_inkling_planner_keeps_explicit_native_selection(self):
        recorded = BUILT_WORKER.parent / "strata-terminal.path"
        if not recorded.is_file():
            self.skipTest("the staged Strata adapter has not been built")
        terminal = Path(recorded.read_text().strip())
        check = """
from unittest.mock import patch
from serve import inkling_kind

common = {'gpu':'0,1', 'backend':'cuda', 'parallel':16, 'drafting':'dspark'}
with patch.object(inkling_kind, 'probe', return_value={'drafter':'dspark', 'spec_opts':'1'}):
    configured = {**common, 'args':['--native','model.strata','--slots','2'],
                  'env':{'IK2_GPUS':'0,1','IK2_EP_SPLIT':'0.3','STRATA_INK_SPEC_BEAM':'3'}}
    args = inkling_kind.engine_args(configured, 'worker', announce=False)
    assert '--gpus' not in args and '--layer-split' not in args, args
    assert '--parallel' not in args and '--draft-beam' not in args, args
    assert args[args.index('--draft')+1] == 'dspark', args
    configured = {**common, 'args':['--native','model.strata','--devices','0,1','--draft-beam','2']}
    args = inkling_kind.engine_args(configured, 'worker', announce=False)
    assert '--gpus' not in args and args.count('--draft-beam') == 1, args
    assert args[args.index('--draft-beam')+1] == '2', args
"""
        result = subprocess.run([sys.executable, "-c", check], cwd=terminal, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_staged_inkling_probe_uses_the_worker_directory(self):
        recorded = BUILT_WORKER.parent / "strata-terminal.path"
        if not recorded.is_file():
            self.skipTest("the staged Strata adapter has not been built")
        terminal = Path(recorded.read_text().strip())
        check = """
from pathlib import Path
import sys
from serve import inkling_kind

root = Path(sys.argv[1])
worker = root / 'strata-inkling'
models = root / 'models'
args = ['--native', 'model.strata', '--vision', 'vision.strata']
assert inkling_kind.probe(str(worker), args, {'cwd': str(root)}) is None
cfg = {'args': args, 'cwd': str(models)}
assert inkling_kind.probe(str(worker), args, cfg)['images'] == '1'
assert '--vision' in inkling_kind.engine_args(cfg, str(worker), announce=False)
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            models.mkdir()
            (models / "model.strata").touch()
            (models / "vision.strata").touch()
            worker = root / "strata-inkling"
            worker.write_text("#!/bin/sh\n[ -f \"$2\" ] && [ -f \"$4\" ] || exit 1\n"
                              "printf 'arch=inkling2_loop images=1\\n'\n", encoding="utf-8")
            worker.chmod(0o700)
            result = subprocess.run([sys.executable, "-c", check, root], cwd=terminal,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_staged_strata_projector_is_resident_even_with_lazy_text(self):
        recorded = BUILT_WORKER.parent / "strata-terminal.path"
        if not recorded.is_file():
            self.skipTest("the staged Strata adapter has not been built")
        terminal = Path(recorded.read_text().strip())
        check = """
from pathlib import Path
import json
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from serve import server
from serve.test_vision_elastic import ElasticVision
from serve.test_vram import VramEngine

class StopAfterStartup(Exception):
    pass

with TemporaryDirectory(prefix="strata-projector-check-") as directory:
    root = Path(directory)
    config = root / "run.json"
    common = {"exe": "/bin/false", "args": [], "backend": "cpu", "tokenizer": directory,
              "lazy_load": True, "scream_backend": "strata"}
    (root / "vocab.json").write_text('{"a":0,"b":1,"ab":2}')
    (root / "merges.txt").write_text("a b")
    (root / "token_type.json").write_text("[1,1,1]")
    for resident in (False, True):
        config.write_text(json.dumps({**common, "vision": {"exe": "/bin/false", "model": "model.gguf",
            "mmproj": "proj.gguf", "scream_resident": resident, "elastic": not resident}}))
        fake_vision = Mock()
        with patch.object(server, "Vision", return_value=fake_vision) as projector, \
             patch.object(server, "start_engine_or_fallback", side_effect=StopAfterStartup) as worker:
            try:
                server._main(["--engine", "strata", "--config", str(config), "--port", "0"])
            except StopAfterStartup:
                assert projector.call_args.kwargs["lazy"] is (not resident), projector.call_args
                assert worker.call_args.args[4] is True, worker.call_args
            else:
                raise AssertionError("did not test worker startup")

    tokenizer = server.ByteTokenizer()
    engine = VramEngine(tokenizer)
    engine.loaded = True
    vision = ElasticVision(directory, elastic=False)
    vision.resident = True
    vision.restart()
    service = server.Service(engine, tokenizer, server.ChatTemplate(server.ROOT / "serve/chat_template.jinja"),
                             vision=vision)
    assert service.unload(idle_for=0) == "unloaded"
    assert vision.alive()
    assert service.unload(idle_for=0) == "not loaded"
    assert vision.alive()
    assert service.unload() == "not loaded"
    assert not vision.alive()
"""
        result = subprocess.run([sys.executable, "-c", check], cwd=terminal, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_staged_split_planner_keeps_native_draft_and_batch_flags(self):
        recorded = BUILT_WORKER.parent / "strata-terminal.path"
        if not recorded.is_file():
            self.skipTest("the staged Strata adapter has not been built")
        terminal = Path(recorded.read_text().strip())
        check = """
from unittest.mock import patch
from serve import server

cfg = {'scream_backend': 'strata-split', 'exe': '/tmp/strata-split-2', 'drafting': 'auto', 'parallel': 6,
       'args': ['--native', 'model.gguf', '--mtp', 'draft', '--spec', '6', '--batch', '2'],
       'user_args': ['--spec', '0', '--batch', '3', '--lookup', '0'], 'vision': {},
       'expert_profile_save': '/tmp/unused.profile', 'draft_costs': True}
with patch.object(server.drafting, 'apply', side_effect=AssertionError('layer drafter planner ran')):
    args = server.engine_args(cfg, cfg['exe'], announce=False)
    assert args == ['--native', 'model.gguf', '--mtp', 'draft',
                    '--spec', '0', '--batch', '3', '--lookup', '0', '--vision'], args
    assert server.mtp_retry_args(cfg, cfg['exe']) is None
    assert server.nodraft_retry_args(cfg, cfg['exe']) is None
    assert not server.scream_split_worker(cfg, '/tmp/scream-strata-layer')
    off = {**cfg, 'drafting': 'off', 'user_args': [], 'args': ['--native', 'model.gguf', '--spec', '6']}
    args = server.engine_args(off, cfg['exe'], announce=False)
    assert args == ['--native', 'model.gguf', '--spec', '0', '--vision', '--batch', '6'], args
"""
        result = subprocess.run([sys.executable, "-c", check], cwd=terminal, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_strata_native_reuses_imported_worker_arguments_and_directory(self):
        recorded = BUILT_WORKER.parent / "strata-terminal.path"
        if not recorded.is_file():
            self.skipTest("the staged Strata argument resolver has not been built")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker"
            worker.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                              "print(json.dumps([sys.argv[1:], os.getcwd(), os.environ['STRATA_TEST_WORKER']]))\n")
            worker.chmod(0o755)
            (root / recorded.name).write_text(recorded.read_text())
            run = root / "strata.json"
            args = ["--native", "relative.strata", "--slots", "4", "--kv-dtype", "fp8", "--ctx", "1024", "--temp", "0.1"]
            run.write_text(json.dumps({"cwd": str(root), "args": [*args, "--threads", "2"],
                                       "args_written": [*args, "--threads", "1"],
                                       "user_args": ["--threads", "8", "--cpu"],
                                       "env": {"STRATA_TEST_WORKER": "imported"}}))
            result = subprocess.run([sys.executable, LAUNCHER, "--profile", "inkling-2-mini-strata", "--mode", "native",
                                     "--binary", worker, "--strata-config", run, "--context", "4096",
                                     "--set", "engine_options.--parallel=1", "--set", "sampling.temperature=0.8"],
                                    capture_output=True, text=True, check=True, timeout=10)
            native, cwd, env = json.loads(result.stdout)
            self.assertEqual((cwd, env), (directory, "imported"))
            self.assertEqual(native[:-1], ["--backend", "strata-inkling", "--native", "relative.strata", "--kv-dtype", "fp8",
                                           "--threads", "2", "--cpu", "--parallel", "1", "--max-context", "4096", "--sampling"])
            temperature, *defaults = native[-1].split()
            self.assertAlmostEqual(float(temperature), 0.8)
            self.assertEqual(defaults, "1.0 64 0.0 1.0 0.0 0.0 64 0".split())

    def test_native_flash_layer_saved_seed_can_be_overridden_or_removed(self):
        recorded = BUILT_WORKER.parent / "strata-terminal.path"
        if not recorded.is_file():
            self.skipTest("the staged Strata argument resolver has not been built")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker"
            worker.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
            worker.chmod(0o755)
            (root / recorded.name).write_text(recorded.read_text())
            run = root / "strata.json"
            run.write_text(json.dumps({"args": ["--seed", "71", "--greedy"]}))
            command = [sys.executable, LAUNCHER, "--profile", "qwen-flash-next-3.8-strata", "--mode", "native",
                       "--binary", worker, "--strata-config", run, "--set", "sampling.min_p=0.125"]
            for seed, legacy, field in (("91", ["--seed", "71"], "91"), ("null", [], "-")):
                with self.subTest(seed=seed):
                    result = subprocess.run([*command, "--set", f"sampling.seed={seed}"], capture_output=True, text=True,
                                            check=True, timeout=10)
                    self.assertEqual(json.loads(result.stdout), ["--backend", "strata", *legacy, "--greedy", "--sampling",
                                                                 f"- - - 0.125 - - - - {field}"])

    def test_megla_native_uses_data_directory_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker"
            worker.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
            worker.chmod(0o755)
            model, draft = root / MEGLA_MODEL, root / MEGLA_DRAFT
            draft.touch()
            command = [sys.executable, LAUNCHER, "--profile", "inkling-2-mini", "--mode", "native",
                       "--binary", worker, "--data-dir", root]
            for switches in ([], ["--num-passes", "2"], ["--set", "engine_options.--mtp=null"]):
                with self.subTest(switches=switches):
                    result = subprocess.run([*command, *switches], capture_output=True, text=True, check=True, timeout=10)
                    native = json.loads(result.stdout)
                    self.assertEqual(native[:4], ["--backend", "ink2_v1", "--model", str(model)])
                    if switches:
                        self.assertNotIn("--mtp", native)
                    else:
                        self.assertEqual(native[native.index("--mtp") + 1], str(draft))

    def test_strata_adapter_receives_resolved_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            serve = root / "serve"
            serve.mkdir()
            worker = root / "scream-p100"
            worker.symlink_to(sys.executable)
            (root / "strata-terminal.path").write_text(str(root) + "\n", encoding="utf-8")
            (serve / "server.py").write_text(
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "args = sys.argv[1:]\n"
                "cfg = json.loads(Path(args[args.index('--config') + 1]).read_text())\n"
                "assert Path(cfg['exe']).resolve() == Path(sys.executable).resolve()\n"
                "if Path(cfg['exe']).name == 'strata-split-2':\n"
                "    fallback = Path(cfg['split_fallback']['exe'])\n"
                "    assert fallback.name == 'scream-strata-layer' and fallback.resolve() == Path(sys.executable).resolve()\n"
                "else:\n"
                "    assert Path(cfg['exe']).name == 'scream-strata-layer' and 'split_fallback' not in cfg\n"
                "assert os.environ.get('STRATA_API_KEY') != 'fixture-worker-only'\n"
                "print(json.dumps([args, cfg, os.environ['FNX_GUARD']]))\n",
                encoding="utf-8",
            )
            doc = {
                "profile": "qwen-flash-next-3.8",
                "settings": {
                    "binary": str(worker), "context": 8192,
                    "strata_settings": {"tokenizer": "tokenizer", "effort_position": "end", "user_args": ["--ctx", "1024"]},
                },
            }
            legacy = root / "strata.json"
            legacy.write_text(json.dumps({
                "host": "127.0.0.2", "port": 8123, "cwd": ".",
                "env": {"FNX_GUARD": "from Strata", "SCREAM_FIXTURE_SPLIT_ONLY": "1", "STRATA_API_KEY": "fixture-worker-only"},
                "split_fallback": {"exe": "component:strata-engine", "args": ["--native", "older.gguf", "--layer-split", "auto"],
                                   "cwd": "layer-worker",
                                   "user_args": ["--model", "older.gguf", "--ctx", "1024"],
                                   "args_written": ["--native", "older.gguf", "--layer-split", "auto"],
                                   "env": {"FNX_GUARD": "from fallback", "SCREAM_FIXTURE_LAYER_ONLY": "1"}},
            }), encoding="utf-8")
            config = root / "profiles.json"
            config.write_text(json.dumps(doc), encoding="utf-8")
            environ = {**os.environ, "XDG_STATE_HOME": str(root / "state"), "SP100_HOST": "127.0.0.3", "SP100_PORT": "9000",
                       "SP100_ENVIRONMENT_PRECEDENCE": "config", "SP100_ENV": '{"FNX_GUARD":"from environment"}'}
            environ.pop("SCREAM_FIXTURE_SPLIT_ONLY", None)
            model = str(root / "selected.gguf")
            resolved = subprocess.run([sys.executable, LAUNCHER, "--config", config, "--strata-config", legacy, "--print-settings"],
                                      env={**environ, "SP100_PORT": "obsolete"}, text=True, capture_output=True,
                                      check=True, timeout=10)
            self.assertEqual(json.loads(resolved.stdout)["settings"]["port"], 8123)
            for overrides, host, port, worker_env, fallback_env in (
                ([], "127.0.0.2", 8123, "from Strata", "from fallback"),
                (["--environment-precedence", "environment"], "127.0.0.3", 9000, "from environment", "from environment"),
                (["--host", "127.0.0.4", "--port", "8124", "--set", 'env.FNX_GUARD="from CLI"'],
                 "127.0.0.4", 8124, "from CLI", "from CLI"),
            ):
                with self.subTest(overrides=overrides):
                    launched = subprocess.run(
                        [sys.executable, LAUNCHER, "--config", config, "--strata-config", legacy, "--model", model,
                         "--set", "engine_options.--batch=2", *overrides],
                        cwd=root, env=environ, text=True, capture_output=True, check=True, timeout=10,
                    )
                    args, cfg, inherited = json.loads(launched.stdout)
                    self.assertEqual(args[-4:], ["--host", host, "--port", str(port)])
                    self.assertEqual((cfg["host"], cfg["port"]), (host, port))
                    common = ["--native", model, "--max-context", "8192"]
                    self.assertEqual(cfg["args"], ["--batch", "2", *common])
                    self.assertEqual(cfg["user_args"], ["--batch", "2", *common])
                    self.assertEqual(cfg["env"].get("FNX_GUARD", inherited), worker_env)
                    self.assertEqual(inherited, "from CLI" if "--set" in overrides else "from environment")
                    self.assertEqual(cfg["env"]["STRATA_API_KEY"], "fixture-worker-only")
                    fallback = cfg["split_fallback"]
                    self.assertEqual(cfg["cwd"], str(root))
                    self.assertEqual(cfg["tokenizer"], str(root / "tokenizer"))
                    self.assertEqual(fallback["cwd"], str(root / "layer-worker"))
                    self.assertEqual(fallback["args"], ["--layer-split", "auto", *common])
                    self.assertEqual(fallback["args_written"], fallback["args"])
                    self.assertEqual(fallback["user_args"], common)
                    self.assertEqual(fallback["env"]["FNX_GUARD"], fallback_env)
                    self.assertIsNone(fallback["env"]["SCREAM_FIXTURE_SPLIT_ONLY"])
                    self.assertEqual(fallback["env"]["SCREAM_FIXTURE_LAYER_ONLY"], "1")
            layer_run = subprocess.run(
                [sys.executable, LAUNCHER, "--config", config, "--strata-config", legacy,
                 "--profile", "qwen-flash-next-3.8-strata", "--model", model],
                cwd=root, env=environ, text=True, capture_output=True, check=True, timeout=10,
            )
            args, layer, inherited = json.loads(layer_run.stdout)
            self.assertEqual(layer["cwd"], str(root / "layer-worker"))
            self.assertEqual(args[-4:], ["--host", "127.0.0.2", "--port", "8123"])
            self.assertEqual(layer["args"], ["--layer-split", "auto", "--native", model, "--max-context", "8192"])
            self.assertEqual(layer["args_written"], layer["args"])
            self.assertEqual(inherited, "from environment")
            self.assertEqual(layer["env"]["FNX_GUARD"], "from fallback")
            self.assertEqual(layer["tokenizer"], str(root / "layer-worker/tokenizer"))
            pending = Path(layer["scream_runtime_settings"]["path"])
            pending.parent.mkdir(parents=True)
            pending.write_text(json.dumps({"version": 1, "next_start": {"sampling.temperature": 0.6, "lazy_load": True,
                                                                         "effort_position": None},
                                           "shared": {"temperature": 0.8}}), encoding="utf-8")
            base = [sys.executable, LAUNCHER, "--config", config, "--strata-config", legacy,
                    "--profile", "qwen-flash-next-3.8-strata", "--model", model]
            restarted = subprocess.run(base, cwd=root, env=environ, text=True, capture_output=True, check=True, timeout=10)
            _, effective, _ = json.loads(restarted.stdout)
            self.assertEqual(effective["sampling"], {"temperature": 0.6})
            self.assertTrue(effective["lazy_load"])
            self.assertNotIn("effort_position", effective)
            self.assertEqual(effective["scream_runtime_settings"]["state"]["shared"], {"temperature": 0.8})
            explicit = subprocess.run([*base, "--sampling", '{"temperature":0.3}', "--print-settings"], cwd=root,
                                      env=environ, text=True, capture_output=True, check=True, timeout=10)
            self.assertEqual(json.loads(explicit.stdout)["settings"]["sampling"], {"temperature": 0.3})
            doc["settings"].update(port=8125, env={"FNX_GUARD": "from profile"})
            config.write_text(json.dumps(doc), encoding="utf-8")
            printed = subprocess.run([sys.executable, LAUNCHER, "--config", config, "--strata-config", legacy,
                                      "--print-settings"], env=environ, text=True, capture_output=True, check=True, timeout=10)
            settings = json.loads(printed.stdout)["settings"]
            self.assertEqual((settings["host"], settings["port"], settings["env"]["FNX_GUARD"]),
                             ("127.0.0.2", 8125, "from profile"))

    def test_megla_adapter_forwards_resolved_arguments_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "weights.gguf"
            model.touch()
            projector = root / "mmproj.gguf"
            projector.touch()
            ready = root / "ready"
            engine = root / "engine"
            engine.write_text(
                f"#!{sys.executable}\n"
                "import os, socket, sys, time\n"
                "assert sys.argv[sys.argv.index('--sampler') + 1] == 'strata-inkling'\n"
                "fields = sys.argv[sys.argv.index('--sampling') + 1].split()\n"
                "assert abs(float(fields[0]) - 0.6) < 1e-6 and fields[2] == '64'\n"
                "assert os.environ['INK2_STRATA_SAMPLER'] == '1'\n"
                "assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'\n"
                "server = socket.socket(socket.AF_UNIX)\n"
                "server.bind(sys.argv[sys.argv.index('--serve') + 1])\n"
                "time.sleep(60)\n",
                encoding="utf-8",
            )
            engine.chmod(0o755)
            (root / "ink2_server.py").write_text(
                "import json, os, sys, time\n"
                "from pathlib import Path\n"
                "assert os.environ['INK2_STRATA_SAMPLER'] == '1'\n"
                "assert os.environ['INK2_STREAM_TOOL_ARGUMENTS'] == '0'\n"
                "assert json.loads(os.environ['INK2_SAMPLING_DEFAULTS']) == {'temperature': 0.6}\n"
                "ready = Path(os.environ['TEST_READY'])\n"
                "pending = ready.with_suffix('.tmp')\n"
                "pending.write_text(json.dumps([sys.argv, os.getpid(), int(os.environ['INK2_ENGINE_PID'])]))\n"
                "pending.replace(ready)\n"
                "time.sleep(60)\n",
                encoding="utf-8",
            )
            launched = subprocess.Popen(
                [sys.executable, LAUNCHER, "--profile", "inkling-2-mini-megla-sampling", "--binary", engine,
                 "--set", "sampling.temperature=0.6",
                 "--tool-streaming", "buffered",
                 "--model", model, "--context", "4000", "--num-passes", "2",
                 "--vision-residency", "on-demand",
                 "--vision-device", VISION_GPU, "--set", f"server_options.--mmproj={json.dumps(str(projector))}",
                 "--set", f"server_options.--vis={json.dumps(sys.executable)}",
                 "--server-directory", root, "--port", "8124", "--host", "127.0.0.2"],
                env={**os.environ, "TEST_READY": str(ready), "INK2_STREAM_TOOL_ARGUMENTS": "1", "CUDA_VISIBLE_DEVICES": "0,1"},
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            )
            try:
                deadline = time.monotonic() + 5
                while not ready.exists() and launched.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(ready.exists(), "HTTP adapter failed to start")
                args, server_pid, engine_pid = json.loads(ready.read_text())
                self.assertEqual(args[args.index("--host") + 1], "127.0.0.2")
                self.assertEqual(args[args.index("--port") + 1], "8124")
                self.assertEqual(args[args.index("--ctx") + 1], "4000")
                self.assertEqual(args[args.index("--vision-device") + 1], VISION_GPU)
                self.assertEqual(args[args.index("--vision-residency") + 1], "on-demand")
                sock = Path(args[args.index("--sock") + 1])
                self.assertTrue(sock.is_socket())
                launched.send_signal(signal.SIGTERM)
                self.assertEqual(launched.wait(timeout=5), 128 + signal.SIGTERM)
                self.assertFalse(sock.parent.exists())
                for pid in (server_pid, engine_pid):
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)
            finally:
                if launched.poll() is None:
                    launched.terminate()
                launched.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
