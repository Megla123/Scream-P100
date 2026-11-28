#!/usr/bin/env python3
"""Launch the native inference executable or its existing HTTP protocol adapter."""
import json
import os
import re
import runpy
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from config import (FLAG, INKLING_FLAG_GROUPS, INSTALL_ROOT, NATIVE_ENV_FLAGS, PROTECTED_WORKER_FLAGS, STRATA_SAMPLING_FLAGS,
                    arguments, child_environment, configuration_sources, mapping, merge_strata_settings,
                    normalize_strata_args, occurrences, read_document, resolve, scalar_setting)
from runtime_settings import clear_deferred_fields, read_state, startup_layer, state_path

MEGLA = INSTALL_ROOT / "engine-inkling2"
MEGLA_MODEL = "Inkling-2-Mini-NVFP4-Q8_0.gguf"
MEGLA_DRAFT = "dspark-Inkling-2-Mini-Q8_0.gguf"
MEGLA_DFLASH = "dflash2-Inkling-2-Mini-Q8_0.gguf"
MEGLA_PROJECTOR = "mmproj-Inkling-2-Mini-BF16.gguf"
MEGLA_BUILD = MEGLA / "build"
ALIASES = {"strata": "scream-strata-layer", "strata-split": "strata-split-2", "strata-inkling": "strata-inkling"}
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

sys.path.insert(0, str(MEGLA / "serve"))
from ink2_sampling import SAMPLING_BASE, sampling_fields, sampling_values
from ink2_vision_device import vision_device


def require_file(path, name, executable=False):
    path = Path(path).absolute()
    if not path.is_file() or (executable and not os.access(path, os.X_OK)):
        raise ValueError(f"{name} missing or unusable: {path}")
    return str(path)


def option_args(options):
    for flag, value in options.items():
        for tokens in occurrences(value):
            yield flag
            yield from tokens


def replace_options(args, options, groups=()):
    """Replace existing flags and their operands without leaving stale aliases in a run config."""
    remove = set(options)
    for group in groups:
        if remove & group:
            remove |= group
    result = []
    skipping = False
    for arg in args:
        flag = re.split(r"[=\s]", arg, maxsplit=1)[0]
        if FLAG.fullmatch(flag):
            skipping = flag in remove
        if not skipping:
            result.append(arg)
    result.extend(option_args(options))
    return result


def engine_options(settings):
    backend = settings["backend"]
    extra = settings["engine_options"]
    overlap = extra.keys() & PROTECTED_WORKER_FLAGS
    if overlap:
        raise ValueError(f"use model, context, num_passes, sampler or sampling for {', '.join(sorted(overlap))}")
    options = {}
    if settings["model"] is not None:
        options["--model" if backend == "ink2_v1" else "--native"] = str(Path(settings["model"]).absolute())
    if settings["context"] is not None:
        options["--ctx" if backend == "ink2_v1" else "--max-context"] = settings["context"]
    if backend == "ink2_v1" and settings["sampler"] == "strata-inkling":
        options["--sampler"] = settings["sampler"]
    if (backend == "ink2_v1" and settings["sampler"] == "strata-inkling") or (
            backend == "strata-inkling" and settings["mode"] == "native"):
        configured = sampling_values(settings["sampling"], max_top_k=64 if backend == "ink2_v1" else 128)
        if configured:
            options["--sampling"] = sampling_fields({**SAMPLING_BASE, **configured})
    if backend in ("strata", "strata-split") and settings["mode"] == "native":
        configured = sampling_values(settings["sampling"])
        if configured:
            if configured.get("penalty_last_n", 0) > 4096:
                raise ValueError("Flash-Next native sampling supports penalty_last_n up to 4096")
            options["--sampling"] = sampling_fields(configured, sparse=True)
    elif backend in ("strata", "strata-split") and settings["sampling"].get("seed") == 0:
        # The HTTP adapter omits a zero seed from request frames. Set the explicit profile seed at startup.
        options["--sampling"] = sampling_fields({"seed": 0}, sparse=True)
    if settings["num_passes"] is not None:
        if backend in ("strata", "strata-split"):
            if settings["num_passes"] != 1:
                raise ValueError("the Flash-Next execution path supports exactly one pass")
        else:
            options["--num-passes"] = settings["num_passes"]
    return {**extra, **options}


def server_options(settings, protected):
    extra = settings["server_options"]
    overlap = extra.keys() & (protected | {"--host", "--port"})
    if overlap:
        raise ValueError(f"launcher-owned server flag: {', '.join(sorted(overlap))}")
    return [*option_args(extra), "--host", settings["host"], "--port", str(settings["port"])]


def strata_flag_groups(backend):
    groups = ({"--model", "--native"}, {"--max-context", "--ctx"})
    if backend == "strata-inkling":
        groups += (*map(set, INKLING_FLAG_GROUPS), {"--gpus", "--devices"}, {"--no-anti-loop", "--repetition-guard"})
    return groups


def strata_arguments(cfg, options, backend):
    groups = strata_flag_groups(backend)
    keys = ("args", "user_args", "args_written") if isinstance(cfg.get("args_written"), list) else ("args", "user_args")
    for key in keys:
        args = normalize_strata_args(cfg.get(key, []), key)
        # Upstream applies user_args over args and uses it to detect explicit device/drafting settings.
        cfg[key] = replace_options(args, options, groups)


def read_strata_run(path, backend):
    cfg = read_document(path)
    saved_layer = cfg.get("split_fallback")
    if backend != "strata" or not isinstance(saved_layer, dict) or not saved_layer.get("exe"):
        return cfg
    # A Strata-generated split file stores the original layer worker alongside the split worker.
    worker_keys = {"exe", "args", "args_written", "env", "layer_split", "replicas", "user_args", "drafting"}
    split_keys = {"split_fallback", "split_engine_note"}
    selected = {key: value for key, value in cfg.items() if key not in worker_keys | split_keys}
    selected.update({key: value for key, value in saved_layer.items() if key in worker_keys | {"cwd"}})
    if selected.get("env") is None:
        selected.pop("env", None)
    return selected


def strata_directory(settings, binary):
    build_dir = Path(binary).parent
    recorded = build_dir / "strata-terminal.path"
    location = recorded.read_text(encoding="utf-8").rstrip("\n") if recorded.is_file() else ""
    staged = build_dir / location if location else None
    directory = settings["server_directory"] or staged or build_dir / "_deps/scream_strata-src/terminal"
    return Path(directory).absolute(), staged


def strata_run_config(run_config, overrides):
    cfg = merge_strata_settings(run_config, overrides)
    cwd = cfg.get("cwd")
    if cwd is not None and (not isinstance(cwd, str) or not cwd.strip() or "\0" in cwd):
        raise ValueError("Strata run configuration cwd must be a nonempty path")
    cfg["cwd"] = os.path.abspath(cwd or os.getcwd())
    return cfg


def strata_native_run(settings, binary, options, run_config):
    cfg = strata_run_config(run_config, settings["strata_settings"])
    backend = settings["backend"]
    if not any(key in cfg for key in ("args", "user_args", "args_written")):
        return list(option_args(options)), cfg["cwd"]
    strata_arguments(cfg, {}, backend)
    directory, _ = strata_directory(settings, binary)
    module = require_file(directory / "strata_control/user_args.py", "Strata run argument resolver")
    user_args = runpy.run_path(module)
    base = user_args["merge"](cfg["args"], user_args["capture"](cfg))
    if backend == "strata-inkling" and "temperature" in settings["sampling"]:
        options = {"--temp": None, **options}
    if backend == "strata":
        removed = dict.fromkeys(flag for flag, key in STRATA_SAMPLING_FLAGS.items() if key in settings["sampling"])
        # Replacing the numeric seed with a sparse value retains any saved --seed mode switch.
        if settings["sampling"].get("seed") is not None:
            removed.pop("--seed", None)
        options = {**removed, **options}
    return replace_options(base, options, strata_flag_groups(backend)), cfg["cwd"]


def megla_worker_options(settings, options, data):
    options = {"--model": str(data / MEGLA_MODEL), **options} if data is not None else options.copy()
    if data is not None and settings["num_passes"] in (None, 3):
        draft = data / MEGLA_DFLASH
        options.setdefault("--mtp", str(draft if draft.is_file() else data / MEGLA_DRAFT))
    draft = list(occurrences(options.get("--mtp")))
    if draft:
        if any(len(values) != 1 for values in draft):
            raise ValueError("Megla --mtp requires a drafter file")
        options["--mtp"] = require_file(draft[-1][0], "Megla drafter")
        options.setdefault("--draft-vocab", str(MEGLA / "data/draft_vocab_32768.bin"))
    return options


class Interrupted(Exception):
    def __init__(self, signum):
        self.signum = signum


@contextmanager
def process(args, env, cwd=None, drain=False):
    child = subprocess.Popen(args, env=env, cwd=cwd, start_new_session=True)
    try:
        yield child
    finally:
        if child.poll() is None:
            try:
                if drain:
                    child.terminate()
                else:
                    os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=345 if drain else 10)
            except ProcessLookupError:
                child.wait()
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()


def serve_megla(settings, binary, options, env):
    data = Path(settings["data_dir"]).absolute() if settings["data_dir"] else None
    if "--serve" in options:
        raise ValueError("the launcher owns the Megla service socket")
    if not settings["model"] and data is None:
        raise ValueError("Megla serving requires model or data_dir")
    context = settings["context"] if settings["context"] is not None else 32768
    if context > 524288:
        raise ValueError("Megla context cannot exceed 524288")
    python = settings["server_python"] or (data / "venv/bin/python" if data else sys.executable)
    python = require_file(python, "Megla HTTP server Python", executable=True)
    server = Path(settings["server_directory"] or MEGLA / "serve") / "ink2_server.py"

    extra = settings["server_options"]
    device = vision_device(settings["vision_device"] or env.get("INK2_VISION_DEVICE", "cpu"))
    options = megla_worker_options(settings, options, data)
    if options.get("--mtp") and Path(options["--mtp"]).name == MEGLA_DFLASH:
        env = {"INK2_D2_QMIN": "0.7", "INK2_OA_MTP": "7", **env}
    model = require_file(options["--model"], "Megla model")
    options["--model"] = model
    options["--ctx"] = context
    http = {
        "--tokenizer": str(MEGLA_BUILD / "llama.cpp/build/bin/llama-tokenize"),
        "--gguf-py": str(MEGLA_BUILD / "llama.cpp/gguf-py"),
    }
    if (MEGLA_BUILD / "ink2_tok").is_file():
        http["--tok"] = str(MEGLA_BUILD / "ink2_tok")
    helper = MEGLA_BUILD / ("ink2_vis" if device == "cpu" else "ink2_vis_cuda")
    if env.get("INK2_PICTURES", "1") != "0":
        if data is not None and (data / MEGLA_PROJECTOR).is_file() and helper.is_file():
            http.update({"--mmproj": str(data / MEGLA_PROJECTOR), "--vis": str(helper)})
        elif extra.get("--mmproj") and "--vis" not in extra and helper.is_file():
            http["--vis"] = str(helper)
    # Apply explicit arguments before flattening, so null/false can suppress adapter defaults.
    http = {**http, **extra}
    projector = list(occurrences(http.get("--mmproj")))
    reader = list(occurrences(http.get("--vis")))
    if projector or reader or device != "cpu" or settings["vision_residency"] == "resident":
        if len(projector) != 1 or len(reader) != 1 or len(projector[0]) != 1 or len(reader[0]) != 1:
            raise ValueError("Megla vision requires one --mmproj and one --vis; GPU placement requires a CUDA projector helper")
        http["--mmproj"] = require_file(projector[0][0], "Megla vision projector")
        http["--vis"] = require_file(reader[0][0], "Megla vision helper", executable=True)
    checked = {**settings, "server_options": http}
    http_args = server_options(checked, {"--model", "--sock", "--ctx", "--vision-device", "--vision-residency"})
    require_file(server, "Megla HTTP server")

    with tempfile.TemporaryDirectory(prefix="scream-p100-") as runtime:
        sock = Path(runtime) / "engine.sock"
        command = [binary, "--backend", "ink2_v1", *option_args(options), "--serve", str(sock)]
        with process(command, env) as engine:
            deadline = time.monotonic() + 180
            while not sock.is_socket():
                if engine.poll() is not None:
                    raise ValueError(f"Megla exited before serving (status {engine.returncode})")
                if time.monotonic() >= deadline:
                    raise ValueError("Megla did not create its service socket within 180 seconds")
                time.sleep(0.1)
            env = {**env, "INK2_ENGINE_PID": str(engine.pid)}
            command = [python, str(server), *http_args, "--sock", str(sock),
                       "--model", model, "--ctx", str(context), "--vision-device", device,
                       "--vision-residency", settings["vision_residency"]]
            with process(command, env) as http_server:
                while http_server.poll() is None and engine.poll() is None:
                    time.sleep(0.1)
                if http_server.returncode is None:
                    raise ValueError(f"Megla exited while serving (status {engine.returncode})")
                return http_server.returncode


def strata_vision(cfg, device, backend):
    inkling = backend == "strata-inkling"
    if inkling and not cfg.get("vision"):
        if device not in (None, "cpu"):
            raise ValueError("Strata Inkling GPU vision requires vision.mmproj and vision.model")
        return
    if inkling:
        device = device or "cpu"
    if device is None:
        return
    if device == "cpu" and not cfg.get("vision"):
        return
    if not inkling and device != "cpu" and cfg.get("backend") not in (None, "cuda"):
        raise ValueError("an NVIDIA vision device requires the Strata CUDA backend")
    current = mapping(cfg.get("vision"), "Strata vision configuration").copy()
    if inkling:
        if not all(isinstance(current.get(key), str) and current[key].strip() for key in ("mmproj", "model")):
            raise ValueError("Strata Inkling external vision requires vision.mmproj and vision.model")
        cwd = Path(cfg.get("cwd") or ".")
        current["mmproj"] = require_file(cwd / current["mmproj"], "Inkling GGUF projector")
        current["model"] = require_file(cwd / current["model"], "Inkling GGUF text vocabulary")
        helper = current.get("exe")
        if helper is not None and (not isinstance(helper, str) or not helper.strip()):
            raise ValueError("vision.exe must be an executable path")
        helper = helper or MEGLA_BUILD / ("ink2_vis" if device == "cpu" else "ink2_vis_cuda")
        current["exe"] = require_file(cwd / helper, "Inkling projector reader", executable=True)
        current["inkling_embeddings"] = True
    if not all(current.get(key) for key in ("exe", "mmproj", "model")):
        raise ValueError("Strata vision placement requires vision.exe, vision.mmproj and vision.model")
    helper_env = mapping(current.get("env") or {}, "Strata vision env").copy()
    helper_env.update(CUDA_VISIBLE_DEVICES="" if device == "cpu" else device, CUDA_DEVICE_ORDER="PCI_BUS_ID")
    cfg["vision"] = {**current, "gpu": device != "cpu", "cuda_device": None if device == "cpu" else device,
                     "env": helper_env, "require_gpu": device != "cpu"}


def strata_vision_residency(cfg, policy):
    vision = cfg.get("vision")
    if not vision:
        return False
    vision = mapping(vision, "Strata vision configuration")
    if policy == "auto" and not vision.get("gpu"):
        return False
    resident = policy != "on-demand"
    cfg["vision"] = {**vision, "elastic": not resident, "scream_resident": resident}
    return True


def serve_strata(settings, binary, options, controls, cli, environ, run_config, document, env, runtime_plan):
    backend = settings["backend"]
    worker = child_environment(settings, controls, environ, cli)
    env.setdefault("STRATA_KEEP_LEFTOVERS", "1")
    directory, staged = strata_directory(settings, binary)
    python = require_file(settings["server_python"] or sys.executable, "Strata HTTP server Python", executable=True)
    server = require_file(directory / "serve/server.py", "Strata HTTP server")
    incremental = settings["tool_streaming"] == "incremental"
    if incremental and not (directory / "serve/inkling_tool_stream.py").is_file():
        raise ValueError("incremental tool streaming requires the Strata adapter staged by the unified build")
    if not (directory / "serve/scream_runtime_settings.py").is_file():
        env["SCREAM_INKLING_TOOL_STREAM"] = "1" if incremental else "0"
    adapter = server_options(settings, {"--engine", "--config"})
    cfg = strata_run_config(run_config, settings["strata_settings"])
    clear_deferred_fields(cfg, runtime_plan["state"]["next_start"])
    external_inkling = backend == "strata-inkling" and bool(cfg.get("vision"))
    strata_vision(cfg, settings["vision_device"], backend)
    managed_projector = strata_vision_residency(cfg, settings["vision_residency"])
    if (managed_projector or settings["vision_device"] not in (None, "cpu") or external_inkling) and (
            not staged or directory.resolve() != Path(staged).resolve()):
        raise ValueError("external Inkling vision, explicit GPU placement and projector residency require the staged Strata adapter")
    if settings["sampling"]:
        defaults = cfg.get("sampling") or {}
        if not isinstance(defaults, dict):
            raise ValueError("Strata run configuration sampling must be an object")
        defaults = defaults.copy()
        for alias, canonical in (("repeat_penalty", "repetition_penalty"), ("repeat_last_n", "penalty_last_n")):
            if canonical in settings["sampling"]:
                defaults.pop(alias, None)
        cfg["sampling"] = {**defaults, **settings["sampling"]}
    if not isinstance(cfg.get("tokenizer"), str) or not cfg["tokenizer"]:
        raise ValueError("Strata serving requires tokenizer in strata_config or strata_settings")
    cfg["tokenizer"] = str(Path(cfg["cwd"]) / cfg["tokenizer"])
    strata_arguments(cfg, options, backend)
    cfg["scream_backend"] = backend
    protected = set(options)
    for group in strata_flag_groups(backend):
        if protected & group:
            protected.update(group)
    cfg["scream_runtime_settings"] = {
        **runtime_plan,
        "incremental_tools": incremental,
        "tuning_flags": sorted(protected),
        "tuning_environment": sorted(settings["env"].keys() | worker.keys()),
    }
    if backend == "strata-inkling":
        cfg["arch"] = "inkling2_loop"
    # Strata classifies the worker by basename. The alias targets the same native binary.
    with tempfile.TemporaryDirectory(prefix="scream-p100-") as runtime:
        runtime = Path(runtime)
        alias = runtime / ALIASES[backend]
        alias.symlink_to(binary)
        fallback = cfg.pop("split_fallback", None)
        if backend == "strata-split" and isinstance(fallback, dict) and fallback.get("exe"):
            fallback_alias = runtime / ALIASES["strata"]
            fallback_alias.symlink_to(binary)
            fallback = {**fallback, "exe": str(fallback_alias), "scream_backend": "strata"}
            if "cwd" in fallback:
                fallback = strata_run_config(fallback, {})
            common_options = {flag: options[flag] for flag in ("--native", "--max-context", "--sampling") if flag in options}
            strata_arguments(fallback, common_options, "strata")
            saved_env = fallback.get("env")
            saved_env = {} if saved_env is None else mapping(saved_env, "Strata split_fallback env")
            fallback_settings, _ = resolve(document, {**cli, "backend": "strata"}, environ, config_defaults={"env": saved_env})
            fallback_env = child_environment(fallback_settings, controls, environ, cli)
            scoped_keys = settings["env"].keys() | fallback_settings["env"].keys()
            if controls["environment_precedence"] == "environment":
                scoped_keys.update({"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"} & fallback_env.keys())
            fallback["env"] = {key: fallback_env.get(key) for key in sorted(scoped_keys)
                               if key in fallback_env or key in worker or key in env}
            cfg["split_fallback"] = fallback
        worker_env = {key: worker.get(key) for key in settings["env"] if worker.get(key) != env.get(key)}
        worker_env.update({key: worker[key] for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER") if key in worker
                           and (key in settings["env"] or controls["environment_precedence"] == "environment")})
        if backend == "strata-inkling":
            worker_env.update({key: worker[key] for key in NATIVE_ENV_FLAGS[backend] if key in worker})
        cfg.update(exe=str(alias), env=worker_env, host=settings["host"], port=settings["port"])
        worker_config = runtime / "strata.json"
        worker_config.write_text(json.dumps(cfg), encoding="utf-8")
        command = [python, server, "--engine", "strata", "--config", str(worker_config), *adapter]
        with process(command, env, cwd=directory, drain=True) as child:
            return child.wait()


def main(argv=None):
    def interrupted(signum, _frame):
        # Cleanup finishes even if a service manager or terminal sends an additional signal.
        for sig in STOP_SIGNALS:
            signal.signal(sig, signal.SIG_IGN)
        raise Interrupted(signum)

    try:
        environ = dict(os.environ)
        ap, doc, cli, print_settings = arguments(argv, environ)
        if "_export_profile" in cli or "_import_profile" in cli:
            from profile_exchange import exchange_profile

            exchange_profile(doc, cli, environ)
            return 0
        sources = configuration_sources(doc, cli, environ)
        controls, config_layers, named = sources
        backend = scalar_setting("backend", config_layers, controls, cli, named)
        mode = scalar_setting("mode", config_layers, controls, cli, named)
        strata = backend in tuple(ALIASES)
        strata_serving = mode == "serve" and strata
        run_config = {}
        config_defaults = {}
        path = scalar_setting("strata_config", config_layers, controls, cli, named)
        if strata and isinstance(path, str) and path.strip() and "\0" not in path:
            run_config = read_strata_run(path, backend)
            config_defaults = {key: run_config[key] for key in ("host", "port") if run_config.get(key) is not None}
            config_defaults["env"] = mapping(run_config.get("env", {}), "Strata run configuration env")
        adapter_environment = {} if strata_serving else None
        runtime_plan = None
        if strata_serving:
            identity_layers = [config_defaults, *config_layers]
            host, port, model = (scalar_setting(key, identity_layers, controls, cli, named)
                                 for key in ("host", "port", "model"))
            identity = [cli.get("_config_path"), controls["profile"], backend, host, port, model, path, os.getcwd()]
            state_file = state_path(environ, identity)
            state = read_state(state_file)
            runtime_plan = {"path": str(state_file), "state": state}
            sources = controls, [*config_layers, startup_layer(state["next_start"])], named
        settings, controls = resolve(doc, cli, environ, config_defaults=config_defaults, sources=sources,
                                     run_config=run_config, adapter_environment=adapter_environment)
        if settings["vision_device"] is not None:
            device = vision_device(settings["vision_device"])
            if backend == "strata-inkling" and mode == "native" and device != "cpu":
                raise ValueError("Strata Inkling GPU projector placement uses the HTTP launcher; native GENI accepts precomputed embeddings")
        if print_settings:
            print(json.dumps({**controls, "settings": settings}, indent=2))
            return 0
        if settings["backend"] is None:
            ap.error("choose a supported model profile or set --backend")
        binary = require_file(settings["binary"], "unified engine", executable=True)
        options = engine_options(settings)
        if settings["mode"] == "serve":
            for sig in STOP_SIGNALS:
                signal.signal(sig, interrupted)
        if strata_serving:
            return serve_strata(settings, binary, options, controls, cli, environ, run_config, doc,
                                adapter_environment, runtime_plan)
        env = child_environment(settings, controls, environ, cli)
        if settings["backend"] == "ink2_v1":
            env["INK2_STRATA_SAMPLER"] = "1" if settings["sampler"] == "strata-inkling" else "0"
            env["INK2_SAMPLING_DEFAULTS"] = json.dumps(settings["sampling"], allow_nan=False)
            if settings["tool_streaming"] != "native":
                env["INK2_STREAM_TOOL_ARGUMENTS"] = "1" if settings["tool_streaming"] == "incremental" else "0"
        if settings["mode"] == "native":
            if strata:
                native_args, cwd = strata_native_run(settings, binary, options, run_config)
                os.chdir(cwd)
            else:
                data = Path(settings["data_dir"]).absolute() if settings["data_dir"] else None
                options = megla_worker_options(settings, options, data)
                native_args = list(option_args(options))
            os.execve(binary, [binary, "--backend", backend, *native_args], env)

        return serve_megla(settings, binary, options, env)
    except Interrupted as error:
        return 128 + error.signum
    except (OSError, ValueError) as error:
        print(f"scream-p100: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
