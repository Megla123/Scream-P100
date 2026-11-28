"""Resolve the Scream P100 launcher configuration before starting any process."""
import argparse
import json
import math
import os
import re
from pathlib import Path

from config_names import (DEFAULT_ENVIRONMENT_PREFIX, ENV_NAME, FLAG, LEGACY_LAUNCHER_PREFIX, LEGACY_SERVER_FLAGS,
                          LEGACY_SETTING_ENV, LEGACY_SETTING_FLAGS, NAMING_DEFAULTS, PREFIX, collision, environment_members,
                          inherited_environment, launcher_environment, naming_controls, normalize_map, select_name,
                          setting_env, setting_flag)
from runtime_settings import FLOAT32_MAX

BACKENDS = ("ink2_v1", "strata-inkling", "strata", "strata-split")
INSTALL_ROOT = Path(__file__).resolve().parents[1]
ENTRY_BINARY = os.environ.get("SCREAM_ENTRY_BINARY")
DEFAULTS = {
    "backend": None,
    "mode": "serve",
    "binary": ENTRY_BINARY or str(INSTALL_ROOT / "build-unified" / "scream-p100"),
    "host": "127.0.0.1",
    "port": 8080,
    "data_dir": None,
    "model": None,
    "context": None,
    "num_passes": None,
    "sampler": "native",
    "sampling": {},
    "tool_streaming": "native",
    "vision_device": None,
    "vision_residency": "auto",
    "server_directory": None,
    "server_python": None,
    "strata_config": None,
    "engine_options": {},
    "server_options": {},
    "env": {},
    "strata_settings": {},
}
CONTROL_DEFAULTS = {"profile": None, "ignore_default_profile": False, "environment_precedence": "config"}
MAP_SETTINGS = {"engine_options", "server_options", "env", "strata_settings", "sampling"}
STRATA_OWNED_FIELDS = {"exe", "args", "env", "host", "port"}
INKLING_FLAG_GROUPS = (("--parallel", "--batch", "--slots"), ("--kv", "--kv-dtype"))
INKLING_EQUIVALENTS = (*INKLING_FLAG_GROUPS, ("--gpus", "--devices"), ("--repetition-guard", "--no-anti-loop"))
PROTECTED_WORKER_FLAGS = {"--model", "--native", "--ctx", "--max-context", "--num-passes", "--sampler", "--sampling"}
STRATA_SAMPLING_FLAGS = {"--temperature": "temperature", "--greedy": "temperature",
                         "--top-p": "top_p", "--top-k": "top_k", "--seed": "seed"}
NATIVE_ENV_FLAGS = {
    "strata-split": {"FNX_ROPE_SCALE": ("--rope-scale",)},
    "strata-inkling": {
        "IK2_EP_SPLIT": ("--ep-split",),
        "IK2_GPUS": ("--gpus", "--devices"),
        "STRATA_INK_SPEC_BEAM": ("--draft-beam",),
        "STRATA_INK_SPEC_COST": ("--draft-cost",),
        "STRATA_INK_SPEC_PLAIN": ("--draft-plain",),
    },
}
SAMPLING_ORDER = (
    "temperature", "top_p", "top_k", "min_p", "repetition_penalty", "frequency_penalty",
    "presence_penalty", "penalty_last_n", "seed",
)
SAMPLING_KEYS = set(SAMPLING_ORDER)
SAMPLING_INTEGER_KEYS = {"top_k", "penalty_last_n", "seed"}
STRATA_SAMPLING_ALIASES = {"repeat_penalty": "repetition_penalty", "repeat_last_n": "penalty_last_n"}
JOINED_FLAG = re.compile(r"^(--[A-Za-z0-9][A-Za-z0-9-]*)(?:\s+|=)(\S.*)$")
LAUNCHER_ENV = {"INK2_ENGINE_PID", "INK2_STRATA_SAMPLER", "INK2_SAMPLING_DEFAULTS", "SCREAM_INKLING_TOOL_STREAM"}


def object_with_unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def json_value(text):
    def invalid_constant(value):
        raise ValueError(f"invalid JSON number {value}")

    return json.loads(text, object_pairs_hook=object_with_unique_keys, parse_constant=invalid_constant)


def mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def check_keys(values, allowed, name):
    unknown = values.keys() - allowed
    if unknown:
        raise ValueError(f"unknown {name}: {', '.join(sorted(unknown))}")


def merge(lower, higher):
    result = lower.copy()
    for key, value in higher.items():
        if key == "strata_settings":
            result[key] = merge_strata_settings(result.get(key, {}), mapping(value, key))
        elif key in ("engine_options", "server_options"):
            overlay = mapping(value, key)
            result[key] = {name: item for name, item in result.get(key, {}).items() if name not in overlay}
            result[key].update(overlay)
        elif key in MAP_SETTINGS:
            result[key] = {**result.get(key, {}), **mapping(value, key)}
        else:
            result[key] = value
    return result


def merge_strata_settings(lower, higher):
    result = lower.copy()
    for key, value in higher.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_strata_settings(result[key], value)
        else:
            result[key] = value
    return result


def packaged_resource(name):
    if ENTRY_BINARY:
        built = Path(ENTRY_BINARY).with_name("scream-" + name.replace("_", "-"))
        if built.is_file():
            return built
    return Path(__file__).with_name(name)


def profiles_from(document, legacy_wins):
    path = packaged_resource("builtin_profiles.json")
    profiles = {}
    for entries in (read_document(path), mapping(document.get("profiles", {}), "profiles")):
        merge_profiles(profiles, entries, legacy_wins)
    if "default" not in profiles:
        raise ValueError(f"{path} must define the default profile")
    return profiles


def normalize_settings(layer, legacy_wins):
    return {key: normalize_map(key, value, legacy_wins) if key in MAP_SETTINGS else value
            for key, value in layer.items()}


def merge_profiles(profiles, entries, legacy_wins):
    for name, value in entries.items():
        if not name.strip():
            raise ValueError("profile names cannot be empty")
        entry = mapping(value, f"profile {name!r}")
        check_keys(entry, {"extends", "settings"}, f"field in profile {name!r}")
        previous = profiles.get(name, {})
        settings = mapping(entry.get("settings", {}), f"settings for {name!r}")
        check_keys(settings, DEFAULTS.keys(), f"setting in profile {name!r}")
        parent = entry.get("extends", previous.get("extends"))
        if parent is not None and (not isinstance(parent, str) or not parent):
            raise ValueError(f"parent of profile {name!r} must be a profile name")
        if name == "default" and parent is not None:
            raise ValueError("the default profile cannot extend another profile")
        profiles[name] = {"extends": parent,
                          "settings": merge(previous.get("settings", {}), normalize_settings(settings, legacy_wins))}


def lineage(profiles, name):
    chain = []
    seen = set()
    while name is not None:
        if name in seen:
            raise ValueError(f"profile inheritance cycle: {' -> '.join([*chain, name])}")
        if name not in profiles:
            raise ValueError(f"unknown profile {name!r}")
        seen.add(name)
        chain.append(name)
        name = profiles[name].get("extends")
    return reversed(chain)


def validate_settings(values):
    if values["backend"] is not None and values["backend"] not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(BACKENDS)}")
    if values["mode"] not in ("serve", "native"):
        raise ValueError("mode must be serve or native")
    sampler, backend = values["sampler"], values["backend"]
    if sampler not in ("native", "strata-inkling"):
        raise ValueError("sampler must be native or strata-inkling")
    if sampler == "strata-inkling" and backend not in (None, "ink2_v1", "strata-inkling"):
        raise ValueError("the Strata Inkling sampler requires an Inkling execution path")
    streaming = values["tool_streaming"]
    if streaming not in ("native", "incremental", "buffered"):
        raise ValueError("tool_streaming must be native, incremental or buffered")
    if streaming != "native" and backend not in (None, "ink2_v1", "strata-inkling"):
        raise ValueError("tool_streaming requires an Inkling execution path")
    if values["vision_residency"] not in ("auto", "resident", "on-demand"):
        raise ValueError("vision_residency must be auto, resident or on-demand")
    if backend == "ink2_v1" and sampler == "native" and any(value is not None for value in values["sampling"].values()):
        raise ValueError("Megla sampling defaults require sampler=strata-inkling")
    check_keys(values["sampling"], SAMPLING_KEYS, "sampling setting")
    for key, value in values["sampling"].items():
        if value is None:
            continue
        if key in SAMPLING_INTEGER_KEYS:
            if type(value) is not int:
                raise ValueError(f"sampling.{key} must be an integer")
        elif type(value) not in (int, float) or not -FLOAT32_MAX <= value <= FLOAT32_MAX:
            raise ValueError(f"sampling.{key} must be a finite number representable by the sampler")
    for key in ("host", "binary", "data_dir", "model", "vision_device", "server_directory", "server_python", "strata_config"):
        value = values[key]
        if value is None and DEFAULTS[key] is None:
            continue
        if not isinstance(value, str) or not value.strip() or "\0" in value:
            raise ValueError(f"{key} must be a nonempty string")
    for key, low, high in (("port", 0, 65535), ("context", 1, 2**31 - 1), ("num_passes", 1, 2**31 - 1)):
        value = values[key]
        if value is None and DEFAULTS[key] is None:
            continue
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{key} must be an integer from {low} to {high}")
    for key, value in values["env"].items():
        if not ENV_NAME.fullmatch(key) or key in LAUNCHER_ENV:
            raise ValueError(f"{key!r} cannot be set in env")
        if value is not None:
            scalar_text(value)
    for group in ("engine_options", "server_options"):
        for key, value in values[group].items():
            if not FLAG.fullmatch(key):
                raise ValueError(f"invalid {group} flag {key!r}")
            try:
                tuple(occurrences(value))
            except ValueError as error:
                raise ValueError(f"{group} {key!r}: {error}") from error
    check_keys(
        {key: None for key in values["strata_settings"] if key in STRATA_OWNED_FIELDS},
        set(),
        "launcher-owned strata_settings field",
    )


def scalar_text(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, str) and "\0" not in value:
        return value
    if type(value) is int or (type(value) is float and math.isfinite(value)):
        return str(value)
    raise ValueError(f"expected a string, number or boolean, received {value!r}")


def occurrences(value):
    if value is False or value is None:
        return
    if value is True:
        yield []
        return
    if not isinstance(value, list):
        value = [value]
    groups = value if any(isinstance(item, list) for item in value) else [value]
    for group in groups:
        if not isinstance(group, list):
            raise ValueError("use an array of arrays for repeated options")
        tokens = [scalar_text(item) for item in group]
        if any(token.startswith("--") for token in tokens):
            raise ValueError("option operands cannot contain another long option")
        yield tokens


def normalize_strata_args(args, key):
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise ValueError(f"Strata run configuration {key} must be an array of strings")
    result = []
    for arg in args:
        joined = JOINED_FLAG.fullmatch(arg)
        # Match Strata's treatment of an operand following a standalone flag.
        if joined and not (result and FLAG.fullmatch(result[-1]) and not arg.startswith(joined[1] + "=")):
            result.extend((joined[1], joined[2].strip()))
        else:
            result.append(arg)
    return result


def env_setting(name, text):
    if name in {"port", "context", "num_passes", "ignore_default_profile", *MAP_SETTINGS}:
        return json_value(text)
    return text


def strata_inkling_layer(layer, native, legacy_wins):
    flags = mapping(layer.get("engine_options", {}), "engine_options")
    if not flags:
        return layer
    aliases = {}
    for group in INKLING_EQUIVALENTS:
        selected = select_name(group[0], group[1:], flags, legacy_wins)
        if selected is None:
            continue
        for flag in group:
            aliases[flag] = None
        # One operand to --gpus is a count. One operand to --devices is an ordinal.
        aliases[selected] = selected if selected == "--devices" else group[0]
    canonical = {}
    for flag, value in flags.items():
        name = aliases.get(flag, flag)
        if name is None:
            continue
        if flag == "--no-anti-loop":
            if value is not None and type(value) is not bool:
                raise ValueError("--no-anti-loop must be a switch")
            value = "off" if value else None
        canonical[name] = value
    layer = {**layer, "engine_options": canonical}
    if not native or "--temp" not in canonical:
        return layer
    args = list(occurrences(canonical.pop("--temp")))
    if any(len(operands) != 1 for operands in args):
        raise ValueError("--temp requires one numeric value")
    try:
        temperature = float(args[-1][0]) if args else None
    except ValueError as error:
        raise ValueError("--temp requires one numeric value") from error
    sampling = mapping(layer.get("sampling", {}), "sampling").copy()
    if "temperature" in sampling:
        collision("sampling.temperature", ["--temp"], "--temp" if legacy_wins else "sampling.temperature")
    if legacy_wins or "temperature" not in sampling:
        sampling["temperature"] = temperature
    return {**layer, "sampling": sampling}


def normalize_sampling_aliases(block, legacy_wins):
    sample = mapping(block, "sampling").copy()
    for alias, name in STRATA_SAMPLING_ALIASES.items():
        if alias in sample:
            picked = select_name(name, (alias,), sample, legacy_wins)
            sample[name] = sample[picked]
            del sample[alias]
    return sample


def strata_sampling_defaults(block):
    if block is None:
        return dict.fromkeys(SAMPLING_KEYS)
    saved = mapping(block, "Strata run configuration sampling").copy()
    for alias, key in STRATA_SAMPLING_ALIASES.items():
        if alias in saved:
            if saved[alias] is not None and saved.get(key) is not None:
                raise ValueError(f"Strata sampling specifies both {key} and {alias}")
            if saved.get(key) is None:
                saved[key] = saved[alias]
    values = {key: value for key, value in saved.items() if key in SAMPLING_KEYS}
    for key in values.keys() & SAMPLING_INTEGER_KEYS:
        value = values[key]
        if type(value) is float and value.is_integer():
            values[key] = int(value)
    return values


def strata_sampling_layer(layer):
    nested = mapping(layer.get("strata_settings", {}), "strata_settings")
    if "sampling" not in nested:
        return layer
    return {**layer, "sampling": {**strata_sampling_defaults(nested["sampling"]),
                                   **mapping(layer.get("sampling", {}), "sampling")}}


def strata_sampling_flags(layer, named_temperature, legacy_wins):
    flags = mapping(layer.get("engine_options", {}), "engine_options")
    named = mapping(layer.get("sampling", {}), "sampling")
    kept, sampling = {}, {}
    for flag, value in flags.items():
        key = STRATA_SAMPLING_FLAGS.get(flag)
        if key is None or (key == "temperature" and not named_temperature):
            kept[flag] = value
            continue
        if flag == "--seed":
            # The legacy native flag enables sampling as well as assigning the seed. A seed in the
            # sparse sampling block assigns the number only, so keep the separate mode switch.
            kept[flag] = value
        args = list(occurrences(value))
        if flag == "--greedy":
            if args not in ([], [[]]):
                raise ValueError("--greedy must be a switch")
            sampling[key] = 0.0 if args else None
            continue
        if any(len(operands) != 1 for operands in args):
            raise ValueError(f"{flag} requires one numeric value")
        try:
            number = (int if key in SAMPLING_INTEGER_KEYS else float)(args[-1][0]) if args else None
        except ValueError as error:
            raise ValueError(f"{flag} requires one numeric value") from error
        # The layer engine treats zero or a wider shortlist as the full 64 candidates.
        if key == "top_k" and number is not None and (number == 0 or number > 64):
            number = 64
        sampling[key] = number
    if "seed" in named and named["seed"] is None:
        kept["--seed"] = None
    for member in sampling.keys() & named.keys():
        aliases = [flag for flag in flags if STRATA_SAMPLING_FLAGS.get(flag) == member]
        key = f"sampling.{member}"
        collision(key, aliases, aliases[-1] if legacy_wins else key)
    return {**layer, "engine_options": kept, "sampling": {**named, **sampling} if legacy_wins else {**sampling, **named}}


def configuration_sources(document, cli, environ):
    document = mapping(document, "configuration")
    check_keys(document, {*CONTROL_DEFAULTS, *NAMING_DEFAULTS, "settings", "profiles"}, "configuration field")
    naming = naming_controls(document, cli, environ, json_value)
    keys = (DEFAULTS.keys() | CONTROL_DEFAULTS.keys()) - LEGACY_SETTING_ENV.keys()
    named = launcher_environment(environ, naming, keys)
    settings = mapping(document.get("settings", {}), "settings")
    check_keys(settings, DEFAULTS.keys(), "setting")
    profiles = profiles_from(document, naming["legacy_name_precedence"] == "legacy")
    for name in profiles:
        tuple(lineage(profiles, name))

    order = cli.get("environment_precedence", document.get("environment_precedence",
                    named.get("environment_precedence", CONTROL_DEFAULTS["environment_precedence"])))
    if order not in ("config", "environment"):
        raise ValueError("environment_precedence must be config or environment")
    controls = {**CONTROL_DEFAULTS, **naming, "environment_precedence": order}
    for key in ("profile", "ignore_default_profile"):
        if key in cli:
            controls[key] = cli[key]
        elif key in named and (order == "environment" or key not in document):
            controls[key] = env_setting(key, named[key])
        elif key in document:
            controls[key] = document[key]
    if type(controls["ignore_default_profile"]) is not bool:
        raise ValueError("ignore_default_profile must be true or false")
    selected = controls["profile"]
    if selected is not None and (not isinstance(selected, str) or not selected):
        raise ValueError("profile must be a profile name")

    chain = list(lineage(profiles, selected))
    if not controls["ignore_default_profile"] and "default" not in chain:
        chain.insert(0, "default")
    controls["profile"] = selected or (chain[-1] if chain else None)

    layers = [settings, *(profiles[name].get("settings", {}) for name in chain)]
    backend = scalar_setting("backend", layers, controls, cli, named)
    named.update(launcher_environment(environ, naming, LEGACY_SETTING_ENV,
                                      backend=backend if isinstance(backend, str) else None))
    return controls, layers, named


def scalar_setting(key, config_layers, controls, cli, named):
    if key in cli:
        return cli[key]
    if controls["environment_precedence"] == "environment" and key in named:
        return env_setting(key, named[key])
    for layer in reversed(config_layers):
        if key in layer:
            return layer[key]
    return env_setting(key, named[key]) if key in named else DEFAULTS[key]


def apply_native_precedence(values, bindings, ranks, run_config, saved_rank, user_rank):
    env, options = values["env"], values["engine_options"]
    saved = merge_strata_settings(run_config, values["strata_settings"])
    args, user, written = (normalize_strata_args(saved[key], key) if isinstance(saved.get(key), list) else []
                           for key in ("args", "user_args", "args_written"))
    explicit_user = {}
    if user_rank is not None:
        bound_flags = {flag for flags in bindings.values() for flag in flags}
        for i, flag in enumerate(user):
            if not isinstance(flag, str) or flag not in bound_flags:
                continue
            if i + 1 >= len(user) or not isinstance(user[i + 1], str) or FLAG.fullmatch(user[i + 1]):
                raise ValueError(f"Strata run configuration user_args {flag} requires one operand")
            explicit_user.pop(flag, None)
            explicit_user[flag] = user[i + 1]
    for key, flags in bindings.items():
        flags_from_user = {flag: arg for flag, arg in explicit_user.items() if flag in flags}
        option_rank = max(ranks["engine_options"].get(flag, -1) for flag in flags)
        if flags_from_user and user_rank > option_rank:
            # These options were provided through the unified configuration, so they take precedence
            # over the imported args and their edit history just like engine_options would.
            for flag in flags:
                if flag in options:
                    options[flag] = None
            for flag, arg in flags_from_user.items():
                options.pop(flag, None)
                options[flag] = arg
                ranks["engine_options"][flag] = user_rank
        overrides_saved = any(flag in options for flag in flags)
        active = {}
        for flag in flags:
            if flag in options:
                if options[flag] is not None and options[flag] is not False:
                    active[flag] = ranks["engine_options"][flag]
            elif not overrides_saved and (flag in args or (flag in user and flag not in written)):
                # Strata's saved user arguments stop applying when removed from edited args.
                active[flag] = saved_rank
        flag_rank = max([*(ranks["engine_options"].get(flag, -1) for flag in flags), *active.values()])
        for flag, rank in active.copy().items():
            if rank < flag_rank:
                options[flag] = None
                del active[flag]
        if not active:
            continue
        env_value = env.get(key)
        blank_is_unset = key in ("IK2_GPUS", "IK2_EP_SPLIT") and env_value == ""
        if env_value is not None and not blank_is_unset and ranks["env"][key] > flag_rank:
            options.update(dict.fromkeys(active))
            continue
        if env_value is not None:
            env[key] = None


def promote_user_options(layers, backend, sampler_flags):
    # Saved user_args is an array replacement. Only the highest source supplying that array wins.
    # Hoist conflicting options into that source so arg serialization cannot reorder their precedence.
    options = {flag for layer in layers for flag in layer.get("engine_options", {})}
    if backend == "strata-inkling":
        for equivalents in INKLING_EQUIVALENTS:
            if options.intersection(equivalents):
                options.update(equivalents)
    options = (options | set(sampler_flags)) - PROTECTED_WORKER_FLAGS
    for index in reversed(range(len(layers))):
        layer = layers[index]
        nested = layer.get("strata_settings", {})
        if "user_args" not in nested:
            continue
        args = normalize_strata_args(nested["user_args"], "user_args")
        promoted = {}
        for i, flag in enumerate(args):
            if flag not in options:
                continue
            end = next((j for j in range(i + 1, len(args)) if FLAG.fullmatch(args[j])), len(args))
            value = args[i + 1:end] or True
            promoted.setdefault(flag, []).append(value)
        if promoted:
            promoted = {key: values[-1] if key in sampler_flags or any(key in group for group in INKLING_EQUIVALENTS)
                        else values[0] if len(values) == 1 else values for key, values in promoted.items()}
            layers[index] = merge({"engine_options": promoted}, layer)
        return


def resolve(document, cli, environ, *, config_defaults=None, sources=None, run_config=None, adapter_environment=None):
    controls, config_layers, named = sources if sources is not None else configuration_sources(document, cli, environ)
    legacy_wins = controls["legacy_name_precedence"] == "legacy"
    config_layers = [config_defaults or {}, *config_layers]
    values = {key: scalar_setting(key, config_layers, controls, cli, named)
              for key in DEFAULTS if key not in MAP_SETTINGS}
    cli = merge_legacy_options(cli, values["backend"], legacy_wins)
    is_strata = values["backend"] in ("strata", "strata-split", "strata-inkling")
    if is_strata and run_config and "sampling" in run_config:
        defaults = {"sampling": strata_sampling_defaults(run_config["sampling"])}
        config_layers[0] = merge(defaults, config_layers[0])
    env_values = {key: normalize_map(key, env_setting(key, named[key]), legacy_wins)
                  for key in MAP_SETTINGS - {"env"} if key in named}
    if values["backend"] == "ink2_v1" and controls["legacy_names"] and "sampling" in env_values:
        env_values["sampling"] = normalize_sampling_aliases(env_values["sampling"], legacy_wins)
    members = environment_members(environ, controls, MAP_SETTINGS - {"env", "strata_settings"}, json_value)
    members = normalize_settings(members, legacy_wins)
    if values["backend"] == "strata-inkling":
        for aliases in INKLING_EQUIVALENTS:
            if members.get("engine_options", {}).keys() & set(aliases):
                for flag in aliases:
                    env_values.get("engine_options", {}).pop(flag, None)
    env_values = merge(env_values, members)
    bindings = NATIVE_ENV_FLAGS.get(values["backend"], {}) if isinstance(values["backend"], str) else {}
    inherited = {key: environ[key] for key in bindings if key in environ and controls["legacy_names"]}
    explicit_env = worker_environment_map(environ, controls, json_value(named["env"]) if "env" in named else {})
    env_options = env_values.get("engine_options", {})
    for legacy, flags in bindings.items():
        if legacy not in inherited or legacy in explicit_env:
            continue
        for flag in flags:
            if flag not in env_options:
                continue
            canonical = setting_env(controls["environment_prefix"], "engine_options") + "__" + flag[2:].replace("-", "_").upper()
            collision(canonical, [legacy], legacy if legacy_wins else canonical)
            if legacy_wins:
                env_options.pop(flag)
    if inherited or explicit_env:
        env_values["env"] = {**inherited, **explicit_env}
    config_first = controls["environment_precedence"] == "config"
    lower = [DEFAULTS, env_values] if config_first else [DEFAULTS]
    higher = [cli] if config_first else [env_values, cli]
    layers = [*lower, *config_layers, *higher]
    layers = [normalize_settings(layer, legacy_wins) for layer in layers]
    if is_strata:
        layers = [strata_sampling_layer(layer) for layer in layers]
    native_layer = values["backend"] == "strata" and values["mode"] == "native"
    native_inkling = values["backend"] == "strata-inkling" and values["mode"] == "native"
    if is_strata:
        sampler_flags = STRATA_SAMPLING_FLAGS if native_layer else {"--temp"} if native_inkling else ()
        promote_user_options(layers, values["backend"], sampler_flags)
    named_layer_sampling = [mapping(layer.get("sampling", {}), "sampling") for layer in layers] if native_layer else []
    normalize_layer_sampling = any(named_layer_sampling)
    named_temperature = any("temperature" in layer for layer in named_layer_sampling)
    ranks = {"env": {}, "engine_options": {}}
    user_rank = None
    adapter_entries = {}
    for position, layer in enumerate(layers):
        if values["backend"] == "strata-inkling":
            layer = strata_inkling_layer(layer, native_inkling, legacy_wins)
        elif normalize_layer_sampling:
            layer = strata_sampling_flags(layer, named_temperature, legacy_wins)
        values = merge(values, {key: value for key, value in layer.items() if key in MAP_SETTINGS})
        if adapter_environment is not None and position != len(lower):
            adapter_entries.update(layer.get("env", {}))
        if bindings:
            for name in ranks:
                ranks[name].update(dict.fromkeys(layer.get(name, {}), position))
            if "user_args" in layer.get("strata_settings", {}):
                user_rank = position
    validate_settings(values)
    if bindings:
        apply_native_precedence(values, bindings, ranks, run_config or {}, saved_rank=len(lower), user_rank=user_rank)
    if adapter_environment is not None:
        adapter_environment.update(child_environment({**values, "env": adapter_entries}, controls, environ, cli))
    return values, controls


def worker_environment_map(environ, controls, whole, *, warn=True):
    legacy_wins = controls["legacy_name_precedence"] == "legacy"
    explicit = normalize_map("env", whole, legacy_wins, warn=warn)
    members = environment_members(environ, controls, {"env"}, json_value, warn=warn).get("env", {})
    if controls["legacy_names"]:
        for key in explicit.keys() - members.keys():
            if key not in environ:
                continue
            canonical = setting_env(controls["environment_prefix"], "env") + "." + key.lower()
            if warn:
                collision(canonical, [key], key if legacy_wins else canonical)
            if legacy_wins:
                explicit[key] = environ[key]
    return {**explicit, **members}


def child_environment(settings, controls, environ, cli):
    env = inherited_environment(environ, controls)
    for aliases in LEGACY_SETTING_ENV.values():
        for alias in aliases.get(settings["backend"], ()):
            env.pop(alias, None)
    legacy_wins = controls["legacy_name_precedence"] == "legacy"
    cli_env = normalize_map("env", cli.get("env", {}), legacy_wins, warn=False)
    prefix = setting_env(controls["environment_prefix"], "env")
    aliases = (setting_env(LEGACY_LAUNCHER_PREFIX, "env"),) if controls["legacy_names"] else ()
    name = select_name(prefix, aliases, environ, legacy_wins, warn=False)
    explicit_env = worker_environment_map(environ, controls, json_value(environ[name]) if name else {}, warn=False)
    native_aliases = NATIVE_ENV_FLAGS.get(settings["backend"], {})
    for key, value in settings["env"].items():
        if (key not in native_aliases and controls["environment_precedence"] == "environment" and key in env
                and key not in cli_env and key not in explicit_env):
            continue
        if value is None:
            env.pop(key, None)
        else:
            env[key] = scalar_text(value)
    return env


def read_document(path):
    return mapping(json_value(Path(path).read_text(encoding="utf-8-sig")), str(path))


def configuration_arguments(parser):
    parser.add_argument("--config", "--cookbook", help=f"configuration JSON (or {setting_env(DEFAULT_ENVIRONMENT_PREFIX, 'config')})")
    parser.add_argument("--environment-prefix", help="environment namespace for launcher settings")
    parser.add_argument("--legacy-names", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--legacy-name-precedence", choices=("canonical", "legacy"))


def legacy_options(args):
    result = {}
    pending = None
    for arg in args:
        flag, separator, operand = arg.partition("=")
        if FLAG.fullmatch(flag):
            group = "server_options" if flag in LEGACY_SERVER_FLAGS else "engine_options"
            options = result.setdefault(group, {})
            pending = options.pop(flag, [])
            options[flag] = pending
            pending.append([operand] if separator else [])
        elif pending is None or arg.startswith("--"):
            raise ValueError(f"unrecognized argument {arg!r}")
        else:
            pending[-1].append(arg)
    return {group: {flag: (values[0] or True) if len(values) == 1 else values for flag, values in flags.items()}
            for group, flags in result.items()}


def merge_legacy_options(cli, backend, legacy_wins):
    if "_legacy_options" not in cli:
        return cli
    cli = cli.copy()
    for group, args in cli.pop("_legacy_options").items():
        modern = normalize_map(group, cli.get(group, {}), legacy_wins)
        canonical_flags = set(modern)
        matches = ()
        if backend == "strata-inkling" and group == "engine_options":
            matches = INKLING_EQUIVALENTS
        preferred = args.copy()
        for equivalents in matches:
            supplied = [flag for flag in args if flag in equivalents]
            if not supplied:
                continue
            for flag in supplied[:-1]:
                preferred.pop(flag)
            selected = supplied[-1]
            value = preferred[selected]
            if isinstance(value, list) and value and isinstance(value[0], list):
                preferred[selected] = value[-1] or True
        for flag, value in preferred.items():
            equivalents = next((names for names in matches if flag in names), (flag,))
            overlap = [other for other in equivalents if other in canonical_flags]
            if overlap:
                canonical = f"{setting_flag(group)} {overlap[0].lstrip('-').replace('-', '_')}"
                collision(canonical, [flag], flag if legacy_wins else canonical)
                if not legacy_wins:
                    continue
                for other in overlap:
                    modern.pop(other, None)
            modern[flag] = value
        cli[group] = modern
    return cli


def cli_map(key, raw, legacy_names):
    try:
        return mapping(json_value(raw), key)
    except ValueError:
        if key != "sampling" or not legacy_names:
            raise
    fields = raw.split()
    if len(fields) != len(SAMPLING_ORDER):
        raise ValueError("--sampling requires a JSON object or the legacy nine sampling operands in one argument")
    return {name: (int if name in SAMPLING_INTEGER_KEYS else float)(field)
            for name, field in zip(SAMPLING_ORDER, fields) if field != "-"}


def arguments(argv, environ=None):
    environ = os.environ if environ is None else environ
    bootstrap = argparse.ArgumentParser(prog="scream-p100", add_help=False, allow_abbrev=False)
    configuration_arguments(bootstrap)
    bootstrap.add_argument("-h", "--help", action="store_true")
    bootstrap.add_argument("--help-settings", nargs="?", const="", metavar="NAME")
    bootstrap.add_argument("--import-profile", "--import-recipe", metavar="FILE")
    discovery, _ = bootstrap.parse_known_args(argv)
    if discovery.help or discovery.help_settings is not None:
        from settings_help import load_reference, reference_text, short_help

        reference = load_reference()
        prefix = discovery.environment_prefix or environ.get(setting_env(DEFAULT_ENVIRONMENT_PREFIX, "environment_prefix"), "")
        prefix = prefix if PREFIX.fullmatch(prefix) else DEFAULT_ENVIRONMENT_PREFIX
        print(short_help(reference) if discovery.help else reference_text(reference, discovery.help_settings, prefix=prefix))
        bootstrap.exit()
    bootstrap_cli = {name: value for name, value in vars(discovery).items() if name in NAMING_DEFAULTS and value is not None}
    initial = naming_controls({}, bootstrap_cli, environ, json_value)
    config_env = setting_env(initial["environment_prefix"], "config")
    default_config = setting_env(DEFAULT_ENVIRONMENT_PREFIX, "config")
    if config_env not in environ:
        config_env = default_config
    legacy_config = (setting_env(LEGACY_LAUNCHER_PREFIX, "config"),) if initial["legacy_names"] else ()
    path = discovery.config
    if not path:
        name = select_name(config_env, legacy_config, environ, initial["legacy_name_precedence"] == "legacy")
        path = environ[name] if name else None
    try:
        document = read_document(path) if path else {}
    except FileNotFoundError:
        if discovery.import_profile is None:
            raise
        document = {}
    naming = naming_controls(document, bootstrap_cli, environ, json_value)
    legacy_wins = naming["legacy_name_precedence"] == "legacy"

    ap = argparse.ArgumentParser(prog="scream-p100", description="Run the unified Scream P100 engine", add_help=False,
                                 allow_abbrev=False, parents=[bootstrap])
    ap.add_argument("--profile", "--recipe", help="profile name, defaults to the default profile")
    ap.add_argument("--ignore-default-profile", "--ignore-default-recipe", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--environment-precedence", choices=("config", "environment"))
    ap.add_argument("--export-profile", "--export-recipe", metavar="FILE")
    ap.add_argument("--parent-profile", "--parent-recipe", metavar="NAME")
    for key in sorted(DEFAULTS):
        ap.add_argument(setting_flag(key), metavar="JSON" if key in MAP_SETTINGS else None)
    if naming["legacy_names"]:
        for flag in LEGACY_SETTING_FLAGS:
            ap.add_argument(flag, dest=flag, help=argparse.SUPPRESS)
    ap.add_argument("--set", action="append", metavar="KEY=JSON", default=[],
                    help="override a setting; use map.FLAG=JSON to override a single map entry")
    ap.add_argument("--print-settings", action="store_true", help="print resolved settings without launching")
    if naming["legacy_names"]:
        parsed, remaining = ap.parse_known_args(argv)
    else:
        parsed, remaining = ap.parse_args(argv), []
    inputs = vars(parsed)
    cli = {key: cli_map(key, inputs[key], naming["legacy_names"]) for key in MAP_SETTINGS if inputs[key] is not None}
    for assignment in parsed.set:
        setting, sep, raw = assignment.partition("=")
        key, dot, member = setting.partition(".")
        try:
            if not sep or key not in DEFAULTS or (dot and (key not in MAP_SETTINGS or not member)):
                raise ValueError(f"invalid --set setting {setting!r}")
            value = json_value(raw)
            cli = merge(cli, {key: {member: value} if dot else value})
        except ValueError as error:
            ap.error(str(error))
    for key in DEFAULTS.keys() - MAP_SETTINGS | CONTROL_DEFAULTS.keys() | NAMING_DEFAULTS.keys():
        available = {setting_flag(key): inputs[key]} if inputs[key] is not None else {}
        aliases = [alias for alias, target in LEGACY_SETTING_FLAGS.items() if target == key and inputs.get(alias) is not None]
        available.update({alias: inputs[alias] for alias in aliases})
        if key in cli and aliases:
            available.setdefault(setting_flag(key), cli[key])
        picked = select_name(setting_flag(key), aliases, available, legacy_wins)
        if picked is not None:
            if picked == setting_flag(key) and inputs[key] is None:
                continue
            value = available[picked]
            try:
                cli[key] = int(value) if key in ("port", "context", "num_passes") else value
            except ValueError:
                ap.error(f"{picked} requires an integer")
    if remaining:
        cli["_legacy_options"] = legacy_options(remaining)
    if path:
        cli["_config_path"] = str(Path(path).absolute())
    if sum((parsed.print_settings, parsed.export_profile is not None, parsed.import_profile is not None)) > 1:
        ap.error("choose one of --print-settings, --export-profile and --import-profile")
    if parsed.parent_profile is not None and parsed.import_profile is None:
        ap.error("--parent-profile requires --import-profile")
    if parsed.export_profile is not None:
        cli["_export_profile"] = parsed.export_profile
    if parsed.import_profile is not None:
        if not path:
            ap.error("--import-profile requires a destination configuration file")
        cli["_import_profile"] = parsed.import_profile
        cli["_parent_profile"] = parsed.parent_profile
    if (parsed.export_profile is not None or parsed.import_profile is not None) and (
            cli.keys() & DEFAULTS.keys() or "_legacy_options" in cli):
        ap.error("profile import and export operate on configuration files, without launch setting overrides")
    return ap, document, cli, parsed.print_settings
