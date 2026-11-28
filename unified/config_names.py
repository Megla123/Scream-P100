"""Names used by the launcher and their translations at the backend boundary."""

import re
import sys

DEFAULT_ENVIRONMENT_PREFIX = "SP100"
LEGACY_LAUNCHER_PREFIX = "SCREAM"
NAMING_DEFAULTS = {
    "environment_prefix": DEFAULT_ENVIRONMENT_PREFIX,
    "legacy_names": False,
    "legacy_name_precedence": "canonical",
}
LEGACY_SETTING_FLAGS = {
    "--ctx": "context",
    "--max-context": "context",
    "--native": "model",
    "--strata": "server_directory",
}
LEGACY_SETTING_ENV = {
    "vision_device": {"ink2_v1": ("INK2_VISION_DEVICE",)},
    "vision_residency": {"ink2_v1": ("INK2_VISION_RESIDENCY",)},
    "sampler": {"ink2_v1": ("INK2_STRATA_SAMPLER",)},
    "sampling": {"ink2_v1": ("INK2_SAMPLING_DEFAULTS",)},
    "tool_streaming": {"ink2_v1": ("INK2_STREAM_TOOL_ARGUMENTS",), "strata-inkling": ("SCREAM_INKLING_TOOL_STREAM",)},
}
LEGACY_ENV_VALUES = {
    "INK2_STRATA_SAMPLER": {"0": "native", "1": "strata-inkling"},
    "INK2_STREAM_TOOL_ARGUMENTS": {"0": "buffered", "1": "incremental"},
    "SCREAM_INKLING_TOOL_STREAM": {"0": "buffered", "1": "incremental"},
}
LEGACY_SERVER_FLAGS = {
    "--alias", "--api-key", "--api-monitor", "--batch-groups", "--batch-telemetry", "--before-load",
    "--fit-max-tokens", "--gguf-py", "--idle-unload", "--lazy", "--max-queue", "--mcp-config",
    "--min-free-vram-mib", "--mmproj", "--no-api-key", "--no-batch-telemetry", "--no-fit-max-tokens",
    "--offline", "--open", "--queue-timeout", "--ready-file", "--replicas", "--shutdown-timeout",
    "--slot-save-path", "--split-serve", "--strict-model", "--tok", "--tokenizer", "--vis",
}
LEGACY_BACKEND_PREFIXES = ("INK2_", "INK_", "IK2_", "FNX_", "SFNX_", "STRATA_", "HY4_", "HY4S_")
# Service-manager inputs, backend policy and transport must survive a change in user-facing names.
SERVICE_ENV = {"STRATA_API_KEY", "STRATA_NO_PRESET", "STRATA_OFFLINE", "STRATA_OWNER_ATTEST_KEY",
               "STRATA_TELEMETRY_KEY"}
SETTING_NAME = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*\Z")
FLAG = re.compile(r"--?[a-zA-Z][a-zA-Z0-9_-]*\Z")
ENV_NAME = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*\Z")
PREFIX = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*\Z")


def setting_flag(name):
    return "--" + name.replace("_", "-")


def setting_env(prefix, name):
    return f"{prefix}_{name.upper()}"


def collision(canonical, aliases, selected):
    print(f"scream-p100: warning: {', '.join((canonical, *aliases))} set the same option; using {selected}",
          file=sys.stderr)


def select_name(canonical, aliases, available, legacy_wins, *, warn=True):
    supplied = [name for name in dict.fromkeys((canonical, *aliases)) if name in available]
    if not supplied:
        return None
    selected = supplied[-1] if legacy_wins else supplied[0]
    if warn and len(supplied) > 1:
        collision(supplied[0], supplied[1:], selected)
    return selected


def option_key(group, name):
    """Translate canonical snake_case entries for the existing worker interfaces."""
    if group == "env":
        return name.upper() if SETTING_NAME.fullmatch(name) else name
    if group in ("engine_options", "server_options"):
        return setting_flag(name) if SETTING_NAME.fullmatch(name) else name
    return name


def normalize_map(group, values, legacy_wins, *, warn=True):
    if not isinstance(values, dict):
        raise ValueError(f"{group} must be an object")
    groups = {}
    for name in values:
        groups.setdefault(option_key(group, name), []).append(name)
    result = {}
    for key, spellings in groups.items():
        spellings.sort(key=lambda name: not bool(SETTING_NAME.fullmatch(name)))
        selected = select_name(spellings[0], spellings[1:], values, legacy_wins, warn=warn)
        result[key] = values[selected]
    return result


def naming_controls(document, cli, environ, parse_json):
    values = {}
    for name, default in NAMING_DEFAULTS.items():
        env_name = setting_env(DEFAULT_ENVIRONMENT_PREFIX, name)
        if name in cli:
            values[name] = cli[name]
        elif name in document:
            values[name] = document[name]
        elif env_name in environ:
            values[name] = parse_json(environ[env_name]) if name == "legacy_names" else environ[env_name]
        else:
            values[name] = default
    prefix = values["environment_prefix"]
    if not isinstance(prefix, str) or not PREFIX.fullmatch(prefix):
        raise ValueError("environment_prefix must contain uppercase letters, digits and single underscores")
    if type(values["legacy_names"]) is not bool:
        raise ValueError("legacy_names must be true or false")
    if values["legacy_name_precedence"] not in ("canonical", "legacy"):
        raise ValueError("legacy_name_precedence must be canonical or legacy")
    return values


def launcher_environment(environ, controls, keys, *, backend=None):
    selected = {}
    for key in keys:
        canonical = setting_env(controls["environment_prefix"], key)
        aliases = ()
        if controls["legacy_names"]:
            native = LEGACY_SETTING_ENV.get(key, {}).get(backend, ())
            aliases = (*native, setting_env(LEGACY_LAUNCHER_PREFIX, key))
        name = select_name(canonical, aliases, environ, controls["legacy_name_precedence"] == "legacy")
        if name is not None:
            value = environ[name]
            selected[key] = LEGACY_ENV_VALUES.get(name, {}).get(value, value)
    return selected


def environment_members(environ, controls, groups, parse_json, *, warn=True):
    """Read per-member variables of the form PREFIX_GROUP__MEMBER."""
    prefix = controls["environment_prefix"]
    legacy_wins = controls["legacy_name_precedence"] == "legacy"
    result = {}
    for group in groups:
        namespace = setting_env(prefix, group) + "__"
        for name, value in environ.items():
            if not name.startswith(namespace):
                continue
            member = name[len(namespace):]
            if not ENV_NAME.fullmatch(member) or member != member.upper():
                raise ValueError(f"invalid configuration environment variable {name}")
            if group == "env":
                legacy = (member,) if controls["legacy_names"] or member in SERVICE_ENV else ()
                selected = select_name(name, legacy, environ, legacy_wins, warn=warn)
                value = environ[selected]
            else:
                member = member.lower()
                value = parse_json(value)
            result.setdefault(group, {})[member] = value
    return result


def inherited_environment(environ, controls):
    if controls["legacy_names"]:
        return dict(environ)
    return {key: value for key, value in environ.items()
            if key in SERVICE_ENV or not key.startswith(LEGACY_BACKEND_PREFIXES)}
