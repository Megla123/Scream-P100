"""Read and render the settings reference only when help or documentation is requested."""

from pathlib import Path
import re
import sys

from config import (CONTROL_DEFAULTS, DEFAULTS, INKLING_FLAG_GROUPS, MAP_SETTINGS, NATIVE_ENV_FLAGS, SAMPLING_KEYS, STRATA_OWNED_FIELDS,
                    packaged_resource, read_document)
from config_names import (DEFAULT_ENVIRONMENT_PREFIX, LEGACY_LAUNCHER_PREFIX, LEGACY_SERVER_FLAGS,
                          LEGACY_SETTING_ENV, LEGACY_SETTING_FLAGS, NAMING_DEFAULTS, setting_env, setting_flag)

KINDS = {"setting", "control", "discovery", "document", "command"}
COMMAND_USAGE = {
    "help": "-h, --help", "help_settings": "--help-settings [NAME]", "set": "--set KEY=JSON",
    "print_settings": "--print-settings", "configured": "--configured", "list_backends": "--list-backends",
    "export_profile": "--export-profile FILE / --export-recipe FILE",
    "import_profile": "--import-profile FILE / --import-recipe FILE",
    "parent_profile": "--parent-profile NAME / --parent-recipe NAME",
}
SECTIONS = (
    ("Launch controls and commands", {"discovery", "control", "document", "command"}, ""),
    ("Launcher settings", {"setting"}, ""),
    ("Sampling", {"setting"}, "sampling."),
    ("Worker options", {"setting"}, "engine_options."),
    ("Server options", {"setting"}, "server_options."),
    ("Worker and server environment", {"setting"}, "env."),
    ("Strata run configuration", {"setting"}, "strata_settings."),
)


def load_reference(path=None):
    path = packaged_resource("settings_reference.json") if path is None else Path(path)
    document = read_document(path)
    entries = document.get("entries")
    version = document.get("version")
    if set(document) != {"version", "entries"} or type(version) is not int or version != 1 or not isinstance(entries, dict):
        raise ValueError(f"invalid settings reference structure in {path}")
    expected = dict.fromkeys((*DEFAULTS, *(f"sampling.{key}" for key in SAMPLING_KEYS)), "setting")
    expected.update(dict.fromkeys((*CONTROL_DEFAULTS, *NAMING_DEFAULTS), "control"))
    expected.update(dict.fromkeys(COMMAND_USAGE, "command"))
    expected["config"] = "discovery"
    if missing := expected.keys() - entries.keys():
        raise ValueError(f"settings reference is missing {', '.join(sorted(missing))}")
    for name, entry in entries.items():
        if not isinstance(entry, dict) or entry.keys() - {"kind", "summary", "values", "related", "aliases", "lookup_aliases"}:
            raise ValueError(f"invalid settings reference entry {name!r}")
        kind = entry.get("kind")
        if kind not in KINDS or expected.get(name, kind) != kind:
            raise ValueError(f"invalid settings reference kind for {name!r}")
        group, dot, _ = name.partition(".")
        if kind == "setting" and name not in expected and (not dot or group not in MAP_SETTINGS - {"sampling"}):
            raise ValueError(f"settings reference {name!r} does not name a configurable setting or map member")
        for field in ("summary", "values"):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                raise ValueError(f"settings reference {name!r} requires text for {field}")
        for field in ("related", "aliases", "lookup_aliases"):
            items = entry.get(field, [])
            if not isinstance(items, list) or any(not isinstance(item, str) or not item.strip() for item in items):
                raise ValueError(f"settings reference {name!r} requires a list of names or notes for {field}")
        if unknown := set(entry.get("related", ())) - entries.keys():
            raise ValueError(f"settings reference {name!r} links to unknown entries {', '.join(sorted(unknown))}")
    return entries


def surfaces(name, kind, prefix=DEFAULT_ENVIRONMENT_PREFIX):
    if kind == "command":
        return (), COMMAND_USAGE.get(name, setting_flag(name)), ""
    if kind == "document":
        return (name,), "", ""
    if kind == "discovery":
        return (), "--config PATH / --cookbook PATH", setting_env(prefix, name)
    if kind == "control":
        cli = {"profile": "--profile NAME / --recipe NAME",
               "ignore_default_profile": "--ignore-default-profile / --ignore-default-recipe",
               "legacy_names": "--legacy-names / --no-legacy-names"}
        if name in NAMING_DEFAULTS:
            prefix = DEFAULT_ENVIRONMENT_PREFIX
        return (name,), cli.get(name, setting_flag(name) + " VALUE"), setting_env(prefix, name)
    paths = (f"settings.{name}", f"profiles.<profile>.settings.{name}")
    group, dot, member = name.partition(".")
    if not dot:
        return paths, setting_flag(name) + (" JSON" if isinstance(DEFAULTS.get(name), dict) else " VALUE"), setting_env(prefix, name)
    if group == "strata_settings":
        # The CLI accepts the first member only. Its JSON object supplies deeper keys.
        first, nested, _ = member.partition(".")
        cli = f"--set '{group}.{first}={'OBJECT' if nested else 'JSON'}'"
        return paths, cli, setting_env(prefix, group) + " (whole JSON object)"
    return paths, f"--set '{name}=JSON'", setting_env(prefix, group) + "__" + member.upper()


def legacy_aliases(name):
    aliases = []
    if name in DEFAULTS or name in CONTROL_DEFAULTS or name == "config":
        aliases.extend(flag for flag, setting in LEGACY_SETTING_FLAGS.items() if setting == name)
        aliases.append(setting_env(LEGACY_LAUNCHER_PREFIX, name))
        aliases.extend(alias for names in LEGACY_SETTING_ENV.get(name, {}).values() for alias in names)
    group, _, member = name.partition(".")
    if member and not member.startswith("<"):
        if group == "env":
            aliases.append(member.upper())
        if group == "server_options" and setting_flag(member) in LEGACY_SERVER_FLAGS:
            aliases.append(setting_flag(member))
        if group == "engine_options":
            flag = setting_flag(member)
            aliases.append(flag)
            for names in INKLING_FLAG_GROUPS:
                if flag == names[0]:
                    aliases.extend(names[1:])
            if flag == "--repetition-guard":
                aliases.append("--no-anti-loop")
            aliases.extend(alias for bindings in NATIVE_ENV_FLAGS.values() for alias, flags in bindings.items() if flag in flags)
    return list(dict.fromkeys(aliases))


def details(name, entry, prefix=DEFAULT_ENVIRONMENT_PREFIX):
    paths, cli, env = surfaces(name, entry["kind"], prefix)
    aliases = list(dict.fromkeys((*legacy_aliases(name), *entry.get("lookup_aliases", ()), *entry.get("aliases", ()))))
    return {label: value for label, value in (
        ("JSON paths", ", ".join(paths)), ("Command line", cli), ("Environment", env), ("Valid values", entry["values"]),
        ("Legacy names and notes", " · ".join(aliases)), ("Related settings", ", ".join(entry.get("related", ()))),
    ) if value}


def short_help(entries):
    lines = ["Usage: scream-p100 [--configured] [options]", "", "Configure a native inference worker or HTTP adapter.", "", "Options"]
    for name, entry in entries.items():
        if "." in name or entry["kind"] == "document":
            continue
        _, cli, _ = surfaces(name, entry["kind"])
        lines.append(f"  {cli:<63} {entry['summary']}")
    return "\n".join([*lines, "", "Use --help-settings for paths, valid values, relationships, and legacy aliases."])


def lookup(entries, query, prefix):
    if query in entries:
        return [(query, entries[query])]
    name = re.sub(r"^(?:settings\.|profiles\..+?\.settings\.)", "", query)
    if name.startswith("--"):
        name = name[2:].replace("-", "_")
    if name in entries:
        return [(name, entries[name])]
    exact = [(key, entry) for key, entry in entries.items()
             if query == surfaces(key, entry["kind"], prefix)[2] or query in legacy_aliases(key)
             or query in entry.get("lookup_aliases", ())]
    if exact:
        return exact
    for namespace in ("engine_options", "server_options", "env"):
        variable_prefix = setting_env(prefix, namespace) + "__"
        if name.startswith(variable_prefix) and name[len(variable_prefix):]:
            name = f"{namespace}.{name[len(variable_prefix):].lower()}"
            break
    group, dot, member = name.partition(".")
    if group == "strata_settings" and member.partition(".")[0] in STRATA_OWNED_FIELDS:
        raise ValueError(f"{name!r} is launcher-owned; see --help-settings strata_settings")
    for pattern in ("<option>", "<variable>", "<field>"):
        if dot and member and (wildcard := f"{group}.{pattern}") in entries:
            return [(name, entries[wildcard])]
    raise ValueError(f"unknown settings reference name {query!r}; use --help-settings to list names")


def reference_text(entries, query=None, *, prefix=DEFAULT_ENVIRONMENT_PREFIX):
    selected = lookup(entries, query, prefix) if query else entries.items()
    blocks = []
    for name, entry in selected:
        lines = [name, f"  {entry['summary']}"]
        lines.extend(f"  {label:<24} {value}" for label, value in details(name, entry, prefix).items())
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def markdown(entries):
    lines = ["# Scream P100 settings reference", "",
             "This reference is generated from [settings_reference.json](settings_reference.json). "
             "The launcher reads the same file when you run `--help` or `--help-settings`. "
             "See the [configuration guide](CONFIGURATION.md) for examples and precedence rules.", "",
             "Worker and adapter options extend this reference through the option maps. The pinned backend defines "
             "their additional flags and limits. See the [worker and adapter interfaces](CONFIGURATION.md#backend-option-interfaces).", ""]
    for title, kinds, prefix in SECTIONS:
        lines.append(f"- [{title}](#{title.lower().replace(' ', '-')})")
    for title, kinds, prefix in SECTIONS:
        lines.extend(("", f"## {title}"))
        for name, entry in entries.items():
            if entry["kind"] not in kinds or prefix and not name.startswith(prefix) or not prefix and entry["kind"] == "setting" and "." in name:
                continue
            lines.extend(("", f"### `{name}`", "", entry["summary"], ""))
            for label, value in details(name, entry).items():
                # Protect flag operands and angle-bracket path placeholders in Markdown.
                if label in {"JSON paths", "Command line", "Environment", "Related settings"}:
                    value = f"`{value}`"
                lines.append(f"- **{label}** {value}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.stdout.write(markdown(load_reference()))
