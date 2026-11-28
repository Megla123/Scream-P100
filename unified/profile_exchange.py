"""Export configured profiles and import them as cookbook entries."""

import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile

from config import (DEFAULTS, MAP_SETTINGS, check_keys, configuration_sources, mapping, normalize_settings,
                    profiles_from, read_document, resolve)


def configured_profile(document, selection, environ, *, sparse=True):
    controls, layers, _ = configuration_sources(document, selection, environ)
    settings, _ = resolve(document, {}, {}, sources=(controls, layers, {}))
    if sparse:
        declared = set().union(*(layer.keys() for layer in layers))
        settings = {key: value for key, value in settings.items() if key in declared or (key in MAP_SETTINGS and value)}
    return controls, settings


def encoded(data):
    return json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def write_json(path, data, *, mode=None):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(encoded(data))
            if mode is not None:
                os.fchmod(file.fileno(), mode)
            file.flush()
            os.fsync(file.fileno())
        if mode is None:
            # Publish a new file without racing another creator or replacing their file.
            os.link(temporary, path)
        else:
            temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def profile_name(name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("a profile name must be a nonempty string")
    return name


def same_value(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_value(value, b[key]) for key, value in a.items())
    return a == b


def map_delta(settings, inherited, *, nested=False):
    changes = {}
    for key, value in settings.items():
        if key not in inherited:
            changes[key] = value
        elif nested and isinstance(value, dict) and isinstance(inherited[key], dict):
            if difference := map_delta(value, inherited[key], nested=True):
                changes[key] = difference
        elif not same_value(value, inherited[key]):
            changes[key] = value
    return changes


def profile_delta(settings, inherited):
    changes = map_delta({key: value for key, value in settings.items() if key not in MAP_SETTINGS}, inherited)
    for key, value in settings.items():
        if key in MAP_SETTINGS and (difference := map_delta(value, inherited.get(key, {}), nested=key == "strata_settings")):
            changes[key] = difference
    return changes


def redundant_paths(settings, delta, prefix=()):
    for key, value in settings.items():
        path = (*prefix, key)
        descend = (not prefix and key in MAP_SETTINGS) or prefix[:1] == ("strata_settings",)
        if descend and isinstance(value, dict) and value:
            yield from redundant_paths(value, delta.get(key, {}), path)
        elif key not in delta:
            yield path


def without_path(settings, path):
    result = settings.copy()
    key, *nested = path
    if nested and (remainder := without_path(result[key], nested)):
        result[key] = remainder
    else:
        del result[key]
    return result


def verify_delta(settings, delta, resolve_overlay):
    expected = resolve_overlay(settings)

    def matches(candidate):
        try:
            return same_value(resolve_overlay(candidate), expected)
        except ValueError:
            return False

    if same_value(settings, delta) or matches(delta):
        return delta
    # An equal value can still override a different spelling supplied by the parent or a nested run setting.
    result = settings
    for path in redundant_paths(settings, delta):
        candidate = without_path(result, path)
        if matches(candidate):
            result = candidate
    return result


def import_profile(path, exported, name, parent):
    if set(exported) != {"version", "profile", "settings"} or type(exported["version"]) is not int or exported["version"] != 1:
        raise ValueError("expected an exported version 1 profile")
    original = profile_name(exported["profile"])
    name = original if name is None else profile_name(name)
    settings = mapping(exported["settings"], "imported profile settings")
    check_keys(settings, DEFAULTS.keys(), "imported setting")
    if parent == name:
        raise ValueError(f"profile {name!r} cannot be its own parent")

    path = path.resolve()
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            document = read_document(path)
            mode = path.stat().st_mode & 0o777
        except FileNotFoundError:
            document, mode = {}, None
        controls, inherited = configured_profile(document, {"profile": parent}, {})
        legacy_wins = controls["legacy_name_precedence"] == "legacy"
        if name in profiles_from(document, legacy_wins):
            raise ValueError(f"profile {name!r} already exists; use --profile to choose a new name")
        settings = normalize_settings(settings, legacy_wins)
        # The parent contributes any fields absent from the exported profile. Explicit null values can override it.
        entry = {"extends": parent} if parent is not None else {}
        entries = mapping(document.get("profiles", {}), "profiles").copy()
        updated = {**document, "profiles": entries}

        def resolve_overlay(overlay):
            entries[name] = {**entry, "settings": overlay}
            return configured_profile(updated, {"profile": name}, {}, sparse=False)[1]

        delta = verify_delta(settings, profile_delta(settings, inherited), resolve_overlay)
        entries[name] = {**entry, "settings": delta}
        write_json(path, updated, mode=mode)
    return name


def exchange_profile(document, cli, environ):
    if "_export_profile" in cli:
        controls, settings = configured_profile(document, cli, environ)
        if controls["profile"] is None:
            raise ValueError("select a profile to export with --profile")
        exported = {"version": 1, "profile": controls["profile"], "settings": settings}
        destination = cli["_export_profile"]
        if destination == "-":
            sys.stdout.write(encoded(exported))
        else:
            write_json(Path(destination), exported)
        return
    path = Path(cli["_config_path"])
    exported = read_document(cli["_import_profile"])
    name = import_profile(path, exported, cli.get("profile"), cli["_parent_profile"])
    print(f"Imported profile {name!r} into {path}")
