"""Deferred settings for one configured HTTP service."""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import fcntl
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import sys
import tempfile
from threading import Lock
from types import MappingProxyType


STARTUP_FIELDS = frozenset({
    "sampling.temperature", "sampling.top_p", "sampling.top_k", "sampling.min_p",
    "reasoning_budget_tokens", "fit_max_tokens", "anthropic_thinking", "effort_position",
    "aliases", "idle_unload_s", "mtp", "lazy_load", "engine_silence_s", "api_monitor",
    "open_browser", "vram_reserve_mib",
})
FLOAT32_MAX = float.fromhex("0x1.fffffep127")
INT32_MAX = 2**31 - 1
STARTUP_LIMITS = {
    "sampling.temperature": FLOAT32_MAX,
    "idle_unload_s": sys.float_info.max,
    "engine_silence_s": sys.float_info.max,
    "vram_reserve_mib": INT32_MAX,
    "reasoning_budget_tokens": INT32_MAX,
}


def state_path(environ, identity):
    root = Path(environ.get("XDG_STATE_HOME", ""))
    if not root.is_absolute():
        root = Path.home() / ".local/state"
    digest = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    return root / "scream-p100" / "instances" / f"{digest}.json"


def read_state(path):
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"version": 1, "next_start": {}, "shared": {}}
    if (not isinstance(state, dict) or set(state) != {"version", "next_start", "shared"}
            or type(state["version"]) is not int or state["version"] != 1
            or not isinstance(state["next_start"], dict) or not isinstance(state["shared"], dict)
            or state["next_start"].keys() - STARTUP_FIELDS):
        raise ValueError(f"invalid deferred settings file {path}")
    return state


def startup_layer(changes):
    settings = {}
    for key, value in changes.items():
        if key.startswith("sampling."):
            settings.setdefault("sampling", {})[key.partition(".")[2]] = value
        elif key == "vram_reserve_mib":
            settings.setdefault("engine_options", {})["--vram-reserve-mib"] = value
        else:
            settings.setdefault("strata_settings", {})[key] = value
    return settings


def clear_deferred_fields(config, changes):
    # The adapter's API removes these fields on null. Keep that tombstone through
    # source resolution, then omit the key if a higher-priority source left it cleared.
    for key, value in changes.items():
        if value is None and config.get(key) is None and "." not in key and key != "vram_reserve_mib":
            config.pop(key, None)


class RuntimeSettings:
    def __init__(self, config, plan, validate_shared):
        self._lock = Lock()
        self._file = Path(plan["path"]) if plan["path"] else None
        self._startup = deepcopy(config)
        self._proposed = self._startup
        self.startup_fields = STARTUP_FIELDS
        if config.get("scream_backend") == "strata-split":
            self.startup_fields = STARTUP_FIELDS - {"vram_reserve_mib"}
        self._shared = MappingProxyType(validate_shared(plan["state"]["shared"]))
        self._pending = None
        self._request = ContextVar("scream_request_settings", default=None)
        self._reserve = (None, False)
        self._device_reserves = {}

    def _persist(self, *, changes=None, shared=None):
        if self._file is None:
            return []
        self._file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self._file.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = read_state(self._file)
            recorded = []
            if changes is not None:
                saved = state["next_start"]
                recorded = [key for key, value in changes.items() if key not in saved or saved[key] != value]
                if not recorded and shared is None:
                    return recorded
                state["next_start"].update(changes)
            if shared is not None:
                state["shared"] = dict(shared)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self._file.parent, delete=False) as file:
                    temporary = Path(file.name)
                    json.dump(state, file, allow_nan=False)
                    file.flush()
                    os.fsync(file.fileno())
                temporary.replace(self._file)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            return recorded

    @contextmanager
    def request(self):
        # Preparing a request may fail or be cancelled, so capturing a snapshot does not publish it.
        with self._lock:
            snapshot = self._shared if self._pending is None else self._pending
        token = self._request.set(snapshot)
        try:
            yield
        finally:
            self._request.reset(token)

    def complete(self):
        snapshot = self._request.get()
        with self._lock:
            if snapshot is not None and self._pending is snapshot:
                self._shared, self._pending = snapshot, None

    def shared(self):
        snapshot = self._request.get()
        return self._shared if snapshot is None else snapshot

    def shared_status(self):
        with self._lock:
            settings = self._shared if self._pending is None else self._pending
            return {"shared": bool(settings), "defaults": dict(settings), "pending": self._pending is not None}

    def queue_shared(self, settings):
        with self._lock:
            self._persist(shared=settings)
            self._pending = MappingProxyType(settings.copy())
        return settings

    def config(self, *, proposed=False):
        with self._lock:
            return deepcopy(self._proposed if proposed else self._startup)

    def queue_startup(self, changes, apply, value_of):
        with self._lock:
            if isinstance(changes, dict):
                if unknown := changes.keys() - self.startup_fields:
                    raise ValueError(f"unsupported startup settings: {', '.join(sorted(unknown))}")
                for key, value in changes.items():
                    if ((type(value) is float and not isfinite(value))
                            or (key in STARTUP_LIMITS and type(value) in (int, float) and value > STARTUP_LIMITS[key])):
                        raise ValueError(f"{key} must be finite and representable by its backend")
            proposed, changed = apply(self._proposed, changes)
            edits = {key: value_of(proposed, key) for key in changes}
            recorded = self._persist(changes=edits)
            self._proposed = proposed
            return deepcopy(proposed), [key for key in changes if key in changed or key in recorded]

    def reserve(self):
        return self._reserve

    def reserve_devices(self):
        return self._device_reserves.copy()

    def record_reserve(self, value, device=None):
        # Strata calls this inside its existing FIFO/control gate after a successful resize or while unloaded.
        if device is None:
            self._reserve = (value, True)
            self._device_reserves.clear()
        else:
            self._device_reserves[device] = value
