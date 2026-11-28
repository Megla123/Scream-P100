"""Validate and encode the shared native Inkling sampler settings."""

import math
import struct


SAMPLING_BASE = {
    "temperature": 0.0, "top_p": 1.0, "top_k": 64, "min_p": 0.0,
    "repetition_penalty": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0,
    "penalty_last_n": 64, "seed": 0,
}
SAMPLING_ALIASES = {"repeat_penalty": "repetition_penalty", "repeat_last_n": "penalty_last_n"}
SAMPLING_INTS = {"top_k": (1, 64), "penalty_last_n": (0, 2**31 - 1), "seed": (0, 2**64 - 1)}


def sampling_values(fields, *, max_top_k=64):
    values = {}
    for name in (*SAMPLING_BASE, *SAMPLING_ALIASES):
        raw = fields.get(name)
        if raw is None:
            continue
        key = SAMPLING_ALIASES.get(name, name)
        if key in values:
            continue  # Prefer the OpenAI/Strata name when both forms are present.
        if key in SAMPLING_INTS:
            low, high = SAMPLING_INTS[key]
            if key == "top_k":
                high = max_top_k
            if type(raw) is not int or not low <= raw <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
            values[key] = raw
            continue
        if type(raw) not in (int, float):
            raise ValueError(f"{name} must be a finite number")
        try:
            value = struct.unpack("f", struct.pack("f", raw))[0]
        except (OverflowError, struct.error):
            raise ValueError(f"{name} is outside the sampler's floating-point range") from None
        valid = {
            "temperature": value >= 0,
            "top_p": 0 < value <= 1,
            "min_p": 0 <= value <= 1,
            "repetition_penalty": value > 0,
            "frequency_penalty": -2 <= value <= 2,
            "presence_penalty": -2 <= value <= 2,
        }
        if not math.isfinite(value) or not valid[key]:
            raise ValueError(f"{name} is outside the supported range")
        values[key] = value
    return values


def sampling_fields(values, *, sparse=False):
    return " ".join(str(values.get(name, "-") if sparse else values[name]) for name in SAMPLING_BASE)
