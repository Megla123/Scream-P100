"""Validate the CPU or single NVIDIA device selector shared by the launchers."""
import re


GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")


def vision_device(value):
    if value == "cpu" or isinstance(value, str) and GPU_UUID.fullmatch(value):
        return value
    raise ValueError("vision_device must be cpu or a full NVIDIA GPU UUID (GPU-...)")
