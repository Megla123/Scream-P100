"""Sample NVIDIA compute allocations for a Linux worker and its descendants."""
import csv
import subprocess
import threading
import time
from pathlib import Path


def smi_rows(query, columns):
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-{query}={columns}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError(f"nvidia-smi {query} query timed out") from error
    if result.returncode:
        raise ValueError(f"nvidia-smi {query} query failed: {result.stderr.strip() or result.stdout.strip()}")
    return (row for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True) if row)


def process_table():
    table = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            # The comm field can contain spaces and closing parentheses.
            fields = (entry / "stat").read_text(errors="replace").rsplit(")", 1)[1].split()
            table[int(entry.name)] = (int(fields[1]), int(fields[19]))
        except (OSError, ValueError, IndexError):
            continue
    return table


def descendants(table, roots):
    children = {}
    for pid, (parent, _) in table.items():
        children.setdefault(parent, []).append(pid)
    pending = list(roots)
    selected = set(roots)
    for pid, start in roots.items():
        if pid not in table or table[pid][1] != start:
            raise ValueError(f"VRAM worker PID {pid} exited or was reused")
    while pending:
        for child in children.get(pending.pop(), ()):
            if child not in selected:
                selected.add(child)
                pending.append(child)
    return selected


def memory_by_gpu(rows, selected, devices):
    totals = dict.fromkeys(devices, 0)
    for uuid, pid, raw in rows:
        if int(pid) not in selected:
            continue
        if uuid not in totals:
            raise ValueError(f"nvidia-smi reported a new GPU in the worker tree: {uuid}")
        if totals[uuid] is None:
            continue
        if not raw.strip().isdecimal():
            totals[uuid] = None
        else:
            totals[uuid] += int(raw)
    return totals


class VramSampler:
    def __init__(self, pids, interval):
        table = process_table()
        self.roots = {}
        for pid in pids:
            if pid not in table:
                raise ValueError(f"VRAM worker PID {pid} is unavailable")
            self.roots[pid] = table[pid][1]
        self.interval = interval
        self.devices = {
            uuid: {"name": name, "total_mib": int(total), "driver": driver}
            for uuid, name, total, driver in smi_rows("gpu", "uuid,name,memory.total,driver_version")
        }
        if not self.devices:
            raise ValueError("nvidia-smi found no NVIDIA GPUs")
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.poll, name="nvidia-vram", daemon=True)
        self.error = None

    def sample(self):
        rows = list(smi_rows("compute-apps", "gpu_uuid,pid,used_gpu_memory"))
        observed = time.perf_counter()
        selected = descendants(process_table(), self.roots)
        self.samples.append((observed, memory_by_gpu(rows, selected, self.devices)))

    def poll(self):
        try:
            while not self.stop.wait(self.interval):
                self.sample()
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            self.error = error

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, exc_type, _exc, _traceback):
        self.stop.set()
        self.thread.join()
        if exc_type is None:
            if self.error is not None:
                raise ValueError(f"VRAM sampling stopped: {self.error}") from self.error
            self.sample()

    def window(self, started, finished):
        selected = [sample for when, sample in self.samples if started <= when <= finished]
        peak = {}
        for uuid in self.devices:
            values = [sample[uuid] for sample in selected]
            peak[uuid] = max(values) if values and None not in values else None
        return {"samples": len(selected), "peak_mib": peak}

    def report(self):
        return {"devices": self.devices, "interval_s": self.interval, "root_pids": sorted(self.roots),
                "before_mib": self.samples[0][1], "after_mib": self.samples[-1][1],
                **self.window(self.samples[0][0], self.samples[-1][0])}
