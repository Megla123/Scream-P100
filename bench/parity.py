#!/usr/bin/env python3
"""Capture and compare identical request sequences on isolated Scream P100 listeners."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import http.client
import ipaddress
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from threading import Event
import time
from urllib.parse import urlsplit

if __package__:
    from .nvidia_vram import VramSampler
else:
    from nvidia_vram import VramSampler


RATES = {"prefill_tokens_per_s": "prompt_per_second", "decode_tokens_per_s": "predicted_per_second"}
NATIVE_METRICS = {"prefill_ms": "prompt_ms", "decode_ms": "predicted_ms",
                  "drafts_offered": "draft_n", "drafts_accepted": "draft_n_accepted"}
METRICS = (*RATES, *NATIVE_METRICS, "headers_s", "first_output_s", "max_sse_gap_s", "max_output_gap_s", "sse_comments", "elapsed_s")
TOOL_METRICS = ("first_tool_argument_s", "tool_argument_frames")
COUNTS = ("prompt_tokens", "completion_tokens", "cached_tokens")
SSE_COMMENT = object()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sse_records(response):
    data = []
    for line in response:
        line = line.rstrip(b"\r\n")
        if not line and data:
            event = b"\n".join(data)
            if event == b"[DONE]":
                yield None
                return
            value = json.loads(event)
            if not isinstance(value, dict):
                raise ValueError("invalid SSE event")
            yield value
            data.clear()
        elif line.startswith(b"data:"):
            data.append(line[5:].removeprefix(b" "))
        elif line.startswith(b":"):
            yield SSE_COMMENT


def add_calls(calls, updates, stream):
    if not isinstance(updates, list):
        raise ValueError("invalid tool call list")
    has_arguments = False
    for offset, update in enumerate(updates):
        if not isinstance(update, dict):
            raise ValueError("invalid tool call event")
        index = update.get("index") if stream else offset
        if type(index) is not int or index < 0:
            raise ValueError("invalid tool call index")
        call = calls.setdefault(index, {"id": None, "name": "", "arguments": ""})
        if "id" in update:
            supplied = update["id"]
            if not isinstance(supplied, str) or not supplied or call["id"] not in (None, supplied):
                raise ValueError("tool call ID changed or was invalid")
            call["id"] = supplied
        function = update.get("function") or {}
        if not isinstance(function, dict):
            raise ValueError("invalid function call event")
        arguments = function.get("arguments", "")
        name = function.get("name", "")
        if not isinstance(arguments, str) or not isinstance(name, str):
            raise ValueError("tool name and argument fragments must be strings")
        call["name"] += name
        call["arguments"] += arguments
        has_arguments |= bool(arguments)
    return has_arguments


def complete_calls(calls):
    if sorted(calls) != list(range(len(calls))) or len({c["id"] for c in calls.values()}) != len(calls):
        raise ValueError("invalid tool call sequence")
    result = []
    for index in range(len(calls)):
        call = calls[index]
        args = json.loads(call["arguments"])
        if not call["id"] or not call["name"] or not isinstance(args, dict):
            raise ValueError("incomplete function call")
        result.append({"name": call["name"], "arguments": args})
    return result


def consume_reply(response, stream, started, headers_at=None):
    records = sse_records(response) if stream else (json.load(response),)
    done, finish, first = not stream, None, None
    first_argument, argument_frames = None, 0
    last_sse, max_sse_gap = headers_at if headers_at is not None else started, None
    last_output, max_output_gap, comments = None, None, 0
    content, reasoning, calls, usage, timings = [], [], {}, {}, {}
    for record in records:
        arrived = time.perf_counter()
        if stream:
            max_sse_gap = max(max_sse_gap or 0.0, arrived - last_sse)
            last_sse = arrived
        if record is SSE_COMMENT:
            comments += 1
            continue
        if record is None and stream:
            done = True
            break
        if not isinstance(record, dict) or record.get("error"):
            raise ValueError("server returned an invalid or error response")
        usage.update(record.get("usage") or {})
        timings.update(record.get("timings") or {})
        choices = record.get("choices") or []
        if not choices:
            continue
        if (not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict)
                or choices[0].get("index", 0) != 0 or finish is not None):
            raise ValueError("unexpected completion choice")
        choice = choices[0]
        message = choice.get("delta" if stream else "message") or {}
        if not isinstance(message, dict):
            raise ValueError("invalid assistant message")
        text = message.get("content") or ""
        thought = message.get("reasoning_content") or ""
        updates = message.get("tool_calls") or []
        has_arguments = add_calls(calls, updates, stream)
        if stream and (text or thought or has_arguments or any((update.get("function") or {}).get("name") for update in updates)):
            if last_output is None:
                first = arrived - started
            else:
                max_output_gap = max(max_output_gap or 0.0, arrived - last_output)
            last_output = arrived
        if stream and has_arguments:
            argument_frames += 1
            if first_argument is None:
                first_argument = arrived - started
        content.append(text)
        reasoning.append(thought)
        finish = choice.get("finish_reason")
    finished = time.perf_counter()
    if not done or not isinstance(finish, str) or finish in ("error", "cancel"):
        raise ValueError("completion has no successful finish and stream terminator")
    completed_calls = complete_calls(calls)
    if finish == "tool_calls" and not completed_calls:
        raise ValueError("tool-call finish has no completed call")
    signature = {"content": "".join(content), "reasoning": "".join(reasoning), "calls": completed_calls, "finish": finish}
    return {"output_sha256": digest(signature), "finish": finish,
            "headers_s": headers_at - started if headers_at is not None else None, "first_output_s": first,
            "max_sse_gap_s": max_sse_gap, "max_output_gap_s": max_output_gap, "sse_comments": comments if stream else None,
            "first_tool_argument_s": first_argument, "tool_argument_frames": argument_frames if stream else None,
            "elapsed_s": finished - started, "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", timings.get("cache_n")),
            **{name: timings.get(key) for name, key in (RATES | NATIVE_METRICS).items()}}


def request_sequence(path):
    sequence = read_json(path)
    if not isinstance(sequence, list) or not sequence:
        raise ValueError("requests must be a nonempty JSON array")
    seen = set()
    for item in sequence:
        name = item.get("name") if isinstance(item, dict) else None
        if not isinstance(name, str) or not name.strip() or name in seen:
            raise ValueError("request names must be nonempty and unique")
        seen.add(name)
        request = item.get("request")
        if (not isinstance(request, dict) or "model" in request or not isinstance(request.get("messages"), list)
                or request.get("n", 1) != 1 or type(request.get("stream", False)) is not bool):
            raise ValueError(f"{name}: supply messages, at most one choice, and use --model for the model alias")
        if request.get("strata_mcp") is True:
            raise ValueError(f"{name}: server-side tool execution is outside the benchmark")
        yield name, {"stream": False, **request}


def capture(options):
    url = urlsplit(options.url)
    if (url.scheme != "http" or not url.hostname or not ipaddress.ip_address(url.hostname).is_loopback
            or not url.port or url.username is not None or url.path not in ("", "/") or url.query or url.fragment):
        raise ValueError("use an explicit HTTP loopback origin with a port")
    if not math.isfinite(options.timeout) or options.timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if not math.isfinite(options.vram_interval) or options.vram_interval < 0.1:
        raise ValueError("VRAM sample interval must be finite and at least 0.1 s")
    if options.concurrency < 1:
        raise ValueError("concurrency must be positive")
    output = Path(options.out)
    if output.exists():
        raise ValueError(f"capture already exists: {output}")
    headers = {"Content-Type": "application/json"}
    if options.token_file:
        headers["Authorization"] = "Bearer " + Path(options.token_file).read_text(encoding="utf-8").strip()
    requests = list(request_sequence(options.requests))
    if options.lookup is not None:
        if not 0 <= options.lookup <= 15:
            raise ValueError("--lookup must be between 0 and 15")
        if any("lookup" in body for _, body in requests):
            raise ValueError("--lookup requires workload requests without a lookup field")
    def issue(name, body, start_group=None):
        payload = {**body, "model": options.model}
        if options.lookup is not None:
            payload["lookup"] = options.lookup
        connection = http.client.HTTPConnection(url.hostname, url.port, timeout=options.timeout)
        try:
            encoded = json_bytes(payload)
            if start_group is not None:
                start_group.wait()
            started = time.perf_counter()
            connection.request("POST", "/v1/chat/completions", encoded, headers)
            with connection.getresponse() as response:
                headers_at = time.perf_counter()
                expected = "text/event-stream" if body["stream"] else "application/json"
                if response.status != 200 or response.headers.get_content_type() != expected:
                    raise ValueError(f"HTTP {response.status} ({response.headers.get_content_type()})")
                result = consume_reply(response, body["stream"], started, headers_at)
        except (OSError, ValueError, TypeError, http.client.HTTPException) as error:
            raise ValueError(f"{name}: {error}") from error
        finally:
            connection.close()
        return {"name": name, "request_sha256": digest(body), **result}, (started, started + result["elapsed_s"])

    rows, windows, groups = [], [], []
    context = VramSampler(options.worker_pid, options.vram_interval) if options.worker_pid else nullcontext()
    executor = (ThreadPoolExecutor(max_workers=min(options.concurrency, len(requests)))
                if options.concurrency > 1 else nullcontext())
    with context as sampler, executor as pool:
        for start in range(0, len(requests), options.concurrency):
            group = requests[start:start + options.concurrency]
            if pool is None:
                results = [issue(*group[0])]
            else:
                start_group = Event()
                try:
                    futures = [pool.submit(issue, name, body, start_group) for name, body in group]
                finally:
                    start_group.set()
                results = [future.result() for future in futures]
            for row, window in results:
                rows.append(row)
                windows.append(window)
                print(f"{row['name']}: {row['finish']} in {row['elapsed_s']:.3f} s", file=sys.stderr, flush=True)
            if pool is not None:
                starts, finishes = zip(*(window for _, window in results))
                groups.append({"request_count": len(group), "elapsed_s": max(finishes) - min(starts)})
    report = {"version": 1, "label": options.label, "model": options.model, "cases": rows}
    if groups:
        report.update(concurrency=options.concurrency, groups=groups)
    if options.lookup is not None:
        report["lookup_override"] = options.lookup
    if sampler is not None:
        report["vram"] = sampler.report()
        for row, (started, finished) in zip(rows, windows):
            row["vram"] = sampler.window(started, finished)
    data = json_bytes(report) + b"\n"
    with tempfile.NamedTemporaryFile(dir=output.parent) as target:
        target.write(data)
        target.flush()
        os.link(target.name, output)


def capture_groups(report, concurrency):
    groups = report.get("groups", [])
    count = len(report["cases"])
    expected = [min(concurrency, count - start) for start in range(0, count, concurrency)] if concurrency > 1 else []
    if not isinstance(groups, list) or [group["request_count"] for group in groups] != expected:
        raise ValueError("invalid capture group boundaries")
    if any(not math.isfinite(group["elapsed_s"]) or group["elapsed_s"] <= 0 for group in groups):
        raise ValueError("invalid capture group duration")
    return groups


def print_metric(key, before, after, comparable):
    old_value = "unavailable" if before is None else f"{before:.4g}"
    new_value = "unavailable" if after is None else f"{after:.4g}"
    percent = f" ({(after / before - 1) * 100:+.1f}%)" if comparable and before and after is not None else ""
    print(f"  {key}: {old_value} / {new_value}{percent}")


def compare(baseline, candidate):
    before, after = read_json(baseline), read_json(candidate)
    if any(not isinstance(report, dict) or report.get("version") != 1 for report in (before, after)):
        raise ValueError("unsupported capture version")
    old, new = before["cases"], after["cases"]
    old_cases, new_cases = ([(row["name"], row["request_sha256"]) for row in capture] for capture in (old, new))
    if not old_cases or old_cases != new_cases:
        raise ValueError("the captures must use the same requests in the same order")
    concurrency = before.get("concurrency", 1)
    if type(concurrency) is not int or concurrency < 1 or type(after.get("concurrency", 1)) is not int:
        raise ValueError("invalid capture concurrency")
    if concurrency != after.get("concurrency", 1):
        raise ValueError("the captures must use the same concurrency")
    old_groups, new_groups = (capture_groups(report, concurrency) for report in (before, after))
    print(f"Baseline {before['label']} | Candidate {after['label']}")
    if "lookup_override" in before or "lookup_override" in after:
        print(f"  Request lookup override: {before.get('lookup_override', 'none')} / {after.get('lookup_override', 'none')}")
    old_vram, new_vram = before.get("vram", {}), after.get("vram", {})
    if old_vram or new_vram:
        print("\nNVIDIA VRAM, worker trees")
        if old_vram.get("devices") != new_vram.get("devices"):
            print("  GPU inventory or driver differs. Match devices before comparing memory.")
        print_vram(old_vram, new_vram, ("before_mib", "peak_mib", "after_mib"))
    different = False
    comparable_timings = []
    for reference, changed in zip(old, new):
        print(f"\n{reference['name']}")
        same = reference["output_sha256"] == changed["output_sha256"]
        different |= not same
        print(f"  Response {'matches' if same else 'differs'} ({reference['finish']} / {changed['finish']})")
        comparable = same and all(reference[key] is not None and reference[key] == changed[key] for key in COUNTS)
        comparable_timings.append(comparable)
        for key in COUNTS:
            print(f"  {key}: {reference[key]} / {changed[key]}")
        metrics = (*METRICS, *TOOL_METRICS) if reference.get("tool_argument_frames") or changed.get("tool_argument_frames") else METRICS
        for key in metrics:
            if key not in reference and key not in changed:
                continue
            print_metric(key, reference.get(key), changed.get(key), comparable)
        if reference.get("vram") or changed.get("vram"):
            print_vram(reference.get("vram", {}), changed.get("vram", {}), ("peak_mib",))
        if not comparable:
            print("  Timing ratios require a matching response and matching token and cache counts.")
    for index, (reference, changed) in enumerate(zip(old_groups, new_groups)):
        start = index * concurrency
        end = start + reference["request_count"]
        comparable = all(comparable_timings[start:end])
        count = end - start
        print(f"\nConcurrent group {index + 1}, {count} {'request' if count == 1 else 'requests'}")
        print_metric("elapsed_s", reference["elapsed_s"], changed["elapsed_s"], comparable)
        rates = []
        for cases, group in ((old, reference), (new, changed)):
            counts = [row["completion_tokens"] for row in cases[start:end]]
            rates.append(sum(counts) / group["elapsed_s"] if None not in counts else None)
        print_metric("completion_tokens_per_s", *rates, comparable)
        if not comparable:
            print("  Group ratios require matching responses and token and cache counts for every request.")
    return int(different)


def print_vram(before, after, fields):
    print(f"  VRAM samples: {before.get('samples', 0)} / {after.get('samples', 0)}")
    for field in fields:
        old, new = before.get(field, {}), after.get(field, {})
        for uuid in sorted(old.keys() | new.keys()):
            a, b = old.get(uuid), new.get(uuid)
            print(f"  {uuid} {field}: {a if a is not None else 'unavailable'} / {b if b is not None else 'unavailable'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("capture")
    for name in ("url", "model", "requests", "label", "out"):
        collect.add_argument("--" + name, required=True)
    collect.add_argument("--token-file", help="read the listener's bearer token from this file")
    collect.add_argument("--timeout", type=float, default=3600, help="HTTP socket timeout in seconds")
    collect.add_argument("--concurrency", type=int, default=1, help="requests to start together (default: 1)")
    collect.add_argument("--lookup", type=int, help="override Inkling prompt lookup for every request (0 disables it)")
    collect.add_argument("--worker-pid", type=int, action="append", help="sample this Linux worker process and its descendants with nvidia-smi (repeatable)")
    collect.add_argument("--vram-interval", type=float, default=0.5, help="delay between VRAM samples in seconds")
    difference = commands.add_parser("compare")
    difference.add_argument("baseline")
    difference.add_argument("candidate")
    options = parser.parse_args(argv)
    try:
        if options.command == "compare":
            return compare(options.baseline, options.candidate)
        capture(options)
        return 0
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(2, f"parity: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
