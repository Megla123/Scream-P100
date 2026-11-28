from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO, StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Thread
import time
import unittest
from unittest.mock import patch

from parity import compare, consume_reply, main
from nvidia_vram import VramSampler, descendants, memory_by_gpu


class ParityTests(unittest.TestCase):
    def test_streamed_tools_have_the_same_signature_as_complete_tools(self):
        def choice(delta, finish=None):
            return {"choices": [{"delta": delta, "finish_reason": finish}]}

        def tool(function, **extra):
            return {"tool_calls": [{"index": 0, "function": function, **extra}]}

        frames = [
            choice({"role": "assistant"}),
            choice({"reasoning_content": "ready"}),
            choice(tool({"name": "edit", "arguments": '{"x":'}, id="stream-id")),
            choice(tool({"arguments": '"é"}'})),
            {**choice({}, "tool_calls"), "timings": {"cache_n": 3, "prompt_per_second": 150}},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4}},
        ]
        stream = BytesIO(b": keepalive\r\n\r\n" + b"".join(b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames)
                         + b"data: [DONE]\n\n")
        complete = BytesIO(json.dumps({"choices": [{"finish_reason": "tool_calls", "message": {
            "reasoning_content": "ready", "tool_calls": [{"id": "complete-id", "function": {"name": "edit", "arguments": '{"x": "é"}'}}],
        }}]}).encode())
        first = consume_reply(stream, True, time.perf_counter())
        second = consume_reply(complete, False, time.perf_counter())
        self.assertEqual(first["output_sha256"], second["output_sha256"])
        self.assertEqual((first["cached_tokens"], first["prefill_tokens_per_s"]), (3, 150))
        self.assertIsNotNone(first["first_output_s"])
        self.assertIsNone(second["first_output_s"])
        self.assertGreaterEqual(first["first_tool_argument_s"], first["first_output_s"])
        self.assertEqual(first["tool_argument_frames"], 2)
        self.assertIsNone(second["tool_argument_frames"])

    def test_sse_traffic_and_model_progress_have_separate_clocks(self):
        response = BytesIO(
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            b': keepalive\n\n'
            b'data: {"choices":[{"delta":{"reasoning_content":"first"}}]}\n\n'
            b': keepalive\n\n'
            b'data: {"choices":[{"delta":{"content":"second"}}]}\n\n'
            b'data: {"choices":[{"finish_reason":"stop"}]}\n\n'
            b'data: [DONE]\n\n'
        )
        with patch("parity.time.perf_counter", side_effect=[10.2, 10.5, 10.8, 11.1, 11.4, 11.5, 11.6, 11.6]):
            result = consume_reply(response, True, 10.0, 10.1)
        self.assertAlmostEqual(result["headers_s"], 0.1)
        self.assertAlmostEqual(result["first_output_s"], 0.8)
        self.assertAlmostEqual(result["max_sse_gap_s"], 0.3)
        self.assertAlmostEqual(result["max_output_gap_s"], 0.6)
        self.assertEqual(result["sse_comments"], 2)

    def test_stream_requires_finish_and_done(self):
        for frames in (
            b'data: {"choices":[{"finish_reason":"stop"}]}\n\n',
            b'data: [DONE]\n\n',
            b'data: {"error":{"message":"worker died"}}\n\ndata: [DONE]\n\n',
        ):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                consume_reply(BytesIO(frames), True, time.perf_counter())

    def test_compare_requires_the_same_workload_and_hides_mismatched_ratios(self):
        baseline = {"name": "work", "request_sha256": "request", "output_sha256": "reply", "finish": "stop",
                    "prompt_tokens": 12, "completion_tokens": 8, "cached_tokens": 0, "elapsed_s": 2,
                    "first_output_s": 1, "prefill_tokens_per_s": 100, "decode_tokens_per_s": 50}
        with TemporaryDirectory() as directory:
            old, new = (Path(directory) / name for name in ("old.json", "new.json"))
            old.write_text(json.dumps({"version": 1, "label": "old", "lookup_override": 0, "cases": [baseline]}))
            for changes, expected, ratio in (({"decode_tokens_per_s": 60}, 0, True),
                                             ({"decode_tokens_per_s": 60, "cached_tokens": 3}, 0, False),
                                             ({"decode_tokens_per_s": 60, "output_sha256": "different"}, 1, False)):
                revised = {**baseline, **changes}
                new.write_text(json.dumps({"version": 1, "label": "new", "lookup_override": 7, "cases": [revised]}))
                with redirect_stdout(StringIO()) as stdout:
                    self.assertEqual(compare(old, new), expected)
                self.assertIn("Request lookup override: 0 / 7", stdout.getvalue())
                self.assertEqual("+20.0%" in stdout.getvalue(), ratio)
            new.write_text(json.dumps({"version": 1, "label": "new", "cases": [{**baseline, "request_sha256": "different"}]}))
            with self.assertRaisesRegex(ValueError, "same requests"):
                compare(old, new)

    def test_capture_overlaps_requests_and_keeps_workload_order(self):
        overlap = Barrier(2, timeout=5)

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                name = request["messages"][0]["content"]
                if name != "tail":
                    overlap.wait()
                if request["stream"]:
                    body = (b'data: {"choices":[{"delta":{"content":"reply"},"finish_reason":"stop"}]}\n\n'
                            b'data: {"usage":{"prompt_tokens":10,"completion_tokens":2}}\n\n'
                            b'data: [DONE]\n\n')
                    content_type = "text/event-stream"
                else:
                    body = (b'{"choices":[{"message":{"content":"reply"},"finish_reason":"stop"}],'
                            b'"usage":{"prompt_tokens":10,"completion_tokens":2}}')
                    content_type = "application/json"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        with TemporaryDirectory() as directory, ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            requests, output = (Path(directory) / name for name in ("workload.json", "capture.json"))
            names = ["streaming", "complete", "tail"]
            requests.write_text(json.dumps([
                {"name": name, "request": {"messages": [{"role": "user", "content": name}], "stream": name == "streaming"}}
                for name in names
            ]))
            listener = Thread(target=server.serve_forever)
            listener.start()
            try:
                with redirect_stderr(StringIO()):
                    self.assertEqual(main(["capture", "--url", f"http://127.0.0.1:{server.server_port}",
                                           "--model", "fixture", "--requests", str(requests), "--label", "parallel",
                                           "--out", str(output), "--concurrency", "2", "--timeout", "10"]), 0)
            finally:
                server.shutdown()
                listener.join()
            report = json.loads(output.read_text())
            self.assertEqual([row["name"] for row in report["cases"]], names)
            self.assertEqual([row["completion_tokens"] for row in report["cases"]], [2, 2, 2])
            self.assertEqual(report["concurrency"], 2)
            self.assertEqual([group["request_count"] for group in report["groups"]], [2, 1])
            self.assertGreaterEqual(report["groups"][0]["elapsed_s"], max(row["elapsed_s"] for row in report["cases"][:2]))

    def test_compare_concurrent_group_throughput_requires_comparable_cases(self):
        case = {"name": "work", "request_sha256": "request", "output_sha256": "reply", "finish": "stop",
                "prompt_tokens": 12, "completion_tokens": 8, "cached_tokens": 0, "elapsed_s": 0.5}
        cases = [case, {**case, "name": "other"}]
        report = {"version": 1, "label": "capture", "cases": cases, "concurrency": 2}
        with TemporaryDirectory() as directory:
            old, new = (Path(directory) / name for name in ("old.json", "new.json"))
            old.write_text(json.dumps({**report, "groups": [{"request_count": 2, "elapsed_s": 2}]}))
            candidate = {**report, "groups": [{"request_count": 2, "elapsed_s": 1}]}
            for changes, exit_code, rate in (({}, 0, "8 / 16 (+100.0%)"), ({"cached_tokens": 1}, 0, "8 / 16"),
                                             ({"output_sha256": "other"}, 1, "8 / 16"),
                                             ({"completion_tokens": None}, 0, "8 / unavailable")):
                new.write_text(json.dumps({**candidate, "cases": [case, {**cases[1], **changes}]}))
                with redirect_stdout(StringIO()) as stdout:
                    self.assertEqual(compare(old, new), exit_code)
                self.assertIn(f"  completion_tokens_per_s: {rate}\n", stdout.getvalue())
            new.write_text(json.dumps({**candidate, "concurrency": 3}))
            with self.assertRaisesRegex(ValueError, "same concurrency"):
                compare(old, new)
            new.write_text(json.dumps({**candidate, "groups": [{"request_count": 1, "elapsed_s": 0.5}] * 2}))
            with self.assertRaisesRegex(ValueError, "group boundaries"):
                compare(old, new)

    def test_vram_excludes_unrelated_and_reused_processes(self):
        table = {8: (1, 100), 9: (8, 110), 10: (9, 120), 11: (1, 105)}
        selected = descendants(table, {8: 100, 9: 110})
        self.assertEqual(selected, {8, 9, 10})
        with self.assertRaisesRegex(ValueError, "exited or was reused"):
            descendants(table, {8: 99})
        rows = [("GPU-a", "8", "100"), ("GPU-a", "9", "20"),
                ("GPU-c", "10", "[N/A]"), ("GPU-a", "11", "300")]
        self.assertEqual(memory_by_gpu(rows, selected, ["GPU-a", "GPU-b", "GPU-c"]),
                         {"GPU-a": 120, "GPU-b": 0, "GPU-c": None})

    def test_vram_reports_sampled_peaks_in_each_request(self):
        sampler = object.__new__(VramSampler)
        sampler.devices = {"GPU-a": {}, "GPU-b": {}}
        sampler.samples = [(1, {"GPU-a": 10, "GPU-b": 50}),
                           (2, {"GPU-a": 40, "GPU-b": None}),
                           (3, {"GPU-a": 25, "GPU-b": 80})]
        self.assertEqual(sampler.window(1.5, 3), {"samples": 2, "peak_mib": {"GPU-a": 40, "GPU-b": None}})
        self.assertEqual(sampler.window(3.1, 3.2), {"samples": 0, "peak_mib": {"GPU-a": None, "GPU-b": None}})


if __name__ == "__main__":
    unittest.main()
