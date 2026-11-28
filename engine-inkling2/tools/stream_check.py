"""Exercise the HTTP handler against a scripted Unix-socket engine, without CUDA or GGUF."""
from contextlib import ExitStack
import fcntl
import http.client
import importlib.util
import json
import os
from pathlib import Path
import resource
import socket
import sys
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SERVER = Path(__file__).resolve().parents[1] / "serve" / "ink2_server.py"


def load_server(directory, *, extra_args=(), settings=None):
    direct = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    extended = iter(range(256, 512))
    vocab = [chr(i if i in direct else next(extended)).encode() for i in range(256)]
    vocab.extend(marker.encode() for marker in (
        "<|message_model|>", "<|content_thinking|>", "<|content_text|>",
        "<|content_invoke_tool_json|>", "<|end_message|>", "<|unused_200054|>",
    ))
    field = SimpleNamespace(parts=vocab, data=range(len(vocab)))
    gguf = SimpleNamespace(GGUFReader=lambda _: SimpleNamespace(fields={"tokenizer.ggml.tokens": field}))
    args = ["--port", "0", "--sock", str(directory / "engine.sock"), "--model", "stub",
            "--tokenizer", "stub", "--gguf-py", str(directory)]
    spec = importlib.util.spec_from_file_location("ink2_server_fixture", SERVER)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, {"INK2_STREAM_TOOL_ARGUMENTS": "1", **(settings or {})}), patch.dict(sys.modules, {"gguf": gguf}):
        with patch.object(sys, "argv", [str(SERVER), *args, *extra_args]), patch.object(sys, "path", sys.path[:]):
            spec.loader.exec_module(module)
    module.TOKENS = SimpleNamespace(ids=lambda _: ([1, 2], []))
    module.KEEPALIVE_SECONDS = 0.05
    module.POLL_SECONDS = 0.02
    return module


def engine_record(conn, **value):
    conn.sendall(json.dumps(value).encode() + b"\n")


def next_frame(response):
    while line := response.readline():
        if line.startswith(b":"):
            return line.strip()
        if line.startswith(b"data: "):
            data = line[6:].strip()
            return data if data == b"[DONE]" else json.loads(data)
    return None


def frames(response):
    result = []
    while (frame := next_frame(response)) is not None:
        result.append(frame)
    return result


def tool_deltas(events):
    return [call for event in events if isinstance(event, dict) and "choices" in event
            for call in event["choices"][0]["delta"].get("tool_calls", [])]


class Streaming(unittest.TestCase):
    def setUp(self):
        self.resources = ExitStack()
        self.addCleanup(self.resources.close)
        directory = Path(self.resources.enter_context(TemporaryDirectory()))
        self.ink = load_server(directory)
        self.listener = self.resources.enter_context(socket.socket(socket.AF_UNIX, socket.SOCK_STREAM))
        self.listener.bind(self.ink.A.sock)
        self.listener.listen()
        self.listener.settimeout(2)
        self.http = self.resources.enter_context(self.ink.ThreadingHTTPServer(("127.0.0.1", 0), self.ink.H))
        self.thread = threading.Thread(target=self.http.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()
        self.addCleanup(self.http.shutdown)

    def request(self, stream=True, **settings):
        client = http.client.HTTPConnection("127.0.0.1", self.http.server_port, timeout=3)
        self.addCleanup(client.close)
        body = dict(model="inkling2-mini", stream=stream, messages=[],
                    tools=[{"type": "function", "function": {"name": "edit"}}])
        body.update(settings)
        client.request("POST", "/v1/chat/completions", json.dumps(body))
        return client

    def engine(self):
        conn, _ = self.listener.accept()
        self.resources.enter_context(conn)
        conn.settimeout(2)
        reader = self.resources.enter_context(conn.makefile("rb"))
        request = json.loads(reader.readline())
        self.assertEqual(request["stream"], 1)
        return conn, request

    def finish(self, conn, eos=True):
        if eos:
            engine_record(conn, t=[self.ink.EOS])
        engine_record(conn, tokens=[], cached=0)
        conn.shutdown(socket.SHUT_WR)

    def tool(self, conn, value, end=True, **outer):
        raw = json.dumps(dict(name="edit", args=value, **outer), ensure_ascii=False).encode()
        engine_record(conn, t=[self.ink.T_MODEL, self.ink.T_TOOL, *raw])
        if end:
            engine_record(conn, t=[self.ink.T_END])

    def test_streamed_tool_arguments_and_close(self):
        response = self.request().getresponse()
        self.resources.enter_context(response)
        first = next_frame(response)
        self.assertEqual(first["choices"][0]["delta"]["role"], "assistant")
        conn, _ = self.engine()
        patch_text = 'print("é\\{stuff}")\n' * 40
        value = {"patch": patch_text, "nested": [{"s": "}"}, 1]}
        raw = json.dumps({"name": "edit", "args": value}, ensure_ascii=False).encode()
        split = raw.index(b"\xc3") + 1
        engine_record(conn, t=[self.ink.T_TOOL, *raw[:split]])
        engine_record(conn, t=[*raw[split:]])
        early_events = []
        while (event := next_frame(response)) not in (None, b": keepalive"):
            early_events.append(event)
        self.assertTrue(tool_deltas(early_events))
        with self.assertRaises(json.JSONDecodeError):
            json.loads("".join(d["function"]["arguments"] for d in tool_deltas(early_events)))
        engine_record(conn, t=[self.ink.T_END])
        self.tool(conn, {"patch": "second"})
        self.finish(conn)
        tail = frames(response)
        deltas = tool_deltas(early_events + tail)
        first_call = [d for d in deltas if d["index"] == 0]
        self.assertGreaterEqual(len(first_call), 3)
        self.assertEqual([d["function"]["name"] for d in first_call if "id" in d], ["edit"])
        self.assertTrue(all("id" not in d and "name" not in d["function"] for d in first_call[1:]))
        self.assertEqual(json.loads("".join(d["function"]["arguments"] for d in first_call)), value)
        self.assertEqual(deltas[-1]["index"], 1)
        self.assertNotEqual(first_call[0]["id"], deltas[-1]["id"])
        self.assertEqual(tail[-2]["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(tail[-1], b"[DONE]")

    def test_unfinished_call_is_an_error(self):
        response = self.resources.enter_context(self.request(max_tokens=400).getresponse())
        conn, req = self.engine()
        prefix = [self.ink.T_TOOL, *b'{"name":"edit","args":{"patch":"']
        engine_record(conn, t=prefix + [ord("x")] * (req["max"] - len(prefix)))
        self.finish(conn, eos=False)
        events = frames(response)
        deltas = tool_deltas(events)
        self.assertTrue(deltas)
        with self.assertRaises(json.JSONDecodeError):
            json.loads("".join(d["function"]["arguments"] for d in deltas))
        self.assertEqual(events[-1]["error"]["code"], "length")
        self.assertEqual(events[-1]["error"]["type"], "invalid_request_error")
        self.assertNotIn(b"[DONE]", events)

    def test_trailing_root_field_cannot_commit_streamed_call(self):
        response = self.resources.enter_context(self.request().getresponse())
        conn, _ = self.engine()
        self.tool(conn, {"patch": "x" * 350}, extra=True)
        self.finish(conn)
        events = frames(response)
        self.assertTrue(tool_deltas(events))
        self.assertIn("error", events[-1])
        self.assertNotIn(b"[DONE]", events)

    def test_buffering_and_nonstream(self):
        self.ink.STREAM_TOOL_ARGUMENTS = False
        response = self.resources.enter_context(self.request().getresponse())
        next_frame(response)
        conn, _ = self.engine()
        value = {"patch": "x" * 350}
        self.tool(conn, value, end=False)
        self.assertEqual(next_frame(response), b": keepalive")
        engine_record(conn, t=[self.ink.T_END])
        self.finish(conn)
        deltas = tool_deltas(frames(response))
        self.assertEqual(len(deltas), 1)
        self.assertEqual(json.loads(deltas[0]["function"]["arguments"]), value)
        self.ink.STREAM_TOOL_ARGUMENTS = True
        client = self.request(stream=False)
        conn, _ = self.engine()
        self.tool(conn, value)
        self.finish(conn)
        response = self.resources.enter_context(client.getresponse())
        choice = json.load(response)["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]), value)

    def test_reordered_envelope_falls_back_to_whole_call(self):
        response = self.resources.enter_context(self.request().getresponse())
        next_frame(response)
        conn, _ = self.engine()
        value = {"patch": "x" * 350}
        raw = json.dumps({"args": value, "name": "edit"}).encode()
        engine_record(conn, t=[self.ink.T_TOOL, *raw])
        self.assertEqual(next_frame(response), b": keepalive")
        engine_record(conn, t=[self.ink.T_END])
        self.finish(conn)
        deltas = tool_deltas(frames(response))
        self.assertEqual(len(deltas), 1)
        self.assertEqual(json.loads(deltas[0]["function"]["arguments"]), value)

    def test_guard_cut_cannot_commit_streamed_call(self):
        response = self.resources.enter_context(self.request().getresponse())
        conn, _ = self.engine()
        block = (b"abcdefghijklmnopqrstuvwxyz0123456789" * 500)
        engine_record(conn, t=[self.ink.T_TOOL, *b'{"name":"edit","args":{"patch":"', *block])
        events = frames(response)
        self.assertTrue(tool_deltas(events))
        self.assertIn("error", events[-1])
        self.assertNotIn(b"[DONE]", events)
        self.assertEqual(conn.recv(1), b"")

    def test_repeated_call_from_history_stays_buffered_and_is_dropped(self):
        history = []
        for i in range(self.ink.LOOP_TURNS):
            history += [
                {"role": "assistant", "tool_calls": [{"function": {"name": "edit", "arguments": '{"patch":"same"}'}}]},
                {"role": "tool", "tool_call_id": f"call_{i}", "content": "same result"},
            ]
        self.ink.tokenize = lambda _: [1, 2]
        response = self.resources.enter_context(self.request(messages=history).getresponse())
        conn, _ = self.engine()
        self.tool(conn, {"patch": "same"})
        self.finish(conn)
        events = frames(response)
        self.assertEqual(tool_deltas(events), [])
        self.assertEqual(events[-2]["timings"]["loop_dropped"], 1)
        self.assertEqual(events[-1], b"[DONE]")

    def test_heartbeat_in_queue_and_prefill_then_cancel(self):
        active_client = self.request()
        active = self.resources.enter_context(active_client.getresponse())
        conn, _ = self.engine()
        next_frame(active)
        self.assertEqual(next_frame(active), b": keepalive")
        queued_finished = threading.Event()
        finish_request = self.http.finish_request

        def finish_queued_request(*args):
            try:
                finish_request(*args)
            finally:
                queued_finished.set()

        self.http.finish_request = finish_queued_request
        queued_client = self.request()
        queued = self.resources.enter_context(queued_client.getresponse())
        next_frame(queued)
        self.assertEqual(next_frame(queued), b": keepalive")
        queued.close()
        queued_client.close()
        self.assertTrue(queued_finished.wait(1), "disconnected request still queued")
        active.close()
        active_client.close()
        self.assertEqual(conn.recv(1), b"")
        self.listener.settimeout(0.15)
        with self.assertRaises(TimeoutError):
            self.listener.accept()

    def test_timeouts_ignore_keepalives(self):
        self.ink.QUEUE_SECONDS = 0.19
        self.ink.ENGINE_SECONDS = 0.5
        active = self.resources.enter_context(self.request().getresponse())
        conn, _ = self.engine()
        queued = self.resources.enter_context(self.request().getresponse())
        events = frames(queued)
        self.assertIn(b": keepalive", events)
        self.assertIn("inference slot", events[-1]["error"]["message"])
        events = frames(active)
        self.assertIn(b": keepalive", events)
        self.assertIn("execution deadline", events[-1]["error"]["message"])
        self.assertEqual(conn.recv(1), b"")

    def test_client_can_finish_writing_and_still_read_its_response(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                client = self.resources.enter_context(socket.create_connection(("127.0.0.1", self.http.server_port), timeout=3))
                body = json.dumps({"model": "inkling2-mini", "stream": stream}).encode()
                client.sendall(
                    (f"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                     f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode() + body
                )
                client.shutdown(socket.SHUT_WR)
                conn, _ = self.engine()
                engine_record(conn, t=[self.ink.T_TEXT, ord("x")])
                self.finish(conn)
                with http.client.HTTPResponse(client) as response:
                    response.begin()
                    self.assertEqual(response.status, 200)
                    if stream:
                        self.assertEqual(frames(response)[-1], b"[DONE]")
                    else:
                        self.assertEqual(json.load(response)["choices"][0]["message"]["content"], "x")

    def test_client_descriptor_above_select_limit(self):
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft_limit != resource.RLIM_INFINITY and soft_limit <= 1024:
            self.skipTest("the process cannot allocate a descriptor above 1023")
        accept = self.http.get_request

        def accept_high_descriptor():
            conn, address = accept()
            with conn:
                return socket.socket(fileno=fcntl.fcntl(conn, fcntl.F_DUPFD, 1024)), address

        self.http.get_request = accept_high_descriptor
        response = self.resources.enter_context(self.request().getresponse())
        conn, _ = self.engine()
        engine_record(conn, t=[self.ink.T_TEXT, ord("x")])
        self.finish(conn)
        self.assertEqual(frames(response)[-1], b"[DONE]")

    def test_validation_before_sse(self):
        with self.request(mtp="not a number").getresponse() as response:
            self.assertEqual(response.status, 400)
            self.assertEqual(response.getheader("Content-Type"), "application/json")
            response.read()


if __name__ == "__main__":
    unittest.main()
