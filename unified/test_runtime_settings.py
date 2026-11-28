import http.client
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Event, Thread
import unittest

from runtime_settings import read_state, RuntimeSettings, STARTUP_FIELDS, startup_layer, state_path


STAGED = Path(__file__).resolve().parents[1] / "build-unified/unified/strata-terminal"


class RuntimeSettingsTest(unittest.TestCase):
    def test_queued_settings_do_not_change_prior_requests(self):
        settings = RuntimeSettings({}, {"path": None, "state": {"next_start": {}, "shared": {"temperature": 0.2}}}, dict)
        with settings.request():
            settings.queue_shared({"temperature": 0.6})
            self.assertTrue(settings.shared_status()["pending"])
            self.assertEqual(settings.shared()["temperature"], 0.2)
            with settings.request():
                self.assertEqual(settings.shared()["temperature"], 0.6)
                settings.complete()
                self.assertFalse(settings.shared_status()["pending"])
            self.assertEqual(settings.shared()["temperature"], 0.2)
        self.assertEqual(settings.shared()["temperature"], 0.6)

    def test_state_only_replays_deferred_allowlisted_fields(self):
        self.assertEqual(startup_layer({"sampling.temperature": 0.4, "vram_reserve_mib": None, "lazy_load": True}),
                         {"sampling": {"temperature": 0.4}, "engine_options": {"--vram-reserve-mib": None},
                          "strata_settings": {"lazy_load": True}})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            path.write_text('{"version":1,"next_start":{"provider":"unknown"},"shared":{}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid deferred settings"):
                read_state(path)

    def test_state_root_is_absolute(self):
        self.assertEqual(state_path({"XDG_STATE_HOME": "relative"}, ["instance"]), state_path({}, ["instance"]))
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(state_path({"XDG_STATE_HOME": temp}, ["instance"]).parents[2], Path(temp))

    @unittest.skipUnless((STAGED / "serve/scream_runtime_settings.py").is_file(), "build the staged Strata adapter")
    def test_staged_api_queues_and_snapshots_runtime_settings(self):
        script = ("from pathlib import Path; import runpy, sys; sys.path.insert(0, str(Path(sys.argv[1]).parent)); "
                  "runpy.run_path(sys.argv[1])['http_fixture']()")
        completed = subprocess.run([sys.executable, "-c", script,
                                    str(Path(__file__).resolve())], cwd=STAGED, timeout=20,
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


def http_fixture():
    from serve import runconfig
    from serve.frontend import ChatTemplate
    from serve.server import ByteTokenizer, MockEngine, QueueWaitExpired, Service, serve, startup_tuning
    from serve.server_parts.http_api import clean_shared_defaults
    from unittest.mock import patch

    assert STARTUP_FIELDS == frozenset(runconfig.SPEC)
    admitted, release = Event(), Event()

    class Recorder(MockEngine):
        running = True

        def alive(self):
            return self.running

        def restart(self):
            self.running = True

        def generate(self, ids, max_new, sampling, cancel, embeddings=None):
            self.observed.append(sampling["temperature"])
            yield from super().generate(ids, max_new, sampling, cancel, embeddings)

    def reject_generation(*_args, **_kwargs):
        yield None
        raise QueueWaitExpired("native queue rejected the request")

    class TestService(Service):
        def with_shared(self, req, api):
            if req.get("messages") and req["messages"][0].get("content") == "paused":
                admitted.set()
                if not release.wait(7):
                    raise RuntimeError("test did not release first request")
            return super().with_shared(req, api)

    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "settings-state.json"
        config = Path(temp) / "run.json"
        startup = {"args": ["--pcie-frac", "0.6"], "env": {"STRATA_PIN_BY_NEED": "1"},
                   "sampling": {"temperature": 0.2}, "scream_runtime_settings": {
                       "tuning_flags": ["--pcie-frac", "--other"], "tuning_environment": ["STRATA_PIN_BY_NEED"]}}
        candidate = {**startup, "args": ["--pcie-frac", "0.1", "--other", "2", "--fast"],
                     "env": {"STRATA_PIN_BY_NEED": "0", "FEATURE": "on"},
                     "sampling": {"temperature": 0.2, "strata_tune": {"pcie_frac": 0.1, "prefill_pipe": 1}}}
        with patch("serve.server.apply_saved_tuning", return_value=candidate):
            tuned = startup_tuning(startup)
        assert tuned["env"] == {"STRATA_PIN_BY_NEED": "1", "FEATURE": "on"}
        assert tuned["args"] == ["--fast", "--pcie-frac", "0.6"]
        assert tuned["sampling"]["strata_tune"] == {"prefill_pipe": 1}
        config.write_text(json.dumps(startup), encoding="utf-8")
        tok = ByteTokenizer()
        engine = Recorder(tok, "ok", max_context=2048)
        engine.observed = []
        svc = TestService(engine, tok, ChatTemplate(Path("serve/chat_template.jinja")), sampling_defaults=startup["sampling"])
        svc.runtime_settings = RuntimeSettings(startup, {"path": str(path), "state": read_state(path)}, clean_shared_defaults)
        svc.config_path = str(config)
        server = serve(svc, port=0)

        def call(path, data=None, status=200):
            conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=12)
            try:
                conn.request("GET" if data is None else "POST", path, None if data is None else json.dumps(data),
                             {"Content-Type": "application/json"})
                response = conn.getresponse()
                parsed = json.load(response)
                assert response.status == status, (response.status, parsed)
                return parsed
            finally:
                conn.close()

        def chat(content, **overrides):
            return call("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": content}],
                                                  **overrides})

        first = []
        thread = Thread(target=lambda: first.append(chat("paused")), daemon=True)
        thread.start()
        try:
            assert admitted.wait(5)
            saved = call("/settings", {"defaults": {"temperature": 0.4}})
            assert saved["pending"] is True and call("/settings")["pending"] is True
            call("/v1/chat/completions", [], status=400)
            with patch.object(svc, "reserve_inference", return_value=None):
                call("/v1/chat/completions", {}, status=503)
            assert call("/settings")["pending"] is True
            with patch.object(engine, "generate", reject_generation):
                for endpoint, request in {
                    "chat/completions": {"messages": [{"role": "user", "content": "ok"}]},
                    "messages": {"messages": [{"role": "user", "content": "ok"}], "max_tokens": 16},
                    "responses": {"input": "ok"},
                }.items():
                    call(f"/v1/{endpoint}", {"model": "m", **request}, status=429)
                    assert call("/settings")["pending"] is True
            config.write_text('{"sampling":{"temperature":0.1}}', encoding="utf-8")
            configured = call("/config", {"set": {"sampling.temperature": 0.7}})
            assert configured["changed"] == ["sampling.temperature"]
            assert call("/config")["keys"] == configured["keys"]
            for key, value in (("sampling.temperature", 1e39), ("sampling.temperature", 10**400),
                               ("vram_reserve_mib", 2**31), ("reasoning_budget_tokens", float("inf")),
                               ("sampling.top_p", float("nan"))):
                bad = call("/config", {"set": {key: value, "idle_unload_s": 10}}, status=400)
                assert "representable" in bad["error"]["message"]
            chat("new")
            assert not call("/settings")["pending"]
            release.set()
            thread.join(8)
            assert first and engine.observed == [0.4, 0.2], (first, engine.observed)
            chat("explicit", temperature=0.9)
            assert engine.observed[-1] == 0.9
            assert read_state(path)["next_start"] == {"sampling.temperature": 0.7}
            assert read_state(path)["shared"] == {"temperature": 0.4}
            again = call("/config", {"set": {"sampling.temperature": 0.7}})
            assert not again["changed"]
            restarted = RuntimeSettings(startup, {"path": str(path), "state": read_state(path)}, clean_shared_defaults)
            _, changed = restarted.queue_startup({"sampling.temperature": 0.2}, runconfig.apply, runconfig.value_of)
            assert changed == ["sampling.temperature"]
            assert read_state(path)["next_start"] == {"sampling.temperature": 0.2}
            assert svc.runtime_settings.reserve() == (None, False)
            svc._record_vram_reserve(100)
            assert svc._engine_reserve() == 100
            engine.running = False
            with patch.object(engine, "vram", create=True) as resize:
                queued = call("/v1/vram", {"reserve_mib": 1000, "device": 1})
                assert queued["status"] == "not loaded"
                with svc.fifo:
                    svc.ensure_loaded()
                assert [call.args for call in resize.call_args_list] == [(100,), (1000,)]
                assert resize.call_args.kwargs == {"device": 1}
        finally:
            release.set()
            thread.join(8)
            server.shutdown()
            server.server_close()
