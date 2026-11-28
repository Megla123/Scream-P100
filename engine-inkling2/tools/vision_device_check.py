"""Check projector selection and lifecycle with a helper that needs no model or GPU."""
import base64
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from stream_check import load_server


GPU = "GPU-abcdef12-3456-7890-abcd-ef0123456789"
HELPER = """#!/usr/bin/env python3
import os
from pathlib import Path
import sys

if "--gpu" in sys.argv and os.environ.get("FAIL_GPU"):
    sys.exit(1)
Path("launch").write_text(os.environ["CUDA_VISIBLE_DEVICES"] + ":" + str("--gpu" in sys.argv))
print("ready 2304", flush=True)
for line in sys.stdin:
    image, rows = map(Path, line.split())
    assert image.read_bytes() == b"picture"
    rows.write_bytes(bytes(2304 * 4))
    print("ok 1 2304 1", flush=True)
"""


class VisionDevice(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.helper = self.directory / "helper"
        self.helper.write_text(HELPER)
        self.helper.chmod(0o755)

    def load(self, *args, **settings):
        module = load_server(self.directory, extra_args=[
            "--vis", os.fspath(self.helper), "--mmproj", "fake", *args,
        ], settings={"INK2_IMG_DIR": str(self.directory), "INK2_VISION_DEVICE": "cpu",
                     "INK2_VISION_RESIDENCY": "auto", **settings})
        self.addCleanup(module.PRIVATE_IMAGES.cleanup)
        self.addCleanup(module.stop_vision)
        return module

    def encode(self, module):
        with TemporaryDirectory(dir=module.PRIVATE_IMAGES.name) as path:
            source = "data:image/png;base64," + base64.b64encode(b"picture").decode()
            return module.encode_image(source, Path(path), lambda: None)

    def launch_log(self, module):
        return (Path(module.PRIVATE_IMAGES.name) / "launch").read_text()

    def test_auto_cpu_starts_on_first_image(self):
        module = self.load("--vision-device", "cpu", INK2_VISION_DEVICE=GPU)
        module.preload_vision()
        self.assertIsNone(module.VIS["proc"])
        self.assertEqual(self.encode(module)[-1], 1)
        self.assertEqual(self.launch_log(module), ":False")

    def test_auto_gpu_preloads_and_reuses_same_process(self):
        module = self.load(INK2_VISION_DEVICE=GPU)
        module.preload_vision()
        before = module.VIS["proc"]
        self.assertEqual(self.launch_log(module), GPU + ":True")
        self.assertEqual(self.encode(module)[-1], 1)
        self.assertIs(module.VIS["proc"], before)

    def test_resident_cpu_preloads(self):
        module = self.load("--vision-residency", "resident")
        module.preload_vision()
        self.assertEqual(self.launch_log(module), ":False")
        self.assertIsNotNone(module.VIS["proc"])

    def test_on_demand_gpu_releases_projector_and_retains_embeddings(self):
        module = self.load("--vision-device", GPU, "--vision-residency", "on-demand")
        module.preload_vision()
        self.assertIsNone(module.VIS["proc"])
        with module.vision_batch():
            value = self.encode(module)
            self.assertEqual(self.launch_log(module), GPU + ":True")
        self.assertIsNone(module.VIS["proc"])
        with module.vision_batch():
            self.assertEqual(value, self.encode(module))
            self.assertIsNone(module.VIS["proc"])

    def test_invalid_gpu_selector_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.load("--vision-device", "0")


if __name__ == "__main__":
    unittest.main()
