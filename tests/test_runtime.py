"""No weights, GUI, GPU, or third-party packages required."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from contextlib import nullcontext
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class Device:
    def __init__(self, name): self.type = name
    def __str__(self): return self.type


class Tensor:
    def __init__(self, device="cpu"): self.device = device
    def to(self, device): return Tensor(device.type)
    def cpu(self): return Tensor("cpu")


class Model:
    def __init__(self, error=None, move_error=False):
        self.error = error
        self.move_error = move_error
        self.calls = []
    def eval(self): return self
    def to(self, device):
        if self.move_error and device.type == "mps":
            raise RuntimeError("MPS device initialization failed")
        self.device = device.type
        return self
    def generate(self, **kwargs):
        self.calls.append((self.device, kwargs))
        if self.device == "mps" and self.error:
            raise self.error
        return Tensor(self.device)


def load_runtime(platform="darwin", cuda=False, mps=True, env=None):
    fake = types.ModuleType("torch")
    fake.device = Device
    fake.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    fake.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    fake.inference_mode = nullcontext
    spec = importlib.util.spec_from_file_location("runtime_under_test", ROOT / "runtime.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch": fake}), patch.object(sys, "platform", platform), patch.dict(os.environ, env or {}, clear=True):
        spec.loader.exec_module(module)
    return module


class DeviceTests(unittest.TestCase):
    def check_device(self, platform, cuda, mps, expected):
        r = load_runtime(platform, cuda, mps)
        with patch.object(sys, "platform", platform), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(r.select_device().type, expected)
    def test_mac_mps(self): self.check_device("darwin", False, True, "mps")
    def test_mac_cpu(self): self.check_device("darwin", False, False, "cpu")
    def test_windows_cuda(self): self.check_device("win32", True, True, "cuda")
    def test_windows_never_mps(self): self.check_device("win32", False, True, "cpu")
    def test_linux_cuda(self): self.check_device("linux", True, False, "cuda")
    def test_force_cpu(self):
        r = load_runtime()
        with patch.dict(os.environ, {"NEWS_DEVICE": "cpu"}, clear=True):
            self.assertEqual(r.select_device().type, "cpu")
    def test_invalid_override(self):
        r = load_runtime()
        with patch.dict(os.environ, {"NEWS_DEVICE": "typo"}, clear=True):
            with self.assertRaises(ValueError): r.select_device()
    def test_mac_rejects_cuda(self):
        r = load_runtime()
        with patch.object(sys, "platform", "darwin"), patch.dict(os.environ, {"NEWS_DEVICE": "cuda"}, clear=True):
            with self.assertRaises(ValueError): r.select_device()
    def test_training_windows_unchanged(self):
        r = load_runtime()
        with patch.object(sys, "platform", "win32"):
            self.assertEqual(r.mac_training_options(), {})
    def test_training_mac_precision_and_batch(self):
        r = load_runtime()
        with patch.object(sys, "platform", "darwin"), patch.dict(os.environ, {}, clear=True):
            options = r.mac_training_options()
        self.assertEqual(options["per_device_train_batch_size"] * options["gradient_accumulation_steps"], 8)
        self.assertFalse(options["fp16"])
        self.assertFalse(options["bf16"])
        self.assertFalse(options["dataloader_pin_memory"])
        self.assertFalse(options["use_cpu"])
    def test_mac_training_cpu_override(self):
        r = load_runtime()
        with patch.object(sys, "platform", "darwin"), patch.dict(os.environ, {"NEWS_DEVICE":"cpu"}, clear=True):
            self.assertTrue(r.mac_training_options()["use_cpu"])
    def test_project_paths(self):
        r = load_runtime()
        with patch.object(sys, "platform", "darwin"):
            self.assertEqual(r.project_path("data/file"), str(ROOT / "data/file"))
        with patch.object(sys, "platform", "win32"):
            self.assertEqual(r.project_path("./model"), "./model")
    def test_model_fallback_only_mac(self):
        r = load_runtime()
        with patch.object(sys, "platform", "darwin"), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(r.model_source("not-present", "owner/model"), "owner/model")
        with patch.object(sys, "platform", "win32"), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(r.model_source("./not-present", "owner/model"), "./not-present")
    def test_model_override(self):
        r = load_runtime()
        with patch.dict(os.environ, {"NEWS_MODEL": "owner/other"}, clear=True):
            self.assertEqual(r.model_source("local"), "owner/other")
    def test_existing_local_model(self):
        r=load_runtime()
        with patch.object(sys, "platform", "darwin"), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(r.model_source("data", "owner/model"), str(ROOT / "data"))


class RunnerTests(unittest.TestCase):
    def test_mps_generation_and_parameters(self):
        r = load_runtime(); model = Model()
        runner = r.ModelRunner(model, Device("mps"))
        source = Tensor()
        output = runner.generate({"input_ids": source}, num_beams=6)
        self.assertEqual(output.device, "cpu")
        self.assertEqual(source.device, "cpu")
        self.assertEqual(model.calls[0][1]["num_beams"], 6)
        self.assertEqual(model.calls[0][1]["input_ids"].device, "mps")
    def test_mps_retry_on_cpu(self):
        r = load_runtime(); model = Model(RuntimeError("MPS backend operation unavailable"))
        runner = r.ModelRunner(model, Device("mps"))
        runner.generate({"input_ids": Tensor()}, num_beams=6)
        self.assertEqual([device for device, _ in model.calls], ["mps", "cpu"])
        runner.generate({"input_ids": Tensor()})
        self.assertEqual([device for device, _ in model.calls], ["mps", "cpu", "cpu"])
    def test_mps_initialization_retry(self):
        r=load_runtime(); runner=r.ModelRunner(Model(move_error=True), Device("mps"))
        self.assertEqual(runner.device.type,"cpu")
    def test_unrelated_error_not_hidden(self):
        r=load_runtime(); runner=r.ModelRunner(Model(RuntimeError("Invalid input shape")),Device("mps"))
        with self.assertRaisesRegex(RuntimeError,"Invalid input shape"):
            runner.generate({"input_ids":Tensor()})
    def test_cuda_errors_not_retried(self):
        r=load_runtime(); model=Model(); runner=r.ModelRunner(model,Device("cuda"))
        def fail(**kwargs): raise RuntimeError("CUDA out of memory")
        model.generate=fail
        with self.assertRaisesRegex(RuntimeError,"CUDA"):
            runner.generate({"input_ids":Tensor()})


class EntryPointTests(unittest.TestCase):
    def test_runtime_imported_before_torch(self):
        for name in ("app.py", "model_trainer.py", "model_trainer_for_other_dataset.py", "test.py"):
            tree=ast.parse((ROOT/name).read_text())
            self.assertEqual(tree.body[0].module,"runtime",name)
    def test_gradio_not_launched_on_import(self):
        tree=ast.parse((ROOT/'app.py').read_text())
        launch=[n for n in tree.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
                and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__']
        self.assertEqual(len(launch),1)


if __name__ == "__main__":
    unittest.main()
