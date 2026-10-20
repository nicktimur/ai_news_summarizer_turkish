"""Platform-aware PyTorch support shared by inference and training."""
import os
import sys
from pathlib import Path

# Must be set before importing torch (also in all entry points).
if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

PROJECT_ROOT = Path(__file__).resolve().parent


def select_device():
    """Keep CUDA/CPU selection on Windows; use MPS/CPU on macOS."""
    override = os.environ.get("NEWS_DEVICE", "auto").lower()
    if override not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("NEWS_DEVICE: auto, cpu, cuda veya mps olmalıdır.")
    if override == "cpu":
        return torch.device("cpu")
    if sys.platform == "darwin":
        available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if override == "cuda":
            raise ValueError("macOS'ta CUDA yerine MPS veya CPU seçin.")
        if override == "mps" and not available:
            raise RuntimeError("MPS kullanılamıyor. Native arm64 Python ve PyTorch kurulumunu kontrol edin.")
        return torch.device("mps" if available else "cpu")
    if override == "mps":
        raise ValueError("MPS yalnızca macOS'ta kullanılabilir.")
    if override == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA bu PyTorch kurulumunda kullanılamıyor.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def project_path(relative):
    """Allow launching macOS scripts from VS Code or any working directory."""
    if sys.platform == "darwin":
        return str(PROJECT_ROOT / relative)
    return relative  # Preserve Windows path behavior.


def model_source(default_local, hub_fallback=None):
    override = os.environ.get("NEWS_MODEL")
    if override:
        expanded = Path(override).expanduser()
        if expanded.is_absolute():
            return str(expanded)
        local = PROJECT_ROOT / expanded
        return str(local) if local.is_dir() else override
    local = project_path(default_local)
    if sys.platform == "darwin" and not Path(local).is_dir() and hub_fallback:
        print(f"Yerel model bulunamadı; Hugging Face modeli kullanılacak: {hub_fallback}", flush=True)
        return hub_fallback
    return local


def mac_training_options():
    """Trainer auto-detects MPS. Preserve Windows batch/precision settings."""
    if sys.platform != "darwin":
        return {}
    device = select_device()
    return dict(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        fp16=False,
        bf16=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        use_cpu=device.type == "cpu",
    )


def _is_mps_failure(exc):
    if isinstance(exc, NotImplementedError):
        return True
    message = str(exc).lower()
    return any(word in message for word in ("mps", "metal", "placeholder storage"))


class ModelRunner:
    """Serialize inference and retry MPS-specific failures on CPU once."""

    def __init__(self, model, device=None):
        from threading import Lock
        self._lock = Lock()
        self.device = select_device() if device is None else device
        self.model = model.eval()
        try:
            self.model.to(self.device)
        except (RuntimeError, NotImplementedError) as exc:
            if self.device.type != "mps" or not _is_mps_failure(exc):
                raise
            self._use_cpu(exc)
        print(f"Özetleme işlem birimi: {self.device}", flush=True)

    def _use_cpu(self, exc):
        print(f"MPS işlemi tamamlanamadı; CPU ile devam ediliyor: {exc}", flush=True)
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def generate(self, inputs, **generation_kwargs):
        # Keep original tokenized tensors on CPU so fallback can reuse them.
        with self._lock, torch.inference_mode():
            try:
                device_inputs = {key: value.to(self.device) for key, value in inputs.items()}
                return self.model.generate(**device_inputs, **generation_kwargs).cpu()
            except (RuntimeError, NotImplementedError) as exc:
                if self.device.type != "mps" or not _is_mps_failure(exc):
                    raise
                device_inputs = None
                self._use_cpu(exc)
                cpu_inputs = {key: value.to(self.device) for key, value in inputs.items()}
                return self.model.generate(**cpu_inputs, **generation_kwargs).cpu()
