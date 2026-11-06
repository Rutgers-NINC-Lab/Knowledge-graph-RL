"""Configuration and device selection for the distributed example workflows."""
import json
import os
from pathlib import Path


def load_config(root):
    path = Path(os.environ.get("KGRL_CONFIG", Path(root) / "configs/example.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def configure(config):
    requested = os.environ.get("KGRL_DEVICE", config["runtime"]["device"])
    # A CPU validation run must not initialize a CUDA context on a busy GPU.
    if requested == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from kgrl.core.device import set_device
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install a CUDA-enabled PyTorch build, or set KGRL_DEVICE=cpu for a local check.")
    torch.set_num_threads(config["runtime"].get("cpu_threads", 1))
    return set_device(requested)


def environment():
    import importlib.metadata
    import platform
    import torch
    from kgrl.core.device import get_device
    device = get_device()
    return {"python": platform.python_version(), "platform": platform.platform(),
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_cuda": torch.version.cuda, "threads": torch.get_num_threads(),
            "packages": {name: importlib.metadata.version(name) for name in ["torch", "numpy", "pandas", "matplotlib"]}}


def synchronize():
    import torch
    from kgrl.core.device import get_device
    if get_device().type == "cuda":
        torch.cuda.synchronize(get_device())
