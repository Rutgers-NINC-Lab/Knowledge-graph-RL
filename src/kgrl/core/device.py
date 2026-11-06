"""Shared torch device state."""
import torch

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device() -> torch.device:
    return _device


def set_device(device_or_name) -> torch.device:
    global _device
    _device = torch.device(device_or_name)
    return _device
