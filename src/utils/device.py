from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceConfig:
    device: str
    amp: bool


def pick_device(requested: str | None = None, amp: bool = True) -> DeviceConfig:
    try:
        import torch

        if requested is None or requested == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = requested

        use_amp = bool(amp and device.startswith("cuda") and torch.cuda.is_available())
        return DeviceConfig(device=device, amp=use_amp)
    except Exception:
        return DeviceConfig(device="cpu", amp=False)
