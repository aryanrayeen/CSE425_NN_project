from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .utils.device import DeviceConfig


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    beta: float = 1.0
    num_workers: int = 0


def _to_loader(x: np.ndarray, batch_size: int, shuffle: bool, num_workers: int, device: str):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    x_t = torch.from_numpy(x)
    ds = TensorDataset(x_t)
    pin = device.startswith("cuda")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin)


def train_mlp_vae(x: np.ndarray, model, cfg: TrainConfig, dev: DeviceConfig) -> dict[str, list[float]]:
    import torch

    try:
        from torch.amp import GradScaler, autocast
    except Exception:  # pragma: no cover
        from torch.cuda.amp import GradScaler, autocast

    from .models.mlp_vae import vae_loss_mse

    model = model.to(dev.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=dev.amp)

    loader = _to_loader(x, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, device=dev.device)

    history: dict[str, list[float]] = {"loss": [], "recon": [], "kl": []}

    model.train()
    for _epoch in range(cfg.epochs):
        losses = []
        recons = []
        kls = []
        for (xb,) in loader:
            xb = xb.to(dev.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=dev.amp):
                recon, mu, logvar, _z = model(xb)
                loss, parts = vae_loss_mse(recon, xb, mu, logvar, beta=cfg.beta)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            losses.append(parts["loss"])
            recons.append(parts["recon"])
            kls.append(parts["kl"])

        history["loss"].append(float(np.mean(losses)))
        history["recon"].append(float(np.mean(recons)))
        history["kl"].append(float(np.mean(kls)))

    return history


def train_mlp_ae(x: np.ndarray, model, cfg: TrainConfig, dev: DeviceConfig) -> dict[str, list[float]]:
    import torch

    try:
        from torch.amp import GradScaler, autocast
    except Exception:  # pragma: no cover
        from torch.cuda.amp import GradScaler, autocast

    from .models.autoencoder import ae_loss_mse

    model = model.to(dev.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=dev.amp)

    loader = _to_loader(x, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, device=dev.device)

    history: dict[str, list[float]] = {"loss": []}

    model.train()
    for _epoch in range(cfg.epochs):
        losses = []
        for (xb,) in loader:
            xb = xb.to(dev.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=dev.amp):
                recon, _z = model(xb)
                loss = ae_loss_mse(recon, xb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        history["loss"].append(float(np.mean(losses)))

    return history


def train_conv_vae(x: np.ndarray, model, cfg: TrainConfig, dev: DeviceConfig) -> dict[str, list[float]]:
    import torch

    try:
        from torch.amp import GradScaler, autocast
    except Exception:  # pragma: no cover
        from torch.cuda.amp import GradScaler, autocast

    from torch.utils.data import DataLoader, TensorDataset

    from .models.conv_vae import conv_vae_loss

    model = model.to(dev.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=dev.amp)

    x_t = torch.from_numpy(x)
    ds = TensorDataset(x_t)
    pin = dev.device.startswith("cuda")
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=pin)

    history: dict[str, list[float]] = {"loss": [], "recon": [], "kl": []}

    model.train()
    for _epoch in range(cfg.epochs):
        losses = []
        recons = []
        kls = []
        for (xb,) in loader:
            xb = xb.to(dev.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=dev.amp):
                recon, mu, logvar, _z = model(xb)
                loss, parts = conv_vae_loss(recon, xb, mu, logvar, beta=cfg.beta)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            losses.append(parts["loss"])
            recons.append(parts["recon"])
            kls.append(parts["kl"])

        history["loss"].append(float(np.mean(losses)))
        history["recon"].append(float(np.mean(recons)))
        history["kl"].append(float(np.mean(kls)))

    return history


def encode_latents_mlp_vae(x: np.ndarray, model, dev: DeviceConfig, batch_size: int = 512) -> np.ndarray:
    import torch

    model = model.to(dev.device)
    model.eval()

    out: list[np.ndarray] = []
    loader = _to_loader(x, batch_size=batch_size, shuffle=False, num_workers=0, device=dev.device)
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(dev.device, non_blocking=True)
            mu, _logvar = model.encode(xb)
            out.append(mu.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def encode_latents_mlp_ae(x: np.ndarray, model, dev: DeviceConfig, batch_size: int = 512) -> np.ndarray:
    import torch

    model = model.to(dev.device)
    model.eval()

    out: list[np.ndarray] = []
    loader = _to_loader(x, batch_size=batch_size, shuffle=False, num_workers=0, device=dev.device)
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(dev.device, non_blocking=True)
            z = model.encoder(xb)
            out.append(z.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def encode_latents_conv_vae(x: np.ndarray, model, dev: DeviceConfig, batch_size: int = 64) -> np.ndarray:
    import torch

    from torch.utils.data import DataLoader, TensorDataset

    model = model.to(dev.device)
    model.eval()

    x_t = torch.from_numpy(x)
    ds = TensorDataset(x_t)
    pin = dev.device.startswith("cuda")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    out: list[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(dev.device, non_blocking=True)
            mu, _logvar = model.encode(xb)
            out.append(mu.detach().cpu().numpy())
    return np.concatenate(out, axis=0)
