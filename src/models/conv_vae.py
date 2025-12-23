from __future__ import annotations

import math

import torch
from torch import nn


class ConvVAE(nn.Module):
    """A small Conv-VAE for 2D spectrogram-like inputs.

    Input shape: (B, 1, n_mels, n_frames)
    """

    def __init__(self, n_mels: int = 128, n_frames: int = 256, latent_dim: int = 32) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.latent_dim = latent_dim

        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, n_frames)
            h = self.enc(dummy)
            self._enc_shape = h.shape[1:]
            flat = int(h.numel())

        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat)

        c, h_m, h_t = self._enc_shape
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z)
        h = h.view(z.shape[0], *self._enc_shape)
        x_hat = self.dec(h)
        # Output may be slightly different size due to stride; center-crop/pad to target.
        x_hat = _resize_to(x_hat, self.n_mels, self.n_frames)
        return x_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


def _resize_to(x: torch.Tensor, n_mels: int, n_frames: int) -> torch.Tensor:
    _, _, h, w = x.shape

    if h < n_mels:
        x = torch.nn.functional.pad(x, (0, 0, 0, n_mels - h))
    if w < n_frames:
        x = torch.nn.functional.pad(x, (0, n_frames - w, 0, 0))

    x = x[:, :, :n_mels, :n_frames]
    return x


def conv_vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
    recon_loss = torch.nn.functional.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + beta * kl
    return loss, {"loss": float(loss.detach().cpu()), "recon": float(recon_loss.detach().cpu()), "kl": float(kl.detach().cpu())}
