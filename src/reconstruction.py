from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils.plot import savefig, set_matplotlib_backend


def save_conv_recon_examples(x: np.ndarray, x_hat: np.ndarray, out_path: Path, n: int = 6) -> None:
    """Save a grid of original vs reconstructed mel-spectrograms.

    x/x_hat shape: (N, 1, n_mels, n_frames)
    """
    set_matplotlib_backend()
    import matplotlib.pyplot as plt

    n = min(n, x.shape[0])
    plt.figure(figsize=(10, 2.5 * n))

    for i in range(n):
        orig = x[i, 0]
        rec = x_hat[i, 0]

        ax1 = plt.subplot(n, 2, 2 * i + 1)
        ax1.imshow(orig, aspect="auto", origin="lower")
        ax1.set_title(f"orig {i}")
        ax1.axis("off")

        ax2 = plt.subplot(n, 2, 2 * i + 2)
        ax2.imshow(rec, aspect="auto", origin="lower")
        ax2.set_title(f"recon {i}")
        ax2.axis("off")

    savefig(out_path)
