from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils.plot import savefig, set_matplotlib_backend


def embed_2d(x: np.ndarray, method: str = "tsne", seed: int = 42) -> np.ndarray:
    method = method.lower()
    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(n_components=2, random_state=seed)
            return reducer.fit_transform(x)
        except Exception:
            method = "tsne"

    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=seed)
    return tsne.fit_transform(x)


def plot_embedding(emb2d: np.ndarray, labels: np.ndarray, title: str, out_path: Path) -> None:
    set_matplotlib_backend()
    import matplotlib.pyplot as plt

    labels = np.asarray(labels)
    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(emb2d[:, 0], emb2d[:, 1], c=labels, s=10, cmap="tab10", alpha=0.85)
    plt.title(title)
    plt.xlabel("dim1")
    plt.ylabel("dim2")
    plt.colorbar(scatter, fraction=0.046, pad=0.04)
    savefig(out_path)
