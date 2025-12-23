from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class BaselineResult:
    name: str
    embeddings: np.ndarray
    cluster_labels: np.ndarray


def pca_kmeans(x: np.ndarray, n_components: int = 32, n_clusters: int = 10, seed: int = 42) -> BaselineResult:
    pca = PCA(n_components=n_components, random_state=seed)
    z = pca.fit_transform(x)
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed)
    y = km.fit_predict(z)
    return BaselineResult(name="pca_kmeans", embeddings=z, cluster_labels=y.astype(np.int64))
