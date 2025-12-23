from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray
    meta: dict[str, object]


def cluster_kmeans(x: np.ndarray, n_clusters: int, seed: int = 42) -> ClusterResult:
    model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed)
    labels = model.fit_predict(x)
    return ClusterResult(labels=labels.astype(np.int64), meta={"method": "kmeans", "n_clusters": n_clusters})


def cluster_agglo(x: np.ndarray, n_clusters: int, linkage: str = "ward") -> ClusterResult:
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(x)
    return ClusterResult(labels=labels.astype(np.int64), meta={"method": "agglo", "n_clusters": n_clusters, "linkage": linkage})


def cluster_dbscan(x: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> ClusterResult:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(x)
    return ClusterResult(labels=labels.astype(np.int64), meta={"method": "dbscan", "eps": eps, "min_samples": min_samples})
