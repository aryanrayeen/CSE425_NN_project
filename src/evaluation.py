from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from .utils.metrics import cluster_purity


@dataclass(frozen=True)
class MetricReport:
    metrics: dict[str, float]


def _num_clusters(y_pred: np.ndarray) -> int:
    # DBSCAN uses -1 for noise; count it as a cluster for reporting only if present.
    return int(len(np.unique(y_pred)))


def evaluate_clustering(x: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray | None = None) -> MetricReport:
    y_pred = np.asarray(y_pred)
    metrics: dict[str, float] = {}

    k = _num_clusters(y_pred)
    metrics["num_clusters"] = float(k)

    # Unsupervised cluster quality metrics require >=2 clusters and < n samples.
    if k >= 2 and k < x.shape[0]:
        metrics["silhouette"] = float(silhouette_score(x, y_pred))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(x, y_pred))
        metrics["davies_bouldin"] = float(davies_bouldin_score(x, y_pred))
    else:
        metrics["silhouette"] = float("nan")
        metrics["calinski_harabasz"] = float("nan")
        metrics["davies_bouldin"] = float("nan")

    if y_true is not None:
        y_true = np.asarray(y_true)
        metrics["ari"] = float(adjusted_rand_score(y_true, y_pred))
        metrics["nmi"] = float(normalized_mutual_info_score(y_true, y_pred))
        metrics["purity"] = float(cluster_purity(y_true, y_pred))

    return MetricReport(metrics=metrics)
