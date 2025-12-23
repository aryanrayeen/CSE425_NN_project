from __future__ import annotations

from collections import Counter

import numpy as np


def cluster_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[0] == 0:
        return float("nan")

    total = 0
    for cluster_id in np.unique(y_pred):
        idx = np.where(y_pred == cluster_id)[0]
        if idx.size == 0:
            continue
        labels = y_true[idx].tolist()
        most_common = Counter(labels).most_common(1)[0][1]
        total += most_common
    return float(total) / float(y_true.shape[0])
