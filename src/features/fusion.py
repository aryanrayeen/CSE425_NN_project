from __future__ import annotations

import numpy as np


def zscore(x: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / (sd + eps), mu, sd


def concat_modalities(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Row mismatch: {a.shape} vs {b.shape}")
    return np.concatenate([a, b], axis=1)
