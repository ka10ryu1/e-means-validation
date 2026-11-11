"""Datasets. Generation logic is unchanged from the validation kit."""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_SEED = 20261111

PROFILES = {
    # smoke: tiny, for tests / CI only (not a performance measurement)
    "smoke": dict(ntrain=3_000, ntest=1_000, dim=16, k=16),
    "quick": dict(ntrain=20_000, ntest=4_000, dim=64, k=128),
    "medium": dict(ntrain=200_000, ntest=10_000, dim=64, k=1_024),
}
DATASETS = ("separated", "gaussian", "imbalanced")


IMBALANCE_EXPONENT = 1.3  # kit default


def synthetic_data(name: str, ntrain: int, ntest: int, dim: int, k: int,
                   imbalance_exponent: float = IMBALANCE_EXPONENT):
    """Independent train/test draws from a fixed generating distribution.

    separated:  k Gaussian blobs (centers ~ N(0, 4^2), unit noise), equal sizes
    gaussian:   a single standard normal (no cluster structure)
    imbalanced: same blobs, cluster sizes follow a power law (rank^-imbalance_exponent;
                the kit uses 1.3, and 0 gives the same sizes as separated)
    """
    rng = np.random.default_rng(DATA_SEED)
    count = ntrain + ntest
    if name == "gaussian":
        data = rng.standard_normal((count, dim)).astype(np.float32)
    else:
        centers = rng.normal(0, 4.0, (k, dim)).astype(np.float32)
        if name == "separated":
            prob = np.full(k, 1 / k)
        elif name == "imbalanced":
            weights = (np.arange(1, k + 1, dtype=np.float64)) ** -imbalance_exponent
            prob = weights / weights.sum()
        else:
            raise ValueError(name)
        group = rng.choice(k, size=count, p=prob)
        data = centers[group] + rng.standard_normal((count, dim)).astype(np.float32)
    return np.ascontiguousarray(data[:ntrain]), np.ascontiguousarray(data[ntrain:])


def real_embeddings(path: Path, ntrain: int, ntest: int):
    """Random disjoint train/test rows from a 2D .npy matrix."""
    x = np.load(path, allow_pickle=False, mmap_mode="r")
    if x.ndim != 2 or x.shape[0] < ntrain + ntest:
        raise ValueError(f"Need >= {ntrain + ntest} 2D embeddings, got {x.shape}")
    rng = np.random.default_rng(DATA_SEED)
    idx = rng.choice(x.shape[0], size=ntrain + ntest, replace=False)
    train = np.ascontiguousarray(x[idx[:ntrain]], dtype=np.float32)
    test = np.ascontiguousarray(x[idx[ntrain:]], dtype=np.float32)
    if not (np.isfinite(train).all() and np.isfinite(test).all()):
        raise ValueError("Input embeddings contain NaN/Inf")
    return train, test


def shared_init(train: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Initial centroids drawn exactly as NumpyEmeans.train does by default
    (k rows of a RandomState(seed) permutation)."""
    idx = np.random.RandomState(seed).permutation(len(train))[:k]
    return np.ascontiguousarray(train[idx], dtype=np.float32)
