"""Timed training and exact scoring (ported from the validation kit).

Only the ``train()`` call is timed. Scoring uses the same exact ``faiss.knn``
(squared L2) for every algorithm and is never timed.
"""
from __future__ import annotations

import time

import numpy as np


def score(faiss, x: np.ndarray, centers: np.ndarray, batch: int = 4096):
    """Exact nearest-final-center mean squared L2 error and empty cluster count."""
    sq_error = 0.0
    counts = np.zeros(len(centers), dtype=np.int64)
    for start in range(0, len(x), batch):
        distances, labels = faiss.knn(x[start:start + batch], centers, 1)
        sq_error += float(np.sum(distances, dtype=np.float64))
        counts += np.bincount(labels.reshape(-1), minlength=len(centers))
    return sq_error / len(x), int(np.count_nonzero(counts == 0))


def validate_centers(centers, k: int, dim: int) -> np.ndarray:
    centers = np.ascontiguousarray(centers, dtype=np.float32)
    if centers.shape != (k, dim) or not np.isfinite(centers).all():
        raise ValueError(
            f"Invalid centroids: shape={centers.shape}, finite={np.isfinite(centers).all()}")
    return centers


KMEANS_INIT_METHODS = {"random": "RANDOM", "kmeanspp": "KMEANS_PLUS_PLUS"}


def run_faiss_kmeans(faiss, train, k, seed, niter, init=None, init_method="random"):
    """Faiss Lloyd k-means on all rows (the 256 points/centroid cap is lifted).

    ``init_method`` selects Faiss' seeding ("random" = Faiss default,
    "kmeanspp" = k-means++); its cost is inside the timed train() call.
    Ignored by Faiss when explicit ``init`` centroids are given.
    """
    n, dim = train.shape
    if init is not None and init_method != "random":
        raise ValueError("explicit init centroids make init_method meaningless")
    model = faiss.Kmeans(
        dim, k, niter=niter, seed=seed, verbose=False,
        max_points_per_centroid=n, min_points_per_centroid=1,
        init_method=getattr(faiss, "ClusteringInitMethod_" + KMEANS_INIT_METHODS[init_method]),
    )
    started = time.perf_counter()
    model.train(train, init_centroids=init)
    elapsed = time.perf_counter() - started
    return validate_centers(model.centroids, k, dim), elapsed, {}


def run_emeans(NumpyEmeans, train, k, seed, nepochs, init=None):
    """Official NumpyEmeans with default hyperparameters (scaling laws)."""
    n, dim = train.shape
    model = NumpyEmeans(dim, k, seed=seed, nepochs=nepochs, verbose=False)
    started = time.perf_counter()
    centers, _ = model.train(train, init_centroids=init)
    elapsed = time.perf_counter() - started
    recipe = model.recipe or {}
    info = {
        "batch_size": recipe.get("batch_size"),
        "alpha0": recipe.get("alpha0"),
        "batch_law": recipe.get("batch_law"),
        "splits": int(model.n_splits),
    }
    return validate_centers(centers, k, dim), elapsed, info


def warmup(faiss, NumpyEmeans, train, k):
    """Untimed 1-epoch run of each algorithm (thread pools, page faults, imports)."""
    sub = np.ascontiguousarray(train[: min(len(train), 50 * k)])
    run_faiss_kmeans(faiss, sub, k, seed=0, niter=1)
    run_emeans(NumpyEmeans, sub, k, seed=0, nepochs=1)
