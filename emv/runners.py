"""Timed training and exact scoring (ported from the validation kit).

Only the ``train()`` call is timed. Scoring uses the same exact ``faiss.knn``
(squared L2) for every algorithm and is never timed.
"""
from __future__ import annotations

import time

import numpy as np


class Timer:
    """Wall time plus CPU time of the whole process and of the calling thread.

    Diagnostics for timing anomalies: if wall time grows but CPU time does
    not, the process was waiting (contention); if CPU time grows with it, the
    work itself ran slower (core type / clock). Overhead: six clock reads.
    """

    def __enter__(self):
        self._wall = time.perf_counter()
        self._cpu = time.process_time()
        self._thread = time.thread_time()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._wall
        self.cpu_seconds = time.process_time() - self._cpu
        self.main_thread_cpu_seconds = time.thread_time() - self._thread
        return False

    def info(self) -> dict:
        return {"cpu_seconds": round(self.cpu_seconds, 6),
                "main_thread_cpu_seconds": round(self.main_thread_cpu_seconds, 6)}


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
    with Timer() as t:
        model.train(train, init_centroids=init)
    return validate_centers(model.centroids, k, dim), t.seconds, t.info()


def kmeanspp_init(faiss, x, k, seed):
    """k-means++ centroids exactly as faiss.Kmeans(init_method=KMEANS_PLUS_PLUS,
    seed=seed) draws them (Faiss seeds its initializer with seed + 1)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    ci = faiss.ClusteringInitialization(x.shape[1], k)
    ci.method = faiss.ClusteringInitMethod_KMEANS_PLUS_PLUS
    ci.seed = seed + 1
    out = np.empty((k, x.shape[1]), dtype=np.float32)
    ci.init_centroids(len(x), faiss.swig_ptr(x), faiss.swig_ptr(out))
    return out


def run_emeans(NumpyEmeans, train, k, seed, nepochs, init=None, init_method="random",
               faiss=None):
    """Official NumpyEmeans with default hyperparameters (scaling laws).

    ``init_method="kmeanspp"`` starts from the same k-means++ centroids as the
    Faiss k-means++ baseline; computing them is inside the timed region, as it
    is for Faiss. "random" keeps NumpyEmeans' own default seeding.
    """
    n, dim = train.shape
    if init_method not in KMEANS_INIT_METHODS:
        raise ValueError(init_method)
    if init_method == "kmeanspp" and (init is not None or faiss is None):
        raise ValueError("kmeanspp init needs faiss and no explicit init centroids")
    model = NumpyEmeans(dim, k, seed=seed, nepochs=nepochs, verbose=False)
    with Timer() as t:
        if init_method == "kmeanspp":
            init = kmeanspp_init(faiss, train, k, seed)
        centers, _ = model.train(train, init_centroids=init)
    elapsed = t.seconds
    recipe = model.recipe or {}
    info = {
        **t.info(),
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
