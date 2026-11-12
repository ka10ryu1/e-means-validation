"""Diagnostics for e-means cluster revival (never on a timed path).

``make_recorder`` subclasses the official ``NumpyEmeans`` and wraps its private
``_choose_splits`` (faiss-src/contrib/e_means/e_means.py) to record every
revival check: which centroids were judged unused (targets), which were split
(donors), their EMA masses, and where they sit relative to the true blobs of
the synthetic data. The wrapper calls the official method first and only reads
model state afterwards; it consumes no random numbers and changes nothing, so
training is bit-identical to the plain class (tested). ``_choose_splits`` is a
private method and may change with ``FAISS_COMMIT``.

``warmup_checks`` > 0 is **our modification, not the official algorithm**: the
first N revival checks are skipped without calling the official method (so no
split and no random draw happens there; later donor draws therefore differ from
the official run). It exists only to test whether revival harms because the EMA
masses still lag at the start. ``warmup_checks=0`` is the official behaviour.
"""
from __future__ import annotations

import sys

import numpy as np

EVENT_KEYS = ["check", "step", "epoch", "step_fraction", "target", "source",
              "target_mass_ratio", "source_mass_ratio",
              "target_blob", "target_blob_ratio", "target_blob_owners", "target_dist2_per_dim",
              "source_blob", "source_blob_ratio", "source_blob_owners"]
CHECK_KEYS = ["check", "step", "epoch", "step_fraction", "skipped", "n_below", "n_donors",
              "nsplit", "threshold_ratio"]


def nearest_blob(faiss, points, true_centers):
    """Index of and squared distance to the nearest true blob center."""
    dist, idx = faiss.knn(np.ascontiguousarray(points, dtype=np.float32), true_centers, 1)
    return idx.ravel(), dist.ravel()


def make_recorder(NumpyEmeans, faiss, true_centers, prob, warmup_checks=0):
    """A NumpyEmeans subclass that logs revival checks into ``.checks`` / ``.events``.

    Masses are reported as ratios to the mean EMA mass at check time (the
    official threshold is ``unused_threshold`` x that mean). ``*_blob`` is the
    true blob nearest to a centroid, ``*_blob_ratio`` that blob's true size
    relative to the mean blob size (``prob * n_blobs``), and ``*_blob_owners``
    the number of centroids whose nearest blob it is, before the split.
    With ``warmup_checks`` > 0 the first checks are skipped (see module doc);
    they are still logged, with ``skipped=True`` and ``nsplit=0``.
    """
    if warmup_checks < 0:
        raise ValueError("warmup_checks must be >= 0")
    module = sys.modules[NumpyEmeans.__module__]
    true_centers = np.ascontiguousarray(true_centers, dtype=np.float32)
    nblobs = len(true_centers)
    blob_ratio = np.asarray(prob, dtype=np.float64) * nblobs

    class RevivalRecorder(NumpyEmeans):
        def begin(self, global_n, init_centroids):
            self.checks, self.events = [], []
            return super().begin(global_n, init_centroids)

        def _choose_splits(self, rng):
            skipped = len(self.checks) < warmup_checks
            source, target = (None, None) if skipped else super()._choose_splits(rng)
            usage = self._usage
            threshold = float(module._numpy_unused_threshold(usage, self.k,
                                                             self._unused_threshold))
            mean = float(usage.sum(dtype=np.float64)) / self.k
            below = usage < threshold
            check = len(self.checks)
            fraction = (self._step + 1) / self.recipe["total_steps"]
            nsplit = 0 if source is None else len(source)
            self.checks.append(dict(
                check=check, step=self._step, epoch=self._epoch, step_fraction=fraction,
                skipped=skipped, n_below=int(below.sum()), n_donors=int(((~below) & (usage > 0)).sum()),
                nsplit=nsplit, threshold_ratio=threshold / mean if mean > 0 else 0.0))
            if nsplit:
                blob, dist2 = nearest_blob(faiss, self._centroids, true_centers)
                owners = np.bincount(blob, minlength=nblobs)
                for s, t in zip(source.tolist(), target.tolist()):
                    self.events.append(dict(
                        check=check, step=self._step, epoch=self._epoch, step_fraction=fraction,
                        target=t, source=s,
                        target_mass_ratio=float(usage[t]) / mean,
                        source_mass_ratio=float(usage[s]) / mean,
                        target_blob=int(blob[t]), target_blob_ratio=float(blob_ratio[blob[t]]),
                        target_blob_owners=int(owners[blob[t]]),
                        target_dist2_per_dim=float(dist2[t]) / self.d,
                        source_blob=int(blob[s]), source_blob_ratio=float(blob_ratio[blob[s]]),
                        source_blob_owners=int(owners[blob[s]])))
            return source, target

    return RevivalRecorder


def coverage(faiss, centers, true_centers, prob):
    """How the centroids are spread over the true blobs.

    ``uncovered_blobs``: blobs that are no centroid's nearest blob;
    ``uncovered_mass``: their share of the data (by true probability);
    ``crowded_blobs``: blobs with 2+ centroids; ``spare_centroids``: centroids
    beyond the first on each blob (k - covered blobs).
    """
    blob, _ = nearest_blob(faiss, centers, true_centers)
    owners = np.bincount(blob, minlength=len(true_centers))
    uncovered = owners == 0
    return dict(uncovered_blobs=int(uncovered.sum()),
                uncovered_mass=float(np.asarray(prob)[uncovered].sum()),
                crowded_blobs=int((owners >= 2).sum()),
                spare_centroids=int(len(centers) - (owners > 0).sum()))


def event_summary(events, checks, threshold_coef):
    """Per-run statistics of the recorded revival events."""
    n = len(events)
    out = dict(checks=len(checks), checks_with_split=sum(c["nsplit"] > 0 for c in checks),
               events=n)
    if not n:
        return out
    ev = {k: np.array([e[k] for e in events]) for k in EVENT_KEYS}
    targets, counts = np.unique(ev["target"], return_counts=True)
    out.update(
        events_epoch0=float(np.mean(ev["epoch"] == 0)),
        events_first10pct=float(np.mean(ev["step_fraction"] <= 0.1)),
        median_step_fraction=float(np.median(ev["step_fraction"])),
        target_mass_ratio_median=float(np.median(ev["target_mass_ratio"])),
        # what the revived centroid was sitting on
        target_on_shared_blob=float(np.mean(ev["target_blob_owners"] >= 2)),
        target_sole_on_blob=float(np.mean(ev["target_blob_owners"] == 1)),
        target_sole_on_small_blob=float(np.mean((ev["target_blob_owners"] == 1)
                                                & (ev["target_blob_ratio"] < threshold_coef))),
        target_blob_ratio_median=float(np.median(ev["target_blob_ratio"])),
        target_dist2_per_dim_median=float(np.median(ev["target_dist2_per_dim"])),
        # where the split copy goes
        source_blob_ratio_median=float(np.median(ev["source_blob_ratio"])),
        source_on_shared_blob=float(np.mean(ev["source_blob_owners"] >= 2)),
        distinct_targets=int(len(targets)),
        max_revivals_per_centroid=int(counts.max()),
        events_on_rerevived=float(np.mean(np.isin(ev["target"], targets[counts >= 2]))),
    )
    return out
