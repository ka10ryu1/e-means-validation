"""Model (not a measurement) of when a sole centroid's EMA mass dips below a threshold.

Used in docs/findings-revival-boundary.md. A blob of r training rows is
assumed to be owned by one centroid that receives all of its rows and no
other rows; the rows sit at random positions of each epoch's permutation.
The mass follows the official update N = (1 - a) N + a n_j from the initial
2**-16, with the official cosine schedule (a0 -> 0.1 a0), and is compared
with the mean mass (total / k, updated the same way from the batch sizes) at
every 5th step. Batch size B and a0 are read from reports/<run>/runs.csv, n
and k from metadata.json. No faiss, no timing; numpy only.

Prints, per run and number of epochs, P(mass ever below thr x mean at a check)
for a grid of blob sizes, and r50 (size at which that probability is 0.5,
interpolated on log size).

Example:
    uv run python model_revival_dip.py rev_bnd_n800k --threshold 0.01
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
INIT_USAGE = 2.0**-16   # base.py _INIT_USAGE
CHECK_EVERY = 5         # base.py _CHECK_UNUSED_EVERY
ALPHA_MIN_FACTOR = 0.1  # base.py _ALPHA_MIN_FACTOR
GRID = (3, 5, 8, 12, 15, 20, 25, 30, 40, 60, 90)


def p_dip(n, k, batch, alpha0, epochs, rows, threshold, sims, rng):
    """Fraction of simulated sole centroids whose mass is ever below threshold."""
    sizes = [min(batch, n - s) for s in range(0, n, batch)]
    per_epoch = len(sizes)
    total = per_epoch * epochs
    edges = np.cumsum([0] + sizes)
    counts = np.empty((sims, total))
    for e in range(epochs):
        pos = (rng.random((sims, rows)) * n).astype(np.int64)
        for s in range(sims):
            counts[s, e * per_epoch:(e + 1) * per_epoch] = np.histogram(pos[s], edges)[0]
    mass = np.full(sims, INIT_USAGE)
    mean_total = INIT_USAGE * k
    ever = np.zeros(sims, bool)
    for step in range(total):
        cosine = 0.5 * (1.0 + math.cos(math.pi * step / max(1, total)))
        a = alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cosine)
        mass = (1 - a) * mass + a * counts[:, step]
        mean_total = (1 - a) * mean_total + a * sizes[step % per_epoch]
        if (step + 1) % CHECK_EVERY == 0:
            ever |= mass < threshold * mean_total / k
    return float(ever.mean())


def r50(grid, probs):
    """Size where the probability crosses 0.5, interpolated on log size."""
    for (r1, p1), (r2, p2) in zip(zip(grid, probs), zip(grid[1:], probs[1:])):
        if p1 >= 0.5 > p2:
            return math.exp(math.log(r1) + (math.log(r2) - math.log(r1)) * (p1 - 0.5) / (p1 - p2))
    return None


def recipes(run):
    """{passes: (batch_size, alpha0)} and settings for a diagnose_revival run."""
    report = ROOT / "reports" / run
    meta = json.loads((report / "metadata.json").read_text(encoding="utf-8"))
    out = {}
    with open(report / "runs.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["algorithm"] == "e_means_pp":
                out.setdefault(int(float(row["passes"])),
                               (int(float(row["batch_size"])), float(row["alpha0"])))
    return meta["settings"], out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="reports/<run> names from diagnose_revival.py")
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--sims", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    print(f"model (not measured): P(ever below {args.threshold:g} x mean) by blob rows, and r50")
    for run in args.runs:
        settings, rec = recipes(run)
        for epochs, (batch, alpha0) in sorted(rec.items()):
            probs = [p_dip(settings["ntrain"], settings["k"], batch, alpha0, epochs, r,
                           args.threshold, args.sims, rng) for r in GRID]
            r = r50(GRID, probs)
            print(f"{run:20} T{epochs:<2} B={batch:<6} a0={alpha0:.3f} "
                  f"r50={'' if r is None else f'{r:.1f}':>5}  "
                  + " ".join(f"{g}:{q:.2f}" for g, q in zip(GRID, probs)), flush=True)


if __name__ == "__main__":
    main()
