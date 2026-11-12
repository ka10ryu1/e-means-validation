"""Record when e-means cluster revival fires and what it moves (quality only).

Trains the official NumpyEmeans from the same k-means++ centroids as the
Faiss k-means++ baseline (e-means++), through a subclass that only logs each
revival check (emv/revival.py; training is bit-identical to the plain class).
Synthetic data only, because the true blobs are needed. No timing is
reported: this is a diagnostic, not a benchmark.

``--warmup-checks N ...`` adds runs that skip the first N revival checks (our
modification, not the official algorithm; see emv/revival.py). 0 is the
official behaviour and is always the only value used with threshold 0.

Example:
    uv run python diagnose_revival.py --profile medium --dataset imbalanced \\
        --imbalance-exponent 1.0 --thresholds 0.1 0 --run-name rev_imb10
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

RUN_KEYS = ["dataset", "imbalance_exponent", "dim", "k", "algorithm", "unused_threshold",
            "warmup_checks", "batch_law", "batch_size", "passes", "seed", "test_mse", "train_mse",
            "empty_clusters", "splits", "checks", "checks_with_split", "events",
            "events_epoch0", "events_first10pct", "median_step_fraction",
            "target_mass_ratio_median", "target_on_shared_blob", "target_sole_on_blob",
            "target_sole_on_small_blob", "target_blob_ratio_median",
            "target_dist2_per_dim_median", "source_blob_ratio_median", "source_on_shared_blob",
            "distinct_targets", "max_revivals_per_centroid", "events_on_rerevived",
            "uncovered_blobs", "uncovered_mass", "crowded_blobs", "spare_centroids"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--faiss-source", type=Path, default=ROOT / "faiss-src")
    p.add_argument("--profile", choices=("smoke", "quick", "medium"), default="medium")
    p.add_argument("--dataset", choices=("separated", "imbalanced"), default="imbalanced")
    p.add_argument("--imbalance-exponent", type=float, default=1.3)
    p.add_argument("--dim", type=int, default=None, help="Override dimension")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", nargs="+", type=float, default=[1, 3, 10])
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.0],
                   help="unused_threshold values (0 disables revival)")
    p.add_argument("--warmup-checks", nargs="+", type=int, default=[0],
                   help="Skip the first N revival checks (our modification; 0 = official)")
    p.add_argument("--batch-law", choices=("auto", "throughput", "quality"), default="auto")
    p.add_argument("--kmeans-niter", type=int, default=25,
                   help="Faiss k-means++ reference iterations (0 to skip)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--run-name", required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    for var in THREAD_ENV:  # before numpy / faiss load their thread pools
        os.environ[var] = str(args.threads)

    import json
    import subprocess
    from datetime import datetime, timezone

    import numpy as np

    from emv import analysis, data, official, revival, runners

    import faiss
    NumpyEmeans = official.load_official_emeans(faiss, args.faiss_source)
    source = official.source_info(faiss, args.faiss_source)
    official.check_pinned(source)
    faiss.omp_set_num_threads(args.threads)

    settings = dict(data.PROFILES[args.profile])
    if args.dim is not None:
        settings["dim"] = args.dim
    k, dim = settings["k"], settings["dim"]
    train, test, truth = data.synthetic_truth(args.dataset, **settings,
                                              imbalance_exponent=args.imbalance_exponent)
    centers_true, prob = truth["centers"], truth["prob"]
    if any(w < 0 for w in args.warmup_checks):
        raise SystemExit("--warmup-checks must be >= 0")
    recorders = {w: revival.make_recorder(NumpyEmeans, faiss, centers_true, prob, warmup_checks=w)
                 for w in {0, *args.warmup_checks}}

    out = ROOT / "results" / args.run_name
    report_dir = ROOT / "reports" / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    def git(*a):
        try:
            return subprocess.check_output(["git", "-C", str(ROOT), *a], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    base = dict(dataset=args.dataset,
                imbalance_exponent=args.imbalance_exponent if args.dataset == "imbalanced" else "",
                dim=dim, k=k)
    rows, events, checks = [], [], []

    def finish(row, centers):
        train_mse, empty = runners.score(faiss, train, centers)
        test_mse, _ = runners.score(faiss, test, centers)
        row.update(train_mse=round(train_mse, 8), test_mse=round(test_mse, 8),
                   empty_clusters=empty,
                   **revival.coverage(faiss, centers, centers_true, prob))
        rows.append(row)
        print(f"  {row['algorithm']:12} thr={row['unused_threshold']!s:>4} "
              f"warmup={row.get('warmup_checks', '')!s:>2} "
              f"passes={row['passes']!s:>3} seed={row['seed']}: test MSE={test_mse:.5f} "
              f"splits={row.get('splits', '')} uncovered={row['uncovered_blobs']}", flush=True)

    for seed in args.seeds:
        init = runners.kmeanspp_init(faiss, train, k, seed)
        finish(dict(base, algorithm="kmeanspp_init", unused_threshold="", passes=0, seed=seed),
               runners.validate_centers(init, k, dim))
        if args.kmeans_niter:
            centers, _, _ = runners.run_faiss_kmeans(faiss, train, k, seed, args.kmeans_niter,
                                                     init_method="kmeanspp")
            finish(dict(base, algorithm="faiss_kmeans_pp", unused_threshold="",
                        passes=args.kmeans_niter, seed=seed), centers)
        for thr, warmup in [(t, w) for t in args.thresholds
                            for w in (args.warmup_checks if t > 0 else [0])]:
            Recorder = recorders[warmup]
            for epochs in args.epochs:
                passes = analysis.normalize_passes(epochs)
                model = Recorder(dim, k, seed=seed, nepochs=passes, verbose=False,
                                 batch_law=args.batch_law, unused_threshold=thr)
                centers, _ = model.train(train, init_centroids=init)
                tag = dict(unused_threshold=thr, warmup_checks=warmup, passes=passes, seed=seed)
                events += [dict(tag, **e) for e in model.events]
                checks += [dict(tag, **c) for c in model.checks]
                summary = revival.event_summary(model.events, model.checks,
                                                model._unused_threshold)
                assert summary["events"] == model.n_splits, "recorder missed a split"
                finish(dict(base, algorithm="e_means_pp", batch_law=model.recipe["batch_law"],
                            batch_size=model.recipe["batch_size"], splits=model.n_splits,
                            **tag, **summary), runners.validate_centers(centers, k, dim))
        analysis.write_csv(out / "runs.csv", rows, RUN_KEYS)

    tag_keys = ["unused_threshold", "warmup_checks", "passes", "seed"]
    if events:
        analysis.write_csv(out / "events.csv", events, tag_keys + revival.EVENT_KEYS)
    if checks:
        analysis.write_csv(out / "checks.csv", checks, tag_keys + revival.CHECK_KEYS)
    analysis.write_csv(report_dir / "runs.csv", rows, RUN_KEYS)
    metadata = dict(
        run_name=args.run_name, command=["python", *sys.argv],
        started_note="quality-only diagnostic; no timing recorded",
        profile=args.profile, settings=settings, dataset=args.dataset,
        imbalance_exponent=args.imbalance_exponent, seeds=args.seeds, epochs=args.epochs,
        thresholds=args.thresholds, warmup_checks=args.warmup_checks, batch_law=args.batch_law, kmeans_niter=args.kmeans_niter,
        threads=args.threads, data_seed=data.DATA_SEED,
        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        faiss_version=getattr(faiss, "__version__", "unknown"), numpy_version=np.__version__,
        repo_commit=git("rev-parse", "HEAD"),
        repo_dirty=bool(git("status", "--porcelain", "--", ".", ":!results", ":!reports")),
        e_means_source=source)
    (report_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    (report_dir / "report.md").write_text(render(rows, metadata), encoding="utf-8")
    print(f"\nRaw events/checks: {out}\nReport: {report_dir / 'report.md'}")


def _median(values):
    import numpy as np
    values = [v for v in values if v != "" and v is not None]
    return float(np.median(values)) if values else None


def render(rows, metadata):
    """Medians over seeds for every (algorithm, threshold, passes)."""
    groups = {}
    for r in rows:
        groups.setdefault((r["algorithm"], str(r["unused_threshold"]),
                           str(r.get("warmup_checks", "")), r["passes"]), []).append(r)

    def f(v, fmt):
        return "" if v is None else format(v, fmt)

    s = metadata["settings"]
    lines = [f"# 復活の診断: {metadata['run_name']}", "",
             f"- データ: {metadata['dataset']}"
             + (f"（指数 {metadata['imbalance_exponent']:g}）"
                if metadata["dataset"] == "imbalanced" else "")
             + f"、n={s['ntrain']:,}、d={s['dim']}、k={s['k']:,}、シード {metadata['seeds']}",
             f"- e-means++（k-means++ 初期重心）、batch_law={metadata['batch_law']}、"
             f"{metadata['threads']} スレッド。品質のみ（時間は記録しない）",
             f"- コード {metadata['repo_commit']}（repo_dirty: {metadata['repo_dirty']}）、"
             f"公式 {metadata['e_means_source'].get('faiss_source_sha')}",
             "- 値はシードの中央値。割合は 1 回の学習内の復活イベントに対する割合。",
             "- 「判定の停止」N > 0 は**本リポジトリの改変**（最初の N 回の復活判定を行わない）で、"
             "公式の挙動ではない。0 が公式。", "",
             "| 構成 | 閾値 | 判定の停止 | エポック/反復 | バッチ | test MSE | 復活数 | うち 1 エポック目 | "
             "発動時点の中央値 | 標的の質量 / 平均 | 標的が共有ブロブ上 | 標的が単独 | 単独かつ小ブロブ | "
             "標的ブロブの大きさ / 平均 | 寄贈元ブロブの大きさ / 平均 | 再復活の割合 | "
             "未被覆ブロブ | 未被覆の質量 | 重複ブロブ |",
             "|" + "---|" * 19]
    for (alg, thr, warmup, passes), g in groups.items():
        m = {key: _median([r.get(key, "") for r in g]) for key in (
            "batch_size", "test_mse", "events", "events_epoch0", "median_step_fraction",
            "target_mass_ratio_median", "target_on_shared_blob", "target_sole_on_blob",
            "target_sole_on_small_blob", "target_blob_ratio_median", "source_blob_ratio_median",
            "events_on_rerevived", "uncovered_blobs", "uncovered_mass", "crowded_blobs")}
        lines.append("| " + " | ".join([
            alg, thr, warmup, str(passes), f(m["batch_size"], ".0f"), f(m["test_mse"], ".5g"),
            f(m["events"], ".0f"), f(m["events_epoch0"], ".0%"), f(m["median_step_fraction"], ".0%"),
            f(m["target_mass_ratio_median"], ".3f"), f(m["target_on_shared_blob"], ".0%"),
            f(m["target_sole_on_blob"], ".0%"), f(m["target_sole_on_small_blob"], ".0%"),
            f(m["target_blob_ratio_median"], ".3f"), f(m["source_blob_ratio_median"], ".2f"),
            f(m["events_on_rerevived"], ".0%"), f(m["uncovered_blobs"], ".0f"),
            f(m["uncovered_mass"], ".2%"), f(m["crowded_blobs"], ".0f")]) + " |")
    lines += ["", "列の定義: `docs/findings-revival.md` を参照。生のイベント列は "
              f"`results/{metadata['run_name']}/events.csv`（Git 管理外）。", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
