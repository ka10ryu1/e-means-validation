#!/usr/bin/env python3
"""Independent, CPU-only Faiss k-means vs official NumpyEmeans benchmark.

Ported from e_means_validation_kit/benchmark.py. All algorithms are scored
with the same exact faiss.knn (squared L2). Training and evaluation timings
are separated. No benchmark numbers are assumed.

    uv run python benchmark.py --profile quick --threads 4
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
PROFILE_NAMES = ("smoke", "quick", "medium")
DATASET_NAMES = ("separated", "gaussian", "imbalanced")
DEFAULT_SEEDS = {"smoke": [0, 1], "quick": [0, 1, 2], "medium": [0, 1, 2, 3, 4]}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--faiss-source", type=Path, default=ROOT / "faiss-src",
                   help="Faiss checkout prepared by scripts/setup_faiss.sh")
    p.add_argument("--allow-unpinned", action="store_true",
                   help="Run even if faiss-src differs from FAISS_COMMIT (recorded in metadata)")
    p.add_argument("--profile", choices=PROFILE_NAMES, default="quick")
    p.add_argument("--datasets", nargs="+", choices=DATASET_NAMES, default=list(DATASET_NAMES))
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Default: smoke 0 1 / quick 0 1 2 / medium 0..4")
    p.add_argument("--repeats", type=int, default=1,
                   help="Re-run every (seed, config) this many times (timing noise)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--k", type=int, default=None, help="Override cluster count")
    p.add_argument("--ntrain", type=int, default=None, help="Override training rows")
    p.add_argument("--ntest", type=int, default=None, help="Override held-out rows")
    p.add_argument("--dim", type=int, default=None,
                   help="Override dimension (synthetic data only)")
    p.add_argument("--kmeans-niters", nargs="+", type=int, default=[5, 10, 25],
                   help="Faiss k-means iteration counts (must include --baseline-niter)")
    p.add_argument("--kmeans-inits", nargs="+", choices=("random", "kmeanspp"),
                   default=["random", "kmeanspp"],
                   help="Faiss k-means seeding: random (Faiss default, required as the "
                        "baseline) and/or kmeanspp (k-means++). kmeanspp is dropped with --init shared")
    p.add_argument("--emeans-inits", nargs="+", choices=("random", "kmeanspp"),
                   default=["random", "kmeanspp"],
                   help="NumpyEmeans seeding: random (official default) and/or kmeanspp "
                        "(same k-means++ centroids as the Faiss k-means++ baseline). "
                        "kmeanspp is dropped with --init shared")
    p.add_argument("--emeans-batch-law", choices=("auto", "throughput", "quality"),
                   default="auto", help="NumpyEmeans batch_law (official default: auto)")
    p.add_argument("--emeans-unused-threshold", type=float, default=None,
                   help="NumpyEmeans unused_threshold for cluster revival "
                        "(official default None = 0.1, or 0 when d <= 2; 0 disables revival)")
    p.add_argument("--imbalance-exponent", type=float, default=1.3,
                   help="Power-law exponent of the imbalanced dataset (kit default 1.3)")
    p.add_argument("--quality-only", action="store_true",
                   help="Run is for quality (MSE) only, e.g. while other work is running; "
                        "the report omits all time and speedup columns")
    p.add_argument("--emeans-epochs", nargs="+", type=float, default=[1, 2, 3, 5, 10],
                   help="NumpyEmeans nepochs values (kit default was 2 3 5)")
    p.add_argument("--baseline-niter", type=int, default=25)
    p.add_argument("--tolerance", type=float, default=0.01,
                   help="Quality match: test MSE <= baseline * (1 + tolerance)")
    p.add_argument("--init", choices=("native", "shared"), default="native",
                   help="native: each library's own seeded init; "
                        "shared: both start from identical centroids")
    p.add_argument("--order", choices=("shuffled", "fixed"), default="shuffled",
                   help="Run order of configurations within each (dataset, seed) block")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--input", type=Path, default=None,
                   help="Optional .npy matrix [n, d]; replaces synthetic datasets")
    p.add_argument("--out", type=Path, default=None,
                   help="Raw results directory (default: results/<run-name>)")
    p.add_argument("--report-dir", type=Path, default=None,
                   help="Git-tracked summary directory (default: reports/<run-name>)")
    p.add_argument("--run-name", default=None)
    args = p.parse_args(argv)
    if args.threads < 1:
        p.error("--threads must be >= 1")
    if args.repeats < 1:
        p.error("--repeats must be >= 1")
    for name in ("ntrain", "ntest", "dim"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            p.error(f"--{name} must be >= 1")
    if args.dim is not None and args.input is not None:
        p.error("--dim cannot be combined with --input (the dimension comes from the file)")
    if args.baseline_niter not in args.kmeans_niters:
        p.error("--kmeans-niters must include --baseline-niter")
    if "random" not in args.kmeans_inits:
        p.error("--kmeans-inits must include random (the baseline)")
    if args.init == "shared" and "kmeanspp" in args.kmeans_inits:
        args.kmeans_inits = ["random"]  # shared centroids replace any seeding method
    if args.init == "shared" and "kmeanspp" in args.emeans_inits:
        args.emeans_inits = ["random"]
    if not args.emeans_inits:
        p.error("--emeans-inits needs at least one value")
    from emv.analysis import normalize_passes
    args.emeans_epochs = [normalize_passes(e) for e in args.emeans_epochs]
    if args.seeds is None:
        args.seeds = DEFAULT_SEEDS[args.profile]
    return args


def main(argv=None):
    args = parse_args(argv)
    # Fix thread counts before numpy / faiss load their thread pools.
    for var in THREAD_ENV:
        os.environ[var] = str(args.threads)

    import json
    import platform
    import subprocess
    import time
    from datetime import datetime, timezone

    import numpy as np

    from emv import analysis, data, official, runners

    try:
        import faiss
    except ImportError as exc:
        raise SystemExit("faiss missing: run `uv sync`") from exc
    try:
        NumpyEmeans = official.load_official_emeans(faiss, args.faiss_source)
    except (ImportError, FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(f"Cannot load official e-means: {exc}") from exc
    source = official.source_info(faiss, args.faiss_source)
    if not args.allow_unpinned:
        try:
            official.check_pinned(source)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
    faiss.omp_set_num_threads(args.threads)

    settings = dict(data.PROFILES[args.profile])
    overrides = {name: getattr(args, name) for name in ("ntrain", "ntest", "dim", "k")
                 if getattr(args, name) is not None}
    settings.update(overrides)
    if not 2 <= settings["k"] <= settings["ntrain"]:
        raise SystemExit("k must be between 2 and ntrain")

    started_at = datetime.now(timezone.utc)
    run_name = args.run_name or f"{args.profile}_{started_at:%Y%m%d-%H%M%S}"
    out = args.out or ROOT / "results" / run_name
    report_dir = args.report_dir or ROOT / "reports" / run_name
    out.mkdir(parents=True, exist_ok=True)

    def git(*a):
        try:
            return subprocess.check_output(["git", "-C", str(ROOT), *a], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    def loadavg():
        try:
            return [float(v) for v in Path("/proc/loadavg").read_text().split()[:3]]
        except (OSError, ValueError):
            return None

    def cpu_model():
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
        return platform.processor() or "unknown"

    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    except Exception:  # informational only; numpy versions differ
        blas = "unknown"

    # real data: the dataset label is the file stem (e.g. sift1m), and the file hash is recorded
    datasets = [args.input.stem] if args.input else args.datasets
    input_sha256 = None
    if args.input:
        import hashlib
        h = hashlib.sha256()
        with args.input.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        input_sha256 = h.hexdigest()
    metadata = {
        "run_name": run_name,
        "started_at": started_at.isoformat(timespec="seconds"),
        "command": ["python", *sys.argv],
        "profile": args.profile, "settings": settings, "settings_overrides": overrides,
        "datasets": datasets,
        "seeds": args.seeds, "repeats": args.repeats, "threads": args.threads,
        "thread_env": {v: os.environ.get(v) for v in THREAD_ENV},
        "quality_only": args.quality_only,
        "emeans_batch_law": args.emeans_batch_law,
        "emeans_unused_threshold": args.emeans_unused_threshold,
        "imbalance_exponent": args.imbalance_exponent,
        "kmeans_niters": args.kmeans_niters, "kmeans_inits": args.kmeans_inits,
        "emeans_inits": args.emeans_inits, "emeans_epochs": args.emeans_epochs,
        "baseline_niter": args.baseline_niter, "tolerance": args.tolerance,
        "init": args.init,
        "init_note": ("Faiss k-means と e-means はそれぞれ独自の乱数でk点を学習データから抽出"
                      "（同一seedでも初期重心は一致しない）" if args.init == "native" else
                      "両者に同一の初期重心（RandomState(seed).permutation(n)[:k] の行）を与える"),
        "order": args.order, "warmup": not args.no_warmup,
        "input": official.display_path(args.input) if args.input else "synthetic",
        "input_sha256": input_sha256,
        "data_seed": data.DATA_SEED,
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "numpy_version": np.__version__, "numpy_blas": blas,
        "python_version": platform.python_version(), "python": sys.version,
        "platform": platform.platform(), "cpu_model": cpu_model(), "cpu_count": os.cpu_count(),
        "repo_commit": git("rev-parse", "HEAD"),
        "repo_dirty": bool(git("status", "--porcelain", "--", ".", ":!results", ":!reports")),
        "e_means_source": source,
        "quality_rule": (f"test MSE <= 基準（Faiss k-means {args.baseline_niter}反復。ランダム初期化、"
                         f"計測時は k-means++ 初期化も）の test MSE × "
                         f"{1 + args.tolerance:g}（中央値どうし／シード別の両方で判定）"),
        "timing": "training only (train() call), time.perf_counter; evaluation excluded",
        "timing_diagnostics": ("cpu_seconds = process CPU time (all threads), "
                               "main_thread_cpu_seconds = calling-thread CPU time, "
                               "loadavg_1m = /proc/loadavg before each run (includes this process). "
                               "CPU clock is not observable inside WSL2 and is not recorded."),
        "loadavg_start": loadavg(),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in ("run_name", "profile", "settings", "seeds",
                                               "threads", "init", "faiss_version")}, indent=2))
    print(f"e-means source: {source['faiss_source_path']} @ {source['faiss_source_sha']}")

    kmeans_algs = {analysis.KMEANS: "random", analysis.KMEANS_PP: "kmeanspp"}
    emeans_algs = {analysis.EMEANS: "random", analysis.EMEANS_PP: "kmeanspp"}
    configs = ([(alg, n) for alg, how in kmeans_algs.items() if how in args.kmeans_inits
                for n in args.kmeans_niters]
               + [(alg, e) for alg, how in emeans_algs.items() if how in args.emeans_inits
                  for e in args.emeans_epochs])
    rows, run_index = [], 0
    for ds_index, dataset in enumerate(datasets):
        t0 = time.perf_counter()
        if args.input:
            train, test = data.real_embeddings(args.input, settings["ntrain"], settings["ntest"])
            settings["dim"] = train.shape[1]
        else:
            train, test = data.synthetic_data(dataset, **settings,
                                              imbalance_exponent=args.imbalance_exponent)
        print(f"\nDataset={dataset} train={train.shape} test={test.shape} "
              f"(generated in {time.perf_counter() - t0:.2f}s, not part of timings)", flush=True)
        if not args.no_warmup:
            runners.warmup(faiss, NumpyEmeans, train, settings["k"])
        for seed in args.seeds:
            init = data.shared_init(train, settings["k"], seed) if args.init == "shared" else None
            for repeat in range(args.repeats):
                order = list(range(len(configs)))
                if args.order == "shuffled":
                    rng = np.random.default_rng([4242, ds_index, seed, repeat])
                    order = list(rng.permutation(len(configs)))
                for position, ci in enumerate(order):
                    alg, passes = configs[ci]
                    load = loadavg()
                    if alg in kmeans_algs:
                        centers, seconds, extra = runners.run_faiss_kmeans(
                            faiss, train, settings["k"], seed, passes, init, kmeans_algs[alg])
                    else:
                        centers, seconds, extra = runners.run_emeans(
                            NumpyEmeans, train, settings["k"], seed, passes, init,
                            emeans_algs[alg], faiss, args.emeans_batch_law,
                            args.emeans_unused_threshold)
                    train_mse, empty = runners.score(faiss, train, centers)
                    test_mse, _ = runners.score(faiss, test, centers)
                    rows.append(dict(
                        dataset=dataset, seed=seed, repeat=repeat, algorithm=alg, passes=passes,
                        init=args.init, run_index=run_index, block_position=position,
                        seconds=round(seconds, 6), cpu_seconds=extra.get("cpu_seconds", ""),
                        main_thread_cpu_seconds=extra.get("main_thread_cpu_seconds", ""),
                        loadavg_1m=load[0] if load else "", train_mse=round(train_mse, 8),
                        test_mse=round(test_mse, 8), empty_clusters=empty,
                        batch_size=extra.get("batch_size", ""), alpha0=extra.get("alpha0", ""),
                        batch_law=extra.get("batch_law", ""), splits=extra.get("splits", ""),
                        unused_threshold=extra.get("unused_threshold", "")))
                    run_index += 1
                    print(f"  seed={seed:>3} rep={repeat} {alg:12} passes={passes:>3}: "
                          f"{seconds:8.3f}s  train MSE={train_mse:.5f}  test MSE={test_mse:.5f}  "
                          f"empty={empty}", flush=True)
                    # preserve partial results if interrupted
                    analysis.write_csv(out / "raw_results.csv", rows, analysis.RAW_KEYS)

    summaries = analysis.summarize(rows)
    comparisons = analysis.compare(rows, summaries, args.baseline_niter, args.tolerance)
    metadata["loadavg_end"] = loadavg()
    metadata["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_outputs(out, report_dir, metadata, summaries, comparisons,
                  analysis.timing_diagnostics(rows))

    title = "Quality match" if args.quality_only else "Quality-matched speed"
    print(f"\n=== {title} (test MSE <= baseline x {1 + args.tolerance:g}, medians) ===")
    for c in comparisons:
        if c["emeans_reached"] and args.quality_only:
            print(f"{c['dataset']:>11} {c['candidate']:<10} vs {c['baseline']:<19}: reached at "
                  f"{c['emeans_min_passes_reached']} epochs (best MSE ratio "
                  f"{c['emeans_best_mse_ratio']}); per-seed reached {c['per_seed_reached']}")
        elif c["emeans_reached"]:
            print(f"{c['dataset']:>11} {c['candidate']:<10} vs {c['baseline']:<19}: "
                  f"{c['speedup_vs_baseline']:.2f}x at e-means "
                  f"{c['emeans_best_passes']} epochs (baseline MSE {c['baseline_median_test_mse']:.5f},"
                  f" e-means MSE {c['emeans_best_median_test_mse']:.5f}); "
                  f"per-seed reached {c['per_seed_reached']}")
        else:
            print(f"{c['dataset']:>11} {c['candidate']:<10} vs {c['baseline']:<19}: did NOT reach baseline "
                  f"held-out MSE within {args.tolerance:.0%} (best ratio "
                  f"{c['emeans_best_mse_ratio']}); per-seed reached {c['per_seed_reached']}")
    print(f"\nRaw CSV + metadata: {out}\nReport: {report_dir / 'report.md'}")


def write_outputs(out, report_dir, metadata, summaries, comparisons, diagnostics=None):
    import json

    from emv import analysis

    report = analysis.render_report(metadata, summaries, comparisons, diagnostics)
    for d in (out, report_dir):
        d.mkdir(parents=True, exist_ok=True)
        analysis.write_csv(d / "summary.csv", summaries)
        analysis.write_csv(d / "comparison.csv", comparisons)
        (d / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
        (d / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
