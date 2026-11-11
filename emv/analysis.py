"""Summaries, quality-matched speedups and the Markdown report.

Everything here is computed from measured rows only; nothing is assumed.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

KMEANS = "faiss_kmeans"          # Lloyd, random init (Faiss default)
KMEANS_PP = "faiss_kmeans_pp"    # Lloyd, k-means++ init
BASELINES = (KMEANS, KMEANS_PP)
EMEANS = "e_means"                # official NumpyEmeans, its own random init
EMEANS_PP = "e_means_pp"          # NumpyEmeans from the Faiss k-means++ centroids
CANDIDATES = (EMEANS, EMEANS_PP)

RAW_KEYS = ["dataset", "seed", "repeat", "algorithm", "passes", "init", "run_index",
            "block_position", "seconds", "cpu_seconds", "main_thread_cpu_seconds",
            "loadavg_1m", "train_mse", "test_mse", "empty_clusters",
            "batch_size", "alpha0", "batch_law", "splits"]


def write_csv(path: Path, rows: list[dict], keys: list[str] | None = None) -> None:
    keys = keys or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_raw_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for key in ("seed", "repeat", "passes", "run_index", "block_position",
                        "empty_clusters"):
                r[key] = int(float(r[key]))
            for key in ("seconds", "train_mse", "test_mse"):
                r[key] = float(r[key])
            for key in ("cpu_seconds", "main_thread_cpu_seconds", "loadavg_1m"):
                # absent in results produced before these diagnostics existed
                r[key] = float(r[key]) if r.get(key) not in (None, "") else None
            rows.append(r)
    return rows


def _stats(values) -> tuple[float, float, float, float]:
    a = np.asarray(values, dtype=np.float64)
    std = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return float(np.median(a)), float(a.min()), float(a.max()), std


def _median_opt(members, key, ndigits):
    vals = [m.get(key) for m in members if m.get(key) not in (None, "")]
    return round(float(np.median(np.asarray(vals, dtype=np.float64))), ndigits) if vals else ""


def timing_diagnostics(rows: list[dict]) -> list[dict]:
    """Per algorithm: median CPU/wall ratios and load average over all runs."""
    out = []
    for alg in sorted({r["algorithm"] for r in rows}):
        rs = [r for r in rows if r["algorithm"] == alg
              and r.get("cpu_seconds") not in (None, "") and float(r["seconds"]) > 0]
        if not rs:
            continue
        cpu = [float(r["cpu_seconds"]) / float(r["seconds"]) for r in rs]
        main = [float(r["main_thread_cpu_seconds"]) / float(r["seconds"]) for r in rs]
        load = [float(r["loadavg_1m"]) for r in rs if r.get("loadavg_1m") not in (None, "")]
        out.append({
            "algorithm": alg, "runs": len(rs),
            "median_process_cpu_per_wall": round(float(np.median(cpu)), 3),
            "median_main_thread_cpu_per_wall": round(float(np.median(main)), 3),
            "median_loadavg_1m": round(float(np.median(load)), 2) if load else "",
            "max_loadavg_1m": round(max(load), 2) if load else "",
        })
    return out


def summarize(rows: list[dict]) -> list[dict]:
    """Per (dataset, algorithm, passes): medians plus spread over seeds x repeats."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["algorithm"], int(row["passes"]))].append(row)
    summaries = []
    for (dataset, alg, passes), members in sorted(grouped.items()):
        sec_med, sec_min, sec_max, sec_std = _stats([r["seconds"] for r in members])
        mse_med, mse_min, mse_max, mse_std = _stats([r["test_mse"] for r in members])
        summaries.append({
            "dataset": dataset,
            "algorithm": alg,
            "passes": passes,
            "runs": len(members),
            "median_seconds": round(sec_med, 5),
            "min_seconds": round(sec_min, 5),
            "max_seconds": round(sec_max, 5),
            "cv_seconds": round(sec_std / sec_med, 4) if sec_med > 0 else "",
            "median_cpu_seconds": _median_opt(members, "cpu_seconds", 5),
            "median_main_thread_cpu_seconds": _median_opt(members, "main_thread_cpu_seconds", 5),
            "median_train_mse": round(float(np.median([r["train_mse"] for r in members])), 8),
            "median_test_mse": round(mse_med, 8),
            "min_test_mse": round(mse_min, 8),
            "max_test_mse": round(mse_max, 8),
            "std_test_mse": round(mse_std, 8),
            "median_empty_clusters": float(np.median([r["empty_clusters"] for r in members])),
            "max_empty_clusters": int(max(r["empty_clusters"] for r in members)),
        })
    return summaries


def _per_seed(rows, dataset):
    """{seed: {(alg, passes): (median seconds, median test mse)}} over repeats."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["dataset"] == dataset:
            acc[r["seed"]][(r["algorithm"], int(r["passes"]))].append(r)
    return {
        seed: {key: (float(np.median([m["seconds"] for m in ms])),
                     float(np.median([m["test_mse"] for m in ms])))
               for key, ms in cfg.items()}
        for seed, cfg in acc.items()
    }


def compare(rows: list[dict], summaries: list[dict], baseline_niter: int = 25,
            tolerance: float = 0.01) -> list[dict]:
    """Quality-matched speed per dataset.

    Baseline: Faiss k-means at ``baseline_niter`` iterations, once per
    baseline algorithm present (random init is required, k-means++ optional);
    "fastest k-means" is searched within the same baseline algorithm. Each
    baseline is compared with every e-means variant present (``candidate``:
    e_means = official default init, e_means_pp = k-means++ init). An e-means
    configuration "reaches" the baseline when its test MSE is
    <= baseline test MSE * (1 + tolerance). Among reaching configurations the
    fastest one is used. Done twice: on medians across seeds (kit rule), and
    per seed (baseline and candidate from the same seed) to expose variance.
    """
    out = []
    for dataset in sorted({r["dataset"] for r in rows}):
        ds = [s for s in summaries if s["dataset"] == dataset]
        for base_alg in BASELINES:
            km = [s for s in ds if s["algorithm"] == base_alg]
            base = next((s for s in km if s["passes"] == baseline_niter), None)
            if base is None:
                if base_alg == KMEANS:
                    raise ValueError(
                        f"{dataset}: no {KMEANS} run with {baseline_niter} iterations")
                continue  # optional baseline (e.g. k-means++) not measured
            for cand in CANDIDATES:
                if any(s["algorithm"] == cand for s in ds):
                    out.append(_compare_one(rows, ds, dataset, base_alg, km, base,
                                            baseline_niter, tolerance, cand))
    return out


def _compare_one(rows, ds, dataset, base_alg, km, base, baseline_niter, tolerance,
                 candidate=EMEANS):
    threshold = base["median_test_mse"] * (1 + tolerance)
    em = [s for s in ds if s["algorithm"] == candidate]
    reached = [s for s in em if s["median_test_mse"] <= threshold]
    best = min(reached, key=lambda s: s["median_seconds"]) if reached else None
    km_best = min((s for s in km if s["median_test_mse"] <= threshold),
                  key=lambda s: s["median_seconds"])
    best_ratio = min(s["median_test_mse"] for s in em) / base["median_test_mse"] if em else None

    seed_speedups, n_seeds = [], 0
    for seed, cfg in sorted(_per_seed(rows, dataset).items()):
        if (base_alg, baseline_niter) not in cfg:
            continue
        n_seeds += 1
        b_sec, b_mse = cfg[(base_alg, baseline_niter)]
        ok = [v for (alg, _), v in cfg.items()
              if alg == candidate and v[1] <= b_mse * (1 + tolerance)]
        if ok:
            seed_speedups.append(b_sec / max(min(v[0] for v in ok), 1e-9))

    return {
        "dataset": dataset,
        "baseline": f"{base_alg}@{baseline_niter}",
        "candidate": candidate,
        "baseline_median_test_mse": base["median_test_mse"],
        "threshold_test_mse": round(threshold, 8),
        "baseline_median_seconds": base["median_seconds"],
        "emeans_reached": best is not None,
        "emeans_best_passes": best["passes"] if best else "",
        "emeans_best_median_seconds": best["median_seconds"] if best else "",
        "emeans_best_median_test_mse": best["median_test_mse"] if best else "",
        "emeans_best_median_empty_clusters": best["median_empty_clusters"] if best else "",
        "speedup_vs_baseline": (round(base["median_seconds"] / max(best["median_seconds"], 1e-9), 3)
                                if best else ""),
        "kmeans_fastest_niter_within_tol": km_best["passes"],
        "kmeans_fastest_median_seconds": km_best["median_seconds"],
        "speedup_vs_fastest_kmeans": (round(km_best["median_seconds"] /
                                            max(best["median_seconds"], 1e-9), 3)
                                      if best else ""),
        "emeans_best_mse_ratio": round(best_ratio, 5) if best_ratio is not None else "",
        "per_seed_reached": f"{len(seed_speedups)}/{n_seeds}",
        "per_seed_speedup_median": round(float(np.median(seed_speedups)), 3) if seed_speedups else "",
        "per_seed_speedup_min": round(min(seed_speedups), 3) if seed_speedups else "",
        "per_seed_speedup_max": round(max(seed_speedups), 3) if seed_speedups else "",
    }


# ---------------------------------------------------------------- report

def _table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(h for _, h in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = []
    for r in rows:
        cells = []
        for key, _ in cols:
            v = r.get(key, "")
            if isinstance(v, bool):
                v = "yes" if v else "**NO**"
            elif isinstance(v, float):
                v = f"{v:.5g}"
            cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])


def _diagnostics_section(m: dict, diagnostics: list[dict] | None) -> list[str]:
    if not diagnostics:
        return []
    return [
        "## 計時の診断",
        "",
        "CPU 時間 / 経過時間 の中央値（プロセス全体・メインスレッド）と、各実行直前の 1 分ロードアベレージ"
        "（本プロセス自身のスレッドを含む）。他の回と比べて経過時間だけが伸び、CPU 時間/経過時間 が下がっていれば"
        "CPU の取り合い（待ち）、比が変わらず時間全体が伸びていればコア自体の遅さ（コア種別・クロック）を示唆する。"
        "WSL2 内では CPU クロックは観測できない。",
        "",
        f"- ロードアベレージ（1/5/15 分） 開始時: {m.get('loadavg_start')}  / 終了時: {m.get('loadavg_end')}",
        "",
        _table(diagnostics, [
            ("algorithm", "algorithm"), ("runs", "runs"),
            ("median_process_cpu_per_wall", "プロセスCPU/経過"),
            ("median_main_thread_cpu_per_wall", "メインスレッドCPU/経過"),
            ("median_loadavg_1m", "load 1m 中央値"), ("max_loadavg_1m", "load 1m 最大"),
        ]),
        "",
    ]


def render_report(metadata: dict, summaries: list[dict], comparisons: list[dict],
                  diagnostics: list[dict] | None = None) -> str:
    m = metadata
    src = m.get("e_means_source", {})
    s = m["settings"]
    lines = [
        f"# e-means 検証レポート: {m.get('run_name', '')}",
        "",
        "> 本レポートは `benchmark.py` の実測値のみから自動生成されています。"
        "CPU上の独立ベンチマークであり、論文のGPU実験（H100, Flash-KMeans比較）の再現ではありません。",
        "",
        "## 実行条件",
        "",
        f"- プロファイル: `{m['profile']}` (ntrain={s['ntrain']}, ntest={s['ntest']}, "
        f"dim={s['dim']}, k={s['k']})",
        f"- データ: {', '.join(m['datasets'])}  / 入力: {m['input']}",
        f"- 乱数シード: {m['seeds']}  / 反復: {m['repeats']}  / 実行順: {m['order']}",
        f"- 初期化: `{m['init']}` — {m['init_note']}",
        f"- スレッド数: {m['threads']} (CPU: {m.get('cpu_model', 'unknown')}, "
        f"論理コア {m.get('cpu_count')})",
        f"- Faiss k-means 反復数: {m['kmeans_niters']}  / e-means エポック数: {m['emeans_epochs']}",
        f"- faiss-cpu {m['faiss_version']}, numpy {m['numpy_version']}, Python {m['python_version']}",
        f"- 公式ソース: facebookresearch/faiss @ `{src.get('faiss_source_sha', '?')}` "
        f"(固定SHAと一致: {src.get('faiss_source_matches_pin')}, "
        f"e_means *.py sha256 `{str(src.get('e_means_py_sha256', ''))[:16]}…`)",
        f"- 実行日時: {m.get('started_at')}  / 本リポジトリ commit: `{m.get('repo_commit')}`"
        f"{' (dirty)' if m.get('repo_dirty') else ''}",
        "",
        "## 同等品質到達時間による比較",
        "",
        f"判定規則: {m['quality_rule']}。到達しなかった場合は **NO** と記録し、倍率は算出しない。",
        "到達判定は test MSE のみに基づく。到達構成に空クラスタが残る場合は品質上の注意が必要（`到達構成の空クラスタ` 列）。\n"
        "`speedup_vs_fastest_kmeans` は、基準と同じ手法（同じ初期化）の k-means が同じ閾値に達する最短反復数との比較（保守的な倍率）。",
        "基準 `faiss_kmeans` は Faiss 既定のランダム初期化、`faiss_kmeans_pp` は k-means++ 初期化（初期化時間も学習時間に含む）。"
        "`e_means` は公式デフォルトの初期化、`e_means_pp` は `faiss_kmeans_pp` と同一の k-means++ 初期重心から開始（初期化時間を含む）。"
        "以下の列名の emeans_* は、この行の e-means 初期化（candidate）を指す。",
        "",
        _table(comparisons, [
            ("dataset", "dataset"),
            ("baseline", "基準"),
            ("candidate", "e-means 初期化"),
            ("baseline_median_test_mse", "基準 test MSE"),
            ("baseline_median_seconds", "基準 秒"),
            ("emeans_reached", "e-means 到達"),
            ("emeans_best_passes", "到達エポック"),
            ("emeans_best_median_seconds", "e-means 秒"),
            ("emeans_best_median_empty_clusters", "到達構成の空クラスタ"),
            ("speedup_vs_baseline", "倍率 (vs 基準)"),
            ("kmeans_fastest_niter_within_tol", "同手法の最短到達反復"),
            ("speedup_vs_fastest_kmeans", "倍率 (vs 同手法最短)"),
            ("emeans_best_mse_ratio", "e-means最良MSE / 基準"),
            ("per_seed_reached", "シード別到達"),
            ("per_seed_speedup_median", "シード別倍率 中央値"),
            ("per_seed_speedup_min", "最小"),
            ("per_seed_speedup_max", "最大"),
        ]),
        "",
        "## 構成別の集計（シード×反復の中央値とばらつき）",
        "",
        _table(summaries, [
            ("dataset", "dataset"), ("algorithm", "algorithm"), ("passes", "passes"),
            ("runs", "runs"), ("median_seconds", "秒 中央値"), ("min_seconds", "秒 min"),
            ("max_seconds", "秒 max"), ("cv_seconds", "秒 CV"),
            ("median_train_mse", "train MSE"), ("median_test_mse", "test MSE"),
            ("min_test_mse", "test min"), ("max_test_mse", "test max"),
            ("median_empty_clusters", "空クラスタ 中央値"), ("max_empty_clusters", "空 max"),
        ]),
        "",
        *_diagnostics_section(metadata, diagnostics),
        "## 注意事項",
        "",
        "- 時間は `train()` 呼び出しのみ（`time.perf_counter`）。データ生成・評価は含まない。",
        "- MSE は全アルゴリズム共通の厳密 `faiss.knn`（二乗L2）で最終重心に再割当てして算出。"
        "e-means の train() が返すラベルは使用していない。",
        "- e-means の学習時間には、公式デフォルト（`capture_assignments=True`）による最終エポックのラベル保存コストが含まれる。",
        "- 小規模データでは Python のオーバーヘッドの比率が大きく、倍率は規模に強く依存し得る。",
        "- シード数が少ない場合、中央値・min/max は参考値として扱うこと。",
        "",
    ]
    return "\n".join(lines)
