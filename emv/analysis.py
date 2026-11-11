"""Summaries, quality-matched speedups and the Markdown report.

Everything here is computed from measured rows only; nothing is assumed.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

KMEANS = "faiss_kmeans"
EMEANS = "e_means"

RAW_KEYS = ["dataset", "seed", "repeat", "algorithm", "passes", "init", "run_index",
            "block_position", "seconds", "train_mse", "test_mse", "empty_clusters",
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
            rows.append(r)
    return rows


def _stats(values) -> tuple[float, float, float, float]:
    a = np.asarray(values, dtype=np.float64)
    std = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return float(np.median(a)), float(a.min()), float(a.max()), std


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

    Baseline: Faiss k-means at ``baseline_niter`` iterations. An e-means
    configuration "reaches" the baseline when its test MSE is
    <= baseline test MSE * (1 + tolerance). Among reaching configurations the
    fastest one is used. Done twice: on medians across seeds (kit rule), and
    per seed (baseline and candidate from the same seed) to expose variance.
    """
    out = []
    for dataset in sorted({r["dataset"] for r in rows}):
        ds = [s for s in summaries if s["dataset"] == dataset]
        base = next((s for s in ds if s["algorithm"] == KMEANS and s["passes"] == baseline_niter),
                    None)
        if base is None:
            raise ValueError(f"{dataset}: no {KMEANS} run with {baseline_niter} iterations")
        threshold = base["median_test_mse"] * (1 + tolerance)
        em = [s for s in ds if s["algorithm"] == EMEANS]
        km = [s for s in ds if s["algorithm"] == KMEANS]
        reached = [s for s in em if s["median_test_mse"] <= threshold]
        best = min(reached, key=lambda s: s["median_seconds"]) if reached else None
        km_best = min((s for s in km if s["median_test_mse"] <= threshold),
                      key=lambda s: s["median_seconds"])
        best_ratio = min(s["median_test_mse"] for s in em) / base["median_test_mse"] if em else None

        seed_speedups, n_seeds = [], 0
        for seed, cfg in sorted(_per_seed(rows, dataset).items()):
            if (KMEANS, baseline_niter) not in cfg:
                continue
            n_seeds += 1
            b_sec, b_mse = cfg[(KMEANS, baseline_niter)]
            ok = [v for (alg, _), v in cfg.items()
                  if alg == EMEANS and v[1] <= b_mse * (1 + tolerance)]
            if ok:
                seed_speedups.append(b_sec / max(min(v[0] for v in ok), 1e-9))

        entry = {
            "dataset": dataset,
            "baseline": f"{KMEANS}@{baseline_niter}",
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
        out.append(entry)
    return out


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


def render_report(metadata: dict, summaries: list[dict], comparisons: list[dict]) -> str:
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
        "`speedup_vs_fastest_kmeans` は、Faiss k-means 自身が同じ閾値に達する最短反復数との比較（保守的な倍率）。",
        "",
        _table(comparisons, [
            ("dataset", "dataset"),
            ("baseline_median_test_mse", "Lloyd25 test MSE"),
            ("baseline_median_seconds", "Lloyd25 秒"),
            ("emeans_reached", "e-means 到達"),
            ("emeans_best_passes", "到達エポック"),
            ("emeans_best_median_seconds", "e-means 秒"),
            ("emeans_best_median_empty_clusters", "到達構成の空クラスタ"),
            ("speedup_vs_baseline", "倍率 (vs Lloyd25)"),
            ("kmeans_fastest_niter_within_tol", "k-means 最短到達反復"),
            ("speedup_vs_fastest_kmeans", "倍率 (vs k-means最短)"),
            ("emeans_best_mse_ratio", "e-means最良MSE / Lloyd25"),
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
