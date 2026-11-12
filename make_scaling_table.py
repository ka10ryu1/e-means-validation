#!/usr/bin/env python3
"""Cross-run tables for the scaling sweep (no new measurements).

Reads reports/<run>/{metadata.json,summary.csv,comparison.csv} of several runs
and prints Markdown tables, one row per run. Values are copied from those
files; the only derived quantity is the per-iteration cost ratio
e-means 1 epoch / (Lloyd 5 iterations / 5), from the summary medians.
The batch settings of the 1-epoch e-means come from results/<run>/raw_results.csv
when it exists (the batch law also depends on the epoch count) (raw data is not in Git); otherwise they are shown as "—".

    uv run python make_scaling_table.py reports/speed_..._medium_t4 reports/scale_*_t4
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DATASETS = ("gaussian", "imbalanced", "separated")
COMPARISONS = (
    ("A", "faiss_kmeans@25", "e_means", "公式デフォルトの e-means 対 Lloyd25"),
    ("B", "faiss_kmeans_pp@25", "e_means_pp", "e-means++ 対 Lloyd++25（初期重心が同一）"),
)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(text: str) -> str:
    """Format a CSV number compactly ('' stays empty, integers lose '.0')."""
    if text == "":
        return ""
    value = float(text)
    return str(int(value)) if value.is_integer() else f"{value:.3g}"


def load_run(report_dir: Path) -> dict:
    meta = json.loads((report_dir / "metadata.json").read_text(encoding="utf-8"))
    summary = {(r["dataset"], r["algorithm"], float(r["passes"])): r
               for r in _read_csv(report_dir / "summary.csv")}
    comparison = {(r["dataset"], r["baseline"], r["candidate"]): r
                  for r in _read_csv(report_dir / "comparison.csv")}
    raw_path = report_dir.parent.parent / "results" / report_dir.name / "raw_results.csv"
    batch = None
    if raw_path.is_file():
        batch = sorted({(r["batch_size"], r["batch_law"], r["unused_threshold"])
                        for r in _read_csv(raw_path)
                        if r["algorithm"] == "e_means" and float(r["passes"]) == 1.0})
    return dict(name=report_dir.name, meta=meta, summary=summary,
                comparison=comparison, batch=batch)


def size_label(run: dict) -> str:
    meta = run["meta"]
    s = meta["settings"]
    label = f"n={s['ntrain']:,} k={s['k']:,} d={s['dim']}"
    law = meta.get("emeans_batch_law", "auto")
    if law != "auto":
        label += f" batch_law={law}"
    return label + f" / {meta.get('threads', '?')} スレッド"


def cost_ratio(run: dict, dataset: str) -> float:
    s = run["summary"]
    epoch = float(s[(dataset, "e_means", 1.0)]["median_seconds"])
    lloyd5 = float(s[(dataset, "faiss_kmeans", 5.0)]["median_seconds"])
    return epoch / (lloyd5 / 5)


def comparison_cell(run: dict, dataset: str, baseline: str, candidate: str) -> str:
    c = run["comparison"].get((dataset, baseline, candidate))
    if c is None:
        return "—"
    seeds = c["per_seed_reached"]
    if c["emeans_reached"] != "True":
        return f"**NO**（{_num(c['emeans_best_mse_ratio'])}、{seeds}）"
    return (f"{_num(c['emeans_min_passes_reached'])} ep / {_num(c['speedup_vs_fastest_kmeans'])}× / "
            f"空 {_num(c['emeans_min_passes_median_empty_clusters'])}（{seeds}）")


def render(runs: list[dict]) -> str:
    out = ["## 規模を振った比較（各レポートの comparison.csv から転記）", "",
           "セル: 到達した最少エポック / 倍率 vs 同手法最短 / その構成の空クラスタ数（シード別到達）。",
           "未到達は **NO**（e-means 最良 MSE ÷ 基準 MSE、シード別到達）。", ""]
    for key, baseline, candidate, title in COMPARISONS:
        out += [f"### {key}. {title}", "",
                "| 規模 | " + " | ".join(DATASETS) + " | レポート |",
                "|---|" + "---|" * (len(DATASETS) + 1)]
        for run in runs:
            cells = [comparison_cell(run, d, baseline, candidate) for d in DATASETS]
            out.append(f"| {size_label(run)} | " + " | ".join(cells) + f" | `{run['name']}` |")
        out.append("")
    out += ["### 1 エポック / 1 反復の時間比と e-means の設定", "",
            "時間比 = e-means 1 エポックの秒 ÷（Lloyd 5 反復の秒 ÷ 5）、各データセットの中央値から。",
            "", "| 規模 | " + " | ".join(DATASETS) + " | 1 エポックの batch_size / batch_law / unused_threshold |",
            "|---|" + "---|" * (len(DATASETS) + 1)]
    for run in runs:
        ratios = [f"{cost_ratio(run, d):.2f}" for d in DATASETS]
        batch = ("; ".join(" / ".join(b) for b in run["batch"]) if run["batch"] else "—")
        out.append(f"| {size_label(run)} | " + " | ".join(ratios) + f" | {batch} |")
    out.append("")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("reports", nargs="+", type=Path, help="reports/<run> directories")
    p.add_argument("--out", type=Path, default=None, help="Also write the Markdown here")
    args = p.parse_args(argv)
    text = render([load_run(d) for d in args.reports])
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
