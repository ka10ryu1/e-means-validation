#!/usr/bin/env python3
"""Tolerance-sensitivity re-read of existing runs (no new measurements).

The judgement rule stays as recorded in each run's metadata (default 1%). This
script recomputes the quality-matched comparison from results/<run>/raw_results.csv
with the same code path (emv.analysis.compare) at several tolerances and prints
one Markdown table per comparison, so one can see how much the "reached /
speedup" verdicts depend on the 1% tolerance. It is a supplementary reading,
not a rule change: reports/<run>/ files are not modified.

As a reproducibility check, the comparison at the run's own recorded tolerance
must equal reports/<run>/comparison.csv; otherwise the script stops.

Speedups are shown only for runs that measured time (not --quality-only) and
had no CPU contention detected; otherwise only quality columns are shown.

    uv run python make_tolerance_table.py reports/speed_..._medium_t4 reports/scale_*_t4 \
        [--tolerances 0.01 0.005 0.001 0] [--out docs/tolerance-sensitivity.md]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from emv import analysis

DEFAULT_TOLERANCES = (0.01, 0.005, 0.001, 0.0)
# (key, baseline algorithm, candidate, title). The baseline iteration count comes
# from each run's metadata (benchmark.py --baseline-niter), not a fixed 25.
COMPARISONS = (
    ("A", "faiss_kmeans", "e_means", "公式デフォルトの e-means 対 Lloyd（基準反復）"),
    ("B", "faiss_kmeans_pp", "e_means_pp", "e-means++ 対 Lloyd++（基準反復、初期重心が同一）"),
)
# Columns compared against the committed comparison.csv (reproducibility check).
CHECK_KEYS = ("emeans_reached", "emeans_min_passes_reached", "kmeans_fastest_niter_within_tol",
              "speedup_vs_fastest_kmeans", "per_seed_reached")


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(value) -> str:
    if value in ("", None):
        return ""
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:.3g}"


def _pct(tol: float) -> str:
    return f"{tol * 100:g}%"


def load_run(report_dir: Path) -> dict:
    meta = json.loads((report_dir / "metadata.json").read_text(encoding="utf-8"))
    raw_path = report_dir.parent.parent / "results" / report_dir.name / "raw_results.csv"
    if not raw_path.is_file():
        raise FileNotFoundError(f"{raw_path} not found (raw data is not in Git)")
    rows = analysis.read_raw_csv(raw_path)
    summaries = analysis.summarize(rows)
    contended = sum(d["contended_runs"] for d in analysis.timing_diagnostics(rows))
    run = dict(name=report_dir.name, meta=meta, rows=rows, summaries=summaries,
               show_speed=not meta.get("quality_only") and contended == 0,
               contended=contended)
    check_reproduction(run, _read_csv(report_dir / "comparison.csv"))
    return run


def comparisons_at(run: dict, tolerance: float) -> dict:
    out = analysis.compare(run["rows"], run["summaries"], run["meta"]["baseline_niter"], tolerance)
    return {(c["dataset"], c["baseline"], c["candidate"]): c for c in out}


def check_reproduction(run: dict, committed: list[dict]) -> None:
    """The recorded-tolerance recomputation must match reports/<run>/comparison.csv."""
    fresh = comparisons_at(run, run["meta"]["tolerance"])
    for c in committed:
        f = fresh.get((c["dataset"], c["baseline"], c["candidate"]))
        if f is None:
            raise ValueError(f"{run['name']}: {c['dataset']} {c['baseline']} {c['candidate']} "
                             "missing on recomputation")
        for key in CHECK_KEYS:
            got = str(f[key]) if key in ("emeans_reached", "per_seed_reached") else _num(f[key])
            want = c[key] if key in ("emeans_reached", "per_seed_reached") else _num(c[key])
            if got != want:
                raise ValueError(f"{run['name']}: {c['dataset']} {c['baseline']} {c['candidate']} "
                                 f"{key}: recomputed {got!r} != committed {want!r}")
    if len(fresh) != len(committed):
        raise ValueError(f"{run['name']}: {len(fresh)} comparisons recomputed, "
                         f"{len(committed)} committed")


def cell(c: dict | None, show_speed: bool) -> str:
    if c is None:
        return "—"
    lloyd = f"L{_num(c['kmeans_fastest_niter_within_tol'])}"
    seeds = c["per_seed_reached"]
    if not c["emeans_reached"]:
        return f"**NO** ({_num(c['emeans_best_mse_ratio'])}) / {lloyd} ({seeds})"
    text = f"{_num(c['emeans_min_passes_reached'])} ep / {lloyd}"
    if show_speed:
        text += f" / {_num(c['speedup_vs_fastest_kmeans'])}×"
    return f"{text} ({seeds})"


def render(runs: list[dict], tolerances: list[float]) -> str:
    out = [
        "## 許容幅を変えた再読（既存の生データから再計算、新しい計測なし）", "",
        "判定規則（基準の 1% 以内）は変えていない。同じ生データ・同じ集計コードで許容幅だけを変えた補足の読み方。",
        "",
        "セル: e-means が到達した最少エポック / 同じ許容幅で到達した Lloyd の最短反復（例: L5・L10・L25。基準列は基準の反復数）"
        " / 倍率 vs 同手法最短（中央値の秒の比）（シード別到達数）。",
        "未到達は **NO**（e-means 最良 MSE ÷ 基準 MSE）。倍率は時間を計測し CPU の取り合いを検出しなかった実行のみ表示。",
        "倍率は「到達した中で最速の e-means 構成」と「同じ許容幅で到達した最速の Lloyd」の比。",
        "",
    ]
    cols = [_pct(t) for t in tolerances]
    for key, base_alg, candidate, title in COMPARISONS:
        out += [f"### {key}. {title}", "",
                "| 実行 | データ | 基準 | " + " | ".join(cols) + " |",
                "|---|---|---|" + "---|" * len(cols)]
        for run in runs:
            niter = run["meta"]["baseline_niter"]
            baseline = f"{base_alg}@{niter}"
            per_tol = [comparisons_at(run, t) for t in tolerances]
            for dataset in sorted({r["dataset"] for r in run["rows"]}):
                cells = [cell(p.get((dataset, baseline, candidate)), run["show_speed"])
                         for p in per_tol]
                if all(c == "—" for c in cells):
                    continue
                out.append(f"| `{run['name']}` | {dataset} | L{niter} | " + " | ".join(cells) + " |")
        out.append("")
    skipped = [r for r in runs if not r["show_speed"]]
    if skipped:
        notes = []
        for r in skipped:
            why = "品質のみ" if r["meta"].get("quality_only") else f"取り合い検出 {r['contended']} 回"
            notes.append(f"`{r['name']}`（{why}）")
        out += ["倍率を表示しなかった実行: " + "、".join(notes), ""]
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("reports", nargs="+", type=Path, help="reports/<run> directories")
    p.add_argument("--tolerances", nargs="+", type=float, default=list(DEFAULT_TOLERANCES))
    p.add_argument("--out", type=Path, default=None, help="Also write the Markdown here")
    args = p.parse_args(argv)
    if any(t < 0 for t in args.tolerances):
        p.error("tolerances must be >= 0")
    text = render([load_run(d) for d in args.reports], args.tolerances)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
