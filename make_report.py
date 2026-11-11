#!/usr/bin/env python3
"""Regenerate summary/comparison CSVs and the Markdown report from raw results.

    uv run python make_report.py results/<run-name> [--report-dir reports/<run-name>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark import ROOT, write_outputs
from emv import analysis


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", type=Path, help="Directory with raw_results.csv and metadata.json")
    p.add_argument("--report-dir", type=Path, default=None)
    args = p.parse_args(argv)
    metadata = json.loads((args.results / "metadata.json").read_text(encoding="utf-8"))
    rows = analysis.read_raw_csv(args.results / "raw_results.csv")
    summaries = analysis.summarize(rows)
    comparisons = analysis.compare(rows, summaries, metadata["baseline_niter"],
                                   metadata["tolerance"])
    report_dir = args.report_dir or ROOT / "reports" / metadata["run_name"]
    write_outputs(args.results, report_dir, metadata, summaries, comparisons)
    print(f"Report: {report_dir / 'report.md'}")


if __name__ == "__main__":
    main()
