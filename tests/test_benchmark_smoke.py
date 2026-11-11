"""End-to-end run of benchmark.py on the tiny `smoke` profile.

This checks that the pipeline produces its files; it is NOT a performance test.
"""
import csv
import json
import subprocess
import sys

from conftest import ROOT


def test_smoke_benchmark_and_report(official_source, tmp_path):
    out, rep = tmp_path / "raw", tmp_path / "rep"
    subprocess.run(
        [sys.executable, "benchmark.py", "--profile", "smoke", "--threads", "1",
         "--datasets", "separated", "imbalanced", "--seeds", "0",
         "--kmeans-niters", "3", "25", "--emeans-epochs", "1", "2",
         "--out", str(out), "--report-dir", str(rep), "--run-name", "pytest-smoke"],
        cwd=ROOT, check=True, capture_output=True, text=True)

    for d in (out, rep):
        for name in ("summary.csv", "comparison.csv", "metadata.json", "report.md"):
            assert (d / name).is_file(), name
    with (out / "raw_results.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 * 1 * 4
    assert {r["algorithm"] for r in rows} == {"faiss_kmeans", "e_means"}
    assert all(float(r["seconds"]) > 0 and float(r["test_mse"]) > 0 for r in rows)

    meta = json.loads((out / "metadata.json").read_text())
    assert meta["threads"] == 1 and meta["thread_env"]["OMP_NUM_THREADS"] == "1"
    assert meta["e_means_source"]["faiss_source_matches_pin"] is True
    assert "e-means 検証レポート" in (rep / "report.md").read_text()

    # make_report.py regenerates identical comparison from raw data
    rep2 = tmp_path / "rep2"
    subprocess.run([sys.executable, "make_report.py", str(out), "--report-dir", str(rep2)],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    assert (rep2 / "comparison.csv").read_text() == (rep / "comparison.csv").read_text()
