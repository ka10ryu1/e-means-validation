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
    # 2 datasets x 1 seed x (2 k-means inits x 2 niters + 2 e-means inits x 2 epochs)
    assert len(rows) == 2 * 1 * 8
    assert {r["algorithm"] for r in rows} == {"faiss_kmeans", "faiss_kmeans_pp",
                                              "e_means", "e_means_pp"}
    assert all(float(r["seconds"]) > 0 and float(r["test_mse"]) > 0 for r in rows)

    meta_text = (out / "metadata.json").read_text()
    assert str(ROOT) not in meta_text, "metadata must not contain absolute repo paths"
    meta = json.loads(meta_text)
    assert meta["threads"] == 1 and meta["thread_env"]["OMP_NUM_THREADS"] == "1"
    assert meta["e_means_source"]["faiss_source_matches_pin"] is True
    assert "e-means 検証レポート" in (rep / "report.md").read_text()

    # make_report.py regenerates identical comparison from raw data
    rep2 = tmp_path / "rep2"
    subprocess.run([sys.executable, "make_report.py", str(out), "--report-dir", str(rep2)],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    assert (rep2 / "comparison.csv").read_text() == (rep / "comparison.csv").read_text()


def test_shared_init_drops_kmeanspp():
    from benchmark import parse_args
    assert parse_args(["--init", "shared"]).kmeans_inits == ["random"]
    assert parse_args([]).kmeans_inits == ["random", "kmeanspp"]
    assert parse_args(["--init", "shared"]).emeans_inits == ["random"]
    assert parse_args([]).emeans_inits == ["random", "kmeanspp"]
