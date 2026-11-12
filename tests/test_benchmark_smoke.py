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
    # timing diagnostics recorded for every run
    assert all(float(r["cpu_seconds"]) > 0 and float(r["main_thread_cpu_seconds"]) > 0
               and float(r["loadavg_1m"]) >= 0 for r in rows)

    meta_text = (out / "metadata.json").read_text()
    assert str(ROOT) not in meta_text, "metadata must not contain absolute repo paths"
    meta = json.loads(meta_text)
    assert meta["threads"] == 1 and meta["thread_env"]["OMP_NUM_THREADS"] == "1"
    assert meta["e_means_source"]["faiss_source_matches_pin"] is True
    assert "e-means 検証レポート" in (rep / "report.md").read_text()
    assert "## 計時の診断" in (rep / "report.md").read_text()
    assert meta["loadavg_start"] is not None and meta["loadavg_end"] is not None

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


def test_quality_only_report_has_no_time_columns(official_source, tmp_path):
    out, rep = tmp_path / "raw", tmp_path / "rep"
    subprocess.run(
        [sys.executable, "benchmark.py", "--profile", "smoke", "--threads", "1",
         "--datasets", "imbalanced", "--seeds", "0", "--kmeans-niters", "25",
         "--emeans-epochs", "1", "--kmeans-inits", "random", "--emeans-inits", "random",
         "--quality-only", "--emeans-unused-threshold", "0", "--emeans-batch-law", "quality",
         "--imbalance-exponent", "2.0",
         "--out", str(out), "--report-dir", str(rep), "--run-name", "pytest-quality"],
        cwd=ROOT, check=True, capture_output=True, text=True)
    report = (rep / "report.md").read_text()
    assert "品質専用の計測" in report
    for header in ("倍率 (vs", "秒 中央値", "基準 秒", "シード別倍率", "計時は"):
        assert header not in report, header
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["quality_only"] and meta["imbalance_exponent"] == 2.0
    with (out / "raw_results.csv").open() as f:
        em = [r for r in csv.DictReader(f) if r["algorithm"] == "e_means"]
    assert em and all(float(r["unused_threshold"]) == 0.0 and r["batch_law"] == "quality"
                      for r in em)


def test_size_overrides(official_source, tmp_path):
    out, rep = tmp_path / "raw", tmp_path / "rep"
    subprocess.run(
        [sys.executable, "benchmark.py", "--profile", "smoke", "--threads", "1",
         "--datasets", "gaussian", "--seeds", "0", "--ntrain", "1200", "--ntest", "300",
         "--dim", "8", "--k", "12", "--kmeans-niters", "25", "--emeans-epochs", "1",
         "--kmeans-inits", "random", "--emeans-inits", "random", "--quality-only",
         "--out", str(out), "--report-dir", str(rep), "--run-name", "pytest-overrides"],
        cwd=ROOT, check=True, capture_output=True, text=True)
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["settings"] == dict(ntrain=1200, ntest=300, dim=8, k=12)
    assert meta["settings_overrides"] == dict(ntrain=1200, ntest=300, dim=8, k=12)
    assert "プロファイルの値を上書き" in (rep / "report.md").read_text()


def test_real_input_labels_dataset_by_file_stem(official_source, tmp_path):
    import hashlib
    import numpy as np
    src = tmp_path / "toy_real.npy"
    np.save(src, np.random.default_rng(0).standard_normal((1500, 8)).astype(np.float32))
    out, rep = tmp_path / "raw", tmp_path / "rep"
    subprocess.run(
        [sys.executable, "benchmark.py", "--profile", "smoke", "--threads", "1",
         "--input", str(src), "--ntrain", "1000", "--ntest", "300", "--k", "8",
         "--seeds", "0", "--kmeans-niters", "25", "--emeans-epochs", "1",
         "--kmeans-inits", "random", "--emeans-inits", "random", "--quality-only",
         "--out", str(out), "--report-dir", str(rep), "--run-name", "pytest-real"],
        cwd=ROOT, check=True, capture_output=True, text=True)
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["datasets"] == ["toy_real"] and meta["settings"]["dim"] == 8
    assert meta["input_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    with (out / "raw_results.csv").open() as f:
        assert {r["dataset"] for r in csv.DictReader(f)} == {"toy_real"}


def test_dim_override_rejected_with_input(tmp_path):
    import pytest
    from benchmark import parse_args
    with pytest.raises(SystemExit):
        parse_args(["--input", str(tmp_path / "x.npy"), "--dim", "8"])


def test_scaling_table_from_committed_report(tmp_path):
    report = ROOT / "reports" / "speed_20261111-1905_medium_t4"
    out = tmp_path / "overview.md"
    subprocess.run([sys.executable, "make_scaling_table.py", str(report), str(report),
                    "--out", str(out)], cwd=ROOT, check=True, capture_output=True, text=True)
    text = out.read_text()
    assert text.count("| n=200,000 k=1,024 d=64 |") == 6  # 2 rows x 3 tables
    # values copied from comparison.csv: e-means vs Lloyd25 on gaussian, 1 epoch, 3.13x
    assert "1 ep / 3.13× / 空 0（5/5）" in text
    assert "**NO**（1.24、0/5）" in text  # e-means++ vs Lloyd++25 on imbalanced
