import json

import pytest

import make_tolerance_table as mtt
from emv import analysis

RAW_KEYS = ["dataset", "seed", "repeat", "algorithm", "passes", "run_index", "block_position",
            "seconds", "cpu_seconds", "main_thread_cpu_seconds", "loadavg_1m",
            "train_mse", "test_mse", "empty_clusters"]


def row(seed, alg, passes, seconds, mse, main_cpu=None):
    return dict(dataset="d", seed=seed, repeat=0, algorithm=alg, passes=passes, run_index=0,
                block_position=0, seconds=seconds, cpu_seconds=seconds,
                main_thread_cpu_seconds=seconds if main_cpu is None else main_cpu,
                loadavg_1m=0.1, train_mse=mse, test_mse=mse, empty_clusters=0)


def rows_fixture():
    # Lloyd5 is 0.6% above Lloyd25, Lloyd10 0.05% above; e-means 1 ep 0.3% above,
    # 3 ep below the baseline.
    out = []
    for seed in (0, 1, 2):
        out += [row(seed, "faiss_kmeans", 25, 10.0, 1.000),
                row(seed, "faiss_kmeans", 10, 4.0, 1.0005),
                row(seed, "faiss_kmeans", 5, 2.0, 1.006),
                row(seed, "e_means", 1, 1.0, 1.003),
                row(seed, "e_means", 3, 2.0, 0.999)]
    return out


def make_run(tmp_path, rows, *, quality_only=False, committed_tol=0.01, tamper=False):
    name = "run_x"
    results, reports = tmp_path / "results" / name, tmp_path / "reports" / name
    results.mkdir(parents=True)
    reports.mkdir(parents=True)
    analysis.write_csv(results / "raw_results.csv", rows, RAW_KEYS)
    parsed = analysis.read_raw_csv(results / "raw_results.csv")
    comps = analysis.compare(parsed, analysis.summarize(parsed), 25, committed_tol)
    if tamper:
        comps[0]["emeans_min_passes_reached"] = 99
    analysis.write_csv(reports / "comparison.csv", comps)
    meta = dict(run_name=name, baseline_niter=25, tolerance=0.01, quality_only=quality_only)
    (reports / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return reports


def test_cells_change_with_tolerance(tmp_path):
    run = mtt.load_run(make_run(tmp_path, rows_fixture()))
    assert run["show_speed"] is True
    text = mtt.render([run], [0.01, 0.005, 0.001, 0.0])
    line = next(l for l in text.splitlines() if l.startswith("| `run_x` | d |"))
    cells = [c.strip() for c in line.strip("|").split("|")][2:]
    # 1%: e-means 1 ep (1.0 s) vs Lloyd5 (2.0 s); 0.5%: Lloyd10 needed (4.0 s);
    # 0.1%: e-means 1 ep (0.3% above) fails -> 3 ep (2.0 s) vs Lloyd10; 0%: Lloyd25 only.
    assert cells == ["1 ep / L5 / 2× (3/3)", "1 ep / L10 / 4× (3/3)",
                     "3 ep / L10 / 2× (3/3)", "3 ep / L25 / 5× (3/3)"]


def test_quality_only_runs_hide_speed(tmp_path):
    run = mtt.load_run(make_run(tmp_path, rows_fixture(), quality_only=True))
    text = mtt.render([run], [0.01])
    assert "×" not in text.split("### A.")[1].split("倍率を表示しなかった")[0]
    assert "品質のみ" in text


def test_contended_runs_hide_speed(tmp_path):
    rows = rows_fixture()
    rows[0]["main_thread_cpu_seconds"] = 5.0  # 50% of wall time -> contention
    run = mtt.load_run(make_run(tmp_path, rows))
    assert run["show_speed"] is False
    assert "取り合い検出 1 回" in mtt.render([run], [0.01])


def test_not_reached_cell(tmp_path):
    rows = [row(s, "faiss_kmeans", 25, 10.0, 1.0) for s in (0, 1)]
    rows += [row(s, "e_means", 1, 1.0, 1.05) for s in (0, 1)]
    run = mtt.load_run(make_run(tmp_path, rows))
    assert "**NO** (1.05) / L25 (0/2)" in mtt.render([run], [0.01])


def test_reproduction_mismatch_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="emeans_min_passes_reached"):
        mtt.load_run(make_run(tmp_path, rows_fixture(), tamper=True))


def test_missing_raw_data_is_an_error(tmp_path):
    reports = make_run(tmp_path, rows_fixture())
    (tmp_path / "results" / "run_x" / "raw_results.csv").unlink()
    with pytest.raises(FileNotFoundError):
        mtt.load_run(reports)
