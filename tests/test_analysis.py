import pytest

from emv import analysis


def row(dataset, seed, alg, passes, seconds, mse):
    return dict(dataset=dataset, seed=seed, repeat=0, algorithm=alg, passes=passes,
                seconds=seconds, train_mse=mse, test_mse=mse, empty_clusters=0)


def test_quality_matched_speedup_reached():
    rows = []
    for seed in (0, 1, 2):
        rows += [row("d", seed, "faiss_kmeans", 25, 10.0, 1.00),
                 row("d", seed, "faiss_kmeans", 5, 2.0, 1.05),
                 row("d", seed, "e_means", 2, 1.0, 1.02),   # 2% worse: not reached
                 row("d", seed, "e_means", 3, 2.5, 1.005)]  # within 1%
    s = analysis.summarize(rows)
    (c,) = analysis.compare(rows, s)
    assert c["emeans_reached"] is True
    assert c["emeans_best_passes"] == 3
    assert c["speedup_vs_baseline"] == pytest.approx(4.0)
    assert c["kmeans_fastest_niter_within_tol"] == 25
    assert c["per_seed_reached"] == "3/3"
    assert c["per_seed_speedup_median"] == pytest.approx(4.0)


def test_not_reached_is_recorded_without_speedup():
    rows = [row("d", 0, "faiss_kmeans", 25, 10.0, 1.0),
            row("d", 0, "e_means", 5, 1.0, 1.2)]
    (c,) = analysis.compare(rows, analysis.summarize(rows))
    assert c["emeans_reached"] is False
    assert c["speedup_vs_baseline"] == ""
    assert c["per_seed_reached"] == "0/1"
    assert c["emeans_best_mse_ratio"] == pytest.approx(1.2)


def test_conservative_speedup_uses_fastest_kmeans():
    rows = [row("d", 0, "faiss_kmeans", 25, 10.0, 1.0),
            row("d", 0, "faiss_kmeans", 10, 4.0, 1.005),
            row("d", 0, "e_means", 2, 2.0, 0.99)]
    (c,) = analysis.compare(rows, analysis.summarize(rows))
    assert c["speedup_vs_baseline"] == pytest.approx(5.0)
    assert c["kmeans_fastest_niter_within_tol"] == 10
    assert c["speedup_vs_fastest_kmeans"] == pytest.approx(2.0)


def test_per_seed_uses_own_baseline():
    # medians reach, but seed 1's e-means misses its own seed baseline
    rows = [row("d", 0, "faiss_kmeans", 25, 10.0, 1.00), row("d", 0, "e_means", 2, 1.0, 1.00),
            row("d", 1, "faiss_kmeans", 25, 10.0, 0.90), row("d", 1, "e_means", 2, 1.0, 1.00)]
    (c,) = analysis.compare(rows, analysis.summarize(rows))
    assert c["per_seed_reached"] == "1/2"


def test_missing_baseline_raises():
    rows = [row("d", 0, "e_means", 2, 1.0, 1.0)]
    with pytest.raises(ValueError):
        analysis.compare(rows, analysis.summarize(rows))


def test_summary_spread():
    rows = [row("d", s, "e_means", 2, sec, 1.0) for s, sec in enumerate((1.0, 2.0, 3.0))]
    (s,) = analysis.summarize(rows)
    assert (s["median_seconds"], s["min_seconds"], s["max_seconds"], s["runs"]) == (2.0, 1.0, 3.0, 3)
    assert s["cv_seconds"] == pytest.approx(0.5)


def test_kmeanspp_baseline_is_compared_separately():
    rows = [row("d", 0, "faiss_kmeans", 25, 10.0, 2.0),
            row("d", 0, "faiss_kmeans", 5, 2.0, 2.0),
            row("d", 0, "faiss_kmeans_pp", 25, 12.0, 1.0),
            row("d", 0, "faiss_kmeans_pp", 5, 3.0, 1.005),
            row("d", 0, "e_means", 2, 1.0, 1.5)]
    rand, pp = analysis.compare(rows, analysis.summarize(rows))
    assert rand["baseline"] == "faiss_kmeans@25" and rand["emeans_reached"] is True
    assert rand["kmeans_fastest_niter_within_tol"] == 5   # searched within random-init only
    assert pp["baseline"] == "faiss_kmeans_pp@25" and pp["emeans_reached"] is False
    assert pp["kmeans_fastest_niter_within_tol"] == 5
    assert pp["emeans_best_mse_ratio"] == pytest.approx(1.5)


def test_each_emeans_variant_gets_its_own_row():
    rows = [row("d", 0, "faiss_kmeans_pp", 25, 10.0, 1.0),
            row("d", 0, "faiss_kmeans", 25, 10.0, 2.0),
            row("d", 0, "e_means", 2, 1.0, 1.5),
            row("d", 0, "e_means_pp", 2, 2.0, 0.99)]
    out = analysis.compare(rows, analysis.summarize(rows))
    got = {(c["baseline"], c["candidate"]): c for c in out}
    assert set(got) == {(b, c) for b in ("faiss_kmeans@25", "faiss_kmeans_pp@25")
                        for c in ("e_means", "e_means_pp")}
    assert got[("faiss_kmeans_pp@25", "e_means")]["emeans_reached"] is False
    pp = got[("faiss_kmeans_pp@25", "e_means_pp")]
    assert pp["emeans_reached"] is True and pp["speedup_vs_baseline"] == pytest.approx(5.0)


def test_timer_separates_waiting_from_work():
    import time

    from emv.runners import Timer
    with Timer() as t:
        time.sleep(0.05)          # waiting: wall time without CPU time
    assert t.seconds >= 0.05 and t.cpu_seconds < 0.03
    # Busy for 0.05 s of main-thread CPU time (not wall time), so the assertions hold
    # even when other processes compete for the CPU and stretch the wall time.
    with Timer() as t:
        x = 0
        while time.thread_time() - t._thread < 0.05:
            x += 1
    assert t.main_thread_cpu_seconds >= 0.05
    assert t.seconds >= 0.99 * t.main_thread_cpu_seconds  # wall time >= own CPU time


def test_timing_diagnostics_and_old_csv_compat():
    rows = [dict(row("d", 0, "e_means", 1, 2.0, 1.0), cpu_seconds=1.0,
                 main_thread_cpu_seconds=0.5, loadavg_1m=3.0),
            dict(row("d", 0, "faiss_kmeans", 25, 1.0, 1.0), cpu_seconds=None,
                 main_thread_cpu_seconds=None, loadavg_1m=None)]   # pre-diagnostics row
    (d,) = analysis.timing_diagnostics(rows)
    assert d["algorithm"] == "e_means"
    assert d["median_process_cpu_per_wall"] == 0.5 and d["median_main_thread_cpu_per_wall"] == 0.25
    s = {x["algorithm"]: x for x in analysis.summarize(rows)}
    assert s["faiss_kmeans"]["median_cpu_seconds"] == "" and s["e_means"]["median_cpu_seconds"] == 1.0


def test_contention_flag():
    base = dict(row("d", 0, "e_means", 1, 1.0, 1.0), cpu_seconds=4.0, loadavg_1m=1.0)
    assert analysis.is_contended(dict(base, main_thread_cpu_seconds=0.99)) is False
    assert analysis.is_contended(dict(base, main_thread_cpu_seconds=0.80)) is True
    assert analysis.is_contended(dict(base, main_thread_cpu_seconds=None)) is None
    (d,) = analysis.timing_diagnostics([dict(base, main_thread_cpu_seconds=0.8),
                                        dict(base, main_thread_cpu_seconds=1.0)])
    assert d["contended_runs"] == 1
    md = {"quality_only": False}
    assert "CPU の取り合いを検出" in "".join(analysis._banners(md, [d]))
    assert analysis._banners(md, [dict(d, contended_runs=0)]) == []


def test_min_passes_reached_is_time_independent():
    rows = [row("d", 0, "faiss_kmeans", 25, 10.0, 1.0),
            row("d", 0, "e_means", 2, 5.0, 1.0),     # reaches, slower
            row("d", 0, "e_means", 3, 1.0, 1.0)]     # reaches, faster
    (c,) = analysis.compare(rows, analysis.summarize(rows))
    assert c["emeans_min_passes_reached"] == 2 and c["emeans_best_passes"] == 3


def test_fractional_epochs_are_not_truncated(tmp_path):
    rows = [row("d", 0, "faiss_kmeans", 25, 10.0, 1.0),
            row("d", 0, "e_means", 1.5, 1.0, 1.2),    # not reached
            row("d", 0, "e_means", 1.9, 2.0, 1.0)]    # reached
    s = analysis.summarize(rows)
    assert sorted(x["passes"] for x in s if x["algorithm"] == "e_means") == [1.5, 1.9]
    (c,) = analysis.compare(rows, s)
    assert c["emeans_min_passes_reached"] == 1.9 and c["per_seed_reached"] == "1/1"
    assert c["per_seed_speedup_median"] == pytest.approx(5.0)
    # round trip through the raw CSV keeps 1.5 and integral values as int
    full = [dict(r, init="native", run_index=i, block_position=i, cpu_seconds="",
                 main_thread_cpu_seconds="", loadavg_1m="", batch_size="", alpha0="",
                 batch_law="", splits="", unused_threshold="") for i, r in enumerate(rows)]
    path = tmp_path / "raw.csv"
    analysis.write_csv(path, full, analysis.RAW_KEYS)
    back = analysis.read_raw_csv(path)
    assert [r["passes"] for r in back] == [25, 1.5, 1.9]
    assert isinstance(back[0]["passes"], int)
