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
