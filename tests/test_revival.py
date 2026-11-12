"""The revival recorder must observe training without changing it."""
import numpy as np

from conftest import ROOT  # noqa: F401  (puts the repo on sys.path)
from emv import data


def test_synthetic_truth_matches_synthetic_data():
    for name in ("separated", "imbalanced"):
        tr, te = data.synthetic_data(name, 500, 100, 8, 10)
        tr2, te2, truth = data.synthetic_truth(name, 500, 100, 8, 10)
        assert np.array_equal(tr, tr2) and np.array_equal(te, te2)
        assert truth["centers"].shape == (10, 8) and truth["train_group"].shape == (500,)
        assert np.isclose(truth["prob"].sum(), 1.0)
        # each row is its blob center plus unit noise
        noise = tr - truth["centers"][truth["train_group"]]
        assert 0.8 < noise.std() < 1.2


def test_recorder_is_bit_identical_and_logs_every_split(official_source):
    import faiss

    from emv import official, revival, runners
    NumpyEmeans = official.load_official_emeans(faiss)
    tr, _, truth = data.synthetic_truth("imbalanced", 4000, 10, 16, 32)
    init = runners.kmeanspp_init(faiss, tr, 32, seed=0)
    Recorder = revival.make_recorder(NumpyEmeans, faiss, truth["centers"], truth["prob"])

    plain = NumpyEmeans(16, 32, seed=0, nepochs=3, batch_size=256, unused_threshold=0.5)
    rec = Recorder(16, 32, seed=0, nepochs=3, batch_size=256, unused_threshold=0.5)
    c_plain, _ = plain.train(tr, init_centroids=init)
    c_rec, _ = rec.train(tr, init_centroids=init)

    assert np.array_equal(c_plain, c_rec)
    assert plain.n_splits == rec.n_splits > 0
    assert len(rec.events) == rec.n_splits
    assert sum(c["nsplit"] for c in rec.checks) == rec.n_splits
    assert len(rec.checks) == rec.recipe["total_steps"] // 5
    s = revival.event_summary(rec.events, rec.checks, 0.5)
    assert s["events"] == rec.n_splits and 0 <= s["target_on_shared_blob"] <= 1
    assert all(e["target_mass_ratio"] < 0.5 <= e["source_mass_ratio"] for e in rec.events)


def test_coverage_counts_blobs():
    import faiss

    from emv import revival
    true = np.array([[0, 0], [10, 0], [0, 10]], dtype=np.float32)
    prob = np.array([0.5, 0.3, 0.2])
    centers = np.array([[0.1, 0], [-0.1, 0], [10, 0.2]], dtype=np.float32)
    cov = revival.coverage(faiss, centers, true, prob)
    assert cov == dict(uncovered_blobs=1, uncovered_mass=0.2, crowded_blobs=1, spare_centroids=1)


def test_warmup_skips_checks_without_drawing(official_source):
    """Skipping every check equals no revival; skipping some delays the first split."""
    import faiss

    from emv import official, revival, runners
    NumpyEmeans = official.load_official_emeans(faiss)
    tr, _, truth = data.synthetic_truth("imbalanced", 4000, 10, 16, 32)
    init = runners.kmeanspp_init(faiss, tr, 32, seed=0)
    kw = dict(seed=0, nepochs=3, batch_size=256)

    def run(warmup, thr):
        cls = revival.make_recorder(NumpyEmeans, faiss, truth["centers"], truth["prob"],
                                    warmup_checks=warmup)
        model = cls(16, 32, unused_threshold=thr, **kw)
        centers, _ = model.train(tr, init_centroids=init)
        return model, centers

    off, c_off = run(0, 0.0)
    total = off.recipe["total_steps"] // 5
    all_skipped, c_all = run(total, 0.5)
    assert off.n_splits == all_skipped.n_splits == 0
    assert np.array_equal(c_off, c_all)
    assert all(c["skipped"] for c in all_skipped.checks)

    full, _ = run(0, 0.5)
    late, _ = run(3, 0.5)
    assert full.n_splits > 0 and not any(c["skipped"] for c in full.checks)
    assert [c["skipped"] for c in late.checks[:4]] == [True, True, True, False]
    assert all(c["nsplit"] == 0 for c in late.checks[:3])
    assert all(e["check"] >= 3 for e in late.events)
    assert sum(c["nsplit"] for c in late.checks) == late.n_splits
