import faiss
import numpy as np
import pytest

from emv import data, runners


@pytest.mark.parametrize("name", data.DATASETS)
def test_synthetic_shapes_and_determinism(name):
    tr, te = data.synthetic_data(name, ntrain=500, ntest=100, dim=8, k=10)
    assert tr.shape == (500, 8) and te.shape == (100, 8)
    assert tr.dtype == np.float32 and tr.flags.c_contiguous
    tr2, te2 = data.synthetic_data(name, ntrain=500, ntest=100, dim=8, k=10)
    np.testing.assert_array_equal(tr, tr2)
    np.testing.assert_array_equal(te, te2)


def test_train_test_are_distinct_draws():
    tr, te = data.synthetic_data("separated", ntrain=500, ntest=100, dim=8, k=10)
    # no test row is a copy of a train row
    d, _ = faiss.knn(te, tr, 1)
    assert d.min() > 0


def test_imbalanced_is_skewed_and_separated_is_balanced():
    k = 20
    rng_counts = {}
    for name in ("separated", "imbalanced"):
        tr, _ = data.synthetic_data(name, ntrain=20_000, ntest=10, dim=16, k=k)
        # recover generating centers with the same RNG stream
        rng = np.random.default_rng(data.DATA_SEED)
        centers = rng.normal(0, 4.0, (k, 16)).astype(np.float32)
        _, lab = faiss.knn(tr, centers, 1)
        rng_counts[name] = np.bincount(lab.ravel(), minlength=k)
    sep, imb = rng_counts["separated"], rng_counts["imbalanced"]
    assert sep.max() / sep.min() < 1.5
    assert imb.max() / imb.min() > 20


def test_shared_init_matches_numpy_emeans_default():
    tr, _ = data.synthetic_data("gaussian", 300, 10, 4, 5)
    init = data.shared_init(tr, 5, seed=3)
    expected = tr[np.random.RandomState(3).permutation(300)[:5]]
    np.testing.assert_array_equal(init, expected)


def test_score_exact_mse_and_empty_clusters():
    x = np.array([[0, 0], [0, 2], [10, 0]], dtype=np.float32)
    centers = np.array([[0, 1], [10, 0], [100, 100]], dtype=np.float32)
    mse, empty = runners.score(faiss, x, centers, batch=2)
    assert mse == pytest.approx((1 + 1 + 0) / 3)
    assert empty == 1


def test_validate_centers_rejects_bad_output():
    with pytest.raises(ValueError):
        runners.validate_centers(np.full((3, 2), np.nan), 3, 2)
    with pytest.raises(ValueError):
        runners.validate_centers(np.zeros((2, 2)), 3, 2)


def test_real_embeddings_split(tmp_path):
    x = np.random.default_rng(0).standard_normal((100, 4)).astype(np.float32)
    path = tmp_path / "x.npy"
    np.save(path, x)
    tr, te = data.real_embeddings(path, 60, 30)
    assert tr.shape == (60, 4) and te.shape == (30, 4)
    with pytest.raises(ValueError):
        data.real_embeddings(path, 90, 30)


def test_imbalance_exponent_default_matches_kit_and_zero_is_balanced():
    a, _ = data.synthetic_data("imbalanced", 500, 10, 8, 10)
    b, _ = data.synthetic_data("imbalanced", 500, 10, 8, 10, imbalance_exponent=1.3)
    np.testing.assert_array_equal(a, b)
    c, _ = data.synthetic_data("imbalanced", 500, 10, 8, 10, imbalance_exponent=0.0)
    d, _ = data.synthetic_data("separated", 500, 10, 8, 10)
    np.testing.assert_array_equal(c, d)
