"""The official NumpyEmeans must come from the pinned checkout, not the wheel."""
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ROOT


def _run(code: str) -> str:
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout


def test_loads_numpy_emeans_from_pinned_source(official_source):
    out = _run(
        "import faiss, sys\n"
        "from emv import official\n"
        "cls = official.load_official_emeans(faiss)\n"
        "print(cls.__name__)\n"
        "print(sys.modules[cls.__module__].__file__)\n"
        "info = official.source_info(faiss)\n"
        "official.check_pinned(info)\n"
        "print(info['faiss_source_sha'])\n")
    name, location, sha = out.split()
    assert name == "NumpyEmeans"
    assert Path(location).resolve().is_relative_to(official_source.resolve() / "contrib" / "e_means")
    assert sha == (ROOT / "FAISS_COMMIT").read_text().strip()


def test_rejects_previously_imported_bundled_copy(official_source):
    code = (
        "import faiss\n"
        "import faiss.contrib.e_means  # wheel copy, imported first\n"
        "from emv import official\n"
        "try:\n"
        "    official.load_official_emeans(faiss)\n"
        "except RuntimeError as e:\n"
        "    print('rejected')\n")
    try:
        out = _run(code)
    except subprocess.CalledProcessError as exc:  # wheel without e_means
        pytest.skip(f"faiss wheel has no bundled e_means: {exc.stderr[-200:]}")
    assert "rejected" in out


def test_missing_source_raises(tmp_path):
    import faiss

    from emv import official
    with pytest.raises(FileNotFoundError):
        official.load_official_emeans(faiss, tmp_path)


def test_official_api_used_by_benchmark(official_source):
    """Guard against upstream API changes (constructor args, recipe, n_splits, init)."""
    out = _run(
        "import faiss, numpy as np\n"
        "from emv import official, runners, data\n"
        "E = official.load_official_emeans(faiss)\n"
        "tr, te = data.synthetic_data('separated', 2000, 200, 8, 8)\n"
        "c, sec, info = runners.run_emeans(E, tr, 8, seed=0, nepochs=2)\n"
        "c2, _, _ = runners.run_emeans(E, tr, 8, seed=0, nepochs=2, init=data.shared_init(tr, 8, 0))\n"
        "assert c.shape == (8, 8) and sec > 0\n"
        "assert isinstance(info['batch_size'], int) and info['splits'] >= 0\n"
        "assert np.array_equal(c, c2), 'shared init should equal the native init'\n"
        "k, _, _ = runners.run_faiss_kmeans(faiss, tr, 8, seed=0, niter=3, init=data.shared_init(tr, 8, 0))\n"
        "assert k.shape == (8, 8)\n"
        "p, _, _ = runners.run_faiss_kmeans(faiss, tr, 8, seed=0, niter=3, init_method='kmeanspp')\n"
        "r, _, _ = runners.run_faiss_kmeans(faiss, tr, 8, seed=0, niter=3)\n"
        "assert p.shape == (8, 8) and not np.array_equal(p, r), 'k-means++ must change the seeding'\n"
        "# e_means_pp starts from exactly the centroids Faiss k-means++ uses\n"
        "q, _, _ = runners.run_faiss_kmeans(faiss, tr, 8, seed=5, niter=3, init=runners.kmeanspp_init(faiss, tr, 8, 5))\n"
        "p5, _, _ = runners.run_faiss_kmeans(faiss, tr, 8, seed=5, niter=3, init_method='kmeanspp')\n"
        "assert np.array_equal(q, p5), 'kmeanspp_init must reproduce Faiss k-means++ seeding'\n"
        "e, sec, _ = runners.run_emeans(E, tr, 8, seed=5, nepochs=2, init_method='kmeanspp', faiss=faiss)\n"
        "e2, _, _ = runners.run_emeans(E, tr, 8, seed=5, nepochs=2, init=runners.kmeanspp_init(faiss, tr, 8, 5))\n"
        "assert np.array_equal(e, e2) and sec > 0\n"
        "print('ok')\n")
    assert out.strip() == "ok"
