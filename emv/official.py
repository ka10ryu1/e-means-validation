"""Load the official e-means implementation from a pinned Faiss checkout.

The loader is taken from the validation kit: it prepends the checkout's
``contrib`` directory to ``faiss.contrib.__path__`` so that
``faiss.contrib.e_means`` resolves to the checkout, without touching
site-packages. It then verifies where the module was actually loaded from.
"""
from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIN_FILE = ROOT / "FAISS_COMMIT"
DEFAULT_SOURCE = ROOT / "faiss-src"
MODULE = "faiss.contrib.e_means"


def pinned_commit() -> str:
    return PIN_FILE.read_text(encoding="utf-8").strip()


def _package_dir(repo: Path) -> Path:
    return repo.expanduser().resolve() / "contrib" / "e_means"


def load_official_emeans(faiss, repo: Path = DEFAULT_SOURCE):
    """Return NumpyEmeans imported from ``repo/contrib/e_means``.

    Raises if the source is missing or if Python resolved the module to any
    other location (e.g. the copy bundled in the faiss-cpu wheel).
    """
    package_dir = _package_dir(repo)
    if not (package_dir / "__init__.py").is_file():
        raise FileNotFoundError(
            f"Official source not found: {package_dir}; run scripts/setup_faiss.sh")
    existing = sys.modules.get(MODULE)
    if existing is not None and package_dir not in Path(existing.__file__).resolve().parents:
        raise RuntimeError(
            f"{MODULE} was already imported from {existing.__file__}; "
            "load the official source before anything else imports e-means")
    import faiss.contrib
    contrib_path = str(package_dir.parent)
    if contrib_path not in faiss.contrib.__path__:
        faiss.contrib.__path__.insert(0, contrib_path)
    importlib.invalidate_caches()
    from faiss.contrib.e_means import NumpyEmeans

    for name in (MODULE, NumpyEmeans.__module__, f"{MODULE}.base"):
        location = Path(sys.modules[name].__file__).resolve()
        if package_dir not in location.parents:
            raise RuntimeError(f"Loaded {name} from {location}, expected source {package_dir}")
    return NumpyEmeans


def tree_digest(package_dir: Path) -> str:
    """sha256 over the sorted (relative path, content) of all .py files."""
    h = hashlib.sha256()
    for path in sorted(package_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        h.update(path.relative_to(package_dir).as_posix().encode())
        h.update(b"\0")
        h.update(path.read_bytes())
    return h.hexdigest()


def _git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_info(faiss, repo: Path = DEFAULT_SOURCE) -> dict:
    """Provenance of the e-means code under test (recorded in metadata.json)."""
    package_dir = _package_dir(repo)
    sha = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain", "--", "contrib/e_means")
    bundled_dir = Path(faiss.__file__).resolve().parent / "contrib" / "e_means"
    digest = tree_digest(package_dir)
    bundled = tree_digest(bundled_dir) if bundled_dir.is_dir() else None
    return {
        "faiss_source_path": str(package_dir),
        "faiss_source_sha": sha or "unknown",
        "faiss_source_pinned_sha": pinned_commit(),
        "faiss_source_matches_pin": sha == pinned_commit(),
        "faiss_source_e_means_dirty": bool(status) if status is not None else None,
        "e_means_py_sha256": digest,
        "bundled_wheel_e_means_present": bundled_dir.is_dir(),
        "bundled_wheel_e_means_identical": (bundled == digest) if bundled else None,
    }


def check_pinned(info: dict) -> None:
    if not info["faiss_source_matches_pin"]:
        raise RuntimeError(
            f"faiss-src is at {info['faiss_source_sha']}, pinned {info['faiss_source_pinned_sha']}; "
            "run scripts/setup_faiss.sh or pass --allow-unpinned")
    if info["faiss_source_e_means_dirty"]:
        raise RuntimeError("faiss-src/contrib/e_means has local modifications; "
                           "pass --allow-unpinned to run anyway")
