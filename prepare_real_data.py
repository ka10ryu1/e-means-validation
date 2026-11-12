#!/usr/bin/env python3
"""Download public ANN datasets and convert them to the .npy matrices used by --input.

    uv run python prepare_real_data.py [sift1m] [deep1m] [--data-dir data]

- sift1m: SIFT1M base set (1,000,000 x 128) from the TEXMEX corpus (INRIA).
  Only sift_base.fvecs is read from the tarball, in memory; nothing is extracted to disk.
- deep1m: the first 1,000,000 rows (x 96) of the DEEP1B base set (Yandex, CC BY 4.0),
  fetched with an HTTP byte range. This is the usual "Deep1M" crop of big-ann-benchmarks.

Vectors are stored as float32 exactly as distributed (no normalization, no PCA).
Downloads go to <data-dir>/raw/, outputs to <data-dir>/<name>.npy (both outside Git).
SHA-256 of the download and of the .npy are checked against the values pinned below;
a mismatch is an error (the source changed, so earlier reports would not be comparable).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

from emv import data

ROOT = Path(__file__).resolve().parent
NROWS = 1_000_000

SOURCES = {
    "sift1m": dict(
        url="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
        raw="sift.tar.gz", size=168_280_445, dim=128,
        license="TEXMEX corpus (INRIA), public research dataset",
        raw_sha256="92f1270c5e3a0cb46b89983e72b0511e4df065c31a9fa0276d8c9b1fca5bc81a",
        npy_sha256="4921953796bc066d3a4be1c6b26e32a27b8931080f21afe30824a63e65cb768d",
    ),
    "deep1m": dict(
        url="https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin",
        raw="deep1b_base_first1M.fbin", size=8 + NROWS * 96 * 4, dim=96,
        license="DEEP1B (Yandex), CC BY 4.0; first 1M rows of base.1B.fbin",
        raw_sha256="1c5169a4eb8dd860727f47ca641e48f06a2253b7a235ecf4cbde7dda3ec893c3",
        npy_sha256="b839dfd137824b6f60a4dab77c0e58b93c0a700ea3780e943c256c048ea95843",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, src: dict, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size == src["size"]:
        print(f"{name}: using existing {dest}")
        return
    req = urllib.request.Request(src["url"])
    if name == "deep1m":
        req.add_header("Range", f"bytes=0-{src['size'] - 1}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"{name}: downloading {src['url']} ({src['size'] / 1e6:.0f} MB)", flush=True)
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 22)
    if tmp.stat().st_size != src["size"]:
        raise RuntimeError(f"{name}: got {tmp.stat().st_size} bytes, expected {src['size']}")
    tmp.replace(dest)


def convert(name: str, src: dict, raw: Path) -> np.ndarray:
    if name == "sift1m":
        with tarfile.open(raw, "r:gz") as tar:
            member = tar.getmember("sift/sift_base.fvecs")
            x = data.parse_fvecs(tar.extractfile(member).read())
    else:
        x = data.parse_fbin_prefix(raw.read_bytes(), NROWS, src["dim"])
    if x.shape != (NROWS, src["dim"]) or not np.isfinite(x).all():
        raise ValueError(f"{name}: unexpected matrix {x.shape} or non-finite values")
    return x


def check(name: str, kind: str, expected: str | None, actual: str) -> None:
    if expected is None:
        print(f"{name}: {kind} sha256 {actual} (not pinned yet)")
    elif expected != actual:
        raise RuntimeError(f"{name}: {kind} sha256 {actual} != pinned {expected}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", metavar="NAME",
                   help=f"datasets to prepare (default: all of {sorted(SOURCES)})")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = p.parse_args(argv)
    args.names = args.names or sorted(SOURCES)
    for name in args.names:
        if name not in SOURCES:
            p.error(f"unknown dataset {name!r}; choose from {sorted(SOURCES)}")
    (args.data_dir / "raw").mkdir(parents=True, exist_ok=True)
    for name in args.names:
        src = SOURCES[name]
        raw = args.data_dir / "raw" / src["raw"]
        download(name, src, raw)
        raw_sha = sha256(raw)
        check(name, "download", src["raw_sha256"], raw_sha)
        out = args.data_dir / f"{name}.npy"
        np.save(out, convert(name, src, raw))
        npy_sha = sha256(out)
        check(name, ".npy", src["npy_sha256"], npy_sha)
        info = {k: src[k] for k in ("url", "license", "dim")}
        info.update(rows=NROWS, raw_sha256=raw_sha, npy_sha256=npy_sha)
        out.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"{name}: wrote {out}")


if __name__ == "__main__":
    main()
