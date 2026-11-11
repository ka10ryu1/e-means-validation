#!/usr/bin/env bash
# Fetch the official Faiss source at the commit pinned in FAISS_COMMIT.
# Only that single commit is downloaded (shallow fetch); nothing is installed
# into site-packages. benchmark.py loads faiss-src/contrib/e_means directly.
#
#   bash scripts/setup_faiss.sh [target_dir]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHA="$(tr -d '[:space:]' < "$ROOT/FAISS_COMMIT")"
DEST="${1:-$ROOT/faiss-src}"
REMOTE="https://github.com/facebookresearch/faiss.git"

if [[ ! "$SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "FAISS_COMMIT must contain a full 40-character SHA, got '$SHA'" >&2
  exit 1
fi

if [[ ! -d "$DEST/.git" ]]; then
  git init -q "$DEST"
  git -C "$DEST" remote add origin "$REMOTE"
fi

if ! git -C "$DEST" cat-file -e "$SHA^{commit}" 2>/dev/null; then
  git -C "$DEST" fetch -q --depth 1 origin "$SHA"
fi
if [[ -n "$(git -C "$DEST" status --porcelain -- contrib/e_means 2>/dev/null)" ]]; then
  echo "Local modifications in $DEST/contrib/e_means; refusing to overwrite." >&2
  exit 1
fi
git -C "$DEST" -c advice.detachedHead=false checkout -q "$SHA"

HEAD="$(git -C "$DEST" rev-parse HEAD)"
if [[ "$HEAD" != "$SHA" ]]; then
  echo "Checkout failed: HEAD=$HEAD expected $SHA" >&2
  exit 1
fi
test -f "$DEST/contrib/e_means/__init__.py" || { echo "contrib/e_means missing at $SHA" >&2; exit 1; }
echo "faiss source ready: $DEST @ $HEAD"
