"""e-means independent validation helpers.

Ported from e_means_validation_kit/benchmark.py. Modules:
- official: load the official NumpyEmeans from a pinned Faiss source checkout
- data:     synthetic / user-provided datasets (train/test split)
- runners:  timed training of Faiss k-means and NumpyEmeans, exact scoring
- analysis: summaries, quality-matched speedups, Markdown report
"""
