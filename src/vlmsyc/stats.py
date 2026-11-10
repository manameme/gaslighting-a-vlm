"""AUROC and a bootstrap clustered by image (SPEC section 8).

Rows of the same image are not independent (one image gives one question per
POPE split, and every question has 12 pushback rows), so confidence intervals
resample whole images, not rows.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n, with tied values sharing their average rank."""
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)               # rank of the last copy of each value
    return (upper - (counts - 1) / 2)[inverse]


def auroc(scores, labels) -> float | None:
    """Probability that a random positive scores higher than a random negative (ties count 1/2).

    Computed from average ranks (Mann-Whitney U). Returns None if either class
    is empty. 0.5 = no information; below 0.5 = the signal works the other way round.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    u = average_ranks(s)[y].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def cluster_bootstrap_ci(
    statistic: Callable[[np.ndarray], float | None],
    clusters,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float | None, float | None]:
    """Percentile CI of statistic(row_indices), resampling clusters with replacement.

    clusters[i] is the cluster (image id) of row i. Each resample draws as many
    clusters as there are, with replacement, and keeps all rows of each drawn
    cluster. Resamples where the statistic is undefined (None) are skipped.
    """
    rng = np.random.default_rng(seed)
    clusters = np.asarray(clusters)
    ids, inverse = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(ids))]
    values = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        idx = np.concatenate([members[k] for k in pick])
        v = statistic(idx)
        if v is not None:
            values.append(v)
    if not values:
        return None, None
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
