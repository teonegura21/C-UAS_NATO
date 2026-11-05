"""Deterministic K-means (numpy only): k-means++ seeding, Lloyd iterations, restarts,
robust standardisation and a sampled silhouette score for choosing K.

Used by the pre-classification step; kept dependency-free because the environment has
numpy but no scikit-learn. Same input + same seed -> same labels, bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Scaler:
    center: np.ndarray
    scale: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.center) / self.scale

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return z * self.scale + self.center


def robust_scaler(x: np.ndarray) -> Scaler:
    """Median / IQR scaling, so a few extreme clutter tracks do not set the unit of a
    feature. Falls back to the standard deviation (then 1) for degenerate columns."""
    center = np.median(x, axis=0)
    q75, q25 = np.percentile(x, [75, 25], axis=0)
    scale = q75 - q25
    std = x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, np.where(std > 1e-12, std, 1.0))
    return Scaler(center, scale)


@dataclass
class KMeansResult:
    labels: np.ndarray      # (n,) int
    centers: np.ndarray     # (k, d) in the space the data was given in
    inertia: float          # sum of squared distances to the assigned center
    n_iter: int


def _sq_dist(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    # |x|^2 - 2 x.c + |c|^2, clipped: cancellation can make it slightly negative.
    d = (x * x).sum(1)[:, None] - 2.0 * x @ c.T + (c * c).sum(1)[None, :]
    return np.maximum(d, 0.0)


def _kmeans_pp(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    centers = np.empty((k, x.shape[1]))
    centers[0] = x[rng.integers(n)]
    closest = _sq_dist(x, centers[:1])[:, 0]
    for j in range(1, k):
        total = closest.sum()
        idx = rng.integers(n) if total <= 0 else rng.choice(n, p=closest / total)
        centers[j] = x[idx]
        closest = np.minimum(closest, _sq_dist(x, centers[j:j + 1])[:, 0])
    return centers


def _lloyd(x, centers, max_iter, tol):
    k = len(centers)
    for it in range(1, max_iter + 1):
        d = _sq_dist(x, centers)
        labels = d.argmin(1)
        new = np.empty_like(centers)
        for j in range(k):
            members = x[labels == j]
            if len(members):
                new[j] = members.mean(0)
            else:
                # Empty cluster: move it to the point worst served by its center.
                far = d[np.arange(len(x)), labels].argmax()
                new[j] = x[far]
                labels[far] = j
        shift = np.sqrt(((new - centers) ** 2).sum(1)).max()
        centers = new
        if shift <= tol:
            break
    d = _sq_dist(x, centers)
    labels = d.argmin(1)
    inertia = float(d[np.arange(len(x)), labels].sum())
    return labels, centers, inertia, it


def kmeans(x: np.ndarray, k: int, n_init: int = 10, max_iter: int = 300,
           tol: float = 1e-7, seed: int = 0) -> KMeansResult:
    """Best of `n_init` k-means++ restarts. Labels are renumbered by descending cluster
    size (ties: by first member index) so cluster ids are stable across runs."""
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("kmeans input contains NaN/inf; impute or drop first")
    if not 1 <= k <= len(x):
        raise ValueError(f"k={k} invalid for {len(x)} points")
    best = None
    for run in range(n_init):
        rng = np.random.default_rng([seed, run])
        res = _lloyd(x, _kmeans_pp(x, k, rng), max_iter, tol)
        if best is None or res[2] < best[2]:
            best = res
    labels, centers, inertia, n_iter = best
    sizes = np.bincount(labels, minlength=k)
    first = np.array([np.flatnonzero(labels == j)[0] if sizes[j] else len(x) for j in range(k)])
    order = sorted(range(k), key=lambda j: (-sizes[j], first[j]))
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    return KMeansResult(remap[labels], centers[order], inertia, n_iter)


def silhouette(x: np.ndarray, labels: np.ndarray, sample: int = 4000, seed: int = 0,
               chunk: int = 1000) -> float:
    """Mean silhouette over a fixed random sample (exact when n <= sample).

    a(i) = mean distance to the other members of its own cluster, b(i) = smallest mean
    distance to another cluster, s(i) = (b - a) / max(a, b); singletons score 0.
    """
    x = np.asarray(x, dtype=float)
    labels = np.asarray(labels)
    n = len(x)
    if n > sample:
        idx = np.sort(np.random.default_rng(seed).choice(n, sample, replace=False))
        x, labels = x[idx], labels[idx]
    ks = np.unique(labels)
    if len(ks) < 2:
        return float("nan")
    onehot = (labels[:, None] == ks[None, :]).astype(float)
    counts = onehot.sum(0)
    scores = np.empty(len(x))
    for s in range(0, len(x), chunk):
        d = np.sqrt(_sq_dist(x[s:s + chunk], x))
        sums = d @ onehot                          # (chunk, k) summed distance per cluster
        own = np.searchsorted(ks, labels[s:s + chunk])
        rows = np.arange(len(own))
        own_count = counts[own] - 1                # exclude the point itself (d=0)
        a = np.where(own_count > 0, sums[rows, own] / np.maximum(own_count, 1), 0.0)
        mean_other = sums / counts[None, :]
        mean_other[rows, own] = np.inf
        b = mean_other.min(1)
        sc = (b - a) / np.maximum(np.maximum(a, b), 1e-300)
        scores[s:s + chunk] = np.where(own_count > 0, sc, 0.0)
    return float(scores.mean())


def scan_k(x: np.ndarray, ks, n_init: int = 10, seed: int = 0, sample: int = 4000):
    """Fit every K in `ks`; return [(k, inertia, silhouette, KMeansResult)]."""
    out = []
    for k in ks:
        res = kmeans(x, k, n_init=n_init, seed=seed)
        out.append((k, res.inertia, silhouette(x, res.labels, sample=sample, seed=seed), res))
    return out
