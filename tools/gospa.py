#!/usr/bin/env python3
"""GOSPA metric library (alpha = 2) and a rectangular min-cost assignment solver.

Scope: pure numpy, no scipy. Nothing here reads competition data; the tracker and the
scoring harness import it. Run `python3 tools/gospa.py` for the numbered self-checks.

Definition (Rahmathullah, Garcia-Fernandez, Svensson 2017), X = truth (n points),
Y = estimates (m points), any dimension D (2D or 3D position; the caller chooses):

    d_p^(c,2)(X,Y) = ( min_gamma  sum_{(i,j) in gamma} d(x_i,y_j)^p
                                  + c^p/2 * (|X| + |Y| - 2|gamma|) )^(1/p)

gamma ranges over partial assignments whose pairs have d < c. Equivalent form used here:
solve a full assignment of min(n,m) pairs on the capped cost min(d,c)^p, then add
c^p/2 * |n-m|. A pair with d >= c costs c^p, i.e. one miss plus one false (c^p/2 each),
so it is reported as unassigned, not as a pair. p >= 1 so the result is a metric.

API
    linear_sum_assignment(cost)   same contract as scipy's, finite n x m, any n, m >= 0
    GospaResult, gospa(x, y, c, p=2.0, alpha=2.0)
    gospa_sequence(step_x, x, step_y, y, n_steps, c, p=2.0)   many time steps at once

Determinism: no randomness; ties resolve by index order (argmin / Hungarian scan order).
"""

from __future__ import annotations

import itertools
import sys
import time
from dataclasses import dataclass

import numpy as np

__all__ = ["linear_sum_assignment", "GospaResult", "gospa", "gospa_sequence"]

_INF = float("inf")
_NP_SWITCH = 96  # columns at which the numpy-vectorised Hungarian beats the list version


# --------------------------------------------------------------------------------------
# Assignment solver: shortest-augmenting-path Hungarian (Jonker-Volgenant style), n <= m
# --------------------------------------------------------------------------------------

def _hungarian_py(a: list[list[float]], n: int, m: int) -> list[int]:
    """Rows n <= cols m. Returns col_of_row (length n). Pure python, fast for small m."""
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [_INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            row = a[i0 - 1]
            ui0 = u[i0]
            delta = _INF
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = row[j - 1] - ui0 - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    col_of_row = [0] * n
    for j in range(1, m + 1):
        if p[j]:
            col_of_row[p[j] - 1] = j - 1
    return col_of_row


def _hungarian_np(a: np.ndarray) -> np.ndarray:
    """Same algorithm with the O(m) inner scans vectorised. a is (n, m), n <= m."""
    n, m = a.shape
    ap = np.zeros((n + 1, m + 1))
    ap[1:, 1:] = a
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int64)
    way = np.zeros(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, _INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = ap[i0] - u[i0] - v
            better = (~used) & (cur < minv)
            minv[better] = cur[better]
            way[better] = j0
            j1 = int(np.argmin(np.where(used, _INF, minv)))
            delta = minv[j1]
            uj = np.flatnonzero(used)
            u[p[uj]] += delta
            v[uj] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    col_of_row = np.zeros(n, dtype=np.int64)
    for j in range(1, m + 1):
        if p[j]:
            col_of_row[p[j] - 1] = j - 1
    return col_of_row


def linear_sum_assignment(cost) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-cost assignment of min(n,m) pairs on a finite n x m matrix.

    Same contract as scipy.optimize.linear_sum_assignment (minimisation): returns
    (row_ind, col_ind) int arrays of length min(n,m), row_ind ascending. Raises
    ValueError on non-2D or non-finite input.
    """
    a = np.asarray(cost, dtype=float)
    if a.ndim != 2:
        raise ValueError("cost must be a 2-D matrix")
    if not np.all(np.isfinite(a)):
        raise ValueError("cost matrix must be finite")
    n, m = a.shape
    if n == 0 or m == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if n == 1:
        return np.zeros(1, dtype=np.int64), np.array([int(np.argmin(a[0]))], dtype=np.int64)
    if m == 1:
        return np.array([int(np.argmin(a[:, 0]))], dtype=np.int64), np.zeros(1, dtype=np.int64)
    transposed = n > m
    if transposed:
        a = a.T
        n, m = m, n
    if m >= _NP_SWITCH:
        cols = _hungarian_np(a)
    else:
        cols = np.array(_hungarian_py(a.tolist(), n, m), dtype=np.int64)
    rows = np.arange(n, dtype=np.int64)
    if transposed:  # a was the transpose: "rows" are the caller's columns
        order = np.argsort(cols, kind="stable")
        return cols[order], rows[order]
    return rows, cols


# --------------------------------------------------------------------------------------
# GOSPA for one pair of point sets
# --------------------------------------------------------------------------------------

@dataclass
class GospaResult:
    distance: float      # GOSPA value, cost ** (1/p)
    cost: float          # loc + c^p/2 * (n_missed + n_false)
    loc: float           # sum of d^p over kept pairs
    n_assigned: int
    n_missed: int
    n_false: int
    pairs: np.ndarray    # (k, 2) int [truth_idx, est_idx], kept pairs (d < c)


def _as_points(a, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return a.reshape(0, a.shape[-1] if a.ndim == 2 else 0)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError(f"{name} must be (n, D)")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    return a


def gospa(x, y, c: float, p: float = 2.0, alpha: float = 2.0) -> GospaResult:
    """GOSPA distance between truth x (n, D) and estimates y (m, D); alpha must be 2."""
    assert alpha == 2, "only alpha = 2 is implemented"
    assert c > 0 and p >= 1, "need c > 0 and p >= 1"
    x = _as_points(x, "x")
    y = _as_points(y, "y")
    n, m = len(x), len(y)
    cp = float(c) ** p
    empty_pairs = np.zeros((0, 2), dtype=np.int64)
    if n == 0 or m == 0:
        cost = cp / 2.0 * (n + m)
        return GospaResult(cost ** (1.0 / p), cost, 0.0, 0, n, m, empty_pairs)
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same dimension")
    d = np.sqrt(((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2))
    if n == 1 or m == 1:  # monotone in d: the nearest partner decides
        flat = int(np.argmin(d))
        i, j = divmod(flat, m)
        rows, cols = np.array([i]), np.array([j])
    else:
        rows, cols = linear_sum_assignment(np.minimum(d, c) ** p)
    dk = d[rows, cols]
    keep = dk < c
    pairs = np.stack([rows[keep], cols[keep]], axis=1).astype(np.int64)
    k = int(keep.sum())
    loc = float((dk[keep] ** p).sum())
    cost = loc + cp / 2.0 * ((n - k) + (m - k))
    return GospaResult(cost ** (1.0 / p), cost, loc, k, n - k, m - k, pairs)


# --------------------------------------------------------------------------------------
# GOSPA over many time steps
# --------------------------------------------------------------------------------------

def gospa_sequence(step_x, x, step_y, y, n_steps: int, c: float, p: float = 2.0,
                   chunk_pairs: int = 400_000) -> dict:
    """GOSPA at every step 0..n_steps-1.

    step_x[i] is the step of truth row x[i]; step_y[j] the step of estimate row y[j]
    (integers in [0, n_steps), rows in any order). Returns per-step arrays distance, cost,
    loc, n_missed, n_false, n_truth, n_est (length n_steps) and 'pairs' (K,3) int
    [step, truth_row, est_row] indexing the INPUT arrays, sorted by step then truth row.

    Method: every (truth, estimate) pair of a step is a candidate; only pairs with d < c
    matter. Edges that touch no other edge are accepted outright (vectorised); steps with
    competing edges solve a small assignment on the rows/columns involved.
    """
    assert c > 0 and p >= 1, "need c > 0 and p >= 1"
    step_x = np.asarray(step_x, dtype=np.int64).reshape(-1)
    step_y = np.asarray(step_y, dtype=np.int64).reshape(-1)
    x = _as_points(x, "x")
    y = _as_points(y, "y")
    if len(x) != len(step_x) or len(y) != len(step_y):
        raise ValueError("step arrays must match the number of rows")
    for s in (step_x, step_y):
        if len(s) and (s.min() < 0 or s.max() >= n_steps):
            raise ValueError("step index outside [0, n_steps)")
    if len(x) and len(y) and x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same dimension")
    cp = float(c) ** p

    n_truth = np.bincount(step_x, minlength=n_steps).astype(np.int64)
    n_est = np.bincount(step_y, minlength=n_steps).astype(np.int64)
    ox = np.argsort(step_x, kind="stable")   # sorted position -> input row
    oy = np.argsort(step_y, kind="stable")
    tstart = np.concatenate([[0], np.cumsum(n_truth)])
    estart = np.concatenate([[0], np.cumsum(n_est)])
    xs, ys = x[ox], y[oy]

    pair_step: list[np.ndarray] = []
    pair_t: list[np.ndarray] = []
    pair_e: list[np.ndarray] = []
    pair_d: list[np.ndarray] = []

    active = np.flatnonzero((n_truth > 0) & (n_est > 0))
    cnt = n_truth[active] * n_est[active]
    # chunk the active steps so the candidate-pair arrays stay small
    bounds = [0]
    acc = 0
    for k, cn in enumerate(cnt):
        acc += int(cn)
        if acc >= chunk_pairs:
            bounds.append(k + 1)
            acc = 0
    if bounds[-1] != len(active):
        bounds.append(len(active))
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        st = active[b0:b1]
        cn = cnt[b0:b1]
        off = np.concatenate([[0], np.cumsum(cn)])[:-1]
        tot = int(cn.sum())
        s_rep = np.repeat(st, cn)
        k = np.arange(tot, dtype=np.int64) - np.repeat(off, cn)
        ne = n_est[s_rep]
        t_idx = tstart[s_rep] + k // ne
        e_idx = estart[s_rep] + k % ne
        d = np.sqrt(((xs[t_idx] - ys[e_idx]) ** 2).sum(axis=1))
        keep = d < c
        pair_step.append(s_rep[keep])
        pair_t.append(t_idx[keep])
        pair_e.append(e_idx[keep])
        pair_d.append(d[keep])

    if pair_step:
        es = np.concatenate(pair_step)
        et = np.concatenate(pair_t)
        ee = np.concatenate(pair_e)
        ed = np.concatenate(pair_d)
    else:
        es = et = ee = np.zeros(0, dtype=np.int64)
        ed = np.zeros(0)

    # edges alone on both their truth and their estimate are pairs for certain
    if len(es):
        deg_t = np.bincount(et, minlength=len(xs))[et]
        deg_e = np.bincount(ee, minlength=len(ys))[ee]
        solo = (deg_t == 1) & (deg_e == 1)
    else:
        solo = np.zeros(0, dtype=bool)
    sel = [np.flatnonzero(solo)]
    contested = np.flatnonzero(~solo)
    if len(contested):
        cs = es[contested]
        brk = np.flatnonzero(np.diff(cs)) + 1
        starts = np.concatenate([[0], brk])
        ends = np.concatenate([brk, [len(cs)]])
        chosen: list[int] = []
        for a0, a1 in zip(starts, ends):
            idx = contested[a0:a1]
            tu, ti = np.unique(et[idx], return_inverse=True)
            eu, ei = np.unique(ee[idx], return_inverse=True)
            if len(idx) == 1:  # cannot happen (it would be solo) but stay safe
                chosen.append(int(idx[0]))
                continue
            mat = np.full((len(tu), len(eu)), cp)
            mat[ti, ei] = ed[idx] ** p
            isedge = np.zeros(mat.shape, dtype=bool)
            isedge[ti, ei] = True
            where = np.full(mat.shape, -1, dtype=np.int64)
            where[ti, ei] = idx
            r, cc = linear_sum_assignment(mat)
            ok = isedge[r, cc]
            chosen.extend(where[r[ok], cc[ok]].tolist())
        sel.append(np.array(chosen, dtype=np.int64))
    sel_idx = np.concatenate(sel) if sel else np.zeros(0, dtype=np.int64)
    ps, pt, pe, pd = es[sel_idx], et[sel_idx], ee[sel_idx], ed[sel_idx]

    n_assigned = np.bincount(ps, minlength=n_steps).astype(np.int64)
    loc = np.bincount(ps, weights=pd ** p, minlength=n_steps) if len(ps) else np.zeros(n_steps)
    n_missed = n_truth - n_assigned
    n_false = n_est - n_assigned
    cost = loc + cp / 2.0 * (n_missed + n_false)
    pairs = np.stack([ps, ox[pt], oy[pe]], axis=1).astype(np.int64) if len(ps) else np.zeros((0, 3), dtype=np.int64)
    if len(pairs):
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    return {
        "distance": cost ** (1.0 / p),
        "cost": cost,
        "loc": loc,
        "n_missed": n_missed,
        "n_false": n_false,
        "n_truth": n_truth,
        "n_est": n_est,
        "pairs": pairs,
    }


# --------------------------------------------------------------------------------------
# Self-checks
# --------------------------------------------------------------------------------------

_FAILS: list[str] = []


def _check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        _FAILS.append(name)


_PERM_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _brute_min(a: np.ndarray) -> float:
    n, m = a.shape
    if n == 0 or m == 0:
        return 0.0
    if n > m:
        a, (n, m) = a.T, (m, n)
    key = (n, m)
    if key not in _PERM_CACHE:
        _PERM_CACHE[key] = np.array(list(itertools.permutations(range(m), n)), dtype=np.int64)
    perms = _PERM_CACHE[key]
    return float(a[np.arange(n)[None, :], perms].sum(axis=1).min())


def check_assignment() -> None:
    print("1. linear_sum_assignment vs brute force over all injections")
    rng = np.random.default_rng(20240601)
    n_cases = 2400
    worst = 0.0
    bad = 0
    shapes_seen = set()
    for t in range(n_cases):
        n = int(rng.integers(0, 8))
        m = int(rng.integers(0, 8))
        if t % 3 == 0:
            a = rng.integers(0, 4, size=(n, m)).astype(float)   # heavy ties
        elif t % 3 == 1:
            a = rng.normal(size=(n, m)) * 10                    # negative costs too
        else:
            a = rng.random((n, m)) * 100
        shapes_seen.add((n, m))
        r, cc = linear_sum_assignment(a)
        k = min(n, m)
        valid = (len(r) == k and len(cc) == k and r.dtype.kind == "i" and cc.dtype.kind == "i"
                 and np.all(np.diff(r) > 0) and len(set(cc.tolist())) == k
                 and (k == 0 or (r.min() >= 0 and r.max() < n and cc.min() >= 0 and cc.max() < m)))
        got = float(a[r, cc].sum()) if k else 0.0
        err = abs(got - _brute_min(a))
        worst = max(worst, err)
        if not valid or err > 1e-9:
            bad += 1
    _check(f"{n_cases} random matrices (0..7 x 0..7, ties, negatives) optimal and valid",
           bad == 0, f"bad={bad} worst_err={worst:.2e} shapes={len(shapes_seen)}")
    _check("0 x m and n x 0 present", {(0, 3), (4, 0), (0, 0)} <= shapes_seen or
           any(s[0] == 0 for s in shapes_seen) and any(s[1] == 0 for s in shapes_seen))
    # numpy-vectorised path exercised directly (small sizes) against brute force
    bad = 0
    for t in range(300):
        n = int(rng.integers(2, 7))
        m = int(rng.integers(n, 8))
        a = rng.integers(0, 5, size=(n, m)).astype(float) if t % 2 else rng.random((n, m)) * 50
        cols = _hungarian_np(a)
        ok = len(set(cols.tolist())) == n and abs(a[np.arange(n), cols].sum() - _brute_min(a)) < 1e-9
        bad += not ok
    _check("numpy Hungarian variant optimal on 300 small matrices", bad == 0, f"bad={bad}")
    # large rectangular: both solvers agree, and dispatch to numpy path works
    a = rng.random((12, 150)) * 100
    r1, c1 = linear_sum_assignment(a)
    c2 = np.array(_hungarian_py(a.tolist(), 12, 150))
    _check("large 12x150 (numpy path) equals list Hungarian cost",
           abs(a[r1, c1].sum() - a[np.arange(12), c2].sum()) < 1e-9)
    a = rng.random((150, 12)) * 100
    r1, c1 = linear_sum_assignment(a)
    c2 = np.array(_hungarian_py(a.T.tolist(), 12, 150))
    _check("transposed 150x12 equals list Hungarian cost, rows ascending",
           abs(a[r1, c1].sum() - a.T[np.arange(12), c2].sum()) < 1e-9 and np.all(np.diff(r1) > 0))
    for bad_in in (np.array([[1.0, np.nan]]), np.array([[np.inf, 1.0]])):
        try:
            linear_sum_assignment(bad_in)
            ok = False
        except ValueError:
            ok = True
        _check("non-finite input raises ValueError", ok)


def check_hand_cases() -> None:
    print("2. hand-computed GOSPA cases")
    z = np.zeros((0, 2))
    cases = [
        ("both empty -> 0", gospa(z, z, 10.0).distance, 0.0),
        ("one missed, c=10 p=2 -> sqrt(50)", gospa([[0, 0]], z, 10.0).distance, np.sqrt(50.0)),
        ("one false, c=10 p=2 -> sqrt(50)", gospa(z, [[5, 5]], 10.0).distance, np.sqrt(50.0)),
        ("one pair at 3 m -> 3", gospa([[0, 0]], [[3, 0]], 10.0).distance, 3.0),
        ("one pair at 20 m, c=10 -> 10 (miss+false)", gospa([[0, 0]], [[20, 0]], 10.0).distance, 10.0),
        ("pair at exactly c -> unassigned, 10", gospa([[0, 0]], [[10, 0]], 10.0).distance, 10.0),
        # crossing: x=(0,0),(10,0), y=(9,0),(1,0): optimal pairs (x0,y1)=1,(x1,y0)=1 -> sqrt(2); naive is sqrt(162)
        ("crossing pairs pick optimal pairing", gospa([[0, 0], [10, 0]], [[9, 0], [1, 0]], 100.0).distance, np.sqrt(2.0)),
        # n != m: pair at 1 plus one false: 1 + 100/2 = 51
        ("1 truth, 2 est (one far): sqrt(1+50)", gospa([[0, 0]], [[1, 0], [50, 0]], 10.0).distance, np.sqrt(51.0)),
        ("2 truth, 1 est: sqrt(4+50)", gospa([[0, 0], [100, 0]], [[2, 0]], 10.0).distance, np.sqrt(54.0)),
        ("p=1: pair 3 + one missed 5 = 8", gospa([[0, 0], [100, 0]], [[3, 0]], 10.0, p=1.0).distance, 8.0),
        ("3-D pair (1,2,2) -> 3", gospa([[0, 0, 0]], [[1, 2, 2]], 10.0).distance, 3.0),
        ("3x3 permuted identical sets -> 0",
         gospa([[0, 0], [5, 5], [9, 1]], [[9, 1], [0, 0], [5, 5]], 10.0).distance, 0.0),
    ]
    for name, got, want in cases:
        _check(name, abs(got - want) < 1e-12, f"got={got:.6f} want={want:.6f}")
    r = gospa([[0, 0], [100, 0], [50, 50]], [[1, 0], [300, 0]], 10.0)
    _check("counts: 3 truth, 2 est, one pair",
           (r.n_assigned, r.n_missed, r.n_false) == (1, 2, 1) and r.pairs.tolist() == [[0, 0]]
           and abs(r.loc - 1.0) < 1e-12 and abs(r.cost - (1 + 50 * 3)) < 1e-9)


def _rand_set(rng, k: int, dim: int = 2, scale: float = 30.0) -> np.ndarray:
    return rng.random((k, dim)) * scale


def check_metric_properties() -> None:
    print("3. permutation invariance, symmetry, triangle inequality")
    rng = np.random.default_rng(7)
    bad_perm = bad_sym = 0
    for _ in range(300):
        n, m = int(rng.integers(0, 7)), int(rng.integers(0, 7))
        x, y = _rand_set(rng, n, 3), _rand_set(rng, m, 3)
        c, p = float(rng.choice([5.0, 10.0, 25.0])), float(rng.choice([1.0, 2.0, 3.0]))
        d0 = gospa(x, y, c, p).distance
        d1 = gospa(x[rng.permutation(n)], y[rng.permutation(m)], c, p).distance
        d2 = gospa(y, x, c, p).distance
        bad_perm += abs(d0 - d1) > 1e-9
        bad_sym += abs(d0 - d2) > 1e-9
    _check("invariant to permutation of x and of y (300 cases, p in 1,2,3)", bad_perm == 0, f"bad={bad_perm}")
    _check("symmetric d(X,Y) = d(Y,X) (300 cases)", bad_sym == 0, f"bad={bad_sym}")
    bad_tri = 0
    worst = -np.inf
    n_tri = 800
    for _ in range(n_tri):
        x, y, z = (_rand_set(rng, int(rng.integers(0, 6)), 2) for _ in range(3))
        c, p = float(rng.choice([4.0, 10.0, 40.0])), float(rng.choice([1.0, 2.0]))
        gap = gospa(x, z, c, p).distance - gospa(x, y, c, p).distance - gospa(y, z, c, p).distance
        worst = max(worst, gap)
        bad_tri += gap > 1e-9
    _check(f"triangle inequality on {n_tri} random triples", bad_tri == 0, f"bad={bad_tri} max_gap={worst:.3e}")


def _random_sequence(rng, n_steps: int, max_t: int, max_e: int, near: bool, scale: float = 500.0):
    nt = rng.integers(0, max_t + 1, size=n_steps)
    ne = rng.integers(0, max_e + 1, size=n_steps)
    step_x = np.repeat(np.arange(n_steps), nt)
    step_y = np.repeat(np.arange(n_steps), ne)
    x = rng.random((len(step_x), 3)) * scale
    y = rng.random((len(step_y), 3)) * scale
    if near and len(x) and len(y):  # estimates cluster around truths of the same step
        first = np.concatenate([[0], np.cumsum(nt)])
        for s in range(n_steps):
            if nt[s] and ne[s]:
                src = x[first[s]:first[s + 1]]
                lo = int(np.sum(ne[:s]))
                k = min(int(ne[s]), 2 * int(nt[s]))
                y[lo:lo + k] = src[rng.integers(0, nt[s], size=k)] + rng.normal(scale=6.0, size=(k, 3))
    # shuffle row order so the input is not step-sorted
    px, py = rng.permutation(len(step_x)), rng.permutation(len(step_y))
    return step_x[px], x[px], step_y[py], y[py]


def check_sequence() -> None:
    print("4. gospa_sequence equals a per-step loop of gospa")
    rng = np.random.default_rng(11)
    for label, near, c, p in (("clustered c=15 p=2", True, 15.0, 2.0),
                              ("uniform c=60 p=1", False, 60.0, 1.0),
                              ("clustered c=1e9 (dense) p=2", True, 1e9, 2.0)):
        n_steps = 300
        sx, x, sy, y = _random_sequence(rng, n_steps, 6, 12, near)
        res = gospa_sequence(sx, x, sy, y, n_steps, c, p)
        worst = 0.0
        bad = 0
        pairs_ok = True
        for s in range(n_steps):
            ix, iy = np.flatnonzero(sx == s), np.flatnonzero(sy == s)
            g = gospa(x[ix], y[iy], c, p)
            for key, val in (("distance", g.distance), ("cost", g.cost), ("loc", g.loc),
                             ("n_missed", g.n_missed), ("n_false", g.n_false),
                             ("n_truth", len(ix)), ("n_est", len(iy))):
                err = abs(float(res[key][s]) - val)
                worst = max(worst, err / max(1.0, abs(val)))
                bad += err > 1e-9 * max(1.0, abs(val))
            got = res["pairs"][res["pairs"][:, 0] == s]
            want = sorted((int(ix[a]), int(iy[b])) for a, b in g.pairs)
            pairs_ok &= sorted(map(tuple, got[:, 1:].tolist())) == want
        _check(f"{label}: 300 steps match per-step gospa", bad == 0 and pairs_ok,
               f"bad={bad} worst_rel={worst:.1e} pairs_equal={pairs_ok}")
    # edge cases: all-empty, one side empty, steps out of order
    z2 = np.zeros((0, 3))
    r = gospa_sequence([], z2, [], z2, 4, 10.0)
    _check("all steps empty -> distance 0", np.all(r["distance"] == 0) and r["pairs"].shape == (0, 3))
    r = gospa_sequence([1, 1], [[0, 0, 0], [1, 1, 1]], [], z2, 3, 10.0)
    _check("truth only at step 1 -> sqrt(2*50)",
           np.allclose(r["distance"], [0, 10.0, 0]) and r["n_missed"].tolist() == [0, 2, 0])
    r = gospa_sequence([2], [[0, 0, 0]], [2], [[3, 0, 0]], 4, 10.0)
    _check("single pair at step 2 -> 3", abs(r["distance"][2] - 3.0) < 1e-12 and r["pairs"].tolist() == [[2, 0, 0]])


def check_timing() -> None:
    print("5. timing: 20,000 steps, <= 8 truths, <= 30 estimates")
    rng = np.random.default_rng(3)
    n_steps = 20_000
    sx, x, sy, y = _random_sequence(rng, n_steps, 8, 30, True, scale=300.0)
    t0 = time.perf_counter()
    res = gospa_sequence(sx, x, sy, y, n_steps, 25.0, 2.0)
    dt = time.perf_counter() - t0
    _check("realistic (clustered estimates, c=25) < 10 s", dt < 10.0,
           f"{dt:.2f} s, {len(sx)} truth rows, {len(sy)} est rows, mean GOSPA {res['distance'].mean():.2f}")
    t0 = time.perf_counter()
    gospa_sequence(sx, x, sy, y, n_steps, 1e9, 2.0)
    dt2 = time.perf_counter() - t0
    _check("worst case (c huge: every pair contested) < 10 s", dt2 < 10.0, f"{dt2:.2f} s")


def main() -> int:
    check_assignment()
    check_hand_cases()
    check_metric_properties()
    check_sequence()
    check_timing()
    if _FAILS:
        print(f"\nFAILED: {len(_FAILS)} check(s): {_FAILS}")
        return 1
    print("\nALL SELF-CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
