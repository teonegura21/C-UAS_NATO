"""Deterministic gradient-boosted decision trees for binary classification (numpy only),
plus probability calibration and metrics. Our stand-in for LightGBM, which the numpy-only
rule excludes. Intended for 'is this track a real drone?' and 'are these two tracks the same
object?' on 1e4-1e5 labelled rows with 20-60 heterogeneous features and many NaNs.

Algorithm (XGBoost/LightGBM-style, second-order):
  * logistic loss; base score = log-odds of the weighted mean of y;
  * per-feature quantile bins fitted on the training set: at most n_bins bins, whose edges
    are actual training values (a split is 'x <= edge'), plus one dedicated missing bin.
    Using data values (not interpolated midpoints) as thresholds makes the fitted model
    exactly invariant to strictly increasing transforms of a feature;
  * histogram split search: per node, per feature, np.bincount of gradients / hessians /
    counts; sibling histograms by subtraction (only the smaller child is counted);
  * the missing bin is tried on both sides of every candidate split and goes to the side
    with the larger gain; that default direction is stored, and a NaN unseen in training at
    a node follows it at predict time (with no training NaN at the node, the side that got
    more training rows);
  * depth-wise growth to max_depth, constrained by min_samples_leaf and min_child_hessian;
    Newton leaf values -G/(H + l2) times learning_rate;
  * row subsampling and column subsampling per tree use a seeded np.random.Generator. The
    model is a pure function of (data, parameters, seed): bit-for-bit reproducible.

Conventions: X float with NaN = missing, y in {0, 1}, sample weights are used as given (not
renormalised, so min_child_hessian scales with them). Gain = 0.5 * (GL^2/(HL+l2) +
GR^2/(HR+l2) - G^2/(H+l2)); a split needs gain > 1e-12.

Calibration: Isotonic (weighted pool-adjacent-violators, non-decreasing, piecewise-linear
between block end-points, output clipped to [1e-6, 1-1e-6]) and Platt scaling. Fit a
calibrator on rows the booster did not train on. Metrics: auc (ties 0.5), logloss, brier,
all optionally weighted.

Run `python3 tools/gbdt.py` for the numbered self-checks (exit status 1 on any failure).
"""

from __future__ import annotations

import json
import sys
import time
from itertools import product

import numpy as np

EPS_P = 1e-6          # clip range of calibrated probabilities
MIN_GAIN = 1e-12


# ----------------------------------------------------------------------------- helpers
def _sigmoid(m: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * m))


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _check_y(y) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).ravel()
    if y.size == 0 or not np.all((y == 0) | (y == 1)):
        raise ValueError("y must be non-empty with values in {0, 1}")
    return y


def _check_w(w, n: int) -> np.ndarray:
    if w is None:
        return np.ones(n)
    w = np.asarray(w, dtype=np.float64).ravel()
    if w.shape[0] != n or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("sample weights must be finite, non-negative and match y")
    return w


# ----------------------------------------------------------------------------- metrics
def auc(y, p, w=None) -> float:
    """Weighted area under the ROC curve (Mann-Whitney), ties in p count 0.5.
    NaN if one class has no weight."""
    y = _check_y(y)
    p = np.asarray(p, dtype=np.float64).ravel()
    w = _check_w(w, y.size)
    _, inv = np.unique(p, return_inverse=True)
    k = inv.max() + 1
    wp = np.bincount(inv, weights=w * y, minlength=k)
    wn = np.bincount(inv, weights=w * (1.0 - y), minlength=k)
    tp, tn = wp.sum(), wn.sum()
    if tp <= 0 or tn <= 0:
        return float("nan")
    neg_below = np.cumsum(wn) - wn
    return float(np.sum(wp * (neg_below + 0.5 * wn)) / (tp * tn))


def logloss(y, p, w=None) -> float:
    y = _check_y(y)
    p = np.clip(np.asarray(p, dtype=np.float64).ravel(), 1e-15, 1.0 - 1e-15)
    w = _check_w(w, y.size)
    return float(-np.sum(w * (y * np.log(p) + (1.0 - y) * np.log1p(-p))) / w.sum())


def brier(y, p, w=None) -> float:
    y = _check_y(y)
    p = np.asarray(p, dtype=np.float64).ravel()
    w = _check_w(w, y.size)
    return float(np.sum(w * (p - y) ** 2) / w.sum())


def _logloss_margin(y: np.ndarray, m: np.ndarray, w: np.ndarray | None = None) -> float:
    """logloss from a raw margin without going through a clipped probability."""
    per = np.maximum(m, 0.0) - y * m + np.log1p(np.exp(-np.abs(m)))
    return float(np.mean(per) if w is None else np.sum(w * per) / w.sum())


# ----------------------------------------------------------------------------- binning
def _fit_edges(X: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """Per-feature split thresholds (training values, strictly increasing). Bin of x is
    searchsorted(edges, x, 'left'), so x <= edges[k] <=> bin <= k. At most n_bins - 1 edges."""
    edges = []
    for j in range(X.shape[1]):
        col = X[:, j]
        s = col[~np.isnan(col)]
        if s.size == 0:
            edges.append(np.empty(0))
            continue
        s = np.sort(s)
        u = np.unique(s)
        if u.size <= n_bins:
            e = u[:-1]
        else:
            m = s.size
            idx = np.clip((np.arange(1, n_bins) * m) // n_bins - 1, 0, m - 1)
            e = np.unique(s[idx])
            e = e[e < s[-1]]
        edges.append(np.ascontiguousarray(e))
    return edges


def _bin(X: np.ndarray, edges: list[np.ndarray], n_bins: int) -> np.ndarray:
    """Feature-major bin matrix (F, n); missing = n_bins."""
    dt = np.uint8 if n_bins <= 255 else np.uint16
    out = np.empty((X.shape[1], X.shape[0]), dtype=dt)
    for j in range(X.shape[1]):
        col = X[:, j]
        b = np.searchsorted(edges[j], col, side="left")
        b[np.isnan(col)] = n_bins
        out[j] = b
    return out


# ----------------------------------------------------------------------------- trees
class _Tree:
    """Flat arrays; node 0 is the root; feat < 0 marks a leaf."""

    __slots__ = ("feat", "thr", "thr_bin", "dleft", "left", "right", "value")

    def __init__(self, feat, thr, thr_bin, dleft, left, right, value):
        self.feat = np.asarray(feat, dtype=np.int64)
        self.thr = np.asarray(thr, dtype=np.float64)
        self.thr_bin = np.asarray(thr_bin, dtype=np.int64)
        self.dleft = np.asarray(dleft, dtype=bool)
        self.left = np.asarray(left, dtype=np.int64)
        self.right = np.asarray(right, dtype=np.int64)
        self.value = np.asarray(value, dtype=np.float64)

    def apply_raw(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        node = np.zeros(n, dtype=np.int64)
        ar = np.arange(n)
        while True:
            f = self.feat[node]
            act = f >= 0
            if not act.any():
                break
            x = X[ar, np.where(act, f, 0)]
            go_left = np.where(np.isnan(x), self.dleft[node], x <= self.thr[node])
            node = np.where(act, np.where(go_left, self.left[node], self.right[node]), node)
        return self.value[node]

    def apply_binned(self, XbT: np.ndarray, idx: np.ndarray, n_bins: int) -> np.ndarray:
        node = np.zeros(idx.size, dtype=np.int64)
        while True:
            f = self.feat[node]
            act = f >= 0
            if not act.any():
                break
            b = XbT[np.where(act, f, 0), idx].astype(np.int64)
            go_left = (b <= self.thr_bin[node]) | ((b == n_bins) & self.dleft[node])
            node = np.where(act, np.where(go_left, self.left[node], self.right[node]), node)
        return self.value[node]

    def to_dict(self) -> dict:
        return {"feat": self.feat.tolist(), "thr": self.thr.tolist(),
                "dleft": self.dleft.tolist(), "left": self.left.tolist(),
                "right": self.right.tolist(), "value": self.value.tolist()}

    @staticmethod
    def from_dict(d: dict) -> "_Tree":
        n = len(d["feat"])
        return _Tree(d["feat"], d["thr"], np.zeros(n, dtype=np.int64), d["dleft"],
                     d["left"], d["right"], d["value"])


class GBDT:
    """Histogram gradient-boosted trees, binary logistic loss, NaN-aware. See module doc."""

    def __init__(self, n_trees=300, learning_rate=0.05, max_depth=4, min_samples_leaf=20,
                 min_child_hessian=1e-3, l2=1.0, n_bins=64, subsample=1.0, colsample=1.0,
                 early_stopping_rounds=None, seed=0):
        if not 2 <= n_bins <= 65534:
            raise ValueError("n_bins must be in [2, 65534]")
        if not (0 < subsample <= 1 and 0 < colsample <= 1):
            raise ValueError("subsample and colsample must be in (0, 1]")
        self.n_trees = int(n_trees)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_child_hessian = float(min_child_hessian)
        self.l2 = float(l2)
        self.n_bins = int(n_bins)
        self.subsample = float(subsample)
        self.colsample = float(colsample)
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = int(seed)
        self.trees_: list[_Tree] = []
        self.base_score_ = 0.0
        self.n_features_ = 0
        self.best_iteration_ = -1
        self.eval_history_: list[float] = []
        self._gain = np.zeros(0)
        self._splits = np.zeros(0)

    # ------------------------------------------------------------------ params
    _PARAMS = ("n_trees", "learning_rate", "max_depth", "min_samples_leaf",
               "min_child_hessian", "l2", "n_bins", "subsample", "colsample",
               "early_stopping_rounds", "seed")

    def get_params(self) -> dict:
        return {k: getattr(self, k) for k in self._PARAMS}

    # ------------------------------------------------------------------ histograms
    def _hist(self, XbS, r, g, h, full):
        """(3, Fs, B+1): gradient sum, hessian sum, row count per feature bin."""
        B = self.n_bins
        Fs = XbS.shape[0]
        gr, hr = g[r], h[r]
        out = np.empty((3, Fs, B + 1))
        sub = XbS if full else XbS[:, r]
        for j in range(Fs):
            col = sub[j]
            out[0, j] = np.bincount(col, weights=gr, minlength=B + 1)
            out[1, j] = np.bincount(col, weights=hr, minlength=B + 1)
            out[2, j] = np.bincount(col, minlength=B + 1)
        return out

    def _best_split(self, hist, G, H, n, nedges):
        """Best (gain, feature, bin, default_left) over candidate splits, or None."""
        B = self.n_bins
        l2 = self.l2
        msl, mch = self.min_samples_leaf, self.min_child_hessian
        gm, hm, cm = hist[0][:, B], hist[1][:, B], hist[2][:, B]
        cg = np.cumsum(hist[0][:, : B - 1], axis=1)
        ch = np.cumsum(hist[1][:, : B - 1], axis=1)
        cc = np.cumsum(hist[2][:, : B - 1], axis=1)
        # axis 0: 0 = missing goes right, 1 = missing goes left
        GL = np.stack([cg, cg + gm[:, None]])
        HL = np.stack([ch, ch + hm[:, None]])
        nL = np.stack([cc, cc + cm[:, None]])
        GR, HR, nR = G - GL, H - HL, n - nL
        valid = ((nL >= msl) & (nR >= msl) & (HL >= mch) & (HR >= mch)
                 & (np.arange(B - 1)[None, None, :] < nedges[None, :, None]))
        if not valid.any():
            return None
        gain = 0.5 * (GL * GL / (HL + l2) + GR * GR / (HR + l2) - G * G / (H + l2))
        gain = np.where(valid, gain, -np.inf)
        d, j, k = np.unravel_index(int(np.argmax(gain)), gain.shape)
        best = float(gain[d, j, k])
        if not best > MIN_GAIN:
            return None
        dleft = bool(d == 1)
        if cm[j] == 0:                         # no training NaN here: follow the bigger side
            dleft = bool(cc[j, k] >= n - cc[j, k])
        return best, int(j), int(k), dleft

    # ------------------------------------------------------------------ one tree
    def _grow(self, XbS, cols, edges, nedges, g, h, rows, margin, full_rows):
        B = self.n_bins
        msl = self.min_samples_leaf
        feat, thr, thrb, dl, left, right, val = [], [], [], [], [], [], []

        def new_node():
            feat.append(-1); thr.append(0.0); thrb.append(0); dl.append(False)
            left.append(-1); right.append(-1); val.append(0.0)
            return len(feat) - 1

        root = new_node()
        stack = [(root, rows, 0, None, float(g[rows].sum()), float(h[rows].sum()))]
        while stack:
            nid, r, depth, hist, G, H = stack.pop()
            split = None
            if depth < self.max_depth and r.size >= 2 * msl:
                if hist is None:
                    hist = self._hist(XbS, r, g, h, full_rows and depth == 0)
                split = self._best_split(hist, G, H, r.size, nedges)
            if split is None:
                v = -G / (H + self.l2) * self.learning_rate
                val[nid] = v
                margin[r] += v
                continue
            gain, j, k, dleft = split
            fj = int(cols[j])
            b = XbS[j][r]
            go_left = (b <= k) | ((b == B) & dleft)
            rl, rr = r[go_left], r[~go_left]
            feat[nid], thr[nid], thrb[nid], dl[nid] = fj, float(edges[fj][k]), k, dleft
            lid, rid = new_node(), new_node()
            left[nid], right[nid] = lid, rid
            self._gain[fj] += gain
            self._splits[fj] += 1
            hl = hr_ = None
            if depth + 1 < self.max_depth and (rl.size >= 2 * msl or rr.size >= 2 * msl):
                if rl.size <= rr.size:
                    hl = self._hist(XbS, rl, g, h, False)
                    hr_ = hist - hl
                else:
                    hr_ = self._hist(XbS, rr, g, h, False)
                    hl = hist - hr_
            stack.append((rid, rr, depth + 1, hr_, float(g[rr].sum()), float(h[rr].sum())))
            stack.append((lid, rl, depth + 1, hl, float(g[rl].sum()), float(h[rl].sum())))
        return _Tree(feat, thr, thrb, dl, left, right, val)

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, sample_weight=None, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        y = _check_y(y)
        if X.shape[0] != y.size:
            raise ValueError("X and y have different lengths")
        n, F = X.shape
        w = _check_w(sample_weight, n)
        if w.sum() <= 0:
            raise ValueError("sample weights sum to zero")
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be given together")
        have_val = X_val is not None
        if have_val:
            X_val = np.asarray(X_val, dtype=np.float64)
            y_val = _check_y(y_val)
            if X_val.shape[1] != F or X_val.shape[0] != y_val.size:
                raise ValueError("validation set shape mismatch")

        rng = np.random.default_rng(self.seed)
        B = self.n_bins
        edges = _fit_edges(X, B)
        nedges_all = np.array([e.size for e in edges], dtype=np.int64)
        XbT = _bin(X, edges, B)
        XbT_val = _bin(X_val, edges, B) if have_val else None
        idx_val = np.arange(X_val.shape[0]) if have_val else None

        pbar = float(np.clip(np.sum(w * y) / w.sum(), EPS_P, 1 - EPS_P))
        self.base_score_ = _logit(pbar)
        self.n_features_ = F
        self.trees_ = []
        self.eval_history_ = []
        self._gain = np.zeros(F)
        self._splits = np.zeros(F)
        margin = np.full(n, self.base_score_)
        vmargin = np.full(X_val.shape[0], self.base_score_) if have_val else None

        n_sub = n if self.subsample >= 1.0 else max(1, int(round(self.subsample * n)))
        n_col = F if self.colsample >= 1.0 else max(1, int(round(self.colsample * F)))
        esr = self.early_stopping_rounds
        best_loss, best_it = np.inf, -1
        imp_hist = []                          # importance snapshot after each tree
        for it in range(self.n_trees):
            p = _sigmoid(margin)
            g = w * (p - y)
            h = w * np.maximum(p * (1.0 - p), 1e-12)
            if n_sub < n:
                rows = np.sort(rng.choice(n, size=n_sub, replace=False))
            else:
                rows = np.arange(n)
            if n_col < F:
                cols = np.sort(rng.choice(F, size=n_col, replace=False))
                XbS, nedges = XbT[cols], nedges_all[cols]
            else:
                cols, XbS, nedges = np.arange(F), XbT, nedges_all
            tree = self._grow(XbS, cols, edges, nedges, g, h, rows, margin, n_sub == n)
            if n_sub < n:                      # out-of-bag rows still need the update
                mask = np.ones(n, dtype=bool)
                mask[rows] = False
                oob = np.flatnonzero(mask)
                margin[oob] += tree.apply_binned(XbT, oob, B)
            self.trees_.append(tree)
            imp_hist.append((self._gain.copy(), self._splits.copy()))
            if have_val:
                vmargin += tree.apply_binned(XbT_val, idx_val, B)
                loss = _logloss_margin(y_val, vmargin)
                self.eval_history_.append(loss)
                if loss < best_loss - 1e-12:
                    best_loss, best_it = loss, it
                elif esr is not None and it - best_it >= esr:
                    break
        if have_val and esr is not None:
            self.trees_ = self.trees_[: best_it + 1]
            if best_it >= 0:                   # importances only for the trees kept
                self._gain, self._splits = imp_hist[best_it]
        self.best_iteration_ = len(self.trees_) - 1
        return self

    # ------------------------------------------------------------------ predict
    def decision_function(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError(f"expected X with {self.n_features_} columns")
        m = np.full(X.shape[0], self.base_score_)
        for t in self.trees_:
            m += t.apply_raw(X)
        return m

    def predict_proba(self, X) -> np.ndarray:
        return _sigmoid(self.decision_function(X))

    def feature_importance(self, kind: str = "gain") -> np.ndarray:
        """Per-feature total split gain ('gain') or number of splits ('split'), raw sums."""
        if kind == "gain":
            return self._gain.copy()
        if kind == "split":
            return self._splits.copy()
        raise ValueError("kind must be 'gain' or 'split'")

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {"format": "gbdt-1", "params": self.get_params(),
                "base_score": self.base_score_, "n_features": self.n_features_,
                "best_iteration": self.best_iteration_,
                "gain": self._gain.tolist(), "splits": self._splits.tolist(),
                "trees": [t.to_dict() for t in self.trees_]}

    @classmethod
    def from_dict(cls, d: dict) -> "GBDT":
        m = cls(**d["params"])
        m.base_score_ = float(d["base_score"])
        m.n_features_ = int(d["n_features"])
        m.best_iteration_ = int(d["best_iteration"])
        m._gain = np.asarray(d["gain"], dtype=np.float64)
        m._splits = np.asarray(d["splits"], dtype=np.float64)
        m.trees_ = [_Tree.from_dict(t) for t in d["trees"]]
        return m


# ----------------------------------------------------------------------------- calibration
class Isotonic:
    """Non-decreasing calibration map p -> P(y=1). Weighted pool-adjacent-violators on the
    distinct values of p; knots at both ends of every pooled block, linear in between, flat
    beyond the ends; output clipped to [1e-6, 1-1e-6]. y may be 0/1 or fractional."""

    def __init__(self):
        self.x_: np.ndarray | None = None
        self.y_: np.ndarray | None = None

    def fit(self, p, y, w=None) -> "Isotonic":
        p = np.asarray(p, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        w = _check_w(w, p.size)
        keep = w > 0
        p, y, w = p[keep], y[keep], w[keep]
        if p.size == 0:
            raise ValueError("no positive-weight rows")
        px, inv = np.unique(p, return_inverse=True)
        wg = np.bincount(inv, weights=w)
        yg = np.bincount(inv, weights=w * y) / wg
        val, wt, lo, hi = [], [], [], []
        for i in range(px.size):
            v, wi, l = yg[i], wg[i], px[i]
            while val and val[-1] >= v:
                v = (val[-1] * wt[-1] + v * wi) / (wt[-1] + wi)
                wi += wt[-1]
                l = lo[-1]
                val.pop(); wt.pop(); lo.pop(); hi.pop()
            val.append(v); wt.append(wi); lo.append(l); hi.append(px[i])
        xs, ys = [], []
        for v, l, h_ in zip(val, lo, hi):
            xs.append(l); ys.append(v)
            if h_ > l:
                xs.append(h_); ys.append(v)
        self.x_ = np.asarray(xs)
        self.y_ = np.clip(np.asarray(ys), EPS_P, 1 - EPS_P)
        return self

    def predict(self, p) -> np.ndarray:
        if self.x_ is None:
            raise RuntimeError("Isotonic not fitted")
        return np.clip(np.interp(np.asarray(p, dtype=np.float64), self.x_, self.y_),
                       EPS_P, 1 - EPS_P)

    def to_dict(self) -> dict:
        return {"x": self.x_.tolist(), "y": self.y_.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Isotonic":
        o = cls()
        o.x_, o.y_ = np.asarray(d["x"]), np.asarray(d["y"])
        return o


def platt_fit(score, y) -> tuple[float, float]:
    """Logistic regression of y on one score: P = sigmoid(a * score + b). Damped Newton
    with a vanishing ridge so separable data still gives finite (a, b)."""
    s = np.asarray(score, dtype=np.float64).ravel()
    y = _check_y(y)
    if s.size != y.size:
        raise ValueError("score and y differ in length")
    ridge = 1e-9
    ybar = float(np.clip(y.mean(), EPS_P, 1 - EPS_P))
    theta = np.array([0.0, _logit(ybar)])

    def loss(t):
        z = t[0] * s + t[1]
        return _logloss_margin(y, z) + 0.5 * ridge * float(t @ t)

    cur = loss(theta)
    for _ in range(100):
        p = _sigmoid(theta[0] * s + theta[1])
        r = p - y
        grad = np.array([np.mean(r * s), np.mean(r)]) + ridge * theta
        wv = p * (1 - p)
        hess = np.array([[np.mean(wv * s * s), np.mean(wv * s)],
                         [np.mean(wv * s), np.mean(wv)]]) + ridge * np.eye(2)
        step = np.linalg.solve(hess + 1e-12 * np.eye(2), grad)
        t = 1.0
        while t > 1e-10:
            cand = theta - t * step
            new = loss(cand)
            if new <= cur:
                break
            t *= 0.5
        else:
            break
        theta, done = cand, cur - new < 1e-14
        cur = new
        if done:
            break
    return float(theta[0]), float(theta[1])


def platt_predict(score, a: float, b: float) -> np.ndarray:
    return _sigmoid(a * np.asarray(score, dtype=np.float64) + b)


# ----------------------------------------------------------------------------- self-checks
_RESULTS: list[tuple[int, str, bool, str]] = []


def _check(num: int, name: str, ok: bool, detail: str) -> None:
    _RESULTS.append((num, name, bool(ok), detail))
    print(f"[{num}] {'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def _check_xor(rng) -> None:
    n = 4000
    Xtr = rng.uniform(-1, 1, (n, 2))
    clean = (Xtr[:, 0] * Xtr[:, 1] > 0).astype(float)
    flip = rng.random(n) < 0.2
    ytr = np.where(flip, 1 - clean, clean)
    Xte = rng.uniform(-1, 1, (n, 2))
    yte = (Xte[:, 0] * Xte[:, 1] > 0).astype(float)      # clean labels: noise-free target
    m = GBDT(n_trees=200, learning_rate=0.1, max_depth=3, min_samples_leaf=20, seed=1)
    a = auc(yte, m.fit(Xtr, ytr).predict_proba(Xte))
    lin = auc(yte, _linear_scores(Xtr, ytr, Xte))
    _check(1, "XOR with 20% label noise", a > 0.85 and lin < 0.6,
           f"held-out AUC vs clean XOR {a:.4f} (> 0.85), linear model {lin:.4f}")


def _linear_scores(Xtr, ytr, Xte):
    A = np.hstack([Xtr, np.ones((Xtr.shape[0], 1))])
    th = np.zeros(A.shape[1])
    for _ in range(50):
        p = _sigmoid(A @ th)
        H = (A * (p * (1 - p))[:, None]).T @ A + 1e-6 * np.eye(A.shape[1])
        th = th - np.linalg.solve(H, A.T @ (p - ytr))
    return np.hstack([Xte, np.ones((Xte.shape[0], 1))]) @ th


def _check_linear(rng) -> None:
    f, ntr, nte = 8, 30000, 50000
    beta = rng.normal(size=f) * 0.6
    def draw(n):
        X = rng.normal(size=(n, f))
        z = X @ beta + 0.3
        y = (rng.random(n) < _sigmoid(z)).astype(float)
        return X, z, y
    Xtr, _, ytr = draw(ntr)
    Xte, zte, yte = draw(nte)
    m = GBDT(n_trees=600, learning_rate=0.05, max_depth=2, min_samples_leaf=100, n_bins=128, seed=2)
    a = auc(yte, m.fit(Xtr, ytr).predict_proba(Xte))
    a0 = auc(yte, zte)
    _check(2, "linear-logistic data", abs(a - a0) < 0.01,
           f"held-out AUC {a:.4f} vs true model {a0:.4f} (diff {abs(a - a0):.4f} < 0.01)")


def _check_missing(rng) -> None:
    def draw(n):
        y = (rng.random(n) < 0.5).astype(float)
        x0 = rng.normal(size=n)                          # pure noise when present
        x0[rng.random(n) < np.where(y == 1, 0.95, 0.05)] = np.nan
        return np.column_stack([x0, rng.normal(size=(n, 3))]), y
    Xtr, ytr = draw(6000)
    Xte, yte = draw(6000)
    m = GBDT(n_trees=100, max_depth=3, seed=3).fit(Xtr, ytr)
    a = auc(yte, m.predict_proba(Xte))
    imp = m.feature_importance("gain")
    _check(3, "signal only through missingness", a > 0.85 and int(np.argmax(imp)) == 0,
           f"held-out AUC {a:.4f} (> 0.85, ideal 0.95), top-gain feature {int(np.argmax(imp))}")


def _synth(rng, n, f, nan_rate=0.1):
    X = rng.normal(size=(n, f))
    z = X[:, 0] + 0.8 * X[:, 1] * X[:, 2] - 0.7 * (X[:, 3] > 0.3) + 0.5 * np.sin(2 * X[:, 4])
    y = (rng.random(n) < _sigmoid(z)).astype(float)
    X[rng.random((n, f)) < nan_rate] = np.nan
    return X, y


def _check_determinism(rng) -> None:
    X, y = _synth(rng, 3000, 10)
    kw = dict(n_trees=40, subsample=0.7, colsample=0.6, max_depth=3)
    p1 = GBDT(seed=5, **kw).fit(X, y).predict_proba(X)
    p2 = GBDT(seed=5, **kw).fit(X, y).predict_proba(X)
    p3 = GBDT(seed=6, **kw).fit(X, y).predict_proba(X)
    d = float(np.max(np.abs(p1 - p3)))
    _check(4, "determinism", np.array_equal(p1, p2) and d > 1e-6,
           f"same seed bit-identical={np.array_equal(p1, p2)}, other seed max|dp|={d:.3g}")


def _check_monotone(rng) -> None:
    X, y = _synth(rng, 4000, 8)
    Xt = X.copy()
    Xt[:, 1] = np.expm1(0.7 * X[:, 1])                   # strictly increasing, NaN stays NaN
    Xt[:, 4] = X[:, 4] ** 3 + 5.0 * X[:, 4]
    Xtest, _ = _synth(rng, 1500, 8)
    Xtest_t = Xtest.copy()
    Xtest_t[:, 1] = np.expm1(0.7 * Xtest[:, 1])
    Xtest_t[:, 4] = Xtest[:, 4] ** 3 + 5.0 * Xtest[:, 4]
    kw = dict(n_trees=60, max_depth=4, seed=7)
    pa = GBDT(**kw).fit(X, y).predict_proba(Xtest)
    pb = GBDT(**kw).fit(Xt, y).predict_proba(Xtest_t)
    d = float(np.max(np.abs(pa - pb)))
    _check(5, "invariance to monotone feature transform", d <= 1e-12,
           f"max|dp| = {d:.3g} (<= 1e-12)")


def _brute_isotonic(p, y, w):
    order = np.argsort(p, kind="stable")
    y, w = y[order], w[order]
    n = y.size
    best_sse, best = np.inf, None
    for cuts in product((0, 1), repeat=n - 1):
        bounds = [0] + [i + 1 for i, c in enumerate(cuts) if c] + [n]
        fit = np.empty(n)
        for a, b in zip(bounds[:-1], bounds[1:]):
            fit[a:b] = np.sum(w[a:b] * y[a:b]) / np.sum(w[a:b])
        if np.any(np.diff(fit) < -1e-15):
            continue
        sse = float(np.sum(w * (y - fit) ** 2))
        if sse < best_sse - 1e-15:
            best_sse, best = sse, fit
    out = np.empty(n)
    out[order] = best
    return out


def _check_isotonic(rng) -> None:
    worst, cases, mono_ok, tie_ok = 0.0, 0, True, True
    for c in range(300):
        n = int(rng.integers(2, 9))
        p = rng.random(n)
        y = (rng.random(n) < 0.5).astype(float) if c % 2 else rng.random(n)
        w = rng.random(n) + 0.1 if c % 3 == 0 else np.ones(n)
        iso = Isotonic().fit(p, y, w)
        got = iso.predict(p)
        ref = np.clip(_brute_isotonic(p, y, w), EPS_P, 1 - EPS_P)
        worst = max(worst, float(np.max(np.abs(got - ref))))
        grid = np.linspace(-0.2, 1.2, 50)
        mono_ok &= bool(np.all(np.diff(iso.predict(grid)) >= -1e-15))
        cases += 1
    pt = np.array([0.3, 0.3, 0.3, 0.6, 0.6, 0.9])
    yt = np.array([0, 1, 1, 0, 1, 1.0])
    gt = Isotonic().fit(pt, yt).predict(pt)
    tie_ok = bool(gt[0] == gt[1] == gt[2] and gt[3] == gt[4] and np.all(np.diff(gt) >= 0))
    iso_ok = worst < 1e-9 and mono_ok and tie_ok
    iso_msg = (f"{cases} cases n<=8, max|diff| {worst:.2g}, non-decreasing={mono_ok}, "
               f"ties pooled={tie_ok}")
    # calibration of a deliberately miscalibrated score, on held-out data
    n = 20000
    z = rng.normal(size=n) * 1.5
    ptrue = _sigmoid(z)
    y = (rng.random(n) < ptrue).astype(float)
    bad = _sigmoid(3.0 * z - 1.0)                          # over-confident and shifted
    half = n // 2
    iso = Isotonic().fit(bad[:half], y[:half])
    a, b = platt_fit(np.log(bad[:half] / (1 - bad[:half])), y[:half])
    l0 = logloss(y[half:], bad[half:])
    l1 = logloss(y[half:], iso.predict(bad[half:]))
    l2_ = logloss(y[half:], platt_predict(np.log(bad[half:] / (1 - bad[half:])), a, b))
    lt = logloss(y[half:], ptrue[half:])
    cal_ok = l1 < l0 - 0.1 and l2_ < l0 - 0.1 and l1 < lt + 0.02
    _check(6, "isotonic = brute-force LS; calibration lowers logloss", iso_ok and cal_ok,
           f"{iso_msg}; held-out logloss raw {l0:.4f}, isotonic {l1:.4f}, platt {l2_:.4f} "
           f"(a={a:.3f}, b={b:.3f}, ideal 1/3 slope), true {lt:.4f}")


def _check_roundtrip(rng) -> None:
    X, y = _synth(rng, 3000, 10)
    m = GBDT(n_trees=50, max_depth=4, subsample=0.8, seed=8).fit(X, y)
    d = json.loads(json.dumps(m.to_dict()))
    m2 = GBDT.from_dict(d)
    Xt, _ = _synth(rng, 1000, 10)
    ok = (np.array_equal(m.predict_proba(Xt), m2.predict_proba(Xt))
          and np.array_equal(m.feature_importance("gain"), m2.feature_importance("gain")))
    # early stopping must leave importances consistent with the kept trees
    Xv, yv = _synth(rng, 1500, 10)
    me = GBDT(n_trees=400, learning_rate=0.3, max_depth=4, early_stopping_rounds=10,
              seed=8).fit(X, y, X_val=Xv, y_val=yv)
    kept = sum(int((t.feat >= 0).sum()) for t in me.trees_)
    es_ok = (len(me.trees_) < 400 and int(me.feature_importance("split").sum()) == kept
             and (me.feature_importance("gain") >= 0).all())
    try:
        GBDT(n_trees=5).fit(X, y, X_val=Xv)
        half_ok = False
    except ValueError:
        half_ok = True
    _check(7, "to_dict/from_dict round trip; early-stopping importances",
           ok and es_ok and half_ok,
           f"identical predictions through JSON: {ok}; early stop kept {len(me.trees_)} trees, "
           f"split importance {int(me.feature_importance('split').sum())} == {kept} nodes: {es_ok}; "
           f"X_val without y_val rejected: {half_ok}")


def _check_speed(rng) -> None:
    X, y = _synth(rng, 60000, 40)
    t0 = time.perf_counter()
    m = GBDT(n_trees=300, learning_rate=0.05, max_depth=4, seed=9).fit(X, y)
    dt = time.perf_counter() - t0
    t1 = time.perf_counter()
    p = m.predict_proba(X)
    dp = time.perf_counter() - t1
    _check(8, "speed 60000 x 40, 300 trees, depth 4", dt < 60.0,
           f"fit {dt:.1f}s (< 60 s), predict {dp:.2f}s, train AUC {auc(y, p):.3f}")


def _check_auc(rng) -> None:
    worst = 0.0
    for c in range(60):
        n = int(rng.integers(5, 80))
        y = (rng.random(n) < 0.4).astype(float)
        if y.sum() in (0, n):
            y[0], y[1] = 0.0, 1.0
        p = np.round(rng.random(n), 1) if c % 2 else rng.random(n)
        w = rng.random(n) + 0.05 if c % 3 == 0 else None
        ww = np.ones(n) if w is None else w
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        num = den = 0.0
        for i in pos:
            for j in neg:
                s = 1.0 if p[i] > p[j] else (0.5 if p[i] == p[j] else 0.0)
                num += ww[i] * ww[j] * s
                den += ww[i] * ww[j]
        worst = max(worst, abs(auc(y, p, w) - num / den))
    _check(9, "auc vs brute-force pairwise count", worst < 1e-12,
           f"60 random cases (ties, weights), max|diff| {worst:.2g}")


def main() -> int:
    rng = np.random.default_rng(20260928)
    _check_xor(rng)
    _check_linear(rng)
    _check_missing(rng)
    _check_determinism(rng)
    _check_monotone(rng)
    _check_isotonic(rng)
    _check_roundtrip(rng)
    _check_speed(rng)
    _check_auc(rng)
    failed = [r for r in _RESULTS if not r[2]]
    if failed:
        print("FAILED: " + ", ".join(f"[{r[0]}] {r[1]}" for r in failed))
        return 1
    print("ALL SELF-CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
