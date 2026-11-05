#!/usr/bin/env python3
"""Learned association score for de-duplication: P(two sensor track segments are the same
real object), with leave-one-iteration-out (LOIO) validation on the 6 testing iterations.

Every extra copy of one drone reported as its own track costs GOSPA a false-track penalty and
every wrong merge costs a miss; the fusion tracker merges segments by this score.

Inputs
------
* data/derived/sensor_tracks_v2/<slug>__tracks.csv + <slug>__grid.npz (tools/sensor_tracks.py;
  schemas in its docstring). Grid values are float32 on disk and are used as float64 here.
* data/derived/sensor_noise_by_iteration.csv: per (sensor_key, testing iteration) bias and
  sigma of the sensor's positions against Dronetag. The "honest total" variance of a sensor
  per axis is its RMS error about zero, s2 = n-weighted mean over iterations of
  (bias^2 + sigma^2): sensor errors are slow biases, so the distance between two sensors'
  tracks of one drone is offset-dominated and the smoother's own grid variance (which
  shrinks with every measurement) badly understates it. A sensor without a finite (bias,
  sigma) row gets DEFAULT_SIGMA_M per axis. The per-sample honest variance per axis is
  V = max(grid var, s2) (it grows in gaps). s2 enters ov_nd_med, ov_share3s and ct_ext_nd
  (the "sigma features") and the hand rules. LOIO, exact: the table that builds the sigma
  features of testing iteration i inside a fit or prediction that must not see a set X of
  testing iterations pools the testing iterations outside X | {i} only. So the features
  written to <slug>__pairs.csv (and scored by the held-out model of fold i) use the table
  without i; the training rows of iteration j for outer fold k use the table without {j, k},
  and inside the inner fold j' of outer fold k without {j, k, j'}; evaluation iterations use
  all 6, and the training rows of the all-6 model use the table without their own iteration.
  The grid variances themselves (var_e / var_n, from sensor_tracks) are fitted upstream on
  all 6 testing iterations (sensor_tracks' docstring): an inherited, documented in-sample part.

Pair types (segments that have grid samples; seg_a < seg_b for overlap pairs)
----------------------------------------------------------------------------
(a) overlap: different sensor_key, >= MIN_OVERLAP_STEPS common grid steps and minimum
    horizontal distance over the common steps < MAX_MIN_DIST_M. Candidates come from grouping
    the grid samples by step and comparing only segments active at the same step.
    Features over the common steps:
      ov_hd_med, ov_hd_p10, ov_hd_p90, ov_hd_max   horizontal distance (m)
      ov_near25, ov_near50, ov_near100             share of common steps within 25/50/100 m
      ov_du_med        median |u_a - u_b| over steps where both are finite (NaN if none)
      ov_nd_med        median of de^2 / (Ve_a + Ve_b) + dn^2 / (Vn_a + Vn_b), V honest (above)
      ov_ndg_med       the same with the smoother's grid variances only
      ov_share3s       share of common steps with that normalised distance <= 9 ("within 3 sigma")
      ov_dv_med        median horizontal |v_a - v_b| (m/s)
      ov_s             common steps * 0.1 s
      ov_frac          common steps / min(grid samples of a, grid samples of b)
      dur_min, dur_max, nmeas_min, nmeas_max   the two segments' duration_s / n_meas (sorted,
                       so the pair is unordered)
      seg_<c>_min, seg_<c>_max  for c in SEG_COLS (tracks.csv per-segment features, all from
                       the sensor reports, none from ground truth): min / max over the two
                       segments ignoring NaN (NaN only if both are NaN). The label says "same"
                       only for two DRONE segments of one flight; two sensors' tracks of one
                       bird or car are label 0, so the score needs to know how drone-like both
                       segments are, not only how close they are.
(b) continuation: same sensor_key, t1_a < t0_b <= t1_a + CONT_MAX_GAP_S (measurement times).
    a's last grid sample is extrapolated at constant velocity to b's first grid sample:
      ct_gap_s         t0_b - t1_a
      ct_dt_s          time between those two grid samples
      ct_ext_dist      horizontal distance between the extrapolation and b's first state
      ct_ext_du        |u| difference of the same (NaN if either has no height)
      ct_ext_nd        normalised: sum over e, n of d^2 / (V_a + V_b + q |dt|^3 / 3),
                       q = sensor_tracks.Q_ACC_H_M2S3 (CV position-variance growth)
      ct_speed_a, ct_speed_b, ct_speed_diff   horizontal speeds (m/s) and |difference|
      ct_head_diff     heading difference (deg, 0..180); NaN if either speed < STATIONARY_MS
      dur_a, dur_b, nmeas_a, nmeas_b, seg_<c>_min, seg_<c>_max (as for overlap)
Both types: sid_<ID> = 1 if either segment's sensor_id is ID (an unordered one-hot).

Labels (testing): 1 if both segments are drone_air / drone_ground with the same best_flight;
0 otherwise (both drone with different flights, or at least one clutter). Evaluation: blank.

Model: one gbdt.GBDT per pair type. Outer LOIO: fold k is fitted on the other 5 testing
iterations; its config (of CONFIGS) is chosen by pooled log-loss of the inner LOIO OOF
predictions over those 5 (fit on 4, predict the 5th), and its gbdt.Isotonic calibrator is fitted
on the same inner OOF predictions of the chosen config. Evaluation pairs: the same procedure
with all 6 testing iterations. Nothing is fitted on evaluation pairs.
Hand rules: overlap "same" if ov_nd_med < CHI2_2_99 (chi-square(2) 99 % gate) and
ov_share3s >= 0.5; continuation "same" if ct_ext_nd < CHI2_2_99.
Headline metrics use deployable sensors only (C.deployable_sensor_ids(): both sensor ids
present in evaluation); LJJX / EUIG / MMJJ / NJQL pairs are a side note.

Outputs (data/derived/pairs/)
-----------------------------
<slug>__pairs.csv  one row per candidate pair, all 9 iterations:
    seg_a, seg_b, type ('overlap' | 'cont'), sensor_key_a, sensor_key_b, the features above
    (the other type's columns blank), the seg_* and sid_<ID> columns (every model input is in
    the table; the sigma features are the held-out iteration's, i.e. exactly what its fold
    model scored), hand (hand-rule 0/1), label (testing; blank on evaluation), p_same
    (calibrated held-out probability; evaluation: the all-6 model), p_raw (the same before
    isotonic calibration; an extra column for diagnostics), fold (testing: index 0..5 of the
    held-out iteration in sorted key order = the iteration itself; evaluation: -1)
pairs.md           per-fold AUC / precision / recall / F1, model vs hand rule, feature
                   importance, candidate recall, plain-language glossary.

Run: python3 tools/pair_features.py [--tracks DIR]
Numbered self-checks at the end; exit status 1 on any failure: (1) leakage guard over every fit
and prediction, including the iterations behind each sigma table; (2) features vs a per-pair
Python loop (also the sigma features rebuilt for a 2-iteration exclusion); (3) candidates vs a
brute-force scan; (4) model beats the hand rule in mean held-out F1 at p > 0.5 (deployable);
(5) calibration: pooled |mean p - observed| <= 0.05 AND every reliability bin with >= CAL_MIN_N
held-out pairs within CAL_BIN_TOL + 2 binomial SE (the pooled mean alone passes for p = 0 at a
1 % base rate); (6) runtime < 20 min.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuas_common as C  # noqa: E402
import gbdt as G  # noqa: E402
import sensor_tracks as ST  # noqa: E402

OUT = C.DERIVED / "pairs"
NOISE_ITER = C.DERIVED / "sensor_noise_by_iteration.csv"

STEP_S = 0.1                 # template grid spacing
MIN_OVERLAP_STEPS = 20       # 2 s of common grid steps
MAX_MIN_DIST_M = 300.0       # overlap candidates: min horizontal distance over the overlap
CONT_MAX_GAP_S = 30.0        # continuation: b starts at most this long after a ends
DEFAULT_SIGMA_M = 20.0       # per-axis honest sigma of a sensor without a noise row
CHI2_2_99 = 9.21             # chi-square(2) 99 % quantile
SIGMA3_ND = 9.0              # "within 3 sigma": normalised squared distance <= 9
HAND_SHARE = 0.5
STATIONARY_MS = ST.STATIONARY_MS
Q_ACC_H = ST.Q_ACC_H_M2S3
N_BRUTE = 300                # random pairs per type for the brute-force feature check
BRUTE_ITER = "testing/S4-I4"  # smallest testing iteration: brute-force candidate scan
SEED = 0
# <= 6 configs; chosen per outer fold on inner-LOIO log-loss only
CONFIGS = [
    dict(n_trees=120, learning_rate=0.1, max_depth=3, min_samples_leaf=20),
    dict(n_trees=120, learning_rate=0.1, max_depth=5, min_samples_leaf=20),
    dict(n_trees=120, learning_rate=0.1, max_depth=4, min_samples_leaf=100,
         subsample=0.8, colsample=0.8),
    dict(n_trees=250, learning_rate=0.05, max_depth=4, min_samples_leaf=50),
]

# per-segment tracks.csv features (sensor-report based) of both segments; the most informative
# GT-free drone-vs-clutter features of tools/existence.py's segment model
SEG_COLS = ["cls_unknown", "cls_conf_mean", "cls_uav_rotary", "det_conf_mean", "range_min_m",
            "range_med_m", "dz_med_m", "dz_p90_m", "speed_mean_ms", "u_med_m"]
SEG_FEATS = [f"seg_{c}_{s}" for c in SEG_COLS for s in ("min", "max")]
SIGMA_FEATS = {"overlap": ["ov_nd_med", "ov_share3s"], "cont": ["ct_ext_nd"]}
CAL_EDGES = [0, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0001]   # reliability bins of p_same
CAL_MIN_N = 30               # self-check 5b: reliability bins with at least this many pairs
CAL_BIN_TOL = 0.10           # ... may miss by this much beyond 2 binomial standard errors
OV_FEATS = ["ov_hd_med", "ov_hd_p10", "ov_hd_p90", "ov_hd_max", "ov_near25", "ov_near50",
            "ov_near100", "ov_du_med", "ov_nd_med", "ov_ndg_med", "ov_share3s", "ov_dv_med",
            "ov_s", "ov_frac", "dur_min", "dur_max", "nmeas_min", "nmeas_max"]
CT_FEATS = ["ct_gap_s", "ct_dt_s", "ct_ext_dist", "ct_ext_du", "ct_ext_nd", "ct_speed_a",
            "ct_speed_b", "ct_speed_diff", "ct_head_diff", "dur_a", "dur_b", "nmeas_a", "nmeas_b"]
TYPES = ("overlap", "cont")


def log(msg: str) -> None:
    print(msg, flush=True)


def md_table(header, rows) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def short(key: str) -> str:
    return key.split("/")[-1] if key.startswith("testing") else key.split("/", 1)[1]


# --------------------------------------------------------------------------- #
# Honest per-sensor variance (LOIO over the testing iterations)
# --------------------------------------------------------------------------- #


_SIGMA_CACHE: dict = {}


def sigma_table(exclude) -> tuple[dict[str, tuple[float, float]], set[str]]:
    """{sensor_key: (s2_e, s2_n)} pooled over the testing rows of sensor_noise_by_iteration.csv
    whose iter_key is not in `exclude` (a set of iteration keys), and the set of iter_keys that
    contributed. Cached by the excluded set."""
    exclude = frozenset(exclude)
    if exclude in _SIGMA_CACHE:
        return _SIGMA_CACHE[exclude]
    t = C.read_columns(NOISE_ITER, numeric={"n", "bias_e_m", "bias_n_m", "sigma_e_m", "sigma_n_m"},
                       usecols=["sensor_key", "iter_key", "n", "bias_e_m", "bias_n_m",
                                "sigma_e_m", "sigma_n_m"])
    ok = ~np.isin(t["iter_key"], sorted(exclude)) & np.isfinite(t["sigma_e_m"]) & np.isfinite(t["sigma_n_m"]) \
        & np.isfinite(t["bias_e_m"]) & np.isfinite(t["bias_n_m"]) & (t["n"] > 0)
    ok &= np.array([str(k).startswith("testing/") for k in t["iter_key"]])
    tab, used = {}, set()
    for sk in sorted(set(t["sensor_key"][ok])):
        m = ok & (t["sensor_key"] == sk)
        w = t["n"][m]
        tab[sk] = (float(np.sum(w * (t["bias_e_m"][m] ** 2 + t["sigma_e_m"][m] ** 2)) / w.sum()),
                   float(np.sum(w * (t["bias_n_m"][m] ** 2 + t["sigma_n_m"][m] ** 2)) / w.sum()))
        used |= set(t["iter_key"][m])
    _SIGMA_CACHE[exclude] = (tab, used)
    return tab, used


def seg_s2(keys: np.ndarray, s2tab: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per sensor_key in `keys`: (s2_e, s2_n) of the table, DEFAULT_SIGMA_M^2 if absent."""
    s2 = np.array([s2tab.get(k, (DEFAULT_SIGMA_M ** 2,) * 2) for k in keys], dtype=np.float64).reshape(-1, 2)
    return s2[:, 0], s2[:, 1]


# --------------------------------------------------------------------------- #
# One iteration's segments and grid
# --------------------------------------------------------------------------- #


class Iter:
    """Segments (indexed by tracks row r) and grid samples (sorted by seg_id, step)."""

    def __init__(self, it: C.Iteration, tracks_dir: Path, s2tab: dict):
        self.key = it.key
        self.testing = it.has_groundtruth
        slug = C.iter_slug(it.key)
        tr = C.read_columns(tracks_dir / f"{slug}__tracks.csv",
                            numeric={"seg_id", "t0", "t1", "n_meas", "duration_s", "frac_matched", *SEG_COLS},
                            usecols=["seg_id", "sensor_key", "sensor_id", "t0", "t1", "n_meas",
                                     "duration_s", "best_flight", "label", "frac_matched"] + SEG_COLS)
        sid = tr["seg_id"].astype(np.int64)
        if np.any(np.diff(sid) <= 0):
            raise ValueError(f"{it.key}: tracks.csv seg_id not strictly increasing")
        self.seg_id = sid
        self.sensor_key = tr["sensor_key"].astype(str)
        self.sensor_id = tr["sensor_id"].astype(str)
        self.t0, self.t1 = tr["t0"], tr["t1"]
        self.n_meas, self.dur = tr["n_meas"], tr["duration_s"]
        self.segf = {c: tr[c].astype(np.float64) for c in SEG_COLS}
        self.label = tr["label"].astype(str)
        self.flight = tr["best_flight"].astype(str)
        self.is_drone = np.isin(self.label, ["drone_air", "drone_ground"])
        self.any_match = np.nan_to_num(tr["frac_matched"]) > 0     # >= 1 row matched to best_flight
        self.keys = np.array(sorted(set(self.sensor_key)))
        self.sens = np.searchsorted(self.keys, self.sensor_key)
        self.s2e, self.s2n = seg_s2(self.keys, s2tab)
        self.s2e, self.s2n = self.s2e[self.sens], self.s2n[self.sens]

        z = np.load(tracks_dir / f"{slug}__grid.npz")
        gseg = z["seg_id"].astype(np.int64)
        self.step = z["step"].astype(np.int64)
        if np.any(np.diff(gseg) < 0):
            raise ValueError(f"{it.key}: grid not sorted by seg_id")
        self.grow = np.searchsorted(sid, gseg)               # tracks row of each grid sample
        if not np.array_equal(sid[self.grow], gseg):
            raise ValueError(f"{it.key}: grid seg_id missing from tracks.csv")
        for k in ("t", "e", "n", "u", "ve", "vn", "vu", "var_e", "var_n"):
            setattr(self, k, z[k].astype(np.float64))
        self.Ve = np.maximum(self.var_e, self.s2e[self.grow])
        self.Vn = np.maximum(self.var_n, self.s2n[self.grow])
        nseg = sid.size
        self.g_cnt = np.bincount(self.grow, minlength=nseg)
        first = np.full(nseg, -1, dtype=np.int64)
        has = self.g_cnt > 0
        first[has] = np.searchsorted(self.grow, np.flatnonzero(has))
        self.g_first = first
        self.nseg = nseg


# --------------------------------------------------------------------------- #
# Overlap candidates
# --------------------------------------------------------------------------- #


def same_step_pairs(step: np.ndarray, row: np.ndarray):
    """Yield (i, j) grid-index arrays covering every pair of grid samples at the same step,
    each pair once, with row[i] < row[j] (a segment has at most one sample per step)."""
    o = np.lexsort((row, step))
    st = step[o]
    starts = np.flatnonzero(np.r_[True, st[1:] != st[:-1]])
    size = np.diff(np.r_[starts, st.size])
    pos = np.arange(st.size) - np.repeat(starts, size)
    rem = np.repeat(size, size) - pos - 1            # later members of the same step group
    by_rem = np.argsort(-rem, kind="stable")
    neg = -rem[by_rem]
    k = 1
    while True:
        m = int(np.searchsorted(neg, -k, side="right"))
        if m == 0:
            return
        idx = by_rem[:m]
        yield o[idx], o[idx + k]
        k += 1


def _reduce(key, cnt, mind):
    o = np.argsort(key, kind="stable")
    ks = key[o]
    st = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    return ks[st], np.add.reduceat(cnt[o], st), np.minimum.reduceat(mind[o], st)


def overlap_stats(D: Iter):
    """All different-sensor segment pairs with >= 1 common step: (key, n_common, min_dist);
    key = row_a * nseg + row_b with row_a < row_b."""
    ks, cs, ms = [], [], []
    for i, j in same_step_pairs(D.step, D.grow):
        ra, rb = D.grow[i], D.grow[j]
        m = D.sens[ra] != D.sens[rb]
        i, j, ra, rb = i[m], j[m], ra[m], rb[m]
        if i.size == 0:
            continue
        d = np.hypot(D.e[i] - D.e[j], D.n[i] - D.n[j])
        k_, c_, m_ = _reduce(ra * D.nseg + rb, np.ones(i.size, dtype=np.int64), d)
        ks.append(k_); cs.append(c_); ms.append(m_)
    if not ks:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    return _reduce(np.concatenate(ks), np.concatenate(cs), np.concatenate(ms))


def overlap_samples(D: Iter, cand: np.ndarray):
    """Grid-index pairs (I, J) of every common step of the candidate pairs (cand sorted keys),
    sorted by (pair, step)."""
    Is, Js = [], []
    for i, j in same_step_pairs(D.step, D.grow):
        key = D.grow[i] * D.nseg + D.grow[j]
        p = np.searchsorted(cand, key)
        p = np.minimum(p, cand.size - 1)
        m = cand[p] == key
        Is.append(i[m]); Js.append(j[m])
    I, J = np.concatenate(Is), np.concatenate(Js)
    o = np.lexsort((D.step[I], D.grow[I] * D.nseg + D.grow[J]))
    return I[o], J[o]


def _pair_onehot(ids_a, ids_b, all_ids):
    return {f"sid_{s}": ((ids_a == s) | (ids_b == s)).astype(np.float64) for s in all_ids}


def seg_pair_features(D: Iter, ra: np.ndarray, rb: np.ndarray) -> dict:
    """seg_<c>_min / _max over the two segments, NaN ignored (NaN only if both are NaN)."""
    f = {}
    for c in SEG_COLS:
        a, b = D.segf[c][ra], D.segf[c][rb]
        f[f"seg_{c}_min"], f[f"seg_{c}_max"] = np.fmin(a, b), np.fmax(a, b)
    return f


# Sigma features: the parts of the overlap / continuation features that depend on the sensor
# error table, kept per iteration so that they can be rebuilt for any excluded set (LOIO).

def ov_sigma_store(D: Iter, cand: np.ndarray, I: np.ndarray, J: np.ndarray) -> dict:
    return dict(P=cand.size, g=np.searchsorted(cand, D.grow[I] * D.nseg + D.grow[J]),
                de2=(D.e[I] - D.e[J]) ** 2, dn2=(D.n[I] - D.n[J]) ** 2,
                vei=D.var_e[I], vej=D.var_e[J], vni=D.var_n[I], vnj=D.var_n[J],
                si=D.sens[D.grow[I]], sj=D.sens[D.grow[J]], keys=D.keys)


def ov_sigma_features(st: dict, s2tab: dict) -> dict:
    s2e, s2n = seg_s2(st["keys"], s2tab)
    nd = st["de2"] / (np.maximum(st["vei"], s2e[st["si"]]) + np.maximum(st["vej"], s2e[st["sj"]])) \
        + st["dn2"] / (np.maximum(st["vni"], s2n[st["si"]]) + np.maximum(st["vnj"], s2n[st["sj"]]))
    cnt = np.bincount(st["g"], minlength=st["P"]).astype(np.float64)
    return dict(ov_nd_med=ST.group_quantile(st["g"], nd, st["P"], (0.5,))[0],
                ov_share3s=np.bincount(st["g"], (nd <= SIGMA3_ND).astype(float), minlength=st["P"]) / cnt)


def ct_sigma_store(D: Iter, A: np.ndarray, B: np.ndarray) -> dict:
    ia = D.g_first[A] + D.g_cnt[A] - 1
    ib = D.g_first[B]
    dt = D.t[ib] - D.t[ia]
    return dict(de=D.e[ia] + D.ve[ia] * dt - D.e[ib], dn=D.n[ia] + D.vn[ia] * dt - D.n[ib],
                grow=Q_ACC_H * np.abs(dt) ** 3 / 3.0, vea=D.var_e[ia], veb=D.var_e[ib],
                vna=D.var_n[ia], vnb=D.var_n[ib], s=D.sens[A], keys=D.keys)


def ct_sigma_features(st: dict, s2tab: dict) -> dict:
    s2e, s2n = seg_s2(st["keys"], s2tab)
    e2, n2 = s2e[st["s"]], s2n[st["s"]]
    return dict(ct_ext_nd=st["de"] ** 2 / (np.maximum(st["vea"], e2) + np.maximum(st["veb"], e2) + st["grow"])
                + st["dn"] ** 2 / (np.maximum(st["vna"], n2) + np.maximum(st["vnb"], n2) + st["grow"]))


def overlap_features(D: Iter, cand: np.ndarray, I: np.ndarray, J: np.ndarray) -> dict:
    P = cand.size
    ra, rb = cand // D.nseg, cand % D.nseg
    g = np.searchsorted(cand, D.grow[I] * D.nseg + D.grow[J])
    de, dn = D.e[I] - D.e[J], D.n[I] - D.n[J]
    d = np.hypot(de, dn)
    du = np.abs(D.u[I] - D.u[J])
    nd = de ** 2 / (D.Ve[I] + D.Ve[J]) + dn ** 2 / (D.Vn[I] + D.Vn[J])
    ndg = de ** 2 / (D.var_e[I] + D.var_e[J]) + dn ** 2 / (D.var_n[I] + D.var_n[J])
    dv = np.hypot(D.ve[I] - D.ve[J], D.vn[I] - D.vn[J])
    cnt = np.bincount(g, minlength=P).astype(np.float64)
    qd = ST.group_quantile(g, d, P, (0.5, 0.1, 0.9, 1.0))
    f = dict(ov_hd_med=qd[0], ov_hd_p10=qd[1], ov_hd_p90=qd[2], ov_hd_max=qd[3])
    for r in (25, 50, 100):
        f[f"ov_near{r}"] = np.bincount(g, (d <= r).astype(float), minlength=P) / cnt
    f["ov_du_med"] = ST.group_quantile(g, du, P, (0.5,))[0]
    f["ov_nd_med"] = ST.group_quantile(g, nd, P, (0.5,))[0]
    f["ov_ndg_med"] = ST.group_quantile(g, ndg, P, (0.5,))[0]
    f["ov_share3s"] = np.bincount(g, (nd <= SIGMA3_ND).astype(float), minlength=P) / cnt
    f["ov_dv_med"] = ST.group_quantile(g, dv, P, (0.5,))[0]
    f["ov_s"] = cnt * STEP_S
    f["ov_frac"] = cnt / np.minimum(D.g_cnt[ra], D.g_cnt[rb])
    f["dur_min"] = np.minimum(D.dur[ra], D.dur[rb])
    f["dur_max"] = np.maximum(D.dur[ra], D.dur[rb])
    f["nmeas_min"] = np.minimum(D.n_meas[ra], D.n_meas[rb])
    f["nmeas_max"] = np.maximum(D.n_meas[ra], D.n_meas[rb])
    f.update(seg_pair_features(D, ra, rb))
    return f


def overlap_hand(f: dict) -> np.ndarray:
    return ((f["ov_nd_med"] < CHI2_2_99) & (f["ov_share3s"] >= HAND_SHARE)).astype(np.float64)


# --------------------------------------------------------------------------- #
# Continuation candidates
# --------------------------------------------------------------------------- #


def cont_candidates(D: Iter) -> tuple[np.ndarray, np.ndarray]:
    """(row_a, row_b): same sensor_key, both with grid samples, t1_a < t0_b <= t1_a + gap."""
    A, B = [], []
    for s in np.unique(D.sens):
        rows = np.flatnonzero((D.sens == s) & (D.g_cnt > 0))
        rows = rows[np.argsort(D.t0[rows], kind="stable")]
        t0 = D.t0[rows]
        lo = np.searchsorted(t0, D.t1[rows], side="right")
        hi = np.searchsorted(t0, D.t1[rows] + CONT_MAX_GAP_S, side="right")
        n = hi - lo
        a = np.repeat(rows, n)
        b_pos = np.repeat(lo - np.cumsum(np.r_[0, n[:-1]]), n) + np.arange(n.sum())
        A.append(a); B.append(rows[b_pos])
    A, B = np.concatenate(A), np.concatenate(B)
    o = np.lexsort((B, A))
    return A[o], B[o]


def cont_features(D: Iter, A: np.ndarray, B: np.ndarray) -> dict:
    ia = D.g_first[A] + D.g_cnt[A] - 1              # a's last grid sample
    ib = D.g_first[B]                               # b's first grid sample
    dt = D.t[ib] - D.t[ia]
    pe, pn = D.e[ia] + D.ve[ia] * dt, D.n[ia] + D.vn[ia] * dt
    pu = D.u[ia] + D.vu[ia] * dt
    de, dn = pe - D.e[ib], pn - D.n[ib]
    grow = Q_ACC_H * np.abs(dt) ** 3 / 3.0
    sa, sb = np.hypot(D.ve[ia], D.vn[ia]), np.hypot(D.ve[ib], D.vn[ib])
    hd = np.abs((np.degrees(np.arctan2(D.ve[ia], D.vn[ia]) - np.arctan2(D.ve[ib], D.vn[ib])) + 180.0)
                % 360.0 - 180.0)
    hd[(sa < STATIONARY_MS) | (sb < STATIONARY_MS)] = np.nan
    f = dict(
        ct_gap_s=D.t0[B] - D.t1[A], ct_dt_s=dt, ct_ext_dist=np.hypot(de, dn),
        ct_ext_du=np.abs(pu - D.u[ib]),
        ct_ext_nd=de ** 2 / (D.Ve[ia] + D.Ve[ib] + grow) + dn ** 2 / (D.Vn[ia] + D.Vn[ib] + grow),
        ct_speed_a=sa, ct_speed_b=sb, ct_speed_diff=np.abs(sa - sb), ct_head_diff=hd,
        dur_a=D.dur[A], dur_b=D.dur[B], nmeas_a=D.n_meas[A], nmeas_b=D.n_meas[B])
    f.update(seg_pair_features(D, A, B))
    return f


def cont_hand(f: dict) -> np.ndarray:
    return (f["ct_ext_nd"] < CHI2_2_99).astype(np.float64)


def pair_labels(D: Iter, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if not D.testing:
        return np.full(A.size, np.nan)
    same = D.is_drone[A] & D.is_drone[B] & (D.flight[A] == D.flight[B])
    return same.astype(np.float64)


# --------------------------------------------------------------------------- #
# Brute-force references (self-checks 2 and 3): plain Python loops over grid samples
# --------------------------------------------------------------------------- #


def _py_quantile(vals: list[float], q: float) -> float:
    if not vals:
        return math.nan
    v = sorted(vals)
    pos = q * (len(v) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (pos - lo) * (v[hi] - v[lo])


def _py_s2(sensor_key: str, exclude) -> tuple[float, float]:
    """Honest per-axis variance recomputed row by row from the CSV (independent of sigma_table)."""
    import csv
    num = {"e": 0.0, "n": 0.0}
    den = 0.0
    with NOISE_ITER.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["sensor_key"] != sensor_key or r["iter_key"] in exclude or not r["iter_key"].startswith("testing/"):
                continue
            vals = [r["bias_e_m"], r["bias_n_m"], r["sigma_e_m"], r["sigma_n_m"], r["n"]]
            if any(v == "" for v in vals) or float(r["n"]) <= 0:
                continue
            be, bn, se, sn, n = map(float, vals)
            num["e"] += n * (be * be + se * se)
            num["n"] += n * (bn * bn + sn * sn)
            den += n
    if den == 0:
        return DEFAULT_SIGMA_M ** 2, DEFAULT_SIGMA_M ** 2
    return num["e"] / den, num["n"] / den


def _py_seg(D: Iter, ra: int, rb: int) -> dict:
    f = {}
    for c in SEG_COLS:
        v = [float(D.segf[c][r]) for r in (ra, rb) if D.segf[c][r] == D.segf[c][r]]
        f[f"seg_{c}_min"] = min(v) if v else math.nan
        f[f"seg_{c}_max"] = max(v) if v else math.nan
    return f


def _samples(D: Iter, r: int) -> list[int]:
    return [i for i in range(int(D.g_first[r]), int(D.g_first[r] + D.g_cnt[r]))]


def brute_overlap(D: Iter, ra: int, rb: int, exclude) -> dict:
    sa, sb = _samples(D, ra), _samples(D, rb)
    at_b = {int(D.step[i]): i for i in sb}
    s2a, s2b = _py_s2(D.sensor_key[ra], exclude), _py_s2(D.sensor_key[rb], exclude)
    d, du, nd, ndg, dv = [], [], [], [], []
    for i in sa:
        j = at_b.get(int(D.step[i]))
        if j is None:
            continue
        de, dn = float(D.e[i]) - float(D.e[j]), float(D.n[i]) - float(D.n[j])
        d.append(math.sqrt(de * de + dn * dn))
        x = abs(float(D.u[i]) - float(D.u[j]))
        if x == x:
            du.append(x)
        vea, vna = max(float(D.var_e[i]), s2a[0]), max(float(D.var_n[i]), s2a[1])
        veb, vnb = max(float(D.var_e[j]), s2b[0]), max(float(D.var_n[j]), s2b[1])
        nd.append(de * de / (vea + veb) + dn * dn / (vna + vnb))
        ndg.append(de * de / (float(D.var_e[i]) + float(D.var_e[j]))
                   + dn * dn / (float(D.var_n[i]) + float(D.var_n[j])))
        dv.append(math.hypot(float(D.ve[i]) - float(D.ve[j]), float(D.vn[i]) - float(D.vn[j])))
    n = len(d)
    f = dict(ov_hd_med=_py_quantile(d, 0.5), ov_hd_p10=_py_quantile(d, 0.1),
             ov_hd_p90=_py_quantile(d, 0.9), ov_hd_max=max(d) if d else math.nan)
    for r in (25, 50, 100):
        f[f"ov_near{r}"] = sum(1 for x in d if x <= r) / n
    f.update(ov_du_med=_py_quantile(du, 0.5), ov_nd_med=_py_quantile(nd, 0.5),
             ov_ndg_med=_py_quantile(ndg, 0.5), ov_share3s=sum(1 for x in nd if x <= SIGMA3_ND) / n,
             ov_dv_med=_py_quantile(dv, 0.5), ov_s=n * STEP_S, ov_frac=n / min(len(sa), len(sb)),
             dur_min=min(D.dur[ra], D.dur[rb]), dur_max=max(D.dur[ra], D.dur[rb]),
             nmeas_min=min(D.n_meas[ra], D.n_meas[rb]), nmeas_max=max(D.n_meas[ra], D.n_meas[rb]))
    f.update(_py_seg(D, ra, rb))
    return f


def brute_cont(D: Iter, ra: int, rb: int, exclude) -> dict:
    sa, sb = _samples(D, ra), _samples(D, rb)
    ia = max(sa, key=lambda i: int(D.step[i]))
    ib = min(sb, key=lambda i: int(D.step[i]))
    dt = float(D.t[ib]) - float(D.t[ia])
    de = float(D.e[ia]) + float(D.ve[ia]) * dt - float(D.e[ib])
    dn = float(D.n[ia]) + float(D.vn[ia]) * dt - float(D.n[ib])
    du = abs(float(D.u[ia]) + float(D.vu[ia]) * dt - float(D.u[ib]))
    s2 = _py_s2(D.sensor_key[ra], exclude)       # same sensor_key for both
    grow = Q_ACC_H * abs(dt) ** 3 / 3.0
    ve_ = max(float(D.var_e[ia]), s2[0]) + max(float(D.var_e[ib]), s2[0]) + grow
    vn_ = max(float(D.var_n[ia]), s2[1]) + max(float(D.var_n[ib]), s2[1]) + grow
    spa = math.hypot(float(D.ve[ia]), float(D.vn[ia]))
    spb = math.hypot(float(D.ve[ib]), float(D.vn[ib]))
    if spa < STATIONARY_MS or spb < STATIONARY_MS:
        hd = math.nan
    else:
        ha = math.degrees(math.atan2(float(D.ve[ia]), float(D.vn[ia])))
        hb = math.degrees(math.atan2(float(D.ve[ib]), float(D.vn[ib])))
        hd = abs(ha - hb)
        hd = 360.0 - hd if hd > 180.0 else hd
    f = dict(ct_gap_s=D.t0[rb] - D.t1[ra], ct_dt_s=dt, ct_ext_dist=math.hypot(de, dn), ct_ext_du=du,
             ct_ext_nd=de * de / ve_ + dn * dn / vn_, ct_speed_a=spa, ct_speed_b=spb,
             ct_speed_diff=abs(spa - spb), ct_head_diff=hd, dur_a=D.dur[ra], dur_b=D.dur[rb],
             nmeas_a=D.n_meas[ra], nmeas_b=D.n_meas[rb])
    f.update(_py_seg(D, ra, rb))
    return f


def brute_candidates(D: Iter) -> tuple[dict, set, set]:
    """Python scan of all segment pairs: {(ra, rb): (n_common, min_dist)} for different-sensor
    pairs with >= 1 common step, the overlap candidate set, and the continuation candidate set."""
    segs = [r for r in range(D.nseg) if D.g_cnt[r] > 0]
    pos = {}
    for r in segs:
        pos[r] = {int(D.step[i]): (float(D.e[i]), float(D.n[i])) for i in _samples(D, r)}
    lo = {r: min(pos[r]) for r in segs}
    hi = {r: max(pos[r]) for r in segs}
    stats, ov, ct = {}, set(), set()
    for x, ra in enumerate(segs):
        for rb in segs[x + 1:]:
            same = D.sensor_key[ra] == D.sensor_key[rb]
            if same:
                for a, b in ((ra, rb), (rb, ra)):
                    if D.t1[a] < D.t0[b] <= D.t1[a] + CONT_MAX_GAP_S:
                        ct.add((a, b))
                continue
            if lo[ra] > hi[rb] or lo[rb] > hi[ra]:
                continue
            pa, pb = pos[ra], pos[rb]
            n, m = 0, math.inf
            for s, (e, nn) in pa.items():
                q = pb.get(s)
                if q is not None:
                    n += 1
                    m = min(m, math.hypot(e - q[0], nn - q[1]))
            if n:
                stats[(ra, rb)] = (n, m)
                if n >= MIN_OVERLAP_STEPS and m < MAX_MIN_DIST_M:
                    ov.add((ra, rb))
    return stats, ov, ct


# --------------------------------------------------------------------------- #
# Model: GBDT + isotonic, nested LOIO
# --------------------------------------------------------------------------- #

CONT_NEAR_M = 150.0          # continuation training rows: every negative with ct_ext_dist < this ...
CONT_FAR_RATE = 0.02         # ... and this share of the farther negatives (weight 1 / rate); all positives


class Trainer:
    """Fits for one pair type. Pool = training rows of the testing iterations (it_idx = index
    of the iteration in `keys`), with sample weights. Fits are cached by (excluded set, config).

    The sigma feature columns (var_idx) are rebuilt per use: in a fit that excludes the set X,
    the rows of iteration i get sigma(i, X | {i}); a prediction for iteration j inside the
    inner loop of outer fold k gets sigma(j, {j, k}). sigma(i, excl) -> (values of the rows of
    i, iteration keys that built the table). Every fit and prediction is recorded with the
    iterations its rows and its sigma tables came from (self-check 1)."""

    def __init__(self, X, y, w, it_idx, keys, var_idx, sigma):
        self.X, self.y, self.w, self.it_idx, self.keys = X, y, w, it_idx, keys
        self.var_idx, self.sigma = var_idx, sigma
        self.cache: dict = {}
        self.records: list[dict] = []          # every fit / prediction, for the leakage guard
        self.n_fits = 0

    def rows_X(self, i: int, excl: frozenset):
        """X of the rows of iteration i with the sigma columns built without excl | {i}."""
        rows = np.flatnonzero(self.it_idx == i)
        X = self.X[rows].copy()
        v, used = self.sigma(i, frozenset(self.keys[j] for j in excl | {i}))
        X[:, self.var_idx] = v
        return rows, X, used

    def fit(self, excl: frozenset, ci: int):
        ck = (excl, ci)
        if ck not in self.cache:
            parts = [self.rows_X(i, excl) for i in range(len(self.keys)) if i not in excl]
            rows = np.concatenate([r for r, _, _ in parts])
            X = np.vstack([x for _, x, _ in parts])
            m = G.GBDT(seed=SEED, **CONFIGS[ci]).fit(X, self.y[rows], sample_weight=self.w[rows])
            self.cache[ck] = (m, frozenset(self.keys[i] for i in np.unique(self.it_idx[rows])),
                              frozenset().union(*[u for _, _, u in parts]))
            self.n_fits += 1
        return self.cache[ck]

    def nested(self, outer: int | None):
        """Inner LOIO over the testing iterations other than `outer`: config choice and
        isotonic calibration on OOF predictions, then the final fit on all of them."""
        inner = [j for j in range(len(self.keys)) if j != outer]
        base = frozenset() if outer is None else frozenset([outer])
        losses, oofs = [], []
        for ci in range(len(CONFIGS)):
            oof = np.full(self.y.size, np.nan)
            for j in inner:
                m, trained, t_sig = self.fit(base | {j}, ci)
                rows, X, p_sig = self.rows_X(j, base)
                oof[rows] = m.predict_proba(X)
                self.records.append(dict(outer=outer, purpose="inner", trained=trained, trained_sigma=t_sig,
                                         predicted=self.keys[j], predicted_sigma=p_sig))
            sel = ~np.isnan(oof)
            losses.append(G.logloss(self.y[sel], oof[sel], self.w[sel]))
            oofs.append(oof)
        best = int(np.argmin(losses))
        sel = ~np.isnan(oofs[best])
        iso = G.Isotonic().fit(oofs[best][sel], self.y[sel], self.w[sel])
        final, trained, t_sig = self.fit(base, best)
        self.records.append(dict(outer=outer, purpose="final", trained=trained, trained_sigma=t_sig,
                                 predicted=None if outer is None else self.keys[outer], predicted_sigma=frozenset()))
        return best, losses, iso, final, oofs[best][sel], self.y[sel], self.w[sel]


def prf(y: np.ndarray, yhat: np.ndarray) -> tuple[float, float, float]:
    tp = float(np.sum((y == 1) & (yhat == 1)))
    fp = float(np.sum((y == 0) & (yhat == 1)))
    fn = float(np.sum((y == 1) & (yhat == 0)))
    prec = tp / (tp + fp) if tp + fp else math.nan
    rec = tp / (tp + fn) if tp + fn else math.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else math.nan
    return prec, rec, f1


def safe_auc(y, s) -> float:
    if y.size == 0 or y.min() == y.max():
        return math.nan
    return G.auc(y, s)


def fmt(v, nd=3) -> str:
    return "" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def strcol(v: np.ndarray) -> list[str]:
    return ["" if x != x else f"{x:.7g}" for x in v.tolist()]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tracks", type=Path, default=C.DERIVED / "sensor_tracks_v2",
                    help="directory with <slug>__tracks.csv + <slug>__grid.npz")
    args = ap.parse_args()
    T0 = time.time()
    rng = np.random.default_rng(SEED)
    fails: list[str] = []

    def check(num: str, name: str, ok: bool, detail: str) -> None:
        log(f"[{'ok' if ok else 'FAIL'}] {num} {name}: {detail}")
        if not ok:
            fails.append(num)

    its = C.iterations()
    test_keys = [it.key for it in its if it.has_groundtruth]
    deploy = C.deployable_sensor_ids()
    all_ids: set[str] = set()
    for it in its:
        all_ids |= set(C.read_columns(args.tracks / f"{C.iter_slug(it.key)}__tracks.csv",
                                      usecols=["sensor_id"])["sensor_id"].astype(str))
    all_ids = sorted(all_ids)
    log(f"sensor ids: {' '.join(all_ids)}; deployable: {' '.join(sorted(deploy))}")

    per: dict[str, dict] = {}          # per iteration: pair tables (features, labels, ...)
    sigma_used: dict[str, set] = {}
    brute_ov_err, brute_ct_err, brute_n = [0.0, ""], [0.0, ""], [0, 0]
    brute_var_err, brute_var_n, store_err = [0.0, ""], 0, 0.0
    cand_check = None
    n_brute_it = -(-N_BRUTE // len(its))

    for it in its:
        t_it = time.time()
        excl = frozenset([it.key]) if it.has_groundtruth else frozenset()
        tab, used = sigma_table(excl)
        sigma_used[it.key] = used
        D = Iter(it, args.tracks, tab)
        k, c, mn = overlap_stats(D)
        cand = k[(c >= MIN_OVERLAP_STEPS) & (mn < MAX_MIN_DIST_M)]
        I, J = overlap_samples(D, cand)
        fo = overlap_features(D, cand, I, J)
        ra, rb = cand // D.nseg, cand % D.nseg
        A, B = cont_candidates(D)
        fc = cont_features(D, A, B)
        rec = dict(D_key=it.key, testing=it.has_groundtruth)
        if it.has_groundtruth:       # sigma features rebuilt per excluded set (training rows)
            rec["store"] = dict(overlap=ov_sigma_store(D, cand, I, J), cont=ct_sigma_store(D, A, B))
            for typ, fn, f in (("overlap", ov_sigma_features, fo), ("cont", ct_sigma_features, fc)):
                g = fn(rec["store"][typ], tab)
                for cname in SIGMA_FEATS[typ]:
                    a_, b_ = g[cname], f[cname]
                    if not np.array_equal(np.isnan(a_), np.isnan(b_)):
                        store_err = math.inf
                    elif a_.size:
                        ok_ = ~np.isnan(a_)
                        store_err = max(store_err, float(np.max(np.abs(a_[ok_] - b_[ok_]) / np.maximum(1.0, np.abs(b_[ok_])), initial=0.0)))
        for typ, a, b, f, hand in (("overlap", ra, rb, fo, overlap_hand(fo)), ("cont", A, B, fc, cont_hand(fc))):
            dep = np.isin(D.sensor_id[a], list(deploy)) & np.isin(D.sensor_id[b], list(deploy))
            f.update(_pair_onehot(D.sensor_id[a], D.sensor_id[b], all_ids))
            y = pair_labels(D, a, b)
            both_drone = D.is_drone[a] & D.is_drone[b]
            rec[typ] = dict(seg_a=D.seg_id[a], seg_b=D.seg_id[b], sk_a=D.sensor_key[a], sk_b=D.sensor_key[b],
                            f=f, hand=hand, y=y, dep=dep, neg_flight=both_drone & (y == 0),
                            n_clutter=(~D.is_drone[a]).astype(int) + (~D.is_drone[b]).astype(int),
                            weak_same=(y == 0) & D.any_match[a] & D.any_match[b] & (D.flight[a] == D.flight[b]),
                            p=np.full(a.size, np.nan), p_raw=np.full(a.size, np.nan),
                            fold=np.full(a.size, -1, dtype=np.int64))
        # candidate recall: true same-flight different-sensor pairs with >= 2 s common steps
        if it.has_groundtruth:
            long_ = c >= MIN_OVERLAP_STEPS
            ta, tb = k[long_] // D.nseg, k[long_] % D.nseg
            yt = pair_labels(D, ta, tb) == 1
            dep_t = np.isin(D.sensor_id[ta], list(deploy)) & np.isin(D.sensor_id[tb], list(deploy))
            found = mn[long_] < MAX_MIN_DIST_M
            # same-sensor simultaneous duplicates (not candidates by definition; side note)
            ks2, cs2 = [], []
            for i, j in same_step_pairs(D.step, D.grow):
                ga, gb = D.grow[i], D.grow[j]
                m = (D.sens[ga] == D.sens[gb]) & D.is_drone[ga] & D.is_drone[gb] & (D.flight[ga] == D.flight[gb])
                if m.any():
                    ks2.append(ga[m] * D.nseg + gb[m])
            ks2 = np.concatenate(ks2) if ks2 else np.zeros(0, np.int64)
            u2, c2 = np.unique(ks2, return_counts=True)
            rec["recall"] = dict(true=int(yt.sum()), found=int((yt & found).sum()),
                                 true_dep=int((yt & dep_t).sum()), found_dep=int((yt & dep_t & found).sum()),
                                 same_sensor_dups=int((c2 >= MIN_OVERLAP_STEPS).sum()),
                                 missed=[(D.sensor_key[x], D.sensor_key[y_], float(m_))
                                         for x, y_, m_ in zip(ta[yt & ~found], tb[yt & ~found], mn[long_][yt & ~found])])
        # self-check 2 sample: brute-force features on random pairs of this iteration; on testing
        # also the sigma features rebuilt without {this, the next testing iteration}
        if it.has_groundtruth:
            other = test_keys[(test_keys.index(it.key) + 1) % len(test_keys)]
            excl_v = frozenset([it.key, other])
            tab_v = sigma_table(excl_v)[0]
            var_f = dict(overlap=ov_sigma_features(rec["store"]["overlap"], tab_v),
                         cont=ct_sigma_features(rec["store"]["cont"], tab_v))
        for typ, brute, err, bi in (("overlap", brute_overlap, brute_ov_err, 0), ("cont", brute_cont, brute_ct_err, 1)):
            src = (ra, rb) if typ == "overlap" else (A, B)
            n = src[0].size
            if n == 0:
                continue
            pick = rng.choice(n, size=min(n_brute_it, n), replace=False)
            for p_ in pick:
                ref = brute(D, int(src[0][p_]), int(src[1][p_]), excl)
                for name, v in ref.items():
                    got = rec[typ]["f"][name][p_]
                    if (v != v) and (got != got):
                        continue
                    e = abs(got - v) / max(1.0, abs(v)) if (v == v and got == got) else math.inf
                    if e > err[0] or (e == err[0] and not err[1]):
                        err[0], err[1] = e, f"{short(it.key)} {name}"
                brute_n[bi] += 1
                if it.has_groundtruth:
                    ref_v = brute(D, int(src[0][p_]), int(src[1][p_]), excl_v)
                    for name in SIGMA_FEATS[typ]:
                        v, got = ref_v[name], var_f[typ][name][p_]
                        if (v != v) and (got != got):
                            continue
                        e = abs(got - v) / max(1.0, abs(v)) if (v == v and got == got) else math.inf
                        if e > brute_var_err[0] or (e == brute_var_err[0] and not brute_var_err[1]):
                            brute_var_err[0], brute_var_err[1] = e, f"{short(it.key)} {name}"
                    brute_var_n += 1
        # self-check 3: brute-force candidate scan on one small testing iteration
        if it.key == BRUTE_ITER:
            t_b = time.time()
            stats, bov, bct = brute_candidates(D)
            vec_stats = {(int(a // D.nseg), int(a % D.nseg)): (int(cc), float(mm)) for a, cc, mm in zip(k, c, mn)}
            vov = {(int(a), int(b)) for a, b in zip(ra, rb)}
            vct = {(int(a), int(b)) for a, b in zip(A, B)}
            stat_ok = stats.keys() == vec_stats.keys() and all(
                stats[q][0] == vec_stats[q][0] and abs(stats[q][1] - vec_stats[q][1]) <= 1e-9 for q in stats)
            cand_check = dict(ov_b=len(bov), ov_v=len(vov), ov_miss=len(bov - vov), ov_extra=len(vov - bov),
                              ct_b=len(bct), ct_v=len(vct), ct_miss=len(bct - vct), ct_extra=len(vct - bct),
                              n_stats=len(stats), stat_ok=stat_ok, secs=time.time() - t_b)
        per[it.key] = rec
        log(f"{it.key}: {D.nseg} segments, {D.e.size} grid samples; overlap candidates {cand.size} "
            f"({I.size} common samples), continuation candidates {A.size}  [{time.time() - t_it:.1f}s]")
        del D, I, J

    # ------------------------------------------------------------------ models
    sid_cols = [f"sid_{s}" for s in all_ids]
    feats = {"overlap": OV_FEATS + SEG_FEATS + sid_cols, "cont": CT_FEATS + SEG_FEATS + sid_cols}
    results: dict[str, dict] = {}
    for typ in TYPES:
        t_m = time.time()
        Xs, ys, ws, idx = [], [], [], []
        for ti, key in enumerate(test_keys):
            r = per[key][typ]
            X = np.column_stack([r["f"][c] for c in feats[typ]])
            y = r["y"]
            if typ == "cont":
                u = np.random.default_rng([SEED, ti]).random(y.size)
                near = r["f"]["ct_ext_dist"] < CONT_NEAR_M
                keep = (y == 1) | near | (u < CONT_FAR_RATE)
                w = np.where((y == 1) | near, 1.0, 1.0 / CONT_FAR_RATE)
            else:
                keep = np.ones(y.size, dtype=bool)
                w = np.ones(y.size)
            r["train_rows"] = int(keep.sum())
            r["keep"] = keep
            Xs.append(X[keep]); ys.append(y[keep]); ws.append(w[keep]); idx.append(np.full(int(keep.sum()), ti))
        sig_fn = ov_sigma_features if typ == "overlap" else ct_sigma_features
        sig_cache: dict = {}

        def sigma(i: int, excl_keys: frozenset, typ=typ, sig_fn=sig_fn, sig_cache=sig_cache):
            ck = (i, excl_keys)
            if ck not in sig_cache:
                r_ = per[test_keys[i]][typ]
                tab_, used_ = sigma_table(excl_keys)
                g_ = sig_fn(per[test_keys[i]]["store"][typ], tab_)
                sig_cache[ck] = (np.column_stack([g_[c] for c in SIGMA_FEATS[typ]])[r_["keep"]], used_)
            return sig_cache[ck]

        tr = Trainer(np.vstack(Xs), np.concatenate(ys), np.concatenate(ws), np.concatenate(idx), test_keys,
                     [feats[typ].index(c) for c in SIGMA_FEATS[typ]], sigma)
        del Xs
        res = dict(folds={}, cal=[], trainer=tr)
        for ti, key in enumerate(test_keys):
            best, losses, iso, final, oof_p, oof_y, oof_w = tr.nested(ti)
            r = per[key][typ]
            X = np.column_stack([r["f"][c] for c in feats[typ]])
            r["p_raw"] = final.predict_proba(X)
            r["p"] = iso.predict(r["p_raw"])
            r["fold"][:] = ti
            res["folds"][key] = dict(best=best, losses=losses)
            log(f"  {typ} fold {short(key)}: config {best} (inner log-loss "
                f"{' '.join(f'{l:.4f}' for l in losses)})  [{time.time() - t_m:.0f}s, {tr.n_fits} fits]")
        best, losses, iso, final, _, _, _ = tr.nested(None)
        res["eval_cfg"], res["eval_losses"], res["final"] = best, losses, final
        for it in its:
            if it.has_groundtruth:
                continue
            r = per[it.key][typ]
            X = np.column_stack([r["f"][c] for c in feats[typ]])
            r["p_raw"] = final.predict_proba(X)
            r["p"] = iso.predict(r["p_raw"])
        gain = final.feature_importance("gain")
        res["importance"] = sorted(zip(feats[typ], gain / max(gain.sum(), 1e-300)), key=lambda t: -t[1])
        res["secs"] = time.time() - t_m
        results[typ] = res
        log(f"{typ}: {tr.y.size} training rows ({int(tr.y.sum())} positive), {tr.n_fits} fits, "
            f"evaluation config {best}  [{res['secs']:.0f}s]")

    # ------------------------------------------------------------------ metrics
    metrics: dict[str, list] = {t: [] for t in TYPES}
    for typ in TYPES:
        score_name = "ov_nd_med" if typ == "overlap" else "ct_ext_nd"
        for key in test_keys:
            r = per[key][typ]
            for scope, m in (("deployable", r["dep"]), ("all", np.ones(r["y"].size, dtype=bool))):
                y, p, h = r["y"][m], r["p"][m], r["hand"][m]
                mp, mr, mf = prf(y, (p > 0.5).astype(float))
                hp, hr, hf = prf(y, h)
                metrics[typ].append(dict(key=key, scope=scope, n=int(y.size), pos=int(y.sum()),
                                         auc=safe_auc(y, p), prec=mp, rec=mr, f1=mf,
                                         h_auc=safe_auc(y, -r["f"][score_name][m]), h_prec=hp, h_rec=hr, h_f1=hf,
                                         n_pred=int((p > 0.5).sum()), n_hand=int(h.sum())))

    # ------------------------------------------------------------------ outputs
    OUT.mkdir(parents=True, exist_ok=True)
    feat_cols = OV_FEATS + CT_FEATS + SEG_FEATS + sid_cols
    out_cols = ["seg_a", "seg_b", "type", "sensor_key_a", "sensor_key_b"] + feat_cols + \
        ["hand", "label", "p_same", "p_raw", "fold"]
    for it in its:
        rec = per[it.key]
        cols: dict[str, list] = {c: [] for c in out_cols}
        for typ in TYPES:
            r = rec[typ]
            n = r["y"].size
            cols["seg_a"] += [str(x) for x in r["seg_a"].tolist()]
            cols["seg_b"] += [str(x) for x in r["seg_b"].tolist()]
            cols["type"] += [typ] * n
            cols["sensor_key_a"] += list(r["sk_a"])
            cols["sensor_key_b"] += list(r["sk_b"])
            for c in feat_cols:
                cols[c] += strcol(r["f"][c]) if c in r["f"] else [""] * n
            cols["hand"] += [str(int(x)) for x in r["hand"].tolist()]
            cols["label"] += ["" if x != x else str(int(x)) for x in r["y"].tolist()]
            cols["p_same"] += strcol(r["p"])
            cols["p_raw"] += strcol(r["p_raw"])
            cols["fold"] += [str(x) for x in r["fold"].tolist()]
        C.write_columns(OUT / f"{C.iter_slug(it.key)}__pairs.csv", cols, out_cols)
    log(f"wrote {len(its)} pair tables to {OUT}")

    # ------------------------------------------------------------------ self-checks
    log("\nSelf-checks")
    # 1. leakage guard
    bad = []
    n_sig_tables: set = set()
    for typ in TYPES:
        for rcd in results[typ]["trainer"].records:
            if any(k.startswith("evaluation/") for k in rcd["trained"]):
                bad.append(f"{typ}: evaluation iteration in a training set")
            if rcd["outer"] is not None and test_keys[rcd["outer"]] in rcd["trained"]:
                bad.append(f"{typ}: held-out {test_keys[rcd['outer']]} in a training set ({rcd['purpose']})")
            if rcd["predicted"] is not None and rcd["predicted"] in rcd["trained"]:
                bad.append(f"{typ}: {rcd['predicted']} predicted by a model trained on it")
            # sigma tables: neither the outer held-out nor the predicted iteration may have built
            # the error table of any training row or of the predicted rows
            for k_ in ([test_keys[rcd["outer"]]] if rcd["outer"] is not None else []) + \
                    ([rcd["predicted"]] if rcd["predicted"] is not None else []):
                if k_ in rcd["trained_sigma"] or k_ in rcd["predicted_sigma"]:
                    bad.append(f"{typ}: {k_}'s noise rows in a sigma table of a fit / prediction "
                               f"({rcd['purpose']}, outer {rcd['outer']})")
            if any(not k.startswith("testing/") for k in rcd["trained_sigma"] | rcd["predicted_sigma"]):
                bad.append(f"{typ}: non-testing rows in a sigma table")
            n_sig_tables.add(rcd["trained_sigma"])
        for key in test_keys:
            r = per[key][typ]
            if r["y"].size and not np.all(r["fold"] == test_keys.index(key)):
                bad.append(f"{typ}: {key} rows not all in their own fold")
        for it in its:
            if not it.has_groundtruth and np.any(per[it.key][typ]["fold"] != -1):
                bad.append(f"{typ}: evaluation rows with a testing fold")
    for key, used in sigma_used.items():
        if key in used or any(not k.startswith("testing/") for k in used):
            bad.append(f"sigma table of {key} used {sorted(used)}")
    n_rec = sum(len(results[t]["trainer"].records) for t in TYPES)
    check("1", "leakage guard (folds, calibration, sigma tables of training and predicted rows, evaluation "
          "never trained on)", not bad,
          f"{n_rec} fit/prediction records checked ({len(n_sig_tables)} distinct training sigma-table unions); "
          + ("; ".join(bad[:3]) if bad else "no violation"))
    # 2. brute-force features
    ok2 = brute_n[0] >= N_BRUTE and brute_n[1] >= N_BRUTE and brute_ov_err[0] <= 1e-6 and brute_ct_err[0] <= 1e-6 \
        and brute_var_n >= N_BRUTE and brute_var_err[0] <= 1e-6 and store_err <= 1e-12
    check("2", "features vs per-pair Python loop", ok2,
          f"overlap {brute_n[0]} pairs, max rel err {brute_ov_err[0]:.2e} ({brute_ov_err[1]}); "
          f"continuation {brute_n[1]} pairs, max rel err {brute_ct_err[0]:.2e} ({brute_ct_err[1]}); "
          f"sigma features rebuilt without 2 iterations: {brute_var_n} testing pairs, max rel err "
          f"{brute_var_err[0]:.2e} ({brute_var_err[1]}); rebuilt with the headline table vs headline "
          f"{store_err:.1e} (<= 1e-12); tolerance 1e-6 relative to max(1, |value|), >= {N_BRUTE} pairs each")
    # 3. candidate generation vs brute-force scan
    cc = cand_check
    ok3 = cc is not None and cc["ov_miss"] == 0 and cc["ov_extra"] == 0 and cc["ct_miss"] == 0 \
        and cc["ct_extra"] == 0 and cc["stat_ok"] and cc["ov_b"] > 0
    check("3", f"candidates vs brute-force scan on {BRUTE_ITER}", ok3,
          "missing iteration" if cc is None else
          f"overlap brute {cc['ov_b']} / vectorised {cc['ov_v']} (missed {cc['ov_miss']}, extra {cc['ov_extra']}); "
          f"continuation brute {cc['ct_b']} / vectorised {cc['ct_v']} (missed {cc['ct_miss']}, extra {cc['ct_extra']}); "
          f"common-step counts and min distances of {cc['n_stats']} overlapping pairs equal: {cc['stat_ok']} "
          f"[{cc['secs']:.0f}s]")
    # 4. model beats the hand rule (mean held-out F1, deployable sensors)
    summ = {}
    for typ in TYPES:
        rows = [m for m in metrics[typ] if m["scope"] == "deployable"]
        both = [m for m in rows if math.isfinite(m["f1"]) and math.isfinite(m["h_f1"])]
        mf = float(np.mean([m["f1"] for m in both])) if both else math.nan
        hf = float(np.mean([m["h_f1"] for m in both])) if both else math.nan
        summ[typ] = (mf, hf, len(both))
    ok4 = all(math.isfinite(summ[t][0]) and summ[t][0] > summ[t][1] for t in TYPES)
    check("4", "model beats the hand rule in mean held-out F1 at p > 0.5 (deployable sensors)", ok4,
          "; ".join(f"{t}: model {summ[t][0]:.3f} vs hand {summ[t][1]:.3f} over {summ[t][2]} folds" for t in TYPES))
    # 5. calibration pooled over held-out folds
    cal = {}
    for typ in TYPES:
        for scope in ("all", "deployable"):
            y = np.concatenate([per[k][typ]["y"][per[k][typ]["dep"] | (scope == "all")] for k in test_keys])
            p = np.concatenate([per[k][typ]["p"][per[k][typ]["dep"] | (scope == "all")] for k in test_keys])
            cal[(typ, scope)] = (float(p.mean()), float(y.mean()), y.size)
    ok5 = all(abs(cal[(t, s)][0] - cal[(t, s)][1]) <= 0.05 for t in TYPES for s in ("all", "deployable"))
    # 5b. the pooled mean alone is near-vacuous at base rates of 1 % / 0.01 % (p = 0 passes):
    # every reliability bin with >= CAL_MIN_N held-out pairs must match its observed share
    # within CAL_BIN_TOL beyond 2 binomial standard errors
    worst = {}
    for typ in TYPES:
        ya = np.concatenate([per[k][typ]["y"] for k in test_keys])
        pa = np.concatenate([per[k][typ]["p"] for k in test_keys])
        w_ = (-math.inf, "no bin")
        for lo, hi in zip(CAL_EDGES[:-1], CAL_EDGES[1:]):
            m = (pa >= lo) & (pa < hi)
            n = int(m.sum())
            if n < CAL_MIN_N:
                continue
            mp_, ob = float(pa[m].mean()), float(ya[m].mean())
            ex = abs(mp_ - ob) - 2.0 * math.sqrt(max(mp_ * (1 - mp_), 1e-12) / n)
            if ex > w_[0]:
                w_ = (ex, f"bin {lo:g}-{min(hi, 1):g}: n {n}, mean p {mp_:.3f} vs observed {ob:.3f}")
        worst[typ] = w_
    ok5 = ok5 and all(worst[t][0] <= CAL_BIN_TOL for t in TYPES)
    check("5", "calibration |mean p - observed| <= 0.05 pooled over held-out folds, and per p_same bin "
          f"(>= {CAL_MIN_N} pairs) within {CAL_BIN_TOL} + 2 SE", ok5,
          "; ".join(f"{t}/{s}: mean p {cal[(t, s)][0]:.4f} vs observed {cal[(t, s)][1]:.4f} (n {cal[(t, s)][2]})"
                    for t in TYPES for s in ("all", "deployable")) + "; worst bin excess over 2 SE: " +
          "; ".join(f"{t} {worst[t][0]:.3f} ({worst[t][1]})" for t in TYPES))
    runtime = time.time() - T0
    check("6", "runtime < 20 min", runtime < 1200, f"{runtime:.0f}s")

    write_md(per, results, metrics, summ, cal, worst, its, test_keys, cand_check, brute_n, runtime, fails)
    log(f"wrote {OUT / 'pairs.md'}")
    if fails:
        log(f"SELF-CHECK FAILURE: {', '.join(fails)}")
        return 1
    log("ALL SELF-CHECKS PASSED")
    return 0


def write_md(per, results, metrics, summ, cal, worst, its, test_keys, cc, brute_n, runtime, fails) -> None:
    L = ["# Pair association score: are two sensor track segments the same object?", ""]
    L.append(f"**Self-checks: {'ALL PASSED' if not fails else 'FAILED ' + ', '.join(fails)}** "
             f"(run log of `python3 tools/pair_features.py`; runtime {runtime / 60:.1f} min).")
    L += ["", "Generated by `tools/pair_features.py` (rules, features and output schema in its docstring). "
          "One table per iteration: `<slug>__pairs.csv`.", "",
          "## What this is, in plain words", "",
          "Each sensor reports its own tracks; one drone seen by three radars and an RF sensor shows up as four "
          "or more separate tracks, and one radar often drops a track and restarts it under a new id. "
          "If the fused picture keeps every copy, the score's track-accuracy part (GOSPA) charges a "
          "false-track penalty for each extra copy; if two different objects are merged, it charges a "
          "missed target. This step gives every candidate pair of track segments a probability `p_same` "
          "that both are the same real object, so the fusion tracker can merge pairs with high p_same.", "",
          "Two kinds of pairs:", "",
          f"* **overlap**: two segments from different sensors that exist at the same time for at least "
          f"{MIN_OVERLAP_STEPS * STEP_S:.0f} s and come within {MAX_MIN_DIST_M:.0f} m of each other at least once;",
          f"* **continuation (cont)**: one sensor's segment ends and another segment of the same sensor starts "
          f"at most {CONT_MAX_GAP_S:.0f} s later (a dropped and restarted track).", "",
          "Glossary:", "",
          "* **GOSPA**: the competition's track-accuracy distance between our tracks and the true drones; "
          "it adds position errors, a penalty per missed drone and a penalty per false track.",
          "* **LOIO (leave-one-iteration-out)**: to test on iteration k, everything (model, its settings, "
          "its calibration, and the sensor error table behind the normalised distances, both for k's own "
          "pairs and for every training pair) is fitted on the other 5 testing iterations only. "
          "The numbers below are therefore what to expect on an unseen iteration (one inherited exception: "
          "the smoother's own grid variances come from sensor_tracks, fitted upstream on all 6).",
          "* **GBDT**: gradient-boosted decision trees (`tools/gbdt.py`), the learned model.",
          "* **calibration / isotonic**: turning the model's raw score into a probability that means what it says "
          "(of pairs given p = 0.8, about 80 % are truly the same). Isotonic regression is a monotone "
          "step-wise map fitted on predictions for iterations the model did not train on.",
          "* **AUC**: probability that a random true same-object pair gets a higher score than a random "
          "different-object pair (1 = perfect ranking, 0.5 = coin flip). It ignores the threshold.",
          "* **precision**: of the pairs we would merge (p > 0.5), the share that truly are one object. "
          "**recall**: of the truly-same pairs, the share we would merge. **F1**: their harmonic mean "
          "(high only if both are high).",
          "* **hand rule**: the no-learning baseline. overlap: same if the median normalised squared distance "
          f"is < {CHI2_2_99} (the chi-square 2-degree-of-freedom 99 % gate: a true pair with the assumed "
          "errors exceeds it only 1 % of the time) and >= 50 % of common steps are within 3 sigma. "
          f"continuation: same if the constant-velocity extrapolation misses b's start by a normalised "
          f"squared distance < {CHI2_2_99}.",
          "* **normalised distance**: squared distance divided by the two sensors' error variances, per axis. "
          "The variance is the larger of the smoother's own variance and the sensor's honest RMS error "
          "about zero (bias^2 + sigma^2 from `sensor_noise_by_iteration.csv`, pooled over the other testing "
          f"iterations; {DEFAULT_SIGMA_M:.0f} m per axis when unknown): sensor errors are slow biases, so two "
          "sensors' tracks of one drone sit a roughly constant 10-40 m apart.",
          "* **deployable sensors**: sensor ids that exist in the evaluation data. LJJX (testing only) relays "
          "the friendly drones' own positions, so pairs involving testing-only sensors (EUIG, LJJX, MMJJ, NJQL) "
          "are excluded from the headline numbers.",
          "* **label**: 1 if both segments were matched to the same Dronetag flight (drone_air or drone_ground "
          "with the same best_flight in sensor_tracks), 0 otherwise (different flights, or clutter involved).", ""]

    L += ["## Pairs per iteration", "",
          "neg-flight: negatives where both segments are drones of different flights (the hard negatives); "
          "neg-clutter: at least one segment is clutter. train rows: continuation negatives farther than "
          f"{CONT_NEAR_M:.0f} m from the extrapolation are subsampled at {CONT_FAR_RATE:.0%} with weight "
          f"{1 / CONT_FAR_RATE:.0f} (keeps the fit fast; the weights keep it unbiased); every pair is still scored.", ""]
    rows = []
    for it in its:
        for typ in TYPES:
            r = per[it.key][typ]
            y = r["y"]
            t = it.has_groundtruth
            rows.append([short(it.key), typ, y.size, int(r["dep"].sum()),
                         int(np.nansum(y)) if t else "", int((r["neg_flight"]).sum()) if t else "",
                         int(np.sum((y == 0) & ~r["neg_flight"])) if t else "", int(r["hand"].sum()),
                         r.get("train_rows", "")])
    L += md_table(["iteration", "type", "pairs", "deployable", "positive", "neg-flight", "neg-clutter",
                   "hand says same", "train rows"], rows)

    L += ["", "## Candidate recall (overlap)", "",
          "Of all different-sensor segment pairs that truly belong to one flight and share >= 2 s of grid "
          f"steps, the share that became candidates (the only remaining condition is min distance < "
          f"{MAX_MIN_DIST_M:.0f} m). A pair missed here can never be merged by the score. "
          "same-sensor dups: pairs of segments of ONE sensor that track the same flight at the same time for "
          ">= 2 s (simultaneous duplicate tracks inside one sensor; not a pair type of this step).", ""]
    rows = []
    tt = tf = td = tfd = 0
    for key in test_keys:
        rc = per[key]["recall"]
        tt += rc["true"]; tf += rc["found"]; td += rc["true_dep"]; tfd += rc["found_dep"]
        rows.append([short(key), rc["true"], rc["found"], fmt(rc["found"] / rc["true"]) if rc["true"] else "",
                     rc["true_dep"], rc["found_dep"], fmt(rc["found_dep"] / rc["true_dep"]) if rc["true_dep"] else "",
                     rc["same_sensor_dups"]])
    rows.append(["all", tt, tf, fmt(tf / tt) if tt else "", td, tfd, fmt(tfd / td) if td else "", ""])
    L += md_table(["iteration", "true pairs", "candidates", "recall", "true (deployable)", "candidates (deployable)",
                   "recall (deployable)", "same-sensor dups"], rows)
    missed = [(short(k), *m) for k in test_keys for m in per[k]["recall"]["missed"]]
    if missed:
        L += ["", "Missed true pairs (iteration, sensors, min distance m): " +
              "; ".join(f"{a} {b}/{c} {d:.0f}" for a, b, c, d in missed[:20])]

    for typ in TYPES:
        L += ["", f"## Held-out results: {typ} pairs", "",
              "Each row: the model fitted without that iteration. Deployable sensors only. "
              "p > 0.5 = would merge. hand AUC ranks by the hand rule's normalised distance.", ""]
        rows = []
        for m in [m for m in metrics[typ] if m["scope"] == "deployable"]:
            rows.append([short(m["key"]), m["n"], m["pos"], fmt(m["auc"]), fmt(m["h_auc"]), m["n_pred"], fmt(m["prec"]),
                         fmt(m["rec"]), fmt(m["f1"]), m["n_hand"], fmt(m["h_prec"]), fmt(m["h_rec"]), fmt(m["h_f1"])])
        L += md_table(["held out", "pairs", "positive", "AUC", "hand AUC", "model merges", "precision", "recall", "F1",
                       "hand merges", "hand precision", "hand recall", "hand F1"], rows)
        mf, hf, nf = summ[typ]
        L += ["", f"Mean held-out F1 over the {nf} folds where both are defined: model **{mf:.3f}**, hand rule "
              f"**{hf:.3f}** (a fold without positives and without merges has no F1)."]
        # pooled
        ys = np.concatenate([per[k][typ]["y"][per[k][typ]["dep"]] for k in test_keys])
        ps = np.concatenate([per[k][typ]["p"][per[k][typ]["dep"]] for k in test_keys])
        hs = np.concatenate([per[k][typ]["hand"][per[k][typ]["dep"]] for k in test_keys])
        pp, pr, pf = prf(ys, (ps > 0.5).astype(float))
        hp, hr, hf_ = prf(ys, hs)
        L += [f"Pooled over all held-out folds (deployable): AUC {fmt(safe_auc(ys, ps))}, precision {fmt(pp)}, "
              f"recall {fmt(pr)}, F1 {fmt(pf)}; hand rule precision {fmt(hp)}, recall {fmt(hr)}, F1 {fmt(hf_)}."]
        rows = []
        for m in [m for m in metrics[typ] if m["scope"] == "all"]:
            rows.append([short(m["key"]), m["n"], m["pos"], fmt(m["auc"]), fmt(m["prec"]), fmt(m["rec"]), fmt(m["f1"]),
                         fmt(m["h_f1"])])
        L += ["", "Side note, all sensors (including testing-only ones; LJJX positives are leakage-prone):", ""]
        L += md_table(["held out", "pairs", "positive", "AUC", "precision", "recall", "F1", "hand F1"], rows)
        # calibration
        L += ["", f"Calibration (all held-out {typ} pairs, all sensors): mean p_same "
              f"{cal[(typ, 'all')][0]:.4f} vs observed same-share {cal[(typ, 'all')][1]:.4f}; deployable "
              f"{cal[(typ, 'deployable')][0]:.4f} vs {cal[(typ, 'deployable')][1]:.4f}. Reliability by p_same bin:", ""]
        ya = np.concatenate([per[k][typ]["y"] for k in test_keys])
        pa = np.concatenate([per[k][typ]["p"] for k in test_keys])
        rows = []
        for lo, hi in zip(CAL_EDGES[:-1], CAL_EDGES[1:]):
            m = (pa >= lo) & (pa < hi)
            if m.any():
                rows.append([f"{lo:g}-{min(hi, 1):g}", int(m.sum()), fmt(float(pa[m].mean()), 4),
                             fmt(float(ya[m].mean()), 4), int(ya[m].sum())])
        L += md_table(["p_same", "pairs", "mean p", "observed share", "true pairs"], rows)
        res = results[typ]
        L += ["", "Config chosen per fold (index into CONFIGS, by inner-LOIO log-loss): " +
              ", ".join(f"{short(k)} {res['folds'][k]['best']}" for k in test_keys) +
              f"; evaluation model {res['eval_cfg']}. CONFIGS: " +
              "; ".join(f"{i}: {c}" for i, c in enumerate(CONFIGS)), "",
              "Most informative features (share of total split gain in the model fitted on all 6 testing "
              "iterations, which scores the evaluation pairs):", ""]
        L += md_table(["feature", "gain share"], [[f, fmt(g)] for f, g in res["importance"][:12]])

    L += ["", "## Why a pair is called 'different': the label breakdown (testing, deployable)", "",
          "The label says 'same' only when BOTH segments are labelled drones of one flight. Two sensors "
          "tracking the same bird, car or unlabelled aircraft are therefore 'different' although they are "
          "one physical object, and no geometric feature can tell them apart from a true drone pair. "
          "weak-same: label 0, but both segments have at least one detection matched to the same flight "
          "(they missed the frac_matched >= 0.5 / frac_best >= 0.8 drone thresholds): probably label noise.", ""]
    rows = []
    for typ in TYPES:
        y = np.concatenate([per[k][typ]["y"][per[k][typ]["dep"]] for k in test_keys])
        nc = np.concatenate([per[k][typ]["n_clutter"][per[k][typ]["dep"]] for k in test_keys])
        ws = np.concatenate([per[k][typ]["weak_same"][per[k][typ]["dep"]] for k in test_keys])
        hd = np.concatenate([per[k][typ]["hand"][per[k][typ]["dep"]] for k in test_keys]) == 1
        pm = np.concatenate([per[k][typ]["p"][per[k][typ]["dep"]] for k in test_keys])
        for name, sel in (("hand rule says same", hd), ("p_same > 0.2", pm > 0.2), ("all candidates", np.ones(y.size, bool))):
            rows.append([typ, name, int(sel.sum()), int((sel & (y == 1)).sum()),
                         int((sel & (y == 0) & (nc == 0)).sum()), int((sel & (nc == 1)).sum()),
                         int((sel & (nc == 2)).sum()), int((sel & ws).sum())])
    L += md_table(["type", "subset", "pairs", "label 1", "0: drone/drone other flight", "0: drone/clutter",
                   "0: clutter/clutter", "of the 0s: weak-same"], rows)
    L += ["", "Drone/drone subset (both segments labelled drone; the question 'same flight?' alone), held-out, "
          "deployable, pooled over folds:", ""]
    rows = []
    for typ in TYPES:
        sel = [per[k][typ]["dep"] & (per[k][typ]["n_clutter"] == 0) for k in test_keys]
        y = np.concatenate([per[k][typ]["y"][m] for k, m in zip(test_keys, sel)])
        p = np.concatenate([per[k][typ]["p"][m] for k, m in zip(test_keys, sel)])
        h = np.concatenate([per[k][typ]["hand"][m] for k, m in zip(test_keys, sel)])
        sc = "ov_nd_med" if typ == "overlap" else "ct_ext_nd"
        s_ = np.concatenate([-per[k][typ]["f"][sc][m] for k, m in zip(test_keys, sel)])
        mp, mr, mf = prf(y, (p > 0.5).astype(float))
        hp, hr, hf_ = prf(y, h)
        rows.append([typ, y.size, int(y.sum()), fmt(safe_auc(y, p)), fmt(safe_auc(y, s_)), fmt(mp), fmt(mr), fmt(mf),
                     fmt(hp), fmt(hr), fmt(hf_)])
    L += md_table(["type", "pairs", "positive", "AUC", "hand AUC", "precision", "recall", "F1", "hand precision",
                   "hand recall", "hand F1"], rows)
    dd = {r_[0]: r_ for r_ in rows}

    L += ["", "## Self-check details", "",
          f"* features recomputed by a per-pair Python loop: {brute_n[0]} overlap and {brute_n[1]} continuation pairs;",
          f"* candidate scan on {BRUTE_ITER}: " + ("missing" if cc is None else
          f"overlap {cc['ov_b']} brute vs {cc['ov_v']} vectorised, continuation {cc['ct_b']} vs {cc['ct_v']}"),
          "", "## Caveats", "",
          f"* **Self-check 4 (model vs hand rule, mean held-out F1 at p > 0.5, deployable)**: overlap model "
          f"{summ['overlap'][0]:.3f} vs hand {summ['overlap'][1]:.3f}, continuation model {summ['cont'][0]:.3f} vs "
          f"hand {summ['cont'][1]:.3f}. The label calls two sensors' tracks of one non-drone object (or of a drone "
          "whose segments missed the drone thresholds) 'different', so the hand rule, which looks at geometry only, "
          "merges many clutter pairs (low precision). The model also sees how drone-like both segments are "
          "(seg_* features: classification shares and confidence, range, height above the sensor, speed), which "
          "is what lets it reach p > 0.5 at all.",
          f"* Within drone/drone pairs (table above) the geometric hand rule reaches F1 {dd['overlap'][10]} on "
          f"overlap pairs and {dd['cont'][10]} on continuation pairs (model {dd['overlap'][7]} / {dd['cont'][7]}): "
          "the remaining question there is 'which flight', and geometry answers most of it.",
          f"* **Self-check 5 (calibration per p_same bin)**: worst bin overlap {worst['overlap'][1]}; continuation "
          f"{worst['cont'][1]}. The pooled mean matches, but the isotonic map is fitted on inner-LOIO predictions "
          "of 5 iterations and the highest raw scores do not transfer cleanly to a new iteration, so p_same above "
          "0.5 overstates the same-drone share. Use p_same as a ranking (AUC in the tables) and pick the "
          "tracker's merge threshold by LOIO GOSPA, not by reading p_same > 0.5 as 'more likely than not'.",
          "* Continuation pairs: about 90 true pairs among 1.7 million candidates (base rate ~5e-5). Almost no "
          "pair reaches p_same > 0.5, so the model's F1 at that threshold is ~0, and the hand rule's is ~0.002 "
          "because it merges tens of thousands of pairs; F1 at p > 0.5 cannot separate them and AUC is the "
          "meaningful comparison for this type.",
          "* p_same is P(both segments are the SAME DRONE), not P(same physical object): two tracks of one bird get "
          "a low p_same. For de-duplicating clutter tracks a tracker should not rely on it; a 'same physical object' "
          "score would need clutter identity labels, which do not exist.",
          "* Few positives: the testing data holds only a few hundred true overlap pairs and about a hundred "
          "true continuations, and one iteration (S3-I1) has no true deployable overlap pair at all; "
          "per-fold numbers are noisy.",
          "* Labels come from sensor_tracks' segment labels (frac_matched >= 0.5, frac_best >= 0.8). A drone "
          "segment that the GT association missed counts as clutter, so some 'false' merges may be true.",
          "* The smoother's grid variances (var_e / var_n, used in ov_ndg_med and as the floor of the honest "
          "variance) come from sensor_tracks, whose total sigma is fitted on all 6 testing iterations: an "
          "inherited in-sample part that this step cannot remove.",
          "* p_same is calibrated for the candidate population defined here; it is not a probability for "
          "arbitrary pairs (a pair outside the gates is not a candidate)."]
    (OUT / "pairs.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
