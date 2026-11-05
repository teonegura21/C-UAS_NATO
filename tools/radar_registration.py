#!/usr/bin/env python3
"""GT-free radar registration: per-iteration estimate of each radar's slow position offset
(east, north, up) and azimuth rotation against tracks that carry the drone's own GPS,
validated on testing against the values measured with Dronetag ground truth.

Headline (2026-10-28 run, registration.md): the largest stable part of the EGOI / NYTF error is an
AZIMUTH ROTATION about the radar site (GT: EGOI +1.06 deg sd 0.05, NYTF +1.68 sd 0.24); UXYJ (+0.40
sd 0.18) and IUKL (-0.57 sd 0.88, not stable) rotate less and most of their horizontal error is
something else (the report gives the share each radar's rotation removes). Rotating each radar
back by the leave-one-iteration-out mean angle (method loio_az) lowers the median 3-D error of
drone radar samples on 6 of 6 testing iterations; the constant east / north offset the brief asked
for does not transfer (it varies from drone to drone because it is a rotation, the pairs are
impure, and the references report late), so it is reported but not recommended. The GT-free
intervals are sampling intervals and under-cover the GT offset (registration.md gives the counts).

Why: the sensor feeds are already tracker outputs whose error is mostly a slowly varying bias
(lag-1 autocorrelation 0.83-0.97 against Dronetag). The radars (EGOI, NYTF, UXYJ, IUKL) are
off by 10-25 m horizontally and more vertically, and that offset changes from iteration to
iteration. An offset calibrated with ground truth on the iteration being scored removes exactly
the error evaluation will have; on evaluation there is no ground truth, so the offset must be
estimated from the iteration's own sensor data. Cyber / passive-RF sensors that decode the
drone's own telemetry (STUG, CEQX, ILYS, IKJX, SMFS: 3-10 m) and UTM relays (VKYY, QTVS) are the
reference: where a radar track and a reference track follow the same moving object, the median
radar - reference difference is the radar's offset (up to the references' own error).

Inputs (estimator: no ground truth, no labels)
----------------------------------------------
* data/derived/sensor_tracks_v2/<slug>__tracks.csv (columns seg_id, iter_key, sensor_id,
  sensor_key only) and <slug>__grid.npz (seg_id, step, e, n, u, ve, vn): every segment of a
  reference or radar sensor of the iteration, on the template's 100 ms grid, local ENU (u is
  sensor_tracks' ellipsoidal height rule). The estimated offset is the RESIDUAL offset of these
  grid positions, i.e. what a downstream fusion would subtract from them.
* data/derived/detections.csv.gz (iter_key, sensor_id, sensor_e, sensor_n, sensor_pos_valid):
  the radar's site = median sensor_e / sensor_n of its valid rows in the iteration (the centre
  of the azimuth rotation and of the range / azimuth model).

Validation inputs (testing only, never seen by the estimator)
-------------------------------------------------------------
* tracks.csv labels (label, best_flight) and truth.load_truth(it, 'all') (Dronetag on the grid).

Reference sets (all deployable: C.deployable_sensor_ids(), checked)
--------------------------------------------------------------------
  noutm: STUG, CEQX, ILYS, IKJX, SMFS            (honest on testing)
  all:   noutm + VKYY, QTVS                       (VKYY / QTVS relay Dronetag itself in testing,
                                                   so their testing validation is leaky; this is
                                                   what evaluation will look like)
Radars: EGOI, NYTF, UXYJ, IUKL; EUIG (testing-only) is estimated and reported separately, never
a reference and never in the deployable summaries.

Estimator, per (iteration, radar, reference set)
------------------------------------------------
1. Reference samples: grid samples of the reference set whose smoothed horizontal speed is
   >= MOVING_MS. A parked drone (or a home point, which STUG also reports) cannot be told apart
   from ground clutter beside it, and a radar track on a moving object is what we want.
2. Candidates: every radar grid sample at the same grid step with horizontal distance
   <= PREGATE_M and horizontal velocity difference |v_r - v_ref| <= DV_GATE_MS. The velocity gate
   is insensitive to a constant position offset, so it does the association work that a
   position gate cannot do when the offset itself is 10-50 m.
3. Gate stages (iterative re-centering): stage 0 keeps candidates whose horizontal residual
   (radar - reference - current offset) is <= GATE0_M around offset 0; later stages keep
   residual <= GATE1_M around the current estimate. Per reference sample the nearest kept radar
   sample is chosen (ties: the earlier radar sample). A pair = (radar segment, reference
   segment) is accepted when it has >= MIN_PAIR_STEPS chosen steps AND they are >= MIN_SHARE of
   the reference segment's moving samples inside the radar segment's step range (one brief
   crossing is not a co-observation). The estimate is recomputed from the accepted pairs; stop
   when no axis moves by more than CONVERGE_M, at most MAX_STAGES stages.
   Why not pair_features' hand rule (ov_nd_med < 9.21 and ov_share3s >= 0.5)? Its normalised
   distance divides by an "honest" variance pooled from GT biases of other iterations (it is
   GT-derived) and it is the gate the bias itself inflates; the velocity gate + re-centred
   position gate is offset-equivariant and GT-free. The purity of the accepted pairs against
   GT labels is reported instead.
   If a later stage accepts no pair, the last stage that did is kept.
4. Co-observation groups: connected components of the accepted pairs over shared radar or
   reference segments (one radar segment matched to the STUG and the QTVS stream of one drone, or
   a reference segment matched to two radar fragments, is ONE group). Offset per axis = weighted
   median of the per-step differences radar - reference over the accepted pairs; a group's steps
   share the weight min(n, PAIR_CAP_STEPS) (one group counts as at most PAIR_CAP_STEPS steps =
   30 s). Vertical: only steps where both have a height, the cap counted on those steps; an axis
   with fewer than MIN_PAIR_STEPS steps has no estimate. Weighted median = the smallest value whose
   cumulative weight reaches half the total.
5. Uncertainty: block bootstrap over the groups, N_BOOT replicates; interval = estimate +-
   t975(G-1) sqrt(G/(G-1)) sd(replicates) per axis (G groups: a small-sample t interval on the
   bootstrap standard error; percentiles of 3-15 blocks are too narrow); only with >= MIN_GROUPS groups (else no interval and reliable = 0: one or two drones
   cannot show the drone-to-drone spread). It is a SAMPLING interval: it misses wrong pairs and
   the references' own error, and under-covers the GT offset on testing (counts in the report).
6. Range / azimuth model (reported, applied only in the effect check): with rel = reference -
   radar site (horizontal), d = b + k rel + delta (rel_n, -rel_e) (b constant offset, k range
   scale, delta azimuth rotation in radians, clockwise), fitted by Huber-weighted IRLS on the
   same steps and weights; only with >= MIN_ROT_GROUPS groups and an azimuth spread of rel
   (circular sd) >= MIN_ROT_AZ_SPREAD_DEG.
7. Lag-aware constant (gt_free_lag_*): radar - reference = b + tau v_ref (Huber IRLS), because
   the cyber / passive-RF references trail the drone by ~0.5-1.7 s (testing GT); needs a heading
   spread >= MIN_LAG_HEAD_SPREAD_DEG. Reported; it does not beat the plain median.
8. Azimuth rotation (gt_free_az_*): weighted median (same pair weights) of the azimuth
   difference radar - reference seen from the radar site over chosen steps whose reference is
   >= MIN_AZ_RANGE_M from the site. Rotation-centred gating (estimate_az): the radar is rotated
   back by the angle found so far and steps 2-4 are rerun until the residual angle is
   < AZ_CONVERGE_DEG (<= AZ_MAX_PASSES), so the pair selection does not depend on the rotation
   itself. Interval: block bootstrap of the last pass (shifted by the accumulated angle). The
   vertical offset of these rows is the median d_u of step 4 (same reference set).

Validation (testing; GT)
------------------------
* GT offset per (iteration, radar): median(radar - truth) per axis over the grid samples of the
  radar's drone_air segments (sensor_tracks_v2 label) where truth.load_truth(it, 'all') has the
  segment's best_flight airborne at that step. Needs >= MIN_GT_SAMPLES samples.
* gt_same: the same on the estimator's own chosen steps (drone segments only), and ref_err:
  median(reference - truth) on those steps: est ~ gt_same - ref_err separates sampling from the
  references' own error.
* Pair purity: share of accepted pairs whose radar and reference segments are both labelled
  drone with one best_flight.
* GT angle per (iteration, radar): median azimuth difference radar - truth about the site on the
  same samples (truth >= MIN_AZ_RANGE_M from the site); gt_after_az: the GT offset left after
  rotating back by the GT angle. Per-flight GT offsets measure the within-iteration spread.
* Stability: per radar, the spread across testing iterations of the GT offset / angle (sd) vs
  its size. LOIO (leave-one-iteration-out) values of testing iteration k = mean GT value of the
  other testing iterations (loio_pooled: offset; loio_az: angle + the pooled d_u; loio_az_const:
  angle + the pooled offset left after the angle); for evaluation the mean over all 6. Their
  interval is a 95 % prediction interval for a new iteration: mean +- t975(n-1) sd sqrt(1 + 1/n)
  (Student t quantile, n = 2-5 pooled iterations).
* Effect check (in memory): every drone_air radar grid sample (deployable radars, truth
  airborne) is corrected by each method (correct(): rotate about the site by -az_deg, then
  subtract d_e, d_n, d_u; no estimate -> no change); per iteration the median 3-D error
  (samples with a height) and horizontal error, and the share within 10 / 25 / 50 m (3-D).
  Methods: none, gt_free_noutm / _all, gt_free_lag_noutm / _all, gt_free_rot (noutm range /
  azimuth model where fitted, else no horizontal change), gt_free_az_noutm / _all, loio_pooled,
  loio_az, loio_az_const, and oracle / oracle_az / oracle_az_const (the iteration's own GT
  values: in-sample upper bounds, not methods).
In-sample caveat inherited from sensor_tracks_v2: its heights (u_ellipsoid offsets) and the
UXYJ clock offset were fitted with GT on all 6 testing iterations, so the vertical GT offsets
on testing are already partly removed in-sample and evaluation vertical offsets may be larger.

Outputs (data/derived/radar_registration/)
------------------------------------------
offsets.csv  one row per (iter_key, sensor_key, method). Correction of a radar position:
             rotate (e, n) about (site_e, site_n) by -az_deg (NaN -> no rotation), then subtract
             (d_e, d_n, d_u) (NaN -> 0); that is correct() below.
    iter_key, sensor_key (the radar's sensor id), method: 'gt_free_noutm' | 'gt_free_all'
                       (constant, step 4) | 'gt_free_lag_noutm' | 'gt_free_lag_all' (step 7) |
                       'gt_free_az_noutm' | 'gt_free_az_all' (step 8) | 'loio_pooled' |
                       'loio_az' | 'loio_az_const' (validation section)
    d_e, d_n, d_u      offset radar - reference (m); NaN = no estimate on that axis; az rows
                       have d_e = d_n = 0
    ci                 interval half-width, the largest over d_e / d_n / d_u with an interval (m);
                       gt_free_*: a bootstrap SAMPLING interval (not calibrated, see step 5);
                       loio_*: t-based 95 % prediction interval
    d_e_lo, d_e_hi, d_n_lo, d_n_hi, d_u_lo, d_u_hi   interval per axis (NaN: < MIN_GROUPS groups)
    n_pairs, n_groups, n_steps, n_steps_u   accepted pairs, their co-observation groups and
                       chosen steps (az rows: the rotation-centred pair set; loio_*: number of
                       testing iterations pooled, in n_pairs; n_groups 0)
    reliable           1 iff the row has an estimate and >= MIN_GROUPS groups (loio_*: >= MIN_GROUPS
                       pooled iterations); downstream use should skip reliable = 0 rows
    refs_used          reference sensor ids with accepted pairs (step 4), '|'-joined
    ci_blocks          'groups' | '' (bootstrap blocks; '' = no interval)
    testing_only       1 for EUIG
    az_deg, az_lo, az_hi  azimuth rotation (deg, clockwise positive) and its 95 % interval (az
                       and loio_az* rows; NaN elsewhere)
    site_e, site_n     the radar site (m, iteration ENU)
    lag_tau_s          relative lag radar - reference of step 7 (lag rows)
    rot_k, rot_delta_deg  range scale and rotation of the range / azimuth model of step 6
                       (gt_free_* rows of the reference set where fitted, else NaN)
    validation columns (testing; blank on evaluation; never inputs): gt_d_e, gt_d_n, gt_d_u,
                       gt_n (GT offset and its sample count), gt_az_deg (GT angle), err_h
                       (horizontal norm of offset - GT offset; NaN on az rows), err_az_deg
                       (|az_deg - GT angle|, az rows), purity
pairs.csv    one row per accepted pair of step 3 (constant estimator): iter_key, sensor_key, refs ('noutm' | 'all'),
             seg_radar, seg_ref, ref_sensor, group (co-observation group id within the
             (iteration, radar, refs) estimate), n_steps, share, med_de, med_dn, med_du,
             same_flight (testing: 1 / 0 by labels; blank on evaluation)
registration.md  plain-language report.

Run: python3 tools/radar_registration.py [--skip-checks]
Numbered self-checks at the end; exit status 1 on any failure: (1) synthetic offsets are
recovered within 2 m and inside the interval (a synthetic radar built from references + a known
offset + slow noise + clutter, and the offset injected into a real radar), and a 1.5 deg
rotation injected into a real radar is recovered within 0.1 deg and inside its interval; (2) the
estimator (constant offset and the azimuth median) vs a plain-Python loop (own groups, own
weighted median) on the testing (iteration, radar) with the most groups where a segment sits in
several pairs, plus a toy whose second gate stage accepts nothing (the stage-0 estimate must be
kept); (3) references deployable and never testing-only, the estimator runs with every
ground-truth loader disabled, every chosen sample / segment of each evaluation estimate exists in
that iteration's own grid / tracks files (read independently; a mutated copy must fail);
(4) determinism (two runs identical).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import zlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuas_clean as Q  # noqa: E402
import cuas_common as C  # noqa: E402
import truth as T  # noqa: E402

TRACKS = C.DERIVED / "sensor_tracks_v2"
OUT = C.DERIVED / "radar_registration"

REF_NOUTM = ("STUG", "CEQX", "ILYS", "IKJX", "SMFS")
UTM = ("VKYY", "QTVS")
REF_SETS = {"noutm": REF_NOUTM, "all": REF_NOUTM + UTM}
RADARS = ("EGOI", "NYTF", "UXYJ", "IUKL", "EUIG")
TESTING_ONLY_RADARS = ("EUIG",)

MOVING_MS = 2.0          # reference samples: smoothed horizontal speed >= this
PREGATE_M = 200.0        # candidate radar samples: horizontal distance <= this (any stage)
DV_GATE_MS = 4.0         # |v_radar - v_ref| horizontal <= this
GATE0_M = 100.0          # stage 0: residual around offset 0
GATE1_M = 50.0           # later stages: residual around the current estimate
MAX_STAGES = 8
CONVERGE_M = 0.05
MIN_PAIR_STEPS = 30      # 3 s of chosen steps
MIN_SHARE = 0.5
PAIR_CAP_STEPS = 300     # a co-observation group weighs at most 30 s of steps
N_BOOT = 400
MIN_GROUPS = 3           # bootstrap interval / reliable flag: >= this many independent co-observation groups
MIN_ROT_GROUPS = 3
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
MIN_ROT_AZ_SPREAD_DEG = 20.0
MIN_LAG_HEAD_SPREAD_DEG = 30.0   # lag-aware fit: circular sd of the reference headings
LAG_IRLS_ITERS = 15
MIN_AZ_RANGE_M = 200.0   # azimuth estimates: steps whose reference (truth) is at least this far from the site
HUBER_K2 = 2.0           # 2-D Huber: weight min(1, HUBER_K2 * s / r), s = median(r) / 1.1774
HUBER_C = 1.345
MIN_GT_SAMPLES = 50
SEED = 0
SYN_OFFSET = (17.0, -9.0, 25.0)
SYN_TOL_M = 2.0
SYN_AZ_DEG = 1.5
AZ_MAX_PASSES = 6
AZ_CONVERGE_DEG = 0.005
SYN_AZ_TOL_DEG = 0.1
PLAIN_ITER = "testing/S4-I4"
REF_LABELS = ("drone_air", "drone_ground")
EFFECT_METHODS = ("none", "gt_free_noutm", "gt_free_all", "gt_free_lag_noutm", "gt_free_lag_all", "gt_free_rot",
                  "gt_free_az_noutm", "gt_free_az_all", "loio_pooled", "loio_az", "loio_az_const",
                  "oracle", "oracle_az", "oracle_az_const")


def log(msg: str) -> None:
    print(msg, flush=True)


def md_table(header, rows) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows]
    return out


def short(key: str) -> str:
    return key.split("/")[-1] if key.startswith("testing") else key.split("/", 1)[1]


def t975(df: int) -> float:
    """97.5 % quantile of Student's t with df degrees of freedom (NaN for df < 1; 1.96 beyond the table)."""
    return float("nan") if df < 1 else T975.get(df, 1.96)


def f1(x, nd=1) -> str:
    return "-" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


# --------------------------------------------------------------------------- #
# Ground-truth guard (self-check 3): the estimator runs with every GT loader disabled
# --------------------------------------------------------------------------- #

_GT_FORBIDDEN = False
LABEL_COLS = ("label", "best_flight", "n_matched", "frac_matched", "frac_best", "frac_airborne")


class GroundTruthAccess(RuntimeError):
    pass


@contextmanager
def gt_forbidden():
    """Within this block, truth.load_truth / Q.load_groundtruth and any read of a GT-derived
    table (groundtruth, det_gt_match) or of label columns raise GroundTruthAccess."""
    global _GT_FORBIDDEN
    saved = (T.load_truth, T._groundtruth, Q.load_groundtruth, C.read_columns)
    orig_read = C.read_columns

    def deny(*a, **k):
        raise GroundTruthAccess("ground truth accessed inside the estimator")

    def guarded_read(path, numeric=frozenset(), usecols=None):
        name = Path(path).name
        if "groundtruth" in name or "det_gt_match" in name:
            raise GroundTruthAccess(f"GT table {name} read inside the estimator")
        if usecols is None or set(usecols) & set(LABEL_COLS):
            raise GroundTruthAccess(f"label columns of {name} read inside the estimator")
        return orig_read(path, numeric, usecols)

    T.load_truth = T._groundtruth = Q.load_groundtruth = deny
    C.read_columns = guarded_read
    _GT_FORBIDDEN = True
    try:
        yield
    finally:
        T.load_truth, T._groundtruth, Q.load_groundtruth, C.read_columns = saved
        _GT_FORBIDDEN = False


# --------------------------------------------------------------------------- #
# Loading (estimator inputs only)
# --------------------------------------------------------------------------- #


def load_iteration(it: C.Iteration) -> dict:
    """Grid samples of the reference and radar segments of one iteration (no labels).
    Returns {sensor_id: {seg, step, e, n, u, ve, vn}} (float64 values, sorted by seg, step),
    plus '_files' (the paths read) and '_iter_key'."""
    slug = C.iter_slug(it.key)
    tpath, gpath = TRACKS / f"{slug}__tracks.csv", TRACKS / f"{slug}__grid.npz"
    tr = C.read_columns(tpath, numeric={"seg_id"}, usecols=["seg_id", "iter_key", "sensor_id"])
    if not np.all(tr["iter_key"] == it.key):
        raise ValueError(f"{tpath.name}: rows of another iteration")
    seg_ids = tr["seg_id"].astype(np.int64)
    sid_of = dict(zip(seg_ids.tolist(), tr["sensor_id"].tolist()))
    wanted = set(REF_SETS["all"]) | set(RADARS)
    with np.load(gpath) as g:
        seg = g["seg_id"].astype(np.int64)
        lut_ids = np.array(sorted(sid_of))
        lut_sid = np.array([sid_of[s] for s in lut_ids], dtype=object)
        sid = lut_sid[np.searchsorted(lut_ids, seg)]
        out = {"_iter_key": it.key, "_files": [tpath, gpath]}
        for s in sorted(wanted):
            m = sid == s
            if not m.any():
                continue
            out[s] = {"seg": seg[m], "step": g["step"][m].astype(np.int64),
                      **{c: g[c][m].astype(np.float64) for c in ("e", "n", "u", "ve", "vn")}}
    return out


def concat(parts: list[dict]) -> dict:
    keys = ("seg", "step", "e", "n", "u", "ve", "vn")
    if not parts:
        return {k: np.zeros(0, np.int64 if k in ("seg", "step") else float) for k in keys} | {"sid": np.zeros(0, object)}
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    out["sid"] = np.concatenate([p["sid"] for p in parts])
    return out


def ref_samples(data: dict, ref_ids) -> dict:
    parts = []
    for s in ref_ids:
        if s in data:
            d = data[s]
            m = np.hypot(d["ve"], d["vn"]) >= MOVING_MS
            p = {k: v[m] for k, v in d.items()}
            p["sid"] = np.full(int(m.sum()), s, dtype=object)
            parts.append(p)
    return concat(parts)


def radar_sites() -> dict[tuple[str, str], tuple[float, float]]:
    det = Q.load_detections(["iter_key", "sensor_id", "sensor_e", "sensor_n", "sensor_pos_valid"])
    out = {}
    ok = det["sensor_pos_valid"] == 1
    for key in sorted(set(det["iter_key"][ok])):
        mk = ok & (det["iter_key"] == key)
        for s in RADARS:
            m = mk & (det["sensor_id"] == s)
            if m.any():
                out[(key, s)] = (float(np.median(det["sensor_e"][m])), float(np.median(det["sensor_n"][m])))
    return out


# --------------------------------------------------------------------------- #
# Estimator
# --------------------------------------------------------------------------- #


def wmedian(v: np.ndarray, w: np.ndarray) -> float:
    """Smallest value whose cumulative weight reaches half the total weight."""
    if v.size == 0 or not np.sum(w) > 0:
        return float("nan")
    o = np.argsort(v, kind="stable")
    cw = np.cumsum(w[o])
    return float(v[o][np.searchsorted(cw, 0.5 * cw[-1], "left")])


def candidates(ref: dict, rad: dict) -> dict:
    """All (reference sample, radar sample) at one grid step within PREGATE_M and DV_GATE_MS."""
    order = np.argsort(rad["step"], kind="stable")
    rs = rad["step"][order]
    lo = np.searchsorted(rs, ref["step"], "left")
    cnt = np.searchsorted(rs, ref["step"], "right") - lo
    tot = int(cnt.sum())
    ri = np.repeat(np.arange(ref["step"].size), cnt)
    off = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    rj = order[np.repeat(lo, cnt) + off]
    de = rad["e"][rj] - ref["e"][ri]
    dn = rad["n"][rj] - ref["n"][ri]
    dv = np.hypot(rad["ve"][rj] - ref["ve"][ri], rad["vn"][rj] - ref["vn"][ri])
    keep = (np.hypot(de, dn) <= PREGATE_M) & (dv <= DV_GATE_MS)
    ri, rj = ri[keep], rj[keep]
    return {"ri": ri, "rj": rj, "de": de[keep], "dn": dn[keep],
            "du": rad["u"][rj] - ref["u"][ri]}


def select(cand: dict, ref: dict, rad: dict, center, gate: float, seg_range: dict) -> dict:
    """Gate around `center`, nearest per reference sample, accept pairs. Returns the chosen
    steps of the accepted pairs: arrays ri, rj, de, dn, du, pair (index into pairs), and pairs
    (list of (seg_radar, seg_ref, ref_sid, n, share))."""
    res = np.hypot(cand["de"] - center[0], cand["dn"] - center[1])
    k = np.nonzero(res <= gate)[0]
    o = np.lexsort((k, res[k], cand["ri"][k]))
    k = k[o]
    first = np.ones(k.size, bool)
    first[1:] = cand["ri"][k][1:] != cand["ri"][k][:-1]
    k = k[first]
    ri, rj = cand["ri"][k], cand["rj"][k]
    sa, sr = rad["seg"][rj], ref["seg"][ri]
    pairs, keep = [], np.zeros(k.size, bool)
    pair_ix = np.full(k.size, -1, np.int64)
    if k.size:
        key = np.stack([sa, sr], 1)
        uk, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        for u_i, ((a, r), n) in enumerate(zip(uk.tolist(), cnt.tolist())):
            if n < MIN_PAIR_STEPS:
                continue
            smin, smax = seg_range[a]
            ref_steps = seg_range[("ref", r)]
            common = int(np.searchsorted(ref_steps, smax, "right") - np.searchsorted(ref_steps, smin, "left"))
            share = n / common
            if share < MIN_SHARE:
                continue
            m = inv == u_i
            pair_ix[m] = len(pairs)
            keep |= m
            pairs.append((a, r, str(ref["sid"][ri[m][0]]), n, share))
    pg = pair_groups(pairs)
    pix = pair_ix[keep]
    return {"ri": ri[keep], "rj": rj[keep], "de": cand["de"][k][keep], "dn": cand["dn"][k][keep],
            "du": cand["du"][k][keep], "pair": pix, "pairs": pairs, "pair_group": pg,
            "group": pg[pix] if pix.size else np.zeros(0, np.int64), "n_groups": int(pg.max()) + 1 if pg.size else 0}


def pair_groups(pairs: list) -> np.ndarray:
    """Co-observation group of each pair: connected components of the graph whose nodes are radar
    and reference segments and whose edges are the accepted pairs. One radar segment paired with
    several reference segments (e.g. the STUG and QTVS streams of one drone), or one reference
    segment with several radar segments, is ONE group: the unit of weighting and of the bootstrap.
    Group ids follow the first pair of each group."""
    parent: dict = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, r, *_ in pairs:
        ra, rr = find(("a", a)), find(("r", r))
        if ra != rr:
            parent[rr] = ra
    ids: dict = {}
    return np.array([ids.setdefault(find(("a", a)), len(ids)) for a, *_ in pairs], dtype=np.int64)


def group_weights(group: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-step weight min(n, cap) / n, n = the group's steps within `mask` (a group weighs at
    most PAIR_CAP_STEPS steps however many pairs / duplicate reference streams it holds)."""
    w = np.zeros(group.size)
    if not mask.any():
        return w
    cnt = np.bincount(group[mask])
    n = cnt[group[mask]].astype(float)
    w[mask] = np.minimum(n, PAIR_CAP_STEPS) / n
    return w


def point_estimate(sel: dict) -> tuple[np.ndarray, int, int]:
    """(d_e, d_n, d_u), horizontal steps, vertical steps."""
    n = sel["pair"].size
    allm = np.ones(n, bool)
    um = np.isfinite(sel["du"])
    wh, wu = group_weights(sel["group"], allm), group_weights(sel["group"], um)
    d = np.full(3, np.nan)
    if n >= MIN_PAIR_STEPS:
        d[0], d[1] = wmedian(sel["de"], wh), wmedian(sel["dn"], wh)
    if um.sum() >= MIN_PAIR_STEPS:
        d[2] = wmedian(sel["du"][um], wu[um])
    return d, int(n), int(um.sum())


def prepare(ref: dict, rad: dict) -> dict:
    seg_range = {}
    if rad["seg"].size:
        o = np.lexsort((rad["step"], rad["seg"]))
        st = rad["step"][o]
        sg = rad["seg"][o]
        us, first = np.unique(sg, return_index=True)
        last = np.r_[first[1:], sg.size] - 1
        for s, a, b in zip(us.tolist(), first.tolist(), last.tolist()):
            seg_range[s] = (int(st[a]), int(st[b]))
    if ref["seg"].size:
        o = np.lexsort((ref["step"], ref["seg"]))
        sg, st = ref["seg"][o], ref["step"][o]
        us, first = np.unique(sg, return_index=True)
        last = np.r_[first[1:], sg.size]
        for s, a, b in zip(us.tolist(), first.tolist(), last.tolist()):
            seg_range[("ref", s)] = st[a:b]
    return seg_range


def estimate(ref: dict, rad: dict, seed_key: str, boot: bool = True) -> dict:
    """The GT-free estimator (steps 2-5 of the docstring) for one radar and one reference set."""
    res = {"d": np.full(3, np.nan), "lo": np.full(3, np.nan), "hi": np.full(3, np.nan),
           "n_steps": 0, "n_steps_u": 0, "pairs": [], "n_groups": 0, "sel": None, "ci_blocks": "", "stages": 0}
    if ref["seg"].size == 0 or rad["seg"].size == 0:
        return res
    cand = candidates(ref, rad)
    seg_range = prepare(ref, rad)
    center, gate = np.zeros(2), GATE0_M
    best = None   # the last stage that accepted pairs: a later empty stage does not discard it
    for stage in range(MAX_STAGES):
        sel = select(cand, ref, rad, center, gate, seg_range)
        d, nh, nu = point_estimate(sel)
        if not np.isfinite(d[0]):
            break
        best = (sel, d, nh, nu)
        res["stages"] = stage + 1
        moved = np.max(np.abs(d[:2] - center)) if stage else np.inf
        center, gate = d[:2].copy(), GATE1_M
        if moved <= CONVERGE_M:
            break
    if best is None:
        return res
    sel, d, nh, nu = best
    sel["step"] = ref["step"][sel["ri"]]
    res.update(d=d, n_steps=nh, n_steps_u=nu, pairs=sel["pairs"], n_groups=sel["n_groups"], sel=sel)
    if boot:
        res["lo"], res["hi"], res["ci_blocks"] = bootstrap(sel, seed_key)
    return res


def boot_blocks(sel: dict) -> np.ndarray | None:
    """Bootstrap blocks = co-observation groups; None (no interval) with < MIN_GROUPS groups:
    one or two co-observed drones cannot show the drone-to-drone spread of the offset."""
    return sel["group"] if sel["n_groups"] >= MIN_GROUPS else None


def boot_half(reps: np.ndarray, n_groups: int) -> np.ndarray:
    """Interval half-width from bootstrap replicates over n_groups blocks: t975(G-1) sqrt(G/(G-1))
    times the replicates' sd (the bootstrap sd of G blocks is short by sqrt((G-1)/G), and with
    3-15 groups a percentile or 1.96 interval is too narrow). NaN where no replicate exists."""
    reps = np.asarray(reps, float)
    reps = reps[:, None] if reps.ndim == 1 else reps
    out = np.full(reps.shape[1], np.nan)
    for ax in range(reps.shape[1]):
        r = reps[:, ax][np.isfinite(reps[:, ax])]
        if r.size > 1:
            out[ax] = t975(n_groups - 1) * math.sqrt(n_groups / (n_groups - 1)) * float(np.std(r, ddof=1))
    return out


def bootstrap(sel: dict, seed_key: str) -> tuple[np.ndarray, np.ndarray, str]:
    rng = np.random.default_rng([SEED, zlib.crc32(seed_key.encode())])
    block = boot_blocks(sel)
    if block is None:
        return np.full(3, np.nan), np.full(3, np.nan), ""
    how = "groups"
    nb = int(block.max()) + 1
    um = np.isfinite(sel["du"])
    base_h = group_weights(sel["group"], np.ones(sel["group"].size, bool))
    base_u = group_weights(sel["group"], um)
    vals = [(sel["de"], base_h, np.ones(sel["de"].size, bool)), (sel["dn"], base_h, np.ones(sel["dn"].size, bool)),
            (sel["du"], base_u, um)]
    orders = [np.argsort(np.where(m, v, np.inf), kind="stable") for v, _, m in vals]
    reps = np.full((N_BOOT, 3), np.nan)
    for b in range(N_BOOT):
        mult = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float)[block]
        for ax, ((v, w, m), o) in enumerate(zip(vals, orders)):
            if m.sum() < MIN_PAIR_STEPS:
                continue
            ww = (w * mult * m)[o]
            cw = np.cumsum(ww)
            if cw[-1] <= 0:
                continue
            reps[b, ax] = v[o][np.searchsorted(cw, 0.5 * cw[-1], "left")]
    d, _, _ = point_estimate(sel)
    half = boot_half(reps, sel["n_groups"])
    return d - half, d + half, how


def fit_rot(sel: dict, ref: dict, site) -> dict | None:
    """Range / azimuth model d = b + k rel + delta (rel_n, -rel_e), Huber IRLS. None if the
    geometry does not support it."""
    if site is None or sel["n_groups"] < MIN_ROT_GROUPS:
        return None
    rel_e = ref["e"][sel["ri"]] - site[0]
    rel_n = ref["n"][sel["ri"]] - site[1]
    az = np.arctan2(rel_e, rel_n)
    R = np.hypot(np.mean(np.cos(az)), np.mean(np.sin(az)))
    spread = math.degrees(math.sqrt(-2.0 * math.log(max(R, 1e-12))))
    if spread < MIN_ROT_AZ_SPREAD_DEG:
        return None
    n = rel_e.size
    w0 = group_weights(sel["group"], np.ones(n, bool))
    A = np.zeros((2 * n, 4))
    A[:n, 0], A[n:, 1] = 1.0, 1.0
    A[:n, 2], A[n:, 2] = rel_e, rel_n
    A[:n, 3], A[n:, 3] = rel_n, -rel_e
    y = np.r_[sel["de"], sel["dn"]]
    w = np.r_[w0, w0]
    x = np.zeros(4)
    for _ in range(30):
        sw = np.sqrt(w)
        x_new = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        r = y - A @ x_new
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        hub = np.minimum(1.0, HUBER_C * s / np.maximum(np.abs(r), 1e-12))
        w = np.r_[w0, w0] * hub
        if np.max(np.abs(x_new - x)) < 1e-6:
            x = x_new
            break
        x = x_new
    r = y - A @ x
    res_rot = np.hypot(r[:n], r[n:])
    return {"b_e": x[0], "b_n": x[1], "k": x[2], "delta": x[3], "az_spread_deg": spread,
            "res_rot_med": float(np.median(res_rot))}


def _lag_solve(de, dn, ve, vn, w0, iters: int = LAG_IRLS_ITERS) -> np.ndarray:
    """Huber-weighted IRLS of (de, dn) = (b_e, b_n) + tau (ve, vn) via the 3x3 normal equations
    (one weight per step for both axes). Returns (b_e, b_n, tau)."""
    w = w0.copy()
    x = np.zeros(3)
    for _ in range(iters):
        sw, swe, swn = w.sum(), (w * ve).sum(), (w * vn).sum()
        N = np.array([[sw, 0.0, swe], [0.0, sw, swn], [swe, swn, (w * (ve * ve + vn * vn)).sum()]])
        rhs = np.array([(w * de).sum(), (w * dn).sum(), (w * (ve * de + vn * dn)).sum()])
        x = np.linalg.solve(N, rhs)
        r = np.hypot(de - x[0] - x[2] * ve, dn - x[1] - x[2] * vn)
        sc = np.median(r[w0 > 0]) / 1.1774 + 1e-9
        w = w0 * np.minimum(1.0, HUBER_K2 * sc / np.maximum(r, 1e-12))
    return x


def fit_lag(sel: dict, ref: dict, seed_key: str, boot: bool = True) -> dict | None:
    """Lag-aware offset: radar - reference = b + tau v_ref. The cyber / passive-RF references
    report the drone late (testing GT, 2026-10-28: along-track error -4 to -31 m at 7-23 m/s,
    ~0.5-1.7 s), so on moving references a plain median mixes that latency into the offset;
    tau absorbs the relative lag (radar minus reference) and b is the constant part. Needs a
    heading spread >= MIN_LAG_HEAD_SPREAD_DEG (else b and tau are confounded): None otherwise.
    Interval: the same block bootstrap as the constant estimate."""
    if sel is None:
        return None
    ve, vn = ref["ve"][sel["ri"]], ref["vn"][sel["ri"]]
    hd = np.arctan2(ve, vn)
    R = np.hypot(np.mean(np.cos(hd)), np.mean(np.sin(hd)))
    spread = math.degrees(math.sqrt(-2.0 * math.log(max(R, 1e-12))))
    if spread < MIN_LAG_HEAD_SPREAD_DEG:
        return None
    w0 = group_weights(sel["group"], np.ones(sel["group"].size, bool))
    x = _lag_solve(sel["de"], sel["dn"], ve, vn, w0)
    out = {"b_e": x[0], "b_n": x[1], "tau": x[2], "head_spread_deg": spread,
           "lo": np.full(3, np.nan), "hi": np.full(3, np.nan),
           "res_lag_med": float(np.median(np.hypot(sel["de"] - x[0] - x[2] * ve, sel["dn"] - x[1] - x[2] * vn)))}
    block = boot_blocks(sel)
    if boot and block is not None:
        rng = np.random.default_rng([SEED, zlib.crc32((seed_key + "|lag").encode())])
        nb = int(block.max()) + 1
        reps = np.full((N_BOOT, 3), np.nan)
        for b in range(N_BOOT):
            mult = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float)[block]
            try:
                reps[b] = _lag_solve(sel["de"], sel["dn"], ve, vn, w0 * mult)
            except np.linalg.LinAlgError:
                pass
        half = boot_half(reps, sel["n_groups"])
        out["lo"], out["hi"] = x - half, x + half
    return out


def az_diff_deg(e, n, re, rn, site) -> np.ndarray:
    """Azimuth of (e, n) minus azimuth of (re, rn), both seen from the radar site, degrees in
    (-180, 180]; positive = the radar puts the object clockwise of the reference."""
    return C.angle_diff_deg(C.azimuth_deg(e - site[0], n - site[1]), C.azimuth_deg(re - site[0], rn - site[1]))


def fit_az(sel: dict, ref: dict, site, seed_key: str, boot: bool = True) -> dict | None:
    """Azimuth-only registration: weighted median over the chosen steps (reference at least
    MIN_AZ_RANGE_M from the site) of the azimuth difference radar - reference about the radar
    site; pair weights as for the offset. Interval: the same block bootstrap."""
    if sel is None or site is None:
        return None
    re_, rn_ = ref["e"][sel["ri"]], ref["n"][sel["ri"]]
    m = np.hypot(re_ - site[0], rn_ - site[1]) >= MIN_AZ_RANGE_M
    if m.sum() < MIN_PAIR_STEPS:
        return None
    th = az_diff_deg(re_ + sel["de"], rn_ + sel["dn"], re_, rn_, site)
    w = group_weights(sel["group"], m)
    out = {"az": wmedian(th[m], w[m]), "lo": np.nan, "hi": np.nan, "n": int(m.sum())}
    block = boot_blocks(sel)
    if boot and block is not None:
        rng = np.random.default_rng([SEED, zlib.crc32((seed_key + "|az").encode())])
        nb = int(block.max()) + 1
        v, ww, bl = th[m], w[m], block[m]
        o = np.argsort(v, kind="stable")
        reps = []
        for _ in range(N_BOOT):
            mult = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float)[bl]
            cw = np.cumsum((ww * mult)[o])
            if cw[-1] > 0:
                reps.append(v[o][np.searchsorted(cw, 0.5 * cw[-1], "left")])
        if reps:
            half = float(boot_half(np.array(reps)[:, None], sel["n_groups"])[0])
            out["lo"], out["hi"] = out["az"] - half, out["az"] + half
    return out


def estimate_az(ref: dict, rad: dict, site, seed_key: str, boot: bool = True) -> dict | None:
    """Azimuth registration with rotation-centred gating: the pair selection of estimate() works
    in a constant-offset frame, which a rotation distorts (a 1.5 deg rotation moves drones at
    different bearings by different amounts and changes which pairs pass the gate). So: estimate,
    fit the angle, rotate the radar back by the angle found so far and repeat on the rotated
    radar (the residual angle is added) until it moves by < AZ_CONVERGE_DEG (<= AZ_MAX_PASSES).
    The interval is the last pass's bootstrap shifted by the accumulated angle."""
    if site is None:
        return None
    total, last, sel = 0.0, None, None
    for _ in range(AZ_MAX_PASSES):
        if total:
            e2, n2, _ = correct(rad["e"], rad["n"], rad["u"], site, total, np.zeros(3))
            r2 = {**rad, "e": e2, "n": n2}
        else:
            r2 = rad
        est = estimate(ref, r2, seed_key, boot=False)
        fa = fit_az(est["sel"], ref, site, seed_key, boot=False)
        if fa is None:
            break
        total += fa["az"]
        last, sel = fa, est["sel"]
        if abs(fa["az"]) < AZ_CONVERGE_DEG:
            break
    if last is None:
        return None
    out = fit_az(sel, ref, site, seed_key, boot=boot)
    base = total - out["az"]
    out.update(az=total, lo=out["lo"] + base, hi=out["hi"] + base, n_pairs=len(sel["pairs"]), n_groups=sel["n_groups"])
    return out


def correct(e, n, u, site, az_deg: float, d) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Registration as written in offsets.csv: rotate (e, n) about the radar site by -az_deg
    (az_deg NaN or no site -> no rotation), then subtract (d_e, d_n, d_u) (NaN -> 0)."""
    d = np.nan_to_num(np.asarray(d, float))
    if site is not None and np.isfinite(az_deg) and az_deg != 0.0:
        a = math.radians(az_deg)
        re_, rn_ = e - site[0], n - site[1]
        e = site[0] + re_ * math.cos(a) - rn_ * math.sin(a)
        n = site[1] + rn_ * math.cos(a) + re_ * math.sin(a)
    return e - d[0], n - d[1], u - d[2]


def apply_rot(rot: dict, e, n, site):
    rel_e, rel_n = e - site[0], n - site[1]
    return (rot["b_e"] + rot["k"] * rel_e + rot["delta"] * rel_n,
            rot["b_n"] + rot["k"] * rel_n - rot["delta"] * rel_e)


def run_estimates(it: C.Iteration, data: dict, sites: dict, boot: bool = True) -> dict:
    """{(radar, refset): estimate dict} for one iteration (GT-free)."""
    out = {}
    for refset, ids in REF_SETS.items():
        ref = ref_samples(data, ids)
        for r in RADARS:
            if r not in data:
                continue
            est = estimate(ref, data[r], f"{it.key}|{r}|{refset}", boot=boot)
            est["refs_used"] = "|".join(sorted({p[2] for p in est["pairs"]}))
            est["rot"] = fit_rot(est["sel"], ref, sites.get((it.key, r))) if est["sel"] is not None else None
            est["lag"] = fit_lag(est["sel"], ref, f"{it.key}|{r}|{refset}", boot=boot)
            est["az"] = estimate_az(ref, data[r], sites.get((it.key, r)), f"{it.key}|{r}|{refset}", boot=boot)
            if est["sel"] is not None:
                sel = est["sel"]
                res_c = np.hypot(sel["de"] - est["d"][0], sel["dn"] - est["d"][1])
                est["res_const_med"] = float(np.median(res_c))
            est["ref"] = ref
            out[(r, refset)] = est
    return out


# --------------------------------------------------------------------------- #
# Validation (testing, GT)
# --------------------------------------------------------------------------- #


def load_labels(it: C.Iteration) -> dict[int, tuple[str, str]]:
    p = TRACKS / f"{C.iter_slug(it.key)}__tracks.csv"
    t = C.read_columns(p, numeric={"seg_id"}, usecols=["seg_id", "label", "best_flight"])
    return {int(s): (str(l), str(f)) for s, l, f in zip(t["seg_id"], t["label"], t["best_flight"])}


class TruthIndex:
    def __init__(self, tr: T.Truth):
        self.tr = tr
        self.fidx = {f["flight_key"]: i for i, f in enumerate(tr.flights)}
        self.nf = max(len(tr.flights), 1)
        self.key = tr.step.astype(np.int64) * self.nf + tr.flight.astype(np.int64)

    def lookup(self, step: np.ndarray, flight_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(row index, valid) of truth rows for (step, flight_key), valid only if airborne."""
        fi = np.array([self.fidx.get(f, -1) for f in flight_keys], dtype=np.int64)
        k = step.astype(np.int64) * self.nf + fi
        pos = np.clip(np.searchsorted(self.key, k), 0, max(self.key.size - 1, 0))
        ok = (fi >= 0) & (self.key.size > 0)
        if self.key.size:
            ok &= self.key[pos] == k
            ok &= self.tr.airborne[pos]
        return pos, ok


def gt_samples(rad: dict, labels: dict, ti: TruthIndex) -> dict:
    """drone_air grid samples of a radar with airborne truth of the segment's best_flight."""
    lab = np.array([labels.get(int(s), ("", ""))[0] for s in rad["seg"]], dtype=object)
    m = lab == "drone_air"
    fk = np.array([labels.get(int(s), ("", ""))[1] for s in rad["seg"][m]], dtype=object)
    pos, ok = ti.lookup(rad["step"][m], fk)
    idx = np.nonzero(m)[0][ok]
    pos = pos[ok]
    tr = ti.tr
    return {"e": rad["e"][idx], "n": rad["n"][idx], "u": rad["u"][idx],
            "te": tr.e[pos], "tn": tr.n[pos], "tu": tr.u[pos], "flight": fk[ok]}


def ref_latency(d: dict, labels: dict, ti: TruthIndex) -> dict | None:
    """Reference sensor vs truth on its moving drone samples (label drone_air / drone_ground,
    speed >= MOVING_MS): median along-track and cross-track error (m), median speed, and the
    implied lag = -along / speed (s; positive = reports late)."""
    lab = np.array([labels.get(int(x), ("", ""))[0] for x in d["seg"]], dtype=object)
    sp = np.hypot(d["ve"], d["vn"])
    m = np.isin(lab, REF_LABELS) & (sp >= MOVING_MS)
    fk = np.array([labels.get(int(x), ("", ""))[1] for x in d["seg"][m]], dtype=object)
    pos, ok = ti.lookup(d["step"][m], fk)
    idx = np.nonzero(m)[0][ok]
    if idx.size < MIN_GT_SAMPLES:
        return None
    de, dn = d["e"][idx] - ti.tr.e[pos[ok]], d["n"][idx] - ti.tr.n[pos[ok]]
    ve, vn, v = d["ve"][idx], d["vn"][idx], sp[idx]
    along, cross = (de * ve + dn * vn) / v, (de * vn - dn * ve) / v
    return {"n": int(idx.size), "along": float(np.median(along)), "cross": float(np.median(cross)),
            "speed": float(np.median(v)), "lag": float(np.median(-along / v))}


def gt_offset(s: dict) -> tuple[np.ndarray, int]:
    if s["e"].size < MIN_GT_SAMPLES:
        return np.full(3, np.nan), int(s["e"].size)
    du = s["u"] - s["tu"]
    du = du[np.isfinite(du)]
    return np.array([np.median(s["e"] - s["te"]), np.median(s["n"] - s["tn"]),
                     np.median(du) if du.size >= MIN_GT_SAMPLES else np.nan]), int(s["e"].size)


def same_diag(est: dict, rad: dict, labels: dict, ti: TruthIndex) -> dict:
    """Pair purity, gt_same (radar - truth on the chosen steps) and ref_err (ref - truth)."""
    out = {"purity": np.nan, "gt_same": np.full(3, np.nan), "ref_err": np.full(3, np.nan), "same_flags": []}
    sel = est["sel"]
    if sel is None:
        return out
    flags = []
    for a, r, *_ in sel["pairs"]:
        la, fa = labels.get(int(a), ("", ""))
        lr, fr = labels.get(int(r), ("", ""))
        flags.append(int(la in REF_LABELS and lr in REF_LABELS and fa == fr))
    out["same_flags"] = flags
    out["purity"] = float(np.mean(flags)) if flags else np.nan
    ref = est["ref"]
    segs = rad["seg"][sel["rj"]]
    lab = np.array([labels.get(int(s), ("", ""))[0] for s in segs], dtype=object)
    fk = np.array([labels.get(int(s), ("", ""))[1] for s in segs], dtype=object)
    m = np.isin(lab, REF_LABELS)
    pos, ok = ti.lookup(sel["step"][m], fk[m])
    if ok.sum() >= MIN_PAIR_STEPS:
        tr = ti.tr
        rj, ri, p = sel["rj"][m][ok], sel["ri"][m][ok], pos[ok]
        g = np.array([np.median(rad["e"][rj] - tr.e[p]), np.median(rad["n"][rj] - tr.n[p]), np.nan])
        f = np.array([np.median(ref["e"][ri] - tr.e[p]), np.median(ref["n"][ri] - tr.n[p]), np.nan])
        du = rad["u"][rj] - tr.u[p]
        if np.isfinite(du).sum() >= MIN_PAIR_STEPS:
            g[2] = np.nanmedian(du)
        dr = ref["u"][ri] - tr.u[p]
        if np.isfinite(dr).sum() >= MIN_PAIR_STEPS:
            f[2] = np.nanmedian(dr)
        out["gt_same"], out["ref_err"] = g, f
    return out


# --------------------------------------------------------------------------- #
# Effect check
# --------------------------------------------------------------------------- #


def effect_stats(e, n, u, te, tn, tu) -> dict:
    eh = np.hypot(e - te, n - tn)
    e3 = np.sqrt((e - te) ** 2 + (n - tn) ** 2 + (u - tu) ** 2)
    e3 = e3[np.isfinite(e3)]
    return {"n": int(eh.size), "n3": int(e3.size), "med_h": float(np.median(eh)) if eh.size else np.nan,
            "med3": float(np.median(e3)) if e3.size else np.nan,
            **{f"w{g}": float(np.mean(e3 <= g)) if e3.size else np.nan for g in (10, 25, 50)}}


# --------------------------------------------------------------------------- #
# Plain-Python recomputation (self-check 2)
# --------------------------------------------------------------------------- #


def plain_estimate(ref: dict, rad: dict) -> tuple[list[float], int, int]:
    R = [dict(seg=int(ref["seg"][i]), step=int(ref["step"][i]), e=float(ref["e"][i]), n=float(ref["n"][i]),
              u=float(ref["u"][i]), ve=float(ref["ve"][i]), vn=float(ref["vn"][i])) for i in range(ref["seg"].size)]
    A = [dict(seg=int(rad["seg"][j]), step=int(rad["step"][j]), e=float(rad["e"][j]), n=float(rad["n"][j]),
              u=float(rad["u"][j]), ve=float(rad["ve"][j]), vn=float(rad["vn"][j])) for j in range(rad["seg"].size)]
    by_step: dict[int, list[int]] = {}
    for j, a in enumerate(A):
        by_step.setdefault(a["step"], []).append(j)
    srange: dict[int, list[int]] = {}
    for a in A:
        r = srange.setdefault(a["seg"], [a["step"], a["step"]])
        r[0], r[1] = min(r[0], a["step"]), max(r[1], a["step"])
    ref_steps: dict[int, list[int]] = {}
    for x in R:
        ref_steps.setdefault(x["seg"], []).append(x["step"])

    def wmed(vals: list[tuple[float, float]]) -> float:
        vals = sorted(vals, key=lambda t: t[0])
        tot = sum(w for _, w in vals)
        c = 0.0
        for v, w in vals:
            c += w
            if c >= 0.5 * tot:
                return v
        return float("nan")

    center, gate = (0.0, 0.0), GATE0_M
    d, nh, nu = [float("nan")] * 3, 0, 0
    for stage in range(MAX_STAGES):
        chosen = []   # (i, j, de, dn, du)
        for i, x in enumerate(R):
            best = None
            for j in by_step.get(x["step"], []):
                a = A[j]
                de, dn = a["e"] - x["e"], a["n"] - x["n"]
                if math.hypot(de, dn) > PREGATE_M or math.hypot(a["ve"] - x["ve"], a["vn"] - x["vn"]) > DV_GATE_MS:
                    continue
                res = math.hypot(de - center[0], dn - center[1])
                if res <= gate and (best is None or res < best[0]):
                    best = (res, j, de, dn, a["u"] - x["u"])
            if best is not None:
                chosen.append((i, best[1], best[2], best[3], best[4]))
        groups: dict[tuple[int, int], list] = {}
        for c in chosen:
            groups.setdefault((A[c[1]]["seg"], R[c[0]]["seg"]), []).append(c)
        acc = {}
        for (sa, sr), lst in groups.items():
            if len(lst) < MIN_PAIR_STEPS:
                continue
            lo, hi = srange[sa]
            common = sum(1 for s in ref_steps[sr] if lo <= s <= hi)
            if len(lst) / common >= MIN_SHARE:
                acc[(sa, sr)] = lst
        # co-observation groups: flood fill over shared radar / reference segments
        comp: dict[tuple[int, int], int] = {}
        for key0 in acc:
            if key0 in comp:
                continue
            gid, todo = len(set(comp.values())), [key0]
            while todo:
                k0 = todo.pop()
                if k0 in comp:
                    continue
                comp[k0] = gid
                todo += [k1 for k1 in acc if k1 not in comp and (k1[0] == k0[0] or k1[1] == k0[1])]
        members: dict[int, list] = {}
        for k0, lst in acc.items():
            members.setdefault(comp[k0], []).extend(lst)
        eh, nh_, uh = [], [], []
        for lst in members.values():
            wh = min(len(lst), PAIR_CAP_STEPS) / len(lst)
            us = [c for c in lst if math.isfinite(c[4])]
            for c in lst:
                eh.append((c[2], wh))
                nh_.append((c[3], wh))
            if us:
                wu = min(len(us), PAIR_CAP_STEPS) / len(us)
                uh += [(c[4], wu) for c in us]
        d_s = [float("nan")] * 3
        if len(eh) >= MIN_PAIR_STEPS:
            d_s[0], d_s[1] = wmed(eh), wmed(nh_)
        if len(uh) >= MIN_PAIR_STEPS:
            d_s[2] = wmed(uh)
        if not math.isfinite(d_s[0]):
            break   # keep the previous stage's estimate
        d, nh, nu = d_s, len(eh), len(uh)
        moved = max(abs(d[0] - center[0]), abs(d[1] - center[1])) if stage else float("inf")
        center, gate = (d[0], d[1]), GATE1_M
        if moved <= CONVERGE_M:
            break
    return d, nh, nu


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-checks", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    its = C.iterations()
    testing = [it for it in its if it.tree == "testing"]
    deploy = C.deployable_sensor_ids()
    log("radar sites from detections ...")
    sites = radar_sites()

    est_all: dict[str, dict] = {}
    val: dict[tuple[str, str], dict] = {}
    gt_samp: dict[tuple[str, str], dict] = {}
    latency: dict[tuple[str, str], dict] = {}
    for it in its:
        t0 = time.time()
        with gt_forbidden():
            data = load_iteration(it)
            est = run_estimates(it, data, sites)
        est_all[it.key] = est
        if it.has_groundtruth:
            labels = load_labels(it)
            ti = TruthIndex(T.load_truth(it, "all"))
            for rid in REF_SETS["all"]:
                if rid in data:
                    lat_ = ref_latency(data[rid], labels, ti)
                    if lat_:
                        latency[(it.key, rid)] = lat_
            for r in RADARS:
                if r not in data:
                    continue
                s = gt_samples(data[r], labels, ti)
                gt_samp[(it.key, r)] = s
                g, ng = gt_offset(s)
                v = {"gt": g, "gt_n": ng, "gt_az": np.nan, "gt_after_az": np.full(3, np.nan)}
                site = sites.get((it.key, r))
                if site is not None and ng >= MIN_GT_SAMPLES:
                    far = np.hypot(s["te"] - site[0], s["tn"] - site[1]) >= MIN_AZ_RANGE_M
                    if far.sum() >= MIN_GT_SAMPLES:
                        v["gt_az"] = float(np.median(az_diff_deg(s["e"][far], s["n"][far], s["te"][far], s["tn"][far], site)))
                        ce, cn, cu = correct(s["e"], s["n"], s["u"], site, v["gt_az"], np.zeros(3))
                        v["gt_after_az"] = gt_offset({**s, "e": ce, "n": cn, "u": cu})[0]
                for refset in REF_SETS:
                    v[refset] = same_diag(est[(r, refset)], data[r], labels, ti)
                val[(it.key, r)] = v
        for k, e in est.items():   # keep the estimator's chosen steps' radar segments for pairs.csv
            if e["sel"] is not None:
                e["sel"]["rad_seg"] = data[k[0]]["seg"][e["sel"]["rj"]]
            e.pop("ref", None)
        del data
        log(f"{it.key}: " + ", ".join(
            f"{r}/{rs}: {len(e['pairs'])}p {f1(e['d'][0])},{f1(e['d'][1])},{f1(e['d'][2])}"
            for (r, rs), e in sorted(est.items()) if e["pairs"]) + f"  ({time.time() - t0:.0f} s)")

    # LOIO pooled GT offsets
    def pooled(radar: str, exclude: str | None) -> tuple[np.ndarray, np.ndarray, int]:
        rows = [val[(it.key, radar)]["gt"] for it in testing
                if it.key != exclude and (it.key, radar) in val and np.isfinite(val[(it.key, radar)]["gt"][0])]
        if not rows:
            return np.full(3, np.nan), np.full(3, np.nan), 0
        a = np.array(rows)
        mean = np.array([np.nanmean(a[:, k]) if np.isfinite(a[:, k]).any() else np.nan for k in range(3)])
        nk = np.isfinite(a).sum(0)
        sd = np.array([np.nanstd(a[:, k], ddof=1) if nk[k] > 1 else np.nan for k in range(3)])
        tq = np.array([t975(int(x) - 1) for x in nk])
        return mean, tq * sd * np.sqrt(1.0 + 1.0 / np.maximum(nk, 1)), len(rows)   # prediction interval

    def pooled_az(radar: str, exclude: str | None) -> tuple[float, float, int]:
        a = np.array([val[(it.key, radar)]["gt_az"] for it in testing
                      if it.key != exclude and (it.key, radar) in val and np.isfinite(val[(it.key, radar)]["gt_az"])])
        if a.size == 0:
            return np.nan, np.nan, 0
        return float(a.mean()), (t975(a.size - 1) * float(a.std(ddof=1)) * math.sqrt(1.0 + 1.0 / a.size) if a.size > 1 else np.nan), \
            int(a.size)   # prediction interval for a new iteration

    def pooled_after_az(radar: str, exclude: str | None) -> np.ndarray:
        a = np.array([val[(it.key, radar)]["gt_after_az"] for it in testing
                      if it.key != exclude and (it.key, radar) in val and np.isfinite(val[(it.key, radar)]["gt_after_az"][0])])
        if a.size == 0:
            return np.full(3, np.nan)
        return np.array([np.nanmean(a[:, k]) if np.isfinite(a[:, k]).any() else np.nan for k in range(3)])

    # offsets.csv
    rows = []
    for it in its:
        for r in RADARS:
            if (r, "noutm") not in est_all[it.key]:
                continue
            v = val.get((it.key, r))
            site = sites.get((it.key, r))
            for method in ("gt_free_noutm", "gt_free_all", "gt_free_lag_noutm", "gt_free_lag_all", "gt_free_az_noutm",
                           "gt_free_az_all", "loio_pooled", "loio_az", "loio_az_const"):
                row = {"iter_key": it.key, "sensor_key": r, "method": method, "testing_only": int(r in TESTING_ONLY_RADARS),
                       "site_e": site[0] if site else np.nan, "site_n": site[1] if site else np.nan,
                       "az_deg": np.nan, "az_lo": np.nan, "az_hi": np.nan}
                excl = it.key if it.has_groundtruth else None
                if method in ("loio_az", "loio_az_const"):
                    az, half_az, npool = pooled_az(r, excl)
                    d = (np.full(3, np.nan) if not np.isfinite(az) else
                         np.r_[0.0, 0.0, pooled(r, excl)[0][2]] if method == "loio_az" else pooled_after_az(r, excl))
                    lo = hi = np.full(3, np.nan)
                    row.update(n_pairs=npool, n_groups=0, n_steps=0, n_steps_u=0, refs_used="", ci_blocks="", lag_tau_s=np.nan,
                               rot_k=np.nan, rot_delta_deg=np.nan, purity=np.nan, az_deg=az,
                               az_lo=az - half_az, az_hi=az + half_az)
                elif method == "loio_pooled":
                    d, half, npool = pooled(r, it.key if it.has_groundtruth else None)
                    lo, hi = d - half, d + half
                    row.update(n_pairs=npool, n_groups=0, n_steps=0, n_steps_u=0, refs_used="", ci_blocks="", lag_tau_s=np.nan,
                               rot_k=np.nan, rot_delta_deg=np.nan, purity=np.nan)
                else:
                    e = est_all[it.key][(r, method.split("_")[-1])]
                    d, lo, hi = e["d"], e["lo"], e["hi"]
                    rot, lag = e["rot"], e["lag"]
                    if "_az_" in method:    # rotation only; vertical from the median
                        a_ = e["az"]
                        d = np.r_[0.0, 0.0, e["d"][2]] if a_ else np.full(3, np.nan)
                        lo, hi = np.r_[np.nan, np.nan, e["lo"][2]], np.r_[np.nan, np.nan, e["hi"][2]]
                        if a_:
                            row.update(az_deg=a_["az"], az_lo=a_["lo"], az_hi=a_["hi"])
                    if "_lag_" in method:   # horizontal from the lag-aware fit, vertical from the median
                        if lag is None:
                            d, lo, hi = np.full(3, np.nan), np.full(3, np.nan), np.full(3, np.nan)
                        else:
                            d = np.array([lag["b_e"], lag["b_n"], e["d"][2]])
                            lo = np.array([lag["lo"][0], lag["lo"][1], e["lo"][2]])
                            hi = np.array([lag["hi"][0], lag["hi"][1], e["hi"][2]])
                    row.update(n_pairs=len(e["pairs"]), n_groups=e["n_groups"], n_steps=e["n_steps"], n_steps_u=e["n_steps_u"],
                               refs_used=e["refs_used"], ci_blocks=e["ci_blocks"],
                               lag_tau_s=lag["tau"] if lag and "_lag_" in method else np.nan,
                               rot_k=rot["k"] if rot else np.nan,
                               rot_delta_deg=math.degrees(rot["delta"]) if rot else np.nan,
                               purity=v[method.split("_")[-1]]["purity"] if v else np.nan)
                    if "_az_" in method and e["az"]:   # the az estimator's own (rotation-centred) pair set
                        row.update(n_pairs=e["az"]["n_pairs"], n_groups=e["az"]["n_groups"], n_steps=e["az"]["n"])
                half = np.maximum(np.abs(hi - d), np.abs(d - lo))
                has = bool(np.isfinite(row["az_deg"]) if "_az" in method else np.isfinite(d[0]))
                row["reliable"] = int(has and (row["n_pairs"] >= MIN_GROUPS if method.startswith("loio")
                                               else row["n_groups"] >= MIN_GROUPS))
                row.update(d_e=d[0], d_n=d[1], d_u=d[2],
                           ci=float(np.nanmax(half)) if np.isfinite(half).any() else np.nan,
                           d_e_lo=lo[0], d_e_hi=hi[0], d_n_lo=lo[1], d_n_hi=hi[1], d_u_lo=lo[2], d_u_hi=hi[2])
                if v:
                    g = v["gt"]
                    is_az = "_az" in method
                    row.update(gt_d_e=g[0], gt_d_n=g[1], gt_d_u=g[2], gt_n=v["gt_n"], gt_az_deg=v["gt_az"],
                               err_h=np.nan if is_az else float(np.hypot(d[0] - g[0], d[1] - g[1])),
                               err_az_deg=abs(row["az_deg"] - v["gt_az"]) if is_az else np.nan)
                else:
                    row.update(gt_d_e=np.nan, gt_d_n=np.nan, gt_d_u=np.nan, gt_n=np.nan, err_h=np.nan,
                               gt_az_deg=np.nan, err_az_deg=np.nan)
                rows.append(row)
    order = ["iter_key", "sensor_key", "method", "d_e", "d_n", "d_u", "ci", "d_e_lo", "d_e_hi", "d_n_lo",
             "d_n_hi", "d_u_lo", "d_u_hi", "n_pairs", "n_groups", "reliable", "n_steps", "n_steps_u", "refs_used", "ci_blocks",
             "testing_only", "az_deg", "az_lo", "az_hi", "site_e", "site_n", "lag_tau_s", "rot_k", "rot_delta_deg",
             "gt_d_e", "gt_d_n", "gt_d_u", "gt_n", "gt_az_deg", "err_h", "err_az_deg", "purity"]
    numeric = {"d_e", "d_n", "d_u", "ci", "d_e_lo", "d_e_hi", "d_n_lo", "d_n_hi", "d_u_lo", "d_u_hi",
               "lag_tau_s", "rot_k", "rot_delta_deg", "gt_d_e", "gt_d_n", "gt_d_u", "gt_n", "err_h", "purity",
               "az_deg", "az_lo", "az_hi", "site_e", "site_n", "gt_az_deg", "err_az_deg"}
    cols = {c: (np.array([float(r_[c]) for r_ in rows]) if c in numeric else [r_[c] for r_ in rows]) for c in order}
    OUT.mkdir(parents=True, exist_ok=True)
    C.write_columns(OUT / "offsets.csv", cols, order)

    # pairs.csv
    prow = {k: [] for k in ("iter_key", "sensor_key", "refs", "seg_radar", "seg_ref", "ref_sensor", "group", "n_steps",
                            "share", "med_de", "med_dn", "med_du", "same_flight")}
    for it in its:
        for (r, refset), e in sorted(est_all[it.key].items()):
            if e["sel"] is None:
                continue
            sel = e["sel"]
            flags = val[(it.key, r)][refset]["same_flags"] if (it.key, r) in val else [None] * len(e["pairs"])
            for pi, ((a, rr, sid, n, share), fl) in enumerate(zip(e["pairs"], flags)):
                m = sel["pair"] == pi
                du = sel["du"][m]
                prow["iter_key"].append(it.key); prow["sensor_key"].append(r); prow["refs"].append(refset)
                prow["seg_radar"].append(a); prow["seg_ref"].append(rr); prow["ref_sensor"].append(sid)
                prow["group"].append(int(sel["pair_group"][pi]))
                prow["n_steps"].append(n)
                prow["share"].append(share); prow["med_de"].append(float(np.median(sel["de"][m])))
                prow["med_dn"].append(float(np.median(sel["dn"][m])))
                prow["med_du"].append(float(np.nanmedian(du)) if np.isfinite(du).any() else np.nan)
                prow["same_flight"].append("" if fl is None else fl)
    for c in ("share", "med_de", "med_dn", "med_du"):
        prow[c] = np.array(prow[c], dtype=float)
    C.write_columns(OUT / "pairs.csv", prow, list(prow))

    # effect check: every method as (az_deg, d) for correct(), or the range / azimuth model
    def method_of(method: str, key: str, r: str):
        """(az_deg, (d_e, d_n, d_u), rot model or None) for the effect check; no estimate -> no shift."""
        z = np.zeros(3)
        if method == "none":
            return np.nan, z, None
        v = val[(key, r)]
        if method == "oracle":
            return np.nan, v["gt"], None
        if method == "oracle_az":
            return v["gt_az"], np.r_[0.0, 0.0, v["gt"][2]], None
        if method == "oracle_az_const":
            return v["gt_az"], v["gt_after_az"], None
        if method == "loio_pooled":
            return np.nan, pooled(r, key)[0], None
        if method == "loio_az":
            return pooled_az(r, key)[0], np.r_[0.0, 0.0, pooled(r, key)[0][2]], None
        if method == "loio_az_const":
            return pooled_az(r, key)[0], pooled_after_az(r, key), None
        refset = "all" if method.endswith("_all") else "noutm"
        e = est_all[key][(r, refset)]
        if "_lag_" in method:
            lag = e["lag"]
            return np.nan, (z if lag is None else np.array([lag["b_e"], lag["b_n"], e["d"][2]])), None
        if "_az_" in method:
            return (e["az"]["az"] if e["az"] else np.nan), np.r_[0.0, 0.0, e["d"][2]], None
        if method == "gt_free_rot" and e["rot"] is not None and (key, r) in sites:
            return np.nan, np.r_[0.0, 0.0, e["d"][2]], e["rot"]
        return np.nan, e["d"], None

    methods = EFFECT_METHODS
    effect = {}
    for scope in ("deployable", "EUIG"):
        for it in testing:
            for meth in methods:
                parts = []
                for r in RADARS:
                    if (r in TESTING_ONLY_RADARS) != (scope == "EUIG") or (it.key, r) not in gt_samp:
                        continue
                    s_ = gt_samp[(it.key, r)]
                    if s_["e"].size == 0:
                        continue
                    az, d, rot = method_of(meth, it.key, r)
                    site = sites.get((it.key, r))
                    if rot is not None:
                        ce, cn = apply_rot(rot, s_["e"], s_["n"], site)
                        fe, fn, fu = s_["e"] - ce, s_["n"] - cn, s_["u"] - np.nan_to_num(d[2])
                    else:
                        fe, fn, fu = correct(s_["e"], s_["n"], s_["u"], site, az, d)
                    parts.append((fe, fn, fu, s_["te"], s_["tn"], s_["tu"]))
                if parts:
                    effect[(scope, it.key, meth)] = effect_stats(*[np.concatenate(x) for x in zip(*parts)])
    for it in testing:   # per radar (key findings: how much of each radar's error the rotation removes)
        for r in RADARS:
            s_ = gt_samp.get((it.key, r))
            if s_ is None or s_["e"].size < MIN_GT_SAMPLES:
                continue
            for meth in ("none", "loio_az"):
                az, d, _ = method_of(meth, it.key, r)
                fe, fn, fu = correct(s_["e"], s_["n"], s_["u"], sites.get((it.key, r)), az, d)
                effect[("radar:" + r, it.key, meth)] = effect_stats(fe, fn, fu, s_["te"], s_["tn"], s_["tu"])

    # self-checks
    checks: list[tuple[str, bool, str]] = []
    if not args.skip_checks:
        checks = self_checks(its, sites, est_all, deploy)
    hetero = {}
    for (key, r), smp in gt_samp.items():
        per = []
        for fl in sorted(set(smp["flight"].tolist())):
            m = smp["flight"] == fl
            if m.sum() >= MIN_GT_SAMPLES:
                per.append((np.median(smp["e"][m] - smp["te"][m]), np.median(smp["n"][m] - smp["tn"][m])))
        hetero[(key, r)] = np.array(per).reshape(-1, 2)
    write_report(its, testing, est_all, val, effect, rows, pooled, checks, sites, time.time() - t_start, hetero, latency)
    log(f"wrote {OUT / 'offsets.csv'}, {OUT / 'pairs.csv'}, {OUT / 'registration.md'}")
    n_fail = sum(not ok for _, ok, _ in checks)
    for name, ok, msg in checks:
        log(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")
    log(f"self-checks: {len(checks) - n_fail}/{len(checks)} passed; runtime {time.time() - t_start:.0f} s")
    return 1 if n_fail else 0


# --------------------------------------------------------------------------- #
# Self-checks
# --------------------------------------------------------------------------- #


def self_checks(its, sites, est_all, deploy) -> list[tuple[str, bool, str]]:
    checks = []
    by_key = {it.key: it for it in its}
    # pick the (testing iteration, radar) with the most 'all' co-observation groups and a vertical estimate
    best = None
    for key, est in est_all.items():
        if not key.startswith("testing"):
            continue
        for (r, rs), e in est.items():
            if rs == "all" and r not in TESTING_ONLY_RADARS and np.isfinite(e["d"][2]):
                if best is None or e["n_groups"] > best[0]:
                    best = (e["n_groups"], key, r)
    # (1a) synthetic radar from references + known offset + slow noise + clutter
    ok1, msgs = True, []
    if best is None:
        ok1, msgs = False, ["no testing (iteration, radar) with a vertical estimate"]
    else:
        _, key, r = best
        it = by_key[key]
        with gt_forbidden():
            data = load_iteration(it)
        ref = ref_samples(data, REF_SETS["all"])
        rng = np.random.default_rng(SEED)
        # all reference samples (moving or not), per segment AR(1) slow error sd 5 m
        parts = []
        for s in REF_SETS["all"]:
            if s in data:
                d = data[s]
                parts.append({**{k: v.copy() for k, v in d.items()}, "sid": np.full(d["seg"].size, s, object)})
        src = concat(parts)
        nse = np.zeros((src["seg"].size, 3))
        for sg in np.unique(src["seg"]):
            idx = np.nonzero(src["seg"] == sg)[0]
            x = np.zeros(3)
            for i in idx:
                x = 0.98 * x + math.sqrt(1 - 0.98 ** 2) * 5.0 * rng.standard_normal(3)
                nse[i] = x
        syn = {"seg": src["seg"] + 10**7, "step": src["step"].copy(),
               "e": src["e"] + SYN_OFFSET[0] + nse[:, 0], "n": src["n"] + SYN_OFFSET[1] + nse[:, 1],
               "u": src["u"] + SYN_OFFSET[2] + nse[:, 2], "ve": src["ve"].copy(), "vn": src["vn"].copy()}
        # clutter: 3 random samples per step near each reference sample, random velocities
        k = 3
        cl = {"seg": np.repeat(np.arange(src["seg"].size) // 50, k) + 2 * 10**7,
              "step": np.repeat(src["step"], k),
              "e": np.repeat(src["e"], k) + rng.uniform(-150, 150, src["seg"].size * k),
              "n": np.repeat(src["n"], k) + rng.uniform(-150, 150, src["seg"].size * k),
              "u": np.repeat(src["u"], k) + rng.uniform(-50, 50, src["seg"].size * k),
              "ve": rng.uniform(-8, 8, src["seg"].size * k), "vn": rng.uniform(-8, 8, src["seg"].size * k)}
        rad = {c: np.concatenate([syn[c], cl[c]]) for c in syn}
        e = estimate(ref, rad, "synthetic")
        err = np.abs(e["d"] - np.array(SYN_OFFSET))
        inside = (e["lo"] <= np.array(SYN_OFFSET)) & (np.array(SYN_OFFSET) <= e["hi"])
        ok_a = bool(np.all(err <= SYN_TOL_M) and np.all(inside))
        msgs.append(f"synthetic radar ({short(key)}): est {np.round(e['d'], 2).tolist()} vs {list(SYN_OFFSET)}, "
                    f"|err| {np.round(err, 2).tolist()}, CI {np.round(e['lo'], 1).tolist()}..{np.round(e['hi'], 1).tolist()} "
                    f"-> {'ok' if ok_a else 'FAIL'}")
        # (1b) inject into the real radar
        base = estimate(ref, data[r], f"{key}|{r}|all")
        rad2 = {c: v.copy() for c, v in data[r].items()}
        rad2["e"] += SYN_OFFSET[0]; rad2["n"] += SYN_OFFSET[1]; rad2["u"] += SYN_OFFSET[2]
        inj = estimate(ref, rad2, f"{key}|{r}|all")
        expect = base["d"] + np.array(SYN_OFFSET)
        err2 = np.abs(inj["d"] - expect)
        inside2 = (inj["lo"] <= expect) & (expect <= inj["hi"])
        ok_b = bool(np.all(err2 <= SYN_TOL_M) and np.all(inside2))
        msgs.append(f"injected into {r} {short(key)} ({len(base['pairs'])} pairs, {base['n_groups']} groups): recovered shift "
                    f"{np.round(inj['d'] - base['d'], 2).tolist()}, |err| {np.round(err2, 2).tolist()}, expected inside CI: "
                    f"{inside2.tolist()} -> {'ok' if ok_b else 'FAIL'}")
        # (1c) inject an azimuth rotation of SYN_AZ_DEG about the radar site into the real radar
        site = sites.get((key, r))
        ok_c = False
        if site is not None:
            base_az = estimate_az(ref, data[r], site, f"{key}|{r}|all")
            rad3 = {c: v.copy() for c, v in data[r].items()}
            rad3["e"], rad3["n"], _ = correct(rad3["e"], rad3["n"], rad3["u"], site, -SYN_AZ_DEG, np.zeros(3))
            inj_az = estimate_az(ref, rad3, site, f"{key}|{r}|all")
            if base_az and inj_az:
                exp_az = base_az["az"] + SYN_AZ_DEG
                ok_c = abs(inj_az["az"] - exp_az) <= SYN_AZ_TOL_DEG and inj_az["lo"] <= exp_az <= inj_az["hi"]
                msgs.append(f"azimuth +{SYN_AZ_DEG:g} deg injected into {r}: recovered {inj_az['az'] - base_az['az']:.3f} deg, "
                            f"expected inside CI [{inj_az['lo']:.2f}..{inj_az['hi']:.2f}]: "
                            f"{inj_az['lo'] <= exp_az <= inj_az['hi']} -> {'ok' if ok_c else 'FAIL'}")
        if not ok_c and site is None:
            msgs.append("azimuth injection: no radar site")
        ok1 = ok_a and ok_b and ok_c
        del data
    checks.append(("1 synthetic offset recovered (<= 2 m, inside its interval)", ok1, "; ".join(msgs)))

    # (2) plain-Python recomputation on the testing (iteration, radar, reference set) with the most
    # co-observation groups among those where one radar segment sits in several pairs (exercises the
    # group weighting); plus a toy case whose second gate stage accepts nothing (stage fallback)
    cand = sorted(((int(any(n > 1 for n in np.bincount(e["sel"]["pair_group"]))), e["n_groups"], len(e["pairs"]), key, r, rs)
                   for key, est in est_all.items() if key.startswith("testing")
                   for (r, rs), e in est.items() if r not in TESTING_ONLY_RADARS and e["sel"] is not None
                   and (key, r) in sites), reverse=True)
    multi, _, _, key2, r, rs = cand[0]
    it = by_key[key2]
    with gt_forbidden():
        data = load_iteration(it)
    ref = ref_samples(data, REF_SETS[rs])
    e = estimate(ref, data[r], "plain", boot=False)
    d_p, nh_p, nu_p = plain_estimate(ref, data[r])
    same = all((math.isnan(a) and math.isnan(b)) or abs(a - b) < 1e-9 for a, b in zip(e["d"].tolist(), d_p))
    ok2 = same and nh_p == e["n_steps"] and nu_p == e["n_steps_u"] and len(e["pairs"]) > 0 and multi == 1
    # azimuth: plain loop over the chosen steps (math.atan2, own groups, own weighted median)
    site = sites.get((key2, r))
    az_np = fit_az(e["sel"], ref, site, "plain", boot=False)
    sel = e["sel"]
    grp: dict[int, int] = {}
    for pi, (a, rr_, *_) in enumerate(sel["pairs"]):   # pairs sharing a radar or a reference segment
        grp[pi] = pi
    changed = True
    while changed:
        changed = False
        for pi, (a, rr_, *_) in enumerate(sel["pairs"]):
            for pj, (b, rb, *_) in enumerate(sel["pairs"]):
                if (a == b or rr_ == rb) and grp[pj] != grp[pi]:
                    grp[pi] = grp[pj] = min(grp[pi], grp[pj])
                    changed = True
    vals, cnt = [], {}
    for i in range(sel["ri"].size):
        x, y = float(ref["e"][sel["ri"][i]]) - site[0], float(ref["n"][sel["ri"][i]]) - site[1]
        if math.hypot(x, y) >= MIN_AZ_RANGE_M:
            g_ = grp[int(sel["pair"][i])]
            cnt[g_] = cnt.get(g_, 0) + 1
    for i in range(sel["ri"].size):
        x, y = float(ref["e"][sel["ri"][i]]) - site[0], float(ref["n"][sel["ri"][i]]) - site[1]
        if math.hypot(x, y) < MIN_AZ_RANGE_M:
            continue
        a_ref = math.degrees(math.atan2(x, y))
        a_rad = math.degrees(math.atan2(x + float(sel["de"][i]), y + float(sel["dn"][i])))
        dd = (a_rad - a_ref + 180.0) % 360.0 - 180.0
        k = cnt[grp[int(sel["pair"][i])]]
        vals.append((dd, min(k, PAIR_CAP_STEPS) / k))
    vals.sort(key=lambda t: t[0])
    tot, c_, az_p = sum(w for _, w in vals), 0.0, float("nan")
    for v_, w_ in vals:
        c_ += w_
        if c_ >= 0.5 * tot:
            az_p = v_
            break
    ok2b = az_np is not None and abs(az_np["az"] - az_p) < 1e-9
    # toy: two drones with offsets (-40, +40) and (+40, -40): stage 0 (gate 100 m around 0) accepts
    # both, the per-axis weighted median is (-40, -40), and the 50 m gate around it accepts neither;
    # the stage-0 estimate must be kept (numpy and plain Python)
    st = np.r_[np.arange(40), np.arange(100, 140)].astype(np.int64)
    toy_ref = {"seg": np.r_[np.full(40, 1), np.full(40, 2)].astype(np.int64), "step": st,
               "e": st * 1.0, "n": np.zeros(80), "u": np.full(80, 50.0), "ve": np.full(80, 10.0), "vn": np.zeros(80),
               "sid": np.full(80, "STUG", dtype=object)}
    off = np.r_[np.full(40, -40.0), np.full(40, 40.0)]
    toy_rad = {"seg": np.r_[np.full(40, 7), np.full(40, 8)].astype(np.int64), "step": st.copy(),
               "e": toy_ref["e"] + off, "n": toy_ref["n"] - off, "u": toy_ref["u"] + 3.0,
               "ve": toy_ref["ve"].copy(), "vn": toy_ref["vn"].copy()}
    te = estimate(toy_ref, toy_rad, "toy", boot=False)
    tp = plain_estimate(toy_ref, toy_rad)[0]
    ok2c = (np.allclose(te["d"], [-40.0, -40.0, 3.0]) and np.allclose(tp, [-40.0, -40.0, 3.0]) and te["stages"] == 1
            and te["n_groups"] == 2)
    checks.append(("2 estimator == plain-Python loop", ok2 and ok2b and ok2c,
                   f"{r}/{rs} {short(key2)}: numpy {np.round(e['d'], 4).tolist()} n={e['n_steps']}/{e['n_steps_u']}, "
                   f"python {[round(x, 4) for x in d_p]} n={nh_p}/{nu_p}, pairs {len(e['pairs'])}, groups {e['n_groups']}, "
                   f"a radar / reference segment in several pairs: {bool(multi)}; azimuth numpy "
                   f"{az_np['az'] if az_np else float('nan'):.6f} vs python {az_p:.6f} deg; empty-second-stage toy: numpy "
                   f"{np.round(te['d'], 3).tolist()} python {[round(x, 3) for x in tp]} (expected [-40, -40, 3], "
                   f"stage-0 kept) -> {'ok' if ok2c else 'FAIL'}"))
    del data

    # (3) references deployable, never testing-only; GT guard works; evaluation uses its own files
    refs_all = set(REF_SETS["all"])
    used = {s for est in est_all.values() for e in est.values() for s in e["refs_used"].split("|") if s}
    ok_ref = refs_all <= set(deploy) and not (used - refs_all) and not (set(RADARS) & refs_all)
    guard_ok = True
    with gt_forbidden():
        for fn in (lambda: T.load_truth(by_key["testing/S1-I1"]), lambda: Q.load_groundtruth(),
                   lambda: C.read_columns(TRACKS / "testing__S4-I4__tracks.csv", usecols=["label"])):
            try:
                fn()
                guard_ok = False
            except GroundTruthAccess:
                pass
    # every chosen step of every evaluation estimate must be a (segment, step) sample of that
    # iteration's own grid file, its radar segment a segment of that radar and its reference
    # segments segments of the reference set in that iteration's tracks.csv (read here
    # independently); a mutation (steps moved past the grid) must be detected
    own_ok, n_chk, mut_caught = True, 0, True
    for it in its:
        if it.tree != "evaluation":
            continue
        slug = C.iter_slug(it.key)
        tr = C.read_columns(TRACKS / f"{slug}__tracks.csv", numeric={"seg_id"}, usecols=["seg_id", "iter_key", "sensor_id"])
        own_ok &= bool(np.all(tr["iter_key"] == it.key))
        sid_of = dict(zip(tr["seg_id"].astype(np.int64).tolist(), tr["sensor_id"].tolist()))
        with np.load(TRACKS / f"{slug}__grid.npz") as g:
            gkey = np.unique(g["seg_id"].astype(np.int64) * 10**8 + g["step"].astype(np.int64))
            smax = int(g["step"].max())

        def member(sel, r_, rs_, shift=0):
            k_ = sel["rad_seg"].astype(np.int64) * 10**8 + sel["step"].astype(np.int64) + shift
            ok_ = bool(np.all(np.isin(k_, gkey)))
            ok_ &= all(sid_of.get(int(a)) == r_ for a in np.unique(sel["rad_seg"]).tolist())
            ok_ &= all(sid_of.get(int(p[1])) == p[2] and p[2] in REF_SETS[rs_] for p in sel["pairs"])
            return ok_

        for (r_, rs_), e_ in est_all[it.key].items():
            if e_["sel"] is None:
                continue
            n_chk += 1
            own_ok &= member(e_["sel"], r_, rs_)
            mut_caught &= not member(e_["sel"], r_, rs_, shift=smax + 1)
    own_ok = own_ok and mut_caught and n_chk > 0
    checks.append(("3 no testing-only / GT input to the estimator", ok_ref and guard_ok and own_ok,
                   f"refs {sorted(refs_all)} deployable: {refs_all <= set(deploy)}; refs used {sorted(used)}; "
                   f"GT guard raises on truth / groundtruth / labels: {guard_ok}; {n_chk} evaluation estimates use only "
                   f"samples / segments of their own iteration's files: {own_ok} (mutation detected: {mut_caught})"))

    # (4) determinism: rerun two iterations (one testing, one evaluation) with bootstrap
    ok4, msgs4 = True, []
    for key in (PLAIN_ITER, "evaluation/DIII/S2-I3"):
        it = by_key[key]
        with gt_forbidden():
            data = load_iteration(it)
            again = run_estimates(it, data, sites)
        for k, e in again.items():
            o = est_all[key][k]
            same = (np.array_equal(e["d"], o["d"], equal_nan=True) and np.array_equal(e["lo"], o["lo"], equal_nan=True)
                    and np.array_equal(e["hi"], o["hi"], equal_nan=True) and len(e["pairs"]) == len(o["pairs"])
                    and e["n_groups"] == o["n_groups"] and (e["az"] is None) == (o["az"] is None)
                    and (e["az"] is None or np.array_equal([e["az"]["az"], e["az"]["lo"], e["az"]["hi"]],
                                                           [o["az"]["az"], o["az"]["lo"], o["az"]["hi"]], equal_nan=True)))
            ok4 &= same
        msgs4.append(f"{short(key)}: {len(again)} estimates identical: {ok4}")
        del data
    checks.append(("4 determinism", ok4, "; ".join(msgs4)))
    return checks


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def write_report(its, testing, est_all, val, effect, rows, pooled, checks, sites, runtime, hetero, latency) -> None:
    L = ["# Radar registration: radar offsets without ground truth", ""]
    n_fail = sum(not ok for _, ok, _ in checks)
    L += [f"**Self-checks: {'all passed' if checks and not n_fail else ('not run' if not checks else f'{n_fail} FAILED')}.** "
          f"Generated by `tools/radar_registration.py` (rules and schemas in its docstring); runtime {runtime:.0f} s.", ""]
    # key findings (numbers from this run)
    def beats(meth, scope="deployable"):
        k = n = 0
        for it in testing:
            if (scope, it.key, "none") in effect:
                n += 1
                k += int(effect[(scope, it.key, meth)]["med3"] < effect[(scope, it.key, "none")]["med3"] - 1e-9)
        return k, n
    def hmed(meth):
        return ", ".join(f"{short(it.key)} {f1(effect[('deployable', it.key, 'none')]['med_h'])} -> "
                         f"{f1(effect[('deployable', it.key, meth)]['med_h'])}" for it in testing
                         if ("deployable", it.key, "none") in effect)
    azs = []
    for r in RADARS:
        a = np.array([val[(it.key, r)]["gt_az"] for it in testing if (it.key, r) in val])
        a = a[np.isfinite(a)]
        if a.size:
            azs.append(f"{r} {a.mean():+.2f} deg (sd {a.std(ddof=1) if a.size > 1 else float('nan'):.2f}, {a.size} iterations)")
    byk = {(r_["iter_key"], r_["sensor_key"], r_["method"]): r_ for r_ in rows}

    def cover(meth):
        """Testing, deployable radars with a GT offset / angle: (estimates, with an interval, GT inside on
        every horizontal axis (az rows: the angle), reliable estimates closer to GT than no correction, reliable)."""
        n_est = n_int = n_in = n_rel_better = n_rel = 0
        for it in testing:
            for r in RADARS:
                rr = byk.get((it.key, r, meth))
                v = val.get((it.key, r))
                if rr is None or v is None or r in TESTING_ONLY_RADARS:
                    continue
                if "_az" in meth:
                    ga = v["gt_az"]
                    if not (np.isfinite(ga) and np.isfinite(rr["az_deg"])):
                        continue
                    n_est += 1
                    if np.isfinite(rr["az_lo"]):
                        n_int += 1
                        n_in += int(rr["az_lo"] <= ga <= rr["az_hi"])
                    continue
                g = v["gt"]
                if not (np.isfinite(g[0]) and np.isfinite(rr["d_e"])):
                    continue
                n_est += 1
                if np.isfinite(rr["d_e_lo"]):
                    n_int += 1
                    n_in += int(rr["d_e_lo"] <= g[0] <= rr["d_e_hi"] and rr["d_n_lo"] <= g[1] <= rr["d_n_hi"])
                if rr["reliable"]:
                    n_rel += 1
                    n_rel_better += int(rr["err_h"] < float(np.hypot(g[0], g[1])))
        return n_est, n_int, n_in, n_rel_better, n_rel

    pur = {}
    for rs in REF_SETS:
        fl = [f for it in testing for r in RADARS if (it.key, r) in val and r not in TESTING_ONLY_RADARS
              for f in val[(it.key, r)][rs]["same_flags"]]
        ng = [est_all[it.key][(r, rs)]["n_groups"] for it in testing for r in RADARS
              if (r, rs) in est_all[it.key] and r not in TESTING_ONLY_RADARS]
        pur[rs] = (sum(fl), len(fl), sum(1 for x in ng if 0 < x < MIN_GROUPS), sum(1 for x in ng if x > 0))
    rot_txt, mostly, partly = [], [], []
    for r in RADARS:
        if r in TESTING_ONLY_RADARS:
            continue
        pr = [(effect[("radar:" + r, it.key, "none")]["med_h"], effect[("radar:" + r, it.key, "loio_az")]["med_h"])
              for it in testing if ("radar:" + r, it.key, "none") in effect]
        if not pr:
            continue
        b0, b1 = float(np.median([x for x, _ in pr])), float(np.median([y for _, y in pr]))
        drop = 1.0 - b1 / b0 if b0 > 0 else np.nan
        rot_txt.append(f"{r} {f1(b0)} -> {f1(b1)} m ({drop:.0%} removed, {len(pr)} iterations, per-iteration "
                       f"after: {f1(min(y for _, y in pr))}..{f1(max(y for _, y in pr))} m)")
        (mostly if drop >= 0.5 else partly).append(r)
    cv = {m: cover(m) for m in ("gt_free_noutm", "gt_free_all", "gt_free_az_noutm", "gt_free_az_all", "loio_az")}
    L += ["## Key findings", "",
          f"* **An azimuth (bearing) rotation is the largest stable part of the error for {', '.join(mostly) or 'no radar'}; "
          f"for {', '.join(partly) or 'no radar'} it is a smaller part.** Seen from the radar, each radar turns the drones by "
          "a nearly fixed angle (GT, testing): " + "; ".join(azs) +
          ". At 1-2 km range, 1 deg is 17-35 m sideways. Median over testing iterations of the radar's median horizontal "
          "error, no correction -> rotated back by the LOIO angle: " + "; ".join(rot_txt) +
          ". Where the drop is below 50 % most of the horizontal error is something else (not modelled here). Because it is "
          "a rotation, the east / north offset changes with where the drone is, so drones in different directions give "
          "different constant offsets (section 1).",
          f"* **Rotating each radar back by the LOIO angle** (`loio_az`: mean GT angle of the other testing iterations) lowers "
          f"the median 3-D error of drone radar samples on {beats('loio_az')[0]} of {beats('loio_az')[1]} testing iterations; "
          f"median horizontal error (m) none -> loio_az, all deployable radars: {hmed('loio_az')}.",
          f"* **The GT-free angle** (`gt_free_az_*`: angle radar - reference about the radar site, from the iteration's own "
          f"co-observations) improves on {beats('gt_free_az_noutm')[0]} of {beats('gt_free_az_noutm')[1]} testing iterations "
          f"without the UTM relays (the honest number; no estimate = no change on the rest). With them it is "
          f"{beats('gt_free_az_all')[0]} of {beats('gt_free_az_all')[1]}, but **on testing VKYY / QTVS are the Dronetag ground "
          "truth itself, so that number is partly GT-fed and optimistic**. Its error against the GT angle is larger than the "
          "LOIO angle's (section 3).",
          f"* **The constant east / north offset** (the brief's estimator) is not worth applying: GT-free constant offsets "
          f"beat no correction on {beats('gt_free_noutm')[0]} / {beats('gt_free_all')[0]} of 6 (noutm / all, the latter "
          f"GT-fed on testing), the LOIO pooled constant on {beats('loio_pooled')[0]} of 6. Reasons: the rotation above, few "
          "co-observed drones, wrong pairs, and the cyber / passive-RF references trail the moving drone (reference latency "
          "table in section 2). A lag-aware fit (`gt_free_lag_*`) does not fix it.",
          f"* **Pair selection is impure and the evidence is thin.** Only {pur['noutm'][0]} of {pur['noutm'][1]} accepted "
          f"pairs (noutm) and {pur['all'][0]} of {pur['all'][1]} (all) are truly one drone by GT labels (deployable radars, "
          f"testing). {pur['noutm'][2]} of {pur['noutm'][3]} (noutm) and {pur['all'][2]} of {pur['all'][3]} (all) estimates "
          f"rest on fewer than {MIN_GROUPS} independent co-observation groups: those have `reliable` = 0 and no interval.",
          "* **The intervals are sampling intervals, not 95 % intervals for the true offset**: they resample the co-observed "
          "drones only and miss wrong pairs and the references' own error. GT inside the interval (testing, deployable, "
          "cases with an interval): " + "; ".join(
              f"{m} {cv[m][2]} of {cv[m][1]} ({cv[m][0]} estimates)" for m in cv) +
          ". Do not gate or weight a correction by `ci` as if it were calibrated. Reliable constant estimates closer to GT "
          f"than no correction: noutm {cv['gt_free_noutm'][3]} of {cv['gt_free_noutm'][4]}, all {cv['gt_free_all'][3]} of "
          f"{cv['gt_free_all'][4]}.",
          "* IUKL's angle is not stable (it changes sign between iterations), so a pooled angle should not be applied "
          "to it; EUIG is testing-only.",
          "* Vertical: only a constant up offset (weighted median) is estimated; no range or elevation dependence of the "
          "vertical error is analysed here.", ""]
    L += ["## Terms", "",
          "* **offset (registration error)**: the slowly varying difference between where a radar puts a drone and where "
          "it is; east / north / up in metres (`d_e`, `d_n`, `d_u`). Subtracting it from the radar's positions is *registration*.",
          "* **reference**: a sensor that reports the drone's own GPS (cyber / passive-RF decoders STUG, CEQX, ILYS, IKJX, SMFS; "
          "UTM relays VKYY, QTVS). `noutm` = without the UTM relays (honest on testing, where VKYY / QTVS are the Dronetag "
          "ground truth itself); `all` = with them (what evaluation looks like).",
          "* **GT-free**: computed without ground truth or labels, only from the iteration's own sensor data, so it can run "
          "on evaluation. **GT offset**: the same offset measured against Dronetag ground truth (testing only; validation).",
          "* **pair**: one radar track segment and one reference segment following the same moving object (velocity within "
          f"{DV_GATE_MS:g} m/s, position within {GATE1_M:g} m of the current offset, >= {MIN_PAIR_STEPS / 10:g} s, "
          f">= {MIN_SHARE:.0%} of the time they overlap). **purity**: share of pairs that are truly one drone (GT labels).",
          "* **co-observation group**: pairs linked by a shared radar or reference segment (e.g. one radar track matched to "
          "both the STUG and the QTVS stream of one drone) count as one independent observation. **reliable**: the estimate "
          f"rests on >= {MIN_GROUPS} groups.",
          "* **weighted median**: the middle value with each group's steps counted at most "
          f"{PAIR_CAP_STEPS / 10:g} s, so one long track (or a drone seen by several references) cannot decide alone.",
          "* **sampling interval (CI)**: from a block bootstrap: redraw the groups with replacement many times and "
          "recompute; estimate +- t(G-1) x the replicates' sd (small-sample t, G groups). It measures the spread between co-observed drones only, not wrong pairs or "
          "the references' own error, so it is NOT a calibrated 95 % interval. **coverage**: how often the GT offset falls "
          "inside.",
          "* **t-based prediction interval** (LOIO rows): mean +- t(n-1) sd sqrt(1 + 1/n) over the n pooled iterations; "
          "t (Student) replaces 1.96 because n is 2-5.",
          "* **LOIO pooled**: leave-one-iteration-out: for testing iteration k, the mean GT offset of the other 5.",
          "* **range / azimuth model**: offset that grows with distance (range scale `k`) or turns with bearing "
          "(azimuth rotation `delta`) instead of a constant.",
          "* **azimuth rotation (`az_deg`)**: the radar reports every bearing turned by the same small angle; correcting "
          "it = rotating the radar's positions about the radar site by -az_deg. **site**: the radar's own position "
          "(median sensor position of its reports).",
          "* **reference latency**: the cyber / passive-RF feeds report where the drone was a moment ago, so on a moving "
          "drone they trail behind it (along-track error).",
          "* **3-D error**: distance between a radar sample and the true drone position; shares within 10 / 25 / 50 m.",
          ""]
    # stability
    L += ["## 1. Is each radar's GT offset stable across testing iterations?", "",
          "GT offset (radar - truth, median over the radar's drone_air samples) per testing iteration, metres e / n / u. "
          "*spread* = sd across iterations, *size* = RMS over iterations; spread << size would make one pooled offset usable.", ""]
    hdr = ["radar"] + [short(it.key) for it in testing] + ["mean", "spread e / n / u", "size e / n / u"]
    trs = []
    for r in RADARS:
        vals = [val[(it.key, r)]["gt"] if (it.key, r) in val else np.full(3, np.nan) for it in testing]
        a = np.array(vals)
        cells = [("-" if not np.isfinite(v[0]) else f"{f1(v[0])} / {f1(v[1])} / {f1(v[2])}") for v in vals]
        fin = np.isfinite(a)
        mean = [np.nanmean(a[:, k]) if fin[:, k].any() else np.nan for k in range(3)]
        sd = [np.nanstd(a[:, k], ddof=1) if fin[:, k].sum() > 1 else np.nan for k in range(3)]
        rms = [np.sqrt(np.nanmean(a[:, k] ** 2)) if fin[:, k].any() else np.nan for k in range(3)]
        trs.append([r + (" (testing-only)" if r in TESTING_ONLY_RADARS else "")] + cells +
                   [" / ".join(f1(x) for x in mean), " / ".join(f1(x) for x in sd), " / ".join(f1(x) for x in rms)])
    L += md_table(hdr, trs) + [""]

    L += ["### Within one iteration: does the offset differ from drone to drone?", "",
          f"GT offset per flight (drone_air samples of that flight, >= {MIN_GT_SAMPLES}), horizontal: the iteration offset, "
          "the spread (sd) of the per-flight offsets and their range. A spread comparable to the iteration offset means one "
          "constant per (iteration, radar) cannot remove most of the error, and an estimate from the few co-observed "
          "drones inherits their own offset.", ""]
    hrows = []
    for it in testing:
        for r in RADARS:
            if (it.key, r) not in hetero or (it.key, r) not in val:
                continue
            h = hetero[(it.key, r)]
            g = val[(it.key, r)]["gt"]
            if h.shape[0] == 0:
                continue
            sd = np.std(h, axis=0, ddof=1) if h.shape[0] > 1 else np.full(2, np.nan)
            hrows.append([short(it.key), r, h.shape[0], f"{f1(g[0])} / {f1(g[1])}", f"{f1(sd[0])} / {f1(sd[1])}",
                          f"{f1(h[:, 0].min())}..{f1(h[:, 0].max())} / {f1(h[:, 1].min())}..{f1(h[:, 1].max())}"])
    L += md_table(["iteration", "radar", "flights", "iteration offset e / n", "per-flight sd e / n",
                   "per-flight range e / n"], hrows) + [""]

    L += ["## 2. GT-free estimate vs GT offset (testing)", "",
          "Per (iteration, radar): the estimate, its sampling interval (none with fewer than "
          f"{MIN_GROUPS} co-observation groups), pairs / groups (purity; '*' = `reliable` 0), and the horizontal error "
          "|estimate - GT|. "
          "`none` = the error of not correcting at all (= |GT offset|). Vertical estimates need a reference with a height "
          "(STUG and SMFS report none).", ""]
    hdr = ["iteration", "radar", "GT e / n / u (n)", "noutm est e / n / u", "noutm CI e / n", "noutm pairs / groups (purity)",
           "all est e / n / u", "all pairs / groups", "lag noutm e / n (tau s)", "LOIO e / n / u", "err_h none", "err_h noutm",
           "err_h all", "err_h lag noutm", "err_h lag all", "err_h LOIO"]
    trs = []
    MS = ("gt_free_noutm", "gt_free_all", "gt_free_lag_noutm", "gt_free_lag_all", "loio_pooled")
    tally = {m: [0, 0, 0] for m in MS}   # beats none, exists, total
    cov = {m: [0, 0, 0, 0] for m in MS[:4]}   # inside e, inside n, -, cases
    cov_u = {m: [0, 0] for m in MS[:4]}
    abs_err = {m: [] for m in MS + ("none",)}
    for it in testing:
        for r in RADARS:
            if (it.key, r) not in val:
                continue
            v = val[(it.key, r)]
            g = v["gt"]
            if not np.isfinite(g[0]):
                continue
            a, b, c = byk[(it.key, r, "gt_free_noutm")], byk[(it.key, r, "gt_free_all")], byk[(it.key, r, "loio_pooled")]
            la, lb = byk[(it.key, r, "gt_free_lag_noutm")], byk[(it.key, r, "gt_free_lag_all")]
            none = float(np.hypot(g[0], g[1]))
            dep = r not in TESTING_ONLY_RADARS
            for m, rr in (("gt_free_noutm", a), ("gt_free_all", b), ("gt_free_lag_noutm", la),
                          ("gt_free_lag_all", lb), ("loio_pooled", c)):
                if dep:
                    tally[m][2] += 1
                    if np.isfinite(rr["err_h"]):
                        tally[m][1] += 1
                        abs_err[m].append(rr["err_h"])
                        tally[m][0] += int(rr["err_h"] < none)
                if m in cov and dep and np.isfinite(rr["d_e"]) and np.isfinite(rr["d_e_lo"]):
                    cov[m][0] += int(rr["d_e_lo"] <= g[0] <= rr["d_e_hi"])
                    cov[m][1] += int(rr["d_n_lo"] <= g[1] <= rr["d_n_hi"])
                    cov[m][3] += 1
                    if np.isfinite(rr["d_u"]) and np.isfinite(g[2]) and np.isfinite(rr["d_u_lo"]):
                        cov_u[m][0] += int(rr["d_u_lo"] <= g[2] <= rr["d_u_hi"])
                        cov_u[m][1] += 1
            if dep:
                abs_err["none"].append(none)
            trs.append([short(it.key), r, f"{f1(g[0])} / {f1(g[1])} / {f1(g[2])} ({v['gt_n']})",
                        f"{f1(a['d_e'])} / {f1(a['d_n'])} / {f1(a['d_u'])}",
                        f"{f1(a['d_e_lo'])}..{f1(a['d_e_hi'])} / {f1(a['d_n_lo'])}..{f1(a['d_n_hi'])}",
                        f"{a['n_pairs']} / {a['n_groups']} ({f1(a['purity'], 2)}){'' if a['reliable'] else ' *'}",
                        f"{f1(b['d_e'])} / {f1(b['d_n'])} / {f1(b['d_u'])}",
                        f"{b['n_pairs']} / {b['n_groups']}{'' if b['reliable'] else ' *'}",
                        f"{f1(la['d_e'])} / {f1(la['d_n'])} ({f1(la['lag_tau_s'], 2)})",
                        f"{f1(c['d_e'])} / {f1(c['d_n'])} / {f1(c['d_u'])}",
                        f1(none), f1(a["err_h"]), f1(b["err_h"]), f1(la["err_h"]), f1(lb["err_h"]), f1(c["err_h"])])
    L += md_table(hdr, trs) + [""]
    L += ["Summary over (testing iteration, deployable radar) cases with a GT offset:", ""]
    srows = []
    for m in MS:
        t_ = tally[m]
        med = np.median(abs_err[m]) if abs_err[m] else np.nan
        srows.append([m, f"{t_[1]} of {t_[2]}", f"{t_[0]} of {t_[1]}", f1(med),
                      (f"e {cov[m][0]}/{cov[m][3]}, n {cov[m][1]}/{cov[m][3]}, u {cov_u[m][0]}/{cov_u[m][1]}"
                       if m in cov else "-")])
    srows.append(["none", "-", "-", f1(np.median(abs_err["none"])) if abs_err["none"] else "-", "-"])
    L += md_table(["method", "estimate exists", "closer to GT than no correction", "median err_h (m)",
                   "GT inside the interval (cases with one)"], srows) + [""]

    # diagnostics: gt_same and ref_err
    L += ["### Where the GT-free error comes from (noutm)", "",
          "`gt_same` = radar - truth on the estimator's own chosen steps (drone segments), `ref_err` = reference - truth on "
          "the same steps. The estimate is about gt_same - ref_err: a gap between GT and gt_same is sampling (the pairs "
          "cover other drones / times than all drone_air samples), a non-zero ref_err is the references' own error.", ""]
    drows = []
    for it in testing:
        for r in RADARS:
            if (it.key, r) not in val:
                continue
            v = val[(it.key, r)]
            for rs in ("noutm", "all"):
                dd = v[rs]
                e = est_all[it.key][(r, rs)]
                if e["sel"] is None:
                    continue
                drows.append([short(it.key), r, rs, " / ".join(f1(x) for x in e["d"]),
                              " / ".join(f1(x) for x in dd["gt_same"]), " / ".join(f1(x) for x in dd["ref_err"]),
                              " / ".join(f1(x) for x in v["gt"]), e["refs_used"], f1(dd["purity"], 2)])
    L += md_table(["iteration", "radar", "refs", "estimate e / n / u", "gt_same", "ref_err", "GT offset", "refs used",
                   "purity"], drows) + [""]

    L += ["### Reference latency (testing, GT)", "",
          "Each reference's own error on its moving drone samples, split along and across the direction of flight; "
          "lag = -along / speed (positive = it reports where the drone was that many seconds ago). VKYY / QTVS are the "
          "Dronetag feed in testing.", ""]
    lrows = [[short(k), rid, v["n"], f1(v["along"]), f1(v["cross"]), f1(v["speed"]), f1(v["lag"], 2)]
             for (k, rid), v in sorted(latency.items())]
    L += md_table(["iteration", "reference", "samples", "along-track (m)", "cross-track (m)", "speed (m/s)",
                   "lag (s)"], lrows) + [""]

    # azimuth rotation
    L += ["## 3. Azimuth rotation per radar", "",
          f"GT angle = median over the radar's drone_air samples (truth >= {MIN_AZ_RANGE_M:g} m from the radar) of the "
          "azimuth difference radar - truth seen from the radar site (degrees, positive = clockwise). GT-free = the same "
          "with the reference instead of truth, on the estimator's chosen steps. `|err|` = |estimate - GT angle|; 'in CI' = "
          "GT angle inside the sampling interval (none with fewer than "
          f"{MIN_GROUPS} groups). LOIO = mean GT angle of the other testing iterations, interval = t-based 95 % "
          "prediction interval for a new iteration.", ""]
    hdr = ["radar"] + [short(it.key) for it in testing] + ["mean", "sd"]
    arows = []
    for r in RADARS:
        a = [val[(it.key, r)]["gt_az"] if (it.key, r) in val else np.nan for it in testing]
        fa = np.array(a)[np.isfinite(a)]
        arows.append([r] + [f1(x, 2) for x in a] + [f1(fa.mean(), 2) if fa.size else "-",
                                                    f1(fa.std(ddof=1), 2) if fa.size > 1 else "-"])
    L += md_table(hdr, arows) + [""]
    arows, acc = [], {m: [[], 0, 0, 0] for m in ("gt_free_az_noutm", "gt_free_az_all", "loio_az")}
    for it in testing:
        for r in RADARS:
            if (it.key, r) not in val or not np.isfinite(val[(it.key, r)]["gt_az"]):
                continue
            ga = val[(it.key, r)]["gt_az"]
            cells = [short(it.key), r, f1(ga, 2)]
            for m in ("gt_free_az_noutm", "gt_free_az_all", "loio_az"):
                rr = byk[(it.key, r, m)]
                ok = np.isfinite(rr["az_deg"])
                inci = ok and np.isfinite(rr["az_lo"]) and rr["az_lo"] <= ga <= rr["az_hi"]
                if ok and r not in TESTING_ONLY_RADARS:
                    acc[m][0].append(abs(rr["az_deg"] - ga))
                    acc[m][1] += int(inci)
                    acc[m][2] += 1
                    acc[m][3] += int(bool(np.isfinite(rr["az_lo"])))
                cells.append(f"{f1(rr['az_deg'], 2)} [{f1(rr['az_lo'], 2)}..{f1(rr['az_hi'], 2)}] ({rr['n_pairs']}"
                             + ("" if m == "loio_az" else f" / {rr['n_groups']}") + ")"
                             if ok else "-")
            arows.append(cells)
    L += md_table(["iteration", "radar", "GT angle", "GT-free noutm [CI] (pairs / groups)", "GT-free all [CI] (pairs / groups)",
                   "LOIO [CI] (iterations)"], arows) + [""]
    L += ["Deployable radars: " + "; ".join(
        f"{m}: estimate in {acc[m][2]} cases, median |err| {f1(np.median(acc[m][0]), 2) if acc[m][0] else '-'} deg, "
        f"GT angle in CI {acc[m][1]} of {acc[m][3]} cases with an interval" for m in acc) + ".", ""]

    # range / azimuth model
    L += ["## 4. Constant offset vs range / azimuth model (GT-free fit)", "",
          "In-sample median horizontal residual of the chosen steps after the constant offset vs after the range / "
          f"azimuth model (fitted only with >= {MIN_ROT_GROUPS} co-observation groups and an azimuth spread >= {MIN_ROT_AZ_SPREAD_DEG:g} deg). "
          "The model always fits at least as well in-sample (more parameters); whether it transfers is the effect check below "
          "(`gt_free_rot`).", ""]
    rrows = []
    for it in its:
        for (r, rs), e in sorted(est_all[it.key].items()):
            if e.get("rot") is None:
                continue
            ro = e["rot"]
            rrows.append([short(it.key), r, rs, len(e["pairs"]), f1(e["res_const_med"]), f1(ro["res_rot_med"]),
                          f"{ro['k'] * 100:.2f} %", f"{math.degrees(ro['delta']):.2f}", f1(ro["az_spread_deg"], 0)])
    L += (md_table(["iteration", "radar", "refs", "pairs", "residual const (m)", "residual range/az (m)", "range scale k",
                    "rotation (deg)", "azimuth spread (deg)"], rrows) if rrows else ["No (iteration, radar) has enough pairs "
                                                                                   "and azimuth spread for the model."]) + [""]

    # effect check
    L += ["## 5. Effect on the radar positions (testing, GT)", "",
          "All drone_air grid samples of the deployable radars (EGOI, NYTF, UXYJ, IUKL) with airborne truth, corrected by "
          "each method (no estimate -> no change). Median 3-D error (m) / share within 10 / 25 / 50 m (3-D). Methods: "
          "`gt_free_*` constant offset (noutm / all references), `gt_free_lag_*` lag-aware constant, `gt_free_rot` range / "
          "azimuth model, `gt_free_az_*` azimuth rotation, all GT-free; `loio_pooled` LOIO constant, `loio_az` LOIO angle, "
          "`loio_az_const` LOIO angle + LOIO constant left after it; vertical: the method's up offset (az methods: the "
          "same reference set's / LOIO up offset). `oracle*` = the iteration's own GT values (in-sample upper bounds, not "
          "methods).", ""]
    methods = EFFECT_METHODS
    main_methods = ("none", "gt_free_noutm", "gt_free_all", "gt_free_az_noutm", "gt_free_az_all", "loio_pooled",
                    "loio_az", "loio_az_const", "oracle", "oracle_az")
    for scope in ("deployable", "EUIG"):
        erows, beats = [], {m: [0, 0] for m in methods}
        for it in testing:
            if (scope, it.key, "none") not in effect:
                continue
            base = effect[(scope, it.key, "none")]
            cells = [short(it.key), str(base["n3"])]
            for m in methods:
                s = effect[(scope, it.key, m)]
                cells.append(f"{f1(s['med3'])} / {s['w10']:.0%} / {s['w25']:.0%} / {s['w50']:.0%}")
                beats[m][1] += 1
                beats[m][0] += int(s["med3"] < base["med3"] - 1e-9)
            erows.append(cells)
        if not erows:
            continue
        keep = [2 + methods.index(m) for m in main_methods]
        erows = [r_[:2] + [r_[k] for k in keep] for r_ in erows]
        L += [f"**{scope}**", ""] + md_table(["iteration", "3-D samples"] + list(main_methods), erows) + [""]
        L += ["Lower median 3-D error than no correction: " + ", ".join(
            f"{m} {beats[m][0]} of {beats[m][1]}" for m in methods if m != "none") + ".", ""]
        hrows = []
        for it in testing:
            if (scope, it.key, "none") not in effect:
                continue
            hrows.append([short(it.key)] + [f1(effect[(scope, it.key, m)]["med_h"]) for m in methods])
        L += ["Median horizontal error (m):", ""] + md_table(["iteration"] + list(methods), hrows) + [""]

    # evaluation
    L += ["## 6. Evaluation estimates", "",
          "`gt_free_all` is the deployable GT-free estimate (references with the UTM relays); `gt_free_noutm` without them; "
          "`loio_pooled` / `loio_az` = mean GT offset / angle of all 6 testing iterations (nothing fitted on evaluation). "
          "'-' = no reference co-observation: no estimate.", ""]
    erows = []
    for it in its:
        if it.tree != "evaluation":
            continue
        for r in RADARS:
            if (it.key, r, "gt_free_all") not in byk:
                continue
            a, b, c = byk[(it.key, r, "gt_free_all")], byk[(it.key, r, "gt_free_noutm")], byk[(it.key, r, "loio_pooled")]
            flag = ("" if a["reliable"] else f" (unreliable: {a['n_groups']} groups)") if np.isfinite(a["d_e"]) \
                else " **no co-observation**"
            erows.append([short(it.key), r + flag, f"{f1(a['d_e'])} / {f1(a['d_n'])} / {f1(a['d_u'])}",
                          f"{f1(a['d_e_lo'])}..{f1(a['d_e_hi'])} / {f1(a['d_n_lo'])}..{f1(a['d_n_hi'])} / "
                          f"{f1(a['d_u_lo'])}..{f1(a['d_u_hi'])}", f"{a['n_pairs']} / {a['n_groups']} / {a['n_steps']}",
                          a["refs_used"] or "-",
                          f"{f1(b['d_e'])} / {f1(b['d_n'])} / {f1(b['d_u'])} ({b['n_pairs']} / {b['n_groups']})",
                          f"{f1(c['d_e'])} / {f1(c['d_n'])} / {f1(c['d_u'])}",
                          f"{f1(byk[(it.key, r, 'gt_free_az_all')]['az_deg'], 2)} [{f1(byk[(it.key, r, 'gt_free_az_all')]['az_lo'], 2)}.."
                          f"{f1(byk[(it.key, r, 'gt_free_az_all')]['az_hi'], 2)}]",
                          f1(byk[(it.key, r, 'gt_free_az_noutm')]['az_deg'], 2),
                          f1(byk[(it.key, r, 'loio_az')]['az_deg'], 2)])
    L += md_table(["iteration", "radar", "all: e / n / u", "all: sampling interval e / n / u", "pairs / groups / steps", "refs used",
                   "noutm (pairs / groups)", "LOIO pooled", "angle GT-free all [CI]", "angle GT-free noutm", "angle LOIO"], erows) + [""]

    L += ["## Self-checks", ""]
    L += md_table(["check", "result", "detail"], [[n, "PASS" if ok else "FAIL", m.replace("|", "/")] for n, ok, m in checks]) \
        if checks else ["Not run (--skip-checks)."]
    L += ["", "## Caveats", "",
          "* The estimate is radar - reference: the references' own error (STUG 3-10 m, slow too) enters it and is not in the "
          "interval. On testing the UTM relays are ground truth, so `gt_free_all` validation there is optimistic.",
          "* sensor_tracks_v2 heights (u_ellipsoid offsets) and the UXYJ clock offset were fitted with GT on all 6 testing "
          "iterations: testing vertical GT offsets are partly removed in-sample.",
          "* Only moving references are used (speed >= 2 m/s); hovering drones give no pairs.", ""]
    (OUT / "registration.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
