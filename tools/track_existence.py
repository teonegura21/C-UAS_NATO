#!/usr/bin/env python3
"""Track-level existence model on the FUSED tracks, and the deployable method F1_fuse
(leave-one-iteration-out, LOIO): report a fused track iff its own calibrated probability of
being a real drone track is high, with a learned fixed-wing / rotary-wing label.

Why: F0_fuse (tools/fuse.py) reports a fused track iff the MAX member-segment existence p is
above a threshold. Its GOSPA cost is 37 % (truth 'all') / 59 % ('airborne') false rows, and the
row purity of its drone tracks is only 0.54-0.74: whether a fused track is worth reporting is a
property of the TRACK (how many sensors agree, how it moves, whether its rows stay on a drone),
not of its best segment. This tool learns that property directly.

Inputs (read only)
------------------
* Fused tracks: fuse.Prep / cluster / fuse / track_table on all 9 iterations with ONE FIXED
  geometry GEOM = edge_gate 'hand', cont_gap_s 10, bridge_s 20. That is fuse's all-6 in-sample
  argmax (the evaluation choice of fuse.md); its tuning objective varies by < 0.002 across the
  geometry grid, so choosing it on all 6 iterations is a negligible selection effect (stated in
  the report). Only deployable segments enter (fuse does this: C.deployable_sensor_ids()).
* Segment features: sensor_tracks_v2/<slug>__tracks.csv (existence.FEATURES columns),
  bearing_support/<slug>__support.csv (camera / acoustic / rf hit features),
  classify/<slug>__pclass.csv (p_fixed per segment: testing = LOIO outer prediction, evaluation
  = final model), existence/<slug>__p.csv through fuse.Prep (ONLY for F0's max-member p, the
  comparison baseline; it is NOT a feature: that would need a second nesting level).
* Ground truth (testing only, targets and diagnostics): truth.load_truth(it, 'all' / 'airborne').

Targets (testing only)
----------------------
For every fused row (bridged rows included), d3 = 3-D distance to the NEAREST truth drone at
that grid step (any drone; no row -> no match). m_<mode> = share of the track's rows with
d3 <= MATCH_M (50 m = the GOSPA cut-off c); y_<mode> = m_<mode> >= Y_SHARE (0.5): a track the
GOSPA could mostly match. mode 'all' -> target y_all, 'airborne' -> y_air. Unlike
fuse.testing_labels' label_drone (>= half of the member grid samples on drone-labelled
segments) this is a statement about the FUSED rows, so a well-labelled track whose fused rows
are > 50 m off (bias, bad height, merge damage) is a negative here; the cross-table is in the
report.

Features (GT-free, one row per fused track; names in model_*.json)
------------------------------------------------------------------
* wm_<c>: n_grid-weighted mean over member segments of every existence.FEATURES column (NaN
  members ignored, NaN if none); max_<c> / min_<c> over members for STRONG.
* fam_<type>: number of member segments per sensor family (sensor_key after the first '_',
  e.g. RADAR, OTHER, PASSIVE-RF--JAMMER--CYBER); sid_<ID>: 1 if a member comes from that
  deployable sensor id.
* n_members, n_sensors, n_clusters, n_steps (fused rows), bridged_share, duration_s.
* fused-grid kinematics: speed_med / p10 / p90 (horizontal, unbridged rows, = track_table),
  slow_share (unbridged rows with speed < 1 m/s), u_med / u_p10 / u_p90 (fused height),
  path_len_m (sum of horizontal row-to-row distances), origin_dist_m (horizontal distance of the
  median fused position from the iteration's local-frame origin).
* has_serial / has_model / has_protocol: the track's resolved metadata is not empty / Unknown.
* bearing support, per family f in (camera, acoustic, rf): bs_<f>_hit_max, bs_<f>_uhit_max
  (max over members of hit_share / unique_hit_share), bs_<f>_hit_wm, bs_<f>_uhit_wm
  (n_reports_overlap-weighted means), bs_<f>_any (1 if any member had an overlapping report).

Model (gbdt.GBDT + gbdt.Isotonic, the nested LOIO of tools/existence.py)
------------------------------------------------------------------------
Combinations: CONFIGS (3 tree settings) x WEIGHTS ('unit' = 1 per track, 'steps' = n_steps
normalised to mean 1 over the training rows). Selection metric: step-weighted (n_steps)
logloss of the raw GBDT output. For held-out testing iteration k: for every j != k a 'pair' model
fitted on testing \\ {k, j} predicts j; the combination with the lowest mean inner metric is
chosen for fold k; the isotonic calibrator is fitted on its pooled inner out-of-fold
predictions (weights as the combination's); the model refitted on testing \\ {k} predicts k
(outer p). p_inner(j, outer_fold = k) = fold k's calibrator applied to the pair model of {k, j}:
never saw k or j, on k's scale. Final model (evaluation): the combination with the lowest mean
6-fold outer metric, fitted on all 6 testing iterations, calibrated on its 6-fold outer OOF.
Naive baseline (self-check 4): training share of y among tracks with the same
(min(n_sensors, 3), main sensor id), Jeffreys (pos + 0.5) / (n + 1); fallback main sensor id,
then the overall share. Main sensor = the member sensor id with the most grid samples (ties
alphabetical, as fuse.track_main_sensor).

Method F1_fuse (deployable)
---------------------------
Report a track iff p_<P_FIELD> >= TAU, P_FIELD in P_FIELDS, TAU in TAUS. For held-out k:
(P_FIELD, TAU) maximises the mean over j != k of (combined_all + combined_airborne) / 2 scoring
j's tracks selected with p_inner(j, outer_fold = k) (re-read from the written file); ties: first
in (P_FIELDS x TAUS) order. Applied to k with k's outer p. Evaluation: the same objective over
all 6 testing iterations with the outer p; evaluation p from the final model.
Labels (constant per track): classification 'Fixed Wing UAS' iff p_fixed (n_grid-weighted mean
member p_fixed of classify) > 0.5 else 'Rotary Wing UAS'; identity / uav_model / protocol /
serial / band exactly as fuse.track_table; classification_confidence = max(p_fixed, 1 - p_fixed).
Rows via fuse.build_rows (track_rows.make_rows), the F0 conventions.
Diagnostics (testing, in memory, never submitted): F1 with F0's classification rule (the
sensors' class strings, cls_fixed_share > 0.5); F1 with every track 'Rotary Wing UAS'; F0's own
rule (max member p >= TAU, fold k's (P_FIELD, TAU) recomputed from fused/tuning.csv) on this
fixed geometry with F0's labels (isolates the geometry change); ORACLE: exactly the y_all
tracks with F1's labels (the ceiling of this track set and this target).

Known second-order leaks (documented, not fixed): j's classify p_fixed and bearing-support
features used while tuning TAU for k came from LOIO models / choices that saw k's labels
(classify and bearing_support have no nested predictions); the sensor_tracks variances,
sensor_noise tables and the honest variance table of j (fuse.Prep) were fitted with k. The
existence p is not a feature, so the nested level that matters for the selection is exact.
VKYY / QTVS relay the Dronetag feed in testing: numbers that use them are optimistic.

Outputs (data/derived/track_existence/)
---------------------------------------
<slug>__tracks.csv   one row per fused track: track, track_id, sensor_ids, main_sensor, n_steps,
    first_step, last_step, serial, model, protocol, identity, class_f0, class_f1, p_fixed,
    pmax_all, pmax_air (F0's max member outer p), f_<feature> for every model feature; testing:
    m_all, m_air, y_all, y_air, label_drone, drone_share (fuse.testing_labels); p_all, p_air
    (calibrated: testing outer LOIO, evaluation final model), fold (held-out key or 'final'),
    reported_f1 (0/1).
<slug>__p_inner.csv  testing only: track, outer_fold (k != this iteration), p_all, p_air.
<slug>__grid.npz     fused rows of this geometry (fuse.fuse): track, step, e, n, u, var_e, var_n,
    var_u, bridged.
model_all.json, model_air.json   {target, features, combo, config, weight, train_iters, gbdt,
    isotonic}.
tuning.csv           k (held-out fold whose p_inner was used, or 'outer'), j, p_field, tau,
    combined_all, track_all, combined_airborne, track_airborne, n_tracks, n_rows, p_fold.
track_existence.md   plain-language report.
Submissions: data/derived/submissions/F1_fuse/ (9 iterations); headline via
score.score_method(log=True), both truth modes.

Run: python3 tools/track_existence.py [--jobs N] [--build-jobs N]
Numbered self-checks at the end; exit status 1 on any failure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines as B  # noqa: E402
import cuas_common as C  # noqa: E402
import existence as E  # noqa: E402
import fuse as F  # noqa: E402
import gbdt as GB  # noqa: E402
import score as SC  # noqa: E402
import sensor_tracks as ST  # noqa: E402
import truth as T  # noqa: E402

OUT = C.DERIVED / "track_existence"
TRACKS_DIR = C.DERIVED / "sensor_tracks_v2"
BS_DIR = C.DERIVED / "bearing_support"
CLS_DIR = C.DERIVED / "classify"
FUSED_TUNING = C.DERIVED / "fused" / "tuning.csv"
FUSED_DIR = C.DERIVED / "fused"
REPORT = OUT / "track_existence.md"
TUNING_CSV = OUT / "tuning.csv"
METHOD = "F1_fuse"
TRUTH_MODES = ("all", "airborne")

GEOM = ("hand", 10.0, 20.0)        # edge_gate, cont_gap_s, bridge_s: fuse's all-6 argmax
MATCH_M = 50.0                     # target: a row within the GOSPA cut-off of a truth drone
Y_SHARE = 0.5                      # y = share of such rows >= this
TARGETS = {"all": "all", "air": "airborne"}      # target -> truth mode
STRONG = ["cls_unknown", "cls_conf_mean", "cls_uav_rotary", "range_min_m", "n_meas",
          "det_conf_mean", "range_med_m", "dz_p90_m", "speed_mean_ms"]
BS_FAMS = ("camera", "acoustic", "rf")
SLOW_MS = 1.0
CONFIGS = {
    "d2": dict(n_trees=200, learning_rate=0.05, max_depth=2, min_samples_leaf=20, seed=0),
    "d3": dict(n_trees=200, learning_rate=0.05, max_depth=3, min_samples_leaf=20, seed=0),
    "d4s": dict(n_trees=300, learning_rate=0.03, max_depth=4, min_samples_leaf=40,
                subsample=0.8, colsample=0.8, seed=0),
}
WEIGHTS = ("unit", "steps")
COMBOS = [(c, w) for c in CONFIGS for w in WEIGHTS]
P_FIELDS = ("all", "air")
TAUS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
SEL_COMBOS = [(pf, tau) for pf in P_FIELDS for tau in TAUS]
JEFFREYS = 0.5
FIXED, ROTARY = "Fixed Wing UAS", "Rotary Wing UAS"
N_CHECK_PER_ITER = 60              # self-check 5: random tracks per testing iteration (>= 300)
RT_TOL_M = 1e-3
CAL_BINS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
RUNTIME_LIMIT_S = 40 * 60
SEED = 0

ITERS = F.ITERS
TESTING = F.TESTING
EVALUATION = F.EVALUATION
ALL_KEYS = TESTING + EVALUATION

# filled before the worker pools fork
D: dict = {}
TUNE: dict = {}
APPLY: dict = {}


def log(msg: str) -> None:
    print(msg, flush=True)


def md_table(header, rows) -> list[str]:
    out = ["| " + " | ".join(str(h) for h in header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def fmt(v, nd: int = 3) -> str:
    return "n/a" if v is None or v != v else f"{v:.{nd}f}"


def short(key: str) -> str:
    return key.split("/", 1)[1]


def combo_name(c: tuple[str, str]) -> str:
    return f"{c[0]}/{c[1]}"


# --------------------------------------------------------------------------- phase A: tracks


def nearest_d3(tr: T.Truth, step: np.ndarray, e: np.ndarray, n: np.ndarray, u: np.ndarray) -> np.ndarray:
    """3-D distance of each row to the nearest truth row at the same grid step (inf if none)."""
    st = np.searchsorted(tr.step, step, side="left")
    en = np.searchsorted(tr.step, step, side="right")
    cnt = en - st
    d = np.full(step.size, np.inf)
    for k in range(int(cnt.max()) if cnt.size else 0):
        m = cnt > k
        i = st[m] + k
        de, dn, du = e[m] - tr.e[i], n[m] - tr.n[i], u[m] - tr.u[i]
        d[m] = np.minimum(d[m], np.sqrt(de * de + dn * dn + du * du))
    return d


def _wmean(mt: np.ndarray, x: np.ndarray, w: np.ndarray, nt: int) -> np.ndarray:
    ok = np.isfinite(x)
    num = np.bincount(mt, np.where(ok, w * np.where(ok, x, 0.0), 0.0), nt)
    den = np.bincount(mt, np.where(ok, w, 0.0), nt)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _reduce(fn, x: np.ndarray, start: np.ndarray) -> np.ndarray:
    return fn.reduceat(x, start) if x.size else np.zeros(0)


def build_task(key: str) -> dict:
    """Fused tracks of the fixed geometry, their GT-free features and (testing) targets."""
    t0 = time.time()
    it = ITERS[key]
    slug = C.iter_slug(key)
    P = F.Prep(it)
    Dg = P.D
    wl = B.load_whitelist()
    Cl = F.cluster(P, GEOM[0], GEOM[1])
    g = F.fuse(P, Cl, GEOM[2])
    tt = F.track_table(P, Cl, g, wl)
    nt = Cl.ntracks
    ms, mt, start = Cl.mem_seg, Cl.mem_track, Cl.mem_start
    w = Dg.g_cnt[ms].astype(float)
    seg = C.read_columns(TRACKS_DIR / f"{slug}__tracks.csv", numeric=set(E.FEATURES) | {"seg_id"},
                         usecols=["seg_id"] + E.FEATURES)
    sup_cols = [f"{f}_{c}" for f in BS_FAMS for c in ("n_reports_overlap", "hit_share", "unique_hit_share")]
    sup = C.read_columns(BS_DIR / f"{slug}__support.csv", numeric=set(sup_cols) | {"seg_id"},
                         usecols=["seg_id"] + sup_cols)
    pc = C.read_columns(CLS_DIR / f"{slug}__pclass.csv", numeric={"seg_id", "p_fixed"})
    for name, tab in (("tracks", seg), ("support", sup), ("pclass", pc)):
        if not np.array_equal(tab["seg_id"].astype(np.int64), Dg.seg_id):
            raise ValueError(f"{key}: {name} seg_id does not match the segment table")
    feats: dict[str, np.ndarray] = {}
    for c in E.FEATURES:
        feats[f"wm_{c}"] = _wmean(mt, seg[c][ms], w, nt)
    for c in STRONG:
        x = seg[c][ms]
        feats[f"max_{c}"] = _reduce(np.fmax, x, start)
        feats[f"min_{c}"] = _reduce(np.fmin, x, start)
    fam_counts: dict[str, np.ndarray] = {}
    sid_present: dict[str, np.ndarray] = {}
    fam = np.array([s.split("_", 1)[1] if "_" in s else s for s in Dg.sensor_key[ms]], dtype=object)
    for f in sorted(set(fam.tolist())):
        fam_counts[f] = np.bincount(mt, (fam == f).astype(float), nt)
    sid = Dg.sensor_id[ms]
    for s in sorted(set(sid.tolist())):
        sid_present[s] = (np.bincount(mt, (sid == s).astype(float), nt) > 0).astype(float)
    feats["n_members"] = tt["n_members"].astype(float)
    feats["n_sensors"] = tt["n_sensors"].astype(float)
    feats["n_clusters"] = tt["n_clusters"].astype(float)
    feats["n_steps"] = tt["n_steps"].astype(float)
    feats["bridged_share"] = tt["n_bridged"] / np.maximum(tt["n_steps"], 1)
    feats["duration_s"] = tt["duration_s"].astype(float)
    tr_ = g["track"]
    ub = ~g["bridged"]
    sp = np.hypot(g["ve"], g["vn"])
    feats["speed_med"], feats["speed_p10"], feats["speed_p90"] = (tt["speed_med_ms"], tt["speed_p10_ms"],
                                                                  tt["speed_p90_ms"])
    with np.errstate(invalid="ignore", divide="ignore"):
        feats["slow_share"] = (np.bincount(tr_[ub], (sp[ub] < SLOW_MS).astype(float), nt)
                               / np.bincount(tr_[ub], minlength=nt))
    q = ST.group_quantile(tr_, g["u"], nt, (0.5, 0.1, 0.9))
    feats["u_med"], feats["u_p10"], feats["u_p90"] = q[0], q[1], q[2]
    same = np.r_[tr_[1:] == tr_[:-1], False]
    dstep = np.zeros(tr_.size)
    if tr_.size > 1:
        dstep[:-1] = np.hypot(np.diff(g["e"]), np.diff(g["n"]))
    feats["path_len_m"] = np.bincount(tr_, np.where(same, dstep, 0.0), nt)
    me = ST.group_quantile(tr_, g["e"], nt, (0.5,))[0]
    mn = ST.group_quantile(tr_, g["n"], nt, (0.5,))[0]
    feats["origin_dist_m"] = np.hypot(me, mn)
    feats["has_serial"] = (tt["serial"] != "").astype(float)
    feats["has_model"] = (tt["model"] != "Unknown").astype(float)
    feats["has_protocol"] = (tt["protocol"] != "Unknown").astype(float)
    for f in BS_FAMS:
        ov = sup[f"{f}_n_reports_overlap"][ms]
        ov = np.nan_to_num(ov)
        for tag, col in (("hit", "hit_share"), ("uhit", "unique_hit_share")):
            x = np.where(ov > 0, sup[f"{f}_{col}"][ms], np.nan)
            feats[f"bs_{f}_{tag}_max"] = _reduce(np.fmax, x, start)
            feats[f"bs_{f}_{tag}_wm"] = _wmean(mt, x, ov, nt)
        feats[f"bs_{f}_any"] = (np.bincount(mt, ov, nt) > 0).astype(float)
    p_fixed = _wmean(mt, pc["p_fixed"][ms], w, nt)
    if not np.all(np.isfinite(p_fixed)):
        raise ValueError(f"{key}: track p_fixed not finite")
    pa, pr, pfold = P.p["outer"]
    pmax_all = F.track_p(Cl, pa, Dg.g_cnt)[0]
    pmax_air = F.track_p(Cl, pr, Dg.g_cnt)[0]
    main = []
    for t in range(nt):
        a, b = start[t], (start[t + 1] if t + 1 < nt else ms.size)
        cnt: dict[str, int] = {}
        for r in ms[a:b].tolist():
            cnt[Dg.sensor_id[r]] = cnt.get(Dg.sensor_id[r], 0) + int(Dg.g_cnt[r])
        main.append(sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0])
    out = {"key": key, "nt": nt, "n_rows": int(tr_.size), "feats": feats, "fam_counts": fam_counts,
           "sid_present": sid_present, "p_fixed": p_fixed, "pmax_all": pmax_all, "pmax_air": pmax_air,
           "p_src_fold": pfold, "main_sensor": np.array(main, dtype=object),
           "sensor_ids": tt["sensor_ids"], "serial": tt["serial"], "model": tt["model"],
           "protocol": tt["protocol"], "lab": tt["lab"], "first_step": tt["first_step"],
           "last_step": tt["last_step"], "stats": Cl.stats, "hash": F.members_hash(Cl),
           "n_mem_nondeploy": int((~P.dep[ms]).sum())}
    if it.has_groundtruth:
        for tgt, mode in TARGETS.items():
            tr = T.load_truth(it, mode)
            d3 = nearest_d3(tr, g["step"], g["e"], g["n"], g["u"])
            near = d3 <= MATCH_M
            out[f"near_{tgt}"] = np.bincount(tr_, near.astype(float), nt)
            out[f"m_{tgt}"] = out[f"near_{tgt}"] / tt["n_steps"]
            out[f"y_{tgt}"] = out[f"m_{tgt}"] >= Y_SHARE
        lab = F.testing_labels(P, Cl, g, T.load_truth(it, "all"))
        out["label_drone"] = lab["label_drone"]
        out["drone_share"] = lab["drone_share"]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{slug}__grid.npz", **{c: g[c] for c in (
        "track", "step", "e", "n", "u", "var_e", "var_n", "var_u", "bridged")})
    del P, Dg, g
    out["seconds"] = time.time() - t0
    return out


# --------------------------------------------------------------------------- phase B: model


def assemble(built: dict) -> None:
    fams = sorted(set().union(*[set(b["fam_counts"]) for b in built.values()]))
    sids = sorted(set().union(*[set(b["sid_present"]) for b in built.values()]))
    base = list(built[ALL_KEYS[0]]["feats"])
    names = base + [f"fam_{f}" for f in fams] + [f"sid_{s}" for s in sids]
    Xs, iter_key = [], []
    for k in ALL_KEYS:
        b = built[k]
        if list(b["feats"]) != base:
            raise SystemExit(f"{k}: feature order differs")
        cols = [b["feats"][c] for c in base]
        cols += [b["fam_counts"].get(f, np.zeros(b["nt"])) for f in fams]
        cols += [b["sid_present"].get(s, np.zeros(b["nt"])) for s in sids]
        Xs.append(np.column_stack(cols) if b["nt"] else np.zeros((0, len(names))))
        iter_key += [k] * b["nt"]
    X = np.vstack(Xs).astype(np.float64)
    if np.isinf(X).any():
        raise SystemExit("infinite feature values")
    iter_key = np.array(iter_key, dtype=object)
    rows = {k: np.flatnonzero(iter_key == k) for k in ALL_KEYS}
    y = {t: np.zeros(X.shape[0]) for t in TARGETS}
    for k in TESTING:
        for t in TARGETS:
            y[t][rows[k]] = built[k][f"y_{t}"].astype(float)
    D.update(X=X, names=names, iter_key=iter_key, rows=rows, y=y, keys=ALL_KEYS, test_keys=TESTING,
             eval_keys=EVALUATION,
             n_steps=np.concatenate([built[k]["feats"]["n_steps"] for k in ALL_KEYS]),
             n_sensors=np.concatenate([built[k]["feats"]["n_sensors"] for k in ALL_KEYS]),
             main=np.concatenate([built[k]["main_sensor"] for k in ALL_KEYS]).astype(str))


def train_rows(train_keys) -> np.ndarray:
    return np.concatenate([D["rows"][k] for k in sorted(train_keys)])


def weights(rows: np.ndarray, scheme: str) -> np.ndarray:
    if scheme == "unit":
        return np.ones(rows.size)
    s = D["n_steps"][rows]
    return s / s.mean()


def fit_task(task: tuple) -> dict:
    """task = (target, combo, train_keys, predict_keys)."""
    target, combo, train_keys, pred_keys = task
    rows = train_rows(train_keys)
    m = GB.GBDT(**CONFIGS[combo[0]]).fit(D["X"][rows], D["y"][target][rows],
                                         sample_weight=weights(rows, combo[1]))
    return {"task": task, "fitted_on": tuple(sorted(set(D["iter_key"][rows].tolist()))),
            "pred": {k: m.predict_proba(D["X"][D["rows"][k]]) for k in pred_keys},
            "model": m.to_dict() if set(train_keys) == set(D["test_keys"]) else None,
            "importance": m.feature_importance("gain")}


def run_tasks(tasks: list[tuple], jobs: int) -> dict:
    if jobs <= 1:
        res = [fit_task(t) for t in tasks]
    else:
        with mp.get_context("fork").Pool(jobs) as pool:
            res = pool.map(fit_task, tasks, chunksize=1)
    return {r["task"]: r for r in res}


def naive_p(target: str, train_keys, rows: np.ndarray) -> np.ndarray:
    tr = train_rows(train_keys)
    y = D["y"][target][tr]
    gk = np.array([f"{min(int(a), 3)}|{b}" for a, b in zip(D["n_sensors"], D["main"])], dtype=object)
    glob = (y.sum() + JEFFREYS) / (y.size + 1.0)
    p = np.full(rows.size, glob)
    for key_arr in (D["main"].astype(object), gk):         # coarse first, then the finer group
        ktr = key_arr[tr]
        for v in np.unique(key_arr[rows]):
            m = ktr == v
            if m.any():
                p[key_arr[rows] == v] = (y[m].sum() + JEFFREYS) / (m.sum() + 1.0)
    return p


def model_phase(jobs: int) -> dict:
    TK, EK = TESTING, EVALUATION
    tasks = []
    for t in TARGETS:
        for c in COMBOS:
            for a, b in combinations(TK, 2):
                tasks.append((t, c, tuple(k for k in TK if k not in (a, b)), (a, b)))
            for k in TK:
                tasks.append((t, c, tuple(x for x in TK if x != k), (k,)))
    log(f"phase B: {len(tasks)} GBDT fits on {jobs} processes ...")
    t1 = time.time()
    R = run_tasks(tasks, jobs)
    log(f"  done in {time.time() - t1:.0f} s")

    def pair(t, c, k, j):
        a, b = sorted((k, j))
        task = (t, c, tuple(x for x in TK if x not in (a, b)), (a, b))
        return R[task], R[task]["pred"][j]

    def outer(t, c, k):
        task = (t, c, tuple(x for x in TK if x != k), (k,))
        return R[task], R[task]["pred"][k]

    s = D["n_steps"]
    guard, sel, inner_tab, p_inner, final_choice, results = [], {}, {}, {}, {}, {}
    for t in TARGETS:
        y = D["y"][t]
        p_cal = np.full(y.size, np.nan)
        p_raw = np.full(y.size, np.nan)
        p_nv = np.full(y.size, np.nan)
        for k in TK:
            inner = {}
            for c in COMBOS:
                vals = []
                for j in TK:
                    if j == k:
                        continue
                    r, pj = pair(t, c, k, j)
                    if k in r["fitted_on"] or j in r["fitted_on"]:
                        guard.append(("combo", k, ("LEAK",), ()))
                    rows = D["rows"][j]
                    vals.append(GB.logloss(y[rows], pj, s[rows]))
                inner[c] = float(np.mean(vals))
            inner_tab[(t, k)] = inner
            best = min(COMBOS, key=lambda c: (inner[c], COMBOS.index(c)))
            sel[(t, k)] = best
            cp, cy, cw, cal_on = [], [], [], set()
            for j in TK:
                if j == k:
                    continue
                r, pj = pair(t, best, k, j)
                cal_on |= set(r["fitted_on"]) | {j}
                rows = D["rows"][j]
                cp.append(pj); cy.append(y[rows])
                cw.append(np.ones(rows.size) if best[1] == "unit" else s[rows])
            iso = GB.Isotonic().fit(np.concatenate(cp), np.concatenate(cy), np.concatenate(cw))
            for j in TK:
                if j != k:
                    p_inner[(t, k, j)] = iso.predict(pair(t, best, k, j)[1])
            r, pk = outer(t, best, k)
            rows = D["rows"][k]
            p_raw[rows] = pk
            p_cal[rows] = iso.predict(pk)
            train_k = tuple(x for x in TK if x != k)
            p_nv[rows] = naive_p(t, train_k, rows)
            guard.append(("outer", k, r["fitted_on"], tuple(sorted(cal_on)), train_k))
        loio = {}
        for c in COMBOS:
            vals = []
            for k in TK:
                _, pk = outer(t, c, k)
                rows = D["rows"][k]
                vals.append(GB.logloss(y[rows], pk, s[rows]))
            loio[c] = float(np.mean(vals))
        final_choice[t] = (min(COMBOS, key=lambda c: (loio[c], COMBOS.index(c))), loio)
        results[t] = dict(p_cal=p_cal, p_raw=p_raw, p_naive=p_nv)

    tasks_f = [(t, final_choice[t][0], tuple(TK), tuple(EK + TK)) for t in TARGETS]
    log("  final models ...")
    RF = run_tasks(tasks_f, min(jobs, len(tasks_f)))
    final = {}
    OUT.mkdir(parents=True, exist_ok=True)
    for t, task in zip(TARGETS, tasks_f):
        c = final_choice[t][0]
        cp, cy, cw = [], [], []
        for k in TK:
            _, pk = outer(t, c, k)
            rows = D["rows"][k]
            cp.append(pk); cy.append(D["y"][t][rows])
            cw.append(np.ones(rows.size) if c[1] == "unit" else s[rows])
        iso = GB.Isotonic().fit(np.concatenate(cp), np.concatenate(cy), np.concatenate(cw))
        r = RF[task]
        final[t] = dict(combo=c, iso=iso, model=r["model"], fitted_on=r["fitted_on"],
                        p_eval={k: iso.predict(r["pred"][k]) for k in EK},
                        importance=r["importance"])
        for k in EK:
            guard.append(("final", k, r["fitted_on"], tuple(TK), tuple(TK)))
        doc = {"target": t, "truth_mode": TARGETS[t], "definition":
               f"y = share of fused rows within {MATCH_M} m 3-D of the nearest truth drone >= {Y_SHARE}",
               "features": D["names"], "combo": combo_name(c), "config": CONFIGS[c[0]], "weight": c[1],
               "train_iters": list(r["fitted_on"]), "geometry": list(GEOM),
               "calibration": "isotonic on the 6-fold LOIO out-of-fold predictions",
               "gbdt": r["model"], "isotonic": iso.to_dict()}
        (OUT / f"model_{t}.json").write_text(json.dumps(doc))
    return dict(R=R, RF=RF, guard=guard, sel=sel, inner_tab=inner_tab, p_inner=p_inner,
                final_choice=final_choice, results=results, final=final)


# --------------------------------------------------------------------------- phase C: scoring


def load_grid(key: str) -> dict:
    z = np.load(OUT / f"{C.iter_slug(key)}__grid.npz")
    g = {c: z[c] for c in z.files}
    del z
    return g


def _parts(it: C.Iteration, rows: dict) -> dict:
    out = {}
    for m in TRUTH_MODES:
        r = SC.score_rows(it, rows, m, configs=(SC.PRIMARY,))
        pr = r[SC.PRIMARY]
        out[m] = dict(r["parts"], cost_share_false=pr["cost_share_false"],
                      cost_share_missed=pr["cost_share_missed"], cost_share_loc=pr["cost_share_loc"])
    return out


def score_task(key: str) -> dict:
    """Tuning scores of every distinct track selection of testing iteration `key`."""
    t0 = time.time()
    it = ITERS[key]
    T_ = TUNE[key]
    g = load_grid(key)
    union = np.zeros(T_["nt"], bool)
    for s in T_["sels"]:
        union |= s
    tt = {"track": np.arange(T_["nt"]), "lab": T_["lab"]}
    rows = F.build_rows(it, g, tt, T_["conf"], union)
    row_track = g["track"][np.flatnonzero(union[g["track"]])]
    res = []
    for s in T_["sels"]:
        rm = s[row_track]
        sub = {c: np.asarray(v)[rm] for c, v in rows.items()}
        p = _parts(it, sub)
        res.append((p["all"]["combined"], p["all"]["track"], p["airborne"]["combined"],
                    p["airborne"]["track"], int(s.sum()), int(rm.sum())))
    return {"key": key, "res": res, "seconds": time.time() - t0}


def apply_task(key: str) -> dict:
    """Write F1_fuse for `key` (round trip checked) and score the diagnostics in memory."""
    t0 = time.time()
    it = ITERS[key]
    A = APPLY[key]
    g = load_grid(key)
    out = {"key": key, "diag": {}}
    nt = A["nt"]
    for name, sel, lab, conf, write in A["variants"]:
        tt = {"track": np.arange(nt), "lab": lab}
        rows = F.build_rows(it, g, tt, conf, sel)
        if write:
            rm = np.flatnonzero(sel[g["track"]])
            path = C.submission_path(METHOD, it)
            rt = B._file_roundtrip(it, rows, path, g["e"][rm], g["n"][rm], g["u"][rm])
            back = rt["back"]
            tb = np.asarray(back["track_id"]).astype(str)
            uniq = True
            if len(tb):
                code = np.unique(tb, return_inverse=True)[1]
                pair = back["step"].astype(np.int64) * (int(code.max()) + 1) + code
                uniq = len(np.unique(pair)) == len(pair)
            out["file"] = {"path": str(path.relative_to(C.REPO)), "rows": len(rows["step"]),
                           "tracks": int(sel.sum()), "problems": rt["problems"], "rt_err_m": rt["rt_err_m"],
                           "rt_labels_ok": rt["rt_labels_ok"], "n_back": rt["n_back"], "unique": uniq}
            if it.has_groundtruth:
                out["diag"][name] = _parts(it, back)
            del back
        elif it.has_groundtruth:
            out["diag"][name] = _parts(it, rows)
        out.setdefault("n", {})[name] = (int(sel.sum()), int(len(rows["step"])))
        del rows
    out["seconds"] = time.time() - t0
    return out


def f0_fold_choices() -> dict:
    """F0's (P_FIELD, TAU) per fold and for evaluation, recomputed from fused/tuning.csv."""
    recs = []
    with FUSED_TUNING.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for c in ("cont_gap_s", "bridge_s", "tau", "combined_all", "combined_airborne"):
                r[c] = float(r[c])
            recs.append(r)
    out = {}
    for k in TESTING:
        out[k] = F.choose(recs, k, [j for j in TESTING if j != k])[0]
    out["outer"] = F.choose(recs, "outer", TESTING)[0]
    return out


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--build-jobs", type=int, default=3)
    args = ap.parse_args()
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("fork")
    phase_s = {}

    # ---- phase A: fused tracks + features + targets
    sizes = {k: int(C.read_columns(TRACKS_DIR / f"{C.iter_slug(k)}__tracks.csv", numeric={"n_grid"},
                                   usecols=["n_grid"])["n_grid"].sum()) for k in ALL_KEYS}
    order = sorted(ALL_KEYS, key=lambda k: -sizes[k])
    log(f"phase A: fused tracks of geometry {GEOM} on {len(order)} iterations, {args.build_jobs} workers")
    built = {}
    with ctx.Pool(args.build_jobs) as pool:
        for n, r in enumerate(pool.imap_unordered(build_task, order), 1):
            built[r["key"]] = r
            extra = ""
            if r["key"] in TESTING:
                extra = f", y_all {int(r['y_all'].sum())}, y_air {int(r['y_air'].sum())}"
            log(f"  [{n}/{len(order)}] {r['key']}: {r['nt']} tracks, {r['n_rows']:,} rows{extra} "
                f"({r['seconds']:.0f}s)")
    phase_s["A"] = time.time() - t_start

    # ---- phase B: model
    t1 = time.time()
    assemble(built)
    log(f"tracks: {D['X'].shape[0]} ({sum(D['rows'][k].size for k in TESTING)} testing), "
        f"{D['X'].shape[1]} features")
    for t in TARGETS:
        log(f"  y_{t}: {int(D['y'][t].sum())} positive testing tracks")
    M = model_phase(args.jobs)
    res, final = M["results"], M["final"]
    # p per track and iteration, p_inner files
    p_track = {}
    for k in ALL_KEYS:
        rows = D["rows"][k]
        p_track[k] = {t: (res[t]["p_cal"][rows] if k in TESTING else final[t]["p_eval"][k]) for t in TARGETS}
    for j in TESTING:
        outs = [x for x in TESTING if x != j]
        nt = built[j]["nt"]
        cols = {"track": np.tile(np.arange(nt), len(outs)), "outer_fold": np.repeat(outs, nt).tolist()}
        for t in TARGETS:
            cols[f"p_{t}"] = np.concatenate([M["p_inner"][(t, o, j)] for o in outs])
        C.write_columns(OUT / f"{C.iter_slug(j)}__p_inner.csv", cols, ["track", "outer_fold", "p_all", "p_air"])
    phase_s["B"] = time.time() - t1

    # ---- phase C1: tuning scores (p_inner re-read from the written files)
    t2 = time.time()
    ledger = []                    # (held-out k or 'outer', j, set of outer_fold labels of the p rows used)
    sel_index = {}                 # (src, j, pf, tau) -> index into TUNE[j]['sels']
    pin_read = {}
    for j in TESTING:
        q = C.read_columns(OUT / f"{C.iter_slug(j)}__p_inner.csv", numeric={"track", "p_all", "p_air"})
        pin_read[j] = q
        b = built[j]
        pf_ = b["p_fixed"]
        lab = dict(b["lab"])
        lab["classification"] = np.where(pf_ > 0.5, FIXED, ROTARY).astype(object)
        sels, hidx = [], {}
        srcs = {"outer": (p_track[j]["all"], p_track[j]["air"], {"outer"})}
        for k in TESTING:
            if k == j:
                continue
            m = q["outer_fold"] == k
            tr_ = q["track"][m].astype(np.int64)
            o = np.argsort(tr_, kind="stable")
            if not np.array_equal(tr_[o], np.arange(b["nt"])):
                raise SystemExit(f"{j}: p_inner fold {k} does not cover every track")
            srcs[k] = (q["p_all"][m][o], q["p_air"][m][o], set(q["outer_fold"][m].tolist()))
        for src, (pa, pr, folds) in srcs.items():
            ledger.append((src, j, folds))
            for pf, tau in SEL_COMBOS:
                s = (pa if pf == "all" else pr) >= tau
                h = hashlib.sha1(np.packbits(s).tobytes()).hexdigest()
                if h not in hidx:
                    hidx[h] = len(sels)
                    sels.append(s)
                sel_index[(src, j, pf, tau)] = hidx[h]
        TUNE[j] = {"nt": b["nt"], "sels": sels, "lab": lab, "conf": np.maximum(pf_, 1 - pf_)}
    njobs = sorted(TESTING, key=lambda k: -sizes[k])
    log(f"phase C1: {sum(len(TUNE[j]['sels']) for j in TESTING)} distinct selections x 2 truth modes on "
        f"{len(TESTING)} testing iterations")
    scores = {}
    with ctx.Pool(min(args.jobs, len(njobs))) as pool:
        for r in pool.imap_unordered(score_task, njobs):
            scores[r["key"]] = r["res"]
            log(f"  {r['key']}: {len(r['res'])} selections ({r['seconds']:.0f}s)")
    records = []
    for (src, j, pf, tau), i in sorted(sel_index.items(), key=lambda kv: (kv[0][0], kv[0][1],
                                                                       P_FIELDS.index(kv[0][2]), kv[0][3])):
        ca, ta, cr, tr_, ntk, nrw = scores[j][i]
        records.append(dict(k=src, j=j, p_field=pf, tau=tau, combined_all=ca, track_all=ta,
                            combined_airborne=cr, track_airborne=tr_, n_tracks=ntk, n_rows=nrw,
                            p_fold="|".join(sorted(next(x[2] for x in ledger if x[0] == src and x[1] == j)))))
    tcols = ["k", "j", "p_field", "tau", "combined_all", "track_all", "combined_airborne", "track_airborne",
             "n_tracks", "n_rows", "p_fold"]
    C.write_columns(TUNING_CSV, {c: (np.array([r[c] for r in records], dtype=float)
                                     if isinstance(records[0][c], float) else [r[c] for r in records])
                                 for c in tcols}, tcols)
    rec_idx = {(r["k"], r["j"], r["p_field"], r["tau"]): r for r in records}

    def choose(src: str, js: list[str]) -> tuple[tuple, float, list]:
        best, bv, used = None, -math.inf, []
        for cb in SEL_COMBOS:
            rs = [rec_idx[(src, j) + cb] for j in js]
            v = float(np.mean([(x["combined_all"] + x["combined_airborne"]) / 2 for x in rs]))
            if v > bv:
                best, bv = cb, v
            used.extend((src, x["j"]) for x in rs)
        return best, bv, used

    choices, used_by = {}, {}
    for k in TESTING:
        cb, v, used = choose(k, [j for j in TESTING if j != k])
        choices[k], used_by[k] = (cb, v), used
    fcb, fv, f_used = choose("outer", TESTING)
    for k in TESTING:
        log(f"  fold {k}: P_FIELD {choices[k][0][0]} TAU {choices[k][0][1]} objective {choices[k][1]:.4f}")
    log(f"  evaluation: P_FIELD {fcb[0]} TAU {fcb[1]} objective {fv:.4f}")
    phase_s["C1"] = time.time() - t2

    # ---- phase C2: apply (files + diagnostics)
    t3 = time.time()
    f0c = f0_fold_choices()
    reported = {}
    for k in ALL_KEYS:
        b = built[k]
        pf, tau = choices[k][0] if k in TESTING else fcb
        s = p_track[k][pf] >= tau
        reported[k] = s
        pfx = b["p_fixed"]
        conf = np.maximum(pfx, 1 - pfx)
        lab1 = dict(b["lab"])
        lab1["classification"] = np.where(pfx > 0.5, FIXED, ROTARY).astype(object)
        var = [("F1", s, lab1, conf, True)]
        if k in TESTING:
            lab0 = dict(b["lab"])                          # F0's class rule (fuse.track_table)
            labr = dict(b["lab"])
            labr["classification"] = np.full(b["nt"], ROTARY, dtype=object)
            c0 = f0c[k]
            s0 = (b["pmax_all"] if c0[3] == "all" else b["pmax_air"]) >= c0[4]
            var += [("F1_f0class", s, lab0, conf, False), ("F1_rotary", s, labr, conf, False),
                    ("F0rule_geom", s0, lab0, conf, False), ("oracle_yall", b["y_all"].astype(bool), lab1, conf,
                                                             False)]
        APPLY[k] = {"nt": b["nt"], "variants": var}
    applied = {}
    ajobs = sorted(ALL_KEYS, key=lambda k: -sizes[k])
    log(f"phase C2: F1 files for {len(ajobs)} iterations + diagnostics")
    with ctx.Pool(min(args.jobs, len(ajobs))) as pool:
        for r in pool.imap_unordered(apply_task, ajobs):
            applied[r["key"]] = r
            f = r["file"]
            log(f"  {r['key']}: F1 {f['tracks']} tracks / {f['rows']:,} rows, problems {len(f['problems'])} "
                f"({r['seconds']:.0f}s)")
    phase_s["C2"] = time.time() - t3

    # ---- tracks.csv
    fnames = D["names"]
    for k in ALL_KEYS:
        b = built[k]
        rows = D["rows"][k]
        it = ITERS[k]
        cols = {"track": np.arange(b["nt"]),
                "track_id": np.array([F.TR.track_id(it, f"{k}/track{t}") for t in range(b["nt"])], dtype=object),
                "sensor_ids": b["sensor_ids"], "main_sensor": b["main_sensor"],
                "n_steps": b["feats"]["n_steps"].astype(np.int64), "first_step": b["first_step"],
                "last_step": b["last_step"], "serial": b["serial"], "model": b["model"], "protocol": b["protocol"],
                "identity": b["lab"]["identity"], "class_f0": b["lab"]["classification"],
                "class_f1": np.where(b["p_fixed"] > 0.5, FIXED, ROTARY).astype(object),
                "p_fixed": b["p_fixed"], "pmax_all": b["pmax_all"], "pmax_air": b["pmax_air"]}
        order = list(cols)
        for i, c in enumerate(fnames):
            cols[f"f_{c}"] = D["X"][rows, i]
            order.append(f"f_{c}")
        if k in TESTING:
            for t in TARGETS:
                cols[f"m_{t}"] = b[f"m_{t}"]
                cols[f"y_{t}"] = b[f"y_{t}"].astype(np.int64)
            cols["label_drone"] = b["label_drone"].astype(np.int64)
            cols["drone_share"] = b["drone_share"]
            order += ["m_all", "y_all", "m_air", "y_air", "label_drone", "drone_share"]
        cols["p_all"], cols["p_air"] = p_track[k]["all"], p_track[k]["air"]
        cols["fold"] = [k if k in TESTING else "final"] * b["nt"]
        cols["reported_f1"] = reported[k].astype(np.int64)
        order += ["p_all", "p_air", "fold", "reported_f1"]
        C.write_columns(OUT / f"{C.iter_slug(k)}__tracks.csv", cols, order)

    # ---- headline scores (files), logged
    headline = {}
    for mode in TRUTH_MODES:
        headline[mode] = SC.score_method(
            METHOD, mode, iters=[ITERS[k] for k in TESTING], log=True,
            note=("track_existence.py F1_fuse: fused tracks (fixed geometry hand/10/20) with track-level "
                  "existence p >= TAU (nested LOIO), learned fixed/rotary label; deployable sensors only"))

    # ---- self-checks
    fails = 0
    check_lines = []

    def check(num, name, ok, detail):
        nonlocal fails
        fails += 0 if ok else 1
        line = f"[{'PASS' if ok else 'FAIL'}] {num}. {name}: {detail}"
        check_lines.append(line)
        log(line)

    TK, EK = TESTING, EVALUATION
    # 1 leakage guard
    bad = []
    for gentry in M["guard"]:
        kind, held = gentry[0], gentry[1]
        if kind == "combo":
            bad.append(f"inner combo metric of {held} used a pair fit that saw it")
            continue
        fitted_on, cal_on, naive_on = gentry[2], gentry[3], gentry[4]
        if held in fitted_on or (held in TK and held in cal_on) or held in naive_on:
            bad.append(f"{kind} {held}")
        if any(x in EK for x in tuple(fitted_on) + tuple(cal_on) + tuple(naive_on)):
            bad.append(f"eval in training ({held})")
    for r in list(M["R"].values()) + list(M["RF"].values()):
        if any(x in EK for x in r["fitted_on"]):
            bad.append(f"eval rows in fit {r['task'][:2]}")
        if set(r["fitted_on"]) != set(r["task"][2]):
            bad.append(f"fit rows {r['fitted_on']} != requested {r['task'][2]}")
        if r["task"] in M["R"] and set(r["fitted_on"]) & set(r["task"][3]):
            bad.append(f"fit {r['task'][:2]} predicts an iteration it was fitted on")
    for k in TK:
        js = {j for _, j in used_by[k]}
        if k in js or js != set(TK) - {k}:
            bad.append(f"{k}: TAU objective used {sorted(js)}")
        for src, j in used_by[k]:
            folds = next(x[2] for x in ledger if x[0] == src and x[1] == j)
            if src != k or folds != {k}:
                bad.append(f"{k}: TAU choice scored {j} with p rows of fold {sorted(folds)}")
    if {j for _, j in f_used} != set(TK) or any(src != "outer" for src, _ in f_used):
        bad.append("evaluation TAU choice did not use the outer p of all 6")
    dep = C.deployable_sensor_ids()
    n_trk_sens = 0
    for k in ALL_KEYS:
        b = built[k]
        ss = set()
        for s in b["sensor_ids"].tolist():
            ss |= set(s.split("|"))
        n_trk_sens += len(ss)
        if not ss <= dep or b["n_mem_nondeploy"]:
            bad.append(f"{k}: non-deployable sensors {sorted(ss - dep)}")
        if k in TK and b["p_src_fold"] != k:
            bad.append(f"{k}: F0 comparison p fold {b['p_src_fold']}")
    check(1, "leakage guard", not bad,
          f"{len(M['guard'])} held-out prediction sets and {len(M['R']) + len(M['RF'])} fits: no held-out "
          f"iteration in its model / calibrator / combination choice / naive training set; the TAU choice of "
          f"each fold k scored only j != k with p_inner rows of outer_fold == k (re-read from file), the "
          f"evaluation choice the outer p of all 6; no evaluation row in any fit; no testing-only sensor in any "
          f"track (sensor ids used: {sorted(set().union(*[set('|'.join(built[k]['sensor_ids'].tolist()).split('|')) for k in ALL_KEYS]))})"
          + (f"; {bad[:5]}" if bad else ""))

    # 2 one p per track per target in [0, 1], files re-read
    bad2 = []
    reread = {}
    for k in ALL_KEYS:
        tf = C.read_columns(OUT / f"{C.iter_slug(k)}__tracks.csv",
                            numeric={"track", "p_all", "p_air", "n_steps", "reported_f1"})
        reread[k] = tf
        nt = built[k]["nt"]
        if not np.array_equal(tf["track"].astype(np.int64), np.arange(nt)):
            bad2.append(f"{k}: track column")
        for t in TARGETS:
            p = tf[f"p_{t}"]
            if p.size != nt or not (np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))):
                bad2.append(f"{k} {t}: p outside [0, 1] / NaN / count")
            elif not np.array_equal(p, p_track[k][t]):
                bad2.append(f"{k} {t}: file p != in-memory p")
        if not np.all(tf["fold"] == (k if k in TK else "final")):
            bad2.append(f"{k}: fold column")
        if not np.array_equal(tf["reported_f1"].astype(bool), reported[k]):
            bad2.append(f"{k}: reported_f1")
    for j in TK:
        q = pin_read[j]
        nt = built[j]["nt"]
        for k in TK:
            if k == j:
                continue
            m = q["outer_fold"] == k
            if m.sum() != nt or np.unique(q["track"][m]).size != nt:
                bad2.append(f"{j}: p_inner fold {k} count")
            for t in TARGETS:
                p = q[f"p_{t}"][m]
                if not (np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))):
                    bad2.append(f"{j}/{k} {t}: p_inner outside [0, 1]")
        if j in set(q["outer_fold"].tolist()):
            bad2.append(f"{j}: p_inner has its own key as outer fold")
    check(2, "exactly one p in [0, 1] per track per target (files re-read)", not bad2,
          f"{sum(built[k]['nt'] for k in ALL_KEYS)} tracks over 9 iterations x 2 targets in tracks.csv, = the "
          f"in-memory values; p_inner: 5 folds x every track per testing iteration" + (f"; {bad2[:5]}" if bad2 else ""))

    # 3 final model reproducible from JSON
    bad3 = []
    for t in TARGETS:
        doc = json.loads((OUT / f"model_{t}.json").read_text())
        m = GB.GBDT.from_dict(doc["gbdt"])
        iso = GB.Isotonic.from_dict(doc["isotonic"])
        if doc["features"] != D["names"]:
            bad3.append(f"{t}: feature list")
        for k in EK:
            p = iso.predict(m.predict_proba(D["X"][D["rows"][k]]))
            if not np.array_equal(p, reread[k][f"p_{t}"]):
                bad3.append(f"{t} {k}: JSON != written (max |d| {np.max(np.abs(p - reread[k][f'p_{t}'])):.2e})")
    check(3, "final model JSON + calibrator reproduce the evaluation predictions bit for bit", not bad3,
          f"model_all.json / model_air.json on {len(EK)} evaluation iterations, compared with the re-read "
          f"tracks.csv" + (f"; {bad3}" if bad3 else ""))

    # 4 model beats naive (mean held-out step-weighted logloss)
    det4, ok4, ll = [], True, {}
    s_ = D["n_steps"]
    for t in TARGETS:
        mm, nn = [], []
        for k in TK:
            rr = D["rows"][k]
            y = D["y"][t][rr]
            mm.append(GB.logloss(y, res[t]["p_cal"][rr], s_[rr]))
            nn.append(GB.logloss(y, res[t]["p_naive"][rr], s_[rr]))
        ll[t] = (mm, nn)
        ok4 &= float(np.mean(mm)) < float(np.mean(nn))
        det4.append(f"y_{t}: model {np.mean(mm):.4f} vs naive {np.mean(nn):.4f}")
    check(4, "model beats the naive per-track baseline (mean held-out step-weighted logloss; every track is "
             "deployable)", ok4, "; ".join(det4))

    # 5 targets recomputed by a plain-python loop
    rng = np.random.default_rng(SEED)
    bad5, n5, nr5 = [], 0, 0
    for k in TK:
        b = built[k]
        g = load_grid(k)
        pick = np.sort(rng.choice(b["nt"], size=min(N_CHECK_PER_ITER, b["nt"]), replace=False))
        st = np.searchsorted(g["track"], pick)
        en = np.searchsorted(g["track"], pick, side="right")
        for tgt, mode in TARGETS.items():
            tr = T.load_truth(ITERS[k], mode)
            bystep: dict[int, list] = {}
            for s, e, n, u in zip(tr.step.tolist(), tr.e.tolist(), tr.n.tolist(), tr.u.tolist()):
                bystep.setdefault(s, []).append((e, n, u))
            for t, a, z in zip(pick.tolist(), st.tolist(), en.tolist()):
                near = 0
                for s, e, n, u in zip(g["step"][a:z].tolist(), g["e"][a:z].tolist(), g["n"][a:z].tolist(),
                                      g["u"][a:z].tolist()):
                    best = math.inf
                    for te, tn, tu in bystep.get(s, ()):
                        de, dn, du = e - te, n - tn, u - tu
                        best = min(best, math.sqrt(de * de + dn * dn + du * du))
                    near += best <= MATCH_M
                m_py = near / (z - a)
                if m_py != b[f"m_{tgt}"][t] or (m_py >= Y_SHARE) != bool(b[f"y_{tgt}"][t]):
                    bad5.append(f"{k} track {t} {tgt}: python {m_py} vs {b[f'm_{tgt}'][t]}")
                if tgt == "all":
                    n5 += 1
                    nr5 += z - a
        del g
    check(5, "targets m_all / m_air recomputed by an independent plain-python loop", n5 >= 300 and not bad5,
          f"{n5} random tracks ({nr5:,} rows) over {len(TK)} testing iterations, both modes equal"
          + (f"; {bad5[:5]}" if bad5 else ""))

    # 6 files validate, round trip, unique
    files = [(k, a["file"]) for k, a in applied.items()]
    bad6 = [(k, f["problems"][:2], f["rt_err_m"], f["unique"], f["rt_labels_ok"], f["n_back"], f["rows"])
            for k, f in files if f["problems"] or not f["unique"] or not (f["rt_err_m"] < RT_TOL_M)
            or f["rt_labels_ok"] is not True or f["n_back"] != f["rows"]]
    diffs = []
    for k in TK:
        dg = applied[k]["diag"]["F1"]
        for mode in TRUTH_MODES:
            hr = next(x for x in headline[mode] if x["iter_key"] == k)
            diffs.append(abs(hr["parts"]["combined"] - dg[mode]["combined"]))
        r = rec_idx[("outer", k) + choices[k][0]]
        diffs.append(abs(r["combined_all"] - dg["all"]["combined"]))
        diffs.append(abs(r["combined_airborne"] - dg["airborne"]["combined"]))
    check(6, "every written file validates; round trip; (step, track_id) unique", not bad6 and max(diffs) < 1e-4,
          f"{len(files)} F1_fuse files: validate ([]), round trip e/n/u max "
          f"{max(f['rt_err_m'] for _, f in files):.1e} m < {RT_TOL_M}, labels / ids identical, (step, track_id) "
          f"unique; tuning-table score of the applied choice = file score = headline (max diff {max(diffs):.1e})"
          + (f"; {bad6[:3]}" if bad6 else ""))

    # 7 determinism of one fold refit
    t7 = "all"
    k7 = TK[0]
    c7 = M["sel"][(t7, k7)]
    task7 = (t7, c7, tuple(x for x in TK if x != k7), (k7,))
    r7 = fit_task(task7)
    ok7 = np.array_equal(r7["pred"][k7], M["R"][task7]["pred"][k7])
    check(7, "determinism of one fold refit", ok7,
          f"outer model of fold {k7} (y_{t7}, {combo_name(c7)}) refitted in the main process: held-out raw "
          f"predictions identical bit for bit to the pool fit")

    runtime = time.time() - t_start
    ok_rt = runtime < RUNTIME_LIMIT_S
    log(f"runtime {runtime:.0f} s ({'<' if ok_rt else '>='} {RUNTIME_LIMIT_S} s)")
    write_report(built, M, p_track, records, choices, (fcb, fv), applied, headline, f0c, ll, check_lines,
                 fails, runtime, phase_s)
    log(f"wrote {REPORT.relative_to(C.REPO)}")
    if fails:
        log(f"SELF-CHECKS FAILED: {fails}")
        return 1
    log("ALL SELF-CHECKS PASSED")
    return 0


# --------------------------------------------------------------------------- report


def calib_table(y, p, w) -> list[list]:
    out = []
    for i in range(len(CAL_BINS) - 1):
        lo, hi = CAL_BINS[i], CAL_BINS[i + 1]
        m = (p >= lo) & ((p < hi) if i < len(CAL_BINS) - 2 else (p <= hi))
        if not m.any():
            out.append([f"{lo:.2f}-{hi:.2f}", 0, 0, "", "", "", ""])
            continue
        out.append([f"{lo:.2f}-{hi:.2f}", int(m.sum()), int(y[m].sum()), fmt(p[m].mean()), fmt(y[m].mean()),
                    fmt(np.sum(w[m] * p[m]) / w[m].sum()), fmt(np.sum(w[m] * y[m]) / w[m].sum())])
    return out


def write_report(built, M, p_track, records, choices, final_sel, applied, headline, f0c, ll, check_lines,
                 fails, runtime, phase_s) -> None:
    TK, EK = TESTING, EVALUATION
    res, final = M["results"], M["final"]
    f0 = F.read_experiments(["F0_fuse"])
    L = ["# Track existence on fused tracks and the method F1_fuse", ""]
    L.append(f"**Self-checks: {'ALL PASSED' if not fails else f'{fails} FAILED'}** (run log of "
             f"`python3 tools/track_existence.py`; runtime {runtime / 60:.1f} min: tracks {phase_s['A']:.0f} s, "
             f"model {phase_s['B']:.0f} s, TAU tuning {phase_s['C1']:.0f} s, files {phase_s['C2']:.0f} s).")
    L.append("")
    L.append("Generated by `tools/track_existence.py` (rules in its docstring). All scores are the local PROXY of "
             "`tools/score.py` (c = 50 m, p = 2, 3-D) on the 6 testing iterations, held out (leave-one-iteration-"
             "out). Every track is built from deployable sensors only, so every number here is 'deployable'. "
             "VKYY / QTVS relay the Dronetag feed in testing (optimistic there).")
    L.append("")
    # summary
    m1 = {mode: float(np.mean([applied[k]["diag"]["F1"][mode]["combined"] for k in TK])) for mode in TRUTH_MODES}
    m0 = {}
    for mode in TRUTH_MODES:
        r = f0.get(("F0_fuse", mode), {})
        m0[mode] = float(np.mean([float(r[k]["combined"]) for k in TK])) if all(k in r for k in TK) else float("nan")
    L.append("## Summary")
    L.append("")
    L.append(f"- **F1_fuse held-out mean combined: {m1['all']:.4f} (truth 'all') / {m1['airborne']:.4f} "
             f"('airborne')** vs F0_fuse {fmt(m0['all'], 4)} / {fmt(m0['airborne'], 4)} "
             f"(latest F0 rows of experiments.csv).")
    for t in TARGETS:
        mm, nn = ll[t]
        L.append(f"- Track model y_{t}: held-out step-weighted logloss {np.mean(mm):.4f} vs naive {np.mean(nn):.4f}; "
                 f"{int(D['y'][t].sum())} positive of {sum(D['rows'][k].size for k in TK)} testing tracks.")
    pool_rows = np.concatenate([D["rows"][k] for k in TK])
    s_ = D["n_steps"]
    for t in TARGETS:
        y = D["y"][t][pool_rows]
        pc = res[t]["p_cal"][pool_rows]
        pm = np.concatenate([built[k][f"pmax_{t}"] for k in TK])
        L.append(f"- Ranking, y_{t} (pooled held-out): AUC track p {GB.auc(y, pc):.3f} vs F0's max-member p "
                 f"{GB.auc(y, pm):.3f} (track-weighted); {GB.auc(y, pc, s_[pool_rows]):.3f} vs "
                 f"{GB.auc(y, pm, s_[pool_rows]):.3f} (step-weighted).")
    dg = {v: {mode: float(np.mean([applied[k]["diag"][v][mode]["combined"] for k in TK])) for mode in TRUTH_MODES}
          for v in ("F0rule_geom", "F1_f0class", "F1", "oracle_yall")}
    verdict = "beats" if m1["all"] > m0["all"] and m1["airborne"] > m0["airborne"] else "does NOT beat"
    L.append(f"- **Verdict: F1 {verdict} F0 on both truth modes.** Same tracks, same labels, only the selection "
             f"changed (F0's rule on this geometry -> track model): {dg['F0rule_geom']['all']:.4f} -> "
             f"{dg['F1_f0class']['all']:.4f} ('all'), {dg['F0rule_geom']['airborne']:.4f} -> "
             f"{dg['F1_f0class']['airborne']:.4f} ('airborne'). Even the ORACLE that reports exactly the y_all tracks "
             f"reaches only {dg['oracle_yall']['all']:.4f} / {dg['oracle_yall']['airborne']:.4f}: on this track set "
             f"perfect track existence is worth about "
             f"{dg['oracle_yall']['all'] - dg['F0rule_geom']['all']:+.3f} / "
             f"{dg['oracle_yall']['airborne'] - dg['F0rule_geom']['airborne']:+.3f} over F0's rule, and the track "
             f"model, trained on only {int(D['y']['all'].sum())} / {int(D['y']['air'].sum())} positive tracks, "
             + ("captures none of it" if dg["F1_f0class"]["all"] <= dg["F0rule_geom"]["all"]
                and dg["F1_f0class"]["airborne"] <= dg["F0rule_geom"]["airborne"] else "captures part of it")
             + " (see the ranking line above, the per-fold table and the calibration table).")
    L.append(f"- Fixed geometry {GEOM} (edge_gate, cont_gap_s, bridge_s) = fuse.py's all-6 in-sample argmax; "
             "fuse.md shows its objective varies by < 0.002 across the geometry grid, so this selection on all 6 "
             "iterations is a negligible optimism (it is not re-chosen per fold).")
    L.append("")
    L.append("## Terms in one line each")
    L.append("")
    for line in [
        "**segment**: one sensor's own track of one object; **fused track**: the segments judged to be one object (tools/fuse.py), one track_id.",
        "**fused row**: one fused track at one 100 ms grid step (bridged rows = short gaps filled by interpolation).",
        "**GOSPA**: the track metric; per step truth drones are matched to rows within c = 50 m; a missed drone or a false row costs c^2/2 each.",
        "**m_all / m_air**: share of a track's fused rows within 50 m (3-D) of the nearest truth drone at that step (truth 'all' incl. parked drones / 'airborne' only).",
        "**y_all / y_air**: the target: m >= 0.5, a track the GOSPA could mostly match.",
        "**label_drone**: fuse.py's label: >= half of the track's member grid samples sit on drone-labelled segments (says nothing about the fused rows' accuracy).",
        "**existence p (track)**: this tool's calibrated probability that a fused track is a y = 1 track.",
        "**F0's max-member p**: the largest segment existence p among a track's members (what F0 thresholds).",
        "**GBDT**: gradient-boosted decision trees (tools/gbdt.py), the classifier; **isotonic calibration**: a non-decreasing map from raw score to observed rate.",
        "**LOIO**: leave-one-iteration-out: everything applied to testing iteration k is fitted and chosen without k.",
        "**nested / p_inner**: predictions for iteration j from a model that saw neither j nor the held-out k, on k's calibration scale; used to tune TAU for k.",
        "**TAU**: the p threshold above which F1 reports a track; **P_FIELD**: which target's p is thresholded (all / air).",
        "**AUC**: chance a random positive track ranks above a random negative one (0.5 coin flip, 1 perfect); step-weighted = each track weighted by its rows.",
        "**logloss**: mean of -log(p given to what happened); lower is better. **naive baseline**: training share of positives among tracks with the same number of sensors (1, 2, 3+) and main sensor.",
        "**combined**: 0.4 track + 0.2 classification + 0.1 identity + 0.3 metadata (label parts are recalls over all eligible truth rows).",
        "**oracle**: uses ground truth, never deployable; here it reports exactly the y_all tracks (the ceiling of this track set and target).",
    ]:
        L.append(f"- {line}")
    L.append("")
    # targets vs label_drone
    L.append("## Target y_all vs fuse.py's label_drone (testing, pooled)")
    L.append("")
    ya = np.concatenate([built[k]["y_all"] for k in TK])
    ld = np.concatenate([built[k]["label_drone"] for k in TK])
    ns = np.concatenate([built[k]["feats"]["n_steps"] for k in TK])
    rows_ = []
    for a in (1, 0):
        for b in (1, 0):
            m = (ya == a) & (ld == b)
            rows_.append([f"y_all={a}", f"label_drone={b}", int(m.sum()), f"{int(ns[m].sum()):,}"])
    L += md_table(["y_all", "label_drone", "tracks", "rows"], rows_)
    L.append("")
    per = []
    for k in TK:
        b = built[k]
        per.append([short(k), b["nt"], int(b["y_all"].sum()), int(b["y_air"].sum()), int(b["label_drone"].sum()),
                    int((b["y_all"] & b["label_drone"]).sum()), int((b["y_all"] & ~b["label_drone"]).sum()),
                    int((~b["y_all"] & b["label_drone"]).sum())])
    L += md_table(["iteration", "tracks", "y_all", "y_air", "label_drone", "both", "y_all only", "label_drone only"],
                  per)
    L.append("")
    L.append("'label_drone only' tracks are drone-labelled by membership but their fused rows are mostly > 50 m from "
             "any drone (or the drone is not in the truth at those steps); 'y_all only' tracks are mostly-clutter "
             "member sets whose fused rows nevertheless sit on a drone.")
    L.append("")
    # model selection
    L.append("## Model selection (nested LOIO)")
    L.append("")
    L.append("Combinations: " + "; ".join(f"`{c}` = {CONFIGS[c]}" for c in CONFIGS) +
             " x weights `unit` / `steps` (n_steps normalised to mean 1). Metric: step-weighted logloss of the raw "
             "GBDT output.")
    L.append("")
    for t in TARGETS:
        rows_ = []
        for k in TK:
            inner = M["inner_tab"][(t, k)]
            rows_.append([short(k)] + [f"{inner[c]:.4f}" for c in COMBOS] + [combo_name(M["sel"][(t, k)])])
        fc, loio = M["final_choice"][t]
        rows_.append(["**6-fold (final)**"] + [f"{loio[c]:.4f}" for c in COMBOS] + [combo_name(fc)])
        L.append(f"### y_{t}")
        L.append("")
        L += md_table(["held-out"] + [combo_name(c) for c in COMBOS] + ["chosen"], rows_)
        L.append("")
    # held-out metrics
    L.append("## Held-out quality of the track p vs F0's max-member p")
    L.append("")
    rows_ = []
    for t in TARGETS:
        for k in TK:
            rr = D["rows"][k]
            y = D["y"][t][rr]
            pm = built[k][f"pmax_{t}"]
            ok = 0 < y.sum() < y.size
            rows_.append([f"y_{t}", short(k), int(y.sum()), fmt(GB.auc(y, res[t]["p_cal"][rr]) if ok else np.nan),
                          fmt(GB.auc(y, pm) if ok else np.nan),
                          fmt(GB.auc(y, res[t]["p_cal"][rr], s_[rr]) if ok else np.nan),
                          fmt(GB.auc(y, pm, s_[rr]) if ok else np.nan),
                          fmt(GB.logloss(y, res[t]["p_cal"][rr], s_[rr]), 4),
                          fmt(GB.logloss(y, res[t]["p_naive"][rr], s_[rr]), 4)])
        y = D["y"][t][pool_rows]
        pm = np.concatenate([built[k][f"pmax_{t}"] for k in TK])
        pc = res[t]["p_cal"][pool_rows]
        rows_.append([f"**y_{t}**", "pooled", int(y.sum()), fmt(GB.auc(y, pc)), fmt(GB.auc(y, pm)),
                      fmt(GB.auc(y, pc, s_[pool_rows])), fmt(GB.auc(y, pm, s_[pool_rows])),
                      fmt(GB.logloss(y, pc, s_[pool_rows]), 4),
                      fmt(GB.logloss(y, res[t]["p_naive"][pool_rows], s_[pool_rows]), 4)])
    L += md_table(["target", "iteration", "positives", "AUC track p", "AUC F0 max p", "AUC track p (step)",
                   "AUC F0 max p (step)", "logloss track p (step)", "logloss naive (step)"], rows_)
    L.append("")
    L.append("F0's max-member p is the existence.py outer p of the segments (target y_all compared with p_drone_all, "
             "y_air with p_drone_air); it was trained for a segment target, so its AUC here measures how well it "
             "ranks fused tracks by THIS target.")
    L.append("")
    # calibration
    L.append("## Calibration (pooled held-out testing tracks)")
    L.append("")
    for t in TARGETS:
        y = D["y"][t][pool_rows]
        L.append(f"### y_{t}")
        L.append("")
        L += md_table(["p bin", "tracks", "positives", "mean p", "observed", "mean p (step)", "observed (step)"],
                      calib_table(y, res[t]["p_cal"][pool_rows], s_[pool_rows]))
        L.append("")
    # importance
    L.append("## Feature importance (final models, share of total split gain, top 20)")
    L.append("")
    for t in TARGETS:
        imp = final[t]["importance"]
        tot = imp.sum() if imp.sum() > 0 else 1.0
        o = np.argsort(-imp, kind="stable")[:20]
        L.append(f"- y_{t} ({combo_name(final[t]['combo'])}): " +
                 ", ".join(f"`{D['names'][i]}` {imp[i] / tot:.3f}" for i in o if imp[i] > 0))
    L.append("")
    L.append(f"{len(D['names'])} features: " + ", ".join(f"`{n}`" for n in D["names"]))
    L.append("")
    # selection
    L.append("## TAU / P_FIELD per fold")
    L.append("")
    rows_ = []
    for k in TK:
        cb, v = choices[k]
        c0 = f0c[k]
        rows_.append([short(k), cb[0], cb[1], f"{v:.4f}", f"{c0[3]} / {c0[4]}"])
    rows_.append(["evaluation (all 6)", final_sel[0][0], final_sel[0][1], f"{final_sel[1]:.4f}",
                  f"{f0c['outer'][3]} / {f0c['outer'][4]}"])
    L += md_table(["held-out", "P_FIELD", "TAU", "objective (inner mean of (all+air)/2)", "F0's P_FIELD / TAU"], rows_)
    L.append("")
    L.append("Objective of each candidate on every inner iteration is in tuning.csv (k = the fold whose p_inner was "
             "used; 'outer' = the held-out p, used for the evaluation choice only).")
    L.append("")
    # scores
    variants = [("F0 (file)", None), ("F0 rule, this geometry", "F0rule_geom"), ("F1, F0 class rule", "F1_f0class"),
                ("F1", "F1"), ("F1, all Rotary", "F1_rotary"), ("ORACLE y_all", "oracle_yall")]
    for mode in TRUTH_MODES:
        L.append(f"## Held-out scores, truth mode '{mode}'")
        L.append("")
        rows_ = []
        f0r = f0.get(("F0_fuse", mode), {})
        means = {v: [] for _, v in variants}
        for k in TK:
            dg = applied[k]["diag"]
            r = [short(k)]
            for lbl, v in variants:
                c = float(f0r[k]["combined"]) if v is None and k in f0r else (dg[v][mode]["combined"] if v else np.nan)
                means[v].append(c)
                r.append(fmt(c))
            d1 = dg["F1"][mode]
            r += [fmt(d1["track"]), fmt(d1["classification"]), fmt(d1["identity"]), fmt(d1["metadata"]),
                  fmt(d1["cost_share_false"], 2), fmt(d1["cost_share_missed"], 2),
                  applied[k]["n"]["F1"][0], f"{applied[k]['n']['F1'][1]:,}"]
            rows_.append(r)
        rows_.append(["**MEAN**"] + [fmt(float(np.mean(means[v]))) for _, v in variants] + [""] * 8)
        L += md_table(["iteration"] + [lbl for lbl, _ in variants] +
                      ["F1 track", "F1 class.", "F1 ident.", "F1 metadata", "F1 false share", "F1 missed share",
                       "F1 tracks", "F1 rows"], rows_)
        L.append("")
        a = {v: float(np.mean(means[v])) for _, v in variants}
        L.append(f"Gain split (mean combined): F1 - F0 = {a['F1'] - a[None]:+.4f} = geometry change "
                 f"(F0 rule on this geometry - F0) {a['F0rule_geom'] - a[None]:+.4f} + track existence model "
                 f"(F1 with F0's class rule - F0 rule on this geometry) {a['F1_f0class'] - a['F0rule_geom']:+.4f} + "
                 f"learned classification (F1 - F1 with F0's class rule) {a['F1'] - a['F1_f0class']:+.4f}. "
                 f"All-Rotary instead of the learned label: {a['F1_rotary'] - a['F1']:+.4f}. Oracle headroom of "
                 f"this track set: {a['oracle_yall'] - a['F1']:+.4f}.")
        L.append("")
    L.append("The 'F0 (file)' column is the latest F0_fuse block of experiments.csv (per-fold geometry); "
             "'F0 rule, this geometry' applies F0's own fold choice of (P_FIELD, TAU) (recomputed from "
             "fused/tuning.csv) to the max-member p of this tool's fixed-geometry tracks, with F0's labels.")
    L.append("")
    # evaluation
    L.append("## What F1 reports in evaluation")
    L.append("")
    rows_ = []
    for k in EK:
        b = built[k]
        s = p_track[k][final_sel[0][0]] >= final_sel[0][1]
        n_rows = int(b["feats"]["n_steps"][s].sum())
        fr = int(np.sum(b["lab"]["identity"][s] == "Friendly"))
        fx = int(np.sum(b["p_fixed"][s] > 0.5))
        f0n = f0rows = ""
        p0 = FUSED_DIR / f"{C.iter_slug(k)}__tracks.csv"
        if p0.exists():
            t0 = C.read_columns(p0, numeric={"reported_f0", "n_steps"}, usecols=["reported_f0", "n_steps"])
            f0n = int(t0["reported_f0"].sum())
            f0rows = f"{int(t0['n_steps'][t0['reported_f0'] > 0].sum()):,}"
        rows_.append([short(k), b["nt"], int(s.sum()), f"{n_rows:,}", fr, fx, f0n, f0rows])
    L += md_table(["iteration", "fused tracks", "F1 tracks", "F1 rows", "Friendly tracks", "Fixed-wing tracks",
                   "F0 tracks", "F0 rows"], rows_)
    L.append("")
    L.append("## Self-checks")
    L.append("")
    for line in check_lines:
        L.append(f"- {line}")
    L.append("")
    L.append("## Caveats")
    L.append("")
    L.append("- Second-order leaks while tuning TAU for fold k (documented, not fixed): j's classify p_fixed and "
             "bearing-support features come from LOIO models / choices that saw k's labels; sensor noise tables, "
             "sensor_tracks variances and fuse's honest variance table of j were fitted with k. The existence p is not "
             "a feature, and the track model / calibrator / TAU nesting itself is exact (self-check 1).")
    L.append("- The geometry is fixed to fuse's all-6 argmax (in-sample, negligible: < 0.002 objective spread).")
    L.append("- 'ident.' in testing measures RED coverage only (no testing drone carries a whitelist serial).")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
