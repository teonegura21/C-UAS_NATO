#!/usr/bin/env python3
"""Multi-sensor fusion tracker (cross-sensor de-duplication) and the first deployable method
F0_fuse, tuned leave-one-iteration-out (LOIO) on the 6 testing iterations.

Every sensor reports its own tracks; one drone seen by three sensors is three segments, and
GOSPA charges every extra copy as a false track. This tool merges the segments of one object
into one fused track (one track_id), averages their positions with honest weights, bridges
short gaps, and reports the tracks whose existence probability (tools/existence.py) is high.

Inputs
------
* data/derived/sensor_tracks_v2/<slug>__tracks.csv + __grid.npz: per-sensor track segments on
  the template's 100 ms grid (tools/sensor_tracks.py), read through pair_features.Iter.
* data/derived/existence/<slug>__p.csv (testing: LOIO outer prediction of that iteration;
  evaluation: final model) and <slug>__p_inner.csv (testing: nested predictions on fold k's
  scale from a model that saw neither k nor the iteration itself).
* pair_features.sigma_table(exclude): the honest per-sensor variance table (testing iteration i
  uses the table without i; evaluation uses all 6). Honest per-sample variance V = max(grid
  var, s2) per horizontal axis (pair_features.Iter.Ve / .Vn).
* Only DEPLOYABLE segments (C.deployable_sensor_ids(): sensor ids present in evaluation) with
  grid samples enter; EUIG, LJJX, MMJJ, NJQL are dropped before clustering. VKYY / QTVS are
  deployable UTM relays; in testing they relay the Dronetag feed, so testing numbers that use
  them are optimistic.
* Inherited in-sample caveat (not fixed here): sensor_noise.csv, the sensor_tracks variances
  and height offsets were fitted on all 6 testing iterations.

Algorithm (per iteration; constants below, tunables in PARAMS)
---------------------------------------------------------------
1. Overlap pairs (pair_features.overlap_stats): every pair of deployable segments of different
   sensor_key with >= MIN_OVERLAP_STEPS common grid steps is "simultaneous". Candidates: also
   min horizontal distance <= MAX_MIN_DIST_M; their ov_nd_med (median normalised squared
   distance) and ov_share3s (share of common steps within 3 sigma) come from
   pair_features.overlap_features. An edge PASSES iff ov_nd_med < nd_gate and ov_share3s >=
   share_gate, where (nd_gate, share_gate) = EDGE_GATES[edge_gate] is a tuned pair: 'hand' is
   pair_features' hand rule (9.21, 0.5), 'loose' (25, 0.25). The two gates are loosened together
   because share >= 0.5 of steps with nd <= 9 already forces the median nd to about 9 or less,
   so raising nd_gate alone admits almost no extra edge. A simultaneous pair that is not a
   candidate or fails is a CONFLICT.
2. Clustering: union-find over passing edges in ascending ov_nd_med (ties by pair key). Two
   clusters merge only if (a) no two members share a sensor_key with grid spans overlapping by
   more than CONFLICT_S (one sensor tracking two things at once) and (b) complete linkage: no
   member pair across them is a conflict. Pairs with fewer common steps impose nothing.
3. Continuation (cluster level, any sensors, only if CONT_GAP_S > 0): cluster X's last fused
   state is extrapolated at constant velocity to cluster Y's first fused state when
   0 < gap <= CONT_GAP_S (gap = grid time between them); the link needs normalised squared
   distance (variance = fused var X + fused var Y + q |dt|^3 / 3, q = sensor_tracks.Q_ACC_H)
   < CONT_ND and distance < CONT_MAX_M; greedy one-to-one in ascending normalised distance
   (ties by cluster index). Clusters of a chain are time-disjoint, so the sensor rule of 2(a)
   holds automatically along a chain (self-check 1 verifies it). A chain is one TRACK.
4. Fused state per (track, step): per axis inverse-variance mean of the member samples at that
   step (weights 1 / V, V honest), var = 1 / sum(1/V), cov_en = 0, written as
   ref + sum(w (x - ref)) / sum(w) with ref = the first sample (a single member reproduces its
   sample exactly). Velocity: the same weighted mean of ve / vn. One-pass outlier rejection:
   with >= 3 samples, the single sample with the largest normalised squared horizontal
   distance to the leave-one-out mean of the others (variance V_i + 1/sum_others(1/V)) is
   dropped if that distance > OUT_ND, then the mean is recomputed. Only the worst one: one
   outlier among 3 samples pulls the leave-one-out means of the two good ones too, so dropping
   EVERY sample above OUT_ND would leave fewer than 2 samples at most such steps. A second
   outlier left in the mean (a step that, after the drop, still has >= 3 samples and one above
   OUT_ND) is counted per iteration (n_resid_out in fuse.md; 0 in most iterations, about 0.4 %
   of the >= 3-sample steps in the worst). Height: the same over the
   kept samples with finite u inside the validator range (submission.ALT_RANGE), weights
   1 / grid var_u. A step without height gets the track median fused height, else the median
   in-range u of the track's main sensor (most grid samples in the track) over every
   deployable segment of the iteration, else the iteration median (deployable segments) --
   never ground truth (the baselines.py rule at track level).
5. Gap bridging: between two covered steps of a track with gap <= BRIDGE_S (grid time between
   them), the steps in between get linear interpolation of e, n, u, ve, vn; var = max(end
   vars) + q (gap/2)^3 / 3 (q = Q_ACC_H for e / n, Q_ACC_U for u). No extrapolation beyond a
   track's first / last covered step (sensor_tracks already adds EXTRAP_S = 1 s).
6. Track labels (GT-free): classification 'Fixed Wing UAS' iff cls_fixed_share > 0.5 else
   'Rotary Wing UAS'; identity 'Friendly' iff the resolved serial is in
   baselines.load_whitelist() else 'Hostile'; model / protocol / serial from the members'
   metadata (majority, below); band 'Unknown'; confidence = p_track.

Methods
-------
F0_fuse (deployable): report a track iff p_track >= TAU, p_track = max over members of the
  segment p_drone_<P_FIELD>. Testing iteration k uses fold k's parameters (chosen without k)
  and k's outer p (p.csv); evaluation uses the parameters chosen on all 6 testing iterations
  and the evaluation p.csv (final model).
O3_fuse_oracle (testing only, ORACLE: uses ground-truth labels): the same fused tracks (fold k's
  geometry) reported iff label_drone. An oracle REFERENCE for this fused geometry, not a strict
  upper bound: it drops every mixed track whose drone share is < 0.5 (taking its drone rows with
  it), so F0 can score higher on some iterations. Compare with O1 / O2.

LOIO tuning (the score table is per (k, j, combo)): every combination of PARAMS is scored on
every testing iteration j in both truth modes with score.score_rows(configs=(PRIMARY,)), with
j's p from p_inner (outer_fold = k) for held-out fold k, and from p.csv ('outer') for the final
choice. Objective of fold k: mean over j != k of (combined_all + combined_airborne) / 2; its
argmax (first in PARAMS order on ties) is applied to k with k's outer p. Evaluation: argmax of
the same objective over all 6 iterations with the outer p. Nothing is fitted on evaluation.

Outputs (data/derived/fused/)
----------------------------
<slug>__tracks.csv   one row per track of the iteration's applied geometry (before the TAU
    selection): track, track_id, n_members, n_clusters, n_sensors, sensor_ids ('|'-joined
    sorted sensor ids), first_step, last_step, n_steps (rows incl. bridged), n_bridged,
    duration_s, p_max_all, p_max_air, p_wmean_all, p_wmean_air (outer p; n_grid-weighted
    means), serial (resolved dropdown serial or ''), model, protocol (dropdown or 'Unknown'),
    cls_fixed_share, speed_med_ms, speed_p10_ms, speed_p90_ms (fused horizontal speed over the
    unbridged steps), u_med_m (median fused height), reported_f0 (0/1),
    and on testing (diagnostics, never used by F0): drone_share (n_grid-weighted share of
    drone-labelled member samples), label_drone (drone_share >= 0.5), flight (n_grid-weighted
    majority best_flight among drone members; '' if not label_drone), purity (share of the
    track's rows within PURITY_M 3-D of that flight's truth position at that step, truth 'all').
<slug>__members.csv  track, cluster, seg_id (sorted).
<slug>__grid.npz     the fused rows of every track before the TAU selection, sorted by (track,
    step): track, step, e, n, u, var_e, var_n, var_u (always finite: rows whose height comes
    from a fallback, u_src 1-3, carry DEFAULT_VAR_U = track_rows' 900 m^2, as the submission
    does; bridged rows the bridging rule), ve, vn, n_used (member samples used;
    0 = bridged), n_rej (samples rejected as outliers), u_src (0 fused, 1 track median,
    2 sensor median, 3 iteration median, 4 bridged interpolation), bridged (bool).
tuning.csv           k ('outer' or the held-out fold key whose p_inner was used), j, edge_gate,
    cont_gap_s, bridge_s, p_field, tau, combined_all, track_all, combined_airborne,
    track_airborne, n_tracks, n_rows, p_fold (the fold label found in the p file).
fuse.md              plain-language report.
Submissions: data/derived/submissions/F0_fuse/ (9 iterations), O3_fuse_oracle/ (6 testing);
headline scores via score.score_method(..., log=True) (appends to experiments.csv).

Performance note: cuas_common.iteration_frame re-reads every raw sensor CSV on each call (about
1 s); score_rows and make_rows call it for every scoring, so this module memoises it per
iteration key in-process (the Frame is a frozen dataclass; results are identical).

Run: python3 tools/fuse.py [--workers N] [--final-workers N]
Numbered self-checks at the end; exit status 1 on any failure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines as B  # noqa: E402
import cuas_common as C  # noqa: E402
import metadata_map as MM  # noqa: E402
import pair_features as PF  # noqa: E402
import score as SC  # noqa: E402
import sensor_tracks as ST  # noqa: E402
import submission as S  # noqa: E402
import track_rows as TR  # noqa: E402
import truth as T  # noqa: E402

OUT = C.DERIVED / "fused"
TRACKS_DIR = C.DERIVED / "sensor_tracks_v2"
EXIST_DIR = C.DERIVED / "existence"
REPORT = OUT / "fuse.md"
TUNING_CSV = OUT / "tuning.csv"
METHOD_F0 = "F0_fuse"
METHOD_O3 = "O3_fuse_oracle"
TRUTH_MODES = ("all", "airborne")

STEP_S = 0.1                       # template grid spacing
MIN_OVERLAP_STEPS = PF.MIN_OVERLAP_STEPS   # 20 steps = 2 s: "simultaneous" pair
MAX_MIN_DIST_M = PF.MAX_MIN_DIST_M         # 300 m: overlap candidate (min horizontal distance)
SHARE_GATE = PF.HAND_SHARE         # 0.5: edge needs >= half the common steps within 3 sigma
EDGE_GATES = {                     # edge rule (nd_gate, share_gate), tuned as one dimension
    "hand": (PF.CHI2_2_99, SHARE_GATE),    # pair_features' hand rule (F1 0.951 on drone pairs)
    "loose": (25.0, 0.25),                 # both loosened: 25-40 % more passing edges
}
CONFLICT_S = 3.0                   # same sensor_key: grid spans may overlap this much (handover
                                   # of a sensor's object ids; segments carry +1 s extrapolation)
CONT_ND = PF.CHI2_2_99             # 9.21: continuation gate (chi-square(2) 99 %)
CONT_MAX_M = 100.0                 # continuation: absolute distance gate
OUT_ND = 13.82                     # outlier rejection: chi-square(2) 99.9 %
Q_H = ST.Q_ACC_H_M2S3              # 2.5 m^2/s^3
Q_U = ST.Q_ACC_U_M2S3              # 0.5 m^2/s^3
DEFAULT_VAR_U = TR.DEFAULT_VAR_U_M2
PURITY_M = 50.0
PARAMS = {                         # LOIO grid: 2 x 3 x 3 x 2 x 6 = 216 combinations
    "edge_gate": ("hand", "loose"),
    "cont_gap_s": (0.0, 10.0, 30.0),
    "bridge_s": (0.0, 5.0, 20.0),
    "p_field": ("all", "air"),
    "tau": (0.02, 0.05, 0.1, 0.2, 0.3, 0.5),
}
COMBO_KEYS = ("edge_gate", "cont_gap_s", "bridge_s", "p_field", "tau")
N_CHECK_PER_ITER = 60              # self-check 2: random unbridged (track, step) per iteration
                                   # (stratified: 10 with a rejected outlier, 30 multi-member)
N_CHECK_BRIDGED = 15               # ... plus bridged ones
RT_TOL_M = 1e-3
RUNTIME_LIMIT_S = 40 * 60
SEED = 0

# --------------------------------------------------------------------------- frame memo

_FRAMES: dict = {}
_frame_orig = C.iteration_frame


def _frame_cached(it: C.Iteration):
    if it.key not in _FRAMES:
        _FRAMES[it.key] = _frame_orig(it)
    return _FRAMES[it.key]


C.iteration_frame = _frame_cached


def log(msg: str) -> None:
    print(msg, flush=True)


def md_table(header, rows) -> list[str]:
    out = ["| " + " | ".join(str(h) for h in header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def _f(v, nd: int = 3) -> str:
    return "n/a" if v is None or v != v else f"{v:.{nd}f}"


def short(key: str) -> str:
    return key.split("/", 1)[1]


ITERS = {i.key: i for i in C.iterations()}
TESTING = [k for k in sorted(ITERS) if ITERS[k].has_groundtruth]
EVALUATION = [k for k in sorted(ITERS) if not ITERS[k].has_groundtruth]


# --------------------------------------------------------------------------- per-iteration data


def load_p(it: C.Iteration, seg_id: np.ndarray) -> dict:
    """{src: (p_all, p_air, fold_label)} per segment row; src 'outer' = p.csv, or the outer
    fold key k for p_inner rows with outer_fold == k (testing only)."""
    slug = C.iter_slug(it.key)
    t = C.read_columns(EXIST_DIR / f"{slug}__p.csv", numeric={"seg_id", "p_drone_all", "p_drone_air"})
    if not np.array_equal(t["seg_id"].astype(np.int64), seg_id):
        raise ValueError(f"{it.key}: p.csv seg_id does not match tracks.csv")
    folds = sorted(set(t["fold"].tolist()))
    out = {"outer": (t["p_drone_all"], t["p_drone_air"], "|".join(folds))}
    if it.has_groundtruth:
        q = C.read_columns(EXIST_DIR / f"{slug}__p_inner.csv",
                           numeric={"seg_id", "p_drone_all", "p_drone_air"})
        for k in sorted(set(q["outer_fold"].tolist())):
            m = q["outer_fold"] == k
            sid = q["seg_id"][m].astype(np.int64)
            o = np.argsort(sid, kind="stable")
            if not np.array_equal(sid[o], seg_id):
                raise ValueError(f"{it.key}: p_inner fold {k} does not cover every segment")
            out[k] = (q["p_drone_all"][m][o], q["p_drone_air"][m][o],
                      "|".join(sorted(set(q["outer_fold"][m].tolist()))))
    return out


class Prep:
    """Segments, grid, honest variances, overlap pairs and p of one iteration."""

    def __init__(self, it: C.Iteration):
        self.it = it
        self.testing = it.has_groundtruth
        self.s2tab, self.s2_used = PF.sigma_table({it.key} if self.testing else ())
        D = PF.Iter(it, TRACKS_DIR, self.s2tab)
        self.D = D
        slug = C.iter_slug(it.key)
        z = np.load(TRACKS_DIR / f"{slug}__grid.npz")
        vu = z["var_u"].astype(np.float64)
        vu[~np.isfinite(vu) | (vu <= 0)] = DEFAULT_VAR_U
        self.var_u = vu
        u = D.u.copy()
        lo, hi = S.ALT_RANGE
        u[~np.isfinite(u) | (u < lo) | (u > hi)] = np.nan
        self.u_use = u
        del z
        t = C.read_columns(TRACKS_DIR / f"{slug}__tracks.csv", numeric={"seg_id", "n_grid", "cls_uav_fixed"},
                           usecols=["seg_id", "n_grid", "cls_uav_fixed", "serial_top", "model_top",
                                    "protocol_top"])
        if not np.array_equal(t["seg_id"].astype(np.int64), D.seg_id) or \
                not np.array_equal(t["n_grid"].astype(np.int64), D.g_cnt):
            raise ValueError(f"{it.key}: tracks.csv n_grid / seg_id disagree with the grid")
        self.cls_fixed = np.nan_to_num(t["cls_uav_fixed"].astype(float))
        self.serial = np.array([MM.resolve_serial(s) for s in t["serial_top"]], dtype=object)
        self.model = np.array([MM.map_value(MM.MODEL, m) for m in t["model_top"]], dtype=object)
        self.protocol = np.array([MM.map_value(MM.PROTOCOL, p) for p in t["protocol_top"]], dtype=object)
        nseg = D.nseg
        self.dep_ids = C.deployable_sensor_ids()
        self.dep = np.isin(D.sensor_id, sorted(self.dep_ids)) & (D.g_cnt > 0)
        has = D.g_cnt > 0
        self.first = np.full(nseg, -1, dtype=np.int64)
        self.last = np.full(nseg, -1, dtype=np.int64)
        self.first[has] = D.step[D.g_first[has]]
        self.last[has] = D.step[D.g_first[has] + D.g_cnt[has] - 1]
        same = D.grow[1:] == D.grow[:-1]
        if np.any(np.diff(D.step)[same] <= 0):
            raise ValueError(f"{it.key}: grid not sorted by step within a segment")
        # height fallbacks (deployable segments only, never ground truth)
        dsm = self.dep[D.grow] & np.isfinite(u)
        self.iter_med = float(np.median(u[dsm])) if dsm.any() else float("nan")
        self.sensor_med = {}
        sid_row = D.sensor_id[D.grow]
        for s in sorted(set(D.sensor_id[self.dep].tolist())):
            x = u[dsm & (sid_row == s)]
            self.sensor_med[s] = float(np.median(x)) if x.size else float("nan")
        # overlap pairs among deployable segments
        ks, cs, ms = PF.overlap_stats(D)
        ra, rb = ks // nseg, ks % nseg
        keep = self.dep[ra] & self.dep[rb] & (cs >= MIN_OVERLAP_STEPS)
        self.sim_keys = ks[keep]
        cand = ms[keep] <= MAX_MIN_DIST_M
        self.cand_keys = self.sim_keys[cand]
        if self.cand_keys.size:
            I, J = PF.overlap_samples(D, self.cand_keys)
            f = PF.overlap_features(D, self.cand_keys, I, J)
            self.nd_med, self.share = f["ov_nd_med"], f["ov_share3s"]
            del I, J, f
        else:
            self.nd_med = self.share = np.zeros(0)
        self.p = load_p(it, D.seg_id)


# --------------------------------------------------------------------------- fusion core


def fuse_groups(P: Prep, idx: np.ndarray, gid: np.ndarray, ng: int) -> dict:
    """Fused state of each group of grid samples (gid sorted, every group 0..ng-1 non-empty)."""
    D = P.D
    e, n = D.e[idx], D.n[idx]
    Ve, Vn = D.Ve[idx], D.Vn[idx]
    we, wn = 1.0 / Ve, 1.0 / Vn
    st = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]])
    if st.size != ng:
        raise ValueError("fuse_groups: empty group")
    cnt = np.diff(np.r_[st, gid.size])
    re, rn = e[st][gid], n[st][gid]
    de, dn = e - re, n - rn

    def sums(k):
        return (np.bincount(gid, we * k, ng), np.bincount(gid, wn * k, ng),
                np.bincount(gid, we * k * de, ng), np.bincount(gid, wn * k * dn, ng))

    keep = np.ones(idx.size)
    Swe, Swn, Se, Sn = sums(keep)
    c3 = cnt[gid] >= 3
    n_rej = np.zeros(ng, dtype=np.int64)
    if c3.any():
        with np.errstate(divide="ignore", invalid="ignore"):
            lwe, lwn = Swe[gid] - we, Swn[gid] - wn
            loo_e = (Se[gid] - we * de) / lwe
            loo_n = (Sn[gid] - wn * dn) / lwn
            nd = (de - loo_e) ** 2 / (Ve + 1.0 / lwe) + (dn - loo_n) ** 2 / (Vn + 1.0 / lwn)
        nd = np.where(c3, nd, -np.inf)
        gmax = np.maximum.reduceat(nd, st)
        worst = c3 & (nd == gmax[gid]) & (gmax[gid] > OUT_ND)
        wi = np.flatnonzero(worst)
        if wi.size:
            _, fi = np.unique(gid[wi], return_index=True)
            drop = wi[fi]
            keep[drop] = 0.0
            n_rej[gid[drop]] = 1
            Swe, Swn, Se, Sn = sums(keep)
    # diagnostic only: steps that still hold an outlier after the single drop
    n_resid = 0
    if n_rej.any():
        k3 = np.bincount(gid, keep, ng)[gid] >= 3
        r3 = c3 & (keep > 0) & k3 & (n_rej[gid] > 0)
        if r3.any():
            with np.errstate(divide="ignore", invalid="ignore"):
                kwe, kwn = keep * we, keep * wn
                lwe, lwn = Swe[gid] - kwe, Swn[gid] - kwn
                loo_e = (Se[gid] - kwe * de) / lwe
                loo_n = (Sn[gid] - kwn * dn) / lwn
                nd2 = (de - loo_e) ** 2 / (Ve + 1.0 / lwe) + (dn - loo_n) ** 2 / (Vn + 1.0 / lwn)
            n_resid = int(np.unique(gid[r3 & (nd2 > OUT_ND)]).size)
    out = {"e": e[st] + Se / Swe, "n": n[st] + Sn / Swn, "var_e": 1.0 / Swe, "var_n": 1.0 / Swn,
           "ve": np.bincount(gid, we * keep * D.ve[idx], ng) / Swe,
           "vn": np.bincount(gid, wn * keep * D.vn[idx], ng) / Swn,
           "n_used": np.bincount(gid, keep, ng).astype(np.int64), "n_rej": n_rej,
           "n_resid_out": n_resid, "n_ge3": int((cnt >= 3).sum())}
    u = P.u_use[idx]
    valid = (keep > 0) & np.isfinite(u)
    wu = 1.0 / P.var_u[idx]
    refu = np.full(ng, np.nan)
    vi = np.flatnonzero(valid)
    if vi.size:
        ug, fi = np.unique(gid[vi], return_index=True)
        refu[ug] = u[vi[fi]]
    du = np.where(valid, u - refu[gid], 0.0)
    Swu = np.bincount(gid, np.where(valid, wu, 0.0), ng)
    Su = np.bincount(gid, np.where(valid, wu, 0.0) * du, ng)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["u"] = np.where(Swu > 0, refu + Su / Swu, np.nan)
        out["var_u"] = np.where(Swu > 0, 1.0 / Swu, np.nan)
    return out


class Clusters:
    pass


def cluster(P: Prep, edge_gate: str, cont_gap_s: float) -> Clusters:
    """Union-find with sensor and complete-linkage constraints, then continuation chains."""
    D = P.D
    nseg = D.nseg
    nd_gate, share_gate = EDGE_GATES[edge_gate]
    passing = (P.nd_med < nd_gate) & (P.share >= share_gate)
    edges, end = P.cand_keys[passing], P.nd_med[passing]
    conflict = set(P.sim_keys.tolist()) - set(edges.tolist())
    order = np.lexsort((edges, end))
    sk = D.sensor_key
    first, last = P.first, P.last
    dep_rows = np.flatnonzero(P.dep)
    parent = {int(r): int(r) for r in dep_rows}
    members = {int(r): [int(r)] for r in dep_rows}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def compatible(MX, MY) -> bool:
        for a in MX:
            for b in MY:
                if sk[a] == sk[b]:
                    if (min(last[a], last[b]) - max(first[a], first[b])) * STEP_S > CONFLICT_S:
                        return False
                elif (min(a, b) * nseg + max(a, b)) in conflict:
                    return False
        return True

    n_merge = n_block = 0
    for k in edges[order].tolist():
        a, b = divmod(int(k), nseg)
        X, Y = find(a), find(b)
        if X == Y:
            continue
        if not compatible(members[X], members[Y]):
            n_block += 1
            continue
        if len(members[X]) < len(members[Y]) or (len(members[X]) == len(members[Y]) and Y < X):
            X, Y = Y, X
        parent[Y] = X
        members[X].extend(members[Y])
        del members[Y]
        n_merge += 1
    cls = [sorted(m) for m in members.values()]
    cls.sort(key=lambda m: (min(first[r] for r in m), m[0]))
    ncl = len(cls)
    seg_cluster = np.full(nseg, -1, dtype=np.int64)
    for c, m in enumerate(cls):
        seg_cluster[m] = c
    cl_first = np.array([min(first[r] for r in m) for m in cls], dtype=np.int64)
    cl_last = np.array([max(last[r] for r in m) for m in cls], dtype=np.int64)

    succ = np.full(ncl, -1, dtype=np.int64)
    pred = np.full(ncl, -1, dtype=np.int64)
    n_cont_cand = 0
    if cont_gap_s > 0 and ncl > 1:
        gap_steps = int(round(cont_gap_s / STEP_S))
        dmask = P.dep[D.grow]
        sidx = np.flatnonzero(dmask)
        scl = seg_cluster[D.grow[sidx]]
        ends = {}
        for name, ref in (("first", cl_first), ("last", cl_last)):
            sel = sidx[D.step[sidx] == ref[scl]]
            c_of = seg_cluster[D.grow[sel]]
            o = np.argsort(c_of, kind="stable")
            sel, c_of = sel[o], c_of[o]
            fg = fuse_groups(P, sel, c_of, ncl)
            ends[name] = fg
        oc = np.argsort(cl_first, kind="stable")
        f_sorted = cl_first[oc]
        lo = np.searchsorted(f_sorted, cl_last, side="right")
        hi = np.searchsorted(f_sorted, cl_last + gap_steps, side="right")
        cntp = hi - lo
        X = np.repeat(np.arange(ncl), cntp)
        Ypos = np.repeat(lo - np.cumsum(np.r_[0, cntp[:-1]]), cntp) + np.arange(cntp.sum())
        Y = oc[Ypos] if X.size else np.zeros(0, dtype=np.int64)
        n_cont_cand = int(X.size)
        if X.size:
            a, b = ends["last"], ends["first"]
            dt = (cl_first[Y] - cl_last[X]) * STEP_S
            grow = Q_H * np.abs(dt) ** 3 / 3.0
            de = a["e"][X] + a["ve"][X] * dt - b["e"][Y]
            dn = a["n"][X] + a["vn"][X] * dt - b["n"][Y]
            nd = de ** 2 / (a["var_e"][X] + b["var_e"][Y] + grow) + dn ** 2 / (a["var_n"][X] + b["var_n"][Y] + grow)
            ok = (dt > 0) & (nd < CONT_ND) & (np.hypot(de, dn) < CONT_MAX_M)
            X, Y, nd = X[ok], Y[ok], nd[ok]
            for i in np.lexsort((Y, X, nd)).tolist():
                x, y = int(X[i]), int(Y[i])
                if succ[x] < 0 and pred[y] < 0:
                    succ[x], pred[y] = y, x
    # chains -> tracks
    heads = [c for c in range(ncl) if pred[c] < 0]
    chains = []
    for h in heads:
        ch, c = [], h
        while c >= 0:
            ch.append(c)
            c = int(succ[c])
        chains.append(ch)
    chains.sort(key=lambda ch: (cl_first[ch[0]], cls[ch[0]][0]))
    cl_track = np.full(ncl, -1, dtype=np.int64)
    for t, ch in enumerate(chains):
        cl_track[ch] = t
    seg_track = np.full(nseg, -1, dtype=np.int64)
    seg_track[dep_rows] = cl_track[seg_cluster[dep_rows]]
    mt, mc, ms = [], [], []
    for t, ch in enumerate(chains):
        for c in ch:
            for r in cls[c]:
                mt.append(t); mc.append(c); ms.append(r)
    Cl = Clusters()
    Cl.edge_gate, Cl.nd_gate, Cl.share_gate, Cl.cont_gap_s = edge_gate, nd_gate, share_gate, cont_gap_s
    Cl.ncl, Cl.ntracks = ncl, len(chains)
    Cl.seg_cluster, Cl.seg_track, Cl.cl_track = seg_cluster, seg_track, cl_track
    Cl.mem_track = np.array(mt, dtype=np.int64)
    Cl.mem_cluster = np.array(mc, dtype=np.int64)
    Cl.mem_seg = np.array(ms, dtype=np.int64)
    o = np.lexsort((Cl.mem_seg, Cl.mem_track))
    Cl.mem_track, Cl.mem_cluster, Cl.mem_seg = Cl.mem_track[o], Cl.mem_cluster[o], Cl.mem_seg[o]
    Cl.mem_start = np.searchsorted(Cl.mem_track, np.arange(Cl.ntracks))
    Cl.chains = chains
    Cl.stats = dict(n_segments=int(dep_rows.size), n_sim_pairs=int(P.sim_keys.size),
                    n_candidates=int(P.cand_keys.size), n_pass=int(edges.size),
                    n_conflict=len(conflict), n_merge=n_merge, n_blocked=n_block, n_clusters=ncl,
                    n_cont_candidates=n_cont_cand, n_cont_links=int((succ >= 0).sum()),
                    n_tracks=len(chains))
    return Cl


def members_hash(Cl: Clusters) -> str:
    h = hashlib.sha256()
    for a in (Cl.mem_track, Cl.mem_cluster, Cl.mem_seg):
        h.update(np.ascontiguousarray(a, dtype=np.int64).tobytes())
    return h.hexdigest()


def track_main_sensor(P: Prep, Cl: Clusters, t: int) -> str:
    segs = Cl.mem_seg[Cl.mem_track == t]
    cnt: dict[str, int] = {}
    for r in segs.tolist():
        s = P.D.sensor_id[r]
        cnt[s] = cnt.get(s, 0) + int(P.D.g_cnt[r])
    return sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def fuse(P: Prep, Cl: Clusters, bridge_s: float) -> dict:
    D = P.D
    idx0 = np.flatnonzero(P.dep[D.grow])
    trk = Cl.seg_track[D.grow[idx0]]
    o = np.lexsort((D.step[idx0], trk))
    idx, tk = idx0[o], trk[o]
    sp = D.step[idx]
    new = np.r_[True, (tk[1:] != tk[:-1]) | (sp[1:] != sp[:-1])]
    gid = np.cumsum(new) - 1
    ng = int(gid[-1]) + 1
    g = fuse_groups(P, idx, gid, ng)
    st = np.flatnonzero(new)
    g["track"], g["step"] = tk[st], sp[st]
    nt = Cl.ntracks
    # heights: track median, else main-sensor median, else iteration median
    tmed = ST.group_quantile(g["track"], g["u"], nt, (0.5,))[0]
    tfb = tmed.copy()
    tsrc = np.ones(nt, dtype=np.int64)
    for t in np.flatnonzero(~np.isfinite(tmed)).tolist():
        v = P.sensor_med.get(track_main_sensor(P, Cl, t), float("nan"))
        if v == v:
            tfb[t], tsrc[t] = v, 2
        else:
            tfb[t], tsrc[t] = P.iter_med, 3
    miss = ~np.isfinite(g["u"])
    g["u_src"] = np.where(miss, tsrc[g["track"]], 0)
    g["u"] = np.where(miss, tfb[g["track"]], g["u"])
    g["var_u"] = np.where(miss, DEFAULT_VAR_U, g["var_u"])     # fallback height: track_rows' default
    g["bridged"] = np.zeros(ng, dtype=bool)
    g["tmed"] = tmed
    # gap bridging
    if bridge_s > 0 and ng > 1:
        same = g["track"][1:] == g["track"][:-1]
        ds = g["step"][1:] - g["step"][:-1]
        gi = np.flatnonzero(same & (ds > 1) & (ds * STEP_S <= bridge_s + 1e-9))
        if gi.size:
            nins = ds[gi] - 1
            rep = np.repeat(gi, nins)
            off = np.arange(nins.sum()) - np.repeat(np.cumsum(nins) - nins, nins) + 1
            frac = off / ds[rep]
            gap = ds[rep] * STEP_S
            add = {"track": g["track"][rep], "step": g["step"][rep] + off}
            for c in ("e", "n", "u", "ve", "vn"):
                a = g[c]
                add[c] = a[rep] + frac * (a[rep + 1] - a[rep])
            for c, q in (("var_e", Q_H), ("var_n", Q_H), ("var_u", Q_U)):
                a = g[c]
                add[c] = np.maximum(a[rep], a[rep + 1]) + q * (gap / 2.0) ** 3 / 3.0
            add["n_used"] = np.zeros(rep.size, dtype=np.int64)
            add["n_rej"] = np.zeros(rep.size, dtype=np.int64)
            add["u_src"] = np.full(rep.size, 4, dtype=np.int64)
            add["bridged"] = np.ones(rep.size, dtype=bool)
            cols = ("track", "step", "e", "n", "u", "var_e", "var_n", "var_u", "ve", "vn", "n_used",
                    "n_rej", "u_src", "bridged")
            for c in cols:
                g[c] = np.concatenate([g[c], add[c]])
            o = np.lexsort((g["step"], g["track"]))
            for c in cols:
                g[c] = g[c][o]
    return g


def _majority(vals, weights) -> str:
    cnt: dict[str, list[float]] = {}
    for v, w in zip(vals, weights):
        if v:
            c = cnt.setdefault(v, [0, 0.0])
            c[0] += 1
            c[1] += w
    if not cnt:
        return ""
    return sorted(cnt.items(), key=lambda kv: (-kv[1][0], -kv[1][1], kv[0]))[0][0]


def track_table(P: Prep, Cl: Clusters, g: dict, whitelist: set[str]) -> dict:
    D = P.D
    nt = Cl.ntracks
    mt, ms = Cl.mem_track, Cl.mem_seg
    w = D.g_cnt[ms].astype(float)
    tt: dict = {"track": np.arange(nt)}
    tt["n_members"] = np.bincount(mt, minlength=nt)
    tt["n_clusters"] = np.array([len(ch) for ch in Cl.chains], dtype=np.int64)
    serial, model, protocol, sids = [], [], [], []
    for t in range(nt):
        a, b = Cl.mem_start[t], (Cl.mem_start[t + 1] if t + 1 < nt else ms.size)
        segs = ms[a:b]
        ww = w[a:b]
        s = _majority(P.serial[segs], ww)
        serial.append(s)
        model.append(_majority(P.model[segs], ww) or MM.model_for_serial(s) or "Unknown")
        protocol.append(_majority(P.protocol[segs], ww) or "Unknown")
        sids.append("|".join(sorted(set(D.sensor_id[segs].tolist()))))
    tt["sensor_ids"] = np.array(sids, dtype=object)
    tt["n_sensors"] = np.array([len(s.split("|")) for s in sids], dtype=np.int64)
    tt["serial"] = np.array(serial, dtype=object)
    tt["model"] = np.array(model, dtype=object)
    tt["protocol"] = np.array(protocol, dtype=object)
    tt["cls_fixed_share"] = np.bincount(mt, w * P.cls_fixed[ms], nt) / np.bincount(mt, w, nt)
    tr = g["track"]
    tt["first_step"] = np.full(nt, -1, dtype=np.int64)
    tt["last_step"] = np.full(nt, -1, dtype=np.int64)
    st = np.searchsorted(tr, np.arange(nt))
    en = np.searchsorted(tr, np.arange(nt), side="right") - 1
    tt["first_step"], tt["last_step"] = g["step"][st], g["step"][en]
    tt["n_steps"] = np.bincount(tr, minlength=nt)
    tt["n_bridged"] = np.bincount(tr, g["bridged"].astype(float), nt).astype(np.int64)
    tt["duration_s"] = (tt["last_step"] - tt["first_step"]) * STEP_S
    ub = ~g["bridged"]
    sp = np.hypot(g["ve"], g["vn"])
    q = ST.group_quantile(tr[ub], sp[ub], nt, (0.5, 0.1, 0.9))
    tt["speed_med_ms"], tt["speed_p10_ms"], tt["speed_p90_ms"] = q[0], q[1], q[2]
    tt["u_med_m"] = ST.group_quantile(tr, g["u"], nt, (0.5,))[0]
    lab = {
        "classification": np.where(tt["cls_fixed_share"] > 0.5, "Fixed Wing UAS", "Rotary Wing UAS").astype(object),
        "identity": np.array(["Friendly" if s and s in whitelist else "Hostile" for s in serial], dtype=object),
        "uav_model/manufacturer": tt["model"],
        "protocol": tt["protocol"],
        "serial_number": np.array([s or "Unknown" for s in serial], dtype=object),
        "signal_frequency_band": np.full(nt, "Unknown", dtype=object),
    }
    tt["lab"] = lab
    return tt


def track_p(Cl: Clusters, p: np.ndarray, n_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(max, n_grid-weighted mean) of the member segments' p per track."""
    ms = Cl.mem_seg
    pmax = np.maximum.reduceat(p[ms], Cl.mem_start) if ms.size else np.zeros(0)
    w = n_grid[ms].astype(float)
    pw = np.bincount(Cl.mem_track, w * p[ms], Cl.ntracks) / np.bincount(Cl.mem_track, w, Cl.ntracks)
    return pmax, pw


def build_rows(it: C.Iteration, g: dict, tt: dict, conf: np.ndarray, sel: np.ndarray | None = None) -> dict:
    nt = len(tt["track"])
    keys = np.array([f"{it.key}/track{t}" for t in range(nt)], dtype=object)
    r = np.arange(g["track"].size) if sel is None else np.flatnonzero(sel[g["track"]])
    tr = g["track"][r]
    labels = {c: v[tr] for c, v in tt["lab"].items()}
    labels["classification_confidence"] = np.clip(conf[tr], 0.0, 1.0)
    return TR.make_rows(it, g["step"][r], keys[tr], g["e"][r], g["n"][r], g["u"][r], g["var_e"][r],
                        g["var_n"][r], g["var_u"][r], 0.0, labels)


def testing_labels(P: Prep, Cl: Clusters, g: dict, tr_all: T.Truth) -> dict:
    D = P.D
    nt = Cl.ntracks
    ms, mt = Cl.mem_seg, Cl.mem_track
    w = D.g_cnt[ms].astype(float)
    share = np.bincount(mt, w * D.is_drone[ms], nt) / np.bincount(mt, w, nt)
    lab = share >= 0.5
    flight = np.full(nt, "", dtype=object)
    for t in np.flatnonzero(lab).tolist():
        a, b = Cl.mem_start[t], (Cl.mem_start[t + 1] if t + 1 < nt else ms.size)
        segs = ms[a:b]
        dm = D.is_drone[segs]
        flight[t] = _majority(D.flight[segs][dm], w[a:b][dm])
    fidx = {f["flight_key"]: i for i, f in enumerate(tr_all.flights)}
    nf = max(len(tr_all.flights), 1)
    tf = np.array([fidx.get(f, -1) if f else -1 for f in flight], dtype=np.int64)
    rk_ok = tf[g["track"]] >= 0
    tkey = tr_all.step.astype(np.int64) * nf + tr_all.flight.astype(np.int64)
    rkey = g["step"][rk_ok] * nf + tf[g["track"][rk_ok]]
    pos = np.minimum(np.searchsorted(tkey, rkey), max(tkey.size - 1, 0))
    hit = (tkey[pos] == rkey) if tkey.size else np.zeros(rkey.size, bool)
    d3 = np.sqrt((g["e"][rk_ok] - tr_all.e[pos]) ** 2 + (g["n"][rk_ok] - tr_all.n[pos]) ** 2
                 + (g["u"][rk_ok] - tr_all.u[pos]) ** 2)
    good = np.zeros(g["track"].size)
    good[np.flatnonzero(rk_ok)] = (hit & (d3 <= PURITY_M)).astype(float)
    purity = np.bincount(g["track"], good, nt) / np.maximum(np.bincount(g["track"], minlength=nt), 1)
    purity[~lab] = np.nan
    return {"drone_share": share, "label_drone": lab, "flight": flight, "flight_idx": tf, "purity": purity}


# --------------------------------------------------------------------------- phase 1: tuning

_PREP: dict = {}


def _prep(key: str) -> Prep:
    if key not in _PREP:
        _PREP.clear()
        _PREP[key] = Prep(ITERS[key])
    return _PREP[key]


def _score_pair(it: C.Iteration, rows: dict) -> tuple:
    out = []
    for m in TRUTH_MODES:
        r = SC.score_rows(it, rows, m, configs=(SC.PRIMARY,))
        out += [r["parts"]["combined"], r["parts"]["track"]]
    return tuple(out)


def tune_task(job: tuple[str, float]) -> dict:
    key, edge_gate = job
    t0 = time.time()
    it = ITERS[key]
    P = _prep(key)
    wl = B.load_whitelist()
    out = {"key": key, "edge_gate": edge_gate, "records": [], "hash": {}, "stats": {}}
    for cg in PARAMS["cont_gap_s"]:
        Cl = cluster(P, edge_gate, cg)
        out["hash"][cg] = members_hash(Cl)
        out["stats"][cg] = Cl.stats
        for bs in PARAMS["bridge_s"]:
            g = fuse(P, Cl, bs)
            tt = track_table(P, Cl, g, wl)
            rows = build_rows(it, g, tt, np.full(Cl.ntracks, 0.5))
            row_track = g["track"]
            cache: dict = {}
            for src, (pa, pr, pfold) in P.p.items():
                for pf in PARAMS["p_field"]:
                    pt = track_p(Cl, pa if pf == "all" else pr, P.D.g_cnt)[0]
                    for tau in PARAMS["tau"]:
                        sel = pt >= tau
                        h = hashlib.sha1(np.packbits(sel).tobytes()).hexdigest()
                        if h not in cache:
                            rm = sel[row_track]
                            sub = {c: np.asarray(v)[rm] for c, v in rows.items()}
                            cache[h] = _score_pair(it, sub) + (int(sel.sum()), int(rm.sum()))
                        ca, ta, cr, tr_, ntk, nrw = cache[h]
                        out["records"].append(dict(
                            k=src, j=key, edge_gate=edge_gate, cont_gap_s=cg, bridge_s=bs, p_field=pf,
                            tau=tau, combined_all=ca, track_all=ta, combined_airborne=cr,
                            track_airborne=tr_, n_tracks=ntk, n_rows=nrw, p_fold=pfold))
            del rows, g, tt
    out["seconds"] = time.time() - t0
    return out


def combo_of(r: dict) -> tuple:
    return tuple(r[c] for c in COMBO_KEYS)


def all_combos() -> list[tuple]:
    out = [()]
    for c in COMBO_KEYS:
        out = [o + (v,) for o in out for v in PARAMS[c]]
    return out


def choose(records: list[dict], src: str, js: list[str]) -> tuple[tuple, float, list]:
    """argmax over all_combos() (first on ties) of mean over js of (comb_all + comb_air) / 2,
    using only records with k == src. Returns (combo, objective, records used)."""
    idx = {}
    for r in records:
        if r["k"] == src and r["j"] in js:
            idx[(combo_of(r), r["j"])] = r
    best, bv, used = None, -math.inf, []
    for cb in all_combos():
        rs = [idx.get((cb, j)) for j in js]
        if any(x is None for x in rs):
            raise ValueError(f"missing score for {cb} / {src}")
        v = float(np.mean([(x["combined_all"] + x["combined_airborne"]) / 2 for x in rs]))
        if v > bv:
            best, bv = cb, v
        used.extend(rs)
    return best, bv, used


# --------------------------------------------------------------------------- phase 2: apply


def check_constraints(P: Prep, Cl: Clusters) -> dict:
    """Self-check 1 from the raw grid file: same-sensor span overlap <= CONFLICT_S and every
    simultaneous different-sensor member pair of a cluster passes the edge rule."""
    it = P.it
    slug = C.iter_slug(it.key)
    z = np.load(TRACKS_DIR / f"{slug}__grid.npz")
    seg = z["seg_id"].astype(np.int64)
    step = z["step"].astype(np.int64)
    e, n = z["e"].astype(np.float64), z["n"].astype(np.float64)
    vae, van = z["var_e"].astype(np.float64), z["var_n"].astype(np.float64)
    t = C.read_columns(TRACKS_DIR / f"{slug}__tracks.csv", numeric={"seg_id"},
                       usecols=["seg_id", "sensor_key", "sensor_id"])
    skey = dict(zip(t["seg_id"].astype(np.int64).tolist(), t["sensor_key"].tolist()))
    sidd = dict(zip(t["seg_id"].astype(np.int64).tolist(), t["sensor_id"].tolist()))
    bnd = np.searchsorted(seg, np.arange(P.D.nseg + 1))
    res = {"same_sensor_bad": 0, "edge_bad": 0, "cross_cluster_sim": 0, "pairs_checked": 0,
           "sim_pairs": 0, "non_deployable": 0, "examples": []}
    s2 = P.s2tab
    dflt = PF.DEFAULT_SIGMA_M ** 2
    for tk in range(Cl.ntracks):
        a0 = Cl.mem_start[tk]
        b0 = Cl.mem_start[tk + 1] if tk + 1 < Cl.ntracks else Cl.mem_seg.size
        segs = Cl.mem_seg[a0:b0].tolist()
        clus = Cl.mem_cluster[a0:b0].tolist()
        for s in segs:
            if sidd[s] not in P.dep_ids:
                res["non_deployable"] += 1
        for x in range(len(segs)):
            for y in range(x + 1, len(segs)):
                a, b = segs[x], segs[y]
                sa, sb = step[bnd[a]:bnd[a + 1]], step[bnd[b]:bnd[b + 1]]
                res["pairs_checked"] += 1
                if skey[a] == skey[b]:
                    ov = (min(sa[-1], sb[-1]) - max(sa[0], sb[0])) * STEP_S
                    if ov > CONFLICT_S:
                        res["same_sensor_bad"] += 1
                        res["examples"].append(("same", tk, a, b, float(ov)))
                    continue
                _, ia, ib = np.intersect1d(sa, sb, return_indices=True)
                if ia.size < MIN_OVERLAP_STEPS:
                    continue
                res["sim_pairs"] += 1
                if clus[x] != clus[y]:
                    res["cross_cluster_sim"] += 1
                    continue
                ia, ib = ia + bnd[a], ib + bnd[b]
                ea, na = s2.get(skey[a], (dflt, dflt))
                eb, nb = s2.get(skey[b], (dflt, dflt))
                de, dn = e[ia] - e[ib], n[ia] - n[ib]
                nd = de ** 2 / (np.maximum(vae[ia], ea) + np.maximum(vae[ib], eb)) \
                    + dn ** 2 / (np.maximum(van[ia], na) + np.maximum(van[ib], nb))
                ok = (np.median(nd) < Cl.nd_gate) and (np.mean(nd <= PF.SIGMA3_ND) >= Cl.share_gate) \
                    and (np.hypot(de, dn).min() <= MAX_MIN_DIST_M)
                if not ok:
                    res["edge_bad"] += 1
                    res["examples"].append(("edge", tk, a, b, float(np.median(nd))))
    return res


def check_fused_python(P: Prep, Cl: Clusters, g: dict, rng: np.random.Generator) -> dict:
    """Self-check 2: plain-python recomputation of random fused rows from the raw grid file:
    e, n, u (m), ve, vn (m/s) and var_e, var_n, var_u (relative) of unbridged rows; bridged
    rows from the plain-python recomputation of their two bracketing unbridged rows."""
    slug = C.iter_slug(P.it.key)
    z = np.load(TRACKS_DIR / f"{slug}__grid.npz")
    raw = {k: z[k] for k in ("seg_id", "step", "e", "n", "u", "ve", "vn", "var_e", "var_n", "var_u")}
    seg = raw["seg_id"].astype(np.int64)
    bnd = np.searchsorted(seg, np.arange(P.D.nseg + 1))
    lo, hi = S.ALT_RANGE
    dflt = PF.DEFAULT_SIGMA_M ** 2
    ub = np.flatnonzero(~g["bridged"])
    br = np.flatnonzero(g["bridged"])
    multi = np.flatnonzero(~g["bridged"] & (g["n_used"] + g["n_rej"] >= 2))
    rej = np.flatnonzero(g["n_rej"] > 0)
    pick = []
    for pool, k in ((rej, N_CHECK_PER_ITER // 6), (multi, N_CHECK_PER_ITER // 2), (ub, N_CHECK_PER_ITER)):
        pool = np.setdiff1d(pool, np.array(pick, dtype=np.int64))
        need = min(k, N_CHECK_PER_ITER - len(pick), pool.size)
        if need > 0:
            pick += rng.choice(pool, size=need, replace=False).tolist()
    pick_b = list(rng.choice(br, size=min(N_CHECK_BRIDGED, br.size), replace=False)) if br.size else []
    stat = {"n_rej": 0, "n_multi": 0}

    def wmean(L, x, v):
        return sum(q[x] / q[v] for q in L) / sum(1.0 / q[v] for q in L)

    def expect(i: int) -> dict:
        """Fused state of unbridged row i, recomputed from the raw grid in plain python."""
        tk, s = int(g["track"][i]), int(g["step"][i])
        segs = Cl.mem_seg[Cl.mem_track == tk].tolist()
        smp = []
        for r in sorted(segs):
            stp = raw["step"][bnd[r]:bnd[r + 1]].tolist()
            if s in stp:
                j = bnd[r] + stp.index(s)
                se, sn = P.s2tab.get(P.D.sensor_key[r], (dflt, dflt))
                uu = float(raw["u"][j])
                vu = float(raw["var_u"][j])
                smp.append(dict(e=float(raw["e"][j]), n=float(raw["n"][j]),
                                ve=float(raw["ve"][j]), vn=float(raw["vn"][j]),
                                Ve=max(float(raw["var_e"][j]), se), Vn=max(float(raw["var_n"][j]), sn),
                                u=uu if (uu == uu and lo <= uu <= hi) else None,
                                vu=vu if (vu == vu and vu > 0) else DEFAULT_VAR_U))
        nrej = 0
        if len(smp) >= 3:
            nds = []
            for a in range(len(smp)):
                oth = [q for b, q in enumerate(smp) if b != a]
                me, mn = wmean(oth, "e", "Ve"), wmean(oth, "n", "Vn")
                ve_ = 1.0 / sum(1.0 / q["Ve"] for q in oth)
                vn_ = 1.0 / sum(1.0 / q["Vn"] for q in oth)
                nds.append((smp[a]["e"] - me) ** 2 / (smp[a]["Ve"] + ve_)
                           + (smp[a]["n"] - mn) ** 2 / (smp[a]["Vn"] + vn_))
            mx = max(nds)
            if mx > OUT_ND:
                smp.pop(nds.index(mx))
                nrej = 1
        x = {"e": wmean(smp, "e", "Ve"), "n": wmean(smp, "n", "Vn"),
             "ve": wmean(smp, "ve", "Ve"), "vn": wmean(smp, "vn", "Vn"),
             "var_e": 1.0 / sum(1.0 / q["Ve"] for q in smp), "var_n": 1.0 / sum(1.0 / q["Vn"] for q in smp),
             "nrej": nrej, "nsmp": len(smp) + nrej}
        withu = [q for q in smp if q["u"] is not None]
        if withu:
            x["u"] = sum(q["u"] / q["vu"] for q in withu) / sum(1.0 / q["vu"] for q in withu)
            x["var_u"] = 1.0 / sum(1.0 / q["vu"] for q in withu)
            return x
        x["var_u"] = DEFAULT_VAR_U
        if g["u_src"][i] == 1:
            x["u"] = statistics.median(g["u"][(g["track"] == tk) & (g["u_src"] == 0)].tolist())
        elif g["u_src"][i] == 2:
            x["u"] = P.sensor_med[track_main_sensor(P, Cl, tk)]
        else:
            x["u"] = P.iter_med
        return x

    err = {"pos": 0.0, "vel": 0.0, "var": 0.0}

    def compare(x: dict, i: int) -> None:
        def d(c, rel=False):
            v = abs(x[c] - float(g[c][i])) / (x[c] if rel else 1.0)
            return v if v == v else math.inf          # a NaN on either side fails
        err["pos"] = max(err["pos"], *(d(c) for c in ("e", "n", "u")))
        err["vel"] = max(err["vel"], *(d(c) for c in ("ve", "vn")))
        err["var"] = max(err["var"], *(d(c, True) for c in ("var_e", "var_n", "var_u")))

    for i in pick:
        x = expect(int(i))
        stat["n_rej"] += x["nrej"]
        stat["n_multi"] += x["nsmp"] >= 2
        compare(x, int(i))
    for i in pick_b:
        tk, s = int(g["track"][i]), int(g["step"][i])
        m = np.flatnonzero((g["track"] == tk) & ~g["bridged"])
        st_ = g["step"][m].tolist()
        a = max(q for q in range(len(st_)) if st_[q] < s)
        b = min(q for q in range(len(st_)) if st_[q] > s)
        xa, xb = expect(int(m[a])), expect(int(m[b]))
        fr = (s - st_[a]) / (st_[b] - st_[a])
        gap = (st_[b] - st_[a]) * STEP_S
        x = {c: xa[c] + fr * (xb[c] - xa[c]) for c in ("e", "n", "u", "ve", "vn")}
        for c, q in (("var_e", Q_H), ("var_n", Q_H), ("var_u", Q_U)):
            x[c] = max(xa[c], xb[c]) + q * (gap / 2) ** 3 / 3
        compare(x, int(i))
    return {"n": len(pick), "n_bridged": len(pick_b), "max_err": err["pos"], "vel_err": err["vel"],
            "var_rel": err["var"], "n_rej": stat["n_rej"], "n_multi": stat["n_multi"]}


def check_singletons(P: Prep, Cl: Clusters, g: dict) -> dict:
    """Self-check 3: single-member unbridged tracks equal their segment's grid samples."""
    D = P.D
    nm = np.bincount(Cl.mem_track, minlength=Cl.ntracks)
    nb = np.bincount(g["track"], g["bridged"].astype(float), Cl.ntracks)
    tr_one = np.flatnonzero((nm == 1) & (nb == 0))
    seg_of = np.full(Cl.ntracks, -1, dtype=np.int64)
    seg_of[Cl.mem_track] = Cl.mem_seg
    bad, n = 0, 0
    starts = np.searchsorted(g["track"], tr_one)
    ends = np.searchsorted(g["track"], tr_one, side="right")
    for t, a, b in zip(tr_one.tolist(), starts.tolist(), ends.tolist()):
        r = seg_of[t]
        ga, gb = D.g_first[r], D.g_first[r] + D.g_cnt[r]
        n += 1
        ok = (b - a == gb - ga and np.array_equal(g["step"][a:b], D.step[ga:gb])
              and np.array_equal(g["e"][a:b], D.e[ga:gb]) and np.array_equal(g["n"][a:b], D.n[ga:gb]))
        uu = P.u_use[ga:gb]
        fin = np.isfinite(uu)
        ok = ok and np.array_equal(g["u"][a:b][fin], uu[fin])
        ok = ok and np.allclose(g["var_e"][a:b], D.Ve[ga:gb], rtol=1e-9, atol=0)
        bad += not ok
    return {"n": n, "bad": bad}


def apply_task(job: tuple) -> dict:
    key, combo, which = job
    t0 = time.time()
    it = ITERS[key]
    edge_gate, cg, bs, pf, tau = combo
    P = Prep(it)
    D = P.D
    wl = B.load_whitelist()
    Cl = cluster(P, edge_gate, cg)
    g = fuse(P, Cl, bs)
    tt = track_table(P, Cl, g, wl)
    pa, pr, pfold = P.p["outer"]
    pmax_all, pw_all = track_p(Cl, pa, D.g_cnt)
    pmax_air, pw_air = track_p(Cl, pr, D.g_cnt)
    p_track = pmax_all if pf == "all" else pmax_air
    sel = p_track >= tau
    out = {"key": key, "combo": combo, "which": which, "hash": members_hash(Cl), "stats": Cl.stats,
           "p_fold": pfold, "n_tracks": Cl.ntracks, "n_rows_all": int(g["track"].size),
           "n_bridged_rows": int(g["bridged"].sum()), "members_per_track": float(np.mean(
               np.bincount(Cl.mem_track, minlength=Cl.ntracks))) if Cl.ntracks else 0.0,
           "max_members": int(np.bincount(Cl.mem_track).max()) if Cl.ntracks else 0,
           "n_rej_steps": int((g["n_rej"] > 0).sum()), "n_resid_out": g["n_resid_out"], "n_ge3": g["n_ge3"],
           "multi_sensor_tracks": int((tt["n_sensors"] >= 2).sum()), "files": {}}
    # testing diagnostics
    lab = None
    if it.has_groundtruth:
        tr_all = T.load_truth(it, "all")
        lab = testing_labels(P, Cl, g, tr_all)
        out["diag"] = diagnostics(P, Cl, g, lab, sel)
    # outputs
    slug = C.iter_slug(key)
    OUT.mkdir(parents=True, exist_ok=True)
    cols = {"track": tt["track"], "track_id": np.array([TR.track_id(it, f"{key}/track{t}")
                                                        for t in range(Cl.ntracks)], dtype=object)}
    for c in ("n_members", "n_clusters", "n_sensors", "sensor_ids", "first_step", "last_step", "n_steps",
              "n_bridged", "duration_s"):
        cols[c] = tt[c]
    cols.update(p_max_all=pmax_all, p_max_air=pmax_air, p_wmean_all=pw_all, p_wmean_air=pw_air)
    for c in ("serial", "model", "protocol", "cls_fixed_share", "speed_med_ms", "speed_p10_ms",
              "speed_p90_ms", "u_med_m"):
        cols[c] = tt[c]
    cols["reported_f0"] = sel.astype(np.int64)
    order = list(cols)
    if lab is not None:
        cols.update(drone_share=lab["drone_share"], label_drone=lab["label_drone"].astype(np.int64),
                    flight=lab["flight"], purity=lab["purity"])
        order += ["drone_share", "label_drone", "flight", "purity"]
    C.write_columns(OUT / f"{slug}__tracks.csv", cols, order)
    C.write_columns(OUT / f"{slug}__members.csv", {"track": Cl.mem_track, "cluster": Cl.mem_cluster,
                                                   "seg_id": D.seg_id[Cl.mem_seg]},
                    ["track", "cluster", "seg_id"])
    np.savez_compressed(OUT / f"{slug}__grid.npz", **{c: g[c] for c in (
        "track", "step", "e", "n", "u", "var_e", "var_n", "var_u", "ve", "vn", "n_used", "n_rej",
        "u_src", "bridged")})
    # submissions
    methods = [(METHOD_F0, sel)]
    if lab is not None:
        methods.append((METHOD_O3, lab["label_drone"]))
    for mname, msel in methods:
        rows = build_rows(it, g, tt, p_track, msel)
        rm = np.flatnonzero(msel[g["track"]])
        path = C.submission_path(mname, it)
        rt = B._file_roundtrip(it, rows, path, g["e"][rm], g["n"][rm], g["u"][rm])
        back = rt["back"]
        tb = np.asarray(back["track_id"]).astype(str)
        uniq = True
        if len(tb):
            code = np.unique(tb, return_inverse=True)[1]
            pair = back["step"].astype(np.int64) * (int(code.max()) + 1) + code
            uniq = len(np.unique(pair)) == len(pair)
        sens = set()
        for t in np.flatnonzero(msel).tolist():
            sens |= set(tt["sensor_ids"][t].split("|"))
        f = {"path": str(path.relative_to(C.REPO)), "rows": len(rows["step"]), "tracks": int(msel.sum()),
             "problems": rt["problems"], "rt_err_m": rt["rt_err_m"], "rt_labels_ok": rt["rt_labels_ok"],
             "n_back": rt["n_back"], "unique": uniq, "sensors": sorted(sens),
             "friendly_tracks": int(sum(1 for t in np.flatnonzero(msel)
                                        if tt["lab"]["identity"][t] == "Friendly")),
             "serial_tracks": int(sum(1 for t in np.flatnonzero(msel) if tt["serial"][t])),
             "model_tracks": int(sum(1 for t in np.flatnonzero(msel) if tt["model"][t] != "Unknown"))}
        if it.has_groundtruth:
            f["mem_scores"] = _score_pair(it, back)
        out["files"][mname] = f
        del rows, back
    # self-check data
    out["c1"] = check_constraints(P, Cl)
    out["c2"] = check_fused_python(P, Cl, g, np.random.default_rng(SEED + len(key)))
    out["c3"] = check_singletons(P, Cl, g)
    out["seconds"] = time.time() - t0
    return out


def diagnostics(P: Prep, Cl: Clusters, g: dict, lab: dict, sel: np.ndarray) -> dict:
    """Purity, fragmentation and missed-truth causes (testing, ground-truth labels)."""
    it, D = P.it, P.D
    out = {"n_label_drone": int(lab["label_drone"].sum())}
    w = np.bincount(g["track"], minlength=Cl.ntracks).astype(float)
    m = lab["label_drone"]
    out["purity_wmean"] = float(np.sum(lab["purity"][m] * w[m]) / w[m].sum()) if m.any() else float("nan")
    for mode in TRUTH_MODES:
        tr = SC.cached_truth(it, mode)
        nf = max(len(tr.flights), 1)
        fidx = {f["flight_key"]: i for i, f in enumerate(tr.flights)}
        tkey = tr.step.astype(np.int64) * nf + tr.flight.astype(np.int64)
        R = tkey.size
        # drone-labelled deployable segment of the flight at that step
        sm = P.dep & D.is_drone
        gi = np.flatnonzero(sm[D.grow])
        sf = np.array([fidx.get(f, -1) for f in D.flight], dtype=np.int64)
        gf = sf[D.grow[gi]]
        ok = gf >= 0
        skey = D.step[gi][ok] * nf + gf[ok]
        cov_seg = np.isin(tkey, skey)
        seg_count = np.zeros(R)
        if skey.size and R:
            pos = np.searchsorted(tkey, skey)
            posc = np.minimum(pos, R - 1)
            h = tkey[posc] == skey
            np.add.at(seg_count, posc[h], 1)
        tf = lab["flight_idx"]
        rr = np.flatnonzero(tf[g["track"]] >= 0)
        rkey = g["step"][rr] * nf + tf[g["track"][rr]]
        has_trk = np.zeros(R, bool)
        trk_count = np.zeros(R)
        dmin = np.full(R, np.inf)
        dmin_rep = np.full(R, np.inf)
        if rr.size and R:
            pos = np.searchsorted(tkey, rkey)
            posc = np.minimum(pos, R - 1)
            h = tkey[posc] == rkey
            p_, r_ = posc[h], rr[h]
            has_trk[p_] = True
            np.add.at(trk_count, p_, 1)
            d3 = np.sqrt((g["e"][r_] - tr.e[p_]) ** 2 + (g["n"][r_] - tr.n[p_]) ** 2 + (g["u"][r_] - tr.u[p_]) ** 2)
            np.minimum.at(dmin, p_, d3)
            rep = sel[g["track"][r_]]
            np.minimum.at(dmin_rep, p_[rep], d3[rep])
        near, near_rep = dmin <= PURITY_M, dmin_rep <= PURITY_M
        n = max(R, 1)
        out[mode] = {
            "n_truth": R,
            "A_no_segment": float(np.sum(~cov_seg & ~has_trk) / n),
            "B_absorbed": float(np.sum(cov_seg & ~has_trk) / n),
            "C_far": float(np.sum(has_trk & ~near) / n),
            "D_not_reported": float(np.sum(near & ~near_rep) / n),
            "E_reported_near": float(np.sum(near_rep) / n),
            "frag_tracks": float(trk_count[trk_count > 0].mean()) if (trk_count > 0).any() else float("nan"),
            "frag_segments": float(seg_count[seg_count > 0].mean()) if (seg_count > 0).any() else float("nan"),
        }
    return out


# --------------------------------------------------------------------------- report


def read_experiments(methods) -> dict:
    """Latest per-iteration rows of `methods` in experiments.csv: {(method, mode): {iter: row}}."""
    out: dict = {}
    if not SC.EXPERIMENTS.exists():
        return out
    with SC.EXPERIMENTS.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["method"] in methods and r["iter_key"].startswith("testing/S"):
                out.setdefault((r["method"], r["truth_mode"]), {})[r["iter_key"]] = r
    return out


def write_report(path: Path, tune: dict, applied: dict, choices: dict, final_choice: tuple,
                 headline: dict, secs: float, phase_s: dict) -> None:
    L: list[str] = []
    L.append("# Fusion tracker F0_fuse: cross-sensor de-duplication, LOIO-tuned\n")
    L.append(f"Generated by `tools/fuse.py` in {secs / 60:.1f} min (tuning {phase_s['tune'] / 60:.1f} min, "
             f"apply + files {phase_s['apply'] / 60:.1f} min). All scores are the local PROXY of "
             f"`tools/score.py` (primary configuration `{SC.PRIMARY}`: GOSPA cut-off c = 50 m, p = 2, 3-D) "
             "on the 6 testing iterations. F0 is held out: the parameters applied to testing iteration k "
             "were chosen on the other 5 iterations with existence probabilities that never saw k. "
             "Caveats: the sensor noise tables, the sensor_tracks variances and height offsets were fitted "
             "on all 6 testing iterations (inherited, in-sample); VKYY / QTVS relay the Dronetag feed in "
             "testing (optimistic there); the honest-variance table of iteration j excludes j only (not "
             "the held-out k), so k's noise statistics enter the clustering of the other iterations "
             "during k's tuning (a small, documented second-order leak).\n")
    L.append("## Plain-language glossary\n")
    for t, d in (
        ("segment", "one sensor's own track (one object id), cut at 30 s gaps, smoothed on the 100 ms grid."),
        ("fused track", "one or more segments judged to be the same object, reported under one track_id."),
        ("cluster", "segments merged because they were seen at the same time at the same place."),
        ("continuation / chain", "a cluster that starts shortly after another ends, where the first one's "
         "position projected forward at constant speed lands on the second; a chain of clusters is one track."),
        ("normalised squared distance", "squared distance divided by the expected variance; about 2 on "
         "average for two reports of the same object, large for different objects."),
        ("gate", "a pass / fail threshold on a distance. 9.21 is the chi-square(2) 99 % gate: "
         "99 % of same-object pairs fall below it if the variances are right."),
        ("edge gate", "the rule that two simultaneous segments are the same object: median normalised "
         "squared distance < nd_gate AND share of common steps within 3 sigma >= share_gate. 'hand' = "
         "(9.21, 0.5), 'loose' = (25, 0.25); both are loosened together because the share rule alone "
         "already keeps the median below about 9."),
        ("honest variance", "a sensor's error variance including its slow bias (pair_features.sigma_table), "
         "not only the smoother's own (too small) variance."),
        ("inverse-variance mean", "an average where each sensor counts in proportion to 1 / its variance, "
         "so precise sensors dominate; the fused variance is 1 / (sum of 1/variance)."),
        ("union-find", "a bookkeeping structure that merges groups pair by pair."),
        ("complete linkage", "two groups merge only if EVERY simultaneous member pair across them agrees, "
         "not just the closest pair (prevents chaining two drones together through a third segment)."),
        ("outlier rejection", "at a step with >= 3 sensors, the one farthest from the mean of the others "
         f"(normalised squared distance > {OUT_ND}) is dropped; only that one, because a single outlier "
         "also pushes the good samples' distances up, and dropping all of them would empty the step."),
        ("bridging", "filling a short gap inside a track by straight-line interpolation."),
        ("existence probability p", "tools/existence.py: probability that a segment is a real drone; "
         "p_track is the maximum over the track's segments."),
        ("TAU", "the threshold on p_track above which F0 reports a track."),
        ("LOIO", "leave-one-iteration-out: every number for iteration k comes from choices made without k."),
        ("p_inner", "existence probabilities of iteration j from a model that saw neither j nor the held-out "
         "k, on k's calibration scale: used to tune TAU for k without leaking k."),
        ("AUC", "area under the ROC curve: probability that a random drone segment ranks above a random "
         "clutter segment (existence model ~0.96)."),
        ("calibration", "whether p = 0.3 really means 30 % of such segments are drones."),
        ("GOSPA", "the track metric: per 100 ms step truth drones are matched to reported rows; matched "
         "pairs cost their squared distance, unmatched truths (missed) and rows (false) cost c^2/2 each."),
        ("track score", "1 - GOSPA cost / worst cost (0..1); 0.4 of the combined score."),
        ("recall", "share of truth rows whose matched row carries the right label (class, identity, "
         "metadata); a missed drone counts as wrong."),
        ("combined", "0.4 track + 0.2 classification + 0.1 identity + 0.3 metadata."),
        ("truth mode all / airborne", "'all' keeps drones parked on the ground (the official truth very "
         "likely does), 'airborne' only flying ones."),
        ("purity", "share of a drone-labelled track's rows within 50 m (3-D) of its own drone."),
        ("fragmentation", "mean number of tracks (or segments) of a drone at the same step, over truth rows "
         "where it has at least one: 1.0 = no duplicates."),
        ("cost shares", "the GOSPA cost split into localisation (matched distance), missed and false rows."),
        ("oracle", "uses ground-truth labels, never deployable. O3 is a reference, not a strict upper "
         "bound: it drops mixed tracks whose drone share is < 0.5, so F0 can beat it on an iteration."),
    ):
        L.append(f"* **{t}**: {d}")
    L.append("")
    L.append("## Constants and the tuning grid\n")
    L.append(f"MIN_OVERLAP_STEPS {MIN_OVERLAP_STEPS} (2 s), MAX_MIN_DIST_M {MAX_MIN_DIST_M:.0f}, EDGE_GATES "
             f"{EDGE_GATES}, CONFLICT_S {CONFLICT_S}, CONT_ND {CONT_ND}, CONT_MAX_M {CONT_MAX_M:.0f}, OUT_ND "
             f"{OUT_ND}, q = {Q_H} (horizontal) / {Q_U} (vertical) m^2/s^3. Grid: " +
             ", ".join(f"{k} {list(v)}" for k, v in PARAMS.items()) +
             f" = {len(all_combos())} combinations, each scored on every testing iteration with 6 "
             "existence-probability sources (outer + 5 inner folds) in both truth modes; identical "
             "track selections are scored once.\n")

    L.append("## Chosen parameters per fold\n")
    rows = []
    for k in TESTING:
        cb, v = choices[k]
        rows.append([short(k)] + [str(x) for x in cb] + [_f(v, 4)])
    rows.append(["evaluation (all 6)"] + [str(x) for x in final_choice[0]] + [_f(final_choice[1], 4)])
    L += md_table(["held-out", "EDGE_GATE", "CONT_GAP_S", "BRIDGE_S", "P_FIELD", "TAU",
                   "objective (inner mean of (all+air)/2)"], rows)
    L.append("")

    L.append("## Segments in, tracks out (applied geometry of each iteration)\n")
    rows = []
    for k in TESTING + EVALUATION:
        a = applied[k]
        s = a["stats"]
        f0 = a["files"][METHOD_F0]
        rows.append([short(k), s["n_segments"], s["n_pass"], s["n_conflict"], s["n_blocked"], s["n_clusters"],
                     s["n_cont_links"], a["n_tracks"], f"{a['members_per_track']:.2f}", a["max_members"],
                     a["multi_sensor_tracks"], f"{a['n_rows_all']:,}", f"{a['n_bridged_rows']:,}",
                     f0["tracks"], f"{f0['rows']:,}"])
    L += md_table(["iteration", "deployable segments", "passing edges", "conflict pairs", "merges blocked",
                   "clusters", "continuation links", "tracks", "members / track", "max members",
                   "multi-sensor tracks", "fused rows", "bridged rows", "F0 tracks", "F0 rows"], rows)
    L.append("")
    L.append("Outlier rejection (one worst sample per step, see the glossary): steps with >= 3 member "
             "samples / steps with a sample rejected / steps that still hold a sample above OUT_ND after "
             "the drop (n_resid_out, left in the mean): " +
             "; ".join(f"{short(k)} {applied[k]['n_ge3']:,} / {applied[k]['n_rej_steps']:,} / "
                       f"{applied[k]['n_resid_out']:,}" for k in TESTING + EVALUATION) + ".\n")
    L.append("## Purity and fragmentation (testing, ground-truth labels)\n")
    rows = []
    for k in TESTING:
        d = applied[k]["diag"]
        rows.append([short(k), d["n_label_drone"], _f(d["purity_wmean"]),
                     _f(d["all"]["frag_segments"], 2), _f(d["all"]["frag_tracks"], 2),
                     _f(d["airborne"]["frag_segments"], 2), _f(d["airborne"]["frag_tracks"], 2)])
    L += md_table(["iteration", "drone-labelled tracks", "purity (row-weighted)",
                   "segments per drone-step (all)", "tracks per drone-step (all)",
                   "segments per drone-step (air)", "tracks per drone-step (air)"], rows)
    L.append("")

    ex = read_experiments({"B2_allpos", "O1_oracle_exist", "O2_oracle_dedup"})
    for mode in TRUTH_MODES:
        L.append(f"## F0 held-out scores, truth mode '{mode}'\n")
        rows = []
        f0 = {r["iter_key"]: r for r in headline[(METHOD_F0, mode)]}
        o3 = {r["iter_key"]: r for r in headline[(METHOD_O3, mode)]}
        acc = {n: [] for n in ("F0c", "F0t", "O3c", "O3t", "B2c", "B2t", "O1c", "O1t", "O2c", "O2t")}
        for k in TESTING:
            a, o = f0[k]["parts"], o3[k]["parts"]
            vals = [a["combined"], a["track"], a["classification"], a["identity"], a["metadata"],
                    o["combined"], o["track"]]
            acc["F0c"].append(a["combined"]); acc["F0t"].append(a["track"])
            acc["O3c"].append(o["combined"]); acc["O3t"].append(o["track"])
            for nm, tag in (("B2_allpos", "B2"), ("O1_oracle_exist", "O1"), ("O2_oracle_dedup", "O2")):
                r = ex.get((nm, mode), {}).get(k)
                c_, t_ = (float(r["combined"]), float(r["track"])) if r else (float("nan"),) * 2
                acc[tag + "c"].append(c_); acc[tag + "t"].append(t_)
                vals += [c_, t_]
            pr = f0[k][SC.PRIMARY]
            vals += [pr["missed_rate"], pr["false_rate"]]
            rows.append([short(k)] + [_f(v) for v in vals] + [f0[k]["n_tracks"], f"{f0[k]['n_est_rows']:,}"])
        mean = lambda x: float(np.nanmean(x)) if len(x) else float("nan")  # noqa: E731
        m0 = SC.summarize(headline[(METHOD_F0, mode)])
        rows.append(["**MEAN**", _f(m0["combined"]), _f(m0["track"]), _f(m0["classification"]),
                     _f(m0["identity"]), _f(m0["metadata"]), _f(mean(acc["O3c"])), _f(mean(acc["O3t"])),
                     _f(mean(acc["B2c"])), _f(mean(acc["B2t"])), _f(mean(acc["O1c"])), _f(mean(acc["O1t"])),
                     _f(mean(acc["O2c"])), _f(mean(acc["O2t"])), "", "", "", ""])
        L += md_table(["iteration", "F0 combined", "F0 track", "F0 class.", "F0 ident.", "F0 metadata",
                       "O3 combined", "O3 track", "B2 combined", "B2 track", "O1 combined", "O1 track",
                       "O2 combined", "O2 track", "F0 missed", "F0 false", "F0 tracks", "F0 rows"], rows)
        L.append("")
        beat = [short(k) for k in TESTING if f0[k]["parts"]["combined"] > o3[k]["parts"]["combined"]]
        L.append("O3 is an oracle reference, not a strict upper bound (it drops mixed tracks with drone "
                 "share < 0.5). Iterations where F0's combined score exceeds O3's: "
                 + (", ".join(beat) if beat else "none") + ".\n")
        cs = [np.mean([r[SC.PRIMARY][c] for r in headline[(METHOD_F0, mode)]])
              for c in ("cost_share_loc", "cost_share_missed", "cost_share_false")]
        cs3 = [np.mean([r[SC.PRIMARY][c] for r in headline[(METHOD_O3, mode)]])
               for c in ("cost_share_loc", "cost_share_missed", "cost_share_false")]
        L.append(f"GOSPA cost shares (mean over iterations): F0 {100 * cs[0]:.0f} % localisation, "
                 f"{100 * cs[1]:.0f} % missed, {100 * cs[2]:.0f} % false; O3 {100 * cs3[0]:.0f} % / "
                 f"{100 * cs3[1]:.0f} % / {100 * cs3[2]:.0f} %. B2 / O1 / O2 are the latest blocks of "
                 "experiments.csv (tools/baselines.py). 'ident.' in testing measures RED coverage only (no "
                 "testing drone carries a whitelist serial).\n")
        L.append(f"Why truth rows are missed, truth mode '{mode}' (share of truth rows, mean over testing "
                 "iterations; uses ground-truth labels, diagnostic only). Each truth row falls in exactly "
                 "one class:\n")
        keys = [("A_no_segment", "no drone-labelled deployable segment of that drone at that step and no "
                 "track of it (true coverage gap)"),
                ("B_absorbed", "a labelled segment exists, but no track labelled with that drone has a row "
                 "there (the segment sits in a track labelled clutter or another drone: merge damage, or "
                 "too small a share of a mixed track)"),
                ("C_far", "a track of that drone has a row there, but none within 50 m (3-D): localisation / "
                 "height / fusion error"),
                ("D_not_reported", "a track of that drone is within 50 m, but F0 did not report it "
                 "(existence threshold)"),
                ("E_reported_near", "F0 reports a row of that drone within 50 m (reachable)")]
        rows = [[d, _f(np.mean([applied[k]["diag"][mode][c] for k in TESTING]))] for c, d in keys]
        L += md_table(["cause", "share"], rows)
        mr = np.mean([r[SC.PRIMARY]["missed_rate"] for r in headline[(METHOD_F0, mode)]])
        L.append(f"\nF0's GOSPA missed rate is {mr:.3f}, i.e. {1 - mr:.3f} of the truth rows are matched; "
                 "the reachable share minus that is lost to the assignment (a reported row matched to "
                 "another drone, or a row of a track labelled with another drone).\n")

    L.append("## Tuning: what the parameters do (outer p, mean over the 6 iterations)\n")
    best = {}
    for r in tune:
        if r["k"] != "outer":
            continue
        best.setdefault(combo_of(r), []).append((r["combined_all"] + r["combined_airborne"]) / 2)
    for cname in COMBO_KEYS:
        i = COMBO_KEYS.index(cname)
        rows = []
        for v in PARAMS[cname]:
            vals = [np.mean(x) for cb, x in best.items() if cb[i] == v and len(x) == len(TESTING)]
            rows.append([str(v), _f(max(vals), 4) if vals else "n/a"])
        L.append(f"Best mean objective with {cname} fixed (in-sample over all 6, outer p; for orientation "
                 "only):\n")
        L += md_table([cname, "best objective"], rows)
        L.append("")

    L.append("## What F0 reports in evaluation\n")
    rows = []
    for k in EVALUATION:
        f = applied[k]["files"][METHOD_F0]
        rows.append([short(k), f["tracks"], f"{f['rows']:,}", f["friendly_tracks"], f["serial_tracks"],
                     f["model_tracks"], ", ".join(f["sensors"])])
    L += md_table(["iteration", "tracks", "rows", "Friendly tracks", "tracks with a serial",
                   "tracks with a model", "sensors used"], rows)
    L.append("")
    L.append("Submissions are in the anonymized longitude frame (de-anonymising before upload is the "
             "owner's step). Files: `data/derived/submissions/F0_fuse/` (9 iterations) and "
             "`data/derived/submissions/O3_fuse_oracle/` (6 testing, ORACLE).\n")
    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=6, help="processes for the tuning phase")
    ap.add_argument("--final-workers", type=int, default=4, help="processes for the apply phase")
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("fork")

    # ---- phase 1: score table
    sizes = {k: int(C.read_columns(TRACKS_DIR / f"{C.iter_slug(k)}__tracks.csv", numeric={"n_grid"},
                                   usecols=["n_grid"])["n_grid"].sum()) for k in TESTING}
    jobs = sorted(((k, nd) for k in TESTING for nd in PARAMS["edge_gate"]), key=lambda x: (-sizes[x[0]], x))
    log(f"phase 1: {len(jobs)} tuning jobs ({len(all_combos())} combinations x 6 p sources x 2 modes "
        f"per testing iteration), {args.workers} workers")
    records, hashes, tstats = [], {}, {}
    with ctx.Pool(args.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(tune_task, jobs), 1):
            records += r["records"]
            for cg, h in r["hash"].items():
                hashes[(r["key"], r["edge_gate"], cg)] = h
                tstats[(r["key"], r["edge_gate"], cg)] = r["stats"][cg]
            log(f"  [{n}/{len(jobs)}] {r['key']} edge_gate {r['edge_gate']}: {len(r['records'])} records "
                f"{r['seconds']:.0f}s")
    records.sort(key=lambda r: (r["k"], r["j"]) + combo_of(r))
    tcols = ["k", "j", "edge_gate", "cont_gap_s", "bridge_s", "p_field", "tau", "combined_all", "track_all",
             "combined_airborne", "track_airborne", "n_tracks", "n_rows", "p_fold"]
    C.write_columns(TUNING_CSV, {c: (np.array([r[c] for r in records], dtype=float)
                                     if isinstance(records[0][c], float) else [r[c] for r in records])
                                 for c in tcols}, tcols)
    t_tune = time.time() - t0

    choices, used_by = {}, {}
    for k in TESTING:
        cb, v, used = choose(records, k, [j for j in TESTING if j != k])
        choices[k], used_by[k] = (cb, v), used
    fcb, fv, fused_used = choose(records, "outer", TESTING)
    final_choice = (fcb, fv)
    for k in TESTING:
        log(f"  fold {k}: {dict(zip(COMBO_KEYS, choices[k][0]))} objective {choices[k][1]:.4f}")
    log(f"  evaluation: {dict(zip(COMBO_KEYS, fcb))} objective {fv:.4f}")

    # ---- phase 2: apply, write outputs + submissions
    t1 = time.time()
    ajobs = [(k, choices[k][0], "fold") for k in TESTING] + [(k, fcb, "final") for k in EVALUATION]
    ajobs.sort(key=lambda j: -sizes.get(j[0], 10 ** 6))
    applied = {}
    with ctx.Pool(args.final_workers) as pool:
        for n, r in enumerate(pool.imap_unordered(apply_task, ajobs), 1):
            applied[r["key"]] = r
            f = r["files"][METHOD_F0]
            log(f"  [{n}/{len(ajobs)}] {r['key']}: {r['n_tracks']} tracks, F0 {f['tracks']} tracks / "
                f"{f['rows']:,} rows, problems {len(f['problems'])}, {r['seconds']:.0f}s")
    t_apply = time.time() - t1

    # ---- headline scores (files), logged
    headline = {}
    for mname, desc in ((METHOD_F0, "fuse.py F0_fuse: fused tracks with p_track >= TAU, LOIO-tuned params "
                         "per fold (held-out), deployable sensors only"),
                        (METHOD_O3, "fuse.py O3_fuse_oracle ORACLE: the same fused tracks (fold geometry) "
                         "reported iff GT label_drone")):
        for mode in TRUTH_MODES:
            headline[(mname, mode)] = SC.score_method(mname, mode, iters=[ITERS[k] for k in TESTING],
                                                      log=True, note=desc)
    secs = time.time() - t0
    write_report(REPORT, records, applied, choices, final_choice, headline, secs,
                 {"tune": t_tune, "apply": t_apply})
    log(f"wrote {REPORT.relative_to(C.REPO)}")

    # ---- self-checks
    checks: list[tuple[str, bool, str]] = []
    c1 = {k: a["c1"] for k, a in applied.items()}
    bad1 = {k: (v["same_sensor_bad"], v["edge_bad"], v["examples"][:2]) for k, v in c1.items()
            if v["same_sensor_bad"] or v["edge_bad"]}
    checks.append((f"clustering constraints re-checked from the raw grid on {sum(v['pairs_checked'] for v in c1.values()):,} "
                   f"member pairs of all 9 iterations ({sum(v['sim_pairs'] for v in c1.values()):,} simultaneous; "
                   f"{sum(v['cross_cluster_sim'] for v in c1.values())} across chain clusters): no same-sensor "
                   f"overlap > {CONFLICT_S} s, every simultaneous pair of a cluster passes the edge rule",
                   not bad1, str(bad1)[:400]))
    c2 = [a["c2"] for a in applied.values()]
    n2 = sum(x["n"] + x["n_bridged"] for x in c2)
    e2 = max(x["max_err"] for x in c2)
    v2 = max(x["var_rel"] for x in c2)
    w2 = max(x["vel_err"] for x in c2)
    checks.append((f"fused state = plain-python recomputation on {n2} random (track, step) rows "
                   f"({sum(x['n_bridged'] for x in c2)} bridged, from their recomputed end rows; "
                   f"{sum(x['n_multi'] for x in c2)} multi-member, {sum(x['n_rej'] for x in c2)} with an "
                   f"outlier rejected): max abs err e/n/u {e2:.2e} m < 1e-6, ve/vn {w2:.2e} m/s < 1e-6, "
                   f"var_e/var_n/var_u relative err {v2:.1e} < 1e-9",
                   n2 >= 300 and e2 < 1e-6 and w2 < 1e-6 and v2 < 1e-9, ""))
    c3 = [a["c3"] for a in applied.values()]
    checks.append((f"{sum(x['n'] for x in c3):,} single-member unbridged tracks reproduce their segment's grid "
                   "samples exactly (e, n, in-range u; var = honest V)",
                   sum(x["n"] for x in c3) > 0 and all(x["bad"] == 0 for x in c3),
                   str([x for x in c3 if x["bad"]])[:200]))
    files = [(k, m, f) for k, a in applied.items() for m, f in a["files"].items()]
    bad4 = [(k, m, f["problems"][:2], f["rt_err_m"], f["unique"], f["rt_labels_ok"], f["n_back"], f["rows"])
            for k, m, f in files if f["problems"] or not f["unique"] or not (f["rt_err_m"] < RT_TOL_M)
            or f["rt_labels_ok"] is not True or f["n_back"] != f["rows"]]
    # in-memory tuning score of the applied combo vs the written file's score
    diffs = []
    rec_idx = {(r["k"], r["j"], combo_of(r)): r for r in records}
    for k in TESTING:
        r = rec_idx[("outer", k, choices[k][0])]
        ms_ = applied[k]["files"][METHOD_F0]["mem_scores"]
        diffs.append(max(abs(r["combined_all"] - ms_[0]), abs(r["combined_airborne"] - ms_[2])))
        for mode, hi in zip(TRUTH_MODES, (0, 2)):
            hr = next(x for x in headline[(METHOD_F0, mode)] if x["iter_key"] == k)
            diffs.append(abs(hr["parts"]["combined"] - ms_[hi]))
    n_ev_o3 = [k for k in EVALUATION if C.submission_path(METHOD_O3, ITERS[k]).exists()]
    checks.append((f"{len(files)} files: (step, track_id) unique, validate ([]), round trip e/n/u < {RT_TOL_M} m "
                   f"(max {max(f['rt_err_m'] for _, _, f in files):.1e}), labels / ids identical; no O3 "
                   f"evaluation file; tuning-table score of the applied combo = file score (max diff "
                   f"{max(diffs):.1e} < 1e-4)",
                   not bad4 and not n_ev_o3 and max(diffs) < 1e-4, str(bad4)[:300] + str(n_ev_o3)))
    # leakage guard
    leak = []
    for k in TESTING:
        js = {r["j"] for r in used_by[k]}
        if k in js or js != set(TESTING) - {k}:
            leak.append(f"{k}: objective used iterations {sorted(js)}")
        if any(r["k"] != k or r["p_fold"] != k for r in used_by[k]):
            leak.append(f"{k}: a score used p not from p_inner outer_fold == k")
        if applied[k]["p_fold"] != k:
            leak.append(f"{k}: applied with p fold {applied[k]['p_fold']}")
        if tuple(applied[k]["combo"]) != tuple(choices[k][0]):
            leak.append(f"{k}: applied combo differs from the fold choice")
    for k in EVALUATION:
        if applied[k]["p_fold"] != "final" or tuple(applied[k]["combo"]) != tuple(fcb):
            leak.append(f"{k}: evaluation p fold {applied[k]['p_fold']} / combo {applied[k]['combo']}")
    if any(r["k"] != "outer" for r in fused_used) or {r["j"] for r in fused_used} != set(TESTING):
        leak.append("evaluation choice used non-outer scores")
    dep = C.deployable_sensor_ids()
    for k, a in applied.items():
        f = a["files"][METHOD_F0]
        if not set(f["sensors"]) <= dep:
            leak.append(f"{k}: F0 uses non-deployable sensors {sorted(set(f['sensors']) - dep)}")
        if a["c1"]["non_deployable"]:
            leak.append(f"{k}: {a['c1']['non_deployable']} non-deployable member segments")
    # independent recomputation of the choices from tuning.csv as written
    tb = C.read_columns(TUNING_CSV, numeric={"cont_gap_s", "bridge_s", "tau", "combined_all",
                                             "combined_airborne"})
    for k in TESTING + ["outer"]:
        acc: dict = {}
        for i in range(len(tb["k"])):
            src = tb["k"][i]
            if (k != "outer" and (src != k or tb["j"][i] == k)) or (k == "outer" and src != "outer"):
                continue
            cb = (tb["edge_gate"][i], float(tb["cont_gap_s"][i]), float(tb["bridge_s"][i]),
                  tb["p_field"][i], float(tb["tau"][i]))
            acc.setdefault(cb, []).append((tb["combined_all"][i] + tb["combined_airborne"][i]) / 2)
        want = 5 if k != "outer" else 6
        bestv = max(sum(v) / len(v) for v in acc.values() if len(v) == want)
        mine = choices[k] if k != "outer" else final_choice
        cbm = tuple(float(x) if not isinstance(x, str) else x for x in mine[0])
        if abs(sum(acc[cbm]) / len(acc[cbm]) - bestv) > 1e-12:
            leak.append(f"{k}: choice is not the argmax of tuning.csv")
    checks.append(("leakage guard: fold k's choice uses only j != k scored with p_inner (outer_fold == k) and "
                   "is applied with k's outer p; evaluation uses the all-6 outer-p choice and p from the final "
                   "model; no testing-only sensor in any F0 track; choices re-derived from tuning.csv",
                   not leak, "; ".join(leak[:4])))
    det = [(k, hashes.get((k, a["combo"][0], a["combo"][1])) == a["hash"]) for k, a in applied.items()
           if k in TESTING]
    checks.append((f"determinism: rebuilding the clusters in another process gives identical member lists "
                   f"on {sum(ok for _, ok in det)}/{len(det)} testing iterations",
                   bool(det) and all(ok for _, ok in det), str([k for k, ok in det if not ok])))
    secs = time.time() - t0
    checks.append((f"runtime {secs / 60:.1f} min < {RUNTIME_LIMIT_S / 60:.0f} min", secs < RUNTIME_LIMIT_S, ""))
    n_fail = 0
    for i, (name, ok, detail) in enumerate(checks, 1):
        print(f"  [{'PASS' if ok else 'FAIL'}] ({i}) {name}" + ("" if ok or not detail else f"\n         {detail}"))
        n_fail += not ok
    for mode in TRUTH_MODES:
        s0, s3 = SC.summarize(headline[(METHOD_F0, mode)]), SC.summarize(headline[(METHOD_O3, mode)])
        log(f"  MEAN [{mode}] F0_fuse combined {s0['combined']:.4f} track {s0['track']:.4f} | "
            f"O3 combined {s3['combined']:.4f} track {s3['track']:.4f}")
    if n_fail:
        print("SELF-CHECK FAILED")
        return 1
    print("ALL SELF-CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
