#!/usr/bin/env python3
"""Leaderboard probes: one-factor variants of F0_fuse, so that each upload answers one unknown
scoring convention.

Why: the rules publish the four part weights but not the GOSPA parameters, whether parked
drones are in the official truth, or whether a correct 'Unknown' earns credit. Only the best
upload per data set counts, so a probe upload costs nothing; if a probe differs from F0 in ONE
factor, the leaderboard's move (up / down) against F0 reveals that factor's convention.

Probes (each identical to F0_fuse -- same tracks, same track_ids, same rows -- except ONE factor)
------------------------------------------------------------------------------------------------
P0_f0    F0_fuse itself: the rows of data/derived/submissions/F0_fuse/ read back and rewritten
         unchanged (self-check 4 compares every cell).
P_park   F0 PLUS rows for drones believed parked on the ground (P0's rows stay, unchanged).
         GT-free rules; constants in PARK_FIXED, tuned ones in PARK_GRID:
         * ground height g of the iteration = median over the deployable ellipsoid anchors
           (sensor_noise.anchor_mask: E datum, sensor_alt 50..120 m) of each sensor's median
           sensor_alt (self-check 5 compares it with the testing take-off heights).
         * anchors, never from a track with a VKYY / QTVS member (those UTM relays carry the
           Dronetag feed in testing, so a testing prediction built on them would be leakage);
           with rf_only = 1 only from tracks with a member of the RF / cyber family (sensor ids
           whose file type contains CYBER or PASSIVE-RF: CEQX, IKJX, ILYS, SMFS, STUG):
             end anchor: an F0-reported track's first (last) grid row when it is LOW (fused
               u - g <= H_LOW_M, or a fallback height: u_src 1-3) and SLOW (median horizontal
               speed over the END_WIN_S nearest rows <= V_SLOW_MS): the drone is about to take
               off (has just landed) there. Interval: the hold_s seconds before the first (after
               the last) row.
             stationary anchor (only if stat_p > 0): a track NOT reported by F0 with
               speed_p90_ms <= V_STAT_MS, n_steps >= MIN_STAT_STEPS, p_max_all >= stat_p
               (existence model: 'drone, airborne or on the ground') and low (u_med_m - g <=
               H_LOW_M or every row a fallback height): STUG home points, parked-drone returns.
               Interval: its own span widened by hold_s on both sides.
           Position: (e, n) of the anchoring row (stationary: median over the track's rows),
           u = g (on the ground; a fallback height can be 30 m off).
         * places: anchors within MERGE_M horizontally are one place (single linkage; position,
           labels = those of the lowest-index anchor; interval = union). An added row is dropped
           at a step where a P0 row lies within MERGE_M horizontally (already reported).
         * an added row carries its place's track labels (fuse.track_table's rules recomputed
           from tracks.csv), covariance 400 / 400 / 900 m^2, confidence = p_max_all, and the
           new track_id track_rows.track_id(it, '<it.key>/park<place>').
         Tuning (experiment design, not a deployed model): every PARK_GRID combination is scored
         under the TUNE_CFGS GOSPA configurations (score.CONFIGS), proxy labels. objective():
         first the number of (iteration, configuration, truth mode) cells with the wanted sign
         (combined delta vs P0 > 0 under 'all', < 0 under 'airborne'), then the mean over
         iterations of the worst cell margin; ties -> the first combination in PARK_GRID order.
         Testing iteration k gets the combination chosen on the other 5 with NESTED inputs
         (fold-k inputs, below); evaluation the one chosen on all 6 with the outer inputs.
           fold-k input of testing iteration j != k: F0's own fold-k rule applied to j with
             p that never saw k: fuse.py's fold-k parameters (edge_gate, cont_gap_s, bridge_s,
             p_field, tau; re-derived from fused/tuning.csv with fuse.choose) rebuild j's fused
             tracks (fuse.Prep / cluster / fuse / track_table), the existence p are
             existence/<slug j>__p_inner.csv rows with outer_fold == k (fuse.load_p), the
             reported set is p_track >= tau (P0 of j under fold k) and p_max_all (stationary
             anchors) comes from the same p. The anchors, P0 rows and deltas of j are built
             from these. (Second-order leak inherited from fuse.py's tuning: j's honest-variance
             table excludes j only.)
           outer input of j: the shipped F0 outputs (fused/ tracks.csv p_max_all / reported_f0
             from p.csv, F0_fuse submission): used for testing files, evaluation's choice and
             the report tables.
         Mirror P_air_only (drop F0 rows believed parked: low and slow, never UTM tracks;
         rf_only variant too) is scored on the outer inputs for comparison, not shipped.
P_meta   metadata best estimate instead of 'Unknown' on EVIDENCE-FREE tracks: tracks whose four
         F0 metadata cells are all 'Unknown', classified 'Rotary Wing UAS', identity 'Hostile'.
         Per iteration, from that iteration's own deployable cleaned sensor reports (no
         duplicates, no self-reports; GT-free inference on its own inputs):
           model    = most frequent mapped model (metadata_map.map_value, else
                      metadata_map.model_for_serial of the serial) over rows whose resolved
                      serial is not whitelisted (baselines.load_whitelist()), EXCLUDING the
                      serial-less rows (metadata_map.resolve_serial == '') whose model is the
                      model (metadata_map.SERIAL_MODEL) of a serial whitelisted in that
                      iteration -- the whitelisted serials the iteration's own reports carry
                      (on evaluation this equals its own Friendly list): such a row is as likely
                      the Blue drone without its serial as a Red copy (evaluation S2-I1 / S2-I3:
                      STUG 'Autel Drone' 154 / 784 rows with the Blue Autel serial, 143 / 353
                      without one). Ties by name; no remaining row -> the probe changes nothing
                      in that iteration.
           protocol = most frequent mapped protocol on the rows carrying that model; none ->
                      metadata_map.protocol_from_model(model); else 'Unknown'.
           serial   = the model's only serial in metadata_map.SERIAL_MODEL when it is not
                      whitelisted (a Hostile track cannot carry the Blue serial), else 'Unknown'.
           band     = 'Unknown': no dropdown protocol implies one dropdown band (OcuSync,
                      Skylink and Wi-Fi run at 2.4 and 5.8 GHz; Crossfire at 868 or 915 MHz;
                      'Video' / 'Digital video link type 2' have no fixed band).
         Fixed Wing tracks keep 'Unknown' (no fixed-wing model in the dropdown); tracks with any
         metadata evidence keep F0's values.
P_ident  identity 'Unknown' instead of 'Hostile' on every non-Friendly row (Friendly unchanged).
P_shift15  every P0 row's position moved by SHIFT_ENU_M = (+15, 0, 0) m: 15 m due EAST in the
         iteration's local ENU frame (C.iteration_frame): (e, n, u) -> (e + 15, n, u), written
         back as latitude / longitude / ellipsoid height (one Newton correction of the inverse,
         self-check 9: read-back ENU minus P0's read-back ENU equals the offset to SHIFT_TOL_M).
         Everything else (tracks, ids, labels, covariances, steps) is P0's. Purpose: bracket
         the unknown GOSPA cut-off c: the relative drop of the score against P0 is large when c
         is small (a 15 m error is close to or beyond c) and small when c is large. Predicted on
         testing for c in SHIFT_CS (p = 2, 3-D and 2-D, both truth modes); probes.md maps an
         observed relative drop to a c range.

Scoring hypotheses emulated locally (testing only)
--------------------------------------------------
* label rules (score_parts, never an edit of score.py): 'proxy' = tools/score.py (truth rows whose
  catalog value is 'Unknown' are not eligible); 'unk' = a correct 'Unknown' earns credit (every
  truth row eligible in the classification, identity and four metadata recalls); 'unk_red' =
  'unk' plus the official identity of every non-whitelisted (RED) drone is 'Unknown'. With
  'proxy' score_parts reproduces score.score_rows exactly (self-check 3).
* GOSPA: GRID = c in {10, 25, 50, 100} m x p in {1, 2} x 3-D / 2-D (16 configurations); the
  hypothesis tables use HYP_CONFIGS (c in {10, 50} m, p = 2, 3-D / 2-D); P0 and P_shift15 are
  also scored under SHIFT_CFGS (c in SHIFT_CS, p = 2, 3-D / 2-D).
* truth modes 'all' and 'airborne' (truth.load_truth).

Outputs
-------
Submissions at C.submission_path(<probe>, it) for the 9 iterations; every file validated.
data/derived/probes/probes.csv: probe, iter_key, truth_mode, label_rule, config, c, p, dims,
    combined, track, classification, identity, metadata, d_combined, d_track,
    d_classification, d_identity, d_metadata (d_ = probe minus P0 under the same hypothesis),
    n_rows, n_rows_p0. Testing only (evaluation has no truth).
data/derived/probes/park_tuning.csv: k ('outer' = shipped F0 inputs, or the held-out fold whose
    nested inputs were used), variant (P_park combination or mirror), iter_key, truth_mode,
    config, d_combined, d_track (proxy labels, fast path), n_added (rows added or removed),
    n_places, park_hit50 / air_hit50 / none50 (outer rows only: rows within 50 m 3-D of a
    parked / airborne truth position at that step, or neither; diagnostics only, never used to
    choose).
data/derived/probes/probes.md: plain-language report, decision table and upload checklist.

Rules: deterministic; numpy + openpyxl only; reads F0's fused outputs (data/derived/fused/) and
its submissions; never re-runs fuse.py's tuning or writes its outputs (it imports fuse to rebuild
the testing iterations' fold-k geometries; self-check 10 verifies that the rebuild reproduces
the shipped F0 tracks and rows of each testing iteration under its own fold); nothing is fitted
on evaluation; no absolute coordinate is printed or written to a report.

Run: python3 tools/probes.py [--workers N]   (numbered self-checks; exit status 1 on a failure)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines as B  # noqa: E402
import cuas_clean as Q  # noqa: E402
import cuas_common as C  # noqa: E402
import fuse as FU  # noqa: E402
import gospa as G  # noqa: E402
import metadata_map as MM  # noqa: E402
import score as SC  # noqa: E402
import sensor_noise as SN  # noqa: E402
import submission as S  # noqa: E402
import track_rows as TR  # noqa: E402
import truth as T  # noqa: E402

# cuas_common.iteration_frame re-reads every raw CSV per call (~1 s); memoise it per key (the
# Frame is a frozen dataclass, results are identical; fuse.py does the same).
_FRAMES: dict = {}
_frame_orig = C.iteration_frame


def _frame_cached(it: C.Iteration):
    if it.key not in _FRAMES:
        _FRAMES[it.key] = _frame_orig(it)
    return _FRAMES[it.key]


C.iteration_frame = _frame_cached

OUT = C.DERIVED / "probes"
FUSED = C.DERIVED / "fused"
F0 = "F0_fuse"
PROBES = ("P0_f0", "P_meta", "P_ident", "P_shift15", "P_park")   # = the upload order
TRUTH_MODES = ("all", "airborne")
LABEL_RULES = ("proxy", "unk", "unk_red")
GRID = tuple((f"c{c:g}_p{p:g}_{d}d", float(c), float(p), d)
             for c in (10, 25, 50, 100) for p in (1, 2) for d in (3, 2))
GRID_BY_NAME = {g[0]: g for g in GRID}
SHIFT_ENU_M = (15.0, 0.0, 0.0)              # P_shift15: 15 m due east, local ENU frame
SHIFT_TOL_M = 1e-6                          # self-check 9
SHIFT_CS = (5, 10, 15, 20, 25, 30, 40, 50, 75, 100)
SHIFT_CFGS = tuple((f"c{c:g}_p2_{d}d", float(c), 2.0, d) for c in SHIFT_CS for d in (3, 2))
CFG_BY_NAME = {**GRID_BY_NAME, **{g[0]: g for g in SHIFT_CFGS}}
SHIFT_EXTRA = tuple(g for g in SHIFT_CFGS if g[0] not in GRID_BY_NAME)
REPRO_TOL_M = 1e-6                          # self-check 10 (file round trip of the rebuilt rows)
HYP_CONFIGS = ("c50_p2_3d", "c50_p2_2d", "c10_p2_3d", "c10_p2_2d")
TUNE_CFGS = tuple(SC.CONFIGS)               # the 6 score.py configurations (PRIMARY first)
PRIMARY = SC.PRIMARY                        # c50_p2_3d
UTM = ("VKYY", "QTVS")
STEP_S = 0.1
PARK_FIXED = {"H_LOW_M": 15.0, "V_SLOW_MS": 2.0, "END_WIN_S": 3.0, "V_STAT_MS": 1.0,
              "MIN_STAT_STEPS": 50, "MERGE_M": 20.0}
PARK_GRID = {"hold_s": (300.0, 900.0, float("inf")), "rf_only": (1, 0), "stat_p": (0.0, 0.05)}
PARK_KEYS = tuple(PARK_GRID)
VAR_H, VAR_U = 400.0, 900.0
HIT_M = 50.0                                # diagnostics radius (park_tuning.csv)
GROUND_TOL_M = 5.0                          # self-check 5
INC_TOL = 1e-9                              # self-check 6
NUMERIC_TRACK = {"track", "n_members", "n_sensors", "first_step", "last_step", "n_steps",
                 "p_max_all", "p_max_air", "cls_fixed_share", "speed_med_ms", "speed_p10_ms",
                 "speed_p90_ms", "u_med_m", "reported_f0", "label_drone"}
EST_COLS = ("classification", "identity") + C.METADATA_COLUMNS
ITERS = {i.key: i for i in C.iterations()}
TESTING = [k for k in sorted(ITERS) if ITERS[k].has_groundtruth]
EVALUATION = [k for k in sorted(ITERS) if not ITERS[k].has_groundtruth]
RUNTIME_LIMIT_S = 30 * 60


def log(msg: str) -> None:
    print(msg, flush=True)


def short(key: str) -> str:
    return key.split("/")[-1]


def combo_name(combo: tuple) -> str:
    h, rf, sp = combo
    return f"hold{'inf' if h == float('inf') else f'{h:g}'}_rf{rf}_stat{sp:g}"


ALL_COMBOS = list(itertools.product(*(PARK_GRID[k] for k in PARK_KEYS)))


def rf_ids() -> frozenset[str]:
    """Sensor ids whose file type contains CYBER or PASSIVE-RF (from file names, GT-free)."""
    return frozenset(sf.sensor_id for it in C.iterations() for sf in C.sensor_files(it)
                     if "CYBER" in sf.sensor_type or "PASSIVE-RF" in sf.sensor_type)


# --------------------------------------------------------------------------- per-iteration data


class Ctx:
    """F0's fused tracks, grid and submission rows of one iteration: the shipped (outer) files,
    or `parts` = (tracks dict, grid dict, rows) rebuilt in memory (fold-k inputs, inner_ctx)."""

    def __init__(self, it: C.Iteration, ground: float, rf: frozenset[str], parts: tuple | None = None):
        self.it = it
        slug = C.iter_slug(it.key)
        if parts is None:
            self.t = C.read_columns(FUSED / f"{slug}__tracks.csv", numeric=NUMERIC_TRACK)
            z = np.load(FUSED / f"{slug}__grid.npz")
            self.g = {k: z[k] for k in ("track", "step", "e", "n", "u", "ve", "vn", "u_src")}
            del z
            self.rows = S.read_submission(it, C.submission_path(F0, it))
        else:
            self.t, self.g, self.rows = parts
        self.nt = len(self.t["track"])
        if not np.array_equal(np.asarray(self.t["track"]).astype(np.int64), np.arange(self.nt)):
            raise ValueError(f"{it.key}: tracks are not indexed 0..n-1")
        key_g = self.g["track"] * (1 << 32) + self.g["step"]
        if np.any(np.diff(key_g) <= 0):
            raise ValueError(f"{it.key}: fused grid not sorted by (track, step)")
        self.n_steps = len(C.template_grid(it)[1])
        self.rep = self.t["reported_f0"] == 1
        sens = [set(s.split("|")) for s in self.t["sensor_ids"]]
        self.utm = np.array([bool(s & set(UTM)) for s in sens])
        self.has_rf = np.array([bool(s & rf) for s in sens])
        self.ground = ground
        tk = self.g["track"]
        ar = np.arange(self.nt)
        self.gs = np.searchsorted(tk, ar)
        self.ge = np.searchsorted(tk, ar, side="right")          # exclusive
        tid = {s: i for i, s in enumerate(self.t["track_id"].tolist())}
        self.row_track = np.array([tid[s] for s in np.asarray(self.rows["track_id"]).tolist()],
                                  dtype=np.int64)
        self.step_b = np.asarray(self.rows["step"], dtype=np.int64)
        self.y_b = rows_enu(it, self.rows)
        # grid row of every F0 row
        key_r = self.row_track * (1 << 32) + self.step_b
        gi = np.searchsorted(key_g, key_r)
        gi = np.minimum(gi, len(key_g) - 1)
        if not np.array_equal(key_g[gi], key_r):
            raise ValueError(f"{it.key}: F0 rows not found in the fused grid")
        self.row_grid = gi


def ground_heights() -> dict[str, float]:
    """{iter_key: g}: median over the deployable ellipsoid anchors (sensor_noise.anchor_mask) of
    each anchor sensor's median sensor_alt. GT-free."""
    d = Q.load_detections(["iter_key", "sensor_id", "sensor_alt", "sensor_alt_datum",
                           "sensor_pos_valid"])
    am = SN.anchor_mask(d) & np.isin(d["sensor_id"], sorted(C.deployable_sensor_ids()))
    out = {}
    for it in C.iterations():
        m = am & (d["iter_key"] == it.key)
        per = [float(np.median(d["sensor_alt"][m & (d["sensor_id"] == s)]))
               for s in sorted(set(d["sensor_id"][m].tolist()))]
        out[it.key] = float(np.median(per)) if per else SN.AREA_H0_M
    return out


def meta_estimates(whitelist: set[str], exclude_blue_model: bool = True) -> dict[str, dict]:
    """Per iteration: the P_meta fill values from its own deployable cleaned reports.
    exclude_blue_model=False is the pre-2026-10-29 rule (kept only to report old vs new fills)."""
    d = Q.load_clean(["iter_key", "sensor_id", "model", "serial", "protocol"])
    m0 = np.isin(d["sensor_id"], sorted(C.deployable_sensor_ids())) & ~d["duplicate"] & ~d["self_report"]
    out = {}
    for key in sorted(ITERS):
        cnt: Counter = Counter()
        prot: dict[str, Counter] = {}
        excl: Counter = Counter()
        idx = np.flatnonzero(m0 & (d["iter_key"] == key)).tolist()
        sers = {i: MM.resolve_serial(d["serial"][i]) for i in idx}
        wl_here = sorted({s for s in sers.values() if s and s in whitelist})
        blue_models = {MM.SERIAL_MODEL[s] for s in wl_here if s in MM.SERIAL_MODEL} if exclude_blue_model else set()
        for i in idx:
            ser = sers[i]
            if ser in whitelist:
                continue
            mo = MM.map_value(MM.MODEL, d["model"][i]) or MM.model_for_serial(d["serial"][i])
            if not mo:
                continue
            if not ser and mo in blue_models:      # possibly the Blue drone without its serial
                excl[mo] += 1
                continue
            cnt[mo] += 1
            p = MM.map_value(MM.PROTOCOL, d["protocol"][i])
            if p:
                prot.setdefault(mo, Counter())[p] += 1
        est = {"model": "Unknown", "protocol": "Unknown", "serial": "Unknown", "band": "Unknown",
               "model_counts": dict(sorted(cnt.items())), "protocol_counts": {},
               "whitelisted_here": wl_here, "excluded_counts": dict(sorted(excl.items()))}
        if cnt:
            model = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            est["model"] = model
            pc = prot.get(model, Counter())
            est["protocol_counts"] = dict(sorted(pc.items()))
            if pc:
                est["protocol"] = sorted(pc.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            else:
                est["protocol"] = MM.protocol_from_model(model) or "Unknown"
            sers = [s for s, mo in MM.SERIAL_MODEL.items() if mo == model]
            if len(sers) == 1 and sers[0] not in whitelist:
                est["serial"] = sers[0]
        out[key] = est
    return out


# --------------------------------------------------------------------------- variant scorer


class TruthView:
    """Truth rows of one (iteration, truth mode) with the label arrays of every label rule."""

    def __init__(self, it: C.Iteration, mode: str):
        tr = SC.cached_truth(it, mode)
        self.it, self.mode = it, mode
        self.n_steps = len(tr.grid_t)
        self.step = np.asarray(tr.step, dtype=np.int64)
        self.x = np.column_stack([tr.e, tr.n, tr.u])
        self.airborne = np.asarray(tr.airborne, dtype=bool)
        fl = tr.flights

        def col(c: str) -> np.ndarray:
            return np.array([fl[f][c] for f in tr.flight], dtype=object)

        self.cls, self.ident = col("classification"), col("identity")
        self.meta = {c: col(c) for c in C.METADATA_COLUMNS}
        self.red = col("side") == "RED"

    def label_sets(self, rule: str) -> dict:
        """{part: (truth values, eligible mask)}; 'metadata' maps column -> (values, mask)."""
        if rule not in LABEL_RULES:
            raise ValueError(f"unknown label rule {rule!r}")
        if rule == "proxy":
            return {"classification": (self.cls, np.isin(self.cls, SC.KNOWN_CLASSES)),
                    "identity": (self.ident, np.isin(self.ident, ("Friendly", "Hostile"))),
                    "metadata": {c: (v, v != "Unknown") for c, v in self.meta.items()}}
        allm = np.ones(len(self.step), dtype=bool)
        ident = self.ident
        if rule == "unk_red":
            ident = np.where(self.red, "Unknown", self.ident).astype(object)
        return {"classification": (self.cls, allm), "identity": (ident, allm),
                "metadata": {c: (v, allm) for c, v in self.meta.items()}}


def rows_enu(it: C.Iteration, rows: dict) -> np.ndarray:
    if len(rows["step"]) == 0:
        return np.zeros((0, 3))
    e, n, u = C.iteration_frame(it).to_enu(np.asarray(rows["latitude"], float),
                                           np.asarray(rows["longitude"], float),
                                           np.asarray(rows["altitude_m"], float))
    return np.column_stack([e, n, u])


class Gosp:
    """GOSPA of one row set against one TruthView under one configuration (as score.score_rows)."""

    def __init__(self, tv: TruthView, step_y: np.ndarray, y: np.ndarray, cfg: tuple):
        name, c, p, dims = cfg
        self.cfg, self.half = cfg, c ** p / 2.0
        seq = G.gospa_sequence(tv.step, tv.x[:, :dims], step_y, y[:, :dims], tv.n_steps, c, p)
        self.cost_step = seq["cost"]
        self.cost = float(seq["cost"].sum())
        self.worst = float((self.half * (seq["n_truth"] + seq["n_est"])).sum())
        self.track = 1.0 - self.cost / self.worst if self.worst > 0 else 1.0
        self.assigned = np.full(len(tv.step), -1, dtype=np.int64)
        if len(seq["pairs"]):
            self.assigned[seq["pairs"][:, 1]] = seq["pairs"][:, 2]


def variant_gospa(tv: TruthView, base: Gosp, step_b: np.ndarray, y_b: np.ndarray,
                  keep: np.ndarray, step_x: np.ndarray, y_x: np.ndarray) -> tuple[float, np.ndarray]:
    """Track score and truth->row assignment of the variant rows [base rows where `keep`] +
    [extra rows], recomputing GOSPA only at the steps the change touches (the fast path of the
    tuning; shipped numbers are scored in full from the written files, self-check 6)."""
    name, c, p, dims = base.cfg
    A = np.unique(np.concatenate([step_x, step_b[~keep]]))
    vidx = np.cumsum(keep) - 1
    n_kept = int(keep.sum())
    a = base.assigned.copy()
    ok = a >= 0
    a[ok] = vidx[a[ok]]
    if A.size == 0:
        return base.track, a
    in_t = np.isin(tv.step, A)
    in_b = np.isin(step_b, A) & keep
    sub_y = np.concatenate([y_b[in_b], y_x])[:, :dims]
    sub_sy = np.searchsorted(A, np.concatenate([step_b[in_b], step_x]))
    seq = G.gospa_sequence(np.searchsorted(A, tv.step[in_t]), tv.x[in_t, :dims], sub_sy, sub_y,
                           len(A), c, p)
    cost = base.cost - float(base.cost_step[A].sum()) + float(seq["cost"].sum())
    worst = base.half * (len(tv.step) + n_kept + len(step_x))
    track = 1.0 - cost / worst if worst > 0 else 1.0
    t_rows = np.flatnonzero(in_t)
    a[t_rows] = -1
    if len(seq["pairs"]):
        b_rows = np.flatnonzero(in_b)
        j = seq["pairs"][:, 2]
        nb = len(b_rows)
        est = np.empty(len(j), dtype=np.int64)
        isb = j < nb
        est[isb] = vidx[b_rows[j[isb]]]
        est[~isb] = n_kept + j[~isb] - nb
        a[t_rows[seq["pairs"][:, 1]]] = est
    return track, a


def score_parts(tv: TruthView, track: float, assigned: np.ndarray, est: dict, rule: str,
                cfg_name: str) -> dict:
    """The four parts + combined, built with score.py's own _recall / _parts (so that rule
    'proxy' reproduces score.score_rows exactly). `est` maps label column -> (R,) strings."""
    ls = tv.label_sets(rule)
    res = {"primary": cfg_name, cfg_name: {"track": track}}
    v, el = ls["classification"]
    res["classification"] = SC._recall(assigned, v, np.asarray(est["classification"], dtype=object), el)
    v, el = ls["identity"]
    res["identity"] = SC._recall(assigned, v, np.asarray(est["identity"], dtype=object), el)
    res["metadata"] = {c: SC._recall(assigned, v, np.asarray(est[c], dtype=object), el)
                       for c, (v, el) in ls["metadata"].items()}
    return SC._parts(res)


def score_all(it: C.Iteration, rows: dict, cfgs=GRID) -> dict:
    """{(mode, rule, cfg): parts} for every truth mode, label rule and configuration."""
    out = {}
    y = rows_enu(it, rows)
    st = np.asarray(rows["step"], dtype=np.int64)
    est = {c: rows[c] for c in EST_COLS}
    for mode in TRUTH_MODES:
        tv = TruthView(it, mode)
        for cfg in cfgs:
            gb = Gosp(tv, st, y, cfg)
            for rule in LABEL_RULES:
                out[(mode, rule, cfg[0])] = score_parts(tv, gb.track, gb.assigned, est, rule, cfg[0])
    return out


# --------------------------------------------------------------------------- probe builders


def track_labels(ctx: Ctx, tracks: np.ndarray, whitelist: set[str]) -> dict:
    """F0's label rule (fuse.track_table) recomputed from tracks.csv for `tracks`."""
    t = ctx.t
    serial = t["serial"][tracks]
    return {
        "classification": np.where(t["cls_fixed_share"][tracks] > 0.5, "Fixed Wing UAS",
                                   "Rotary Wing UAS").astype(object),
        "identity": np.array(["Friendly" if s and s in whitelist else "Hostile" for s in serial],
                             dtype=object),
        "uav_model/manufacturer": t["model"][tracks].astype(object),
        "protocol": t["protocol"][tracks].astype(object),
        "serial_number": np.array([s or "Unknown" for s in serial], dtype=object),
        "signal_frequency_band": np.full(len(tracks), "Unknown", dtype=object),
        "classification_confidence": np.clip(t["p_max_all"][tracks], 0.0, 1.0),
    }


def _end_speed(ctx: Ctx, k: int, first: bool) -> float:
    a, b = int(ctx.gs[k]), int(ctx.ge[k])
    w = int(round(PARK_FIXED["END_WIN_S"] / STEP_S))
    sl = slice(a, min(a + w, b)) if first else slice(max(a, b - w), b)
    sp = np.hypot(ctx.g["ve"][sl], ctx.g["vn"][sl])
    sp = sp[np.isfinite(sp)]
    return float(np.median(sp)) if sp.size else float("nan")


def park_anchors(ctx: Ctx, rf_only: int, stat_p: float) -> list[dict]:
    """GT-free anchors of parked drones (see the module docstring)."""
    g, t = ctx.g, ctx.t
    H_LOW, V_SLOW = PARK_FIXED["H_LOW_M"], PARK_FIXED["V_SLOW_MS"]
    ok_src = ~ctx.utm & (ctx.has_rf if rf_only else True)
    out = []
    for k in np.flatnonzero(ctx.rep & ok_src).tolist():
        for first in (True, False):
            r = int(ctx.gs[k]) if first else int(ctx.ge[k]) - 1
            low = bool(g["u"][r] - ctx.ground <= H_LOW) or bool(1 <= g["u_src"][r] <= 3)
            v = _end_speed(ctx, k, first)
            if low and v == v and v <= V_SLOW:
                out.append({"track": k, "kind": "pre" if first else "post",
                            "e": float(g["e"][r]), "n": float(g["n"][r]), "step": int(g["step"][r])})
    if stat_p > 0:
        cand = (~ctx.rep & ok_src & (t["speed_p90_ms"] <= PARK_FIXED["V_STAT_MS"])
                & (t["n_steps"] >= PARK_FIXED["MIN_STAT_STEPS"]) & (t["p_max_all"] >= stat_p))
        for k in np.flatnonzero(cand).tolist():
            a, b = int(ctx.gs[k]), int(ctx.ge[k])
            fb = (g["u_src"][a:b] >= 1) & (g["u_src"][a:b] <= 3)
            if (t["u_med_m"][k] - ctx.ground <= H_LOW) or bool(fb.all()):
                out.append({"track": k, "kind": "stat", "e": float(np.median(g["e"][a:b])),
                            "n": float(np.median(g["n"][a:b])), "s0": int(g["step"][a]),
                            "s1": int(g["step"][b - 1])})
    return out


def near_rows(step_q: np.ndarray, e: np.ndarray, n: np.ndarray, step_b: np.ndarray,
              y_b: np.ndarray, radius: float) -> np.ndarray:
    """Per query row: is a base row (step_b sorted) within `radius` horizontally at its step?"""
    lo, hi = np.searchsorted(step_b, step_q), np.searchsorted(step_b, step_q, side="right")
    near = np.zeros(step_q.size, dtype=bool)
    for k in range(int((hi - lo).max()) if step_q.size else 0):
        idx = lo + k
        v = idx < hi
        j = np.where(v, idx, 0)
        near |= v & (np.hypot(y_b[j, 0] - e, y_b[j, 1] - n) < radius)
    return near


def park_extra(ctx: Ctx, combo: tuple) -> dict:
    """The rows P_park adds: step, e, n, u, track (whose labels), place; plus n_places, anchors."""
    hold_s, rf_only, stat_p = combo
    anc = park_anchors(ctx, rf_only, stat_p)
    S_ = ctx.n_steps
    h = S_ if hold_s == float("inf") else int(round(hold_s / STEP_S))
    out = {"step": np.zeros(0, np.int64), "e": np.zeros(0), "n": np.zeros(0), "u": np.zeros(0),
           "track": np.zeros(0, np.int64), "place": np.zeros(0, np.int64), "n_places": 0,
           "anchors": anc}
    if not anc:
        return out
    xy = np.array([[a["e"], a["n"]] for a in anc])
    adj = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1]) < PARK_FIXED["MERGE_M"]
    lab = np.arange(len(anc))
    while True:                                   # single linkage: propagate the minimum label
        new = np.array([lab[adj[i]].min() for i in range(len(anc))])
        if np.array_equal(new, lab):
            break
        lab = new
    places = sorted(set(lab.tolist()))
    st, ee, nn, tk, pl = [], [], [], [], []
    for pi, rep in enumerate(places):
        mask = np.zeros(S_, dtype=bool)
        for i in np.flatnonzero(lab == rep).tolist():
            a = anc[i]
            if a["kind"] == "pre":
                mask[max(0, a["step"] - h):a["step"]] = True
            elif a["kind"] == "post":
                mask[a["step"] + 1:min(S_, a["step"] + 1 + h)] = True
            else:
                mask[max(0, a["s0"] - h):min(S_, a["s1"] + 1 + h)] = True
        s = np.flatnonzero(mask)
        st.append(s)
        ee.append(np.full(s.size, anc[rep]["e"]))
        nn.append(np.full(s.size, anc[rep]["n"]))
        tk.append(np.full(s.size, anc[rep]["track"], dtype=np.int64))
        pl.append(np.full(s.size, pi, dtype=np.int64))
    step, e, n = np.concatenate(st), np.concatenate(ee), np.concatenate(nn)
    track, place = np.concatenate(tk), np.concatenate(pl)
    keep = ~near_rows(step, e, n, ctx.step_b, ctx.y_b, PARK_FIXED["MERGE_M"])
    o = np.lexsort((place[keep], step[keep]))
    out.update(step=step[keep][o], e=e[keep][o], n=n[keep][o], u=np.full(int(keep.sum()), ctx.ground),
               track=track[keep][o], place=place[keep][o], n_places=len(places))
    return out


def park_rows(ctx: Ctx, ex: dict, whitelist: set[str]) -> dict:
    keys = np.array([f"{ctx.it.key}/park{p}" for p in ex["place"].tolist()], dtype=object)
    return TR.make_rows(ctx.it, ex["step"], keys, ex["e"], ex["n"], ex["u"], VAR_H, VAR_H, VAR_U,
                        0.0, track_labels(ctx, ex["track"], whitelist))


def air_only_keep(ctx: Ctx, rf_only: int) -> np.ndarray:
    """Mirror probe: keep mask over F0 rows, dropping rows believed parked (low: fused u - g <=
    H_LOW_M or a fallback height; slow: horizontal speed <= V_SLOW_MS), never on UTM tracks."""
    g, gi = ctx.g, ctx.row_grid
    low = (g["u"][gi] - ctx.ground <= PARK_FIXED["H_LOW_M"]) | ((g["u_src"][gi] >= 1) & (g["u_src"][gi] <= 3))
    slow = np.hypot(g["ve"][gi], g["vn"][gi]) <= PARK_FIXED["V_SLOW_MS"]
    src = ~ctx.utm[ctx.row_track] & (ctx.has_rf[ctx.row_track] if rf_only else True)
    return ~(low & slow & src)


def evidence_free(rows: dict) -> np.ndarray:
    """Per row: does its track carry no metadata evidence (all four cells 'Unknown' on every row
    of the track) and is it 'Rotary Wing UAS' / 'Hostile' (on every row)?"""
    tid = np.asarray(rows["track_id"]).astype(str)
    if tid.size == 0:
        return np.zeros(0, dtype=bool)
    _, code = np.unique(tid, return_inverse=True)
    row_ok = np.ones(tid.size, dtype=bool)
    for c in C.METADATA_COLUMNS:
        row_ok &= np.asarray(rows[c]) == "Unknown"
    row_ok &= (np.asarray(rows["classification"]) == "Rotary Wing UAS") & (np.asarray(rows["identity"]) == "Hostile")
    bad = np.bincount(code, weights=(~row_ok).astype(float), minlength=int(code.max()) + 1)
    return bad[code] == 0


def meta_rows(rows: dict, est: dict) -> dict:
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rows.items()}
    m = evidence_free(rows)
    fill = {"uav_model/manufacturer": est["model"], "protocol": est["protocol"],
            "serial_number": est["serial"], "signal_frequency_band": est["band"]}
    for c, v in fill.items():
        a = np.asarray(out[c], dtype=object).copy()
        a[m] = v
        out[c] = a
    return out


def ident_rows(rows: dict) -> dict:
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rows.items()}
    a = np.asarray(out["identity"], dtype=object).copy()
    a[a != "Friendly"] = "Unknown"
    out["identity"] = a
    return out


def shift_rows(it: C.Iteration, rows: dict) -> dict:
    """P_shift15: every row's local ENU position + SHIFT_ENU_M, back to latitude / longitude /
    ellipsoid height; one Newton correction of the (Bowring, sub-millimetre) inverse."""
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rows.items()}
    if len(rows["step"]) == 0:
        return out
    fr = C.iteration_frame(it)
    tgt = rows_enu(it, rows) + np.asarray(SHIFT_ENU_M)
    lat, lon, h = fr.to_geodetic(tgt[:, 0], tgt[:, 1], tgt[:, 2])
    back = np.column_stack(fr.to_enu(lat, lon, h))
    x = tgt + (tgt - back)
    lat, lon, h = fr.to_geodetic(x[:, 0], x[:, 1], x[:, 2])
    out["latitude"], out["longitude"], out["altitude_m"] = (np.asarray(lat, float), np.asarray(lon, float),
                                                            np.asarray(h, float))
    return out


def file_order(rows: dict) -> dict:
    """Rows in submission.write_submission's order: (step, track_id)."""
    if len(rows["step"]) == 0:
        return rows
    _, code = np.unique(np.asarray(rows["track_id"], dtype=object).astype(str), return_inverse=True)
    o = np.lexsort((code, np.asarray(rows["step"], dtype=np.int64)))
    return {k: np.asarray(v)[o] for k, v in rows.items()}


# --------------------------------------------------------------------------- tuning (testing)


def objective(recs: dict, js: list[str], combo: tuple) -> tuple[int, float]:
    """(cells with the wanted sign, mean over iterations of the worst-cell margin) of `combo`
    over iterations `js`; recs[(j, combo)][(mode, cfg)] = combined delta vs P0 (proxy)."""
    n_ok, margins = 0, []
    for j in js:
        d = recs[(j, combo)]
        cells = [d[("all", c[0])] for c in TUNE_CFGS] + [-d[("airborne", c[0])] for c in TUNE_CFGS]
        n_ok += sum(v > 0 for v in cells)
        margins.append(min(cells))
    return n_ok, float(np.mean(margins))


def choose(recs: dict, js: list[str]) -> tuple:
    best, best_v = None, None
    for combo in ALL_COMBOS:                      # PARK_GRID order: the first wins ties
        v = objective(recs, js, combo)
        if best_v is None or v > best_v:
            best, best_v = combo, v
    return best


def _hits(tv_all: TruthView, n_steps: int, step: np.ndarray, xyz: np.ndarray) -> tuple[int, int, int]:
    """Rows within HIT_M (3-D) of a parked / airborne truth position at their step, or neither."""
    if step.size == 0:
        return 0, 0, 0
    tr = tv_all
    nf = int(np.max(SC.cached_truth(tr.it, "all").flight)) + 1 if len(tr.step) else 1
    P = np.full((n_steps, nf, 3), np.nan)
    A = np.zeros((n_steps, nf), dtype=bool)
    fl = SC.cached_truth(tr.it, "all").flight
    P[tr.step, fl] = tr.x
    A[tr.step, fl] = tr.airborne
    d = np.linalg.norm(P[step] - xyz[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    j = d.argmin(axis=1)
    dm = d[np.arange(len(j)), j]
    air = A[step, j]
    hit = dm < HIT_M
    return int((hit & ~air).sum()), int((hit & air).sum()), int((~hit).sum())


def f0_fold_rules() -> dict:
    """{fold key k: F0's fold-k parameters (edge_gate, cont_gap_s, bridge_s, p_field, tau)} and
    'outer': the evaluation choice, re-derived from fused/tuning.csv with fuse.choose (the same
    argmax fuse.py applied; its own self-check re-derives it from the same file)."""
    tb = C.read_columns(FU.TUNING_CSV, numeric={"cont_gap_s", "bridge_s", "tau", "combined_all",
                                                "combined_airborne"})
    recs = [{"k": tb["k"][i], "j": tb["j"][i], "edge_gate": tb["edge_gate"][i],
             "cont_gap_s": float(tb["cont_gap_s"][i]), "bridge_s": float(tb["bridge_s"][i]),
             "p_field": tb["p_field"][i], "tau": float(tb["tau"][i]),
             "combined_all": float(tb["combined_all"][i]), "combined_airborne": float(tb["combined_airborne"][i])}
            for i in range(len(tb["k"]))]
    out = {k: FU.choose(recs, k, [j for j in TESTING if j != k])[0] for k in TESTING}
    out["outer"] = FU.choose(recs, "outer", TESTING)[0]
    return out


def rebuild(it: C.Iteration, P, rule: tuple, src: str, whitelist: set[str], geo_cache: dict) -> tuple:
    """F0's fused tracks of `it` under fuse.py parameters `rule` with existence p of source `src`
    ('outer' = p.csv, or a fold key k = p_inner rows with outer_fold == k): (Ctx parts, provenance)."""
    eg, cg, bs, pf, tau = rule
    if (eg, cg, bs) not in geo_cache:
        Cl = FU.cluster(P, eg, cg)
        g = FU.fuse(P, Cl, bs)
        geo_cache[(eg, cg, bs)] = (Cl, g, FU.track_table(P, Cl, g, whitelist))
    Cl, g, tt = geo_cache[(eg, cg, bs)]
    pa, pr, plabel = P.p[src]
    pmax_all = FU.track_p(Cl, pa, P.D.g_cnt)[0]
    pmax_air = FU.track_p(Cl, pr, P.D.g_cnt)[0]
    pt = pmax_all if pf == "all" else pmax_air
    sel = pt >= tau
    rows = file_order(FU.build_rows(it, g, tt, pt, sel))
    nt = Cl.ntracks
    t = {"track": np.arange(nt),
         "track_id": np.array([TR.track_id(it, f"{it.key}/track{i}") for i in range(nt)], dtype=object),
         "sensor_ids": tt["sensor_ids"], "reported_f0": sel.astype(np.int64), "p_max_all": pmax_all,
         "p_max_air": pmax_air}
    for c in ("speed_p90_ms", "n_steps", "u_med_m", "cls_fixed_share", "serial", "model", "protocol"):
        t[c] = tt[c]
    gg = {c: g[c] for c in ("track", "step", "e", "n", "u", "ve", "vn", "u_src")}
    prov = {"p_label": plabel, "mem_start": np.asarray(Cl.mem_start, dtype=np.int64).copy(),
            "mem_seg_id": np.asarray(P.D.seg_id[Cl.mem_seg], dtype=np.int64).copy(),
            "pmax_all": pmax_all, "pmax_air": pmax_air, "sel": sel, "rule": rule}
    return (t, gg, rows), prov


def _delta_fn(ctx: Ctx, tvs: dict):
    """delta(keep, step_x, y_x, lab_x) -> {(mode, cfg): (d combined, d track)} vs ctx's P0 rows
    (proxy labels, TUNE_CFGS, fast path)."""
    est0 = {c: np.asarray(ctx.rows[c], dtype=object) for c in EST_COLS}
    bases = {}
    for m in TRUTH_MODES:
        for cfg in TUNE_CFGS:
            gb = Gosp(tvs[m], ctx.step_b, ctx.y_b, cfg)
            bases[(m, cfg[0])] = (gb, score_parts(tvs[m], gb.track, gb.assigned, est0, "proxy", cfg[0]))

    def delta(keep: np.ndarray, step_x: np.ndarray, y_x: np.ndarray, lab_x: dict) -> dict:
        est = {c: np.concatenate([est0[c][keep], np.asarray(lab_x[c], dtype=object)]) for c in EST_COLS}
        out = {}
        for (m, cn), (gb, p0) in bases.items():
            tr_, a = variant_gospa(tvs[m], gb, ctx.step_b, ctx.y_b, keep, step_x, y_x)
            pv = score_parts(tvs[m], tr_, a, est, "proxy", cn)
            out[(m, cn)] = (pv["combined"] - p0["combined"], pv["track"] - p0["track"])
        return out

    return delta


def park_deltas(ctx: Ctx, tvs: dict, whitelist: set[str], hits: bool) -> tuple[dict, object]:
    delta = _delta_fn(ctx, tvs)
    allk = np.ones(len(ctx.step_b), dtype=bool)
    park = {}
    for combo in ALL_COMBOS:
        ex = park_extra(ctx, combo)
        y_x = np.column_stack([ex["e"], ex["n"], ex["u"]])
        lab = track_labels(ctx, ex["track"], whitelist)
        park[combo] = {"d": delta(allk, ex["step"], y_x, lab), "n_added": int(ex["step"].size),
                       "n_places": ex["n_places"],
                       "hits": _hits(tvs["all"], ctx.n_steps, ex["step"], y_x) if hits else ("", "", "")}
    return park, delta


def _anchor_sig(ctx: Ctx) -> tuple:
    return tuple(tuple(sorted(tuple(sorted(a.items())) for a in park_anchors(ctx, rf_, sp)))
                 for rf_ in PARK_GRID["rf_only"] for sp in PARK_GRID["stat_p"])


def inner_task(job: tuple) -> dict:
    """Fold-k inputs of testing iteration j (j != k): F0's fold-k rule with p_inner(outer_fold == k);
    P_park deltas of every PARK_GRID combination against that P0."""
    key, k, rule, ground, rf, whitelist = job
    t0 = time.time()
    it = ITERS[key]
    P = FU.Prep(it)
    geo: dict = {}
    parts, prov = rebuild(it, P, rule, k, whitelist, geo)
    ctx = Ctx(it, ground, rf, parts)
    tvs = {m: TruthView(it, m) for m in TRUTH_MODES}
    park, _ = park_deltas(ctx, tvs, whitelist, hits=False)
    # teeth: the same fold-k rule with the OUTER p (models that saw k) gives other anchors?
    parts_o, prov_o = rebuild(it, P, rule, "outer", whitelist, geo)
    anchors_differ = _anchor_sig(Ctx(it, ground, rf, parts_o)) != _anchor_sig(ctx)
    return {"key": key, "k": k, "park": park, "prov": prov, "anchors_differ": anchors_differ,
            "n_rows_p0": int(len(parts[2]["step"])), "seconds": time.time() - t0}


def tune_task(job: tuple) -> dict:
    key, ground, rf, whitelist, own_rule = job
    t0 = time.time()
    it = ITERS[key]
    ctx = Ctx(it, ground, rf)
    tvs = {m: TruthView(it, m) for m in TRUTH_MODES}
    park, delta = park_deltas(ctx, tvs, whitelist, hits=True)
    res = {"key": key, "park": park, "mirror": {}}
    # self-check 10: fuse.py's own fold-j rule with the outer p, rebuilt here, reproduces the
    # shipped F0 tracks.csv / grid.npz / submission rows of j (validates the fold-k rebuild)
    P = FU.Prep(it)
    (t_r, g_r, rows_r), _ = rebuild(it, P, own_rule, "outer", whitelist, {})
    rep = {"n_tracks": len(t_r["track"]) == ctx.nt}
    if rep["n_tracks"]:
        rep["reported"] = bool(np.array_equal(t_r["reported_f0"], ctx.t["reported_f0"].astype(np.int64)))
        rep["p_max_all"] = float(np.max(np.abs(t_r["p_max_all"] - ctx.t["p_max_all"]))) if ctx.nt else 0.0
        rep["grid_keys"] = bool(np.array_equal(g_r["track"], ctx.g["track"]) and np.array_equal(g_r["step"], ctx.g["step"]))
        rep["grid_pos_m"] = float(max(np.max(np.abs(g_r[c] - ctx.g[c])) for c in ("e", "n", "u"))) \
            if rep["grid_keys"] and len(g_r["step"]) else (0.0 if rep["grid_keys"] else float("inf"))
        same_rows = len(rows_r["step"]) == len(ctx.rows["step"])
        rep["rows_keys"] = bool(same_rows and np.array_equal(np.asarray(rows_r["step"]), np.asarray(ctx.rows["step"]))
                                and np.array_equal(np.asarray(rows_r["track_id"]).astype(str),
                                                   np.asarray(ctx.rows["track_id"]).astype(str)))
        rep["rows_pos_m"] = float(np.max(np.abs(rows_enu(it, rows_r) - ctx.y_b))) \
            if rep["rows_keys"] and len(ctx.y_b) else (0.0 if rep["rows_keys"] else float("inf"))
        rep["rows_labels"] = bool(rep["rows_keys"] and _cols_equal(rows_r, ctx.rows, EST_COLS))
    rep["ok"] = bool(rep["n_tracks"] and rep["reported"] and rep["p_max_all"] <= 1e-9 and rep["grid_keys"]
                     and rep["grid_pos_m"] <= REPRO_TOL_M and rep["rows_keys"] and rep["rows_pos_m"] <= REPRO_TOL_M
                     and rep["rows_labels"]) if rep["n_tracks"] else False
    res["repro"] = rep
    del P
    empty = {c: np.zeros(0, dtype=object) for c in EST_COLS}
    for rf_only in (0, 1):
        keep = air_only_keep(ctx, rf_only)
        d = delta(keep, np.zeros(0, np.int64), np.zeros((0, 3)), empty)
        res["mirror"][rf_only] = {"d": d, "n_added": -int((~keep).sum()), "n_places": 0,
                                  "hits": _hits(tvs["all"], ctx.n_steps, ctx.step_b[~keep], ctx.y_b[~keep])}
    # self-check 5 data: ground vs the truth's take-off ellipsoid height (verification only)
    tr = SC.cached_truth(it, "all")
    res["ground"] = (ground, float(np.median(tr.u - tr.height)))
    res["seconds"] = time.time() - t0
    return res


# --------------------------------------------------------------------------- files


def _cols_equal(a: dict, b: dict, cols) -> bool:
    for c in cols:
        x, y = np.asarray(a[c]), np.asarray(b[c])
        if x.shape != y.shape:
            return False
        if x.dtype.kind == "f" or y.dtype.kind == "f":
            if not np.array_equal(x.astype(float), y.astype(float)):
                return False
        elif not np.array_equal(x.astype(str), y.astype(str)):
            return False
    return True


def _subset(rows: dict, m: np.ndarray) -> dict:
    return {k: np.asarray(v)[m] for k, v in rows.items()}


def file_task(job: tuple) -> dict:
    """Build one probe file, write + validate + read back, check its factor, score it (testing)."""
    probe, key, combo, ground, rf, whitelist, meta = job
    t0 = time.time()
    it = ITERS[key]
    ctx = Ctx(it, ground, rf)
    p0 = ctx.rows
    info: dict = {"probe": probe, "key": key, "combo": combo, "n_rows_p0": int(len(p0["step"]))}
    if probe == "P0_f0":
        rows = p0
    elif probe == "P_ident":
        rows = ident_rows(p0)
    elif probe == "P_meta":
        rows = meta_rows(p0, meta)
        info["meta"] = {k: meta[k] for k in ("model", "protocol", "serial", "band")}
        ef = evidence_free(p0)
        info["n_free_rows"] = int(ef.sum())
        info["n_free_tracks"] = len(set(np.asarray(p0["track_id"])[ef].tolist()))
    elif probe == "P_shift15":
        rows = shift_rows(it, p0)
    elif probe == "P_park":
        ex = park_extra(ctx, combo)
        add = park_rows(ctx, ex, whitelist)
        rows = {k: np.concatenate([np.asarray(p0[k]), np.asarray(add[k])]) for k in S.TRACK_FIELDS}
        info.update(n_added=int(ex["step"].size), n_places=ex["n_places"], n_anchors=len(ex["anchors"]),
                    anchor_kinds=dict(Counter(a["kind"] for a in ex["anchors"])),
                    added_sensors=sorted({s for k in set(ex["track"].tolist())
                                          for s in ctx.t["sensor_ids"][k].split("|")}))
    else:
        raise ValueError(probe)
    path = C.submission_path(probe, it)
    S.write_submission(it, rows, path)
    info["path"] = str(path.relative_to(C.REPO))
    info["problems"] = S.validate_submission(it, path)
    back = S.read_submission(it, path)
    info["n_rows"] = int(len(back["step"]))
    info["n_tracks"] = len(set(np.asarray(back["track_id"]).tolist()))
    body = ("step",) + S.BODY
    # ---- self-check 1 / 4: the factor, and nothing else
    f: dict = {}
    if probe == "P0_f0":
        f["equal_all"] = _cols_equal(back, p0, body)
        # F0's rows against the fused grid (independent: the rows of the reported tracks)
        rep_ids = set(ctx.t["track_id"][ctx.rep].tolist())
        f["ids_are_reported"] = set(np.asarray(back["track_id"]).tolist()) == rep_ids
        f["n_rows_eq_grid"] = int(len(back["step"])) == int(np.isin(ctx.g["track"], np.flatnonzero(ctx.rep)).sum())
        gi = ctx.row_grid
        f["pos_err_m"] = float(np.abs(ctx.y_b - np.column_stack([ctx.g["e"][gi], ctx.g["n"][gi], ctx.g["u"][gi]])).max()) \
            if len(gi) else 0.0
        lab = track_labels(ctx, ctx.row_track, whitelist)
        f["labels_rule_ok"] = all(np.array_equal(np.asarray(p0[c]).astype(str), lab[c].astype(str))
                                  for c in EST_COLS)
        f["ok"] = f["equal_all"] and f["ids_are_reported"] and f["n_rows_eq_grid"] and f["pos_err_m"] < 1e-3 \
            and f["labels_rule_ok"]
    elif probe == "P_ident":
        same = [c for c in body if c != "identity"]
        ident0, ident1 = np.asarray(p0["identity"]).astype(str), np.asarray(back["identity"]).astype(str)
        changed = ident0 != ident1
        f["others_equal"] = _cols_equal(back, p0, same)
        f["changed_only_nonfriendly"] = bool(np.all(ident0[changed] != "Friendly"))
        f["all_nonfriendly_unknown"] = bool(np.all(ident1[ident0 != "Friendly"] == "Unknown"))
        f["friendly_kept"] = bool(np.all(ident1[ident0 == "Friendly"] == "Friendly"))
        f["n_changed"] = int(changed.sum())
        f["ok"] = f["others_equal"] and f["changed_only_nonfriendly"] and f["all_nonfriendly_unknown"] \
            and f["friendly_kept"]
    elif probe == "P_meta":
        same = [c for c in body if c not in C.METADATA_COLUMNS]
        f["others_equal"] = _cols_equal(back, p0, same)
        chg = np.zeros(len(p0["step"]), dtype=bool)
        for c in C.METADATA_COLUMNS:
            chg |= np.asarray(p0[c]).astype(str) != np.asarray(back[c]).astype(str)
        free = evidence_free(p0)
        f["changed_only_free"] = bool(np.all(free[chg]))
        f["evidence_kept"] = _cols_equal(_subset(back, ~free), _subset(p0, ~free), C.METADATA_COLUMNS)
        f["free_filled"] = all(np.all(np.asarray(back[c])[free] == v) for c, v in (
            ("uav_model/manufacturer", meta["model"]), ("protocol", meta["protocol"]),
            ("serial_number", meta["serial"]), ("signal_frequency_band", meta["band"])))
        f["n_changed"] = int(chg.sum())
        f["ok"] = f["others_equal"] and f["changed_only_free"] and f["evidence_kept"] and f["free_filled"]
    elif probe == "P_shift15":
        pos = ("latitude", "longitude", "altitude_m")
        f["others_equal"] = _cols_equal(back, p0, [c for c in body if c not in pos])
        off = rows_enu(it, back) - ctx.y_b - np.asarray(SHIFT_ENU_M)
        f["offset_err_m"] = float(np.abs(off).max()) if off.size else 0.0
        f["n_changed"] = int(len(back["step"]))
        f["ok"] = f["others_equal"] and f["offset_err_m"] <= SHIFT_TOL_M
    elif probe == "P_park":
        ids0 = set(np.asarray(p0["track_id"]).tolist())
        in0 = np.isin(np.asarray(back["track_id"]).astype(str), sorted(ids0))
        f["p0_subset_unchanged"] = _cols_equal(_subset(back, in0), p0, body)
        f["added_count"] = int((~in0).sum()) == info["n_added"]
        f["added_ids_new"] = not (set(np.asarray(back["track_id"])[~in0].tolist()) & ids0)
        ex2 = park_extra(ctx, combo)          # determinism: rebuilt identically
        f["deterministic"] = all(np.array_equal(ex2[c], ex[c]) for c in ("step", "e", "n", "place", "track"))
        f["ok"] = f["p0_subset_unchanged"] and f["added_count"] and f["added_ids_new"] and f["deterministic"]
    info["factor"] = f
    # ---- testing: score in full from the read-back file
    if it.has_groundtruth:
        info["scores"] = score_all(it, back, GRID + (SHIFT_EXTRA if probe in ("P0_f0", "P_shift15") else ()))
        if probe in ("P0_f0", "P_park", "P_shift15"):   # self-check 3: proxy == score.score_rows
            diffs = []
            for mode in TRUTH_MODES:
                for cn in ("c50_p2_3d", "c10_p1_2d") + (("c15_p2_3d",) if probe == "P_shift15" else ()):
                    ref = SC.score_rows(it, back, mode, configs=(CFG_BY_NAME[cn],), primary=cn)["parts"]
                    mine = info["scores"][(mode, "proxy", cn)]
                    for k in ("combined", "track", "classification", "identity", "metadata"):
                        nan_r, nan_m = ref[k] != ref[k], mine[k] != mine[k]
                        diffs.append(0.0 if nan_r and nan_m else float("inf") if nan_r or nan_m
                                     else abs(ref[k] - mine[k]))
            info["proxy_max_diff"] = float(max(diffs))
        if probe == "P_park":                   # self-check 6: fast path == full path
            y_back = rows_enu(it, back)
            add_m = ~in0
            est0 = {c: np.asarray(p0[c], dtype=object) for c in EST_COLS}
            est = {c: np.concatenate([est0[c], np.asarray(back[c], dtype=object)[add_m]]) for c in EST_COLS}
            keep = np.ones(len(ctx.step_b), dtype=bool)
            mx = 0.0
            for mode in TRUTH_MODES:
                tv = TruthView(it, mode)
                for cfg in TUNE_CFGS:
                    gb = Gosp(tv, ctx.step_b, ctx.y_b, cfg)
                    tr_, a = variant_gospa(tv, gb, ctx.step_b, ctx.y_b, keep,
                                           np.asarray(back["step"], np.int64)[add_m], y_back[add_m])
                    fast = score_parts(tv, tr_, a, est, "proxy", cfg[0])
                    full = info["scores"][(mode, "proxy", cfg[0])]
                    mx = max(mx, abs(fast["combined"] - full["combined"]), abs(fast["track"] - full["track"]))
            info["fast_full_max_diff"] = mx
    info["seconds"] = time.time() - t0
    return info


# --------------------------------------------------------------------------- self-check 3 (synthetic)


def unk_selftest() -> dict:
    """The 'unk' rule on synthetic rows: truth copied (every part 1 under every rule), then every
    metadata cell set to 'Unknown' (proxy metadata 0; 'unk' metadata = the share of truth rows whose
    catalog value is 'Unknown', recomputed independently)."""
    it = ITERS["testing/S1-I1"]
    tr = SC.cached_truth(it, "all")
    rows = SC._rows_from_truth(tr)
    tv = TruthView(it, "all")
    y = rows_enu(it, rows)
    gb = Gosp(tv, np.asarray(rows["step"], np.int64), y, GRID_BY_NAME[PRIMARY])
    est = {c: rows[c] for c in EST_COLS}
    perfect = {r: score_parts(tv, gb.track, gb.assigned, est, r, PRIMARY) for r in ("proxy", "unk")}
    est_u = dict(est)
    for c in C.METADATA_COLUMNS:
        est_u[c] = np.full(len(rows["step"]), "Unknown", dtype=object)
    mu = {r: score_parts(tv, gb.track, gb.assigned, est_u, r, PRIMARY)["metadata"] for r in ("proxy", "unk")}
    shares = []
    for c in C.METADATA_COLUMNS:
        vals = [tr.flights[f][c] for f in tr.flight.tolist()]
        shares.append(sum(v == "Unknown" for v in vals) / len(vals))
    exp_unk = float(np.mean(shares))
    ok = (all(abs(perfect[r][k] - 1.0) < 1e-12 for r in perfect for k in ("track", "classification", "identity"))
          and abs(perfect["unk"]["metadata"] - 1.0) < 1e-12 and mu["proxy"] == 0.0
          and abs(mu["unk"] - exp_unk) < 1e-12)
    return {"ok": ok, "perfect": perfect, "meta_unknown": mu, "expected_unk": exp_unk}


# --------------------------------------------------------------------------- report


def _f(v, nd: int = 4, sign: bool = False) -> str:
    if v is None or v != v:
        return "n/a"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def md_table(header, rows) -> list[str]:
    out = ["| " + " | ".join(str(h) for h in header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def write_csvs(files: dict, tune: dict, inner: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "probes.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        parts = ("combined", "track", "classification", "identity", "metadata")
        w.writerow(["probe", "iter_key", "truth_mode", "label_rule", "config", "c", "p", "dims"]
                   + list(parts) + [f"d_{k}" for k in parts] + ["n_rows", "n_rows_p0"])
        for key in TESTING:
            base = files[("P0_f0", key)]["scores"]
            for probe in PROBES:
                inf = files[(probe, key)]
                for (mode, rule, cn), pa in sorted(inf["scores"].items()):
                    b = base[(mode, rule, cn)]
                    _, c, p, d = CFG_BY_NAME[cn]
                    w.writerow([probe, key, mode, rule, cn, f"{c:g}", f"{p:g}", d]
                               + [f"{pa[k]:.6g}" if pa[k] == pa[k] else "" for k in parts]
                               + [f"{pa[k] - b[k]:.6g}" if pa[k] == pa[k] and b[k] == b[k] else "" for k in parts]
                               + [inf["n_rows"], inf["n_rows_p0"]])
    with (OUT / "park_tuning.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["k", "variant", "iter_key", "truth_mode", "config", "d_combined", "d_track", "n_added",
                    "n_places", "park_hit50", "air_hit50", "none50"])
        blocks = [("outer", key, [(combo_name(c), v) for c, v in tune[key]["park"].items()]
                   + [(f"air_only_rf{k}", v) for k, v in tune[key]["mirror"].items()]) for key in TESTING]
        blocks += [(k, j, [(combo_name(c), v) for c, v in inner[(k, j)].items()])
                   for k in TESTING for j in TESTING if j != k]
        for src, key, items in blocks:
            for name, v in items:
                for (mode, cn), (dc, dt) in sorted(v["d"].items()):
                    w.writerow([src, name, key, mode, cn, f"{dc:.6g}", f"{dt:.6g}", v["n_added"], v["n_places"]]
                               + list(v["hits"]))


def _d(files: dict, probe: str, key: str, mode: str, rule: str, cn: str) -> float:
    return files[(probe, key)]["scores"][(mode, rule, cn)]["combined"] - \
        files[("P0_f0", key)]["scores"][(mode, rule, cn)]["combined"]


def _pct(v: float) -> str:
    return "n/a" if v != v else f"{100 * v:.1f} %"


def shift_tables(files: dict) -> dict:
    """P_shift15 vs P0 on testing (proxy labels): per (mode, dims, c) the deltas and relative
    drops (P0 - P_shift15) / P0 of the combined score per iteration, and the drop bands."""
    out = {"rel": {}, "d": {}, "bands": {}, "monotone": {}}
    for mode in TRUTH_MODES:
        for d in (3, 2):
            meds = []
            for c in SHIFT_CS:
                cn = f"c{c:g}_p2_{d}d"
                ds, rs = [], []
                for k in TESTING:
                    b = files[("P0_f0", k)]["scores"][(mode, "proxy", cn)]["combined"]
                    v = files[("P_shift15", k)]["scores"][(mode, "proxy", cn)]["combined"]
                    ds.append(v - b)
                    rs.append((b - v) / b if b > 0 else float("nan"))
                out["d"][(mode, d, c)] = ds
                out["rel"][(mode, d, c)] = rs
                meds.append(float(np.nanmedian(rs)))
            out["monotone"][(mode, d)] = all(meds[i] >= meds[i + 1] for i in range(len(meds) - 1))
            for i, c in enumerate(SHIFT_CS):
                hi = (meds[i - 1] + meds[i]) / 2 if i > 0 else float("inf")
                lo = (meds[i] + meds[i + 1]) / 2 if i + 1 < len(meds) else 0.0
                out["bands"][(mode, d, c)] = (lo, hi, meds[i])
    return out


def write_report(files: dict, tune: dict, choices: dict, checks: list, meta: dict, ghs: dict,
                 seconds: float, choices_old: dict, meta_old: dict, rules: dict, c7: dict) -> None:
    L: list[str] = []
    A = L.append
    A("# Leaderboard probes: one factor per upload")
    A("")
    A(f"Generated by `tools/probes.py` in {seconds / 60:.1f} min. Every probe file is F0_fuse (the "
      "current best deployable method) with ONE factor changed, so the leaderboard's move against "
      "F0 on the same data set answers one question about the official scoring. All local numbers "
      "are the PROXY of `tools/score.py` on the 6 testing iterations (plus emulated variants); "
      "they predict the direction of a move, not the official value.")
    A("")
    A("## Plain-language glossary")
    A("")
    for t, d in (
        ("probe", "a submission built only to learn a scoring rule from the leaderboard, not to win."),
        ("P0 / F0", "the reference: F0_fuse's rows, unchanged. Every probe is compared with it."),
        ("delta (d_)", "probe score minus P0 score, same data set and same scoring assumptions."),
        ("relative drop", "(P0 score - probe score) / P0 score: the share of P0's score a probe loses."),
        ("truth mode 'all'", "the official truth includes drones parked on the ground (about 55 % of the Dronetag rows)."),
        ("truth mode 'airborne'", "the official truth only has drones more than 5 m above take-off."),
        ("GOSPA", "the track-accuracy metric: per 100 ms it pairs our tracks with true drones; a pair costs its distance (to the power p), an unpaired truth or track costs c^p / 2."),
        ("c", "the GOSPA cut-off distance: a track farther than c from every drone counts as false and that drone as missed."),
        ("p", "the GOSPA exponent: p = 2 punishes large errors more than p = 1."),
        ("3-D / 2-D", "whether height errors count in GOSPA (3-D) or only the horizontal position (2-D)."),
        ("label rule 'proxy'", "our scorer: a truth row whose catalog value is 'Unknown' is not counted in that label's recall."),
        ("label rule 'unk'", "the alternative: a correct 'Unknown' earns credit (every truth row counts)."),
        ("label rule 'unk_red'", "'unk' plus: the organisers label every non-whitelisted drone 'Unknown' instead of 'Hostile'."),
        ("recall", "share of true drone rows whose paired track carries the right label (a missed drone counts as wrong)."),
        ("LOIO", "leave-one-iteration-out: a choice applied to testing iteration k was made on the other 5."),
        ("nested inputs", "for held-out k, the other iterations' F0 tracks rebuilt with F0's fold-k parameters and existence probabilities from models that never saw k (p_inner)."),
        ("anchor / place", "a ground position where P_park believes a drone is parked; nearby anchors are merged into one place."),
        ("ground height g", "the iteration's ground level, from the ellipsoid heights the ground-mounted sensors report about themselves."),
        ("RF / cyber family", "sensors that read the drone's own radio or telemetry (CEQX, IKJX, ILYS, SMFS, STUG)."),
        ("evidence-free track", "an F0 track whose four metadata cells are all 'Unknown'."),
        ("best-only board", "a leaderboard that shows only the best score uploaded so far per data set."),
    ):
        A(f"* **{t}**: {d}")
    A("")
    A("## What each probe changes")
    A("")
    rows = []
    for key in TESTING + EVALUATION:
        pk, pm, pi = files[("P_park", key)], files[("P_meta", key)], files[("P_ident", key)]
        rows.append([short(key), files[("P0_f0", key)]["n_rows"], pm["factor"]["n_changed"],
                     f"{pm['meta']['model']} / {pm['meta']['protocol']} / {pm['meta']['serial']}",
                     f"{meta_old[key]['model']} / {meta_old[key]['protocol']} / {meta_old[key]['serial']}",
                     pi["factor"]["n_changed"], files[("P_shift15", key)]["factor"]["n_changed"],
                     combo_name(pk["combo"]), f"+{pk['n_added']} rows, {pk['n_places']} places"])
    L += md_table(["iteration", "P0 rows", "P_meta rows changed", "P_meta fill (model / protocol / serial)",
                   "P_meta fill before the Blue-model fix", "P_ident rows changed", "P_shift15 rows moved 15 m east",
                   "P_park setting", "P_park adds"], rows)
    A("")
    A("P_meta's fill comes from each iteration's own reports (counts of mapped hostile models). Rows "
      "WITHOUT a serial whose model is that of a serial whitelisted in the iteration (the whitelisted "
      "serials its reports carry; on evaluation this is its own Friendly list) are excluded: they may "
      "be the Blue drone reported without its serial.")
    A("")
    for key in TESTING + EVALUATION:
        m = meta[key]
        A(f"* {short(key)}: models {m['model_counts'] or 'none'}; excluded serial-less rows "
          f"{m['excluded_counts'] or 'none'} (whitelisted serials reported here: "
          f"{', '.join(m['whitelisted_here']) or 'none'}); protocols on the chosen model "
          f"{m['protocol_counts'] or 'none'}.")
    A("")
    A("## Predicted deltas on testing (combined score, probe minus P0)")
    A("")
    A("One row per scoring hypothesis; one column per testing iteration; the last column counts the "
      "iterations where the delta is positive. A probe is informative when the sign flips between "
      "hypotheses and is stable across iterations.")
    for probe in ("P_park", "P_meta", "P_ident"):
        A("")
        A(f"### {probe}")
        A("")
        rr = []
        for mode in TRUTH_MODES:
            for rule in LABEL_RULES:
                for cn in HYP_CONFIGS:
                    ds = [_d(files, probe, k, mode, rule, cn) for k in TESTING]
                    rr.append([mode, rule, cn] + [_f(d, 4, True) for d in ds]
                              + [f"{sum(d > 1e-12 for d in ds)} up / {sum(d < -1e-12 for d in ds)} down"])
        L += md_table(["truth", "labels", "GOSPA"] + [short(k) for k in TESTING] + ["of 6"], rr)
    # ---- P_shift15
    sh = shift_tables(files)
    A("")
    A("## P_shift15: bracketing the GOSPA cut-off c")
    A("")
    A(f"P_shift15 is P0 with every position moved {SHIFT_ENU_M[0]:g} m due east (local ENU); nothing "
      "else changes. A 15 m error costs little when c is much larger than 15 m and nearly a missed "
      "drone plus a false track when c is 15 m or less, so the relative drop of the score against P0 "
      "says how large c is. Proxy labels (labels are unchanged; the label parts move only because the "
      "pairing of tracks with drones changes), p = 2.")
    A("")
    A("Predicted delta (P_shift15 minus P0, combined score) per testing iteration, with the relative "
      "drop in parentheses:")
    A("")
    rr = []
    for mode in TRUTH_MODES:
        for d in (3, 2):
            for c in SHIFT_CS:
                ds, rs = sh["d"][(mode, d, c)], sh["rel"][(mode, d, c)]
                rr.append([mode, f"{d}-D", f"{c:g}"] + [f"{_f(x, 4, True)} ({_pct(r)})" for x, r in zip(ds, rs)]
                          + [f"{sum(x < -1e-12 for x in ds)}/6"])
    L += md_table(["truth", "GOSPA", "c (m)"] + [short(k) for k in TESTING] + ["down"], rr)
    A("")
    A("Decision table: find the observed relative drop (P0 score - P_shift15 score) / P0 score on the "
      "leaderboard in the column of the scoring hypothesis you believe (or read across all four). Each "
      "cell is the band of drops closest to that c (band edges = midpoints between the medians over the "
      "6 testing iterations of neighbouring c values); 'median [min .. max]' shows the spread across "
      "iterations: where neighbouring c values' spreads overlap, the bracket is only that coarse.")
    A("")
    rr = []
    hyps = [(m, d) for m in TRUTH_MODES for d in (3, 2)]
    for c in SHIFT_CS:
        row = [f"c ~ {c:g} m"]
        for m, d in hyps:
            lo, hi, med = sh["bands"][(m, d, c)]
            rs = [r for r in sh["rel"][(m, d, c)] if r == r]
            band = (f">= {_pct(lo)}" if hi == float("inf") else f"{_pct(lo)} .. {_pct(hi)}")
            row.append(f"{band}; median {_pct(med)} [{_pct(min(rs))} .. {_pct(max(rs))}]")
        rr.append(row)
    L += md_table(["c bracket"] + [f"'{m}' {d}-D" for m, d in hyps], rr)
    nonmono = [f"'{m}' {d}-D" for (m, d), ok in sh["monotone"].items() if not ok]
    n_down = sum(x < -1e-12 for v in sh["d"].values() for x in v)
    n_all = sum(len(v) for v in sh["d"].values())
    A("")
    A(f"P_shift15 is predicted BELOW P0 in {n_down} of {n_all} (iteration, hypothesis, c) cells, so on a "
      "best-only board it is never visible: upload it only where every upload's score is shown. "
      + ("Every column's median drop decreases with c, so the bands are ordered. " if not nonmono else
         f"Non-monotone median drop in {', '.join(nonmono)}: read those bands with care. ")
      + "Evaluation's drop also depends on how close F0's evaluation tracks are to the truth (unknown); "
      "the testing spread above is the honest uncertainty of the bracket.")
    # ---- P_park LOIO
    A("")
    A("## P_park: robustness over the full GOSPA grid (LOIO, nested inputs)")
    A("")
    A("Each testing iteration k uses the setting chosen on the other 5 with NESTED inputs: their F0 "
      "tracks rebuilt with F0's fold-k parameters and existence probabilities that never saw k "
      "(p_inner, outer_fold == k). The previous version chose fold k on the SHIPPED F0 outputs of the "
      "other iterations, whose existence p came from models that had seen k ('old' column). Counts are "
      "over the 16 GOSPA configurations (c 10/25/50/100 m, p 1/2, 3-D/2-D), proxy labels, scored from "
      "the written file.")
    A("")
    rr = []
    for key in TESTING:
        sc, b = files[("P_park", key)]["scores"], files[("P0_f0", key)]["scores"]
        da = [sc[("all", "proxy", g[0])]["combined"] - b[("all", "proxy", g[0])]["combined"] for g in GRID]
        dr = [sc[("airborne", "proxy", g[0])]["combined"] - b[("airborne", "proxy", g[0])]["combined"] for g in GRID]
        h = tune[key]["park"][choices[key]]["hits"]
        rr.append([short(key), combo_name(choices[key]), combo_name(choices_old[key]),
                   "/".join(f"{x:g}" if isinstance(x, float) else str(x) for x in rules[key]),
                   f"{sum(d > 0 for d in da)}/16", f"{sum(d < 0 for d in dr)}/16", _f(min(da), 4, True),
                   _f(max(da), 4, True), _f(min(dr), 4, True), _f(max(dr), 4, True), f"{h[0]} / {h[1]} / {h[2]}"])
    L += md_table(["iteration", "setting (nested, chosen without it)", "old setting (shipped inputs)",
                   "F0 fold rule (edge / cont / bridge / p / tau)", "'all' up", "'airborne' down",
                   "min d 'all'", "max d 'all'", "min d 'airborne'", "max d 'airborne'",
                   "added rows near parked / airborne truth / neither (50 m)"], rr)
    n_changed = sum(choices[k] != choices_old[k] for k in TESTING)
    A("")
    A(f"Nested inputs changed the fold choice on {n_changed} of 6 folds. Evaluation uses the setting "
      f"chosen on all 6 with the shipped (outer) inputs, as F0's own evaluation choice does: "
      f"**{combo_name(choices['all'])}**.")
    A("")
    A("Tuning table (all 6 testing iterations, outer inputs, fast path, the 6 score.py configurations x 2 "
      "truth modes = 72 cells; 'cells' = cells with the wanted sign, 'margin' = mean over iterations of the "
      "worst cell):")
    A("")
    recs_c = {(k, c): {mc: v[0] for mc, v in tune[k]["park"][c]["d"].items()} for k in TESTING for c in ALL_COMBOS}
    rr = []
    for c in ALL_COMBOS:
        n_ok, mg = objective(recs_c, TESTING, c)
        rr.append([combo_name(c), f"{n_ok}/72", _f(mg, 5, True),
                   sum(tune[k]["park"][c]["n_added"] for k in TESTING)])
    L += md_table(["setting", "cells", "margin", "rows added (6 iterations)"], rr)
    A("")
    A("Mirror probe P_air_only (drop F0 rows believed parked; wanted signs reversed: down under 'all', "
      "up under 'airborne'), not shipped:")
    A("")
    rr = []
    for rf_only in (0, 1):
        cells = 0
        for k in TESTING:
            d = tune[k]["mirror"][rf_only]["d"]
            cells += sum(d[("all", c[0])][0] < 0 for c in TUNE_CFGS) + sum(d[("airborne", c[0])][0] > 0 for c in TUNE_CFGS)
        rr.append([f"air_only_rf{rf_only}", f"{cells}/72",
                   sum(-tune[k]["mirror"][rf_only]["n_added"] for k in TESTING)])
    L += md_table(["variant", "cells with the wanted sign", "rows removed (6 iterations)"], rr)
    best_mirror = max(int(r[1].split("/")[0]) for r in rr)
    n_park = objective(recs_c, TESTING, choices["all"])[0]
    A("")
    A(f"The cell counts are NOT a fair race: P_park's {n_park}/72 is the best of {len(ALL_COMBOS)} "
      f"settings counted on the same 6 iterations it was chosen on (in-sample), the mirror's "
      f"{best_mirror}/72 the best of 2. The held-out evidence for shipping P_park is the LOIO robustness "
      "table above (each row's setting chosen without that iteration). The mirror also moves the "
      "score far less: F0 reports few parked drones, so removing them changes little.")
    # ---- park numbers used by the limits text
    park_sign = {}
    for c in (10, 25, 50, 100):
        for d in (3, 2):
            cn = f"c{c:g}_p2_{d}d"
            ua = sum(_d(files, "P_park", k, "all", "proxy", cn) > 1e-12 for k in TESTING)
            dr = sum(_d(files, "P_park", k, "airborne", "proxy", cn) < -1e-12 for k in TESTING)
            park_sign[(c, d)] = (ua, dr)
    A("")
    A("P_park sign counts on the held-out testing iterations (proxy, p = 2): 'all' up / 'airborne' down, of 6:")
    A("")
    L += md_table(["c (m)", "3-D: 'all' up", "3-D: 'airborne' down", "2-D: 'all' up", "2-D: 'airborne' down"],
                  [[f"{c:g}", f"{park_sign[(c, 3)][0]}/6", f"{park_sign[(c, 3)][1]}/6", f"{park_sign[(c, 2)][0]}/6",
                    f"{park_sign[(c, 2)][1]}/6"] for c in (10, 25, 50, 100)])
    # ---- P_meta per data set
    A("")
    A("## P_meta: what each data set can tell")
    A("")
    A("Under 'proxy' a wrong guess scores exactly like 'Unknown' (a truth row whose catalog value is "
      "'Unknown' is not scored; a known truth value is missed either way), so P_meta can only go UP or "
      "stay EQUAL there. Under 'unk' a guess on a truth-'Unknown' row loses the credit 'Unknown' had, so "
      "P_meta can go either way. Hence: DOWN means 'a correct Unknown earns credit' (only 'unk' allows "
      "it); UP means the guess was right on some scored rows, which both rules allow; EQUAL is "
      "inconclusive. Testing iterations with a non-empty fill (PRIMARY c50_p2_3d):")
    A("")
    rr = []
    for k in TESTING:
        if files[("P_meta", k)]["factor"]["n_changed"] == 0:
            continue
        vals = {(m, r): _d(files, "P_meta", k, m, r, PRIMARY) for m in TRUTH_MODES for r in ("proxy", "unk")}
        sep = all(np.sign(round(vals[(m, "proxy")], 12)) != np.sign(round(vals[(m, "unk")], 12)) for m in TRUTH_MODES)
        rr.append([short(k), meta[k]["model"], _f(vals[("all", "proxy")], 4, True), _f(vals[("all", "unk")], 4, True),
                   _f(vals[("airborne", "proxy")], 4, True), _f(vals[("airborne", "unk")], 4, True),
                   "yes" if sep else "no"])
    L += md_table(["testing iteration", "fill model", "'all' proxy", "'all' unk", "'airborne' proxy",
                   "'airborne' unk", "separates 'proxy' from 'unk' (sign differs in both truth modes)"], rr)
    A("")
    rr = []
    for key in EVALUATION:
        pm = files[("P_meta", key)]
        ds = key.split("/")[1] + " " + short(key)
        if pm["factor"]["n_changed"] == 0:
            rr.append([ds, "none (" + (", ".join(f"{m} {n}" for m, n in meta[key]["excluded_counts"].items())
                                       + " serial-less rows excluded" if meta[key]["excluded_counts"] else
                                       "no hostile model reported") + ")", 0,
                       "identical to P0: do not upload; no information"])
        else:
            same = [short(k) for k in TESTING if meta[k]["model"] == meta[key]["model"]
                    and files[("P_meta", k)]["factor"]["n_changed"] > 0]
            rr.append([ds, f"{meta[key]['model']} / {meta[key]['protocol']} / {meta[key]['serial']}",
                       pm["factor"]["n_changed"],
                       "DOWN -> 'unk' (Unknown credited); UP -> the guess pays (favours 'proxy' where the "
                       f"same-model testing iterations {', '.join(same) or 'none'} went down under 'unk'); "
                       "EQUAL -> inconclusive"])
    L += md_table(["data set", "fill (model / protocol / serial)", "rows changed", "reading"], rr)
    # ---- decision table
    A("")
    A("## Decision table")
    A("")
    A("Compare each probe's leaderboard score with P0's on the SAME data set (D = DI S1-I3, DII S2-I1, "
      "DIII S2-I3). One data set is one vote; read the three together. Two columns: what an outcome "
      "means when the board shows every upload's score, and when it shows only the best so far (then "
      "the board rises only if the probe beats every earlier upload of that data set; 'unchanged' only "
      "means 'not above the best so far', and after an earlier probe went up it does not even mean "
      "'not above P0').")
    A("")
    ps = park_sign
    L += md_table(["probe", "outcome vs P0", "board shows every score", "board shows only the best", "then"], [
        ["P_meta", "up", "the guess was right on scored rows: a best guess pays (both label rules allow it)",
         "same (the board rises above P0)", "fill evidence-free tracks with the per-iteration best estimate"],
        ["P_meta", "down", "a correct 'Unknown' earns credit ('unk'; impossible under 'proxy')",
         "not visible: 'unchanged' = down or equal", "keep 'Unknown' where there is no evidence"],
        ["P_meta", "equal", "inconclusive: the filled tracks matched no scored row",
         "'unchanged' = down or equal: cannot tell", "keep 'Unknown' (no gain either way)"],
        ["P_ident", "up", "the official identity of non-whitelisted drones is 'Unknown' (rule 'unk_red'); "
         "'Unknown' merely being credited ('unk', truth 'Hostile') predicts down on 6/6",
         "same, if it beats the best so far (on DI P_meta may already have raised the board)",
         "write 'Unknown' for non-whitelisted tracks"],
        ["P_ident", "down", "non-whitelisted drones are 'Hostile' in the official truth",
         "not visible: 'unchanged' = down or equal (or hidden below an earlier probe that went up)", "keep 'Hostile'"],
        ["P_ident", "equal", "identity is not scored on these rows", "'unchanged': cannot tell", "keep 'Hostile'"],
        ["P_shift15", "down by a relative drop r", "c range from the P_shift15 decision table",
         "never visible (predicted below P0 everywhere): no information", "score locally with the bracketed c"],
        ["P_park", "up", "the official truth includes parked drones (mode 'all') and c is not small",
         "same, if it beats the best so far (on DI possibly P_meta)",
         "report parked drones (build a proper ground-drone tracker); score locally with truth 'all'"],
        ["P_park", "down", f"the official truth is airborne only, OR c is small (at c = 10 m, 3-D: 'all' up only "
         f"{ps[(10, 3)][0]}/6 on testing)", "not visible: 'unchanged' = down or equal",
         "never report parked drones; score locally with truth 'airborne'; consider dropping low-and-slow rows"],
        ["P_park", "equal", "the added rows were not scored or matched nothing (inconclusive)",
         "'unchanged': cannot tell", "check that the file was accepted; ask the organisers"],
    ])
    A("")
    ev_add = ", ".join(f"{short(k)} +{files[('P_park', k)]['n_added']}" for k in EVALUATION)
    none_sh = []
    for k in TESTING:
        h = tune[k]["park"][choices[k]]["hits"]
        tot = sum(h)
        if tot:
            none_sh.append(h[2] / tot)
    A("Expected direction from testing (proxy, c = 50 m, 3-D): P_park up if 'all' and down if "
      "'airborne'; P_meta up or unchanged under 'proxy', mostly down under 'unk' (table above); P_ident "
      "down under 'proxy' and 'unk', up only under 'unk_red'; P_shift15 down, by an amount that "
      "brackets c. Limits:")
    A("")
    A(f"* (a) P_park separates the truth modes reliably only for a large c. On the held-out testing "
      f"iterations (p = 2, 3-D) 'all' up / 'airborne' down: c = 10 m {ps[(10, 3)][0]}/6 and {ps[(10, 3)][1]}/6, "
      f"c = 25 m {ps[(25, 3)][0]}/6 and {ps[(25, 3)][1]}/6, c = 50 m {ps[(50, 3)][0]}/6 and {ps[(50, 3)][1]}/6, "
      f"c = 100 m {ps[(100, 3)][0]}/6 and {ps[(100, 3)][1]}/6 (2-D in the table above). Read a P_park "
      "'down' as 'airborne only, or c small'; P_shift15 tells which.")
    A(f"* (b) The evaluation P_park (setting {combo_name(choices['all'])}) adds {ev_add} rows; with "
      "hold = inf these are permanent static tracks over the whole iteration. On testing "
      f"{_pct(min(none_sh)) if none_sh else 'n/a'} to {_pct(max(none_sh)) if none_sh else 'n/a'} of the "
      "added rows are within 50 m of neither a parked nor an airborne truth position, so the leaderboard "
      "delta of P_park also carries this false-track cost: an 'up' is the net of found parked drones minus "
      "false tracks, and a 'down' under 'all' is possible when the false tracks dominate.")
    A("* (c) Probes moved the 'wrong' way on single testing iterations (S3-I2 for P_park): one data set "
      "alone is weak evidence, three agreeing data sets are strong. A move opposite to every hypothesis "
      "above means the proxy itself is wrong on that part: report it before acting.")
    # ---- checklist
    A("")
    A("## Upload checklist")
    A("")
    A("1. De-anonymise every file first (the owner's step): all coordinates here are in the "
      "ANONYMIZED longitude frame. Keep the file name (scenario_N_iteration_M.xlsx).")
    A("2. Find out whether the leaderboard shows every upload's score or only the best one, and use the "
      "matching column of the decision table.")
    A("3. Per data set, upload in this order and write down the board's score after each upload. Each "
      "probe is compared with P0's score, never with the probe before it:")
    A("")
    rr = []
    for key in EVALUATION:
        i = 0
        for probe in PROBES:
            f = files[(probe, key)]
            skip = probe != "P0_f0" and probe != "P_park" and f["factor"].get("n_changed", 1) == 0
            i += 0 if skip else 1
            rr.append([key.split("/")[1] + " " + short(key), "skip" if skip else i, probe,
                       "identical to P0: do not upload" if skip else f["path"], f["n_rows"]])
    L += md_table(["data set", "order", "probe", "file", "rows"], rr)
    A("")
    A("4. Why this order: P0 first (the reference); P_meta, P_ident and P_shift15 are one-factor "
      "probes of P0 expected to move little or down; P_park LAST, because under truth 'all' it is "
      "expected to go up the most, and on a best-only board every probe uploaded after it would be "
      "hidden unless it beat P_park. P_shift15 is informative only when every score is shown. On DI, "
      "P_meta may go up and then hide a smaller P_ident rise on a best-only board (see the decision table).")
    A("5. After the answers: combine the winning factors into one submission (the probes change "
      "disjoint things: metadata cells of evidence-free tracks, identity cells of non-Friendly rows, "
      "added rows; P_shift15 is never part of it).")
    A("")
    A("## Inputs, rules and caveats")
    A("")
    A("* Inputs: F0's fused tracks (`data/derived/fused/`), F0's submission files, the cleaned "
      "detections (metadata counts, sensor heights), fuse.py's tuning table (fold rules) and the "
      "existence p (p.csv, p_inner.csv). fuse.py's tuning is not re-run; nothing is fitted on "
      "evaluation; the P_park setting is chosen on testing (LOIO with nested inputs for testing files, "
      "all 6 with outer inputs for evaluation).")
    A(f"* LOIO provenance (self-check 7): {c7['n_pairs']} (fold, iteration) inputs rebuilt from p_inner "
      f"(outer_fold == fold); with the outer p instead, the reported set differs on {c7['n_sel_differs']} "
      f"and the anchors on {c7['n_anchor_differs']} of them. Inherited second-order leak: j's honest-"
      "variance table excludes j only (as in fuse.py's own tuning).")
    A("* Ground heights g (m, local frame): " + ", ".join(f"{short(k)} {v:.1f}" for k, v in ghs.items())
      + ". Testing take-off heights from the truth (checking only): "
      + ", ".join(f"{short(k)} {tune[k]['ground'][1]:.1f}" for k in TESTING) + ".")
    A("* P_park never uses tracks with a VKYY / QTVS member (UTM relays of the Dronetag feed in testing).")
    A("* The proxy's label truth comes from `tools/drone_catalog.csv` (our own mapping); band truth is "
      "'Unknown' for every testing flight, so band never moves the 'proxy' metadata part.")
    A("* Magnitudes on evaluation differ from testing; the decisions use the sign, and P_shift15 the "
      "size of the relative drop with the testing spread as its uncertainty.")
    A("* Inherited caveat: F0's sensor noise tables and height offsets were fitted on all 6 testing "
      "iterations (fuse.md).")
    A("")
    A("## Self-checks")
    A("")
    for name, ok, detail in checks:
        A(f"* [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    A("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probes.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- LOIO provenance


def _tune_dispatch(job: tuple) -> dict:
    kind, args = job
    return tune_task(args) if kind == "tune" else inner_task(args)


def _track_max(vals: np.ndarray, mem_start: np.ndarray) -> np.ndarray:
    return np.maximum.reduceat(vals, mem_start) if vals.size else np.zeros(0)


def loio_provenance(inner_res: dict, rules: dict) -> dict:
    """Self-check 7: recompute every (k, j) fold-k input from the raw existence csv files."""
    out = {"no_self": all(k != j for k, j in inner_res), "bad": [], "n_pairs": 0, "n_outer_differs": 0,
           "n_sel_differs": 0, "n_anchor_differs": 0}
    want = {(k, j) for k in TESTING for j in TESTING if j != k}
    if set(inner_res) != want:
        out["bad"].append(f"pairs {sorted(set(inner_res) ^ want)[:3]}")
    for j in TESTING:
        slug = C.iter_slug(j)
        q = C.read_columns(FU.EXIST_DIR / f"{slug}__p_inner.csv", numeric={"seg_id", "p_drone_all", "p_drone_air"})
        po = C.read_columns(FU.EXIST_DIR / f"{slug}__p.csv", numeric={"seg_id", "p_drone_all", "p_drone_air"})
        omap_a = dict(zip(po["seg_id"].astype(np.int64).tolist(), po["p_drone_all"].tolist()))
        omap_r = dict(zip(po["seg_id"].astype(np.int64).tolist(), po["p_drone_air"].tolist()))
        for k in TESTING:
            if k == j or (k, j) not in inner_res:
                continue
            r = inner_res[(k, j)]
            pv = r["prov"]
            m = q["outer_fold"] == k
            imap_a = dict(zip(q["seg_id"][m].astype(np.int64).tolist(), q["p_drone_all"][m].tolist()))
            imap_r = dict(zip(q["seg_id"][m].astype(np.int64).tolist(), q["p_drone_air"][m].tolist()))
            segs = pv["mem_seg_id"].tolist()
            if any(sg not in imap_a for sg in segs):
                out["bad"].append(f"{short(k)}/{short(j)}: member without p_inner row")
                continue
            pf, tau = rules[k][3], rules[k][4]
            ia = _track_max(np.array([imap_a[sg] for sg in segs]), pv["mem_start"])
            ir = _track_max(np.array([imap_r[sg] for sg in segs]), pv["mem_start"])
            isel = (ia if pf == "all" else ir) >= tau
            if pv["p_label"] != k or tuple(pv["rule"]) != tuple(rules[k]) or not np.array_equal(ia, pv["pmax_all"]) \
                    or not np.array_equal(ir, pv["pmax_air"]) or not np.array_equal(isel, pv["sel"]):
                out["bad"].append(f"{short(k)}/{short(j)}: inputs differ from p_inner outer_fold == k")
                continue
            out["n_pairs"] += 1
            oa = _track_max(np.array([omap_a[sg] for sg in segs]), pv["mem_start"])
            orr = _track_max(np.array([omap_r[sg] for sg in segs]), pv["mem_start"])
            osel = (oa if pf == "all" else orr) >= tau
            sel_d = not np.array_equal(osel, pv["sel"])
            out["n_sel_differs"] += int(sel_d)
            out["n_outer_differs"] += int(sel_d or not np.array_equal(oa, pv["pmax_all"]))
            out["n_anchor_differs"] += int(r["anchors_differ"])
    return out


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    whitelist = B.load_whitelist()
    rf = rf_ids()
    log(f"RF / cyber family: {sorted(rf)}; whitelist: {len(whitelist)} serials")
    ghs = ground_heights()
    log("ground heights g: " + ", ".join(f"{short(k)} {v:.1f}" for k, v in ghs.items()))
    meta = meta_estimates(whitelist)
    meta_old = meta_estimates(whitelist, exclude_blue_model=False)
    for k in sorted(meta):
        log(f"  P_meta {short(k)}: {meta[k]['model']} / {meta[k]['protocol']} / {meta[k]['serial']}"
            f"  counts {meta[k]['model_counts']} excluded {meta[k]['excluded_counts']}"
            f"  (old rule: {meta_old[k]['model']} / {meta_old[k]['protocol']} / {meta_old[k]['serial']})")
    rules = f0_fold_rules()
    for k, r in rules.items():
        log(f"  F0 fold rule {short(k) if k != 'outer' else 'evaluation'}: {r}")
    ctx_mp = mp.get_context("fork")
    # ---- tuning on testing: outer inputs (+ self-check 10) and nested fold-k inputs
    jobs_t = [("tune", (k, ghs[k], rf, whitelist, rules[k])) for k in TESTING]
    jobs_t += [("inner", (j, k, rules[k], ghs[j], rf, whitelist)) for k in TESTING for j in TESTING if j != k]
    jobs_t.sort(key=lambda x: (x[1][0] not in ("testing/S3-I3", "testing/S3-I2", "testing/S3-I1"), x[0] != "tune",
                               x[1][0], x[1][1] if x[0] == "inner" else ""))
    log(f"tuning P_park: {len(ALL_COMBOS)} settings; {len(TESTING)} outer + {len(jobs_t) - len(TESTING)} "
        f"nested (fold k, iteration j) inputs, {a.workers} workers")
    with ctx_mp.Pool(a.workers) as pool:
        res = pool.map(_tune_dispatch, jobs_t, chunksize=1)
    tune = {r["key"]: r for r in res if "k" not in r}
    inner_res = {(r["k"], r["key"]): r for r in res if "k" in r}
    for k in TESTING:
        log(f"  {short(k)} outer tuned in {tune[k]['seconds']:.0f} s; nested inputs "
            + ", ".join(f"fold {short(kk)} {inner_res[(kk, k)]['seconds']:.0f} s" for kk in TESTING if kk != k))
    inner = {kj: r["park"] for kj, r in inner_res.items()}
    recs_outer = {(j, c): {mc: v[0] for mc, v in tune[j]["park"][c]["d"].items()} for j in TESTING for c in ALL_COMBOS}
    recs_fold = {k: {(j, c): {mc: v[0] for mc, v in inner[(k, j)][c]["d"].items()}
                     for j in TESTING if j != k for c in ALL_COMBOS} for k in TESTING}
    choices = {k: choose(recs_fold[k], [j for j in TESTING if j != k]) for k in TESTING}
    choices["all"] = choose(recs_outer, TESTING)
    choices_old = {k: choose(recs_outer, [j for j in TESTING if j != k]) for k in TESTING}
    for k, c in choices.items():
        log(f"  choice for {short(k) if k != 'all' else 'evaluation (all 6)'}: {combo_name(c)}"
            + (f" (old, outer inputs: {combo_name(choices_old[k])})" if k != "all" else ""))
    # ---- files
    jobs = []
    for key in TESTING + EVALUATION:
        for probe in PROBES:
            combo = (choices[key] if key in choices else choices["all"]) if probe == "P_park" else None
            jobs.append((probe, key, combo, ghs[key], rf, whitelist, meta[key]))
    jobs.sort(key=lambda j: (j[1] not in ("testing/S3-I3", "testing/S3-I2", "testing/S3-I1"), j[0] != "P_park"))
    log(f"writing {len(jobs)} probe files with {a.workers} workers")
    with ctx_mp.Pool(a.workers) as pool:
        out = pool.map(file_task, jobs, chunksize=1)
    files = {(r["probe"], r["key"]): r for r in out}
    for key in TESTING + EVALUATION:
        log(f"  {short(key)}: " + "; ".join(
            f"{p} {files[(p, key)]['n_rows']} rows {files[(p, key)]['seconds']:.0f} s" for p in PROBES))
    # ---- self-checks
    checks = []
    f1 = {(p, k): files[(p, k)]["factor"]["ok"] for p in PROBES[1:] for k in TESTING + EVALUATION}
    bad = [f"{p}@{short(k)}" for (p, k), ok in f1.items() if not ok]
    checks.append(("1 every probe differs from P0 only in its factor (P_ident: identity of non-Friendly rows; "
                   "P_meta: metadata of evidence-free tracks; P_shift15: positions only; P_park: P0 rows "
                   "unchanged + new-id rows, deterministic)",
                   not bad, f"all {len(f1)} files" if not bad else f"failed: {bad}"))
    probs = {(p, k): files[(p, k)]["problems"] for p in PROBES for k in TESTING + EVALUATION}
    badv = [f"{p}@{short(k)}: {v[:2]}" for (p, k), v in probs.items() if v]
    checks.append(("2 every file validates (submission.validate_submission)", not badv,
                   f"{len(probs)} files" if not badv else "; ".join(badv)))
    pm = max(files[(p, k)]["proxy_max_diff"] for p in ("P0_f0", "P_park", "P_shift15") for k in TESTING)
    us = unk_selftest()
    checks.append(("3 variant scorer with label rule 'proxy' equals score.score_rows exactly (P0, P_park, P_shift15; "
                   "6 iterations x 2 modes x c50_p2_3d / c10_p1_2d, P_shift15 also c15_p2_3d); 'unk' on synthetic "
                   "rows (truth copy = 1; "
                   "all-Unknown metadata: proxy 0, unk = share of truth 'Unknown')",
                   pm == 0.0 and us["ok"],
                   f"max |diff| {pm:.3g}; unk metadata {us['meta_unknown']['unk']:.6f} vs expected "
                   f"{us['expected_unk']:.6f}, proxy {us['meta_unknown']['proxy']:.3g}"))
    f4 = {k: files[("P0_f0", k)]["factor"] for k in TESTING + EVALUATION}
    bad4 = [short(k) for k, v in f4.items() if not v["ok"]]
    checks.append(("4 P0 rows equal F0_fuse rows (every cell), track ids = F0's reported tracks, positions = "
                   "fused grid (< 1e-3 m), labels = F0's track rule",
                   not bad4, f"max position error {max(v['pos_err_m'] for v in f4.values()):.2e} m"
                   if not bad4 else f"failed: {bad4}"))
    gd = {k: tune[k]["ground"] for k in TESTING}
    worst = max(abs(a_ - b_) for a_, b_ in gd.values())
    checks.append((f"5 GT-free ground height within {GROUND_TOL_M:g} m of the testing take-off height (truth, "
                   "checking only)", worst <= GROUND_TOL_M, f"worst |diff| {worst:.2f} m"))
    fd = max(files[("P_park", k)]["fast_full_max_diff"] for k in TESTING)
    checks.append((f"6 tuning fast path equals the full score of the written P_park file (6 configs x 2 modes, "
                   f"tolerance {INC_TOL:g})", fd <= INC_TOL, f"max |diff| {fd:.2e}"))
    # 7 LOIO with nested inputs: fold k's records exist only for j != k; each (k, j) input was
    # built from p_inner rows with outer_fold == k (recomputed here from the raw csv); the same
    # recomputation with the outer p.csv must differ on at least one pair (the check has teeth)
    c7 = loio_provenance(inner_res, rules)
    ok7 = c7["no_self"] and not c7["bad"] and c7["n_outer_differs"] >= 1
    checks.append(("7 LOIO, nested inputs: fold k's P_park choice reads only j != k, and every (k, j) input "
                   "(F0 fold-k reported set, p_max_all) equals its recomputation from p_inner.csv rows with "
                   "outer_fold == k; the same recomputation from the outer p.csv differs on >= 1 pair (teeth)",
                   ok7, f"{c7['n_pairs']} pairs match p_inner; outer p differs on {c7['n_outer_differs']} pairs "
                   f"(reported set on {c7['n_sel_differs']}), anchors differ on {c7['n_anchor_differs']}"
                   if ok7 else f"failed: self {not c7['no_self']}; {c7['bad'][:4]}; outer differs "
                   f"{c7['n_outer_differs']}"))
    # 8 P_park signs on the held-out iteration (LOIO, proxy, PRIMARY) -- reported, must be computed
    sig = []
    for k in TESTING:
        sc, b = files[("P_park", k)]["scores"], files[("P0_f0", k)]["scores"]
        sig.append((sc[("all", "proxy", PRIMARY)]["combined"] - b[("all", "proxy", PRIMARY)]["combined"],
                    sc[("airborne", "proxy", PRIMARY)]["combined"] - b[("airborne", "proxy", PRIMARY)]["combined"]))
    n_sep = sum(da > 0 and dr < 0 for da, dr in sig)
    checks.append(("8 P_park separates the truth modes on the held-out iteration (PRIMARY: up under 'all', "
                   "down under 'airborne') on at least 4 of 6", n_sep >= 4, f"{n_sep} of 6"))
    f9 = {k: files[("P_shift15", k)]["factor"] for k in TESTING + EVALUATION}
    bad9 = [short(k) for k, v in f9.items() if not v["ok"]]
    checks.append((f"9 P_shift15 differs from P0 only in position, by exactly {SHIFT_ENU_M} m ENU (read back, "
                   f"tolerance {SHIFT_TOL_M:g} m; every other cell equal): 9 files", not bad9,
                   f"max |offset error| {max(v['offset_err_m'] for v in f9.values()):.2e} m"
                   if not bad9 else f"failed: {bad9}"))
    rp = {k: tune[k]["repro"] for k in TESTING}
    bad10 = [k for k, v in rp.items() if not v["ok"]]
    checks.append(("10 fold-k rebuild is faithful: fuse.py's fold-j parameters + outer p, rebuilt here, reproduce "
                   "j's shipped F0 tracks (count, reported set, p_max_all), fused grid and F0_fuse rows "
                   f"(ids, labels, positions < {REPRO_TOL_M:g} m) on every testing iteration", not bad10,
                   f"max grid {max(v.get('grid_pos_m', 0.0) for v in rp.values()):.2e} m, rows "
                   f"{max(v.get('rows_pos_m', 0.0) for v in rp.values()):.2e} m"
                   if not bad10 else f"failed: {[(short(b), rp[b]) for b in bad10]}"))
    secs = time.time() - t0
    checks.append((f"11 runtime under {RUNTIME_LIMIT_S / 60:.0f} min", secs < RUNTIME_LIMIT_S, f"{secs / 60:.1f} min"))
    write_csvs(files, tune, inner)
    write_report(files, tune, choices, checks, meta, ghs, secs, choices_old=choices_old, meta_old=meta_old,
                 rules=rules, c7=c7)
    for name, ok, detail in checks:
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    log(S.ANON_REMINDER)
    log(f"wrote {(OUT / 'probes.md').relative_to(C.REPO)}, probes.csv, park_tuning.csv in {secs / 60:.1f} min")
    if not all(ok for _, ok, _ in checks):
        log("SELF-CHECK FAILED")
        return 1
    log("ALL SELF-CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
