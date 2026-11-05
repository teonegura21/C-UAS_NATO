#!/usr/bin/env python3
"""Calibration + ground-truth association + "which sensors see which drone" (testing tree).

Only the six testing iterations have Dronetag ground truth, so everything here is estimated
on them. Pipeline (all deterministic, numpy only):

A. CALIBRATION against GT -> data/derived/sensor_calibration.csv
   * time offset per position sensor (scan -6..+6 s, 0.25 s; score = share of usable
     detections whose nearest GT flight at t+d is within 30 m). Adopted (pooled) only when the
     pooled peak beats d=0 by >= 10 % relative AND the informative per-iteration argmaxes
     spread <= 1.0 s. t_cal = t + offset.
   * azimuth offset per (sensor, iteration) for bearing sensors (measured az) and radars
     (derived az); scan -180..+180 deg, 0.5 deg. ROBUST score: every (object_id, 1-s bin) is ONE
     sample (hit if any of its rows has some active flight within 5 deg), samples are averaged
     per object and objects are weighted equally. Per-iteration offsets are applied only when
     n >= 50 samples, peak >= 2 x chance and >= 0.3, AND no more than 50 % of the peak's score
     comes from a single object (>= 2 objects present) or from constant-bearing runs (az spread
     < 0.5 deg: a stare at one stationary drone cannot identify an offset). A pooled offset
     (valid beyond testing) needs >= 2 agreeing iterations (within 3 deg) that are also >= 2/3 of
     the sensor's informative iterations (n >= 50). GTYT / OEPS (45-degree sectors) get the
     dedicated +22.5 deg test instead. Radars additionally get a "radar_bias" row measured on
     position-matched rows only (estimate_only: radars associate by position).
   * every calibration row has a `source` (scan | radar_bias | sector_test) and a `status`
     (applied | estimate_only | rejected); `adopted` == (status == 'applied');
     (sensor_key, iter_key, param, source) is unique.
B. ASSOCIATION -> det_gt_match.csv.gz (with gt_height / gt_airborne), track_labels.csv
   (air/ground/ambiguous drone labels)
C. COVERAGE / CO-DETECTION -> flight_coverage.csv, codetection_sets.csv, codetection_pairs.csv
   (variants all, no_gt_feed, no_gt_feed_air)
D. REPORT -> gt_association.md

Interpolation of a GT flight is Q.interp_flight's (linear in time, NaN outside the span),
plus one rule: inside a GT gap longer than GAP_MAX_S (60 s) the flight is treated as not active
(Dronetag logs have gaps of up to 741 s; interpolating across them invents a track). 'Airborne'
means GT height (above the take-off point) > 5 m at the matched time.

Usage: python3 tools/gt_associate.py
"""

from __future__ import annotations

import collections
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import cuas_clean as Q  # noqa: E402
import cuas_common as C  # noqa: E402

OUT = C.DERIVED
T0 = time.time()

# ----------------------------------------------------------------------------- parameters
TIME_GRID = np.arange(-6.0, 6.0 + 1e-9, 0.25)
TIME_GATE_M = 30.0
TIME_REL_GAIN = 1.10          # pooled peak must beat d=0 by >= 10 %
TIME_MAX_SPREAD_S = 1.0       # informative per-iteration argmaxes
TIME_INFORMATIVE_MIN_HITS = 20
TIME_INFORMATIVE_PEAK_RATIO = 1.2   # peak score / median score over the scan
AZ_STEP = 0.5
AZ_GRID = np.arange(-180.0, 180.0 + 1e-9, AZ_STEP)      # 721 points (+-180 are the same angle)
AZ_N = 720
PLATEAU_FRAC = 0.98
AZ_ZERO_IDX = 360
AZ_WINDOW = 5.0
AZ_MIN_N = 15                 # sector test only
AZ_ADOPT_MIN_N = 50           # scan: (object, 1-s) samples needed for per-iteration adoption
AZ_POOL_FRAC = 2.0 / 3.0      # pooled az offset: agreeing share of informative iterations
MAX_SUPPORT_SHARE = 0.5       # single object / constant-bearing run share of the peak score
AZ_OBJ_MIN_SAMPLES = 5        # objects with fewer 1-s samples do not vote (their hit fraction is noise)
AZ_CONSENSUS_DEG = 10.0       # object-balanced and sample-weighted (stare-free) scans must agree this closely
STARE_SPREAD_DEG = 0.5
STARE_MIN_SAMPLES = 10
STARE_GAP_S = 10.0
AZ_CHANCE_MULT = 2.0
AZ_MIN_SCORE = 0.3
AZ_POOL_AGREE = 3.0
SECTOR_HALF = 22.5
SECTOR_GAIN = 1.10
ZERO_PREF = 0.9
PRESENCE_ONLY = ("YXAB", "ORFH")
GAP_MAX_S = 60.0              # a GT gap longer than this is "not flying" (was 6 s; see report 1d)
GAP_OLD_S = 6.0
AIR_H_M = 5.0                 # airborne = GT height above take-off > 5 m
CHANCE_SHIFTS = (-60.0, 60.0)
POS_GATES = (25.0, 50.0, 100.0)      # matched_25 / matched / matched_100
BEAR_GATE_SCALE = (0.5, 1.0, 2.0)
RADAR_BIAS_MAX_RESID_M = 50.0
RADAR_BIAS_MIN_RANGE_M = 300.0
RADAR_BIAS_MIN_N = 100
COV_MIN_FRAC = 0.05
SECTOR_SENSORS = ("GTYT", "OEPS")
CAL_ORDER = ["sensor_key", "iter_key", "param", "source", "status", "value", "score_at_value", "score_at_zero",
             "chance", "n", "adopted", "applies_to", "note"]
GT_FEED = tuple(sorted(Q.GT_FEED_SENSORS))

DET_COLS = ["iter_key", "tree", "file", "row", "sensor_id", "sensor_type", "sensor_key", "object_id",
            "t", "meas_kind", "e", "n", "az_deg", "az_err_deg", "az_ref", "sensor_e", "sensor_n",
            "sensor_pos_valid"]


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def short_iter(ik: str) -> str:
    return ik.split("/", 1)[1]


# ----------------------------------------------------------------------------- GT flights
class Flight:
    """One Dronetag flight: sorted samples + gap-aware interpolation."""

    def __init__(self, gid: int, gt: dict, mask: np.ndarray):
        order = np.argsort(gt["t"][mask], kind="stable")
        self.gid = gid
        self.iter_key = str(gt["iter_key"][mask][0])
        self.key = str(gt["flight_key"][mask][0])
        self.side = str(gt["side"][mask][0])
        self.name = str(gt["drone_name"][mask][0])
        self.t = gt["t"][mask][order]
        self.e = gt["e"][mask][order]
        self.n = gt["n"][mask][order]
        self.h = gt["height"][mask][order]
        self.seg = np.diff(self.t)
        self.t0, self.t1 = float(self.t[0]), float(self.t[-1])
        self.seconds = np.unique(np.round(self.t).astype(np.int64))
        self.label = f"{short_iter(self.iter_key)}:{self.key}"

    def _bad(self, t: np.ndarray) -> np.ndarray:
        bad = (t < self.t0) | (t > self.t1)
        i = np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.seg) - 1)
        bad |= (self.seg[i] > GAP_MAX_S) & (t > self.t[i]) & (t < self.t[i + 1])
        return bad

    def at(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(e, n) at times t; NaN outside the span or inside a GT gap > GAP_MAX_S."""
        e = np.interp(t, self.t, self.e)
        n = np.interp(t, self.t, self.n)
        bad = self._bad(t)
        e[bad] = np.nan
        n[bad] = np.nan
        return e, n

    def height_at(self, t: np.ndarray) -> np.ndarray:
        """GT 'height' (above take-off) at times t; same activity rule as `at`."""
        h = np.interp(t, self.t, self.h)
        h[self._bad(t)] = np.nan
        return h

    def air_L(self) -> np.ndarray:
        """bool[L] over the flight's second grid (index = second - first second): GT height > AIR_H_M
        at the GT sample second (False for seconds without a GT sample)."""
        base = int(self.seconds[0])
        out = np.zeros(int(self.seconds[-1] - base + 1), dtype=bool)
        out[self.seconds - base] = np.interp(self.seconds.astype(float), self.t, self.h) > AIR_H_M
        return out


def build_flights(gt: dict) -> tuple[list[Flight], dict[str, list[Flight]]]:
    flights, by_iter = [], collections.defaultdict(list)
    for ik in sorted(set(gt["iter_key"].tolist())):
        for fk in sorted(set(gt["flight_key"][gt["iter_key"] == ik].tolist())):
            f = Flight(len(flights), gt, (gt["iter_key"] == ik) & (gt["flight_key"] == fk))
            flights.append(f)
            by_iter[ik].append(f)
    return flights, dict(by_iter)


# ----------------------------------------------------------------------------- helpers
def pick_argmax(score: np.ndarray, circular: bool = False, with_raw: bool = False):
    """Index of the scan maximum, taken as the CENTRE of the near-maximal run (score >= PLATEAU_FRAC x
    max, contiguous around the exact argmax). A flat-topped curve (slow drones, 30 m / 5 deg gates) has
    no unique argmax: the first/median tied sample would be an arbitrary edge of the plateau.
    `circular`: the scan wraps (azimuth, 720 distinct samples). `with_raw`: also return the raw
    np.argmax (first maximal sample) so reports can show both."""
    score = np.asarray(score, float)
    n = len(score)
    j = int(np.argmax(score))
    thr = PLATEAU_FRAC * score[j]

    def ok(k: int) -> bool:
        return score[k % n] >= thr if circular else (0 <= k < n and score[k] >= thr)

    hi = 0
    while hi < n - 1 and ok(j + hi + 1):
        hi += 1
    lo = 0
    while hi - lo < n - 1 and ok(j + lo - 1):
        lo -= 1
    c = int(np.floor(j + (lo + hi) / 2.0 + 0.5))
    c = c % n if circular else c
    return (c, j) if with_raw else c


def circ_median(vals: list[float]) -> float:
    v = np.asarray(vals, float)
    ref = v[0]
    d = C.angle_diff_deg(v, ref)
    return float(C.angle_diff_deg(ref + np.median(d), 0.0))


def az_hit_matrix(az: np.ndarray, azf: np.ndarray, halfw: float) -> np.ndarray:
    """bool[n, 720]: for each detection and scan offset o_j = -180 + AZ_STEP*j, whether some flight
    lies within `halfw` deg of az + o_j. `azf` is (n, F) flight azimuths, NaN = inactive."""
    n = len(az)
    w = int(np.floor(2 * halfw / AZ_STEP)) + 1
    ks = np.arange(w)
    hits = np.zeros((n, AZ_N), dtype=bool)
    for s in range(0, n, 20000):
        a, f = az[s:s + 20000], azf[s:s + 20000]
        c = C.angle_diff_deg(f, a[:, None])                  # offset that aligns az with flight
        valid = np.isfinite(c)
        cc = np.where(valid, c, 0.0)
        jlo = np.ceil((cc - halfw + 180.0) / AZ_STEP - 1e-9).astype(np.int64)
        jhi = np.floor((cc + halfw + 180.0) / AZ_STEP + 1e-9).astype(np.int64)
        j = jlo[:, :, None] + ks[None, None, :]
        sel = valid[:, :, None] & (j <= jhi[:, :, None])
        rows = np.broadcast_to(np.arange(len(a))[:, None, None] + s, j.shape)
        hits[rows[sel], j[sel] % AZ_N] = True
    return hits


def az_hits(az: np.ndarray, azf: np.ndarray, halfw: float) -> tuple[np.ndarray, int]:
    """Row-level counts (legacy score, kernel self-check): number of detections with some flight within
    `halfw` of az + o_j. Detections with no active flight are dropped. Returns (counts[721], n)."""
    ok = np.isfinite(azf).any(axis=1) & np.isfinite(az)
    az, azf = az[ok], azf[ok]
    if len(az) == 0:
        return np.zeros(AZ_N + 1, dtype=np.int64), 0
    counts = az_hit_matrix(az, azf, halfw).sum(axis=0).astype(np.int64)
    return np.append(counts, counts[0]), len(az)


def az_scan_group(az: np.ndarray, azf: np.ndarray, obj: np.ndarray, t: np.ndarray, halfw: float,
                  want_stare: bool) -> dict:
    """Robust azimuth scan of one (sensor, iteration). Each (object_id, 1-s bin) is ONE sample (hit at
    offset o if any of its rows has an active flight within `halfw` of az + o); the score at o is the
    mean over objects of the object's hit fraction (objects weighted equally; objects with fewer than
    AZ_OBJ_MIN_SAMPLES samples do not vote). `score_b` is the cross-check estimator: all samples of all
    objects pooled, constant-bearing (stare) samples removed. Also returns what the support test needs:
    per-object hit counts, and the same restricted to constant-bearing samples. Rows without an active
    flight are dropped."""
    ok = np.isfinite(azf).any(axis=1) & np.isfinite(az)
    az, azf, obj, t = az[ok], azf[ok], obj[ok], t[ok]
    n_rows = len(az)
    empty = dict(score=np.zeros(AZ_N + 1), score_b=np.zeros(AZ_N + 1), rows=np.zeros(AZ_N + 1), n_rows=0, n=0, k=0,
                 n_eff=0, k_eff=0, keep=np.zeros(0, bool), obj_n=np.zeros(0), obj_hits=np.zeros((0, AZ_N)),
                 stare_hits=np.zeros((0, AZ_N)), stare_n=np.zeros(0), n_stare=0)
    if n_rows == 0:
        return empty
    _, oidx = np.unique(obj, return_inverse=True)
    tb = np.floor(t - t.min()).astype(np.int64)
    ukey, sinv = np.unique(oidx.astype(np.int64) * (1 << 32) + tb, return_inverse=True)
    ns = len(ukey)
    order = np.argsort(sinv, kind="stable")
    az_s, azf_s, sinv_s = az[order], azf[order], sinv[order]
    starts = np.searchsorted(sinv_s, np.arange(ns))
    S = np.zeros((ns, AZ_N), dtype=bool)
    rows_cnt = np.zeros(AZ_N, dtype=np.int64)
    s0 = 0
    while s0 < ns:
        s1 = int(np.searchsorted(starts, starts[s0] + 20000, side="left"))
        s1 = min(max(s1, s0 + 1), ns)
        a, b = int(starts[s0]), (int(starts[s1]) if s1 < ns else n_rows)
        hits = az_hit_matrix(az_s[a:b], azf_s[a:b], halfw)
        rows_cnt += hits.sum(axis=0)
        S[s0:s1] = np.logical_or.reduceat(hits, starts[s0:s1] - a, axis=0)
        s0 = s1
    sobj = (ukey >> 32).astype(np.int64)
    K = int(sobj.max()) + 1
    ostart = np.searchsorted(sobj, np.arange(K))
    obj_n = np.bincount(sobj, minlength=K).astype(float)
    obj_hits = np.add.reduceat(S, ostart, axis=0, dtype=np.int32).astype(float)
    stare_hits = np.zeros_like(obj_hits)
    stare_n = np.zeros(K)
    n_stare = 0
    if want_stare:
        saz, st = az_s[starts], (ukey & ((1 << 32) - 1)).astype(float)
        stare = np.zeros(ns, dtype=bool)
        for k in range(K):
            a, b = int(ostart[k]), (int(ostart[k + 1]) if k + 1 < K else ns)
            if b - a < STARE_MIN_SAMPLES:
                continue
            cut = np.flatnonzero(np.diff(st[a:b]) > STARE_GAP_S) + 1
            for lo, hi in zip(np.r_[0, cut], np.r_[cut, b - a]):
                if hi - lo < STARE_MIN_SAMPLES:
                    continue
                d = C.angle_diff_deg(saz[a + lo:a + hi], saz[a + lo])
                if d.max() - d.min() < STARE_SPREAD_DEG:
                    stare[a + lo:a + hi] = True
        n_stare = int(stare.sum())
        if n_stare:
            stare_hits = np.add.reduceat(S & stare[:, None], ostart, axis=0, dtype=np.int32).astype(float)
            stare_n = np.add.reduceat(stare.astype(np.int64), ostart).astype(float)
    keep = obj_n >= AZ_OBJ_MIN_SAMPLES
    score = (obj_hits[keep] / obj_n[keep][:, None]).mean(axis=0) if keep.any() else np.zeros(AZ_N)
    den = float((obj_n - stare_n).sum())
    score_b = (obj_hits - stare_hits).sum(axis=0) / den if den > 0 else np.zeros(AZ_N)
    return dict(score=np.append(score, score[0]), score_b=np.append(score_b, score_b[0]),
                rows=np.append(rows_cnt, rows_cnt[0]).astype(float), n_rows=n_rows, n=ns, k=K,
                n_eff=int(obj_n[keep].sum()), k_eff=int(keep.sum()), keep=keep, obj_n=obj_n, obj_hits=obj_hits,
                stare_hits=stare_hits, stare_n=stare_n, n_stare=n_stare)


def support_shares(r: dict, j: int) -> tuple[float, float]:
    """(largest single-object share, constant-bearing share) of the object-balanced score at scan index j
    (only objects that vote, i.e. with >= AZ_OBJ_MIN_SAMPLES samples)."""
    kp = r["keep"]
    if r["k_eff"] == 0:
        return float("nan"), float("nan")
    c = r["obj_hits"][kp, j % AZ_N] / r["obj_n"][kp]
    tot = float(c.sum())
    if tot <= 0:
        return float("nan"), float("nan")
    return float(c.max() / tot), float((r["stare_hits"][kp, j % AZ_N] / r["obj_n"][kp]).sum() / tot)


def flight_azimuths(flights: list[Flight], t: np.ndarray, se: np.ndarray, sn: np.ndarray) -> np.ndarray:
    out = np.full((len(t), len(flights)), np.nan)
    for k, f in enumerate(flights):
        fe, fn = f.at(t)
        out[:, k] = C.azimuth_deg(fe - se, fn - sn)
        out[~np.isfinite(fe), k] = np.nan
    return out


# ----------------------------------------------------------------------------- A. calibration
def scan_time_offsets(D: dict, by_iter: dict[str, list[Flight]]) -> dict:
    """{(sensor_key, iter): (hits[len(TIME_GRID)], n)} for usable, near position rows."""
    res = {}
    sel_all = D["usable_pos"] & ~D["far"]
    for ik, fl in by_iter.items():
        rows = np.flatnonzero(sel_all & (D["iter_key"] == ik))
        if len(rows) == 0:
            continue
        t, e, n = D["t"][rows], D["e"][rows], D["n"][rows]
        sks, sidx = np.unique(D["sensor_key"][rows], return_inverse=True)
        hits = np.zeros((len(sks), len(TIME_GRID)), dtype=np.int64)
        for j, d in enumerate(TIME_GRID):
            dmin = np.full(len(rows), np.inf)
            for f in fl:
                fe, fn = f.at(t + d)
                dist = np.hypot(e - fe, n - fn)
                dist[~np.isfinite(dist)] = np.inf
                np.minimum(dmin, dist, out=dmin)
            hits[:, j] = np.bincount(sidx[dmin <= TIME_GATE_M], minlength=len(sks))
        cnt = np.bincount(sidx, minlength=len(sks))
        for k, sk in enumerate(sks):
            res[(str(sk), ik)] = (hits[k], int(cnt[k]))
    return res


def decide_time_offsets(scan: dict) -> tuple[list[dict], dict[str, float], list[dict]]:
    """Calibration rows + adopted pooled offsets {sensor_key: seconds} + report details."""
    rows, adopted, detail = [], {}, []
    i0 = int(np.argmin(np.abs(TIME_GRID)))
    for sk in sorted({k[0] for k in scan}):
        its = sorted(k[1] for k in scan if k[0] == sk)
        per_arg, informative = {}, []
        pooled_h = np.zeros(len(TIME_GRID), dtype=np.int64)
        pooled_n = 0
        for ik in its:
            h, n = scan[(sk, ik)]
            pooled_h += h
            pooled_n += n
            j, jr = pick_argmax(h, with_raw=True)
            sc = h / max(n, 1)
            inf = bool(h[j] >= TIME_INFORMATIVE_MIN_HITS and sc[j] >= TIME_INFORMATIVE_PEAK_RATIO * max(np.median(sc), 1e-12))
            per_arg[ik] = (float(TIME_GRID[j]), inf, float(TIME_GRID[jr]))
            if inf:
                informative.append(float(TIME_GRID[j]))
            rows.append(crow(sk, ik, "time_offset_s", "scan", "estimate_only", float(TIME_GRID[j]), sc[j], sc[i0],
                             float(sc.mean()), n, "testing",
                             f"per-iteration plateau-centre argmax (raw argmax {TIME_GRID[jr]:+.2f}); "
                             f"hits_at_peak={int(h[j])}; informative={'yes' if inf else 'no'}"))
        j, jr = pick_argmax(pooled_h, with_raw=True)
        sc = pooled_h / max(pooled_n, 1)
        peak, zero = sc[j], sc[i0]
        spread = (max(informative) - min(informative)) if informative else float("nan")
        need = min(2, len(its))
        if len(informative) < need:
            ok, why = False, f"only {len(informative)} informative iteration(s) (need {need}); using 0"
        elif not (peak >= TIME_REL_GAIN * zero) or pooled_h[j] < TIME_INFORMATIVE_MIN_HITS:
            ok, why = False, (f"pooled peak {peak:.4f} at {TIME_GRID[j]:+.2f}s is not >= 1.10 x score(0)={zero:.4f}"
                              f" (or < {TIME_INFORMATIVE_MIN_HITS} hits); using 0")
        elif spread > TIME_MAX_SPREAD_S:
            ok, why = False, f"per-iteration argmaxes spread {spread:.2f}s > 1.0s; using 0"
        else:
            ok, why = True, (f"adopted: pooled peak {peak:.4f} vs score(0) {zero:.4f} "
                             f"(x{peak / max(zero, 1e-12):.2f}); informative argmax spread {spread:.2f}s")
        why += f" [plateau-centre argmax {TIME_GRID[j]:+.2f}, raw argmax {TIME_GRID[jr]:+.2f}]"
        if ok:
            adopted[sk] = float(TIME_GRID[j])
        rows.append(crow(sk, "*", "time_offset_s", "scan", "applied" if ok else "rejected", float(TIME_GRID[j]), peak,
                         zero, float(sc.mean()), pooled_n, "all", why))
        detail.append(dict(sensor_key=sk, pooled=float(TIME_GRID[j]), pooled_raw=float(TIME_GRID[jr]), peak=peak,
                           zero=zero, n=pooled_n, ok=ok, why=why, per_iter=per_arg, spread=spread))
    return rows, adopted, detail


def scan_azimuth(D: dict, by_iter: dict[str, list[Flight]], toff: dict[str, float]) -> dict:
    """{(sensor_key, iter): dict(r5=<az_scan_group 5 deg>, rs=<sector scan>|None, kind, sector_resid)} for
    bearing sensors (measured az) and radars (derived az)."""
    res = {}
    is_bearing = np.isin(D["meas_kind"], Q.BEARING_KINDS) & (D["sensor_pos_valid"] == 1) & np.isfinite(D["az_deg"])
    is_radar = (D["sensor_type"] == "RADAR") & D["usable_pos"] & ~D["far"] & (D["sensor_pos_valid"] == 1) \
        & np.isfinite(D["az_deg"])
    for ik, fl in by_iter.items():
        base = (is_bearing | is_radar) & (D["iter_key"] == ik)
        for sk in sorted(set(D["sensor_key"][base].tolist())):
            rows = np.flatnonzero(base & (D["sensor_key"] == sk))
            t = D["t"][rows] + toff.get(sk, 0.0)
            azf = flight_azimuths(fl, t, D["sensor_e"][rows], D["sensor_n"][rows])
            kind = "radar" if sk.split("_")[1] == "RADAR" else "bearing"
            out = dict(kind=kind, rs=None,
                       r5=az_scan_group(D["az_deg"][rows], azf, D["object_id"][rows], t, AZ_WINDOW, kind == "bearing"))
            if sk.split("_")[0] in SECTOR_SENSORS:
                out["rs"] = az_scan_group(D["az_deg"][rows], azf, D["object_id"][rows], t, SECTOR_HALF, False)
                c0 = C.angle_diff_deg(azf, D["az_deg"][rows][:, None])
                okr = np.isfinite(c0).any(axis=1)
                c0 = c0[okr]
                out["sector_resid"] = c0[np.arange(len(c0)), np.argmin(np.where(np.isfinite(c0), np.abs(c0), np.inf), axis=1)]
            res[(sk, ik)] = out
    return res


def crow(sensor_key, iter_key, param, source, status, value, score_at_value, score_at_zero, chance, n, applies_to,
         note, **extra) -> dict:
    """One calibration row. `adopted` is DERIVED from status ('applied')."""
    assert source in ("scan", "radar_bias", "sector_test") and status in ("applied", "estimate_only", "rejected")
    return dict(sensor_key=sensor_key, iter_key=iter_key, param=param, source=source, status=status, value=value,
                score_at_value=score_at_value, score_at_zero=score_at_zero, chance=chance, n=int(n),
                adopted=int(status == "applied"), applies_to=applies_to, note=note, **extra)


def decide_az_offsets(scan: dict) -> tuple[list[dict], dict, dict, list[dict]]:
    """Calibration rows; per-iteration applied offsets {(sensor_key, iter): deg}; pooled applied
    offsets {sensor_key: deg}; report detail."""
    rows, per_iter, pooled, detail = [], {}, {}, []
    nan = float("nan")
    for sk in sorted({k[0] for k in scan}):
        sid = sk.split("_")[0]
        its = sorted(k[1] for k in scan if k[0] == sk)
        kind = scan[(sk, its[0])]["kind"]
        sector = sid in SECTOR_SENSORS
        presence_only = sid in PRESENCE_ONLY
        adopted_it, it_detail = {}, {}
        for ik in its:
            r5 = scan[(sk, ik)]["r5"]
            sc, n = r5["score"], r5["n_eff"]
            j, j_raw = pick_argmax(sc[:AZ_N], circular=True, with_raw=True)
            peak, zero, chance = float(sc[j]), float(sc[AZ_ZERO_IDX]), float(sc.mean())
            legacy_j = pick_argmax(r5["rows"][:AZ_N], circular=True) if r5["n_rows"] else 0
            jb = pick_argmax(r5["score_b"][:AZ_N], circular=True)
            vb = float(AZ_GRID[jb])
            if r5["score_b"][AZ_ZERO_IDX] >= ZERO_PREF * r5["score_b"][jb]:
                vb = 0.0
            passes_thr = n >= AZ_ADOPT_MIN_N and peak >= AZ_CHANCE_MULT * chance and peak >= AZ_MIN_SCORE
            value, score_v = float(AZ_GRID[j]), peak
            note = (f"{kind} robust az scan (1-s samples, objects equal), window 5 deg, n_samples={n} "
                    f"(of {r5['n']}; objects with >= {AZ_OBJ_MIN_SAMPLES} samples vote), n_objects={r5['k_eff']} "
                    f"(of {r5['k']}), n_rows={r5['n_rows']}; plateau-centre argmax {AZ_GRID[j]:+.1f} "
                    f"(raw argmax {AZ_GRID[j_raw]:+.1f}; legacy row-weighted {AZ_GRID[legacy_j]:+.1f}; "
                    f"cross-check sample-weighted stare-free {vb:+.1f})")
            if passes_thr and zero >= ZERO_PREF * peak and abs(value) > 0:
                note += (f"; scan peak at {value:+.1f} is indistinguishable from 0 (score@0 >= {ZERO_PREF:g} x peak, "
                         f"the 5-deg window cannot resolve smaller offsets): offset set to 0")
                value, score_v = 0.0, zero
            jf = int(round((value + 180.0) / AZ_STEP)) % AZ_N
            sh_obj, sh_stare = support_shares(r5, jf)
            if kind == "bearing" and not sector and not presence_only:
                note += f"; support: top object {sh_obj:.2f}, constant-bearing {sh_stare:.2f} of the score at the offset"
            if sid == "EMXG" and abs(value) > 90:
                note += "; EMXG FLIP (rotated ~160-170 deg in this iteration)"
            if sid == "FGIX":
                note += "; az_ref magnetic (FGIX)"
            status = "applied" if passes_thr else "rejected"
            if sector:
                status = "rejected"
                note += "; sector sensor: governed by the +22.5 sector test (sector_test rows)"
            elif presence_only:
                status = "rejected"
                note += ("; PRESENCE-ONLY override: bearings do not track any drone (physics check), so no offset "
                         "is applied even where the scan clears the thresholds" if passes_thr else
                         "; presence-only sensor (bearing does not track any drone)")
            elif kind == "radar":
                status = "rejected"
                note += ("; radar scan is reported only: clutter-dominated rows and a 5-deg window cannot resolve a "
                         "~1-deg bias; see the radar_bias rows")
            elif not passes_thr:
                why = []
                if n < AZ_ADOPT_MIN_N:
                    why.append(f"n_samples<{AZ_ADOPT_MIN_N}")
                if peak < AZ_CHANCE_MULT * chance:
                    why.append(f"peak<{AZ_CHANCE_MULT:g}xchance")
                if peak < AZ_MIN_SCORE:
                    why.append(f"peak<{AZ_MIN_SCORE}")
                note += "; REJECTED (" + ",".join(why) + ")"
            else:
                why = []
                dv = abs(float(C.angle_diff_deg(value, vb)))
                if dv > AZ_CONSENSUS_DEG:
                    why.append(f"estimators disagree: object-balanced {value:+.1f} vs sample-weighted stare-free "
                               f"{vb:+.1f} ({dv:.0f} deg apart > {AZ_CONSENSUS_DEG:g})")
                if r5["k_eff"] >= 2 and sh_obj > MAX_SUPPORT_SHARE:
                    why.append(f"{sh_obj:.0%} of the support comes from one object")
                if sh_stare > MAX_SUPPORT_SHARE:
                    why.append(f"{sh_stare:.0%} of the support comes from constant-bearing runs (az spread < "
                               f"{STARE_SPREAD_DEG:g} deg)")
                if why:
                    status = "rejected"
                    note += "; REJECTED (" + "; ".join(why) + ")"
                else:
                    note += "; applied per-iteration (testing only)"
            rows.append(crow(sk, ik, "az_offset_deg", "scan", status, value, score_v, zero, chance, n, "testing", note))
            it_detail[ik] = dict(value=value, status=status, n=n, k=r5["k_eff"], raw=float(AZ_GRID[j_raw]),
                                 centre=float(AZ_GRID[j]), legacy=float(AZ_GRID[legacy_j]), vb=vb, sh_obj=sh_obj,
                                 sh_stare=sh_stare, score=score_v, zero=zero, chance=chance, n_rows=r5["n_rows"])
            if status == "applied":
                adopted_it[ik] = value
                per_iter[(sk, ik)] = value
        if sector:
            for ik in its:
                r = scan[(sk, ik)]
                rs = r["rs"]
                sc, ns = rs["score"], rs["n_eff"]
                chance = float(sc.mean())
                s0, s225 = float(sc[AZ_ZERO_IDX]), float(sc[AZ_ZERO_IDX + int(SECTOR_HALF / AZ_STEP)])
                best_off, best = (22.5, s225) if s225 >= SECTOR_GAIN * s0 else (0.0, s0)
                ok = ns >= AZ_MIN_N and best >= AZ_CHANCE_MULT * chance and best >= AZ_MIN_SCORE
                sr = r["sector_resid"]
                stat = (f"; nearest-flight az minus label: median {np.median(sr):+.1f}, p10 {np.percentile(sr, 10):+.1f}, "
                        f"p90 {np.percentile(sr, 90):+.1f} deg" if len(sr) else "")
                note = (f"sector hypothesis, robust samples, window +-{SECTOR_HALF:g} deg: score(0)={s0:.3f}, "
                        f"score(+22.5)={s225:.3f} (+22.5 needs >= {SECTOR_GAIN:g} x score(0)), n_samples={ns} "
                        f"(>= {AZ_MIN_N} needed here; the n>=50 / support rules are for the free scan){stat}; "
                        + ("applied per-iteration (testing only)" if ok else "REJECTED (below 2xchance / 0.3 / n)"))
                rows.append(crow(sk, ik, "az_offset_deg", "sector_test", "applied" if ok else "rejected", best_off,
                                 best, s0, chance, ns, "testing", note))
                if ok:
                    adopted_it[ik] = best_off
                    per_iter[(sk, ik)] = best_off
            # pooled sector score: sample-weighted average of the per-iteration scores
            ns = sum(scan[(sk, ik)]["rs"]["n_eff"] for ik in its)
            sc = sum(scan[(sk, ik)]["rs"]["score"] * scan[(sk, ik)]["rs"]["n_eff"] for ik in its) / max(ns, 1)
            s0, s225 = float(sc[AZ_ZERO_IDX]), float(sc[AZ_ZERO_IDX + int(SECTOR_HALF / AZ_STEP)])
            better = 22.5 if s225 >= SECTOR_GAIN * s0 else 0.0
            rows.append(crow(sk, "*", "az_offset_deg", "sector_test", "estimate_only", better, max(s0, s225), s0,
                             float(sc.mean()), ns, "testing",
                             f"sector hypothesis pooled (+-{SECTOR_HALF:g} deg window): score(0)={s0:.3f} vs "
                             f"score(+22.5)={s225:.3f}, only {ns} samples: reported, not applied to evaluation data"))
            detail.append(dict(sensor_key=sk, kind="sector", adopted_it=adopted_it, it=it_detail, pooled=None,
                               s0=s0, s225=s225, n=ns))
            continue
        if kind == "radar":
            rows.append(crow(sk, "*", "az_offset_deg", "scan", "rejected", nan, nan, nan, nan,
                             sum(scan[(sk, i)]["r5"]["n_eff"] for i in its), "all",
                             "radar: no pooled value from the scan; the pooled radar azimuth bias is the radar_bias row"))
            detail.append(dict(sensor_key=sk, kind=kind, adopted_it={}, it=it_detail, pooled=None, members=[]))
            continue
        # pooled sensor-level offset: largest cluster (within AZ_POOL_AGREE) of applied iterations, which must
        # also be >= 2/3 of the sensor's informative iterations (n_samples >= AZ_ADOPT_MIN_N)
        informative = [ik for ik in its if it_detail[ik]["n"] >= AZ_ADOPT_MIN_N]
        best_members = []
        for ik, v in adopted_it.items():
            mem = [i2 for i2, v2 in adopted_it.items() if abs(float(C.angle_diff_deg(v2, v))) <= AZ_POOL_AGREE]
            if len(mem) > len(best_members):
                best_members = mem
        need = max(2, int(np.ceil(AZ_POOL_FRAC * len(informative) - 1e-9)))
        n_all = sum(scan[(sk, i)]["r5"]["n_eff"] for i in its)
        if presence_only:
            why = "presence-only sensor (no drone-correlated bearing)"
        elif len(best_members) >= need:
            why = ""
        elif not adopted_it:
            why = f"no applied iteration ({len(informative)} informative)"
        elif len(best_members) < 2:
            why = f"only {len(adopted_it)} applied iteration(s); none agree within {AZ_POOL_AGREE:g} deg in >= 2"
        else:
            why = (f"only {len(best_members)} of {len(informative)} informative iterations (n_samples >= "
                   f"{AZ_ADOPT_MIN_N}) agree within {AZ_POOL_AGREE:g} deg (need >= 2/3 = {need}); evaluation iterations "
                   f"need a per-iteration check")
        if not why:
            val = circ_median([adopted_it[i] for i in best_members])
            j = int(round((val + 180.0) / AZ_STEP)) % AZ_N
            ns_m = sum(scan[(sk, i)]["r5"]["n_eff"] for i in best_members)
            sc = sum(scan[(sk, i)]["r5"]["score"] * scan[(sk, i)]["r5"]["n_eff"] for i in best_members) / max(ns_m, 1)
            excl = [short_iter(i) for i in its if i not in best_members]
            note = (f"pooled over {len(best_members)} of {len(informative)} informative iterations agreeing within "
                    f"{AZ_POOL_AGREE:g} deg ({', '.join(short_iter(i) for i in best_members)})"
                    + (f"; NOT in cluster: {', '.join(excl)}" if excl else ""))
            rows.append(crow(sk, "*", "az_offset_deg", "scan", "applied", round(val, 3), float(sc[j]),
                             float(sc[AZ_ZERO_IDX]), float(sc.mean()), ns_m, "all", note))
            pooled[sk] = val
            detail.append(dict(sensor_key=sk, kind=kind, adopted_it=adopted_it, it=it_detail, pooled=val,
                               members=best_members, informative=informative))
        else:
            if sid == "FGIX":
                why += "; FGIX az ~80 deg off in S3-I2 (per-iteration only)"
            if sid == "EMXG":
                nfl = sum(1 for v in adopted_it.values() if abs(v) > 90)
                why += (f"; EMXG flips by ~160-170 deg in some iterations ({nfl} of {len(adopted_it)} applied iterations "
                        f"are flipped, |offset| > 90 deg): evaluation iterations need a per-iteration check")
            rows.append(crow(sk, "*", "az_offset_deg", "scan", "rejected", nan, nan, nan, nan, n_all, "all",
                             "no pooled offset: " + why))
            detail.append(dict(sensor_key=sk, kind=kind, adopted_it=adopted_it, it=it_detail, pooled=None,
                               members=best_members, informative=informative, why=why))
    return rows, per_iter, pooled, detail


# ----------------------------------------------------------------------------- B. association
def associate(D: dict, by_iter: dict[str, list[Flight]], toff: dict[str, float],
              az_it: dict, shift: float = 0.0, only: np.ndarray | None = None) -> dict:
    """Per-row association for every testing detection. Rows that cannot be evaluated get an
    exclusion reason instead of a match record."""
    N = len(D["t"])
    t_cal = D["t"].copy()
    for sk, off in toff.items():
        t_cal[D["sensor_key"] == sk] += off
    t_cal = np.round(t_cal, 3) + shift            # 1 ms resolution = what det_gt_match stores
    kind = np.zeros(N, dtype=np.int8)          # 0 excluded, 1 pos, 2 bearing
    reason = np.full(N, "", dtype=object)
    is_pos = np.isin(D["meas_kind"], Q.POSITION_KINDS)
    is_bear = np.isin(D["meas_kind"], Q.BEARING_KINDS)
    kind[is_pos & D["usable_pos"] & ~D["far"]] = 1
    reason[is_pos & ~D["usable_pos"]] = "pos_not_usable"
    reason[is_pos & D["usable_pos"] & D["far"]] = "pos_far"
    bear_ok = is_bear & (D["sensor_pos_valid"] == 1) & np.isfinite(D["az_deg"])
    reason[is_bear & (D["sensor_pos_valid"] != 1)] = "bearing_no_sensor_position"
    reason[is_bear & (D["sensor_pos_valid"] == 1) & ~np.isfinite(D["az_deg"])] = "bearing_no_az"
    reason[~is_pos & ~is_bear] = "meas_kind_other"
    az_cal = np.full(N, np.nan)
    for (sk, ik), off in az_it.items():
        m = bear_ok & (D["sensor_key"] == sk) & (D["iter_key"] == ik)
        kind[m] = 2
        az_cal[m] = D["az_deg"][m] + off
    reason[bear_ok & (kind != 2)] = "bearing_sensor_unusable"
    gate = np.full(N, np.nan)
    gate[kind == 1] = POS_GATES[1]
    ae = D["az_err_deg"]
    g = np.clip(np.maximum(AZ_WINDOW, 2.0 * np.where(np.isfinite(ae), ae, 0.0)), AZ_WINDOW, 30.0)
    gate[kind == 2] = g[kind == 2]

    if only is not None:                       # chance-match runs: restrict to a row subset
        kind[~only] = 0
    best = np.full(N, -1, dtype=np.int64)
    gt_h = np.full(N, np.nan)
    resid = np.full(N, np.nan)
    second = np.full(N, np.nan)
    nin = np.zeros(N, dtype=np.int64)
    for ik, fl in by_iter.items():
        for kd in (1, 2):
            rows = np.flatnonzero((kind == kd) & (D["iter_key"] == ik))
            if len(rows) == 0:
                continue
            tt = t_cal[rows]
            M = np.full((len(rows), len(fl)), np.inf)
            if kd == 1:
                for k, f in enumerate(fl):
                    fe, fn = f.at(tt)
                    d = np.hypot(D["e"][rows] - fe, D["n"][rows] - fn)
                    M[:, k] = np.where(np.isfinite(d), d, np.inf)
            else:
                azf = flight_azimuths(fl, tt, D["sensor_e"][rows], D["sensor_n"][rows])
                d = np.abs(C.angle_diff_deg(az_cal[rows][:, None], azf))
                M = np.where(np.isfinite(d), d, np.inf)
            k_best = np.argmin(M, axis=1)
            r_best = M[np.arange(len(rows)), k_best]
            if len(fl) > 1:
                r_second = np.partition(M, 1, axis=1)[:, 1]
            else:
                r_second = np.full(len(rows), np.inf)
            act = np.isfinite(r_best)
            gids = np.array([f.gid for f in fl])
            best[rows[act]] = gids[k_best[act]]
            for k, f in enumerate(fl):
                sel = act & (k_best == k)
                if sel.any():
                    gt_h[rows[sel]] = f.height_at(tt[sel])
            resid[rows[act]] = r_best[act]
            second[rows[act]] = np.where(np.isfinite(r_second), r_second, np.nan)[act]
            nin[rows] = (M <= gate[rows][:, None]).sum(axis=1)
    return dict(t_cal=t_cal, kind=kind, reason=reason, best=best, resid=resid, second=second, nin=nin,
                gate=gate, az_cal=az_cal, gt_height=gt_h)


def gate_at(A: dict, i: int) -> np.ndarray:
    """Matching gate at sensitivity level i (0: tight, 1: main, 2: loose)."""
    return np.where(A["kind"] == 1, POS_GATES[i], A["gate"] * BEAR_GATE_SCALE[i])


def match_flags(A: dict, i: int) -> tuple[np.ndarray, np.ndarray]:
    """(matched, ambiguous) at sensitivity level i."""
    g = gate_at(A, i)
    with np.errstate(invalid="ignore"):
        matched = (A["kind"] > 0) & (A["best"] >= 0) & (A["resid"] <= g)
        amb = matched & (A["second"] <= g)
    return matched, amb


# ----------------------------------------------------------------------------- C. coverage
def coverage_arrays(D: dict, A: dict, flights: list[Flight], matched: np.ndarray, amb: np.ndarray,
                    key: str) -> dict:
    """{(gid, key_value): bool[len(flight.seconds)]}: seconds with >= 1 matched non-ambiguous
    detection of that sensor within +-1 s. `key` is the column to group by."""
    sel = np.flatnonzero(matched & ~amb)
    out = {}
    vals = D[key][sel]
    for f in flights:
        s = sel[A["best"][sel] == f.gid]
        if len(s) == 0:
            continue
        L = int(f.seconds[-1] - f.seconds[0] + 1)
        base = int(f.seconds[0])
        in_gt = np.zeros(L, dtype=bool)
        in_gt[f.seconds - base] = True
        v = D[key][s]
        tc = A["t_cal"][s]
        for kv in sorted(set(v.tolist())):
            m = v == kv
            lo = np.clip(np.ceil(tc[m] - 1.0).astype(np.int64) - base, 0, L)
            hi = np.clip(np.floor(tc[m] + 1.0).astype(np.int64) - base, -1, L - 1)
            ok = hi >= lo
            lo, hi = lo[ok], hi[ok]
            if len(lo) == 0:
                continue
            diff = np.bincount(lo, minlength=L + 1)[: L + 1] - np.bincount(hi + 1, minlength=L + 1)[: L + 1]
            cov = (np.cumsum(diff)[:L] > 0) & in_gt
            if cov.any():
                out[(f.gid, str(kv))] = cov
    return out


def codetection(flights: list[Flight], cov_id: dict, drop: frozenset = frozenset(), air_only: bool = False):
    """Per flight-second sets of distinct sensor ids. Returns (sensors, masks_by_flight) where
    masks_by_flight[gid] is an int64 array (bitmask per GT second, 0 = nobody). `air_only`: keep only the GT
    seconds with height > AIR_H_M."""
    sensors = sorted({k[1] for k in cov_id if k[1] not in drop})
    bit = {s: 1 << i for i, s in enumerate(sensors)}
    masks = {}
    for f in flights:
        L = int(f.seconds[-1] - f.seconds[0] + 1)
        m = np.zeros(L, dtype=np.int64)
        for (gid, sid), cov in cov_id.items():
            if gid == f.gid and sid in bit:
                m[cov] |= bit[sid]
        in_gt = np.zeros(L, dtype=bool)
        in_gt[f.seconds - int(f.seconds[0])] = True
        masks[f.gid] = m[in_gt & f.air_L()] if air_only else m[in_gt]
    return sensors, masks


def summarise_sets(flights: list[Flight], sensors: list[str], masks: dict) -> dict:
    """Exact and at-least counts per combo (bitmask)."""
    by_gid = {f.gid: f for f in flights}
    exact = collections.Counter()                              # mask -> seconds
    exact_fl = collections.defaultdict(collections.Counter)    # mask -> gid -> seconds
    for gid, m in masks.items():
        vals, cnt = np.unique(m, return_counts=True)
        for v, c in zip(vals.tolist(), cnt.tolist()):
            if v:
                exact[v] += c
                exact_fl[v][gid] += c
    atleast = collections.Counter()
    atleast_fl = collections.defaultdict(collections.Counter)
    for mk, c in exact.items():
        bits = [i for i in range(len(sensors)) if mk >> i & 1]
        for r in range(1, len(bits) + 1):
            for sub in combinations(bits, r):
                sm = 0
                for i in sub:
                    sm |= 1 << i
                atleast[sm] += c
                for gid, cg in exact_fl[mk].items():
                    atleast_fl[sm][gid] += cg
    nonempty = sum(int((m > 0).sum()) for m in masks.values())
    n_gt_sec = sum(len(m) for m in masks.values())
    return dict(exact=exact, exact_fl=exact_fl, atleast=atleast, atleast_fl=atleast_fl, nonempty=nonempty,
                n_gt_sec=n_gt_sec, sensors=sensors, by_gid=by_gid)


def combo_str(sensors: list[str], mk: int) -> str:
    return "|".join(sensors[i] for i in range(len(sensors)) if mk >> i & 1)


def popcount(x: int) -> int:
    return bin(x).count("1")


# ----------------------------------------------------------------------------- main
def gap_stats(flights: list[Flight]) -> list[dict]:
    """Per flight: GT gaps longer than GAP_OLD_S / GAP_MAX_S (count, total seconds, longest)."""
    out = []
    for f in flights:
        g = f.seg
        out.append(dict(label=f"{f.side[0]}:{f.name} ({short_iter(f.iter_key)})", n6=int((g > GAP_OLD_S).sum()),
                        s6=float(g[g > GAP_OLD_S].sum()), n60=int((g > GAP_MAX_S).sum()),
                        s60=float(g[g > GAP_MAX_S].sum()), longest=float(g.max()) if len(g) else 0.0))
    return out


def gap_effect(D: dict, by_iter: dict, toff: dict, az_it: dict, A: dict, matched: np.ndarray) -> dict:
    """Re-associate with the old 6 s gap rule (all else equal) and count the rows whose match changes."""
    global GAP_MAX_S
    keep, GAP_MAX_S = GAP_MAX_S, GAP_OLD_S
    try:
        A6 = associate(D, by_iter, toff, az_it)
    finally:
        GAP_MAX_S = keep
    m6, _ = match_flags(A6, 1)
    gained = matched & ~m6
    lost = m6 & ~matched
    moved = matched & m6 & (A["best"] != A6["best"])
    per = collections.Counter(D["sensor_key"][gained].tolist())
    return dict(gained=int(gained.sum()), lost=int(lost.sum()), moved=int(moved.sum()), per_sensor=per,
                matched_old=int(m6.sum()), matched_new=int(matched.sum()))


def chance_matches(D: dict, by_iter: dict, toff: dict, az_it: dict, A: dict, matched: np.ndarray, m25: np.ndarray) -> list[dict]:
    """True match rate of each radar vs the rate with the detection times shifted by +-CHANCE_SHIFTS
    (the GT track is unchanged, so any match at a shifted time is a chance coincidence)."""
    radar = (D["sensor_type"] == "RADAR") & (A["kind"] == 1)
    shifted = []
    for sh in CHANCE_SHIFTS:
        As = associate(D, by_iter, toff, az_it, shift=sh, only=radar)
        ms, _ = match_flags(As, 1)
        m25s, _ = match_flags(As, 0)
        shifted.append((sh, ms, m25s))
    out = []
    for sk in sorted(set(D["sensor_key"][radar].tolist())):
        m = radar & (D["sensor_key"] == sk)
        n = int(m.sum())
        r = dict(sensor_key=sk, n=n, true50=float((matched & m).sum() / n), true25=float((m25 & m).sum() / n),
                 shifts={sh: (float((ms & m).sum() / n), float((m25s & m).sum() / n)) for sh, ms, m25s in shifted})
        r["chance50"] = float(np.mean([v[0] for v in r["shifts"].values()]))
        r["chance25"] = float(np.mean([v[1] for v in r["shifts"].values()]))
        out.append(r)
    return out


def independent_combos(gt: dict, path: Path, spec: dict) -> dict:
    """Self-check 3 helper: recompute exact co-detection combo counts from the WRITTEN det_gt_match.csv.gz
    and the raw ground truth with plain python sets (no masks, no Flight class, no coverage_arrays).
    spec = {variant: (drop_ids, air_only)}; returns {variant: Counter(combo_string -> seconds)}."""
    import csv
    import gzip
    secs, air = {}, {}
    for ik, fk in sorted(set(zip(gt["iter_key"].tolist(), gt["flight_key"].tolist()))):
        m = (gt["iter_key"] == ik) & (gt["flight_key"] == fk)
        sec = np.unique(np.round(gt["t"][m]).astype(np.int64))
        h = Q.interp_flight(gt, m, sec.astype(float), cols=("height",))[0]
        secs[(ik, fk)] = set(sec.tolist())
        air[(ik, fk)] = {int(s) for s, hh in zip(sec.tolist(), h.tolist()) if hh > AIR_H_M}
    seen = collections.defaultdict(lambda: collections.defaultdict(set))      # (iter, flight) -> second -> {ids}
    with gzip.open(path, "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["matched"] != "1" or r["ambiguous"] != "0" or not r["best_flight"]:
                continue
            key = (r["iter_key"], r["best_flight"])
            tc = float(r["t_cal"])
            sid = r["sensor_key"].split("_")[0]
            for s in range(int(np.ceil(tc - 1.0)), int(np.floor(tc + 1.0)) + 1):
                if s in secs[key]:
                    seen[key][s].add(sid)
    out = {}
    for variant, (drop, air_only) in spec.items():
        cnt = collections.Counter()
        for key, by_sec in seen.items():
            for s, ids in by_sec.items():
                ids = ids - set(drop)
                if ids and (not air_only or s in air[key]):
                    cnt["|".join(sorted(ids))] += 1
        out[variant] = cnt
    return out


def main() -> None:
    log("loading detections (testing) and ground truth")
    D = Q.load_clean(DET_COLS)
    keep = D["tree"] == "testing"
    D = {k: v[keep] for k, v in D.items()}
    N = len(D["t"])
    gt = Q.load_groundtruth()
    flights, by_iter = build_flights(gt)
    log(f"testing detections {N:,}; GT flights {len(flights)} in {len(by_iter)} iterations")

    # sanity: our gap-aware interpolation equals Q.interp_flight wherever a flight is active
    f0 = flights[0]
    tt = np.linspace(f0.t0 - 5, f0.t1 + 5, 997)
    qe, qn = Q.interp_flight(gt, (gt["iter_key"] == f0.iter_key) & (gt["flight_key"] == f0.key), tt, cols=("e", "n"))
    fe, fn = f0.at(tt)
    act = np.isfinite(fe)
    assert np.allclose(fe[act], qe[act]) and np.allclose(fn[act], qn[act]) and np.isfinite(qe[act]).all()

    # ---- A. calibration
    log("A1. time-offset scan")
    tscan = scan_time_offsets(D, by_iter)
    cal_rows, toff, tdetail = decide_time_offsets(tscan)
    log("adopted time offsets: " + (", ".join(f"{k}={v:+.2f}s" for k, v in sorted(toff.items())) or "none"))
    log("A2. azimuth-offset scan (robust: 1-s samples, objects equal)")
    ascan = scan_azimuth(D, by_iter, toff)
    az_rows, az_it, az_pooled, adetail = decide_az_offsets(ascan)
    cal_rows += az_rows
    log("applied pooled az offsets: " + (", ".join(f"{k}={v:+.1f}" for k, v in sorted(az_pooled.items())) or "none"))

    # direct check of the interval-marking azimuth kernel on one group
    k_chk = sorted(ascan)[0]
    rows = np.flatnonzero((D["sensor_key"] == k_chk[0]) & (D["iter_key"] == k_chk[1]) & np.isfinite(D["az_deg"]))[:300]
    azf = flight_azimuths(by_iter[k_chk[1]], D["t"][rows] + toff.get(k_chk[0], 0.0), D["sensor_e"][rows], D["sensor_n"][rows])
    c_fast, n_fast = az_hits(D["az_deg"][rows], azf, AZ_WINDOW)
    okr = np.isfinite(azf).any(axis=1)
    c_ref = np.array([(np.nanmin(np.abs(C.angle_diff_deg(D["az_deg"][rows][okr][:, None] + o, azf[okr])), axis=1)
                       <= AZ_WINDOW + 1e-9).sum() if okr.any() else 0 for o in AZ_GRID])
    kernel_diff = int(np.abs(c_fast - c_ref).max())
    log(f"self-check az kernel vs direct evaluation on {k_chk}: max count difference {kernel_diff}")
    assert kernel_diff <= 2

    # ---- B. association
    log("B. association")
    A = associate(D, by_iter, toff, az_it)
    matched, amb = match_flags(A, 1)
    m25, _ = match_flags(A, 0)
    m100, _ = match_flags(A, 2)
    inc = A["kind"] > 0
    gid_arr = A["best"]
    f_side = np.array([f.side for f in flights] + [""], dtype=object)
    f_name = np.array([f.name for f in flights] + [""], dtype=object)
    f_key = np.array([f.key for f in flights] + [""], dtype=object)

    # radar azimuth bias from position-matched rows (estimate_only calibration rows)
    cal_rows += radar_bias_rows(D, A, flights, matched)

    log("B2. GT-gap rule effect and chance matches")
    gstats = gap_stats(flights)
    geff = gap_effect(D, by_iter, toff, az_it, A, matched)
    chance = chance_matches(D, by_iter, toff, az_it, A, matched, m25)

    # ---- write det_gt_match
    idx = np.flatnonzero(inc)
    o = np.lexsort((D["row"][idx], D["file"][idx], D["iter_key"][idx]))
    idx = idx[o]
    gi = np.where(gid_arr[idx] >= 0, gid_arr[idx], len(flights))
    unit = np.where(A["kind"][idx] == 1, "m", "deg").astype(object)
    gth = A["gt_height"][idx]
    match_cols = dict(
        iter_key=D["iter_key"][idx], file=D["file"][idx], row=D["row"][idx].astype(np.int64).tolist(),
        sensor_key=D["sensor_key"][idx], object_id=D["object_id"][idx], t=D["t"][idx],
        t_cal=A["t_cal"][idx], kind=np.where(A["kind"][idx] == 1, "pos", "bearing").astype(object),
        best_flight=f_key[gi], side=f_side[gi], drone_name=f_name[gi],
        resid=np.round(A["resid"][idx], 3), resid_unit=unit, second_resid=np.round(A["second"][idx], 3),
        n_in_gate=A["nin"][idx].tolist(), matched=matched[idx].astype(np.int8).tolist(),
        matched_25=m25[idx].astype(np.int8).tolist(), matched_100=m100[idx].astype(np.int8).tolist(),
        ambiguous=amb[idx].astype(np.int8).tolist(), gt_feed=D["gt_feed"][idx].astype(np.int8).tolist(),
        gt_height=np.round(gth, 2), gt_airborne=(gth > AIR_H_M).astype(np.int8).tolist())
    order = ["iter_key", "file", "row", "sensor_key", "object_id", "t", "t_cal", "kind", "best_flight", "side",
             "drone_name", "resid", "resid_unit", "second_resid", "n_in_gate", "matched", "matched_25",
             "matched_100", "ambiguous", "gt_feed", "gt_height", "gt_airborne"]
    C.write_columns(OUT / "det_gt_match.csv.gz", match_cols, order)
    log(f"wrote det_gt_match.csv.gz ({len(idx):,} rows)")

    # ---- self-check 1: every testing detection is in det_gt_match once or excluded with a reason
    n_dup = len(idx) - len({(a, b, c) for a, b, c in zip(match_cols["iter_key"], match_cols["file"], match_cols["row"])})
    excl = collections.Counter(A["reason"][~inc].tolist())
    n_excl = int((~inc).sum())
    assert (A["reason"][~inc] != "").all(), "excluded rows without a reason"
    assert len(idx) + n_excl == N and n_dup == 0
    chk1 = (f"self-check 1: {N:,} testing detections = {len(idx):,} in det_gt_match (duplicates {n_dup}) + "
            f"{n_excl:,} excluded [" + ", ".join(f"{k}={v:,}" for k, v in sorted(excl.items())) + "]")
    log(chk1)

    # ---- self-check 4: recompute residuals of a deterministic sample with Q.interp_flight
    chk4 = self_check_residuals(D, A, flights, gt)
    log(chk4)

    # ---- track labels
    track_rows = track_labels(D, A, matched, amb, flights, f_side, f_name, f_key)
    C.write_columns(OUT / "track_labels.csv", track_rows["cols"], track_rows["order"])
    n_tracks_expected = len({(a, b, c) for a, b, c in zip(D["iter_key"].tolist(), D["sensor_key"].tolist(), D["object_id"].tolist())})
    assert len(track_rows["cols"]["iter_key"]) == n_tracks_expected
    chk2 = f"self-check 2: track_labels has {n_tracks_expected:,} rows = every distinct (iter, sensor, object) of testing"
    log(chk2)

    # ---- C. coverage and co-detection
    log("C. coverage / co-detection")
    cov_key = coverage_arrays(D, A, flights, matched, amb, "sensor_key")
    cov_id = coverage_arrays(D, A, flights, matched, amb, "sensor_id")
    cov_rows = coverage_table(D, A, flights, matched, amb, cov_key, inc)
    C.write_columns(OUT / "flight_coverage.csv", cov_rows["cols"], cov_rows["order"])

    variants = {}
    for variant, drop, air_only in (("all", frozenset(), False), ("no_gt_feed", frozenset(GT_FEED), False),
                                    ("no_gt_feed_air", frozenset(GT_FEED), True)):
        sensors, masks = codetection(flights, cov_id, drop, air_only)
        variants[variant] = summarise_sets(flights, sensors, masks)
        variants[variant]["masks"] = masks
    write_sets_and_pairs(variants)

    # self-check 3: counts recomputed from the written det_gt_match + raw GT, independent of the masks
    indep = independent_combos(gt, OUT / "det_gt_match.csv.gz",
                               {"all": ((), False), "no_gt_feed": (GT_FEED, False), "no_gt_feed_air": (GT_FEED, True)})
    chk3 = []
    for variant, V in variants.items():
        total = sum(V["exact"].values())
        assert total == V["nonempty"], (variant, total, V["nonempty"])
        mine = collections.Counter({combo_str(V["sensors"], mk): c for mk, c in V["exact"].items()})
        assert mine == indep[variant], (variant, (mine - indep[variant]).most_common(3), (indep[variant] - mine).most_common(3))
        chk3.append(f"{variant}: {len(mine)} distinct combos / {total:,} seconds identical to the independent recount"
                    f" (of {V['n_gt_sec']:,} GT flight-seconds)")
    chk3 = ("self-check 3: co-detection combo counts == independent recount from det_gt_match.csv.gz + raw GT "
            "(python sets, no masks); " + "; ".join(chk3))
    log(chk3)

    # gate sensitivity
    sens = []
    for i, gname in enumerate(("tight (25 m / 0.5x gate)", "main (50 m / 1x gate)", "loose (100 m / 2x gate)")):
        mt, at = match_flags(A, i)
        ci = coverage_arrays(D, A, flights, mt, at, "sensor_id")
        sn, mk = codetection(flights, ci, frozenset())
        S = summarise_sets(flights, sn, mk)
        sn2, mk2 = codetection(flights, ci, frozenset(GT_FEED))
        S2 = summarise_sets(flights, sn2, mk2)
        sens.append(dict(name=gname, matched=mt, amb=at, S=S, S2=S2, cov_id=ci))

    # ---- calibration table: unique keys, adopted == (status == applied)
    for r in cal_rows:
        assert r["adopted"] == int(r["status"] == "applied")
    keys = [(r["sensor_key"], r["iter_key"], r["param"], r["source"]) for r in cal_rows]
    dup = [k for k, c in collections.Counter(keys).items() if c > 1]
    assert not dup, f"duplicate calibration keys: {dup[:5]}"
    chk5 = (f"self-check 5: sensor_calibration.csv has {len(cal_rows)} rows, (sensor_key, iter_key, param, source) unique, "
            f"adopted == (status == 'applied') on every row")
    log(chk5)

    write_report(D, A, flights, by_iter, matched, amb, m25, m100, cal_rows, tdetail, adetail, toff, az_pooled,
                 az_it, track_rows, cov_rows, variants, sens, excl, [chk1, chk2, chk3, chk4, chk5], kernel_diff,
                 gstats, geff, chance)
    cal_cols = dict(
        sensor_key=[r["sensor_key"] for r in cal_rows], iter_key=[r["iter_key"] for r in cal_rows],
        param=[r["param"] for r in cal_rows], source=[r["source"] for r in cal_rows],
        status=[r["status"] for r in cal_rows], value=np.array([r["value"] for r in cal_rows], float).round(4),
        score_at_value=np.array([r["score_at_value"] for r in cal_rows], float).round(5),
        score_at_zero=np.array([r["score_at_zero"] for r in cal_rows], float).round(5),
        chance=np.array([r["chance"] for r in cal_rows], float).round(5),
        n=[int(r["n"]) for r in cal_rows], adopted=[int(r["adopted"]) for r in cal_rows],
        applies_to=[r["applies_to"] for r in cal_rows], note=[r["note"] for r in cal_rows])
    sort = sorted(range(len(cal_rows)), key=lambda i: (cal_cols["sensor_key"][i], cal_cols["param"][i],
                                                       cal_cols["iter_key"][i] == "*", cal_cols["iter_key"][i],
                                                       cal_cols["source"][i], i))
    cal_cols = {k: (v[sort] if isinstance(v, np.ndarray) else [v[i] for i in sort]) for k, v in cal_cols.items()}
    C.write_columns(OUT / "sensor_calibration.csv", cal_cols, CAL_ORDER)
    log("done")
    print()
    print("SELF-CHECKS:")
    for c in (chk1, chk2, chk3, chk4, chk5):
        print("  " + c)
    print("ADOPTED TIME OFFSETS:", {k: v for k, v in sorted(toff.items())})
    print("APPLIED POOLED AZ OFFSETS:", {k: round(v, 2) for k, v in sorted(az_pooled.items())})
    print("CALIBRATION STATUS COUNTS:", dict(collections.Counter((r["param"], r["source"], r["status"]) for r in cal_rows)))
    print("REJECTED POOLED AZ (iter *):", sorted(r["sensor_key"] for r in cal_rows
                                                 if r["param"] == "az_offset_deg" and r["iter_key"] == "*"
                                                 and r["source"] == "scan" and r["status"] == "rejected"))
    print("LABEL COUNTS PER SENSOR (drone_air / drone_ground / drone_ambiguous / mixed / unmatched / no_usable_rows):")
    lc = collections.defaultdict(collections.Counter)
    for sk, lb in zip(track_rows["sensor"].tolist(), track_rows["label"].tolist()):
        lc[sk][lb] += 1
    for sk in sorted(lc):
        print(f"   {sk:22s} " + " / ".join(str(lc[sk][lb]) for lb in LABELS))
    for variant, V in variants.items():
        print(f"TOP 5 CO-DETECTION COMBOS [{variant}] (exact, by seconds):")
        for mk, c in V["exact"].most_common(5):
            print(f"   {c:6d}  {combo_str(V['sensors'], mk)}")
    for variant, V in variants.items():
        mx = max((popcount(mk) for mk in V["exact"]), default=0)
        ex = sorted(((V["exact"][mk], combo_str(V["sensors"], mk)) for mk in V["exact"] if popcount(mk) == mx), reverse=True)
        print(f"MAX DISTINCT SENSORS ON ONE DRONE IN ONE SECOND [{variant}]: {mx} -> {ex[:3]}")
    print(f"runtime {time.time() - T0:.1f}s")


def self_check_residuals(D: dict, A: dict, flights: list[Flight], gt: dict, n_sample: int = 1500) -> str:
    """Recompute resid of a deterministic sample of matched-candidate rows with Q.interp_flight
    (the controller's interpolation) and the plain geometry; they must agree."""
    rng = np.random.default_rng(20260926)
    out = []
    for kd, name in ((1, "pos"), (2, "bearing")):
        cand = np.flatnonzero((A["kind"] == kd) & (A["best"] >= 0))
        pick = rng.choice(cand, size=min(n_sample, len(cand)), replace=False)
        worst = 0.0
        for r in pick.tolist():
            f = flights[int(A["best"][r])]
            e, n = Q.interp_flight(gt, (gt["iter_key"] == f.iter_key) & (gt["flight_key"] == f.key),
                                   np.array([A["t_cal"][r]]), cols=("e", "n"))
            if kd == 1:
                v = float(np.hypot(D["e"][r] - e[0], D["n"][r] - n[0]))
            else:
                azf = float(C.azimuth_deg(e[0] - D["sensor_e"][r], n[0] - D["sensor_n"][r]))
                v = abs(float(C.angle_diff_deg(A["az_cal"][r], azf)))
            worst = max(worst, abs(v - A["resid"][r]))
        assert worst < 1e-6, (name, worst)
        out.append(f"{name} {len(pick)} rows max |resid diff| {worst:.2e}")
    return "self-check 4: resid recomputed with Q.interp_flight on a seeded sample: " + "; ".join(out)


# ----------------------------------------------------------------------------- B helpers
def radar_bias_rows(D: dict, A: dict, flights: list[Flight], matched: np.ndarray) -> list[dict]:
    """GT-minus-derived azimuth on radar rows that sit within RADAR_BIAS_MAX_RESID_M of their flight
    and at least RADAR_BIAS_MIN_RANGE_M from the sensor. Reported as az_offset_deg (the value to ADD to
    the radar az); adopted pooled only when >= 2 iterations agree within AZ_POOL_AGREE deg."""
    rows = []
    sel = (D["sensor_type"] == "RADAR") & (A["kind"] == 1) & matched & (A["resid"] <= RADAR_BIAS_MAX_RESID_M) \
        & (D["sensor_pos_valid"] == 1) & np.isfinite(D["az_deg"])
    fl_by_gid = {f.gid: f for f in flights}
    for sk in sorted(set(D["sensor_key"][sel].tolist())):
        per = {}
        for ik in sorted(set(D["iter_key"][sel & (D["sensor_key"] == sk)].tolist())):
            r = np.flatnonzero(sel & (D["sensor_key"] == sk) & (D["iter_key"] == ik))
            res = np.full(len(r), np.nan)
            rng = np.full(len(r), np.nan)
            for gid in np.unique(A["best"][r]):
                m = A["best"][r] == gid
                fe, fn = fl_by_gid[int(gid)].at(A["t_cal"][r][m])
                azg = C.azimuth_deg(fe - D["sensor_e"][r][m], fn - D["sensor_n"][r][m])
                res[m] = C.angle_diff_deg(azg, D["az_deg"][r][m])
                rng[m] = np.hypot(fe - D["sensor_e"][r][m], fn - D["sensor_n"][r][m])
            ok = np.isfinite(res) & (rng >= RADAR_BIAS_MIN_RANGE_M)
            if ok.sum() == 0:
                continue
            per[ik] = (float(np.median(res[ok])), int(ok.sum()), float(np.mean(np.abs(res[ok] - np.median(res[ok])) <= AZ_WINDOW)))
        good = {ik: v for ik, v in per.items() if v[1] >= RADAR_BIAS_MIN_N}
        for ik, (med, n, within) in per.items():
            rows.append(crow(sk, ik, "az_offset_deg", "radar_bias", "estimate_only", med, within, float("nan"),
                             float("nan"), n, "testing",
                             f"radar_bias: median(GT az - derived az) on position-matched rows (resid<=50 m, range>=300 m); "
                             f"sign: ADD to az; score_at_value = share of those rows within 5 deg of the median"
                             + ("" if n >= RADAR_BIAS_MIN_N else f"; n<{RADAR_BIAS_MIN_N}, indicative only")))
        mem = []
        for ik, v in good.items():
            cl = [i2 for i2, v2 in good.items() if abs(v2[0] - v[0]) <= AZ_POOL_AGREE]
            if len(cl) > len(mem):
                mem = cl
        if len(mem) >= 2:
            val = float(np.median([good[i][0] for i in mem]))
            rows.append(crow(sk, "*", "az_offset_deg", "radar_bias", "estimate_only", round(val, 3), float("nan"),
                             float("nan"), float("nan"), sum(good[i][1] for i in mem), "all",
                             f"radar_bias pooled over {len(mem)} agreeing iterations "
                             f"({', '.join(short_iter(i) for i in mem)}); ADD to the derived az in the fusion step; "
                             f"estimate_only: radar positions are associated by position, this az is not applied here"))
        else:
            rows.append(crow(sk, "*", "az_offset_deg", "radar_bias", "rejected", float("nan"), float("nan"),
                             float("nan"), float("nan"), sum(v[1] for v in per.values()), "all",
                             "radar_bias: no pooled value (fewer than 2 iterations with n>=100 agree within 3 deg)"))
    return rows


LABELS = ["drone_air", "drone_ground", "drone_ambiguous", "mixed", "unmatched", "no_usable_rows"]


def track_labels(D, A, matched, amb, flights, f_side, f_name, f_key) -> dict:
    """Per (iteration, sensor, object) label. best_flight / frac_best use NON-ambiguous matched rows only;
    frac_air = share of the non-ambiguous matched rows whose GT height (best flight at t_cal) is > 5 m."""
    N = len(D["t"])
    ui, ii = np.unique(D["iter_key"], return_inverse=True)
    us, si = np.unique(D["sensor_key"], return_inverse=True)
    uo, oi = np.unique(D["object_id"], return_inverse=True)
    tk = (ii.astype(np.int64) * len(us) + si) * len(uo) + oi
    utk, tinv = np.unique(tk, return_inverse=True)
    T = len(utk)
    n_total = np.bincount(tinv, minlength=T)
    inc = A["kind"] > 0
    n_det = np.bincount(tinv[inc], minlength=T)
    n_match = np.bincount(tinv[matched], minlength=T)
    un = matched & ~amb
    n_un = np.bincount(tinv[un], minlength=T)
    air = un & (A["gt_height"] > AIR_H_M)
    n_air = np.bincount(tinv[air], minlength=T)
    n_pos = np.bincount(tinv[A["kind"] == 1], minlength=T)
    n_bear = np.bincount(tinv[A["kind"] == 2], minlength=T)
    is_bear_kind = np.isin(D["meas_kind"], Q.BEARING_KINDS)
    nb_all = np.bincount(tinv[is_bear_kind], minlength=T)
    F = len(flights)
    pair = tinv[un].astype(np.int64) * F + A["best"][un]
    up, pc = np.unique(pair, return_counts=True)
    best = np.full(T, -1, dtype=np.int64)
    bcount = np.zeros(T, dtype=np.int64)
    for p, c in zip(up.tolist(), pc.tolist()):          # up is sorted -> ties keep the lowest flight id
        t_i, f_i = divmod(p, F)
        if c > bcount[t_i]:
            bcount[t_i], best[t_i] = c, f_i
    first = np.zeros(T, dtype=np.int64)
    first[tinv[::-1]] = np.arange(N)[::-1]              # first row of each track
    frac_m = np.where(n_det > 0, n_match / np.maximum(n_det, 1), np.nan)
    frac_b = np.where(n_un > 0, bcount / np.maximum(n_un, 1), np.nan)
    frac_a = np.where(n_un > 0, n_air / np.maximum(n_un, 1), np.nan)
    hi = (n_det > 0) & (frac_m >= 0.5)
    amb_dom = hi & (n_un < 0.5 * n_match)
    ok_drone = hi & ~amb_dom & (frac_b >= 0.7)
    label = np.full(T, "mixed", dtype=object)
    label[ok_drone & (frac_a >= 0.5)] = "drone_air"
    label[ok_drone & ~(frac_a >= 0.5)] = "drone_ground"
    label[amb_dom] = "drone_ambiguous"
    label[(n_det > 0) & (frac_m < 0.1)] = "unmatched"
    label[n_det == 0] = "no_usable_rows"
    kind = np.where((n_pos > 0) & (n_bear > 0), "mixed",
                    np.where(n_pos > 0, "pos", np.where(n_bear > 0, "bearing",
                                                        np.where(nb_all > 0, "bearing", "pos")))).astype(object)
    bi = np.where(best >= 0, best, F)
    rows_first = first
    gtf = D["gt_feed"][rows_first].astype(np.int8)
    cols = dict(iter_key=D["iter_key"][rows_first], sensor_key=D["sensor_key"][rows_first],
                object_id=D["object_id"][rows_first], kind=kind, n_total=n_total.tolist(), n_det=n_det.tolist(),
                n_matched=n_match.tolist(), frac_matched=np.round(frac_m, 4), best_flight=f_key[bi],
                frac_best=np.round(frac_b, 4), side=f_side[bi], drone_name=f_name[bi], label=label,
                gt_feed=gtf.tolist(), n_matched_unamb=n_un.tolist(), n_matched_air=n_air.tolist(),
                frac_air=np.round(frac_a, 4))
    order = ["iter_key", "sensor_key", "object_id", "n_det", "n_matched", "frac_matched", "best_flight",
             "frac_best", "side", "drone_name", "label", "gt_feed", "kind", "n_total", "n_matched_unamb",
             "n_matched_air", "frac_air"]
    # stable output order
    o = np.lexsort((cols["object_id"], cols["sensor_key"], cols["iter_key"]))
    cols = {k: (v[o] if isinstance(v, np.ndarray) else [v[i] for i in o]) for k, v in cols.items()}
    return dict(cols=cols, order=order, label=cols["label"], sensor=cols["sensor_key"], iter=cols["iter_key"],
                n_det=np.asarray(cols["n_det"]), best=cols["best_flight"])


def coverage_table(D, A, flights, matched, amb, cov_key, inc) -> dict:
    present = collections.defaultdict(set)              # iteration -> sensor_keys with any row
    for ik, sk in set(zip(D["iter_key"].tolist(), D["sensor_key"].tolist())):
        present[ik].add(sk)
    rows = collections.defaultdict(list)
    sel_m = np.flatnonzero(matched)
    for f in flights:
        s = sel_m[A["best"][sel_m] == f.gid]
        L_sec = len(f.seconds)
        air = f.air_L()
        n_air = int(air.sum())
        for sk in sorted(present[f.iter_key]):
            ms = s[D["sensor_key"][s] == sk]
            cov = cov_key.get((f.gid, sk))
            kinds = set(A["kind"][ms].tolist())
            unit = "deg" if kinds == {2} else "m"
            unamb = ms[~amb[ms]]
            rows["iter_key"].append(f.iter_key)
            rows["flight_key"].append(f.key)
            rows["side"].append(f.side)
            rows["drone_name"].append(f.name)
            rows["flight_duration_s"].append(round(f.t1 - f.t0, 1))
            rows["sensor_key"].append(sk)
            rows["n_matched_det"].append(int(len(ms)))
            rows["n_tracks"].append(int(len(set(D["object_id"][ms].tolist()))))
            rows["coverage_frac"].append(round(float(cov.sum() / L_sec), 4) if cov is not None else 0.0)
            rows["median_resid"].append(round(float(np.median(A["resid"][ms])), 3) if len(ms) else float("nan"))
            rows["resid_unit"].append(unit if len(ms) else "")
            rows["n_unambiguous"].append(int(len(unamb)))
            rows["n_gt_seconds"].append(int(L_sec))
            rows["airborne_seconds"].append(n_air)
            rows["coverage_frac_air"].append(round(float((cov & air).sum() / n_air), 4) if n_air and cov is not None
                                             else (0.0 if n_air else ""))
    cols = {k: (np.array(v, float) if k in ("median_resid",) else v) for k, v in rows.items()}
    order = ["iter_key", "flight_key", "side", "drone_name", "flight_duration_s", "sensor_key", "n_matched_det",
             "n_tracks", "coverage_frac", "median_resid", "resid_unit", "n_unambiguous", "n_gt_seconds",
             "airborne_seconds", "coverage_frac_air"]
    return dict(cols=cols, order=order)


def write_sets_and_pairs(variants: dict) -> None:
    sets = collections.defaultdict(list)
    pairs = collections.defaultdict(list)
    for variant, V in variants.items():
        sensors, by_gid = V["sensors"], V["by_gid"]
        combos = set(V["atleast"]) | set(V["exact"])
        ordered = sorted(combos, key=lambda m: (-V["exact"].get(m, 0), -V["atleast"][m], combo_str(sensors, m)))
        for mk in ordered:
            exact_n = V["exact"].get(mk, 0)
            gids = sorted(V["exact_fl"][mk]) if exact_n else []
            at_g = sorted(V["atleast_fl"][mk])
            sets["variant"].append(variant)
            sets["combo"].append(combo_str(sensors, mk))
            sets["size"].append(popcount(mk))
            sets["n_seconds"].append(exact_n)
            sets["n_flights"].append(len(gids))
            sets["flights"].append(";".join(by_gid[g].label for g in gids))
            sets["drones"].append(";".join(sorted({f"{by_gid[g].side}:{by_gid[g].name}" for g in gids})))
            sets["n_seconds_atleast"].append(V["atleast"][mk])
            sets["n_flights_atleast"].append(len(at_g))
        # pairs
        cnt = collections.Counter()
        for mk, c in V["exact"].items():
            for i in range(len(sensors)):
                if mk >> i & 1:
                    cnt[(i,)] += c
        for a, b in combinations(range(len(sensors)), 2):
            na, nb = cnt[(a,)], cnt[(b,)]
            both = V["atleast"].get((1 << a) | (1 << b), 0)
            if na == 0 or nb == 0:
                continue
            pairs["variant"].append(variant)
            pairs["sensor_a"].append(sensors[a])
            pairs["sensor_b"].append(sensors[b])
            pairs["n_seconds_both"].append(both)
            pairs["n_seconds_a"].append(na)
            pairs["n_seconds_b"].append(nb)
            pairs["jaccard"].append(round(both / (na + nb - both), 5))
    C.write_columns(OUT / "codetection_sets.csv", dict(sets),
                    ["variant", "combo", "size", "n_seconds", "n_flights", "flights", "drones",
                     "n_seconds_atleast", "n_flights_atleast"])
    C.write_columns(OUT / "codetection_pairs.csv", dict(pairs),
                    ["variant", "sensor_a", "sensor_b", "n_seconds_both", "n_seconds_a", "n_seconds_b", "jaccard"])


# ----------------------------------------------------------------------------- D. report
def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def write_report(D, A, flights, by_iter, matched, amb, m25, m100, cal_rows, tdetail, adetail, toff, az_pooled,
                 az_it, track_rows, cov_rows, variants, sens, excl, checks, kernel_diff, gstats, geff, chance) -> None:
    L = []
    L.append("# Ground-truth association, sensor calibration and co-detection (testing tree)\n")
    L.append("Generated by `tools/gt_associate.py` (deterministic). Only the six testing iterations have Dronetag "
             "ground truth, so every number below is measured there. Gt-feed sensors (VKYY, QTVS, LJJX) relay the "
             "Dronetag feed itself: their matches are label leakage if used to evaluate against GT, hence the "
             "`no_gt_feed` variants of every co-detection result. **Airborne** means GT `height` (above the take-off "
             "point) > 5 m at the matched time: Dronetag 'flights' include long ground periods (a Skyranger is a ground "
             "vehicle carrying a Dronetag; FPV drones are airborne 1-41 % of their file), so a 'drone' label is "
             "time-resolved (`drone_air` / `drone_ground`).\n")
    L.append("## Self-checks\n")
    for c in checks:
        L.append(f"- {c}")
    L.append(f"- azimuth interval kernel vs brute-force count difference: {kernel_diff} (<= 2 allowed)\n")

    # ---------------- calibration
    L.append("## 1. Calibration\n")
    L.append("Every row of `sensor_calibration.csv` has a `source` (`scan` | `radar_bias` | `sector_test`) and a `status`: "
             "`applied` (used by this association and valid to apply to evaluation data where `applies_to` = `all`), "
             "`estimate_only` (measured, reported, deliberately not applied here) or `rejected` (failed its criteria). "
             "`adopted` == (`status` == `applied`). `(sensor_key, iter_key, param, source)` is unique.\n")
    cnt = collections.Counter((r["param"], r["source"], r["status"]) for r in cal_rows)
    L.append(md_table(["param", "source", "status", "rows"], [[a, b, c, n] for (a, b, c), n in sorted(cnt.items())]))
    L.append("\n### 1a. Time offsets (position sensors)\n")
    L.append("Score(d) = share of usable detections (usable_pos, not far) whose nearest GT flight at t+d is within "
             "30 m; scan -6..+6 s in 0.25 s. `t_cal = t + offset`. Adopt the pooled offset iff its peak is >= 1.10 x "
             "score(0) AND the informative per-iteration argmaxes spread <= 1.0 s (informative = >= 20 hits at the "
             "peak and peak >= 1.2 x the median score of the scan; at least 2 informative iterations required when "
             "the sensor appears in >= 2 iterations). The reported argmax is the **plateau centre** (see 1d); the raw "
             "`np.argmax` is shown beside it.\n")
    rows = []
    for d in tdetail:
        per = "; ".join(f"{short_iter(ik)}:{v[0]:+.2f}" + ("" if v[1] else "?") + (f" (raw {v[2]:+.2f})" if abs(v[2] - v[0]) > 1e-9 else "")
                        for ik, v in d["per_iter"].items())
        rows.append([d["sensor_key"], f"{d['pooled']:+.2f}", f"{d['pooled_raw']:+.2f}", f"{d['peak']:.4f}", f"{d['zero']:.4f}",
                     f"{d['n']:,}", "ADOPTED" if d["ok"] else "rejected (0)", per])
    L.append(md_table(["sensor", "pooled argmax s (plateau centre)", "pooled raw argmax s", "score@peak", "score@0", "n",
                       "decision", "per-iteration argmax ('?' = not informative; raw if different)"], rows))
    L.append("\nAdopted: " + (", ".join(f"{k} {v:+.2f} s" for k, v in sorted(toff.items())) or "none") + ".\n")

    L.append("### 1b. Azimuth offsets (robust scan)\n")
    L.append("Bearing sensors use measured az, radars the derived az; scan -180..+180 in 0.5 deg. **Score** at an "
             "offset: each (object_id, 1-s time bin) is ONE sample, a hit when any of its rows has some active GT flight "
             "within 5 deg of az + offset; the score is the mean over objects of the object's hit fraction, so **objects "
             "are weighted equally**; objects with fewer than 5 samples do not vote (their hit fraction is noise; a "
             "variant that let them vote moved WVFX S3-I1 to +162 deg). The previous version counted rows, so a drone "
             "staring at one stationary drone for 1000 s outvoted everything else. chance = mean score over the scan. "
             "**Per-iteration offset applied** iff n_samples (of voting objects) >= 50, peak >= 2 x chance, peak >= 0.3, "
             "AND at the offset (after the 'indistinguishable from 0' snap: score@0 >= 0.9 x peak -> 0) no more than 50 % of "
             "the score comes from one object (checked when >= 2 objects vote) and no more than 50 % from constant-bearing "
             "runs (>= 10 consecutive samples, gaps <= 10 s, az spread < 0.5 deg), AND a cross-check estimator agrees: "
             "all samples pooled with the constant-bearing runs removed (sample-weighted) must put its own (snapped) "
             "argmax within 10 deg of the object-balanced one (the two estimators disagreeing means the iteration does not "
             "identify an offset). **Pooled offset** (applies_to = all, for evaluation data) iff >= 2 applied iterations "
             "agree within 3 deg AND they are >= 2/3 of the sensor's informative iterations (n_samples >= 50). GTYT/OEPS "
             "use the sector test instead (window +-22.5 deg, offset 0 vs +22.5, same robust samples; n_samples >= 15 "
             "there because these sensors have a handful of tracks; the support/consensus rules are not applied to it). "
             "Radar scans are reported only: clutter-dominated rows and a 5-deg window cannot resolve a ~1-deg bias.\n")
    rows = []
    for d in adetail:
        for ik, v in sorted(d.get("it", {}).items()):
            if d["kind"] == "radar":
                continue
            rows.append([d["sensor_key"], short_iter(ik), v["n"], v["k"], f"{v['centre']:+.1f}", f"{v['raw']:+.1f}",
                         f"{v['legacy']:+.1f}", f"{v['vb']:+.1f}", f"{v['value']:+.1f}", f"{v['score']:.3f}", f"{v['zero']:.3f}",
                         f"{v['chance']:.3f}", "" if v["sh_obj"] != v["sh_obj"] else f"{v['sh_obj']:.2f}",
                         "" if v["sh_stare"] != v["sh_stare"] else f"{v['sh_stare']:.2f}", v["status"]])
    L.append(md_table(["sensor", "iter", "n samples (voting objects)", "voting objects", "plateau-centre argmax", "raw argmax", "legacy row-weighted argmax",
                       "cross-check: sample-weighted, stares removed", "offset used", "score@offset", "score@0", "chance", "top-object share", "constant-bearing share",
                       "status"], rows))
    L.append("\n(sector sensors: see the `sector_test` rows of the csv; the offset used there is 0 or +22.5.)\n")
    L.append("#### Pooled (evaluation-data) azimuth decisions\n")
    rows = []
    for r in sorted((r for r in cal_rows if r["param"] == "az_offset_deg" and r["iter_key"] == "*" and r["source"] != "radar_bias"),
                    key=lambda r: r["sensor_key"]):
        rows.append([r["sensor_key"], r["source"], r["status"], "" if r["value"] != r["value"] else f"{r['value']:+.2f}",
                     r["n"], r["note"]])
    L.append(md_table(["sensor", "source", "status", "value deg", "n", "note"], rows))
    L.append("")
    # before / after
    L.append("#### Before / after the robust scan (constant-bearing stares)\n")
    prev = {("WVFX_CAMERA", "testing/S1-I1"): "+49.0 applied", ("NJQL_CAMERA", "testing/S3-I1"): "-119.5 applied"}
    rows = []
    byk = {d["sensor_key"]: d for d in adetail}
    for sk, ik in (("WVFX_CAMERA", "testing/S1-I1"), ("NJQL_CAMERA", "testing/S3-I1")):
        v = byk[sk]["it"][ik]
        rows.append([sk, short_iter(ik), prev[(sk, ik)], f"{v['legacy']:+.1f} (row-weighted rescan)",
                     f"{v['centre']:+.1f} (raw {v['raw']:+.1f})", f"{v['value']:+.1f} {v['status']}",
                     f"{v['sh_obj']:.2f} / {v['sh_stare']:.2f}", f"{v['score']:.3f} vs {v['zero']:.3f}@0"])
    L.append(md_table(["sensor", "iter", "previous version (rows)", "legacy argmax now", "robust argmax", "robust result",
                       "top-object / constant-bearing share", "robust score@offset vs @0"], rows))
    L.append("")
    emx = byk.get("EMXG_ACOUSTIC")
    if emx:
        rows = [[short_iter(ik), v["n"], v["k"], f"{v['centre']:+.1f} (raw {v['raw']:+.1f})", f"{v['value']:+.1f}", v["status"]]
                for ik, v in sorted(emx["it"].items())]
        L.append("EMXG per iteration (flips must survive where real; S4-I4 rested on n=28 rows and has "
                 + ", ".join(f"{v['n']}" for ik, v in sorted(emx["it"].items()) if ik.endswith("S4-I4"))
                 + " samples here, so it is below the n >= 50 adoption bar and is rejected; the evaluation tree cannot "
                 "use a per-iteration EMXG offset for it either):\n")
        L.append(md_table(["iter", "n samples", "objects", "robust argmax", "offset used", "status"], rows))
        L.append("")
    L.append("#### Radar azimuth bias from position-matched rows (`source = radar_bias`, `status = estimate_only`)\n")
    L.append("These are the offsets the fusion step should ADD to a radar's derived azimuth (radar rows are associated by "
             "position, so the bias is not applied here). Filter: `source == radar_bias`, `iter_key == *` for the pooled value.\n")
    rows = []
    for r in sorted((r for r in cal_rows if r["source"] == "radar_bias"),
                    key=lambda r: (r["sensor_key"], r["iter_key"] == "*", r["iter_key"])):
        rows.append([r["sensor_key"], short_iter(r["iter_key"]) if r["iter_key"] != "*" else "*",
                     "" if r["value"] != r["value"] else f"{r['value']:+.2f}", r["n"], r["status"]])
    L.append(md_table(["sensor", "iter", "offset to ADD deg (= -bias)", "n", "status"], rows))
    L.append("\nPer-iteration az offsets applied by the association (testing only): "
             + (", ".join(f"{k[0].split('_')[0]}@{short_iter(k[1])} {v:+.1f}" for k, v in sorted(az_it.items())) or "none") + ".\n")

    L.append("### 1c. GT-gap rule\n")
    L.append(f"Dronetag logs have gaps (up to 741 s). Interpolating across a gap invents a track, so a flight is treated as "
             f"*not active* inside any GT gap longer than **{GAP_MAX_S:g} s** (it was {GAP_OLD_S:g} s; a 49 s Quantum "
             f"Vector gap removed ~777 genuine LJJX rows). Gaps <= {GAP_MAX_S:g} s are linearly interpolated. Effect, "
             f"re-associating with the old {GAP_OLD_S:g} s rule and everything else equal: "
             f"**{geff['gained']:,} rows are matched now that were not** (matched rows {geff['matched_old']:,} -> "
             f"{geff['matched_new']:,}), {geff['lost']:,} rows lose a match, {geff['moved']:,} rows change best flight.\n")
    top = geff["per_sensor"].most_common(8)
    L.append("Rows newly matched per sensor: " + (", ".join(f"{k} {v:,}" for k, v in top) or "none") + ".\n")
    rows = [[g["label"], g["n6"], f"{g['s6']:.0f}", g["n60"], f"{g['s60']:.0f}", f"{g['longest']:.0f}"]
            for g in gstats if g["n6"]]
    L.append(md_table(["flight", f"gaps > {GAP_OLD_S:g} s", "their total s", f"gaps > {GAP_MAX_S:g} s (inactive)",
                       "their total s", "longest gap s"], rows))
    L.append("\nGT seconds are the seconds that have a GT sample, so seconds inside a gap are never counted in coverage or "
             "co-detection, whichever rule is used.\n")

    L.append("### 1d. Plateau-centre argmax rule\n")
    L.append(f"Scores are flat-topped (30 m / 5 deg gates, slow drones), so the exact `np.argmax` is an arbitrary edge of "
             f"the plateau. Every time/azimuth offset reported as 'argmax' is the **centre of the contiguous run of scan "
             f"points with score >= {PLATEAU_FRAC:g} x the peak** around the exact argmax (azimuth runs wrap). The raw "
             f"argmax (first maximal point) is shown next to it in 1a/1b and in the csv notes (full per-row reasoning is in the `note` column).\n")

    L.append("### 1e. Chance matches (radars)\n")
    L.append("Each radar's usable position rows are re-matched with their times shifted by -60 s and +60 s (GT unchanged); "
             "any match is then a chance coincidence of a clutter/other-object detection with a drone position. Compared "
             "with the true match rate at the same gate (precision@gate, section 2). Caveat: a hovering drone would match "
             "after a shift too, which makes this an upper bound on the chance rate.\n")
    rows = []
    for r in chance:
        rows.append([r["sensor_key"], f"{r['n']:,}", f"{r['true50']:.4f}",
                     f"{r['shifts'][-60.0][0]:.4f} / {r['shifts'][60.0][0]:.4f}", f"{r['chance50']:.4f}",
                     f"{r['true50'] / max(r['chance50'], 1e-9):.1f}x", f"{r['true25']:.4f}", f"{r['chance25']:.4f}"])
    L.append(md_table(["radar", "usable rows", "true rate @50 m", "shifted -60 s / +60 s @50 m", "chance rate @50 m (mean)",
                       "true / chance", "true @25 m", "chance @25 m"], rows))
    L.append("")

    # ---------------- precision
    L.append("## 2. Per-sensor precision (share of usable rows matched to a drone, gate 50 m / bearing gate)\n")
    sks = sorted(set(D["sensor_key"].tolist()))
    rows = []
    for sk in sks:
        m = D["sensor_key"] == sk
        inc = m & (A["kind"] > 0)
        n_in = int(inc.sum())
        n_m = int((matched & m).sum())
        n_a = int((amb & m).sum())
        kinds = "bearing" if (A["kind"][inc] == 2).any() else "pos"
        rows.append([sk, "gt_feed" if sk.split("_")[0] in GT_FEED else "", int(m.sum()), n_in, n_m,
                     f"{n_m / n_in:.3f}" if n_in else "", n_a,
                     f"{(m25 & m).sum() / n_in:.3f}" if n_in else "", f"{(m100 & m).sum() / n_in:.3f}" if n_in else "",
                     kinds if n_in else "-"])
    L.append(md_table(["sensor", "note", "rows", "usable rows", "matched", "precision@gate", "ambiguous",
                       "prec@tight", "prec@loose", "kind"], rows))
    L.append("\nExcluded rows (not in det_gt_match): " + ", ".join(f"{k}={v:,}" for k, v in sorted(excl.items())) + ".\n")
    L.append("Ambiguous (>= 2 flights inside the gate) detections are matched but never count toward coverage / "
             "co-detection. Tight/loose gates: position 25 / 100 m, bearing 0.5x / 2x the angular gate.\n")

    # ---------------- recall matrix
    L.append("## 3. Per-flight recall (coverage_frac = share of the flight's GT seconds with a matched, "
             "non-ambiguous detection of that sensor within +-1 s)\n")
    cc = cov_rows["cols"]
    by_flight = collections.defaultdict(dict)
    for i in range(len(cc["iter_key"])):
        by_flight[(cc["iter_key"][i], cc["flight_key"][i])][cc["sensor_key"][i]] = cc["coverage_frac"][i]
    sid_all = sorted({cc["sensor_key"][i] for i in range(len(cc["iter_key"]))})
    head = ["iter", "flight", "dur s"] + [s.split("_")[0] for s in sid_all]
    rows = []
    fl_by_key = {(f.iter_key, f.key): f for f in flights}
    for (ik, fk), d in sorted(by_flight.items()):
        f = fl_by_key[(ik, fk)]
        rows.append([short_iter(ik), f"{f.side[0]}:{f.name} {fk.split('_')[3]}-{fk.split('_')[4]}", int(f.t1 - f.t0)]
                    + [(f"{d[s]:.2f}" if s in d and d[s] > 0 else ("." if s in d else "")) for s in sid_all])
    L.append(md_table(head, rows))
    L.append("\n`.` = sensor present in that iteration, zero coverage; blank = sensor absent from the iteration.\n")
    L.append("### 3b. Airborne recall (coverage over the GT seconds with height > 5 m only)\n")
    air_by = {}
    for i in range(len(cc["iter_key"])):
        air_by.setdefault((cc["iter_key"][i], cc["flight_key"][i]), (cc["airborne_seconds"][i], cc["n_gt_seconds"][i]))
    cfa = collections.defaultdict(dict)
    for i in range(len(cc["iter_key"])):
        v = cc["coverage_frac_air"][i]
        cfa[(cc["iter_key"][i], cc["flight_key"][i])][cc["sensor_key"][i]] = v
    rows = []
    for (ik, fk), d in sorted(cfa.items()):
        f = fl_by_key[(ik, fk)]
        na, ng = air_by[(ik, fk)]
        rows.append([short_iter(ik), f"{f.side[0]}:{f.name} {fk.split('_')[3]}-{fk.split('_')[4]}", ng, na,
                     f"{na / ng:.2f}"] + [(f"{d[s]:.2f}" if s in d and d[s] != "" and d[s] > 0 else ("." if s in d and d[s] != "" else ""))
                                           for s in sid_all])
    L.append(md_table(["iter", "flight", "GT s", "airborne s", "air share"] + [s.split("_")[0] for s in sid_all], rows))
    never = [r[1] for r in rows if r[3] == 0]
    L.append("\nFlights never above 5 m (no airborne seconds): " + (", ".join(f"{r[0]} {r[1]}" for r in rows if r[3] == 0) or "none") + ".\n")

    # ---------------- drones seen by multiple sensors
    L.append("## 4. Drones seen by multiple distinct sensors (coverage_frac >= 0.05)\n")
    rows = []
    for (ik, fk), d in sorted(by_flight.items()):
        f = fl_by_key[(ik, fk)]
        seen = sorted(s.split("_")[0] for s, v in d.items() if v >= COV_MIN_FRAC)
        seen_ng = [s for s in seen if s not in GT_FEED]
        rows.append([short_iter(ik), f"{f.side}:{f.name} ({fk.split('_')[3]}-{fk.split('_')[4]})", len(seen), len(seen_ng),
                     ", ".join(seen)])
    L.append(md_table(["iter", "flight", "# sensors", "# excl. gt-feed", "sensors"], rows))
    L.append("")

    # ---------------- intersections
    L.append("## 5. Biggest co-detection intersections\n")
    L.append("S = set of distinct sensor ids with a matched non-ambiguous detection of the same flight within +-1 s "
             "of a whole GT second. `exact` counts seconds whose S equals the combo; `at-least` counts seconds in "
             "which at least these sensors co-detect (superset counting over every observed S).\n")
    L.append("Variants: `all` (every sensor), `no_gt_feed` (VKYY, QTVS, LJJX removed: no label leakage), `no_gt_feed_air` "
             "(also restricted to the GT seconds with height > 5 m: co-detection of drones that are actually flying).\n")
    for variant, V in variants.items():
        sensors = V["sensors"]
        L.append(f"### Variant `{variant}` ({V['nonempty']:,} of {V['n_gt_sec']:,} GT flight-seconds have a non-empty S; "
                 f"{len(V['exact'])} distinct S)\n")
        L.append("Top 15 exact sets by seconds:\n")
        rows = []
        for mk, c in V["exact"].most_common(15):
            gids = sorted(V["exact_fl"][mk])
            rows.append([combo_str(sensors, mk).replace("|", "+"), popcount(mk), c, len(gids),
                         "; ".join(sorted({f"{V['by_gid'][g].side[0]}:{V['by_gid'][g].name}" for g in gids}))[:90]])
        L.append(md_table(["combo", "size", "seconds", "flights", "drones"], rows))
        L.append("\nTop 15 combos per size by at-least seconds (superset counting), sizes 2..max:\n")
        maxsz = max((popcount(m) for m in V["exact"]), default=0)
        for k in range(2, maxsz + 1):
            cand = sorted(((c, m) for m, c in V["atleast"].items() if popcount(m) == k),
                          key=lambda x: (-x[0], combo_str(sensors, x[1])))[:15]
            if not cand:
                continue
            L.append(f"**size {k}** (best exact: " + (lambda b: f"{combo_str(sensors, b[1]).replace('|', '+')} "
                                                       f"{b[0]} s" if b else "none")(
                max(((V['exact'][m], m) for m in V['exact'] if popcount(m) == k), default=None, key=lambda x: x[0] if x else 0)) + ")\n")
            rows = [[combo_str(sensors, m).replace("|", "+"), c, V["exact"].get(m, 0), len(V["atleast_fl"][m])]
                    for c, m in cand]
            L.append(md_table(["combo", "at-least seconds", "exact seconds", "flights"], rows))
            L.append("")
        mx = [(V["exact"][m], combo_str(sensors, m)) for m in V["exact"] if popcount(m) == maxsz]
        L.append(f"Maximum distinct sensors co-detecting one drone in one second: **{maxsz}** "
                 f"({'; '.join(f'{c} s: {s}' for c, s in sorted(mx, reverse=True)[:5])}).\n")
    # pairs
    L.append("### Strongest pairs (Jaccard over flight-seconds, variant `all`)\n")
    V = variants["all"]
    sensors = V["sensors"]
    cnt1 = collections.Counter()
    for mk, c in V["exact"].items():
        for i in range(len(sensors)):
            if mk >> i & 1:
                cnt1[i] += c
    pr = []
    for a, b in combinations(range(len(sensors)), 2):
        both = V["atleast"].get((1 << a) | (1 << b), 0)
        if both:
            pr.append((both / (cnt1[a] + cnt1[b] - both), both, sensors[a], sensors[b], cnt1[a], cnt1[b]))
    pr.sort(reverse=True)
    L.append(md_table(["pair", "jaccard", "seconds both", "seconds a", "seconds b"],
                      [[f"{a}+{b}", f"{j:.3f}", bo, na, nb] for j, bo, a, b, na, nb in pr[:20]]))
    L.append("")

    # ---------------- track labels
    L.append("## 6. Track labels per sensor\n")
    cnt = collections.defaultdict(collections.Counter)
    for sk, lb in zip(track_rows["sensor"].tolist(), track_rows["label"].tolist()):
        cnt[sk][lb] += 1
    rows = [[sk] + [cnt[sk][lb] for lb in LABELS] + [sum(cnt[sk].values())] for sk in sorted(cnt)]
    tot = [sum(cnt[sk][lb] for sk in cnt) for lb in LABELS]
    rows.append(["**all**"] + tot + [sum(tot)])
    L.append(md_table(["sensor"] + LABELS + ["tracks"], rows))
    L.append("\nLet frac_matched = matched / usable rows (ambiguous matches included), n_unamb = non-ambiguous matched rows, "
             "frac_best = share of n_unamb on the best flight (best flight and frac_best come from NON-ambiguous rows only), "
             "frac_air = share of n_unamb whose GT height at t_cal is > 5 m. `unmatched`: frac_matched < 0.1; "
             "`drone_ambiguous`: frac_matched >= 0.5 but n_unamb < 50 % of the matched rows (two flights inside the gate, "
             "the flight cannot be told); `drone_air` / `drone_ground`: frac_matched >= 0.5, n_unamb >= 50 % of matched, "
             "frac_best >= 0.7 and frac_air >= 0.5 / < 0.5; every other track with usable rows is `mixed`; `no_usable_rows`: "
             "every row of the track was excluded.\n")

    # ---------------- sensitivity
    L.append("## 7. Gate sensitivity (position 25 / 50 / 100 m; bearing 0.5x / 1x / 2x gate)\n")
    rows = []
    for sk in sks:
        m = D["sensor_key"] == sk
        n_in = int((m & (A["kind"] > 0)).sum())
        if not n_in:
            continue
        rows.append([sk] + [f"{(s['matched'] & m).sum() / n_in:.3f}" for s in sens] + [n_in])
    L.append(md_table(["sensor"] + [s["name"] for s in sens] + ["usable rows"], rows))
    L.append("")
    rows = []
    for s in sens:
        for variant, S in (("all", s["S"]), ("no_gt_feed", s["S2"])):
            mx = max((popcount(m) for m in S["exact"]), default=0)
            top = S["exact"].most_common(1)
            multi = 0
            per_flight = collections.defaultdict(set)
            for (gid, sid), cov in s["cov_id"].items():
                if (variant == "all" or sid not in GT_FEED) and cov.sum() / len(S["by_gid"][gid].seconds) >= COV_MIN_FRAC:
                    per_flight[gid].add(sid)
            multi = sum(1 for g, v in per_flight.items() if len(v) >= 2)
            rows.append([s["name"], variant, S["nonempty"], mx, multi,
                         (combo_str(S["sensors"], top[0][0]).replace("|", "+") + f" ({top[0][1]} s)") if top else ""])
    L.append(md_table(["gate", "variant", "flight-seconds with S != empty", "max sensors", "flights with >= 2 sensors (cov>=5%)",
                       "top exact combo"], rows))
    L.append("")
    (OUT / "gt_association.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
