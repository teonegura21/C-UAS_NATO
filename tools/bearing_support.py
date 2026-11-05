#!/usr/bin/env python3
"""Bearing support: does a camera / acoustic / RF direction-finding sensor point at this
position track?  Plus a ground-truth-free azimuth registration of every bearing sensor.

Bearing-only sensors report an azimuth (degrees clockwise from north) and for some an
elevation, seen from a known sensor position: cameras WVFX / FGIX, acoustic EMXG, passive-RF
direction finders ORFH / YXAB / OEPS / GTYT.  Their azimuths carry per-iteration offsets
(WVFX +5 deg in testing S1-I1, EMXG flipped by -156 / -172 deg in S3-I2 / S3-I3, FGIX -83.5 deg in
S3-I2).  data/derived/sensor_calibration.csv has them, but made WITH ground truth, so it cannot
exist for evaluation.  This tool estimates the offset from the position tracks alone and then
asks, for every position-track segment of data/derived/sensor_tracks_v2/, how many bearing
reports point at it.

Inputs (read only)
* detections: cuas_clean.load_clean, bearing rows (meas_kind in Q.BEARING_KINDS) that are not
  cuas_clean 'duplicate' rows, with a valid sensor position and a finite azimuth.  Only sensors
  of deployable ids (C.deployable_sensor_ids()) are used: NJQL (testing-only camera) is dropped
  completely, it never gets a registration row or a feature.
* existence probability p: existence/<slug>__p.csv (own iteration) and, for the nested LOIO choices, __p_inner.csv.
* position tracks: sensor_tracks_v2/<slug>__tracks.csv + __grid.npz (Kalman/RTS-smoothed grid
  samples, local ENU metres).  No other file feeds the registration.
* sensor_calibration.csv: ONLY for the comparison columns gt_offset_deg / agree and self-check
  3.  Never an input of an estimate.

Registration (per iteration and bearing sensor_key; own iteration's data only)
* candidates: for every bearing report, every position-segment grid sample at the nearest grid
  step (|t_report - t_grid| <= 0.05 s) whose horizontal distance to the bearing sensor is within
  [REG_RANGE_MIN_M, RANGE_MAX_M], from a REGISTRATION POOL segment: sensor id deployable and not a
  ground-truth relay (VKYY, QTVS, LJJX = Q.GT_FEED_SENSORS), so EUIG / MMJJ / LJJX never enter.
* residual r = wrap180(az_meas - az_pred), az_pred = atan2(e - e_s, n - n_s) in degrees.
* histogram of r in 1-degree bins, circularly smoothed (triangular kernel, half-width SMOOTH_HW).
  The brief's plain weight (1 / number of candidates of the report) gives significance 3-5 and a wrong
  peak in most testing rows, because ~90 % of the 50-100 candidates per report are radar clutter, so
  the delivered pair weight is
      w = p_s / sum(p over the report's candidates)      p = existence p_drone_all of the segment
          x 1 / (reports of this sensor in the same second)   (a sensor-second counts once)
          x 1 / (1 + (sigma_ang / prec_sigma0)^2)    sigma_ang = track sd / range, in degrees
  and one segment may supply at most seg_cap to any single bin (a parked pair or a 15 s coincidence
  cannot carry the peak).  existence p is a leave-one-iteration-out prediction on testing and the
  final model's on evaluation, i.e. it never saw the registered iteration's labels.  The plain-weight
  estimate is still computed (registration.csv plain_offset_deg / plain_significance, not used).
* peak b = argmax of the smoothed histogram, refined by the centroid of the raw histogram within
  +-CENTROID_HW bins and then by MS_ITER mean-shift steps over +-ms_hw degrees (averages the position
  sensors' independent 10-25 m biases).  offset_deg = -b: the value to ADD to the sensor azimuth
  (same sign as sensor_calibration.csv).
* significance = smoothed peak / max(median smoothed bin, SIG_FLOOR_SHARE x total weight / 360 x kernel).
  second_ratio = best smoothed bin more than SECOND_EXCL_DEG from the peak / the peak.
  Accepted when significance >= sig_min, reports with a candidate >= MIN_REPORTS, the peak (+-PEAK_WIN_DEG)
  is carried by >= MIN_SEG_PEAK segments from >= MIN_SENS_PEAK position sensors, second_ratio <= max_second
  (an ambiguous histogram is no registration) and the TIME-BLOCK test holds: the sensor's reports are cut into
  SPLIT_K blocks of equal report counts and each block is scanned alone; a block with >= SPLIT_MIN_REPORTS reports
  and its own significance >= SPLIT_SIG whose peak is more than SPLIT_TOL_DEG from the full-data offset contradicts
  it (split mode 2: none may contradict, 1: at most one, 0: test off).  A per-iteration constant offset must show up
  everywhere (testing S1-I1 WVFX does not: about -113 early, -10 later).  Otherwise offset 0 ('unregistered').
* ORFH and YXAB (PRESENCE_ONLY_IDS: their bearing does not follow drones, brief and sensor_calibration.csv) are
  scanned and reported but never accepted, so their offset is always 0.
* Coarse-sector RF sensors (GTYT, OEPS; az_err 25 deg, one sector per report) are never scanned:
  only offset 0 is tested.  share(o) = share of reports with a candidate of existence p >= SECTOR_P_MIN
  within the gate after adding o; significance = share(0) / median share over o in steps of
  SECTOR_STEP_DEG; accepted when >= SECTOR_SIG_MIN, fewer than SECTOR_RANK_MAX offsets beat offset 0 and
  >= SECTOR_MIN_REPORTS reports.  Offset is 0 either way; the flag only says whether zero got support.
* LOIO choices (choose_fold; loio_choices.csv): the estimator shape (seg_cap, prec_sigma0, ms_hw; EST_GRID), the
  acceptance rule (sig_min, max_second, split mode; ACC_GRID) and the azimuth gate floors (GATE_GRID) applied to a
  testing iteration k are chosen on the OTHER testing iterations: shape + rule maximise sum(+1 accepted and within
  AGREE_DEG of a GT row, -2 accepted and wrong, 0 unaccepted) over their sensor_calibration.csv rows (n >= GT_MIN_N)
  with each training iteration's p_inner for fold k (a model that saw neither k nor it); gates maximise the pooled
  drone-vs-clutter AUC of hit_share (four radars, covered segments) with the offsets that rule gives those training
  iterations.  Ties go to the a-priori default (first grid entry).  Evaluation iterations use the choice made on all
  six testing iterations (nothing is ever fitted on evaluation).  NOT chosen LOIO: the constants in the header of the
  file (range limits, elevation gates, K_SIGMA, MIN_*, block parameters, SECTOR_*) and the grids; they are physical or
  a-priori, read off pooled testing residuals.

Support features (per position segment; offsets applied, accepted or 0)
* a (report, segment) pair exists when the segment has a grid sample at the report's nearest step
  within RANGE_MAX_M of the sensor.  HIT: |r| <= gate and, where the report has an elevation and the
  segment a height, |el_meas - el_pred| <= EL_GATE_DEG[family].  gate = max(K_SIGMA x az_err_deg,
  AZ_GATE_<family>) per report (az_err blank -> the family constant).
* per segment and family f in (camera, acoustic, rf), summed over the sensors of the family:
    f_n_reports_overlap   pairs (reports of the family while the segment has a sample in range)
    f_n_hits, f_hit_share = hits / overlap (NaN without overlap)
    f_med_abs_r           median |r| over hits, degrees
    f_n_unique_hits       hits whose segment is the angularly closest candidate of that report
                          (smallest |r| among the deployable-sensor segments; ties: lowest seg_id;
                          a segment of a testing-only sensor counts as closest only when strictly
                          closer than all deployable ones, so testing mimics evaluation)
    f_unique_hit_share    f_n_unique_hits / overlap
  camera_uav_share: share of the camera hits whose report class text (cls / cls_sub) contains
    'uav' or 'drone'.  rf_band / rf_protocol / rf_model: most common metadata_map.map_value string
    over the RF hit reports (exact dropdown entries or ''; ties: alphabetical).
* The azimuth gate floors are LOIO choices (above); the other constants are fixed.

Outputs (data/derived/bearing_support/)
  <slug>__support.csv  seg_id, iter_key, sensor_key + the feature columns above, one row per
                       segment of the iteration's tracks.csv
  registration.csv     iter_key, sensor_key, offset_deg, significance, n_reports, accepted,
                       gt_offset_deg (testing 'applied' rows of sensor_calibration.csv, else blank),
                       agree (accepted offset within AGREE_DEG of gt; blank when unaccepted or no gt),
                       eff_agree (offset actually used, 0 if unaccepted, within AGREE_DEG of gt), raw_offset_deg,
                       status (accepted / unregistered / presence_only), blocks (n:peak:significance per time block),
                       n_blocks_contradict, az_spread_deg (info), config (which LOIO choice), method, n_pairs,
                       peak_share, n_seg_peak, n_sens_peak, src_sensors / src_iters (MEASURED from the candidate pairs)
  loio_choices.csv     the chosen estimator shape, acceptance rule and gate floors per fold
  bearing_support.md   plain-language report

Run: python3 tools/bearing_support.py     (about 1 minute; numbered self-checks, exit 1 on failure)
"""

from __future__ import annotations

import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuas_clean as Q  # noqa: E402
import cuas_common as C  # noqa: E402
import gbdt  # noqa: E402
import metadata_map as MM  # noqa: E402
import truth as T  # noqa: E402

OUT = C.DERIVED / "bearing_support"
TRACKS = C.DERIVED / "sensor_tracks_v2"
CALIBRATION = C.DERIVED / "sensor_calibration.csv"
EXISTENCE = C.DERIVED / "existence"

FAMILY_OF_ID = {"WVFX": "camera", "FGIX": "camera", "NJQL": "camera", "EMXG": "acoustic",
                "ORFH": "rf", "YXAB": "rf", "OEPS": "rf", "GTYT": "rf"}
FAMILIES = ("camera", "acoustic", "rf")
SECTOR_IDS = frozenset({"GTYT", "OEPS"})

RANGE_MAX_M = 3000.0        # candidates / support: farther segments are out of a Class I UAS detection reach
REG_RANGE_MIN_M = 100.0     # registration only: below this a 10-25 m position bias is > 6 deg
GRID_TOL_S = 0.05
SMOOTH_HW = 3               # triangular kernel half-width, 1-degree bins
CENTROID_HW = 3
SIG_FLOOR_SHARE = 1.0e-3    # of the total weight, floor of the null level (sparse histograms)
MIN_REPORTS = 50            # reports with >= 1 candidate
MIN_SEG_PEAK = 3            # distinct position segments carrying the peak ...
MIN_SENS_PEAK = 2           # ... from at least this many distinct position sensors (corroboration)
MIN_SEG_W = 0.5             # a segment carries the peak when its capped weight within PEAK_WIN_DEG is >= this
PEAK_WIN_DEG = 4.0
SECOND_EXCL_DEG = 15           # a rival peak must lie farther than this from the main one
MS_ITER = 10
SPLIT_K = 3                 # time blocks (equal report counts) of the temporal-consistency test
SPLIT_MIN_REPORTS = 40      # a block is testable with at least this many reports with a candidate ...
SPLIT_SIG = 10.0            # ... and its own peak significance at least this
SPLIT_TOL_DEG = 20.0        # a testable block contradicts the full-data offset when its peak is farther than this
SECTOR_STEP_DEG = 5.0
SECTOR_P_MIN = 0.3          # sector test: only candidates with existence p_drone_all >= this count
SECTOR_SIG_MIN = 1.5        # share(0) / median share over offsets
SECTOR_RANK_MAX = 9         # and fewer than this many of the 72 offsets may beat offset 0
SECTOR_MIN_REPORTS = 30
K_SIGMA = 1.0
EL_GATE_DEG = {"camera": 15.0, "acoustic": 30.0, "rf": 90.0}
AGREE_DEG = 3.0
GT_MIN_N = 50
SEED = 0
# Sensors whose bearing does not track any drone (brief; sensor_calibration.csv "presence-only"): the scan
# is still run and reported, but it is never accepted, so their offset is always 0.
PRESENCE_ONLY_IDS = frozenset({"ORFH", "YXAB"})

# Choices made leave-one-iteration-out (see choose_fold): estimator shape, acceptance rule, azimuth gate floors.
# The FIRST entry of each grid is the a-priori default (ties go to the entry closest to it, i.e. the earliest).
EST_DEFAULT = dict(seg_cap=3.0, prec_sigma0=2.0, ms_hw=6.0)
EST_GRID = [EST_DEFAULT] + [dict(seg_cap=c, prec_sigma0=p, ms_hw=m) for c in (3.0, 1.0, 10.0) for p in (2.0, 5.0) for m in (6.0, 3.0)
                            if not (c == 3.0 and p == 2.0 and m == 6.0)]
ACC_DEFAULT = dict(sig_min=10.0, max_second=0.75, split=2)      # split: 0 off, 1 at most one contradicting block, 2 none
ACC_GRID = [ACC_DEFAULT] + [dict(sig_min=s, max_second=m, split=sp) for s in (10.0, 5.0, 20.0) for m in (0.75, 0.5, 1.01) for sp in (2, 1, 0)
                            if not (s == 10.0 and m == 0.75 and sp == 2)]
GATE_DEFAULT = {"camera": 5.0, "acoustic": 10.0, "rf": 10.0}    # azimuth gate floor (deg): gate = max(K_SIGMA x az_err, floor)
GATE_GRID = {"camera": (5.0, 3.0, 8.0, 12.0, 20.0), "acoustic": (10.0, 5.0, 15.0, 25.0), "rf": (10.0, 5.0, 20.0, 40.0)}
UTIL_RIGHT, UTIL_WRONG = 1.0, -2.0      # selection utility of an accepted offset that agrees / does not agree with the GT row
AUC_GROUP = ("EGOI", "NYTF", "UXYJ", "IUKL")

SUP_COLS = ["n_reports_overlap", "n_hits", "hit_share", "med_abs_r", "n_unique_hits", "unique_hit_share"]


def log(msg: str) -> None:
    print(msg, flush=True)


def wrap180(x):
    return C.angle_diff_deg(x, 0.0)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

def predict_angles(e, n, u, es, ns, us) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(azimuth deg clockwise from north in (-180, 180], elevation deg, horizontal range m) of a
    target at (e, n, u) seen from (es, ns, us).  NaN u or us -> NaN elevation."""
    de = np.asarray(e, float) - es
    dn = np.asarray(n, float) - ns
    rng = np.hypot(de, dn)
    az = np.degrees(np.arctan2(de, dn))
    with np.errstate(invalid="ignore"):
        el = np.degrees(np.arctan2(np.asarray(u, float) - us, rng))
    return az, el, rng


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #

def load_reports(det: dict, it_key: str, deployable: frozenset[str]) -> dict | None:
    """Bearing reports of one iteration (see module docstring), sorted by (sensor_key, t)."""
    m = ((det["iter_key"] == it_key) & np.isin(det["meas_kind"], Q.BEARING_KINDS) & ~det["duplicate"]
         & (det["sensor_pos_valid"] == 1) & np.isfinite(det["az_deg"]) & np.isfinite(det["sensor_e"])
         & np.isfinite(det["sensor_n"]))
    m &= np.array([sid in FAMILY_OF_ID and sid in deployable for sid in det["sensor_id"]])
    if not m.any():
        return None
    o = np.lexsort((det["t"][m], det["sensor_key"][m]))
    R = {k: det[k][m][o] for k in ("sensor_key", "sensor_id", "iter_key", "t", "az_deg", "el_deg", "az_err_deg", "sensor_e",
                                   "sensor_n", "sensor_u", "cls", "cls_sub", "freq_mhz", "model", "protocol")}
    key = np.array([f"{k}\x1f{int(np.floor(t))}" for k, t in zip(R["sensor_key"], R["t"])], dtype=object)
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    R["rw"] = 1.0 / cnt[inv]              # a sensor-second weighs 1, however many reports it holds
    R["fam"] = np.array([FAMILIES.index(FAMILY_OF_ID[s]) for s in R["sensor_id"]])
    low = [f"{a} {b}".lower() for a, b in zip(R["cls"], R["cls_sub"])]
    R["uav"] = np.array([("uav" in s) or ("drone" in s) for s in low])
    R["band"] = np.array([MM.map_value(MM.BAND, f"{f:g}") if np.isfinite(f) else "" for f in R["freq_mhz"]], dtype=object)
    R["proto"] = np.array([MM.map_value(MM.PROTOCOL, s) for s in R["protocol"]], dtype=object)
    R["mdl"] = np.array([MM.map_value(MM.MODEL, s) for s in R["model"]], dtype=object)
    return R


class Pool:
    """Position-track segments of one iteration and their grid samples sorted by step."""

    def __init__(self, it: C.Iteration, deployable: frozenset[str]):
        slug = C.iter_slug(it.key)
        tr = C.read_columns(TRACKS / f"{slug}__tracks.csv", {"seg_id"},
                            usecols=["seg_id", "iter_key", "sensor_key", "sensor_id", "object_id", "label", "best_flight"])
        self.it_key = it.key
        self.seg_id = tr["seg_id"].astype(np.int64)
        if not np.array_equal(self.seg_id, np.arange(self.seg_id.size)):
            raise SystemExit(f"{it.key}: seg_id is not 0..S-1 in row order")
        self.sensor_key, self.sensor_id = tr["sensor_key"], tr["sensor_id"]
        self.seg_iter = tr["iter_key"]                  # as written in tracks.csv, used to MEASURE the provenance
        self.label, self.best_flight = tr["label"], tr["best_flight"]
        self.n_seg = self.seg_id.size
        # registration pool: deployable sensors, no ground-truth relay
        self.reg_ok = np.array([(s in deployable) and (s not in Q.GT_FEED_SENSORS) for s in self.sensor_id])
        self.elig = np.array([s in deployable for s in self.sensor_id])      # candidates for 'closest'
        ex = EXISTENCE / f"{slug}__p.csv"
        if ex.exists():
            pc = C.read_columns(ex, {"seg_id", "p_drone_all"}, usecols=["seg_id", "p_drone_all"])
            if not np.array_equal(pc["seg_id"].astype(np.int64), self.seg_id):
                raise SystemExit(f"{it.key}: existence seg_id mismatch")
            self.p = np.nan_to_num(pc["p_drone_all"], nan=0.0)
        else:
            self.p = np.ones(self.n_seg)
        self.inner: dict[str, np.ndarray] = {}           # testing only: outer fold -> nested p on that fold's scale
        pin = EXISTENCE / f"{slug}__p_inner.csv"
        if pin.exists():
            pi = C.read_columns(pin, {"seg_id", "p_drone_all"}, usecols=["seg_id", "outer_fold", "p_drone_all"])
            for fold in sorted(set(pi["outer_fold"].tolist())):
                mm = pi["outer_fold"] == fold
                sid_ = pi["seg_id"][mm].astype(np.int64)
                if not np.array_equal(sid_, np.arange(self.n_seg)):
                    raise SystemExit(f"{it.key}: p_inner for fold {fold} is not seg_id 0..S-1")
                self.inner[str(fold)] = np.nan_to_num(pi["p_drone_all"][mm], nan=0.0)
        z = np.load(TRACKS / f"{slug}__grid.npz")
        o = np.argsort(z["step"], kind="stable")
        self.g_seg = z["seg_id"][o].astype(np.int64)
        self.g_step = z["step"][o].astype(np.int64)
        self.g_e, self.g_n, self.g_u = (z[k][o].astype(float) for k in ("e", "n", "u"))
        self.g_sd = np.sqrt(0.5 * (z["var_e"][o].astype(float) + z["var_n"][o].astype(float)))
        self.grid_t = C.template_grid(it)[1]
        self.starts = np.searchsorted(self.g_step, np.arange(self.grid_t.size + 1))


def nearest_step(grid_t: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Nearest grid index, -1 where |t - grid| > GRID_TOL_S."""
    i = np.clip(np.searchsorted(grid_t, t), 1, grid_t.size - 1)
    j = np.where(np.abs(t - grid_t[i - 1]) <= np.abs(t - grid_t[i]), i - 1, i)
    return np.where(np.abs(t - grid_t[j]) <= GRID_TOL_S + 1e-9, j, -1)


def build_pairs(pool: Pool, R: dict, az: np.ndarray | None = None, chunk: int = 3_000_000) -> dict:
    """All (report, segment) pairs within RANGE_MAX_M: rep, seg, gi (grid row), r_raw (az residual
    before any offset), el_r (elevation residual, NaN if undefined), rng.  `az` overrides the
    report azimuths (used by the injection check)."""
    az = R["az_deg"] if az is None else az
    step = nearest_step(pool.grid_t, R["t"])
    ok = np.flatnonzero(step >= 0)
    cnt = np.zeros(step.size, np.int64)
    cnt[ok] = pool.starts[step[ok] + 1] - pool.starts[step[ok]]
    out = {k: [] for k in ("rep", "seg", "gi", "r_raw", "el_r", "rng", "sd")}
    i = 0
    while i < ok.size:
        c = np.cumsum(cnt[ok[i:]])
        k = max(1, int(np.searchsorted(c, chunk, side="right")))
        rep = ok[i:i + k]
        i += k
        n = cnt[rep]
        tot = int(n.sum())
        if tot == 0:
            continue
        rep_p = np.repeat(rep, n)
        off = np.repeat(np.cumsum(n) - n, n)
        gi = pool.starts[step[rep_p]] + (np.arange(tot) - off)
        a, el, rng = predict_angles(pool.g_e[gi], pool.g_n[gi], pool.g_u[gi], R["sensor_e"][rep_p],
                                    R["sensor_n"][rep_p], R["sensor_u"][rep_p])
        keep = rng <= RANGE_MAX_M
        rep_p, gi, a, el, rng = rep_p[keep], gi[keep], a[keep], el[keep], rng[keep]
        out["rep"].append(rep_p)
        out["seg"].append(pool.g_seg[gi])
        out["gi"].append(gi)
        out["r_raw"].append(wrap180(az[rep_p] - a))
        out["el_r"].append(R["el_deg"][rep_p] - el)
        out["rng"].append(rng)
        out["sd"].append(pool.g_sd[gi])
    return {k: (np.concatenate(v) if v else np.zeros(0, np.int64 if k in ("rep", "seg", "gi") else float))
            for k, v in out.items()}


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #

def _smooth_circ(h: np.ndarray, hw: int) -> np.ndarray:
    k = np.concatenate([np.arange(1, hw + 2), np.arange(hw, 0, -1)]).astype(float)
    k /= k.sum()
    out = np.zeros_like(h)
    for j, wj in zip(range(-hw, hw + 1), k):
        out += wj * np.roll(h, j)
    return out


def scan_offset(rep: np.ndarray, r_raw: np.ndarray, seg: np.ndarray | None = None, q: np.ndarray | None = None,
                rw: np.ndarray | None = None, seg_sensor: np.ndarray | None = None, cap: float = np.inf,
                prec: np.ndarray | None = None, est: dict = EST_DEFAULT, az_rep: np.ndarray | None = None) -> dict:
    """Azimuth offset (value to ADD) from candidate residuals.

    `rep` identifies the report of each candidate pair.  Pair weight = q / sum of q over the report's
    candidates (q = 1: the plain 1 / number of candidates) x rw (per-pair report weight, e.g. 1 over
    the sensor's reports in the same second).  With `seg` the weight one segment supplies to any single 1-degree
    bin is capped at `cap`, so no single short coincidence or parked pair can carry the histogram.  `seg_sensor` (position sensor
    of each pair) only feeds the corroboration counts n_seg_peak / n_sens_peak.  est["ms_hw"] is the mean-shift window.
    `az_rep` (reported azimuth of each pair's report) only feeds the information column az_spread_deg, the weighted circular
    sd of the reported azimuths of the pairs within PEAK_WIN_DEG of the peak (a camera that stares in one direction gives ~5-10)."""
    if rep.size == 0:
        return dict(offset=0.0, significance=0.0, peak_share=0.0, n_reports=0, n_pairs=0, n_seg_peak=0, n_sens_peak=0, second_ratio=1.0)
    q = np.ones(rep.size) if q is None else q
    _, inv, cnt = np.unique(rep, return_inverse=True, return_counts=True)
    sq = np.bincount(inv, weights=q)
    live = sq[inv] > 0
    w = np.where(live, q / np.where(sq[inv] > 0, sq[inv], 1.0), 0.0)
    if rw is not None:
        w = w * rw
    if prec is not None:
        w = w * prec
    b = np.floor(r_raw + 180.0).astype(np.int64) % 360
    if seg is not None and np.isfinite(cap):          # one segment supplies at most `cap` to any single 1-degree bin
        _, sinv = np.unique(seg, return_inverse=True)
        code = sinv.astype(np.int64) * 360 + b
        uc, cinv = np.unique(code, return_inverse=True)
        wb = np.bincount(cinv, weights=w)
        w = w * np.minimum(1.0, cap / np.maximum(wb, 1e-300))[cinv]
    h = np.bincount(b, weights=w, minlength=360)
    s = _smooth_circ(h, SMOOTH_HW)
    pk = int(np.argmax(s))
    idx = (pk + np.arange(-CENTROID_HW, CENTROID_HW + 1))
    hw = h[idx % 360]
    centre = pk - 180.0 + 0.5
    total = float(w.sum())
    if hw.sum() <= 0 or total <= 0:
        return dict(offset=0.0, significance=0.0, peak_share=0.0, n_reports=int((sq > 0).sum()), n_pairs=int(rep.size),
                    n_seg_peak=0, n_sens_peak=0, second_ratio=1.0)
    rival = s.copy()
    rival[(pk + np.arange(-SECOND_EXCL_DEG, SECOND_EXCL_DEG + 1)) % 360] = 0.0
    second = float(rival.max() / s[pk])
    bias = centre + float(np.sum(hw * np.arange(-CENTROID_HW, CENTROID_HW + 1)) / hw.sum())
    for _ in range(MS_ITER):                      # mean-shift: average the biases of the position sensors
        d = wrap180(r_raw - bias)
        inw = np.abs(d) <= est["ms_hw"]
        if not inw.any() or w[inw].sum() <= 0:
            break
        bias = float(wrap180(bias + np.sum(w[inw] * d[inw]) / w[inw].sum()))
    null = max(float(np.median(s)), SIG_FLOOR_SHARE * total / 360.0 * (2 * SMOOTH_HW + 1))
    n_seg = n_sens = 0
    spread = float("nan")
    nearp = np.abs(wrap180(r_raw - bias)) <= PEAK_WIN_DEG
    if az_rep is not None and w[nearp].sum() > 0:
        zc = np.sum(w[nearp] * np.exp(1j * np.radians(az_rep[nearp]))) / w[nearp].sum()
        spread = float(np.degrees(np.sqrt(-2.0 * np.log(max(abs(zc), 1e-9)))))
    if seg is not None:
        near = nearp
        ws_near = {}
        for sg, ww, sn in zip(seg[near], w[near], (seg_sensor[near] if seg_sensor is not None else seg[near])):
            ws_near.setdefault(int(sg), [0.0, sn])[0] += ww
        good = [v for v in ws_near.values() if v[0] >= MIN_SEG_W]
        n_seg, n_sens = len(good), len({v[1] for v in good})
    return dict(offset=float(wrap180(-bias)), significance=float(s[pk] / null), peak_share=float(hw.sum() / total),
                n_reports=int((sq > 0).sum()), n_pairs=int(rep.size), n_seg_peak=n_seg, n_sens_peak=n_sens,
                second_ratio=second, az_spread_deg=spread)


def sector_test(rep: np.ndarray, r_raw: np.ndarray, gate: float, p: np.ndarray) -> dict:
    """Zero-offset hypothesis for a coarse-sector sensor.  share(o) = share of reports that have a
    candidate with existence p >= SECTOR_P_MIN within `gate` after adding o (reports counted: those
    with any candidate).  significance = share(0) / median share over o in steps of SECTOR_STEP_DEG;
    rank0 = number of offsets that beat offset 0."""
    empty = dict(offset=0.0, significance=0.0, peak_share=0.0, n_reports=0, n_pairs=0, rank0=99)
    if rep.size == 0:
        return empty
    o = np.argsort(rep, kind="stable")
    rep, r_raw, p = rep[o], r_raw[o], p[o]
    first = np.flatnonzero(np.concatenate([[True], rep[1:] != rep[:-1]]))
    offs = np.arange(-180.0, 180.0, SECTOR_STEP_DEG)
    ok = p >= SECTOR_P_MIN
    shares = np.array([float(np.mean(np.maximum.reduceat(((np.abs(wrap180(r_raw + x)) <= gate) & ok).astype(float), first)))
                       for x in offs])
    i0 = int(np.flatnonzero(offs == 0.0)[0])
    s0 = float(shares[i0])
    med = max(float(np.median(shares)), 1.0 / (10.0 * first.size))
    return dict(offset=0.0, significance=s0 / med, peak_share=s0, n_reports=int(first.size), n_pairs=int(rep.size),
                rank0=int((shares > s0).sum()))


def el_gate_of(R: dict) -> np.ndarray:
    return np.array([EL_GATE_DEG[FAMILIES[f]] for f in R["fam"]])


def gate_of(R: dict, gates: dict = GATE_DEFAULT, rep=None) -> np.ndarray:
    """Per-report azimuth gate max(K_SIGMA x az_err, family floor in `gates`)."""
    base = np.array([gates[FAMILIES[f]] for f in R["fam"]])
    err = np.nan_to_num(R["az_err_deg"], nan=0.0)
    g = np.maximum(K_SIGMA * err, base)
    return g if rep is None else g[rep]


def precision(sd_m: np.ndarray, rng_m: np.ndarray, sigma0: float = EST_DEFAULT["prec_sigma0"]) -> np.ndarray:
    """Pair weight 1 / (1 + (sigma_ang / sigma0)^2): angular sd of the position track seen from
    the bearing sensor (the track's total variance includes the slow bias)."""
    ang = np.degrees(sd_m / np.maximum(rng_m, 1.0))
    return 1.0 / (1.0 + (ang / sigma0) ** 2)


def time_blocks(t_rep: np.ndarray, rep: np.ndarray, k: int = SPLIT_K) -> list[np.ndarray]:
    """Pair masks of k time blocks holding (about) equal numbers of reports."""
    u, first = np.unique(rep, return_index=True)
    qs = np.quantile(np.sort(t_rep[first]), np.linspace(0.0, 1.0, k + 1))
    out = []
    for b in range(k):
        out.append((t_rep >= qs[b]) & ((t_rep < qs[b + 1]) if b < k - 1 else (t_rep <= qs[b + 1])))
    return out


def register_stats(pool: Pool, R: dict, P: dict, sk: str, est: dict = EST_DEFAULT, weighted: bool = True,
                   p: np.ndarray | None = None) -> dict:
    """Everything the acceptance rule needs for one bearing sensor in one iteration, independent of the
    acceptance thresholds: peak, significance, corroboration counts, second_ratio, the SPLIT_K time-block
    peaks, provenance.  `p` overrides the pool's existence probability (used for the nested p_inner)."""
    p = pool.p if p is None else p
    smask = R["sensor_key"] == sk
    rmask = smask[P["rep"]] & pool.reg_ok[P["seg"]] & (P["rng"] >= REG_RANGE_MIN_M)
    rep, r_raw, seg = P["rep"][rmask], P["r_raw"][rmask], P["seg"][rmask]
    sid = R["sensor_id"][smask][0]
    if sid in SECTOR_IDS:
        gate = float(gate_of(R)[smask].max())
        res = sector_test(rep, r_raw, gate, p[seg])
        res["method"] = "sector"
        res["sector_ok"] = bool(res["significance"] >= SECTOR_SIG_MIN and res["rank0"] < SECTOR_RANK_MAX
                                and res["n_reports"] >= SECTOR_MIN_REPORTS)
        res["blocks"] = []
    else:
        prec = precision(P["sd"][rmask], P["rng"][rmask], est["prec_sigma0"])
        kw = dict(seg=seg, seg_sensor=pool.sensor_key[seg], prec=prec, est=est, az_rep=R["az_deg"][rep])
        if weighted:
            kw.update(q=p[seg], rw=R["rw"][rep], cap=est["seg_cap"])
        res = scan_offset(rep, r_raw, **kw)
        res["method"] = "scan"
        res["blocks"] = []
        if weighted and rep.size:
            t_rep = R["t"][rep]
            for m in time_blocks(t_rep, rep):
                sub = {k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in kw.items()}
                b = scan_offset(rep[m], r_raw[m], **sub)
                res["blocks"].append((int(b["n_reports"]), float(b["offset"]), float(b["significance"])))
    res["n_total_reports"] = int(smask.sum())
    res["src_sensors"] = sorted(set(pool.sensor_key[seg].tolist()))
    res["src_iters"] = sorted(set(pool.seg_iter[seg].tolist()) | (set(R["iter_key"][smask].tolist()) if "iter_key" in R and seg.size else set()))
    res["src_bearing"] = sk
    res["sid"] = str(sid)
    return res


def n_contradicting_blocks(res: dict) -> int:
    """Testable time blocks (>= SPLIT_MIN_REPORTS reports, own significance >= SPLIT_SIG) whose peak is farther than
    SPLIT_TOL_DEG from the full-data offset."""
    return sum(1 for (n, off, sig) in res["blocks"]
               if n >= SPLIT_MIN_REPORTS and sig >= SPLIT_SIG and abs(float(wrap180(off - res["offset"]))) > SPLIT_TOL_DEG)


def accept_rule(res: dict, acc: dict = ACC_DEFAULT) -> bool:
    """Accept the estimated offset?  Pure function of register_stats() output and the acceptance parameters."""
    if res["sid"] in PRESENCE_ONLY_IDS:
        return False
    if res["method"] == "sector":
        return res["sector_ok"]
    ok = (res["significance"] >= acc["sig_min"] and res["n_reports"] >= MIN_REPORTS
          and res["n_seg_peak"] >= MIN_SEG_PEAK and res["n_sens_peak"] >= MIN_SENS_PEAK
          and res["second_ratio"] <= acc["max_second"])
    if acc["split"] == 2:
        ok = ok and n_contradicting_blocks(res) == 0
    elif acc["split"] == 1:
        ok = ok and n_contradicting_blocks(res) <= 1
    return bool(ok)


def register_sensor(pool: Pool, R: dict, P: dict, sk: str, est: dict = EST_DEFAULT, acc: dict = ACC_DEFAULT,
                    weighted: bool = True, p: np.ndarray | None = None) -> dict:
    """Registration of one bearing sensor in one iteration (stats + acceptance); see register_stats."""
    res = register_stats(pool, R, P, sk, est, weighted, p)
    if res["sid"] in PRESENCE_ONLY_IDS:
        res["status"] = "presence_only"
    res["accepted"] = accept_rule(res, acc)
    res.setdefault("status", "accepted" if res["accepted"] else "unregistered")
    res.setdefault("n_seg_peak", 0)
    res.setdefault("n_sens_peak", 0)
    res.setdefault("second_ratio", 1.0)
    res["effective_offset"] = res["offset"] if res["accepted"] else 0.0
    return res


# --------------------------------------------------------------------------- #
# support features
# --------------------------------------------------------------------------- #

def group_median(g: np.ndarray, v: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n, np.nan)
    if g.size == 0:
        return out
    o = np.lexsort((v, g))
    vs = v[o]
    cnt = np.bincount(g, minlength=n)
    start = np.cumsum(cnt) - cnt
    m = cnt > 0
    out[m] = 0.5 * (vs[start[m] + (cnt[m] - 1) // 2] + vs[start[m] + cnt[m] // 2])
    return out


def top_string(seg: np.ndarray, vals: np.ndarray, n: int) -> np.ndarray:
    """Most common non-empty string per segment (ties: alphabetical), '' if none."""
    out = np.full(n, "", dtype=object)
    keep = np.array([bool(v) for v in vals], dtype=bool)
    if not keep.any():
        return out
    seg, vals = seg[keep], vals[keep]
    uniq = sorted(set(vals.tolist()))
    vid = np.array([uniq.index(v) for v in vals])
    code = seg.astype(np.int64) * len(uniq) + vid
    u, c = np.unique(code, return_counts=True)
    s_u, v_u = u // len(uniq), u % len(uniq)
    o = np.lexsort((v_u, -c, s_u))          # per segment: highest count, then smallest string id
    first = np.flatnonzero(np.concatenate([[True], s_u[o][1:] != s_u[o][:-1]]))
    for j in first:
        out[s_u[o][j]] = uniq[v_u[o][j]]
    return out


def support_features(pool: Pool, R: dict, P: dict, offsets: dict[str, float], gates: dict = GATE_DEFAULT,
                     fams: tuple[str, ...] = FAMILIES) -> dict:
    """Per-segment feature columns (arrays of length pool.n_seg) from the pair table, for the families `fams`
    (azimuth gate floors `gates`)."""
    S = pool.n_seg
    rep, seg = P["rep"], P["seg"]
    off = np.array([offsets.get(k, 0.0) for k in R["sensor_key"]])
    r = wrap180(P["r_raw"] + off[rep])
    absr = np.abs(r)
    el_r = P["el_r"]
    el_ok = ~(np.isfinite(el_r) & np.isfinite(R["el_deg"][rep])) | (np.abs(el_r) <= el_gate_of(R)[rep])
    hit = (absr <= gate_of(R, gates)[rep]) & el_ok
    # closest eligible candidate per report (ties: lowest seg index)
    n_rep = R["t"].size
    best_abs = np.full(n_rep, np.inf)
    best_seg = np.full(n_rep, -1, np.int64)
    e = pool.elig[seg]
    if e.any():
        ie = np.flatnonzero(e)
        o = np.lexsort((seg[ie], absr[ie], rep[ie]))
        ie = ie[o]
        first = np.flatnonzero(np.concatenate([[True], rep[ie][1:] != rep[ie][:-1]]))
        best_abs[rep[ie][first]] = absr[ie][first]
        best_seg[rep[ie][first]] = seg[ie][first]
    uniq = np.where(e, seg == best_seg[rep], absr < best_abs[rep]) & hit
    fam = R["fam"][rep]
    cols: dict[str, np.ndarray] = {}
    for fi, f in enumerate(FAMILIES):
        if f not in fams:
            continue
        mf = fam == fi
        n_ov = np.bincount(seg[mf], minlength=S).astype(float)
        n_hit = np.bincount(seg[mf & hit], minlength=S).astype(float)
        n_uni = np.bincount(seg[mf & uniq], minlength=S).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            cols[f"{f}_n_reports_overlap"] = n_ov
            cols[f"{f}_n_hits"] = n_hit
            cols[f"{f}_hit_share"] = np.where(n_ov > 0, n_hit / n_ov, np.nan)
            cols[f"{f}_med_abs_r"] = group_median(seg[mf & hit], absr[mf & hit], S)
            cols[f"{f}_n_unique_hits"] = n_uni
            cols[f"{f}_unique_hit_share"] = np.where(n_ov > 0, n_uni / n_ov, np.nan)
        if f == "camera":
            h = mf & hit
            n_uav = np.bincount(seg[h], weights=R["uav"][rep[h]].astype(float), minlength=S)
            cols["camera_uav_share"] = np.where(n_hit > 0, n_uav / np.maximum(n_hit, 1), np.nan)
        if f == "rf":
            h = mf & hit
            cols["rf_band"] = top_string(seg[h], R["band"][rep[h]], S)
            cols["rf_protocol"] = top_string(seg[h], R["proto"][rep[h]], S)
            cols["rf_model"] = top_string(seg[h], R["mdl"][rep[h]], S)
    return cols


FEATURE_ORDER = (["seg_id", "iter_key", "sensor_key"]
                 + [f"{f}_{c}" for f in FAMILIES for c in SUP_COLS]
                 + ["camera_uav_share", "rf_band", "rf_protocol", "rf_model"])


# --------------------------------------------------------------------------- #
# calibration table (comparison only)
# --------------------------------------------------------------------------- #

def gt_offsets() -> dict[tuple[str, str], tuple[float, int]]:
    """(iter_key, sensor_key) -> (value, n) of 'applied' per-iteration az_offset_deg rows."""
    cal = C.read_columns(CALIBRATION, {"value", "n"})
    out = {}
    for i in range(len(cal["sensor_key"])):
        if (cal["param"][i] == "az_offset_deg" and cal["status"][i] == "applied"
                and cal["iter_key"][i] not in ("*", "") and cal["iter_key"][i].startswith("testing")):
            out[(cal["iter_key"][i], cal["sensor_key"][i])] = (float(cal["value"][i]), int(cal["n"][i]))
    return out


# --------------------------------------------------------------------------- #
# one iteration end to end
# --------------------------------------------------------------------------- #

DET_COLS = ["cls", "cls_sub", "freq_mhz", "model", "protocol", "az_deg", "el_deg", "az_err_deg", "sensor_u"]


class Ctx:
    """One iteration's bearing reports, position pool and (report, segment) pair table."""

    def __init__(self, it: C.Iteration, det: dict, deployable: frozenset[str]):
        self.it = it
        self.R = load_reports(det, it.key, deployable)
        self.pool = Pool(it, deployable)
        self.P = build_pairs(self.pool, self.R) if self.R is not None else None
        self.sensors = sorted(set(self.R["sensor_key"].tolist())) if self.R is not None else []


def row_utility(res: dict, acc: dict, gt_val: float) -> float:
    """Selection utility of one GT row: an accepted offset that agrees with the GT row is worth UTIL_RIGHT, an accepted
    one that does not UTIL_WRONG (a wrong rotation is worse than none), an unaccepted one 0."""
    if not accept_rule(res, acc):
        return 0.0
    return UTIL_RIGHT if abs(float(wrap180(res["offset"] - gt_val))) <= AGREE_DEG else UTIL_WRONG


def choose_fold(held: str | None, ctxs: dict[str, Ctx], gt: dict, testing_keys: list[str]) -> dict:
    """Choices applied to iteration `held` (a testing key) or, with held=None, to the evaluation iterations.
    Everything is chosen on the OTHER testing iterations only (held-out one excluded):
      1. estimator shape (EST_GRID) x acceptance rule (ACC_GRID): maximise the summed selection utility on the GT rows
         (n >= GT_MIN_N, scan sensors) of the training iterations; ties go to the entry closest to the a-priori default;
      2. azimuth gate floors (GATE_GRID): per family, the floor with the highest pooled within-sensor AUC (drone vs
         clutter, hit_share of the covered segments of the four radars) on the training iterations, computed with the
         offsets that the chosen (1) gives those training iterations.
    The existence probability of a training iteration j is p_inner(j, fold=held) = a model that saw neither held nor j
    (held=None: the ordinary leave-one-out p).  The GT rows are only used to score candidate rules."""
    train = [k for k in testing_keys if k != held]
    def pf(j):
        return ctxs[j].pool.inner[held] if held else ctxs[j].pool.p
    rows = [(j, sk, v) for (j, sk), (v, n) in sorted(gt.items())
            if j in train and n >= GT_MIN_N and sk.split("_")[0] not in SECTOR_IDS and sk in ctxs[j].sensors]
    best = None
    for ei, est in enumerate(EST_GRID):
        sts = [register_stats(ctxs[j].pool, ctxs[j].R, ctxs[j].P, sk, est, True, pf(j)) for j, sk, _ in rows]
        for ai, acc in enumerate(ACC_GRID):
            u = sum(row_utility(st, acc, v) for st, (_, _, v) in zip(sts, rows))
            if best is None or (-u, ei, ai) < best[0]:
                best = ((-u, ei, ai), ei, ai, u)
    _, ei, ai, util = best
    est, acc = EST_GRID[ei], ACC_GRID[ai]
    offs = {}
    for j in train:
        c = ctxs[j]
        offs[j] = {sk: register_sensor(c.pool, c.R, c.P, sk, est, acc, True, pf(j))["effective_offset"] for sk in c.sensors}
    gates, gate_auc = dict(GATE_DEFAULT), {}
    for fam in FAMILIES:
        aucs = []
        for g in GATE_GRID[fam]:
            gg = dict(GATE_DEFAULT)
            gg[fam] = g
            ys, sc = [], []
            for j in train:
                c = ctxs[j]
                if c.R is None:
                    continue
                F = support_features(c.pool, c.R, c.P, offs[j], gg, (fam,))
                m = np.isin(c.pool.sensor_id, AUC_GROUP) & (F[f"{fam}_n_reports_overlap"] > 0)
                ys.append(np.array([str(x).startswith("drone") for x in c.pool.label[m]], float))
                sc.append(F[f"{fam}_hit_share"][m])
            y = np.concatenate(ys) if ys else np.zeros(0)
            aucs.append(gbdt.auc(y, np.concatenate(sc)) if y.size and 0 < y.sum() < y.size else float("nan"))
        key = [(-round(a, 3) if a == a else 0.0, i) for i, a in enumerate(aucs)]
        gates[fam] = GATE_GRID[fam][min(key)[1]]
        gate_auc[fam] = aucs
    return dict(held=held, train=train, est=est, acc=acc, gates=gates, util=util, n_rows=len(rows), est_idx=ei, acc_idx=ai,
                rows=[(j, sk, v) for j, sk, v in rows], p_kind=[("inner:" + held) if held else "outer" for _ in rows],
                gate_auc=gate_auc)


def process_iteration(ctx: Ctx, choice: dict, gt: dict) -> dict:
    """Registration rows + per-segment feature columns of one iteration under `choice` (from choose_fold)."""
    it, R, P, pool = ctx.it, ctx.R, ctx.P, ctx.pool
    S = pool.n_seg
    meta = dict(label=pool.label, best_flight=pool.best_flight, sensor_id=pool.sensor_id)
    if R is None:
        return dict(regs=[], feats=_empty_features(pool), meta=meta, n_reports=0, choice=choice, offsets={})
    regs, offsets = [], {}
    for sk in ctx.sensors:
        res = register_sensor(pool, R, P, sk, choice["est"], choice["acc"], True)
        plain = register_sensor(pool, R, P, sk, EST_DEFAULT, ACC_DEFAULT, False)
        g = gt.get((it.key, sk))
        eff, raw = res["effective_offset"], res["offset"]
        close = (lambda v: abs(float(wrap180(v - g[0]))) <= AGREE_DEG) if g else None
        res.update(iter_key=it.key, sensor_key=sk, raw_offset=res["offset"], offset=eff,
                   config=f"{'loio:' + choice['held'] if choice['held'] else 'all-testing'} est{choice['est_idx']} acc{choice['acc_idx']}",
                   plain_offset=plain["effective_offset"] if plain["accepted"] else plain["offset"],
                   plain_significance=plain["significance"],
                   gt_offset=g[0] if g else float("nan"), gt_n=g[1] if g else 0,
                   agree=(close(raw) if (g and res["accepted"]) else None),     # accepted offsets only
                   eff_agree=(close(eff) if g else None),                                  # offset actually used (0 if unaccepted)
                   raw_agree=(close(raw) if g else None))
        regs.append(res)
        offsets[sk] = eff
    feats = support_features(pool, R, P, offsets, choice["gates"])
    feats.update(seg_id=pool.seg_id, iter_key=np.full(S, it.key, dtype=object), sensor_key=pool.sensor_key)
    return dict(regs=regs, feats=feats, meta=meta, n_reports=int(R["t"].size), n_pairs=int(P["rep"].size), choice=choice, offsets=offsets)


def _empty_features(pool: Pool) -> dict:
    S = pool.n_seg
    f: dict[str, np.ndarray] = {}
    for fam in FAMILIES:
        for c in SUP_COLS:
            f[f"{fam}_{c}"] = np.zeros(S) if c.startswith("n_") else np.full(S, np.nan)
    f["camera_uav_share"] = np.full(S, np.nan)
    for c in ("rf_band", "rf_protocol", "rf_model"):
        f[c] = np.full(S, "", dtype=object)
    f.update(seg_id=pool.seg_id, iter_key=np.full(S, pool.it_key, dtype=object), sensor_key=pool.sensor_key)
    return f


def write_support(slug: str, feats: dict) -> None:
    cols = dict(feats)
    for k in list(cols):
        if k.endswith(("n_reports_overlap", "n_hits", "n_unique_hits")):
            cols[k] = np.asarray(cols[k]).astype(np.int64)
    C.write_columns(OUT / f"{slug}__support.csv", cols, FEATURE_ORDER)


REG_ORDER = ["iter_key", "sensor_key", "offset_deg", "significance", "n_reports", "accepted", "gt_offset_deg", "agree",
             "raw_offset_deg", "method", "n_total_reports", "n_pairs", "peak_share", "n_seg_peak", "n_sens_peak",
             "gt_n", "plain_offset_deg", "plain_significance", "src_sensors", "src_iters", "status", "eff_agree", "n_blocks_contradict",
             "blocks", "config", "az_spread_deg"]


def write_registration(regs: list[dict]) -> None:
    def col(f, dtype=object):
        return np.array([f(r) for r in regs], dtype=dtype)
    cols = dict(
        iter_key=col(lambda r: r["iter_key"]), sensor_key=col(lambda r: r["sensor_key"]),
        offset_deg=col(lambda r: round(r["offset"], 3), float), significance=col(lambda r: round(r["significance"], 3), float),
        n_reports=col(lambda r: r["n_reports"], np.int64), accepted=col(lambda r: int(r["accepted"]), np.int64),
        gt_offset_deg=col(lambda r: r["gt_offset"], float),
        agree=col(lambda r: "" if r["agree"] is None else int(r["agree"])),
        raw_offset_deg=col(lambda r: round(r["raw_offset"], 3), float), method=col(lambda r: r["method"]),
        n_total_reports=col(lambda r: r["n_total_reports"], np.int64), n_pairs=col(lambda r: r["n_pairs"], np.int64),
        peak_share=col(lambda r: round(r["peak_share"], 4), float), n_seg_peak=col(lambda r: r["n_seg_peak"], np.int64),
        n_sens_peak=col(lambda r: r["n_sens_peak"], np.int64), gt_n=col(lambda r: r["gt_n"], np.int64),
        plain_offset_deg=col(lambda r: round(r["plain_offset"], 3), float),
        plain_significance=col(lambda r: round(r["plain_significance"], 3), float),
        src_sensors=col(lambda r: "|".join(r["src_sensors"])), src_iters=col(lambda r: "|".join(r["src_iters"])),
        status=col(lambda r: r["status"]), eff_agree=col(lambda r: "" if r["eff_agree"] is None else int(r["eff_agree"])),
        n_blocks_contradict=col(lambda r: n_contradicting_blocks(r), np.int64),
        blocks=col(lambda r: "|".join(f"{n}:{o:+.1f}:{sg:.0f}" for n, o, sg in r["blocks"])),
        config=col(lambda r: r["config"]), az_spread_deg=col(lambda r: round(r.get("az_spread_deg", float("nan")), 1), float))
    C.write_columns(OUT / "registration.csv", cols, REG_ORDER)


# --------------------------------------------------------------------------- #
# synthetic world (self-checks 1 and 2)
# --------------------------------------------------------------------------- #

def synthetic_world(true_offset: float, seed: int, n_drone: int = 5, n_clutter: int = 20, dur_s: int = 600,
                    az_noise: float = 0.7, false_share: float = 0.3):
    """(pool, R, truth) of a fictitious iteration.  A camera at the origin reports the azimuth of a random
    visible drone minus `true_offset` (so the correct offset to add is `true_offset`), with noise, and
    false_share of the reports point at random directions.  Each drone is seen by two radars with their own
    constant bias; clutter segments come from a third radar."""
    rng = np.random.default_rng(seed)
    G = dur_s * 10
    grid_t = 1.0e9 + 0.1 * np.arange(G)
    segs = []                       # (sensor_key, sensor_id, label, e[G], n[G], p)
    truth_xy = []
    for _ in range(n_drone):
        r0, th = rng.uniform(300, 1500), rng.uniform(0, 2 * np.pi)
        v = rng.uniform(3, 12) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        z = r0 * np.exp(1j * th) + v * np.arange(G) * 0.1
        truth_xy.append(z)
        for sk, sid in (("NYTF_RADAR", "NYTF"), ("UXYJ_RADAR", "UXYJ")):
            b = rng.normal(0, 8.0, 2)
            segs.append((sk, sid, "drone_air", z.real + b[0] + rng.normal(0, 1, G), z.imag + b[1] + rng.normal(0, 1, G), 0.8))
    for _ in range(n_clutter):
        r0, th = rng.uniform(150, 2500), rng.uniform(0, 2 * np.pi)
        v = rng.uniform(0, 8) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        z = r0 * np.exp(1j * th) + v * np.arange(G) * 0.1
        segs.append(("EGOI_RADAR", "EGOI", "clutter", z.real, z.imag, 0.05))
    S = len(segs)
    pool = Pool.__new__(Pool)
    pool.it_key = "synthetic/x"
    pool.seg_iter = np.full(S, "synthetic/x", dtype=object)
    pool.inner = {}
    pool.seg_id = np.arange(S)
    pool.n_seg = S
    pool.sensor_key = np.array([s[0] for s in segs], dtype=object)
    pool.sensor_id = np.array([s[1] for s in segs], dtype=object)
    pool.label = np.array([s[2] for s in segs], dtype=object)
    pool.best_flight = np.full(S, "", dtype=object)
    pool.reg_ok = np.ones(S, bool)
    pool.elig = np.ones(S, bool)
    pool.p = np.array([s[5] for s in segs])
    pool.grid_t = grid_t
    pool.g_seg = np.repeat(np.arange(S), G)
    pool.g_step = np.tile(np.arange(G), S)
    pool.g_e = np.concatenate([s[3] for s in segs])
    pool.g_n = np.concatenate([s[4] for s in segs])
    pool.g_u = np.full(S * G, 60.0)
    pool.g_sd = np.full(S * G, 10.0)
    o = np.argsort(pool.g_step, kind="stable")
    for k in ("g_seg", "g_step", "g_e", "g_n", "g_u", "g_sd"):
        setattr(pool, k, getattr(pool, k)[o])
    pool.starts = np.searchsorted(pool.g_step, np.arange(G + 1))
    n_rep = 3000
    t = np.sort(rng.uniform(0, dur_s - 1, n_rep))
    step = np.minimum((t * 10).astype(int), G - 1)
    az = np.empty(n_rep)
    for i in range(n_rep):
        if rng.random() < false_share:
            az[i] = rng.uniform(-180, 180)
        else:
            z = truth_xy[rng.integers(n_drone)][step[i]]
            az[i] = np.degrees(np.arctan2(z.real, z.imag)) - true_offset + rng.normal(0, az_noise)
    R = dict(sensor_key=np.full(n_rep, "WVFX_CAMERA", dtype=object), sensor_id=np.full(n_rep, "WVFX", dtype=object),
             t=grid_t[step], az_deg=wrap180(az), el_deg=np.full(n_rep, np.nan), az_err_deg=np.full(n_rep, np.nan),
             sensor_e=np.zeros(n_rep), sensor_n=np.zeros(n_rep), sensor_u=np.full(n_rep, 10.0),
             fam=np.zeros(n_rep, np.int64), rw=np.ones(n_rep), uav=np.zeros(n_rep, bool),
             band=np.full(n_rep, "", dtype=object), proto=np.full(n_rep, "", dtype=object),
             mdl=np.full(n_rep, "", dtype=object))
    return pool, R, truth_xy


# --------------------------------------------------------------------------- #
# plain-python reference for the support features (self-check 4)
# --------------------------------------------------------------------------- #

def reference_context(it: C.Iteration):
    """Fresh, loop-friendly view of the grid file: by_step[step] = [(seg, e, n, u)] ascending seg, and
    by_seg[seg] = {step: (e, n, u)}.  Independent of Pool's arrays."""
    z = np.load(TRACKS / f"{C.iter_slug(it.key)}__grid.npz")
    seg, step = z["seg_id"].tolist(), z["step"].tolist()
    e, n, u = z["e"].astype(float).tolist(), z["n"].astype(float).tolist(), z["u"].astype(float).tolist()
    by_step: dict[int, list] = {}
    by_seg: dict[int, dict] = {}
    for sg, st, ee, nn, uu in zip(seg, step, e, n, u):
        by_step.setdefault(st, []).append((sg, ee, nn, uu))
        by_seg.setdefault(sg, {})[st] = (ee, nn, uu)
    for lst in by_step.values():
        lst.sort(key=lambda x: x[0])
    return by_step, by_seg


def _py_wrap(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0


def reference_features(seg: int, R: dict, offsets: dict, grid_t: list, by_step: dict, by_seg: dict,
                       elig: list[bool], gates: dict) -> dict:
    """Features of one segment by plain loops over reports (see module docstring for the rules)."""
    import bisect
    out = {}
    for fi, fam in enumerate(FAMILIES):
        n_ov = n_hit = n_uni = n_uav = 0
        absr_hits: list[float] = []
        bands: Counter = Counter()
        protos: Counter = Counter()
        mdls: Counter = Counter()
        for i in range(len(R["t"])):
            if R["fam"][i] != fi:
                continue
            t = float(R["t"][i])
            j = bisect.bisect_left(grid_t, t)
            cands = [k for k in (j - 1, j) if 0 <= k < len(grid_t)]
            if not cands:
                continue
            step = min(cands, key=lambda k: (abs(grid_t[k] - t), k))
            if abs(grid_t[step] - t) > GRID_TOL_S + 1e-9:
                continue
            sample = by_seg.get(seg, {}).get(step)
            if sample is None:
                continue
            es, ns, us = float(R["sensor_e"][i]), float(R["sensor_n"][i]), float(R["sensor_u"][i])
            off = offsets.get(R["sensor_key"][i], 0.0)
            gate = max(K_SIGMA * (0.0 if R["az_err_deg"][i] != R["az_err_deg"][i] else float(R["az_err_deg"][i])),
                       gates[fam])
            elgate = EL_GATE_DEG[fam]

            def pair(e, n, u):
                de, dn = e - es, n - ns
                rng = math.hypot(de, dn)
                az = math.degrees(math.atan2(de, dn))
                r = _py_wrap(float(R["az_deg"][i]) + off - az)
                el_r = float("nan")
                if R["el_deg"][i] == R["el_deg"][i] and u == u and us == us:
                    el_r = float(R["el_deg"][i]) - math.degrees(math.atan2(u - us, rng))
                return rng, abs(r), el_r

            rng, ar, el_r = pair(*sample)
            if rng > RANGE_MAX_M:
                continue
            n_ov += 1
            if not (ar <= gate and (el_r != el_r or abs(el_r) <= elgate)):
                continue
            n_hit += 1
            absr_hits.append(ar)
            # closest deployable candidate of this report (ties: lowest seg)
            best, best_seg = math.inf, -1
            for sg, e2, n2, u2 in by_step.get(step, []):
                if not elig[sg]:
                    continue
                r2, a2, _ = pair(e2, n2, u2)
                if r2 <= RANGE_MAX_M and a2 < best:
                    best, best_seg = a2, sg
            if (elig[seg] and best_seg == seg) or (not elig[seg] and ar < best):
                n_uni += 1
            n_uav += int(bool(R["uav"][i]))
            for ctr, key in ((bands, "band"), (protos, "proto"), (mdls, "mdl")):
                if R[key][i]:
                    ctr[R[key][i]] += 1
        out[f"{fam}_n_reports_overlap"] = n_ov
        out[f"{fam}_n_hits"] = n_hit
        out[f"{fam}_hit_share"] = n_hit / n_ov if n_ov else float("nan")
        s = sorted(absr_hits)
        out[f"{fam}_med_abs_r"] = (0.5 * (s[(len(s) - 1) // 2] + s[len(s) // 2])) if s else float("nan")
        out[f"{fam}_n_unique_hits"] = n_uni
        out[f"{fam}_unique_hit_share"] = n_uni / n_ov if n_ov else float("nan")
        if fam == "camera":
            out["camera_uav_share"] = n_uav / n_hit if n_hit else float("nan")
        if fam == "rf":
            for name, ctr in (("rf_band", bands), ("rf_protocol", protos), ("rf_model", mdls)):
                out[name] = min(ctr.items(), key=lambda kv: (-kv[1], kv[0]))[0] if ctr else ""
    return out


# --------------------------------------------------------------------------- #
# ground-truth diagnostics (testing only; NEVER an input of an estimate)
# --------------------------------------------------------------------------- #

def dronetag_scan(it: C.Iteration, det: dict, deployable: frozenset[str], sk: str, est: dict = EST_DEFAULT) -> dict:
    """Same scan with the Dronetag (airborne) flight positions as the candidates: an upper bound on what
    any position-track registration can reach with this histogram method.  Also the peaks of the SPLIT_K time
    blocks ('blocks': n_reports, offset, significance) to show whether a single per-iteration offset exists."""
    R = load_reports(det, it.key, deployable)
    tr = T.load_truth(it, "airborne")
    sel = np.flatnonzero(R["sensor_key"] == sk)
    step = nearest_step(tr.grid_t, R["t"][sel])
    o = np.argsort(tr.step, kind="stable")
    ts = tr.step[o]
    reps, rr, fl = [], [], []
    for k, i in enumerate(sel):
        if step[k] < 0:
            continue
        a, b = np.searchsorted(ts, step[k], "left"), np.searchsorted(ts, step[k], "right")
        for j in o[a:b]:
            az, _, rng = predict_angles(tr.e[j], tr.n[j], tr.u[j], R["sensor_e"][i], R["sensor_n"][i], R["sensor_u"][i])
            if REG_RANGE_MIN_M <= rng <= RANGE_MAX_M:
                reps.append(i)
                rr.append(float(wrap180(R["az_deg"][i] - az)))
                fl.append(tr.flight[j])
    reps, rr, fl = np.array(reps, np.int64), np.array(rr), np.array(fl, np.int64)
    res = scan_offset(reps, rr, seg=fl, rw=R["rw"][reps] if reps.size else None, est=est)
    res["blocks"] = []
    if reps.size:
        for m in time_blocks(R["t"][reps], reps):
            b = scan_offset(reps[m], rr[m], seg=fl[m], rw=R["rw"][reps[m]], est=est)
            res["blocks"].append((int(b["n_reports"]), float(b["offset"]), float(b["significance"])))
    return res


def label_oracle_scan(it: C.Iteration, det: dict, deployable: frozenset[str], sk: str, est: dict = EST_DEFAULT) -> dict:
    """Same scan restricted to the position segments labelled drone (testing labels)."""
    R = load_reports(det, it.key, deployable)
    pool = Pool(it, deployable)
    P = build_pairs(pool, R)
    m = (R["sensor_key"] == sk)[P["rep"]] & pool.reg_ok[P["seg"]] & (P["rng"] >= REG_RANGE_MIN_M)
    rep, seg = P["rep"][m], P["seg"][m]
    lab = np.array([str(x).startswith("drone") for x in pool.label[seg]]).astype(float)
    return scan_offset(rep, P["r_raw"][m], seg=seg, q=lab, rw=R["rw"][rep], seg_sensor=pool.sensor_key[seg],
                       cap=est["seg_cap"], prec=precision(P["sd"][m], P["rng"][m], est["prec_sigma0"]), est=est)


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #

def stack(results: dict[str, dict]) -> dict[str, np.ndarray]:
    keys = list(results.values())[0]["feats"].keys()
    out = {k: np.concatenate([r["feats"][k] for r in results.values()]) for k in keys}
    for m in ("label", "best_flight", "sensor_id"):
        out[m] = np.concatenate([r["meta"][m] for r in results.values()])
    return out


def auc_rows(F: dict, fams=FAMILIES) -> list[dict]:
    """Within-sensor AUC (drone vs clutter, testing, deployable non-relay sensors) of {family}_hit_share."""
    test = np.array([str(k).startswith("testing") for k in F["iter_key"]])
    lab = np.array([str(x) for x in F["label"]])
    drone = np.array([x.startswith("drone") for x in lab])
    drone_air = lab == "drone_air"
    nonrelay = ~np.isin(F["sensor_id"], list(Q.GT_FEED_SENSORS))
    dep = np.isin(F["sensor_id"], sorted(C.deployable_sensor_ids()))
    groups = [(sid, F["sensor_id"] == sid) for sid in ("EGOI", "NYTF", "UXYJ", "IUKL")]
    groups.append(("4 radars pooled", np.isin(F["sensor_id"], ["EGOI", "NYTF", "UXYJ", "IUKL"])))
    groups.append(("all deployable, no relay", dep & nonrelay))
    rows = []
    for fam in fams:
        hs, ov = F[f"{fam}_hit_share"], F[f"{fam}_n_reports_overlap"]
        for name, g in groups:
            base = test & g
            for tag, y in (("drone", drone), ("drone_air", drone_air)):
                cov = base & (ov > 0)
                a_cov = gbdt.auc(y[cov].astype(float), hs[cov]) if cov.sum() > 0 else float("nan")
                a_all = gbdt.auc(y[base].astype(float), np.nan_to_num(hs[base], nan=0.0)) if base.sum() > 0 else float("nan")
                rows.append(dict(family=fam, group=name, target=tag, n=int(base.sum()), n_drone=int((base & y).sum()),
                                 n_cov=int(cov.sum()), n_drone_cov=int((cov & y).sum()), auc_cov=a_cov, auc_all=a_all))
    return rows


def rf_consistency(F: dict, testing_iters: list[C.Iteration]) -> dict:
    """Band / protocol / model strings that reach drone segments against the catalog entry of the flight."""
    meta = {}
    for it in testing_iters:
        for f in T.load_truth(it, "all").flights:
            meta[(it.key, f["flight_key"])] = f
    cnt = {k: Counter() for k in ("rf_band", "rf_protocol", "rf_model")}
    verdict = {k: Counter() for k in cnt}
    field = {"rf_band": "signal_frequency_band", "rf_protocol": "protocol", "rf_model": "uav_model/manufacturer"}
    lab = np.array([str(x) for x in F["label"]])
    for i in np.flatnonzero(np.array([x.startswith("drone") for x in lab]) & np.array([str(k).startswith("testing") for k in F["iter_key"]])):
        for k in cnt:
            v = F[k][i]
            if not v:
                continue
            cnt[k][v] += 1
            fm = meta.get((F["iter_key"][i], F["best_flight"][i]))
            cat = fm[field[k]] if fm else "(no flight)"
            verdict[k]["match" if v == cat else ("catalog Unknown" if cat in ("Unknown", "Not Applicable") else "differs")] += 1
    return dict(counts=cnt, verdict=verdict)


def fmt(x, nd=2) -> str:
    return "-" if x is None or (isinstance(x, float) and x != x) else (f"{x:.{nd}f}" if isinstance(x, (float, np.floating)) else str(x))


def md_table(header, rows) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


# --------------------------------------------------------------------------- #
# self-checks
# --------------------------------------------------------------------------- #

def mini_pool(e: float, n: float, u: float = 50.0, grid_n: int = 20) -> Pool:
    """One stationary segment, for geometry checks."""
    pool = Pool.__new__(Pool)
    pool.it_key, pool.n_seg = "synthetic/mini", 1
    pool.seg_iter, pool.inner = np.array(["synthetic/mini"], dtype=object), {}
    pool.seg_id = np.zeros(1, np.int64)
    pool.sensor_key, pool.sensor_id = np.array(["NYTF_RADAR"], dtype=object), np.array(["NYTF"], dtype=object)
    pool.label, pool.best_flight = np.array(["clutter"], dtype=object), np.array([""], dtype=object)
    pool.reg_ok = pool.elig = np.ones(1, bool)
    pool.p = np.ones(1)
    pool.grid_t = 1.0e9 + 0.1 * np.arange(grid_n)
    pool.g_seg, pool.g_step = np.zeros(grid_n, np.int64), np.arange(grid_n)
    pool.g_e, pool.g_n, pool.g_u = np.full(grid_n, e), np.full(grid_n, n), np.full(grid_n, u)
    pool.g_sd = np.full(grid_n, 5.0)
    pool.starts = np.searchsorted(pool.g_step, np.arange(grid_n + 1))
    return pool


def check_geometry() -> tuple[bool, str]:
    ok = True
    msgs = []
    # 1a closed-form table (sensor at (1000, -500, 20))
    es, ns, us = 1000.0, -500.0, 20.0
    table = [((0, 100, 0), 0.0, 0.0), ((100, 0, 0), 90.0, 0.0), ((0, -100, 0), 180.0, 0.0), ((-100, 0, 0), -90.0, 0.0),
             ((100, 100, 100), 45.0, 35.26438968275465), ((-100, -100, -100), -135.0, -35.26438968275465),
             ((0, 0, 1000), 0.0, 90.0)]
    worst = 0.0
    for (de, dn, du), az_exp, el_exp in table:
        az, el, rng = predict_angles(es + de, ns + dn, us + du, es, ns, us)
        d = abs(float(wrap180(az - az_exp)))
        worst = max(worst, d, abs(float(el) - el_exp), abs(float(rng) - math.hypot(de, dn)))
    if worst > 1e-9:
        ok = False
    msgs.append(f"closed-form table max err {worst:.2e}")
    # 1b wrap
    w = wrap180(np.array([190.0, -190.0, 180.0, -180.0, 360.0, 540.0, 0.0]))
    if not np.allclose(w, [-170, 170, 180, 180, 0, 180, 0]):
        ok = False
    msgs.append(f"wrap180 -> {np.round(w, 1).tolist()}")
    # 1c round trip through the independent forward model C.bearing_offset
    rg = np.random.default_rng(SEED)
    az_t, el_t, r_t = rg.uniform(-180, 180, 2000), rg.uniform(-60, 60, 2000), rg.uniform(50, 3000, 2000)
    de, dn, du = C.bearing_offset(az_t, el_t, r_t)
    az, el, rng = predict_angles(es + de, ns + dn, us + du, es, ns, us)
    err = max(float(np.abs(wrap180(az - az_t)).max()), float(np.abs(el - el_t).max()), float(np.abs(rng - r_t * np.cos(np.radians(el_t))).max()))
    if err > 1e-6:
        ok = False
    msgs.append(f"round trip (2000 random az/el/range) max err {err:.2e}")
    # 1d wrap at +-180 through build_pairs: report -179.9, target at +179.95 -> residual -0.15 +- , both ways
    for az_meas, tgt_az in ((-179.9, 179.95), (179.9, -179.95), (0.05, -0.05), (-0.05, 359.95)):
        pool = mini_pool(2000.0 * math.sin(math.radians(tgt_az)), 2000.0 * math.cos(math.radians(tgt_az)))
        R = dict(t=np.array([pool.grid_t[3]]), az_deg=np.array([az_meas]), el_deg=np.array([np.nan]),
                 sensor_e=np.zeros(1), sensor_n=np.zeros(1), sensor_u=np.full(1, 20.0))
        P = build_pairs(pool, R)
        exp = float(wrap180(az_meas - tgt_az))
        if P["rep"].size != 1 or abs(float(P["r_raw"][0]) - exp) > 1e-6 or abs(exp) > 0.2:
            ok = False
            msgs.append(f"wrap case {az_meas}/{tgt_az}: got {P['r_raw']} expected {exp}")
    msgs.append("wrap cases at +-180 / north OK" if ok else "wrap case FAILED")
    return ok, "; ".join(msgs)


def check_injection_synthetic() -> tuple[bool, str]:
    """Known offsets in a synthetic world (the world is far less peaked than real data, so the acceptance
    thresholds are not used here: the estimate must be within 1 deg and its significance at least 2.5x that of
    the same world with every report pointing in a random direction)."""
    ok, msgs = True, []
    for true_off, seed in ((37.0, 1), (-160.0, 2), (0.0, 3)):
        pool, R, _ = synthetic_world(true_off, seed)
        res = register_sensor(pool, R, build_pairs(pool, R), "WVFX_CAMERA")
        poolN, RN, _ = synthetic_world(true_off, seed, false_share=1.0)
        null = register_sensor(poolN, RN, build_pairs(poolN, RN), "WVFX_CAMERA")
        err = abs(float(wrap180(res["offset"] - true_off)))
        good = err <= 1.0 and res["significance"] >= 2.5 * null["significance"]
        ok &= good
        msgs.append(f"inject {true_off:+.0f}: est {res['offset']:+.2f} (err {err:.2f}, sig {res['significance']:.1f} vs {null['significance']:.1f} with no signal)")
    return ok, "; ".join(msgs)


def semi_real_world(ctx: Ctx, true_offset: float, seed: int, n_rep: int = 1500, noise: float = 1.0, false_share: float = 0.3) -> dict:
    """Reports of a fictitious camera, placed at the real WVFX/EMXG site of the iteration, over the REAL position tracks
    and REAL existence p of that iteration (testing labels pick which segments the camera 'sees': a test fixture, not an
    estimator input).  Report azimuth = true azimuth of a random airborne-drone segment sample minus `true_offset` plus
    noise; `false_share` of the reports point at random directions.  So the offset to add is `true_offset`."""
    rng = np.random.default_rng(seed)
    pool, R0 = ctx.pool, ctx.R
    site = R0["sensor_key"] == sorted(set(R0["sensor_key"]))[0]
    es, ns, us = (float(np.median(R0[k][site])) for k in ("sensor_e", "sensor_n", "sensor_u"))
    ok = np.flatnonzero(np.array([str(x) == "drone_air" for x in pool.label])[pool.g_seg])
    az, _, rg = predict_angles(pool.g_e[ok], pool.g_n[ok], pool.g_u[ok], es, ns, us)
    ok = ok[(rg >= 150.0) & (rg <= 2500.0)]
    pick = ok[rng.integers(0, ok.size, n_rep)]
    az, _, _ = predict_angles(pool.g_e[pick], pool.g_n[pick], pool.g_u[pick], es, ns, us)
    az = az - true_offset + rng.normal(0.0, noise, n_rep)
    false = rng.random(n_rep) < false_share
    az[false] = rng.uniform(-180.0, 180.0, int(false.sum()))
    t = pool.grid_t[pool.g_step[pick]]
    o = np.argsort(t, kind="stable")
    return dict(sensor_key=np.full(n_rep, "WVFX_CAMERA", dtype=object), sensor_id=np.full(n_rep, "WVFX", dtype=object),
                iter_key=np.full(n_rep, pool.it_key, dtype=object), t=t[o], az_deg=wrap180(az[o]), el_deg=np.full(n_rep, np.nan),
                az_err_deg=np.full(n_rep, np.nan), sensor_e=np.full(n_rep, es), sensor_n=np.full(n_rep, ns), sensor_u=np.full(n_rep, us),
                fam=np.zeros(n_rep, np.int64), rw=np.ones(n_rep))


def check_injection_real(ctxs: dict[str, Ctx]) -> tuple[bool, str]:
    """Known offsets on real position tracks: the estimate must be within 1 deg, and its significance at least 2.5x the
    significance of the same tracks with every report pointing in a random direction.  The estimator's weights, cap and
    mean-shift all see real clutter and real track geometry here (an azimuth shift of real reports would be a tautology:
    the whole histogram moves by the shift)."""
    ok, msgs = True, []
    for ik, inj, seed in (("testing/S3-I3", 37.0, 11), ("testing/S3-I1", -160.0, 12), ("testing/S4-I4", 90.0, 13)):
        ctx = ctxs[ik]
        R = semi_real_world(ctx, inj, seed)
        res = register_sensor(ctx.pool, R, build_pairs(ctx.pool, R), "WVFX_CAMERA")
        RN = semi_real_world(ctx, inj, seed, false_share=1.0)
        null = register_sensor(ctx.pool, RN, build_pairs(ctx.pool, RN), "WVFX_CAMERA")
        err = abs(float(wrap180(res["offset"] - inj)))
        good = err <= 1.0 and res["significance"] >= 2.5 * null["significance"]
        ok &= good
        msgs.append(f"{ik} inject {inj:+.0f}: est {res['offset']:+.2f} (err {err:.2f}, sig {res['significance']:.1f} vs {null['significance']:.1f} with no signal, accepted {int(res['accepted'])})")
    return ok, "; ".join(msgs)


def gt_breakdown(regs: list[dict]) -> dict:
    """Outcome of every GT row (n >= GT_MIN_N): accepted & agree / accepted & wrong / unaccepted (+ whether the effective offset 0
    is right and whether the raw peak was close)."""
    rows = [r for r in regs if r["gt_n"] >= GT_MIN_N]
    acc_ok = [r for r in rows if r["accepted"] and r["agree"]]
    acc_bad = [r for r in rows if r["accepted"] and not r["agree"]]
    un = [r for r in rows if not r["accepted"]]
    return dict(rows=rows, acc_ok=acc_ok, acc_bad=acc_bad, unaccepted=un,
                un_zero_right=[r for r in un if r["eff_agree"]], un_raw_close=[r for r in un if r["raw_agree"]],
                eff_ok=[r for r in rows if r["eff_agree"]])


def check_gt_agreement(regs: list[dict]) -> tuple[bool, str, list[str]]:
    """GT-free ACCEPTED offsets against the GT rows with n >= GT_MIN_N.  A row counts as agreeing only when the sensor was
    registered (accepted) and the accepted offset is within AGREE_DEG of the GT value; an unregistered row (offset 0 by
    default) is not an agreement, even where the GT value happens to be 0.  The share of rows that agree must be >= 80 %."""
    d = gt_breakdown(regs)
    rows = d["rows"]
    lines = []
    for r in rows:
        verdict = ("AGREE" if r["agree"] else "DISAGREE (accepted, wrong)") if r["accepted"] else \
                  f"not registered (effective 0 {'= GT' if r['eff_agree'] else '!= GT'}, raw {'close to' if r['raw_agree'] else 'far from'} GT)"
        lines.append(f"    {r['iter_key']:<14} {r['sensor_key']:<16} GT {r['gt_offset']:+7.1f} (n {r['gt_n']:>4})  ours {r['offset']:+7.1f}"
                     f" (raw {r['raw_offset']:+7.1f}, sig {r['significance']:6.1f}, {r['method']}, status {r['status']}, "
                     f"blocks off by >{SPLIT_TOL_DEG:g}: {n_contradicting_blocks(r)})  {verdict}")
    n, k = len(rows), len(d["acc_ok"])
    ok = n > 0 and k >= 0.8 * n
    prec = k / max(len(d["acc_ok"]) + len(d["acc_bad"]), 1)
    msg = (f"{k}/{n} GT rows (n>={GT_MIN_N}) have an accepted offset within {AGREE_DEG:g} deg ({100 * k / max(n, 1):.0f} %, need 80 %); "
           f"accepted rows: {len(d['acc_ok'])} right, {len(d['acc_bad'])} wrong (precision {100 * prec:.0f} %); "
           f"{len(d['unaccepted'])} rows not registered ({len(d['un_zero_right'])} of them have GT 0 so the default offset 0 is right, "
           f"{len(d['un_raw_close'])} had a raw peak within {AGREE_DEG:g} deg of GT); offset actually used is right in {len(d['eff_ok'])}/{n}")
    return ok, msg, lines


def provenance_violations(regs: list[dict], testing_only_ids: set[str]) -> list[str]:
    bad = set(Q.GT_FEED_SENSORS) | set(testing_only_ids)
    out = []
    for r in regs:
        for sk in r["src_sensors"]:
            if sk.split("_")[0] in bad:
                out.append(f"{r['iter_key']} {r['sensor_key']}: position sensor {sk} is a relay / testing-only sensor")
        if r["sensor_key"].split("_")[0] in bad:
            out.append(f"{r['iter_key']}: bearing sensor {r['sensor_key']} is testing-only")
        if any(i != r["iter_key"] for i in r["src_iters"]):
            out.append(f"{r['iter_key']} {r['sensor_key']}: candidates from other iterations {r['src_iters']}")
    return out


def check_provenance(regs: list[dict], ctxs: dict[str, Ctx], deployable: frozenset[str]) -> tuple[bool, str]:
    """src_sensors / src_iters are MEASURED from the candidate pairs that entered each registration (the sensor key and the
    iter_key column of tracks.csv for every candidate segment, the iter_key of every bearing report).  The detector must be
    able to fire: doctored rows, the real code path with the relay filter removed, and the real code path with a foreign
    iteration's segment table spliced in."""
    testing_only = {sf.sensor_id for it in C.iterations() for sf in C.sensor_files(it)} - set(deployable)
    viol = provenance_violations(regs, testing_only)
    n_eval = sum(r["iter_key"].startswith("evaluation") for r in regs)
    mislabelled = [ik for ik, c in ctxs.items() if (c.pool.seg_iter != ik).any()]
    doctored = [dict(regs[0], src_sensors=regs[0]["src_sensors"] + ["LJJX_FUSION-NODE"]), dict(regs[0], src_iters=["testing/S1-I1", "evaluation/DI/S1-I3"])]
    fired_a = len(provenance_violations(doctored, testing_only)) == 2
    fired_b = fired_c = False
    for ik, c in ctxs.items():
        if not c.it.has_groundtruth or c.R is None or not any(s in testing_only | set(Q.GT_FEED_SENSORS) for s in c.pool.sensor_id):
            continue
        sk = c.sensors[0]
        pool = Pool.__new__(Pool)
        pool.__dict__.update(c.pool.__dict__)
        pool.reg_ok = np.ones(pool.n_seg, bool)                       # break the relay filter on purpose
        res = register_sensor(pool, c.R, c.P, sk)
        res.update(iter_key=ik, sensor_key=sk)
        fired_b = len(provenance_violations([res], testing_only)) > 0
        pool2 = Pool.__new__(Pool)
        pool2.__dict__.update(c.pool.__dict__)
        pool2.seg_iter = np.where(np.arange(pool2.n_seg) % 2 == 0, "evaluation/DI/S1-I3", pool2.seg_iter)   # a foreign table spliced in
        res2 = register_sensor(pool2, c.R, c.P, sk)
        res2.update(iter_key=ik, sensor_key=sk)
        fired_c = len(provenance_violations([res2], testing_only)) > 0
        break
    ok = (not viol and not mislabelled and fired_a and fired_b and fired_c
          and not any(r["sensor_key"].startswith("NJQL") for r in regs))
    return ok, (f"{len(regs)} registrations ({n_eval} evaluation) used position sensors "
                f"{sorted({s for r in regs for s in r['src_sensors']})}; violations {len(viol)}; tracks.csv iter_key differs from the "
                f"iteration in {len(mislabelled)} iterations; the iteration keys of the candidates were measured; detector fires on "
                f"doctored rows {fired_a}, with the relay filter removed {fired_b}, with a foreign segment table {fired_c}")


def check_loio(results: dict, ctxs: dict[str, Ctx]) -> tuple[bool, str]:
    """Leave-one-iteration-out integrity of the choices.  For every testing iteration k: k is not among the training
    iterations of its choice, the training rows used p_inner(fold=k) (a model that saw neither k nor the row's own
    iteration; for those iterations it differs from the ordinary p), the choice that was applied to k is the one made
    for fold k; evaluation iterations got the all-testing choice.  The detector must fire on a doctored choice."""
    problems = []
    n_diff = 0
    for ik, res in results.items():
        ch = res["choice"]
        if ik.startswith("testing"):
            if ch["held"] != ik or ik in ch["train"]:
                problems.append(f"{ik}: choice held {ch['held']} train {ch['train']}")
            if any(j == ik for j, _, _ in ch["rows"]):
                problems.append(f"{ik}: its own GT row was used to choose its configuration")
            for (j, sk, _), kind in zip(ch["rows"], ch["p_kind"]):
                pin = ctxs[j].pool.inner.get(ik)
                if kind != "inner:" + ik or pin is None:
                    problems.append(f"{ik}: training row {j} {sk} used p kind {kind}")
                elif not np.allclose(pin, ctxs[j].pool.p):
                    n_diff += 1
        else:
            if ch["held"] is not None or ch["train"] != sorted(k for k in ctxs if k.startswith("testing")):
                problems.append(f"{ik}: evaluation choice is not the all-testing choice")
    def violations(ch: dict, ik: str) -> int:
        return int(ch["held"] != ik) + int(ik in ch["train"]) + int(any(j == ik for j, _, _ in ch["rows"]))
    doctored = dict(results["testing/S1-I1"]["choice"], held="testing/S2-I2")
    fired = violations(doctored, "testing/S1-I1") > 0 and violations(dict(results["testing/S1-I1"]["choice"], train=["testing/S1-I1"]), "testing/S1-I1") > 0
    ok = not problems and fired and n_diff > 0
    return ok, (f"{sum(k.startswith('testing') for k in results)} testing iterations each chose estimator / acceptance rule / gates without its own rows "
                f"(training rows used p_inner of the held-out fold, which differs from the ordinary p in {n_diff} rows); "
                f"{sum(k.startswith('evaluation') for k in results)} evaluation iterations use the choice made on all 6 testing iterations; "
                f"problems {problems if problems else 'none'}; detector fires on a doctored choice {fired}")


def check_features(results: dict, ctxs: dict[str, Ctx], n_target: int = 200) -> tuple[bool, str]:
    rnd = random.Random(SEED)
    n_seg = n_pairs = n_hits = n_bad = 0
    detail = []
    for ik in ("testing/S1-I1", "evaluation/DIII/S2-I3"):
        res = results[ik]
        F = res["feats"]
        ctx = ctxs[ik]
        R, pool = ctx.R, ctx.pool
        offsets = {r["sensor_key"]: r["offset"] for r in res["regs"]}
        gates = res["choice"]["gates"]
        by_step, by_seg = reference_context(ctx.it)
        ov = sum(F[f"{f}_n_reports_overlap"] for f in FAMILIES)
        hh = sum(F[f"{f}_n_hits"] for f in FAMILIES)
        cand_any = [int(x) for x in np.flatnonzero(ov > 0)]
        cand_hit = [int(x) for x in np.flatnonzero(hh > 0)]
        pick = set(rnd.sample(cand_any, min(len(cand_any), n_target // 2)))
        pick |= set(rnd.sample(cand_hit, min(len(cand_hit), n_target // 2)))
        grid_list = pool.grid_t.tolist()
        elig = pool.elig.tolist()
        for sg in sorted(pick):
            ref = reference_features(sg, R, offsets, grid_list, by_step, by_seg, elig, gates)
            n_seg += 1
            n_pairs += sum(ref[f"{f}_n_reports_overlap"] for f in FAMILIES)
            n_hits += sum(ref[f"{f}_n_hits"] for f in FAMILIES)
            for k, v in ref.items():
                got = F[k][sg]
                same = (v == got) if isinstance(v, str) else ((v != v and got != got) or abs(float(v) - float(got)) <= 1e-9 * max(1.0, abs(float(v))))
                if not same:
                    n_bad += 1
                    if len(detail) < 5:
                        detail.append(f"{ik} seg {sg} {k}: loop {v} vs vectorised {got}")
        del by_step, by_seg
    ok = n_seg >= n_target and n_pairs >= n_target and n_bad == 0
    msg = f"{n_seg} segments / {n_pairs} (segment, report) pairs ({n_hits} hits) recomputed with plain loops; {n_bad} feature mismatches"
    if detail:
        msg += " :: " + " | ".join(detail)
    return ok, msg


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def write_report(regs: list[dict], results: dict, F: dict, checks: list[tuple[int, str, bool, str]], gt_lines: list[str],
                 diag: dict, auc: list[dict], rfc: dict, cover: list[list], choices: dict, secs: float) -> None:
    L: list[str] = []
    ok_all = all(c[2] for c in checks)
    failed = [c[0] for c in checks if not c[2]]
    L += ["# Bearing support and azimuth registration", "",
          f"**Self-checks: {'all passed' if ok_all else 'NOT all passed: check ' + ', '.join(map(str, failed)) + ' (see below, kept failing on purpose)'}.** Run time {secs:.0f} s. "
          "Generated by `tools/bearing_support.py`; rules, constants and output schemas are in its docstring.", "",
          "## Plain-language glossary", "",
          "* **Bearing sensor**: a sensor that reports only a direction (azimuth = compass angle clockwise from north, sometimes also elevation) from a known spot: cameras (WVFX, FGIX), the acoustic array (EMXG), radio direction finders (ORFH, YXAB, OEPS, GTYT).",
          "* **Position track segment**: one stretch of one position sensor's track (radar, RF position, cyber, UTM), from `sensor_tracks_v2`; this tool scores each one.",
          "* **Azimuth offset / registration**: the fixed angle a bearing sensor is rotated by (EMXG is flipped by about -160 to -172 degrees in some iterations). Registration estimates it WITHOUT ground truth by asking which rotation makes the bearings line up with the position tracks. The value is what you ADD to the reported azimuth.",
          "* **Residual r**: reported azimuth minus the azimuth you would predict from a track's position, wrapped to -180..180 degrees.",
          "* **Significance**: height of the best bin of the residual histogram divided by the typical bin. About 1 means no structure; large means a clear peak.",
          "* **Time blocks**: the reports of one sensor cut into three time slices with equal report counts. A single per-iteration offset must show up in every slice that has enough data; a slice with its own clear peak (significance >= 10) more than 20 degrees away contradicts it.",
          "* **Accepted / unregistered**: accepted = the offset passed every test and is applied; unregistered = it did not, so offset 0 is used. **Presence-only**: ORFH and YXAB, whose bearing does not follow drones, are never accepted.",
          "* **Gate**: the largest angular miss still counted as 'the sensor points at this track'. **Hit**: a report inside the gate (and elevation gate). **hit_share**: hits divided by the reports that could have seen the track.",
          "* **GOSPA** (the track-accuracy score): not used here. **LOIO** (leave-one-iteration-out): every choice applied to testing iteration k was made on the other five testing iterations only, so the testing numbers are honest estimates; evaluation iterations get the choice made on all six testing iterations.",
          "* **AUC**: probability that a random drone segment scores higher than a random clutter segment (0.5 = useless, 1 = perfect).",
          "* **Existence probability p**: `data/derived/existence/*__p.csv`, the earlier model's guess that a segment is a drone (testing = leave-one-iteration-out predictions, evaluation = final model). When tuning for held-out k, other iterations use `p_inner` (a model that saw neither k nor that iteration).", ""]
    L += ["## What was built", "",
          "Per iteration and bearing sensor: an azimuth offset estimated from the position tracks only (no ground truth, no relay feeds VKYY / QTVS / LJJX, no testing-only sensors EUIG / MMJJ, and the testing-only camera NJQL is dropped entirely). Then, per position segment, camera / acoustic / RF hit features. "
          "Fixed (physical or a-priori) constants: " + ", ".join(f"{k} {v}" for k, v in (("range max", f"{RANGE_MAX_M:g} m"), ("registration range min", f"{REG_RANGE_MIN_M:g} m"),
                                                            ("el gates", EL_GATE_DEG), ("K_SIGMA", K_SIGMA), ("min reports", MIN_REPORTS),
                                                            ("min peak segments / sensors", f"{MIN_SEG_PEAK} / {MIN_SENS_PEAK}"),
                                                            ("time blocks", f"{SPLIT_K} (testable from {SPLIT_MIN_REPORTS} reports, own significance >= {SPLIT_SIG:g}, tolerance {SPLIT_TOL_DEG:g} deg)"))) + ".", "",
          "**Deviation from the brief's weighting (and why).** The brief weights each candidate 1 / (number of candidates of the report). With 50-100 candidates per report, 90 % of them radar clutter, that histogram has significance 3-5 and its peak is wrong in most testing rows. The delivered estimator therefore (a) weights candidates by the existence probability p, (b) counts each bearing sensor-second once, (c) caps what any one segment can contribute, (d) weights pairs by the angular precision of the position track, (e) averages the position sensors' independent biases by a mean-shift, (f) accepts only a peak carried by >= 3 segments from >= 2 position sensors, above a significance threshold, with no strong rival peak and no contradicting time block. "
          "The brief's plain weighting is still computed and shown as 'plain' in the table.", "",
          "**What is chosen leave-one-iteration-out.** The estimator shape (segment cap, angular precision scale, mean-shift window), the acceptance rule (significance threshold, rival-peak ratio, time-block mode) and the azimuth gate floors of the three families are chosen per testing iteration k on the other testing iterations: shape and rule maximise a utility (+1 for an accepted offset within 3 degrees of a GT row, -2 for an accepted wrong one, 0 for unaccepted) over their GT rows (with p_inner of fold k); gates maximise the pooled drone-vs-clutter AUC of `hit_share` over the four radars. Ties go to the a-priori default. Evaluation uses the choice made on all six testing iterations. "
          "Still fixed by hand (not LOIO): the constants listed above, the sector-sensor test (SECTOR_*), and the grids themselves. With only 6 to 9 GT rows the selection is noisy (see the table below), and the gate choice sits on the edge of its grid for rf.", ""]
    L += ["### Choices (one row per fold)", ""]
    rows = []
    for held, ch in choices.items():
        rows.append([held or "evaluation (all 6)", f"{ch['est']['seg_cap']:g} / {ch['est']['prec_sigma0']:g} / {ch['est']['ms_hw']:g}",
                     f"{ch['acc']['sig_min']:g} / {ch['acc']['max_second']:g} / {ch['acc']['split']}", f"{ch['gates']['camera']:g} / {ch['gates']['acoustic']:g} / {ch['gates']['rf']:g}",
                     ch["n_rows"], f"{ch['util']:+.0f}"])
    L += md_table(["applied to", "segment cap / precision scale / mean-shift window", "min significance / max rival ratio / block mode (0 off, 1 <=1 bad block, 2 none)",
                   "gate floors camera / acoustic / rf (deg)", "GT rows", "utility on them"], rows)
    L += ["", "## Self-checks", ""]
    for num, name, ok, msg in checks:
        L.append(f"{num}. **{'PASS' if ok else 'FAIL'}** {name}: {msg}")
    L += ["", "Check 3 rows (every GT 'applied' row with n >= 50):", "", "```"] + gt_lines + ["```", ""]
    L += ["## Registration, every iteration and bearing sensor", "",
          "`offset` is the applied value (0 when unregistered or presence-only); `raw` the peak estimate; `blocks` = reports : peak : significance in each of the three time blocks; `az sd` = spread (degrees) of the reported azimuths behind the peak, information only: a camera that stares in one direction gives about 5-10 and its 'peak' is then a coincidence with whatever track lies in that direction. `peak seg/sens` = position segments / sensors carrying the peak. `plain` = the brief's unweighted estimate (accepted-or-raw). `agree` is only filled for ACCEPTED offsets with a GT row; `used ok` asks whether the offset actually used (0 if unregistered) is within 3 degrees of GT. GT columns exist for testing only (`sensor_calibration.csv`, made WITH ground truth, used for comparison only).", ""]
    rows = []
    for r in regs:
        rows.append([r["iter_key"], r["sensor_key"], r["method"], r["status"], fmt(r["offset"], 1), fmt(r["raw_offset"], 1), fmt(r["significance"], 1),
                     r["n_reports"], f"{r['n_seg_peak']}/{r['n_sens_peak']}", " ".join(f"{n}:{o:+.0f}:{sg:.0f}" for n, o, sg in r["blocks"]), fmt(r.get("az_spread_deg", float("nan")), 0),
                     fmt(r["gt_offset"], 1) + (f" ({r['gt_n']})" if r["gt_n"] else ""), "" if r["agree"] is None else ("yes" if r["agree"] else "NO"),
                     "" if r["eff_agree"] is None else ("yes" if r["eff_agree"] else "NO"), fmt(r["plain_offset"], 1)])
    L += md_table(["iteration", "sensor", "method", "status", "offset", "raw", "signif.", "reports", "peak seg/sens", "blocks", "az sd", "GT (n)", "agree", "used ok", "plain"], rows)
    L += ["", "### Why check 3 behaves as it does (testing, ground truth used for diagnosis only)", "",
          "Same histogram method, different candidate sets, against the GT offset of `sensor_calibration.csv`. 'Dronetag' = the real flight positions as candidates (the best any position-track registration could do with this method), with the peaks of its three time blocks; 'labelled drones' = only the position segments labelled drone; 'delivered' = the GT-free estimate (raw peak).", ""]
    rows = []
    for (ik, sk), d in sorted(diag.items()):
        r = next(x for x in regs if x["iter_key"] == ik and x["sensor_key"] == sk)
        rows.append([ik, sk, fmt(r["gt_offset"], 1), fmt(d["dronetag"]["offset"], 1) + f" (sig {d['dronetag']['significance']:.0f})",
                     " ".join(f"{o:+.0f}" for n, o, sg in d["dronetag"]["blocks"]),
                     fmt(d["label"]["offset"], 1) + f" (sig {d['label']['significance']:.0f})", fmt(r["raw_offset"], 1) + f" (sig {r['significance']:.0f})"])
    L += md_table(["iteration", "sensor", "GT offset", "Dronetag candidates", "Dronetag, 3 time blocks", "labelled drones", "delivered (raw)"], rows)
    def _ag(off, gtv):
        return abs(float(wrap180(off - gtv))) <= AGREE_DEG
    n_d = len(diag)
    def _gtof(ik, sk):
        return next(x for x in regs if x["iter_key"] == ik and x["sensor_key"] == sk)["gt_offset"]
    k_dt = sum(_ag(d["dronetag"]["offset"], _gtof(ik, sk)) for (ik, sk), d in diag.items())
    k_lb = sum(_ag(d["label"]["offset"], _gtof(ik, sk)) for (ik, sk), d in diag.items())
    k_rw = sum(_ag(next(x for x in regs if x["iter_key"] == ik and x["sensor_key"] == sk)["raw_offset"], _gtof(ik, sk)) for (ik, sk) in diag)
    L += ["", f"Within {AGREE_DEG:g} degrees of the calibration value: Dronetag candidates {k_dt}/{n_d}, labelled drones {k_lb}/{n_d}, delivered raw peak {k_rw}/{n_d}. "
          "Two findings. (1) Even the true flight positions do not reproduce every calibration row: testing S1-I1 WVFX has TWO regimes (with the real flight positions the first two time blocks peak near -115 and the last near 0), so no single per-iteration offset exists there and the calibration row (+5) describes only part of the reports; the time-block test is what stops a GT-free estimator from picking the early one. "
          "(2) For EMXG S3-I2 the GT-free peak (-164) sits 7.7 degrees from the calibration value (-156.5); the same calibration tool's own cross-checks give -163, and the radars' position bias (10-25 m, i.e. 2-5 degrees at 300-500 m) sets a floor on GT-free accuracy. "
          "(3) Some wrong high-significance peaks are stares: testing S3-I2 WVFX reports azimuths 43-61 degrees almost all the time (az sd 9) and its +169 peak is a cluster of radar clutter that happens to lie in that direction, consistent in all three time blocks; testing S2-I2 EMXG is similar. A GT-free test cannot tell such a stare from a hovering drone. "
          "An acoustic array can also point 180 degrees away from a real drone (testing S2-I2), which is why ambiguous histograms are rejected.", ""]
    L += ["## Support features: coverage", ""]
    L += md_table(["iteration", "segments", "bearing reports", "camera: segs with overlap / with hit", "acoustic: overlap / hit", "rf: overlap / hit"], cover)
    L += ["", "## Does the support separate drones from clutter? (testing, within sensor)", "",
          "AUC of `<family>_hit_share`, drone (drone_air + drone_ground) vs clutter. 'covered' = only segments that had >= 1 overlapping report (n_cov); 'all' = segments without overlap scored 0. Segment labels are sensor_tracks_v2's (testing GT, used only to score). The gate floors used for testing iteration k were chosen without k (LOIO, on exactly this AUC over the other iterations), so these numbers are honest for the gates; the registration offsets are LOIO too.", ""]
    rows = [[a["family"], a["group"], a["target"], a["n"], a["n_drone"], a["n_cov"], a["n_drone_cov"], fmt(a["auc_cov"], 3), fmt(a["auc_all"], 3)] for a in auc]
    L += md_table(["family", "group", "target", "segments", "drone", "n_cov", "drone cov", "AUC covered", "AUC all"], rows)
    lab = np.array([str(x) for x in F["label"]])
    test = np.array([str(k).startswith("testing") for k in F["iter_key"]])
    drone = np.array([x.startswith("drone") for x in lab]) & test
    L += ["", "### Drone segments with any hit (testing)", ""]
    rows = []
    for fam in FAMILIES:
        for tag, m in (("drone (air + ground)", drone), ("drone_air", (lab == "drone_air") & test)):
            rows.append([fam, tag, int(m.sum()), int((m & (F[f'{fam}_n_reports_overlap'] > 0)).sum()), int((m & (F[f'{fam}_n_hits'] > 0)).sum())])
    L += md_table(["family", "segments", "n", "with overlap", "with >= 1 hit"], rows)
    L += ["", "## Metadata strings that reach drone segments (testing)", "",
          "Most common mapped (exact dropdown) string among the RF hit reports of a segment, compared with the catalog entry of the segment's best-matching Dronetag flight (`truth.py` flight labels). 'catalog Unknown' = the catalog has no value to compare with.", ""]
    for k in ("rf_band", "rf_protocol", "rf_model"):
        L.append(f"* **{k}**: strings {dict(rfc['counts'][k])}; against the catalog {dict(rfc['verdict'][k])}")
    L += ["", "Camera class text: `camera_uav_share` = share of camera hits whose class string contains 'uav' / 'drone' (WVFX writes 'UAV Rotary Wing' / 'UAV Fixed Wing'; FGIX and EMXG only 'Air Vehicle', so they count as not-UAV).", "",
          "## Caveats", "",
          "* Constants not chosen LOIO (listed above) were read off pooled testing residuals or set a priori; the same values are used on evaluation, which never tunes anything.",
          "* With 6 to 9 GT rows the LOIO selection of the acceptance rule is noisy; different folds pick different rules (table above). That is the honest cost of having no ground truth on evaluation.",
          "* ORFH / YXAB are presence-only by the brief and by `sensor_calibration.csv` (their bearings do not follow drones); a stable peak such as YXAB -34.5 in testing S2-I2 and evaluation DII is more likely a fixed ground source than a registration, and is never applied.",
          "* The sensor_noise.csv / sensor_tracks variances and height offsets were fitted on all six testing iterations (inherited in-sample caveat).",
          "* VKYY / QTVS are deployable UTM relays whose testing segments relay Dronetag, so any testing statistic involving them is optimistic; they are excluded from registration and from the AUC groups, but they do receive support features.",
          "* Offsets are never applied to anything outside this tool; downstream users should read `registration.csv`, `loio_choices.csv` and `<slug>__support.csv`."]
    (OUT / "bearing_support.md").write_text("\n".join(L) + "\n")


def coverage_rows(results: dict) -> list[list]:
    rows = []
    for ik, r in results.items():
        F = r["feats"]
        row = [ik, int(F["seg_id"].size), r["n_reports"]]
        for fam in FAMILIES:
            row.append(f"{int((F[f'{fam}_n_reports_overlap'] > 0).sum())} / {int((F[f'{fam}_n_hits'] > 0).sum())}")
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    deployable = C.deployable_sensor_ids()
    gt = gt_offsets()
    log("loading detections ...")
    det = Q.load_clean(DET_COLS)
    log(f"  done in {time.time() - t0:.0f} s")
    ctxs: dict[str, Ctx] = {}
    for it in C.iterations():
        ctxs[it.key] = Ctx(it, det, deployable)
    testing_keys = sorted(k for k in ctxs if k.startswith("testing"))
    log(f"contexts built [{time.time() - t0:.0f} s]; choosing configurations leave-one-iteration-out ...")
    choices: dict = {}
    for held in testing_keys + [None]:
        choices[held] = choose_fold(held, ctxs, gt, testing_keys)
        ch = choices[held]
        log(f"  fold {held or 'all testing (for evaluation)'}: est#{ch['est_idx']} acc#{ch['acc_idx']} utility {ch['util']:+.1f} on {ch['n_rows']} rows, "
            f"gates {ch['gates']}  [{time.time() - t0:.0f} s]")
    results: dict[str, dict] = {}
    for ik, ctx in ctxs.items():
        choice = choices[ik] if ik.startswith("testing") else choices[None]
        res = process_iteration(ctx, choice, gt)
        results[ik] = res
        write_support(C.iter_slug(ik), res["feats"])
        log(f"{ik}: {res['n_reports']} bearing reports, {len(res['regs'])} sensors registered "
            + ", ".join(f"{r['sensor_key'].split('_')[0]} {r['offset']:+.1f}{'*' if r['accepted'] else ''}" for r in res["regs"])
            + f"  [{time.time() - t0:.0f} s]")
    regs = [r for res in results.values() for r in res["regs"]]
    write_registration(regs)
    write_choices(choices)

    # diagnostics (testing GT; not inputs)
    its = {i.key: i for i in C.iterations()}
    diag = {}
    for (ik, sk), (v, n) in sorted(gt.items()):
        if sk.split("_")[0] in SECTOR_IDS or n < GT_MIN_N:
            continue
        est = results[ik]["choice"]["est"]
        diag[(ik, sk)] = dict(dronetag=dronetag_scan(its[ik], det, deployable, sk, est), label=label_oracle_scan(its[ik], det, deployable, sk, est))
    del det

    # self-checks
    checks: list[tuple[int, str, bool, str]] = []
    ok1, m1 = check_geometry()
    checks.append((1, "azimuth / elevation geometry against closed form", ok1, m1))
    ok2a, m2a = check_injection_synthetic()
    ok2b, m2b = check_injection_real(ctxs)
    checks.append((2, "registration recovers an injected offset within 1 deg (synthetic world; real position tracks)", ok2a and ok2b, m2a + " || " + m2b))
    ok3, m3, gt_lines = check_gt_agreement(regs)
    checks.append((3, "GT-free ACCEPTED offsets agree (3 deg) with the GT 'applied' rows, >= 80 %", ok3, m3))
    ok4, m4 = check_features(results, ctxs)
    checks.append((4, "features equal a plain-python loop", ok4, m4))
    ok5, m5 = check_provenance(regs, ctxs, deployable)
    checks.append((5, "no relay / testing-only sensor and no foreign iteration entered a registration (measured)", ok5, m5))
    ok6, m6 = check_loio(results, ctxs)
    checks.append((6, "every choice (estimator, acceptance rule, gates) was made without the iteration it is applied to", ok6, m6))

    F = stack(results)
    auc = auc_rows(F)
    rfc = rf_consistency(F, [i for i in C.iterations() if i.has_groundtruth])
    write_report(regs, results, F, checks, gt_lines, diag, auc, rfc, coverage_rows(results), choices, time.time() - t0)

    print()
    for num, name, ok, msg in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] ({num}) {name}\n        {msg}")
    print("\ncheck 3 rows:")
    print("\n".join(gt_lines))
    print("\nreport-only (not gated):")
    for a in auc:
        if a["target"] == "drone" and a["group"] in ("EGOI", "NYTF", "UXYJ", "IUKL", "4 radars pooled", "all deployable, no relay"):
            print(f"  AUC {a['family']:<8} {a['group']:<26} covered {fmt(a['auc_cov'], 3)} (n {a['n_cov']}, drone {a['n_drone_cov']})  all {fmt(a['auc_all'], 3)}")
    lab = np.array([str(x) for x in F["label"]])
    dr = np.array([x.startswith("drone") for x in lab]) & np.array([str(k).startswith("testing") for k in F["iter_key"]])
    for fam in FAMILIES:
        print(f"  drone segments (testing) with a {fam} hit: {int((dr & (F[f'{fam}_n_hits'] > 0)).sum())} of {int(dr.sum())}"
              f" ({int((dr & (F[f'{fam}_n_reports_overlap'] > 0)).sum())} had overlap)")
    for k in ("rf_band", "rf_protocol", "rf_model"):
        print(f"  {k}: strings {dict(rfc['counts'][k])} vs catalog {dict(rfc['verdict'][k])}")
    print(f"\nwrote {OUT}/ ({len(results)} support files, registration.csv, loio_choices.csv, bearing_support.md) in {time.time() - t0:.0f} s")
    bad = [c[0] for c in checks if not c[2]]
    if bad:
        print(f"SELF-CHECKS FAILED: {bad}")
        return 1
    print("all self-checks passed")
    return 0


def write_choices(choices: dict) -> None:
    cols: dict[str, list] = {k: [] for k in ("applied_to", "train_iterations", "est_idx", "seg_cap", "prec_sigma0", "ms_hw_deg", "acc_idx",
                                              "sig_min", "max_second_ratio", "split_mode", "gate_camera", "gate_acoustic", "gate_rf",
                                              "n_gt_rows", "utility")}
    for held, ch in choices.items():
        cols["applied_to"].append(held or "evaluation (all 6 testing iterations)")
        cols["train_iterations"].append("|".join(ch["train"]))
        cols["est_idx"].append(ch["est_idx"])
        cols["seg_cap"].append(ch["est"]["seg_cap"])
        cols["prec_sigma0"].append(ch["est"]["prec_sigma0"])
        cols["ms_hw_deg"].append(ch["est"]["ms_hw"])
        cols["acc_idx"].append(ch["acc_idx"])
        cols["sig_min"].append(ch["acc"]["sig_min"])
        cols["max_second_ratio"].append(ch["acc"]["max_second"])
        cols["split_mode"].append(ch["acc"]["split"])
        for f in FAMILIES:
            cols[f"gate_{f}"].append(ch["gates"][f])
        cols["n_gt_rows"].append(ch["n_rows"])
        cols["utility"].append(round(ch["util"], 2))
    C.write_columns(OUT / "loio_choices.csv", {k: np.array(v, dtype=object) for k, v in cols.items()}, list(cols))


if __name__ == "__main__":
    sys.exit(main())
