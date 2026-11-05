#!/usr/bin/env python3
"""Per-sensor measurement-error model (bias and spread in E, N, U), measured on TESTING against
Dronetag ground truth, plus a ground-truth-free rule that turns a reported height into height
above the WGS84 ellipsoid.

Inputs : data/derived/det_gt_match.csv.gz (gt_associate.py), detections.csv.gz (Q.clean),
         groundtruth.csv.gz, sensor_calibration.csv (applied azimuth offsets), sensor_sites.csv.
Outputs: data/derived/sensor_noise.csv, sensor_noise_by_iteration.csv, sensor_noise.md,
         sensor_noise_loio.csv (leave-one-iteration-out refits, schema below).

Rules (read before using the numbers)
* Rows: det_gt_match kind == 'pos', matched == 1 (horizontal distance to the best flight at the
  calibrated time <= 50 m), ambiguous == 0, gt_airborne == 1 (GT height > 5 m). Only testing rows;
  nothing is fitted on the evaluation tree. Evaluation rows only pass through the same inference
  code (`u_ellipsoid`) and are printed as a coverage check.
* Residual = detection - GT in the iteration ENU frame (C.iteration_frame, e/n/u already in
  detections.csv.gz), GT interpolated at t_cal with Q.interp_flight. GT `u` is checked to be
  geo_alt based (height above the WGS84 ellipsoid) up to the < 2 m earth-curvature term.
* The horizontal gate TRUNCATES the residuals: a detection further than 50 m from every drone is
  not a row here, so sigma_e / sigma_n / sigma_h / p90_h are lower bounds for sensors whose true
  error is of the order of the gate (radars). Statistics are robust (median, 1.4826 * MAD).
  `sigma_h_100m_m` / `n_100m` repeat sigma_h with a 100 m gate as a sensitivity check.
  The gate is horizontal only: the vertical residual is never truncated.
* Pooled bias and sigma are over all rows of a sensor (rows, not iterations, are the unit; sigma
  is about the pooled median so it includes between-iteration bias scatter, which is what an R
  matrix for an unseen iteration needs). `sigma_u_m` is the spread of the vertical residual AFTER
  `u_ellipsoid` (the height a fusion would actually use); `sigma_u_raw_m` is before.
* stable = every iteration with >= MIN_ITER_ROWS rows has |bias_e|,|bias_n| within TOL_H of the
  pooled value (needs >= 2 such iterations). applies_to is 'all' only when stable; the vertical
  analogue is u_stable / u_applies_to (after-rule bias, tolerance TOL_U).

Height-to-ellipsoid rule (`u_ellipsoid`, no ground truth at inference)
    u_ell = alt_c + ref_shift + offset(sensor)
    ref_shift = H_site(iteration, sensor) - sensor_alt   only for sensors whose reference switches
                between "height above ground" (sensor_alt < AGL_MAX_M) and the ellipsoid (NYTF in
                testing, flagged u_agl_switch); else 0. The shift equals "dz_sensor + H_site" minus
                the sensor's own ellipsoid height.
    H_site = median sensor_alt of the ellipsoid anchors (E datum, 50..120 m, i.e. radars / RF
             receivers that report their ellipsoid height) at the same physical site in that
             iteration (sensor_sites.csv), else of the whole iteration, else AREA_H0_M.
    offset(sensor) is fitted on testing as minus the median of per-iteration median residuals and
    adopted only if >= 2 iterations support it AND leave-one-iteration-out (LOIO) shows it lowers
    the mean |held-out median bias|; otherwise 0. Sensors whose reported altitude does not follow
    the drone (regression slope vs GT < 0.3) are flagged u_informative = 0: offset 0, u_ellipsoid
returns NaN for them and their sigma_u_m is NaN (STUG in testing).
`u_ellipsoid` needs the whole iteration (the H_site anchors are other sensors' rows); with a subset it
warns when an AGL-reference row falls back to AREA_H0_M (self-check 8 measures the effect).
bias_u_m (both CSVs) is the height bias of the reported altitude after ref_shift and before the
offset (bias_u_noshift_m: without ref_shift, i.e. the altitude exactly as reported; bias_u_after_m: after
the offset). sigma_u_m is the spread AFTER the rule in both CSVs (sigma_u_raw_m: before the offset).
Sensors with fewer than MIN_FOR_SIGMA rows get NaN sigma / p50 / p90 (a handful of rows gives an
unreliable and anisotropic sigma); sensor_tracks.py then uses its default. u_applies_to additionally
needs the leave-one-iteration-out held-out |bias| < TOL_U. Relay sensors (gt_feed) always get
applies_to 'testing'.
`u_ellipsoid` returns height above the ellipsoid (alt based). The ENU `u` of the iteration frame
(h0 = 0) differs from it by the earth-curvature term u_c - alt_c (-2 .. 0 m within 3.5 km).

Leave-one-iteration-out table (sensor_noise_loio.csv; read by sensor_tracks.py corrections
'height_loio' / 'var_loio'; sensor_noise.csv and sensor_noise_by_iteration.csv do not change)
    One row per (sensor_key, held_out): held_out = k for every TESTING iteration k in which the
    sensor has base rows (a sensor with no base rows in k is not affected by k: its '*' row is
    already free of k), plus held_out = '*' (the full pool, equal to sensor_noise.csv).
    sensor_key, held_out
    n, n_u             base rows (finite horizontal / height residual) used, i.e. WITHOUT k
    u_offset_m         fit_offsets on the per-iteration medians of the other iterations (0 when fewer
                       than 2 supporting iterations remain or LOIO does not adopt it; 0 for sensors
                       whose altitude does not follow the drone without k, u_informative = 0)
    u_adopted          1 if that offset was adopted
    u_support          supporting iterations of that fit (>= MIN_ITER_ROWS height rows)
    u_informative      informativeness flag (slope of reported alt vs GT geo_alt >= SLOPE_MIN; -1 = cannot
                       tell, < 20 rows) recomputed on the base rows WITHOUT k (k rows) / of the full fit
                       ('*' rows, = sensor_noise.csv); 0 -> offset 0 and sigma_u NaN in that row
    sigma_e_m, sigma_n_m, sigma_u_m   block_stats of the base rows of the other iterations (same
                       MIN_FOR_SIGMA rule; sigma_u after the held-out offset)
    held_out_med_du_loio_m, held_out_med_du_none_m   (k rows) median |height residual| on k itself
                       with the LOIO offset / with offset 0 (after the reference shift)
    u_eval_apply       ('*' rows) PRE-REGISTERED decision for evaluation: 1 iff the full rule is
                       adopted AND the LOIO offset gives a lower median |du| on the held-out iteration
                       than offset 0 on >= ceil(EVAL_SHARE x n_support) of the n_support supporting
                       iterations AND the pooled median |du| over the held-out samples of those
                       iterations is lower with the LOIO offsets than with 0; else 0 (evaluation then
                       uses offset 0). Consequence (spec-literal): a sensor with n_support = 2 always
                       gets 0, since each fold has 1 supporting iteration left and fit_offsets then
                       returns 0 (ILYS_PASSIVE-RF, STUG_CYBER in testing).
    n_better, n_support, pooled_med_du_loio_m, pooled_med_du_none_m   ('*' rows) the inputs of that rule
    Floats are rounded to 3 decimals as in sensor_noise.csv. Self-checks 9 ('*' rows == sensor_noise.csv)
    and 10 (independent plain-Python recomputation of every held-out row and of u_eval_apply, with a
    mutation test) guard it.

Seeded, deterministic, < 1 minute. Prints numbered self-checks; exits non-zero if one fails.
"""

from __future__ import annotations

import csv
import gzip
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuas_clean as Q  # noqa: E402
import cuas_common as C  # noqa: E402

OUT_CSV = C.DERIVED / "sensor_noise.csv"
OUT_ITER = C.DERIVED / "sensor_noise_by_iteration.csv"
OUT_MD = C.DERIVED / "sensor_noise.md"
OUT_LOIO = C.DERIVED / "sensor_noise_loio.csv"
MATCH = C.DERIVED / "det_gt_match.csv.gz"
CALIB = C.DERIVED / "sensor_calibration.csv"
SITES = C.DERIVED / "sensor_sites.csv"

GATE_M = 50.0                 # det_gt_match position gate
GATE2_M = 100.0               # sensitivity gate
MIN_ITER_ROWS = 30            # iteration enters stability / offset fitting
CHECK_ROWS = 50               # check 3 applies to (sensor, iteration) with >= this many rows
TOL_H = 5.0                   # m, per-iteration horizontal bias vs pooled
TOL_U = 5.0                   # m, vertical bias after rule
AGL_MAX_M = 10.0              # sensor_alt in (0, AGL_MAX_M): reference is height above ground
ANCHOR_RANGE = (50.0, 120.0)  # ellipsoid height (m) of ground-level sensors in this area (testing: 67..79)
AREA_H0_M = 76.0              # median ellipsoid height of the anchors over testing (checked in self-check 5)
SLOPE_MIN = 0.3               # regression slope of reported alt vs GT geo_alt below this: not informative
AZ_SUSPECT_DEG = 2.0          # |median azimuth residual| after the applied offset above this: offset suspect
MIN_FOR_SIGMA = 30            # fewer rows: sigma / p50 / p90 are NaN (sensor_tracks then uses its default)
EVAL_SHARE = 2.0 / 3.0        # u_eval_apply: the LOIO offset must win on >= ceil(EVAL_SHARE x n_support) folds
SEED = 20261028

EXPLAIN = {
    "relay": "relay of the Dronetag feed: the vertical offset depends on the drone model (it reproduces that "
             "drone's own altitude source), so no sensor-level rule can be learned",
    "uninformative": "reported altitude does not follow the drone (slope vs GT < 0.3)",
}


# ----------------------------------------------------------------------------- helpers
def robust_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def med(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def fmt(v, nd=1) -> str:
    return "-" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def md_table(header, rows) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def short_iter(ik: str) -> str:
    return ik.split("/", 1)[1] if "/" in ik else ik


def check(name: str, ok: bool, msg: str, fails: list) -> None:
    print(f"self-check {name}: {'OK' if ok else 'FAILED'}  {msg}")
    if not ok:
        fails.append(name)


# ----------------------------------------------------------------------------- inference-side rule
_SITE_MAP: dict | None = None
_RULES: dict | None = None


def _site_map() -> dict[tuple[str, str], str]:
    """{(iter_key, sensor_key): site_id} from sensor_sites.csv."""
    global _SITE_MAP
    if _SITE_MAP is None:
        m = {}
        with SITES.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                for sk in r["sensor_keys"].split("|"):
                    m[(r["iter_key"], sk)] = r["site_id"]
        _SITE_MAP = m
    return _SITE_MAP


def load_rules(path: Path = OUT_CSV) -> dict[str, dict]:
    """{sensor_key: dict(offset, datum, agl_switch, informative)} from sensor_noise.csv."""
    global _RULES
    if _RULES is None or path != OUT_CSV:
        rules = {}
        with path.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rules[r["sensor_key"]] = dict(
                    offset=float(r["u_offset_m"] or 0.0), datum=r["u_datum"], agl_switch=r["u_agl_switch"] == "1",
                    informative=r["u_informative"] != "0")
        if path == OUT_CSV:
            _RULES = rules
        return rules
    return _RULES


def anchor_mask(det: dict) -> np.ndarray:
    sa = np.asarray(det["sensor_alt"], float)
    with np.errstate(invalid="ignore"):
        return (np.asarray(det["sensor_pos_valid"]) == 1) & (np.asarray(det["sensor_alt_datum"]) == "E") \
            & (sa >= ANCHOR_RANGE[0]) & (sa <= ANCHOR_RANGE[1])


def site_heights(det: dict) -> dict[tuple[str, str], float]:
    """{(iter_key, sensor_key): H_site}: ellipsoid height of the sensor's physical site, from the
    anchors co-located with it (same site in sensor_sites.csv), else the iteration's anchors, else
    AREA_H0_M. Computed from the rows of `det` itself (pass the whole iteration)."""
    return _site_heights_src(det)[0]


def _site_heights_src(det: dict):
    """(heights, source) with source[(iter, sensor)] in {'site', 'iteration', 'area'}."""
    ik, sk, sa = np.asarray(det["iter_key"]), np.asarray(det["sensor_key"]), np.asarray(det["sensor_alt"], float)
    am = anchor_mask(det)
    per = defaultdict(list)                         # (iter, anchor sensor) -> median sensor_alt
    for key in {(a, b) for a, b in zip(ik[am].tolist(), sk[am].tolist())}:
        m = am & (ik == key[0]) & (sk == key[1])
        per[key[0]].append((key[1], float(np.median(sa[m]))))
    sites = _site_map()
    out, src = {}, {}
    for key in {(a, b) for a, b in zip(ik.tolist(), sk.tolist())}:
        anchors = per.get(key[0], [])
        site = sites.get(key)
        near = [h for s, h in anchors if site is not None and sites.get((key[0], s)) == site and s != key[1]]
        if near:
            out[key], src[key] = float(np.median(near)), "site"
        elif anchors:
            out[key] = float(np.median([h for s, h in anchors if s != key[1]] or [h for _, h in anchors]))
            src[key] = "iteration"
        else:
            out[key], src[key] = AREA_H0_M, "area"
    return out, src


def ref_shift(det: dict, heights: dict | None = None, rules: dict | None = None) -> np.ndarray:
    """Per-row H_site - sensor_alt for sensors flagged u_agl_switch whose sensor_alt is a height above
    ground (0 < sensor_alt < AGL_MAX_M); 0 elsewhere."""
    rules = load_rules() if rules is None else rules
    ik, sk = np.asarray(det["iter_key"]), np.asarray(det["sensor_key"])
    sa = np.asarray(det["sensor_alt"], float)
    shift = np.zeros(len(sa))
    flagged = [s for s, r in rules.items() if r["agl_switch"]]
    if not flagged:
        return shift
    src = {}
    if heights is None:
        heights, src = _site_heights_src(det)
    with np.errstate(invalid="ignore"):
        agl = np.isin(sk, flagged) & (np.asarray(det["sensor_pos_valid"]) == 1) & (sa > 0) & (sa < AGL_MAX_M)
    fell_back = sorted(f"{a}/{b}" for a, b in {(a, b) for a, b in zip(ik[agl].tolist(), sk[agl].tolist())}
                       if src.get((a, b)) == "area")
    if fell_back:
        warnings.warn(f"u_ellipsoid: no ellipsoid anchors in the rows passed for {', '.join(fell_back)}; H_site falls "
                      f"back to AREA_H0_M = {AREA_H0_M:g} m (pass the whole iteration, or heights=site_heights(full))",
                      RuntimeWarning, stacklevel=3)
    for key in {(a, b) for a, b in zip(ik[agl].tolist(), sk[agl].tolist())}:
        m = agl & (ik == key[0]) & (sk == key[1])
        shift[m] = heights.get(key, AREA_H0_M) - sa[m]
    return shift


def u_ellipsoid(det: dict, rules: dict | None = None, heights: dict | None = None) -> np.ndarray:
    """Height above the WGS84 ellipsoid (m) of every row of `det` (a Q.clean()-ed dict with iter_key,
    sensor_key, alt_c, alt_datum, sensor_alt, sensor_alt_datum, sensor_pos_valid); NaN where alt is
    unknown (alt_c NaN). No ground truth is used. Rule: alt_c + ref_shift + offset(sensor); sensors
    without a fitted rule get offset 0 (see module docstring). `heights` = site_heights(det)."""
    rules = load_rules() if rules is None else rules
    alt = np.asarray(det["alt_c"], float)
    sk = np.asarray(det["sensor_key"])
    off = np.zeros(len(alt))
    for s in np.unique(sk):
        r = rules.get(str(s))
        if r is not None:
            m = sk == s
            dat = np.asarray(det["alt_datum"])[m]
            # offset valid for the datum it was fitted on ('' = unknown, treated as the fitted one)
            off[m] = np.where((dat == r["datum"]) | (dat == ""), r["offset"], 0.0)
    out = alt + ref_shift(det, heights, rules) + off
    for s in np.unique(sk):                      # altitude that does not follow the drone: no usable height
        r = rules.get(str(s))
        if r is not None and not r["informative"]:
            out[sk == s] = np.nan
    return out


# ----------------------------------------------------------------------------- data
def read_match_stream() -> Counter:
    """Independent recount straight from the gzip text (no numpy): rows kind=pos, matched=1,
    ambiguous=0, gt_airborne=1 per sensor_key."""
    cnt: Counter = Counter()
    with gzip.open(MATCH, "rt", newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            if r["kind"] == "pos" and r["matched"] == "1" and r["ambiguous"] == "0" and r["gt_airborne"] == "1":
                cnt[r["sensor_key"]] += 1
    return cnt


def load_joined():
    """det_gt_match joined to cleaned detections (testing rows), GT interpolated at t_cal."""
    t0 = time.time()
    cols = ["iter_key", "file", "row", "sensor_key", "meas_kind", "e", "n", "u", "alt", "alt_datum", "sensor_alt",
            "sensor_alt_datum", "lat", "lon", "sensor_pos_valid", "sensor_e", "sensor_n", "az_deg"]
    D = Q.load_clean(cols)
    M = C.read_columns(MATCH, {"t", "t_cal", "resid", "second_resid", "matched", "ambiguous", "gt_feed", "gt_height",
                               "gt_airborne", "row"})
    idx = {(a, b, int(c)): i for i, (a, b, c) in enumerate(zip(D["iter_key"], D["file"], D["row"]))}
    j = np.array([idx[(a, b, int(c))] for a, b, c in zip(M["iter_key"], M["file"], M["row"])], dtype=np.int64)
    assert len(set(j.tolist())) == len(j)
    # (iter, sensor, row) of the joined table must agree with det_gt_match
    assert (D["sensor_key"][j] == M["sensor_key"]).all()
    J = {k: v[j] for k, v in D.items()}
    gt = Q.load_groundtruth()
    N = len(j)
    G = {c: np.full(N, np.nan) for c in ("e", "n", "u", "geo_alt", "height")}
    bf = M["best_flight"]
    for ik in np.unique(M["iter_key"]):
        mi = M["iter_key"] == ik
        for fk in np.unique(bf[mi]):
            if fk == "":
                continue
            sel = mi & (bf == fk) & np.isfinite(M["t_cal"])
            fm = (gt["iter_key"] == ik) & (gt["flight_key"] == fk)
            if not fm.any() or not sel.any():
                continue
            vals = Q.interp_flight(gt, fm, M["t_cal"][sel], cols=tuple(G))
            for c, v in zip(G, vals):
                G[c][sel] = v
    print(f"loaded {N:,} joined rows ({time.time() - t0:.1f} s)")
    return D, J, M, G, gt, j


# ----------------------------------------------------------------------------- statistics
def block_stats(re, rn, ru_raw, ru_after, ru_noshift=None) -> dict:
    """Robust stats of one group of residual arrays. `ru_raw` = height residual after the reference shift,
    before the offset (named raw for the offset fit); `ru_noshift` = the altitude exactly as reported."""
    n = int(np.isfinite(re).sum())
    nu = int(np.isfinite(ru_raw).sum())
    h = np.hypot(re, rn)
    be, bn, bu, bua = med(re), med(rn), med(ru_raw), med(ru_after)
    ok = n >= MIN_FOR_SIGMA
    oku = nu >= MIN_FOR_SIGMA
    se, sn = (robust_sigma(re), robust_sigma(rn)) if ok else (float("nan"),) * 2
    return dict(n=n, n_u=nu, bias_e=be, bias_n=bn, bias_u=bu, bias_u_after=bua, sigma_e=se, sigma_n=sn,
                bias_u_noshift=med(ru_noshift) if ru_noshift is not None else float("nan"),
                sigma_u=robust_sigma(ru_after) if oku else float("nan"),
                sigma_u_raw=robust_sigma(ru_raw) if oku else float("nan"),
                sigma_h=float(np.hypot(se, sn)) if ok else float("nan"),
                p50_h=float(np.percentile(h[np.isfinite(h)], 50)) if ok else float("nan"),
                p90_h=float(np.percentile(h[np.isfinite(h)], 90)) if ok else float("nan"))


def informative_flag(geo_alt: np.ndarray, alt_c: np.ndarray) -> tuple[float, int]:
    """(slope, informative) of reported alt vs GT geo_alt on rows where both are finite: informative = 1 if the
    slope is >= SLOPE_MIN or cannot be fitted (< 20 rows or GT spread <= 1 m), 0 below, -1 with < 20 rows."""
    ok = np.isfinite(geo_alt) & np.isfinite(alt_c)
    x, y = geo_alt[ok], alt_c[ok]
    if ok.sum() >= 20 and np.std(x) > 1.0:
        slope = float(np.polyfit(x, y, 1)[0])
    else:
        slope = float("nan")
    informative = 1 if (not np.isfinite(slope) or slope >= SLOPE_MIN) else 0
    if ok.sum() < 20:
        informative = -1                           # cannot tell
    return slope, informative


def fit_offsets(raw_by_iter: dict[str, tuple[float, int, int]]):
    """raw_by_iter: iter -> (median raw residual, n_u, n_flights). Returns (offset, support, loio_before,
    loio_after, adopted, why). offset = -median of the supporting iteration medians."""
    sup = {i: v[0] for i, v in raw_by_iter.items() if v[1] >= MIN_ITER_ROWS and np.isfinite(v[0])}
    if len(sup) < 2:
        return 0.0, len(sup), float("nan"), float("nan"), False, f"{len(sup)} iteration(s) with >= {MIN_ITER_ROWS} height rows (need 2)"
    vals = np.array(list(sup.values()))
    off_all = -float(np.median(vals))
    before = float(np.mean(np.abs(vals)))
    after = []
    for i, v in sup.items():
        others = np.array([w for k, w in sup.items() if k != i])
        after.append(abs(v - float(np.median(others))))
    after = float(np.mean(after))
    if after < before - 1.0:
        return off_all, len(sup), before, after, True, "LOIO mean |held-out bias| lower than without offset"
    return 0.0, len(sup), before, after, False, "LOIO does not improve on offset 0"


def az_offsets_applied() -> dict[tuple[str, str], float]:
    out = {}
    with CALIB.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["param"] == "az_offset_deg" and r["status"] == "applied" and r["iter_key"] != "*":
                out[(r["sensor_key"], r["iter_key"])] = float(r["value"])
    return out


# ----------------------------------------------------------------------------- leave-one-iteration-out
LOIO_ORDER = ["sensor_key", "held_out", "n", "n_u", "u_offset_m", "u_adopted", "u_support", "u_informative", "sigma_e_m",
              "sigma_n_m", "sigma_u_m", "held_out_med_du_loio_m", "held_out_med_du_none_m", "u_eval_apply",
              "n_better", "n_support", "pooled_med_du_loio_m", "pooled_med_du_none_m"]
LOIO_FLOATS = ("u_offset_m", "sigma_e_m", "sigma_n_m", "sigma_u_m", "held_out_med_du_loio_m",
               "held_out_med_du_none_m", "pooled_med_du_loio_m", "pooled_med_du_none_m")


def loio_table(sensors, rule, raw_iter, its_by_sensor, base, sk, ik, re, rn, ru_ref, ru_raw, full_stats,
               geo_alt, alt_c) -> list[dict]:
    """Rows of sensor_noise_loio.csv (unrounded): per sensor one row per held-out testing iteration it has base
    rows in (offset, informativeness flag and sigmas all refitted without it), plus the '*' row (full pool,
    `full_stats[s]` = the block_stats of sensor_noise.csv) with the pre-registered evaluation decision
    u_eval_apply (module docstring)."""
    nan = float("nan")
    out = []
    for s in sensors:
        inf0 = rule[s]["informative"] == 0
        ms = base & (sk == s)
        k_rows, off_k = [], {}
        for k in its_by_sensor[s]:
            if not str(k).startswith("testing"):
                continue
            off, support, _, _, adopted, _ = fit_offsets({i: v for i, v in raw_iter[s].items() if i != k})
            m = ms & (ik != k)
            _, inf_k = informative_flag(geo_alt[m], alt_c[m])
            if inf_k == 0:
                off, adopted = 0.0, False
            off_k[k] = off
            ra = ru_ref[m] + off
            ra[~np.isfinite(ru_raw[m])] = np.nan
            if inf_k == 0:
                ra[:] = np.nan
            st = block_stats(re[m], rn[m], ru_ref[m], ra, ru_raw[m])
            mk = ms & (ik == k) & np.isfinite(ru_raw)
            k_rows.append(dict(sensor_key=s, held_out=k, n=st["n"], n_u=st["n_u"], u_offset_m=off, u_adopted=int(adopted),
                               u_support=support, u_informative=inf_k, sigma_e_m=st["sigma_e"], sigma_n_m=st["sigma_n"],
                               sigma_u_m=st["sigma_u"],
                               held_out_med_du_loio_m=med(np.abs(ru_ref[mk] + off)),
                               held_out_med_du_none_m=med(np.abs(ru_ref[mk])),
                               u_eval_apply="", n_better="", n_support="", pooled_med_du_loio_m=nan,
                               pooled_med_du_none_m=nan))
        sup = [i for i, v in raw_iter[s].items() if v[1] >= MIN_ITER_ROWS and np.isfinite(v[0]) and i in off_k]
        n_better, d_loio, d_none = 0, [], []
        for k in sup:
            mk = ms & (ik == k) & np.isfinite(ru_raw)
            a, b = np.abs(ru_ref[mk] + off_k[k]), np.abs(ru_ref[mk])
            n_better += med(a) < med(b)
            d_loio.append(a)
            d_none.append(b)
        pl = med(np.concatenate(d_loio)) if d_loio else nan
        pn = med(np.concatenate(d_none)) if d_none else nan
        need = int(np.ceil(EVAL_SHARE * len(sup) - 1e-9))
        apply = int(bool(rule[s]["adopted"]) and not inf0 and len(sup) >= 2 and n_better >= need and pl < pn)
        st = full_stats[s]
        out.append(dict(sensor_key=s, held_out="*", n=st["n"], n_u=st["n_u"], u_offset_m=rule[s]["offset"],
                        u_adopted=int(bool(rule[s]["adopted"])), u_support=rule[s]["support"],
                        u_informative=rule[s]["informative"], sigma_e_m=st["sigma_e"],
                        sigma_n_m=st["sigma_n"], sigma_u_m=st["sigma_u"], held_out_med_du_loio_m=nan,
                        held_out_med_du_none_m=nan, u_eval_apply=apply, n_better=n_better, n_support=len(sup),
                        pooled_med_du_loio_m=pl, pooled_med_du_none_m=pn))
        out += k_rows
    return out


def write_loio(rows: list[dict]) -> None:
    cols = {k: [r[k] for r in rows] for k in LOIO_ORDER}
    for k in LOIO_FLOATS:
        cols[k] = np.round(np.array(cols[k], dtype=float), 3)
    for k in ("n", "n_u", "u_adopted", "u_support", "u_informative", "u_eval_apply", "n_better", "n_support"):
        cols[k] = [str(int(v)) if v != "" else "" for v in cols[k]]
    C.write_columns(OUT_LOIO, cols, LOIO_ORDER)


def _pmed(xs: list) -> float:
    """Plain-Python median (no numpy) of a non-empty list."""
    v = sorted(xs)
    h = len(v) // 2
    return float(v[h]) if len(v) % 2 else (v[h - 1] + v[h]) / 2.0


def _prsig(xs: list) -> float:
    m = _pmed(xs)
    return 1.4826 * _pmed([abs(x - m) for x in xs])


def loio_plain(rows_by_sensor: dict, informative: dict, adopted_full: dict, exclude_k: bool = True) -> dict:
    """Independent recomputation of sensor_noise_loio.csv with plain Python loops over the base rows
    (rows_by_sensor[s] = list of (iter_key, re, rn, ru_ref, ru_raw, geo_alt, alt_c) floats, NaN = missing).
    Re-implements the offset fit (supporting iterations, -median, LOIO adoption), the informativeness slope
    (least squares), the robust sigmas and the u_eval_apply rule without calling fit_offsets / block_stats /
    informative_flag. exclude_k=False is the mutation: held-out rows computed WITH k.
    Returns {(s, held_out): dict}."""
    def fin(x):
        return x == x

    def fit(per_iter: dict) -> tuple[float, int, int]:
        sup = {i: _pmed(v) for i, v in per_iter.items() if len(v) >= MIN_ITER_ROWS}
        if len(sup) < 2:
            return 0.0, 0, len(sup)
        vals = list(sup.values())
        off = -_pmed(vals)
        before = sum(abs(v) for v in vals) / len(vals)
        after = sum(abs(v - _pmed([w for j, w in sup.items() if j != i])) for i, v in sup.items()) / len(sup)
        return (off, 1, len(sup)) if after < before - 1.0 else (0.0, 0, len(sup))

    def inform(use: list) -> int:
        xy = [(r[5], r[6]) for r in use if fin(r[5]) and fin(r[6])]
        if len(xy) < 20:
            return -1
        mx = sum(x for x, _ in xy) / len(xy)
        my = sum(y for _, y in xy) / len(xy)
        sxx = sum((x - mx) ** 2 for x, _ in xy)
        if (sxx / len(xy)) ** 0.5 <= 1.0:
            return 1                                 # slope cannot be fitted: treated as informative
        slope = sum((x - mx) * (y - my) for x, y in xy) / sxx
        return 1 if slope >= SLOPE_MIN else 0

    out = {}
    for s, rows in rows_by_sensor.items():
        its = sorted({r[0] for r in rows if r[0].startswith("testing")})
        off_k = {}
        for k in its:
            use = [r for r in rows if r[0] != k] if exclude_k else list(rows)
            per = defaultdict(list)
            for r in use:
                if fin(r[3]):
                    per[r[0]].append(r[3])
            off, ad, sup = fit(per)
            inf_k = inform(use)
            if inf_k == 0:
                off, ad = 0.0, 0
            off_k[k] = off
            e_ = [r[1] for r in use if fin(r[1])]
            n_ = [r[2] for r in use if fin(r[2])]
            u_ = [r[3] + off for r in use if fin(r[4])]
            n_u = sum(1 for r in use if fin(r[3]))
            ok, oku = len(e_) >= MIN_FOR_SIGMA, n_u >= MIN_FOR_SIGMA and inf_k != 0
            out[(s, k)] = dict(n=len(e_), n_u=n_u, u_offset_m=off, u_adopted=ad, u_support=sup, u_informative=inf_k,
                               sigma_e_m=_prsig(e_) if ok else float("nan"), sigma_n_m=_prsig(n_) if ok else float("nan"),
                               sigma_u_m=_prsig(u_) if oku and u_ else float("nan"))
        per_all = defaultdict(list)
        for r in rows:
            if fin(r[3]):
                per_all[r[0]].append(r[3])
        sup = [i for i in sorted(per_all) if len(per_all[i]) >= MIN_ITER_ROWS and i in off_k]
        nb, al, an = 0, [], []
        for k in sup:
            a = [abs(x + off_k[k]) for x in per_all[k]]
            b = [abs(x) for x in per_all[k]]
            nb += _pmed(a) < _pmed(b)
            al += a
            an += b
        need = -(-2 * len(sup) // 3)                 # ceil(2/3 n) in integers
        pl, pn = (_pmed(al), _pmed(an)) if al else (float("nan"), float("nan"))
        out[(s, "*")] = dict(u_eval_apply=int(adopted_full[s] and informative[s] != 0 and len(sup) >= 2 and nb >= need
                                              and pl < pn), n_better=nb, n_support=len(sup),
                             pooled_med_du_loio_m=pl, pooled_med_du_none_m=pn)
    return out


# ----------------------------------------------------------------------------- main
def main() -> int:
    t_start = time.time()
    fails: list[str] = []
    D, J, M, G, gt, j = load_joined()
    N = len(J["e"])
    sk, ik = J["sensor_key"], M["iter_key"]
    pos = M["kind"] == "pos"
    base = pos & (M["matched"] == 1) & (M["ambiguous"] == 0) & (M["gt_airborne"] == 1)

    # ---- GT u is geo_alt based
    du = G["u"][np.isfinite(G["u"])] - G["geo_alt"][np.isfinite(G["u"])]
    gt_curv = (float(du.min()), float(np.median(du)), float(du.max()))

    # ---- residuals (raw frame) and rule
    re = J["e"] - G["e"]
    rn = J["n"] - G["n"]
    cur = J["u"] - J["alt_c"]                      # earth-curvature term of the frame (u_c - alt_c at alt != 0)
    cur = np.where(np.isfinite(cur), cur, J["u"] - J["alt"])
    ru_raw = J["u_c"] - G["u"]

    # rule inputs: ref shift needs the iteration's anchors, so site heights come from the FULL detection table
    heights = site_heights(D)
    rules_stub = {}
    # AGL-switch sensors: a sensor with sensor_alt in (0, AGL_MAX_M) in some iterations and an anchor-range
    # value in others (testing)
    sa_all = D["sensor_alt"]
    agl_switch = set()
    for s in np.unique(D["sensor_key"][np.isin(D["meas_kind"], Q.POSITION_KINDS)]):
        ms = (D["sensor_key"] == s) & (D["sensor_pos_valid"] == 1) & np.char.startswith(D["iter_key"].astype(str), "testing")
        v = sa_all[ms]
        if ((v > 0) & (v < AGL_MAX_M)).any() and ((v >= ANCHOR_RANGE[0]) & (v <= ANCHOR_RANGE[1])).any():
            agl_switch.add(str(s))
    for s in agl_switch:
        rules_stub[s] = dict(offset=0.0, datum="E", agl_switch=True, informative=True)
    shift_all = ref_shift(D, heights, rules_stub)
    shift = shift_all[j]
    ru_ref = ru_raw + shift                        # height after the reference correction, before the offset

    # ---- per (sensor, iteration) and per sensor stats
    sensors = sorted(np.unique(sk[base]).tolist())
    gtf = {s: int(np.round(np.nanmax(M["gt_feed"][base & (sk == s)]))) for s in sensors}
    its_by_sensor = {s: sorted(np.unique(ik[base & (sk == s)]).tolist()) for s in sensors}
    raw_iter = {s: {} for s in sensors}
    for s in sensors:
        for i in its_by_sensor[s]:
            m = base & (sk == s) & (ik == i)
            raw_iter[s][i] = (med(ru_ref[m]), int(np.isfinite(ru_ref[m]).sum()), len(np.unique(M["best_flight"][m])))
    rule = {}
    for s in sensors:
        off, support, lb, la, adopted, why = fit_offsets(raw_iter[s])
        m = base & (sk == s)
        dat = Counter(J["alt_datum"][m & np.isfinite(ru_raw)].tolist())
        datum = dat.most_common(1)[0][0] if dat else ""
        # informativeness: slope of reported alt vs GT geo_alt over all rows with a height
        slope, informative = informative_flag(G["geo_alt"][m], J["alt_c"][m])
        if informative == 0:
            off, adopted, why = 0.0, False, "altitude does not follow the drone: u_ellipsoid returns NaN"
        rule[s] = dict(offset=off, support=support, loio_before=lb, loio_after=la, adopted=adopted, why=why,
                       datum=datum, slope=slope, informative=informative, agl=int(s in agl_switch))
    off_row = np.zeros(N)
    for s in sensors:
        off_row[sk == s] = rule[s]["offset"]
    ru_after = ru_ref + off_row
    ru_after[~np.isfinite(ru_raw)] = np.nan
    for s in sensors:                              # no usable height: nothing "after"
        if rule[s]["informative"] == 0:
            ru_after[sk == s] = np.nan

    # ---- rule exercised through the public inference function (checks it reproduces the fitted residuals)
    rules_pub = {s: dict(offset=rule[s]["offset"], datum=rule[s]["datum"], agl_switch=bool(rule[s]["agl"]),
                         informative=rule[s]["informative"] != 0) for s in sensors}
    Jr = {k: J[k] for k in ("iter_key", "sensor_key", "alt_c", "alt_datum", "sensor_alt", "sensor_alt_datum",
                           "sensor_pos_valid")}
    u_pub = u_ellipsoid(Jr, rules_pub, heights)
    ru_pub = (u_pub + cur) - G["u"]
    same = np.isfinite(ru_after[base]) == np.isfinite(ru_pub[base])
    diff_pub = float(np.nanmax(np.abs(ru_pub[base] - ru_after[base]))) if same.all() else float("inf")

    iter_rows, sens_rows = [], []
    iter_stats: dict[tuple[str, str], dict] = {}
    for s in sensors:
        for i in its_by_sensor[s]:
            m = base & (sk == s) & (ik == i)
            st = block_stats(re[m], rn[m], ru_ref[m], ru_after[m], ru_raw[m])
            st["n_flights"] = len(np.unique(M["best_flight"][m]))
            st["n_flights_u"] = len(np.unique(M["best_flight"][m & np.isfinite(ru_raw)]))
            st["sensor_alt"] = med(J["sensor_alt"][m])
            iter_stats[(s, i)] = st
            iter_rows.append((s, i, st))

    # 100 m-gate sensitivity: resid <= 100, second flight further than 100 m, airborne
    base100 = pos & (M["resid"] <= GATE2_M) & ~(M["second_resid"] <= GATE2_M) & (M["gt_airborne"] == 1) & np.isfinite(re)
    cand = pos & (M["gt_airborne"] == 1) & np.isfinite(M["resid"])

    full_stats = {}
    for s in sensors:
        m = base & (sk == s)
        st = block_stats(re[m], rn[m], ru_ref[m], ru_after[m], ru_raw[m])
        full_stats[s] = st
        ev = [i for i in its_by_sensor[s] if iter_stats[(s, i)]["n"] >= MIN_ITER_ROWS]
        if len(ev) >= 2:
            stable = int(all(abs(iter_stats[(s, i)]["bias_e"] - st["bias_e"]) <= TOL_H
                             and abs(iter_stats[(s, i)]["bias_n"] - st["bias_n"]) <= TOL_H for i in ev))
        else:
            stable = float("nan")
        evu = [i for i in its_by_sensor[s] if iter_stats[(s, i)]["n_u"] >= MIN_ITER_ROWS]
        if len(evu) >= 2:
            u_stable = int(all(abs(iter_stats[(s, i)]["bias_u_after"] - st["bias_u_after"]) <= TOL_U for i in evu))
        else:
            u_stable = float("nan")
        m100 = base100 & (sk == s)
        sg100 = robust_sigma(re[m100]), robust_sigma(rn[m100])
        r = rule[s]
        applies = "all" if (stable == 1 and not gtf[s]) else "testing"
        loio_eff = r["loio_after"] if r["adopted"] else r["loio_before"]      # honest held-out |bias| of the rule as applied
        u_applies = "all" if (u_stable == 1 and r["informative"] != 0 and not gtf[s]
                              and np.isfinite(loio_eff) and loio_eff < TOL_U) else "testing"
        note = []
        if gtf[s]:
            note.append("relays the Dronetag feed (leakage vs GT; sub-metre agreement expected, not a real error model)")
        if st["n"] < MIN_FOR_SIGMA:
            note.append(f"n < {MIN_FOR_SIGMA}: sigma not estimated")
        if len(its_by_sensor[s]) == 1:
            note.append("single iteration")
        if r["informative"] == 0:
            note.append(f"reported altitude does not follow the drone (slope {r['slope']:.2f})")
        if st["p90_h"] >= 0.8 * GATE_M:
            note.append("p90_h near the 50 m gate: sigma_h is a lower bound")
        if u_stable == 1 and u_applies == "testing" and r["informative"] != 0 and not gtf[s]:
            note.append(f"vertical rule fits in-sample but LOIO held-out abs bias {loio_eff:.1f} m >= {TOL_U:g} m: u_applies_to testing")
        if r["agl"]:
            note.append("height reference switches between above-ground and ellipsoid across iterations")
        rd = f"{r['why']}" if r["support"] >= 2 else r["why"]
        if r["agl"]:
            u_rule = (f"u_ell = alt + (H_site - sensor_alt if 0 < sensor_alt < {AGL_MAX_M:g} else 0) "
                      f"{r['offset']:+.1f}")
        else:
            u_rule = f"u_ell = alt {r['offset']:+.1f}"
        u_rule += f" [datum {r['datum'] or '?'}; {'fitted' if r['adopted'] else 'offset not adopted: ' + rd}]"
        sens_rows.append(dict(
            sensor_key=s, n=st["n"], n_iterations=len(its_by_sensor[s]), n_u=st["n_u"],
            bias_e_m=st["bias_e"], bias_n_m=st["bias_n"], bias_u_m=st["bias_u"], bias_u_after_m=st["bias_u_after"],
            bias_u_noshift_m=st["bias_u_noshift"], loio_held_out_u_m=loio_eff,
            sigma_e_m=st["sigma_e"], sigma_n_m=st["sigma_n"], sigma_u_m=st["sigma_u"], sigma_u_raw_m=st["sigma_u_raw"],
            sigma_h_m=st["sigma_h"], p50_h_m=st["p50_h"], p90_h_m=st["p90_h"],
            n_100m=int(np.isfinite(re[m100]).sum()), sigma_h_100m_m=float(np.hypot(*sg100)) if m100.sum() >= MIN_FOR_SIGMA else float("nan"),
            gate_frac=float(m.sum() / max(1, (cand & (sk == s)).sum())),
            u_rule=u_rule, u_offset_m=r["offset"], u_datum=r["datum"], u_agl_switch=r["agl"], u_informative=r["informative"],
            u_slope=r["slope"], u_stable=u_stable, u_applies_to=u_applies, stable=stable, applies_to=applies,
            gt_feed=gtf[s], note="; ".join(note)))
    order_sens = ["sensor_key", "n", "n_iterations", "bias_e_m", "bias_n_m", "bias_u_m", "sigma_e_m", "sigma_n_m",
                  "sigma_u_m", "sigma_h_m", "p50_h_m", "p90_h_m", "u_rule", "stable", "applies_to", "note",
                  "n_u", "bias_u_after_m", "sigma_u_raw_m", "n_100m", "sigma_h_100m_m", "gate_frac", "u_offset_m",
                  "u_datum", "u_agl_switch", "u_informative", "u_slope", "u_stable", "u_applies_to", "gt_feed",
                  "bias_u_noshift_m", "loio_held_out_u_m"]
    cols = {k: [r[k] for r in sens_rows] for k in order_sens}
    for k in order_sens:
        if k not in ("sensor_key", "u_rule", "applies_to", "note", "u_datum", "u_applies_to"):
            cols[k] = np.array(cols[k], dtype=float)
    for k in ("u_agl_switch", "u_informative", "gt_feed"):      # exact ints in the file
        cols[k] = [str(int(v)) for v in cols[k]]
    for k in ("n", "n_iterations", "n_u", "n_100m"):
        cols[k] = [str(int(v)) for v in cols[k]]
    for k in ("stable", "u_stable"):
        cols[k] = [("" if not np.isfinite(v) else str(int(v))) for v in cols[k]]
    for k in ("bias_e_m", "bias_n_m", "bias_u_m", "sigma_e_m", "sigma_n_m", "sigma_u_m", "sigma_h_m", "p50_h_m",
              "p90_h_m", "bias_u_after_m", "sigma_u_raw_m", "sigma_h_100m_m", "gate_frac", "u_offset_m", "u_slope", "bias_u_noshift_m", "loio_held_out_u_m"):
        cols[k] = np.round(cols[k], 3)
    C.write_columns(OUT_CSV, cols, order_sens)

    order_it = ["sensor_key", "iter_key", "n", "n_u", "n_flights", "bias_e_m", "bias_n_m", "bias_u_m", "sigma_e_m",
                "sigma_n_m", "sigma_u_m", "sigma_h_m", "p50_h_m", "p90_h_m", "bias_u_after_m",
                "sigma_u_raw_m", "bias_u_noshift_m", "sensor_alt_m", "u_offset_m"]
    ic = defaultdict(list)
    for s, i, st in iter_rows:
        vals = dict(sensor_key=s, iter_key=i, n=st["n"], n_u=st["n_u"], n_flights=st["n_flights"],
                    bias_e_m=st["bias_e"], bias_n_m=st["bias_n"], bias_u_m=st["bias_u"], sigma_e_m=st["sigma_e"],
                    sigma_n_m=st["sigma_n"], sigma_u_m=st["sigma_u"], sigma_h_m=st["sigma_h"], p50_h_m=st["p50_h"],
                    p90_h_m=st["p90_h"], bias_u_after_m=st["bias_u_after"], sigma_u_raw_m=st["sigma_u_raw"],
                    bias_u_noshift_m=st["bias_u_noshift"], sensor_alt_m=st["sensor_alt"], u_offset_m=rule[s]["offset"])
        for k in order_it:
            ic[k].append(vals[k])
    for k in order_it:
        if k not in ("sensor_key", "iter_key"):
            ic[k] = np.round(np.array(ic[k], dtype=float), 3)
    for k in ("n", "n_u", "n_flights"):
        ic[k] = [str(int(v)) for v in ic[k]]
    C.write_columns(OUT_ITER, ic, order_it)
    loio_rows = loio_table(sensors, rule, raw_iter, its_by_sensor, base, sk, ik, re, rn, ru_ref, ru_raw, full_stats,
                           G["geo_alt"], J["alt_c"])
    write_loio(loio_rows)
    global _RULES
    _RULES = None                                  # force a reload from the file just written

    # ---- vertical exceptions (check 3). The after-rule bias is re-derived from the PUBLIC u_ellipsoid output
    # (ru_pub), not from the fit arrays, so a broken rule cannot hide behind its own bookkeeping.
    def pub_bias(s_, i_):
        mm_ = base & (sk == s_) & (ik == i_) & np.isfinite(ru_pub)
        return med(ru_pub[mm_]), int(mm_.sum())

    def classify(s_, i_, b_after):
        """(category, verified, text). 'unexplained' = no tested explanation applies."""
        m_ = base & (sk == s_) & (ik == i_) & np.isfinite(ru_raw)
        fl_ = np.unique(M["best_flight"][m_])
        gap_ = [med(G["geo_alt"][m_ & (M["best_flight"] == f)] - G["height"][m_ & (M["best_flight"] == f)]) for f in fl_]
        ctx = f"{len(fl_)} flight(s)"
        if len(gap_) >= 2 and max(gap_) - min(gap_) > 10:
            ctx += (f"; GT geo_alt - baro height differs by {max(gap_) - min(gap_):.0f} m between the flights here "
                    f"(candidate cause, not verified: section 5)")
        if gtf[s_]:
            return "relay", True, EXPLAIN["relay"] + f" [{ctx}]"
        if rule[s_]["informative"] == 0:
            return "uninformative", True, EXPLAIN["uninformative"] + f" [{ctx}]"
        if rule[s_]["support"] < 2:
            return "single_iteration", True, f"fewer than 2 iterations with >= {MIN_ITER_ROWS} height rows: no offset fitted [{ctx}]"
        others = [v[0] for k_, v in raw_iter[s_].items() if k_ != i_ and v[1] >= MIN_ITER_ROWS and np.isfinite(v[0])]
        if others:
            miss = abs(raw_iter[s_][i_][0] - float(np.median(others)))
            if miss > TOL_U:
                return "iteration_specific", True, (f"iteration-specific: this iteration's height bias differs by {miss:.1f} m "
                                                    f"from the median of the other iterations (> {TOL_U:g} m), so no constant "
                                                    f"sensor offset predicts it (LOIO) [{ctx}]")
        sup_ = [v[0] for v in raw_iter[s_].values() if v[1] >= MIN_ITER_ROWS and np.isfinite(v[0])]
        if len(sup_) >= 2 and max(sup_) - min(sup_) > 2 * TOL_U:
            return "iteration_spread", True, (f"per-iteration height medians of this sensor span {min(sup_):.1f} .. "
                                              f"{max(sup_):.1f} m (> {2 * TOL_U:g} m), so no single constant offset can put every "
                                              f"iteration within {TOL_U:g} m [{ctx}]")
        if abs(b_after) - TOL_U < 1.0:
            return "marginal", True, f"|bias| {abs(b_after):.1f} m exceeds the {TOL_U:g} m criterion by < 1 m [{ctx}]"
        return "unexplained", False, f"no tested explanation [{ctx}]"

    exceptions, exc_keys_pub = [], set()
    n_ok_after = n_ok_before = n_elig = 0
    sum_after = sum_before = 0.0
    for (s, i), st in sorted(iter_stats.items()):
        if st["n_u"] < CHECK_ROWS:
            continue
        b_after, _ = pub_bias(s, i)
        if not gtf[s] and rule[s]["informative"] != 0:        # rule-effectiveness sample: real, informative sensors
            n_elig += 1
            n_ok_after += abs(b_after) < TOL_U
            n_ok_before += abs(st["bias_u_noshift"]) < TOL_U
            sum_after += abs(b_after)
            sum_before += abs(st["bias_u_noshift"])
        if abs(b_after) >= TOL_U:
            exc_keys_pub.add((s, i))
            cat, ver, why = classify(s, i, b_after)
            exceptions.append((s, i, st["n_u"], st["bias_u"], b_after, why, cat, ver))

    # ======================================================== self-checks
    print()
    # 1: n vs independent recount
    cnt = read_match_stream()
    mism = {s: (cnt.get(s, 0), r["n"]) for s, r in zip(sensors, sens_rows) if cnt.get(s, 0) != r["n"]}
    missing = set(cnt) - set(sensors)
    check("1", not mism and not missing and sum(cnt.values()) == sum(r["n"] for r in sens_rows),
          f"per-sensor n equals a streaming recount of det_gt_match ({len(sensors)} sensors, "
          f"{sum(r['n'] for r in sens_rows):,} rows; mismatches {mism or 'none'}; missing {sorted(missing) or 'none'})", fails)

    # 2: 1000 residuals recomputed from raw lat/lon/alt with to_enu and GT lat/lon/geo_alt
    rng = np.random.default_rng(SEED)
    cand_idx = np.flatnonzero(base & np.isfinite(ru_raw))
    pick = rng.choice(cand_idx, size=1000, replace=False)
    frames = {it.key: C.iteration_frame(it) for it in C.iterations() if it.key in set(ik[pick].tolist())}
    gt_cache = {}
    worst = dict(e=0.0, n=0.0, u=0.0)
    for r in pick.tolist():
        fr = frames[ik[r]]
        de, dn, du_ = fr.to_enu(J["lat"][r], J["lon"][r], J["alt"][r])
        fk = M["best_flight"][r]
        key = (ik[r], fk)
        if key not in gt_cache:
            fm = (gt["iter_key"] == ik[r]) & (gt["flight_key"] == fk)
            o = np.argsort(gt["t"][fm], kind="stable")
            gt_cache[key] = {c: gt[c][fm][o] for c in ("t", "lat", "lon", "geo_alt")}
        g = gt_cache[key]
        t = M["t_cal"][r]
        k = int(np.searchsorted(g["t"], t, side="right"))
        k0, k1 = max(k - 1, 0), min(k, len(g["t"]) - 1)
        if k0 == k1 or g["t"][k1] == g["t"][k0]:
            w = 0.0
        else:
            w = (t - g["t"][k0]) / (g["t"][k1] - g["t"][k0])
        ee, nn, uu = fr.to_enu(g["lat"][[k0, k1]], g["lon"][[k0, k1]], g["geo_alt"][[k0, k1]])
        ge, gn, gu = [(1 - w) * a[0] + w * a[1] for a in (ee, nn, uu)]
        worst["e"] = max(worst["e"], abs(float(de) - ge - re[r]))
        worst["n"] = max(worst["n"], abs(float(dn) - gn - rn[r]))
        worst["u"] = max(worst["u"], abs(float(du_) - gu - ru_raw[r]))
    check("2", max(worst.values()) < 1e-6,
          f"1000 seeded residuals recomputed from raw lat/lon/alt via C.iteration_frame(it).to_enu vs GT: "
          f"max |diff| e {worst['e']:.1e} n {worst['n']:.1e} u {worst['u']:.1e} m", fails)

    # 3: vertical median after the rule (re-derived from the public u_ellipsoid; the rule must demonstrably help)
    elig = [(k, v) for k, v in iter_stats.items() if v["n_u"] >= CHECK_ROWS]
    bad_fit = {k for k, v in elig if abs(v["bias_u_after"]) >= TOL_U}
    cats = Counter(e[6] for e in exceptions)
    ok3 = (exc_keys_pub == bad_fit and all(e[7] for e in exceptions) and n_ok_after > n_ok_before
           and sum_after < sum_before)
    check("3", ok3,
          f"|median vertical residual after u_ellipsoid| < {TOL_U:g} m in {len(elig) - len(exceptions)} of {len(elig)} "
          f"(sensor, iteration) pairs with >= {CHECK_ROWS} rows; {len(exceptions)} exception(s), all with a tested "
          f"explanation {dict(cats)} (unexplained: {cats.get('unexplained', 0)}); exception set from the public function "
          f"{'==' if exc_keys_pub == bad_fit else '!='} the fit's; rule effectiveness on {n_elig} non-relay informative pairs: "
          f"within tolerance {n_ok_before} -> {n_ok_after}, mean |bias| {sum_before / max(n_elig, 1):.1f} -> "
          f"{sum_after / max(n_elig, 1):.1f} m (as reported -> after rule); max |public - fit| {diff_pub:.1e} m", fails)
    check("3b", diff_pub < 1e-9 and gt_curv[0] > -2.5 and gt_curv[2] < 0.01,
          f"GT u is geo_alt based: u - geo_alt min/median/max {gt_curv[0]:.2f}/{gt_curv[1]:.2f}/{gt_curv[2]:.2f} m "
          f"(earth curvature); public u_ellipsoid == internal fit", fails)

    # 4: runtime (evaluated at the end)
    # 5: AREA_H0_M against the testing anchors; evaluation coverage of the rule
    tm = np.char.startswith(D["iter_key"].astype(str), "testing")
    am = anchor_mask(D) & tm
    per_anchor = [float(np.median(D["sensor_alt"][am & (D["iter_key"] == a) & (D["sensor_key"] == b)]))
                  for a, b in sorted({(a, b) for a, b in zip(D["iter_key"][am].tolist(), D["sensor_key"][am].tolist())})]
    area_med = float(np.median(per_anchor))
    check("5", abs(area_med - AREA_H0_M) <= 1.0,
          f"AREA_H0_M = {AREA_H0_M:g} m vs median of {len(per_anchor)} testing anchor heights {area_med:.1f} m", fails)
    check("6", all(np.isfinite(r["sigma_e_m"]) or r["n"] < MIN_FOR_SIGMA for r in sens_rows)
          and all(c in cols for c in ("sigma_e_m", "sigma_n_m", "sigma_u_m", "sensor_key", "n", "n_iterations", "bias_e_m",
                                      "bias_n_m", "bias_u_m", "sigma_h_m", "p50_h_m", "p90_h_m", "u_rule", "stable",
                                      "applies_to", "note")),
          f"sensor_noise.csv has the required columns; sigma finite whenever n >= {MIN_FOR_SIGMA}", fails)

    # ---- evaluation coverage of the rule (inference only; nothing is fitted)
    ev_lines = []
    rules_final = load_rules()
    pos_ev = np.isin(D["meas_kind"], Q.POSITION_KINDS) & ~tm
    u_all = u_ellipsoid({k: D[k] for k in ("iter_key", "sensor_key", "alt_c", "alt_datum", "sensor_alt", "sensor_alt_datum",
                                           "sensor_pos_valid")}, rules_final, heights)
    ev_sensors = sorted(np.unique(D["sensor_key"][pos_ev]).tolist())
    unseen = [s for s in ev_sensors if s not in rules_final]
    for i in sorted(np.unique(D["iter_key"][~tm]).tolist()):
        m = pos_ev & (D["iter_key"] == i) & np.isfinite(u_all)
        ev_lines.append((i, int(m.sum()), med(u_all[m] - D["alt_c"][m]), sorted(np.unique(D["sensor_key"][m]).tolist())))

    # 8: u_ellipsoid depends on the rows passed (anchors): a subset must warn, the full table must not
    keys = ("iter_key", "sensor_key", "alt_c", "alt_datum", "sensor_alt", "sensor_alt_datum", "sensor_pos_valid")
    agl_rows = np.isin(D["sensor_key"], sorted(agl_switch)) & (D["sensor_alt"] > 0) & (D["sensor_alt"] < AGL_MAX_M) \
        & (D["sensor_pos_valid"] == 1) & np.isin(D["meas_kind"], Q.POSITION_KINDS)
    sub_msg, sub_ok = "no AGL-reference rows to test", True
    if agl_rows.any():
        sub_diffs, n_warn = [], 0
        for i in sorted(np.unique(D["iter_key"][agl_rows]).tolist()):
            sel = agl_rows & (D["iter_key"] == i)
            sub = {k: D[k][sel] for k in keys}
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter("always")
                u_sub = u_ellipsoid(sub, rules_final)
            n_warn += any("falls back to AREA_H0_M" in str(w_.message) for w_ in wl)
            with warnings.catch_warnings(record=True) as wf:
                warnings.simplefilter("always")
                fm_ = D["iter_key"] == i
                u_full = u_ellipsoid({k: D[k][fm_] for k in keys}, rules_final)[sel[fm_]]
            sub_ok &= not any("falls back" in str(w_.message) for w_ in wf)
            sub_diffs.append(float(np.nanmax(np.abs(u_sub - u_full))))
        sub_ok &= n_warn >= 1
        sub_msg = (f"NYTF-only subsets fall back to AREA_H0_M with a RuntimeWarning in {n_warn} of "
                   f"{len(sub_diffs)} AGL iteration(s) (max change vs the whole iteration {max(sub_diffs):.1f} m); "
                   f"whole-iteration calls do not warn")
    check("8", sub_ok, "u_ellipsoid needs the whole iteration: " + sub_msg, fails)

    # 9: the '*' rows of sensor_noise_loio.csv reproduce sensor_noise.csv exactly (both read back from disk)
    lo_b = C.read_columns(OUT_LOIO)
    sn_b = C.read_columns(OUT_CSV)
    sn_idx = {k: i for i, k in enumerate(sn_b["sensor_key"])}
    bad9 = []
    n_star = 0
    for i, (s_, h_) in enumerate(zip(lo_b["sensor_key"], lo_b["held_out"])):
        if h_ != "*":
            continue
        n_star += 1
        j_ = sn_idx.get(s_)
        for c_ in ("n", "n_u", "u_offset_m", "u_informative", "sigma_e_m", "sigma_n_m", "sigma_u_m"):
            if j_ is None or lo_b[c_][i] != sn_b[c_][j_]:
                bad9.append(f"{s_}.{c_}")
    check("9", n_star == len(sn_b["sensor_key"]) and not bad9,
          f"sensor_noise_loio.csv '*' rows == sensor_noise.csv as written ({n_star} of {len(sn_b['sensor_key'])} sensors; "
          f"columns n, n_u, u_offset_m, u_informative, sigma_e/n/u_m compared as text; mismatches {bad9[:6] or 'none'})", fails)

    # 10: every held_out = k row (and u_eval_apply) recomputed with plain Python loops over the base rows without k;
    # mutation: the same recomputation WITH k must disagree with the file somewhere (the check can fail)
    rows_by_s = defaultdict(list)
    for r_ in np.flatnonzero(base).tolist():
        rows_by_s[str(sk[r_])].append((str(ik[r_]), float(re[r_]), float(rn[r_]), float(ru_ref[r_]), float(ru_raw[r_]),
                                       float(G["geo_alt"][r_]), float(J["alt_c"][r_])))
    infm = {s_: int(v_) for s_, v_ in zip(sn_b["sensor_key"], sn_b["u_informative"])}
    adf = {s_: float(v_ or 0.0) != 0.0 for s_, v_ in zip(sn_b["sensor_key"], sn_b["u_offset_m"])}
    ref10 = loio_plain(rows_by_s, infm, adf)
    mut10 = loio_plain(rows_by_s, infm, adf, exclude_k=False)

    def num(v_):
        return float(v_) if v_ != "" else float("nan")

    def diff(a_, b_):
        if a_ != a_ or b_ != b_:
            return 0.0 if (a_ != a_) == (b_ != b_) else float("inf")
        return abs(a_ - b_)

    worst10, bad10, n_k, mut_hits = 0.0, [], 0, 0
    keys_file = set()
    for i, (s_, h_) in enumerate(zip(lo_b["sensor_key"], lo_b["held_out"])):
        keys_file.add((s_, h_))
        rf = ref10.get((s_, h_))
        if rf is None:
            bad10.append(f"{s_}/{h_} missing in recomputation")
            continue
        if h_ == "*":
            for c_ in ("u_eval_apply", "n_better", "n_support"):
                if int(lo_b[c_][i]) != int(rf[c_]):
                    bad10.append(f"{s_}/*.{c_}")
            for c_ in ("pooled_med_du_loio_m", "pooled_med_du_none_m"):
                d_ = diff(num(lo_b[c_][i]), rf[c_])
                worst10 = max(worst10, d_)
                if d_ > 5e-4 + 1e-6:
                    bad10.append(f"{s_}/*.{c_}")
            continue
        n_k += 1
        for c_ in ("n", "n_u", "u_adopted", "u_support", "u_informative"):
            if int(lo_b[c_][i]) != int(rf[c_]):
                bad10.append(f"{s_}/{short_iter(h_)}.{c_}")
        hit = False
        for c_ in ("u_offset_m", "sigma_e_m", "sigma_n_m", "sigma_u_m"):
            d_ = diff(num(lo_b[c_][i]), rf[c_])
            worst10 = max(worst10, d_)
            if d_ > 5e-4 + 1e-6:                     # file is rounded to 3 decimals
                bad10.append(f"{s_}/{short_iter(h_)}.{c_}")
            hit |= diff(num(lo_b[c_][i]), mut10[(s_, h_)][c_]) > 5e-4 + 1e-6
        hit |= int(lo_b["n"][i]) != int(mut10[(s_, h_)]["n"])
        mut_hits += hit
    missing10 = set(ref10) - keys_file
    n_apply = sum(1 for s_, h_, a_ in zip(lo_b["sensor_key"], lo_b["held_out"], lo_b["u_eval_apply"]) if h_ == "*" and a_ == "1")
    check("10", not bad10 and not missing10 and n_k > 0 and mut_hits > 0,
          f"{n_k} held-out rows + {n_star} '*' decisions recomputed with plain Python loops over the base rows without "
          f"the held-out iteration (offset fit, informativeness flag, sigmas, u_eval_apply): max |diff| {worst10:.1e} (tolerance 5e-4, the "
          f"file's rounding); mismatches {bad10[:6] or 'none'}; rows missing in the file {sorted(missing10)[:4] or 'none'}; "
          f"mutation (recomputed WITH the held-out iteration) disagrees on {mut_hits} of {n_k} rows (must be > 0); "
          f"u_eval_apply = 1 for {n_apply} sensor(s)", fails)

    # ======================================================== bearing sensors
    az_off = az_offsets_applied()
    bear = (M["kind"] == "bearing") & (M["matched"] == 1) & (M["ambiguous"] == 0) & (M["gt_airborne"] == 1)
    az_gt = np.full(N, np.nan)
    for a in np.unique(M["iter_key"][bear]):
        for f in np.unique(M["best_flight"][bear & (M["iter_key"] == a)]):
            sel = bear & (M["iter_key"] == a) & (M["best_flight"] == f)
            fm = (gt["iter_key"] == a) & (gt["flight_key"] == f)
            fe, fn = Q.interp_flight(gt, fm, M["t_cal"][sel], cols=("e", "n"))
            az_gt[sel] = C.azimuth_deg(fe - J["sensor_e"][sel], fn - J["sensor_n"][sel])
    az_res = np.full(N, np.nan)
    for s, i in sorted({(a, b) for a, b in zip(sk[bear].tolist(), ik[bear].tolist())}):
        m = bear & (sk == s) & (ik == i)
        off = az_off.get((s, i))
        if off is None:
            continue
        az_res[m] = C.angle_diff_deg(J["az_deg"][m] + off, az_gt[m])
    bm = bear & np.isfinite(az_res)
    az_check = float(np.nanmax(np.abs(np.abs(az_res[bm]) - M["resid"][bm]))) if bm.any() else 0.0
    check("7", az_check < 2e-3, f"bearing residuals recomputed from the applied azimuth offsets reproduce det_gt_match resid "
          f"(max |diff| {az_check:.1e} deg on {int(bm.sum()):,} rows; unit deg)", fails)
    bear_rows = []
    for s in sorted(np.unique(sk[bm]).tolist()):
        m = bm & (sk == s)
        its_b = sorted(np.unique(ik[m]).tolist())
        centred = np.concatenate([az_res[m & (ik == i)] - med(az_res[m & (ik == i)]) for i in its_b])
        susp = [short_iter(i) for i in its_b if abs(med(az_res[m & (ik == i)])) > AZ_SUSPECT_DEG]
        bear_rows.append((s, "pooled", int(m.sum()), med(az_res[m]), robust_sigma(az_res[m]), robust_sigma(centred),
                          float(np.percentile(np.abs(az_res[m]), 90)),
                          ("offset suspect in " + ", ".join(susp)) if susp else ""))
        for i in its_b:
            mi = m & (ik == i)
            mdi = med(az_res[mi])
            bear_rows.append((s, short_iter(i), int(mi.sum()), mdi, robust_sigma(az_res[mi]), robust_sigma(az_res[mi] - mdi),
                              float(np.percentile(np.abs(az_res[mi]), 90)),
                              "offset suspect" if abs(mdi) > AZ_SUSPECT_DEG else ""))
    n_unapplied = int(((M["kind"] == "bearing") & (M["matched"] == 1) & ~np.isfinite(az_res) & (M["ambiguous"] == 0)
                       & (M["gt_airborne"] == 1)).sum())

    # ---- GT geo_alt vs baro height per flight (section 5)
    gap_rows = []
    for a in sorted(np.unique(gt["iter_key"]).tolist()):
        for f in sorted(np.unique(gt["flight_key"][gt["iter_key"] == a]).tolist()):
            fm = (gt["iter_key"] == a) & (gt["flight_key"] == f)
            air = fm & (gt["height"] > 5)
            if air.sum() >= 20:
                gap_rows.append((short_iter(a), f.split("_", 4)[-1] if f.count("_") >= 4 else f,
                                 gt["flight_key"][fm][0].split("_")[1], int(air.sum()),
                                 med((gt["geo_alt"] - gt["height"])[air]),
                                 med((gt["geo_alt"] - gt["height"])[fm & (np.abs(gt["height"]) < 2)])))
    # between-iteration spread of the vertical median: against geo_alt vs against baro height (radars)
    spread_rows = []
    for s in sensors:
        its = [i for i in its_by_sensor[s] if iter_stats[(s, i)]["n_u"] >= MIN_ITER_ROWS]
        if len(its) < 2 or gtf[s]:
            continue
        a_geo, a_h = [], []
        for i in its:
            m = base & (sk == s) & (ik == i) & np.isfinite(ru_raw)
            a_geo.append(med(ru_ref[m]))
            a_h.append(med((J["alt_c"] + shift - G["height"])[m]))
        spread_rows.append((s, len(its), float(np.std(a_geo)), float(np.std(a_h))))

    # ======================================================== report
    runtime = time.time() - t_start
    check("4", runtime < 300, f"runtime {runtime:.0f} s < 300 s", fails)
    write_md(sensors, sens_rows, iter_rows, rule, iter_stats, exceptions, ev_lines, unseen, heights, D, bear_rows,
             n_unapplied, gap_rows, spread_rows, gt_curv, agl_switch, area_med, fails, loio_rows)
    print()
    print(f"wrote {OUT_CSV.relative_to(C.REPO)} ({len(sens_rows)} sensors), {OUT_ITER.relative_to(C.REPO)} "
          f"({len(iter_rows)} rows), {OUT_MD.relative_to(C.REPO)}")
    if fails:
        print("SELF-CHECKS FAILED:", ", ".join(fails))
        return 1
    print("ALL SELF-CHECKS PASSED")
    return 0


def write_md(sensors, sens_rows, iter_rows, rule, iter_stats, exceptions, ev_lines, unseen, heights, D, bear_rows,
             n_unapplied, gap_rows, spread_rows, gt_curv, agl_switch, area_med, fails, loio_rows=()) -> None:
    L = ["# Sensor measurement-error model (testing vs Dronetag ground truth)\n"]
    L.append("Generated by `tools/sensor_noise.py` (deterministic). Local ENU metres only; no absolute coordinates. "
             "Outputs: `sensor_noise.csv` (one row per sensor, the file `tools/sensor_tracks.py` reads: `sigma_e_m`, "
             "`sigma_n_m`, `sigma_u_m`), `sensor_noise_by_iteration.csv`, this report.\n")
    L.append("## 0. Method and caveats\n")
    L.append(f"* Rows: position detections with `matched == 1` (horizontal distance to the best Dronetag flight at the "
             f"calibrated time <= {GATE_M:g} m), `ambiguous == 0`, `gt_airborne == 1` (GT height > 5 m). Testing only, "
             f"{sum(r['n'] for r in sens_rows):,} rows, {len(sensors)} sensors. Residual = detection - GT in the iteration ENU "
             f"frame, GT linearly interpolated at `t_cal`.")
    L.append(f"* **The 50 m association gate truncates the horizontal residuals.** A detection > 50 m from every drone is "
             f"not in the sample, so `sigma_e/n/h` and `p90_h` are lower bounds for sensors whose true error approaches the "
             f"gate (radars with a 1-2 degree azimuth bias at 400 m range already reach 14 m). `gate_frac` = share of "
             f"airborne candidate rows (finite residual) that are in the sample (clutter and truncation are not "
             f"separable); `sigma_h_100m_m` repeats sigma_h with a 100 m gate. The gate is horizontal only, so vertical "
             f"statistics are not truncated.")
    L.append("* Statistics are robust: bias = median, sigma = 1.4826 * MAD, `sigma_h_m` = hypot(sigma_e, sigma_n), "
             "`p50_h_m` / `p90_h_m` = percentiles of the raw horizontal error (not de-biased). Pooled values use all rows of a "
             "sensor (so sigma includes between-iteration bias scatter).")
    L.append(f"* `stable`: every iteration with >= {MIN_ITER_ROWS} rows has |bias_e|, |bias_n| within {TOL_H:g} m of the pooled "
             f"bias (needs >= 2 such iterations); `applies_to = all` only then, else `testing`. `u_stable` / `u_applies_to`: "
             f"same on the vertical after-rule bias with tolerance {TOL_U:g} m (and the altitude must follow the drone).")
    L.append(f"* GT `u` is geo_alt based (height above the WGS84 ellipsoid): u - geo_alt is {gt_curv[0]:.2f} .. {gt_curv[2]:.2f} m "
             f"(median {gt_curv[1]:.2f}), the earth-curvature term of the h0 = 0 frame.")
    L.append("* Relay sensors (`gt_feed` = 1: LJJX, QTVS, VKYY) reproduce the Dronetag feed; their sigma is not a sensor error "
             "model and they are label leakage if used for evaluation.\n")

    L.append("## 1. Per-sensor error model (pooled over testing iterations)\n")
    rows = []
    for r in sens_rows:
        rows.append([r["sensor_key"], r["n"], r["n_iterations"], fmt(r["bias_e_m"]), fmt(r["bias_n_m"]), fmt(r["bias_u_m"]),
                     fmt(r["sigma_e_m"]), fmt(r["sigma_n_m"]), fmt(r["sigma_u_m"]), fmt(r["sigma_h_m"]),
                     fmt(r["p50_h_m"]), fmt(r["p90_h_m"]), fmt(r["sigma_h_100m_m"]), fmt(r["gate_frac"], 2),
                     "" if not np.isfinite(r["stable"]) else int(r["stable"]), r["applies_to"], r["note"]])
    L += md_table(["sensor", "n", "iters", "bias E", "bias N", "bias U (after ref shift, before offset)", "sigma E", "sigma N", "sigma U (after rule)",
                   "sigma H", "p50 H", "p90 H", "sigma H (100 m gate)", "gate_frac", "stable", "applies_to", "note"], rows)
    L.append("\nAll values in metres. `bias U` is the height residual of the reported altitude after the reference "
             "shift (non-zero only for NYTF-type sensors whose altitude is above ground in some iterations) and before the "
             "offset, pooled over rows (so it mixes iterations: for NYTF it averages the shifted iterations, see section 3; "
             "the unshifted value is `bias_u_noshift_m` in the CSV); `sigma U` is the spread after `u_ellipsoid`. "
             f"Sensors with fewer than {MIN_FOR_SIGMA} rows have no sigma / p50 / p90 (empty cells): too few rows for a "
             "reliable, isotropic estimate.\n")

    L.append("## 2. Height-to-ellipsoid rule\n")
    L.append("`u_ellipsoid(det)` in `tools/sensor_noise.py` returns the height above the WGS84 ellipsoid with no ground truth:\n")
    L.append("```")
    L.append("u_ell = alt_c + ref_shift + offset(sensor)")
    L.append(f"ref_shift = H_site(iteration, sensor) - sensor_alt   if sensor is flagged agl_switch and 0 < sensor_alt < {AGL_MAX_M:g}")
    L.append("          = 0                                       otherwise")
    L.append("```")
    L.append(f"* `H_site` = median `sensor_alt` of the ellipsoid anchors (E datum, {ANCHOR_RANGE[0]:g}..{ANCHOR_RANGE[1]:g} m: "
             f"EGOI, ILYS, STUG, IUKL and NYTF when it reports an ellipsoid height) at the same physical site "
             f"(`sensor_sites.csv`) in that iteration, else the anchors of the iteration, else `AREA_H0_M` = {AREA_H0_M:g} m "
             f"(testing median {area_med:.1f} m). This is `dz_sensor + H_site` with the sensor's own ellipsoid height replaced by "
             f"the site's.")
    L.append(f"* AGL-switch sensors found in testing: {', '.join(sorted(agl_switch)) or 'none'} (`sensor_alt` about 2 m in some iterations, "
             f"about 79 m in others). The same flag is applied to evaluation, where NYTF reports 2 m in DIII/S2-I3.")
    L.append(f"* `offset(sensor)`: minus the median of the per-iteration median residuals (iterations with >= {MIN_ITER_ROWS} height rows), "
             f"adopted only with >= 2 supporting iterations and a lower leave-one-iteration-out (LOIO) mean |held-out bias|, "
             f"else 0. It is a testing-fitted constant (the area geoid and the sensor's vertical calibration are assumed "
             f"unchanged in evaluation; the evaluation sensors sit in the same area).")
    L.append("* Datum: `G` (geoid) altitudes get their own fitted offset when the sensor was seen (EUIG); an unseen sensor gets 0.\n")
    rows = []
    for r in sens_rows:
        ru = rule[r["sensor_key"]]
        rows.append([r["sensor_key"], r["u_rule"], ru["support"], fmt(ru["loio_before"]), fmt(ru["loio_after"]),
                     fmt(ru["slope"], 2), {1: "yes", 0: "NO", -1: "?"}[ru["informative"]], r["u_applies_to"]])
    L += md_table(["sensor", "rule", "supporting iterations", "LOIO mean abs bias, offset 0 (m)", "LOIO, fitted offset (m)",
                   "slope alt vs GT", "alt follows drone", "u applies_to"], rows)
    L.append("\n`H_site` per iteration for the AGL-switch sensors (testing and evaluation, from the same inference code):\n")
    rows = []
    for (it, s), h in sorted(heights.items()):
        if s in agl_switch:
            m = (D["iter_key"] == it) & (D["sensor_key"] == s)
            sa = med(D["sensor_alt"][m])
            rows.append([it, s, fmt(sa), fmt(h), fmt(h - sa) if 0 < sa < AGL_MAX_M else "0 (ellipsoid reference)"])
    L += md_table(["iteration", "sensor", "sensor_alt", "H_site", "applied shift"], rows)
    L.append("\nEvaluation coverage of the rule (inference only, no fitting): rows with a finite `u_ellipsoid`, median applied "
             "change `u_ell - alt` per iteration.\n")
    L += md_table(["iteration", "rows with u_ell", "median (u_ell - alt) (m)", "sensors"],
                  [[i, n, fmt(m_), ", ".join(s)] for i, n, m_, s in ev_lines])
    if unseen:
        L.append(f"\nEvaluation position sensors with no testing rule (offset 0 applied): {', '.join(unseen)}.")
    L.append("")

    L.append("## 3. Vertical bias per (sensor, iteration): before and after the rule\n")
    rows = []
    for s, i, st in iter_rows:
        rows.append([s, short_iter(i), st["n_u"], st["n_flights_u"], fmt(st["sensor_alt"]), fmt(st["bias_u_noshift"]),
                     fmt(st["bias_u"]), fmt(st["bias_u_after"]), fmt(st["sigma_u_raw"]), fmt(st["sigma_u"])])
    L += md_table(["sensor", "iteration", "n (height)", "flights", "sensor_alt", "bias as reported (m)",
                   "bias after ref shift (m)", "bias after rule (m)", "sigma before offset", "sigma after rule"], rows)
    L.append("\nAs reported = the detection `u` (ENU-frame, vs GT geo_alt); after ref shift = NYTF-type AGL correction only; "
             "after rule = `u_ellipsoid` (ENU-frame equivalent). The offset is fitted on the same testing iterations "
             "(in-sample); the LOIO column in section 2 and the `loio_held_out_u_m` CSV column are the honest transfer "
             "estimate.\n")

    L.append(f"## 4. Vertical exceptions (|median after rule| >= {TOL_U:g} m, >= {CHECK_ROWS} rows)\n")
    if exceptions:
        L.append("Explanations are limited to tested statements (relay flag, altitude slope, iteration count, "
                 "leave-one-iteration-out miss); a GT-reference gap between flights is shown as a *candidate* only. "
                 "`bias before` = after the reference shift, before the offset.\n")
        L += md_table(["sensor", "iteration", "n", "bias before", "bias after", "category", "explanation"],
                      [[s, short_iter(i), n, fmt(b), fmt(a), c_, w] for s, i, n, b, a, w, c_, _v in exceptions])
    else:
        L.append("None.")
    L.append("")

    L.append("## 5. The GT reference itself: geo_alt vs barometric height\n")
    L.append("Dronetag `height` (above take-off, barometric) and `geo_alt` (ellipsoid, GNSS) are different sources. Their "
             "difference is not a constant per flight: it differs between flights and between the airborne and ground "
             "parts of one flight by tens of metres, so it is NOT simply the take-off ellipsoid height. Whether the GT "
             "geo_alt or the baro height is the less reliable vertical reference is not settled by this report. Median "
             "(geo_alt - height) over airborne samples, and at ground level (|height| < 2 m):\n")
    L += md_table(["iteration", "flight", "side", "airborne samples", "geo_alt - height (airborne)", "geo_alt - height (ground)"],
                  [[a, b, c, d, fmt(e_), fmt(f_)] for a, b, c, d, e_, f_ in gap_rows])
    L.append("\nBetween-iteration standard deviation (m) of the per-iteration median vertical offset of each sensor "
             "(>= 2 iterations with >= 30 height rows, relays excluded), measured against geo_alt (the reference used "
             "everywhere above) and against the barometric `height` above take-off (which has an unknown per-flight datum, so "
             "this comparison is only indicative). Values are mixed: some sensors scatter less against geo_alt, others "
             "against baro height; there is no consistent evidence that the geo_alt reference is the cause of the "
             "iteration-to-iteration scatter:\n")
    L += md_table(["sensor", "iterations", "std vs geo_alt", "std vs baro height"],
                  [[s, n, fmt(a), fmt(b_)] for s, n, a, b_ in spread_rows])
    L.append("")

    L.append("## 6. Bearing sensors: azimuth residual (degrees, after the applied offsets)\n")
    L.append("Signed residual = (az_deg + applied offset from `sensor_calibration.csv`) - azimuth of the GT drone seen from the "
             "sensor, at `t_cal`; rows `matched == 1`, non-ambiguous, airborne. Truncated by the angular gate (5 deg, or twice the "
             "reported azimuth error up to 30 deg) so the spread is a lower bound. Report only; not used by the position model."
             + (f" {n_unapplied:,} matched bearing rows belong to (sensor, iteration) groups with no applied offset and are "
                f"not shown." if n_unapplied else " Every matched bearing row belongs to a group with an applied offset.")
             + " The `pooled` rows mix iterations: their robust std includes the between-iteration scatter of the median "
               "residual. `within-iteration std` is the robust std about each iteration's own median and is the better "
               "estimate of the azimuth noise. A median residual of more than 2 deg after the applied offset (flag "
               "`offset suspect`) means the offset of that iteration does not centre the data, typically because the angular "
               "gate cut the residual distribution (p90 close to the gate); the std of such a group is not a noise estimate.\n")
    L += md_table(["sensor", "iteration", "n", "median residual (deg)", "robust std (deg)", "within-iteration std (deg)",
                   "p90 abs residual (deg)", "flag"],
                  [[s, i, n, fmt(a, 2), fmt(b, 2), fmt(w_, 2), fmt(c_, 2), f_] for s, i, n, a, b, w_, c_, f_ in bear_rows])
    L.append("\n## 7. Self-checks\n")
    L.append("See the console output of `python3 tools/sensor_noise.py` (1 n recount, 2 residual recomputation from raw "
             "lat/lon/alt, 3 vertical medians after the rule, 3b GT u definition, 4 runtime, 5 area height, 6 columns, "
             "7 bearing residual reproduction, 8 subset warning, 9 LOIO '*' rows = sensor_noise.csv, 10 LOIO rows "
             "recomputed independently). " + ("FAILED: " + ", ".join(fails) if fails else "All passed at write time."))
    if loio_rows:
        L.append("\n## 8. Leave-one-iteration-out refits (`sensor_noise_loio.csv`)\n")
        L.append("Leave-one-iteration-out (LOIO) = refit every number without one testing iteration, so that the value "
                 "applied to that iteration has never seen its ground truth. `tools/sensor_tracks.py` uses these rows with "
                 "the corrections `height_loio` (height offset) and `var_loio` (error spread); without corrections nothing "
                 "here is used. Held-out offset = the offset rule (section 2) fitted on the other iterations only; "
                 "|du| = absolute height residual after the reference shift, median over the held-out iteration's rows, "
                 "with the held-out offset vs with no offset.\n")
        L.append(f"Pre-registered evaluation decision `u_eval_apply` (fixed before looking at any result): the full "
                 f"offset is applied to evaluation only if the full rule is adopted, the held-out offset beats offset 0 "
                 f"on at least ceil({EVAL_SHARE:.3g} x n) of the n supporting iterations, and the pooled median |du| over "
                 f"those held-out rows is lower with the held-out offsets than with 0. Otherwise evaluation uses 0. "
                 f"A sensor with only 2 supporting iterations always gets 0: without one of them a single iteration "
                 f"is left, and the offset rule needs 2 to adopt an offset, so every held-out offset is 0.\n")
        L.append("Informative = the reported altitude follows the drone (slope vs ground truth >= "
                 f"{SLOPE_MIN:g}; -1 = too few rows to tell, treated as informative), recomputed without the held-out "
                 "iteration; 0 means no usable height (offset 0, no height sigma).\n")
        rows = []
        for r in loio_rows:
            if r["held_out"] == "*":
                continue
            rows.append([r["sensor_key"], short_iter(r["held_out"]), r["n"], r["u_informative"], fmt(r["u_offset_m"]), r["u_adopted"],
                         fmt(r["held_out_med_du_loio_m"]), fmt(r["held_out_med_du_none_m"]), fmt(r["sigma_e_m"]),
                         fmt(r["sigma_n_m"]), fmt(r["sigma_u_m"])])
        L += md_table(["sensor", "held out", "rows used (without it)", "informative", "held-out offset (m)", "adopted",
                       "held-out median |du|, held-out offset (m)", "held-out median |du|, no offset (m)",
                       "sigma E (m)", "sigma N (m)", "sigma U (m)"], rows)
        L.append("")
        rows = [[r["sensor_key"], fmt(r["u_offset_m"]), r["u_adopted"], r["n_support"], r["n_better"],
                 fmt(r["pooled_med_du_loio_m"]), fmt(r["pooled_med_du_none_m"]), r["u_eval_apply"]]
                for r in loio_rows if r["held_out"] == "*"]
        L += md_table(["sensor", "full offset (m)", "full rule adopted", "supporting iterations", "held-out wins",
                       "pooled median |du|, held-out offsets (m)", "pooled median |du|, no offset (m)",
                       "u_eval_apply"], rows)
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
