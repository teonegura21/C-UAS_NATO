#!/usr/bin/env python3
"""Radar error model in SENSOR coordinates: latency, azimuth rotation, range error and an
elevation-like vertical term per (radar, testing iteration), its stability across iterations, a
leave-one-iteration-out (LOIO) correction measured per iteration, a pre-registered decision table
for the next correction wave, a GT-free azimuth sanity check on evaluation, and a library API
(correct, params_for).

Why: an east / north mean of an azimuth error (1 deg = 35 m at 2 km, its direction set by where the
drone is relative to the radar) or of a latency (2 s x 10 m/s = 20 m along the flight) flips sign
from iteration to iteration although the cause is constant. Modelled in the radar's own
coordinates (range, azimuth, height) the same causes become constants that can transfer.

Rules
-----
* Testing only for every fit; LOIO everywhere: a value applied to testing iteration k is computed
  without k; evaluation uses the values pooled over all testing iterations; nothing is fitted on
  evaluation (step 5 estimates an angle from an evaluation iteration's own sensor data, GT-free,
  as a sanity check only).
* Radars: EGOI, NYTF, UXYJ, IUKL (deployable); EUIG is testing-only: fitted and reported, marked
  testing_only = 1, never a deployable decision.
* Corrections use MEASURED quantities only (measured position, the row's sensor site, the sensor's
  reported velocity or, where it reports none, the velocity differenced from its own reports).
  Truth enters only the fits and the error measurement.

Inputs
------
* data/derived/detections.csv.gz through cuas_clean.load_clean: position rows with
  sensor_tracks.position_mask (usable_pos, not duplicate, not far). Time t_cal = t + the APPLIED
  clock offset of sensor_tracks.time_offsets() (UXYJ -3.5 s, LJJX -2 s); every latency below is
  relative to t_cal.
* Heights: sensor_noise.u_ellipsoid with the RADARS' fitted offsets set to 0 ("base" height: the
  ground-truth-free datum / above-ground rule only). Reason: the sensor_noise offsets (UXYJ
  +23.9 m, IUKL +40.4 m, EUIG +11.2 m) were fitted with GT on all six testing iterations, so a
  vertical model on top of them would be in-sample; with them removed the constant c_u is fitted
  LOIO and replaces them (the in-sample offset is kept as the reference method 'u_sn_insample').
  correct() therefore expects u in the base frame: base_height() gives it.
* data/derived/sensor_tracks_v2/<slug>__tracks.csv (testing labels): segments labelled drone_air
  with a best_flight. ALL raw rows of such a segment (same sensor_key and object_id, t_cal in
  [t0, t1]; the count must equal the segment's n_rows, self-check 7) are compared with that
  flight's Dronetag truth interpolated at t_cal (cuas_clean.interp_flight on groundtruth.csv.gz:
  e, n, u, ve, vn, vu). No per-row distance gate (a gate truncates the residuals); rows with
  |horizontal residual| > GROSS_M (resp. |vertical| > GROSS_M) are left out of the fits only.

Model per (radar, iteration)
----------------------------
Residual r = p_meas - p_truth. Basis at the truth position: l = horizontal line-of-sight unit
(site -> target), t = l rotated 90 deg clockwise = the direction of increasing azimuth, R = horizontal
range. The horizontal residual is taken in polar form, r_t = R x (azimuth of the measurement - azimuth
of the truth, radians, wrapped) and r_l = measured range - true range: to first order these equal
r . t and r . l, and they make correct() the exact inverse of the fitted model (the projection would
leak -R daz^2 / 2 into s_r and daz x s_r into daz; self-check 1 showed this at daz = 1.2 deg).
    r_t = daz R - tau v_t                       (v = truth velocity; v_t = v . t, v_l = v . l)
    r_l = b_r + s_r R - tau v_l
    r_u = c_u + del R                           (fitted separately; the -tau v_u term is left out
                                                 of the fit: drone vertical speeds are small; the
                                                 correction applies tau v_u when tau is applied)
daz (deg, positive = the radar puts the target clockwise of the truth; = -radar_bias of
sensor_calibration.csv), tau (s, positive = the radar reports late), b_r (m, range bias at R = 0),
s_r (range scale), c_u (m, height bias at R = 0), del (deg, elevation-like bias: height error
growing with range). Internally s_r and del are fitted about the fit's mean range R0 (centred
column), so the collinearity numbers mean what they say; b_r / c_u are converted back to R = 0.
* Huber IRLS (c = HUBER_C on a 1.4826 MAD scale per row type t / l / u), base weight per sample
  min(1, SEG_CAP / n_segment): one long segment weighs at most SEG_CAP samples.
* Identifiability (pre-registered, before the fit): tau is dropped if the weighted rms truth speed
  < MIN_SPEED_RMS or its variance inflation (VIF) > VIF_MAX; s_r / del are dropped if the weighted
  range sd < MIN_R_SD or VIF > VIF_MAX (worst first, then recomputed). VIF_j = j-th diagonal of
  the inverse of the column-normalised information matrix X'WX (1 = orthogonal to the rest);
  corr = its off-diagonals. daz, b_r, c_u are always kept.
* Interval: block bootstrap over segments (N_BOOT replicates, each refits the same terms), half
  width t975(B-1) sqrt(B/(B-1)) sd(replicates), B = segments; none with B < MIN_CI_BLOCKS,
  few_blocks = 1 when B < FEW_BLOCKS (a sampling interval: the slow bias is shared within a
  flight, so segments of one flight are not independent).
* Models: every horizontal term subset that the decision procedure needs ('h:daz',
  'h:daz+tau', ..., 'h:daz+tau+b_r+s_r' = h full; s_r only with b_r), 'v:c_u', 'v:c_u+del' (v
  full), and the hover cross-check 'hover:daz' (truth speed < HOVER_MS, R >= HOVER_MIN_R: no
  latency can act there).

Stability, LOIO and decisions
-----------------------------
* A fit is valid with >= MIN_FIT_N samples in >= MIN_FIT_SEGS segments. Pooled value of a term =
  mean of the valid per-iteration estimates (an iteration where the term was dropped does not
  count). LOIO value for testing k = that mean without k; evaluation = all testing iterations.
  CI of the pooled mean: mean +- t975(n-1) sd / sqrt(n); prediction interval for a new iteration:
  mean +- t975(n-1) sd sqrt(1 + 1/n).
* Validation (validation.csv) per (radar, testing k): median horizontal / |vertical| / 3-D error and
  share within 10 / 25 / 50 m (horizontal and 3-D) of: raw; loio:<model> and insample:<model> for
  every model; loio:full / insample:full (h full + v full); loio:decided (final term set, LOIO
  values; the set itself was chosen on all iterations); loio:decided_nested (the whole decision
  procedure rerun without k, then LOIO values: no selection optimism; both use the deployed
  sensor_noise height where no vertical term is final); u_sn_insample (the height
  deployed today: sensor_noise offset, in-sample).
* PRE-REGISTERED decision rule (decisions.csv), sequential per radar in the order daz, tau, b_r,
  s_r (horizontal) and c_u, del (vertical); A = the terms already accepted (start empty): term X
  is tested as A + X (s_r brings b_r with it, del brings c_u) against A. 'apply' iff the LOIO
  correction with A + X has a strictly lower median horizontal error (vertical for c_u / del) than
  with A on >= ceil(2/3 n) of the radar's n testing iterations (those with >= MIN_VAL_N rows and a
  LOIO value) AND the pooled CI of X in model A + X excludes 0 (needs >= 2 iterations). Then
  A = A + X. Column 'decision' is that test. Column 'final' (what params_for applies) = the term is
  in the final set: a term brought in by another (b_r by s_r, c_u by del) is final even when its own
  row is 'reject', with the value of the accepted model (final_value). Non-regression gate (added
  after review, not pre-registered): the accepted vertical set's LOIO correction must have a lower
  median |vertical error| than the deployed sensor_noise height on >= ceil(2/3 n) iterations
  (column vs_deployed), else the vertical set is dropped and the deployed height is kept.

Reference latency (step 1, reference_latency.csv)
--------------------------------------------------
STUG, ILYS, CEQX (decoded drone telemetry) on their drone_air segments: r_e = b_e - tau ve,
r_n = b_n - tau vn (truth velocity), Huber IRLS, same weights / bootstrap; LOIO tau per sensor =
mean of the other iterations; error of the raw and the LOIO-latency-corrected reports (measured
velocity). Decision per sensor (pooled row, added after review): apply iff the LOIO correction
lowers the median error on >= ceil(2/3 n) iterations AND the pooled CI excludes 0; a sensor
without a valid fit gets tau 0. Step 5's primary variant is 'lat' only if every reference passes.

GT-free azimuth on evaluation (step 5)
--------------------------------------
radar_registration.estimate_az (rotation-centred co-observation pairs on the sensor_tracks_v2
grid, GT loaders disabled) of each radar against the references STUG / CEQX / ILYS / IKJX / SMFS
('noutm') and + VKYY / QTVS ('all'), with the references' grid positions moved forward by their
latency (e += tau ve, n += tau vn; LOIO tau on testing, pooled on evaluation), and without it.
Each estimate is compared with the prediction interval of the az-only model ('h:daz'): LOIO on
testing (plus the iteration's GT value), pooled over all testing iterations on evaluation.

Outputs (data/derived/radar_model/)
-----------------------------------
params.csv   one row per (sensor_key, iter_key, kind, model):
    kind: 'fit' (in-sample, testing iteration) | 'loio' (pooled over the other testing iterations,
          for testing iter_key) | 'pooled' (all testing; iter_key '*'; evaluation) |
          'decided' / 'decided_loio' (pooled / LOIO values of the final terms only, model
          'decided'; the rows params_for() reads). loio / decided_loio rows exist for EVERY
          testing key of a radar with data; a key without drone_air rows gets the full pool | 'gt_free_noutm' | 'gt_free_all' |
          'gt_free_noutm_nolat' | 'gt_free_all_nolat' (step 5, model 'h:daz')
    daz_deg, tau_s, b_r_m, s_r, c_u_m, del_deg and <p>_lo, <p>_hi (fit: bootstrap interval;
          loio / pooled: CI of the mean; gt_free: estimator interval); NaN = not in the model /
          not identified
    pi_lo_daz, pi_hi_daz  prediction interval of daz for a new iteration (loio / pooled rows;
          gt_free rows: the interval the estimate is checked against)
    inside        gt_free rows: 1 if the estimate lies inside [pi_lo_daz, pi_hi_daz], 0 if not, blank
    gt_daz_deg    gt_free rows on testing: the iteration's in-sample GT daz (h:daz)
    n_samples, n_segs, n_samples_u, few_blocks, terms, dropped ('|'-joined), R0_m, scale_t_m,
          scale_l_m, scale_u_m, vif ('term:value;...'), corr ('a~b:value;...'),
          rms_speed_ms, r_sd_m, corr_ru_vu (vertical residual vs truth vertical speed after the v
          fit), n_iter, iters ('|'-joined testing iterations pooled), testing_only, n_pairs, n_groups
validation.csv  (sensor_key, iter_key, method, n, n_u, med_h_m, med_v_m, med_3d_m, h_le10, h_le25,
          h_le50, d3_le10, d3_le25, d3_le50, testing_only)
decisions.csv   (sensor_key, axis, term, baseline, with_terms, n_iters, n_improved, need,
          improved_iters, pooled_value, ci_lo, ci_hi, ci_excludes_0, n_ci_iters, decision,
          vs_deployed, final, final_value, testing_only)
reference_latency.csv  (sensor_key, iter_key, kind 'fit' | 'loio' | 'pooled', tau_s, tau_lo,
          tau_hi, b_e_m, b_n_m, n_samples, n_segs, few_blocks, rms_speed_ms, n_iter, iters,
          n_val, med_h_raw_m, med_h_corr_m, n_vel, decision (pooled rows))
radar_model.md  plain-language report.

API (next wave)
---------------
params_for(sensor_key, held_out_iter=None, terms='decided') -> dict (incl. u_frame 'base' | 'any')
correct(sensor_key, e, n, u, t, ve, vn, vu, site_e, site_n, site_u, params) -> (e, n, u)
base_height(det, rules=None) -> base ellipsoid height for every row (radar offsets zeroed)
measured_velocity(seg, t, x) -> per-row velocity differenced from a sensor's own reports

Run: python3 tools/radar_model.py [--skip-checks]. Numbered self-checks; exit 1 on any failure:
(1) synthetic radar from real truth tracks with injected tau 1.7 s, daz +1.2 deg, b_r +8 m,
s_r 0.004, del 0.3 deg, c_u -12 m + slow and white noise: every parameter inside its interval;
(2) basis / correction geometry vs closed form (north, east, azimuth wrap, exact inverse);
(3) plain-Python recomputation of one real fit; (4) LOIO guard (params for k never include k,
mutating k leaves them unchanged; corrections run with every GT loader disabled and do not move
when the truth is mutated; params_for answers every testing key from its own LOIO row); (5) the
known EGOI / NYTF grid numbers (mean per-iteration median horizontal error 27.1 -> 8.4 m / 25.4 ->
10.1 m, 5 of 5): raw within 1 m, the radar_bias LOIO angle (the angle they came from) through
correct() within RB_TOL_M, and this model's ungated azimuth-only LOIO 5 of 5 better and no worse
than the known number + 1 m; (6) determinism; (7) segment row mapping reproduces tracks.csv n_rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuas_clean as Q  # noqa: E402
import cuas_common as C  # noqa: E402
import truth as T  # noqa: E402


def _import_retry(name: str):
    """sensor_noise / sensor_tracks are being extended by another unit; an import that fails with a
    syntax or name error mid-edit is retried for a few minutes."""
    import importlib
    for attempt in range(10):
        try:
            return importlib.import_module(name)
        except (SyntaxError, NameError, ImportError, IndentationError) as exc:   # pragma: no cover
            print(f"import {name} failed ({exc!r}); retry {attempt + 1}/10 in 30 s", flush=True)
            sys.modules.pop(name, None)
            time.sleep(30)
    return importlib.import_module(name)


SN = _import_retry("sensor_noise")
ST = _import_retry("sensor_tracks")
RR = _import_retry("radar_registration")

TRACKS = C.DERIVED / "sensor_tracks_v2"
OUT = C.DERIVED / "radar_model"
SITES_CSV = C.DERIVED / "sensor_sites.csv"

RADAR_KEYS = ("EGOI_RADAR", "NYTF_RADAR", "UXYJ_RADAR", "IUKL_RADAR", "EUIG_RADAR")
TESTING_ONLY = ("EUIG_RADAR",)
REF_KEYS = ("STUG_CYBER", "ILYS_PASSIVE-RF", "CEQX_CYBER")
H_TERMS = ("daz", "tau", "b_r", "s_r")
V_TERMS = ("c_u", "del")
PARAMS = ("daz_deg", "tau_s", "b_r_m", "s_r", "c_u_m", "del_deg")
TERM_PARAM = {"daz": "daz_deg", "tau": "tau_s", "b_r": "b_r_m", "s_r": "s_r", "c_u": "c_u_m", "del": "del_deg"}

GROSS_M = 200.0          # rows with |residual| above this are left out of the fits (not of the validation)
HUBER_C = 1.345
IRLS_MAX = 50
IRLS_TOL = 1e-10
SEG_CAP = 120            # one segment weighs at most this many samples
N_BOOT = 400
MIN_FIT_N = 50
MIN_FIT_SEGS = 2
MIN_REF_N = 10           # reference latency fits
MIN_CI_BLOCKS = 3
FEW_BLOCKS = 5
VIF_MAX = 10.0
MIN_SPEED_RMS = 1.5      # m/s: weighted rms truth speed below this -> tau not identifiable
MIN_R_SD = 150.0         # m: weighted range sd below this -> s_r / del not identifiable
HOVER_MS = 1.0
HOVER_MIN_R = 200.0
MIN_VAL_N = 50
VEL_WIN_S = 2.0          # differenced velocity: least-squares slope over +- this window of the segment's reports
DECIDE_SHARE = 2.0 / 3.0
SEED = 20261029
SYN = {"tau_s": 1.7, "daz_deg": 1.2, "b_r_m": 8.0, "s_r": 0.004, "del_deg": 0.3, "c_u_m": -12.0}
KNOWN_GRID = {"EGOI": (27.1, 8.4), "NYTF": (25.4, 10.1)}   # calib_stability C5 (skeptic-verified 2026-10-28)
KNOWN_MIN_GRID = 300
KNOWN_TOL_M = 1.0
RB_TOL_M = 0.25          # the known numbers (0.1 m precision) through the radar_bias angle + correct()

_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
         11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
         20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def t975(df: int) -> float:
    """97.5 % Student t quantile (table, linear in 1/df between entries; 1.96 beyond 120)."""
    if df < 1:
        return float("nan")
    if df in _T975:
        return _T975[df]
    ks = sorted(_T975)
    if df > ks[-1]:
        return 1.96
    hi = next(k for k in ks if k > df)
    lo = max(k for k in ks if k < df)
    f = (1 / df - 1 / hi) / (1 / lo - 1 / hi)
    return _T975[hi] + f * (_T975[lo] - _T975[hi])


def log(msg: str) -> None:
    print(msg, flush=True)


def short(key: str) -> str:
    return key.split("/")[-1] if key.startswith("testing") else key.split("/", 1)[1]


def f_(x, nd=1) -> str:
    return "-" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def md_table(header, rows) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows]
    return out


# --------------------------------------------------------------------------- #
# Geometry and the correction (API)
# --------------------------------------------------------------------------- #


def basis(e, n, se, sn):
    """(R, l_e, l_n, t_e, t_n): horizontal range, line-of-sight unit site -> point, and its
    clockwise normal (the direction in which the azimuth grows)."""
    dx, dy = np.asarray(e, float) - se, np.asarray(n, float) - sn
    R = np.hypot(dx, dy)
    with np.errstate(invalid="ignore", divide="ignore"):
        le, ln = dx / R, dy / R
    return R, le, ln, ln, -le


def _p(params: dict, name: str) -> float:
    v = params.get(name, 0.0) if params else 0.0
    return float(v) if v is not None and np.isfinite(v) else 0.0


def correct(sensor_key, e, n, u, t, ve, vn, vu, site_e, site_n, site_u, params: dict):
    """Undo the modelled radar error of measured positions. All inputs are MEASURED: e, n (local ENU,
    m), u (base ellipsoid height, base_height()), t (t_cal; not used by the current model, latency
    acts through the velocity), ve, vn, vu (the sensor's reported velocity or measured_velocity();
    NaN -> no latency correction on that axis), site_e, site_n (the row's sensor site), site_u (not
    used: the vertical term scales with horizontal range). params: dict with any of daz_deg, tau_s,
    b_r_m, s_r, c_u_m, del_deg (missing / NaN = 0), e.g. params_for(). Order: latency (p += tau v),
    then the exact polar inverse (azimuth - daz, range (R - b_r) / (1 + s_r)), then the height
    (u - c_u - del R_corrected). Returns (e, n, u) arrays."""
    if params and params.get("sensor_key") not in (None, "", sensor_key):
        raise ValueError(f"params are for {params.get('sensor_key')}, not {sensor_key}")
    e, n, u = (np.asarray(x, float) for x in (e, n, u))
    tau = _p(params, "tau_s")
    if tau:
        e = e + tau * np.nan_to_num(np.asarray(ve, float))
        n = n + tau * np.nan_to_num(np.asarray(vn, float))
        u = u + tau * np.nan_to_num(np.asarray(vu, float))
    daz, b_r, s_r = math.radians(_p(params, "daz_deg")), _p(params, "b_r_m"), _p(params, "s_r")
    se, sn = np.asarray(site_e, float), np.asarray(site_n, float)
    dx, dy = e - se, n - sn
    R = np.hypot(dx, dy)
    if daz or b_r or s_r:
        az = np.arctan2(dx, dy) - daz
        Rc = np.maximum((R - b_r) / (1.0 + s_r), 0.0)
        e, n, R = se + Rc * np.sin(az), sn + Rc * np.cos(az), Rc
    c_u, dl = _p(params, "c_u_m"), math.radians(_p(params, "del_deg"))
    if c_u or dl:
        u = u - c_u - dl * R
    return e, n, u


def distort(e, n, u, ve, vn, vu, se, sn, params: dict):
    """Forward model (synthetic data and self-check 2): truth -> measurement, the exact inverse of
    correct() with the same params (latency applied last, as a measurement of p - tau v)."""
    daz, b_r, s_r = math.radians(_p(params, "daz_deg")), _p(params, "b_r_m"), _p(params, "s_r")
    dx, dy = np.asarray(e, float) - se, np.asarray(n, float) - sn
    R = np.hypot(dx, dy)
    az = np.arctan2(dx, dy) + daz
    Rm = R * (1.0 + s_r) + b_r
    em, nm = se + Rm * np.sin(az), sn + Rm * np.cos(az)
    um = np.asarray(u, float) + _p(params, "c_u_m") + math.radians(_p(params, "del_deg")) * R
    tau = _p(params, "tau_s")
    return em - tau * ve, nm - tau * vn, um - tau * vu


def measured_velocity(seg, t, x) -> np.ndarray:
    """Velocity of x differenced from a sensor's own reports: least-squares slope over the reports of
    the same segment within +- VEL_WIN_S of each report (NaN with < 2 distinct times). Rows must be
    sorted by (seg, t)."""
    seg, t, x = np.asarray(seg), np.asarray(t, float), np.asarray(x, float)
    out = np.full(t.size, np.nan)
    if t.size == 0:
        return out
    brk = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1], True])
    for a, b in zip(brk[:-1], brk[1:]):
        tt = t[a:b] - t[a]
        xx = x[a:b]
        ok = np.isfinite(xx)
        if ok.sum() < 2:
            continue
        tt_o, xx_o = tt[ok], xx[ok]
        c1 = np.r_[0.0, np.cumsum(np.ones_like(tt_o))]
        ct = np.r_[0.0, np.cumsum(tt_o)]
        ctt = np.r_[0.0, np.cumsum(tt_o * tt_o)]
        cx = np.r_[0.0, np.cumsum(xx_o)]
        ctx = np.r_[0.0, np.cumsum(tt_o * xx_o)]
        lo = np.searchsorted(tt_o, tt - VEL_WIN_S, "left")
        hi = np.searchsorted(tt_o, tt + VEL_WIN_S, "right")
        k = c1[hi] - c1[lo]
        st, stt, sx, stx = ct[hi] - ct[lo], ctt[hi] - ctt[lo], cx[hi] - cx[lo], ctx[hi] - ctx[lo]
        den = k * stt - st * st
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.where(den > 1e-9 * np.maximum(k, 1) ** 2, (k * stx - st * sx) / den, np.nan)
        out[a:b] = v
    return out


def base_height(det: dict, rules: dict | None = None) -> np.ndarray:
    """sensor_noise.u_ellipsoid with the radars' fitted offsets set to 0 (pass the whole table)."""
    rules = SN.load_rules() if rules is None else rules
    r0 = {k: (dict(v, offset=0.0) if k in RADAR_KEYS else v) for k, v in rules.items()}
    return SN.u_ellipsoid(det, r0)


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #


def wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    nrm = np.sqrt((Xw * Xw).sum(0))
    nrm[nrm == 0] = 1.0
    beta = np.linalg.lstsq(Xw / nrm, y * sw, rcond=None)[0]
    return beta / nrm


def irls(X, y, w, rtype, beta0=None):
    """Huber IRLS: weight w * min(1, HUBER_C / |res / s_type|), s_type = 1.4826 MAD per row type."""
    beta = wls(X, y, w) if beta0 is None else beta0
    types = np.unique(rtype)
    scale = {}
    for _ in range(IRLS_MAX):
        res = y - X @ beta
        hw = np.ones_like(y)
        for k in types:
            m = (rtype == k) & (w > 0)
            s = 1.4826 * float(np.median(np.abs(res[m]))) if m.any() else 1.0
            s = max(s, 0.05)
            scale[int(k)] = s
            z = np.abs(res[rtype == k]) / s
            hw[rtype == k] = np.where(z <= HUBER_C, 1.0, HUBER_C / np.maximum(z, 1e-12))
        new = wls(X, y, w * hw)
        if np.all(np.abs(new - beta) <= IRLS_TOL * (1.0 + np.abs(beta))):
            beta = new
            break
        beta = new
    return beta, scale


def vif_corr(X: np.ndarray, w: np.ndarray, names: list[str]) -> tuple[dict, dict]:
    M = (X * w[:, None]).T @ X
    d = np.sqrt(np.diag(M))
    d[d == 0] = 1.0
    Cn = M / np.outer(d, d)
    try:
        inv = np.linalg.inv(Cn)
        vif = {nm: float(inv[i, i]) for i, nm in enumerate(names)}
    except np.linalg.LinAlgError:
        vif = {nm: float("inf") for nm in names}
    corr = {f"{names[i]}~{names[j]}": float(Cn[i, j]) for i in range(len(names)) for j in range(i + 1, len(names))}
    return vif, corr


def seg_weights(seg: np.ndarray) -> np.ndarray:
    _, inv, cnt = np.unique(seg, return_inverse=True, return_counts=True)
    return np.minimum(1.0, SEG_CAP / cnt[inv].astype(float))


def residual_frame(S: dict, m: np.ndarray) -> dict:
    """Residuals and truth velocity projected on the basis at the truth position."""
    R, le, ln, tE, tN = basis(S["te"][m], S["tn"][m], S["se"][m], S["sn"][m])
    dxm, dym = S["e"][m] - S["se"][m], S["n"][m] - S["sn"][m]
    daz_obs = np.radians(C.angle_diff_deg(C.azimuth_deg(dxm, dym), C.azimuth_deg(R * le, R * ln)))
    ve, vn = S["tve"][m], S["tvn"][m]
    # polar residuals: r_t = R x (azimuth error), r_l = range error; to first order these are the
    # projections r . t and r . l, and correct() is their exact inverse (no daz^2 R leakage into s_r)
    return {"R": R, "rt": R * daz_obs, "rl": np.hypot(dxm, dym) - R, "vt": ve * tE + vn * tN,
            "vl": ve * le + vn * ln, "speed": np.hypot(ve, vn), "ru": S["u"][m] - S["tu"][m],
            "tvu": S["tvu"][m], "seg": S["seg"][m]}


def h_design(F: dict, terms, R0: float):
    """Stacked t-rows and l-rows; columns in the order of `terms`; rows whose design is all zero
    (l-rows of a daz-only model) are removed."""
    N = F["R"].size
    cols = []
    for tm in terms:
        if tm == "daz":
            cols.append(np.r_[F["R"], np.zeros(N)])
        elif tm == "tau":
            cols.append(np.r_[-F["vt"], -F["vl"]])
        elif tm == "b_r":
            cols.append(np.r_[np.zeros(N), np.ones(N)])
        elif tm == "s_r":
            cols.append(np.r_[np.zeros(N), F["R"] - R0])
    X = np.column_stack(cols)
    y = np.r_[F["rt"], F["rl"]]
    rtype = np.r_[np.zeros(N, int), np.ones(N, int)]
    sample = np.r_[np.arange(N), np.arange(N)]
    keep = np.any(X != 0, axis=1)
    return X[keep], y[keep], rtype[keep], sample[keep]


def v_design(F: dict, terms, R0: float):
    N = F["R"].size
    cols = [np.ones(N) if tm == "c_u" else F["R"] - R0 for tm in terms]
    return np.column_stack(cols), F["ru"].copy(), np.full(N, 2, int), np.arange(N)


def map_params(axis: str, terms, beta: np.ndarray, R0: float) -> dict:
    b = dict(zip(terms, beta))
    out = {}
    if axis == "h":
        if "daz" in b:
            out["daz_deg"] = math.degrees(b["daz"])
        if "tau" in b:
            out["tau_s"] = float(b["tau"])
        if "s_r" in b:
            out["s_r"] = float(b["s_r"])
        if "b_r" in b:
            out["b_r_m"] = float(b["b_r"] - b.get("s_r", 0.0) * R0)
    else:
        if "del" in b:
            out["del_deg"] = math.degrees(b["del"])
        if "c_u" in b:
            out["c_u_m"] = float(b["c_u"] - b.get("del", 0.0) * R0)
    return out


def fit_model(S: dict, model: str, seed_key: str, boot: bool = True) -> dict:
    """One model on one (sensor, iteration) row set. model = 'h:<terms>' | 'v:<terms>' | 'hover:daz'."""
    axis, spec = model.split(":")
    want = spec.split("+")
    res = {"model": model, "terms": [], "dropped": [], "valid": False, "n_samples": 0, "n_segs": 0,
           "vals": {}, "lo": {}, "hi": {}, "vif": {}, "corr": {}, "scale": {}, "R0": float("nan"),
           "few_blocks": 0, "rms_speed": float("nan"), "r_sd": float("nan"), "corr_ru_vu": float("nan")}
    rh = np.hypot(S["e"] - S["te"], S["n"] - S["tn"])
    if axis in ("h", "hover"):
        m = np.isfinite(rh) & (rh <= GROSS_M)
        if axis == "hover":
            R_t = np.hypot(S["te"] - S["se"], S["tn"] - S["sn"])
            m &= (np.hypot(S["tve"], S["tvn"]) < HOVER_MS) & (R_t >= HOVER_MIN_R)
    else:
        ru = S["u"] - S["tu"]
        m = np.isfinite(ru) & (np.abs(ru) <= GROSS_M) & np.isfinite(S["te"])
    if axis != "v":
        m &= np.isfinite(S["tve"]) & np.isfinite(S["tvn"])
    F = residual_frame(S, m)
    N = F["R"].size
    nseg = np.unique(F["seg"]).size
    res.update(n_samples=int(N), n_segs=int(nseg))
    if N < MIN_FIT_N or nseg < MIN_FIT_SEGS:
        return res
    w_s = seg_weights(F["seg"])
    R0 = float(np.sum(w_s * F["R"]) / np.sum(w_s))
    res["R0"] = R0
    res["rms_speed"] = float(np.sqrt(np.sum(w_s * F["speed"] ** 2) / np.sum(w_s)))
    res["r_sd"] = float(np.sqrt(np.sum(w_s * (F["R"] - R0) ** 2) / np.sum(w_s)))
    design = v_design if axis == "v" else h_design
    terms = [tm for tm in (V_TERMS if axis == "v" else H_TERMS) if tm in want]
    # identifiability (pre-registered)
    dropped = []
    if "tau" in terms and res["rms_speed"] < MIN_SPEED_RMS:
        terms.remove("tau")
        dropped.append("tau(speed)")
    for tm in ("s_r", "del"):
        if tm in terms and res["r_sd"] < MIN_R_SD:
            terms.remove(tm)
            dropped.append(f"{tm}(range sd)")
    while True:
        X, y, rtype, smp = design(F, terms, R0)
        vif, corr = vif_corr(X, w_s[smp], terms)
        bad = [(vif[tm], tm) for tm in terms if tm in ("tau", "s_r", "del") and vif[tm] > VIF_MAX]
        if not bad:
            break
        tm = max(bad)[1]
        terms.remove(tm)
        dropped.append(f"{tm}(vif {max(bad)[0]:.0f})")
    res.update(terms=terms, dropped=dropped, vif=vif, corr=corr)
    beta, scale = irls(X, y, w_s[smp], rtype)
    res["scale"] = scale
    res["vals"] = map_params("v" if axis == "v" else "h", terms, beta, R0)
    res["valid"] = True
    if axis == "v":
        r_after = y - X @ beta
        ok = np.isfinite(F["tvu"])
        if ok.sum() > 10 and np.std(F["tvu"][ok]) > 0:
            res["corr_ru_vu"] = float(np.corrcoef(r_after[ok], F["tvu"][ok])[0, 1])
    if not boot:
        return res
    segs, sinv = np.unique(F["seg"], return_inverse=True)
    B = segs.size
    res["few_blocks"] = int(B < FEW_BLOCKS)
    if B < MIN_CI_BLOCKS:
        return res
    rng = np.random.default_rng([SEED, zlib.crc32((seed_key + "|" + model).encode())])
    names = list(res["vals"])
    reps = np.full((N_BOOT, len(names)), np.nan)
    row_seg = sinv[smp]
    for b in range(N_BOOT):
        mult = np.bincount(rng.integers(0, B, B), minlength=B).astype(float)
        wb = w_s[smp] * mult[row_seg]
        if not np.any(wb > 0):
            continue
        bb, _ = irls(X, y, wb, rtype, beta0=beta.copy())
        mp = map_params("v" if axis == "v" else "h", terms, bb, R0)
        reps[b] = [mp[nm] for nm in names]
    half = t975(B - 1) * math.sqrt(B / (B - 1)) * np.nanstd(reps, axis=0, ddof=1)
    for i, nm in enumerate(names):
        res["lo"][nm], res["hi"][nm] = res["vals"][nm] - half[i], res["vals"][nm] + half[i]
    return res


def fit_ref_latency(S: dict, seed_key: str, boot: bool = True) -> dict:
    """Reference sensor: r_e = b_e - tau ve, r_n = b_n - tau vn (truth velocity), Huber IRLS."""
    out = {"valid": False, "tau": float("nan"), "lo": float("nan"), "hi": float("nan"), "b": (np.nan, np.nan),
           "n_samples": 0, "n_segs": 0, "few_blocks": 0, "rms_speed": float("nan")}
    re, rn = S["e"] - S["te"], S["n"] - S["tn"]
    m = np.isfinite(re) & np.isfinite(rn) & (np.hypot(re, rn) <= GROSS_M) & np.isfinite(S["tve"]) & np.isfinite(S["tvn"])
    N = int(m.sum())
    seg = S["seg"][m]
    out.update(n_samples=N, n_segs=int(np.unique(seg).size))
    if N < MIN_REF_N:
        return out
    w_s = seg_weights(seg)
    ve, vn = S["tve"][m], S["tvn"][m]
    out["rms_speed"] = float(np.sqrt(np.sum(w_s * (ve ** 2 + vn ** 2)) / np.sum(w_s)))
    if out["rms_speed"] < MIN_SPEED_RMS:
        return out
    z = np.zeros(N)
    X = np.r_[np.column_stack([np.ones(N), z, -ve]), np.column_stack([z, np.ones(N), -vn])]
    y = np.r_[re[m], rn[m]]
    rtype = np.zeros(2 * N, int)
    w = np.r_[w_s, w_s]
    beta, _ = irls(X, y, w, rtype)
    out.update(valid=True, tau=float(beta[2]), b=(float(beta[0]), float(beta[1])))
    segs, sinv = np.unique(seg, return_inverse=True)
    B = segs.size
    out["few_blocks"] = int(B < FEW_BLOCKS)
    if boot and B >= MIN_CI_BLOCKS:
        rng = np.random.default_rng([SEED, zlib.crc32((seed_key + "|ref").encode())])
        rs = np.r_[sinv, sinv]
        reps = []
        for _ in range(N_BOOT):
            mult = np.bincount(rng.integers(0, B, B), minlength=B).astype(float)
            wb = w * mult[rs]
            reps.append(irls(X, y, wb, rtype, beta0=beta.copy())[0][2])
        half = t975(B - 1) * math.sqrt(B / (B - 1)) * float(np.std(reps, ddof=1))
        out["lo"], out["hi"] = out["tau"] - half, out["tau"] + half
    return out


# --------------------------------------------------------------------------- #
# Pooling / LOIO
# --------------------------------------------------------------------------- #


def pool(fits: dict[str, dict], exclude: str | None = None) -> dict:
    """{param: dict(mean, n, iters, sd, ci_lo, ci_hi, pi_lo, pi_hi)} over the valid fits of `fits`
    ({iter_key: fit}) without iteration `exclude`."""
    out = {}
    names = sorted({nm for f in fits.values() if f.get("valid") for nm in f["vals"]})
    for nm in names:
        its = sorted(k for k, f in fits.items() if k != exclude and f.get("valid") and nm in f["vals"]
                     and np.isfinite(f["vals"][nm]))
        v = np.array([fits[k]["vals"][nm] for k in its])
        if v.size == 0:
            continue
        mean = float(v.mean())
        d = {"mean": mean, "n": int(v.size), "iters": its, "sd": float("nan"), "ci_lo": float("nan"),
             "ci_hi": float("nan"), "pi_lo": float("nan"), "pi_hi": float("nan")}
        if v.size >= 2:
            sd = float(v.std(ddof=1))
            tq = t975(v.size - 1)
            d.update(sd=sd, ci_lo=mean - tq * sd / math.sqrt(v.size), ci_hi=mean + tq * sd / math.sqrt(v.size),
                     pi_lo=mean - tq * sd * math.sqrt(1 + 1 / v.size), pi_hi=mean + tq * sd * math.sqrt(1 + 1 / v.size))
        out[nm] = d
    return out


def pooled_params(fits: dict[str, dict], exclude: str | None) -> dict:
    return {nm: d["mean"] for nm, d in pool(fits, exclude).items()}


# --------------------------------------------------------------------------- #
# Loading (testing rows of drone_air segments + truth)
# --------------------------------------------------------------------------- #


def load_labels_full(it: C.Iteration) -> dict:
    return C.read_columns(TRACKS / f"{C.iter_slug(it.key)}__tracks.csv",
                          numeric={"seg_id", "t0", "t1", "n_rows"},
                          usecols=["seg_id", "iter_key", "sensor_key", "object_id", "t0", "t1", "n_rows", "label",
                                   "best_flight", "gt_feed"])


def load_rows(testing: list[C.Iteration], keys) -> tuple[dict, dict, list]:
    """{(sensor_key, iter_key): S} with the raw rows of every drone_air segment and the flight's truth
    at t_cal; also the full cleaned table's base heights (for the API check) and the mapping log."""
    det = Q.load_clean(["alt_datum", "sensor_alt_datum", "sensor_u"])
    rules = SN.load_rules()
    h_base = base_height(det, rules)
    h_sn = SN.u_ellipsoid(det, rules)
    toff = ST.time_offsets()
    pm = ST.position_mask(det)
    gt = Q.load_groundtruth()
    out, maplog = {}, []
    for it in testing:
        key = it.key
        m = (det["iter_key"] == key) & pm & np.isin(det["sensor_key"], list(keys))
        idx = np.flatnonzero(m)
        fr = C.iteration_frame(it)
        u_b = np.asarray(fr.to_enu(det["lat"][idx], det["lon"][idx], h_base[idx])[2], float)
        u_s = np.asarray(fr.to_enu(det["lat"][idx], det["lon"][idx], h_sn[idx])[2], float)
        sk_ = det["sensor_key"][idx]
        t_cal = det["t"][idx].copy()
        for sk, off in toff.items():
            t_cal[sk_ == sk] += off
        groups: dict[tuple, np.ndarray] = {}
        ident = np.array([f"{a}\x1f{b}" for a, b in zip(sk_, det["object_id"][idx])], dtype=object)
        order = np.argsort(ident, kind="stable")
        uid, first = np.unique(ident[order], return_index=True)
        bounds = np.r_[first, order.size]
        for j, u_ in enumerate(uid):
            groups[u_] = order[bounds[j]:bounds[j + 1]]
        lab = load_labels_full(it)
        fl_mask = {}
        per: dict[str, list] = {}
        for s in range(lab["seg_id"].size):
            sk = lab["sensor_key"][s]
            if sk not in keys or lab["label"][s] != "drone_air" or not lab["best_flight"][s]:
                continue
            g = groups.get(f"{sk}\x1f{lab['object_id'][s]}")
            if g is None:
                maplog.append((key, sk, int(lab["seg_id"][s]), int(lab["n_rows"][s]), 0))
                continue
            sel = g[(t_cal[g] >= lab["t0"][s] - 1e-4) & (t_cal[g] <= lab["t1"][s] + 1e-4)]
            maplog.append((key, sk, int(lab["seg_id"][s]), int(lab["n_rows"][s]), int(sel.size)))
            sel = sel[np.lexsort((np.arange(sel.size), t_cal[sel]))]
            fk = lab["best_flight"][s]
            if fk not in fl_mask:
                fl_mask[fk] = (gt["iter_key"] == key) & (gt["flight_key"] == fk)
            tr = Q.interp_flight(gt, fl_mask[fk], t_cal[sel], cols=("e", "n", "u", "ve", "vn", "vu"))
            r = idx[sel]
            seg = np.full(sel.size, int(lab["seg_id"][s]), np.int64)
            ve_r, vn_r, vu_r = det["ve_c"][r], det["vn_c"][r], det["vu_c"][r]
            ve_d = measured_velocity(seg, t_cal[sel], det["e"][r])
            vn_d = measured_velocity(seg, t_cal[sel], det["n"][r])
            vu_d = measured_velocity(seg, t_cal[sel], u_b[sel])
            per.setdefault(sk, []).append({
                "seg": seg, "t": t_cal[sel], "e": det["e"][r], "n": det["n"][r], "u": u_b[sel], "u_sn": u_s[sel],
                "ve": np.where(np.isfinite(ve_r), ve_r, ve_d), "vn": np.where(np.isfinite(vn_r), vn_r, vn_d),
                "vu": np.where(np.isfinite(vu_r), vu_r, vu_d), "v_reported": np.isfinite(ve_r),
                "se": det["sensor_e"][r], "sn": det["sensor_n"][r], "su": det["sensor_u"][r],
                "te": tr[0], "tn": tr[1], "tu": tr[2], "tve": tr[3], "tvn": tr[4], "tvu": tr[5],
                "flight": np.full(sel.size, fk, dtype=object)})
        for sk, parts in per.items():
            S = {c: np.concatenate([p[c] for p in parts]) for c in parts[0]}
            ok = np.isfinite(S["te"]) & np.isfinite(S["tn"]) & np.isfinite(S["se"]) & np.isfinite(S["sn"])
            out[(sk, key)] = {c: v[ok] for c, v in S.items()}
    return out, {"det_n": int(det["t"].size)}, maplog


# --------------------------------------------------------------------------- #
# Validation metrics
# --------------------------------------------------------------------------- #


def err_stats(e, n, u, S: dict) -> dict:
    eh = np.hypot(e - S["te"], n - S["tn"])
    ev = np.abs(u - S["tu"])
    okv = np.isfinite(ev)
    e3 = np.sqrt(eh[okv] ** 2 + ev[okv] ** 2)
    nan = float("nan")
    return {"n": int(eh.size), "n_u": int(okv.sum()),
            "med_h_m": float(np.median(eh)) if eh.size else nan,
            "med_v_m": float(np.median(ev[okv])) if okv.any() else nan,
            "med_3d_m": float(np.median(e3)) if e3.size else nan,
            **{f"h_le{c}": float(np.mean(eh <= c)) if eh.size else nan for c in (10, 25, 50)},
            **{f"d3_le{c}": float(np.mean(e3 <= c)) if e3.size else nan for c in (10, 25, 50)}}


def apply(S: dict, sk: str, params: dict):
    return correct(sk, S["e"], S["n"], S["u"], S["t"], S["ve"], S["vn"], S["vu"], S["se"], S["sn"], S["su"],
                   params)


def model_name(axis: str, terms) -> str:
    order = H_TERMS if axis == "h" else V_TERMS
    return f"{axis}:" + "+".join(tm for tm in order if tm in terms)


H_FULL = model_name("h", H_TERMS)
V_FULL = model_name("v", V_TERMS)


class Fits:
    """Memoised per-(sensor, iteration, model) fits."""

    def __init__(self, rows: dict, boot: bool = True):
        self.rows, self.boot, self.memo = rows, boot, {}

    def get(self, sk: str, ik: str, model: str) -> dict:
        k = (sk, ik, model)
        if k not in self.memo:
            self.memo[k] = fit_model(self.rows[(sk, ik)], model, f"{sk}|{ik}", boot=self.boot)
        return self.memo[k]

    def by_iter(self, sk: str, model: str, iters) -> dict[str, dict]:
        return {ik: self.get(sk, ik, model) for ik in iters}


def decide(sk: str, iters: list[str], fits: Fits, val_cache: dict) -> tuple[list[dict], set, set]:
    """Sequential pre-registered decisions for one radar. Returns (rows, accepted h terms, accepted v terms)."""
    rows = []
    rows_by = {ik: fits.rows[(sk, ik)] for ik in iters}
    elig = [ik for ik in iters if rows_by[ik]["e"].size >= MIN_VAL_N]

    def med(model_h: str | None, model_v: str | None, ik: str, axis: str) -> float:
        key = (sk, tuple(iters), ik, model_h, model_v)        # the LOIO pool depends on `iters` (nested calls)
        if key not in val_cache:
            p = {}
            if model_h:
                p.update(pooled_params(fits.by_iter(sk, model_h, iters), ik))
            if model_v:
                p.update(pooled_params(fits.by_iter(sk, model_v, iters), ik))
            S = rows_by[ik]
            e, n, u = apply(S, sk, p)
            val_cache[key] = err_stats(e, n, u, S)
        return val_cache[key]["med_h_m" if axis == "h" else "med_v_m"]

    for axis, order in (("h", H_TERMS), ("v", V_TERMS)):
        A: set = set()
        for tm in order:
            add = {tm} | ({"b_r"} if tm == "s_r" else set()) | ({"c_u"} if tm == "del" else set())
            with_ = A | add
            m_with = model_name(axis, with_)
            m_base = model_name(axis, A) if A else None
            have = [ik for ik in elig if pooled_params(fits.by_iter(sk, m_with, iters), ik).get(TERM_PARAM[tm]) is not None]
            imp = []
            for ik in have:
                a = med(m_with if axis == "h" else None, m_with if axis == "v" else None, ik, axis)
                b = med(m_base if axis == "h" else None, m_base if axis == "v" else None, ik, axis)
                if np.isfinite(a) and np.isfinite(b) and a < b:
                    imp.append(ik)
            need = math.ceil(DECIDE_SHARE * len(have)) if have else 0
            pl = pool(fits.by_iter(sk, m_with, iters)).get(TERM_PARAM[tm], {})
            lo, hi = pl.get("ci_lo", float("nan")), pl.get("ci_hi", float("nan"))
            excl = bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))
            ok = len(have) >= 2 and len(imp) >= need and excl
            rows.append({"sensor_key": sk, "axis": axis, "term": tm, "baseline": m_base or "raw", "with_terms": m_with,
                         "n_iters": len(have), "n_improved": len(imp), "need": need,
                         "improved_iters": "|".join(short(i) for i in imp), "pooled_value": pl.get("mean", float("nan")),
                         "ci_lo": lo, "ci_hi": hi, "ci_excludes_0": int(excl), "n_ci_iters": pl.get("n", 0),
                         "decision": "apply" if ok else "reject", "testing_only": int(sk in TESTING_ONLY)})
            if ok:
                A = with_
        if axis == "h":
            acc_h = set(A)
        else:
            acc_v = set(A)
    # Non-regression of the accepted vertical set against the height deployed today (sensor_noise
    # offset, in-sample, so the check is conservative); same count rule as the primary condition.
    # Added after review (2026-10-29), not part of the pre-registered primary rule.
    gate = ""
    if acc_v:
        mv = model_name("v", acc_v)
        have = [ik for ik in elig if pooled_params(fits.by_iter(sk, mv, iters), ik)]
        wins = [ik for ik in have
                if med(None, mv, ik, "v") < err_stats(rows_by[ik]["e"], rows_by[ik]["n"], rows_by[ik]["u_sn"],
                                                       rows_by[ik])["med_v_m"]]
        need = math.ceil(DECIDE_SHARE * len(have)) if have else 0
        passed = bool(have) and len(wins) >= need
        gate = f"{len(wins)}/{len(have)} need {need}: {'pass' if passed else 'FAIL'}"
        if not passed:
            acc_v = set()
    final_vals = {}
    for axis, acc in (("h", acc_h), ("v", acc_v)):
        if acc:
            final_vals.update(pooled_params(fits.by_iter(sk, model_name(axis, acc), iters), None))
    for r in rows:
        inf = r["term"] in (acc_h if r["axis"] == "h" else acc_v)
        r["vs_deployed"] = gate if r["axis"] == "v" else ""
        r["final"] = "apply" if inf else "reject"
        r["final_value"] = final_vals.get(TERM_PARAM[r["term"]], float("nan")) if inf else float("nan")
    return rows, acc_h, acc_v


# --------------------------------------------------------------------------- #
# Params rows
# --------------------------------------------------------------------------- #

PARAM_ORDER = (["sensor_key", "iter_key", "kind", "model"]
               + [c for p in PARAMS for c in (p, p + "_lo", p + "_hi")]
               + ["pi_lo_daz", "pi_hi_daz", "inside", "gt_daz_deg", "n_samples", "n_segs", "n_samples_u", "few_blocks",
                  "terms", "dropped", "R0_m", "scale_t_m", "scale_l_m", "scale_u_m", "vif", "corr", "rms_speed_ms",
                  "r_sd_m", "corr_ru_vu", "n_iter", "iters", "testing_only", "n_pairs", "n_groups"])


def blank_row(sk, ik, kind, model) -> dict:
    r = {c: float("nan") for c in PARAM_ORDER}
    r.update(sensor_key=sk, iter_key=ik, kind=kind, model=model, inside="", terms="", dropped="", vif="", corr="",
             iters="", testing_only=int(sk in TESTING_ONLY), n_iter=0)
    return r


def fit_row(sk, ik, f: dict) -> dict:
    r = blank_row(sk, ik, "fit", f["model"])
    for nm, v in f["vals"].items():
        r[nm] = v
        r[nm + "_lo"] = f["lo"].get(nm, float("nan"))
        r[nm + "_hi"] = f["hi"].get(nm, float("nan"))
    axis = f["model"].split(":")[0]
    r.update(n_samples=f["n_samples"], n_segs=f["n_segs"], few_blocks=f["few_blocks"], terms="|".join(f["terms"]),
             dropped="|".join(f["dropped"]), R0_m=f["R0"], rms_speed_ms=f["rms_speed"], r_sd_m=f["r_sd"],
             corr_ru_vu=f["corr_ru_vu"], vif=";".join(f"{k}:{v:.2f}" for k, v in f["vif"].items()),
             corr=";".join(f"{k}:{v:.3f}" for k, v in f["corr"].items()), n_iter=1, iters=ik if f["valid"] else "")
    if axis == "v":
        r["n_samples_u"] = f["n_samples"]
        r["scale_u_m"] = f["scale"].get(2, float("nan"))
    else:
        r["scale_t_m"] = f["scale"].get(0, float("nan"))
        r["scale_l_m"] = f["scale"].get(1, float("nan"))
    return r


def pool_row(sk, ik, kind, model, pl: dict) -> dict:
    r = blank_row(sk, ik, kind, model)
    its = set()
    for nm, d in pl.items():
        r[nm], r[nm + "_lo"], r[nm + "_hi"] = d["mean"], d["ci_lo"], d["ci_hi"]
        its |= set(d["iters"])
        if nm == "daz_deg":
            r["pi_lo_daz"], r["pi_hi_daz"] = d["pi_lo"], d["pi_hi"]
    r["iters"] = "|".join(sorted(its))
    r["n_iter"] = len(its)
    return r


def read_params(path: Path | None = None) -> list[dict]:
    path = OUT / "params.csv" if path is None else path
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


_PARAMS_CACHE: dict = {}


def params_for(sensor_key: str, held_out_iter: str | None = None, terms: str = "decided") -> dict:
    """Correction parameters of one radar (sensor_key 'EGOI_RADAR' or id 'EGOI'). held_out_iter = a
    testing iteration key -> the LOIO values without it; None or an evaluation key -> the values
    pooled over all testing iterations. terms='decided': only the terms of the final accepted set
    (decisions.csv column 'final', which includes bundled terms and the vertical non-regression
    gate; others 0); terms='full': every term of h full + v full. Returns {param: value,
    'sensor_key', 'source', 'u_frame'}; u_frame 'base' = c_u / del are set, so correct() needs the
    base height (base_height()); 'any' = correct() leaves u unchanged, so pass the deployed
    (sensor_noise) height. A testing iteration where the radar has no drone_air rows still has its
    own decided_loio / loio rows (= the pool of the other iterations). Unknown radar -> all zeros
    (no correction, source 'none')."""
    if "params" not in _PARAMS_CACHE:
        _PARAMS_CACHE["params"] = read_params()
    rows = _PARAMS_CACHE["params"]
    sk = sensor_key if "_" in sensor_key else next((k for k in RADAR_KEYS if k.startswith(sensor_key + "_")), sensor_key)
    loio = held_out_iter is not None and held_out_iter.startswith("testing/")
    out = {p: 0.0 for p in PARAMS}
    out["sensor_key"], out["source"] = sk, "none"
    if terms == "decided":
        want = [("decided_loio", held_out_iter, "decided")] if loio else [("decided", "*", "decided")]
    elif terms == "full":
        kind = "loio" if loio else "pooled"
        ik = held_out_iter if loio else "*"
        want = [(kind, ik, H_FULL), (kind, ik, V_FULL)]
    else:
        raise ValueError("terms must be 'decided' or 'full'")
    for kind, ik, model in want:
        for r in rows:
            if r["sensor_key"] == sk and r["kind"] == kind and r["iter_key"] == ik and r["model"] == model:
                for p in PARAMS:
                    if r[p] not in ("", None):
                        out[p] = float(r[p])
                out["source"] = f"{kind}:{ik}"
    out["u_frame"] = "base" if (out["c_u_m"] or out["del_deg"]) else "any"
    return out


# --------------------------------------------------------------------------- #
# Step 5: GT-free azimuth on every iteration
# --------------------------------------------------------------------------- #


def gt_free_az(its: list[C.Iteration], ref_tau: dict, sites: dict) -> dict:
    """{(iter_key, radar_id, variant): estimate_az dict}; variant in noutm / all (+ '_nolat')."""
    out = {}
    tau_id = {k.split("_")[0]: k for k in REF_KEYS}
    for it in its:
        with RR.gt_forbidden():
            data = RR.load_iteration(it)
            lat = {}
            for sid, sk in tau_id.items():
                if sid in data:
                    tau = ref_tau.get((sk, it.key), 0.0)
                    d = data[sid]
                    lat[sid] = {**d, "e": d["e"] + tau * d["ve"], "n": d["n"] + tau * d["vn"]}
            data_lat = {**data, **lat}
            for rid in RR.RADARS:
                if rid not in data:
                    continue
                site = sites.get((it.key, rid))
                for refset, ids in RR.REF_SETS.items():
                    for suffix, dd in (("", data_lat), ("_nolat", data)):
                        ref = RR.ref_samples(dd, ids)
                        out[(it.key, rid, refset + suffix)] = RR.estimate_az(ref, dd[rid], site,
                                                                            f"{it.key}|{rid}|{refset}{suffix}|rm")
    return out


# --------------------------------------------------------------------------- #
# Self-checks
# --------------------------------------------------------------------------- #


def check_synthetic(testing: list[C.Iteration]) -> tuple[bool, str]:
    """(1) Synthetic radar from real truth tracks (every testing flight, airborne records, 60-s
    segments) about each iteration's EGOI (else NYTF) site; injected SYN params + per-segment AR(1)
    slow noise (sd 4 m horizontal / 6 m vertical, rho 0.9 per record) + white noise (2 / 3 m)."""
    gt = Q.load_groundtruth()
    sites = RR.radar_sites()
    rng = np.random.default_rng([SEED, 1])
    parts = []
    seg_base = 0
    for it in testing:
        site = sites.get((it.key, "EGOI")) or sites.get((it.key, "NYTF"))
        if site is None:
            continue
        mi = gt["iter_key"] == it.key
        for fk in sorted(set(gt["flight_key"][mi].tolist())):
            fm = mi & (gt["flight_key"] == fk)
            tg = np.sort(gt["t"][fm])
            air = Q.interp_flight(gt, fm, tg, cols=("height",))[0] > 5.0
            tg = tg[air]
            if tg.size < 60:
                continue
            te, tn, tu, tve, tvn, tvu = Q.interp_flight(gt, fm, tg, cols=("e", "n", "u", "ve", "vn", "vu"))
            lag = Q.interp_flight(gt, fm, tg - SYN["tau_s"], cols=("e", "n", "u"))
            R = np.hypot(te - site[0], tn - site[1])
            ok = np.isfinite(lag[0]) & np.isfinite(tve) & np.isfinite(tvn) & np.isfinite(tvu) & (R >= 200)
            if ok.sum() < 60:
                continue
            tg, te, tn, tu, tve, tvn, tvu = (x[ok] for x in (tg, te, tn, tu, tve, tvn, tvu))
            le_, ln_, lu_ = (x[ok] for x in lag)
            # measurement of the lagged truth through the polar distortion (no latency term inside distort)
            p0 = {k: v for k, v in SYN.items() if k != "tau_s"}
            em, nm, um = distort(le_, ln_, lu_, 0, 0, 0, site[0], site[1], p0)
            seg = seg_base + ((tg - tg[0]) // 60).astype(np.int64)
            seg_base = int(seg.max()) + 1
            slow = np.zeros((tg.size, 3))
            x = np.zeros(3)
            prev = None
            for i in range(tg.size):
                if prev is None or seg[i] != prev:
                    x = rng.standard_normal(3) * (4.0, 4.0, 6.0)
                else:
                    x = 0.9 * x + math.sqrt(1 - 0.81) * rng.standard_normal(3) * (4.0, 4.0, 6.0)
                slow[i] = x
                prev = seg[i]
            white = rng.standard_normal((tg.size, 3)) * (2.0, 2.0, 3.0)
            parts.append({"seg": seg, "t": tg, "e": em + slow[:, 0] + white[:, 0], "n": nm + slow[:, 1] + white[:, 1],
                          "u": um + slow[:, 2] + white[:, 2], "se": np.full(tg.size, site[0]),
                          "sn": np.full(tg.size, site[1]), "su": np.zeros(tg.size), "te": te, "tn": tn, "tu": tu,
                          "tve": tve, "tvn": tvn, "tvu": tvu})
    S = {c: np.concatenate([p[c] for p in parts]) for c in parts[0]}
    fh = fit_model(S, H_FULL, "synthetic")
    fv = fit_model(S, V_FULL, "synthetic")
    msgs, ok = [], fh["valid"] and fv["valid"]
    for f in (fh, fv):
        for nm, v in f["vals"].items():
            lo, hi = f["lo"].get(nm, np.nan), f["hi"].get(nm, np.nan)
            inside = bool(np.isfinite(lo) and lo <= SYN[nm] <= hi)
            ok &= inside
            msgs.append(f"{nm} {v:.4g} [{lo:.4g}, {hi:.4g}] vs {SYN[nm]:g} {'ok' if inside else 'MISS'}")
    missing = [nm for nm in SYN if nm not in fh["vals"] and nm not in fv["vals"]]
    ok &= not missing
    return ok, (f"{S['e'].size} samples, {np.unique(S['seg']).size} segments; " + "; ".join(msgs)
                + (f"; not identified: {missing}" if missing else "") + f"; dropped {fh['dropped'] + fv['dropped']}")


def check_geometry() -> tuple[bool, str]:
    """(2) basis vs closed form, correction wrap and exact inverse of distort()."""
    bad = []
    R, le, ln, tE, tN = basis(np.array([0.0, 1000.0, -500.0]), np.array([1000.0, 0.0, 0.0]), 0.0, 0.0)
    if not np.allclose([le[0], ln[0], tE[0], tN[0], R[0]], [0, 1, 1, 0, 1000]):
        bad.append("north")
    if not np.allclose([le[1], ln[1], tE[1], tN[1]], [1, 0, 0, -1]):
        bad.append("east")
    if not np.allclose([le[2], ln[2], tE[2], tN[2]], [-1, 0, 0, 1]):
        bad.append("west")
    rng = np.random.default_rng([SEED, 2])
    az = rng.uniform(0, 2 * np.pi, 200)
    Rr = rng.uniform(100, 5000, 200)
    e, n = 30 + Rr * np.sin(az), -40 + Rr * np.cos(az)
    _, le, ln, tE, tN = basis(e, n, 30.0, -40.0)
    h = 1e-7
    de_ = (Rr * np.sin(az + h) - Rr * np.sin(az)) / h / Rr
    dn_ = (Rr * np.cos(az + h) - Rr * np.cos(az)) / h / Rr
    if not (np.allclose(tE, de_, atol=1e-5) and np.allclose(tN, dn_, atol=1e-5)):
        bad.append("t = d(unit)/d(az)")
    # wrap: a point at azimuth 0.5 deg rotated back by +1 deg lands at 359.5 deg, range kept
    pe, pn = 1000 * math.sin(math.radians(0.5)), 1000 * math.cos(math.radians(0.5))
    ce, cn, _ = correct("X", [pe], [pn], [0.0], [0.0], [0.0], [0.0], [0.0], 0.0, 0.0, 0.0, {"daz_deg": 1.0})
    a = float(C.azimuth_deg(ce, cn)[0])
    if abs(a - 359.5) > 1e-9 or abs(math.hypot(ce[0], cn[0]) - 1000) > 1e-9:
        bad.append(f"wrap {a}")
    if abs(float(C.angle_diff_deg(359.9, 0.1)) + 0.2) > 1e-9:
        bad.append("angle_diff wrap")
    # exact inverse with every term
    p = dict(SYN)
    u = rng.uniform(50, 300, 200)
    ve, vn, vu = rng.normal(0, 8, 200), rng.normal(0, 8, 200), rng.normal(0, 2, 200)
    em, nm, um = distort(e, n, u, ve, vn, vu, 30.0, -40.0, p)
    ce, cn, cu = correct("X", em, nm, um, 0, ve, vn, vu, 30.0, -40.0, 0.0, p)
    err = max(np.max(np.abs(ce - e)), np.max(np.abs(cn - n)), np.max(np.abs(cu - u)))
    # forward model first order = the linear model at the truth position
    p1 = {"daz_deg": 0.01, "b_r_m": 0.5, "s_r": 1e-5}
    em1, nm1, _ = distort(e, n, u, 0, 0, 0, 30.0, -40.0, p1)
    lin_e = e + math.radians(0.01) * Rr * tE + (0.5 + 1e-5 * Rr) * le
    lin_n = n + math.radians(0.01) * Rr * tN + (0.5 + 1e-5 * Rr) * ln
    lin_err = max(np.max(np.abs(em1 - lin_e)), np.max(np.abs(nm1 - lin_n)))
    if err > 1e-6:
        bad.append(f"inverse {err:.2e}")
    if lin_err > 1e-3:
        bad.append(f"linear {lin_err:.2e}")
    return not bad, (f"north / east / west closed form, t = d(unit)/d(az) on 200 random points, wrap 0.5 -> 359.5 deg, "
                     f"correct(distort(p)) = p max err {err:.1e} m, first-order model err {lin_err:.1e} m; "
                     f"failures {bad or 'none'}")


def plain_fit(S: dict) -> list[float]:
    """(3) Independent plain-Python version of fit_model(S, H_FULL) without bootstrap (no numpy in the
    arithmetic): basis, gross exclusion, segment cap, centred range, Huber IRLS by normal equations."""
    rows = []
    for i in range(S["e"].size):
        te, tn, se, sn = float(S["te"][i]), float(S["tn"][i]), float(S["se"][i]), float(S["sn"][i])
        re, rn = float(S["e"][i]) - te, float(S["n"][i]) - tn
        ve, vn = float(S["tve"][i]), float(S["tvn"][i])
        if not (math.isfinite(re) and math.isfinite(rn) and math.isfinite(ve) and math.isfinite(vn)):
            continue
        if math.hypot(re, rn) > GROSS_M:
            continue
        a = math.atan2(te - se, tn - sn)
        R = math.hypot(te - se, tn - sn)
        l = (math.sin(a), math.cos(a))
        t = (math.cos(a), -math.sin(a))
        em, nm = float(S["e"][i]) - se, float(S["n"][i]) - sn
        dz = (math.atan2(em, nm) - a + math.pi) % (2 * math.pi) - math.pi
        rows.append((int(S["seg"][i]), R, R * dz, math.hypot(em, nm) - R, ve * t[0] + vn * t[1],
                     ve * l[0] + vn * l[1]))
    cnt: dict = {}
    for r in rows:
        cnt[r[0]] = cnt.get(r[0], 0) + 1
    w = [min(1.0, SEG_CAP / cnt[r[0]]) for r in rows]
    R0 = sum(wi * r[1] for wi, r in zip(w, rows)) / sum(w)
    X, y, wt, ty = [], [], [], []
    for wi, (_, R, rt, rl, vt, vl) in zip(w, rows):
        X.append([R, -vt, 0.0, 0.0]); y.append(rt); wt.append(wi); ty.append(0)
        X.append([0.0, -vl, 1.0, R - R0]); y.append(rl); wt.append(wi); ty.append(1)

    def solve(ww):
        p = 4
        A = [[sum(ww[k] * X[k][i] * X[k][j] for k in range(len(X))) for j in range(p)] for i in range(p)]
        b = [sum(ww[k] * X[k][i] * y[k] for k in range(len(X))) for i in range(p)]
        M = [A[i] + [b[i]] for i in range(p)]
        for c in range(p):
            piv = max(range(c, p), key=lambda r_: abs(M[r_][c]))
            M[c], M[piv] = M[piv], M[c]
            for r_ in range(p):
                if r_ != c:
                    f = M[r_][c] / M[c][c]
                    M[r_] = [a_ - f * b_ for a_, b_ in zip(M[r_], M[c])]
        return [M[i][p] / M[i][i] for i in range(p)]

    def median(v):
        v = sorted(v)
        k = len(v)
        return v[k // 2] if k % 2 else 0.5 * (v[k // 2 - 1] + v[k // 2])

    beta = solve(wt)
    for _ in range(IRLS_MAX):
        res = [y[k] - sum(X[k][j] * beta[j] for j in range(4)) for k in range(len(X))]
        s = {tt: max(1.4826 * median([abs(res[k]) for k in range(len(X)) if ty[k] == tt]), 0.05) for tt in (0, 1)}
        hw = [1.0 if abs(res[k]) / s[ty[k]] <= HUBER_C else HUBER_C / (abs(res[k]) / s[ty[k]]) for k in range(len(X))]
        new = solve([wt[k] * hw[k] for k in range(len(X))])
        done = all(abs(a_ - b_) <= IRLS_TOL * (1 + abs(b_)) for a_, b_ in zip(new, beta))
        beta = new
        if done:
            break
    return [math.degrees(beta[0]), beta[1], beta[2] - beta[3] * R0, beta[3]]


# --------------------------------------------------------------------------- #
# Check 5: known grid numbers
# --------------------------------------------------------------------------- #


def site_csv() -> dict[tuple[str, str], tuple[float, float]]:
    out = {}
    with SITES_CSV.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not r["site_e"]:
                continue
            for sk in r["sensor_keys"].split("|"):
                out[(r["iter_key"], sk)] = (float(r["site_e"]), float(r["site_n"]))
    return out


def grid_known(testing, daz_loio: dict, rb_vals: dict) -> dict:
    """Mean per-iteration median horizontal error of EGOI / NYTF drone_air grid samples (airborne truth),
    iterations with >= KNOWN_MIN_GRID samples: raw, rotated by this model's LOIO daz (h:daz) and by
    the LOIO radar_bias angle of sensor_calibration.csv pooled over the same qualifying iterations
    (the original protocol), via correct()."""
    sites = site_csv()
    res = {}
    for rid in KNOWN_GRID:
        sk = f"{rid}_RADAR"
        per = []
        samples = {}
        for it in testing:
            data = RR.load_iteration(it)
            if rid not in data or (it.key, sk) not in sites:
                continue
            labels = RR.load_labels(it)
            ti = RR.TruthIndex(T.load_truth(it, "airborne"))
            s = RR.gt_samples(data[rid], labels, ti)
            if s["e"].size >= KNOWN_MIN_GRID:
                samples[it.key] = s
        vals = rb_vals.get(sk, {})
        rb_loio = {}
        for k in samples:
            use = [v for j, v in vals.items() if j != k and j in samples]
            if use:
                rb_loio[(sk, k)] = -float(np.mean(use))      # radar_bias is the angle to ADD: daz = -radar_bias
        for it in testing:
            if it.key not in samples:
                continue
            s = samples[it.key]
            se, sn = sites[(it.key, sk)]
            raw = float(np.median(np.hypot(s["e"] - s["te"], s["n"] - s["tn"])))
            out = [raw]
            for ang in (daz_loio.get((sk, it.key)), rb_loio.get((sk, it.key))):
                if ang is None:
                    out.append(float("nan"))
                    continue
                ce, cn, _ = correct(sk, s["e"], s["n"], s["u"], 0, 0, 0, 0, se, sn, 0, {"daz_deg": ang})
                out.append(float(np.median(np.hypot(ce - s["te"], cn - s["tn"]))))
            per.append((it.key, s["e"].size, *out))
        res[rid] = per
    return res


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-checks", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    its = C.iterations()
    testing = [it for it in its if it.has_groundtruth]
    tkeys = [it.key for it in testing]
    OUT.mkdir(parents=True, exist_ok=True)

    log("loading drone_air rows + truth (testing) ...")
    rows, _, maplog = load_rows(testing, set(RADAR_KEYS) | set(REF_KEYS))
    log(f"  {len(rows)} (sensor, iteration) row sets, {sum(S['e'].size for S in rows.values())} rows "
        f"({time.time() - t_start:.0f} s)")

    # ---- step 1: reference latency
    log("step 1: reference latency ...")
    ref_fit = {(sk, ik): fit_ref_latency(rows[(sk, ik)], f"{sk}|{ik}") for sk in REF_KEYS for ik in tkeys
               if (sk, ik) in rows}
    ref_rows, ref_tau, ref_dec = [], {}, {}
    for sk in REF_KEYS:
        fits_sk = {ik: {"valid": f["valid"], "vals": {"tau_s": f["tau"]}} for (s, ik), f in ref_fit.items() if s == sk}
        for ik in tkeys:
            pl = pool(fits_sk, ik).get("tau_s")
            tau_l = pl["mean"] if pl else float("nan")
            ref_tau[(sk, ik)] = tau_l if np.isfinite(tau_l) else 0.0      # LOIO: used by step 5 on testing k
            if (sk, ik) not in rows:
                continue
            f = ref_fit[(sk, ik)]
            S = rows[(sk, ik)]
            eh_raw = np.hypot(S["e"] - S["te"], S["n"] - S["tn"])
            ce = S["e"] + ref_tau[(sk, ik)] * np.nan_to_num(S["ve"])
            cn = S["n"] + ref_tau[(sk, ik)] * np.nan_to_num(S["vn"])
            eh_c = np.hypot(ce - S["te"], cn - S["tn"])
            base = {"sensor_key": sk, "iter_key": ik, "n_val": int(S["e"].size),
                    "med_h_raw_m": float(np.median(eh_raw)), "med_h_corr_m": float(np.median(eh_c)),
                    "n_vel": int(np.isfinite(S["ve"]).sum())}
            ref_rows.append({**base, "kind": "fit", "tau_s": f["tau"], "tau_lo": f["lo"], "tau_hi": f["hi"],
                             "b_e_m": f["b"][0], "b_n_m": f["b"][1], "n_samples": f["n_samples"], "n_segs": f["n_segs"],
                             "few_blocks": f["few_blocks"], "rms_speed_ms": f["rms_speed"], "n_iter": int(f["valid"]),
                             "iters": ik if f["valid"] else ""})
            ref_rows.append({**base, "kind": "loio", "tau_s": tau_l, "tau_lo": pl["ci_lo"] if pl else np.nan,
                             "tau_hi": pl["ci_hi"] if pl else np.nan, "b_e_m": np.nan, "b_n_m": np.nan, "n_samples": 0,
                             "n_segs": 0, "few_blocks": 0, "rms_speed_ms": np.nan, "n_iter": pl["n"] if pl else 0,
                             "iters": "|".join(pl["iters"]) if pl else ""})
        pl = pool(fits_sk).get("tau_s")
        tau_all = pl["mean"] if pl else 0.0
        for it in its:
            if not it.has_groundtruth:
                ref_tau[(sk, it.key)] = tau_all
        # decision (same count + CI rule as the radar terms; added after review, 2026-10-29)
        lo_rows = [r for r in ref_rows if r["sensor_key"] == sk and r["kind"] == "loio" and np.isfinite(r["tau_s"])]
        n_imp = sum(r["med_h_corr_m"] < r["med_h_raw_m"] for r in lo_rows)
        need = math.ceil(DECIDE_SHARE * len(lo_rows)) if lo_rows else 0
        excl = bool(pl and np.isfinite(pl["ci_lo"]) and (pl["ci_lo"] > 0 or pl["ci_hi"] < 0))
        if not pl:
            dec = "reject: no valid fit (tau 0)"
        else:
            dec = ("apply" if len(lo_rows) >= 2 and n_imp >= need and excl else "reject") + \
                f": LOIO improves {n_imp}/{len(lo_rows)} (need {need}); CI excludes 0: {'yes' if excl else 'no'}"
        ref_dec[sk] = dec
        ref_rows.append({"sensor_key": sk, "iter_key": "*", "kind": "pooled", "decision": dec,
                         "tau_s": pl["mean"] if pl else np.nan,
                         "tau_lo": pl["ci_lo"] if pl else np.nan, "tau_hi": pl["ci_hi"] if pl else np.nan,
                         "b_e_m": np.nan, "b_n_m": np.nan, "n_samples": 0, "n_segs": 0, "few_blocks": 0,
                         "rms_speed_ms": np.nan, "n_iter": pl["n"] if pl else 0,
                         "iters": "|".join(pl["iters"]) if pl else "", "n_val": 0, "med_h_raw_m": np.nan,
                         "med_h_corr_m": np.nan, "n_vel": 0})

    # ---- steps 2-4: radar fits, LOIO, decisions
    log(f"steps 2-4: radar fits + decisions ({time.time() - t_start:.0f} s) ...")
    fits = Fits(rows)
    val_cache: dict = {}
    decisions, accepted = [], {}
    radar_iters = {sk: [ik for ik in tkeys if (sk, ik) in rows] for sk in RADAR_KEYS}
    for sk in RADAR_KEYS:
        iters = radar_iters[sk]
        if not iters:
            continue
        for model in (H_FULL, "h:daz", V_FULL, "hover:daz"):
            fits.by_iter(sk, model, iters)
        d_rows, acc_h, acc_v = decide(sk, iters, fits, val_cache)
        decisions += d_rows
        accepted[sk] = (acc_h, acc_v)
        log(f"  {sk}: h {sorted(acc_h) or '-'}, v {sorted(acc_v) or '-'} ({time.time() - t_start:.0f} s)")

    # params rows
    prow = []
    for (sk, ik, model), f in sorted(fits.memo.items()):
        prow.append(fit_row(sk, ik, f))
    for sk in RADAR_KEYS:
        iters = radar_iters[sk]
        if not iters:
            continue
        models = sorted({m for (s, _, m) in fits.memo if s == sk})
        for model in models:
            byi = fits.by_iter(sk, model, iters)
            for ik in tkeys:                # every testing key: one without drone_air rows gets the plain pool
                prow.append(pool_row(sk, ik, "loio", model, pool(byi, ik)))
            prow.append(pool_row(sk, "*", "pooled", model, pool(byi)))
        if sk not in accepted:
            continue
        acc_h, acc_v = accepted[sk]
        mh = model_name("h", acc_h) if acc_h else None
        mv = model_name("v", acc_v) if acc_v else None
        for ik in tkeys + ["*"]:
            pl = {}
            for mdl in (mh, mv):
                if mdl:
                    pl.update(pool(fits.by_iter(sk, mdl, iters), None if ik == "*" else ik))
            r = pool_row(sk, ik, "decided" if ik == "*" else "decided_loio", "decided", pl)
            r["terms"] = "|".join(sorted(acc_h | acc_v))
            prow.append(r)

    # ---- validation
    log(f"validation ({time.time() - t_start:.0f} s) ...")
    vrows, nested = [], {}
    for sk in RADAR_KEYS:
        iters = radar_iters[sk]
        models = sorted({m for (s, _, m) in fits.memo if s == sk and not m.startswith("hover")})
        acc_h, acc_v = accepted.get(sk, (set(), set()))
        for ik in iters:
            S = rows[(sk, ik)]
            meth = {"raw": {}}
            for model in models:
                meth["loio:" + model] = pooled_params(fits.by_iter(sk, model, iters), ik)
                f = fits.get(sk, ik, model)
                meth["insample:" + model] = dict(f["vals"]) if f["valid"] else {}
            meth["loio:full"] = {**meth["loio:" + H_FULL], **meth["loio:" + V_FULL]}
            meth["insample:full"] = {**meth["insample:" + H_FULL], **meth["insample:" + V_FULL]}
            dec = {}
            if acc_h:
                dec.update(pooled_params(fits.by_iter(sk, model_name("h", acc_h), iters), ik))
            if acc_v:
                dec.update(pooled_params(fits.by_iter(sk, model_name("v", acc_v), iters), ik))
            meth["loio:decided"] = dec
            # term SELECTION without k too: the decision procedure rerun on the other iterations only
            sub = [i for i in iters if i != ik]
            nh, nv = decide(sk, sub, fits, val_cache)[1:] if sub else (set(), set())
            nest = {}
            for axis, acc in (("h", nh), ("v", nv)):
                if acc:
                    nest.update(pooled_params(fits.by_iter(sk, model_name(axis, acc), iters), ik))
            meth["loio:decided_nested"] = nest
            nested[(sk, ik)] = sorted(nh | nv)
            for name, p in meth.items():
                e, n, u = apply(S, sk, p)
                if name.startswith("loio:decided") and not (p.get("c_u_m") or p.get("del_deg")):
                    u = S["u_sn"]           # no final vertical term: the deployed height is kept (u_frame 'any')
                vrows.append({"sensor_key": sk, "iter_key": ik, "method": name, **err_stats(e, n, u, S),
                              "testing_only": int(sk in TESTING_ONLY)})
            vrows.append({"sensor_key": sk, "iter_key": ik, "method": "u_sn_insample",
                          **err_stats(S["e"], S["n"], S["u_sn"], S), "testing_only": int(sk in TESTING_ONLY)})

    # ---- step 5: GT-free azimuth
    log(f"step 5: GT-free azimuth vs the testing interval ({time.time() - t_start:.0f} s) ...")
    sites = RR.radar_sites()
    gfz = gt_free_az(its, ref_tau, sites)
    for (ik, rid, variant), est in sorted(gfz.items()):
        sk = f"{rid}_RADAR"
        if sk not in RADAR_KEYS:
            continue
        r = blank_row(sk, ik, "gt_free_" + variant, "h:daz")
        iters = radar_iters.get(sk, [])
        byi = fits.by_iter(sk, "h:daz", iters) if iters else {}
        pl = pool(byi, ik if ik in tkeys else None).get("daz_deg") if byi else None
        if pl:
            r["pi_lo_daz"], r["pi_hi_daz"] = pl["pi_lo"], pl["pi_hi"]
            r["n_iter"], r["iters"] = pl["n"], "|".join(pl["iters"])
        if ik in tkeys and (sk, ik) in rows:
            f = fits.get(sk, ik, "h:daz")
            r["gt_daz_deg"] = f["vals"].get("daz_deg", np.nan) if f["valid"] else np.nan
        if est is not None:
            r["daz_deg"], r["daz_deg_lo"], r["daz_deg_hi"] = est["az"], est["lo"], est["hi"]
            r["n_samples"], r["n_pairs"], r["n_groups"] = est["n"], est.get("n_pairs", 0), est.get("n_groups", 0)
            if np.isfinite(r["pi_lo_daz"]):
                r["inside"] = int(r["pi_lo_daz"] <= est["az"] <= r["pi_hi_daz"])
        prow.append(r)

    # ---- write
    C.write_columns(OUT / "params.csv", {c: _col([r[c] for r in prow]) for c in PARAM_ORDER}, list(PARAM_ORDER))
    VORDER = ["sensor_key", "iter_key", "method", "n", "n_u", "med_h_m", "med_v_m", "med_3d_m", "h_le10", "h_le25",
              "h_le50", "d3_le10", "d3_le25", "d3_le50", "testing_only"]
    C.write_columns(OUT / "validation.csv", {c: _col([r[c] for r in vrows]) for c in VORDER}, VORDER)
    DORDER = ["sensor_key", "axis", "term", "baseline", "with_terms", "n_iters", "n_improved", "need", "improved_iters",
              "pooled_value", "ci_lo", "ci_hi", "ci_excludes_0", "n_ci_iters", "decision", "vs_deployed", "final",
              "final_value", "testing_only"]
    C.write_columns(OUT / "decisions.csv", {c: _col([r[c] for r in decisions]) for c in DORDER}, DORDER)
    RORDER = ["sensor_key", "iter_key", "kind", "tau_s", "tau_lo", "tau_hi", "b_e_m", "b_n_m", "n_samples", "n_segs",
              "few_blocks", "rms_speed_ms", "n_iter", "iters", "n_val", "med_h_raw_m", "med_h_corr_m", "n_vel", "decision"]
    C.write_columns(OUT / "reference_latency.csv", {c: _col([r.get(c, "") for r in ref_rows]) for c in RORDER}, RORDER)
    _PARAMS_CACHE.clear()

    # ---- self-checks
    checks: list[tuple[str, bool, str]] = []
    known = None
    if not args.skip_checks:
        log(f"self-checks ({time.time() - t_start:.0f} s) ...")
        checks, known = self_checks(testing, rows, fits, decisions, maplog, prow)
    write_report(its, testing, rows, fits, radar_iters, decisions, accepted, vrows, ref_rows, prow, checks, known,
                 time.time() - t_start, nested, ref_dec)
    n_fail = sum(not ok for _, ok, _ in checks)
    for name, ok, msg in checks:
        log(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")
    log(f"wrote {OUT}/params.csv, validation.csv, decisions.csv, reference_latency.csv, radar_model.md")
    log(f"self-checks: {len(checks) - n_fail}/{len(checks)} passed; runtime {time.time() - t_start:.0f} s")
    return 1 if n_fail else 0


def _col(vals: list) -> np.ndarray | list:
    if all(isinstance(v, (float, np.floating)) for v in vals):
        return np.array(vals, dtype=float)
    return ["" if (isinstance(v, float) and v != v) else (repr(float(v)) if isinstance(v, (float, np.floating)) else v)
            for v in vals]


def self_checks(testing, rows, fits: Fits, decisions, maplog, prow):
    checks = []
    tkeys = [it.key for it in testing]
    # (1)
    ok, msg = check_synthetic(testing)
    checks.append(("1 synthetic radar: every injected parameter inside its interval", ok, msg))
    # (2)
    ok, msg = check_geometry()
    checks.append(("2 basis / correction geometry vs closed form", ok, msg))
    # (3) plain-python recomputation of the real fit with the most samples (EGOI or NYTF)
    cand = [(rows[(sk, ik)]["e"].size, sk, ik) for (sk, ik) in rows if sk in ("EGOI_RADAR", "NYTF_RADAR")]
    _, sk3, ik3 = sorted(cand)[len(cand) // 2]          # a median-size set keeps the plain loop quick
    f = fits.get(sk3, ik3, H_FULL)
    if f["valid"] and f["terms"] == list(H_TERMS):
        pv = plain_fit(rows[(sk3, ik3)])
        mv = [f["vals"][p] for p in ("daz_deg", "tau_s", "b_r_m", "s_r")]
        dev = max(abs(a - b) / (1e-6 + abs(b)) for a, b in zip(mv, pv))
        checks.append(("3 plain-Python recomputation of one fit", dev < 1e-6,
                       f"{sk3} {short(ik3)} h full ({f['n_samples']} samples): numpy {np.round(mv, 6).tolist()} vs plain "
                       f"{np.round(pv, 6).tolist()}, max relative deviation {dev:.1e} (< 1e-6)"))
    else:
        checks.append(("3 plain-Python recomputation of one fit", False,
                       f"{sk3} {short(ik3)}: full model not fitted with all four terms ({f['terms']})"))
    # (4) LOIO guard
    bad = []
    for r in prow:
        if r["kind"] in ("loio", "decided_loio") and r["iter_key"] in r["iters"].split("|"):
            bad.append(f"{r['sensor_key']} {r['iter_key']} {r['model']}")
    for sk in ("EGOI_RADAR", "NYTF_RADAR", "UXYJ_RADAR", "IUKL_RADAR"):
        iters = [ik for ik in tkeys if (sk, ik) in rows]
        if len(iters) < 3:
            continue
        byi = fits.by_iter(sk, H_FULL, iters)
        k = iters[0]
        ref_k = pooled_params(byi, k)
        mut = {ik: ({**f, "vals": {nm: v + 1e6 for nm, v in f["vals"].items()}} if ik == k else f) for ik, f in byi.items()}
        if pooled_params(mut, k) != ref_k:
            bad.append(f"{sk}: mutating {short(k)} changed its own LOIO params")
        if pooled_params(mut, iters[1]) == pooled_params(byi, iters[1]):
            bad.append(f"{sk}: mutating {short(k)} did not reach the LOIO params of {short(iters[1])}")
        # recompute the LOIO mean by hand
        for nm, v in ref_k.items():
            vals = [byi[i]["vals"][nm] for i in iters if i != k and byi[i]["valid"] and nm in byi[i]["vals"]]
            if abs(sum(vals) / len(vals) - v) > 1e-12:
                bad.append(f"{sk} {nm} mean")
    # corrections with every GT loader disabled, and invariant to a mutated truth
    sk, ik = "EGOI_RADAR", next(i for i in tkeys if ("EGOI_RADAR", i) in rows)
    S = rows[(sk, ik)]
    p = pooled_params(fits.by_iter(sk, H_FULL, [i for i in tkeys if (sk, i) in rows]), ik)
    meas = {c: S[c] for c in ("e", "n", "u", "t", "ve", "vn", "vu", "se", "sn", "su")}
    with RR.gt_forbidden():
        a = correct(sk, meas["e"], meas["n"], meas["u"], meas["t"], meas["ve"], meas["vn"], meas["vu"], meas["se"],
                    meas["sn"], meas["su"], p)
    S2 = {**S, "te": S["te"] + 100.0, "tn": S["tn"] - 50.0, "tu": S["tu"] + 9.0, "tve": S["tve"] * 2}
    b = apply(S2, sk, p)
    if not all(np.array_equal(x, y, equal_nan=True) for x, y in zip(a, b)):
        bad.append("correction moved with the truth")
    # params_for (written params.csv) answers every testing key with its own LOIO row; a key without drone_air
    # rows (e.g. IUKL S1-I1) gets the pool of all the other iterations, i.e. the evaluation values
    n_pf = 0
    for sk_ in ("EGOI_RADAR", "NYTF_RADAR", "UXYJ_RADAR", "IUKL_RADAR"):
        if not any((sk_, i) in rows for i in tkeys):
            continue
        pool_all = params_for(sk_, None)
        for k in tkeys:
            pf = params_for(sk_, k)
            n_pf += 1
            if pf["source"] != f"decided_loio:{k}":
                bad.append(f"params_for({short(sk_)}, {short(k)}) source {pf['source']}")
            if (sk_, k) not in rows and any(abs(pf[q] - pool_all[q]) > 1e-12 for q in PARAMS):
                bad.append(f"params_for({short(sk_)}, {short(k)}) without rows differs from the full pool")
    n_loio = sum(r["kind"] in ("loio", "decided_loio") for r in prow)
    checks.append(("4 LOIO guard", not bad,
                   f"{n_loio} loio / decided_loio rows never list their own iteration; mutating iteration k leaves its "
                   f"LOIO params unchanged and reaches the others; hand recomputed means; the correction runs with every "
                   f"GT loader disabled and is unchanged by mutated truth; params_for answers all {n_pf} (radar, testing "
                   f"key) pairs from their own LOIO row; failures {bad or 'none'}"))
    # (5) known grid numbers
    daz_loio = {}
    for rid in KNOWN_GRID:
        sk_ = f"{rid}_RADAR"
        iters = [i for i in tkeys if (sk_, i) in rows]
        byi = fits.by_iter(sk_, "h:daz", iters)
        for i in tkeys:
            d = pooled_params(byi, i).get("daz_deg")
            if d is not None:
                daz_loio[(sk_, i)] = d
    known = grid_known(testing, daz_loio, ST.radar_bias_values())
    ok5, parts = True, []
    for rid, (k_raw, k_after) in KNOWN_GRID.items():
        per = known[rid]
        raw = float(np.mean([x[2] for x in per])) if per else float("nan")
        mod = float(np.mean([x[3] for x in per])) if per else float("nan")
        rbv = float(np.mean([x[4] for x in per])) if per else float("nan")
        n_imp = sum(x[3] < x[2] for x in per)
        n_imp_rb = sum(x[4] < x[2] for x in per)
        # (a) protocol + correct(): the known numbers came from the radar_bias angle (a median on rows gated at
        # 50 m); fed through correct() it must reproduce them (two-sided, RB_TOL_M). (b) this model's ungated
        # Huber angle (the spec forbids the gate) must improve 5 of 5 and be no worse than the known number + 1 m.
        good = (len(per) == 5 and abs(raw - k_raw) <= KNOWN_TOL_M and n_imp_rb == 5 and abs(rbv - k_after) <= RB_TOL_M
                and n_imp == 5 and mod <= k_after + KNOWN_TOL_M)
        ok5 &= good
        parts.append(f"{rid}: {len(per)} iterations, raw {raw:.2f} (known {k_raw}); radar_bias LOIO angle through "
                     f"correct() {rbv:.2f} (known {k_after}, tolerance +-{RB_TOL_M}, {n_imp_rb} of {len(per)} improved); "
                     f"this model's LOIO daz {mod:.2f} ({n_imp} of {len(per)} improved, must be <= {k_after + KNOWN_TOL_M:.1f})")
    parts.append("the gap between the two angles is the estimator, not the geometry: radar_bias is a median on rows "
                 "gated at 50 m, which shrinks a larger rotation; this model's angle is an ungated Huber fit")
    checks.append(("5 azimuth-only LOIO: the known EGOI / NYTF grid numbers reproduced through correct() (5 of 5), "
                   "this model's angle at least as good", ok5, "; ".join(parts)))
    # (6) determinism: refit one radar's models from scratch (with bootstrap) and redo its decisions
    sk6 = "UXYJ_RADAR"
    iters6 = [i for i in tkeys if (sk6, i) in rows]
    f2 = Fits(rows)
    for model in (H_FULL, "h:daz", V_FULL, "hover:daz"):
        f2.by_iter(sk6, model, iters6)
    d2, _, _ = decide(sk6, iters6, f2, {})
    same = all(_fit_digest(fits.memo[k]) == _fit_digest(f2.memo[k]) for k in f2.memo)
    d1 = [r for r in decisions if r["sensor_key"] == sk6]
    same &= _digest(d1) == _digest(d2)
    checks.append(("6 determinism", same, f"{sk6}: {len(f2.memo)} fits (with bootstrap intervals) and its decisions "
                                         f"recomputed from scratch are bit-identical: {same}"))
    # (7) segment mapping
    bad7 = [m for m in maplog if m[3] != m[4]]
    checks.append(("7 segment row mapping reproduces tracks.csv n_rows", not bad7,
                   f"{len(maplog)} drone_air segments of the radars / references; mismatches {bad7[:5] or 'none'}"))
    return checks, known


def _fit_digest(f: dict) -> str:
    keys = ("vals", "lo", "hi", "terms", "dropped", "n_samples", "n_segs")
    return hashlib.sha256(repr([(k, sorted(f[k].items()) if isinstance(f[k], dict) else f[k]) for k in keys]).encode()).hexdigest()


def _digest(rows: list[dict]) -> str:
    return hashlib.sha256(repr([sorted(r.items()) for r in rows]).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def write_report(its, testing, rows, fits, radar_iters, decisions, accepted, vrows, ref_rows, prow, checks, known,
                 runtime, nested, ref_dec) -> None:
    L = []
    n_fail = sum(not ok for _, ok, _ in checks)
    L.append("# Radar error model in sensor coordinates\n")
    L.append(("**Self-checks: all passed.** " if checks and not n_fail else
              (f"**Self-checks: {n_fail} FAILED.** " if checks else "**Self-checks skipped.** "))
             + f"Generated by `tools/radar_model.py` (rules, model and schemas in its docstring); runtime {runtime:.0f} s.\n")
    L.append("## Terms\n")
    for t_, d in (
            ("radar site", "where the radar stands (the row's sensor_e / sensor_n)."),
            ("range R", "horizontal distance from the radar site to the target."),
            ("azimuth", "compass direction of the target seen from the radar, clockwise from north."),
            ("daz", "azimuth error in degrees: positive means the radar puts the drone clockwise of where it is. "
                    "At 2 km, 1 deg is 35 m sideways."),
            ("tau (latency)", "how many seconds late the sensor reports: the position shown is where the drone was "
                              "tau seconds ago, so the error points backwards along the flight, tau x speed metres."),
            ("b_r, s_r", "range error: a constant b_r metres plus s_r x range (s_r = 0.004 is 4 m per km)."),
            ("c_u, del", "height error: a constant c_u metres plus an elevation-like term del (deg) x range."),
            ("residual", "measured position minus the Dronetag truth at the same time."),
            ("Huber IRLS", "a least-squares fit that down-weights large residuals so a few bad rows cannot drag it."),
            ("segment", "one continuous piece of one sensor track (sensor_tracks_v2); drone_air = matched to one "
                        "Dronetag flight and mostly airborne."),
            ("block bootstrap", "refit on segments drawn at random with replacement, 400 times; the spread of the "
                                "refits gives the interval (CI). Few segments make it unreliable (few_blocks)."),
            ("VIF", "variance inflation: how much a term's uncertainty grows because it looks like another term "
                    "(1 = independent; above 10 the term is dropped)."),
            ("LOIO", "leave-one-iteration-out: the value used on testing iteration k is the mean over the OTHER "
                     "testing iterations, so k is never used to correct itself."),
            ("prediction interval", "where the value of a NEW iteration should fall given the spread between iterations."),
            ("median error", "half of the rows are closer than this to the truth."),
    ):
        L.append(f"* **{t_}**: {d}")
    L.append("")

    # headline
    dec_by = {}
    for d in decisions:
        dec_by.setdefault(d["sensor_key"], []).append(d)
    vb = {(r["sensor_key"], r["iter_key"], r["method"]): r for r in vrows}
    L.append("## Headline\n")
    for sk in RADAR_KEYS:
        if sk not in accepted:
            continue
        acc_h, acc_v = accepted[sk]
        iters = radar_iters[sk]
        el = [ik for ik in iters if rows[(sk, ik)]["e"].size >= MIN_VAL_N]
        raw = [vb[(sk, ik, "raw")]["med_h_m"] for ik in el]
        az = [vb[(sk, ik, "loio:h:daz")]["med_h_m"] for ik in el]
        dec = [vb[(sk, ik, "loio:decided")]["med_h_m"] for ik in el]
        nst = [vb[(sk, ik, "loio:decided_nested")]["med_h_m"] for ik in el]
        n_az = sum(a < r for a, r in zip(az, raw))
        n_dec = sum(a < r for a, r in zip(dec, raw))
        n_nst = sum(a < r for a, r in zip(nst, raw))
        gate = next((d["vs_deployed"] for d in dec_by.get(sk, []) if d["axis"] == "v" and d["vs_deployed"]), "")
        vsn = [vb[(sk, ik, "u_sn_insample")]["med_v_m"] for ik in el]
        vdec = [vb[(sk, ik, "loio:decided")]["med_v_m"] for ik in el]
        L.append(f"* **{sk.split('_')[0]}**{' (testing-only, analysis only)' if sk in TESTING_ONLY else ''}: final "
                 f"terms {', '.join(sorted(acc_h | acc_v)) or 'none'}. Mean of per-iteration median horizontal error "
                 f"raw {f_(np.mean(raw))} m -> azimuth-only LOIO {f_(np.mean(az))} m ({n_az} of {len(el)} better) -> "
                 f"decided LOIO {f_(np.mean(dec))} m ({n_dec} of {len(el)} better than raw) -> decided with the "
                 f"term selection also made without the held-out iteration {f_(np.mean(nst))} m ({n_nst} of {len(el)} "
                 f"better than raw; selections {'; '.join(short(ik) + ': ' + ('+'.join(nested[(sk, ik)]) or 'none') for ik in el)}). "
                 f"Vertical: deployed (sensor_noise, in-sample) height {f_(np.mean(vsn))} m, final {f_(np.mean(vdec))} m"
                 + (f"; vertical gate vs the deployed height {gate}" if gate else "") + ".")
    L.append("")
    L.append("'decided LOIO' uses LOIO parameter values but the term SET was chosen by the decision table, which saw "
             "every iteration (mild selection optimism); the 'selection also without k' number reruns the whole "
             "decision procedure on the other iterations for each held-out k and is the honest one. A vertical set "
             "that loses to the deployed sensor_noise height (gate FAIL) is dropped: the deployed height is kept "
             "(params_for u_frame 'any').\n")

    # reference latency
    L.append("## 1. Reference latency (STUG, ILYS, CEQX)\n")
    L.append("Fit on drone_air segments: residual = constant offset - tau x truth velocity. Positive tau = reports "
             "late. The LOIO row uses the other iterations only; med raw / corr = median horizontal error of the "
             "reports before / after moving them forward by the LOIO tau with the sensor's own (or differenced) "
             "velocity; n_vel = reports with a velocity (the rest cannot be corrected).\n")
    tab = []
    for r in ref_rows:
        if r["kind"] == "pooled":
            continue
        tab.append([r["sensor_key"].split("_")[0], short(r["iter_key"]), r["kind"], f_(r["tau_s"], 2),
                    f"{f_(r['tau_lo'], 2)}..{f_(r['tau_hi'], 2)}", r["n_samples"] if r["kind"] == "fit" else r["n_iter"],
                    r["n_segs"] if r["kind"] == "fit" else "-", "yes" if r.get("few_blocks") else "",
                    f_(r["med_h_raw_m"]), f_(r["med_h_corr_m"]) if r["kind"] == "loio" else "-", r["n_vel"]])
    L += md_table(["sensor", "iteration", "kind", "tau s", "interval", "samples / iters", "segments", "few blocks",
                   "med raw m", "med corr m", "n_vel"], tab)
    L.append("")
    for r in ref_rows:
        if r["kind"] == "pooled":
            L.append(f"* {r['sensor_key']}: pooled tau {f_(r['tau_s'], 2)} s (CI {f_(r['tau_lo'], 2)}..{f_(r['tau_hi'], 2)}, "
                     f"{r['n_iter']} iterations). Decision: **{ref_dec[r['sensor_key']]}**.")
    lat_primary = bool(ref_dec) and all(d.startswith("apply") for d in ref_dec.values())
    L.append("")
    L.append("Decision rule for the references (same as for the radar terms, added after review): apply the latency "
             "iff the LOIO correction lowers the median error on >= ceil(2/3 n) of the sensor's iterations AND the "
             "pooled CI excludes 0. A sensor without a valid fit (too few drone_air samples) gets tau 0. "
             + ("Every reference passes, so step 5's primary variant is 'lat' (references moved forward by their tau)."
                if lat_primary else
                "Not every reference passes, so the latency correction is NOT supported by the data: step 5's primary "
                "variant is 'nolat' (references as reported); the 'lat' variant (estimated taus applied anyway) is "
                "kept as analysis only.") + "\n")

    # per-iteration params
    L.append("## 2-3. Per-iteration fits (full model, in-sample)\n")
    L.append("One row per radar and testing iteration: the full horizontal model (daz, tau, b_r, s_r) and the "
             "vertical model (c_u, del), each with its bootstrap interval. '-' = not identified (dropped, reason "
             "in the next table) or not enough data. daz hover = the azimuth fitted only on hovering samples "
             "(truth speed < 1 m/s), where a latency cannot act: it should agree with daz.\n")
    tab = []
    for sk in RADAR_KEYS:
        for ik in radar_iters[sk]:
            fh, fv, fz, fo = (fits.get(sk, ik, m) for m in (H_FULL, V_FULL, "h:daz", "hover:daz"))

            def pv(f, nm, nd=2):
                if not f["valid"] or nm not in f["vals"]:
                    return "-"
                return f"{f['vals'][nm]:.{nd}f} ({f_(f['lo'].get(nm, np.nan), nd)}..{f_(f['hi'].get(nm, np.nan), nd)})"
            tab.append([sk.split("_")[0], short(ik), fh["n_samples"], fh["n_segs"], pv(fz, "daz_deg"), pv(fh, "daz_deg"),
                        pv(fh, "tau_s"), pv(fh, "b_r_m", 1), pv(fh, "s_r", 4), pv(fv, "c_u_m", 1), pv(fv, "del_deg"),
                        pv(fo, "daz_deg") + (f" n={fo['n_samples']}" if fo["valid"] else "")])
    L += md_table(["radar", "iter", "samples", "segments", "daz az-only", "daz full", "tau s", "b_r m", "s_r",
                   "c_u m", "del deg", "daz hover"], tab)
    hov = [(sk, ik) for sk in RADAR_KEYS for ik in radar_iters[sk] if fits.get(sk, ik, "hover:daz")["valid"]]
    n_fit = sum(len(radar_iters[sk]) for sk in RADAR_KEYS)
    L.append("")
    L.append(f"The hover cross-check is available for {len(hov)} of {n_fit} fits "
             f"({', '.join(sk.split('_')[0] + ' ' + short(ik) for sk, ik in hov) or 'none'}); elsewhere there are fewer "
             f"than {MIN_FIT_N} hovering samples (truth speed < {HOVER_MS:g} m/s) at >= {HOVER_MIN_R:g} m range in "
             f">= {MIN_FIT_SEGS} segments. So daz is NOT hover-validated for the other fits: the latency / azimuth "
             f"separation there rests on the VIF / correlation numbers below.\n")
    L.append("### Collinearity and dropped terms\n")
    L.append("VIF per term of the full horizontal model, the largest absolute correlation between two terms, and "
             "the terms dropped before the fit (pre-registered rule: rms speed < 1.5 m/s, range sd < 150 m or VIF > 10).\n")
    tab = []
    for sk in RADAR_KEYS:
        for ik in radar_iters[sk]:
            fh, fv = fits.get(sk, ik, H_FULL), fits.get(sk, ik, V_FULL)
            if not fh["valid"]:
                tab.append([sk.split("_")[0], short(ik), "-", "-", "-", "-", "-"])
                continue
            mc = max(fh["corr"].items(), key=lambda kv: abs(kv[1])) if fh["corr"] else ("-", float("nan"))
            tab.append([sk.split("_")[0], short(ik), " ".join(f"{k} {v:.1f}" for k, v in fh["vif"].items()),
                        f"{mc[0]} {mc[1]:+.2f}", f_(fh["rms_speed"]), f_(fh["r_sd"], 0),
                        ", ".join(fh["dropped"] + (fv["dropped"] if fv["valid"] else [])) or "none"])
    L += md_table(["radar", "iter", "VIF", "largest corr", "rms speed m/s", "range sd m", "dropped"], tab)
    L.append("")

    # stability
    L.append("## 4. Stability across iterations\n")
    L.append("Pooled = mean of the per-iteration estimates (all testing iterations), its 95 % CI (is the mean "
             "different from 0?) and sd between iterations. A term is stable when sd is small next to the mean.\n")
    tab = []
    for sk in RADAR_KEYS:
        iters = radar_iters[sk]
        if not iters:
            continue
        for model in ("h:daz", H_FULL, V_FULL):
            pl = pool(fits.by_iter(sk, model, iters))
            for nm, d in pl.items():
                tab.append([sk.split("_")[0], model, nm, f_(d["mean"], 4 if nm == "s_r" else 2),
                            f"{f_(d['ci_lo'], 4 if nm == 's_r' else 2)}..{f_(d['ci_hi'], 4 if nm == 's_r' else 2)}",
                            f_(d["sd"], 4 if nm == "s_r" else 2), d["n"]])
    L += md_table(["radar", "model", "param", "pooled", "CI", "sd", "iterations"], tab)
    L.append("")

    # validation
    L.append("## 4. LOIO correction, per iteration\n")
    L.append("Median horizontal error (m) of every raw report of the radar's drone_air segments, before and after "
             "each correction (all values LOIO except 'in-sample full', which uses the iteration's own fit and is a "
             "reference only). 'decided (nested)' = the term set also chosen without the held-out iteration. Shares = fraction of reports within 25 m horizontally. Vertical: median |height error| "
             "(m); 'sn offset' = the current in-sample sensor_noise height offset, for comparison; 'v decided' = that "
             "deployed height wherever no vertical term is final.\n")
    tab = []
    for sk in RADAR_KEYS:
        for ik in radar_iters[sk]:
            g = lambda m, c="med_h_m": vb.get((sk, ik, m), {}).get(c, float("nan"))  # noqa: E731
            tab.append([sk.split("_")[0], short(ik), vb[(sk, ik, "raw")]["n"], f_(g("raw")), f_(g("loio:h:daz")),
                        f_(g("loio:full")), f_(g("loio:decided")), f_(g("loio:decided_nested")), f_(g("insample:full")),
                        f"{f_(100 * g('raw', 'h_le25'), 0)} / {f_(100 * g('loio:decided', 'h_le25'), 0)} %",
                        f_(g("raw", "med_v_m")), f_(g("u_sn_insample", "med_v_m")), f_(g("loio:" + V_FULL, "med_v_m")),
                        f_(g("loio:decided", "med_v_m"))])
    L += md_table(["radar", "iter", "rows", "raw", "az-only", "full", "decided", "decided (nested)", "in-sample full",
                   "<= 25 m raw / dec",
                   "v raw", "v sn offset", "v full", "v decided"], tab)
    L.append("")

    # decisions
    L.append("## 4. Decision table (pre-registered)\n")
    L.append("Sequential per radar: daz, tau, b_r, s_r (horizontal), then c_u, del (vertical). A term is tested "
             "as 'accepted so far + term' against 'accepted so far'. **apply** iff that LOIO correction has a lower "
             "median error on at least ceil(2/3 n) of the radar's n testing iterations AND the 95 % CI of the term's "
             "mean over iterations excludes 0. EUIG is testing-only: its rows are analysis, never a deployable decision. "
             "'decision' is that pre-registered test of the row's own term. **'final' is what params_for() applies**: "
             "a term can be final without its own row passing when another accepted term brought it in (s_r brings "
             "b_r, del brings c_u; the bundled term's value is then the one of the accepted model, column final "
             "value, not the row's pooled value). The vertical set must also pass a non-regression gate (column "
             "vs deployed, added after review): its LOIO correction must beat the height deployed today "
             "(sensor_noise offset, in-sample, so the gate is conservative) on >= ceil(2/3 n) iterations, else the "
             "deployed height is kept and the vertical terms are not final.\n")
    tab = [[d["sensor_key"].split("_")[0], d["axis"], d["term"], d["baseline"], d["with_terms"],
            f"{d['n_improved']} / {d['n_iters']} (need {d['need']})", f_(d["pooled_value"], 4 if d["term"] == "s_r" else 2),
            f"{f_(d['ci_lo'], 4 if d['term'] == 's_r' else 2)}..{f_(d['ci_hi'], 4 if d['term'] == 's_r' else 2)}",
            "yes" if d["ci_excludes_0"] else "no", d["decision"], d["vs_deployed"] or "-",
            f"**{d['final']}**" + (" (testing-only)" if d["testing_only"] else ""),
            f_(d["final_value"], 4 if d["term"] == "s_r" else 2)]
           for d in decisions]
    L += md_table(["radar", "axis", "term", "baseline", "with", "improved", "pooled", "CI", "CI excl. 0", "decision",
                   "vs deployed", "final", "final value"], tab)
    L.append("")

    # step 5
    L.append("## 5. GT-free azimuth per iteration vs the testing interval\n")
    L.append("radar_registration's co-observation estimator (no ground truth) of each radar's azimuth error against "
             "the references, with the references moved forward by their latency ('lat') or not. 'noutm' = cyber / "
             "passive-RF references only; 'all' adds VKYY / QTVS, which on testing are the Dronetag feed itself (so "
             "testing 'all' numbers are partly ground truth). The interval is the az-only model's prediction interval "
             "for a new iteration (LOIO on testing, all 6 on evaluation). GT daz = the iteration's own fitted value.\n")
    tab = []
    gf = {(r["sensor_key"], r["iter_key"], r["kind"]): r for r in prow if r["kind"].startswith("gt_free")}
    for it in its:
        for sk in RADAR_KEYS:
            if (sk, it.key, "gt_free_noutm") not in gf:
                continue
            cells = [sk.split("_")[0], short(it.key)]
            for kind in ("gt_free_noutm", "gt_free_noutm_nolat", "gt_free_all", "gt_free_all_nolat"):
                r = gf.get((sk, it.key, kind))
                if r is None or not np.isfinite(r["daz_deg"]):
                    cells.append("-")
                else:
                    ins = {1: " in", 0: " OUT"}.get(r["inside"], "")
                    cells.append(f"{r['daz_deg']:.2f}{ins} (g{int(r['n_groups'])})")
            r0 = gf[(sk, it.key, "gt_free_noutm")]
            cells += [f"{f_(r0['pi_lo_daz'], 2)}..{f_(r0['pi_hi_daz'], 2)}", f_(r0["gt_daz_deg"], 2)]
            tab.append(cells)
    L += md_table(["radar", "iteration", "noutm lat", "noutm raw", "all lat", "all raw", "interval", "GT daz"], tab)
    L.append("")
    for kind in ("gt_free_noutm", "gt_free_noutm_nolat"):
        ins = [r for r in prow if r["kind"] == kind and r["inside"] != "" and not r["iter_key"].startswith("testing")
               and not r["testing_only"]]
        prim = (kind == "gt_free_noutm") == lat_primary
        L.append(f"* Evaluation, noutm {'with' if kind == 'gt_free_noutm' else 'without'} reference latency"
                 f"{' (PRIMARY, per the step-1 decisions)' if prim else ' (analysis only)'}: "
                 f"{sum(r['inside'] == 1 for r in ins)} of {len(ins)} estimates inside the testing interval.")
    L.append("\n'g' = co-observation groups (independent drones); with 1-2 groups the estimate rests on one or two "
             "drones.\n")
    for kind in ("gt_free_noutm", "gt_free_noutm_nolat", "gt_free_all", "gt_free_all_nolat"):
        tr = [r for r in prow if r["kind"] == kind and r["iter_key"].startswith("testing") and not r["testing_only"]
              and np.isfinite(r["daz_deg"]) and np.isfinite(r["gt_daz_deg"])]
        if tr:
            err = np.abs(np.array([r["daz_deg"] - r["gt_daz_deg"] for r in tr]))
            L.append(f"* testing, {kind[8:]}: {len(tr)} estimates, median |GT-free - GT daz| {np.median(err):.2f} deg, "
                     f"{sum(r['inside'] == 1 for r in tr)} inside the LOIO interval"
                     + (" (VKYY / QTVS are ground truth on testing)" if "all" in kind else ""))
    L.append("")

    if known:
        L.append("## Self-check 5 detail (grid protocol of the earlier calibration-stability finding)\n")
        tab = []
        for rid, per in known.items():
            for x in per:
                tab.append([rid, short(x[0]), x[1], f_(x[2]), f_(x[3]), f_(x[4])])
        L += md_table(["radar", "iter", "grid samples", "raw", "this model LOIO daz", "radar_bias LOIO"], tab)
        L.append("")

    L.append("## Caveats\n")
    L.append("* Few drones: each (radar, iteration) rests on a handful of flights; segment blocks of one flight share "
             "its slow bias, so the bootstrap intervals are sampling intervals and probably too narrow.")
    L.append("* Heights are the base ellipsoid heights (radar offsets of sensor_noise removed); the AGL / datum rule "
             "itself came from the same testing data.")
    L.append("* The UXYJ clock offset (-3.5 s) is already applied; its tau is relative to that.")
    L.append("* Corrections use the measured velocity: IUKL reports none and UXYJ no vertical speed, so theirs is "
             "differenced from the reports (+-2 s), which is noisier.\n")
    L.append("## Self-checks\n")
    L += md_table(["check", "result", "detail"], [[n, "PASS" if ok else "FAIL", m] for n, ok, m in checks])
    (OUT / "radar_model.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
