#!/usr/bin/env python3
"""Per-sensor track segments on the template's 100 ms grid: constant-velocity Kalman filter +
Rauch-Tung-Striebel smoother, with per-segment features and (testing) ground-truth labels.

This is the layer every baseline and tracker builds on: each sensor's own track (objectId),
cut into segments, smoothed and resampled on the scoring grid with a covariance.

Version 2 (2026-10-28, outputs in data/derived/sensor_tracks_v2/): heights above the WGS84
ellipsoid (rule 1 below) and a two-part noise model (rule 2), because the sensor feeds are
already tracker outputs whose error is mostly a slowly varying bias (lag-1 autocorrelation
0.83-0.97 against Dronetag), not white noise: a white-noise smoother with R = total variance
cannot beat linear interpolation of the reports.

Inputs and rules
----------------
* Measurements: data/derived/detections.csv.gz, ALL iterations, cuas_clean.clean (whole
  iterations: duplicates and site heights are found among the rows passed). A position
  measurement is `meas_kind in POSITION_KINDS & usable_pos & ~far` (the same set gt_associate
  scores, kind == 'pos'; usable_pos excludes cuas_clean's exact 'duplicate' rows). e / n are in
  C.iteration_frame(it).
* Heights (rule 1): h = sensor_noise.u_ellipsoid(det), the GT-free height above the WGS84
  ellipsoid, computed on the whole cleaned detection table (it infers site heights from other
  sensors' rows of the same iteration), then u_ell = C.iteration_frame(it).to_enu(lat, lon, h)[2],
  the iteration ENU up coordinate (the scorer compares ENU computed from lat / lon /
  altitude_m). NaN h (unknown altitude, alt == 0 sentinel, pos2d sensors such as SMFS, sensors
  whose altitude does not follow the drone such as STUG) -> the row has no height. The height as
  reported (u_c) is kept only as features (u_raw_med_m, u_med_m, u_std_m) and in the vertical
  error report.
* Clock: t_cal = t + value for data/derived/sensor_calibration.csv rows with
  param == 'time_offset_s', status == 'applied', applies_to == 'all' (read, not hard-coded).
* Grid: C.template_grid(it)[1] (unix seconds, 100 ms).
* Noise (rule 2), per sensor_key and axis, two independent parts:
  (a) white part sigma_white: the GT-free second-difference estimator on TESTING segments only
      (fixed constants applied to every iteration; nothing is fitted on evaluation). For three
      consecutive measurement epochs of one segment (t0 < t1 < t2, both gaps <= WHITE_MAX_DT_S)
      r = (z1 - (w0 z0 + w2 z2)) / sqrt(1 + w0^2 + w2^2), w0 = (t2-t1)/(t2-t0), w2 = (t1-t0)/(t2-t0),
      i.e. the miss of the middle report from the straight line through its neighbours,
      normalised so that white noise of sd s gives sd(r) = s; a smooth trajectory and a slow
      bias cancel. sigma_white = 1.4826 * MAD(r) pooled over the sensor's testing triples (the
      vertical axis on u_ell, epochs with a height only). Fewer than WHITE_MIN_N triples (or no
      testing data) -> DEFAULT_SIGMA_H_M / DEFAULT_SIGMA_U_M. Floored at SIGMA_FLOOR_M (a relay's
      sigma 0 would make the smoother singular at a dt = 0 grid epoch, and Dronetag itself is
      ~3 m accurate). Written to white_noise.csv. The filter's measurement variance is
      R = sigma_white^2.
  (b) slow part: var_slow = max(sigma_total^2 - sigma_white^2, 0), sigma_total = sigma_e_m /
      sigma_n_m / sigma_u_m from data/derived/sensor_noise.csv (a row with this iter_key wins
      over an iteration-free / '*' row; blank sigma_e / sigma_n -> no row; a blank sigma_u alone
      -> DEFAULT_SIGMA_U_M), floored at SIGMA_FLOOR_M. Without a row: the GT-feed relays
      (Q.GT_FEED_SENSORS: VKYY, QTVS; the table leaves them blank, n < 30) get
      sigma_total = SIGMA_FLOOR_M (they agree with Dronetag to ~0.1 m, kept from version 1);
      every other sensor gets DEFAULT_SIGMA_H_M (e and n) and DEFAULT_SIGMA_U_M.
  REPORTED variance: var_e / var_n / var_u in grid.npz = smoothed (RTS) variance + var_slow.
  It is meant as the total error variance of the smoothed position (the smoother removes the
  white part only; the slow bias is untouched by smoothing) and as the weight a fusion uses. It
  is NOT calibrated: sensor_noise.csv truncates residuals at a 50 m gate (its sigma is a lower
  bound near the gate), and the run's calibration report (testing, in-sample) shows several
  sensors over-confident (variance too small: IUKL, STUG, EGOI, UXYJ at the 2026-10-28 run)
  and the GT relays under-confident. A fusion should re-calibrate per sensor (LOIO).
* Fitted upstream on ALL six testing iterations with ground truth (not leave-one-iteration-out):
  sigma_total (sensor_noise.csv sigma_e/n/u_m), the u_ellipsoid height offsets (sensor_noise.csv
  u_offset_m) and the clock offsets (sensor_calibration.csv). sensor_noise.csv has no iter_key
  column, so every iteration gets the same values. Applied to testing iteration k they were fitted
  WITH k: the calibration and vertical-error reports below are in-sample (optimistic), and a
  downstream LOIO validation that uses var_* or u from testing grid.npz inherits this. The
  vertical report also prints sensor_noise.csv loio_held_out_u_m (the held-out mean |median
  height bias| per iteration, a different metric) as the honest transfer reference.
  total_sigma_table already prefers a row whose iter_key equals the iteration, so per-held-out
  rows in sensor_noise.csv would be picked up without a change here.
  The variant corrections below (height_loio, var_loio, clock_loio) remove this in-sample use.
  `noise_src` records where sigma_total came from, `white_src` where sigma_white came from.
  Measurement noise is diagonal in (e, n), so the two horizontal axes decouple exactly and
  `cov_en` is 0 (kept in the schema for a future non-diagonal model).
* Segments: identity (iter_key, sensor_key, object_id), sorted by t_cal, split where two
  consecutive measurements are more than SPLIT_GAP_S apart. Measured within-object gaps
  (2026-10-28, all iterations): gaps > 30 s are 5 of 3,811 on GT-matched EGOI tracks and 0 on
  matched NYTF / IUKL / UXYJ / EUIG tracks, while 31 matched EGOI gaps lie in 20-30 s; 30 s
  keeps drone tracks whole. Rows of one segment at an identical t_cal are averaged into one
  measurement epoch (counted in `n_dup_merged`; the variance is NOT divided by the count).
  cuas_clean's 'duplicate' rows are removed before this (usable_pos): same sensor, object,
  t rounded to 1 ms, position to 1e-7 deg / 1 cm and bearing to 1e-3 deg. Most are copies at an
  identical timestamp (version 1 merged them here), but some (EGOI, LJJX; all but 4 / 5 of the
  dropped rows in evaluation DI / DII) are the same position stamped up to ~1 ms apart, which
  version 1 kept as separate epochs; the run prints both counts per iteration. The remaining
  n_dup_merged rows are distinct reports at one identical timestamp.
* Filter: per segment and axis a white-noise-acceleration constant-velocity model, state
  (position, velocity), process-noise density Q_ACC_H_M2S3 for e and n, Q_ACC_U_M2S3 for u:
  Q(dt) = q [[dt^3/3, dt^2/2], [dt^2/2, dt]]. From groundtruth.csv.gz (airborne 1 s Dronetag
  steps, testing) the MLE q = mean(dv^2 / dt) is 2.67 (ve), 2.55 (vn), 0.54 (vu) m^2/s^3, so
  2.5 and 0.5 (the run re-prints the estimate). Initialisation (broad prior, not a two-point
  difference): position mean = the first finite measurement with sd INIT_POS_SD_M (diffuse, so
  that measurement dominates), velocity 0 with sd INIT_VEL_SD_MS = 50 (weakly informative: the GT drone speeds
  reach p99 31 m/s; sd 20 pulled the first long gap of sparse segments toward rest, which
  failed check 5 on VKYY, 2026-10-28). The vertical filter runs only when the segment has >= 2 epochs with a u_ell
  height, else u / vu / var_u are NaN.
  The grid steps in [t0, t1 + EXTRAP_S] are inserted as no-measurement epochs (a grid step at
  the same time as a measurement comes after it), so each gets the smoothed state (after t1:
  the prediction). `--causal` keeps the forward filter only (no smoothing; for the case
  offline smoothing is not allowed) and writes the *_causal files.
* Labels (testing only; evaluation segments get empty labels): rows joined to
  data/derived/det_gt_match.csv.gz on (iter_key, file, row).
    n_matched     rows with matched == 1;  frac_matched = n_matched / n_rows
    best_flight   majority flight among the matched rows (ties: the smallest name)
    frac_best     matched rows of best_flight / n_matched
    frac_airborne mean gt_airborne over the matched rows of best_flight
    label         drone_air    if frac_matched >= 0.5 and frac_best >= 0.8 and frac_airborne >= 0.5
                  drone_ground if frac_matched >= 0.5 and frac_best >= 0.8 and frac_airborne <  0.5
                  clutter      otherwise
  On the GT-feed sensors (VKYY, QTVS, LJJX: `gt_feed` = 1) the label is leakage-prone.

Corrections (variant trees; none by default, and then nothing below runs: outputs are exactly the plain run's)
--------------------------------------------------------------------------------------------------------------
Read from C.DERIVED / variant.json when present ({"corrections": [...], "az_sensors": [...],
"drop_sensors": [...]}, written by run.py --variant-config); the CLI --corrections / --az-sensors /
--drop-sensors overrides it (no names = none). LOIO = leave-one-iteration-out: what is applied to testing
iteration k was fitted without k; evaluation uses values from all six testing iterations.
* height_loio: u_ellipsoid with a per-iteration rules dict (site heights from the whole table): testing k
  takes the held_out = k offset AND informativeness flag of sensor_noise_loio.csv (both refitted without
  k's ground truth; no such row: the sensor has no base rows in k, so its full fit never saw k), evaluation
  the full offset only where the pre-registered u_eval_apply = 1, else 0, and the full flag. Datum (the
  majority alt_datum label the sensor itself reports) and the AGL switch (GT-free: sensor_alt values) stay
  those of sensor_noise.csv. u_eval_apply is 0 for every sensor with only 2 supporting iterations (each fold
  has 1 left and the offset rule needs 2): ILYS_PASSIVE-RF loses its +41.6 m on evaluation.
* var_loio: sigma_total from sensor_noise_loio.csv held_out = k rows (testing k; '*' when the sensor has no
  base rows in k) and '*' rows (evaluation, = sensor_noise.csv); noise_src = 'sensor_noise_loio.csv'.
* clock_loio: for every sensor with an applied pooled time offset (UXYJ_RADAR; LJJX, testing only):
  testing k uses the median of the sensor's per-iteration time_offset_s scan values of the OTHER testing
  iterations (scan rows noted informative=no are not estimates and are skipped; none left -> 0);
  evaluation keeps the pooled applied value.
* az_rotate (az_sensors, e.g. EGOI_RADAR NYTF_RADAR): position rows (e, n), reported velocities (ve, vn) and
  az_deg of those sensors are rotated about their site (sensor_e, sensor_n) by delta_k = mean of the
  per-iteration radar_bias az_offset_deg rows of sensor_calibration.csv of the OTHER testing iterations
  (the angle to ADD to the azimuth of the reported position, clockwise-positive); evaluation: mean over
  all testing iterations. Applied before segmenting. lat / lon are not rotated, so u_ell keeps the
  unrotated earth-curvature term (< 2 cm); the far / duplicate flags of cuas_clean are those of the
  unrotated rows.
* drop_sensors (sensor ids or keys, e.g. VKYY QTVS): their position rows are removed before segmenting
  (after the heights, which need the whole table), so no segment of them exists downstream.
Extra self-checks of a run with corrections: 8 LOIO guard (each applied parameter recomputed with the csv
module from the source tables, without k for testing k; a leaky mutation must be flagged), 9 azimuth vs GT
(rotated median (GT az - derived az) = raw median - delta_k within AZ_CHECK_TOL_DEG per radar and testing
iteration; GT used by this check only), 10 synthetic +1.5 deg rotation recovered, 11 rotated rows keep their
range to the site, 12 no segment of a dropped sensor, 13 the per-iteration height path with the plain
rules reproduces the whole-table h_ell exactly.

Outputs (--out DIR, default data/derived/sensor_tracks_v2/; slug = C.iter_slug(key))
-------------------------------------------------------------------------------------
<slug>__tracks.csv   one row per segment (seg_id ascending = sorted by sensor_key, object_id, part):
    seg_id         int, unique within the iteration ((iter_key, seg_id) is global)
    iter_key, sensor_key, sensor_id, object_id
    part           0, 1, ... = index of the segment within its (sensor_key, object_id)
    t0, t1         first / last measurement time (t_cal, unix s)
    n_meas         measurement epochs (after merging identical timestamps)
    n_rows         detection rows (= n_meas + n_dup_merged)
    n_dup_merged   rows merged into an earlier row at an identical timestamp
    has_u          1 if the vertical filter ran (>= 2 epochs with a u_ell height)
    n_grid         grid samples written for the segment
    noise_src      source of sigma_total: 'sensor_noise.csv', 'gt_feed_floor' or 'default'
    white_src      source of sigma_white: 'testing' (estimated) or 'default'
    features (all NaN when undefined):
      duration_s     t1 - t0
      meas_rate_hz   n_meas / duration_s
      t0_rel_s, t1_rel_s  t0 / t1 minus the first grid time
      path_len_m     sum of the smoothed horizontal steps between measurement epochs
      speed_mean_ms, speed_p10_ms, speed_p90_ms, speed_max_ms
                     mean / 10th percentile / 90th percentile / max of the smoothed horizontal
                     speed at the measurement epochs
      vu_abs_p90_ms  90th percentile of |smoothed vertical speed| at the measurement epochs
                     (NaN when has_u = 0)
      accel_p90_ms2  p90 of |dv| / dt between consecutive measurement epochs (smoothed velocity)
      turn_p90_degs  p90 of |d heading| / dt where both speeds >= STATIONARY_MS
      stationary_share  share of measurement epochs with smoothed speed < STATIONARY_MS
      (the kinematic features above need n_meas >= 2; they come from the smoothed state at
       the measurement epochs, so they exist also for segments outside the grid span)
      dz_med_m, dz_p10_m, dz_p90_m, dz_std_m   dz_sensor (alt - sensor_alt) over the rows
      u_ell_med_m, u_ell_p10_m, u_ell_p90_m    median / p10 / p90 over the rows of u_ell, the
                                 ENU up coordinate of the height above the ellipsoid (rule 1)
      u_raw_med_m                median over the rows of u_c, the ENU up of the height AS
                                 REPORTED (no datum / offset rule)
      u_med_m, u_std_m           median / std of u_c over the rows (version-1 names; u_med_m
                                 equals u_raw_med_m)
      range_med_m, range_min_m, range_max_m   horizontal distance to the own sensor
                                 (rows with sensor_pos_valid == 1; NaN for QTVS / VKYY)
      det_conf_mean              mean finite det_conf
      cls_uav, cls_uav_rotary, cls_uav_fixed, cls_uav_transmitter, cls_air_vehicle, cls_bird,
      cls_manned, cls_ground, cls_unknown, cls_blank
                                 share of rows per class, the sensor's own label: the subclass
                                 when it is set and not Unknown / Other, else the class
                                 (UAV Transmitter + EM Transmitter -> uav_transmitter; Animal,
                                 Bird -> bird; Manned * -> manned; Land Vehicle, Sea Vessel,
                                 Fixed Object, Home -> ground; Unknown, Other -> unknown;
                                 empty class -> blank). The shares sum to 1.
      cls_conf_mean              mean finite cls_conf
      has_model, has_serial, has_protocol, has_freq   1 if any row has a value
      model_top, serial_top, protocol_top   most common non-empty value (ties: smallest), '' if none
      gt_feed                    1 for VKYY / QTVS / LJJX
    labels: n_matched, frac_matched, best_flight, frac_best, frac_airborne, label
            (empty for evaluation)
<slug>__grid.npz     one entry per (segment, grid step), sorted by seg_id then step:
    seg_id (int32), step (int32, index into C.template_grid(it)), t (float64 unix s = grid[step]),
    e, n, u (m, local ENU of the iteration; u from u_ell), ve, vn, vu (m/s),
    var_e, var_n, var_u (m^2: smoothed variance + var_slow, the total error variance),
    cov_en (m^2, 0: decoupled axes); these ten are float32 on disk.
    u / vu / var_u are NaN when has_u = 0.
white_noise.csv      one row per sensor_key with position rows anywhere: sensor_key,
                     sigma_white_e_m, sigma_white_n_m, sigma_white_u_m (after the floor / default),
                     n (horizontal testing triples), method (estimator, gap limit, unfloored
                     estimates and vertical n, or 'default ...'). Identical for any --iter subset.
README.md            self-check verdict, noise model, segments / labels per sensor, error,
                     calibration and vertical tables; rewritten only by a run over all
                     iterations (an --iter subset leaves it). With corrections: a section
                     listing them, every applied value and the azimuth / per-iteration error tables.
corrections.csv      only with corrections: iter_key, sensor_key, correction ('<name>:<parameter>'),
                     value, source (one row per applied number; a plain run removes a stale one).

Self-check 5 (GT, testing): per sensor with >= CHECK5_MIN_GRID drone_air grid samples, the median
horizontal error of the smoothed grid samples <= CHECK5_RATIO x the median error of linear
interpolation of the raw measurements at the same grid samples, and pooled smoothed <= linear
both over all sensors and over the deployable sensors only (C.deployable_sensor_ids(): without
the testing-only GT relay LJJX, which carries the all-sensor pooled gain). Measured 2026-10-28:
on deployable sensors the smoother is only ~0.3 % better than linear interpolation (the feeds
are already tracker outputs); its value is the velocity and the variance, not accuracy.
Reported, not gating: covariance calibration (z = (err_e^2/var_e + err_n^2/var_n)/2 per grid
sample; a calibrated Gaussian gives z ~ chi2(2)/2: mean 1, median ln 2 = 0.693) and the
vertical error of heights as reported vs after rule 1 (both in-sample, see above).

Run: python3 tools/sensor_tracks.py [--iter KEY ...] [--causal] [--out DIR]   (default: all iterations)
     [--corrections NAME ...] [--az-sensors SENSOR_KEY ...] [--drop-sensors SENSOR_ID ...]
Numbered self-checks at the end; exit status 1 on any failure.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuas_clean as Q  # noqa: E402
import cuas_common as C  # noqa: E402
import sensor_noise as SN  # noqa: E402

OUT_DEFAULT = C.DERIVED / "sensor_tracks_v2"
OUT_V1 = C.DERIVED / "sensor_tracks"          # version-1 outputs, read only (n_dup_merged comparison)
CALIBRATION = C.DERIVED / "sensor_calibration.csv"
NOISE = C.DERIVED / "sensor_noise.csv"
NOISE_LOIO = C.DERIVED / "sensor_noise_loio.csv"           # leave-one-iteration-out refits (corrections only)
NOISE_BY_ITER = C.DERIVED / "sensor_noise_by_iteration.csv"  # read by the LOIO guard (check 8) only
VARIANT_JSON = C.DERIVED / "variant.json"                   # corrections of a variant tree (run.py --variant-config)
GT_MATCH = C.DERIVED / "det_gt_match.csv.gz"

SPLIT_GAP_S = 30.0          # split a sensor track where consecutive measurements are further apart
EXTRAP_S = 1.0              # grid samples up to t1 + EXTRAP_S (prediction)
Q_ACC_H_M2S3 = 2.5          # horizontal white-noise acceleration density (GT MLE 2.55-2.67)
Q_ACC_U_M2S3 = 0.5          # vertical (GT MLE 0.54)
DEFAULT_SIGMA_H_M = 20.0    # measurement sd per horizontal axis when sensor_noise.csv has no row
DEFAULT_SIGMA_U_M = 30.0
SIGMA_FLOOR_M = 1.0         # lower bound on any sigma; also the sigma of a GT-feed relay without a table row
INIT_POS_SD_M = 1000.0      # diffuse position prior around the first measurement
INIT_VEL_SD_MS = 50.0       # broad velocity prior, mean 0 (> p99 GT speed 31 m/s)
STATIONARY_MS = 1.0
AIR_GT_H_M = 5.0            # q estimate: GT steps above this height (as gt_associate's airborne)
LABEL_FRAC_MATCHED = 0.5
LABEL_FRAC_BEST = 0.8
LABEL_FRAC_AIR = 0.5
WHITE_MAX_DT_S = 2.0        # second-difference triples: both gaps <= this (keeps the trajectory term small)
WHITE_MIN_N = 30            # fewer testing triples: sigma_white = default
CHECK5_MIN_GRID = 500       # check 5 gates sensors with at least this many drone_air grid samples
CHECK5_RATIO = 1.05         # smoothed median <= ratio x linear-interpolation median
CALIB_REF_MEAN, CALIB_REF_MED = 1.0, float(np.log(2.0))      # chi2(2)/2: mean 1, median ln 2
CALIB_U_REF_MED = 0.4549364231195724                        # chi2(1) median (err_u^2 / var_u)
CALIB_FACTOR = 1.5          # report wording only: median z beyond ln 2 x / / 1.5 -> 'over' / 'under'-confident
SEED = 0
CORRECTIONS = ("height_loio", "var_loio", "clock_loio", "az_rotate", "drop_sensors")
AZ_CHECK_TOL_DEG = 0.02     # check 9: rotated median (GT az - derived az) = raw median - delta within this
RADAR_BIAS_MAX_RESID_M = 50.0   # check 9 row set = gt_associate's radar_bias rows: matched, resid <= 50 m,
RADAR_BIAS_MIN_RANGE_M = 300.0  # GT range from the radar >= 300 m

DET_COLS = ["iter_key", "file", "row", "object_id", "t", "det_conf", "cls", "cls_conf", "cls_sub",
            "model", "serial", "protocol", "freq_mhz", "alt_datum", "sensor_alt_datum"] + Q._CLEAN_NEEDS

CLASS_SHARES = ("uav", "uav_rotary", "uav_fixed", "uav_transmitter", "air_vehicle", "bird",
                "manned", "ground", "unknown", "blank")
_CLASS_MAP = {
    "UAV": "uav", "FIXED_WING": "uav_fixed", "UAV Rotary Wing": "uav_rotary",
    "UAV Fixed Wing": "uav_fixed", "UAV Transmitter": "uav_transmitter",
    "EM Transmitter": "uav_transmitter", "Air Vehicle": "air_vehicle", "Animal": "bird",
    "Bird": "bird", "Manned Fixed Wing": "manned", "Manned Rotary Wing": "manned",
    "Land Vehicle": "ground", "Sea Vessel": "ground", "Fixed Object": "ground", "Home": "ground",
    "Unknown": "unknown", "Other": "unknown",
}

FEATURES = ["duration_s", "meas_rate_hz", "t0_rel_s", "t1_rel_s", "path_len_m", "speed_mean_ms",
            "speed_p10_ms", "speed_p90_ms", "speed_max_ms", "vu_abs_p90_ms", "accel_p90_ms2", "turn_p90_degs",
            "stationary_share", "dz_med_m", "dz_p10_m", "dz_p90_m", "dz_std_m", "u_ell_med_m", "u_ell_p10_m",
            "u_ell_p90_m", "u_raw_med_m", "u_med_m", "u_std_m", "range_med_m",
            "range_min_m", "range_max_m", "det_conf_mean"] + [f"cls_{c}" for c in CLASS_SHARES] + \
           ["cls_conf_mean", "has_model", "has_serial", "has_protocol", "has_freq", "model_top",
            "serial_top", "protocol_top", "gt_feed"]
LABELS = ["n_matched", "frac_matched", "best_flight", "frac_best", "frac_airborne", "label"]
GRID_F32 = ("e", "n", "u", "ve", "vn", "vu", "var_e", "var_n", "var_u", "cov_en")   # stored as float32 (~1 mm at 15 km)
HEAD = ["seg_id", "iter_key", "sensor_key", "sensor_id", "object_id", "part", "t0", "t1", "n_meas",
        "n_rows", "n_dup_merged", "has_u", "n_grid", "noise_src", "white_src"]


def log(msg: str) -> None:
    print(msg, flush=True)


def md_table(header, rows) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


# --------------------------------------------------------------------------- #
# Batched one-axis constant-velocity Kalman filter + RTS smoother
# --------------------------------------------------------------------------- #


def cv_axis(t: np.ndarray, z: np.ndarray, r: np.ndarray, start: np.ndarray, length: np.ndarray,
            q: float, causal: bool = False) -> tuple[np.ndarray, ...]:
    """One axis of a white-noise-acceleration CV model for many independent lanes at once.

    Lane j owns the epochs start[j] .. start[j] + length[j] - 1 of the flat arrays, sorted by
    time. z is the measured position (NaN = no measurement at that epoch), r its variance.
    Prior per lane: position = the lane's first finite z (sd INIT_POS_SD_M), velocity 0
    (sd INIT_VEL_SD_MS), at the lane's first epoch. Returns (x, v, Ppp, Ppv, Pvv) per epoch:
    smoothed (RTS), or filtered when causal. Every lane needs >= 1 finite z.
    """
    N = len(t)
    fin = np.isfinite(z)
    lane_of = np.repeat(np.arange(len(start)), length)
    fidx = np.flatnonzero(fin)
    first_lane, first_pos = np.unique(lane_of[fidx], return_index=True)
    if len(first_lane) != len(start):
        raise ValueError("cv_axis: a lane has no finite measurement")
    m0 = z[fidx[first_pos]]

    order = np.argsort(-length, kind="stable")      # longest first: active lanes = a prefix
    st, ln, m0 = start[order], length[order], m0[order]
    L = int(ln[0]) if len(ln) else 0
    n_act = np.searchsorted(-ln, -np.arange(L), side="left")   # lanes with length > k

    xp, vp, ppp, ppv, pvv = (np.empty(N) for _ in range(5))     # predicted (prior at epoch)
    xf, vf, fpp, fpv, fvv = (np.empty(N) for _ in range(5))     # filtered
    zz = np.where(fin, z, 0.0)
    rr = np.where(fin, r, 1.0)
    for k in range(L):
        i = st[:n_act[k]] + k
        if k == 0:
            xp[i], vp[i] = m0, 0.0
            ppp[i], ppv[i], pvv[i] = INIT_POS_SD_M ** 2, 0.0, INIT_VEL_SD_MS ** 2
        else:
            j = i - 1
            dt = t[i] - t[j]
            xp[i] = xf[j] + dt * vf[j]
            vp[i] = vf[j]
            ppp[i] = fpp[j] + 2.0 * dt * fpv[j] + dt * dt * fvv[j] + q * dt ** 3 / 3.0
            ppv[i] = fpv[j] + dt * fvv[j] + q * dt * dt / 2.0
            pvv[i] = fvv[j] + q * dt
        has = fin[i]
        c, b, d = ppp[i], ppv[i], pvv[i]
        s = c + rr[i]
        k1 = np.where(has, c / s, 0.0)
        k2 = np.where(has, b / s, 0.0)
        y = np.where(has, zz[i] - xp[i], 0.0)
        xf[i] = xp[i] + k1 * y
        vf[i] = vp[i] + k2 * y
        fpp[i] = (1.0 - k1) * c
        fpv[i] = (1.0 - k1) * b
        fvv[i] = d - k2 * b
    if causal:
        return xf, vf, fpp, fpv, fvv

    # RTS, in place: slot k holds the filtered state until step k overwrites it with the smoothed one.
    for k in range(L - 2, -1, -1):
        i = st[:n_act[k + 1]] + k
        i1 = i + 1
        dt = t[i1] - t[i]
        a11 = fpp[i] + dt * fpv[i]          # A = Pf(k) F^T
        a12 = fpv[i]
        a21 = fpv[i] + dt * fvv[i]
        a22 = fvv[i]
        c, b, d = ppp[i1], ppv[i1], pvv[i1]
        det = c * d - b * b
        g11 = (a11 * d - a12 * b) / det     # G = A Pp(k+1)^-1
        g12 = (a12 * c - a11 * b) / det
        g21 = (a21 * d - a22 * b) / det
        g22 = (a22 * c - a21 * b) / det
        dx = xf[i1] - xp[i1]
        dv = vf[i1] - vp[i1]
        d11, d12, d22 = fpp[i1] - c, fpv[i1] - b, fvv[i1] - d
        t11 = g11 * d11 + g12 * d12
        t12 = g11 * d12 + g12 * d22
        t21 = g21 * d11 + g22 * d12
        t22 = g21 * d12 + g22 * d22
        nx = xf[i] + g11 * dx + g12 * dv
        nv = vf[i] + g21 * dx + g22 * dv
        npp = fpp[i] + t11 * g11 + t12 * g12
        npv = fpv[i] + t11 * g21 + t12 * g22
        nvv = fvv[i] + t21 * g21 + t22 * g22
        xf[i], vf[i], fpp[i], fpv[i], fvv[i] = nx, nv, npp, npv, nvv
    return xf, vf, fpp, fpv, fvv


def batch_wls(t: np.ndarray, z: np.ndarray, r: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force information-form MAP of one lane under the same model and prior as cv_axis:
    returns (mean (K, 2), covariance (2K, 2K)). Needs strictly increasing t."""
    K = len(t)
    J = np.zeros((2 * K, 2 * K))
    h = np.zeros(2 * K)
    fin = np.isfinite(z)
    m0 = np.array([z[np.flatnonzero(fin)[0]], 0.0])
    P0i = np.diag([1.0 / INIT_POS_SD_M ** 2, 1.0 / INIT_VEL_SD_MS ** 2])
    J[:2, :2] += P0i
    h[:2] += P0i @ m0
    for k in range(K - 1):
        dt = t[k + 1] - t[k]
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Qk = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        A = np.zeros((2, 2 * K))
        A[:, 2 * k:2 * k + 2] = -F
        A[:, 2 * k + 2:2 * k + 4] = np.eye(2)
        J += A.T @ np.linalg.inv(Qk) @ A
    for k in np.flatnonzero(fin):
        J[2 * k, 2 * k] += 1.0 / r[k]
        h[2 * k] += z[k] / r[k]
    cov = np.linalg.inv(J)
    return (cov @ h).reshape(K, 2), cov


# --------------------------------------------------------------------------- #
# Grouped statistics over contiguous groups (rows sorted by group)
# --------------------------------------------------------------------------- #


def group_quantile(g: np.ndarray, v: np.ndarray, n_groups: int, qs) -> np.ndarray:
    """Per-group quantiles (linear interpolation, as np.percentile) of finite v; NaN for groups
    without finite values. Returns (len(qs), n_groups)."""
    ok = np.isfinite(v)
    g, v = g[ok], v[ok]
    o = np.lexsort((v, g))
    g, v = g[o], v[o]
    cnt = np.bincount(g, minlength=n_groups)
    st = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    out = np.full((len(qs), n_groups), np.nan)
    has = cnt > 0
    for a, qv in enumerate(qs):
        pos = qv * (cnt[has] - 1)
        lo = np.floor(pos).astype(np.int64)
        hi = np.minimum(lo + 1, cnt[has] - 1)
        frac = pos - lo
        vl, vh = v[st[has] + lo], v[st[has] + hi]
        out[a, has] = vl + frac * (vh - vl)
    return out


def group_mean_std(g: np.ndarray, v: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(v)
    cnt = np.bincount(g[ok], minlength=n_groups).astype(float)
    s1 = np.bincount(g[ok], v[ok], minlength=n_groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / cnt
        dev = np.zeros(len(v))
        dev[ok] = (v[ok] - mean[g[ok]]) ** 2
        std = np.sqrt(np.bincount(g[ok], dev[ok], minlength=n_groups) / cnt)
    return mean, std


def group_top(g: np.ndarray, vals: np.ndarray, n_groups: int) -> list[str]:
    """Most common non-empty string per group (ties: smallest), '' if none."""
    out = [""] * n_groups
    cnt: Counter = Counter((int(a), b) for a, b in zip(g, vals) if b != "")
    best: dict[int, tuple[int, str]] = {}
    for (a, b), c in cnt.items():
        cur = best.get(a)
        if cur is None or c > cur[0] or (c == cur[0] and b < cur[1]):
            best[a] = (c, b)
    for a, (_, b) in best.items():
        out[a] = b
    return out


def class_share_of(cls: np.ndarray, sub: np.ndarray) -> np.ndarray:
    """Per row the CLASS_SHARES group of the sensor's own label (subclass when it says more)."""
    out = np.empty(len(cls), dtype=object)
    cache: dict[tuple[str, str], str] = {}
    for i, (c, s) in enumerate(zip(cls, sub)):
        key = (c, s)
        grp = cache.get(key)
        if grp is None:
            cn, sn = C.norm_class(c) if c else "", C.norm_class(s) if s else ""
            if sn and sn not in ("Unknown", "Other"):
                grp = _CLASS_MAP.get(sn, "unknown")
            elif cn:
                grp = _CLASS_MAP.get(cn, "unknown")
            else:
                grp = "blank"
            cache[key] = grp
        out[i] = grp
    return out


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def time_offsets() -> dict[str, float]:
    cal = C.read_columns(CALIBRATION, {"value"})
    m = (cal["param"] == "time_offset_s") & (cal["status"] == "applied") & (cal["applies_to"] == "all")
    return {str(k): float(v) for k, v in zip(cal["sensor_key"][m], cal["value"][m])}


def total_sigma_table(iter_key: str, sensor_keys, sigma_rows: dict | None = None,
                      src_label: str = "sensor_noise.csv") -> dict[str, tuple[float, float, float, str]]:
    """sensor_key -> (sigma_total_e, sigma_total_n, sigma_total_u, source), floored at SIGMA_FLOOR_M.
    `sigma_rows` ({sensor_key: (sigma_e, sigma_n, sigma_u)}, correction 'var_loio') replaces the
    sensor_noise.csv rows; the rest of the rule (blank -> floor / default) is the same."""
    rows: dict[str, tuple[int, float, float, float]] = {}
    if sigma_rows is not None:
        rows = {sk: (1, float(v[0]), float(v[1]), float(v[2])) for sk, v in sigma_rows.items()}
    elif NOISE.exists():
        tab = C.read_columns(NOISE, {"sigma_e_m", "sigma_n_m", "sigma_u_m"})
        iks = tab.get("iter_key", np.array([""] * len(tab["sensor_key"]), dtype=object))
        for sk, ik, se, sn, su in zip(tab["sensor_key"], iks, tab["sigma_e_m"], tab["sigma_n_m"], tab["sigma_u_m"]):
            prio = 2 if ik == iter_key else (1 if ik in ("", "*", "all") else 0)
            if prio and (sk not in rows or prio > rows[sk][0]):
                rows[sk] = (prio, se, sn, su)
    out = {}
    for sk in sensor_keys:
        if sk in rows and np.isfinite(rows[sk][1]) and np.isfinite(rows[sk][2]):
            _, se, sn, su = rows[sk]
            su = float(su) if np.isfinite(su) else DEFAULT_SIGMA_U_M
            out[sk] = (max(float(se), SIGMA_FLOOR_M), max(float(sn), SIGMA_FLOOR_M), max(su, SIGMA_FLOOR_M),
                       src_label)
        elif str(sk).split("_", 1)[0] in Q.GT_FEED_SENSORS:
            out[sk] = (SIGMA_FLOOR_M, SIGMA_FLOOR_M, SIGMA_FLOOR_M, "gt_feed_floor")
        else:
            out[sk] = (DEFAULT_SIGMA_H_M, DEFAULT_SIGMA_H_M, DEFAULT_SIGMA_U_M, "default")
    return out


def noise_table(iter_key: str, sensor_keys, white: dict, sigma_rows: dict | None = None) -> dict[str, dict]:
    """sensor_key -> dict(white=(e, n, u) sd, slow=(e, n, u) variance, src, white_src).
    `white` = estimate_white(...) (sensors missing there get the defaults); `sigma_rows`: see total_sigma_table."""
    tot = (total_sigma_table(iter_key, sensor_keys) if sigma_rows is None
           else total_sigma_table(iter_key, sensor_keys, sigma_rows, NOISE_LOIO.name))
    out = {}
    for sk in sensor_keys:
        w = white.get(sk)
        wsd = w["sigma"] if w is not None else (DEFAULT_SIGMA_H_M, DEFAULT_SIGMA_H_M, DEFAULT_SIGMA_U_M)
        wsrc = w["src"] if w is not None else "default"
        slow = tuple(max(tot[sk][c] ** 2 - wsd[c] ** 2, 0.0) for c in range(3))
        out[sk] = dict(white=wsd, slow=slow, total=tot[sk][:3], src=tot[sk][3], white_src=wsrc)
    return out


def second_diff(t: np.ndarray, z: np.ndarray, seg: np.ndarray, max_dt: float) -> np.ndarray:
    """Normalised second differences of the epochs (sorted by seg, t): for each three consecutive
    epochs of one segment with both gaps in (0, max_dt] and finite z,
    r = (z1 - (w0 z0 + w2 z2)) / sqrt(1 + w0^2 + w2^2) (white noise of sd s -> sd(r) = s)."""
    ok = np.isfinite(z)
    t, z, seg = t[ok], z[ok], seg[ok]
    if len(t) < 3:
        return np.empty(0)
    da, db = t[1:-1] - t[:-2], t[2:] - t[1:-1]
    use = (seg[:-2] == seg[1:-1]) & (seg[1:-1] == seg[2:]) & (da > 0) & (db > 0) & (da <= max_dt) & (db <= max_dt)
    da, db = da[use], db[use]
    w0, w2 = db / (da + db), da / (da + db)
    r = z[1:-1][use] - w0 * z[:-2][use] - w2 * z[2:][use]
    return r / np.sqrt(1.0 + w0 ** 2 + w2 ** 2)


def mad_sigma(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


def estimate_white(preps: list[dict], all_keys) -> tuple[dict, dict]:
    """sigma_white per sensor_key from the TESTING preps (GT-free). Returns (white, table) with
    white[sk] = dict(sigma=(e, n, u) after floor / default, src) and table = white_noise.csv columns."""
    res: dict[str, list[list[np.ndarray]]] = {}
    for p in preps:
        if not p["testing"]:
            continue
        sk_ep = p["seg_sk"][p["me_seg"]]
        for sk in np.unique(sk_ep):
            m = sk_ep == sk
            acc = res.setdefault(str(sk), [[], [], []])
            for c, z in enumerate((p["me_e"], p["me_n"], p["me_u"])):
                acc[c].append(second_diff(p["me_t"][m], z[m], p["me_seg"][m], WHITE_MAX_DT_S))
    white, tab = {}, {k: [] for k in ("sensor_key", "sigma_white_e_m", "sigma_white_n_m", "sigma_white_u_m",
                                       "n", "method")}
    for sk in sorted(set(all_keys) | set(res)):
        r = [np.concatenate(x) if x else np.empty(0) for x in res.get(sk, [[], [], []])]
        nh, nu = min(len(r[0]), len(r[1])), len(r[2])
        raw = [mad_sigma(x) for x in r]
        if nh >= WHITE_MIN_N:
            se, sn = max(raw[0], SIGMA_FLOOR_M), max(raw[1], SIGMA_FLOOR_M)
            su = max(raw[2], SIGMA_FLOOR_M) if nu >= WHITE_MIN_N else DEFAULT_SIGMA_U_M
            src = "testing"
            meth = (f"second_difference_mad testing gaps<={WHITE_MAX_DT_S:g}s; unfloored e {raw[0]:.3f} n {raw[1]:.3f} "
                    f"u {raw[2]:.3f} m; n_u {nu}; floor {SIGMA_FLOOR_M:g} m"
                    + ("" if nu >= WHITE_MIN_N else f"; u default (n_u < {WHITE_MIN_N})"))
        else:
            se, sn, su = DEFAULT_SIGMA_H_M, DEFAULT_SIGMA_H_M, DEFAULT_SIGMA_U_M
            src = "default"
            meth = f"default (testing triples {nh} < {WHITE_MIN_N})"
        white[sk] = dict(sigma=(se, sn, su), src=src, raw=raw, n=nh, n_u=nu)
        for k, v in zip(tab, (sk, se, sn, su, nh, meth)):
            tab[k].append(v)
    for k in ("sigma_white_e_m", "sigma_white_n_m", "sigma_white_u_m"):
        tab[k] = np.array(tab[k], dtype=float)
    return white, tab


def gt_q_estimate(gt: dict) -> tuple[float, float, float, int]:
    """MLE white-noise-acceleration density from 1 s airborne Dronetag velocity steps."""
    o = np.lexsort((gt["t"], gt["flight_key"]))
    fk, t, h = gt["flight_key"][o], gt["t"][o], gt["height"][o]
    dt = np.diff(t)
    ok = (fk[1:] == fk[:-1]) & (h[1:] > AIR_GT_H_M) & (h[:-1] > AIR_GT_H_M) & (dt > 0.5) & (dt < 1.5)
    est = [float(np.mean(np.diff(gt[c][o])[ok] ** 2 / dt[ok])) for c in ("ve", "vn", "vu")]
    return est[0], est[1], est[2], int(ok.sum())


# --------------------------------------------------------------------------- #
# Corrections (variant trees): honest leave-one-iteration-out calibration, radar azimuth rotation,
# sensor drop. With none selected nothing below runs and the outputs are exactly the plain run's.
# --------------------------------------------------------------------------- #


def load_corrections(args) -> dict:
    """{'names': sorted tuple, 'az_sensors': tuple, 'drop_sensors': tuple, 'origin': str}. The CLI
    (--corrections / --az-sensors / --drop-sensors) overrides variant.json; neither -> none."""
    if args.corrections is not None or args.az_sensors or args.drop_sensors:
        cfg = dict(corrections=args.corrections or [], az_sensors=args.az_sensors or [],
                   drop_sensors=args.drop_sensors or [])
        origin = "command line"
    elif VARIANT_JSON.is_file():
        import json
        cfg = json.loads(VARIANT_JSON.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise SystemExit(f"{VARIANT_JSON}: expected a JSON object")
        origin = str(VARIANT_JSON)
    else:
        return dict(names=(), az_sensors=(), drop_sensors=(), origin="none")
    names = set(cfg.get("corrections", []))
    az, drop = tuple(cfg.get("az_sensors", [])), tuple(cfg.get("drop_sensors", []))
    unknown = names - set(CORRECTIONS)
    if unknown:
        raise SystemExit(f"unknown corrections {sorted(unknown)} ({origin}); supported: {', '.join(CORRECTIONS)}")
    if "az_rotate" in names and not az:
        raise SystemExit("az_rotate needs a non-empty az_sensors list")
    if az and "az_rotate" not in names:
        raise SystemExit("az_sensors given without the az_rotate correction")
    if "drop_sensors" in names and not drop:
        raise SystemExit("drop_sensors named without a list of sensor ids")
    if drop:
        names.add("drop_sensors")
    return dict(names=tuple(c for c in CORRECTIONS if c in names), az_sensors=az, drop_sensors=drop, origin=origin)


def _short(ik: str) -> str:
    return ik.split("/", 1)[1] if "/" in ik else ik


def read_loio() -> dict[tuple[str, str], dict]:
    """(sensor_key, held_out) -> row of sensor_noise_loio.csv (floats; NaN = blank)."""
    if not NOISE_LOIO.exists():
        raise SystemExit(f"{NOISE_LOIO} missing: run tools/sensor_noise.py in this tree first")
    tab = C.read_columns(NOISE_LOIO, {"u_offset_m", "sigma_e_m", "sigma_n_m", "sigma_u_m", "u_eval_apply", "n",
                                      "u_informative"})
    return {(sk, h): dict(u_offset_m=float(o), sigma=(float(se), float(sn), float(su)), u_eval_apply=a, n=n,
                          informative=int(fi) != 0)
            for sk, h, o, se, sn, su, a, n, fi in zip(tab["sensor_key"], tab["held_out"], tab["u_offset_m"],
                                                      tab["sigma_e_m"], tab["sigma_n_m"], tab["sigma_u_m"],
                                                      tab["u_eval_apply"], tab["n"], tab["u_informative"])}


def calibration_rows() -> dict[str, np.ndarray]:
    return C.read_columns(CALIBRATION, {"value"})


def clock_plan(keys: list[str], testing_keys: list[str], active: bool) -> dict[str, dict[str, tuple[float, str]]]:
    """iter_key -> {sensor_key: (t offset s, source)}. Today: the pooled applied offsets for every
    iteration. clock_loio: testing k gets the median of the sensor's per-iteration scan values of the
    OTHER testing iterations (rows whose note says informative=no are not estimates and are skipped;
    none left -> 0); evaluation keeps the pooled applied value."""
    pooled = time_offsets()
    plan = {k: {sk: (v, "pooled applied (sensor_calibration.csv)") for sk, v in pooled.items()} for k in keys}
    if not active:
        return plan
    cal = calibration_rows()
    informative = np.array(["informative=no" not in str(x) for x in cal["note"]], dtype=bool)
    m = (cal["param"] == "time_offset_s") & (cal["source"] == "scan") & np.char.startswith(
        cal["iter_key"].astype(str), "testing/") & informative
    for k in keys:
        if k not in testing_keys:
            continue
        for sk in pooled:
            sel = m & (cal["sensor_key"] == sk) & (cal["iter_key"] != k)
            vals = cal["value"][sel]
            if len(vals):
                plan[k][sk] = (float(np.median(vals)), "median of scan rows " + ", ".join(
                    _short(i) for i in sorted(cal["iter_key"][sel].tolist())))
            else:
                plan[k][sk] = (0.0, "no informative scan row outside this iteration")
    return plan


def height_plan(keys: list[str], testing_keys: list[str], loio: dict | None) -> dict[str, dict]:
    """iter_key -> u_ellipsoid rules dict with per-iteration offsets (correction height_loio): testing k
    takes the held_out = k offset and informativeness flag of sensor_noise_loio.csv (no such row: the sensor
    has no base rows in k, so its full fit never saw k), evaluation the full offset where the pre-registered
    u_eval_apply is 1, else 0, and the full flag. Datum and AGL switch stay those of sensor_noise.csv."""
    base = SN.load_rules(NOISE)
    plan = {}
    for k in keys:
        rules = {}
        for sk, r in base.items():
            r2 = dict(r)
            if k in testing_keys:
                row = loio.get((sk, k))
                r2["offset"] = row["u_offset_m"] if row is not None else r["offset"]
                r2["informative"] = row["informative"] if row is not None else r["informative"]
                r2["src"] = f"sensor_noise_loio.csv held_out={_short(k)}" if row is not None else \
                    "sensor_noise.csv (no base rows in this iteration)"
            else:
                apply = loio[(sk, "*")]["u_eval_apply"] == 1
                r2["offset"] = r["offset"] if apply else 0.0
                r2["src"] = "u_eval_apply = 1: full offset" if apply else "u_eval_apply = 0: offset 0"
            rules[sk] = r2
        plan[k] = rules
    return plan


def sigma_plan(keys: list[str], testing_keys: list[str], loio: dict) -> dict[str, tuple[dict, dict]]:
    """iter_key -> ({sensor_key: (sigma_e, sigma_n, sigma_u)}, {sensor_key: source}) for var_loio."""
    plan = {}
    for k in keys:
        rows, src = {}, {}
        for (sk, h), r in loio.items():
            if h != "*":
                continue
            if k in testing_keys and (sk, k) in loio:
                rows[sk], src[sk] = loio[(sk, k)]["sigma"], f"sensor_noise_loio.csv held_out={_short(k)}"
            else:
                rows[sk] = r["sigma"]
                src[sk] = ("sensor_noise_loio.csv * (no base rows in this iteration)" if k in testing_keys
                           else "sensor_noise_loio.csv * (all testing iterations)")
        plan[k] = (rows, src)
    return plan


def radar_bias_values() -> dict[str, dict[str, float]]:
    """sensor_key -> {testing iter_key: radar_bias az_offset_deg} (the angle to ADD to the azimuth derived
    from the reported position about the radar site, gt_associate)."""
    cal = calibration_rows()
    m = (cal["param"] == "az_offset_deg") & (cal["source"] == "radar_bias") & (cal["iter_key"] != "*")
    out: dict[str, dict[str, float]] = {}
    for sk, ik, v in zip(cal["sensor_key"][m], cal["iter_key"][m], cal["value"][m]):
        out.setdefault(str(sk), {})[str(ik)] = float(v)
    return out


def az_plan(keys: list[str], testing_keys: list[str], sensors) -> dict[str, dict[str, tuple[float, str]]]:
    """iter_key -> {sensor_key: (delta deg, source)}: testing k = mean of the sensor's radar_bias values of
    the OTHER testing iterations; evaluation = mean over all testing iterations."""
    rb = radar_bias_values()
    plan = {k: {} for k in keys}
    for sk in sensors:
        vals = rb.get(sk)
        if not vals:
            raise SystemExit(f"az_rotate: no per-iteration radar_bias rows for {sk} in {CALIBRATION.name}")
        for k in keys:
            use = {i: v for i, v in vals.items() if (k not in testing_keys or i != k)}
            if not use:
                raise SystemExit(f"az_rotate: {sk} has radar_bias only in {k}; nothing to transfer")
            plan[k][sk] = (float(np.mean(list(use.values()))), "mean radar_bias of " + ", ".join(_short(i) for i in sorted(use)))
    return plan


def rotate_about(e, n, se, sn, delta_deg):
    """Rotate points (e, n) about (se, sn) so that their azimuth (clockwise from north) grows by delta_deg."""
    a = np.radians(delta_deg)
    c, s = np.cos(a), np.sin(a)
    de, dn = e - se, n - sn
    return se + c * de + s * dn, sn - s * de + c * dn


def rotate_vec(ve, vn, delta_deg):
    a = np.radians(delta_deg)
    c, s = np.cos(a), np.sin(a)
    return c * ve + s * vn, -s * ve + c * vn


def az_synth_check() -> tuple[bool, str]:
    """(10) A +1.5 deg azimuth bias injected on synthetic site / target geometry is recovered by
    median(GT az - derived az) and undone by rotate_about; the convention is clockwise-positive."""
    rng = np.random.default_rng(SEED + 3)
    N = 5000
    se, sn = rng.uniform(-800, 800, N), rng.uniform(-800, 800, N)
    rng_m, az = rng.uniform(300, 4000, N), rng.uniform(0, 360, N)
    te, tn = se + rng_m * np.sin(np.radians(az)), sn + rng_m * np.cos(np.radians(az))
    tve, tvn = rng.normal(0, 10, N), rng.normal(0, 10, N)
    inj = 1.5
    re_, rn_ = rotate_about(te, tn, se, sn, -inj)               # reported = truth seen 1.5 deg anticlockwise
    rve, rvn = rotate_vec(tve, tvn, -inj)
    est = float(np.median(C.angle_diff_deg(C.azimuth_deg(te - se, tn - sn), C.azimuth_deg(re_ - se, rn_ - sn))))
    ce, cn = rotate_about(re_, rn_, se, sn, est)
    cve, cvn = rotate_vec(rve, rvn, est)
    pos_err = float(np.max(np.hypot(ce - te, cn - tn)))
    vel_err = float(np.max(np.hypot(cve - tve, cvn - tvn)))
    ne, nn = rotate_about(np.array([0.0]), np.array([100.0]), 0.0, 0.0, 90.0)      # north + 90 deg cw -> east
    conv = abs(ne[0] - 100.0) < 1e-9 and abs(nn[0]) < 1e-9
    ok = abs(est - inj) < 1e-9 and pos_err < 1e-6 and vel_err < 1e-9 and conv
    return ok, (f"synthetic azimuth bias +{inj} deg on {N:,} targets at 300-4000 m from random sites: recovered "
                f"{est:+.9f} deg; after rotate_about max position error {pos_err:.1e} m (< 1e-6), velocity {vel_err:.1e} m/s; "
                f"north +90 deg -> east {'yes' if conv else 'NO'} (clockwise-positive)")


def _csv_dicts(path: Path) -> list[dict]:
    import csv
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def loio_guard(names, keys, testing_keys, applied: dict) -> list[str]:
    """(8) Recompute every parameter applied to each iteration from the source tables with the csv module
    (independent of the loaders above) and return the mismatches. Testing k: the value must be the one
    obtainable WITHOUT k; evaluation: only pooled / pre-registered values. Corrections that are not active
    must leave today's values. `applied[k]` = dict(toff, rules, sigma (or None), az)."""
    import statistics
    probs = []
    cal = _csv_dicts(CALIBRATION)
    pooled_t = {r["sensor_key"]: float(r["value"]) for r in cal if r["param"] == "time_offset_s"
                and r["status"] == "applied" and r["applies_to"] == "all"}
    sn = {r["sensor_key"]: r for r in _csv_dicts(NOISE)}
    loio = {(r["sensor_key"], r["held_out"]): r for r in _csv_dicts(NOISE_LOIO)} if NOISE_LOIO.exists() else {}
    byit: dict[str, dict[str, tuple[float, int]]] = {}
    for r in _csv_dicts(NOISE_BY_ITER):
        byit.setdefault(r["sensor_key"], {})[r["iter_key"]] = (float(r["bias_u_m"]) if r["bias_u_m"] else float("nan"),
                                                               int(r["n_u"]), int(r["n"]))
    rb: dict[str, dict[str, float]] = {}
    for r in cal:
        if r["param"] == "az_offset_deg" and r["source"] == "radar_bias" and r["iter_key"] != "*":
            rb.setdefault(r["sensor_key"], {})[r["iter_key"]] = float(r["value"])

    def fnum(x):
        return float(x) if x != "" else float("nan")

    for k in keys:
        a = applied[k]
        is_t = k in testing_keys
        # clock
        if set(a["toff"]) != set(pooled_t):
            probs.append(f"{_short(k)} clock sensors {sorted(a['toff'])} != pooled {sorted(pooled_t)}")
        for sk, v in a["toff"].items():
            exp = pooled_t.get(sk, float("nan"))
            if "clock_loio" in names and is_t:
                vals = [float(r["value"]) for r in cal if r["sensor_key"] == sk and r["param"] == "time_offset_s"
                        and r["source"] == "scan" and r["iter_key"].startswith("testing/") and r["iter_key"] != k
                        and "informative=no" not in r["note"]]
                exp = statistics.median(vals) if vals else 0.0
            if not abs(v - exp) <= 1e-12:
                probs.append(f"{_short(k)} clock {sk} applied {v:+.3f} expected {exp:+.3f}")
        # height offsets
        for sk, r in a["rules"].items():
            full = fnum(sn[sk]["u_offset_m"]) if sk in sn else 0.0
            exp = full
            if "height_loio" in names and sk in sn:
                if is_t:
                    if k in byit.get(sk, {}):
                        sup = {i: b for i, (b, nu, _) in byit[sk].items() if i != k and nu >= SN.MIN_ITER_ROWS and b == b}
                        exp = 0.0
                        lk = loio.get((sk, k))
                        if lk is None:
                            probs.append(f"{_short(k)} height {sk}: base rows in k but no held_out row")
                        inf_exp = lk is not None and lk["u_informative"] != "0"
                        if len(sup) >= 2 and inf_exp:
                            vals = list(sup.values())
                            before = sum(abs(x) for x in vals) / len(vals)
                            after = sum(abs(x - statistics.median([w for j, w in sup.items() if j != i]))
                                        for i, x in sup.items()) / len(sup)
                            exp = -statistics.median(vals) if after < before - 1.0 else 0.0
                    # else: the sensor has no base rows in k, its full fit never saw k (exp = full)
                else:
                    star = loio.get((sk, "*"))
                    if star is None or fnum(star["u_offset_m"]) != full:
                        probs.append(f"{_short(k)} height {sk}: '*' row missing or != sensor_noise.csv")
                    exp = full if star is not None and star["u_eval_apply"] == "1" else 0.0
            if not abs(r["offset"] - exp) <= 2e-3:      # by-iteration medians are rounded to 3 decimals
                probs.append(f"{_short(k)} height {sk} applied {r['offset']:+.3f} expected {exp:+.3f}")
            # informativeness flag: held_out = k row's on testing k with base rows in k, else the full fit's
            inf_full = sn[sk]["u_informative"] != "0" if sk in sn else bool(r["informative"])
            inf_e = inf_full
            if "height_loio" in names and is_t and sk in sn and k in byit.get(sk, {}) and (sk, k) in loio:
                inf_e = loio[(sk, k)]["u_informative"] != "0"
            if bool(r["informative"]) != inf_e:
                probs.append(f"{_short(k)} height {sk} informative {bool(r['informative'])} expected {inf_e}")
        # sigma_total
        if "var_loio" in names:
            for sk, sig in a["sigma"].items():
                row = loio.get((sk, k)) if is_t else None
                if is_t and row is None and k in byit.get(sk, {}):
                    probs.append(f"{_short(k)} sigma {sk}: sensor has base rows in k but no held_out row was used")
                if row is None:
                    row = loio.get((sk, "*"))
                    srow = sn.get(sk)
                    if srow is None or any(fnum(row[c]) != fnum(srow[c]) and not (fnum(row[c]) != fnum(row[c])
                                                                              and fnum(srow[c]) != fnum(srow[c]))
                                           for c in ("sigma_e_m", "sigma_n_m", "sigma_u_m")):
                        probs.append(f"{_short(k)} sigma {sk}: '*' row != sensor_noise.csv")
                elif int(row["n"]) != int(sn[sk]["n"]) - byit[sk][k][2]:      # rows of k really left out
                    probs.append(f"{_short(k)} sigma {sk}: held-out n {row['n']} != {sn[sk]['n']} - {byit[sk][k][2]}")
                exp = tuple(fnum(row[c]) for c in ("sigma_e_m", "sigma_n_m", "sigma_u_m"))
                if any(not (x == y or (x != x and y != y)) for x, y in zip(sig, exp)):
                    probs.append(f"{_short(k)} sigma {sk} applied {sig} expected {exp}")
        elif a["sigma"] is not None:
            probs.append(f"{_short(k)} sigma rows replaced although var_loio is off")
        # azimuth
        for sk, d in a["az"].items():
            vals = [v for i, v in rb.get(sk, {}).items() if not (is_t and i == k)]
            exp = statistics.fmean(vals) if vals else float("nan")
            if not abs(d - exp) <= 1e-12:
                probs.append(f"{_short(k)} az {sk} applied {d:+.4f} expected {exp:+.4f}")
        if "az_rotate" not in names and a["az"]:
            probs.append(f"{_short(k)} azimuth rotated although az_rotate is off")
    return probs


U_KEYS = ("iter_key", "sensor_key", "alt_c", "alt_datum", "sensor_alt", "sensor_alt_datum", "sensor_pos_valid")


def apply_corrections(det: dict, corr: dict, all_its: list) -> tuple[dict, dict]:
    """Apply the selected corrections to the cleaned detections (det["h_ell"] already = the plain rule on the
    whole table). Order: heights (per-iteration rules, site heights of the whole table), sensor drop, azimuth
    rotation. Returns (det, info) with the per-iteration parameter plans, the corrections.csv records and
    what the checks need."""
    names = set(corr["names"])
    keys = [i.key for i in all_its]
    tkeys = [i.key for i in all_its if i.has_groundtruth]
    loio = read_loio() if names & {"height_loio", "var_loio"} else None
    rec: list[tuple] = []
    pm = position_mask(det)
    present = {k: set(det["sensor_key"][(det["iter_key"] == k) & pm].tolist()) for k in keys}
    info = dict(names=names, keys=keys, tkeys=tkeys, records=rec, rot=[], h13=None, n_dropped=0, present=present)

    info["toff"] = clock_plan(keys, tkeys, "clock_loio" in names)
    if "clock_loio" in names:
        for k in keys:
            for sk, (v, src) in sorted(info["toff"][k].items()):
                if sk in present[k]:
                    rec.append((k, sk, "clock_loio:time_offset_s", v, src))

    base_rules = SN.load_rules(NOISE)
    if "height_loio" in names:
        rules_by = height_plan(keys, tkeys, loio)
        heights = SN.site_heights(det)
        h_new, h_full = np.full(len(det["t"]), np.nan), np.full(len(det["t"]), np.nan)
        for k in keys:
            m = det["iter_key"] == k
            sub = {c: det[c][m] for c in U_KEYS}
            h_new[m] = SN.u_ellipsoid(sub, rules_by[k], heights)
            h_full[m] = SN.u_ellipsoid(sub, base_rules, heights)
        ref = det["h_ell"]
        same_nan = bool(np.array_equal(np.isnan(h_full), np.isnan(ref)))
        both = np.isfinite(h_full) & np.isfinite(ref)
        info["h13"] = (same_nan, float(np.max(np.abs(h_full[both] - ref[both]))) if both.any() else 0.0, int(both.sum()),
                       int(np.sum(np.isfinite(h_new) & (h_new != ref))))
        det["h_ell"] = h_new
        for k in keys:
            for sk in sorted(rules_by[k]):
                if sk in present[k]:
                    rec.append((k, sk, "height_loio:u_offset_m", rules_by[k][sk]["offset"], rules_by[k][sk]["src"]))
                    rec.append((k, sk, "height_loio:u_informative", int(bool(rules_by[k][sk]["informative"])),
                                rules_by[k][sk]["src"]))
    else:
        rules_by = {k: base_rules for k in keys}
    info["rules"] = rules_by

    info["sigma"] = sigma_plan(keys, tkeys, loio) if "var_loio" in names else None
    if info["sigma"] is not None:
        for k in keys:
            rows, src = info["sigma"][k]
            for sk in sorted(rows):
                if sk in present[k]:
                    tot = total_sigma_table(k, [sk], rows, NOISE_LOIO.name)[sk]
                    for c, ax in enumerate("enu"):
                        rec.append((k, sk, f"var_loio:sigma_total_{ax}_m", tot[c], f"{src[sk]} -> {tot[3]}"))

    if "drop_sensors" in names:
        drop = list(corr["drop_sensors"])
        known = set(det["sensor_id"].tolist()) | set(det["sensor_key"].tolist())
        if set(drop) - known:
            raise SystemExit(f"drop_sensors: unknown sensor ids {sorted(set(drop) - known)}")
        dm = (np.isin(det["sensor_id"], drop) | np.isin(det["sensor_key"], drop)) & np.isin(det["meas_kind"], Q.POSITION_KINDS)
        for (k, sk), n in sorted(Counter(zip(det["iter_key"][dm].tolist(), det["sensor_key"][dm].tolist())).items()):
            rec.append((k, sk, "drop_sensors:rows_removed", n, "position rows removed before segmenting"))
        info["n_dropped"] = int(dm.sum())
        det = {c: v[~dm] for c, v in det.items()}

    info["az"] = az_plan(keys, tkeys, corr["az_sensors"]) if "az_rotate" in names else {k: {} for k in keys}
    if "az_rotate" in names:
        unknown = set(corr["az_sensors"]) - set(det["sensor_key"].tolist())
        if unknown:
            raise SystemExit(f"az_sensors: unknown sensor keys {sorted(unknown)}")
        for c in ("e", "n", "ve", "vn", "az_deg"):
            det[c] = np.array(det[c], dtype=float, copy=True)
        posk = np.isin(det["meas_kind"], Q.POSITION_KINDS)
        for k in keys:
            for sk, (d, src) in sorted(info["az"][k].items()):
                m = (det["iter_key"] == k) & (det["sensor_key"] == sk) & posk
                ok = m & (det["sensor_pos_valid"] == 1) & np.isfinite(det["e"]) & np.isfinite(det["n"])
                idx = np.flatnonzero(ok)
                if len(idx) == 0:
                    continue
                e0, n0 = det["e"][idx].copy(), det["n"][idx].copy()
                det["e"][idx], det["n"][idx] = rotate_about(e0, n0, det["sensor_e"][idx], det["sensor_n"][idx], d)
                iv = idx[np.isfinite(det["ve"][idx]) & np.isfinite(det["vn"][idx])]
                det["ve"][iv], det["vn"][iv] = rotate_vec(det["ve"][iv], det["vn"][iv], d)
                ia = idx[np.isfinite(det["az_deg"][idx])]
                det["az_deg"][ia] = (det["az_deg"][ia] + d) % 360.0
                n_skip = int(m.sum() - ok.sum())
                info["rot"].append(dict(key=k, sk=sk, delta=d, idx=idx, e0=e0, n0=n0, n_skip=n_skip))
                rec.append((k, sk, "az_rotate:delta_deg", d, src))
                rec.append((k, sk, "az_rotate:rows_rotated", len(idx),
                            f"{n_skip} position rows without a valid site left as reported"))
    return det, info


def applied_params(info: dict, leaky: bool = False) -> dict:
    """Per iteration the parameters the run applied (check 8). leaky=True is the mutation: every testing
    iteration gets the in-sample (all-testing) value of each active correction."""
    out = {}
    pooled = time_offsets()
    base_rules = SN.load_rules(NOISE)
    loio = read_loio() if info["sigma"] is not None else None
    rb = radar_bias_values()
    for k in info["keys"]:
        toff = {sk: v for sk, (v, _) in info["toff"][k].items()}
        rules = info["rules"][k]
        sigma = info["sigma"][k][0] if info["sigma"] is not None else None
        az = {sk: d for sk, (d, _) in info["az"][k].items()}
        if leaky and k in info["tkeys"]:
            if "clock_loio" in info["names"]:
                toff = dict(pooled)
            if "height_loio" in info["names"]:
                rules = base_rules
            if sigma is not None:
                sigma = {sk: loio[(sk, "*")]["sigma"] for sk in sigma}
            az = {sk: float(np.mean(list(rb[sk].values()))) for sk in az}
        out[k] = dict(toff=toff, rules=rules, sigma=sigma, az=az)
    return out


def az_gt_check(det: dict, info: dict, gt: dict) -> tuple[bool, str, list]:
    """(9) GT only here: per (radar, testing iteration) the median (GT az - derived az) over gt_associate's
    radar_bias row set, before / after the rotation, must satisfy rotated = raw - delta (within
    AZ_CHECK_TOL_DEG). Also the median horizontal error of the airborne matched rows before / after."""
    sens = sorted({r["sk"] for r in info["rot"]})
    tab = C.read_columns(GT_MATCH, {"row", "matched", "t_cal", "resid", "gt_airborne"},
                         usecols=["iter_key", "file", "row", "sensor_key", "kind", "matched", "best_flight", "t_cal",
                                  "resid", "gt_airborne"])
    m = (tab["kind"] == "pos") & np.isin(tab["sensor_key"], sens)
    lut = {(a, b, int(c)): i for i, (a, b, c) in enumerate(zip(tab["iter_key"][m], tab["file"][m], tab["row"][m]))}
    sub = {c: tab[c][m] for c in ("matched", "best_flight", "t_cal", "resid", "gt_airborne")}
    rb = radar_bias_values()
    rows, bad = [], []
    for r in info["rot"]:
        if r["key"] not in info["tkeys"]:
            continue
        idx = r["idx"]
        j = np.array([lut.get((r["key"], f, int(w)), -1) for f, w in zip(det["file"][idx], det["row"][idx])])
        has = j >= 0
        jj = np.where(has, j, 0)
        mt = has & (sub["matched"][jj] == 1) & np.isfinite(sub["t_cal"][jj])
        ge, gn = np.full(len(idx), np.nan), np.full(len(idx), np.nan)
        for f in np.unique(sub["best_flight"][jj][mt]):
            s_ = mt & (sub["best_flight"][jj] == f)
            fm = (gt["iter_key"] == r["key"]) & (gt["flight_key"] == f)
            ge[s_], gn[s_] = Q.interp_flight(gt, fm, sub["t_cal"][jj][s_], cols=("e", "n"))
        se, sn = det["sensor_e"][idx], det["sensor_n"][idx]
        e1, n1 = det["e"][idx], det["n"][idx]
        rng_gt = np.hypot(ge - se, gn - sn)
        with np.errstate(invalid="ignore"):
            selb = mt & (sub["resid"][jj] <= RADAR_BIAS_MAX_RESID_M) & (rng_gt >= RADAR_BIAS_MIN_RANGE_M)
        azg = C.azimuth_deg(ge - se, gn - sn)
        raw = C.angle_diff_deg(azg, C.azimuth_deg(r["e0"] - se, r["n0"] - sn))
        rot = C.angle_diff_deg(azg, C.azimuth_deg(e1 - se, n1 - sn))
        air = mt & (sub["gt_airborne"][jj] == 1) & np.isfinite(ge)
        h_raw = float(np.median(np.hypot(r["e0"] - ge, r["n0"] - gn)[air])) if air.any() else float("nan")
        h_rot = float(np.median(np.hypot(e1 - ge, n1 - gn)[air])) if air.any() else float("nan")
        if not selb.any():
            rows.append((r["key"], r["sk"], r["delta"], 0, float("nan"), float("nan"), float("nan"),
                         rb.get(r["sk"], {}).get(r["key"], float("nan")), int(air.sum()), h_raw, h_rot))
            continue
        rm, om = float(np.median(raw[selb])), float(np.median(rot[selb]))
        dev = abs(om - (rm - r["delta"]))
        if not dev <= AZ_CHECK_TOL_DEG:
            bad.append(f"{_short(r['key'])} {r['sk']} {dev:.3f}")
        rows.append((r["key"], r["sk"], r["delta"], int(selb.sum()), rm, om, dev,
                     rb.get(r["sk"], {}).get(r["key"], float("nan")), int(air.sum()), h_raw, h_rot))
    n_eval = sum(1 for x in rows if x[3] > 0)
    ok = not bad and n_eval > 0
    msg = (f"azimuth implementation vs GT on {n_eval} (radar, testing iteration) groups (gt_associate's radar_bias "
           f"rows: matched, resid <= {RADAR_BIAS_MAX_RESID_M:g} m, GT range >= {RADAR_BIAS_MIN_RANGE_M:g} m): rotated median "
           f"(GT az - derived az) = raw median - delta_k within {AZ_CHECK_TOL_DEG} deg; worst "
           f"{max((x[6] for x in rows if x[3] > 0), default=float('nan')):.4f} deg; failures {bad or 'none'}")
    return ok, msg, rows


def correction_checks(det: dict, info: dict, gt: dict | None) -> list[tuple[str, bool, str]]:
    """Self-checks 8-11 and 13 of a run with corrections (12, the sensor drop, runs after the writes)."""
    out = []
    names = info["names"]
    calib = names & {"height_loio", "var_loio", "clock_loio", "az_rotate"}
    probs = loio_guard(names, info["keys"], info["tkeys"], applied_params(info))
    mut = loio_guard(names, info["keys"], info["tkeys"], applied_params(info, leaky=True)) if calib else []
    # the rotation really applied to the rows: azimuth change about the site == delta
    worst_rot = 0.0
    for r in info["rot"]:
        se, sn = det["sensor_e"][r["idx"]], det["sensor_n"][r["idx"]]
        far = np.hypot(r["e0"] - se, r["n0"] - sn) > 1.0
        da = C.angle_diff_deg(C.azimuth_deg(det["e"][r["idx"]] - se, det["n"][r["idx"]] - sn),
                              C.azimuth_deg(r["e0"] - se, r["n0"] - sn))
        if far.any():
            worst_rot = max(worst_rot, float(np.max(np.abs(da[far] - r["delta"]))))
    n_par = sum(len(v) for v in (info["toff"].values()))
    out.append(("8", not probs and (not calib or len(mut) > 0) and worst_rot < 1e-6,
                f"LOIO guard ({', '.join(sorted(names))}): every parameter applied to each of {len(info['keys'])} iterations "
                f"recomputed from sensor_calibration.csv / sensor_noise*.csv with the csv module (testing k: without k; "
                f"evaluation: pooled or the pre-registered u_eval_apply; inactive corrections: today's values; "
                f"held-out sigma rows: n = full n - rows of k): {len(probs)} mismatch(es) {probs[:4] or ''}; mutation "
                f"(in-sample values on testing) flagged {len(mut)} (must be > 0 when a calibration correction is active); "
                f"rotation actually applied to the rows: max |azimuth change - delta| {worst_rot:.1e} deg "
                f"({n_par} clock entries checked)"))
    if "az_rotate" in names:
        if gt is not None:
            ok9, msg9, rows9 = az_gt_check(det, info, gt)
            info["az_rows"] = rows9
            out.append(("9", ok9, msg9))
        out.append(("10",) + az_synth_check())
        worst_r, n_r, n_skip = 0.0, 0, 0
        for r in info["rot"]:
            se, sn = det["sensor_e"][r["idx"]], det["sensor_n"][r["idx"]]
            d_ = np.abs(np.hypot(det["e"][r["idx"]] - se, det["n"][r["idx"]] - sn) - np.hypot(r["e0"] - se, r["n0"] - sn))
            worst_r = max(worst_r, float(d_.max()))
            n_r += len(r["idx"])
            n_skip += r["n_skip"]
        out.append(("11", n_r > 0 and worst_r < 1e-6,
                    f"{n_r:,} rotated position rows keep their horizontal range to the site: max |change| {worst_r:.1e} m "
                    f"(< 1e-6); {n_skip} position rows of the rotated sensors had no valid site and were left as reported"))
    if info["h13"] is not None:
        same_nan, d13, n13, n_ch = info["h13"]
        out.append(("13", same_nan and d13 == 0.0 and n13 > 0,
                    f"height_loio keeps the rest of u_ellipsoid: the per-iteration path (site heights of the whole table) "
                    f"with the plain sensor_noise.csv rules reproduces the whole-table h_ell exactly on {n13:,} rows "
                    f"(max |diff| {d13:.1e} m, NaN pattern {'equal' if same_nan else 'DIFFERENT'}); with the LOIO rules "
                    f"{n_ch:,} rows change"))
    return out


def per_iteration_error_report(errs_it: dict) -> str:
    """Median horizontal error of the smoothed drone_air grid samples per (sensor, testing iteration)."""
    keys = sorted({k for _, k in errs_it})
    sens = sorted({s for s, _ in errs_it})
    lines = ["    drone_air smoothed grid samples: median horizontal error (m) per sensor and testing iteration "
             "(n grid samples in brackets); compare variant trees with this table:",
             f"    {'sensor':6s} " + " ".join(f"{_short(k):>16s}" for k in keys)]
    for s in sens:
        cells = []
        for k in keys:
            v = errs_it.get((s, k))
            if v:
                g = np.concatenate(v)
                cells.append(f"{np.median(g):8.2f} ({len(g):>5d})" if len(g) else f"{'-':>16s}")
            else:
                cells.append(f"{'-':>16s}")
        lines.append(f"    {s:6s} " + " ".join(cells))
    return "\n".join(lines)


def write_corrections(out: Path, info: dict, keys: list[str]) -> None:
    rec = [r for r in info["records"] if r[0] in set(keys)]
    cols = {"iter_key": [r[0] for r in rec], "sensor_key": [r[1] for r in rec], "correction": [r[2] for r in rec],
            "value": np.array([float(r[3]) for r in rec], dtype=float), "source": [r[4] for r in rec]}
    C.write_columns(out / "corrections.csv", cols, ["iter_key", "sensor_key", "correction", "value", "source"])


def corr_section(info: dict, corr: dict, errs_it: dict) -> list[str]:
    """README section of a run with corrections."""
    L = ["## Corrections of this variant", "",
         f"Selected ({corr['origin']}): **{', '.join(corr['names'])}**"
         + (f"; az_sensors {', '.join(corr['az_sensors'])}" if corr["az_sensors"] else "")
         + (f"; drop_sensors {', '.join(corr['drop_sensors'])}" if corr["drop_sensors"] else "")
         + ". Every applied number per (iteration, sensor) is in `corrections.csv` (iter_key, sensor_key, correction, "
           "value, source). Leave-one-iteration-out (LOIO): a value applied to testing iteration k was fitted without "
           "k's ground truth; evaluation gets values fitted on all six testing iterations (or the pre-registered "
           "`u_eval_apply` decision of `sensor_noise_loio.csv` for the height offset).", "",
         "* `height_loio`: height offset of the ellipsoid rule and its informativeness flag (does the reported altitude "
         "follow the drone) refitted without k (`sensor_noise_loio.csv`); evaluation: the full offset only where "
         "`u_eval_apply` = 1, else 0 (a sensor with only 2 supporting iterations always gets 0, since each fold leaves 1 "
         "and the offset rule needs 2). The datum label and the AGL switch (both read from the sensor's own reports) "
         "are those of the full fit.",
         "* `var_loio`: total error spread (`sigma_total`) refitted without k; evaluation: the full table.",
         "* `clock_loio`: clock offsets of the sensors with an applied pooled offset = median of the per-iteration scan "
         "values of the other testing iterations; evaluation: the pooled value.",
         "* `az_rotate`: reported radar positions (and velocities) rotated about the radar site by delta_k = mean of the "
         "per-iteration radar azimuth bias (`radar_bias` rows of `sensor_calibration.csv`, the angle to ADD, "
         "clockwise-positive) of the other testing iterations; evaluation: the mean over all testing iterations.",
         "* `drop_sensors`: position rows of these sensors removed before segmenting.", ""]
    by = {}
    for k, sk, c, v, _ in info["records"]:
        by.setdefault((sk, c), {})[k] = v
    keys = info["keys"]
    rows = [[sk, c] + [fmt(by[(sk, c)][k], 3) if k in by[(sk, c)] else "-" for k in keys] for sk, c in sorted(by)]
    L += md_table(["sensor", "correction"] + [_short(k) for k in keys], rows)
    if info.get("az_rows"):
        L += ["", "Radar azimuth check (testing, GT used only by this check): median (GT azimuth - azimuth of the "
              "reported position) over gt_associate's radar_bias rows before / after the rotation, and the median "
              "horizontal error of the matched airborne rows (the same rows before and after).", ""]
        L += md_table(["iteration", "radar", "delta applied (deg)", "rows", "median az residual raw (deg)",
                       "after (deg)", "|after - (raw - delta)|", "sensor_calibration value (deg)", "airborne rows",
                       "median horizontal error raw (m)", "after (m)"],
                      [[_short(k), sk, fmt(d, 4), n, fmt(rm, 4), fmt(om, 4), fmt(dv, 4), fmt(cv, 4), na, fmt(hr), fmt(ho)]
                       for k, sk, d, n, rm, om, dv, cv, na, hr, ho in info["az_rows"]])
    if errs_it:
        tk = sorted({k for _, k in errs_it})
        L += ["", "Median horizontal error (m) of the smoothed drone_air grid samples per sensor and testing iteration "
              "(the same table in another variant tree gives the per-iteration comparison):", ""]
        L += md_table(["sensor"] + [_short(k) for k in tk],
                      [[s_] + [fmt(float(np.median(np.concatenate(errs_it[(s_, k)])))) if errs_it.get((s_, k)) else "-"
                               for k in tk] for s_ in sorted({s_ for s_, _ in errs_it})])
    return L


# --------------------------------------------------------------------------- #
# One iteration
# --------------------------------------------------------------------------- #


def position_mask(det: dict) -> np.ndarray:
    return np.isin(det["meas_kind"], Q.POSITION_KINDS) & det["usable_pos"] & ~det["far"]


def prepare(it: C.Iteration, det: dict, toff: dict[str, float]) -> dict:
    """Position rows of one iteration sorted into segments and measurement epochs (no filtering).
    det needs h_ell (sensor_noise.u_ellipsoid of the whole table); u_ell is computed here."""
    key = it.key
    m = (det["iter_key"] == key) & position_mask(det)
    D = {k: v[m] for k, v in det.items()}
    n_in = int(m.sum())
    if n_in == 0:
        raise SystemExit(f"{key}: no position rows")
    D["u_ell"] = np.asarray(C.iteration_frame(it).to_enu(D["lat"], D["lon"], D["h_ell"])[2], float)

    t_cal = D["t"].copy()
    for sk, off in toff.items():
        t_cal[D["sensor_key"] == sk] += off
    ident = np.array([f"{a}\x1f{b}" for a, b in zip(D["sensor_key"], D["object_id"])], dtype=object)
    _, inv = np.unique(ident, return_inverse=True)
    o = np.lexsort((t_cal, inv))
    D = {k: v[o] for k, v in D.items()}
    t_cal, inv = t_cal[o], inv[o]

    dt_row = np.diff(t_cal)
    same_obj = inv[1:] == inv[:-1]
    new_seg = np.concatenate([[True], ~same_obj | (dt_row > SPLIT_GAP_S)])
    seg_row = np.cumsum(new_seg) - 1
    S = int(seg_row[-1]) + 1
    new_ep = np.concatenate([[True], new_seg[1:] | (dt_row != 0)])
    ep_row = np.cumsum(new_ep) - 1
    E = int(ep_row[-1]) + 1

    # measurement epochs: mean of the rows at one timestamp
    cnt_ep = np.bincount(ep_row, minlength=E).astype(float)
    me_e = np.bincount(ep_row, D["e"], minlength=E) / cnt_ep
    me_n = np.bincount(ep_row, D["n"], minlength=E) / cnt_ep

    def ep_mean(v):
        fin = np.isfinite(v)
        cu = np.bincount(ep_row[fin], minlength=E).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.bincount(ep_row[fin], v[fin], minlength=E) / cu

    me_u, me_ur = ep_mean(D["u_ell"]), ep_mean(D["u_c"])
    first_ep_row = np.flatnonzero(new_ep)
    me_t = t_cal[first_ep_row]
    me_seg = seg_row[first_ep_row]

    first_row = np.flatnonzero(new_seg)
    seg_inv = inv[first_row]
    part = np.zeros(S, dtype=np.int64)
    for s in range(1, S):
        part[s] = part[s - 1] + 1 if seg_inv[s] == seg_inv[s - 1] else 0
    n_meas = np.bincount(me_seg, minlength=S)
    ep_st = np.concatenate([[0], np.cumsum(n_meas)[:-1]])
    return dict(it=it, key=key, testing=it.has_groundtruth, D=D, n_in=n_in, S=S, E=E, seg_row=seg_row,
                me_t=me_t, me_seg=me_seg, me_e=me_e, me_n=me_n, me_u=me_u, me_ur=me_ur,
                seg_sk=D["sensor_key"][first_row], seg_sid=D["sensor_id"][first_row],
                seg_obj=D["object_id"][first_row], part=part, n_meas=n_meas,
                n_rows=np.bincount(seg_row, minlength=S), t0=me_t[ep_st], t1=me_t[ep_st + n_meas - 1])


def process_iteration(P: dict, white: dict, causal: bool, gtm: dict | None, sigma_rows: dict | None = None) -> dict:
    it, key, D, n_in, S, E = P["it"], P["key"], P["D"], P["n_in"], P["S"], P["E"]
    seg_row, me_t, me_seg = P["seg_row"], P["me_t"], P["me_seg"]
    me_e, me_n, me_u, me_ur = P["me_e"], P["me_n"], P["me_u"], P["me_ur"]
    seg_sk, seg_sid, seg_obj, part = P["seg_sk"], P["seg_sid"], P["seg_obj"], P["part"]
    n_meas, n_rows, t0, t1 = P["n_meas"], P["n_rows"], P["t0"], P["t1"]
    n_dup = n_in - E
    _, grid = C.template_grid(it)
    has_u = np.bincount(me_seg, np.isfinite(me_u), minlength=S) >= 2
    has_ur = np.bincount(me_seg, np.isfinite(me_ur), minlength=S) >= 2

    # grid epochs
    lo = np.searchsorted(grid, t0, side="left")
    hi = np.searchsorted(grid, t1 + EXTRAP_S, side="right")
    n_grid = np.maximum(hi - lo, 0)
    G = int(n_grid.sum())
    g_seg = np.repeat(np.arange(S), n_grid)
    g_off = np.concatenate([[0], np.cumsum(n_grid)[:-1]])
    g_step = lo[g_seg] + (np.arange(G) - g_off[g_seg])

    # merged epoch list: segment, time, measurement before grid at ties
    a_seg = np.concatenate([me_seg, g_seg])
    a_t = np.concatenate([me_t, grid[g_step]])
    a_grid = np.concatenate([np.zeros(E, bool), np.ones(G, bool)])
    oo = np.lexsort((a_grid, a_t, a_seg))
    a_seg, a_t, a_grid = a_seg[oo], a_t[oo], a_grid[oo]
    nan_g = np.full(G, np.nan)
    z_e = np.concatenate([me_e, nan_g])[oo]
    z_n = np.concatenate([me_n, nan_g])[oo]
    z_u = np.concatenate([me_u, nan_g])[oo]
    a_step = np.concatenate([np.full(E, -1), g_step])[oo]
    a_len = n_meas + n_grid
    a_st = np.concatenate([[0], np.cumsum(a_len)[:-1]])

    noise = noise_table(key, sorted(set(seg_sk.tolist())), white, sigma_rows)
    wsd = np.array([noise[s]["white"] for s in seg_sk], dtype=float).reshape(S, 3)
    slow = np.array([noise[s]["slow"] for s in seg_sk], dtype=float).reshape(S, 3)
    re, rn, ru = (wsd[a_seg, c] ** 2 for c in range(3))           # R = white^2
    xe, ve, pe, _, _ = cv_axis(a_t, z_e, re, a_st, a_len, Q_ACC_H_M2S3, causal)
    xn, vn, pn, _, _ = cv_axis(a_t, z_n, rn, a_st, a_len, Q_ACC_H_M2S3, causal)

    def vertical(z_all: np.ndarray, has: np.ndarray):
        x_, v_, p_ = (np.full(len(a_t), np.nan) for _ in range(3))
        if has.any():
            idx = np.flatnonzero(has[a_seg])
            ln_u = a_len[has]
            st_u = np.concatenate([[0], np.cumsum(ln_u)[:-1]])
            x_[idx], v_[idx], p_[idx], _, _ = cv_axis(a_t[idx], z_all[idx], ru[idx], st_u, ln_u, Q_ACC_U_M2S3, causal)
        return x_, v_, p_

    xu, vu, pu = vertical(z_u, has_u)
    gi = np.flatnonzero(a_grid)
    gs = a_seg[gi]
    grid_out = dict(seg_id=gs.astype(np.int32), step=a_step[gi].astype(np.int32), t=a_t[gi],
                    e=xe[gi], n=xn[gi], u=xu[gi], ve=ve[gi], vn=vn[gi], vu=vu[gi],
                    var_e=pe[gi] + slow[gs, 0], var_n=pn[gi] + slow[gs, 1], var_u=pu[gi] + slow[gs, 2],
                    cov_en=np.zeros(len(gi)))
    u_raw_grid = None
    if it.has_groundtruth:                          # diagnostic only: the same filter on heights as reported
        z_ur = np.concatenate([me_ur, nan_g])[oo]
        u_raw_grid = vertical(z_ur, has_ur)[0][gi]

    # ---- features
    mi = np.flatnonzero(~a_grid)                       # measurement epochs, sorted by (seg, t)
    ms = a_seg[mi]
    sp = np.hypot(ve[mi], vn[mi])
    multi = n_meas >= 2
    F: dict[str, np.ndarray | list] = {}
    dur = t1 - t0
    F["duration_s"] = dur
    with np.errstate(invalid="ignore", divide="ignore"):
        F["meas_rate_hz"] = np.where(dur > 0, n_meas / dur, np.nan)
    F["t0_rel_s"] = t0 - grid[0]
    F["t1_rel_s"] = t1 - grid[0]
    step_same = ms[1:] == ms[:-1]
    ds = np.hypot(np.diff(xe[mi]), np.diff(xn[mi]))
    F["path_len_m"] = np.where(multi, np.bincount(ms[1:][step_same], ds[step_same], minlength=S), np.nan)
    smean, _ = group_mean_std(ms, sp, S)
    sq = group_quantile(ms, sp, S, (0.1, 0.9, 1.0))
    F["speed_mean_ms"] = np.where(multi, smean, np.nan)
    F["speed_p10_ms"] = np.where(multi, sq[0], np.nan)
    F["speed_p90_ms"] = np.where(multi, sq[1], np.nan)
    F["speed_max_ms"] = np.where(multi, sq[2], np.nan)
    F["vu_abs_p90_ms"] = np.where(multi & has_u, group_quantile(ms, np.abs(vu[mi]), S, (0.9,))[0], np.nan)
    dtm = np.diff(a_t[mi])
    okd = step_same & (dtm > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.hypot(np.diff(ve[mi]), np.diff(vn[mi])) / dtm
        hd = np.degrees(np.arctan2(ve[mi], vn[mi]))
        turn = np.abs(C.angle_diff_deg(hd[1:], hd[:-1])) / dtm
    turn_ok = okd & (sp[1:] >= STATIONARY_MS) & (sp[:-1] >= STATIONARY_MS)
    F["accel_p90_ms2"] = group_quantile(ms[1:][okd], acc[okd], S, (0.9,))[0]
    F["turn_p90_degs"] = group_quantile(ms[1:][turn_ok], turn[turn_ok], S, (0.9,))[0]
    F["stationary_share"] = np.where(multi, np.bincount(ms, sp < STATIONARY_MS, minlength=S) / n_meas, np.nan)

    dzq = group_quantile(seg_row, D["dz_sensor"], S, (0.5, 0.1, 0.9))
    _, dzs = group_mean_std(seg_row, D["dz_sensor"], S)
    F["dz_med_m"], F["dz_p10_m"], F["dz_p90_m"], F["dz_std_m"] = dzq[0], dzq[1], dzq[2], dzs
    ueq = group_quantile(seg_row, D["u_ell"], S, (0.5, 0.1, 0.9))
    F["u_ell_med_m"], F["u_ell_p10_m"], F["u_ell_p90_m"] = ueq[0], ueq[1], ueq[2]
    uq = group_quantile(seg_row, D["u_c"], S, (0.5,))[0]
    _, us = group_mean_std(seg_row, D["u_c"], S)
    F["u_raw_med_m"], F["u_med_m"], F["u_std_m"] = uq, uq.copy(), us
    with np.errstate(invalid="ignore"):
        rng = np.where(D["sensor_pos_valid"] == 1, np.hypot(D["e"] - D["sensor_e"], D["n"] - D["sensor_n"]), np.nan)
    rq = group_quantile(seg_row, rng, S, (0.5, 0.0, 1.0))
    F["range_med_m"], F["range_min_m"], F["range_max_m"] = rq[0], rq[1], rq[2]
    F["det_conf_mean"] = group_mean_std(seg_row, D["det_conf"], S)[0]
    grp = class_share_of(D["cls"], D["cls_sub"])
    for c in CLASS_SHARES:
        F[f"cls_{c}"] = np.bincount(seg_row, grp == c, minlength=S) / n_rows
    F["cls_conf_mean"] = group_mean_std(seg_row, D["cls_conf"], S)[0]
    for c in ("model", "serial", "protocol"):
        F[f"has_{c}"] = (np.bincount(seg_row, D[c] != "", minlength=S) > 0).astype(np.int64).tolist()
    F["has_freq"] = (np.bincount(seg_row, np.isfinite(D["freq_mhz"]), minlength=S) > 0).astype(np.int64).tolist()
    for c in ("model", "serial", "protocol"):
        F[f"{c}_top"] = group_top(seg_row, D[c], S)
    F["gt_feed"] = np.isin(seg_sid, list(Q.GT_FEED_SENSORS)).astype(np.int64).tolist()

    # ---- labels (testing)
    Lb: dict[str, np.ndarray | list] = {k: np.full(S, np.nan) for k in ("frac_matched", "frac_best", "frac_airborne")}
    Lb["n_matched"] = [""] * S
    Lb["best_flight"] = [""] * S
    Lb["label"] = [""] * S
    n_missing = 0
    if gtm is not None and it.has_groundtruth:
        lut = gtm.get(key, {})
        matched = np.zeros(n_in, bool)
        air = np.zeros(n_in, bool)
        flight = np.empty(n_in, dtype=object)
        for i, (f, r) in enumerate(zip(D["file"], D["row"])):
            rec = lut.get((f, int(r)))
            if rec is None:
                n_missing += 1
                flight[i] = ""
                continue
            matched[i], flight[i], air[i] = rec
        nm = np.bincount(seg_row, matched, minlength=S)
        cnt_pf = Counter((int(s), f) for s, f, mm in zip(seg_row, flight, matched) if mm)
        best: dict[int, tuple[int, str]] = {}
        for (s, f), c in cnt_pf.items():
            cur = best.get(s)
            if cur is None or c > cur[0] or (c == cur[0] and f < cur[1]):
                best[s] = (c, f)
        bf = [best[s][1] if s in best else "" for s in range(S)]
        nb = np.array([best[s][0] if s in best else 0 for s in range(S)], dtype=float)
        bf_row = np.array(bf, dtype=object)[seg_row]
        on_best = matched & (flight == bf_row)
        nair = np.bincount(seg_row, on_best & air, minlength=S)
        with np.errstate(invalid="ignore", divide="ignore"):
            fm = nm / n_rows
            fb = np.where(nm > 0, nb / nm, np.nan)
            fa = np.where(nb > 0, nair / nb, np.nan)
        drone = (fm >= LABEL_FRAC_MATCHED) & (fb >= LABEL_FRAC_BEST)
        lab = np.where(drone & (fa >= LABEL_FRAC_AIR), "drone_air", np.where(drone, "drone_ground", "clutter"))
        Lb.update(n_matched=nm.astype(np.int64).tolist(), frac_matched=fm, best_flight=bf, frac_best=fb,
                  frac_airborne=fa, label=lab.astype(object).tolist())

    tracks = dict(seg_id=np.arange(S).tolist(), iter_key=[key] * S, sensor_key=seg_sk, sensor_id=seg_sid,
                  object_id=seg_obj, part=part.tolist(), t0=t0, t1=t1, n_meas=n_meas.tolist(),
                  n_rows=n_rows.tolist(), n_dup_merged=(n_rows - n_meas).tolist(),
                  has_u=has_u.astype(np.int64).tolist(), n_grid=n_grid.tolist(),
                  noise_src=[noise[s]["src"] for s in seg_sk], white_src=[noise[s]["white_src"] for s in seg_sk])
    tracks.update(F)
    tracks.update(Lb)

    return dict(key=key, tracks=tracks, grid=grid_out, grid_t=grid, n_in=n_in, n_dup=n_dup,
                n_missing=n_missing, noise=noise, meas=dict(seg=me_seg, t=me_t, e=me_e, n=me_n, se=xe[mi], sn=xn[mi]),
                u_raw_grid=u_raw_grid, t0=t0, t1=t1, S=S)


# --------------------------------------------------------------------------- #
# Self-checks on synthetic data
# --------------------------------------------------------------------------- #


def synth_check() -> tuple[bool, str]:
    """(1) CV truth + Gaussian noise: RMSE and NEES of the smoothed 2D position."""
    rng = np.random.default_rng(SEED)
    lanes, T, sigma, q = 100, 300.0, DEFAULT_SIGMA_H_M, Q_ACC_H_M2S3
    tf = np.round(np.arange(0.0, T + 1e-9, 0.1), 1)            # truth / grid epochs, 0.1 s
    is_meas = np.isclose(tf % 1.0, 0.0)                        # measurement every 1 s
    K = len(tf)
    dt = 0.1
    Lq = np.linalg.cholesky(q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]]))
    res = {}
    for ax in ("e", "n"):
        x = np.zeros((lanes, K))
        v = np.zeros((lanes, K))
        x[:, 0] = rng.normal(0, 200, lanes)
        v[:, 0] = rng.normal(0, INIT_VEL_SD_MS, lanes)
        for k in range(1, K):
            w = rng.standard_normal((lanes, 2)) @ Lq.T
            x[:, k] = x[:, k - 1] + dt * v[:, k - 1] + w[:, 0]
            v[:, k] = v[:, k - 1] + w[:, 1]
        z = np.where(is_meas[None, :], x + rng.normal(0, sigma, (lanes, K)), np.nan)
        t = np.tile(tf, lanes)
        st = np.arange(lanes) * K
        ln = np.full(lanes, K)
        r = np.full(lanes * K, sigma ** 2)
        xs, _, ps, _, _ = cv_axis(t, z.ravel(), r, st, ln, q, causal=False)
        xf, _, _, _, _ = cv_axis(t, z.ravel(), r, st, ln, q, causal=True)
        res[ax] = (x.ravel(), z.ravel(), xs, ps, xf)
    mm = np.tile(is_meas, lanes)
    raw = np.sqrt(np.mean([(res[a][1][mm] - res[a][0][mm]) ** 2 for a in "en"]) )
    smo = np.sqrt(np.mean([(res[a][2][mm] - res[a][0][mm]) ** 2 for a in "en"]))
    fil = np.sqrt(np.mean([(res[a][4][mm] - res[a][0][mm]) ** 2 for a in "en"]))
    nees = sum((res[a][2] - res[a][0]) ** 2 / res[a][3] for a in "en")
    mnees = float(np.mean(nees))
    ok = smo < 0.6 * raw and fil < raw and 1.5 <= mnees <= 2.6
    return ok, (f"synthetic CV ({lanes} lanes x {T:.0f} s, 1 Hz, sigma {sigma:.0f} m, q {q}): RMSE raw {raw:.2f} m, "
                f"filtered {fil:.2f}, smoothed {smo:.2f} (ratio {smo / raw:.3f} < 0.6); mean NEES smoothed 2D {mnees:.3f} "
                f"(all {len(nees):,} epochs, target [1.5, 2.6])")


def wls_check() -> tuple[bool, str]:
    """(2) cv_axis (several lanes of different length) == batch information-form solution."""
    rng = np.random.default_rng(SEED + 1)
    lanes = []
    for K in (9, 5, 12, 1):
        t = np.cumsum(rng.uniform(0.05, 3.0, K)) + 1000.0
        z = rng.normal(0, 50, K) + 7.0 * (t - t[0])
        if K > 4:
            z[[2, K - 1]] = np.nan                                # no-measurement epochs, also the last
        r = rng.uniform(4, 400, K)
        lanes.append((t, z, r))
    t = np.concatenate([a[0] for a in lanes])
    z = np.concatenate([a[1] for a in lanes])
    r = np.concatenate([a[2] for a in lanes])
    ln = np.array([len(a[0]) for a in lanes])
    st = np.concatenate([[0], np.cumsum(ln)[:-1]])
    xs, vs, pp, pv, pvv = cv_axis(t, z, r, st, ln, Q_ACC_H_M2S3)
    worst = 0.0
    for j, (tl, zl, rl) in enumerate(lanes):
        mean, cov = batch_wls(tl, zl, rl, Q_ACC_H_M2S3)
        s = slice(st[j], st[j] + ln[j])
        k2 = 2 * np.arange(ln[j])
        worst = max(worst, np.abs(xs[s] - mean[:, 0]).max(), np.abs(vs[s] - mean[:, 1]).max(),
                    np.abs(pp[s] - cov[k2, k2]).max(), np.abs(pv[s] - cov[k2, k2 + 1]).max(),
                    np.abs(pvv[s] - cov[k2 + 1, k2 + 1]).max())
    return worst < 1e-6, f"RTS vs batch WLS on {len(lanes)} lanes ({', '.join(map(str, ln))} epochs): max |diff| {worst:.2e} (< 1e-6)"


def white_check() -> tuple[bool, str]:
    """(6) second_diff + mad_sigma recover a known white sd on irregular sampling, despite a
    smooth manoeuvring trajectory and a large slowly varying (smooth) bias."""
    rng = np.random.default_rng(SEED + 2)
    lanes, K = 200, 300
    worst, parts = 0.0, []
    for sigma in (1.0, 5.0, 20.0):
        t = np.cumsum(rng.uniform(0.5, 1.5, (lanes, K)), axis=1)
        traj = 8.0 * t + 40.0 * np.sin(t / 15.0)                           # 8 m/s + gentle weave
        bias = 25.0 * np.sin(t / 40.0 + rng.uniform(0, 6.28, (lanes, 1)))   # slow bias, 25 m amplitude
        z = traj + bias + rng.normal(0, sigma, (lanes, K))
        seg = np.repeat(np.arange(lanes), K)
        r = second_diff(t.ravel(), z.ravel(), seg, WHITE_MAX_DT_S)
        est = mad_sigma(r)
        worst = max(worst, abs(est / sigma - 1.0))
        parts.append(f"sd {sigma:g} -> {est:.3f}")
    return worst < 0.05, (f"white-noise estimator on synthetic tracks (irregular 0.5-1.5 s sampling, smooth weave + "
                          f"25 m slow bias): {', '.join(parts)}; worst relative error {worst:.3f} (< 0.05)")


def u_ell_check(det: dict) -> tuple[bool, str]:
    """(7) rule-1 conversion vs build_detections' own ENU u: on position rows where u_ellipsoid
    changed nothing (h_ell == alt_c), u_ell must equal u_c; elsewhere u_ell - u_c must equal
    h_ell - alt_c (the frame's up axis is height to < 1e-3 m per m at these ranges)."""
    m = position_mask(det) & np.isfinite(det["h_ell"]) & np.isfinite(det["alt_c"]) & np.isfinite(det["u_c"])
    worst_same, worst_shift, n_same, n_shift = 0.0, 0.0, 0, 0
    for it in C.iterations():
        mi = m & (det["iter_key"] == it.key)
        if not mi.any():
            continue
        u_ell = np.asarray(C.iteration_frame(it).to_enu(det["lat"][mi], det["lon"][mi], det["h_ell"][mi])[2], float)
        d = det["h_ell"][mi] - det["alt_c"][mi]
        du = u_ell - det["u_c"][mi]
        same = d == 0
        n_same += int(same.sum())
        n_shift += int((~same).sum())
        if same.any():
            worst_same = max(worst_same, float(np.abs(du[same]).max()))
        if (~same).any():
            worst_shift = max(worst_shift, float(np.abs(du[~same] - d[~same]).max()))
    ok = n_same > 0 and n_shift > 0 and worst_same < 0.01 and worst_shift < 0.05
    return ok, (f"u_ell vs detections u: {n_same:,} rows with no height rule, max |u_ell - u_c| {worst_same:.4f} m (< 0.01); "
                f"{n_shift:,} rows with a shift, max |(u_ell - u_c) - (h_ell - alt_c)| {worst_shift:.4f} m (< 0.05)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def load_gt_match(keys: list[str]) -> dict:
    tab = C.read_columns(GT_MATCH, {"row", "matched", "gt_airborne"},
                         usecols=["iter_key", "file", "row", "kind", "matched", "best_flight", "gt_airborne"])
    out: dict[str, dict] = {k: {} for k in keys}
    for ik, f, r, kd, mt, bf, ga in zip(tab["iter_key"], tab["file"], tab["row"], tab["kind"], tab["matched"],
                                        tab["best_flight"], tab["gt_airborne"]):
        if ik in out and kd == "pos":
            out[ik][(f, int(r))] = (mt == 1, bf, ga == 1)
    return out


def independent_counts(det: dict, key: str, toff: dict[str, float]) -> tuple[int, Counter, int]:
    """Check-3 reference counts straight from the detections, without the segmentation:
    (position rows, position rows per sensor_key, distinct (sensor_key, object_id, t_cal)).
    Rows at one timestamp of one object always share a segment (a split needs a gap > 0), so
    the distinct triples must equal sum(n_meas)."""
    m = (det["iter_key"] == key) & np.isin(det["meas_kind"], Q.POSITION_KINDS) & det["usable_pos"] & ~det["far"]
    sk, obj, t = det["sensor_key"][m], det["object_id"][m], det["t"][m].astype(np.float64)
    off = np.array([toff.get(s, 0.0) for s in sk.tolist()], dtype=np.float64)
    trip = {(a, b, float(c)) for a, b, c in zip(sk.tolist(), obj.tolist(), (t + off).tolist())}
    return int(m.sum()), Counter(sk.tolist()), len(trip)


def gt_errors(res: dict, gt: dict) -> list[tuple]:
    """Per drone_air segment: (sensor_id, dict of errors vs GT of best_flight):
    raw       horizontal, measurement epochs (raw position at t_cal)
    smo_meas  horizontal, smoothed state at the same measurement epochs
    grid      horizontal, smoothed grid samples (the check-5 quantity)
    lin       horizontal, linear interpolation of the raw measurements at the same grid samples
              (np.interp: constant after t1), the naive resampler the smoother replaces
    far       per grid sample: more than 1 s from the nearest measurement epoch
    z         per grid sample (err_e^2 / var_e + err_n^2 / var_n) / 2 (covariance calibration)
    zu        per grid sample with a height: err_u^2 / var_u
    du_ell, du_raw  per grid sample with a height: smoothed u - GT u, heights after rule 1 / as
              reported (the same filter run on u_c)."""
    tr, gr, me = res["tracks"], res["grid"], res["meas"]
    out = []
    g_st = np.searchsorted(gr["seg_id"], np.arange(res["S"] + 1))
    m_st = np.searchsorted(me["seg"], np.arange(res["S"] + 1))
    in_iter = gt["iter_key"] == res["key"]
    for s in range(res["S"]):
        if tr["label"][s] != "drone_air":
            continue
        fm = in_iter & (gt["flight_key"] == tr["best_flight"][s])
        a, b = m_st[s], m_st[s + 1]
        mt, me_e, me_n = me["t"][a:b], me["e"][a:b], me["n"][a:b]
        ge, gn = Q.interp_flight(gt, fm, mt, cols=("e", "n"))
        raw = np.hypot(me_e - ge, me_n - gn)
        smo_meas = np.hypot(me["se"][a:b] - ge, me["sn"][a:b] - gn)
        a, b = g_st[s], g_st[s + 1]
        tg = gr["t"][a:b]
        ge, gn, gu = Q.interp_flight(gt, fm, tg, cols=("e", "n", "u"))
        ee, en = gr["e"][a:b] - ge, gr["n"][a:b] - gn
        grid = np.hypot(ee, en)
        lin = np.hypot(np.interp(tg, mt, me_e) - ge, np.interp(tg, mt, me_n) - gn)
        j = np.clip(np.searchsorted(mt, tg), 1, max(len(mt) - 1, 1))
        near = np.minimum(np.abs(tg - mt[j - 1]), np.abs(tg - mt[np.minimum(j, len(mt) - 1)]))
        z = (ee ** 2 / gr["var_e"][a:b] + en ** 2 / gr["var_n"][a:b]) / 2.0
        du_ell = gr["u"][a:b] - gu
        du_raw = res["u_raw_grid"][a:b] - gu
        zu = du_ell ** 2 / gr["var_u"][a:b]
        okm, okg = np.isfinite(raw), np.isfinite(grid)
        out.append((tr["sensor_id"][s], dict(raw=raw[okm], smo_meas=smo_meas[okm], grid=grid[okg], lin=lin[okg],
                                             far=near[okg] > 1.0, z=z[okg], zu=zu[np.isfinite(zu)],
                                             du_ell=du_ell[np.isfinite(du_ell)], du_raw=du_raw[np.isfinite(du_raw)])))
    return out


def dropped_duplicates(det: dict, key: str, toff: dict[str, float]) -> tuple[int, int]:
    """Position rows of `key` that would be measurements but for cuas_clean's 'duplicate' flag:
    (dropped rows, of which a kept row of the same sensor / object has the identical t_cal).
    The rest are near-copies stamped up to 1 ms apart (version 1 kept them as separate epochs)."""
    base = ((det["iter_key"] == key) & np.isin(det["meas_kind"], Q.POSITION_KINDS) & ~det["far"]
            & ~det["null_island"] & ~det["self_report"] & np.isfinite(det["e"]) & np.isfinite(det["n"]))
    kept, drop = base & det["usable_pos"], base & det["duplicate"]

    def trip(m):
        sk = det["sensor_key"][m].tolist()
        off = [toff.get(s, 0.0) for s in sk]
        return [(a, b, float(c) + o) for a, b, c, o in zip(sk, det["object_id"][m].tolist(), det["t"][m].tolist(), off)]

    seen = set(trip(kept))
    return int(drop.sum()), sum(1 for x in trip(drop) if x in seen)


def v1_dup_counts(slug: str, suffix: str) -> tuple[int, int] | None:
    """(sum n_rows, sum n_dup_merged) of the version-1 tracks.csv, if present (read only)."""
    path = OUT_V1 / f"{slug}__tracks{suffix}.csv"
    if not path.exists():
        return None
    v = C.read_columns(path, {"n_rows", "n_dup_merged"}, usecols=["n_rows", "n_dup_merged"])
    return int(v["n_rows"].sum()), int(v["n_dup_merged"].sum())


def summarise_errors(errs: dict[str, list]) -> list[dict]:
    """Per sensor_id: medians / percentiles of the gt_errors arrays (pooled over segments)."""
    tab = []
    for s_id in sorted(errs):
        ev = {k: np.concatenate([e[k] for e in errs[s_id]]) for k in errs[s_id][0]}
        if len(ev["raw"]) == 0 or len(ev["grid"]) == 0:
            continue

        def md(x):
            return float(np.median(x)) if len(x) else float("nan")

        tab.append(dict(sensor=s_id, segs=len(errs[s_id]), n_raw=len(ev["raw"]), n_grid=len(ev["grid"]),
                        raw=md(ev["raw"]), grid=md(ev["grid"]), lin=md(ev["lin"]),
                        raw90=float(np.percentile(ev["raw"], 90)), grid90=float(np.percentile(ev["grid"], 90)),
                        smo_meas=md(ev["smo_meas"]), far=float(np.mean(ev["far"])),
                        z_med=md(ev["z"]), z_mean=float(np.mean(ev["z"])), n_zu=len(ev["zu"]),
                        zu_med=md(ev["zu"]), zu_mean=float(np.mean(ev["zu"])) if len(ev["zu"]) else float("nan"),
                        n_ur=len(ev["du_raw"]), ur_abs=md(np.abs(ev["du_raw"])), ur_bias=md(ev["du_raw"]),
                        n_ue=len(ev["du_ell"]), ue_abs=md(np.abs(ev["du_ell"])), ue_bias=md(ev["du_ell"]),
                        gt_feed=s_id in Q.GT_FEED_SENSORS, grid_all=ev["grid"], lin_all=ev["lin"]))
    for r in tab:
        r["gated"] = r["n_grid"] >= CHECK5_MIN_GRID
        r["ok5"] = (not r["gated"]) or r["grid"] <= CHECK5_RATIO * r["lin"]
    return tab


def calib_word(z_med: float) -> str:
    """'over' (variance too small) / 'under' (too large) / 'ok' from the median z vs ln 2, factor CALIB_FACTOR."""
    if not np.isfinite(z_med):
        return "-"
    return "over" if z_med > CALIB_FACTOR * CALIB_REF_MED else ("under" if z_med < CALIB_REF_MED / CALIB_FACTOR else "ok")


def loio_height_reference() -> dict[str, float]:
    """sensor_id -> sensor_noise.csv loio_held_out_u_m (NaN when absent)."""
    if not NOISE.exists():
        return {}
    tab = C.read_columns(NOISE, {"loio_held_out_u_m"})
    if "loio_held_out_u_m" not in tab:
        return {}
    return {str(k).split("_", 1)[0]: float(v) for k, v in zip(tab["sensor_key"], tab["loio_held_out_u_m"])}


def fmt(v, nd=2) -> str:
    return "-" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--iter", nargs="*", default=None, help="iteration keys (default: all)")
    ap.add_argument("--causal", action="store_true", help="forward filter only (no RTS smoothing)")
    ap.add_argument("--out", default=str(OUT_DEFAULT), help="output directory (default: data/derived/sensor_tracks_v2)")
    ap.add_argument("--corrections", nargs="*", default=None, metavar="NAME",
                    help=f"override variant.json: corrections to apply ({', '.join(CORRECTIONS)}); no names = none")
    ap.add_argument("--az-sensors", nargs="*", default=None, metavar="SENSOR_KEY", help="sensor keys for az_rotate")
    ap.add_argument("--drop-sensors", nargs="*", default=None, metavar="SENSOR_ID", help="sensor ids to drop")
    args = ap.parse_args()
    corr = load_corrections(args)
    out = Path(args.out)
    if out.resolve() == OUT_V1.resolve():
        raise SystemExit(f"--out {out} is the version-1 directory; choose another")
    t_start = time.time()
    all_its = C.iterations()
    its = all_its
    if args.iter:
        unknown = set(args.iter) - {i.key for i in its}
        if unknown:
            raise SystemExit(f"unknown iteration keys: {sorted(unknown)}")
        its = [i for i in its if i.key in set(args.iter)]
    keys = [i.key for i in its]
    suffix = "_causal" if args.causal else ""
    mode = "causal (filter only)" if args.causal else "smoothed (RTS)"
    log(f"sensor_tracks v2: {len(its)} iteration(s), mode {mode}, out {out}")
    if corr["names"]:
        log(f"corrections ({corr['origin']}): {', '.join(corr['names'])}"
            + (f"; az_sensors {', '.join(corr['az_sensors'])}" if corr["az_sensors"] else "")
            + (f"; drop_sensors {', '.join(corr['drop_sensors'])}" if corr["drop_sensors"] else ""))

    checks: list[tuple[str, bool, str]] = []
    for n, fn in (("1", synth_check), ("2", wls_check), ("6", white_check)):
        ok, msg = fn()
        checks.append((n, ok, msg))
    log(f"synthetic checks done ({time.time() - t_start:.1f} s)")

    # all iterations are cleaned: duplicates / site heights need whole iterations, and the white
    # noise is always estimated on every testing iteration (identical constants for any --iter)
    det = Q.clean(Q.load_detections(DET_COLS))
    det["h_ell"] = SN.u_ellipsoid(det)
    toff = time_offsets()
    cinfo = None
    toff_by = {i.key: toff for i in all_its}
    if corr["names"]:
        det, cinfo = apply_corrections(det, corr, all_its)
        toff_by = {k: {sk: v for sk, (v, _) in pl.items()} for k, pl in cinfo["toff"].items()}
    checks.append(("7",) + u_ell_check(det))
    log(f"detections: {len(det['t']):,} rows ({int(det['duplicate'].sum()):,} exact duplicates excluded from "
        f"usable_pos); applied clock offsets: " + ", ".join(f"{k} {v:+.2f} s" for k, v in sorted(toff.items()))
        + (" (pooled; clock_loio: per iteration, see corrections.csv)" if cinfo and "clock_loio" in cinfo["names"] else ""))
    testing_all = [i for i in all_its if i.has_groundtruth]
    need = {i.key for i in its} | {i.key for i in testing_all}
    preps = {i.key: prepare(i, det, toff_by[i.key]) for i in all_its if i.key in need}
    all_keys = sorted(set(det["sensor_key"][position_mask(det)].tolist()))
    white, wtab = estimate_white([preps[i.key] for i in testing_all], all_keys)
    out.mkdir(parents=True, exist_ok=True)
    C.write_columns(out / "white_noise.csv", wtab, list(wtab))
    tot = total_sigma_table("", all_keys)
    log("noise model (m): sensor_key | white e n u (src) | total e n u (src) | slow sd e n u")
    for sk in all_keys:
        w, tt = white[sk], tot[sk]
        sl = [np.sqrt(max(tt[c] ** 2 - w["sigma"][c] ** 2, 0.0)) for c in range(3)]
        log(f"  {sk:32s} | {w['sigma'][0]:6.2f} {w['sigma'][1]:6.2f} {w['sigma'][2]:6.2f} ({w['src']}, n {w['n']}) | "
            f"{tt[0]:6.2f} {tt[1]:6.2f} {tt[2]:6.2f} ({tt[3]}) | {sl[0]:6.2f} {sl[1]:6.2f} {sl[2]:6.2f}")

    testing = [i for i in its if i.has_groundtruth]
    gt = Q.load_groundtruth() if testing else None
    gtm = load_gt_match([i.key for i in testing]) if testing else None
    if cinfo is not None:
        checks += correction_checks(det, cinfo, gt)
    if gt is not None:
        qe, qn, qu, nq = gt_q_estimate(gt)
        log(f"GT process-noise estimate (n={nq} airborne 1 s steps): q_e {qe:.2f}, q_n {qn:.2f}, q_u {qu:.2f} m^2/s^3; "
            f"used Q_ACC_H {Q_ACC_H_M2S3}, Q_ACC_U {Q_ACC_U_M2S3}")

    order = HEAD + FEATURES + LABELS
    results = []
    acc_rows, span_bad, span_dup, n_grid_tot, missing, dup_rows = [], 0, 0, 0, 0, []
    errs: dict[str, list] = {}
    errs_it: dict[tuple[str, str], list] = {}
    for it in its:
        ti = time.time()
        res = process_iteration(preps[it.key], white, args.causal, gtm,
                                cinfo["sigma"][it.key][0] if cinfo is not None and cinfo["sigma"] is not None else None)
        slug = C.iter_slug(it.key)
        C.write_columns(out / f"{slug}__tracks{suffix}.csv", res["tracks"], order)
        np.savez_compressed(out / f"{slug}__grid{suffix}.npz",
                            **{k: (v.astype(np.float32) if k in GRID_F32 else v) for k, v in res["grid"].items()})
        tr, gr = res["tracks"], res["grid"]
        # check 3: row accounting, written tracks.csv / grid.npz read back vs counts taken from the detections
        n_ref, per_sk, n_trip = independent_counts(det, it.key, toff_by[it.key])
        back = C.read_columns(out / f"{slug}__tracks{suffix}.csv", {"n_meas", "n_rows", "n_dup_merged", "n_grid"},
                              usecols=["sensor_key", "n_meas", "n_rows", "n_dup_merged", "n_grid"])
        with np.load(out / f"{slug}__grid{suffix}.npz") as z:
            n_npz = len(z["seg_id"])
        sk_rows = Counter()
        for s, nr in zip(back["sensor_key"], back["n_rows"]):
            sk_rows[s] += int(nr)
        problems = []
        if int(back["n_rows"].sum()) != n_ref:
            problems.append(f"sum n_rows {int(back['n_rows'].sum()):,} != {n_ref:,} position rows")
        if sk_rows != per_sk:
            problems.append("per-sensor rows differ: " + ", ".join(
                f"{s} {sk_rows[s]} vs {per_sk[s]}" for s in sorted(set(sk_rows) | set(per_sk)) if sk_rows[s] != per_sk[s]))
        if int(back["n_meas"].sum()) != n_trip:
            problems.append(f"sum n_meas {int(back['n_meas'].sum()):,} != {n_trip:,} distinct (sensor, object, t_cal)")
        if np.any(back["n_rows"] != back["n_meas"] + back["n_dup_merged"]):
            problems.append("n_rows != n_meas + n_dup_merged on some segment")
        if int(back["n_grid"].sum()) != n_npz:
            problems.append(f"sum n_grid {int(back['n_grid'].sum()):,} != {n_npz:,} grid.npz rows")
        acc_rows.append((it.key, n_ref, int(back["n_meas"].sum()), int(back["n_dup_merged"].sum()), problems))
        dup_rows.append((it.key, n_ref, int(back["n_dup_merged"].sum()), v1_dup_counts(slug, suffix),
                         dropped_duplicates(det, it.key, toff_by[it.key])))
        # check 4: grid samples inside spans, unique (seg, step), grid time = template time
        sid = gr["seg_id"]
        tg = res["grid_t"][gr["step"]]
        span_bad += int(np.sum((tg < res["t0"][sid]) | (tg > res["t1"][sid] + EXTRAP_S) | (tg != gr["t"])))
        span_dup += len(sid) - len(np.unique(sid.astype(np.int64) * (len(res["grid_t"]) + 1) + gr["step"]))
        n_grid_tot += len(sid)
        missing += res["n_missing"]
        if gt is not None and it.has_groundtruth:
            for s_id, ev in gt_errors(res, gt):
                errs.setdefault(s_id, []).append(ev)
                errs_it.setdefault((s_id, it.key), []).append(ev["grid"])
        log(f"  {it.key}: {res['n_in']:,} position rows -> {res['S']:,} segments, {len(sid):,} grid samples; "
            f"noise {dict(Counter(v['src'] for v in res['noise'].values()))}, white "
            f"{dict(Counter(v['white_src'] for v in res['noise'].values()))}; {time.time() - ti:.1f} s")
        results.append(dict(key=res["key"], tracks=tr))
        del res
    del preps
    if cinfo is not None and "drop_sensors" in cinfo["names"]:
        drop = set(corr["drop_sensors"])
        left = Counter(f"{r['key']}:{sid}" for r in results for sid, skk in zip(r["tracks"]["sensor_id"], r["tracks"]["sensor_key"])
                       if sid in drop or skk in drop)
        left_rows = int((position_mask(det) & (np.isin(det["sensor_id"], list(drop)) | np.isin(det["sensor_key"], list(drop)))).sum())
        checks.append(("12", not left and left_rows == 0 and cinfo["n_dropped"] > 0,
                       f"drop_sensors {sorted(drop)}: {cinfo['n_dropped']:,} position rows removed before segmenting; "
                       f"segments of dropped sensors left in the written tracks: {sum(left.values())} "
                       f"{dict(left) if left else ''}; position rows left: {left_rows}"))

    bad3 = [a for a in acc_rows if a[4]]
    checks.append(("3", not bad3, "row accounting (position rows counted from the detections = sum n_meas + "
                   "sum n_dup_merged of the written tracks.csv; also per sensor_key, sum n_meas = distinct "
                   "(sensor, object, t_cal), sum n_grid = grid.npz rows): " + "; ".join(
                       f"{k.split('/', 1)[1]} {n:,} = {m:,} + {d:,}" + (f" [{'; '.join(p)}]" if p else "")
                       for k, n, m, d, p in acc_rows)))
    checks.append(("4", span_bad == 0 and span_dup == 0,
                   f"{n_grid_tot:,} grid samples: {span_bad} outside [t0, t1 + {EXTRAP_S:.0f} s] or off-grid, "
                   f"{span_dup} duplicate (seg, step)"))
    dup_lines = ["    n_dup_merged (rows merged at an identical timestamp) version 1 -> version 2; cuas_clean 'duplicate' "
                 "position rows dropped upstream = same t_cal as a kept row + near-copies (t up to 1 ms apart):"]
    for k, nr, nd, v1, (ndr, nsame) in dup_rows:
        dup_lines.append(f"    {k:26s} rows {v1[0] if v1 else '-':>8} -> {nr:>8,}   n_dup_merged {v1[1] if v1 else '-':>7} -> "
                         f"{nd:>7,}   dropped {ndr:>7,} = {nsame:>7,} same t + {ndr - nsame:>5,} near")
    log("\n".join(dup_lines))

    # check 5: drone_air smoothed grid error vs linear interpolation, per sensor and pooled
    tab5 = summarise_errors(errs)
    deploy = C.deployable_sensor_ids()
    loio_u = loio_height_reference()
    pooled = None
    if tab5:
        g_all = np.concatenate([r["grid_all"] for r in tab5])
        l_all = np.concatenate([r["lin_all"] for r in tab5])
        dm = [r for r in tab5 if r["sensor"] in deploy]
        g_dep = np.concatenate([r["grid_all"] for r in dm]) if dm else np.empty(0)
        l_dep = np.concatenate([r["lin_all"] for r in dm]) if dm else np.empty(0)
        pooled = dict(n=len(g_all), grid=float(np.median(g_all)), lin=float(np.median(l_all)),
                      n_dep=len(g_dep), grid_dep=float(np.median(g_dep)) if len(g_dep) else float("nan"),
                      lin_dep=float(np.median(l_dep)) if len(l_dep) else float("nan"))
    if testing:
        lines = [f"    {'sensor':6s} {'segs':>5s} {'raw n':>7s} {'grid n':>8s} {'raw med':>8s} {'grid med':>9s} "
                 f"{'lin@grid':>9s} {'grid/lin':>9s} | {'raw p90':>8s} {'grid p90':>9s} {'sm@meas':>8s} {'grid>1s':>8s}"]
        for r in tab5:
            verdict = ("" if r["ok5"] else "  FAIL") if r["gated"] else f"  (n < {CHECK5_MIN_GRID}: not gated)"
            lines.append(f"    {r['sensor']:6s} {r['segs']:5d} {r['n_raw']:7d} {r['n_grid']:8d} {r['raw']:8.2f} "
                         f"{r['grid']:9.2f} {r['lin']:9.2f} {r['grid'] / r['lin']:9.3f} | {r['raw90']:8.2f} "
                         f"{r['grid90']:9.2f} {r['smo_meas']:8.2f} {100 * r['far']:7.0f}%"
                         + ("  gt_feed" if r["gt_feed"] else "") + verdict)
        if pooled:
            lines.append(f"    pooled all sensors: n {pooled['n']:,}, grid median {pooled['grid']:.3f} m vs linear "
                         f"{pooled['lin']:.3f} m" + ("" if pooled["grid"] <= pooled["lin"] else "  FAIL"))
            lines.append(f"    pooled deployable sensors (present in evaluation; no LJJX / EUIG): n {pooled['n_dep']:,}, "
                         f"grid median {pooled['grid_dep']:.3f} m vs linear {pooled['lin_dep']:.3f} m "
                         f"({100 * (1 - pooled['grid_dep'] / pooled['lin_dep']):+.2f} % gain)"
                         + ("" if pooled["grid_dep"] <= pooled["lin_dep"] else "  FAIL"))
        lines.append(f"    (pass rule: grid med <= {CHECK5_RATIO} x lin@grid med for sensors with >= {CHECK5_MIN_GRID} "
                     "grid samples, and pooled grid med <= pooled lin med over all and over deployable sensors. raw = "
                     "measurement epochs; sm@meas = smoothed "
                     "(causal: filtered) state at the measurement epochs; grid>1s = share of grid samples more than 1 s "
                     "from a measurement)")
        ok5 = (bool(tab5) and missing == 0 and all(r["ok5"] for r in tab5) and pooled is not None
               and pooled["grid"] <= pooled["lin"] and pooled["n_dep"] > 0 and pooled["grid_dep"] <= pooled["lin_dep"])
        checks.append(("5", ok5, f"drone_air: median horizontal error of {'filtered' if args.causal else 'smoothed'} "
                       f"grid samples vs linear interpolation (m; det_gt_match rows missing: {missing}):\n" + "\n".join(lines)))
        rep = [f"    covariance calibration on drone_air grid samples (z = (err_e^2/var_e + err_n^2/var_n)/2; a calibrated "
               f"Gaussian gives mean {CALIB_REF_MEAN:.3f}, median {CALIB_REF_MED:.3f}; zu = err_u^2/var_u: mean 1, median "
               f"{CALIB_U_REF_MED:.3f}). Vertical: median |u - GT u| / median (u - GT u) of the smoothed height. "
               "IN-SAMPLE: sigma_total, the height offsets and the clock offsets were fitted on all six testing "
               "iterations; loio = sensor_noise.csv loio_held_out_u_m (held-out mean |median height bias|, the "
               "honest transfer reference, a different metric):",
               f"    {'sensor':6s} {'grid n':>8s} {'z med':>7s} {'z mean':>8s} {'calib':>6s} | {'zu n':>7s} {'zu med':>7s} "
               f"{'zu mean':>8s} | {'raw n':>7s} {'|du| raw':>9s} {'bias raw':>9s} | {'ell n':>7s} {'|du| ell':>9s} "
               f"{'bias ell':>9s} {'loio':>6s}"]
        for r in tab5:
            rep.append(f"    {r['sensor']:6s} {r['n_grid']:8d} {fmt(r['z_med'], 3):>7s} {fmt(r['z_mean'], 3):>8s} "
                       f"{calib_word(r['z_med']):>6s} | "
                       f"{r['n_zu']:7d} {fmt(r['zu_med'], 3):>7s} {fmt(r['zu_mean'], 3):>8s} | {r['n_ur']:7d} "
                       f"{fmt(r['ur_abs']):>9s} {fmt(r['ur_bias']):>9s} | {r['n_ue']:7d} {fmt(r['ue_abs']):>9s} "
                       f"{fmt(r['ue_bias']):>9s} {fmt(loio_u.get(r['sensor'], float('nan'))):>6s}")
        log("\nREPORT (not gating)\n" + "\n".join(rep))
        if cinfo is not None:
            log(per_iteration_error_report(errs_it))
    else:
        checks.append(("5", True, "skipped (no testing iteration selected)"))

    checks.sort(key=lambda c: int(c[0]))
    if cinfo is not None:
        write_corrections(out, cinfo, keys)
    elif (out / "corrections.csv").exists():
        (out / "corrections.csv").unlink()            # a plain run leaves no stale record of an earlier variant
    if set(keys) == {i.key for i in all_its}:
        write_readme(out, results, tab5, pooled, toff, mode, suffix, n_grid_tot, checks, white, tot, dup_rows,
                     corr_section(cinfo, corr, errs_it) if cinfo is not None else None)
    else:
        log(f"README{suffix}.md not rewritten (subset run: {len(keys)} of {len(all_its)} iterations)")

    print("\nSELF-CHECKS")
    for n, ok, msg in checks:
        print(f"self-check {n} {'OK  ' if ok else 'FAIL'} {msg}")
    log(f"total {time.time() - t_start:.1f} s; wrote {out}")
    if all(ok for _, ok, _ in checks):
        print("ALL SELF-CHECKS PASSED")
        return 0
    print("SELF-CHECKS FAILED")
    return 1


def write_readme(out: Path, results: list[dict], tab5: list, pooled: dict | None, toff: dict, mode: str, suffix: str,
                 n_grid: int, checks: list[tuple[str, bool, str]], white: dict, tot: dict, dup_rows: list,
                 corr_lines: list[str] | None = None) -> None:
    """Written only by a run over all iterations; states the self-check verdict."""
    failed = [n for n, ok, _ in checks if not ok]
    L = [f"# Per-sensor track segments, version 2: {mode}", "",
         ("**Self-checks: all passed.**" if not failed else
          f"**Self-checks FAILED: {', '.join(failed)}** (see the tables below and the run log; the files were still "
          "written)."), "",
         "Generated by `tools/sensor_tracks.py` (schemas, feature list and label rule in its docstring). Each sensor's own "
         "track (its objectId) is cut into segments, passed through a Kalman filter and a Rauch-Tung-Striebel (RTS) "
         "smoother (a filter run forward, then backward, so every time step uses the reports before and after it), and "
         "written on the submission template's 100 ms grid with a variance per axis.", "",
         "## What changed from version 1", "",
         "* **Heights** are now heights above the WGS84 ellipsoid (`sensor_noise.u_ellipsoid`, no ground truth used), "
         "converted to the iteration's local up coordinate. The height as reported is kept as the feature `u_raw_med_m`.",
         "* **Two-part noise model.** The sensor feeds are already the output of each sensor's own tracker, so their error "
         "is mostly a slowly drifting offset (bias) plus a small jitter. The filter now only removes the jitter (the "
         "*white* part, measured from the reports alone); the drift cannot be removed by smoothing, so its variance is "
         "added back to the reported variance. `var_e / var_n / var_u` = smoothed variance + slow variance: meant as the "
         "total error variance and the fusion weight, but **not calibrated** (see the calibration section: several "
         "sensors' variances are clearly too small, the ground-truth relays' too large; the total spread from "
         "`sensor_noise.csv` is cut at a 50 m gate, so it is a lower bound for the noisier radars). A fusion should "
         "re-calibrate it per sensor, leave-one-iteration-out.",
         "* **In-sample caveat.** The total spread (`sensor_noise.csv` sigma_e / sigma_n / sigma_u), the height offsets of "
         "`sensor_noise.u_ellipsoid` and the clock offsets were fitted with Dronetag ground truth on all six testing "
         "iterations at once, then applied to every iteration, including the testing one being scored. So the calibration "
         "and vertical tables below are in-sample (optimistic), and any leave-one-iteration-out validation that uses "
         "`var_*` or `u` from the testing grid files inherits this small leak. The vertical table adds the held-out "
         "column of `sensor_noise.csv` as the honest reference.",
         "* **Duplicate rows** are removed upstream (cuas_clean 'duplicate': same sensor, object, time to 1 ms, position "
         "to 1 cm and bearing). Most are copies at an identical timestamp; some are the same position stamped up to 1 ms "
         "apart (see the table under Segments).",
         "* **Self-check 5** now compares the smoothed track with linear interpolation of the raw reports (the "
         "simplest alternative), instead of with the raw reports.", "",
         f"Constants: SPLIT_GAP_S {SPLIT_GAP_S:.0f} s, EXTRAP_S {EXTRAP_S:.0f} s, Q_ACC_H {Q_ACC_H_M2S3} m²/s³, "
         f"Q_ACC_U {Q_ACC_U_M2S3} m²/s³, prior sd position {INIT_POS_SD_M:.0f} m / velocity {INIT_VEL_SD_MS:.0f} m/s, "
         f"white-noise triples with gaps <= {WHITE_MAX_DT_S:g} s, at least {WHITE_MIN_N} triples, floor "
         f"{SIGMA_FLOOR_M:g} m. Clock offsets applied: " + ", ".join(f"{k} {v:+.2f} s" for k, v in sorted(toff.items()))
         + f". Total grid samples: {n_grid:,}.", "",
         "## Noise model per sensor (metres, standard deviations)", "",
         "*white*: jitter from report to report, estimated on testing without ground truth: for three consecutive "
         "reports, how far the middle one misses the straight line through its neighbours (scaled so pure jitter of sd s "
         "gives s), robust spread (1.4826 x median absolute deviation) over all testing triples. This is the filter's "
         "measurement noise. *total*: the error spread against Dronetag ground truth from `sensor_noise.csv` (or the "
         f"default {DEFAULT_SIGMA_H_M:g} / {DEFAULT_SIGMA_U_M:g} m). *slow* = sqrt(max(total² - white², 0)): the drifting "
         "part, added to the reported variance.", ""]
    rows = []
    for sk in sorted(white):
        w, tt = white[sk], tot[sk]
        sl = [np.sqrt(max(tt[c] ** 2 - w["sigma"][c] ** 2, 0.0)) for c in range(3)]
        rows.append([sk, w["n"], fmt(w["raw"][0]), fmt(w["raw"][1]), fmt(w["raw"][2]),
                     " / ".join(f"{v:.2f}" for v in w["sigma"]), w["src"], " / ".join(f"{v:.2f}" for v in tt[:3]), tt[3],
                     " / ".join(f"{v:.2f}" for v in sl)])
    L += md_table(["sensor_key", "triples", "white e (unfloored)", "white n (unfloored)", "white u (unfloored)",
                   "white used e / n / u", "white src", "total e / n / u", "total src", "slow e / n / u"], rows)
    L += ["", "## Segments per sensor", ""]
    seg = Counter()
    nrows = Counter()
    trees = sorted({r["key"].split("/")[0] for r in results})
    for r in results:
        tree = r["key"].split("/")[0]
        tr = r["tracks"]
        for sid, nr in zip(tr["sensor_id"], tr["n_rows"]):
            seg[(sid, tree)] += 1
            nrows[(sid, tree)] += nr
    sensors = sorted({s for s, _ in seg})
    L += md_table(["sensor"] + [f"{t} segments" for t in trees] + [f"{t} rows" for t in trees],
                  [[s] + [seg[(s, t)] for t in trees] + [nrows[(s, t)] for t in trees] for s in sensors])
    L += ["", "Rows merged at an identical timestamp (`n_dup_merged`), version 1 -> version 2, and the position rows "
          "now dropped upstream as duplicates, split into copies with the same timestamp as a kept row (version 1 merged "
          "these, so they explain the fall of `n_dup_merged`) and near-copies with the same position but a timestamp up "
          "to 1 ms different (version 1 kept these as separate reports; they explain the row drop in evaluation DI and "
          "DII, where `n_dup_merged` barely changes):", ""]
    L += md_table(["iteration", "position rows v1", "position rows v2", "n_dup_merged v1", "n_dup_merged v2",
                   "dropped duplicates", "same timestamp", "near-copies (< 1 ms)"],
                  [[k, f"{v1[0]:,}" if v1 else "-", f"{nr:,}", f"{v1[1]:,}" if v1 else "-", f"{nd:,}", f"{ndr:,}",
                    f"{nsame:,}", f"{ndr - nsame:,}"] for k, nr, nd, v1, (ndr, nsame) in dup_rows])
    labs = Counter()
    for r in results:
        if r["key"].startswith("testing"):
            for sid, lab in zip(r["tracks"]["sensor_id"], r["tracks"]["label"]):
                labs[(sid, lab)] += 1
    if labs:
        L += ["", "## Labels per sensor (testing)", "",
              f"drone = frac_matched >= {LABEL_FRAC_MATCHED} and frac_best >= {LABEL_FRAC_BEST}; air = frac_airborne >= "
              f"{LABEL_FRAC_AIR}. GT-feed sensors (VKYY, QTVS, LJJX) relay the Dronetag feed: their labels are leakage.", ""]
        ls = sorted({s for s, _ in labs})
        L += md_table(["sensor", "drone_air", "drone_ground", "clutter"],
                      [[s, labs[(s, "drone_air")], labs[(s, "drone_ground")], labs[(s, "clutter")]] for s in ls])
    if tab5:
        L += ["", "## Horizontal error on drone_air segments (testing, self-check 5)", "",
              "Medians unless stated, metres (the median is the middle value; p90 is the value 90 % of samples stay "
              "below). raw: each report vs the best flight's ground truth at the report time. grid: each grid sample of "
              "the smoothed segment (including gaps and the 1 s extrapolation) vs ground truth at the grid time. lin: "
              "linear interpolation of the raw reports at the same grid samples (the simplest alternative). sm@meas: the "
              "smoothed state at the report times. grid>1s: share of grid samples more than 1 s from a report. "
              f"check 5: grid <= {CHECK5_RATIO} x lin for sensors with >= {CHECK5_MIN_GRID} grid samples.", ""]
        L += md_table(["sensor", "segments", "raw n", "grid n", "raw", "grid", "lin", "grid / lin", "raw p90", "grid p90",
                       "sm@meas", "grid>1s", "gt_feed", "check 5"],
                      [[r["sensor"], r["segs"], r["n_raw"], r["n_grid"], fmt(r["raw"]), fmt(r["grid"]), fmt(r["lin"]),
                        f"{r['grid'] / r['lin']:.3f}", fmt(r["raw90"]), fmt(r["grid90"]), fmt(r["smo_meas"]),
                        f"{100 * r['far']:.0f} %", "yes" if r["gt_feed"] else "",
                        ("ok" if r["ok5"] else "FAIL") if r["gated"] else "not gated"] for r in tab5])
        if pooled:
            gain = 100 * (1 - pooled["grid_dep"] / pooled["lin_dep"])
            L += ["", f"Pooled over all sensors: grid {pooled['grid']:.3f} m vs lin {pooled['lin']:.3f} m "
                  f"({pooled['n']:,} samples). Pooled over the deployable sensors only (those present in evaluation, so "
                  f"without the testing-only EUIG and the relay LJJX, which carries most of the all-sensor gain): grid "
                  f"{pooled['grid_dep']:.3f} m vs lin {pooled['lin_dep']:.3f} m ({pooled['n_dep']:,} samples, "
                  f"{gain:+.2f} %). Both are gated. Plainly: on the sensors we can use in evaluation the smoother is "
                  "about as accurate as joining the reports with straight lines, because the feeds are already the "
                  "output of each sensor's own tracker. What it adds is a velocity and a variance at every grid step.",
                  "", "A sensor with few, widely spaced reports (VKYY: one main segment of 9 reports with gaps up to 27 s) "
                  "can do worse than straight lines: the constant-velocity smoother draws a smooth curve (a cubic spline) "
                  "through the reports, which overshoots when the drone changes speed during a long gap."]
        L += ["", "## Covariance calibration (testing, drone_air grid samples; reported, not gated)", "",
              "Calibration asks whether the reported variance matches the actual error. Per grid sample z = (err_e²/var_e "
              "+ err_n²/var_n) / 2. If the errors were Gaussian with exactly the reported variance, z would have mean "
              f"{CALIB_REF_MEAN:.3f} and median {CALIB_REF_MED:.3f} (= ln 2, a chi-square with 2 degrees of freedom "
              "divided by 2). Larger values: the variance is too small (over-confident); smaller: too large. zu = "
              f"err_u²/var_u, reference mean 1, median {CALIB_U_REF_MED:.3f} (chi-square, 1 degree of freedom). "
              f"Verdict column: 'over' = over-confident (median z above {CALIB_FACTOR:g} x ln 2, the variance is too "
              f"small), 'under' = under-confident (below ln 2 / {CALIB_FACTOR:g}), else 'ok'. These numbers are "
              "in-sample (the total spread was fitted on the same testing data), so the truth is at best this good.", "",
              "Over-confident here: " + (", ".join(f"{r['sensor']} (median z {r['z_med']:.2f}, variance about "
                                                    f"{r['z_med'] / CALIB_REF_MED:.1f}x too small)"
                                                    for r in tab5 if calib_word(r["z_med"]) == "over") or "none")
              + ". Under-confident: " + (", ".join(f"{r['sensor']} ({r['z_med']:.2f})" for r in tab5
                                                   if calib_word(r["z_med"]) == "under") or "none")
              + ". A mean far above the median means heavy tails: long stretches where the sensor's error is much "
              "larger than its typical spread (the slow bias wanders), which a Gaussian weight does not expect.", ""]
        L += md_table(["sensor", "grid n", "z median", "z mean", "verdict", "zu n", "zu median", "zu mean"],
                      [[r["sensor"], r["n_grid"], fmt(r["z_med"], 3), fmt(r["z_mean"], 3), calib_word(r["z_med"]),
                        r["n_zu"], fmt(r["zu_med"], 3), fmt(r["zu_mean"], 3)] for r in tab5])
        L += ["", "## Vertical error (testing, drone_air grid samples; reported, not gated)", "",
              "The same smoother run on the heights as reported (raw) and on the heights after the ellipsoid rule (ell, "
              "what grid.npz holds), compared with the ground-truth up coordinate. |du| = median absolute error, bias = "
              "median signed error (positive: above the drone). STUG's altitude does not follow the drone, so it has no "
              "rule-1 height. In-sample: the height offsets (EUIG, ILYS, IUKL, UXYJ) were fitted on these same testing "
              "iterations. 'held-out' is `sensor_noise.csv` loio_held_out_u_m: the offset rule fitted without one "
              "iteration and measured on it (mean over iterations of |median height bias|), the honest guide to how "
              "the rule transfers; it is a different metric from 'ell abs', so compare it with the absolute value of 'ell bias'.", ""]
        loio_u = loio_height_reference()
        L += md_table(["sensor", "raw n", "raw abs", "raw bias", "ell n", "ell abs", "ell bias", "held-out abs bias"],
                      [[r["sensor"], r["n_ur"], fmt(r["ur_abs"]), fmt(r["ur_bias"]), r["n_ue"], fmt(r["ue_abs"]),
                        fmt(r["ue_bias"]), fmt(loio_u.get(r["sensor"], float("nan")))] for r in tab5])
    if corr_lines:
        L += [""] + corr_lines
    L += ["", "## Self-checks", ""]
    L += [f"* {n} {'OK' if ok else 'FAIL'}: {msg.splitlines()[0]}" for n, ok, msg in checks]
    (out / f"README{suffix}.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
