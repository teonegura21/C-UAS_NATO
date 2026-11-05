#!/usr/bin/env python3
"""Compact, documented dataset: one clean picture of the data organised by sensor and by drone.

Built from the derived tree that ``cuas_common.DERIVED`` points to: the main tree (data/derived)
by default, or a corrected variant tree when ``CUAS_DERIVED`` is set (``run.py --derived``). It
never reads the stale v1 ``sensor_tracks/`` directory or any scratch tree; every input path is
printed. Output: ``<tree>/compact/`` with five files, each column documented in the README.

Tables (all numbers are metres, seconds or shares; positions are LOCAL East / North metres in
the iteration's own frame; no latitude / longitude is ever written)
* sensors.csv           one row per (iteration, sensor): what the sensor is, how many reports it
                        produced, how long it was active, how its objects live, how many track
                        segments it has, its error model, and (testing) its error against the drones.
* drones.csv            testing only: one row per Dronetag flight (one drone on one flight).
* drones_by_sensor.csv  testing only: per flight and sensor, the seconds that sensor saw that drone.
* segments.csv          one row per sensor-track segment of sensor_tracks_v2 (all 9 iterations).
* objects.csv           one row per fused track of fused/ (all 9 iterations).
* README.md             what each table is, every column in one line, its source, the tree it
                        was built from, the row counts, a glossary.

Inputs (all inside the active tree; the first six are required)
  detections.csv.gz, groundtruth.csv.gz, sensor_tracks_v2/<slug>__tracks.csv + __grid.npz,
  fused/<slug>__tracks.csv + __members.csv, friendly_whitelist.csv; optional (columns stay blank
  and the README says so): existence/<slug>__p.csv, classify/<slug>__pclass.csv,
  bearing_support/<slug>__support.csv, sensor_noise.csv, sensor_tracks_v2/white_noise.csv,
  sensor_tracks_v2/corrections.csv, variant.json.

Definitions that need care
* usable reports: position sensors: ``Q.clean`` usable_pos and not far (exactly the rows that
  become segments); bearing-only sensors: non-duplicate bearing rows with a finite azimuth.
* object = one ``object_id`` of one sensor (that sensor's own track id). Lifetime = last minus
  first usable report. Reports per second per object = (n - 1) / lifetime for objects with at
  least two reports and a positive lifetime; the sensor row gives the median over its objects.
* active seconds = number of distinct whole seconds that contain at least one usable report.
* drone_air segment = a testing segment labelled so by sensor_tracks.py (>= 50 % of its rows
  match one Dronetag flight within the matching gate, >= 80 % of those to the same flight, and
  >= 50 % airborne). Error of a sample = horizontal / vertical distance between the segment's
  smoothed grid sample and the truth position of its best flight at the same 100 ms step.
  Heights carry the in-sample offsets fitted on all six testing iterations (optimistic).
* Fused-track labels (objects.csv classification / identity) use fuse.track_table's rule: Fixed
  Wing iff cls_fixed_share > 0.5, else Rotary Wing; identity Friendly iff the resolved serial is
  on the evaluation whitelist (friendly_whitelist.csv), else Hostile.

Self-checks (numbered, can fail; exit 1): 1 row counts and keys equal the source tables (segments,
objects, drones, sensors, report totals); 2 plain-loop recomputation of four sensors.csv columns on
one iteration equals the vectorised values; 3 no latitude / longitude value in any output (scan
against every coordinate in the inputs, with a planted control that must be caught); 4 a
non-existent CUAS_DERIVED fails cleanly with a message; 5 two full runs are byte-identical; 6 the
README documents exactly the columns the files have; 7 internal consistency (segment shares in
[0, 1], seconds seen <= airborne seconds, key uniqueness, joins complete); 8 inputs are the active
tree's v2 tables, never the stale v1 directory.

Run: python3 tools/compact.py [--out DIR] [--inputs-only] [--no-det-check]   (< 10 min)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuas_clean as Q  # noqa: E402
import cuas_common as C  # noqa: E402
import truth as T  # noqa: E402

TREE = C.DERIVED
OUT = TREE / "compact"
TRACKS_V2 = TREE / "sensor_tracks_v2"
FUSED = TREE / "fused"
STALE_V1 = C.BASE_DERIVED / "sensor_tracks"
READ_LOG: list[str] = []   # every file read while building the tables (read_dicts and the grid.npz loads), for check 8
STEP_S = 0.1
SHORT_LIFE_S = 5.0
FAMILY = {"RADAR": "radar", "CAMERA": "camera", "ACOUSTIC": "acoustic", "CYBER": "cyber",
          "OTHER": "UTM", "FUSION-NODE": "fusion-node"}
CSV_TOKEN = re.compile(r"-?\d+\.\d{7,}")   # outputs hold <= 6 decimals, so 7+ decimals can only be an unrounded input value
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def family_of(sensor_key: str) -> str:
    typ = sensor_key.split("_", 1)[1]
    return "passive-RF" if typ.startswith("PASSIVE-RF") else FAMILY.get(typ, typ.lower())


# --------------------------------------------------------------------------- #
# Column specs: name, kind (s str, i int, f float, u unix seconds), plain-language line, source
# --------------------------------------------------------------------------- #

ITER_COLS = [
    ("iter_key", "s", "Iteration id, e.g. testing/S1-I1 or evaluation/DII/S2-I1.", "iteration folder name"),
    ("scenario", "s", "Scenario number (S1..S4). Iterations of one scenario are correlated.", "iteration folder name"),
    ("same_scenario_in_other_tree", "i",
     "1 if the other tree (testing vs evaluation) has an iteration of the same scenario: {{SAME_SCEN}}",
     "computed from the iteration list"),
]

CLS_SHARES = ("cls_uav", "cls_uav_rotary", "cls_uav_fixed", "cls_uav_transmitter", "cls_air_vehicle", "cls_bird",
              "cls_manned", "cls_ground", "cls_unknown", "cls_blank")

SENSORS = ITER_COLS + [
    ("sensor_key", "s", "Sensor id and type as in the file names, e.g. EGOI_RADAR.", "sensor file names"),
    ("sensor_id", "s", "Four-letter anonymised sensor id.", "sensor file names"),
    ("family", "s", "radar / passive-RF / cyber / camera / acoustic / UTM / fusion-node.", "type part of the file name"),
    ("measurement", "s", "'position' (reports a 3-D or 2-D position) or 'bearing-only' (azimuth, maybe elevation).",
     "detections.csv.gz meas_kind"),
    ("meas_kinds", "s", "The meas_kind values seen, '+'-joined (pos3d, pos2d, pos2d_effective, bearing, bearing_el).",
     "detections.csv.gz meas_kind"),
    ("deployable", "i", "1 if the sensor id also exists in an evaluation iteration; 0 = testing-only ({{TESTING_ONLY}}). "
     "Never score or tune on testing-only sensors.", "cuas_common.deployable_sensor_ids()"),
    ("gt_feed", "i", "1 for sensors that relay the Dronetag feed itself ({{GT_FEED}}): comparing them with "
     "ground truth is leakage.", "cuas_clean.GT_FEED_SENSORS"),
    ("site_e_m", "f", "Sensor site, East metres in the iteration frame (median of the reported sensor positions); "
     "blank if the sensor reports no valid site (UTM services).", "detections.csv.gz sensor_e"),
    ("site_n_m", "f", "Sensor site, North metres in the iteration frame.", "detections.csv.gz sensor_n"),
    ("files", "i", "Number of raw sensor files of this sensor in the iteration.", "detections.csv.gz file"),
    ("rows_raw", "i", "Report rows in the raw files.", "detections.csv.gz"),
    ("rows_duplicate", "i", "Rows that are exact copies of an earlier row (same object, time, position).",
     "cuas_clean.flag_duplicates"),
    ("rows_usable", "i", "Rows that can be used as measurements (see README: usable reports).", "cuas_clean + rule above"),
    ("active_s", "i", "Whole seconds that contain at least one usable report.", "usable rows' times"),
    ("span_s", "f", "Last minus first usable report time.", "usable rows' times"),
    ("objects", "i", "Distinct object ids (the sensor's own track ids) among the usable rows.", "usable rows' object_id"),
    ("reports_per_s_median", "f", "Median over objects of reports per second, (n - 1) / lifetime.", "usable rows"),
    ("object_life_median_s", "f", "Median object lifetime (last minus first usable report).", "usable rows"),
    ("objects_lt5s_share", "f", "Share of objects whose lifetime is under 5 s.", "usable rows"),
    ("segments", "i", "Track segments of this sensor in sensor_tracks_v2 (objects cut at gaps > 30 s).", "sensor_tracks_v2 tracks.csv"),
    ("seg_dur_median_s", "f", "Median segment duration.", "sensor_tracks_v2 tracks.csv duration_s"),
    ("seg_dur_p90_s", "f", "90th percentile of segment duration.", "sensor_tracks_v2 tracks.csv duration_s"),
    ("seg_single_meas_share", "f", "Share of segments with exactly one measurement epoch.", "sensor_tracks_v2 tracks.csv n_meas"),
    ("sigma_e_m", "f", "Total error spread East, from the testing error model (m).", "sensor_noise.csv sigma_e_m"),
    ("sigma_n_m", "f", "Total error spread North (m).", "sensor_noise.csv sigma_n_m"),
    ("sigma_u_m", "f", "Total error spread Up after the height rule (m).", "sensor_noise.csv sigma_u_m"),
    ("height_offset_m", "f", "Height offset added to the reported height to reach the ellipsoid (m).", "sensor_noise.csv u_offset_m"),
    ("sigma_white_e_m", "f", "White (report-to-report jitter) part of the East error, estimated without ground truth (m).",
     "sensor_tracks_v2/white_noise.csv"),
    ("sigma_white_n_m", "f", "White part of the North error (m).", "sensor_tracks_v2/white_noise.csv"),
    ("noise_src", "s", "Where the segments' total sigma came from (most common over the sensor's segments).",
     "sensor_tracks_v2 tracks.csv noise_src"),
    ("corrections", "s", "Corrections applied to this sensor in this tree, '|'-joined names, or 'none'. 'clock_pooled' = the "
     "fixed pooled clock offset every tree applies (value in corrections_detail); the other names come from a variant "
     "tree's corrections.csv (the main tree has none).", "sensor_calibration.csv (applied, all) + sensor_tracks_v2/corrections.csv"),
    ("corrections_detail", "s", "Each applied number as name:parameter=value, ';'-joined; blank if none.",
     "sensor_calibration.csv + sensor_tracks_v2/corrections.csv"),
    ("in_sample_calibration", "s", "Which of this sensor's calibration numbers were fitted with ground truth on ALL six testing "
     "iterations and then applied to every iteration, including the testing one being scored (so testing errors are optimistic): "
     "'|'-joined from clock (pooled clock offset), height_offset, sigma; 'none' if all are absent or replaced by "
     "leave-one-iteration-out versions in a variant tree.", "sensor_noise.csv + sensor_calibration.csv + corrections.csv"),
    ("seg_drone_air", "i", "Testing: segments labelled drone_air (blank on evaluation).", "sensor_tracks_v2 tracks.csv label"),
    ("seg_drone_ground", "i", "Testing: segments on a parked drone.", "sensor_tracks_v2 tracks.csv label"),
    ("seg_clutter", "i", "Testing: segments that match no flight.", "sensor_tracks_v2 tracks.csv label"),
    ("air_samples", "i", "Testing: 100 ms grid samples of drone_air segments that have a truth position.", "grid.npz + truth.py"),
    ("err_h_median_m", "f", "Testing: median horizontal error of those samples (m).", "grid.npz + truth.py"),
    ("err_v_median_m", "f", "Testing: median vertical error of those samples (m, in-sample height offsets).", "grid.npz + truth.py"),
]

DRONES = ITER_COLS + [
    ("flight_key", "s", "Dronetag file stem; unique per flight.", "FriendDronesList file name"),
    ("drone_name", "s", "Drone name from the Dronetag file name.", "FriendDronesList file name"),
    ("side", "s", "BLUE (friendly) or RED (hostile).", "FriendDronesList file name"),
    ("identity", "s", "Friendly (BLUE) or Hostile (RED).", "truth.IDENTITY"),
    ("classification", "s", "Catalog class (Fixed Wing UAS / Rotary Wing UAS / ...), evidence-based.", "tools/drone_catalog.csv"),
    ("model", "s", "Catalog model / manufacturer label (dropdown).", "tools/drone_catalog.csv"),
    ("protocol", "s", "Catalog protocol (dropdown).", "tools/drone_catalog.csv"),
    ("serial", "s", "Catalog serial (dropdown).", "tools/drone_catalog.csv"),
    ("band", "s", "Catalog signal frequency band (dropdown).", "tools/drone_catalog.csv"),
    ("truth_rows_all", "i", "Rows of the flight on the 100 ms grid, parked periods included.", "truth.load_truth(all)"),
    ("truth_rows_airborne", "i", "Of those, rows above {{AIRBORNE_M}} m height over take-off.", "truth.load_truth(airborne rule)"),
    ("airborne_s", "f", "Airborne seconds (rows x 0.1 s).", "truth.load_truth"),
    ("start_rel_s", "f", "First truth row, seconds after the first grid time.", "truth.load_truth"),
    ("end_rel_s", "f", "Last truth row, seconds after the first grid time.", "truth.load_truth"),
    ("speed_median_airborne_ms", "f", "Median horizontal speed while airborne (m/s).", "truth.load_truth"),
    ("max_height_m", "f", "Largest height above take-off (m).", "truth.load_truth"),
    ("seen_any_deployable_s", "f", "Airborne seconds in which at least one deployable sensor has a drone_air segment of this flight. "
     "Sensors that relay the Dronetag feed ({{GT_FEED}}) are NOT counted: they would be leakage (they stay visible in drones_by_sensor.csv).",
     "grid.npz + sensor_tracks_v2 labels"),
    ("seen_2plus_deployable_s", "f", "Airborne seconds seen by two or more different deployable sensors (Dronetag-relay sensors excluded).", "same"),
    ("n_deployable_sensors_seen", "i", "Deployable sensors that saw this flight at all (airborne), Dronetag-relay sensors excluded.", "same"),
]

DRONES_BY_SENSOR = ITER_COLS + [
    ("flight_key", "s", "Dronetag file stem (joins drones.csv).", "FriendDronesList file name"),
    ("sensor_key", "s", "Sensor (joins sensors.csv).", "sensor file names"),
    ("deployable", "i", "1 deployable, 0 testing-only sensor.", "cuas_common.deployable_sensor_ids()"),
    ("gt_feed", "i", "1 if the sensor relays the Dronetag feed (leakage).", "cuas_clean.GT_FEED_SENSORS"),
    ("seconds_seen", "f", "Airborne seconds in which this sensor has a drone_air segment sample whose best flight is this flight "
     "(distinct 100 ms steps x 0.1).", "grid.npz + sensor_tracks_v2 labels"),
    ("share_of_airborne", "f", "seconds_seen / the flight's airborne seconds.", "computed"),
    ("n_segments", "i", "Drone_air segments of this sensor that follow this flight.", "sensor_tracks_v2 tracks.csv"),
    ("median_error_m", "f", "Median horizontal error of those samples (m).", "grid.npz + truth.py"),
    ("median_error_v_m", "f", "Median vertical error of those samples (m).", "grid.npz + truth.py"),
]

SEGMENTS = ITER_COLS[:2] + [
    ("seg_id", "i", "Segment id, unique within the iteration.", "sensor_tracks_v2 tracks.csv"),
    ("sensor_key", "s", "Sensor (joins sensors.csv).", "tracks.csv"),
    ("sensor_id", "s", "Four-letter sensor id.", "tracks.csv"),
    ("family", "s", "Sensor family.", "from sensor_key"),
    ("deployable", "i", "1 deployable, 0 testing-only sensor.", "cuas_common.deployable_sensor_ids()"),
    ("object_id", "s", "The sensor's own track id the segment was cut from.", "tracks.csv"),
    ("part", "i", "Index of the segment within its object (0, 1, ...).", "tracks.csv"),
    ("t0", "u", "First measurement time (unix seconds, clock-corrected).", "tracks.csv"),
    ("t1", "u", "Last measurement time (unix seconds).", "tracks.csv"),
    ("t0_rel_s", "f", "t0 minus the first grid time.", "tracks.csv"),
    ("t1_rel_s", "f", "t1 minus the first grid time.", "tracks.csv"),
    ("duration_s", "f", "t1 - t0.", "tracks.csv"),
    ("n_meas", "i", "Measurement epochs (after merging identical timestamps).", "tracks.csv"),
    ("n_rows", "i", "Report rows in the segment.", "tracks.csv"),
    ("n_grid", "i", "100 ms grid samples written for the segment.", "tracks.csv"),
    ("meas_rate_hz", "f", "n_meas / duration_s.", "tracks.csv"),
    ("path_len_m", "f", "Length of the smoothed horizontal path.", "tracks.csv"),
    ("speed_mean_ms", "f", "Mean smoothed horizontal speed.", "tracks.csv"),
    ("speed_p90_ms", "f", "90th percentile of the speed.", "tracks.csv"),
    ("speed_max_ms", "f", "Maximum speed.", "tracks.csv"),
    ("vu_abs_p90_ms", "f", "90th percentile of |vertical speed|.", "tracks.csv"),
    ("accel_p90_ms2", "f", "90th percentile of |change of velocity| / dt.", "tracks.csv"),
    ("turn_p90_degs", "f", "90th percentile of turn rate (deg/s).", "tracks.csv"),
    ("stationary_share", "f", "Share of epochs with speed under the stationary threshold.", "tracks.csv"),
    ("u_ell_med_m", "f", "Median height above the ellipsoid, as ENU up in the iteration frame (m).", "tracks.csv"),
    ("dz_med_m", "f", "Median reported height minus the sensor's own height (m).", "tracks.csv"),
    ("range_med_m", "f", "Median horizontal distance to the own sensor (m); blank for UTM.", "tracks.csv"),
    ("det_conf_mean", "f", "Mean detection confidence reported by the sensor.", "tracks.csv"),
    ("cls_uav", "f", "Share of rows the sensor itself calls UAV (no subclass).", "tracks.csv"),
    ("cls_uav_rotary", "f", "Share of rows the sensor calls rotary-wing UAV.", "tracks.csv"),
    ("cls_uav_fixed", "f", "Share of rows the sensor calls fixed-wing UAV.", "tracks.csv"),
    ("cls_bird", "f", "Share of rows the sensor calls bird / animal.", "tracks.csv"),
    ("cls_ground", "f", "Share of rows the sensor calls a ground object (land vehicle, home, ...).", "tracks.csv"),
    ("cls_unknown", "f", "Share of rows the sensor calls unknown / other.", "tracks.csv"),
    ("cls_uav_transmitter", "f", "Share of rows the sensor calls a UAV / EM transmitter (RF emitter, not a body).", "tracks.csv"),
    ("cls_air_vehicle", "f", "Share of rows the sensor calls a generic air vehicle.", "tracks.csv"),
    ("cls_manned", "f", "Share of rows the sensor calls a manned aircraft.", "tracks.csv"),
    ("cls_blank", "f", "Share of rows with no class at all. The ten cls_* shares (cls_uav ... cls_blank) sum to 1; "
     "all zero would mean a missing value, not 'no class'.", "tracks.csv"),
    ("cls_conf_mean", "f", "Mean classification confidence.", "tracks.csv"),
    ("model_top", "s", "Most common model string reported (blank if none).", "tracks.csv"),
    ("serial_top", "s", "Most common serial reported (blank if none).", "tracks.csv"),
    ("protocol_top", "s", "Most common protocol reported (blank if none).", "tracks.csv"),
    ("gt_feed", "i", "1 for ground-truth relay sensors (leakage on testing).", "tracks.csv"),
    ("noise_src", "s", "Where the total sigma came from.", "tracks.csv"),
    ("white_src", "s", "Where the white sigma came from.", "tracks.csv"),
    ("p_drone_all", "f", "Existence model: probability the segment is a real drone (truth incl. parked). Testing: leave-one-iteration-out prediction.",
     "existence/<slug>__p.csv"),
    ("p_drone_air", "f", "Same, truth = airborne drones only.", "existence/<slug>__p.csv"),
    ("p_fixed", "f", "Classifier: probability the drone is fixed wing (testing: leave-one-iteration-out).", "classify/<slug>__pclass.csv"),
    ("camera_hit_share", "f", "Share of overlapping camera bearing reports that point at the segment.", "bearing_support/<slug>__support.csv"),
    ("camera_n_overlap", "i", "Camera reports overlapping the segment in time.", "bearing_support"),
    ("acoustic_hit_share", "f", "Same for acoustic bearings.", "bearing_support"),
    ("acoustic_n_overlap", "i", "Acoustic reports overlapping in time.", "bearing_support"),
    ("rf_hit_share", "f", "Same for passive-RF direction finders.", "bearing_support"),
    ("rf_n_overlap", "i", "RF reports overlapping in time.", "bearing_support"),
    ("n_matched", "i", "Testing: report rows matched to a Dronetag flight.", "tracks.csv"),
    ("frac_matched", "f", "Testing: n_matched / n_rows.", "tracks.csv"),
    ("best_flight", "s", "Testing: the flight most rows match (joins drones.csv flight_key).", "tracks.csv"),
    ("frac_best", "f", "Testing: share of matched rows on best_flight.", "tracks.csv"),
    ("frac_airborne", "f", "Testing: share of those rows with the drone airborne.", "tracks.csv"),
    ("label", "s", "Testing: drone_air / drone_ground / clutter (rule in sensor_tracks.py).", "tracks.csv"),
]

OBJECTS = ITER_COLS[:2] + [
    ("track", "i", "Fused track index, unique within the iteration.", "fused/<slug>__tracks.csv"),
    ("track_id", "s", "UUID the submission uses for this track.", "fused tracks.csv"),
    ("sensor_ids", "s", "Sensor ids of the member segments, '|'-joined.", "fused tracks.csv"),
    ("n_members", "i", "Member segments fused into the track.", "fused tracks.csv"),
    ("n_clusters", "i", "Simultaneous clusters chained into the track (continuations).", "fused tracks.csv"),
    ("n_sensors", "i", "Distinct sensor ids among the members.", "fused tracks.csv"),
    ("member_seg_ids", "s", "Member segment ids ('|'-joined; join segments.csv on iter_key + seg_id).", "fused members.csv"),
    ("first_step", "i", "First grid step of the track.", "fused tracks.csv"),
    ("last_step", "i", "Last grid step.", "fused tracks.csv"),
    ("t0_rel_s", "f", "First step, seconds after the first grid time.", "template grid"),
    ("t1_rel_s", "f", "Last step, seconds after the first grid time.", "template grid"),
    ("duration_s", "f", "Duration of the track.", "fused tracks.csv"),
    ("n_bridged", "i", "Grid rows interpolated across short gaps.", "fused tracks.csv"),
    ("speed_med_ms", "f", "Median fused horizontal speed.", "fused tracks.csv"),
    ("speed_p10_ms", "f", "10th percentile of the speed.", "fused tracks.csv"),
    ("speed_p90_ms", "f", "90th percentile of the speed.", "fused tracks.csv"),
    ("u_med_m", "f", "Median fused height (ENU up, m).", "fused tracks.csv"),
    ("p_max_all", "f", "Largest member p_drone_all.", "fused tracks.csv"),
    ("p_max_air", "f", "Largest member p_drone_air.", "fused tracks.csv"),
    ("classification", "s", "Submitted class: Fixed Wing UAS if cls_fixed_share > 0.5, else Rotary Wing UAS.", "fuse.track_table rule"),
    ("identity", "s", "Friendly if the serial is on the evaluation whitelist, else Hostile.", "fuse.track_table rule"),
    ("model", "s", "Model / manufacturer label of the members (or Unknown).", "fused tracks.csv"),
    ("protocol", "s", "Protocol label (or Unknown).", "fused tracks.csv"),
    ("serial", "s", "Resolved serial (blank = none).", "fused tracks.csv"),
    ("cls_fixed_share", "f", "Weighted share of fixed-wing votes among the members.", "fused tracks.csv"),
    ("reported_f0", "i", "1 if the F0 method reports this track (existence above its threshold).", "fused tracks.csv"),
    ("drone_share", "f", "Testing: share of member samples labelled drone.", "fused tracks.csv"),
    ("label_drone", "i", "Testing: 1 if drone_share >= 0.5.", "fused tracks.csv"),
    ("flight", "s", "Testing: majority flight of a drone track (joins drones.csv flight_key).", "fused tracks.csv"),
    ("purity", "f", "Testing: share of the track's rows within 50 m (3-D) of that flight.", "fused tracks.csv"),
]

SPECS = {"sensors": SENSORS, "drones": DRONES, "drones_by_sensor": DRONES_BY_SENSOR,
         "segments": SEGMENTS, "objects": OBJECTS}
TABLE_TITLES = {
    "sensors": "One row per (iteration, sensor): identity, volume, activity, object lifetimes, segments, error model, testing error.",
    "drones": "Testing only. One row per Dronetag flight: who the drone is, how much truth exists, how well deployable sensors saw it.",
    "drones_by_sensor": "Testing only. Per flight and sensor: seconds that sensor followed that drone and its error.",
    "segments": "One row per sensor-track segment (sensor_tracks_v2), all 9 iterations, with model outputs and testing labels.",
    "objects": "One row per fused track (fused/), all 9 iterations, with members, labels and testing diagnostics.",
}


# --------------------------------------------------------------------------- #
# cells
# --------------------------------------------------------------------------- #


def cell(v, kind: str) -> str:
    if v is None:
        return ""
    if kind == "s":
        return str(v)
    if kind == "i":
        if isinstance(v, float) and v != v:
            return ""
        return str(int(v))
    v = float(v)
    if v != v or v in (math.inf, -math.inf):
        return ""
    return f"{v:.3f}" if kind == "u" else repr(round(v, 6))


def write_table(path: Path, spec, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow([s[0] for s in spec])
        for r in rows:
            w.writerow([cell(r.get(name), kind) for name, kind, *_ in spec])


def fnum(s: str) -> float:
    if s is None or s == "":
        return math.nan
    try:
        return float(s)
    except ValueError:
        return math.nan


def read_dicts(path: Path) -> list[dict]:
    READ_LOG.append(os.path.abspath(path))
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def nanmed(a, q=50):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else math.nan


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #


def slug(it) -> str:
    return C.iter_slug(it.key)


def required_paths(its) -> list[Path]:
    req = [TREE / "detections.csv.gz", TREE / "groundtruth.csv.gz", TREE / "friendly_whitelist.csv"]
    for it in its:
        s = slug(it)
        req += [TRACKS_V2 / f"{s}__tracks.csv", TRACKS_V2 / f"{s}__grid.npz",
                FUSED / f"{s}__tracks.csv", FUSED / f"{s}__members.csv"]
    return req


def require_inputs() -> int:
    """0 if every required input exists; else print what is missing and return 2 (clean failure)."""
    if not TREE.is_dir():
        print(f"compact.py: the derived tree {TREE} does not exist (CUAS_DERIVED={os.environ.get('CUAS_DERIVED', '')!r}). "
              "Build it first (tools/run.py) or unset CUAS_DERIVED.", file=sys.stderr)
        return 2
    its = C.iterations()
    missing = [p for p in required_paths(its) if not p.exists()]
    if missing:
        print(f"compact.py: {len(missing)} required input(s) missing in {TREE}:", file=sys.stderr)
        for p in missing[:12]:
            print(f"  missing: {rel(p)}", file=sys.stderr)
        return 2
    return 0


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(C.REPO))
    except ValueError:
        return Path(p).name


def scenario_flags(its) -> dict[str, int]:
    ev = {i.scenario for i in its if i.tree == "evaluation"}
    te = {i.scenario for i in its if i.tree == "testing"}
    return {i.key: int(i.scenario in ev and i.scenario in te) for i in its}


def iter_cols(it, flags) -> dict:
    return {"iter_key": it.key, "scenario": f"S{it.scenario}", "same_scenario_in_other_tree": flags[it.key]}


# --------------------------------------------------------------------------- #
# detections: per (iteration, sensor) volume / activity / objects
# --------------------------------------------------------------------------- #

DET_COLS = ["iter_key", "file", "row", "sensor_id", "sensor_key", "object_id", "t", "meas_kind",
            "sensor_pos_valid", "sensor_e", "sensor_n", "lat", "lon", "sensor_lat", "sensor_lon"]


def load_detections():
    cols = Q.load_clean(DET_COLS)
    return cols


def coordinate_values(det, gt) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Every latitude / longitude float in the inputs (exact-value leak scan), and the lat / lon bands of the
    sites (+-0.1 deg around the 1st..99th percentile of the valid sensor sites) for the column-band scan."""
    vals = [det[c] for c in ("lat", "lon", "sensor_lat", "sensor_lon")] + [gt["lat"], gt["lon"]]
    v = np.concatenate([np.asarray(x, float) for x in vals])
    ok = det["sensor_pos_valid"] == 1
    bands = []
    for c in ("sensor_lat", "sensor_lon"):
        x = np.asarray(det[c], float)[ok]
        x = x[np.isfinite(x)]
        bands.append((float(np.percentile(x, 1)) - 0.1, float(np.percentile(x, 99)) + 0.1))
    return np.unique(v[np.isfinite(v)]), bands


def group_minmax(codes: np.ndarray, vals: np.ndarray, n: int):
    order = np.argsort(codes, kind="stable")
    c, v = codes[order], vals[order]
    starts = np.searchsorted(c, np.arange(n))
    return np.minimum.reduceat(v, starts), np.maximum.reduceat(v, starts), np.diff(np.append(starts, c.size))


def usable_mask(det) -> np.ndarray:
    kind = det["meas_kind"]
    pos = np.isin(kind, Q.POSITION_KINDS)
    bearing = np.isin(kind, Q.BEARING_KINDS) & ~det["duplicate"] & np.isfinite(det["az_deg"])
    return (pos & det["usable_pos"] & ~det["far"]) | bearing


def sensor_volume(det) -> dict:
    """(iter_key, sensor_key) -> dict of the detection-derived sensor columns."""
    usable = usable_mask(det)
    keys = np.array([a + "\t" + b for a, b in zip(det["iter_key"], det["sensor_key"])], dtype=object)
    uniq, inv = np.unique(keys, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(uniq.size + 1))
    out = {}
    for gi, k in enumerate(uniq):
        idx = order[bounds[gi]:bounds[gi + 1]]
        iter_key, sensor_key = k.split("\t")
        u = idx[usable[idx]]
        t = det["t"][u]
        kinds = sorted(set(det["meas_kind"][idx].tolist()))
        ok = det["sensor_pos_valid"][idx] == 1
        rec = dict(
            sensor_id=str(det["sensor_id"][idx[0]]),
            measurement="bearing-only" if all(kk in Q.BEARING_KINDS for kk in kinds) else "position",
            meas_kinds="+".join(kinds),
            site_e_m=float(np.median(det["sensor_e"][idx][ok])) if ok.any() else math.nan,
            site_n_m=float(np.median(det["sensor_n"][idx][ok])) if ok.any() else math.nan,
            files=len(set(det["file"][idx].tolist())),
            rows_raw=int(idx.size), rows_duplicate=int(det["duplicate"][idx].sum()), rows_usable=int(u.size),
            gt_feed=int(det["sensor_id"][idx[0]] in Q.GT_FEED_SENSORS),
            _idx_usable=u)
        if u.size:
            rec["active_s"] = int(np.unique(np.floor(t).astype(np.int64)).size)
            rec["span_s"] = float(t.max() - t.min())
            ocodes = np.unique(det["object_id"][u], return_inverse=True)[1]
            nobj = int(ocodes.max()) + 1
            tmin, tmax, cnt = group_minmax(ocodes, t, nobj)
            life = tmax - tmin
            rate_ok = (cnt >= 2) & (life > 0)
            rec.update(objects=nobj, reports_per_s_median=nanmed((cnt[rate_ok] - 1) / life[rate_ok]) if rate_ok.any() else math.nan,
                       object_life_median_s=float(np.median(life)), objects_lt5s_share=float(np.mean(life < SHORT_LIFE_S)))
        else:
            rec.update(objects=0)
        out[(iter_key, sensor_key)] = rec
    return out


# --------------------------------------------------------------------------- #
# segments + model outputs
# --------------------------------------------------------------------------- #


def optional(path: Path):
    return path if path.exists() else None


def load_segments(its, flags, deployable) -> dict[str, list[dict]]:
    out = {}
    for it in its:
        s = slug(it)
        rows = read_dicts(TRACKS_V2 / f"{s}__tracks.csv")
        p = optional(TREE / "existence" / f"{s}__p.csv")
        pc = optional(TREE / "classify" / f"{s}__pclass.csv")
        bs = optional(TREE / "bearing_support" / f"{s}__support.csv")
        pmap = {r["seg_id"]: r for r in read_dicts(p)} if p else {}
        cmap = {r["seg_id"]: r for r in read_dicts(pc)} if pc else {}
        bmap = {r["seg_id"]: r for r in read_dicts(bs)} if bs else {}
        recs = []
        for r in rows:
            d = iter_cols(it, flags)
            d.update(seg_id=int(r["seg_id"]), sensor_key=r["sensor_key"], sensor_id=r["sensor_id"],
                     family=family_of(r["sensor_key"]), deployable=int(r["sensor_id"] in deployable),
                     object_id=r["object_id"])
            for name, kind, *_ in SEGMENTS:
                if name in d or name in ("p_drone_all", "p_drone_air", "p_fixed") or name.startswith(("camera_", "acoustic_", "rf_")):
                    continue
                if name in r:
                    d[name] = r[name] if kind == "s" else fnum(r[name])
            pr = pmap.get(r["seg_id"])
            if pr:
                d["p_drone_all"], d["p_drone_air"] = fnum(pr["p_drone_all"]), fnum(pr["p_drone_air"])
            cr = cmap.get(r["seg_id"])
            if cr:
                d["p_fixed"] = fnum(cr["p_fixed"])
            br = bmap.get(r["seg_id"])
            if br:
                for fam in ("camera", "acoustic", "rf"):
                    d[f"{fam}_hit_share"] = fnum(br[f"{fam}_hit_share"])
                    d[f"{fam}_n_overlap"] = fnum(br[f"{fam}_n_reports_overlap"])
            recs.append(d)
        out[it.key] = recs
    return out


def load_noise():
    """sensor_key -> dict from sensor_noise.csv (prefers a row of the iteration if the table gets an iter_key column)."""
    p = optional(TREE / "sensor_noise.csv")
    noise = {}
    if p:
        for r in read_dicts(p):
            noise.setdefault(r["sensor_key"], []).append(r)
    return noise


def noise_for(noise, sensor_key, iter_key):
    rows = noise.get(sensor_key, [])
    for r in rows:
        if r.get("iter_key") == iter_key:
            return r
    for r in rows:
        if r.get("iter_key", "") in ("", "*"):
            return r
    return {}


def load_white():
    p = optional(TRACKS_V2 / "white_noise.csv")
    return {r["sensor_key"]: r for r in read_dicts(p)} if p else {}


def load_pooled_clock() -> dict:
    """sensor_key -> pooled clock offset (s) that sensor_tracks applies in every tree (status applied, applies_to all)."""
    p = optional(TREE / "sensor_calibration.csv")
    out = {}
    if p:
        for r in read_dicts(p):
            if r["param"] == "time_offset_s" and r["status"] == "applied" and r["applies_to"] == "all":
                out[r["sensor_key"]] = fnum(r["value"])
    return out


def load_corrections():
    p = optional(TRACKS_V2 / "corrections.csv")
    out: dict = {}
    if p:
        for r in read_dicts(p):
            names, detail = out.setdefault((r["iter_key"], r["sensor_key"]), (set(), []))
            names.add(r["correction"].split(":")[0])
            detail.append(f"{r['correction']}={fnum(r['value']):g}")
    return out


# --------------------------------------------------------------------------- #
# testing: samples against truth
# --------------------------------------------------------------------------- #


def testing_samples(it, segs: list[dict], tr):
    """Grid samples of drone_air segments joined to the truth of their best flight at the same step.
    Returns dict of arrays: sensor_key (obj), flight (int), seg (int), step, err_h, err_v, airborne (bool)."""
    fidx = {m["flight_key"]: i for i, m in enumerate(tr.flights)}
    seg_flight = {}
    seg_sensor = {}
    for r in segs:
        if r.get("label") == "drone_air" and r.get("best_flight") in fidx:
            seg_flight[r["seg_id"]] = fidx[r["best_flight"]]
            seg_sensor[r["seg_id"]] = r["sensor_key"]
    empty = dict(sensor_key=np.array([], dtype=object), flight=np.zeros(0, np.int64), seg=np.zeros(0, np.int64),
                 step=np.zeros(0, np.int64), err_h=np.zeros(0), err_v=np.zeros(0), airborne=np.zeros(0, bool))
    if not seg_flight:
        return empty
    READ_LOG.append(os.path.abspath(TRACKS_V2 / f"{slug(it)}__grid.npz"))
    with np.load(TRACKS_V2 / f"{slug(it)}__grid.npz") as z:
        sid, step = z["seg_id"].astype(np.int64), z["step"].astype(np.int64)
        e, n, u = (z[c].astype(float) for c in ("e", "n", "u"))
    n_steps = int(tr.grid_t.size)
    lut = np.full(int(max(sid.max(), max(seg_flight)) + 1), -1, np.int64)
    for s, f in seg_flight.items():
        lut[s] = f
    fl = lut[sid]
    keep = fl >= 0
    sid, step, e, n, u, fl = sid[keep], step[keep], e[keep], n[keep], u[keep], fl[keep]
    tkey = tr.flight.astype(np.int64) * n_steps + tr.step
    order = np.argsort(tkey, kind="stable")
    ts = tkey[order]
    skey = fl * n_steps + step
    pos = np.clip(np.searchsorted(ts, skey), 0, max(ts.size - 1, 0))
    found = ts[pos] == skey if ts.size else np.zeros(skey.size, bool)
    ti = order[pos[found]]
    sid, step, e, n, u, fl = sid[found], step[found], e[found], n[found], u[found], fl[found]
    err_h = np.hypot(e - tr.e[ti], n - tr.n[ti])
    with np.errstate(invalid="ignore"):
        err_v = np.abs(u - tr.u[ti])
    sk = np.array([seg_sensor[int(s)] for s in sid], dtype=object) if sid.size else np.array([], dtype=object)
    return dict(sensor_key=sk, flight=fl, seg=sid, step=step, err_h=err_h, err_v=err_v, airborne=tr.airborne[ti])


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #


def build_all(its, flags, deployable):
    det = load_detections()
    gt = T._groundtruth()
    coords, bands = coordinate_values(det, gt)
    n_det_rows = int(len(det["t"]))
    log(f"detections loaded and cleaned: {n_det_rows:,} rows")
    vol = sensor_volume(det)
    log(f"per-sensor volumes: {len(vol)} (iteration, sensor) pairs")
    segs = load_segments(its, flags, deployable)
    noise, white, corr = load_noise(), load_white(), load_corrections()
    pooled_clock = load_pooled_clock()
    log(f"segments read: {sum(len(v) for v in segs.values()):,}")

    samples, truths = {}, {}
    for it in its:
        if it.has_groundtruth:
            truths[it.key] = T.load_truth(it, "all")
            samples[it.key] = testing_samples(it, segs[it.key], truths[it.key])
    log("testing samples joined to truth")

    sensors = []
    for it in its:
        s_it = {k[1]: v for k, v in vol.items() if k[0] == it.key}
        seg_by_sensor: dict[str, list[dict]] = {}
        for r in segs[it.key]:
            seg_by_sensor.setdefault(r["sensor_key"], []).append(r)
        sm = samples.get(it.key)
        for sk in sorted(s_it):
            v = s_it[sk]
            rec = iter_cols(it, flags)
            rec.update({k: x for k, x in v.items() if not k.startswith("_")})
            rec.update(sensor_key=sk, family=family_of(sk), deployable=int(v["sensor_id"] in deployable))
            ss = seg_by_sensor.get(sk, [])
            durs = np.array([r["duration_s"] for r in ss], float)
            rec["segments"] = len(ss)
            if ss:
                rec.update(seg_dur_median_s=float(np.median(durs)), seg_dur_p90_s=float(np.percentile(durs, 90)),
                           seg_single_meas_share=float(np.mean([r["n_meas"] == 1 for r in ss])))
                srcs = sorted(set(r["noise_src"] for r in ss))
                cnt = {s: sum(r["noise_src"] == s for r in ss) for s in srcs}
                rec["noise_src"] = max(srcs, key=lambda s: (cnt[s], s))
            nz = noise_for(noise, sk, it.key)
            rec.update(sigma_e_m=fnum(nz.get("sigma_e_m")), sigma_n_m=fnum(nz.get("sigma_n_m")),
                       sigma_u_m=fnum(nz.get("sigma_u_m")), height_offset_m=fnum(nz.get("u_offset_m")))
            wz = white.get(sk, {})
            rec.update(sigma_white_e_m=fnum(wz.get("sigma_white_e_m")), sigma_white_n_m=fnum(wz.get("sigma_white_n_m")))
            cn, cd = corr.get((it.key, sk), (set(), []))
            cn, cd = set(cn), list(cd)
            pc = pooled_clock.get(sk)
            clock_pooled = pc is not None and not any(n.startswith("clock") for n in cn)
            if clock_pooled:
                cn.add("clock_pooled")
                cd.append(f"clock_pooled:time_offset_s={pc:g}")
            rec["corrections"] = "|".join(sorted(cn)) if cn else "none"
            rec["corrections_detail"] = ";".join(cd)
            ins = []
            if clock_pooled:
                ins.append("clock")
            ho = rec.get("height_offset_m")
            if isinstance(ho, float) and ho == ho and ho != 0 and "height_loio" not in cn:
                ins.append("height_offset")
            if any(isinstance(rec.get(k), float) and rec[k] == rec[k] for k in ("sigma_e_m", "sigma_n_m", "sigma_u_m")) \
                    and "var_loio" not in cn:
                ins.append("sigma")
            rec["in_sample_calibration"] = "|".join(ins) if ins else "none"
            if it.has_groundtruth:
                lab = [r.get("label") for r in ss]
                rec.update(seg_drone_air=lab.count("drone_air"), seg_drone_ground=lab.count("drone_ground"),
                           seg_clutter=lab.count("clutter"))
                m = sm["sensor_key"] == sk
                rec["air_samples"] = int(m.sum())
                if m.any():
                    rec["err_h_median_m"] = float(np.median(sm["err_h"][m]))
                    rec["err_v_median_m"] = nanmed(sm["err_v"][m])
            sensors.append(rec)

    drones, by_sensor = [], []
    for it in its:
        if not it.has_groundtruth:
            continue
        tr, sm = truths[it.key], samples[it.key]
        rel0 = tr.grid_t[0]
        for fi, meta in enumerate(tr.flights):
            m = tr.flight == fi
            air = m & tr.airborne
            rec = iter_cols(it, flags)
            rec.update(flight_key=meta["flight_key"], drone_name=meta["drone_name"], side=meta["side"], identity=meta["identity"],
                       classification=meta["classification"], model=meta["uav_model/manufacturer"], protocol=meta["protocol"],
                       serial=meta["serial_number"], band=meta["signal_frequency_band"],
                       truth_rows_all=int(m.sum()), truth_rows_airborne=int(air.sum()), airborne_s=air.sum() * STEP_S)
            if m.any():
                rec.update(start_rel_s=float(tr.grid_t[tr.step[m].min()] - rel0), end_rel_s=float(tr.grid_t[tr.step[m].max()] - rel0),
                           max_height_m=float(tr.height[m].max()))
            if air.any():
                rec["speed_median_airborne_ms"] = float(np.median(tr.speed[air]))
            sel = (sm["flight"] == fi) & sm["airborne"]
            sens_here = sorted(set(sm["sensor_key"][sel].tolist()))
            steps_by_sensor = {}
            for sk in sens_here:
                ms = sel & (sm["sensor_key"] == sk)
                st = np.unique(sm["step"][ms])
                steps_by_sensor[sk] = st
                sid = sk.split("_")[0]
                b = iter_cols(it, flags)
                b.update(flight_key=meta["flight_key"], sensor_key=sk, deployable=int(sid in deployable),
                         gt_feed=int(sid in Q.GT_FEED_SENSORS), seconds_seen=st.size * STEP_S,
                         share_of_airborne=(st.size / air.sum()) if air.sum() else math.nan,
                         n_segments=int(np.unique(sm["seg"][ms]).size), median_error_m=float(np.median(sm["err_h"][ms])),
                         median_error_v_m=nanmed(sm["err_v"][ms]))
                by_sensor.append(b)
            dep = [sk for sk in sens_here if sk.split("_")[0] in deployable and sk.split("_")[0] not in Q.GT_FEED_SENSORS]
            if dep:
                allst = np.concatenate([steps_by_sensor[sk] for sk in dep])
                uq, cnt = np.unique(allst, return_counts=True)
                rec.update(seen_any_deployable_s=uq.size * STEP_S, seen_2plus_deployable_s=int((cnt >= 2).sum()) * STEP_S)
            else:
                rec.update(seen_any_deployable_s=0.0, seen_2plus_deployable_s=0.0)
            rec["n_deployable_sensors_seen"] = len(dep)
            drones.append(rec)

    whitelist = load_whitelist()
    seg_rows = [r for it in its for r in segs[it.key]]
    objects = build_objects(its, flags, whitelist)
    log(f"drones {len(drones)}, drones_by_sensor {len(by_sensor)}, segments {len(seg_rows):,}, objects {len(objects):,}")

    del gt
    return dict(sensors=sensors, drones=drones, drones_by_sensor=by_sensor, segments=seg_rows, objects=objects), \
        dict(det=det, vol=vol, n_det_rows=n_det_rows, coords=coords, bands=bands, samples=samples, truths=truths, segs=segs)


def load_whitelist() -> set[str]:
    out = set()
    for r in read_dicts(TREE / "friendly_whitelist.csv"):
        if r["iter_key"].startswith("evaluation") and r["serial_mapped"]:
            out.add(r["serial_mapped"])
    return out


def build_objects(its, flags, whitelist) -> list[dict]:
    rows = []
    for it in its:
        s = slug(it)
        grid = C.template_grid(it)[1]
        mem: dict[str, list[int]] = {}
        for r in read_dicts(FUSED / f"{s}__members.csv"):
            mem.setdefault(r["track"], []).append(int(r["seg_id"]))
        for r in read_dicts(FUSED / f"{s}__tracks.csv"):
            d = iter_cols(it, flags)
            for name, kind, *_ in OBJECTS:
                if name in d or name in ("member_seg_ids", "classification", "identity", "t0_rel_s", "t1_rel_s"):
                    continue
                if name in r:
                    d[name] = r[name] if kind == "s" else fnum(r[name])
            d["member_seg_ids"] = "|".join(str(x) for x in sorted(mem.get(r["track"], [])))
            fs, ls = int(r["first_step"]), int(r["last_step"])
            d["t0_rel_s"], d["t1_rel_s"] = float(grid[fs] - grid[0]), float(grid[ls] - grid[0])
            d["classification"] = "Fixed Wing UAS" if fnum(r["cls_fixed_share"]) > 0.5 else "Rotary Wing UAS"
            d["identity"] = "Friendly" if r["serial"] and r["serial"] in whitelist else "Hostile"
            rows.append(d)
    return rows


# --------------------------------------------------------------------------- #
# README
# --------------------------------------------------------------------------- #

GLOSSARY = [
    ("ENU / local metres", "East-North-Up distances in metres from a per-iteration origin. All positions here are of this kind; real coordinates are never written (the data's longitudes are anonymised anyway)."),
    ("iteration", "One recording session of one scenario (e.g. testing S1-I1). Testing iterations have ground truth, evaluation iterations do not."),
    ("sensor / family", "One physical sensor (a four-letter id) and its kind: radar, passive-RF direction finder, cyber, camera, acoustic, UTM service, fusion node."),
    ("deployable", "A sensor id that also appears in the evaluation data. Testing-only sensors ({{TESTING_ONLY}}) must not drive anything that is later applied to evaluation."),
    ("object", "A sensor's own track id (object_id). One real drone can be several objects, and several sensors each have their own."),
    ("segment", "An object cut at gaps over 30 s, smoothed, and written on the 100 ms scoring grid (sensor_tracks_v2)."),
    ("fused track / object (objects.csv)", "Segments of different sensors that were judged to be the same drone and merged into one track (fuse.py)."),
    ("Dronetag flight", "The ground-truth GPS log of one drone on one flight; BLUE = friendly, RED = hostile. Only testing has them."),
    ("drone_air / drone_ground / clutter", "Testing label of a segment: it follows an airborne drone / a parked drone / nothing known."),
    ("airborne", "More than {{AIRBORNE_M}} m above the take-off height."),
    ("p_drone, p_fixed", "Model outputs: probability the segment is a real drone / a fixed-wing drone. On testing they are leave-one-iteration-out predictions, so a model never saw the iteration it scores."),
    ("leave-one-iteration-out (LOIO)", "Fit on the other testing iterations, predict the held-out one."),
    ("duplicate row", "A row identical to an earlier one (same object, time, position); {{DUP_PCT}} % of all rows."),
    ("gt_feed", "A sensor that relays the Dronetag feed itself; its agreement with ground truth is leakage, not accuracy."),
    ("grid sample", "One 100 ms step of the submission template's time grid."),
]


def fill_map(tables: dict, its, flags) -> dict:
    """Facts quoted in README prose, computed from the tables / modules instead of typed in."""
    def short(i):
        return i.key.split("/")[-1]
    te_same = [short(i) for i in its if i.tree == "testing" and flags[i.key]]
    te_other = [short(i) for i in its if i.tree == "testing" and not flags[i.key]]
    ev = [short(i) for i in its if i.tree == "evaluation"]
    raw = sum(r["rows_raw"] for r in tables["sensors"])
    dup = sum(r["rows_duplicate"] for r in tables["sensors"])
    testing_only = sorted({r["sensor_id"] for r in tables["sensors"] if not r["deployable"]})
    return {
        "{{SAME_SCEN}}": f"testing {' and '.join(te_same)} (with evaluation {', '.join(ev)}) and all evaluation iterations; "
                         f"0 for {', '.join(te_other)}.",
        "{{SCEN_T}}": " and ".join(te_same),
        "{{SCEN_E}}": ", ".join(ev),
        "{{TESTING_ONLY}}": ", ".join(testing_only),
        "{{GT_FEED}}": ", ".join(sorted(Q.GT_FEED_SENSORS)),
        "{{AIRBORNE_M}}": f"{T.AIRBORNE_HEIGHT_M:g}",
        "{{DUP_PCT}}": f"{100 * dup / raw:.0f}" if raw else "?",
    }


def readme_text(tables: dict, ctx: dict, its, flags, deployable, out_dir_label: str) -> str:
    fm = fill_map(tables, its, flags)

    def fill(t: str) -> str:
        for k, v in fm.items():
            t = t.replace(k, v)
        return t

    L = ["# Compact dataset: the data by sensor and by drone", ""]
    vj = optional(TREE / "variant.json")
    tree_rel = rel(TREE)
    L += ["Built by `tools/compact.py` from the derived tree `" + tree_rel + "/` "
          + ("(a variant tree; variant.json: `" + json.dumps(json.loads(vj.read_text()), sort_keys=True) + "`)." if vj
             else ("(the main tree)." if TREE == C.BASE_DERIVED else "(a tree without variant.json, selected with CUAS_DERIVED).")),
          "Inputs read: " + ", ".join(f"`{x}`" for x in (
              "detections.csv.gz", "groundtruth.csv.gz", "friendly_whitelist.csv", "sensor_tracks_v2/", "fused/",
              "existence/", "classify/", "bearing_support/", "sensor_noise.csv", "sensor_tracks_v2/white_noise.csv",
              "sensor_tracks_v2/corrections.csv") if (TREE / x).exists()) + ".",
          "Optional inputs missing from this tree leave their columns blank: "
          + (", ".join(f"`{x}`" for x in ("existence/", "classify/", "bearing_support/", "sensor_noise.csv",
                                           "sensor_tracks_v2/white_noise.csv", "sensor_tracks_v2/corrections.csv")
                       if not (TREE / x).exists() and x != "sensor_tracks_v2/corrections.csv") or "none") + ".",
          ("This tree has a `corrections.csv` (variant corrections, listed per sensor in the corrections column)."
           if (TREE / "sensor_tracks_v2/corrections.csv").exists() else
           "No `corrections.csv` in this tree (no variant corrections). The corrections column still lists the fixed clock offsets "
           "every tree applies (`clock_pooled`), and says `none` only for sensors without one."),
          "Calibration caveat: the clock offsets, the total error spreads (sigma) and the height offsets are fitted with ground truth on "
          "all six testing iterations at once and then applied to every iteration. Testing errors computed with them are in-sample "
          "(optimistic); the column `in_sample_calibration` marks which sensors carry such numbers.",
          "The stale v1 directory `sensor_tracks/` is never read. No latitude or longitude appears anywhere; positions are local East / North metres.",
          "Blank cell = not defined (e.g. an error against ground truth on an evaluation iteration).", ""]
    L += ["## Row counts", "", "| table | rows |", "|---|---|"]
    for name in SPECS:
        L.append(f"| {name}.csv | {len(tables[name]):,} |")
    L.append("")
    L += ["## Scenarios (read before pooling)", "",
          fill("Testing iterations {{SCEN_T}} come from the same scenarios as evaluation {{SCEN_E}}; the iterations of one scenario "
               "are correlated with each other (the three S3 iterations in particular). ")
          + "Count wins per iteration, and mind the per-scenario counts below. Column `same_scenario_in_other_tree` marks this.", "",
          "| scenario | testing iterations | evaluation iterations | flights (testing) | segments | fused tracks |", "|---|---|---|---|---|---|"]
    for sc in sorted({i.scenario for i in its}):
        te = [i.key for i in its if i.scenario == sc and i.tree == "testing"]
        ev = [i.key for i in its if i.scenario == sc and i.tree == "evaluation"]
        keys = set(te + ev)
        L.append(f"| S{sc} | {len(te)} ({', '.join(k.split('/')[-1] for k in te) or '-'}) | {len(ev)} ({', '.join(k.split('/')[-1] for k in ev) or '-'}) | "
                 f"{sum(1 for r in tables['drones'] if r['scenario'] == f'S{sc}')} | "
                 f"{sum(1 for r in tables['segments'] if r['iter_key'] in keys):,} | {sum(1 for r in tables['objects'] if r['iter_key'] in keys):,} |")
    L.append("")
    for name, spec in SPECS.items():
        L += [f"## {name}.csv", "", TABLE_TITLES[name], "", "| column | meaning | source |", "|---|---|---|"]
        for cname, kind, desc, src in spec:
            L.append(f"| `{cname}` | {fill(desc)} | {fill(src)} |")
        L.append("")
    L += ["## Glossary", ""] + [f"* **{a}**: {fill(b)}" for a, b in GLOSSARY] + [""]
    L += ["## How the numbers are defined", "",
          "* Usable reports: position sensors: not a duplicate, not a self-report (within 5 m of the own site), not null-island, finite position, within 15 km of the origin (exactly the rows that become segments); bearing-only sensors: non-duplicate bearing rows with a finite azimuth.",
          "* Reports per second per object: (n - 1) / lifetime over objects with at least two usable reports and positive lifetime; the sensor value is the median over objects.",
          "* Active seconds: distinct whole seconds containing at least one usable report.",
          "* Error against ground truth (testing): smoothed grid samples of drone_air segments minus the truth position of the segment's best flight at the same 100 ms step. Heights use offsets fitted on all six testing iterations, so vertical errors are optimistic.",
          "* Seconds seen (drones_by_sensor.csv): distinct 100 ms steps, while the drone is airborne, covered by a drone_air segment of that sensor whose best flight is this one, times 0.1 s.",
          ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# self-checks
# --------------------------------------------------------------------------- #


def plain_sensor_check(it_key: str, rec_by_sensor: dict, det, vol, samples_it, tr, segs_it) -> list[str]:
    """Check 2: recompute four sensors.csv columns of one iteration with plain loops."""
    problems = []
    usable = usable_mask(det)
    iters, sks, objs, ts = det["iter_key"], det["sensor_key"], det["object_id"], det["t"]
    life: dict = {}
    for i in np.flatnonzero(usable & (iters == it_key)):
        d = life.setdefault(sks[i], {}).setdefault(objs[i], [ts[i], ts[i]])
        if ts[i] < d[0]:
            d[0] = ts[i]
        if ts[i] > d[1]:
            d[1] = ts[i]
    truth_xy = {}
    if tr is not None:
        for j in range(tr.step.size):
            truth_xy[(int(tr.flight[j]), int(tr.step[j]))] = (tr.e[j], tr.n[j])
    fidx = {m["flight_key"]: i for i, m in enumerate(tr.flights)} if tr is not None else {}
    seg_info = {r["seg_id"]: r for r in segs_it}
    errs: dict = {}
    if tr is not None:
        with np.load(TRACKS_V2 / f"{C.iter_slug(it_key)}__grid.npz") as z:
            sid, st, e, n = z["seg_id"], z["step"], z["e"].astype(float), z["n"].astype(float)
        for a, b, x, y in zip(sid.tolist(), st.tolist(), e.tolist(), n.tolist()):
            r = seg_info[a]
            if r.get("label") != "drone_air" or r.get("best_flight") not in fidx:
                continue
            tx = truth_xy.get((fidx[r["best_flight"]], b))
            if tx is not None:
                errs.setdefault(r["sensor_key"], []).append(math.hypot(x - tx[0], y - tx[1]))
    import statistics
    for sk, rec in rec_by_sensor.items():
        lv = [b - a for a, b in life.get(sk, {}).values()]
        exp_obj = len(lv)
        got = rec.get("objects", 0)
        if exp_obj != got:
            problems.append(f"{sk} objects: plain {exp_obj} vs vectorised {got}")
        if lv:
            med = statistics.median(lv)
            if abs(med - rec["object_life_median_s"]) > 1e-9 * max(1.0, abs(med)):
                problems.append(f"{sk} object_life_median_s: plain {med} vs vectorised {rec['object_life_median_s']}")
            sh = sum(1 for x in lv if x < SHORT_LIFE_S) / len(lv)
            if abs(sh - rec["objects_lt5s_share"]) > 1e-12:
                problems.append(f"{sk} objects_lt5s_share: plain {sh} vs {rec['objects_lt5s_share']}")
        if tr is not None:
            ev = errs.get(sk, [])
            if (len(ev) != rec.get("air_samples", 0)):
                problems.append(f"{sk} air_samples: plain {len(ev)} vs {rec.get('air_samples')}")
            elif ev:
                med = statistics.median(ev)
                if abs(med - rec["err_h_median_m"]) > 1e-9 * max(1.0, med):
                    problems.append(f"{sk} err_h_median_m: plain {med} vs {rec['err_h_median_m']}")
    return problems


def scan_for_coordinates(paths: list[Path], coords: np.ndarray, bands: list[tuple[float, float]]) -> list[str]:
    """Leak scan, three tests. (a) any decimal token with >= 7 decimals equal (1e-9) to a latitude / longitude of the
    inputs (outputs are rounded to 6 decimals, so such a token is an unrounded input value); (b) a column name that
    looks like a coordinate; (c) a numeric column of which more than half of the cells lie in the sites' latitude
    or longitude band (catches a rounded or shifted coordinate column; no metric column does)."""
    bad = []
    for p in paths:
        text = p.read_text(encoding="utf-8")
        for m in CSV_TOKEN.finditer(text):
            v = float(m.group())
            j = np.searchsorted(coords, v)
            if any(0 <= k < coords.size and abs(coords[k] - v) < 1e-9 for k in (j - 1, j)):
                bad.append(f"{p.name}: exact coordinate {m.group()}")
                break
        if p.suffix != ".csv":
            continue
        rows = list(csv.reader(text.splitlines()))
        header = rows[0]
        for ci, name in enumerate(header):
            if re.fullmatch(r"(sensor_)?(lat|lon|lng|latitude|longitude)(_deg)?", name.lower()):
                bad.append(f"{p.name}: coordinate-like column name {name}")
            vals = []
            for r in rows[1:]:
                try:
                    vals.append(float(r[ci]))
                except (ValueError, IndexError):
                    pass
            if len(vals) >= 20:
                a = np.array(vals)
                for lo, hi in bands:
                    if np.mean((a >= lo) & (a <= hi)) > 0.5:
                        bad.append(f"{p.name}: column {name} lies in a latitude / longitude band")
    return bad


def run_checks(tables, ctx, its, flags, deployable, out: Path, det_check: bool) -> int:
    fails = 0

    def verdict(n: int, name: str, probs: list[str]) -> None:
        nonlocal fails
        if probs:
            fails += 1
            print(f"FAIL {n} {name}")
            for p in probs[:8]:
                print(f"     {p}")
        else:
            print(f"ok   {n} {name}")

    # 1 counts and keys against the sources
    pr = []
    src_seg = {}
    for it in its:
        with (TRACKS_V2 / f"{slug(it)}__tracks.csv").open(newline="", encoding="utf-8") as fh:
            rd = csv.reader(fh)
            h = next(rd)
            i_seg = h.index("seg_id")
            src_seg[it.key] = [int(r[i_seg]) for r in rd]
    if sorted((r["iter_key"], r["seg_id"]) for r in tables["segments"]) != sorted((k, s) for k, v in src_seg.items() for s in v):
        pr.append("segments keys differ from the tracks.csv rows")
    pr_cnt = sum(len(v) for v in src_seg.values())
    if len(tables["segments"]) != pr_cnt:
        pr.append(f"segments {len(tables['segments'])} != sum of tracks.csv rows {pr_cnt}")
    src_obj = []
    for it in its:
        with (FUSED / f"{slug(it)}__tracks.csv").open(newline="", encoding="utf-8") as fh:
            rd = csv.reader(fh)
            h = next(rd)
            i_t = h.index("track")
            src_obj += [(it.key, int(r[i_t])) for r in rd]
    if sorted((r["iter_key"], r["track"]) for r in tables["objects"]) != sorted(src_obj):
        pr.append(f"objects keys differ from fused tracks ({len(tables['objects'])} vs {len(src_obj)})")
    flights = [(it.key, gf.flight_key) for it in its if it.has_groundtruth for gf in C.groundtruth_files(it)]
    if sorted((r["iter_key"], r["flight_key"]) for r in tables["drones"]) != sorted(flights):
        pr.append(f"drones keys differ from Dronetag files ({len(tables['drones'])} vs {len(flights)})")
    if sum(r["rows_raw"] for r in tables["sensors"]) != ctx["n_det_rows"]:
        pr.append("sum of sensors rows_raw != detections rows")
    pairs = {(r["iter_key"], r["sensor_key"]) for r in tables["sensors"]}
    if pairs != set(ctx["vol"]) or len(pairs) != len(tables["sensors"]):
        pr.append("sensors keys differ from the detection groups")
    for it in its:
        nf = {}
        for sfile in C.sensor_files(it):
            nf[sfile.sensor_key] = nf.get(sfile.sensor_key, 0) + 1
        for r in tables["sensors"]:
            if r["iter_key"] == it.key and r["sensor_key"] in nf and r["files"] != nf[r["sensor_key"]]:
                pr.append(f"{it.key} {r['sensor_key']} files {r['files']} != sensor_files {nf[r['sensor_key']]}")
    seg_sensor_n = {}
    for r in tables["segments"]:
        seg_sensor_n[(r["iter_key"], r["sensor_key"])] = seg_sensor_n.get((r["iter_key"], r["sensor_key"]), 0) + 1
    for r in tables["sensors"]:
        if r["segments"] != seg_sensor_n.get((r["iter_key"], r["sensor_key"]), 0):
            pr.append(f"{r['iter_key']} {r['sensor_key']} segments count mismatch")
    if sum(r["segments"] for r in tables["sensors"]) != len(tables["segments"]):
        pr.append("sum of sensors.segments != segments rows (a segment's sensor has no detection rows)")
    verdict(1, "row counts and keys equal the source tables", pr)

    # 2 plain-loop recomputation (testing iteration with the most sensors that have segments)
    key = "testing/S1-I1"
    det = ctx.pop("det")
    recs = {r["sensor_key"]: r for r in tables["sensors"] if r["iter_key"] == key}
    itobj = next(i for i in its if i.key == key)
    prob = plain_sensor_check(key, recs, det, ctx["vol"], ctx["samples"][key], ctx["truths"][key], ctx["segs"][key])
    verdict(2, f"plain-loop recomputation of objects / object_life_median_s / objects_lt5s_share / air_samples + err_h_median_m on {key}", prob)
    del det, itobj
    import gc
    gc.collect()

    # 3 no coordinates (with a planted control)
    files = sorted(out.glob("*.csv")) + [out / "README.md"]
    prob = scan_for_coordinates(files, ctx["coords"], ctx["bands"])
    ctl = Path(tempfile.mkdtemp(prefix="compact_ctl_"))
    try:
        sample = ctx["coords"][ctx["coords"].size // 2]
        lat_band = ctx["bands"][0]
        mid = (lat_band[0] + lat_band[1]) / 2
        (ctl / "exact.csv").write_text(f"a,b\n1,{repr(float(sample))}\n", encoding="utf-8")
        (ctl / "name.csv").write_text("lat,x\n1,2\n", encoding="utf-8")
        (ctl / "band.csv").write_text("a,b\n" + "".join(f"1,{mid + i * 1e-4:.6f}\n" for i in range(30)), encoding="utf-8")
        caught = all(scan_for_coordinates([ctl / n], ctx["coords"], ctx["bands"]) for n in ("exact.csv", "name.csv", "band.csv"))
        clean_ok = not scan_for_coordinates([ctl / "exact.csv"], np.array([0.5]), [])
    finally:
        shutil.rmtree(ctl, ignore_errors=True)
    if not caught or not clean_ok:
        prob.append("the scanner missed a planted coordinate (control failed)")
    verdict(3, f"no latitude / longitude in any output ({ctx['coords'].size:,} input coordinates scanned; planted exact / name / band controls caught)", prob)

    # 4 clean failure on a missing tree
    miss = Path(tempfile.gettempdir()) / "compact_no_such_tree_dir"
    shutil.rmtree(miss, ignore_errors=True)
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--inputs-only"], capture_output=True, text=True,
                       env={**os.environ, "CUAS_DERIVED": str(miss)})
    prob = []
    if r.returncode == 0:
        prob.append("exit status 0 on a missing tree")
    if "does not exist" not in r.stderr or "Traceback" in r.stderr:
        prob.append(f"unclear failure message: {r.stderr[-200:]!r}")
    if miss.exists():
        prob.append("the missing tree was created")
    verdict(4, "CUAS_DERIVED pointing to a missing tree fails cleanly with a message", prob)

    # 6 README documents exactly the columns
    prob = []
    readme = (out / "README.md").read_text(encoding="utf-8")
    for name, spec in SPECS.items():
        with (out / f"{name}.csv").open(newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        if header != [s[0] for s in spec]:
            prob.append(f"{name}.csv header differs from the spec")
        sect = readme.split(f"## {name}.csv", 1)[1].split("\n## ", 1)[0]
        docs = re.findall(r"^\| `([^`]+)` \|", sect, flags=re.M)
        if docs != header:
            prob.append(f"{name}: README documents {len(docs)} columns, file has {len(header)}")
    verdict(6, "README documents exactly the columns of every file", prob)

    # 7 consistency
    prob = []
    for r in tables["segments"]:
        for c in ("stationary_share",) + CLS_SHARES + (
                  "p_drone_all", "p_drone_air", "p_fixed", "camera_hit_share", "acoustic_hit_share", "rf_hit_share"):
            v = r.get(c)
            if isinstance(v, float) and v == v and not (-1e-9 <= v <= 1 + 1e-9):
                prob.append(f"segments {r['iter_key']} {r['seg_id']} {c}={v}")
                break
    bad_sum = [r for r in tables["segments"] if abs(sum(r[c] for c in CLS_SHARES) - 1.0) > 1e-3]
    if bad_sum:
        prob.append(f"{len(bad_sum)} segments whose ten cls_* shares do not sum to 1, e.g. {bad_sum[0]['iter_key']} seg {bad_sum[0]['seg_id']}")
    if len({(r["iter_key"], r["seg_id"]) for r in tables["segments"]}) != len(tables["segments"]):
        prob.append("duplicate segment keys")
    dr = {(r["iter_key"], r["flight_key"]): r for r in tables["drones"]}
    for b in tables["drones_by_sensor"]:
        d = dr.get((b["iter_key"], b["flight_key"]))
        if d is None:
            prob.append(f"drones_by_sensor flight {b['flight_key']} not in drones.csv")
        elif b["seconds_seen"] > d["airborne_s"] + 1e-9:
            prob.append(f"{b['flight_key']} {b['sensor_key']} seconds_seen {b['seconds_seen']} > airborne {d['airborne_s']}")
    for d in dr.values():
        if d["seen_any_deployable_s"] > d["airborne_s"] + 1e-9 or d["seen_2plus_deployable_s"] > d["seen_any_deployable_s"] + 1e-9:
            prob.append(f"{d['flight_key']} seen seconds inconsistent")
    segkeys = {(r["iter_key"], r["seg_id"]) for r in tables["segments"]}
    for o in tables["objects"]:
        if o["member_seg_ids"] and any((o["iter_key"], int(x)) not in segkeys for x in o["member_seg_ids"].split("|")):
            prob.append(f"object {o['iter_key']} {o['track']} has a member segment missing from segments.csv")
            break
    if any(r.get("measurement") is None for r in tables["sensors"]):
        prob.append("sensor without measurement type")
    for r in tables["sensors"]:
        if r["rows_duplicate"] > r["rows_raw"] or r["rows_usable"] > r["rows_raw"] - 0:
            prob.append(f"{r['iter_key']} {r['sensor_key']} row counts inconsistent")
    verdict(7, "internal consistency (shares in [0, 1], seconds seen <= airborne, joins complete, key uniqueness)", prob)

    # 8 inputs are the active tree's v2 (what was actually opened, not what the constants say)
    prob = []
    v2 = os.path.abspath(TREE / "sensor_tracks_v2")
    stale = {os.path.abspath(STALE_V1), os.path.realpath(STALE_V1), os.path.abspath(TREE / "sensor_tracks")}
    if os.path.abspath(TRACKS_V2) != v2:
        prob.append(f"TRACKS_V2 {rel(TRACKS_V2)} is not <active tree>/sensor_tracks_v2")
    read = set(READ_LOG)
    for f in sorted(read):
        if any(f == st or f.startswith(st + os.sep) or os.path.realpath(f).startswith(st + os.sep) for st in stale):
            prob.append(f"read from the stale v1 directory: {rel(Path(f))}")
    tr_read = {f for f in read if os.path.basename(f).endswith("__tracks.csv") and os.path.dirname(f) == v2}
    grid_read = {f for f in read if os.path.basename(f).endswith("__grid.npz") and os.path.dirname(f) == v2}
    if len(tr_read) != len(its):
        prob.append(f"{len(tr_read)} v2 tracks.csv files were read, expected {len(its)} (one per iteration)")
    if len(grid_read) < sum(1 for i in its if i.has_groundtruth):
        prob.append(f"{len(grid_read)} v2 grid.npz files were read, expected at least one per testing iteration")
    other = {f for f in read if os.path.basename(f).endswith("__tracks.csv") and os.path.dirname(f) != v2
             and os.path.basename(os.path.dirname(f)) != "fused"}
    if other:
        prob.append(f"tracks.csv read from outside the active sensor_tracks_v2: {sorted(rel(Path(f)) for f in other)[:3]}")
    if not (TRACKS_V2 / "README.md").exists() and TREE == C.BASE_DERIVED:
        prob.append("main tree sensor_tracks_v2 has no README.md")
    verdict(8, f"segment inputs actually read are {rel(TRACKS_V2)} ({len(tr_read)} tracks.csv, {len(grid_read)} grid.npz; stale v1 sensor_tracks/ never opened)", prob)

    # 5 determinism: a second full run into a scratch directory, byte-compared
    if det_check:
        scratch = Path(tempfile.mkdtemp(prefix="compact_det_"))
        try:
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--out", str(scratch), "--no-det-check", "--no-checks"],
                               capture_output=True, text=True)
            prob = []
            if r.returncode != 0:
                prob.append(f"second run failed: {r.stderr[-300:]}")
            else:
                for f in sorted(out.iterdir()):
                    g = scratch / f.name
                    if not g.exists() or g.read_bytes() != f.read_bytes():
                        prob.append(f"{f.name} differs between two runs")
                if {x.name for x in scratch.iterdir()} != {x.name for x in out.iterdir()}:
                    prob.append("file sets differ")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        verdict(5, "two full runs are byte-identical", prob)
    return fails


# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--inputs-only", action="store_true", help="only check that the inputs exist")
    ap.add_argument("--no-det-check", action="store_true", help="skip self-check 5 (the second run)")
    ap.add_argument("--no-checks", action="store_true", help="build only (used by the determinism check)")
    args = ap.parse_args()

    rc = require_inputs()
    if rc or args.inputs_only:
        return rc
    its = C.iterations()
    flags = scenario_flags(its)
    deployable = C.deployable_sensor_ids()
    log(f"tree: {rel(TREE)}  ({len(its)} iterations)")
    for p in (TREE / "detections.csv.gz", TRACKS_V2, FUSED):
        log(f"input: {rel(p)}")

    tables, ctx = build_all(its, flags, deployable)
    for name, spec in SPECS.items():
        tables[name].sort(key=lambda r: ([r["iter_key"]] + [r.get(k) for k in SORT_KEYS[name]]))
        write_table(args.out / f"{name}.csv", spec, tables[name])
    (args.out / "README.md").write_text(readme_text(tables, ctx, its, flags, deployable, "compact"), encoding="utf-8")
    for name in SPECS:
        log(f"{name}.csv: {len(tables[name]):,} rows")
    if args.no_checks:
        return 0
    print("\nSelf-checks")
    fails = run_checks(tables, ctx, its, flags, deployable, args.out, not args.no_det_check)
    print(f"\n{'ALL SELF-CHECKS PASSED' if not fails else str(fails) + ' SELF-CHECK(S) FAILED'}; "
          f"row counts: " + ", ".join(f"{n} {len(tables[n]):,}" for n in SPECS))
    return 1 if fails else 0


SORT_KEYS = {"sensors": ["sensor_key"], "drones": ["flight_key"], "drones_by_sensor": ["flight_key", "sensor_key"],
             "segments": ["seg_id"], "objects": ["track"]}

if __name__ == "__main__":
    sys.exit(main())
