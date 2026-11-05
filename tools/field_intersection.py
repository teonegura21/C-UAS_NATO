#!/usr/bin/env python3
"""Field identification, intersection mapping and clustering of the C-UAS sensors.

Starting from README/README_Sensor.pdf:

1. SPEC matrix   - which fields the PDF documents per sensor (parsed with pdftotext, exact).
2. DATA matrix   - which CSV columns each sensor id really has, and how full they are.
3. SEMANTIC layer - raw columns mapped to attributes that carry real information
                    (placeholders such as range=200/1, 0,0 sensor sites, 5000/500 errors, constant
                    per-detection scores are NOT counted; constant error columns are kept as *_nominal;
                    the raw `id` column is split into entity_id / id_label).
4. INTERSECTIONS - pairwise Jaccard + Formal Concept Analysis (all closed attribute sets) on
                    the semantic, data and spec matrices; Pareto front, per-k table, top concepts.
5. K-MEANS       - the sensors clustered on their semantic profile (K by silhouette).
6. Markdown report: executive summary first, then the full detail sections.

field_matrix_data.csv has, for every raw column c, two adjacent columns: `c` (fill rate = non-empty
rows / all rows of the sensor) and `c__present` (1 when c is in the header of at least one file of the
sensor, else 0), so 'absent' (present 0) and 'present but empty' (present 1, fill 0) differ.

Run:  python3 tools/field_intersection.py        (about 30 s; reads every sensor CSV once)
Outputs (data/derived/, gitignored): field_matrix_spec.csv field_matrix_data.csv
field_matrix_semantic.csv field_pairwise.csv field_concepts.csv sensor_families.csv
field_intersection.md
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuas_common as C  # noqa: E402
import kmeans as K  # noqa: E402

OUT = C.DERIVED
PDF = C.REPO / "README" / "README_Sensor.pdf"

# --------------------------------------------------------------------------- #
# 1. SPEC parse
# --------------------------------------------------------------------------- #

SPEC_TO_DATA = {"WCFX": "WVFX", "EMXGA": "EMXG"}   # the PDF spells two ids differently

# Manual reading of the PDF text (bullets counted by hand, see report). The parse must match.
SPEC_EXPECTED = {
    "IUKL": 22, "UXYJ": 29, "EGOI": 27, "NYTF": 25, "EUIG": 24,                    # radar
    "ILYS": 21, "MMJJ": 20, "ORFH": 17, "SMFS": 16, "IKJX": 20, "OEPS": 16, "GTYT": 17,  # RF-DF
    "WCFX": 23, "NJQL": 15, "FGIX": 19,                                           # EO/IR
    "EMXGA": 17,                                                                  # acoustic
    "CEQX": 32, "STUG": 29,                                                       # cyber
    "VKYY": 26, "QTVS": 26,                                                       # other (UTM)
}
CATEGORY_ORDER = ["radar", "rf_df", "eo_ir", "acoustic", "cyber", "other"]
_HEADING = [("radar data", "radar"), ("radiofrequency", "rf_df"), ("eo/ir", "eo_ir"),
            ("acoustic", "acoustic"), ("cyber", "cyber"), ("other", "other")]
_RE_H2 = re.compile(r"^\s*1\.1\.2\.\d+\s+(.+?)\s*$")
_RE_SENSOR = re.compile(r"^\s*(?:[ivx]+\.|[a-z]\))\s+([A-Z]{4,5}):?\s*$")
_RE_FIELD = re.compile(r"^\s*-\s+([A-Za-z][A-Za-z0-9_]*)(?:\s*\[[^\]]*\])?\s*:")


def parse_spec() -> dict[str, tuple[str, list[str]]]:
    """{spec sensor id: (category, [field, ...])} in document order."""
    text = subprocess.run(["pdftotext", "-layout", str(PDF), "-"], check=True,
                          capture_output=True, text=True).stdout
    spec: dict[str, tuple[str, list[str]]] = {}
    category = cur = None
    for line in text.splitlines():
        if re.match(r"^\s*1\.1\.3\s+Classification", line) and "...." not in line:
            break
        m = _RE_H2.match(line)
        if m and "...." not in line:
            low = m.group(1).lower()
            category = next(c for key, c in _HEADING if key in low)
            cur = None
            continue
        if category is None:
            continue
        m = _RE_SENSOR.match(line)
        if m:
            cur = m.group(1)
            if cur in spec:
                raise ValueError(f"sensor {cur} appears twice in the spec")
            spec[cur] = (category, [])
            continue
        m = _RE_FIELD.match(line)
        if m and cur is not None:
            fields = spec[cur][1]
            if m.group(1) in fields:
                raise ValueError(f"{cur}: duplicate field {m.group(1)}")
            fields.append(m.group(1))
    return spec


# --------------------------------------------------------------------------- #
# 3. Semantic attributes
# --------------------------------------------------------------------------- #

# Present on (almost) every row of every sensor: excluded when ranking intersections and from K-means.
ALWAYS = ["time", "track_id", "sensor_position", "node_id", "report_id"]
# K-means features: everything except the attributes that are constant 1 on every sensor
# (sensor_position stays in: it is 0 for QTVS and VKYY).
KMEANS_EXCLUDE = ["time", "track_id", "node_id", "report_id"]

CORE_ATTRS = [
    "time", "track_id", "node_id", "report_id", "sensor_position", "sensor_altitude",
    "object_position", "object_altitude", "position_error", "position_error_nominal",
    "altitude_error", "altitude_error_nominal",
    "range", "azimuth", "azimuth_coarse", "elevation", "azimuth_error", "azimuth_error_nominal",
    "elevation_error", "elevation_error_nominal",
    "velocity_horizontal", "velocity_vertical", "velocity_error", "velocity_error_nominal",
    "classification", "classification_subclass", "classification_confidence", "detection_confidence",
    "track_info", "object_info", "rf_signal", "state", "behaviour", "fusion_references",
    "task_id", "destination_id", "associated_file", "free_text", "colour", "entity_id", "id_label",
]
# identity / kinematic attributes found INSIDE nested cells (objectInfo, trackInfo, signal)
NESTED_ATTRS = [
    "heading", "speed", "climb_angle", "rcs", "covariance", "track_confidence", "radar_track_id",
    "model", "serial_number", "protocol", "frequency", "frequency_band", "signal_amplitude",
    "rf_technology", "rf_channel", "drone_mac", "pilot_mac", "ap_mac", "angular_rates", "range_rate",
    "camera_source", "track_following", "mitigation_flags", "drone_flags", "operator_id",
    "flight_description", "dji_uuid",
]
NESTED_MAP = {
    "az": "heading", "heading": "heading", "speed": "speed", "speed_ms": "speed",
    "el": "climb_angle", "rcs": "rcs", "confidence": "track_confidence", "trackId": "radar_track_id",
    "model": "model", "droneMaker": "model", "factory_name": "model",
    "serial number": "serial_number", "serial_number": "serial_number",
    "protocol": "protocol", "technology": "rf_technology", "channel": "rf_channel",
    "droneMac": "drone_mac", "pilotMac": "pilot_mac", "apMac": "ap_mac",
    "rae_azimuth_rate": "angular_rates", "rae_elevation_rate": "angular_rates",
    "rae_range_rate": "range_rate", "SourceCamera": "camera_source", "CameraID": "camera_source",
    "Following": "track_following", "fendable": "mitigation_flags", "mitigable": "mitigation_flags",
    "can_takeover": "mitigation_flags", "support_mitigation": "mitigation_flags",
    "Is remote ID": "drone_flags", "Is modified drone": "drone_flags",
    "Remote ID operator ID": "operator_id", "Remote ID flight description": "flight_description",
    "DJI UUID": "dji_uuid",
}
SIGNAL_MAP = {"centreFrequency": "frequency", "startFrequency": "frequency_band",
              "stopFrequency": "frequency_band", "amplitude": "signal_amplitude"}
NESTED_WRAPPERS = {"Drone Event Info"}     # CEQX: trackInfo wraps a JSON document in 'value'
MIN_NESTED = 0.01                          # an identity attribute counts at >= 1 % of the rows
BINARY_THRESHOLD = 0.5
MAX_TIES = 4
KEYS = ("n_attributes", "n_discriminative")   # all attributes / without the ones every sensor has

_TYPE = re.compile(r"'type': '([^']*)'")
_TYPE_VOID = re.compile(r"'type': '([^']*)', 'value': (?:''|'nan'|'None'|nan|None)[,}]")
_TYPE_CONF = re.compile(r"'type': '([^']*)', 'confidence': ([-0-9.eE+]+)")
_TRACK_CONF = re.compile(r"'type': 'confidence', 'value': '([-0-9.eE+]+)'")
_SIGKEY = re.compile(r"'([A-Za-z]+)':")
_CEQX_KEYS = re.compile(r'"(serial_number|factory_name|speed_ms|can_takeover|support_mitigation)":')
_UNKNOWN_CLASSES = {"", "unknown", "other"}
_EMPTYISH = {"", "[]", "{}"}
ID_RATIO = 0.5          # entity_id when distinct ids >= ID_RATIO x distinct objectIds (and > 1)

# Quantities checked for being constant over ALL rows of a sensor.
#   placeholder: a per-detection measured/estimated score; one distinct value = no information -> usability 0
#   nominal:     an error column; one distinct value = nominal spec value, kept usable as <attr>_nominal
CONST_QTY = {
    "range": ("range", "placeholder"), "up": ("velocity_vertical", "placeholder"),
    "dconf": ("detection_confidence", "placeholder"), "cconf": ("classification_confidence", "placeholder"),
    "tconf": ("track_confidence", "placeholder"),
    "perr": ("position_error", "nominal"), "zerr": ("altitude_error", "nominal"),
    "aerr": ("azimuth_error", "nominal"), "eerr": ("elevation_error", "nominal"),
    "verr": ("velocity_error", "nominal"),
}
PLACEHOLDER_ATTRS = [a for a, k in CONST_QTY.values() if k == "placeholder"]
NOMINAL_ATTRS = [a for a, k in CONST_QTY.values() if k == "nominal"]

# (attribute, raw source, rule, attributes whose per-sensor constants are listed in the last column)
PLACEHOLDER_RULES = [
    ("range", "rangeBearing_range", "constant over the whole sensor (FGIX 200, NJQL 1) = placeholder, not a range",
     ["range"]),
    ("sensor_position", "sensor_location_x/y", "(0, 0) = no sensor site (QTVS, VKYY UTM services)", []),
    ("sensor_altitude", "sensor_location_z", "z = 0 (many RF sensors) or no valid sensor position", []),
    ("object_altitude", "location_z", "z = 0 treated as 'unknown altitude'", []),
    ("position_error", "location_x/yError",
     "must be > 0 and not the STUG placeholder (5000, 5000, 500); constant over the sensor -> position_error_nominal, "
     "varying -> position_error", ["position_error"]),
    ("altitude_error", "location_zError",
     "must be > 0; STUG placeholder 500 and IKJX/MMJJ constant 2000 excluded; constant -> altitude_error_nominal, "
     "varying -> altitude_error", ["altitude_error"]),
    ("azimuth", "rangeBearing_azimuth", "fine bearing: azimuthError < 20 deg (or absent); az = 0 with error 0 excluded", []),
    ("azimuth_coarse", "rangeBearing_azimuth", "45-degree sectors, azimuthError >= 20 deg (OEPS, GTYT)", []),
    ("azimuth_error", "rangeBearing_azimuthError",
     "value must be > 0 (0.0 = no uncertainty reported); constant -> azimuth_error_nominal", ["azimuth_error"]),
    ("elevation_error", "rangeBearing_elevationError",
     "value must be > 0; constant -> elevation_error_nominal", ["elevation_error"]),
    ("velocity_error", "enuVelocity_*RateError",
     "any of the three > 0; constant -> velocity_error_nominal", ["velocity_error"]),
    ("velocity_horizontal", "enuVelocity_east/northRate",
     "east and north both exactly 0 treated as 'no velocity' (QTVS, VKYY)", []),
    ("velocity_vertical", "enuVelocity_upRate",
     "column constant over the whole sensor (UXYJ, QTVS, VKYY: always 0) = placeholder", ["velocity_vertical"]),
    ("classification", "classification", "at least one class other than Unknown / Other (UXYJ reports only Unknown 0.0)", []),
    ("classification_subclass", "classification", "a non-Unknown class below the top level (e.g. UAV Rotary Wing)", []),
    ("classification_confidence", "classification",
     "only on rows whose classification is usable (not Unknown/Other): the first confidence > 0 of a usable entry; "
     "a single distinct value over the sensor = placeholder (usability 0)", ["classification_confidence"]),
    ("detection_confidence", "detectionConfidence",
     "per-detection score; a single distinct value over the sensor = placeholder (usability 0)",
     ["detection_confidence"]),
    ("track_confidence", "trackInfo 'confidence'",
     "per-detection score (radar track quality); a single distinct value = placeholder (usability 0)",
     ["track_confidence"]),
    ("entity_id", "id",
     f"distinct id values >= {ID_RATIO} x distinct objectId values of the sensor (and more than one): per-object identifier", []),
    ("id_label", "id",
     f"distinct id values < {ID_RATIO} x distinct objectId values (2+ values): categorical text such as protocol, "
     "vendor or base64 labels; a single constant id value counts as neither", []),
    ("nested attributes", "objectInfo / trackInfo / signal", "a key counts when >= 1 % of the sensor's rows carry it", []),
]

# raw column -> semantic attribute, for the 'present in the data but not usable' diff
RAW_TO_SEM = [
    ("rangeBearing_range", "range"), ("sensor_location_z", "sensor_altitude"),
    ("sensor_location_x", "sensor_position"), ("location_z", "object_altitude"),
    ("location_xError", "position_error"), ("location_zError", "altitude_error"),
    ("rangeBearing_azimuth", "azimuth"), ("rangeBearing_azimuthError", "azimuth_error"),
    ("rangeBearing_elevationError", "elevation_error"), ("enuVelocity_eastRate", "velocity_horizontal"),
    ("enuVelocity_upRate", "velocity_vertical"), ("enuVelocity_eastRateError", "velocity_error"),
    ("classification", "classification"), ("classification", "classification_confidence"),
    ("detectionConfidence", "detection_confidence"),
]


def num(s: str):
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if v == v else None


def nested_attr(key: str) -> str | None:
    if key in NESTED_WRAPPERS:
        return None
    if key in NESTED_MAP:
        return NESTED_MAP[key]
    if key.startswith("covariance"):
        return "covariance"
    return "nested_" + re.sub(r"\W+", "_", key.lower()).strip("_")


# --------------------------------------------------------------------------- #
# 2 + 3. One pass over every sensor CSV: header union, fill rates, semantic presence
# --------------------------------------------------------------------------- #


def _fmt_val(v) -> str:
    return "(" + ", ".join(f"{x:g}" for x in v) + ")" if isinstance(v, tuple) else f"{v:g}"


def _track(s: dict, key: str, value) -> None:
    """Count one observation of a quantity and remember (at most) its first two distinct values."""
    s["vcnt"][key] += 1
    vals = s["vals"][key]
    if len(vals) < 2:
        vals.add(value)


def scan_sensors() -> dict[str, dict]:
    S: dict[str, dict] = {}
    for it in C.iterations():
        for sf in C.sensor_files(it):
            s = S.setdefault(sf.sensor_id, dict(
                type=sf.sensor_type, files=0, rows=0, ragged=0, hdr=[], ne=Counter(), attr=Counter(),
                vals=defaultdict(set), vcnt=Counter(), const={}, col_files=Counter(),
                ids=set(), oids=set(), id_rows=0, id_kind="none", unmapped=Counter()))
            if s["type"] != sf.sensor_type:
                raise ValueError(f"{sf.sensor_id} has two types: {s['type']} / {sf.sensor_type}")
            s["files"] += 1
            _scan_file(sf.path, s)
    for sid, s in S.items():
        # constant-value rule: one distinct value over all rows of the sensor
        for key, (a, kind) in CONST_QTY.items():
            n = s["vcnt"][key]
            if not n:
                continue
            if len(s["vals"][key]) > 1:
                s["attr"][a] = n
            elif kind == "placeholder":
                s["attr"][a] = 0
                s["const"][a] = (kind, _fmt_val(next(iter(s["vals"][key]))), n)
            else:
                s["attr"][a] = 0
                s["attr"][a + "_nominal"] = n
                s["const"][a] = (kind, _fmt_val(next(iter(s["vals"][key]))), n)
        # raw `id` column: per-object identifier vs categorical label vs constant
        n_id, n_oid = len(s["ids"]), len(s["oids"])
        if n_id == 1:
            s["id_kind"] = "constant"
        elif n_id >= 2 and n_id >= ID_RATIO * n_oid:
            s["id_kind"] = "entity_id"
            s["attr"]["entity_id"] = s["id_rows"]
        elif n_id >= 2:
            s["id_kind"] = "id_label"
            s["attr"]["id_label"] = s["id_rows"]
        for a in NESTED_ATTRS:                      # 1 % threshold for nested identity attributes
            if s["attr"][a] and s["attr"][a] / s["rows"] < MIN_NESTED:
                s["attr"][a] = 0
        for a in [a for a in s["attr"] if a.startswith("nested_")]:
            if s["attr"][a] / s["rows"] < MIN_NESTED:
                s["attr"][a] = 0
    return S


def _scan_file(path: Path, s: dict) -> None:
    header, rd = C.iter_rows(path)    # unwraps the quoted CEQX lines like every other tool
    header = ["row" if (i == 0 and not h) else h for i, h in enumerate(header)]
    for h in header:
        if h not in s["hdr"]:
            s["hdr"].append(h)
        s["col_files"][h] += 1
    width = len(header)
    idx = {h: i for i, h in enumerate(header)}
    g = lambda n: idx.get(n, -1)  # noqa: E731  (-1 hits the '' sentinel appended to each row)
    i_ts, i_oid, i_node, i_rep = g("timestamp"), g("objectId"), g("nodeId"), g("reportId")
    i_sx, i_sy, i_sz = g("sensor_location_x"), g("sensor_location_y"), g("sensor_location_z")
    i_lx, i_ly, i_lz = g("location_x"), g("location_y"), g("location_z")
    i_ex, i_ey, i_ez = g("location_xError"), g("location_yError"), g("location_zError")
    i_rng, i_az, i_el = g("rangeBearing_range"), g("rangeBearing_azimuth"), g("rangeBearing_elevation")
    i_aze, i_ele = g("rangeBearing_azimuthError"), g("rangeBearing_elevationError")
    i_ve, i_vn, i_vu = g("enuVelocity_eastRate"), g("enuVelocity_northRate"), g("enuVelocity_upRate")
    i_vee, i_vne, i_vue = g("enuVelocity_eastRateError"), g("enuVelocity_northRateError"), g("enuVelocity_upRateError")
    i_cls, i_conf = g("classification"), g("detectionConfidence")
    i_ti, i_oi, i_sig = g("trackInfo"), g("objectInfo"), g("signal")
    i_state, i_beh = g("state"), g("behaviour")
    i_ad, i_dd = g("associatedDetection"), g("derivedDetection")
    i_task, i_dest, i_file = g("taskId"), g("destinationId"), g("associatedFile")
    i_free, i_col, i_id = g("additionalInformation"), g("colour"), g("id")
    ne, attr = s["ne"], s["attr"]
    for row in rd:
        if row is None:
            s["ragged"] += 1
            continue
        s["rows"] += 1
        for h, v in zip(header, row):
            if v:
                ne[h] += 1
        row.append("")
        p: list[str] = []
        if row[i_ts]:
            p.append("time")
        if row[i_oid]:
            p.append("track_id")
        if row[i_node]:
            p.append("node_id")
        if row[i_rep]:
            p.append("report_id")
        sx, sy = num(row[i_sx]), num(row[i_sy])
        if sx is not None and sy is not None and not (sx == 0 and sy == 0):
            p.append("sensor_position")
            sz = num(row[i_sz])
            if sz is not None and sz != 0:
                p.append("sensor_altitude")
        if num(row[i_lx]) is not None and num(row[i_ly]) is not None:
            p.append("object_position")
        lz = num(row[i_lz])
        if lz is not None and lz != 0:
            p.append("object_altitude")
        ex, ey, ez = num(row[i_ex]), num(row[i_ey]), num(row[i_ez])
        placeholder = ex == 5000 and ey == 5000 and ez == 500
        if ex is not None and ey is not None and ex > 0 and ey > 0 and not placeholder:
            _track(s, "perr", (ex, ey))
        if ez is not None and ez > 0 and not placeholder and ez != 2000:
            _track(s, "zerr", ez)
        r = num(row[i_rng])
        if r is not None:
            _track(s, "range", r)
        az, aze = num(row[i_az]), num(row[i_aze])
        if az is not None:
            if aze is not None and aze >= 20:
                p.append("azimuth_coarse")
            elif not (az == 0 and aze == 0):
                p.append("azimuth")
        if aze is not None and aze > 0:
            _track(s, "aerr", aze)
        if num(row[i_el]) is not None:
            p.append("elevation")
        ele = num(row[i_ele])
        if ele is not None and ele > 0:
            _track(s, "eerr", ele)
        ve, vn = num(row[i_ve]), num(row[i_vn])
        if ve is not None and vn is not None and not (ve == 0 and vn == 0):
            p.append("velocity_horizontal")
        vu = num(row[i_vu])
        if vu is not None:
            _track(s, "up", vu)
        verr = tuple(v if (v := num(row[i])) is not None and v > 0 else 0.0 for i in (i_vee, i_vne, i_vue))
        if any(verr):
            _track(s, "verr", verr)
        cell = row[i_cls]
        if cell not in _EMPTYISH:
            cls_ok = any(t.lower() not in _UNKNOWN_CLASSES for t in _TYPE.findall(cell))
            if cls_ok:
                p.append("classification")
            pos = cell.find("'subClass'")
            if pos >= 0 and any(t.lower() not in _UNKNOWN_CLASSES for t in _TYPE.findall(cell[pos:])):
                p.append("classification_subclass")
            if cls_ok and "'confidence'" in cell:      # confidence only counts where the class is usable
                for ctype, cval in _TYPE_CONF.findall(cell):
                    cv = num(cval)
                    if ctype.lower() not in _UNKNOWN_CLASSES and cv is not None and cv > 0:
                        _track(s, "cconf", cv)
                        break
        dc = num(row[i_conf])
        if dc is not None:
            _track(s, "dconf", dc)
        for i, name in ((i_ti, "track_info"), (i_oi, "object_info"), (i_sig, "rf_signal"),
                        (i_state, "state"), (i_beh, "behaviour"), (i_task, "task_id"),
                        (i_dest, "destination_id"), (i_file, "associated_file"),
                        (i_free, "free_text"), (i_col, "colour")):
            if row[i] not in _EMPTYISH:
                p.append(name)
        if row[i_oid]:
            s["oids"].add(row[i_oid])
        if row[i_id] not in _EMPTYISH:
            s["id_rows"] += 1
            s["ids"].add(row[i_id])
        if row[i_ad] not in _EMPTYISH or row[i_dd] not in _EMPTYISH:
            p.append("fusion_references")
        nested: set[str] = set()
        for i in (i_ti, i_oi):
            c = row[i]
            if c:
                void = set(_TYPE_VOID.findall(c))
                for k in _TYPE.findall(c):
                    if k in void:                       # e.g. WVFX rae_range_rate = 'nan' on every row
                        continue
                    a = nested_attr(k)
                    if a == "track_confidence":      # per-detection score: constancy checked below
                        continue
                    if a:
                        nested.add(a)
                        if a.startswith("nested_"):
                            s["unmapped"][k] += 1
                for k in _CEQX_KEYS.findall(c):
                    nested.add(NESTED_MAP[k])
        c = row[i_ti]
        if c and "'confidence'" in c:
            m = _TRACK_CONF.search(c)
            if m and num(m.group(1)) is not None:
                _track(s, "tconf", num(m.group(1)))
        c = row[i_sig]
        if c:
            for k in set(_SIGKEY.findall(c)):
                if k in SIGNAL_MAP:
                    nested.add(SIGNAL_MAP[k])
        attr.update(p)
        attr.update(nested)


# --------------------------------------------------------------------------- #
# 4. Formal concept analysis
# --------------------------------------------------------------------------- #


def fca(sensors: list[str], attrs: list[str], has: dict[str, set[str]]) -> list[tuple[int, list[str]]]:
    """All closed attribute sets (intents) of the context, as (bitmask, extent). Closure under
    intersection of the sensors' attribute sets, plus the full attribute set."""
    bit = {a: 1 << i for i, a in enumerate(attrs)}
    masks = {s: sum(bit[a] for a in has[s]) for s in sensors}
    full = (1 << len(attrs)) - 1
    intents = {full}
    for s in sensors:
        g = masks[s]
        intents |= {c & g for c in intents} | {g}
    out = []
    for m in intents:
        out.append((m, [s for s in sensors if masks[s] & m == m]))
    return out


def concepts_table(level: str, sensors, attrs, has, rows, drop_cols: set[str]) -> tuple[list[dict], dict]:
    t0 = time.perf_counter()
    raw = fca(sensors, attrs, has)
    res = []
    for mask, ext in raw:
        names = [a for i, a in enumerate(attrs) if mask >> i & 1]
        n_disc = sum(1 for a in names if a not in drop_cols)
        res.append(dict(level=level, n_sensors=len(ext), n_attributes=len(names),
                        area=len(ext) * len(names), total_rows=sum(rows[s] for s in ext),
                        sensors="|".join(ext), attributes="|".join(names),
                        n_discriminative=n_disc, area_discriminative=len(ext) * n_disc,
                        _mask=mask))
    res.sort(key=lambda d: (-d["n_sensors"], -d["n_attributes"], d["attributes"]))
    checks = self_check(level, sensors, attrs, has, res)
    checks["seconds"] = time.perf_counter() - t0
    return res, checks


def self_check(level, sensors, attrs, has, res) -> dict:
    """Independent verification with plain sets (no bitmasks)."""
    intent_sets = {frozenset(d["attributes"].split("|")) if d["attributes"] else frozenset() for d in res}
    assert len(intent_sets) == len(res), f"{level}: duplicate intents"
    for d in res:                                  # extent exactly the sensors containing the intent
        intent = set(d["attributes"].split("|")) if d["attributes"] else set()
        ext = [s for s in sensors if intent <= has[s]]
        assert ext == (d["sensors"].split("|") if d["sensors"] else []), f"{level}: extent mismatch {intent}"
        if ext:                                    # closed: intent is exactly what its extent shares
            assert set.intersection(*(set(has[s]) for s in ext)) == intent, f"{level}: not closed {intent}"
        else:
            assert intent == set(attrs), f"{level}: empty extent but not the full set"
    for s in sensors:                              # every sensor's own set is an intent
        assert frozenset(has[s]) in intent_sets, f"{level}: object intent missing {s}"
    masks = [d["_mask"] for d in res]              # closure: intersection of any two intents is an intent
    mset = set(masks)
    for i, a in enumerate(masks):
        for b in masks[i + 1:]:
            assert (a & b) in mset, f"{level}: closure incomplete"
    return dict(n_concepts=len(res), n_pairs=len(masks) * (len(masks) - 1) // 2)


def pareto(res: list[dict], key: str = "n_attributes") -> list[tuple[int, int, list[dict]]]:
    """Non-dominated (n_sensors, `key`) points over concepts with >= 1 sensor."""
    live = [d for d in res if d["n_sensors"] >= 1]
    max_n = max(d["n_sensors"] for d in live)
    best_ge = [0] * (max_n + 2)
    for n in range(max_n, 0, -1):
        best_ge[n] = max([best_ge[n + 1]] + [d[key] for d in live if d["n_sensors"] == n])
    pts: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for d in live:
        n, a = d["n_sensors"], d[key]
        if a == best_ge[n] and best_ge[n + 1] < a:
            pts[(n, a)].append(d)
    return [(n, a, pts[(n, a)]) for (n, a) in sorted(pts, reverse=True)]


def per_k(res: list[dict], n_max: int, key: str = "n_attributes") -> list[tuple[int, int, list[dict]]]:
    out = []
    live = [d for d in res if d["n_sensors"] >= 1]
    for k in range(2, n_max + 1):
        cand = [d for d in live if d["n_sensors"] >= k]
        if not cand:
            continue
        best = max(d[key] for d in cand)
        out.append((k, best, [d for d in cand if d[key] == best]))
    return out


def top_concepts(res, key, n=10, min_sensors=3, min_disc=4):
    cand = [d for d in res if d["n_sensors"] >= min_sensors and d["n_discriminative"] >= min_disc]
    if key == "area":
        cand.sort(key=lambda d: (-d["area_discriminative"], -d["n_sensors"], -d["total_rows"], d["attributes"]))
    else:
        cand.sort(key=lambda d: (-d["total_rows"], -d["area_discriminative"], d["attributes"]))
    return cand[:n]


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def mdtable(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def fmt_attrs(names: str, drop: set[str] | None = None) -> str:
    items = [a for a in names.split("|") if a and not (drop and a in drop)]
    return ", ".join(items) if items else "(none)"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    t_start = time.perf_counter()

    # ---------------- 1. SPEC ---------------- #
    spec = parse_spec()
    print("== 1. SPEC parse (README_Sensor.pdf): documented fields per sensor ==")
    bad = []
    for sid, (cat, fields) in spec.items():
        exp = SPEC_EXPECTED.get(sid)
        flag = "ok" if exp == len(fields) else f"MISMATCH (manual reading {exp})"
        if exp != len(fields):
            bad.append(sid)
        print(f"  {sid:6s} {cat:9s} {len(fields):3d} fields  {flag}")
    assert not bad and set(spec) == set(SPEC_EXPECTED), f"spec parse differs from manual reading: {bad}"
    print(f"  {len(spec)} sensors, {sum(len(f) for _, f in spec.values())} field entries; parse == manual reading")
    spec_data: dict[str, tuple[str, set[str]]] = {
        SPEC_TO_DATA.get(sid, sid): (cat, set(f)) for sid, (cat, f) in spec.items()}
    spec_fields = sorted({f for _, fs in spec_data.values() for f in fs})
    write_csv(OUT / "field_matrix_spec.csv", ["sensor_id", "category"] + spec_fields,
              [[sid, cat] + [int(f in fs) for f in spec_fields] for sid, (cat, fs) in sorted(spec_data.items())])

    # ---------------- 2. DATA + 3. SEMANTIC (one pass) ---------------- #
    print("== 2/3. scanning every sensor CSV ==")
    t0 = time.perf_counter()
    S = scan_sensors()
    print(f"  scanned {sum(s['files'] for s in S.values())} files, {sum(s['rows'] for s in S.values())} rows "
          f"({sum(s['ragged'] for s in S.values())} ragged rows skipped) in {time.perf_counter() - t0:.1f} s")
    sensors = sorted(S)
    rows_of = {sid: S[sid]["rows"] for sid in sensors}
    category_of = {sid: spec_data[sid][0] if sid in spec_data else "undocumented" for sid in sensors}
    type_of = {sid: S[sid]["type"] for sid in sensors}
    raw_cols = sorted({c for s in S.values() for c in s["hdr"]} - {"row"})
    fill = {sid: {c: round(S[sid]["ne"][c] / S[sid]["rows"], 4) if c in S[sid]["hdr"] else 0.0
                  for c in raw_cols} for sid in sensors}
    present = {sid: {c: int(c in S[sid]["hdr"]) for c in raw_cols} for sid in sensors}
    write_csv(OUT / "field_matrix_data.csv",
              ["sensor_id", "sensor_type", "n_files", "n_rows"] + [h for c in raw_cols for h in (c, c + "__present")],
              [[sid, type_of[sid], S[sid]["files"], S[sid]["rows"]] +
               [x for c in raw_cols for x in (fill[sid][c], present[sid][c])] for sid in sensors])
    # columns that are in the header of only some of a sensor's files
    partial_cols = {sid: sorted(c for c in raw_cols if 0 < S[sid]["col_files"][c] < S[sid]["files"])
                    for sid in sensors}

    # spec vs data
    diff = {}
    for sid in sensors:
        hdr = set(S[sid]["hdr"]) - {"row"}
        empty = sorted(c for c in hdr if S[sid]["ne"][c] == 0)
        if sid in spec_data:
            doc = spec_data[sid][1]
            diff[sid] = dict(absent=sorted(doc - hdr), undocumented=sorted(hdr - doc), empty=empty)
        else:
            diff[sid] = dict(absent=None, undocumented=sorted(hdr), empty=empty)

    # semantic
    all_attrs = CORE_ATTRS + NESTED_ATTRS
    extra = sorted({a for s in S.values() for a, n in s["attr"].items() if n and a not in all_attrs})
    all_attrs = all_attrs + extra
    frac = {sid: {a: round(S[sid]["attr"][a] / S[sid]["rows"], 4) for a in all_attrs} for sid in sensors}
    sem_attrs = [a for a in all_attrs if any(frac[sid][a] > 0 for sid in sensors)]
    sem_bin = {sid: {a for a in sem_attrs if frac[sid][a] >= BINARY_THRESHOLD} for sid in sensors}
    write_csv(OUT / "field_matrix_semantic.csv", ["sensor_id", "view"] + sem_attrs,
              [[sid, "fraction"] + [frac[sid][a] for a in sem_attrs] for sid in sensors] +
              [[sid, "binary"] + [int(a in sem_bin[sid]) for a in sem_attrs] for sid in sensors])
    dropped = [a for a in all_attrs if a not in sem_attrs]
    if dropped:
        print("  semantic attributes with no sensor carrying them (dropped):", ", ".join(dropped))

    data_bin = {sid: {c for c in raw_cols if fill[sid][c] > 0} for sid in sensors}
    spec_sensors = sorted(spec_data)
    spec_bin = {sid: set(spec_data[sid][1]) for sid in spec_sensors}

    # ---------------- 4. INTERSECTIONS ---------------- #
    print("== 4. intersections ==")
    levels = {
        "semantic": (sensors, sem_attrs, sem_bin, set(ALWAYS)),
        "data": (sensors, raw_cols, data_bin, set.intersection(*data_bin.values())),
        "spec": (spec_sensors, spec_fields, spec_bin, set.intersection(*spec_bin.values())),
    }
    pair_rows = []
    for lv, (sens, attrs, has, _) in levels.items():
        for i, a in enumerate(sens):
            for b in sens[i + 1:]:
                inter, uni = has[a] & has[b], has[a] | has[b]
                pair_rows.append([lv, a, b, len(inter), len(uni),
                                  round(len(inter) / len(uni), 4) if uni else 0.0,
                                  "|".join(x for x in attrs if x in inter)])
    write_csv(OUT / "field_pairwise.csv",
              ["level", "sensor_a", "sensor_b", "intersection", "union", "jaccard", "shared"], pair_rows)

    concepts: dict[str, list[dict]] = {}
    checks = {}
    for lv, (sens, attrs, has, drop) in levels.items():
        concepts[lv], checks[lv] = concepts_table(lv, sens, attrs, has, rows_of, drop)
        c = checks[lv]
        print(f"  {lv:8s} {len(sens)} sensors x {len(attrs)} attributes -> {c['n_concepts']} concepts "
              f"({c['seconds']:.1f} s)")
    cfields = ["level", "n_sensors", "n_attributes", "area", "total_rows", "sensors", "attributes",
               "n_discriminative", "area_discriminative"]
    write_csv(OUT / "field_concepts.csv", cfields,
              [[d[f] for f in cfields] for lv in levels for d in concepts[lv]])

    fronts = {(lv, key): pareto(concepts[lv], key) for lv in levels for key in KEYS}
    perk = {(lv, key): per_k(concepts[lv], len(levels[lv][0]), key) for lv in levels for key in KEYS}
    for lv in levels:
        for key in KEYS:
            print(f"-- Pareto front ({lv}, {key}): n_sensors x attributes")
            for n, a, ds in fronts[(lv, key)]:
                ex = ds[0]["sensors"] if len(ds) == 1 else f"{len(ds)} concepts, e.g. {ds[0]['sensors']}"
                print(f"   {n:2d} sensors x {a:2d} attributes   {ex}")

    # ---------------- 5. K-MEANS ---------------- #
    print(f"== 5. K-means on the semantic profiles (excluded features: {', '.join(KMEANS_EXCLUDE)}) ==")
    feat_attrs = [a for a in sem_attrs if a not in KMEANS_EXCLUDE]
    X = np.array([[frac[sid][a] for a in feat_attrs] for sid in sensors], dtype=float)
    scan = K.scan_k(X, range(2, 9), n_init=20, seed=0)
    best = max(scan, key=lambda t: (round(t[2], 12), -t[0]))
    k_best, res_best = best[0], best[3]
    res6 = next(t[3] for t in scan if t[0] == 6)
    sil6 = next(t[2] for t in scan if t[0] == 6)
    for k, inertia, sil, _ in scan:
        print(f"   K={k}  inertia={inertia:8.4f}  silhouette={sil:7.4f}{'   <- best' if k == k_best else ''}"
              f"{'   (PDF categories)' if k == 6 else ''}")
    print(f"   chosen K={k_best} (silhouette {best[2]:.4f}); K=6 silhouette {sil6:.4f}")
    dist_best = np.sqrt(((X - res_best.centers[res_best.labels]) ** 2).sum(1))
    dist6 = np.sqrt(((X - res6.centers[res6.labels]) ** 2).sum(1))
    fam_best = {sid: int(res_best.labels[i]) + 1 for i, sid in enumerate(sensors)}
    fam6 = {sid: int(res6.labels[i]) + 1 for i, sid in enumerate(sensors)}
    write_csv(OUT / "sensor_families.csv",
              ["sensor_id", "sensor_type", "pdf_category", "family_k_best", "family_k6",
               "distance_to_center", "distance_to_center_k6"],
              [[sid, type_of[sid], category_of[sid], fam_best[sid], fam6[sid],
                round(float(dist_best[i]), 4), round(float(dist6[i]), 4)] for i, sid in enumerate(sensors)])

    # ---------------- 6. report ---------------- #
    md = build_md(locals())
    (OUT / "field_intersection.md").write_text(md, encoding="utf-8")

    # reload the CSVs and cross-check the in-memory tables the report was built from
    with (OUT / "field_concepts.csv").open(newline="", encoding="utf-8") as fh:
        rd = list(csv.DictReader(fh))
    assert len(rd) == sum(len(concepts[lv]) for lv in levels)
    with (OUT / "field_matrix_semantic.csv").open(newline="", encoding="utf-8") as fh:
        rd = list(csv.DictReader(fh))
    for r in rd:
        if r["view"] == "binary":
            assert {a for a in sem_attrs if r[a] == "1"} == sem_bin[r["sensor_id"]]
    with (OUT / "sensor_families.csv").open(newline="", encoding="utf-8") as fh:
        rd = list(csv.DictReader(fh))
    assert {r["sensor_id"]: int(r["family_k_best"]) for r in rd} == fam_best
    with (OUT / "field_matrix_data.csv").open(newline="", encoding="utf-8") as fh:
        rd = list(csv.DictReader(fh))
    assert [r["sensor_id"] for r in rd] == sensors
    for r in rd:
        for c in raw_cols:
            assert float(r[c]) == fill[r["sensor_id"]][c] and int(r[c + "__present"]) == present[r["sensor_id"]][c]

    print("== self-checks ==")
    for lv in levels:
        c = checks[lv]
        print(f"   {lv:8s} PASS extent==sensors containing intent, intent==shared attributes of extent, "
              f"closure complete ({c['n_concepts']} concepts, {c['n_pairs']} intent pairs)")
    print("   PASS spec parse == manual reading; CSV reload matches in-memory tables")
    verify_ids_and_flags(S, sensors, raw_cols, fill, present, frac)
    print("   PASS independent recount of id / objectId / detectionConfidence distinct values == scan; "
          "present flags consistent with fill rates; entity_id/id_label and *_nominal/per-detection exclusive")
    print(f"== done: wrote 7 files to {OUT} ==")
    print(f"runtime: {time.perf_counter() - t_start:.1f} s")
    return 0


def verify_ids_and_flags(S, sensors, raw_cols, fill, present, frac) -> None:
    """Independent re-read of id / objectId / detectionConfidence (plain csv.reader, no shared helpers
    with the scan) plus consistency of the flags the report relies on."""
    seen = {sid: dict(id=set(), oid=set(), dc=set()) for sid in sensors}
    for it in C.iterations():
        for sf in C.sensor_files(it):
            with sf.path.open(newline="", encoding="utf-8") as fh:
                rd = csv.reader(fh)
                header = next(rd)
                ix = {h: i for i, h in enumerate(header)}
                for row in rd:
                    if len(row) == 1 and len(header) > 1:   # whole line wrapped in quotes (one CEQX file)
                        row = next(csv.reader([row[0]]), row)
                    if len(row) != len(header):
                        continue
                    for key, col in (("id", "id"), ("oid", "objectId")):
                        if col in ix and row[ix[col]] not in _EMPTYISH:
                            seen[sf.sensor_id][key].add(row[ix[col]])
                    if "detectionConfidence" in ix and row[ix["detectionConfidence"]]:
                        seen[sf.sensor_id]["dc"].add(float(row[ix["detectionConfidence"]]))
    for sid in sensors:
        s = S[sid]
        assert seen[sid]["id"] == s["ids"] and seen[sid]["oid"] == s["oids"], f"{sid}: id recount differs"
        assert min(len(seen[sid]["dc"]), 2) == len(s["vals"]["dconf"]), f"{sid}: detectionConfidence recount differs"
        assert (len(seen[sid]["dc"]) == 1) == ("detection_confidence" in s["const"]), f"{sid}: dc constancy"
        assert not (frac[sid]["entity_id"] and frac[sid]["id_label"]), f"{sid}: entity_id and id_label both set"
        for a in NOMINAL_ATTRS:
            assert not (frac[sid][a] and frac[sid][a + "_nominal"]), f"{sid}: {a} both nominal and varying"
        for a in PLACEHOLDER_ATTRS:
            if a in s["const"]:
                assert frac[sid][a] == 0, f"{sid}: constant placeholder {a} still usable"
        for c in raw_cols:
            assert present[sid][c] >= (fill[sid][c] > 0), f"{sid}.{c}: filled but not present"


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #


def sem_cell(frac_s: dict, a: str) -> str:
    """Matrix symbol: ✓ usable on >= 50 % of rows, n = nominal (constant) error on >= 50 %, ~ partial."""
    f, fn = frac_s.get(a, 0.0), frac_s.get(a + "_nominal", 0.0)
    if f >= BINARY_THRESHOLD:
        return "✓"
    if fn >= BINARY_THRESHOLD:
        return "n"
    if f > 0:
        return "~"
    return "~n" if fn > 0 else ""


def and_join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def drop_text(lv: str, drop: set[str], sensors: list[str], sem_bin: dict) -> str:
    """The exclusion line for the discriminative counts of one level."""
    names = ", ".join(sorted(drop)) if drop else "none"
    if lv != "semantic":
        return f"excluded: present on every sensor of this level: {names}"
    absent = [f"{a} is absent for {and_join([s for s in sensors if a not in sem_bin[s]])}"
              for a in sorted(drop) if any(a not in sem_bin[s] for s in sensors)]
    return (f"excluded by rule (present on all or nearly all sensors): {names}"
            + (f"; {'; '.join(absent)}" if absent else ""))


def const_text(attrs: list[str], sensors: list[str], S: dict) -> str:
    """Per-sensor constant handling for the rule table's last column."""
    parts = []
    for a in attrs:
        kind = next(k for aa, k in CONST_QTY.values() if aa == a)
        groups: dict[str, list[str]] = defaultdict(list)
        varying = []
        for sid in sensors:
            if a in S[sid]["const"]:
                groups[S[sid]["const"][a][1]].append(sid)
            elif S[sid]["vcnt"][next(k for k, (aa, _) in CONST_QTY.items() if aa == a)]:
                varying.append(sid)
        for val in sorted(groups):
            label = f"constant placeholder (value {val})" if kind == "placeholder" else f"nominal (constant {val})"
            parts.append(f"{label}: {', '.join(groups[val])}")
        if varying:
            parts.append("varying (per-detection): " + ", ".join(varying))
    return "; ".join(parts) or "-"


def build_md(v: dict) -> str:
    sensors, frac, sem_bin, sem_attrs = v["sensors"], v["frac"], v["sem_bin"], v["sem_attrs"]
    type_of, category_of, rows_of, S = v["type_of"], v["category_of"], v["rows_of"], v["S"]
    fam_best, fam6, k_best, scan, res_best = v["fam_best"], v["fam6"], v["k_best"], v["scan"], v["res_best"]
    feat_attrs, X = v["feat_attrs"], v["X"]
    concepts, fronts, perk, levels = v["concepts"], v["fronts"], v["perk"], v["levels"]
    diff, spec_data, spec = v["diff"], v["spec_data"], v["spec"]
    fill, raw_cols, checks = v["fill"], v["raw_cols"], v["checks"]
    present, partial_cols = v["present"], v["partial_cols"]
    res6 = v["res6"]
    n = len(sensors)
    out: list[str] = []
    W = out.append

    members: dict[int, list[str]] = defaultdict(list)
    for sid in sensors:
        members[fam_best[sid]].append(sid)
    order = sorted(sensors, key=lambda s: (fam_best[s], CATEGORY_ORDER.index(category_of[s])
                                           if category_of[s] in CATEGORY_ORDER else 99, s))
    m6: dict[int, list[str]] = defaultdict(list)
    for sid in sensors:
        m6[fam6[sid]].append(sid)
    cats = CATEGORY_ORDER + ["undocumented"]
    sem_sens, sem_attrs_l, _, sem_drop = levels["semantic"]
    tot_rows = sum(rows_of.values())

    W("# Field intersection and sensor clustering\n")
    W("Generated by `tools/field_intersection.py` from `README/README_Sensor.pdf` and every sensor CSV "
      f"(testing + evaluation trees; byte-identical `(1)` duplicates skipped): {n} sensor ids in the data, "
      f"{len(spec)} documented in the PDF, {sum(S[s]['files'] for s in sensors)} files, {tot_rows} detection rows "
      f"({sum(S[s]['ragged'] for s in sensors)} ragged rows skipped). Tables come from the CSVs written "
      "next to this file (`field_matrix_*.csv`, `field_pairwise.csv`, `field_concepts.csv`, `sensor_families.csv`).\n")

    # =================================================================== #
    # EXECUTIVE SUMMARY
    # =================================================================== #
    sm_start = len(out)
    W("## Executive summary\n")
    W("An attribute counts for a sensor only when it carries information (usability rules in the detail section "
      "(i)): placeholders, constant per-detection scores, unknown-only classes and the like are not counted.\n")
    W("### 1. Sensor x attribute matrix, grouped by K-means family\n")
    cols = [("spos", "sensor_position"), ("pos", "object_position"), ("oalt", "object_altitude"),
            ("salt", "sensor_altitude"), ("rng", "range"), ("az", "azimuth"), ("azc", "azimuth_coarse"),
            ("el", "elevation"), ("perr", "position_error"), ("zerr", "altitude_error"), ("aerr", "azimuth_error"),
            ("vh", "velocity_horizontal"), ("vv", "velocity_vertical"), ("cls", "classification"),
            ("sub", "classification_subclass"), ("ccf", "classification_confidence"),
            ("dcf", "detection_confidence"), ("tcf", "track_confidence"), ("ent", "entity_id"), ("lbl", "id_label")]
    shown = {a for _, a in cols} | {a + "_nominal" for _, a in cols} | {"time", "track_id", "node_id", "report_id"}
    W("`✓` usable on >= 50 % of the sensor's rows, `~` on 1..49 %, `n` = error column kept as a nominal constant "
      "(`*_nominal`, not per-detection), blank = absent or unusable. Columns: " +
      ", ".join(f"`{h}` {a}" for h, a in cols) + ". `other` = remaining usable attributes (>= 50 %).\n")
    mrows = []
    for s in order:
        other = [a for a in sem_attrs if a in sem_bin[s] and a not in shown]
        mrows.append([f"F{fam_best[s]}", s, type_of[s]] + [sem_cell(frac[s], a) for _, a in cols] +
                     [", ".join(other)])
    W(mdtable(["fam", "sensor", "type"] + [h for h, _ in cols] + ["other"], mrows))
    W("")
    W("Families (K = %d, silhouette %.4f); attributes shared by every member (besides the always-present ones):\n"
      % (k_best, max(t[2] for t in scan)))
    for f in sorted(members):
        shared = sorted(set.intersection(*(sem_bin[s] for s in members[f])) - set(ALWAYS))
        W(f"- **F{f}** ({len(members[f])}): {', '.join(members[f])} - " +
          (", ".join(shared[:9]) + (f" (+{len(shared) - 9} more)" if len(shared) > 9 else "") if shared else "none"))
    W("")

    W("### 2. Top identifiable intersection sets (semantic level, 22 sensors)\n")
    W(f"Discriminative counts leave out the attributes {drop_text('semantic', sem_drop, sem_sens, sem_bin)}. "
      f"{checks['semantic']['n_concepts']} concepts enumerated and verified. "
      "Pareto front (no set has both more sensors and more attributes):\n")
    prow = []
    for nn, a, ds in fronts[("semantic", "n_discriminative")]:
        for d in ds:
            prow.append([nn, a, d["total_rows"], d["sensors"].replace("|", ", "), fmt_attrs(d["attributes"], sem_drop)])
    W(mdtable(["sensors", "attributes", "total rows", "which sensors", "attribute set"], prow))
    W("")
    W("Top 5 by area (sensors x discriminative attributes, >= 3 sensors, >= 4 attributes):\n")
    W(mdtable(["#", "sensors", "attrs", "area", "total rows", "which sensors", "attribute set"],
              [[i, d["n_sensors"], d["n_discriminative"], d["area_discriminative"], d["total_rows"],
                d["sensors"].replace("|", ", "), fmt_attrs(d["attributes"], sem_drop)]
               for i, d in enumerate(top_concepts(concepts["semantic"], "area", 5), 1)]))
    W("")

    W("### 3. Families (K-means on the semantic fractions)\n")
    W(f"Features: {len(feat_attrs)} semantic fractions per sensor; excluded: {', '.join(KMEANS_EXCLUDE)} "
      "(sensor_position stays in). Deterministic k-means++ (seed 0, 20 restarts), K by silhouette.\n")
    W(mdtable(["K", "inertia", "silhouette", ""],
              [[k, f"{ine:.4f}", f"{sil:.4f}",
                ("best" if k == k_best else "") + (" PDF's 6 categories" if k == 6 else "")]
               for k, ine, sil, _ in scan]))
    W("")
    W(mdtable(["K = 6 group", "sensors", "PDF categories inside"],
              [[f"G{f}", ", ".join(m6[f]), ", ".join(f"{c} {sum(1 for s in m6[f] if category_of[s] == c)}"
                                                       for c in cats if any(category_of[s] == c for s in m6[f]))]
               for f in sorted(m6)]))
    W("")

    W("### 4. Spec versus data\n")
    nab = sum(len(d["absent"]) for d in diff.values() if d["absent"] is not None)
    absent_txt = "; ".join(f"{sid} {', '.join(d['absent'])}" for sid, d in diff.items() if d["absent"])
    undoc: dict[str, list[str]] = defaultdict(list)
    for sid, d in diff.items():
        if d["absent"] is not None:
            for c in d["undocumented"]:
                undoc[c].append(sid)
    n_undoc = sum(len(x) for x in undoc.values())
    unusable = sorted({(sid, a) for sid in sensors for col, a in RAW_TO_SEM
                       if col in S[sid]["hdr"] and fill[sid][col] > 0 and a in frac[sid]
                       and frac[sid][a] + frac[sid].get(a + "_nominal", 0.0) < BINARY_THRESHOLD})
    W(f"- Ids: the PDF's WCFX is WVFX in the data and EMXGA is EMXG; LJJX (FUSION-NODE) and YXAB (PASSIVE-RF) "
      f"are in the data but not in the PDF ({len(spec)} documented, {n} present).")
    W(f"- {nab} documented fields are absent from the CSVs: {absent_txt}. {n_undoc} columns are in the CSVs but "
      f"undocumented: " + "; ".join(f"{c} ({', '.join(x)})" for c, x in sorted(undoc.items())) + ". "
      f"{sum(len(d['empty']) for d in diff.values())} columns are present but empty on every row.")
    W(f"- Present-in-the-header is not the same as usable: {len(unusable)} (sensor, attribute) pairs have a raw "
      "column with data but the attribute is usable on < 50 % of the rows (constant placeholders, z = 0, "
      "Unknown-only classes, (0, 0) sensor sites; list in section (ii)).")
    const_pl = sorted((sid, a) for sid in sensors for a in PLACEHOLDER_ATTRS if a in S[sid]["const"])
    W(f"- Constant placeholders found: " + "; ".join(
        f"{a}: " + ", ".join(f"{sid} ({S[sid]['const'][a][1]})" for sid, aa in const_pl if aa == a)
        for a in PLACEHOLDER_ATTRS if any(aa == a for _, aa in const_pl)) + ".")
    nom = sorted((sid, a) for sid in sensors for a in NOMINAL_ATTRS if a in S[sid]["const"])
    W("- Constant (nominal) error columns kept usable as `*_nominal`: " + "; ".join(
        f"{a}: " + ", ".join(f"{sid} ({S[sid]['const'][a][1]})" for sid, aa in nom if aa == a)
        for a in NOMINAL_ATTRS if any(aa == a for _, aa in nom)) + ".")
    ent = [s for s in sensors if S[s]["id_kind"] == "entity_id"]
    lab = [s for s in sensors if S[s]["id_kind"] == "id_label"]
    cst = [s for s in sensors if S[s]["id_kind"] == "constant"]
    W(f"- Raw `id`: per-object `entity_id` on {', '.join(ent)}; categorical `id_label` on {', '.join(lab)}; "
      f"a single constant value on {', '.join(cst) or 'none'} (counts in section (vi)).")
    W("")

    W("### 5. Implications for the unified detection table\n")
    impl = implications(v, members)
    for line in impl["short"]:
        W(line)
    W("")
    summary_lines = len(out) - sm_start
    assert summary_lines <= 120, f"executive summary has {summary_lines} lines (> 120)"
    v["summary_lines"] = summary_lines

    # =================================================================== #
    # DETAIL SECTIONS
    # =================================================================== #
    W("---\n")
    W("# Detail sections\n")

    # ---- (i) semantic table ----
    W("## (i) Which sensor carries which attribute\n")
    W("Attributes are semantic, not raw column names: a field counts only when it carries information "
      "(see the usability rules). `✓` = present and usable on >= 50 % of the sensor's rows; "
      "`~` = on 1..49 % of rows. Sensors are grouped by K-means family (section iv). "
      "`*_nominal` attributes are error columns that are constant over the whole sensor (nominal spec value, "
      "not a per-detection estimate); the unsuffixed error attributes are per-detection.\n")
    W(f"Sensors per family (K = {k_best}):\n")
    for f in sorted(members):
        W(f"- **F{f}**: " + ", ".join(f"{s} ({type_of[s]})" for s in members[f]))
    W("")
    W("Usability rules applied when mapping raw columns to attributes (last column: what the data shows for "
      "constant values, computed over all rows of each sensor):\n")
    W(mdtable(["attribute", "raw source", "rule", "constant handling observed"],
              [[a, r, t, const_text(ca, sensors, S)] for a, r, t, ca in PLACEHOLDER_RULES]))
    W("")
    rows = []
    for a in sem_attrs:
        cells = []
        for s in order:
            f = frac[s][a]
            cells.append("✓" if f >= BINARY_THRESHOLD else ("~" if f > 0 else ""))
        cnt = sum(1 for s in sensors if a in sem_bin[s])
        rows.append([a] + cells + [cnt])
    W(mdtable(["attribute"] + [f"{s}<br>F{fam_best[s]}" for s in order] + ["n"], rows))
    W("")
    W(mdtable(["sensor", "type", "PDF category", "family", "rows", "usable attributes (>=0.5)",
               "of which discriminative"],
              [[s, type_of[s], category_of[s], f"F{fam_best[s]}", rows_of[s], len(sem_bin[s]),
                len(sem_bin[s] - set(ALWAYS))] for s in order]))
    W("")

    # ---- (ii) discrepancies ----
    W("## (ii) Spec versus data\n")
    W(f"Ids: the PDF's WCFX is WVFX in the data and EMXGA is EMXG. LJJX (FUSION-NODE) and YXAB (PASSIVE-RF) "
      f"are in the data but not in the PDF. The first CSV column is an unnamed index and is ignored.\n")
    W("`field_matrix_data.csv` carries, for every raw column `c`, the fill rate `c` (non-empty rows / all rows of "
      "the sensor) and next to it `c__present` (1 = `c` is in the header of at least one file of the sensor). "
      "So `present = 0` means absent from the sensor's files, `present = 1` with fill 0 means present but empty "
      "on every row.\n")
    anyp = {sid: cs for sid, cs in partial_cols.items() if cs}
    W("Columns in the header of only some of a sensor's files: " +
      ("; ".join(f"{sid}: {', '.join(cs)}" for sid, cs in anyp.items()) if anyp else "none") + ".\n")
    W("Raw column differences per sensor (documented but absent from the CSV / in the CSV but undocumented / "
      "present in the header but empty on every row):\n")
    drows = []
    for sid in sensors:
        d = diff[sid]
        if d["absent"] is None:
            drows.append([sid, "(not in PDF)", "all " + str(len(d["undocumented"])) + ": " +
                          ", ".join(d["undocumented"]), ", ".join(d["empty"]) or "none"])
        else:
            drows.append([sid, ", ".join(d["absent"]) or "none", ", ".join(d["undocumented"]) or "none",
                          ", ".join(d["empty"]) or "none"])
    W(mdtable(["sensor", "documented, absent in data", "in data, undocumented", "present, always empty"], drows))
    W("")
    W(f"Totals: {nab} documented-but-absent field entries over {len(spec)} sensors; "
      f"{sum(len(d['empty']) for d in diff.values())} present-but-always-empty columns.\n")
    W("Columns present in the data whose semantic attribute is nevertheless unusable on most rows "
      "(placeholders - fill rate of the raw column vs usable fraction of the attribute, nominal constants "
      "counted as usable):\n")
    urows = []
    for sid in sensors:
        for col, a in RAW_TO_SEM:
            if col in S[sid]["hdr"] and fill[sid][col] > 0 and a in frac[sid]:
                usable = round(frac[sid][a] + frac[sid].get(a + "_nominal", 0.0), 4)
                if usable >= BINARY_THRESHOLD:
                    continue
                if a == "azimuth" and frac[sid]["azimuth_coarse"] >= BINARY_THRESHOLD:
                    a_txt = f"azimuth (only coarse: azimuth_coarse={frac[sid]['azimuth_coarse']})"
                else:
                    a_txt = a
                key = next((k for k, (aa, _) in CONST_QTY.items() if aa == a), None)
                if a in S[sid]["const"] and S[sid]["const"][a][0] == "placeholder":
                    why = f"constant placeholder (value {S[sid]['const'][a][1]})"
                elif key in ("cconf", "dconf", "tconf") and not S[sid]["vcnt"][key]:
                    why = ("no confidence in the usable classification entries" if key == "cconf"
                           else "no value reported")
                else:
                    why = ""
                urows.append([sid, col, fill[sid][col], a_txt, usable, why])
    W(mdtable(["sensor", "raw column", "raw fill", "attribute", "usable fraction", "reason"], urows))
    W("")

    # ---- (iii) intersections ----
    W("## (iii) Headline intersections\n")
    W("A *concept* is a closed attribute set (intent) with the set of sensors that have all of it (extent); "
      "every concept of the three matrices is enumerated and verified. Counts of concepts: "
      + ", ".join(f"{lv} {checks[lv]['n_concepts']}" for lv in levels) + ".\n")
    for lv in ("semantic", "data", "spec"):
        sens, attrs, has, drop = levels[lv]
        dtxt = drop_text(lv, drop, sens, has)
        W(f"### {lv} level ({len(sens)} sensors x {len(attrs)} attributes)\n")
        W(f"Discriminative counts leave out the attributes {dtxt}.\n")
        for key, label in (("n_discriminative", "discriminative attributes only"), ("n_attributes", "all attributes")):
            dd = drop if key == "n_discriminative" else None
            W(f"**(a) Pareto front, {label}**: no concept has both more sensors and more attributes.\n")
            prow = []
            for nn, a, ds in fronts[(lv, key)]:
                for d in ds:
                    prow.append([nn, a, d["total_rows"], d["sensors"].replace("|", ", "), fmt_attrs(d["attributes"], dd)])
            W(mdtable(["sensors", "attributes", "total rows", "which sensors", "attribute set"], prow))
            W("")
            W(f"**(b) Largest attribute set shared by at least k sensors, {label}** (ties: largest data volume first):\n")
            krow = []
            for k, a, ds in perk[(lv, key)]:
                ds = sorted(ds, key=lambda d: (-d["total_rows"], -d["n_sensors"], d["attributes"]))
                for d in ds[:MAX_TIES]:
                    krow.append([k, a, len(ds), d["n_sensors"], d["sensors"].replace("|", ", "), d["total_rows"],
                                 fmt_attrs(d["attributes"], dd)])
            W(mdtable(["k", "max attributes", "tied concepts", "actual sensors", "sensors", "total rows",
                       "attribute set"], krow))
            W(f"(At most {MAX_TIES} tied concepts per k are shown, largest data volume first; all are in "
              "`field_concepts.csv`.)")
            W("")
        W(f"**(c) Top 10 concepts** with >= 3 sensors and >= 4 discriminative attributes "
          f"(not counted: {dtxt}).\n")
        for key, title in (("area", "by area = sensors x discriminative attributes"),
                           ("rows", "by total data volume (rows of the extent sensors)")):
            W(f"Ranked {title}:\n")
            trow = []
            for i, d in enumerate(top_concepts(concepts[lv], key), 1):
                trow.append([i, d["n_sensors"], d["n_discriminative"], d["area_discriminative"], d["total_rows"],
                             d["sensors"].replace("|", ", "), fmt_attrs(d["attributes"], drop)])
            W(mdtable(["#", "sensors", "disc. attrs", "area", "total rows", "which sensors",
                       "discriminative attributes"], trow))
            W("")

    W("### Attribute coverage (semantic, binary >= 0.5)\n")
    cov = sorted(sem_attrs, key=lambda a: (-sum(1 for s in sensors if a in sem_bin[s]), a))
    W(mdtable(["attribute", "n sensors", "which sensors"],
              [[a, sum(1 for s in sensors if a in sem_bin[s]),
                ", ".join(s for s in sensors if a in sem_bin[s])] for a in cov]))
    W("")
    W("### Most similar sensor pairs (Jaccard on the semantic sets, top 12)\n")
    pj = []
    for i, a in enumerate(sensors):
        for b in sensors[i + 1:]:
            inter, uni = sem_bin[a] & sem_bin[b], sem_bin[a] | sem_bin[b]
            pj.append((len(inter) / len(uni) if uni else 0.0, a, b, len(inter), len(uni)))
    pj.sort(key=lambda t: (-t[0], t[1], t[2]))
    W(mdtable(["sensor A", "sensor B", "|intersection|", "|union|", "Jaccard"],
              [[a, b, ni, nu, f"{j:.4f}"] for j, a, b, ni, nu in pj[:12]]))
    W("")

    # ---- (iv) K-means ----
    W("## (iv) K-means families\n")
    W(f"Features: the {len(feat_attrs)} semantic presence fractions per sensor ({n} sensors). Excluded: "
      f"{', '.join(KMEANS_EXCLUDE)} (constant 1 on every sensor); sensor_position is kept as a feature "
      "(it is 0 for QTVS and VKYY). Deterministic k-means++ (seed 0, 20 restarts). "
      "K chosen by the highest silhouette.\n")
    W(mdtable(["K", "inertia", "silhouette", ""],
              [[k, f"{ine:.4f}", f"{sil:.4f}",
                ("best" if k == k_best else "") + (" PDF's 6 categories" if k == 6 else "")]
               for k, ine, sil, _ in scan]))
    W("")
    cent = res_best.centers
    for f in sorted(members):
        c = cent[f - 1]
        high = [(feat_attrs[j], c[j]) for j in np.argsort(-c) if c[j] >= BINARY_THRESHOLD]
        W(f"**Family F{f}** ({len(members[f])} sensors): " + ", ".join(
            f"{s} ({type_of[s]})" for s in members[f]))
        W("")
        W("- centroid attributes >= 0.5: " + ", ".join(f"{a} {x:.2f}" for a, x in high))
        W("")
    W("Family (K = %d) versus PDF category:\n" % k_best)
    W(mdtable(["family"] + cats + ["total"],
              [[f"F{f}"] + [sum(1 for s in members[f] if category_of[s] == c) for c in cats] + [len(members[f])]
               for f in sorted(members)]))
    W("")
    W("K = 6 (same count as the PDF's categories) versus PDF category:\n")
    W(mdtable(["family"] + cats + ["total", "sensors"],
              [[f"G{f}"] + [sum(1 for s in m6[f] if category_of[s] == c) for c in cats] + [len(m6[f]), ", ".join(m6[f])]
               for f in sorted(m6)]))
    W("")
    types = sorted(set(type_of.values()))
    W(f"Family (K = {k_best}) versus data sensor type:\n")
    W(mdtable(["family"] + types,
              [[f"F{f}"] + [sum(1 for s in members[f] if type_of[s] == t) for t in types] for f in sorted(members)]))
    W("")

    # ---- (v) implications ----
    W("## (v) What this means for one unified detection table\n")
    for line in impl["full"]:
        W(line)
    W("")

    # ---- (vi) constants and ids ----
    W("## (vi) Constant per-detection quantities and the raw `id` column\n")
    W("Every quantity below takes a single distinct value over all rows of its sensor. Placeholders "
      "(scores) have usability 0; nominal (error) columns stay usable as `*_nominal`.\n")
    crow = []
    for sid in sensors:
        for key, (a, kind) in CONST_QTY.items():
            if a in S[sid]["const"]:
                _, val, cnt = S[sid]["const"][a]
                crow.append([sid, a, "constant placeholder" if kind == "placeholder" else "nominal (constant)",
                             val, cnt, round(cnt / S[sid]["rows"], 4)])
    W(mdtable(["sensor", "quantity", "treatment", "value", "rows carrying it", "fraction of sensor rows"], crow))
    W("")
    W(f"Raw `id` column: `entity_id` when the sensor has at least {ID_RATIO} x as many distinct `id` values as "
      "distinct `objectId` values (and more than one `id` value), `id_label` when it has 2+ values but fewer, "
      "`constant` when it has exactly one value (counts as neither), `none` when the column is empty. "
      "Distinct counts are over all rows of the sensor, empty cells excluded.\n")
    irow = []
    for sid in sensors:
        s = S[sid]
        n_id, n_oid = len(s["ids"]), len(s["oids"])
        ex = ", ".join(x if len(x) <= 28 else x[:25] + "..." for x in sorted(s["ids"])[:3])
        irow.append([sid, s["rows"], n_oid, n_id, f"{n_id / n_oid:.2f}" if n_oid else "-", s["id_rows"],
                     s["id_kind"], ex])
    W(mdtable(["sensor", "rows", "distinct objectId", "distinct id", "id / objectId", "rows with id", "classified as",
               "first id values (sorted)"], irow))
    W("")
    return "\n".join(out)


def implications(v: dict, members: dict) -> dict:
    """The unified-table bullets, as a short list (executive summary) and a full list (section v)."""
    sensors, frac, sem_bin, sem_attrs, S = v["sensors"], v["frac"], v["sem_bin"], v["sem_attrs"], v["S"]
    type_of, k_best, scan, concepts, rows_of = v["type_of"], v["k_best"], v["scan"], v["concepts"], v["rows_of"]
    n = len(sensors)
    full: list[str] = []
    short: list[str] = []
    full_cov = [a for a in sem_attrs if all(a in sem_bin[s] for s in sensors)]
    near = [a for a in sem_attrs if 0 < sum(1 for s in sensors if a in sem_bin[s]) < n
            and sum(1 for s in sensors if a in sem_bin[s]) >= n - 4]
    b = (f"- **Core columns, usable for all {n} sensors**: {', '.join(full_cov) or '(none)'}. "
         f"Attributes missing for only 1-4 sensors: "
         + (", ".join(f"{a} (missing: {', '.join(s for s in sensors if a not in sem_bin[s])})" for a in near) or "none")
         + ".")
    full.append(b)
    short.append(b)
    geo_abs = [s for s in sensors if "object_position" in sem_bin[s]]
    geo_bearing = [s for s in sensors if "object_position" not in sem_bin[s]
                   and ({"azimuth", "azimuth_coarse"} & sem_bin[s])]
    geo_none = [s for s in sensors if s not in geo_abs and s not in geo_bearing]
    b = (f"- **Geometry splits the sensors in three groups** (this decides how a position is derived): "
         f"absolute object position (object_position) - {', '.join(geo_abs)} ({len(geo_abs)}); "
         f"bearing only (azimuth, no object position) - {', '.join(geo_bearing)} ({len(geo_bearing)}); "
         f"no usable geometry - {', '.join(geo_none) or 'none'}.")
    full.append(b)
    short.append(b)
    bear_rng = [s for s in geo_bearing if "range" in sem_bin[s]]
    bear_el = [s for s in geo_bearing if "elevation" in sem_bin[s]]
    bear_coarse = [s for s in geo_bearing if "azimuth_coarse" in sem_bin[s]]
    full.append(f"  Among the bearing-only sensors: with a real range - {', '.join(bear_rng) or 'none'}; "
                f"with elevation - {', '.join(bear_el) or 'none'}; coarse 45-degree sectors - "
                f"{', '.join(bear_coarse) or 'none'}.")
    vel = [s for s in sensors if "velocity_horizontal" in sem_bin[s]]
    vel_up = [s for s in vel if "velocity_vertical" in sem_bin[s]]
    b = (f"- **Velocity** (east/north) is usable for {len(vel)} sensors ({', '.join(vel)}); a real vertical rate for "
         f"{len(vel_up)} ({', '.join(vel_up)}). Sensors whose nested track info gives speed/heading instead: "
         f"{', '.join(s for s in sensors if {'speed', 'heading'} <= sem_bin[s] and s not in vel) or 'none'}.")
    full.append(b)
    short.append(b)
    cls = [s for s in sensors if "classification" in sem_bin[s]]
    part = [s for s in sensors if 0 < frac[s]["classification"] < BINARY_THRESHOLD]
    never = [s for s in sensors if frac[s]["classification"] == 0]
    sub = [s for s in sensors if "classification_subclass" in sem_bin[s]]
    b = (f"- **Classification** is usable on >= 50 % of the rows for {len(cls)} of {n} sensors "
         f"({', '.join(cls)}); with a sub-class for {len(sub)} ({', '.join(sub)}). Partial: "
         + (", ".join(f"{s} classifies {frac[s]['classification']:.0%} of rows" for s in part) or "none")
         + f". Never classifies: {', '.join(never) or 'none'}.")
    full.append(b)
    short.append(b)
    dcf = [s for s in sensors if "detection_confidence" in sem_bin[s]]
    ccf = [s for s in sensors if "classification_confidence" in sem_bin[s]]
    tcf = [s for s in sensors if "track_confidence" in sem_bin[s]]
    const_sc = sorted({(a, s) for s in sensors for a in PLACEHOLDER_ATTRS
                       if a in S[s]["const"] and a in ("detection_confidence", "classification_confidence",
                                                      "track_confidence")})
    def scores(a: str, bin_list: list[str]) -> str:
        pt = [s for s in sensors if 0 < frac[s][a] < BINARY_THRESHOLD]
        return (", ".join(bin_list) or "none") + (f" (partial: {', '.join(f'{s} {frac[s][a]:.0%}' for s in pt)})" if pt else "")
    b = (f"- **Confidence scores** carry information only where they vary: detection_confidence on "
         f"{scores('detection_confidence', dcf)}; classification_confidence on {scores('classification_confidence', ccf)} "
         f"(counted on usable-class rows only); track_confidence on {scores('track_confidence', tcf)}. "
         "Constant placeholders (no information): " +
         "; ".join(f"{a}: " + ", ".join(f"{s} ({S[s]['const'][a][1]})" for aa, s in const_sc if aa == a)
                   for a in ("detection_confidence", "classification_confidence", "track_confidence")
                   if any(aa == a for aa, _ in const_sc)) + ".")
    full.append(b)
    short.append(b)
    ident = [s for s in sensors if {"serial_number", "drone_mac", "pilot_mac", "model", "entity_id"} & sem_bin[s]]
    ent = [s for s in sensors if "entity_id" in sem_bin[s]]
    lab = [s for s in sensors if "id_label" in sem_bin[s]]
    b = (f"- **Identity-bearing fields** (serial number, MAC, model/maker or a per-object entity id): "
         f"{', '.join(ident)} ({len(ident)}); these are the candidates for matching a detection to a known drone "
         f"without relying on geometry alone. Per-object `entity_id`: {', '.join(ent)}; categorical `id_label` "
         f"(not an identity): {', '.join(lab)}.")
    full.append(b)
    short.append(b)
    err = [s for s in sensors if "position_error" in sem_bin[s]]
    errn = [s for s in sensors if "position_error_nominal" in sem_bin[s]]
    errx = [s for s in sensors if s not in err and s not in errn]
    b = (f"- **Position uncertainty**: per-detection for {len(err)} sensors ({', '.join(err)}); a nominal constant "
         f"for {len(errn)} ({', '.join(f'{s} {S[s]['const']['position_error'][1]}' for s in errn)}); none for "
         f"{', '.join(errx)}, where the table must carry a sensor-type default or the angular errors.")
    full.append(b)
    short.append(b)
    top_sem = top_concepts(concepts["semantic"], "area", 3)
    b = ("- **Largest identifiable groups** (by area, see (iii)): " + "; ".join(
        f"{{{d['sensors'].replace('|', ', ')}}} share {d['n_discriminative']} discriminative attributes"
        for d in top_sem) + ".")
    full.append(b)
    short.append(b)
    best = max(t[2] for t in scan)
    close = [t for t in scan if t[0] != k_best and best - t[2] <= 0.01]
    b = (f"- The best K-means partition has K = {k_best} (silhouette {best:.4f}); "
         + (("K values within 0.01 of the best silhouette: " + ", ".join(f"K = {t[0]} ({t[2]:.4f})" for t in close)
             + " - the choice of K is therefore not sharp. ") if close else "")
         + "The contingency tables in (iv) show how far the families follow the PDF categories. "
         + ("The silhouette is below 0.25, i.e. the cluster structure is weak: the families are a convenient grouping, "
            "not sharply separated types. " if best < 0.25 else "")
         + "A unified table should therefore be keyed on attribute sets "
           "(geometry type, velocity, classification depth, identity) rather than on sensor category.")
    full.append(b)
    short.append(b)
    return dict(full=full, short=short)


if __name__ == "__main__":
    raise SystemExit(main())
