#!/usr/bin/env python3
"""Build the unified, analysis-ready detection / ground-truth tables (step U1).

Outputs (all under data/derived/, gitignored):
  iteration_refs.csv     one row per iteration: local ENU frame origin + file counts
  detections.csv.gz      ONE ROW PER non-ragged row of every sensor file (both trees)
  groundtruth.csv.gz     one row per Dronetag row (testing tree)
  build_log.md           per-sensor statistics, placeholders, duplicates, info keys

Everything geometric goes through tools/cuas_common.py (local ENU frame per iteration,
per-row RAD/DEG coordinate systems, ...). Files are processed in parallel (one process
per sensor file). Exit status is non-zero if any self-check fails.

Run:  python3 tools/build_detections.py
"""

from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuas_common as C  # noqa: E402

OUT = C.DERIVED

DET_COLUMNS = [
    "iter_key", "tree", "file", "row", "team", "sensor_id", "sensor_type", "sensor_key", "node_id",
    "object_id", "report_id", "t",
    "sensor_lat", "sensor_lon", "sensor_alt", "sensor_alt_datum", "sensor_pos_valid",
    "sensor_e", "sensor_n", "sensor_u",
    "meas_kind", "lat", "lon", "alt", "alt_datum", "e", "n", "u",
    "az_deg", "el_deg", "range_m", "bearing_source", "az_ref", "az_err_deg", "el_err_deg",
    "err_x_m", "err_y_m", "err_z_m", "ve", "vn", "vu", "det_conf",
    "cls", "cls_conf", "cls_sub", "cls_sub_conf",
    "rcs", "freq_mhz", "signal_amp", "model", "serial", "protocol",
    "n_assoc", "assoc_nodes", "state", "behaviour", "colour", "info",
]
STR_COLS = {
    "iter_key", "tree", "file", "row", "team", "sensor_id", "sensor_type", "sensor_key", "node_id",
    "object_id", "report_id", "sensor_alt_datum", "meas_kind", "alt_datum", "bearing_source",
    "az_ref", "cls", "cls_sub", "model", "serial", "protocol", "assoc_nodes", "state",
    "behaviour", "colour", "info",
}
INT_COLS = {"sensor_pos_valid", "n_assoc"}
FLOAT_COLS = [c for c in DET_COLUMNS if c not in STR_COLS and c not in INT_COLS]

GT_COLUMNS = ["iter_key", "flight_key", "side", "drone_name", "t", "lat", "lon", "geo_alt",
              "baro_alt", "height", "e", "n", "u", "ve", "vn", "vu", "heading", "h_acc",
              "v_acc", "speed_acc", "flight_id"]
GT_STR = {"iter_key", "flight_key", "side", "drone_name", "flight_id"}

M_PER_DEG_LAT = 110574.0
M_PER_DEG_LON_EQ = 111320.0


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def num(cols: dict, name: str, n: int) -> np.ndarray:
    """Column -> float64, blanks / junk -> NaN, missing column -> NaN."""
    cells = cols.get(name)
    if cells is None:
        return np.full(n, np.nan)
    try:
        return np.array([c if c else "nan" for c in cells], dtype=float)
    except ValueError:
        return C.to_float(cells, n)


def datum_letter(cells: list[str] | None, n: int) -> list[str]:
    if cells is None:
        return [""] * n
    return ["E" if c.endswith("_E") else "G" if c.endswith("_G") else "" for c in cells]


def norm_ws(s: str) -> str:
    return " ".join(str(s).split())


def looks_like_serial(s: str) -> bool:
    """Drone serials in the data are single-case alphanumerics (>=10 chars, letters + digits);
    random per-signal ids ('kXW5dCu49ZJHokn') are mixed-case and protocol names contain '-'/space."""
    if len(s) < 10 or not s.isalnum():
        return False
    if s != s.upper() and s != s.lower():
        return False
    return any(ch.isdigit() for ch in s) and any(ch.isalpha() for ch in s)


# sensors whose `id` column carries a drone serial (verified: same serials appear in
# CEQX / ILYS / STUG / IKJX / MMJJ, differing only in case or a vendor prefix)
SERIAL_ID_SENSORS = {"CEQX", "ILYS", "STUG", "IKJX", "MMJJ"}
# sensors whose `id` column is protocol / drone-type text
PROTOCOL_ID_SENSORS = {"ORFH", "YXAB"}


class Parsers:
    """Per-process caches of literal parsing keyed by the exact cell string."""

    def __init__(self):
        self.kv_cache: dict[tuple[str, str], dict] = {}
        self.cls_cache: dict[str, tuple] = {}
        self.sig_cache: dict[str, tuple] = {}
        self.assoc_cache: dict[tuple[str, str], tuple] = {}
        self.state_cache: dict[str, str] = {}
        self.file_cache: dict[str, str] = {}

    # ---- objectInfo / trackInfo / additionalInformation -> flat dict
    def kv(self, col: str, cell: str) -> dict:
        key = (col, cell)
        hit = self.kv_cache.get(key)
        if hit is not None:
            return hit
        d: dict = {}
        if cell:
            obj = C.parse_literal(cell)
            if isinstance(obj, list):
                for ent in obj:
                    if not isinstance(ent, dict) or "type" not in ent:
                        continue
                    typ = str(ent["type"])
                    val = ent.get("value", "")
                    if typ == "Drone Event Info":
                        self._drone_event(d, val)
                    else:
                        d[typ] = val
                    if "error" in ent:
                        d[typ + "_err"] = ent["error"]
            elif isinstance(obj, dict):
                d.update({str(k): v for k, v in obj.items()})
            else:
                d["_" + col] = cell
        self.kv_cache[key] = d
        return d

    @staticmethod
    def _drone_event(d: dict, val) -> None:
        """CEQX trackInfo 'Drone Event Info': a JSON blob or just the serial string."""
        if isinstance(val, str) and val[:1] == "{":
            try:
                j = json.loads(val)
            except ValueError:
                d["drone_event_info"] = val
                return
            if j.get("serial_number"):
                d["serial number"] = j["serial_number"]
            if j.get("factory_name"):
                d["model"] = j["factory_name"]
            for src, dst in (("id", "ceqx_event_id"), ("can_takeover", "can_takeover"),
                             ("support_mitigation", "support_mitigation"),
                             ("speed_ms", "speed_ms")):
                if j.get(src) is not None:
                    d[dst] = j[src]
        elif val:
            d["serial number"] = val

    # ---- classification
    def cls(self, cell: str) -> tuple:
        hit = self.cls_cache.get(cell)
        if hit is None:
            hit = self.cls_cache[cell] = C.top_class(cell)
        return hit

    # ---- signal -> (freq_mhz, amp, start, stop) of the strongest entry
    def signal(self, cell: str) -> tuple:
        hit = self.sig_cache.get(cell)
        if hit is not None:
            return hit
        res = (math.nan, math.nan, math.nan, math.nan)
        obj = C.parse_literal(cell)
        if isinstance(obj, list):
            ents = [e for e in obj if isinstance(e, dict)]
            if ents:
                best = max(ents, key=lambda e: _f(e.get("amplitude"), -math.inf))
                res = (_f(best.get("centreFrequency")), _f(best.get("amplitude")),
                       _f(best.get("startFrequency")), _f(best.get("stopFrequency")))
        self.sig_cache[cell] = res
        return res

    # ---- associatedDetection / derivedDetection -> (count, node ids)
    def assoc(self, a: str, b: str) -> tuple[int, str]:
        key = (a, b)
        hit = self.assoc_cache.get(key)
        if hit is not None:
            return hit
        n = 0
        nodes: set[str] = set()
        for cell in key:
            obj = C.parse_literal(cell)
            if isinstance(obj, list):
                for e in obj:
                    if isinstance(e, dict):
                        n += 1
                        if e.get("nodeId"):
                            nodes.add(str(e["nodeId"]))
        res = (n, "|".join(sorted(nodes)))
        self.assoc_cache[key] = res
        return res

    # ---- state / behaviour: list of dicts -> 'type=value|type', else raw string
    def state(self, cell: str) -> str:
        hit = self.state_cache.get(cell)
        if hit is not None:
            return hit
        out = cell
        obj = C.parse_literal(cell)
        if isinstance(obj, list):
            parts = []
            for e in obj:
                if isinstance(e, dict) and "type" in e:
                    parts.append(f"{e['type']}={e['value']}" if e.get("value") not in (None, "")
                                 else str(e["type"]))
            out = "|".join(parts)
        self.state_cache[cell] = out
        return out

    # ---- associatedFile -> media types (URLs are not identity evidence)
    def files(self, cell: str) -> str:
        hit = self.file_cache.get(cell)
        if hit is not None:
            return hit
        obj = C.parse_literal(cell)
        out = ""
        if isinstance(obj, list):
            out = "|".join(sorted({str(e.get("type", "")) for e in obj if isinstance(e, dict)}))
        self.file_cache[cell] = out
        return out


def _f(v, default=math.nan) -> float:
    try:
        x = float(v)
        return x if x == x else default
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# per-file worker
# --------------------------------------------------------------------------- #


def process_file(task: dict) -> dict:
    path = Path(task["path"])
    frame = C.Frame(*task["frame"])
    placeholder = bool(task["placeholder"])      # True: rangeBearing_range is a placeholder
    sid = task["sensor_id"]
    tb = C.read_table(path)
    cols, n = tb.columns, tb.n_rows
    out: dict = {"_n_rows": n, "_ragged": tb.ragged_rows, "_file": task["rel"]}
    if n == 0:
        out["cols"] = {c: ([] if c in STR_COLS or c in INT_COLS else np.zeros(0)) for c in DET_COLUMNS}
        out["info_keys"] = Counter()
        return out
    P = Parsers()
    nan = np.full(n, np.nan)

    def const(v):
        return [v] * n

    # ---- identity / time
    t = C.parse_times(cols["timestamp"])
    node_id = cols.get("nodeId") or const("")
    object_id = cols.get("objectId") or const("")
    report_id = cols.get("reportId") or const("")

    # ---- sensor position
    sx, sy, sz = num(cols, "sensor_location_x", n), num(cols, "sensor_location_y", n), \
        num(cols, "sensor_location_z", n)
    scs = cols.get("sensor_location_coordinateSystem") or "LAT_LNG_DEG_M"
    scs = scs[0] if isinstance(scs, list) and len(set(scs)) == 1 else scs
    slat, slon = C.latlon_deg(sx, sy, scs)
    valid = C.sensor_position_valid(slat, slon)
    with np.errstate(invalid="ignore"):
        se, sn, su = frame.to_enu(slat, slon, np.nan_to_num(sz))
    su = np.where(np.isfinite(sz), su, np.nan)
    se, sn, su = (np.where(valid, a, np.nan) for a in (se, sn, su))

    # ---- measurement
    kind = np.full(n, "", dtype=object)
    lat, lon, alt, e, nn, u = (np.full(n, np.nan) for _ in range(6))
    alt_datum = const("")
    az, el, rng = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    bsrc = np.full(n, "", dtype=object)
    azref = np.full(n, "", dtype=object)
    if "location_x" in cols:
        x, y, z = num(cols, "location_x", n), num(cols, "location_y", n), num(cols, "location_z", n)
        cs = cols.get("location_coordinateSystem") or "LAT_LNG_DEG_M"
        cs = cs[0] if isinstance(cs, list) and len(set(cs)) == 1 else cs
        lat, lon = C.latlon_deg(x, y, cs)
        alt = z
        f2 = np.isfinite(lat) & np.isfinite(lon)
        f3 = f2 & np.isfinite(alt)
        kind[f3] = "pos3d"
        kind[f2 & ~f3] = "pos2d"
        lat, lon = np.where(f2, lat, np.nan), np.where(f2, lon, np.nan)
        with np.errstate(invalid="ignore"):
            e, nn, u = frame.to_enu(lat, lon, np.nan_to_num(alt))
        e, nn = np.where(f2, e, np.nan), np.where(f2, nn, np.nan)
        u = np.where(f3, u, np.nan)
        alt_datum = datum_letter(cols.get("location_datum"), n)
        has = f2 & valid
        with np.errstate(invalid="ignore"):
            de, dn, du = e - se, nn - sn, u - su
            dh = np.hypot(de, dn)
            az = np.where(has, C.azimuth_deg(de, dn), np.nan)
            p3 = has & (kind == "pos3d") & np.isfinite(du)
            el = np.where(p3, np.degrees(np.arctan2(du, dh)), np.nan)
            rng = np.where(p3, np.hypot(dh, du), np.where(has & (kind == "pos2d"), dh, np.nan))
        bsrc[has] = "derived"
        azref[has] = "true"
    if "rangeBearing_azimuth" in cols:
        raz = num(cols, "rangeBearing_azimuth", n)
        rel_ = num(cols, "rangeBearing_elevation", n)
        rrg = num(cols, "rangeBearing_range", n)
        magnetic = np.array(["MAGNETIC" in c for c in cols["rangeBearing_datum"]]) \
            if "rangeBearing_datum" in cols else np.zeros(n, bool)
        fz = np.isfinite(raz) & (kind == "")
        real_range = fz & np.isfinite(rrg) & valid & (not placeholder)
        kind[fz] = "bearing"
        kind[fz & np.isfinite(rel_)] = "bearing_el"
        kind[real_range] = "range_bearing"
        az = np.where(fz, raz % 360.0, az)
        el = np.where(fz, rel_, el)
        rng = np.where(fz & np.isfinite(rrg) & (not placeholder), rrg, rng)
        bsrc[fz] = "measured"
        azref[fz] = np.where(magnetic[fz], "magnetic", "true")
        if real_range.any():
            do, dno, duo = C.bearing_offset(az, el, rng)
            ee, nr, uu = se + do, sn + dno, su + duo
            la, lo, hh = frame.to_geodetic(ee, nr, uu)
            lat, lon, alt = (np.where(real_range, a, b) for a, b in ((la, lat), (lo, lon), (hh, alt)))
            e, nn, u = (np.where(real_range, a, b) for a, b in ((ee, e), (nr, nn), (uu, u)))
    az_err = num(cols, "rangeBearing_azimuthError", n)
    el_err = num(cols, "rangeBearing_elevationError", n)

    # ---- errors (CEQX reports x/y in degrees)
    ex, ey, ez = (num(cols, f"location_{a}Error", n) for a in "xyz")
    if sid == "CEQX":
        ex = ex * M_PER_DEG_LON_EQ * np.cos(np.radians(lat))
        ey = ey * M_PER_DEG_LAT
    ve, vn, vu = (num(cols, f"enuVelocity_{a}Rate", n) for a in ("east", "north", "up"))
    det_conf = num(cols, "detectionConfidence", n)

    # ---- per-row string-derived fields
    def col_or_blank(name):
        return cols.get(name) or const("")

    cls_c = col_or_blank("classification")
    obj_c, trk_c, add_c = col_or_blank("objectInfo"), col_or_blank("trackInfo"), \
        col_or_blank("additionalInformation")
    sig_c, id_c = col_or_blank("signal"), col_or_blank("id")
    task_c, dest_c = col_or_blank("taskId"), col_or_blank("destinationId")
    file_c, state_c, beh_c = col_or_blank("associatedFile"), col_or_blank("state"), \
        col_or_blank("behaviour")
    asc_c, drv_c = col_or_blank("associatedDetection"), col_or_blank("derivedDetection")
    colour = col_or_blank("colour")

    cls = [""] * n
    cls_conf, cls_sub_conf = np.full(n, np.nan), np.full(n, np.nan)
    cls_sub = [""] * n
    rcs, freq, amp = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    model, serial, protocol = [""] * n, [""] * n, [""] * n
    n_assoc, assoc_nodes = [0] * n, [""] * n
    state_o, beh_o, info_o = [""] * n, [""] * n, [""] * n
    kc: Counter = Counter()
    is_serial_sensor = sid in SERIAL_ID_SENSORS
    is_proto_sensor = sid in PROTOCOL_ID_SENSORS
    dumps = json.dumps
    for i in range(n):
        c = cls_c[i]
        if c:
            cls[i], cls_conf[i], cls_sub[i], cls_sub_conf[i] = P.cls(c)
        info: dict = {}
        for col, cell in (("objectInfo", obj_c[i]), ("trackInfo", trk_c[i]),
                          ("additionalInformation", add_c[i])):
            if cell:
                for k, v in P.kv(col, cell).items():
                    info[k if k not in info else f"{k}@{col}"] = v
        idv = id_c[i]
        if idv:
            info["_id"] = idv
        if task_c[i]:
            info["_taskId"] = task_c[i]
        if dest_c[i]:
            info["_destinationId"] = dest_c[i]
        if file_c[i]:
            ft = P.files(file_c[i])
            if ft:
                info["_file_types"] = ft
        sg = (math.nan,) * 4
        if sig_c[i]:
            sg = P.signal(sig_c[i])
            freq[i], amp[i] = sg[0], sg[1]
            if sg[2] == sg[2]:
                info["sig_start_mhz"], info["sig_stop_mhz"] = sg[2], sg[3]
        rv = info.get("rcs")
        if rv is not None:
            rcs[i] = _f(rv)
        model[i] = norm_ws(info.get("model", ""))
        ser = info.get("serial number", "")
        pro = info.get("protocol") or info.get("technology") or ""
        if idv:
            if not ser and is_serial_sensor and looks_like_serial(idv):
                ser = idv
            if not pro:
                if is_proto_sensor:
                    pro = idv
                elif is_serial_sensor and not looks_like_serial(idv) and ("-" in idv or " " in idv):
                    pro = idv
        serial[i] = norm_ws(ser).upper()
        protocol[i] = norm_ws(pro)
        if asc_c[i] or drv_c[i]:
            n_assoc[i], assoc_nodes[i] = P.assoc(asc_c[i], drv_c[i])
        if state_c[i]:
            state_o[i] = P.state(state_c[i])
        if beh_c[i]:
            beh_o[i] = P.state(beh_c[i])
        if info:
            info_o[i] = dumps(info, sort_keys=True, separators=(",", ":"))
            kc.update(info.keys())

    res = {
        "iter_key": const(task["iter_key"]), "tree": const(task["tree"]), "file": const(task["rel"]),
        "row": cols["row"], "team": const(task["team"]), "sensor_id": const(sid),
        "sensor_type": const(task["sensor_type"]), "sensor_key": const(task["sensor_key"]),
        "node_id": node_id, "object_id": object_id, "report_id": report_id, "t": t,
        "sensor_lat": slat, "sensor_lon": slon, "sensor_alt": sz,
        "sensor_alt_datum": datum_letter(cols.get("sensor_location_datum"), n),
        "sensor_pos_valid": valid.astype(int).tolist(), "sensor_e": se, "sensor_n": sn, "sensor_u": su,
        "meas_kind": kind.tolist(), "lat": lat, "lon": lon, "alt": alt, "alt_datum": alt_datum,
        "e": e, "n": nn, "u": u, "az_deg": az, "el_deg": el, "range_m": rng,
        "bearing_source": bsrc.tolist(), "az_ref": azref.tolist(), "az_err_deg": az_err,
        "el_err_deg": el_err, "err_x_m": ex, "err_y_m": ey, "err_z_m": ez, "ve": ve, "vn": vn,
        "vu": vu, "det_conf": det_conf, "cls": cls, "cls_conf": cls_conf, "cls_sub": cls_sub,
        "cls_sub_conf": cls_sub_conf, "rcs": rcs, "freq_mhz": freq, "signal_amp": amp,
        "model": model, "serial": serial, "protocol": protocol, "n_assoc": n_assoc,
        "assoc_nodes": assoc_nodes, "state": state_o, "behaviour": beh_o, "colour": colour,
        "info": info_o,
    }
    assert set(res) == set(DET_COLUMNS), set(DET_COLUMNS) ^ set(res)
    assert all(len(v) == n for v in res.values())
    res["sensor_lat"] = np.asarray(res["sensor_lat"], float)
    res["sensor_lon"] = np.asarray(res["sensor_lon"], float)
    out["cols"] = res
    out["info_keys"] = kc
    return out


def frame_of(it_index: int):
    return C.iteration_frame(ITERATIONS[it_index])


def scan_ranges(task: tuple[str, str]) -> tuple[str, set]:
    """Distinct finite rangeBearing_range values of one file (for placeholder detection)."""
    skey, path = task
    vals: set[float] = set()
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if "rangeBearing_range" not in header:
            return skey, vals
        ix = header.index("rangeBearing_range")
        for row in reader:
            if len(row) == len(header) and row[ix]:
                try:
                    v = float(row[ix])
                except ValueError:
                    continue
                if v == v:
                    vals.add(v)
    return skey, vals


ITERATIONS = C.iterations()


# --------------------------------------------------------------------------- #
# ground truth
# --------------------------------------------------------------------------- #


def build_groundtruth(its, frames):
    cols: dict[str, list] = {c: [] for c in GT_COLUMNS}
    raw_rows = 0
    for it, fr in zip(its, frames):
        for gf in C.groundtruth_files(it):
            with gf.path.open(newline="", encoding="utf-8") as fh:      # independent row count
                raw_rows += sum(1 for _ in csv.reader(fh)) - 1
            tb = C.read_table(gf.path)
            n = tb.n_rows
            if n == 0:
                continue
            lat, lon = num(tb.columns, "latitude", n), num(tb.columns, "longitude", n)
            ga = num(tb.columns, "geo_altitude", n)
            with np.errstate(invalid="ignore"):
                e, nn, u = fr.to_enu(lat, lon, np.nan_to_num(ga))
            u = np.where(np.isfinite(ga), u, np.nan)
            vals = {
                "iter_key": [it.key] * n, "flight_key": [gf.flight_key] * n,
                "side": [gf.side] * n, "drone_name": [gf.drone_name] * n,
                "t": C.parse_times(tb.columns["timestamp"]), "lat": lat, "lon": lon,
                "geo_alt": ga, "baro_alt": num(tb.columns, "altitude", n),
                "height": num(tb.columns, "height", n), "e": e, "n": nn, "u": u,
                "ve": num(tb.columns, "velocity_x", n), "vn": num(tb.columns, "velocity_y", n),
                "vu": num(tb.columns, "velocity_z", n), "heading": num(tb.columns, "heading", n),
                "h_acc": num(tb.columns, "horizontal_accuracy", n),
                "v_acc": num(tb.columns, "vertical_accuracy", n),
                "speed_acc": num(tb.columns, "speed_accuracy", n),
                "flight_id": tb.columns.get("flight_id") or [""] * n,
            }
            for c in GT_COLUMNS:
                v = vals[c]
                cols[c].extend(v.tolist() if isinstance(v, np.ndarray) else v)
    out = {c: (cols[c] if c in GT_STR else np.array(cols[c], dtype=float)) for c in GT_COLUMNS}
    return out, raw_rows


# --------------------------------------------------------------------------- #
# log
# --------------------------------------------------------------------------- #


def mmm(a: np.ndarray) -> str:
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "-"
    return f"{a.min():.6g} / {np.median(a):.6g} / {a.max():.6g}"


def write_log(path: Path, det: dict, per_file: list[dict], dup_skipped: list[str],
              placeholders: dict, ranges_seen: dict, n_total: int, runtime: float,
              info_keys: dict) -> None:
    sk = np.array(det["sensor_key"], dtype=object)
    kind = np.array(det["meas_kind"], dtype=object)
    L = ["# build_detections log", "",
         f"Generated by `tools/build_detections.py` in {runtime:.0f} s. Rows: {n_total}.", ""]
    L += ["## Duplicate files skipped (browser re-downloads, byte-identical to their sibling)", ""]
    L += [f"- `{d}`" for d in dup_skipped] or ["- none"]
    L += ["", "## Placeholder ranges (rangeBearing_range with a single distinct finite value; range_m set to NaN)", ""]
    for k in sorted(ranges_seen):
        vals = sorted(ranges_seen[k])
        tag = "PLACEHOLDER" if k in placeholders else "real"
        L.append(f"- {k}: {len(vals)} distinct value(s) {vals[:5]} -> {tag}")
    if not ranges_seen:
        L.append("- none of the sensors has a rangeBearing_range column with values")
    L += ["", f"## Rows with no usable measurement (meas_kind empty): {int((kind == '').sum())}", ""]
    for k in sorted(set(sk[kind == ""])):
        L.append(f"- {k}: {int(((kind == '') & (sk == k)).sum())}")
    L += ["", "## Suspicious rows (kept as reported; NOT fixed or dropped)", ""]
    posk = np.isin(kind, ["pos3d", "pos2d"])
    zero = posk & (det["lat"] == 0) & (det["lon"] == 0)
    L.append(f"- target position exactly lat=0,lon=0 (e/n/range become millions of metres): {int(zero.sum())} rows"
             + "".join(f"; {k}: {int((zero & (sk == k)).sum())}" for k in sorted(set(sk[zero]))))
    for f in sorted({det["file"][i] for i in np.flatnonzero(zero)[:50]}):
        L.append(f"  - `{f}`")
    far = posk & ((np.abs(det["e"]) > 50000) | (np.abs(det["n"]) > 50000)) & ~zero
    L.append("- target further than 50 km from the iteration origin (excluding lat/lon 0,0): "
             + (", ".join(f"{k}={int((far & (sk == k)).sum())}" for k in sorted(set(sk[far]))) or "none"))
    near = posk & (det["range_m"] < 5)
    L.append("- positioned target within 5 m of its own sensor (position carries no target information): "
             + (", ".join(f"{k}={int((near & (sk == k)).sum())}/{int((posk & (sk == k)).sum())}"
                          for k in sorted(set(sk[near]))) or "none"))
    zalt = np.isfinite(det["sensor_alt"]) & (det["sensor_alt"] == 0)
    L.append("- sensor_location_z == 0 on >50% of rows (unknown sensor altitude; derived el_deg/range_m use it as-is): "
             + (", ".join(k for k in sorted(set(sk)) if zalt[sk == k].mean() > 0.5) or "none"))
    ragged_files = [(f["_file"], f["_ragged"]) for f in per_file if f["_ragged"]]
    L += ["", "## Ragged rows skipped (field count != header)", ""]
    L += [f"- `{f}`: {r}" for f, r in ragged_files] or ["- none"]
    L += ["", "## Per-sensor statistics", "",
          "min / median / max of finite values. err_x_m is in METRES for every sensor "
          "(CEQX converted from degrees); STUG errors are the 5000/500 placeholder as reported; "
          "IKJX/MMJJ xError is reported as-is (0.026949, unit unconfirmed).", ""]
    stat_cols = ["az_deg", "el_deg", "range_m", "alt", "err_x_m", "det_conf", "rcs", "freq_mhz"]
    for k in sorted(set(sk)):
        m = sk == k
        files = [f for f in per_file if f["_skey"] == k]
        L += [f"### {k}", "",
              f"- files: {len(files)}, rows: {int(m.sum())}, ragged skipped: {sum(f['_ragged'] for f in files)}",
              "- meas_kind: " + ", ".join(f"{a or '(none)'}={b}" for a, b in sorted(Counter(kind[m]).items())),
              "- NaN counts: " + ", ".join(f"{c}={int(np.isnan(det[c][m]).sum())}" for c in ("e", "n", "u", "az_deg")),
              f"- sensor_pos_valid=0 rows: {int((np.asarray(det['sensor_pos_valid'])[m] == 0).sum())}"]
        for c in stat_cols:
            L.append(f"- {c}: {mmm(det[c][m])}")
        top = info_keys[k].most_common(15)
        L.append("- info keys (top 15): " + (", ".join(f"{a}={b}" for a, b in top) or "(none)"))
        L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# sanity probe
# --------------------------------------------------------------------------- #


def probe(det: dict, gt: dict, it_key: str, sensors: list[str], dt_max: float = 1.0) -> list[str]:
    g_it = np.array(gt["iter_key"], dtype=object) == it_key
    gt_t, gt_e, gt_n = gt["t"][g_it], gt["e"][g_it], gt["n"][g_it]
    order = np.argsort(gt_t)
    gt_t, gt_e, gt_n = gt_t[order], gt_e[order], gt_n[order]
    ik = np.array(det["iter_key"], dtype=object) == it_key
    sid = np.array(det["sensor_id"], dtype=object)
    lines = []
    for s in sensors:
        m = ik & (sid == s) & np.isfinite(det["e"]) & np.isfinite(det["n"])
        if not m.any():
            lines.append(f"{s}: no positioned detections in {it_key}")
            continue
        t, e, n = det["t"][m], det["e"][m], det["n"][m]
        lo = np.searchsorted(gt_t, t - dt_max, "left")
        hi = np.searchsorted(gt_t, t + dt_max, "right")
        w = int((hi - lo).max()) if len(t) else 0
        if w == 0:
            lines.append(f"{s}: {len(t)} detections, none within |dt|<={dt_max}s of a ground-truth point")
            continue
        idx = lo[:, None] + np.arange(w)[None, :]
        ok = idx < hi[:, None]
        idx = np.minimum(idx, len(gt_t) - 1)
        d = np.hypot(gt_e[idx] - e[:, None], gt_n[idx] - n[:, None])
        d = np.where(ok, d, np.inf).min(axis=1)
        d = d[np.isfinite(d)]
        lines.append(f"{s}: n_det={len(t)} matched={len(d)} median={np.median(d):.1f} m "
                     f"p10={np.percentile(d, 10):.1f} p90={np.percentile(d, 90):.1f} min={d.min():.1f}")
    return lines


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    its = ITERATIONS
    ctx = mp.get_context("fork")
    nproc = min(20, mp.cpu_count())

    sfiles = [(it, C.sensor_files(it)) for it in its]
    dup_skipped = []
    for it, fs in sfiles:
        kept = {sf.path for sf in fs}
        for p in sorted((it.path / "SensorData").glob("*.csv")):
            if p not in kept:
                dup_skipped.append(f"{p.relative_to(C.DATA).as_posix()}  [{it.key}]")

    with ctx.Pool(nproc) as pool:
        frames = pool.map(frame_of, range(len(its)))
        print(f"[{time.time() - t0:5.1f}s] frames done", flush=True)
        # placeholder-range detection across the whole dataset, per sensor_key
        rseen: dict[str, set] = defaultdict(set)
        for skey, vals in pool.map(scan_ranges, [(sf.sensor_key, str(sf.path))
                                                  for _, fs in sfiles for sf in fs]):
            rseen[skey] |= vals
        ranges_seen = {k: v for k, v in rseen.items() if v}
        placeholders = {k for k, v in ranges_seen.items() if len(v) == 1}
        print(f"[{time.time() - t0:5.1f}s] ranges scanned; placeholders: {sorted(placeholders)}", flush=True)

        tasks = []
        for it, fs, fr in zip(its, [f for _, f in sfiles], frames):
            for sf in fs:
                tasks.append({
                    "path": str(sf.path), "rel": sf.path.relative_to(C.DATA).as_posix(),
                    "iter_key": it.key, "tree": it.tree, "team": sf.team, "sensor_id": sf.sensor_id,
                    "sensor_type": sf.sensor_type, "sensor_key": sf.sensor_key,
                    "frame": (fr.lat0, fr.lon0, fr.h0), "placeholder": sf.sensor_key in placeholders,
                })
        # biggest files first for load balancing; results are re-sorted into task order
        order = sorted(range(len(tasks)), key=lambda i: -Path(tasks[i]["path"]).stat().st_size)
        res_sorted = pool.map(process_file, [tasks[i] for i in order], chunksize=1)
    results: list[dict] = [None] * len(tasks)  # type: ignore[list-item]
    for i, r in zip(order, res_sorted):
        r["_skey"] = tasks[i]["sensor_key"]
        results[i] = r
    print(f"[{time.time() - t0:5.1f}s] files parsed: {len(results)}", flush=True)

    # ---- concatenate
    det: dict = {}
    for c in DET_COLUMNS:
        if c in STR_COLS or c in INT_COLS:
            lst: list = []
            for r in results:
                lst.extend(r["cols"][c])
            det[c] = lst
        else:
            det[c] = np.concatenate([np.asarray(r["cols"][c], dtype=float) for r in results])
    n_total = len(det["t"])
    expected_rows = sum(r["_n_rows"] for r in results)

    # ---- groundtruth
    gt, gt_raw_rows = build_groundtruth(its, frames)
    n_gt = len(gt["t"])

    # ---- iteration refs
    with (OUT / "iteration_refs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["iter_key", "tree", "dataset", "scenario", "iteration", "lat0", "lon0", "h0",
                    "n_sensor_files", "n_gt_files"])
        for (it, fs), fr in zip(sfiles, frames):
            w.writerow([it.key, it.tree, it.dataset, it.scenario, it.iteration, repr(fr.lat0),
                        repr(fr.lon0), repr(fr.h0), len(fs), len(C.groundtruth_files(it))])

    # ---- write tables
    C.write_columns(OUT / "detections.csv.gz", det, DET_COLUMNS)
    print(f"[{time.time() - t0:5.1f}s] detections written", flush=True)
    C.write_columns(OUT / "groundtruth.csv.gz", gt, GT_COLUMNS)

    # ---- info keys per sensor (for log + frequency-field audit)
    info_keys: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        info_keys[r["_skey"]].update(r["info_keys"])
    freq_like = sorted({f"{s}:{k}" for s, c in info_keys.items() for k in c
                        if "freq" in k.lower() and not k.startswith("sig_")})

    runtime_pre_log = time.time() - t0
    write_log(OUT / "build_log.md", det, results, dup_skipped, placeholders, ranges_seen,
              n_total, runtime_pre_log, info_keys)
    if freq_like:
        print("frequency-like info keys found:", freq_like)

    # ---- self-checks
    kind = np.array(det["meas_kind"], dtype=object)
    posk = np.isin(kind, ["pos3d", "pos2d"])
    brk = np.isin(kind, ["bearing", "bearing_el"])
    checks = [
        ("(a) detections rows == sum of non-ragged rows of all sensor files",
         n_total == expected_rows, f"{n_total} vs {expected_rows}"),
        ("(b) t finite everywhere", bool(np.isfinite(det["t"]).all()), f"{int((~np.isfinite(det['t'])).sum())} bad"),
        ("(c) pos3d/pos2d rows have finite e,n",
         bool((np.isfinite(det["e"][posk]) & np.isfinite(det["n"][posk])).all()),
         f"{int(posk.sum())} rows"),
        ("(d) bearing rows have finite az_deg", bool(np.isfinite(det["az_deg"][brk]).all()),
         f"{int(brk.sum())} rows"),
        ("(e) groundtruth rows == sum of Dronetag rows", n_gt == gt_raw_rows, f"{n_gt} vs {gt_raw_rows}"),
        ("(f) pos3d rows have finite u", bool(np.isfinite(det["u"][kind == "pos3d"]).all()),
         f"{int((kind == 'pos3d').sum())} rows"),
    ]
    ok = True
    print("\nSELF-CHECKS")
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  ({detail})")
        ok &= passed

    # manifest cross-check (informational)
    mism = manifest_mismatches(results)
    print(f"  [info] manifest row-count mismatches (rows+ragged vs manifest): {mism}")

    print("\nSUMMARY")
    print(f"  detections: {n_total} rows, {len(results)} files, {len(its)} iterations, "
          f"meas_kind={dict(Counter(kind.tolist()))}")
    print(f"  groundtruth: {n_gt} rows")
    print("\nPROBE testing/S1-I1: median horizontal distance (m) to nearest ground-truth point, |dt|<=1 s")
    for line in probe(det, gt, "testing/S1-I1", ["STUG", "CEQX", "IUKL", "UXYJ", "NYTF", "EGOI"]):
        print("  " + line)
    total = time.time() - t0
    print(f"\nruntime: {total:.1f} s")
    log = OUT / "build_log.md"
    txt = log.read_text(encoding="utf-8").replace(f"in {runtime_pre_log:.0f} s", f"in {total:.0f} s", 1)
    log.write_text(txt, encoding="utf-8")
    print("ALL CHECKS PASSED" if ok else "SELF-CHECK FAILURE")
    return 0 if ok else 1


def manifest_mismatches(results: list[dict]) -> int:
    bad = 0
    for tree, root in C.TREES.items():
        mf = root / "ANONYMIZATION_MANIFEST.json"
        if not mf.exists():
            continue
        rows = {f["path"]: f["rows"] for f in json.loads(mf.read_text())["files"]}
        for r in results:
            rel = r["_file"]
            if not rel.startswith(root.name + "/"):
                continue
            key = rel[len(root.name) + 1:]
            if key not in rows:
                continue
            if rows[key] != r["_n_rows"] + r["_ragged"]:
                bad += 1
    return bad


if __name__ == "__main__":
    sys.exit(main())
