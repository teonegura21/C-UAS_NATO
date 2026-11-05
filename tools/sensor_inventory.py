#!/usr/bin/env python3
"""Inventory of every sensor file: who reports what, where, and which devices combine several sensors.

Reads the RAW sensor CSVs through tools/cuas_common.py and writes, under data/derived/:

  sensor_inventory.csv  one row per (iteration, sensor_key, team, file): coverage, sensor site,
                        capability flags, measurement kind, class mix, identity evidence
  sensor_sites.csv      per iteration, sensors whose median positions are within 30 m
                        (single linkage) grouped into sites S1..Sn (+ NOSITE rows)
  fusion_links.csv      associatedDetection / derivedDetection references between sensors
  sensor_devices.md     readable report (Q1..Q5); every number is recomputed from the three CSVs

Usage:  python3 tools/sensor_inventory.py
"""

from __future__ import annotations

import base64
import binascii
import csv
import math
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import cuas_common as C  # noqa: E402

SITE_THRESHOLD_M = 30.0
MOBILE_SPREAD_M = 10.0
SENS_THRESHOLDS_M = (10.0, 30.0, 100.0)
ALIAS_MIN_FRAC = 0.90      # share of referenced objectIds owned by one sensor_key to alias an unknown nodeId

OUT_INV = C.DERIVED / "sensor_inventory.csv"
OUT_SITES = C.DERIVED / "sensor_sites.csv"
OUT_LINKS = C.DERIVED / "fusion_links.csv"
OUT_MD = C.DERIVED / "sensor_devices.md"

INV_COLS = [
    "iter_key", "tree", "team", "sensor_id", "sensor_type", "modalities", "sensor_key", "file",
    "rows", "ragged_rows", "t_start", "t_end", "duration_s", "node_ids", "n_objects",
    "sensor_lat", "sensor_lon", "sensor_alt", "sensor_alt_datum", "sensor_pos_valid",
    "sensor_e", "sensor_n", "sensor_u", "pos_spread_m", "frac_gt10m_from_median", "mobile",
    "has_position", "has_altitude", "has_azimuth", "has_elevation", "has_real_range",
    "has_velocity", "has_classification", "has_rcs", "has_frequency", "has_identity",
    "meas_kind", "top_classes",
    # extras (after the contract columns)
    "class_counts", "identity_evidence", "sub_sources", "site_id", "frac_alt_zero",
]
FLAGS = ["has_position", "has_altitude", "has_azimuth", "has_elevation", "has_real_range",
         "has_velocity", "has_classification", "has_rcs", "has_frequency", "has_identity"]
SITE_COLS = ["iter_key", "site_id", "site_lat", "site_lon", "site_e", "site_n", "n_sensor_ids",
             "sensor_keys", "teams", "modalities", "n_modalities", "n_modalities_cross", "multi_modal",
             "n_files", "max_intra_m"]
LINK_COLS = ["iter_key", "fusing_sensor_key", "relation", "referenced_sensor_key",
             "n_reports", "n_distinct_objects", "n_refs", "resolution", "objid_owner_frac"]

# nested-field "type" tags that count as identity evidence (model / serial / protocol / maker ...)
IDENTITY_TYPES = {"model", "serial number", "protocol", "DJI UUID", "droneMac", "droneMaker",
                  "technology", "Drone Event Info", "Remote ID operator ID"}
# nested-field "type" tags whose values are summarised in sub_sources
SUBSOURCE_TYPES = ("SourceCamera", "CameraID", "protocol", "technology", "droneMaker")
_TYPE_RE = re.compile(r"""['"]type['"]:\s*['"]([^'"]+)['"]""")
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_MAC_RE = re.compile(r"^([0-9a-f]{2}:){5}[0-9a-f]{2}$", re.I)
_TRACK_RE = re.compile(r"^ES\.\d+\.\d+$")
_PROTO_RE = re.compile(r"ocusync|dji|sv-|eleron|zala|phantom|mavic|spark|parrot", re.I)
_SERIAL_RE = re.compile(r"^[0-9A-Za-z]{10,24}$")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "" if not math.isfinite(v) else repr(float(v))
    return str(v)


def iso(t: float) -> str:
    return str(np.datetime64(int(round(t * 1e6)), "us")) + "Z"


def id_kind(value: str) -> tuple[str, str]:
    """Classify an `id` cell -> (kind, decoded). Kinds: uuid track mac b64text protocol serial opaque other."""
    if _UUID_RE.match(value):
        return "uuid", ""
    if _TRACK_RE.match(value):
        return "track", ""
    if _MAC_RE.match(value):
        return "mac", ""
    if _PROTO_RE.search(value):
        return "protocol", ""
    if len(value) >= 4 and len(value) % 4 == 0 and re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", value):
        try:
            dec = base64.b64decode(value, validate=True).decode("ascii")
            if dec.isalpha():
                return "b64text", dec
        except (binascii.Error, UnicodeDecodeError, ValueError):
            pass
    if _SERIAL_RE.match(value) and any(ch.isdigit() for ch in value) and any(ch.isalpha() for ch in value) \
            and (value == value.upper() or value == value.lower()):
        return "serial", ""
    if _SERIAL_RE.match(value):
        return "opaque", ""
    return "other", ""


def nested_types(cells: Counter) -> tuple[Counter, dict[str, Counter]]:
    """Over a column of nested-literal cells -> (row count per 'type' tag, value counts of SUBSOURCE_TYPES)."""
    types: Counter = Counter()
    subs: dict[str, Counter] = defaultdict(Counter)
    for cell, cnt in cells.items():
        if not cell:
            continue
        tags = set(_TYPE_RE.findall(cell))
        for tag in tags:
            types[tag] += cnt
        if tags & set(SUBSOURCE_TYPES):
            obj = C.parse_literal(cell)
            if isinstance(obj, list):
                for e in obj:
                    if isinstance(e, dict) and e.get("type") in SUBSOURCE_TYPES and e.get("value") not in (None, ""):
                        subs[e["type"]][str(e["value"])] += cnt
    return types, subs


def fmt_counter(c: Counter, limit: int | None = None) -> str:
    items = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
    if limit:
        items = items[:limit]
    return "|".join(f"{k}:{v}" for k, v in items)


# --------------------------------------------------------------------------- #
# Per-file scan
# --------------------------------------------------------------------------- #


def scan_file(it: C.Iteration, frame: C.Frame, sf: C.SensorFile):
    t = C.read_table(sf.path)
    n, cols = t.n_rows, t.columns
    rec: dict = {
        "iter_key": it.key, "tree": it.tree, "team": sf.team, "sensor_id": sf.sensor_id,
        "sensor_type": sf.sensor_type, "modalities": "|".join(sf.sensor_type.split("--")),
        "sensor_key": sf.sensor_key, "file": str(sf.path.relative_to(C.DATA)),
        "rows": n, "ragged_rows": t.ragged_rows,
    }
    fl = lambda name: C.to_float(cols.get(name), n)  # noqa: E731
    nfin = lambda name: int(np.isfinite(fl(name)).sum()) if name in cols else 0  # noqa: E731

    # --- time / ids -------------------------------------------------------
    if n:
        ts = C.parse_times(cols["timestamp"])
        rec["t_start"], rec["t_end"] = iso(float(ts.min())), iso(float(ts.max()))
        rec["duration_s"] = round(float(ts.max() - ts.min()), 3)
    rec["node_ids"] = "|".join(sorted(set(cols.get("nodeId", [])) - {""}))
    rec["n_objects"] = len(set(cols.get("objectId", [])) - {""})

    # --- sensor position --------------------------------------------------
    rec["sensor_pos_valid"] = 0
    if n:
        cs = cols.get("sensor_location_coordinateSystem") or [""] * n
        lat, lon = C.latlon_deg(fl("sensor_location_x"), fl("sensor_location_y"), cs)
        ok = C.sensor_position_valid(lat, lon)
        if ok.any():
            rec["sensor_pos_valid"] = 1
            lat_m, lon_m = float(np.median(lat[ok])), float(np.median(lon[ok]))
            z = fl("sensor_location_z")[ok]
            z = z[np.isfinite(z)]
            alt_m = float(np.median(z)) if z.size else math.nan
            rec.update(sensor_lat=lat_m, sensor_lon=lon_m, sensor_alt=alt_m)
            datum = Counter(np.asarray(cols.get("sensor_location_datum") or [""] * n, dtype=object)[ok])
            top = datum.most_common(1)
            rec["sensor_alt_datum"] = top[0][0].replace("LOCATION_DATUM_", "") if top and top[0][0] else ""
            pts = np.unique(np.column_stack([lat[ok], lon[ok]]), axis=0)
            h = alt_m if math.isfinite(alt_m) else 0.0
            e, nn, u = frame.to_enu(np.append(pts[:, 0], lat_m), np.append(pts[:, 1], lon_m), h)
            rec.update(sensor_e=float(e[-1]), sensor_n=float(nn[-1]), sensor_u=float(u[-1]),
                       pos_spread_m=float(np.hypot(e[:-1] - e[-1], nn[:-1] - nn[-1]).max()))
            rec["mobile"] = int(rec["pos_spread_m"] > MOBILE_SPREAD_M)
            # share of ROWS (not unique points) more than MOBILE_SPREAD_M from the median position
            er, nr, _ur = frame.to_enu(lat[ok], lon[ok], h)
            rec["frac_gt10m_from_median"] = float(
                (np.hypot(er - e[-1], nr - nn[-1]) > MOBILE_SPREAD_M).mean())

    # --- measurement capabilities ----------------------------------------
    rec["has_position"] = int(nfin("location_x") > 0 and nfin("location_y") > 0)
    # altitude only counts if location_z carries finite NON-ZERO values (all-zero z is a placeholder)
    z_all = fl("location_z")
    z_fin = z_all[np.isfinite(z_all)]
    rec["has_altitude"] = int(z_fin.size > 0 and bool((z_fin != 0).any()))
    if z_fin.size:
        rec["frac_alt_zero"] = float((z_fin == 0).mean())
    rec["has_azimuth"] = int(nfin("rangeBearing_azimuth") > 0)
    rec["has_elevation"] = int(nfin("rangeBearing_elevation") > 0)
    rng = fl("rangeBearing_range")
    rng = rng[np.isfinite(rng)]
    rec["has_real_range"] = int(rng.size > 0 and np.unique(rng).size > 1)

    objinfo, trackinfo = Counter(cols.get("objectInfo", [])), Counter(cols.get("trackInfo", []))
    o_types, o_subs = nested_types(objinfo)
    t_types, t_subs = nested_types(trackinfo)
    types = o_types + t_types
    rec["has_velocity"] = int(nfin("enuVelocity_eastRate") > 0 or "speed" in types)
    rec["has_rcs"] = int("rcs" in types)
    rec["has_frequency"] = int(any("requency" in c for c in set(cols.get("signal", []))))

    # --- classification ---------------------------------------------------
    cls: Counter = Counter()
    for cell, cnt in Counter(cols.get("classification", [])).items():
        typ = C.top_class(cell)[0]
        if typ:
            cls[typ] += cnt
    rec["has_classification"] = int(bool(cls))
    tot = sum(cls.values())
    rec["top_classes"] = "|".join(f"{k}:{v / tot:.2f}" for k, v in
                                  sorted(cls.items(), key=lambda kv: (-kv[1], kv[0]))[:3])
    rec["class_counts"] = fmt_counter(cls)

    # --- identity evidence + sub-sources ---------------------------------
    evidence: list[str] = []
    for src, tys in (("objectInfo", o_types), ("trackInfo", t_types)):
        evidence += [f"{src}:{tag}" for tag in sorted(set(tys) & IDENTITY_TYPES)]
    subs: dict[str, Counter] = defaultdict(Counter)
    for d in (o_subs, t_subs):
        for k, c in d.items():
            subs[k] += c
    kinds: Counter = Counter()
    for val, cnt in Counter(cols.get("id", [])).items():
        if not val:
            continue
        kind, dec = id_kind(val)
        kinds[kind] += cnt
        if kind == "b64text":
            subs["id_b64"][dec] += cnt
    evidence += [f"id:{k}" for k in sorted(kinds) if k in ("serial", "mac", "protocol")]
    rec["identity_evidence"] = "|".join(evidence)
    rec["has_identity"] = int(bool(evidence))
    rec["sub_sources"] = "|".join(f"{k}={'+'.join(f'{v}:{c}' for v, c in sorted(subs[k].items(), key=lambda kv: (-kv[1], kv[0]))[:6])}"
                                  for k in sorted(subs))

    # --- measurement kind --------------------------------------------------
    if rec["has_position"] and rec["has_altitude"]:
        rec["meas_kind"] = "pos3d"
    elif rec["has_position"] and z_fin.size:
        rec["meas_kind"] = "pos2d_effective"    # location_z present but identically 0
    elif rec["has_position"]:
        rec["meas_kind"] = "pos2d"
    elif rec["has_azimuth"] and rec["has_elevation"]:
        rec["meas_kind"] = "bearing_el"
    elif rec["has_azimuth"]:
        rec["meas_kind"] = "bearing"
    else:
        rec["meas_kind"] = "none"

    # --- fusion references (resolved to sensor_key later: needs the global nodeId map) ----
    refs: dict[str, dict[str, dict]] = {}   # relation -> referenced nodeId -> {rows, refs, objs}
    obj_ids = cols.get("objectId", [""] * n)
    for relation in ("associatedDetection", "derivedDetection"):
        cells = cols.get(relation)
        if not cells:
            continue
        cache: dict[str, list] = {}
        acc: dict[str, dict] = {}
        for cell, _oid in zip(cells, obj_ids):
            if not cell:
                continue
            if cell not in cache:
                obj = C.parse_literal(cell)
                cache[cell] = [(e.get("nodeId", ""), e.get("objectId", "")) for e in obj
                               if isinstance(e, dict)] if isinstance(obj, list) else []
            seen = set()
            for nid, oid in cache[cell]:
                a = acc.setdefault(nid, {"rows": 0, "refs": 0, "objs": set()})
                a["refs"] += 1
                a["objs"].add(oid)
                seen.add(nid)
            for nid in seen:
                acc[nid]["rows"] += 1
        if acc:
            refs[relation] = acc
    node_pairs = sorted(set(cols.get("nodeId", [])) - {""})
    return rec, refs, node_pairs, sorted(set(obj_ids) - {""})


def scan_iteration(it: C.Iteration):
    frame = C.iteration_frame(it)
    return it.key, frame, [scan_file(it, frame, sf) for sf in C.sensor_files(it)]


def count_rows_independently(sf: C.SensorFile) -> tuple[int, int]:
    """Row counts without C.read_table (used only for the self-check)."""
    with sf.path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        width = len(next(reader))
        good = bad = 0
        for row in reader:
            if len(row) == 1 and width > 1:     # whole line wrapped in quotes (one CEQX file)
                row = next(csv.reader([row[0]]), row)
            if len(row) == width:
                good += 1
            else:
                bad += 1
    return good, bad


# --------------------------------------------------------------------------- #
# Sites
# --------------------------------------------------------------------------- #


def cluster(points: np.ndarray, threshold: float) -> list[list[int]]:
    """Single-linkage connected components of horizontal distance <= threshold (indices)."""
    m = len(points)
    parent = list(range(m))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(m):
        for j in range(i + 1, m):
            if math.hypot(*(points[i] - points[j])) <= threshold:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(m):
        groups[find(i)].append(i)
    return list(groups.values())


def modality_gain(members: list[dict]) -> tuple[int, int, int, int]:
    """(n distinct sensor ids, |union of modalities|, n_modalities_cross, multi_modal 0/1) for a group of files.

    n_modalities_cross = |union| - (modality count of the largest single sensor id): what the group
    adds beyond its best member. MULTI-MODAL = >= 2 distinct sensor ids AND n_modalities_cross >= 1,
    i.e. the union is strictly larger than the modality set of any single member."""
    per_id: dict[str, set[str]] = defaultdict(set)
    for r in members:
        per_id[r["sensor_id"]] |= set(r["modalities"].split("|"))
    union = set().union(*per_id.values()) if per_id else set()
    cross = len(union) - max((len(v) for v in per_id.values()), default=0)
    return len(per_id), len(union), cross, int(len(per_id) >= 2 and cross >= 1)


def build_sites(inv_rows: list[dict], frames: dict[str, C.Frame]):
    """Assign site_id to each valid-position inventory row; return site rows."""
    site_rows: list[dict] = []
    by_iter: dict[str, list[dict]] = defaultdict(list)
    for r in inv_rows:
        by_iter[r["iter_key"]].append(r)
    for key in sorted(by_iter):
        rows = by_iter[key]
        valid = [r for r in rows if r["sensor_pos_valid"]]
        pts = np.array([[r["sensor_e"], r["sensor_n"]] for r in valid]) if valid else np.zeros((0, 2))
        comps = cluster(pts, SITE_THRESHOLD_M) if valid else []
        comp_info = []
        for idx in comps:
            c = pts[idx].mean(axis=0)
            intra = max((math.hypot(*(pts[a] - pts[b])) for a in idx for b in idx if a < b), default=0.0)
            comp_info.append((round(float(c[1]), 2), round(float(c[0]), 2), idx, c, intra))
        comp_info.sort(key=lambda x: (x[0], x[1], sorted(valid[i]["file"] for i in x[2])))
        for k, (_n, _e, idx, c, intra) in enumerate(comp_info, 1):
            sid = f"S{k}"
            mem = [valid[i] for i in idx]
            for r in mem:
                r["site_id"] = sid
            lat, lon, _h = frames[key].to_geodetic(c[0], c[1], 0.0)
            mods = sorted({m for r in mem for m in r["modalities"].split("|")})
            n_ids, _nu, cross, mm = modality_gain(mem)
            site_rows.append({
                "iter_key": key, "site_id": sid, "site_lat": float(lat), "site_lon": float(lon),
                "site_e": float(c[0]), "site_n": float(c[1]),
                "n_sensor_ids": n_ids,
                "sensor_keys": "|".join(sorted({r["sensor_key"] for r in mem})),
                "teams": "|".join(sorted({r["team"] for r in mem})),
                "modalities": "|".join(mods), "n_modalities": len(mods),
                "n_modalities_cross": cross, "multi_modal": mm,
                "n_files": len(mem), "max_intra_m": intra})
        nosite: dict[tuple, list[dict]] = defaultdict(list)
        for r in rows:
            if not r["sensor_pos_valid"]:
                r["site_id"] = "NOSITE"
                nosite[(r["sensor_key"], r["team"])].append(r)
        for (skey, team), mem in sorted(nosite.items()):
            mods = sorted({m for r in mem for m in r["modalities"].split("|")})
            site_rows.append({
                "iter_key": key, "site_id": "NOSITE", "n_sensor_ids": 1, "sensor_keys": skey,
                "teams": team, "modalities": "|".join(mods), "n_modalities": len(mods),
                "n_modalities_cross": 0, "multi_modal": 0, "n_files": len(mem)})
    return site_rows


# --------------------------------------------------------------------------- #
# Writing tables
# --------------------------------------------------------------------------- #


def write_table(path: Path, order: list[str], rows: list[dict]) -> None:
    cols = {c: [fmt(r.get(c)) for r in rows] for c in order}
    C.write_columns(path, cols, order)


# --------------------------------------------------------------------------- #
# Report (reads the CSVs back: every number comes from them)
# --------------------------------------------------------------------------- #


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x).replace("|", ", ") for x in r) + " |")
    return "\n".join(out)


def short_iter(key: str) -> str:
    tree, rest = key.split("/", 1)
    return ("T " + rest) if tree == "testing" else ("E " + rest.replace("/", " "))


def parse_counts(s: str) -> Counter:
    c: Counter = Counter()
    for part in filter(None, s.split("|")):
        k, v = part.rsplit(":", 1)
        c[k] += int(v)
    return c


def f2(x: float) -> str:
    return f"{round(float(x), 1) + 0.0:.1f}"


def frac(n: int, tot: int) -> str:
    return "<0.01" if 0 < n / tot < 0.005 else f"{n / tot:.2f}"


def build_report() -> str:
    num = {"rows", "ragged_rows", "duration_s", "n_objects", "sensor_lat", "sensor_lon", "sensor_alt",
           "sensor_pos_valid", "sensor_e", "sensor_n", "sensor_u", "pos_spread_m", "frac_gt10m_from_median",
           "frac_alt_zero", "mobile", *FLAGS}
    inv = C.read_columns(OUT_INV, num)
    sites = C.read_columns(OUT_SITES, {"n_sensor_ids", "n_modalities", "n_modalities_cross",
                                       "multi_modal", "n_files", "max_intra_m", "site_e", "site_n"})
    links = C.read_columns(OUT_LINKS, {"n_reports", "n_distinct_objects", "n_refs", "objid_owner_frac"})
    N = len(inv["iter_key"])
    I = [{k: inv[k][i] for k in inv} for i in range(N)]
    iters = sorted({r["iter_key"] for r in I})
    sensor_ids = sorted({r["sensor_id"] for r in I})
    sensor_keys = sorted({r["sensor_key"] for r in I})
    L = []
    w = L.append

    def caps_union(rs):
        return ", ".join(f.replace("has_", "") for f in FLAGS if any(r[f] for r in rs)) or "-"

    def kinds_of(rs):
        return "/".join(sorted({r["meas_kind"] for r in rs}))

    w("# Sensor devices: multi-sensor inventory")
    w("")
    w("Generated by `tools/sensor_inventory.py` from the raw sensor CSVs. Every number below is recomputed "
      "from `sensor_inventory.csv`, `sensor_sites.csv` and `fusion_links.csv` in this folder.")
    w("")
    w(f"- {N} sensor files (byte-identical `(1).csv` duplicates skipped), {len(sensor_keys)} sensor keys "
      f"({len(sensor_ids)} sensor ids), {len(iters)} iterations "
      f"({sum(i.startswith('testing') for i in iters)} testing, {sum(i.startswith('evaluation') for i in iters)} evaluation), "
      f"{int(sum(r['rows'] for r in I))} detection rows, {int(sum(r['ragged_rows'] for r in I))} ragged rows skipped.")
    w(f"- Site rule: single linkage on horizontal distance <= {SITE_THRESHOLD_M:.0f} m between per-file median sensor "
      f"positions, inside one iteration ENU frame. A file with `mobile=1` (position spread > {MOBILE_SPREAD_M:.0f} m) is "
      "represented by its median only.")
    w("- Capability flags (0/1): `position` = location_x/y present; `altitude` = location_z has finite NON-ZERO values (`frac_alt_zero` = share of finite location_z that is exactly 0); `azimuth`/`elevation` "
      "= rangeBearing columns present; `real_range` = rangeBearing_range with more than one distinct value; "
      "`velocity` = enuVelocity or a speed tag; `classification`, `rcs` and `frequency` = field present; "
      "`identity` = model / serial / protocol / maker / MAC evidence in objectInfo, trackInfo or the id column.")
    w("- `meas_kind`: pos3d = position + altitude, pos2d = position only (no location_z), pos2d_effective = position with a "
      "location_z column that is identically 0 (inventory-only label; `detections.csv.gz` keeps its own meas_kind), "
      "bearing_el = azimuth + elevation, bearing = azimuth only.")
    w("- Class fractions are over the rows that carry a classification (`C.top_class`: the entry with the highest own "
      "confidence, else the highest subclass confidence).")
    w("- MULTI-MODAL site: >= 2 distinct sensor ids AND the union of their modalities is strictly larger than the "
      "modality set of any single member (`n_modalities_cross` = union size minus the largest member's modality count; "
      "a site of several identical multi-modal devices is therefore NOT multi-modal).")
    w("- Team is the operator, not the sensor: the same sensor id can appear under several teams.")
    w("")

    # ------------------------------------------------------------------ Q1
    w("## Q1. Multi-modal devices")
    w("")
    multi = sorted({r["sensor_key"] for r in I if "--" in r["sensor_type"]})
    w(f"Sensor keys whose type string combines several modalities: {', '.join(f'`{k}`' for k in multi) or 'none'}.")
    w("")
    rows_q1 = []
    for k in multi:
        rs = [r for r in I if r["sensor_key"] == k]
        ev = sorted({e for r in rs for e in r["identity_evidence"].split("|") if e})
        subs = sorted({s for r in rs for s in r["sub_sources"].split("|") if s})
        cc: Counter = Counter()
        for r in rs:
            cc += parse_counts(r["class_counts"])
        tot = sum(cc.values())
        rows_q1.append([k, rs[0]["modalities"], len({r["iter_key"] for r in rs}), len(rs),
                        int(sum(r["rows"] for r in rs)), kinds_of(rs), caps_union(rs),
                        ", ".join(ev) or "-", "; ".join(subs) or "-",
                        ", ".join(f"{c} {frac(n, tot)}" for c, n in cc.most_common(3)) or "-"])
    w(md_table(["sensor_key", "modalities", "iterations", "files", "rows", "meas_kind", "capabilities",
                "identity evidence", "sub_sources", "classes (fraction of classified rows)"], rows_q1))
    w("")
    w("Reading: each multi-modal key is one reporting node (one nodeId per sensor id) that emits a single stream of "
      "the measurement kind shown; `sub_sources` is empty for them, i.e. no column tells which of the combined "
      "modalities produced a row.")
    w("")
    w("### Single-type devices that still carry several distinct sub-sources")
    w("")
    w("`sub_sources` lists distinct values (with row counts) of camera id, protocol / technology / maker tags and "
      "decoded base64 `id` labels per file, summed per sensor key below.")
    w("")
    rows_ss = []
    for k in sensor_keys:
        rs = [r for r in I if r["sensor_key"] == k and r["sub_sources"]]
        if not rs:
            continue
        agg: dict[str, Counter] = defaultdict(Counter)
        for r in rs:
            for part in r["sub_sources"].split("|"):
                key, vals = part.split("=", 1)
                agg[key] += parse_counts(vals.replace("+", "|"))
        for key in sorted(agg):
            rows_ss.append([k, key, len(agg[key]), ", ".join(f"{v} ({c})" for v, c in agg[key].most_common(6))])
    w(md_table(["sensor_key", "tag", "distinct values", "values (rows)"], rows_ss))
    w("")

    # ------------------------------------------------------------------ Q2
    w("## Q2. Co-located multi-sensor sites per iteration")
    w("")
    S = [{k: sites[k][i] for k in sites} for i in range(len(sites["iter_key"]))]

    def mlabel(r) -> str:
        """sensor_key, annotated when the file's sensor position is not fixed."""
        if r["mobile"]:
            return (f"{r['sensor_key']} (mobile, {r['team']}: spread {f2(r['pos_spread_m'])} m, "
                    f"{100 * r['frac_gt10m_from_median']:.0f}% of rows >{MOBILE_SPREAD_M:.0f} m from median)")
        return r["sensor_key"]

    def site_members(key, sid):
        return [r for r in I if r["iter_key"] == key and r["site_id"] == sid]

    mob = [r for r in I if r["sensor_pos_valid"] and r["mobile"]]
    w(f"### Mobile sensor files (position spread > {MOBILE_SPREAD_M:.0f} m)")
    w("")
    w("A site uses the per-file MEDIAN sensor position. For a mobile file `pos_spread_m` is the largest distance of any "
      f"reported sensor position from that median and `rows >{MOBILE_SPREAD_M:.0f} m` the share of rows farther than "
      f"{MOBILE_SPREAD_M:.0f} m from it; a site containing such a file is only as good as the median.")
    w("")
    if mob:
        w(md_table(["iteration", "sensor", "team", "rows", "pos_spread_m", f"rows >{MOBILE_SPREAD_M:.0f} m from median", "site"],
                   [[short_iter(r["iter_key"]), r["sensor_key"], r["team"], int(r["rows"]), f2(r["pos_spread_m"]),
                     f"{r['frac_gt10m_from_median']:.2f}", r["site_id"]]
                    for r in sorted(mob, key=lambda r: (-r["pos_spread_m"], r["iter_key"], r["sensor_key"]))]))
    else:
        w("none")
    w("")
    for key in iters:
        ss = [s for s in S if s["iter_key"] == key]
        real = [s for s in ss if s["site_id"] != "NOSITE"]
        nos = [s for s in ss if s["site_id"] == "NOSITE"]
        multi_s = [s for s in real if s["n_sensor_ids"] >= 2]
        mm = [s for s in multi_s if s["multi_modal"]]
        w(f"### {key}")
        w("")
        w(f"{len(real)} sites with a valid position, {len(multi_s)} of them hold two or more distinct sensor ids, "
          f"{len(mm)} of them genuinely multi-modal (the union of modalities exceeds that of any single member); {len(nos)} sensor/team entries have no site position"
          + (f" ({', '.join(f'{s['sensor_keys']} ({s['teams']})' for s in nos)})." if nos else "."))
        w("")
        if multi_s:
            w(md_table(["site", "east m", "north m", "sensor ids", "sensors", "teams", "modalities",
                        "n_modalities_cross", "max intra m", "flag"],
                       [[s["site_id"], f2(s["site_e"]), f2(s["site_n"]), int(s["n_sensor_ids"]),
                         "; ".join(sorted({mlabel(r) for r in site_members(key, s["site_id"])})),
                         s["teams"], s["modalities"], int(s["n_modalities_cross"]), f2(s["max_intra_m"]),
                         "MULTI-MODAL" if s["multi_modal"] else "no modality gain"] for s in multi_s]))
            w("")
        singles = [s for s in real if s["n_sensor_ids"] < 2]
        if singles:
            w("Single-sensor sites: " + "; ".join(
                f"{s['site_id']} {'; '.join(sorted({mlabel(r) for r in site_members(key, s['site_id'])}))} "
                f"({s['teams']}, {f2(s['site_e'])} E / {f2(s['site_n'])} N)"
                for s in singles) + ".")
            w("")

    # threshold sensitivity, recomputed from the inventory's sensor_e / sensor_n
    w("### Threshold sensitivity (recomputed from sensor_inventory.csv)")
    w("")
    w(f"Units are the inventory files with a valid position. 'max intra' is the largest pairwise distance inside "
      f"one group (single linkage can chain beyond the threshold); 'min inter' is the smallest distance between two "
      f"different groups. Max intra-site distance at the chosen {SITE_THRESHOLD_M:.0f} m threshold, over all iterations: "
      f"{max(s['max_intra_m'] for s in S if s['site_id'] != 'NOSITE'):.1f} m.")
    w("")
    groups_by: dict[tuple[str, float], list[tuple[str, ...]]] = {}
    sens_rows = []
    for key in iters:
        units = [r for r in I if r["iter_key"] == key and r["sensor_pos_valid"]]
        pts = np.array([[r["sensor_e"], r["sensor_n"]] for r in units])
        row = [key, len(units)]
        for th in SENS_THRESHOLDS_M:
            comps = cluster(pts, th)
            labels = [tuple(sorted({f"{units[i]['team']}/{units[i]['sensor_key']}" for i in c})) for c in comps]
            groups_by[(key, th)] = labels
            multi_sensor = sum(len({units[i]["sensor_id"] for i in c}) >= 2 for c in comps)
            multi_mod = sum(modality_gain([units[i] for i in c])[3] for c in comps)
            intra = max((float(np.hypot(*(pts[a] - pts[b]))) for c in comps for a in c for b in c if a < b), default=0.0)
            inter = min((float(np.hypot(*(pts[a] - pts[b]))) for ci, c in enumerate(comps) for cj, d in enumerate(comps)
                         if ci < cj for a in c for b in d), default=math.nan)
            row += [len(comps), multi_sensor, multi_mod, f2(intra), "-" if math.isnan(inter) else f2(inter)]
        sens_rows.append(row)
    hdr = ["iteration", "units"]
    for th in SENS_THRESHOLDS_M:
        t = f"{th:.0f} m"
        hdr += [f"sites @{t}", f"multi-sensor @{t}", f"multi-modal @{t}", f"max intra @{t}", f"min inter @{t}"]
    w(md_table(hdr, sens_rows))
    w("")
    for lo, hi in ((SENS_THRESHOLDS_M[0], SENS_THRESHOLDS_M[1]), (SENS_THRESHOLDS_M[1], SENS_THRESHOLDS_M[2])):
        w(f"Groups that exist at {hi:.0f} m but not at {lo:.0f} m (each is a merge of smaller groups; "
          f"unchanged groups omitted):")
        w("")
        any_change = False
        for key in iters:
            a, b = set(groups_by[(key, lo)]), set(groups_by[(key, hi)])
            if a != b:
                any_change = True
                w(f"- {key}: " + "; ".join("{" + ", ".join(g) + "}" for g in sorted(b - a)))
        if not any_change:
            w("- no iteration changes")
        w("")

    # ------------------------------------------------------------------ Q3
    w("## Q3. Fusion nodes (outputs that reference other detections)")
    w("")
    LK = [{k: links[k][i] for k in links} for i in range(len(links["iter_key"]))]
    if not LK:
        w("No associatedDetection / derivedDetection references found.")
    else:
        def base_key(l) -> str:
            return l["referenced_sensor_key"].split(" (alias", 1)[0]

        def link_type(l) -> str:
            if l["resolution"] == "unknown":
                return "UNKNOWN"
            return ("self" if base_key(l) == l["fusing_sensor_key"] else "other sensor") + (
                " (via objectId alias)" if l["resolution"] == "objectId_alias" else "")

        w("`n_reports` = rows of the fusing sensor holding at least one reference to the referenced sensor; "
          "`n_refs` = total references; `n_distinct_objects` = distinct referenced objectIds; "
          "`owner share` = fraction of those objectIds that belong to the referenced sensor key in the same iteration. "
          "`resolution`: `nodeId` = the referenced nodeId is reported by a sensor in the data; `objectId_alias` = the "
          f"nodeId is absent from the data but >= {100 * ALIAS_MIN_FRAC:.0f} % of the referenced objectIds belong to one "
          "sensor key of the same iteration, so the nodeId is treated as an alias of that sensor's node; `unknown` = neither.")
        w("")
        w(md_table(["iteration", "fusing sensor", "relation", "referenced sensor", "resolution", "type", "n_reports",
                    "n_refs", "n_distinct_objects", "owner share"],
                   [[short_iter(l["iter_key"]), l["fusing_sensor_key"], l["relation"], l["referenced_sensor_key"],
                     l["resolution"], link_type(l), int(l["n_reports"]), int(l["n_refs"]), int(l["n_distinct_objects"]),
                     "-" if math.isnan(l["objid_owner_frac"]) else f"{l['objid_owner_frac']:.2f}"] for l in LK]))
        w("")
        cross = [l for l in LK if l["resolution"] != "unknown" and base_key(l) != l["fusing_sensor_key"]]
        unk = [l for l in LK if l["resolution"] == "unknown"]
        alias = [l for l in LK if l["resolution"] == "objectId_alias"]
        w(f"{len(LK)} link rows: {len(LK) - len(cross) - len(unk)} reference the fusing sensor's own objects "
          f"({sum(l['resolution'] == 'nodeId' and base_key(l) == l['fusing_sensor_key'] for l in LK)} by a known nodeId, "
          f"{sum(base_key(l) == l['fusing_sensor_key'] for l in alias)} through an alias nodeId), "
          f"{len(cross)} reference another sensor present in the data, {len(unk)} reference a node that could not be resolved.")
        w("")
        if cross:
            w("Cross-sensor references exist: " + "; ".join(
                f"{short_iter(l['iter_key'])} {l['fusing_sensor_key']} -> {l['referenced_sensor_key']}" for l in cross) + ".")
        else:
            fusers = sorted({l["fusing_sensor_key"] for l in LK})
            w(f"Conclusion: no cross-sensor fusion exists in the data. The only sensors that emit references ({', '.join(fusers)}) "
              "reference their own objects, "
              + ("including STUG, whose associatedDetection nodeId is absent from the data but resolves to STUG's own "
                 "objectIds; " if any(l["fusing_sensor_key"].startswith("STUG") and l["resolution"] == "objectId_alias" for l in LK) else "")
              + "so no sensor's output can be taken as a pre-fused track of other sensors.")
        weak = [l for l in LK if l["resolution"] == "nodeId" and l["objid_owner_frac"] < ALIAS_MIN_FRAC]
        if weak:
            w("")
            w("Self references by nodeId whose referenced objectIds are not present in the data of that iteration "
              "(owner share below the alias threshold): " + "; ".join(
                  f"{short_iter(l['iter_key'])} {l['fusing_sensor_key']} ({int(l['n_distinct_objects'])} objectIds, "
                  f"owner share {l['objid_owner_frac']:.2f})" for l in weak) + ".")
        w("")
    w("Implicit relays (not referenced via associatedDetection): sensor keys whose `id` column carries decoded base64 feed labels:")
    w("")
    relays = [r for r in rows_ss if r[1] == "id_b64"]
    w(md_table(["sensor_key", "feed labels (rows)"], [[r[0], r[3]] for r in relays]) if relays else "none found")
    w("")

    # ------------------------------------------------------------------ Q4
    w("## Q4. Per sensor id across all iterations")
    w("")
    rows_q4 = []
    for sid in sensor_ids:
        rs = [r for r in I if r["sensor_id"] == sid]
        cc: Counter = Counter()
        for r in rs:
            cc += parse_counts(r["class_counts"])
        tot = sum(cc.values())
        mob = sum(r["mobile"] for r in rs if r["sensor_pos_valid"])
        npos = sum(r["sensor_pos_valid"] for r in rs)
        rows_q4.append([sid, "/".join(sorted({r["sensor_type"] for r in rs})), ", ".join(sorted({r["team"] for r in rs})),
                        len({r["iter_key"] for r in rs}), len(rs), int(sum(r["rows"] for r in rs)),
                        int(sum(r["n_objects"] for r in rs)), f"{int(mob)}/{int(npos)}",
                        kinds_of(rs), caps_union(rs),
                        ", ".join(f"{c} {frac(n, tot)}" for c, n in cc.most_common(4)) or "-"])
    w(md_table(["sensor id", "type", "teams", "iterations", "files", "rows", "objects (sum of per-file distinct)",
                "mobile files / files with position", "meas_kind", "capabilities", "classes (fraction of classified rows)"], rows_q4))
    w("")

    # ------------------------------------------------------------------ Q5
    w("## Q5. Presence matrix (rows per sensor key; `.` = absent)")
    w("")
    w("Columns: " + "; ".join(f"`{short_iter(k)}` = {k}" for k in iters))
    w("")
    rows_q5 = []
    for k in sensor_keys:
        row = [k]
        for it in iters:
            rs = [r for r in I if r["sensor_key"] == k and r["iter_key"] == it]
            row.append(int(sum(r["rows"] for r in rs)) if rs else ".")
        rows_q5.append(row)
    w(md_table(["sensor key"] + [short_iter(k) for k in iters], rows_q5))
    w("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    t0 = time.time()
    its = C.iterations()
    with ProcessPoolExecutor(max_workers=min(len(its), 12)) as pool:
        results = list(pool.map(scan_iteration, its))

    frames = {key: fr for key, fr, _ in results}
    inv_rows: list[dict] = []
    raw_refs: list[tuple[str, str, dict]] = []
    node_map: dict[str, set[str]] = defaultdict(set)
    obj_owner: dict[tuple[str, str], set[str]] = defaultdict(set)     # (iteration, objectId) -> sensor_keys
    for key, _fr, recs in results:
        for rec, refs, nodes, objs in recs:
            inv_rows.append(rec)
            for nid in nodes:
                node_map[nid].add(rec["sensor_key"])
            for oid in objs:
                obj_owner[(key, oid)].add(rec["sensor_key"])
            if refs:
                raw_refs.append((key, rec["sensor_key"], refs))
    inv_rows.sort(key=lambda r: (r["iter_key"], r["team"], r["sensor_id"], r["file"]))

    site_rows = build_sites(inv_rows, frames)

    # fusion links: resolve nodeIds over ALL data; an unknown nodeId is resolved by objectId ownership
    # (the sensor_key owning >= ALIAS_MIN_FRAC of the referenced objectIds in the same iteration)
    def owner_share(key: str, objs: set[str]) -> Counter:
        c: Counter = Counter()
        for oid in objs:
            for sk in obj_owner.get((key, oid), ()):
                c[sk] += 1
        return c

    agg: dict[tuple, dict] = {}
    for key, fkey, refs in raw_refs:
        for relation, per_node in refs.items():
            for nid, a in per_node.items():
                keys = node_map.get(nid)
                if keys:
                    rkey, how = "|".join(sorted(keys)), "nodeId"
                else:
                    objs = {o for o in a["objs"] if o}
                    best = owner_share(key, objs).most_common()
                    best = sorted(best, key=lambda kv: (-kv[1], kv[0]))[:1]
                    if objs and best and best[0][1] / len(objs) >= ALIAS_MIN_FRAC:
                        rkey, how = f"{best[0][0]} (alias node {nid[:8]})", "objectId_alias"
                    else:
                        rkey, how = f"UNKNOWN:{nid[:8]}", "unknown"
                g = agg.setdefault((key, fkey, relation, rkey, how), {"rows": 0, "refs": 0, "objs": set()})
                g["rows"] += a["rows"]
                g["refs"] += a["refs"]
                g["objs"] |= a["objs"]
    link_rows = []
    for k, g in sorted(agg.items()):
        objs = {o for o in g["objs"] if o}
        base = k[3].split(" (alias", 1)[0]
        own = owner_share(k[0], objs)
        share = (own[base] / len(objs)) if objs and "|" not in base and not base.startswith("UNKNOWN") else (
            (own.most_common(1)[0][1] / len(objs)) if objs and own else math.nan)
        link_rows.append({"iter_key": k[0], "fusing_sensor_key": k[1], "relation": k[2],
                          "referenced_sensor_key": k[3], "n_reports": g["rows"],
                          "n_distinct_objects": len(g["objs"]), "n_refs": g["refs"],
                          "resolution": k[4], "objid_owner_frac": round(share, 4)})

    write_table(OUT_INV, INV_COLS, inv_rows)
    write_table(OUT_SITES, SITE_COLS, site_rows)
    write_table(OUT_LINKS, LINK_COLS, link_rows)
    OUT_MD.write_text(build_report(), encoding="utf-8")

    # ------------------------------------------------------------------ self-check
    print(f"sensor_inventory.csv : {len(inv_rows)} rows")
    print(f"sensor_sites.csv     : {len(site_rows)} rows")
    print(f"fusion_links.csv     : {len(link_rows)} rows")
    print(f"sensor_devices.md    : {len(OUT_MD.read_text().splitlines())} lines")
    ok = True
    exp_good = exp_bad = exp_files = 0
    for it in its:
        for sf in C.sensor_files(it):
            g, b = count_rows_independently(sf)
            exp_good += g
            exp_bad += b
            exp_files += 1
    inv_back = C.read_columns(OUT_INV, {"rows", "ragged_rows", "sensor_pos_valid"})
    got_good, got_bad = int(inv_back["rows"].sum()), int(inv_back["ragged_rows"].sum())
    c1 = (got_good == exp_good and got_bad == exp_bad and len(inv_rows) == exp_files)
    print(f"[check] inventory files={len(inv_rows)} rows={got_good} ragged={got_bad} vs "
          f"sensor_files files={exp_files} non-ragged={exp_good} ragged={exp_bad}: {'OK' if c1 else 'FAIL'}")
    ok &= c1
    sites_back = C.read_columns(OUT_SITES, {"n_files"})
    site_index = {(sites_back["iter_key"][i], sites_back["site_id"][i]): i for i in range(len(sites_back["iter_key"]))}
    bad = 0
    n_valid = 0
    inv_full = C.read_columns(OUT_INV, {"sensor_pos_valid"})
    for i in range(len(inv_full["iter_key"])):
        if inv_full["sensor_pos_valid"][i]:
            n_valid += 1
            j = site_index.get((inv_full["iter_key"][i], inv_full["site_id"][i]))
            if (j is None or inv_full["site_id"][i] == "NOSITE"
                    or inv_full["sensor_key"][i] not in sites_back["sensor_keys"][j].split("|")
                    or inv_full["team"][i] not in sites_back["teams"][j].split("|")):
                bad += 1
    c2 = bad == 0
    print(f"[check] valid-position sensor files={n_valid}, without a matching site row={bad}: {'OK' if c2 else 'FAIL'}")
    ok &= c2
    multi_node = {n[:8]: sorted(v) for n, v in node_map.items() if len(v) > 1}
    c3 = not multi_node
    print(f"[check] nodeId -> sensor_key map: {len(node_map)} nodeIds, non-1:1={len(multi_node)}: {'OK' if c3 else 'FAIL'}")
    ok &= c3
    print(f"[check] runtime {time.time() - t0:.1f}s: {'ALL OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
