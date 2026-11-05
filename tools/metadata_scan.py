#!/usr/bin/env python3
"""Which sensors report the four Fusion Completeness metadata fields, with what values, and
which values belong to which Dronetag drone (testing).

Fields (C.METADATA_COLUMNS): uav_model/manufacturer, protocol, serial_number,
signal_frequency_band. Raw value sources, all taken from data/derived/detections.csv.gz
(faithful copy of the sensor CSVs; check 3 re-reads the raw CSVs independently):

* model       `model` column (STUG objectInfo model, CEQX Drone Event Info factory_name) and the
              GTYT objectInfo `droneMaker` ('Parrot'), which only exists in the `info` JSON.
* serial      `serial` column (objectInfo / trackInfo serial number, or the `id` column of
              CEQX / ILYS / STUG / IKJX / MMJJ when it looks like a serial); upper-cased.
* protocol    `protocol` column (objectInfo protocol / technology, or the `id` column text of
              ORFH / YXAB, or non-serial `id` text of the serial sensors).
* band        `freq_mhz` = centreFrequency of the strongest `signal` entry (OEPS, GTYT only).

metadata_values.csv `raw_value` is the value after build_detections' normalisation (whitespace
collapsed, serials upper-cased, GTYT droneMaker injected into the model field); `raw_spellings`
lists the spellings found in the raw cells whenever they differ (CEQX writes lower-case serials).

Mapping raw -> dropdown lives only in tools/metadata_map.py. This script never changes
tools/drone_catalog.csv; the report lists PROPOSED catalog values.

Outputs (data/derived/):
  metadata_values.csv         (tree, sensor_key, field, raw_value) -> counts, mapped, rule, near_match
  drone_metadata_testing.csv  per (iteration, flight) and per drone x field: mapped values on
                              Dronetag-matched detections, counts, sensors, purity
  friendly_whitelist.csv      evaluation Friendly(Blue)UAS.xlsx rows with mapped serial / model
  metadata_report.md          synthesis (coverage, proposals, evaluation values, unmatched objects)

Evidence levels for the drone tables (drone-class rows only: Home / UAV Transmitter / EM Transmitter
rows sit at the pilot's position, which the matcher pairs with whatever took off nearby):
  det    detections with matched == 1 and ambiguous == 0 in det_gt_match.csv.gz (the strict rule)
  obj    every row of an (iteration, sensor, object) track that track_labels.csv labels
         drone_air or drone_ground (its best_flight is the flight)
  ser    rows whose object carries a serial that >= 3 det rows (>= 80 %) tie to one flight: the
         serial is the device identity, so the object's other fields inherit the flight
  union  per row: det, else ser, else obj (used for the PROPOSED catalog values and coverage B)
Only airborne evidence counts (the flight's GT height > AIR_H_M at the record's second): a drone
parked on the ground gets matched to whatever sits near it, so ground rows are tallied separately.
Testing tree only for anything that touches ground truth; evaluation rows are only counted.

Run:  python3 tools/metadata_scan.py   (exits non-zero if a self-check fails)
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_detections as BD  # noqa: E402  (norm_ws, looks_like_serial, id-column sensor sets)
import cuas_clean as Q  # noqa: E402
import cuas_common as C  # noqa: E402
import metadata_map as MM  # noqa: E402

OUT = C.DERIVED
FIELDS = list(C.METADATA_COLUMNS)
MODEL, PROTOCOL, SERIAL, BAND = FIELDS
SERIAL_NEAR = "serial_number_near"     # drone tables only: strict serial, else the near-match
AIR_H_M = 5.0                          # same airborne rule as tools/gt_associate.py
HOLD_S = 5                             # 'held' coverage: a detection also covers +-HOLD_S seconds
PROPOSE_MIN_PURITY = 0.8
OBJ_LABELS = ("drone_air", "drone_ground")
CHECKS: list[tuple[bool, str]] = []


def log(msg: str) -> None:
    print(msg, flush=True)


def check(ok: bool, msg: str) -> None:
    CHECKS.append((bool(ok), msg))
    log(f"self-check {len(CHECKS)}: {'PASS' if ok else 'FAIL'}: {msg}")


# --------------------------------------------------------------------------- #
# 1. raw values per detection row
# --------------------------------------------------------------------------- #

DET_COLS = ["iter_key", "tree", "file", "row", "sensor_key", "object_id", "t", "model", "serial",
            "protocol", "freq_mhz", "cls", "cls_sub", "info"]


def fmt_freq(f: float) -> str:
    """Centre frequency in MHz as a short canonical string ('5745', '2473.5')."""
    return ("%.3f" % f).rstrip("0").rstrip(".")


def raw_fields(det: dict) -> dict[str, np.ndarray]:
    """{field: object array of raw strings, '' where the row carries nothing}."""
    n = len(det["t"])
    model = np.array(det["model"], dtype=object)
    info = det["info"]
    has_maker = np.array(["droneMaker" in s for s in info], dtype=bool)
    for i in np.where(has_maker & (model == ""))[0]:
        maker = BD.norm_ws(json.loads(info[i]).get("droneMaker", ""))
        model[i] = maker
    band = np.full(n, "", dtype=object)
    fin = np.isfinite(det["freq_mhz"])
    band[fin] = [fmt_freq(f) for f in det["freq_mhz"][fin]]
    return {MODEL: model, PROTOCOL: np.array(det["protocol"], dtype=object),
            SERIAL: np.array(det["serial"], dtype=object), BAND: band}


# --------------------------------------------------------------------------- #
# 2. metadata_values.csv
# --------------------------------------------------------------------------- #

def build_values(det: dict, raw: dict[str, np.ndarray]) -> tuple[list[dict], dict]:
    """One row per (tree, sensor_key, field, raw_value). Returns (rows, source_totals)."""
    acc: dict[tuple, dict] = {}
    totals: Counter = Counter()
    for f in FIELDS:
        for i in np.where(raw[f] != "")[0]:
            key = (det["tree"][i], det["sensor_key"][i], f, raw[f][i])
            a = acc.get(key)
            if a is None:
                a = acc[key] = {"n": 0, "obj": set(), "it": set()}
            a["n"] += 1
            a["obj"].add((det["iter_key"][i], det["sensor_key"][i], det["object_id"][i]))
            a["it"].add(det["iter_key"][i])
    rows = []
    for (tree, sk, f, rv), a in sorted(acc.items()):
        mapped, rule = MM.map_with_rule(f, rv)
        rows.append({"tree": tree, "sensor_key": sk, "field": f, "raw_value": rv,
                     "n_rows": a["n"], "n_objects": len(a["obj"]), "n_iterations": len(a["it"]),
                     "mapped": mapped, "rule": rule,
                     "near_match": MM.serial_near(rv) if f == SERIAL else ""})
    for f in FIELDS:
        m = raw[f] != ""
        for tree, sk in set(zip(det["tree"][m], det["sensor_key"][m])):
            totals[(tree, sk, f)] = int(((det["tree"] == tree) & (det["sensor_key"] == sk) & m).sum())
    return rows, dict(totals)


# --------------------------------------------------------------------------- #
# 3. ground-truth association (testing)
# --------------------------------------------------------------------------- #

def load_match() -> dict[tuple, tuple]:
    """(iter_key, file, row) -> (best_flight, drone_name, side, t_cal, matched, ambiguous, gt_airborne)."""
    m = C.read_columns(OUT / "det_gt_match.csv.gz", {"row", "t_cal", "matched", "ambiguous", "gt_airborne"},
                       usecols=["iter_key", "file", "row", "best_flight", "drone_name", "side",
                                "t_cal", "matched", "ambiguous", "gt_airborne"])
    return {(a, b, int(c)): (f, dn, sd, tc, int(mt), int(am), int(ga))
            for a, b, c, f, dn, sd, tc, mt, am, ga in zip(
                m["iter_key"], m["file"], m["row"], m["best_flight"], m["drone_name"], m["side"],
                m["t_cal"], m["matched"], m["ambiguous"], m["gt_airborne"])}


def load_obj_labels() -> dict[tuple, tuple]:
    """(iter_key, sensor_key, object_id) -> (flight_key, drone_name, side) for drone_air / drone_ground."""
    out = {}
    with (OUT / "track_labels.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["label"] in OBJ_LABELS and r["best_flight"]:
                out[(r["iter_key"], r["sensor_key"], r["object_id"])] = (
                    r["best_flight"], r["drone_name"], r["side"])
    return out


NON_DRONE_CLASSES = frozenset({"Home", "UAV Transmitter", "EM Transmitter"})
SER_MIN_ROWS, SER_MIN_SHARE = 3, 0.8


def metadata_rows(det, raw) -> list[dict]:
    """One record per detection row (both trees) that carries any of the four fields."""
    meta = np.zeros(len(det["t"]), dtype=bool)
    for f in FIELDS:
        meta |= raw[f] != ""
    out = []
    for i in np.where(meta)[0]:
        rawd = {f: raw[f][i] for f in FIELDS}
        mapped = {f: (MM.map_value(f, rawd[f]) if rawd[f] != "" else None) for f in FIELDS}
        sid = (MM.resolve_serial(rawd[SERIAL]) or rawd[SERIAL]) if rawd[SERIAL] else ""
        mapped[SERIAL_NEAR] = (mapped[SERIAL] or MM.serial_near(rawd[SERIAL])) if rawd[SERIAL] else None
        if mapped[PROTOCOL] is None and MM.protocol_from_model(rawd[MODEL]):
            mapped[PROTOCOL] = MM.protocol_from_model(rawd[MODEL])
            rawd[PROTOCOL] = rawd[MODEL]           # protocol named in the model slot (STUG DIY CROSSFIRE)
        out.append({"i": int(i), "tree": det["tree"][i], "iter_key": det["iter_key"][i],
                    "file": det["file"][i], "row": int(det["row"][i]), "t": float(det["t"][i]),
                    "sensor_key": det["sensor_key"][i], "object": det["object_id"][i],
                    "cls": det["cls"][i], "cls_sub": det["cls_sub"][i], "raw": rawd,
                    "mapped": mapped, "sid": sid,
                    "drone_class": det["cls"][i] not in NON_DRONE_CLASSES})
    return out


def drone_observations(mrows, match, objlab, air) -> tuple[list[dict], dict]:
    """Evidence records (testing, drone-class rows only) at four levels.

    det    matched == 1 and ambiguous == 0 in det_gt_match.csv.gz
    obj    the row's (iteration, sensor, object) is labelled drone_air / drone_ground in track_labels.csv
    ser    the row's object carries a serial whose drone-class det rows (same iteration) vote for
           one flight (>= SER_MIN_ROWS rows, >= SER_MIN_SHARE of them): the serial is the device
           identity, so its other fields / other objects inherit the flight
    union  one record per row: det, else ser, else obj
    Only AIRBORNE evidence is kept: a record counts when its second is one of the flight's airborne
    seconds (`air`, GT height > AIR_H_M). A drone parked on the ground is matched to whatever sits
    near it (S4-I4 'Autel Evo 2' <- a Mavic 2 serial), so ground rows say nothing about the drone's
    identity; they are tallied in info['ground_excluded'] instead.
    Home / UAV Transmitter / EM Transmitter rows are skipped: their position is the pilot's, which
    the matcher happily pairs with a drone that took off nearby (S2-I2 'Anafi' <- Autel serial).
    Returns (records, info) with info = {'serial_flight': {(iter, sid): (flight, n, share)},
    'skipped_matched': n non-drone-class rows that were matched,
    'ground_excluded': {(iter, drone_name, side): rows whose every candidate flight second is on the ground}}."""
    tst = [r for r in mrows if r["tree"] == "testing"]
    recs: list[dict] = []
    skipped = 0
    det_by_row: dict = {}
    det_ground: dict = {}                      # row index -> (iter, name, side) of a ground-only strict match

    def airborne(it, fk, sec):
        return sec in air.get((it, fk), ())

    for r in tst:
        mt = match.get((r["iter_key"], r["file"], r["row"]))
        if not (mt and mt[4] == 1 and mt[5] == 0 and mt[0]):
            continue
        if not r["drone_class"]:
            skipped += 1
            continue
        rec = {**r, "level": "det", "flight_key": mt[0], "drone_name": mt[1],
               "side": mt[2], "second": int(round(mt[3]))}
        if mt[6] == 1 and airborne(r["iter_key"], mt[0], rec["second"]):   # matcher flag AND air seconds
            det_by_row[r["i"]] = rec
        else:
            det_ground[r["i"]] = (r["iter_key"], mt[1], mt[2])
    votes: dict = defaultdict(Counter)
    for rec in det_by_row.values():
        if rec["sid"]:
            votes[(rec["iter_key"], rec["sid"])][(rec["flight_key"], rec["drone_name"], rec["side"])] += 1
    serial_flight = {}
    for k, c in votes.items():
        (fl, n), = c.most_common(1)
        tot = sum(c.values())
        if n >= SER_MIN_ROWS and n / tot >= SER_MIN_SHARE:
            serial_flight[k] = (fl, n, n / tot)
    obj_sids: dict = defaultdict(set)
    for r in tst:
        if r["sid"]:
            obj_sids[(r["iter_key"], r["sensor_key"], r["object"])].add(r["sid"])
    union: dict = {}
    ground_excl: Counter = Counter()
    for r in tst:
        if not r["drone_class"]:
            continue
        i = r["i"]
        det_rec = det_by_row.get(i)
        gc = det_ground.get(i)                 # identity of a dropped (ground) candidate
        if det_rec:
            recs.append(det_rec)
        flights = {serial_flight[(r["iter_key"], s)][0] for s in obj_sids.get((r["iter_key"], r["sensor_key"], r["object"]), ())
                   if (r["iter_key"], s) in serial_flight}
        ser_rec = None
        if len(flights) == 1:
            fl = next(iter(flights))
            ser_rec = {**r, "level": "ser", "flight_key": fl[0], "drone_name": fl[1], "side": fl[2],
                       "second": int(round(r["t"]))}
            if airborne(r["iter_key"], fl[0], ser_rec["second"]):
                recs.append(ser_rec)
            else:
                ser_rec = None
                gc = gc or (r["iter_key"], fl[1], fl[2])
        ol = objlab.get((r["iter_key"], r["sensor_key"], r["object"]))
        obj_rec = None
        if ol:
            obj_rec = {**r, "level": "obj", "flight_key": ol[0], "drone_name": ol[1], "side": ol[2], "second": int(round(r["t"]))}
            if airborne(r["iter_key"], ol[0], obj_rec["second"]):
                recs.append(obj_rec)
            else:
                obj_rec = None
                gc = gc or (r["iter_key"], ol[1], ol[2])
        pick = det_rec or ser_rec or obj_rec
        if pick:
            union[i] = {**pick, "level": "union"}
        elif gc:
            ground_excl[gc] += 1
    recs += [union[i] for i in sorted(union)]
    return recs, {"serial_flight": serial_flight, "skipped_matched": skipped,
                  "ground_excluded": dict(ground_excl)}


def group_rows(recs: list[dict], scope: str) -> list[dict]:
    """Aggregate observations to value rows. scope: flight | drone | drone_name."""
    def gkey(r):
        if scope == "flight":
            return (r["iter_key"], r["flight_key"], r["drone_name"], r["side"])
        if scope == "drone":
            return ("ALL", "", r["drone_name"], r["side"])
        return ("ALL", "", r["drone_name"], "")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        groups[(r["level"],) + gkey(r)].append(r)
    out = []
    for gk in sorted(groups):
        level, iter_key, flight_key, drone_name, side = gk
        rs = groups[gk]
        sides = "|".join(sorted({r["side"] for r in rs})) if scope == "drone_name" else side
        n_flights = len({(r["iter_key"], r["flight_key"]) for r in rs})
        for f in FIELDS + [SERIAL_NEAR]:
            cnt: Counter = Counter()
            objs: dict = defaultdict(set)
            sens: dict = defaultdict(Counter)
            rawv: dict = defaultdict(Counter)
            for r in rs:
                v = r["mapped"].get(f)
                if v is None:
                    continue
                cnt[v] += 1
                objs[v].add((r["iter_key"], r["sensor_key"], r["object"]))
                sens[v][r["sensor_key"]] += 1
                rawv[v][r["raw"].get(SERIAL if f == SERIAL_NEAR else f, "")] += 1
            n_map = sum(c for v, c in cnt.items() if v)
            top = max((c for v, c in cnt.items() if v), default=0)
            for v in sorted(cnt, key=lambda x: (-cnt[x], x)):
                out.append({
                    "scope": scope, "level": level, "iter_key": iter_key, "flight_key": flight_key,
                    "drone_name": drone_name, "side": sides, "n_flights": n_flights, "field": f,
                    "value": v, "n_rows": cnt[v], "n_objects": len(objs[v]),
                    "n_sensors": len(sens[v]),
                    "sensors": "|".join(f"{k.split('_')[0]}:{c}" for k, c in sorted(sens[v].items())),
                    "share": round(cnt[v] / n_map, 4) if v and n_map else "",
                    "purity": round(top / n_map, 4) if n_map else "",
                    "raw_values": "|".join(f"{k}:{c}" for k, c in rawv[v].most_common(6)),
                })
    return out


def airborne_seconds(gt: dict) -> dict[tuple, set]:
    """(iter_key, flight_key) -> set of integer seconds with GT height > AIR_H_M."""
    out = {}
    for ik, fk in sorted(set(zip(gt["iter_key"].tolist(), gt["flight_key"].tolist()))):
        m = (gt["iter_key"] == ik) & (gt["flight_key"] == fk)
        sec = np.unique(np.round(gt["t"][m]).astype(np.int64))
        h = Q.interp_flight(gt, m, sec.astype(float), cols=("height",))[0]
        out[(ik, fk)] = {int(s) for s, hh in zip(sec.tolist(), h.tolist()) if hh > AIR_H_M}
    return out


def coverage(recs: list[dict], air: dict, flights: dict, level: str) -> list[dict]:
    """Per flight: share of its airborne seconds covered by >= 1 `level` metadata record (any
    field, then per field with a dropdown value). `held` also counts +-HOLD_S seconds."""
    secs: dict = defaultdict(lambda: defaultdict(set))   # (it, fk) -> tag -> seconds
    sens: dict = defaultdict(Counter)
    for r in recs:
        if r["level"] != level:
            continue
        key = (r["iter_key"], r["flight_key"])
        sens[key][r["sensor_key"].split("_")[0]] += 1
        tags = [f for f in FIELDS if r["mapped"].get(f)]
        for tg in (["any"] + tags) if tags else ():       # 'any' needs >= 1 dropdown value
            secs[key][tg].add(r["second"])
    rows = []
    for key in sorted(air):
        a = air[key]
        rec = {"iter_key": key[0], "flight_key": key[1], **flights[key], "airborne_s": len(a)}
        if a:
            for tg in ["any"] + FIELDS:
                s_ = secs[key].get(tg, set())
                held = {x + d for x in s_ for d in range(-HOLD_S, HOLD_S + 1)}
                rec[f"cov_{tg}"] = len(a & s_) / len(a)
                rec[f"covheld_{tg}"] = len(a & held) / len(a)
        rec["sensors"] = "|".join(f"{k}:{c}" for k, c in sorted(sens[key].items()))
        rows.append(rec)
    return rows


def unmatched_objects(mrows, match) -> list[dict]:
    """Testing objects carrying metadata where none of their metadata-bearing rows is matched to a
    Dronetag flight (matched == 1, ambiguous or not), grouped by sensor / class / values."""
    per_obj: dict = defaultdict(list)
    for r in mrows:
        if r["tree"] == "testing":
            per_obj[(r["iter_key"], r["sensor_key"], r["object"])].append(r)
    agg: dict = defaultdict(lambda: {"obj": set(), "rows": 0})
    for k, rs in sorted(per_obj.items()):
        if any((match.get((r["iter_key"], r["file"], r["row"])) or (0,) * 7)[4] == 1 for r in rs):
            continue
        r0 = rs[0]
        sig = (k[1], r0["cls"], r0["cls_sub"],
               *("|".join(sorted({r["raw"][f] for r in rs if r["raw"][f]})) for f in (MODEL, SERIAL, PROTOCOL, BAND)))
        agg[sig]["obj"].add(k)
        agg[sig]["rows"] += len(rs)
    return [{"sensor_key": sig[0], "cls": sig[1], "cls_sub": sig[2], "model": sig[3], "serial": sig[4],
             "protocol": sig[5], "freq_mhz": sig[6], "n_objects": len(a["obj"]), "n_rows": a["rows"]}
            for sig, a in sorted(agg.items(), key=lambda kv: (-kv[1]["rows"], kv[0]))]


def device_identity(mrows, serial_flight) -> list[dict]:
    """Per (tree, resolved serial): which models / protocols / sensors report together with it
    (object level: an object's model and protocol values are credited to every serial it carries)."""
    objs: dict = defaultdict(list)
    for r in mrows:
        objs[(r["tree"], r["iter_key"], r["sensor_key"], r["object"])].append(r)
    dev: dict = defaultdict(lambda: {"models": Counter(), "protocols": Counter(), "sensors": Counter(),
                                     "iters": set(), "objs": set(), "rows": 0, "classes": Counter()})
    for (tree, ik, sk, ob), rs in objs.items():
        sids = {r["sid"] for r in rs if r["sid"]}
        if not sids:
            continue
        models = Counter(r["raw"][MODEL] for r in rs if r["raw"][MODEL])
        protos = Counter(r["raw"][PROTOCOL] for r in rs if r["raw"][PROTOCOL])
        for sid in sids:
            d = dev[(tree, sid)]
            d["models"].update(models)
            d["protocols"].update(protos)
            d["sensors"][sk.split("_")[0]] += sum(1 for r in rs if r["sid"] == sid)
            d["iters"].add(ik)
            d["objs"].add((ik, sk, ob))
            d["rows"] += sum(1 for r in rs if r["sid"] == sid)
            d["classes"][rs[0]["cls"]] += 1
    out = []
    for (tree, sid), d in sorted(dev.items()):
        fl = sorted({(ik, f"{v[0][1]} ({v[0][2]})") for (ik, s2), v in serial_flight.items() if s2 == sid}) \
            if tree == "testing" else []
        out.append({"tree": tree, "serial": sid, "rows": d["rows"], "objects": len(d["objs"]),
                    "iterations": len(d["iters"]), "sensors": d["sensors"], "models": d["models"],
                    "protocols": d["protocols"], "classes": d["classes"], "flights": fl})
    return out


# --------------------------------------------------------------------------- #
# 4. whitelist
# --------------------------------------------------------------------------- #

def read_whitelists() -> list[dict]:
    import openpyxl
    out = []
    for it in C.iterations():
        if it.tree != "evaluation":
            continue
        for p in sorted((it.path / "FriendDronesList").glob("*.xlsx")):
            wb = openpyxl.load_workbook(p, read_only=True)
            try:
                rows = list(wb.active.iter_rows(values_only=True))
            finally:
                wb.close()
            header = [str(h).strip() if h is not None else "" for h in rows[0]]
            im, isr = header.index("Manufacturer"), header.index("Serial Number")
            for r in rows[1:]:
                if r[im] is None and r[isr] is None:
                    continue
                man = "" if r[im] is None else str(r[im]).strip()
                ser = "" if r[isr] is None else str(r[isr]).strip()
                out.append({"iter_key": it.key, "manufacturer": man, "serial": ser,
                            "serial_mapped": MM.map_value(SERIAL, ser),
                            "model_mapped": MM.map_value(MODEL, man)})
    return out


# --------------------------------------------------------------------------- #
# 5. independent raw recount (self-check 3)
# --------------------------------------------------------------------------- #

def raw_recount(sf: C.SensorFile, tb: C.Table, spell: Counter | None = None) -> Counter:
    """Counter of (field, raw_value) for one sensor file, straight from the raw cells
    (C.read_table). Same value rules as build_detections.py, re-implemented from the cells.
    `spell`, when given, also counts (field, normalised value, spelling in the cell) so that the
    table can show case variants that the upper-casing of serials hides."""
    n = tb.n_rows
    cols = tb.columns
    sid = sf.sensor_id
    cnt: Counter = Counter()
    for i in range(n):
        model = serial = proto = ""
        for col in ("objectInfo", "trackInfo", "additionalInformation"):
            cell = cols.get(col, [""] * n)[i]
            obj = C.parse_literal(cell) if cell else None
            if not isinstance(obj, list):
                continue
            for e in obj:
                if not isinstance(e, dict) or "type" not in e:
                    continue
                typ, val = str(e["type"]), e.get("value", "")
                if typ == "Drone Event Info":
                    if isinstance(val, str) and val[:1] == "{":
                        j = json.loads(val)
                        serial = serial or str(j.get("serial_number") or "")
                        model = model or str(j.get("factory_name") or "")
                    elif val:
                        serial = serial or str(val)
                elif typ == "model":
                    model = model or str(val)
                elif typ == "droneMaker":
                    model = model or str(val)
                elif typ == "serial number":
                    serial = serial or str(val)
                elif typ == "protocol":
                    proto = proto or str(val)
                elif typ == "technology":
                    proto = proto or str(val)
        idv = cols.get("id", [""] * n)[i]
        if idv:
            if not serial and sid in BD.SERIAL_ID_SENSORS and BD.looks_like_serial(idv):
                serial = idv
            if not proto:
                if sid in BD.PROTOCOL_ID_SENSORS:
                    proto = idv
                elif sid in BD.SERIAL_ID_SENSORS and not BD.looks_like_serial(idv) \
                        and ("-" in idv or " " in idv):
                    proto = idv
        sig = cols.get("signal", [""] * n)[i]
        freq = ""
        if sig:
            ents = [e for e in (C.parse_literal(sig) or []) if isinstance(e, dict)]
            if ents:
                best = max(ents, key=lambda e: float(e.get("amplitude", float("-inf"))))
                freq = fmt_freq(float(best["centreFrequency"]))
        for f, v, cell in ((MODEL, BD.norm_ws(model), BD.norm_ws(model)),
                           (SERIAL, BD.norm_ws(serial).upper(), BD.norm_ws(serial)),
                           (PROTOCOL, BD.norm_ws(proto), BD.norm_ws(proto)), (BAND, freq, freq)):
            if v:
                cnt[(f, v)] += 1
                if spell is not None:
                    spell[(f, v, cell)] += 1
    return cnt


def recount_all(rows: list[dict]) -> tuple[bool, str, dict]:
    """Recount every file of every metadata-bearing sensor key from the raw CSVs and compare
    with metadata_values.csv rows (per tree, sensor_key, field, raw_value). Also returns
    {(tree, sensor_key, field, raw_value): {spelling in the raw cell: rows}}."""
    bearing = {(r["tree"], r["sensor_key"]) for r in rows}
    bearing_keys = {k for _, k in bearing}
    want: Counter = Counter()
    for r in rows:
        want[(r["tree"], r["sensor_key"], r["field"], r["raw_value"])] += r["n_rows"]
    got: Counter = Counter()
    spelled: dict = defaultdict(Counter)
    n_files = 0
    per_sensor_files: Counter = Counter()
    for it in C.iterations():
        for sf in C.sensor_files(it):
            if sf.sensor_key not in bearing_keys:
                continue
            tb = C.read_table(sf.path)
            n_files += 1
            per_sensor_files[sf.sensor_key] += 1
            spell: Counter = Counter()
            for (f, v), c in raw_recount(sf, tb, spell).items():
                got[(it.tree, sf.sensor_key, f, v)] += c
            for (f, v, cell), c in spell.items():
                spelled[(it.tree, sf.sensor_key, f, v)][cell] += c
    ok = want == got
    detail = f"{len(bearing_keys)} metadata-bearing sensors, {n_files} raw files, {sum(got.values()):,} values"
    if not ok:
        diff = [(k, want.get(k, 0), got.get(k, 0)) for k in sorted(set(want) | set(got))
                if want.get(k, 0) != got.get(k, 0)]
        detail += f"; MISMATCH {diff[:5]}"
    return ok, detail, dict(spelled)


# --------------------------------------------------------------------------- #
# 6. report
# --------------------------------------------------------------------------- #

def md_table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(x).replace("|", "/") if x != "" else "-" for x in r) + " |")
    return out + [""]


def write_csv(path: Path, rows: list[dict], order: list[str]) -> None:
    C.write_columns(path, {k: [r.get(k, "") for r in rows] for k in order}, order)


def build_report(values, drone_rows, cov_det, cov_union, whitelist, unmatched, devices, det_stats, info,
                 flights) -> str:
    L: list[str] = []
    L.append("# Class metadata: sensor values, dropdown mapping, per-drone evidence\n")
    L.append("Generated by `tools/metadata_scan.py` (mapping rules: `tools/metadata_map.py`). "
             "Tables: `metadata_values.csv`, `drone_metadata_testing.csv`, `friendly_whitelist.csv`. "
             "Ground truth is used on the testing tree only; the evaluation tree is shown as values, "
             "never as rows.\n")

    mrows_all = info["rows"]
    # ---- headline
    tot = Counter()
    for r in values:
        tot[(r["tree"], r["field"])] += r["n_rows"]
    srcs = defaultdict(set)
    for r in values:
        srcs[r["field"]].add(r["sensor_key"].split("_")[0])
    L.append("## 0. Findings\n")
    for f in FIELDS:
        L.append(f"* **{f}**: reported by {', '.join(sorted(srcs[f]))} "
                 f"({tot[('testing', f)]:,} testing rows, {tot[('evaluation', f)]:,} evaluation rows).")
    L.append("* Radars, EO/IR, acoustic, UTM services (QTVS, VKYY), SMFS, LJJX and EUIG carry none of the "
             "four fields: metadata can only come from the CYBER / PASSIVE-RF families and the relay of "
             "a serial through STUG, CEQX, ILYS, IKJX, MMJJ.")
    ser_t = {d["serial"] for d in devices if d["tree"] == "testing"}
    ser_e = {d["serial"] for d in devices if d["tree"] == "evaluation"}
    wl_ser = {w["serial_mapped"] for w in whitelist}
    L.append("* A serial is the device identity: the same serial appears on several sensors and implies the "
             "model (section 3). Serials seen: testing " + ", ".join(f"`{x}`" for x in sorted(ser_t))
             + "; evaluation " + ", ".join(f"`{x}`" for x in sorted(ser_e))
             + "; in both trees " + (", ".join(f"`{x}`" for x in sorted(ser_t & ser_e)) or "none") + ".")
    sides: dict = defaultdict(set)
    names: dict = defaultdict(set)
    for d in devices:
        for _, nm in d["flights"]:
            m = re.fullmatch(r"(.*) \((BLUE|RED)\)", nm)
            sides[d["serial"]].add(m[2])
            names[d["serial"]].add(m[1])
    L.append("* Identity from the serial: " + "; ".join(
        f"`{x}` -> {'/'.join(sorted(sides[x]))} flight ({', '.join(sorted(names[x]))})" for x in sorted(sides))
        + ". Whitelist serials: " + ", ".join(f"`{x}`" for x in sorted(wl_ser))
        + ". Every testing serial attributed to a BLUE flight is a whitelist serial and the others are not, "
        "so `serial in whitelist` => Friendly is consistent with the testing ground truth"
        + ("" if all(sides[x] == {"BLUE"} for x in sides if x in wl_ser) and not any(
            "BLUE" in sides[x] for x in sides if x not in wl_ser) else " (CHECK: it is not here)") + ".")
    multi = {x: sorted(n) for x, n in names.items() if len(n) > 1}
    if multi:
        L.append("* WARNING: a serial is attributed to several Dronetag drone names ("
                 + "; ".join(f"`{x}`: {', '.join(n)}" for x, n in sorted(multi.items()))
                 + "): Dronetag names are not a stable airframe identity, so key the catalog by serial / "
                 "model, not by drone name.")
    gx = info["ground_excluded"]
    if gx:
        L.append("* Ground-parked evidence is EXCLUDED from the drone tables and proposals (a parked drone is "
                 "matched to whatever sits near it): " + "; ".join(
                     f"{ik.replace('testing/', '')} {dn} ({sd}) {n} rows" for (ik, dn, sd), n in sorted(gx.items()))
                 + ". Only seconds with GT height > 5 m count.")
    near = sorted({(r["raw_value"], r["near_match"]) for r in values if r["near_match"] and r["tree"] == "evaluation"})
    L.append("* Evaluation serials that only match the dropdown after removing a vendor prefix / suffix: "
             + ", ".join(f"`{a}` ~ `{b}`" for a, b in near)
             + ". `map_value` stays strict; `resolve_serial` / `serial_number_near` resolve them.")
    L.append(f"* Ground-truth attribution is thin: of {sum(1 for r in info['rows'] if r['tree'] == 'testing'):,} "
             f"testing metadata rows only {info['n_det']:,} are matched to a flight unambiguously "
             f"(drone-class rows), {info['skipped']:,} more matched rows are Home / UAV Transmitter / "
             "EM Transmitter positions (the pilot, not the drone) and are excluded. Bearing-only sensors "
             "(OEPS, GTYT, ORFH, YXAB) are never unambiguous, so no band or bearing-sensor protocol "
             "can be tied to a Dronetag flight by position.\n")

    L.append("## 1. Which sensors give which fields\n")
    sens_tab = []
    for (tree, sk, f), n in sorted(det_stats["rows"].items()):
        vs = [r for r in values if (r["tree"], r["sensor_key"], r["field"]) == (tree, sk, f)]
        sens_tab.append([tree, sk, f, n, det_stats["objects"][(tree, sk, f)],
                         det_stats["iterations"][(tree, sk, f)], len(vs),
                         sum(r["n_rows"] for r in vs if r["mapped"])])
    L += md_table(["tree", "sensor", "field", "rows", "objects", "iterations", "distinct raw values",
                   "rows with a dropdown value"], sens_tab)

    L.append("## 2. Mapping table (raw value -> dropdown)\n")
    L.append("Rows are detection rows (testing / evaluation). A blank mapped cell means the raw value "
             "has no dropdown equivalent (`map_value` returns ''; whether to write 'Unknown' is a fusion "
             "decision). ORFH / YXAB put drone-type text in the protocol slot; it is kept under `protocol`.\n")
    agg2: dict = defaultdict(lambda: [0, 0, set(), ("", "", "")])
    for r in values:
        a = agg2[(r["field"], r["raw_value"])]
        a[0 if r["tree"] == "testing" else 1] += r["n_rows"]
        a[2].add(r["sensor_key"].split("_")[0])
        a[3] = (r["mapped"], r["rule"], r["near_match"])
    for f in FIELDS:
        L.append(f"### {f}\n")
        tab = []
        for (ff, rv), a in sorted(agg2.items(), key=lambda kv: (-(kv[1][0] + kv[1][1]), kv[0])):
            if ff == f:
                row = [rv, a[0], a[1], "/".join(sorted(a[2])), a[3][0], a[3][1]]
                tab.append(row + ([a[3][2]] if f == SERIAL else []))
        L += md_table(["raw", "rows test", "rows eval", "sensors", "mapped", "rule"]
                      + (["near dropdown serial"] if f == SERIAL else []), tab)
    cat: dict = {}
    with (Path(__file__).resolve().parent / "drone_catalog.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cat[r["drone_name"]] = r["uav_model/manufacturer"]
    gt_names: dict = defaultdict(set)
    for (ik, _), v in flights.items():
        gt_names[ik].add(v["drone_name"])
    L.append("Iteration-level corroboration of the model aliases (testing; `tools/drone_catalog.csv` read "
             "only): a raw model value is corroborated in an iteration when a Dronetag drone whose "
             "catalog model equals the mapped value flew there.\n")
    tab = []
    seen_it: dict = defaultdict(set)
    for r in mrows_all:
        if r["tree"] == "testing" and r["raw"][MODEL] and r["mapped"][MODEL]:
            seen_it[(r["raw"][MODEL], r["mapped"][MODEL])].add(r["iter_key"])
    for (rv, mp), its in sorted(seen_it.items()):
        cells = []
        for ik in sorted(its):
            hit = sorted(n for n in gt_names[ik] if cat.get(n) == mp)
            cells.append(f"{ik.replace('testing/', '')}: {', '.join(hit) if hit else 'no catalog drone with this model'}")
        tab.append([rv, mp, "; ".join(cells)])
    L += md_table(["raw model", "mapped", "iterations seen -> Dronetag drones of that model"], tab)
    mo = sorted({r["raw_value"] for r in values if r["field"] == MODEL and r["rule"].startswith("model:maker_only")})
    if mo:
        L.append("Maker-only model rules (rule `model:maker_only:*`; the raw value names a maker, the dropdown "
                 "has one entry for it, so any other model of that maker would be mislabelled): "
                 + ", ".join(f"`{x}`" for x in mo) + ". Treat as weak evidence.\n")
    L.append("Frequency: GTYT reports channel 149 as centre 5745 MHz; OEPS reports the centre of each "
             "burst (2.4 GHz, 5.2 GHz and 5.8 GHz hops). 2200-2500 never wins: every observed frequency "
             "lies inside a narrower band.\n")

    L.append("## 3. Device identity: serial <-> model <-> protocol\n")
    L.append("Per resolved serial, over objects that carry both (rows = detection rows with that serial). "
             "Testing flights come from the serial vote (>= 3 matched drone-class rows, >= 80 % for one "
             "flight). `metadata_map.SERIAL_MODEL` encodes the serial -> model implication and "
             "self-check 7 re-verifies it against every object below.\n")
    tab = []
    for d in devices:
        mp = MM.model_for_serial(d["serial"])
        tab.append([d["tree"], d["serial"], MM.SERIAL_MODEL.get(MM.resolve_serial(d["serial"]), "-"),
                    d["rows"], d["objects"], d["iterations"],
                    "/".join(f"{k}:{v}" for k, v in sorted(d["sensors"].items())),
                    "; ".join(f"{k} ({v})" for k, v in d["models"].most_common()),
                    "; ".join(f"{k} ({v})" for k, v in d["protocols"].most_common()),
                    "; ".join(f"{k} {v}" for k, v in d["classes"].most_common()),
                    "; ".join(f"{a.replace('testing/', '')} {b}" for a, b in d["flights"])])
    L += md_table(["tree", "serial (resolved)", "implied model", "rows", "objects", "iters", "sensors",
                   "raw models on the same objects", "raw protocols", "object classes", "flights (testing)"], tab)

    L.append("## 4. Per-drone metadata on testing: PROPOSED values for tools/drone_catalog.csv\n")
    L.append("Evidence level `union` = strict match (det), else serial vote (ser), else track label (obj); "
             "drone-class rows only. A value is PROPOSED when it is the top value with purity >= "
             f"{PROPOSE_MIN_PURITY:.0%} and >= 3 rows; `mixed` otherwise (the competing values follow); "
             "a serial that is the top value of two or more drone names is flagged `SHARED` and is NOT proposed. "
             "Only airborne rows count (GT height > 5 m). "
             "`n` = detection rows, `fl` = flights, `sens` = reporting sensors. The serial column "
             "uses the resolved serial (`serial_number_near`), so vendor-prefixed strings count.\n")
    dnames = sorted({r["drone_name"] for r in drone_rows if r["scope"] == "drone_name"})
    shared: dict = defaultdict(set)
    for r in drone_rows:
        if r["scope"] == "drone_name" and r["level"] == "union" and r["field"] == SERIAL_NEAR and r["n_rows"] >= 3:
            shared[r["value"]].add(r["drone_name"])
    tab = []
    for dn in dnames:
        sd = "|".join(sorted({r["side"] for r in drone_rows if r["scope"] == "drone" and r["drone_name"] == dn}))
        tab.append([dn, sd] + [proposal_cell(drone_rows, dn, f, shared) for f in (MODEL, PROTOCOL, SERIAL_NEAR, BAND)])
    L += md_table(["drone name", "sides", "uav_model/manufacturer", "protocol", "serial_number",
                   "signal_frequency_band"], tab)
    none_ = sorted((c["drone_name"], c["side"], c["iter_key"].replace("testing/", ""))
                   for c in cov_union if c["airborne_s"] and c["cov_any"] == 0)
    L.append("Flights with airborne seconds but no airborne drone-class metadata record carrying a dropdown "
             "value (the sensors do not report them in that iteration): "
             + ("; ".join(f"{n} ({sd}) {it}" for n, sd, it in none_) or "none") + ".\n")

    L.append("### Per flight, by evidence level\n")
    fl_cells: dict = defaultdict(dict)
    for r in drone_rows:
        if r["scope"] != "flight" or not r["value"] or r["field"] == SERIAL_NEAR:
            continue
        key = (r["iter_key"], r["drone_name"], r["side"], r["level"])
        fl_cells[key].setdefault(r["field"], []).append(f"{r['value']} ({r['n_rows']}; {r['sensors']})")
    tab = []
    for key in sorted(fl_cells):
        c = fl_cells[key]
        tab.append([key[0].replace("testing/", ""), f"{key[1]} ({key[2]})", key[3]]
                   + [("; ".join(c.get(f, [])[:3]) or "-") for f in FIELDS])
    L += md_table(["iteration", "flight", "level", "model", "protocol", "serial (exact)", "band"], tab)

    L.append("## 5. Coverage: share of each flight's airborne seconds with a metadata detection\n")
    L.append("Airborne seconds = GT samples with height > 5 m. A second is covered when >= 1 record "
             "with a dropdown value for the field falls in it. `any` = any of the four fields. "
             f"Cells read `strict / held` where held also counts +-{HOLD_S} s. Table A uses only the "
             "strict matcher (matched, unambiguous, drone class); table B the `union` level.\n")
    for title, cv in (("A. strict (det)", cov_det), ("B. union (det + serial vote + track label)", cov_union)):
        L.append(f"### {title}\n")
        tab = []
        for c in cv:
            if not c["airborne_s"]:
                continue
            tab.append([c["iter_key"].replace("testing/", ""), f"{c['drone_name']} ({c['side']})", c["airborne_s"]]
                       + [f"{c['cov_' + t]:.0%}/{c['covheld_' + t]:.0%}" for t in ["any"] + FIELDS]
                       + [c.get("sensors", "")])
        L += md_table(["iteration", "flight", "air s", "any", "model", "protocol", "serial", "band",
                       "reporting sensors (rows)"], tab)

    L.append("## 6. Evaluation tree: values present (values only)\n")
    tab = []
    for f in FIELDS:
        vs = defaultdict(lambda: [0, set()])
        for r in values:
            if r["tree"] == "evaluation" and r["field"] == f:
                key = r["mapped"] or f"(no dropdown) {r['raw_value']}"
                vs[key][0] += r["n_rows"]
                vs[key][1].add(r["sensor_key"].split("_")[0])
        for k, (n, s_) in sorted(vs.items(), key=lambda kv: (-kv[1][0], kv[0])):
            tab.append([f, k, n, "/".join(sorted(s_))])
    L += md_table(["field", "value", "rows", "sensors"], tab)
    L.append("Friendly(Blue)UAS whitelist (`friendly_whitelist.csv`):\n")
    L += md_table(["iteration", "manufacturer", "serial", "serial_mapped", "model_mapped"],
                  [[w["iter_key"], w["manufacturer"], w["serial"], w["serial_mapped"], w["model_mapped"]]
                   for w in whitelist])
    ev_serials = {d["serial"] for d in devices if d["tree"] == "evaluation"}
    wl = {w["serial_mapped"] for w in whitelist}
    L.append(f"Evaluation serials seen by sensors: {', '.join(f'`{x}`' for x in sorted(ev_serials))}; "
             f"on the whitelist: {', '.join(f'`{x}`' for x in sorted(ev_serials & wl))}; "
             f"not on the whitelist (hostile or unknown): {', '.join(f'`{x}`' for x in sorted(ev_serials - wl))}.\n")

    L.append("## 7. Metadata on objects NOT matched to any Dronetag flight (testing)\n")
    BEARING_ONLY = ("YXAB", "ORFH", "OEPS", "GTYT")
    pos_um = [u for u in unmatched if u["cls"] not in NON_DRONE_CLASSES
              and u["sensor_key"].split("_")[0] not in BEARING_ONLY]
    um_ser = sorted({x for u in pos_um for x in u["serial"].split("|") if x})
    um_mod = sorted({f"{u['sensor_key'].split('_')[0]} '{x}'" for u in pos_um for x in u["model"].split("|") if x})
    L.append("Objects whose metadata-bearing rows are all unmatched. They are either drones without a "
             "Dronetag (drone class with a position"
             + (f"; serials seen: {', '.join(f'`{x}`' for x in um_ser)}" if um_ser else "")
             + "), pilot controllers ('UAV Transmitter', 'EM Transmitter') or the take-off point ('Home'). "
             "YXAB / ORFH rows are bearing-only and carry no usable position, so they cannot be matched at all.\n")
    cls_tot: Counter = Counter()
    for u in unmatched:
        grp = ("controller / Home" if u["cls"] in NON_DRONE_CLASSES else
               "bearing-only (no usable position)" if u["sensor_key"].split("_")[0] in BEARING_ONLY
               else "drone-class with position")
        cls_tot[grp] += u["n_objects"]
    L.append("Unmatched metadata objects by kind: " + ", ".join(f"{k} {v}" for k, v in sorted(cls_tot.items()))
             + ". The 'drone-class with position' ones are drones that no Dronetag covers (or whose GT "
             "position the matcher could not reach)"
             + (f", model strings: {', '.join(um_mod)}" if um_mod else "")
             + "; they must still be reported with their own metadata.\n")
    L += md_table(["sensor", "class", "subclass", "model", "serial", "protocol", "freq MHz", "objects", "rows"],
                  [[u["sensor_key"].split("_")[0], u["cls"], u["cls_sub"], u["model"], u["serial"],
                    u["protocol"], u["freq_mhz"], u["n_objects"], u["n_rows"]] for u in unmatched])
    return "\n".join(L) + "\n"


def proposal_cell(drone_rows, dn, field, shared) -> str:
    rs = [r for r in drone_rows if r["scope"] == "drone_name" and r["level"] == "union" and r["field"] == field
          and r["drone_name"] == dn and r["value"]]
    if not rs:
        return "- (no evidence)"
    top = rs[0]
    flag = "PROPOSED" if top["purity"] >= PROPOSE_MIN_PURITY and top["n_rows"] >= 3 else "mixed"
    if field == SERIAL_NEAR and flag == "PROPOSED" and len(shared.get(top["value"], ())) > 1:
        flag = "SHARED"                    # the same serial is also the top evidence of another drone name
    txt = f"{flag} {top['value']} ({top['share']:.0%}, n={top['n_rows']}, fl={top['n_flights']}, {top['sensors']})"
    if len(rs) > 1:
        txt += " vs " + ", ".join(f"{r['value']} {r['n_rows']}" for r in rs[1:4])
    if field == SERIAL_NEAR and len(shared.get(top["value"], ())) > 1:
        txt += " SHARED with " + ", ".join(sorted(shared[top["value"]] - {dn}))
    return txt


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    t0 = time.time()
    det = Q.load_detections(usecols=DET_COLS)
    n_det = len(det["t"])
    log(f"loaded {n_det:,} detections in {time.time() - t0:.0f} s")
    raw = raw_fields(det)

    # ---- metadata_values.csv
    values, totals = build_values(det, raw)
    recount_ok, recount_detail, spelled = recount_all(values)
    for r in values:
        sp = spelled.get((r["tree"], r["sensor_key"], r["field"], r["raw_value"]), {})
        r["raw_spellings"] = ("|".join(f"{k}:{c}" for k, c in sorted(sp.items(), key=lambda kv: (-kv[1], kv[0])))
                              if any(k != r["raw_value"] for k in sp) else "")
    write_csv(OUT / "metadata_values.csv", values,
              ["tree", "sensor_key", "field", "raw_value", "n_rows", "n_objects", "n_iterations",
               "mapped", "rule", "near_match", "raw_spellings"])
    log(f"metadata_values.csv: {len(values)} rows")
    det_stats = {"objects": defaultdict(int), "iterations": defaultdict(int), "rows": Counter()}
    for f in FIELDS:
        m = raw[f] != ""
        o: dict = defaultdict(set)
        iset: dict = defaultdict(set)
        for tr, sk, ob, it in zip(det["tree"][m], det["sensor_key"][m], det["object_id"][m], det["iter_key"][m]):
            o[(tr, sk, f)].add((it, ob))
            iset[(tr, sk, f)].add(it)
            det_stats["rows"][(tr, sk, f)] += 1
        for k in o:
            det_stats["objects"][k] = len(o[k])
            det_stats["iterations"][k] = len(iset[k])

    # ---- friendly whitelist
    whitelist = read_whitelists()
    write_csv(OUT / "friendly_whitelist.csv", whitelist,
              ["iter_key", "manufacturer", "serial", "serial_mapped", "model_mapped"])

    # ---- testing association
    match = load_match()
    objlab = load_obj_labels()
    mrows = metadata_rows(det, raw)
    gt = Q.load_groundtruth()
    air = airborne_seconds(gt)
    recs, info = drone_observations(mrows, match, objlab, air)
    by_level = Counter(r["level"] for r in recs)
    log("drone observations (drone-class testing rows): "
        + ", ".join(f"{k} {by_level[k]:,}" for k in ("det", "ser", "obj", "union"))
        + f"; matched Home/Transmitter rows excluded: {info['skipped_matched']:,}"
        + f"; ground-only rows excluded: {sum(info['ground_excluded'].values()):,}")
    drone_rows = []
    for scope in ("flight", "drone", "drone_name"):
        drone_rows += group_rows(recs, scope)
    write_csv(OUT / "drone_metadata_testing.csv", drone_rows,
              ["scope", "level", "iter_key", "flight_key", "drone_name", "side", "n_flights", "field",
               "value", "n_rows", "n_objects", "n_sensors", "sensors", "share", "purity", "raw_values"])
    flights = {}
    for ik, fk, dn, sd in set(zip(gt["iter_key"].tolist(), gt["flight_key"].tolist(),
                                  gt["drone_name"].tolist(), gt["side"].tolist())):
        flights[(ik, fk)] = {"drone_name": dn, "side": sd}
    cov_det = coverage(recs, air, flights, "det")
    cov_union = coverage(recs, air, flights, "union")
    unmatched = unmatched_objects(mrows, match)
    devices = device_identity(mrows, info["serial_flight"])

    report_info = {"rows": mrows, "n_det": by_level["det"], "skipped": info["skipped_matched"],
                   "ground_excluded": info["ground_excluded"]}
    report = build_report(values, drone_rows, cov_det, cov_union, whitelist, unmatched, devices,
                          det_stats, report_info, flights)
    (OUT / "metadata_report.md").write_text(report, encoding="utf-8")
    log(f"wrote metadata_values.csv, drone_metadata_testing.csv, friendly_whitelist.csv, metadata_report.md "
        f"({time.time() - t0:.0f} s)")

    # ------------------------------------------------------------------ self-checks
    # 1. n_rows per (tree, sensor_key, field) = independent count of non-empty raw cells
    tab_tot: Counter = Counter()
    for r in values:
        tab_tot[(r["tree"], r["sensor_key"], r["field"])] += r["n_rows"]
    indep: Counter = Counter()
    src = {MODEL: (det["model"] != "") | np.array(["droneMaker" in s for s in det["info"]], dtype=bool),
           PROTOCOL: det["protocol"] != "", SERIAL: det["serial"] != "", BAND: np.isfinite(det["freq_mhz"])}
    for f, mask in src.items():
        for tree, sk in set(zip(det["tree"][mask], det["sensor_key"][mask])):
            indep[(tree, sk, f)] = int(((det["tree"] == tree) & (det["sensor_key"] == sk) & mask).sum())
    check(tab_tot == indep and sum(tab_tot.values()) > 0,
          f"metadata_values n_rows sum to the source row totals for all {len(indep)} (tree, sensor_key, field) "
          f"cells ({sum(tab_tot.values()):,} values)")

    # 2. every non-empty mapped value is a dropdown string
    bad = []
    for r in values:
        if r["mapped"] and r["mapped"] not in C.DROPDOWNS[r["field"]]:
            bad.append(r)
        if r["near_match"] and r["near_match"] not in C.DROPDOWNS[SERIAL]:
            bad.append(r)
    for r in drone_rows:
        fld = SERIAL if r["field"] == SERIAL_NEAR else r["field"]
        if r["value"] and r["value"] not in C.DROPDOWNS[fld]:
            bad.append(r)
    for w in whitelist:
        if w["serial_mapped"] and w["serial_mapped"] not in C.DROPDOWNS[SERIAL]:
            bad.append(w)
        if w["model_mapped"] and w["model_mapped"] not in C.DROPDOWNS[MODEL]:
            bad.append(w)
    check(not bad, f"every non-empty mapped value (values, near matches, drone tables, whitelist) is in "
                   f"C.DROPDOWNS ({len(values)} + {len(drone_rows)} + {len(whitelist)} rows)")

    # 3. raw recount
    check(recount_ok, f"recount from the raw CSVs via C.read_table equals metadata_values.csv ({recount_detail})")

    # 4. map_value unit cases: (field, raw, expected)
    cases = [
        (PROTOCOL, "ocusync", "DJI Ocusyinc"), (PROTOCOL, "DJI Ocusync", "DJI Ocusyinc"),
        (PROTOCOL, "DJI OcuSyncCCF", "DJI Ocusyinc"), (PROTOCOL, "sv-ocusync", "DJI Ocusyinc"),
        (PROTOCOL, "wifi", "Wifi"), (PROTOCOL, "Autel Skylink", "Autel Skylink"),
        (PROTOCOL, "Video", "Video"), (PROTOCOL, "Digital Video Link Type 2", "Digital video link type 2"),
        (PROTOCOL, "DIY CROSSFIRE", "DIY Crossfire"),
        (PROTOCOL, "Eleron 3", ""), (PROTOCOL, "Zala Aero", ""),
        (PROTOCOL, "DJI Spark / Phantom 3 SE / Mavic Air / Mavic Mini", ""), (PROTOCOL, "", ""),
        (BAND, "5745", "5725 – 5850"), (BAND, "5178", "5150 – 5250"), (BAND, "2473.5", "2400 – 2483.5"),
        (BAND, "2416.5", "2400 – 2483.5"), (BAND, "2300", "2200 – 2500"), (BAND, "915", "902 – 928"),
        (BAND, "5838.5", "5725 – 5850"), (BAND, "1575", ""), (BAND, "x", ""),
        (SERIAL, "F5FHD233M00DEYG6", "F5FHD233M00DEYG6"), (SERIAL, " 1748fev2hl7824051124 ", "1748FEV2HL7824051124"),
        (SERIAL, "1ZNBJAB00C00MQ", "1ZNBJAB00C00MQ"), (SERIAL, "1581F45TB21BR1AE03FP", ""),
        (SERIAL, "F1ZNBJAB00C00MQ0", ""), (SERIAL, "ZZZ", ""),
        (MODEL, "DJI Mavic 3 Enterprise", "DJI Mavic 3 Enterprise"), (MODEL, "MTRC 300R", "MTRC 300R"),
        (MODEL, "Autel Drone", "Autel Drone"), (MODEL, "Mavic 3", "DJI Mavic 3"),
        (MODEL, "Mavic 3 E", "DJI Mavic 3 Enterprise"), (MODEL, "Anafi SE", "Parrot Anafi"),
        (MODEL, "DJI Matrice 300R", "MTRC 300R"), (MODEL, "Autel Evo 2", "Autel Drone"),
        (MODEL, "Parrot", "Parrot Anafi"), (MODEL, "Remote ID", ""), (MODEL, "DRI Drone", ""), (MODEL, "DJI Mavic 2 Pro", ""),
    ]
    fails = [(f, v, MM.map_value(f, v), e) for f, v, e in cases if MM.map_value(f, v) != e]
    near_cases = [("1581F45TB21BR1AE03FP", "F45TB21BR1AE03FP"), ("F1ZNBJAB00C00MQ0", "1ZNBJAB00C00MQ"),
                  ("F5FHD233M00DEYG6", ""), ("ZZZ", "")]
    fails += [("near", a, MM.serial_near(a), b) for a, b in near_cases if MM.serial_near(a) != b]
    check(not fails, f"map_value unit cases ({len(cases)} + {len(near_cases)} near-match cases){'' if not fails else ' FAILED ' + str(fails)}")

    # 5. whitelist rows all resolved
    check(len(whitelist) == 5 and all(w["serial_mapped"] and w["model_mapped"] for w in whitelist),
          f"all {len(whitelist)} whitelist rows map to a dropdown serial and model")

    # 6. coverage self-consistency: airborne seconds equal flight_coverage.csv
    fc = {}
    with (OUT / "flight_coverage.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            fc[(r["iter_key"], r["flight_key"])] = int(r["airborne_seconds"])
    mism = [(k, len(air[k]), fc.get(k)) for k in air if fc.get(k) != len(air[k])]
    check(not mism, f"airborne seconds of {len(air)} flights equal flight_coverage.csv"
                    f"{'' if not mism else ' MISMATCH ' + str(mism[:3])}")

    # 7. SERIAL_MODEL is consistent with every object carrying a serial and a mapped model, and the whitelist
    objs: dict = defaultdict(list)
    for r in mrows:
        objs[(r["tree"], r["iter_key"], r["sensor_key"], r["object"])].append(r)
    conflicts, n_pairs, seen_sid = [], 0, set()
    for k, rs in objs.items():
        sids = {r["sid"] for r in rs if r["sid"]}
        models = {r["mapped"][MODEL] for r in rs if r["mapped"][MODEL]}
        for sid in sids:
            want_m = MM.SERIAL_MODEL.get(sid)
            if want_m and models:
                n_pairs += 1
                seen_sid.add(sid)
                if models != {want_m}:
                    conflicts.append((k[0], k[2][:4], sid, sorted(models)))
    wl_bad = [w for w in whitelist if MM.SERIAL_MODEL.get(w["serial_mapped"]) != w["model_mapped"]]
    check(not conflicts and not wl_bad and len(MM.SERIAL_MODEL) == 4,
          f"SERIAL_MODEL agrees with {n_pairs} (object, serial) pairs carrying a mapped model "
          f"({len(seen_sid)} serials) and all {len(whitelist)} whitelist rows"
          f"{'' if not conflicts and not wl_bad else ' CONFLICTS ' + str(conflicts[:3]) + str(wl_bad[:2])}")

    # 8. every hand-written alias / protocol rule is backed by a raw value in the data or whitelist
    seen_low = {(r["field"], r["raw_value"].lower()) for r in values}
    seen_low |= {(MODEL, w["manufacturer"].lower()) for w in whitelist}
    seen_low |= {(PROTOCOL, r["raw_value"].lower()) for r in values if r["field"] == MODEL}   # DIY CROSSFIRE
    unbacked = [("model", k) for k in MM._MODEL_RULES if (MODEL, k) not in seen_low] \
        + [("protocol", k) for k in MM._PROTOCOL_RULES if (PROTOCOL, k) not in seen_low]
    check(not unbacked, f"all {len(MM._MODEL_RULES)} model aliases and {len(MM._PROTOCOL_RULES)} protocol rules "
                        f"are backed by a raw value seen in the data / whitelist{'' if not unbacked else ' UNBACKED ' + str(unbacked)}")

    # 9. drone-table evidence is airborne-only; the Evo 2 parked-drone case is gone from the proposals
    grd = [r for r in recs if r["second"] not in air.get((r["iter_key"], r["flight_key"]), ())]
    parked = [r for r in drone_rows if r["scope"] == "drone_name" and r["drone_name"] == "Autel Evo 2"]
    check(not grd and not parked,
          f"all {len(recs):,} drone-table records fall in airborne seconds of their flight; "
          f"{sum(info['ground_excluded'].values()):,} ground-only rows excluded; no rows for the ground-only 'Autel Evo 2'")

    failed = [m for ok_, m in CHECKS if not ok_]
    log("ALL SELF-CHECKS PASSED" if not failed else f"SELF-CHECKS FAILED: {len(failed)}")
    log(f"total {time.time() - t0:.0f} s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
