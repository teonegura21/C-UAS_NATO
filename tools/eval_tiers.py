#!/usr/bin/env python3
"""Evaluation-only field ladder and tier-by-tier K-means.

Scope: the three EVALUATION iterations only (DI/S1-I3, DII/S2-I1, DIII/S2-I3). The
testing tree is not read.

1. Field ladder. For every field documented in README/README_Sensor.pdf, the share of
   the PDF-documented sensors present in the evaluation data that have it (also the share
   in the actual evaluation CSV headers). Fields are grouped into tiers by that share:
   100 % (present for every sensor), then the next most common, down to fields only one
   sensor has.
2. K-means per tier, each using only the fields of that tier, on evaluation data:
   K1  tier 1   sensor_location_x/y        -> sensor positions -> physical sites
   K2  tier 2   + classification           -> class profile per sensor -> what sensors report
   K3  tier 3   + location_x/y (object)    -> detection positions -> coarse regions several sensors cover;
               plus a same-time test (not K-means): detections of different sensors within 1 s and 50 m
   K4  tier 7   + enuVelocity              -> track kinematics + the sensors' own class shares
Tier numbers are looked up in the ladder, not hard-coded.

Outputs (no row-level data): data/derived/eval/ field_ladder.csv, k1_sites.csv,
k2_class_profiles.csv, k3_detection_clusters.csv, k4_track_clusters.csv,
k4_track_assignments.csv, EVAL_TIERS.md.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coobservation as CO  # noqa: E402  (pair_hits / group_labels: verified same-time matching)
import cuas_clean as Q  # noqa: E402
import cuas_common as C  # noqa: E402
import kmeans as K  # noqa: E402

OUT = C.DERIVED / "eval"
SEED = 0
SITE_RADIUS_M = 30.0      # K1: a cluster is "one site" when every member is within this of its centre
COOBS_GATE_M = 50.0       # same-time test: different sensors within this distance ...
COOBS_DT_S = CO.MAX_DT_S  # ... and this time difference (1 s)
K3_BIN_S = 1.0            # K3: one sample per (sensor, object, 1-s bin), so high-rate sensors do not dominate

CLASS_GROUPS = {          # canonical class -> group used by K2 / K4 profiles
    "UAV": "uav", "UAV Rotary Wing": "uav", "UAV Fixed Wing": "uav", "FIXED_WING": "uav",
    "UAV Transmitter": "transmitter", "EM Transmitter": "transmitter",
    "Air Vehicle": "air_vehicle",
    "Manned Fixed Wing": "manned", "Manned Rotary Wing": "manned",
    "Animal": "bio", "Bird": "bio",
    "Land Vehicle": "ground", "Sea Vessel": "ground", "Fixed Object": "ground", "Home": "ground",
}
GROUPS = ["uav", "transmitter", "air_vehicle", "manned", "bio", "ground", "unknown_or_none"]


def class_group(cls: str, sub: str) -> str:
    """Most specific usable class: the subclass when it says more than the class."""
    for name in (sub, cls):
        if name and name not in ("Unknown", "Other"):
            return CLASS_GROUPS.get(name, "unknown_or_none")
    return "unknown_or_none"


def md_table(header, rows) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def scan(x, ks, n_init=20, sample=4000):
    res = K.scan_k(x, [k for k in ks if k < len(x)], n_init=n_init, seed=SEED, sample=sample)
    best = max(res, key=lambda r: (r[2], -r[0]))
    return res, best


# --------------------------------------------------------------------------- #
# 1. Field ladder
# --------------------------------------------------------------------------- #


def field_ladder(evals):
    headers: dict[str, set[str]] = defaultdict(set)
    rows = Counter()
    for it in evals:
        for sf in C.sensor_files(it):
            header, it_rows = C.iter_rows(sf.path)
            headers[sf.sensor_id].update(h for h in header if h)
            rows[sf.sensor_id] += sum(1 for r in it_rows if r is not None)
    spec = list(csv.DictReader((C.DERIVED / "field_matrix_spec.csv").open()))
    fields = [k for k in spec[0] if k not in ("sensor_id", "category")]
    pdf = {r["sensor_id"]: {f for f in fields if r[f] == "1"} for r in spec}
    category = {r["sensor_id"]: r["category"] for r in spec}
    in_eval = sorted(headers)
    documented = [s for s in in_eval if s in pdf]
    table = []
    for f in sorted(set().union(*(pdf[s] for s in documented))):
        have = [s for s in documented if f in pdf[s]]
        have_data = [s for s in in_eval if f in headers[s]]
        table.append(dict(field=f, n_pdf=len(have), pct_pdf=round(100 * len(have) / len(documented), 1),
                          n_data=len(have_data), pct_data=round(100 * len(have_data) / len(in_eval), 1),
                          sensors=" ".join(have), missing=" ".join(s for s in documented if s not in have),
                          sensors_data=" ".join(have_data)))
    table.sort(key=lambda r: (-r["n_pdf"], -r["n_data"], r["field"]))
    levels = sorted({r["n_pdf"] for r in table}, reverse=True)
    for r in table:
        r["tier"] = levels.index(r["n_pdf"]) + 1
    extra = sorted(set().union(*headers.values()) - set().union(*(pdf[s] for s in documented)))
    return table, documented, [s for s in in_eval if s not in pdf], category, rows, extra


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def load_eval(evals):
    cols = ["tree", "object_id", "t", "cls", "cls_sub", "sensor_lat", "sensor_lon", "sensor_type"] + Q._CLEAN_NEEDS
    det = Q.load_detections(cols)
    keep = det["tree"] == "evaluation"
    # Clean AFTER filtering, so the per-sensor placeholder rules see evaluation rows only.
    det = Q.clean({k: v[keep] for k, v in det.items()})
    frames = [C.iteration_frame(it) for it in evals]
    common = C.Frame(float(np.median([f.lat0 for f in frames])), float(np.median([f.lon0 for f in frames])), 0.0)
    with np.errstate(invalid="ignore"):
        det["E"], det["N"], _ = common.to_enu(det["lat"], det["lon"], np.zeros(len(det["lat"])))
        det["SE"], det["SN"], _ = common.to_enu(det["sensor_lat"], det["sensor_lon"], np.zeros(len(det["lat"])))
    return det


# --------------------------------------------------------------------------- #
# K1: sensor positions -> sites
# --------------------------------------------------------------------------- #


def k1_sites(det):
    units = []
    valid = det["sensor_pos_valid"] == 1
    for key in sorted({(i, s) for i, s in zip(det["iter_key"], det["sensor_id"])}):
        m = (det["iter_key"] == key[0]) & (det["sensor_id"] == key[1])
        mv = m & valid
        typ = det["sensor_type"][m][0]
        if not mv.any():
            units.append(dict(iter_key=key[0], sensor_id=key[1], sensor_type=typ, e=np.nan, n=np.nan, rows=int(m.sum())))
            continue
        units.append(dict(iter_key=key[0], sensor_id=key[1], sensor_type=typ, e=float(np.median(det["SE"][mv])),
                          n=float(np.median(det["SN"][mv])), rows=int(m.sum())))
    placed = [u for u in units if np.isfinite(u["e"])]
    x = np.array([[u["e"], u["n"]] for u in placed])
    res, best = scan(x, range(2, 13), sample=10_000)
    k_site = None
    by_k = {}
    for k in range(2, len(x)):
        r = K.kmeans(x, k, n_init=20, seed=SEED)
        rad = max(np.hypot(*(x[r.labels == j] - r.centers[j]).T).max() for j in range(k) if (r.labels == j).any())
        by_k[k] = (r, rad)
        if rad <= SITE_RADIUS_M:
            k_site = k
            break
    return units, placed, x, res, best, k_site, by_k


def describe_clusters(placed, x, labels, centers):
    out = []
    for j in range(len(centers)):
        idx = np.flatnonzero(labels == j)
        if not len(idx):
            continue
        mem = [placed[i] for i in idx]
        ids = sorted({u["sensor_id"] for u in mem})
        mods = sorted({m for u in mem for m in u["sensor_type"].split("--")})
        out.append(dict(cluster=j, e=round(float(centers[j][0]), 1), n=round(float(centers[j][1]), 1),
                        radius_m=round(float(np.hypot(*(x[idx] - centers[j]).T).max()), 1),
                        n_sensor_ids=len(ids), sensors=" ".join(ids), modalities=" ".join(mods),
                        iterations=" ".join(sorted({u["iter_key"].split("/")[1] for u in mem}))))
    return out


# --------------------------------------------------------------------------- #
# K2: class profiles
# --------------------------------------------------------------------------- #


def k2_profiles(det, sensors_with_cls):
    grp = np.array([class_group(c, s) for c, s in zip(det["cls"], det["cls_sub"])], dtype=object)
    units, feats = [], []
    for key in sorted({(i, s) for i, s in zip(det["iter_key"], det["sensor_id"]) if s in sensors_with_cls}):
        m = (det["iter_key"] == key[0]) & (det["sensor_id"] == key[1])
        cnt = Counter(grp[m])
        f = [cnt[g] / m.sum() for g in GROUPS]
        units.append(dict(iter_key=key[0], sensor_id=key[1], rows=int(m.sum()), **{g: round(v, 4) for g, v in zip(GROUPS, f)}))
        feats.append(f)
    x = np.array(feats)
    res, best = scan(x, range(2, 9), sample=10_000)
    return units, x, res, best


# --------------------------------------------------------------------------- #
# K3: detection positions of the location sensors
# --------------------------------------------------------------------------- #


def k3_detections(det, location_sensors):
    m = det["usable_pos"] & ~det["far"] & np.isin(det["sensor_id"], location_sensors)
    idx = np.flatnonzero(m)
    key = np.array([f"{s}|{o}|{int(t // K3_BIN_S)}" for s, o, t in
                    zip(det["sensor_id"][idx], det["object_id"][idx], det["t"][idx])])
    _, first = np.unique(key, return_index=True)
    idx = idx[np.sort(first)]
    x = np.column_stack([det["E"][idx], det["N"][idx]])
    res, best = scan(x, range(2, 16), n_init=10)
    return idx, x, res, best


def k3_describe(det, idx, x, r):
    grp = np.array([class_group(c, s) for c, s in zip(det["cls"][idx], det["cls_sub"][idx])], dtype=object)
    out = []
    for j in range(len(r.centers)):
        mm = r.labels == j
        sens = Counter(det["sensor_id"][idx][mm])
        d = np.hypot(*(x[mm] - r.centers[j]).T)
        dz = det["dz_sensor"][idx][mm]
        out.append(dict(cluster=j, n_samples=int(mm.sum()), e=round(float(r.centers[j][0])), n=round(float(r.centers[j][1])),
                        radius_p50_m=round(float(np.median(d))), radius_p90_m=round(float(np.percentile(d, 90))),
                        height_med_m=round(float(np.nanmedian(dz)), 1) if np.isfinite(dz).any() else "",
                        n_sensor_ids=len(sens), sensors=" ".join(f"{s}:{c}" for s, c in sens.most_common()),
                        iterations=" ".join(f"{k.split('/')[1]}:{v}" for k, v in sorted(Counter(det["iter_key"][idx][mm]).items())),
                        top_class=" ".join(f"{g}:{100 * c / mm.sum():.0f}%" for g, c in Counter(grp[mm]).most_common(3))))
    return out


# --------------------------------------------------------------------------- #
# K4: tracks of the velocity sensors (kinematics + class evidence)
# --------------------------------------------------------------------------- #

K4_FEATURES = ["log_duration", "log_speed_med", "log_speed_p90", "straightness", "log_turn_rate",
               "frac_uav", "frac_air_vehicle", "frac_bio", "frac_unknown"]


def k4_tracks(velocity_sensors):
    rows = [r for r in csv.DictReader((C.DERIVED / "track_features.csv").open())
            if r["tree"] == "evaluation" and r["sensor_id"] in velocity_sensors]
    f = lambda r, k: float(r[k]) if r[k] != "" else np.nan  # noqa: E731
    feats = np.array([[np.log10(f(r, "duration_s")), np.log10(1 + f(r, "speed_med")), np.log10(1 + f(r, "speed_p90")),
                       f(r, "straightness"), np.log10(1 + f(r, "turn_rate_med")), f(r, "frac_uav"),
                       f(r, "frac_air_vehicle"), f(r, "frac_bio"), f(r, "frac_unknown")] for r in rows])
    ok = np.isfinite(feats).all(1)
    rows = [r for r, o in zip(rows, ok) if o]
    raw = feats[ok]
    scaler = K.robust_scaler(raw)
    x = scaler.transform(raw)
    res, best = scan(x, range(3, 11), n_init=10)
    # How much of the structure is behaviour and how much is the sensors' own labels:
    _, best_kin = scan(x[:, :5], range(2, 11), n_init=10)
    _, best_cls = scan(x[:, 5:], range(2, 11), n_init=10)
    return rows, raw, x, res, best, int((~ok).sum()), best_kin, best_cls


def k4_describe(rows, raw, r):
    out = []
    for j in range(len(r.centers)):
        mm = r.labels == j
        sub = [rows[i] for i in np.flatnonzero(mm)]
        med = np.median(raw[mm], axis=0)
        out.append(dict(cluster=j, n_tracks=int(mm.sum()),
                        duration_s=round(10 ** med[0], 1), speed_med=round(10 ** med[1] - 1, 1),
                        speed_p90=round(10 ** med[2] - 1, 1), straightness=round(med[3], 2),
                        turn_rate=round(10 ** med[4] - 1, 1),
                        frac_uav=round(float(raw[mm][:, 5].mean()), 2), frac_air_vehicle=round(float(raw[mm][:, 6].mean()), 2),
                        frac_bio=round(float(raw[mm][:, 7].mean()), 2), frac_unknown=round(float(raw[mm][:, 8].mean()), 2),
                        has_serial=sum(x["has_serial"] in ("1", "True") for x in sub),
                        sensors=" ".join(f"{s}:{c}" for s, c in Counter(x["sensor_id"] for x in sub).most_common()),
                        iterations=" ".join(f"{k.split('/')[1]}:{v}" for k, v in sorted(Counter(x["iter_key"] for x in sub).items()))))
    return out


# --------------------------------------------------------------------------- #
# Same-time intersections (not K-means): which sensors see something at the same place and time
# --------------------------------------------------------------------------- #


def same_time_intersections(det, location_sensors):
    """Per evaluation iteration: detections of different sensors within COOBS_DT_S and
    COOBS_GATE_M of each other. Uses the verified matching functions of coobservation.py,
    with NO clock calibration (offsets were measured on testing; this analysis is evaluation-only).

    Returns pooled pair counts {(a, b): [n_a, n_a_with_b]} and combo counts of the 1-s groups."""
    m = det["usable_pos"] & ~det["far"] & np.isin(det["sensor_id"], location_sensors)
    pairs = defaultdict(lambda: [0, 0])
    combos = Counter()
    combo_iters = defaultdict(set)
    n_groups = Counter()
    for it in sorted(set(det["iter_key"][m])):
        mi = m & (det["iter_key"] == it)
        order = np.argsort(det["t"][mi], kind="stable")
        t, e, n = det["t"][mi][order], det["E"][mi][order], det["N"][mi][order]
        sid = det["sensor_id"][mi][order]
        names = sorted(set(sid))
        sc = np.array([names.index(x) for x in sid], dtype=np.int64)
        hit, _ = CO.pair_hits(t, e, n, sc, len(names), gates=(COOBS_GATE_M,))
        cnt = np.bincount(sc, minlength=len(names))
        for a in range(len(names)):
            for b in range(len(names)):
                if a != b:
                    pairs[(names[a], names[b])][0] += int(cnt[a])
                    pairs[(names[a], names[b])][1] += int(hit[COOBS_GATE_M][a, b])
        tb = np.floor(t - np.floor(t.min())).astype(np.int64)
        gid = CO.group_labels(tb, e, n, sc)
        members = defaultdict(set)
        for g, c in zip(gid.tolist(), sc.tolist()):
            members[g].add(names[c])
        for ss in members.values():
            n_groups[len(ss)] += 1
            if len(ss) >= 2:
                key = " + ".join(sorted(ss))
                combos[key] += 1
                combo_iters[key].add(it.split("/")[1])
    return pairs, combos, combo_iters, n_groups


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


PLOT_SAMPLE = 8000        # K3 points kept for drawing (stratified by cluster, fixed seed)


def write_plot_data(res1, best1, k_site, r_site, placed, units, res2, best2, units2, res3, best3, det, idx3, x3,
                    res3b, best3b, res4, best4, rows4, raw4, best_kin, best_cls, pairs, combos, n_groups, loc_sensors) -> None:
    """Small JSON for tools/eval_charts.py: K scans, cluster labels and a drawing sample."""
    import json

    def scan_rows(res, best):
        return dict(k_best=best[0], rows=[dict(k=k, inertia=round(i, 3), silhouette=round(s, 4)) for k, i, s, _ in res])

    lab3 = best3[3].labels
    rng = np.random.default_rng(SEED)
    pick = []
    for j in range(best3[0]):
        members = np.flatnonzero(lab3 == j)
        take = max(1, round(PLOT_SAMPLE * len(members) / len(lab3)))
        pick.extend(np.sort(rng.choice(members, min(take, len(members)), replace=False)).tolist())
    pick = np.array(sorted(pick))
    data = dict(
        scans=dict(k1=scan_rows(res1, best1), k2=scan_rows(res2, best2), k3=scan_rows(res3, best3),
                   k3_no_egoi=scan_rows(res3b, best3b), k4=scan_rows(res4, best4)),
        k1=dict(k_site=k_site, site_radius_m=SITE_RADIUS_M,
                units=[dict(iter=u["iter_key"].split("/")[1], sensor=u["sensor_id"], type=u["sensor_type"],
                            e=round(u["e"], 1), n=round(u["n"], 1), area=int(a), site=int(s))
                       for u, a, s in zip(placed, best1[3].labels, r_site.labels)],
                no_site=sorted({u["sensor_id"] for u in units if not np.isfinite(u["e"])}),
                area_centers=np.round(best1[3].centers, 1).tolist(), site_centers=np.round(r_site.centers, 1).tolist()),
        k2=dict(groups=GROUPS, units=units2),
        k3=dict(n_total=int(len(x3)), centers=np.round(best3[3].centers).tolist(),
                sizes=np.bincount(lab3, minlength=best3[0]).tolist(),
                points=[[round(float(x3[i, 0])), round(float(x3[i, 1])), int(lab3[i]), str(det["sensor_id"][idx3][i])] for i in pick]),
        k4=dict(features=K4_FEATURES, centers_scaled=np.round(best4[3].centers, 3).tolist(),
                sil_kinematic_only=dict(k=best_kin[0], silhouette=round(best_kin[2], 4)),
                sil_class_only=dict(k=best_cls[0], silhouette=round(best_cls[2], 4)),
                tracks=[dict(sensor=r["sensor_id"], iter=r["iter_key"].split("/")[1], cluster=int(lab), speed_src=r["speed_src"],
                             f=[round(float(v), 4) for v in raw]) for r, lab, raw in zip(rows4, best4[3].labels, raw4)]),
        same_time=dict(gate_m=COOBS_GATE_M, dt_s=COOBS_DT_S, sensors=loc_sensors,
                       pairs=[dict(a=a, b=b, n_a=v[0], n_a_with_b=v[1]) for (a, b), v in sorted(pairs.items())],
                       combos=[dict(combo=k, n_groups=v) for k, v in combos.most_common()],
                       groups_by_size={str(k): v for k, v in sorted(n_groups.items())}),
    )
    (OUT / "plot_data.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")


def write_csv(name, rows):
    if rows:
        C.write_columns(OUT / name, {k: [r[k] for r in rows] for k in rows[0]}, list(rows[0]))


def main() -> int:
    evals = [it for it in C.iterations() if it.tree == "evaluation"]
    eval_keys = {it.key for it in evals}
    ladder, documented, undocumented, category, rows, extra = field_ladder(evals)
    write_csv("field_ladder.csv", ladder)
    tiers = defaultdict(list)
    for r in ladder:
        tiers[r["tier"]].append(r)
    n_doc, n_all = len(documented), len(documented) + len(undocumented)

    det = load_eval(evals)
    # Leak guard: every row that reached the analysis belongs to an evaluation iteration.
    assert len(det["iter_key"]) and set(det["iter_key"]) <= eval_keys, "non-evaluation rows reached the analysis"

    by_field = {r["field"]: r for r in ladder}

    def have(field, source="pdf"):
        return by_field[field]["sensors" if source == "pdf" else "sensors_data"].split()

    def tier_of(field):
        r = by_field[field]
        return f"tier {r['tier']} ({r['pct_pdf']:.0f} %)"

    # K1
    units, placed, x1, res1, best1, k_site, by_k = k1_sites(det)
    sites_best = describe_clusters(placed, x1, best1[3].labels, best1[3].centers)
    r_site, rad_site = by_k[k_site]
    sites_fine = describe_clusters(placed, x1, r_site.labels, r_site.centers)
    write_csv("k1_sites.csv", [dict(resolution=f"K*={best1[0]}", **s) for s in sites_best] +
              [dict(resolution=f"site K={k_site}", **s) for s in sites_fine])
    # K2: every evaluation sensor whose CSV actually has a classification column (incl. YXAB, not in the PDF)
    cls_sensors = have("classification", "data")
    units2, x2, res2, best2 = k2_profiles(det, set(cls_sensors))
    for u, lab in zip(units2, best2[3].labels):
        u["cluster"] = int(lab)
    write_csv("k2_class_profiles.csv", units2)
    # K3
    loc_sensors = have("location_x", "data")
    idx3, x3, res3, best3 = k3_detections(det, loc_sensors)
    k3 = k3_describe(det, idx3, x3, best3[3])
    write_csv("k3_detection_clusters.csv", k3)
    egoi = det["sensor_id"][idx3] == "EGOI"
    res3b, best3b = scan(x3[~egoi], range(2, 11), n_init=10)
    k3b = k3_describe(det, idx3[~egoi], x3[~egoi], best3b[3])
    write_csv("k3_detection_clusters_without_egoi.csv", k3b)
    pairs, combos, combo_iters, n_groups = same_time_intersections(det, loc_sensors)
    write_csv("same_time_pairs.csv", [dict(sensor_a=a, sensor_b=b, n_a=v[0], n_a_with_b=v[1],
                                           frac=round(v[1] / v[0], 4) if v[0] else 0.0)
                                      for (a, b), v in sorted(pairs.items()) if v[1]])
    write_csv("same_time_combos.csv", [dict(combo=k, size=k.count("+") + 1, n_groups=v,
                                            iterations=" ".join(sorted(combo_iters[k]))) for k, v in combos.most_common()])
    # K4
    vel_sensors = have("enuVelocity_eastRate", "data")
    rows4, raw4, x4, res4, best4, dropped4, best_kin, best_cls = k4_tracks(set(vel_sensors))
    assert {r["iter_key"] for r in rows4} <= eval_keys, "non-evaluation tracks reached K4"
    k4 = k4_describe(rows4, raw4, best4[3])
    write_csv("k4_track_clusters.csv", k4)
    write_csv("k4_track_assignments.csv", [dict(iter_key=r["iter_key"], sensor_id=r["sensor_id"], object_id=r["object_id"],
                                                cluster=int(lab)) for r, lab in zip(rows4, best4[3].labels)])

    write_plot_data(res1, best1, k_site, r_site, placed, units, res2, best2, units2, res3, best3, det, idx3, x3,
                    res3b, best3b, res4, best4, rows4, raw4, best_kin, best_cls, pairs, combos, n_groups, loc_sensors)

    # ---------------- report ----------------
    L = ["# Evaluation data: field ladder and tier-by-tier K-means", "",
         "Generated by `tools/eval_tiers.py`. **Evaluation iterations only** "
         f"({', '.join(it.key for it in evals)}); no testing row, label or calibration is used. No row-level data is written. "
         "Charts and a glossary: `EVAL_CLUSTERS.html` (from `tools/eval_charts.py`).", "",
         f"Sensors in the evaluation data: {n_all} ({n_doc} documented in README_Sensor.pdf: {', '.join(documented)}; "
         f"not in the PDF: {', '.join(undocumented) or '-'}).", "",
         "## 1. Field ladder (README_Sensor.pdf fields, share of the evaluation sensors that have them)", "",
         f"A tier groups the fields that the same NUMBER of sensors have; the sensors themselves can differ per field, so each "
         f"field lists its own. `PDF` counts the {n_doc} documented sensors; `data` counts the actual evaluation CSV headers of all {n_all}.", ""]
    for t in sorted(tiers):
        rs = tiers[t]
        L.append(f"**Tier {t} — {rs[0]['n_pdf']}/{n_doc} sensors ({rs[0]['pct_pdf']:.0f} %)**")
        L.append("")
        rows_t = []
        for r in rs:
            miss = r["missing"].split()
            who = ("all" if not miss else
                   (f"all except {', '.join(miss)}" if len(miss) <= n_doc // 2 else f"only {r['sensors'].replace(' ', ', ')}"))
            rows_t.append([r["field"], f"{r['n_pdf']}/{n_doc}", f"{r['n_data']}/{n_all}", who])
        L += md_table(["field", "PDF", "data", "which sensors (PDF)"], rows_t)
        L.append("")
    if extra:
        L += [f"Columns in the evaluation files that the PDF does not document for these sensors: {', '.join(extra)}.", ""]

    L += ["## 2. K-means per tier", "",
          "Each K-means uses only the fields of its tier. K is chosen by the best silhouette score "
          "(1 = perfectly separated groups, 0 = no structure).", ""]

    L += [f"### K1 — {tier_of('sensor_location_x')}: sensor_location_x / y -> where the sensors are", "",
          f"Units: one point per (iteration, sensor) = median sensor position, all three iterations in one common local frame "
          f"(metres east / north). {len(placed)} placed units; "
          f"{', '.join(sorted({u['sensor_id'] for u in units if not np.isfinite(u['e'])}))} report position 0,0 (no site).", ""]
    L += md_table(["K", "silhouette"], [[k, f"{s:.3f}"] for k, _, s, _ in res1])
    L += ["", f"Best K = {best1[0]} (silhouette {best1[2]:.3f}): the coarse areas.", ""]
    L += md_table(["area", "centre E,N (m)", "radius (m)", "distinct sensors", "sensors", "modalities", "iterations"],
                  [[s["cluster"], f"{s['e']:.0f}, {s['n']:.0f}", f"{s['radius_m']:.0f}", s["n_sensor_ids"], s["sensors"], s["modalities"], s["iterations"]] for s in sites_best])
    L += ["", f"Site resolution: the smallest K where every member is within {SITE_RADIUS_M:.0f} m of its cluster centre is K = {k_site} "
          f"(largest radius {rad_site:.1f} m; two members of one site can therefore be up to ~{2 * SITE_RADIUS_M:.0f} m apart). "
          "Sites hosting several distinct sensors are the multi-sensor sites:", ""]
    L += md_table(["site", "centre E,N (m)", "radius (m)", "distinct sensors", "sensors", "modalities", "iterations"],
                  [[s["cluster"], f"{s['e']:.0f}, {s['n']:.0f}", f"{s['radius_m']:.0f}", s["n_sensor_ids"], s["sensors"], s["modalities"], s["iterations"]]
                   for s in sorted(sites_fine, key=lambda s: (-s["n_sensor_ids"], s["cluster"]))])

    L += ["", f"### K2 — {tier_of('classification')}: + classification -> what each sensor reports", "",
          f"Units: (iteration, sensor) for the {len(cls_sensors)} evaluation sensors whose files have a classification column "
          f"({', '.join(cls_sensors)}); features = share of its rows in each class group ({', '.join(GROUPS)}; the subclass is used "
          "when it is more specific; `unknown_or_none` also holds empty classifications).", ""]
    L += md_table(["K", "silhouette"], [[k, f"{s:.3f}"] for k, _, s, _ in res2])
    L += ["", f"Best K = {best2[0]} (silhouette {best2[2]:.3f}).", ""]
    prof_rows = []
    for j in range(best2[0]):
        mem = [u for u in units2 if u["cluster"] == j]
        if not mem:
            continue
        mean = {g: np.mean([u[g] for u in mem]) for g in GROUPS}
        top = ", ".join(f"{g} {100 * v:.0f}%" for g, v in sorted(mean.items(), key=lambda kv: -kv[1]) if v >= 0.05)
        prof_rows.append([j, len(mem), " ".join(sorted({u["sensor_id"] for u in mem})), top])
    L += md_table(["group", "units", "sensors", "mean class mix"], prof_rows)

    egoi_share = egoi.mean()
    L += ["", f"### K3 — {tier_of('location_x')}: + location_x / y -> which regions the detections of several sensors cover", "",
          f"Units: detections of the {len(loc_sensors)} sensors with an object position ({', '.join(loc_sensors)}), usable rows only "
          f"(no self-reports, no targets > {Q.FAR_M / 1000:.0f} km), one sample per (sensor, object, {K3_BIN_S:.0f} s): "
          f"{len(x3):,} samples, {100 * egoi_share:.0f} % of them EGOI. Features: east, north (m). Time is not used, so a cluster is a "
          "REGION over the whole recording, not an object.", ""]
    L += md_table(["K", "silhouette"], [[k, f"{s:.3f}"] for k, _, s, _ in res3])
    L += ["", f"Best K = {best3[0]} (silhouette {best3[2]:.3f}, weak: K=11 scores almost the same). Regions:", ""]
    k3_header = ["cluster", "samples", "centre E,N (m)", "radius p50/p90 (m)", "height (m)", "distinct sensors", "sensors:samples", "iterations", "class mix"]
    k3_row = lambda c: [c["cluster"], c["n_samples"], f"{c['e']}, {c['n']}", f"{c['radius_p50_m']}/{c['radius_p90_m']}", c["height_med_m"],  # noqa: E731
                        c["n_sensor_ids"], c["sensors"], c["iterations"], c["top_class"]]
    L += md_table(k3_header, [k3_row(c) for c in sorted(k3, key=lambda c: (-c["n_sensor_ids"], -c["n_samples"]))])
    L += ["", f"Without EGOI ({(~egoi).sum():,} samples): best K = {best3b[0]} (silhouette {best3b[2]:.3f}).", ""]
    L += md_table(k3_header, [k3_row(c) for c in sorted(k3b, key=lambda c: (-c["n_sensor_ids"], -c["n_samples"]))])
    tot = sum(n_groups.values())
    multi = sum(v for k, v in n_groups.items() if k >= 2)
    L += ["", f"#### Same-time intersections (not K-means): different sensors within {COOBS_DT_S:.0f} s and {COOBS_GATE_M:.0f} m", "",
          f"The real test of whether two sensors see the same object: a detection of sensor A counts as shared with B when B has a "
          f"detection within {COOBS_DT_S:.0f} s and {COOBS_GATE_M:.0f} m. Same detections as K3, per iteration, no clock calibration "
          f"(UXYJ's ~3.5 s lag, measured on testing, is NOT applied, so UXYJ pairs are undercounted). "
          f"Detections grouped per 1-s bin (single linkage at {COOBS_GATE_M:.0f} m): {tot:,} groups, {multi:,} of them "
          f"({100 * multi / max(tot, 1):.1f} %) seen by >= 2 distinct sensors.", ""]
    L += md_table(["sensors seen together", "groups", "iterations"],
                  [[k, v, " ".join(sorted(combo_iters[k]))] for k, v in combos.most_common(15)])
    L += ["", "Strongest pairs (share of A's detections that B also has at the same time and place):", ""]
    top_pairs = sorted(((a, b, v) for (a, b), v in pairs.items() if v[1] >= 20), key=lambda x: -x[2][1] / x[2][0])[:15]
    L += md_table(["A", "B", "A detections", "with B", "share"], [[a, b, v[0], v[1], f"{100 * v[1] / v[0]:.1f} %"] for a, b, v in top_pairs])

    speed_src = defaultdict(Counter)
    for r in rows4:
        speed_src[r["sensor_id"]][r["speed_src"]] += 1
    contrib = Counter(r["sensor_id"] for r in rows4)
    L += ["", f"### K4 — {tier_of('enuVelocity_eastRate')}: + enuVelocity -> object pre-classification", "",
          f"Units: tracks (sensor objectId) of the {len(vel_sensors)} evaluation sensors with velocity columns ({', '.join(vel_sensors)}) from "
          f"`track_features.csv`, evaluation rows only: {len(rows4)} tracks ({dropped4} dropped for non-finite features). "
          f"Tracks per sensor: {', '.join(f'{s} {contrib.get(s, 0)}' for s in vel_sensors)}. "
          f"Speed source (reported velocity vs finite differences of positions; QTVS/VKYY/most STUG velocities are zero placeholders): "
          f"{'; '.join(f'{s}: ' + ', '.join(f'{k} {v}' for k, v in c.items()) for s, c in sorted(speed_src.items()))}. "
          f"Features: {', '.join(K4_FEATURES)} (robust-scaled).", ""]
    L += md_table(["K", "silhouette"], [[k, f"{s:.3f}"] for k, _, s, _ in res4])
    L += ["", f"Best K = {best4[0]} (silhouette {best4[2]:.3f}). **What drives it:** the same K-means on the 5 kinematic features alone "
          f"reaches only silhouette {best_kin[2]:.3f} (best K {best_kin[0]}); on the 4 class shares alone {best_cls[2]:.3f} (K {best_cls[0]}). "
          "So these clusters are mainly 'what the sensor itself called the track' (UAV / Air Vehicle, bird, unknown); the motion "
          "features add little separation. Treat cluster 1 as 'tracks the sensors labelled UAV / Air Vehicle', a candidate list, not a verified class.", ""]
    L += md_table(["cluster", "tracks", "duration (s)", "speed med/p90 (m/s)", "straightness", "turn (deg/s)", "UAV", "Air Vehicle", "bio", "unknown", "with serial", "sensors", "iterations"],
                  [[c["cluster"], c["n_tracks"], c["duration_s"], f"{c['speed_med']}/{c['speed_p90']}", c["straightness"], c["turn_rate"],
                    c["frac_uav"], c["frac_air_vehicle"], c["frac_bio"], c["frac_unknown"], c["has_serial"], c["sensors"], c["iterations"]] for c in k4])
    L += ["", "Class shares are the mean fraction of a track's rows that the sensor itself labelled so (not exclusive: an 'Air Vehicle' "
          "class with a 'UAV ...' subclass counts in both). No ground truth exists for evaluation, so clusters are described, not scored."]
    (OUT / "EVAL_TIERS.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"evaluation sensors: {n_doc} documented + {len(undocumented)} undocumented; tiers: {len(tiers)}")
    for t in sorted(tiers):
        print(f"  tier {t}: {tiers[t][0]['n_pdf']}/{n_doc} sensors  {', '.join(r['field'] for r in tiers[t])}")
    print(f"K1 sites: K*={best1[0]} sil={best1[2]:.3f}; site resolution K={k_site} (radius {rad_site:.1f} m)")
    print(f"K2 class profiles: {len(units2)} units, K*={best2[0]} sil={best2[2]:.3f}")
    print(f"K3 detections: {len(x3)} samples, K*={best3[0]} sil={best3[2]:.3f}; without EGOI K*={best3b[0]} sil={best3b[2]:.3f}")
    print(f"same-time: {tot} groups, {multi} multi-sensor; top: {combos.most_common(3)}")
    print(f"K4 tracks: {len(rows4)} tracks, K*={best4[0]} sil={best4[2]:.3f}; kinematic-only {best_kin[2]:.3f}, class-only {best_cls[2]:.3f}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
