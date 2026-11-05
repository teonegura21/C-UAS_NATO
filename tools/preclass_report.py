"""Owner-facing synthesis: data identification, sensor intersections, K-means pre-classification.

Reads the derived tables written by the other tools (field_intersection, sensor_inventory,
gt_associate, coobservation, cluster_tracks) plus a few quality-layer columns of
detections.csv.gz, joins the track K-means clusters with the ground-truth track labels and
writes

  data/derived/PRECLASS_REPORT.md     the report (<= 250 lines, every number computed here)
  data/derived/track_preclass.csv     one row per track_clusters row (cluster, drone_likeness, label)

drone_likeness of a cluster = share of TESTING in_fit tracks of that cluster labelled 'drone'
by gt_associate (>= 50 % of the track's rows within the gate of one Dronetag flight, one flight
>= 70 % of the matches), with a Wilson 95 % interval. in_fit excludes the ground-truth relay
sensors (VKYY / QTVS / LJJX), so the likeness is not label leakage. The value written for an
evaluation track is the cluster's testing value (a transferred prior, not a measurement); for a
testing track it is in-sample.

Run: python3 tools/preclass_report.py
"""

from __future__ import annotations

import csv
import itertools
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuas_clean as Q  # noqa: E402
import cuas_common as C  # noqa: E402

D = C.DERIVED
OUT_MD = D / "PRECLASS_REPORT.md"
OUT_CSV = D / "track_preclass.csv"

MAX_LINES = 250
Z95 = 1.96
LIFT_RULE = 3.0            # task rule: clusters with likeness > base x 3
LIFT_RELAXED = 1.5         # reported next to it because the x3 rule selects nothing
COVER_MIN = 0.05           # flight "seen" by a sensor (same threshold as gt_association.md)
GT_FEEDS = ("LJJX", "QTVS", "VKYY")
MANNED_MPS = 50.0          # cluster median speed at/above which a cluster reads as manned aircraft
MODEL_ALIAS = {"Autel Evo 2": "Autel Evo2", "DJI Mavic 3": "Mavic 3", "Parrot Anafi": "Anafi",
               "Vector": "Quantum Vector"}
FILL_MIN = 0.5             # attribute counts as usable above this share of rows
PARTIAL_MIN = 0.05


# ---------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------
def rd(name: str) -> list[dict]:
    with open(D / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fnum(x, default=float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + Z95 ** 2 / n
    mid = (p + Z95 ** 2 / (2 * n)) / den
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def pct(x: float, d: int = 1) -> str:
    return "-" if x != x else f"{100 * x:.{d}f}%"


def num(x) -> str:
    return f"{int(round(x)):,}"


def md_table(headers: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def add_table(L: list[str], headers: list[str], rows: list[list]) -> None:
    """Append a markdown table with a blank line before and after (renderers need both)."""
    if L and L[-1] != "":
        L.append("")
    L.extend(md_table(headers, rows))
    L.append("")


def parse_md_table(text: str, heading: str) -> list[dict]:
    """First markdown table after the first line matching `heading` (regex)."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if re.search(heading, ln)), None)
    if start is None:
        return []
    i = start
    while i < len(lines) and not lines[i].startswith("|"):
        i += 1
    if i >= len(lines):
        return []
    header = [c.strip() for c in lines[i].strip().strip("|").split("|")]
    rows = []
    for ln in lines[i + 2:]:
        if not ln.startswith("|"):
            break
        cells = [c.strip().replace("**", "") for c in ln.strip().strip("|").split("|")]
        rows.append(dict(zip(header, cells)))
    return rows


def sid(sensor_key: str) -> str:
    return sensor_key.split("_")[0]


def short_combo(s: str) -> str:
    return s.replace("|", "+")


# ---------------------------------------------------------------------------------------------
# tracks: join clusters with labels
# ---------------------------------------------------------------------------------------------
class Tracks:
    def __init__(self):
        cl = rd("track_clusters.csv")
        lb = {(x["iter_key"], x["sensor_key"], x["object_id"]): x for x in rd("track_labels.csv")}
        tf = {(x["iter_key"], x["sensor_key"], x["object_id"]): x for x in rd("track_features.csv")}
        self.rows = cl
        self.n = len(cl)
        keys = [(x["iter_key"], x["sensor_key"], x["object_id"]) for x in cl]
        self.dup_keys = self.n - len(set(keys))
        self.it = np.array([x["iter_key"] for x in cl], dtype=object)
        self.tree = np.array([x["tree"] for x in cl], dtype=object)
        self.sensor = np.array([sid(x["sensor_key"]) for x in cl], dtype=object)
        self.in_fit = np.array([x["in_fit"] == "1" for x in cl])
        self.testing = self.tree == "testing"
        self.k_star = np.array([int(x["cluster_kstar"]) for x in cl])
        self.k6 = np.array([int(x["cluster_k6"]) for x in cl])
        self.label = np.array([lb[k]["label"] if (t == "testing" and k in lb) else ""
                               for k, t in zip(keys, self.tree)], dtype=object)
        self.unlabelled_testing = int(np.sum(self.testing & (self.label == "")))
        self.side = np.array([lb[k]["side"] if k in lb and self.label[i] == "drone" else ""
                              for i, k in enumerate(keys)], dtype=object)
        self.model = np.array([MODEL_ALIAS.get(lb[k]["drone_name"], lb[k]["drone_name"])
                               if k in lb and self.label[i] == "drone" else ""
                               for i, k in enumerate(keys)], dtype=object)
        self.drone = self.label == "drone"
        self.fit_test = self.testing & self.in_fit
        self.labels_in_file = sum(1 for k in lb if k[0].startswith("testing"))
        feats = ["duration_s", "dz_med", "speed_med", "speed_p90", "straightness", "range_med",
                 "det_conf_med"]
        self.feat = {f: np.array([fnum(tf[k][f]) if k in tf else np.nan for k in keys]) for f in feats}
        self.cls_top = np.array([tf[k]["cls_top"] if k in tf else "" for k in keys], dtype=object)
        self.k_star_ids = sorted(set(self.k_star[self.in_fit].tolist()))
        self.k6_ids = sorted(set(self.k6[self.in_fit].tolist()))

    def clusters(self, kcol: np.ndarray, ids: list[int]) -> dict[int, dict]:
        """Per-cluster statistics. base = drone share over all testing in_fit tracks."""
        base_n = int(self.fit_test.sum())
        base_k = int((self.fit_test & self.drone).sum())
        base = base_k / base_n
        out = {}
        for c in ids:
            m = kcol == c
            ft = m & self.fit_test
            n_fit = int(ft.sum())
            k = int((ft & self.drone).sum())
            lo, hi = wilson(k, n_fit)
            lik = k / n_fit if n_fit else float("nan")
            mix = Counter(self.label[ft].tolist())
            dr = ft & self.drone
            models = Counter(self.model[dr].tolist()).most_common(3)
            cls = Counter(self.cls_top[m & self.in_fit].tolist())
            nc = max(int((m & self.in_fit).sum()), 1)
            out[c] = dict(
                n_test=int((m & self.testing).sum()), n_eval=int((m & ~self.testing).sum()),
                n_fit_test=n_fit, k=k, lik=lik, lo=lo, hi=hi, lift=lik / base if base else float("nan"),
                mix=mix, blue=int((dr & (self.side == "BLUE")).sum()), red=int((dr & (self.side == "RED")).sum()),
                models=models, base=base,
                animal_share=(cls["Animal"] + cls["Bird"] + cls["Land Vehicle"]) / nc,
                med={f: float(np.nanmedian(v[m & self.in_fit])) for f, v in self.feat.items()},
            )
        return out

    @staticmethod
    def reading(s: dict) -> str:
        if s["med"]["speed_med"] >= MANNED_MPS:
            return "manned-aircraft-like (fast)"
        if s["lik"] != s["lik"]:
            return "n/a"
        base = s["base"]
        if s["lik"] >= LIFT_RULE * base and s["lo"] > base:
            return "drone-like"
        if s["lo"] > base and s["lik"] >= LIFT_RELAXED * base:
            return "drone-enriched"
        if s["lo"] > base:
            return "weakly enriched"
        if s["hi"] < base:
            return "clutter-like (depleted)"
        return "indeterminate (CI covers base)"


def pr_by_sensor(tr: Tracks, kcol: np.ndarray, mult: float) -> dict:
    """Precision / recall per sensor when 'cluster likeness > mult x base' is the drone call.
    Returns in-sample and leave-one-iteration-out (LOIO) counts. Testing in_fit tracks only."""
    m = tr.fit_test
    cl, dr, sen, it = kcol[m], tr.drone[m], tr.sensor[m], tr.it[m]

    def select(mask):
        base = dr[mask].mean()
        sel = set()
        for c in np.unique(cl[mask]):
            mc = mask & (cl == c)
            if dr[mc].mean() > mult * base:
                sel.add(int(c))
        return sel

    res = {}
    allm = np.ones(len(cl), dtype=bool)
    sel_in = select(allm)
    pred_in = np.isin(cl, list(sel_in)) if sel_in else np.zeros(len(cl), dtype=bool)
    pred_lo = np.zeros(len(cl), dtype=bool)
    for h in sorted(set(it.tolist())):
        sel = select(it != h)
        if sel:
            pred_lo |= (it == h) & np.isin(cl, list(sel))
    for s in sorted(set(sen.tolist())) + ["ALL"]:
        ms = np.ones(len(cl), dtype=bool) if s == "ALL" else sen == s
        res[s] = dict(n=int(ms.sum()), pos=int((ms & dr).sum()),
                      pred_in=int((ms & pred_in).sum()), tp_in=int((ms & pred_in & dr).sum()),
                      pred_lo=int((ms & pred_lo).sum()), tp_lo=int((ms & pred_lo & dr).sum()))
    return dict(sel=sorted(sel_in), per=res)


# ---------------------------------------------------------------------------------------------
# detections: data-quality counts, far-row physics, serial sharing
# ---------------------------------------------------------------------------------------------
def detection_facts() -> dict:
    det = Q.load_clean(["tree", "serial", "model", "t"])
    sid_ = det["sensor_id"]
    kind = det["meas_kind"]
    is_pos = np.isin(kind, Q.POSITION_KINDS)
    sp = np.hypot(det["ve"], det["vn"])
    facts = {"n_rows": len(sid_), "n_pos": int(is_pos.sum())}
    per = {}
    for s in sorted(set(sid_.tolist())):
        m = sid_ == s
        mp = m & is_pos
        npos = int(mp.sum())
        if not npos:
            continue
        rawv = mp & np.isfinite(det["ve"]) & np.isfinite(det["vn"])
        per[s] = dict(
            n_pos=npos, null_island=int((mp & det["null_island"]).sum()),
            self_report=int((mp & det["self_report"]).sum()), far=int((mp & det["far"]).sum()),
            alt0=int((mp & (det["alt"] == 0)).sum()),
            vzero=int((rawv & np.isnan(det["ve_c"])).sum()), vfin=int(rawv.sum()),
            vu_dead=int((m & np.isfinite(det["vu"]) & ~np.isfinite(det["vu_c"])).sum()),
            usable=int((mp & det["usable_pos"] & ~det["far"]).sum()),
        )
    facts["per"] = per
    far = det["far"]
    farinfo = {}
    for s in sorted(set(sid_[far].tolist())):
        m = far & (sid_ == s)
        v = sp[m][np.isfinite(sp[m])]
        a = det["alt"][m][np.isfinite(det["alt"][m])]
        farinfo[s] = dict(n=int(m.sum()), speed_med=float(np.median(v)) if len(v) else float("nan"),
                          alt_med=float(np.median(a)) if len(a) else float("nan"),
                          fast=float((v >= MANNED_MPS).mean()) if len(v) else float("nan"))
    facts["far"] = farinfo
    # serial sharing: normalise by containment of the core serial
    ser = det["serial"]
    ms = ser != ""
    pair_rows = Counter(zip(sid_[ms].tolist(), ser[ms].tolist()))
    pair_iters = defaultdict(set)
    for s, v, k in zip(sid_[ms].tolist(), ser[ms].tolist(), det["iter_key"][ms].tolist()):
        pair_iters[(s, v)].add(k)
    pair_model = defaultdict(Counter)
    for s, v, mo in zip(sid_[ms].tolist(), ser[ms].tolist(), det["model"][ms].tolist()):
        if mo:
            pair_model[v][mo] += 1
    variants = sorted({v for _, v in pair_rows}, key=lambda x: (len(x), x))
    parent = {v: v for v in variants}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in itertools.combinations(variants, 2):
        if len(a) >= 10 and a in b:
            parent[find(b)] = find(a)
    groups = defaultdict(lambda: dict(variants=set(), sensors=Counter(), iters=set(), models=Counter()))
    for (s, v), n in pair_rows.items():
        g = groups[find(v)]
        g["variants"].add(v)
        g["sensors"][s] += n
        g["iters"] |= pair_iters[(s, v)]
        g["models"].update(pair_model[v])
    facts["serial_groups"] = {k: groups[k] for k in sorted(groups)}
    facts["serial_rows"] = int(ms.sum())
    return facts


# ---------------------------------------------------------------------------------------------
# report sections
# ---------------------------------------------------------------------------------------------
def sec_identification(L: list[str], inv: list[dict], sem: list[dict], fam: list[dict]) -> dict:
    L.append("## 1. Data identification")
    sensors = sorted({x["sensor_id"] for x in inv})
    frac = {x["sensor_id"]: x for x in sem if x["view"] == "fraction"}
    per = {}
    for s in sensors:
        rows = [x for x in inv if x["sensor_id"] == s]
        iters = {x["iter_key"] for x in rows}
        per[s] = dict(type=rows[0]["sensor_type"], rows=sum(int(x["rows"]) for x in rows),
                      kinds="/".join(sorted({x["meas_kind"] for x in rows})),
                      t=len({i for i in iters if i.startswith("testing")}),
                      e=len({i for i in iters if i.startswith("evaluation")}),
                      real_range=any(x["has_real_range"] == "1" for x in rows))
    tok = [("pos", "object_position"), ("alt", "object_altitude"), ("az", "azimuth"), ("az45", "azimuth_coarse"),
           ("el", "elevation"), ("vh", "velocity_horizontal"), ("vz", "velocity_vertical"),
           ("perr", "position_error"), ("cls", "classification"), ("sub", "classification_subclass"),
           ("dconf", "detection_confidence"), ("rcs", "rcs"), ("freq", "frequency"),
           ("amp", "signal_amplitude")]
    id_attrs = ("entity_id", "serial_number", "model", "drone_mac", "pilot_mac")
    table = []
    for s in sensors:
        f = frac[s]
        parts = []
        for name, attr in tok:
            v = fnum(f.get(attr))
            if v >= FILL_MIN:
                parts.append(name)
            elif v >= PARTIAL_MIN:
                parts.append(f"{name}~{100 * v:.0f}%")
        idv = max(fnum(f.get(a), 0.0) for a in id_attrs)
        if idv >= FILL_MIN:
            parts.append("id")
        if fnum(f.get("sensor_position")) < FILL_MIN:
            parts.append("NO-SITE")
        p = per[s]
        table.append([s, p["type"], p["kinds"], " ".join(parts), num(p["rows"]), f"{p['t']}T+{p['e']}E"])
    tot_rows = sum(p["rows"] for p in per.values())
    n_pos = sum(1 for s in sensors if per[s]["kinds"].startswith("pos"))
    its = {x["iter_key"] for x in inv}
    KEY.update(n_sensors=len(sensors), n_rows=tot_rows, n_pos=n_pos, n_files=len(inv), n_iters=len(its),
               n_test_it=sum(i.startswith("testing") for i in its), n_eval_it=sum(i.startswith("evaluation") for i in its))
    L.append(f"{len(sensors)} sensor ids, {len(inv)} files, {num(tot_rows)} rows in {len({x['iter_key'] for x in inv})} "
             f"iterations (T = testing with GT, E = evaluation). Kinds: pos3d/pos2d = absolute position "
             f"({n_pos} sensors; pos2d_effective = z column all 0), bearing_el = az+el, bearing = az only. Fields usable on >= {FILL_MIN:.0%} of rows "
             f"(`~N%` = partly usable; placeholders and constants removed): pos position, alt altitude, az/el bearing, "
             f"az45 45-degree sector, vh/vz velocity, perr per-detection position error, cls/sub class/subclass, "
             f"dconf varying detection confidence, id serial/model/MAC/entity id, NO-SITE = no sensor position.")
    L.append("")
    add_table(L, ["sensor", "type", "kind", "usable fields", "rows", "iters"], table)
    n_range = [s for s in sensors if per[s]["real_range"]]
    L.append("")
    # README vs data
    spec = {x["sensor_id"]: x for x in rd("field_matrix_spec.csv")}
    data = {x["sensor_id"]: x for x in rd("field_matrix_data.csv")}
    spec_cols = [c for c in next(iter(spec.values())) if c not in ("sensor_id", "category")]
    data_cols = [c[: -len("__present")] for c in next(iter(data.values())) if c.endswith("__present")]
    absent, undoc, empty = [], [], []
    for s in sensors:
        if s not in spec:
            continue
        for c in data_cols:
            doc = spec[s].get(c, "0") == "1"
            pres = data[s][c + "__present"] == "1"
            if doc and not pres:
                absent.append(f"{s}.{c}")
            if pres and not doc:
                undoc.append(f"{s}.{c}")
            if doc and pres and fnum(data[s][c]) == 0:
                empty.append(f"{s}.{c}")
    undocumented_ids = [x["sensor_id"] for x in fam if x["pdf_category"] == "undocumented"]
    conc = {"location_z": "object_altitude", "enuVelocity_upRate": "velocity_vertical",
            "enuVelocity_eastRate": "velocity_horizontal", "detectionConfidence": "detection_confidence",
            "classification": "classification"}
    unusable = defaultdict(list)
    for s in sensors:
        for col, attr in conc.items():
            if fnum(data[s].get(col)) >= FILL_MIN and fnum(frac[s].get(attr)) < FILL_MIN:
                unusable[attr].append(s)
    fi = (D / "field_intersection.md").read_text(encoding="utf-8")
    ids_line = next((ln for ln in fi.splitlines() if ln.startswith("- Ids:")), "")
    ids_note = re.search(r"the PDF's WCFX is WVFX[^;.]*", ids_line)
    L.append("README_Sensor.pdf versus data (field_matrix_spec / field_matrix_data / field_matrix_semantic):")
    const_range = [s for s in sensors if fnum(data[s].get("rangeBearing_range")) >= FILL_MIN and not per[s]["real_range"]]
    L.append(f"- Ids: {len(undocumented_ids)} data sensors are not in the PDF ({', '.join(undocumented_ids)}); "
             f"{ids_note.group(0) if ids_note else 'PDF ids differ'}. "
             f"No sensor has a real range ({len(n_range)} sensors with > 1 distinct rangeBearing_range); "
             f"constant range placeholders on {', '.join(const_range)}.")
    L.append(f"- {len(absent)} documented (sensor, field) pairs are absent from the CSVs ({', '.join(absent)}); "
             f"{len(undoc)} CSV columns are undocumented ({', '.join(undoc)}); {len(empty)} documented fields are empty on every row.")
    L.append("- Present but not usable (> 50 % of raw rows filled, < 50 % usable after placeholder rules): "
             + "; ".join(f"{a}: {', '.join(v)}" for a, v in sorted(unusable.items())) + ".")
    return per


def sec_field_intersections(L: list[str]) -> None:
    conc = rd("field_concepts.csv")
    excl = {"time", "track_id", "node_id", "report_id", "sensor_position"}

    def disc(x):
        return [a for a in x["attributes"].split("|") if a not in excl]

    sem = [x for x in conc if x["level"] == "semantic"]
    best = {}
    for x in sem:
        key = (int(x["n_sensors"]), int(x["n_discriminative"]))
        if key not in best or int(x["total_rows"]) > int(best[key]["total_rows"]):
            best[key] = x
    pareto = [v for k, v in best.items()
              if k[1] >= 1 and k[0] >= 2 and not any((k2[0] >= k[0] and k2[1] >= k[1]) and k2 != k for k2 in best)]
    pareto.sort(key=lambda x: (-int(x["n_sensors"]), -int(x["n_discriminative"])))
    L.append("### 2a. Field intersections (field_concepts.csv: sensors sharing usable attribute sets)")
    L.append("Pareto front of semantic concepts (no set has both more sensors and more discriminative attributes; "
             "time/track_id/node_id/report_id/sensor_position excluded):")
    L.append("")
    rows = [[x["n_sensors"], x["n_discriminative"], num(int(x["total_rows"])), short_combo(x["sensors"]),
             ", ".join(disc(x))] for x in pareto]
    add_table(L, ["sensors", "attrs", "rows", "which sensors", "attribute set"], rows)
    L.append("")
    area = sorted([x for x in sem if int(x["n_sensors"]) >= 3 and int(x["n_discriminative"]) >= 4],
                  key=lambda x: (-int(x["area_discriminative"]), -int(x["n_sensors"]), x["sensors"]))[:2]
    lv = []
    for level in ("data", "spec"):
        xs = sorted([x for x in conc if x["level"] == level],
                    key=lambda x: (-int(x["area_discriminative"]), -int(x["n_sensors"]), x["sensors"]))
        lv.append(f"{level}-level raw columns: {xs[0]['n_sensors']} sensors x {xs[0]['n_discriminative']} columns "
                  f"({short_combo(xs[0]['sensors'])})")
    KEY["top_area"] = (int(area[0]["n_sensors"]), int(area[0]["n_discriminative"]), short_combo(area[0]["sensors"]))
    L.append("Largest by area: " + "; ".join(f"{x['n_sensors']} sensors x {x['n_discriminative']} attrs "
                                             f"({short_combo(x['sensors'])})" for x in area)
             + ". Raw: " + "; ".join(lv) + ".")


def sec_devices(L: list[str], per: dict, facts: dict) -> None:
    inv = rd("sensor_inventory.csv")
    sites = rd("sensor_sites.csv")
    fl = rd("fusion_links.csv")
    L.append("")
    L.append("### 2b. Device / site intersections")
    byit = defaultdict(list)
    for x in sites:
        byit[x["iter_key"]].append(x)
    pair_cnt = Counter()
    n_sites = n_multi = n_mm = 0
    biggest = []
    for it in sorted(byit):
        ss = byit[it]
        multi = [x for x in ss if int(x["n_sensor_ids"]) >= 2]
        n_sites += len(ss)
        n_multi += len(multi)
        n_mm += sum(x["multi_modal"] == "1" for x in multi)
        for x in multi:
            ids = sorted({sid(k) for k in x["sensor_keys"].split("|")})
            biggest.append((len(ids), it.replace("evaluation/", "E ").replace("testing/", "T "), "+".join(ids)))
            for a_, b_ in itertools.combinations(ids, 2):
                pair_cnt[(a_, b_)] += 1
    biggest.sort(key=lambda r: (-r[0], r[1]))
    intra = [float(x["max_intra_m"]) for x in sites if int(x["n_sensor_ids"]) >= 2]
    L.append(f"Sites (sensor files within 30 m inside one iteration): {n_sites} sites in {len(byit)} iterations, {n_multi} hold >= 2 "
             f"sensor ids (max spread {max(intra):.0f} m), {n_mm} are multi-modal (union of modalities larger than any member's). "
             f"Biggest: " + "; ".join(f"{n} sensors at {it}: {c}" for n, it, c in biggest[:3]) + ". Most often co-located pairs "
             "(iterations): " + ", ".join(f"{a_}+{b_} {n}" for (a_, b_), n in pair_cnt.most_common(8)) + ".")
    mm_keys = sorted({x["sensor_key"] for x in inv if "--" in x["sensor_type"]})
    mm_rows = {k: sum(int(x["rows"]) for x in inv if x["sensor_key"] == k) for k in mm_keys}
    mob = sorted({x["sensor_key"] for x in inv if x["mobile"] == "1"})
    cross = [x for x in fl if sid(x["referenced_sensor_key"].split(" ")[0]) != sid(x["fusing_sensor_key"])]
    emit = sorted({sid(x["fusing_sensor_key"]) for x in fl})
    L.append(f"Multi-modal devices (one node, one stream, no sub-source tag): "
             + ", ".join(f"{k.split('_')[0]} ({num(v)} rows)" for k, v in mm_rows.items())
             + f". Mobile (position spread > 10 m) sensor keys: {', '.join(sorted({k.split('_')[0] for k in mob}))}. "
             f"Fusion: {len(fl)} reference rows from {', '.join(emit)}, {len(cross)} reference another sensor "
             f"=> no pre-fused cross-sensor tracks exist.")
    sg = facts["serial_groups"]
    multi_ser = {k: g for k, g in sg.items() if len(g["sensors"]) >= 2}
    KEY.update(n_sites=n_sites, n_multi_sites=n_multi, n_mm_sites=n_mm, n_serials=len(sg), n_serials_multi=len(multi_ser),
               serial_sensors="/".join(sorted({s_ for g in multi_ser.values() for s_ in g["sensors"]})),
               serial_max=max((len(g["sensors"]) for g in sg.values()), default=0))
    L.append(f"Shared drone serials ({num(facts['serial_rows'])} rows carry a serial; variants merged when the shorter "
             f"is contained in the longer, >= 10 chars): {len(sg)} distinct devices, {len(multi_ser)} seen by >= 2 sensors.")
    rows = []
    for k, g in sorted(multi_ser.items(), key=lambda kv: (-len(kv[1]["sensors"]), kv[0])):
        mo = g["models"].most_common(1)
        rows.append([k, len(g["variants"]), "+".join(sorted(g["sensors"])), num(sum(g["sensors"].values())),
                     len(g["iters"]), mo[0][0] if mo else "-"])
    add_table(L, ["core serial", "variants", "sensors", "rows", "iters", "model"], rows)


def sec_observation(L: list[str]) -> dict:
    fc = rd("flight_coverage.csv")
    L.append("")
    L.append("### 2c. Observation intersections (which drones are seen by which sensors)")
    flights = defaultdict(dict)
    meta = {}
    for x in fc:
        k = (x["iter_key"], x["flight_key"])
        flights[k][sid(x["sensor_key"])] = fnum(x["coverage_frac"])
        meta[k] = (x["side"], x["drone_name"], int(x["n_gt_seconds"]))
    tot_s = sum(v[2] for v in meta.values())
    nf = len(flights)
    seen_all = {k: sorted(s for s, v in d.items() if v >= COVER_MIN) for k, d in flights.items()}
    seen_nf = {k: [s for s in v if s not in GT_FEEDS] for k, v in seen_all.items()}
    hist = Counter(len(v) for v in seen_nf.values())
    cnt_all = Counter(len(v) for v in seen_all.values())
    L.append(f"{nf} Dronetag flights ({tot_s:,} GT flight-seconds), seen = coverage >= {COVER_MIN:.0%} of the flight's GT seconds. "
             f"Distinct sensors per flight excluding the GT feeds (LJJX/QTVS/VKYY), flights: "
             + ", ".join(f"{k}: {hist[k]}" for k in sorted(hist)) + f". With the feeds: max {max(cnt_all)}.")
    best = sorted(seen_nf, key=lambda k: (-len(seen_nf[k]), k))[:5]
    L.append("Best-covered flights (no feeds): " + "; ".join(
        f"{k[0].split('/')[-1]} {meta[k][0][0]}:{meta[k][1]} = {'+'.join(seen_nf[k])}" for k in best) + ".")
    pres = defaultdict(lambda: [0, 0])
    for k, d in flights.items():
        for s, v in d.items():
            pres[s][0] += 1
            pres[s][1] += v >= COVER_MIN
    L.append("Flights seen / flights in iterations where the sensor ran: "
             + ", ".join(f"{s} {a[1]}/{a[0]}" for s, a in sorted(pres.items(), key=lambda kv: (-kv[1][1], kv[0]))
                         if a[1] > 0) + ".")
    KEY.update(n_flights=nf, flights_two=sum(v for k, v in hist.items() if k >= 2), flights_zero=hist.get(0, 0))
    ct = rd("codetection_sets.csv")
    var = {v: {x["combo"]: x for x in ct if x["variant"] == v} for v in ("all", "no_gt_feed")}
    parts = []
    for variant, title in (("all", "with the GT feeds (leaks if scored against GT)"), ("no_gt_feed", "without them")):
        rows = list(var[variant].values())
        s1, s2, s3 = (sum(int(x["n_seconds"]) for x in rows if int(x["size"]) >= k) for k in (1, 2, 3))
        parts.append(f"{title}: {s1:,} ({pct(s1 / tot_s)}) have >= 1 sensor, {s2:,} ({pct(s2 / tot_s)}) >= 2, {s3:,} ({pct(s3 / tot_s)}) >= 3")
    L.append("")
    L.append("Co-detection (GT flight-seconds with a matched detection from distinct sensors, of "
             f"{tot_s:,}): " + "; ".join(parts) + ". Biggest exact combinations (seconds; no-feed counts can exceed 'all' because removing a feed sensor merges supersets), then the largest sets:")
    pick = []
    nofeed = sorted([x for x in var["no_gt_feed"].values() if int(x["size"]) >= 2], key=lambda x: (-int(x["n_seconds"]), x["combo"]))
    withfeed = sorted([x for x in var["all"].values() if int(x["size"]) >= 2 and any(f in x["combo"].split("|") for f in GT_FEEDS)],
                      key=lambda x: (-int(x["n_seconds"]), x["combo"]))
    for x in nofeed[:5] + withfeed[:2]:
        pick.append(x["combo"])
    for v in ("no_gt_feed", "all"):
        big = max(var[v].values(), key=lambda x: (int(x["size"]), int(x["n_seconds"]), x["combo"]))
        pick.append(big["combo"])
    rows = []
    for c in dict.fromkeys(pick):
        a_, n_ = var["all"].get(c), var["no_gt_feed"].get(c)
        ref = n_ or a_
        rows.append([short_combo(c), ref["size"], a_["n_seconds"] if a_ else "-", n_["n_seconds"] if n_ else "n/a (feed)",
                     ref["n_flights"], ref["n_seconds_atleast"] if ref is n_ else a_["n_seconds_atleast"]])
    add_table(L, ["combo", "size", "seconds (all)", "seconds (no feed)", "flights", "seconds incl. supersets"], rows)
    # coobservation without GT
    cs = rd("coobs_sets.csv")
    L.append("")
    L.append("GT-free co-observation (coobs_sets.csv; 1-s bins, 50 m, single linkage, GT feeds removed; time offsets applied):")
    agg = {}
    for tree in ("testing", "evaluation"):
        rows = [x for x in cs if x["tree"] == tree and x["variant"] == "no_gt_feed"]
        n1 = sum(int(x["n_groups"]) for x in rows if x["size"] == "1")
        n2 = sum(int(x["n_groups"]) for x in rows if int(x["size"]) >= 2)
        g1 = sum(int(x["n_groups_on_gt"] or 0) for x in rows if x["size"] == "1")
        g2 = sum(int(x["n_groups_on_gt"] or 0) for x in rows if int(x["size"]) >= 2)
        agg[tree] = (n1, n2, g1, g2)
    t1, t2, g1, g2 = agg["testing"]
    e1, e2, _, _ = agg["evaluation"]
    p1, p2 = g1 / t1, g2 / t2
    p_ge2_given_gt = g2 / (g1 + g2)
    L.append(f"- VALIDATION (testing): P(group lies on a GT drone | seen by >= 2 sensors) = {pct(p2)} ({g2:,}/{t2:,}) versus "
             f"{pct(p1)} ({g1:,}/{t1:,}) for single-sensor groups: lift x{p2 / p1:.1f}; coverage P(>= 2 sensors | on GT) = "
             f"{pct(p_ge2_given_gt)}. Lower bounds: only the {nf} tagged flights count as drones.")
    L.append(f"- Evaluation (no GT): {e2:,} of {e1 + e2:,} groups ({pct(e2 / (e1 + e2))}) are multi-sensor, versus "
             f"{pct(t2 / (t1 + t2))} on testing.")
    tcs = {x["combo"]: x for x in cs if x["tree"] == "testing" and x["variant"] == "no_gt_feed" and int(x["size"]) >= 2}
    ecs = {x["combo"]: x for x in cs if x["tree"] == "evaluation" and x["variant"] == "no_gt_feed" and int(x["size"]) >= 2}
    allc = set(tcs) | set(ecs)
    score = lambda c: -(int(tcs[c]["n_groups"]) if c in tcs else 0) - (int(ecs[c]["n_groups"]) if c in ecs else 0)
    rows = []
    for c in sorted(allc, key=lambda c: (score(c), c))[:9]:
        t = tcs.get(c)
        e = ecs.get(c)
        rows.append([short_combo(c), len(c.split("|")),
                     num(int(t["n_groups"])) if t else "0", pct(fnum(t["frac_groups_on_gt_drone"])) if t else "-",
                     num(int(e["n_groups"])) if e else "0"])
    add_table(L, ["combo", "size", "testing groups", "on GT drone", "evaluation groups"], rows)
    hi = sorted([x for x in tcs.values() if int(x["n_groups"]) >= 100 and fnum(x["frac_groups_on_gt_drone"]) == fnum(x["frac_groups_on_gt_drone"])],
                key=lambda x: (-fnum(x["frac_groups_on_gt_drone"]), x["combo"]))[:3]
    L.append("Highest GT share (testing, >= 100 groups): " + "; ".join(
        f"{short_combo(x['combo'])} {pct(fnum(x['frac_groups_on_gt_drone']))} ({x['n_groups']})" for x in hi) + ".")
    KEY.update(lift=p2 / p1, p1=p1, p2=p2, top_combos=[short_combo(x["combo"]) for x in nofeed[:3]],
               hi_combos=[short_combo(x["combo"]) for x in hi[:2]])
    return dict(lift=p2 / p1, p2=p2, p1=p1, nf_hist=hist, pres=pres, n_flights=nf)


def sec_kmeans(L: list[str], tr: Tracks, fam: list[dict]) -> dict:
    L.append("")
    L.append("## 3. K-means")
    fi = (D / "field_intersection.md").read_text(encoding="utf-8")
    kt = (D / "kmeans_tracks.md").read_text(encoding="utf-8")
    fsil = {int(r["K"]): (fnum(r["silhouette"]), r.get("") or "") for r in parse_md_table(fi, r"^### 3\. Families")}
    kbest = max(fsil, key=lambda k: fsil[k][0]) if fsil else None
    groups = defaultdict(list)
    for x in fam:
        groups[x["family_k_best"]].append(x)
    L.append("### 3a. Sensor families (K-means on 59 semantic field fractions per sensor)")
    L.append(f"Silhouette by K: " + ", ".join(f"K={k} {v[0]:.3f}" for k, v in sorted(fsil.items()))
             + f". Best K = {kbest} ({fsil[kbest][0]:.3f}); the PDF has 6 categories (K=6 silhouette {fsil[6][0]:.3f}). "
             "Families (by field profile, not by sensor type):")
    L.append("")
    rows = []
    for g in sorted(groups, key=lambda g: (-len(groups[g]), g)):
        xs = groups[g]
        cats = Counter(x["pdf_category"] for x in xs)
        rows.append([f"F{g}", len(xs), ", ".join(x["sensor_id"] for x in xs),
                     ", ".join(f"{c} {n}" for c, n in sorted(cats.items()))])
    add_table(L, ["family", "n", "sensors", "PDF categories inside"], rows)

    # track clusters
    scan = parse_md_table(kt, r"^## K scan")
    ks_m = re.search(r"K\* = \*\*(\d+)\*\*[^.]*?silhouette ([0-9.]+)", kt)
    elbow = re.search(r"Elbow[^\n]*\): K = (\d+)", kt)
    ks = len(tr.k_star_ids)
    sil = {int(r["K"]): fnum(r["silhouette"]) for r in scan}
    L.append("")
    L.append("### 3b. Track clusters (cluster_tracks.py: 8 kinematic/height features per track, one feature space for all sensors)")
    n_tr = tr.n
    n_fit = int(tr.in_fit.sum())
    L.append(f"{num(n_tr)} tracks ({int(tr.testing.sum()):,} testing, {int((~tr.testing).sum()):,} evaluation), {num(n_fit)} in the "
             f"fit (relay sensors and non-finite features excluded). Silhouette by K: "
             + ", ".join(f"{k}: {v:.3f}" for k, v in sorted(sil.items()) if k <= 8)
             + f"; K* = {ks} (argmax over K=3..12{', silhouette ' + ks_m.group(2) if ks_m else ''}), elbow K = "
             f"{elbow.group(1) if elbow else '?'}. Silhouettes of 0.17-0.26 mean a continuum, not separated groups. "
             f"Join: {tr.labels_in_file:,} testing label rows; {tr.unlabelled_testing} testing tracks without a label row.")
    base_n = int(tr.fit_test.sum())
    base_k = int((tr.fit_test & tr.drone).sum())
    blo, bhi = wilson(base_k, base_n)
    L.append(f"Base rate: {base_k} of {base_n:,} testing in_fit tracks are labelled 'drone' = {pct(base_k / base_n, 2)} "
             f"[{pct(blo, 2)}, {pct(bhi, 2)}]; {int((tr.fit_test & (tr.label == 'mixed')).sum())} 'mixed', "
             f"{int((tr.fit_test & (tr.label == 'unmatched')).sum()):,} 'unmatched'. Rule: drone-like = likeness > base x {LIFT_RULE:g} "
             f"(= {pct(LIFT_RULE * base_k / base_n)}).")
    out = {}
    for name, kcol, ids in (("K*", tr.k_star, tr.k_star_ids), (f"K=6", tr.k6, tr.k6_ids)):
        st = tr.clusters(kcol, ids)
        out[name] = st
        L.append("")
        L.append(f"**{name}** (testing/eval = tracks per tree; drone, mixed, unmatched = testing in_fit tracks; likeness = drone / testing in_fit "
                 "tracks, Wilson 95 %; lift = likeness / base; speed, dz = cluster medians; reading = mechanical rule on lift and interval):")
        rows = []
        for c in ids:
            s = st[c]
            mix = s["mix"]
            m = s["med"]
            md = ", ".join(f"{mo} {n}" for mo, n in s["models"]) or "-"
            rows.append([c, num(s["n_test"]), num(s["n_eval"]), mix["drone"],
                         f"{mix['mixed']}/{num(mix['unmatched'])}/{mix['no_usable_rows']}",
                         f"{pct(s['lik'], 2)} [{pct(s['lo'], 2)}, {pct(s['hi'], 2)}]", f"{s['lift']:.2f}",
                         f"{s['blue']}/{s['red']}", md, f"{m['speed_med']:.1f}", f"{m['dz_med']:.0f}",
                         pct(s["animal_share"]), Tracks.reading(s)])
        add_table(L, ["cl", "testing", "eval", "drone", "mixed/unmatched/no-rows", "drone_likeness [95 %]", "lift",
                      "BLUE/RED", "drone models (top 3)", "speed m/s", "dz m", "sensor says animal/bird/vehicle",
                      "reading (from numbers)"], rows)
        by = defaultdict(list)
        for c in ids:
            by[Tracks.reading(st[c])].append(str(c))
        L.append(f"{name} readings: " + "; ".join(f"{r}: c{', c'.join(v)}" for r, v in sorted(by.items())) + ".")
    bird = tr.in_fit & (tr.cls_top == "Bird")
    ani = tr.in_fit & (tr.cls_top == "Animal")
    KEY["best6"] = max(s_["lift"] for s_ in out["K=6"].values() if s_["lift"] == s_["lift"])
    KEY["best_star"] = max(s_["lift"] for s_ in out["K*"].values() if s_["lift"] == s_["lift"])
    L.append(f"Bird / clutter: the sensors call only {int(bird.sum())} tracks 'Bird' ({int((bird & tr.fit_test & tr.drone).sum())} drone-labelled) and "
             f"{int(ani.sum())} 'Animal' ({int((ani & tr.fit_test & tr.drone).sum())}); no K-means cluster isolates them (animal/bird/vehicle share "
             f"{min(s_['animal_share'] for s_ in out['K=6'].values()):.1%} to {max(s_['animal_share'] for s_ in out['K=6'].values()):.1%} across K=6 clusters).")
    return out


KEY: dict = {}             # headline numbers collected by the sections for the summary at the top
MIN_SENSOR_TRACKS = 20     # per-sensor tables list sensors with at least this many testing in_fit tracks


def sec_sensor_cluster(L: list[str], tr: Tracks, stats: dict) -> dict:
    L.append("")
    cnt = Counter(tr.sensor[tr.fit_test].tolist())
    sens = sorted({s for s in tr.sensor[tr.fit_test & tr.drone] if cnt[s] >= MIN_SENSOR_TRACKS})
    small = sorted(s for s in set(tr.sensor[tr.fit_test & tr.drone]) if cnt[s] < MIN_SENSOR_TRACKS)
    L.append("**Per-sensor drone_likeness inside each cluster** (drone/tracks, testing in_fit; sensors with >= 1 drone track and "
             f">= {MIN_SENSOR_TRACKS} tracks; omitted for size: " + ", ".join(f"{s} n={cnt[s]}" for s in small) + "):")
    res = {}
    for name, kcol, ids in (("K*", tr.k_star, tr.k_star_ids), ("K=6", tr.k6, tr.k6_ids)):
        rows = []
        for s in sens:
            ms = tr.fit_test & (tr.sensor == s)
            cells = []
            for c in ids:
                m = ms & (kcol == c)
                n = int(m.sum())
                k = int((m & tr.drone).sum())
                cells.append("" if n == 0 else f"{pct(k / n)} ({k}/{n})")
            n = int(ms.sum())
            k = int((ms & tr.drone).sum())
            rows.append([s, f"{pct(k / n)} ({k}/{n})"] + cells)
        add_table(L, [f"{name} sensor", "all"] + [f"c{c}" for c in ids], rows)
    L.append("**Precision / recall of 'track is drone if its cluster likeness > m x base'** (per sensor, testing in_fit tracks; "
             "LOIO = clusters re-selected without the evaluated testing iteration; last column = the sensor's own class "
             "cls_top == 'Air Vehicle' for comparison, nothing fitted):")
    for name, kcol in (("K*", tr.k_star), ("K=6", tr.k6)):
        r3 = pr_by_sensor(tr, kcol, LIFT_RULE)
        best = max(s["lift"] for s in stats[name].values() if s["lift"] == s["lift"])
        L.append(f"- {name}, m = {LIFT_RULE:g}: clusters selected = {', '.join(map(str, r3['sel'])) if r3['sel'] else 'NONE'} "
                 f"(best cluster lift {best:.2f}, so every precision/recall at m = {LIFT_RULE:g} is 0/undefined).")
    rr = {name: pr_by_sensor(tr, kcol, LIFT_RELAXED) for name, kcol in (("K*", tr.k_star), ("K=6", tr.k6))}
    L.append(f"- Relaxed m = {LIFT_RELAXED:g}: K* selects {rr['K*']['sel'] or 'none'}, K=6 selects {rr['K=6']['sel'] or 'none'}; "
             "the table uses it.")
    rows = []
    for s in sens + ["ALL"]:
        a = rr["K=6"]["per"][s]
        base = a["pos"] / a["n"]
        ms = tr.fit_test if s == "ALL" else tr.fit_test & (tr.sensor == s)
        av = ms & (tr.cls_top == "Air Vehicle")
        tp_av = int((av & tr.drone).sum())

        def pr(tp, pred, pos):
            return f"{pct(tp / pred) if pred else '-'}/{pct(tp / pos, 0) if pos else '-'}"

        rows.append([s, a["n"], a["pos"], pct(base),
                     pr(a["tp_in"], a["pred_in"], a["pos"]), pr(a["tp_lo"], a["pred_lo"], a["pos"]),
                     pr(tp_av, int(av.sum()), a["pos"])])
        if s == "ALL":
            unk = ms & (tr.cls_top == "Unknown")
            res["av"] = dict(p=tp_av / int(av.sum()), r=tp_av / a["pos"], base=base,
                             unk_p=tr.drone[unk].mean(), unk_n=int(unk.sum()))
    add_table(L, ["sensor", "tracks", "drone", "base", f"K=6 P/R in-sample (m={LIFT_RELAXED:g})", "K=6 P/R LOIO",
                  "cls=Air Vehicle P/R"], rows)
    res["relaxed"] = rr
    return res


def sec_sensor_class(L: list[str], tr: Tracks) -> None:
    m = tr.fit_test
    base = tr.drone[m].mean()
    rows = []
    for c, n in Counter(tr.cls_top[m].tolist()).most_common():
        k = int((m & (tr.cls_top == c) & tr.drone).sum())
        if n >= 30:
            rows.append(f"{c} {k}/{n} ({pct(k / n)}, x{(k / n) / base:.1f})")
    inner = []
    for s in sorted(set(tr.sensor[m])):
        av = m & (tr.sensor == s) & (tr.cls_top == "Air Vehicle")
        rest = m & (tr.sensor == s) & (tr.cls_top == "Unknown")
        if av.sum() >= 20 and rest.sum() >= 20:
            inner.append(f"{s} {pct(tr.drone[av].mean())} vs {pct(tr.drone[rest].mean())}")
    L.append("")
    L.append("Sensor's own class (cls_top; drone share, lift vs base; testing in_fit tracks): " + "; ".join(rows)
             + ". Within a radar, 'Air Vehicle' versus 'Unknown' drone share: " + ", ".join(inner) + ".")


def sec_evaluation(L: list[str], tr: Tracks, stats: dict, facts: dict) -> None:
    L.append("")
    L.append("**How evaluation tracks distribute** (no GT; the testing likeness is transferred as a prior):")
    ev = ~tr.testing
    its = sorted(set(tr.it[ev].tolist()))
    sens_rate = {}
    for s in sorted(set(tr.sensor[tr.fit_test].tolist())):
        m = tr.fit_test & (tr.sensor == s)
        sens_rate[s] = tr.drone[m].mean()
    rows = []
    for it in its:
        m = ev & (tr.it == it)
        row = [it.replace("evaluation/", ""), int(m.sum())]
        row.append("/".join(str(int((m & (tr.k_star == c)).sum())) for c in tr.k_star_ids))
        row.append("/".join(str(int((m & (tr.k6 == c)).sum())) for c in tr.k6_ids))
        fit = m & tr.in_fit
        e_sensor = sum(sens_rate.get(s, 0.0) for s in tr.sensor[fit])
        lik6 = {c: stats["K=6"][c]["lik"] for c in tr.k6_ids}
        e_cl = sum(lik6[c] for c in tr.k6[fit])
        top = Counter(tr.sensor[m].tolist()).most_common(3)
        row += [f"{e_sensor:.0f}", f"{e_cl:.0f}", ", ".join(f"{s} {n}" for s, n in top)]
        rows.append(row)
    add_table(L, ["iteration", "tracks", "K* c0/c1/c2".replace("c0/c1/c2", "/".join(f"c{c}" for c in tr.k_star_ids)),
                   "K=6 " + "/".join(f"c{c}" for c in tr.k6_ids), "expected drone tracks (sensor rate)",
                   "(K=6 cluster rate)", "top sensors (tracks)"], rows)
    fr = facts["far"]
    eg = fr.get("EGOI")
    if eg:
        L.append(f"Manned aircraft: none of the clusters is manned-aircraft-like (largest cluster median speed "
                 f"{max(s['med']['speed_med'] for s in stats['K=6'].values()):.1f} m/s, rule >= {MANNED_MPS:g} m/s) because the "
                 f"15 km gate removes them first: {num(eg['n'])} EGOI rows ({pct(eg['n'] / facts['per']['EGOI']['n_pos'])} of its "
                 f"position rows) lie beyond 15 km with median speed {eg['speed_med']:.0f} m/s and altitude {eg['alt_med']:.0f} m "
                 f"({pct(eg['fast'], 0)} of them >= {MANNED_MPS:g} m/s). These are the manned-aircraft class and are a separate, "
                 "kinematically trivial group.")


def sec_calibration(L: list[str], facts: dict) -> None:
    cal = rd("sensor_calibration.csv")
    L.append("")
    L.append("## 4. Calibration and data-quality rules the fusion/tracking step must apply")
    tadopt = [x for x in cal if x["param"] == "time_offset_s" and x["adopted"] == "1" and x["iter_key"] == "*"]
    trej = [x for x in cal if x["param"] == "time_offset_s" and x["iter_key"] == "*" and x["adopted"] != "1"]
    L.append("Calibration (gt_association: estimated on testing GT; apply to evaluation by sensor identity):")
    L.append("- Time: " + "; ".join(f"{sid(x['sensor_key'])} t_cal = t {fnum(x['value']):+.2f} s (score {fnum(x['score_at_value']):.2f} vs "
                                   f"{fnum(x['score_at_zero']):.2f} at 0)" for x in tadopt)
             + f". {len(trej)} other position sensors: no offset adopted (gain < 1.10x or not consistent across iterations).")
    radar = [x for x in cal if x["param"] == "az_offset_deg" and x["iter_key"] == "*" and x["note"].startswith("radar_bias")]
    L.append("- Radar azimuth offset to ADD to the derived azimuth (informational; radar positions are matched by position): "
             + ", ".join(f"{sid(x['sensor_key'])} {fnum(x['value']):+.2f}" for x in radar) + " deg.")
    flips = [x for x in cal if x["param"] == "az_offset_deg" and x["adopted"] == "1" and x["iter_key"] != "*"
             and abs(fnum(x["value"])) > 20]
    emx = [x for x in cal if x["sensor_key"].startswith("EMXG") and x["param"] == "az_offset_deg" and x["iter_key"] != "*"
           and x["adopted"] == "1"]
    emx_flip = [x for x in emx if abs(fnum(x["value"])) > 20]
    L.append("- Bearing sensors, per-iteration azimuth offsets (testing only): "
             + ", ".join(f"{sid(x['sensor_key'])}@{x['iter_key'].split('/')[-1]} {fnum(x['value']):+.1f}" for x in flips)
             + f". EMXG is rotated in {len(emx_flip)} of {len(emx)} testing iterations, so its evaluation azimuth cannot be trusted "
             "without a per-iteration check (co-observation with a radar).")
    pres = sorted({sid(x["sensor_key"]) for x in cal if "presence-only" in x["note"].lower()})
    sect = sorted({sid(x["sensor_key"]) for x in cal if "sector" in x["note"].lower()})
    KEY["flip_sensors"] = sorted({sid(x["sensor_key"]) for x in flips})
    L.append(f"- Presence-only (bearing tracks no drone): {', '.join(pres)}. 45-degree sector labels: {', '.join(sect)}.")
    per = facts["per"]
    L.append("")
    L.append("Quality layer (cuas_clean.py), share of each sensor's position rows:")
    rows = []
    for s, p in per.items():
        if p["null_island"] + p["self_report"] + p["far"] + p["alt0"] + p["vzero"] + p["vu_dead"] == 0:
            continue
        rows.append([s, num(p["n_pos"]), pct(p["null_island"] / p["n_pos"]), pct(p["self_report"] / p["n_pos"]),
                     pct(p["far"] / p["n_pos"]), pct(p["alt0"] / p["n_pos"]),
                     pct(p["vzero"] / p["vfin"]) if p["vfin"] else "-", pct(p["usable"] / p["n_pos"])])
    add_table(L, ["sensor", "pos rows", "null island", "self-report (<5 m)", "far (>15 km)", "alt = 0", "(ve,vn)=(0,0) of finite",
                   "usable"], rows)
    gt = Q.GT_FEED_SENSORS
    L.append("Rules: drop null-island and self-report rows; flag > 15 km; alt == 0 and (0,0) velocities mean unknown (NaN); "
             "pool height as dz = alt - sensor_alt (NYTF switches height datum between iterations); "
             f"{', '.join(sorted(gt))} relay the Dronetag feed (use as inputs, never as evidence when scoring against GT); "
             "radar clutter dominates, so a single radar track needs a second sensor or cluster/identity evidence.")
    vu = [s for s, p in per.items() if p["vu_dead"]]
    L.append(f"Dead vertical velocity (vu always 0, set to NaN): {', '.join(vu) if vu else 'no sensor'}.")


def sec_next(L: list[str], tr: Tracks, obs: dict, cls: dict, prec: list[dict], facts: dict, stats: dict) -> None:
    L.append("")
    L.append("## 5. Recommended next steps (weights: GOSPA tracking 0.4, fusion completeness 0.3, classification 0.2, identification 0.1)")
    pr = {x["sensor"]: fnum(x["precision@gate"]) for x in prec}
    clutter = ", ".join(f"{s} {pct(pr[s])}" for s in ("EGOI", "NYTF", "IUKL", "EUIG", "UXYJ") if s in pr)
    rr = cls["relaxed"]["K=6"]["per"]["ALL"]
    p_in = rr["tp_in"] / rr["pred_in"] if rr["pred_in"] else float("nan")
    p_lo = rr["tp_lo"] / rr["pred_lo"] if rr["pred_lo"] else float("nan")
    r_in = rr["tp_in"] / rr["pos"]
    av = cls["av"]
    best6 = max(s["lift"] for s in stats["K=6"].values() if s["lift"] == s["lift"])
    L.append(f"1. Tracking (0.4): radar clutter is the false-track source (share of rows within 50 m of a drone: {clutter}). "
             f"Gate tracks on multi-sensor agreement first: co-observation by >= 2 sensors raises the GT hit rate x{obs['lift']:.1f} "
             f"({pct(obs['p1'])} -> {pct(obs['p2'])}) and exists on evaluation too. Kinematic K-means is only a weak prior (best cluster "
             f"lift {best6:.2f}, m = {LIFT_RELAXED:g} call: precision {pct(p_in)} in-sample / {pct(p_lo)} LOIO vs base {pct(av['base'])}, "
             f"recall {pct(r_in, 0)}); do not gate on it.")
    nf = obs["nf_hist"]
    two = sum(v for k, v in nf.items() if k >= 2)
    L.append(f"2. Fusion completeness (0.3): apply the LJJX/UXYJ time offsets and per-iteration bearing offsets, then fuse the strongest "
             f"combinations ({', '.join(KEY['top_combos'])} by co-detected seconds; {', '.join(KEY['hi_combos'])} by GT share; see 2c). "
             f"Only {two} of {sum(nf.values())} flights are seen by >= 2 non-feed "
             f"sensors and {nf.get(0, 0)} by none: completeness is bounded by sensor coverage more than by association.")
    L.append(f"3. Classification (0.2): the radar's own class carries the signal that K-means lacks: 'Air Vehicle' tracks are drones "
             f"{pct(av['p'])} of the time (recall {pct(av['r'], 0)}) against base {pct(av['base'])}, while 'Unknown' (most tracks) is "
             f"{pct(av['unk_p'])}. Use class + multi-sensor support + modality (cyber/RF/camera/acoustic reports are drone reports); keep "
             "far, fast EGOI tracks as a separate manned-aircraft class.")
    sg = facts["serial_groups"]
    multi = sum(1 for g in sg.values() if len(g["sensors"]) >= 2)
    L.append(f"4. Identification (0.1): only {len(sg)} distinct serials exist and {multi} are seen by >= 2 sensors ({KEY['serial_sensors']}): "
             "merge by serial containment and attach model/protocol to the fused track.")
    L.append(f"5. Before scoring on evaluation: re-estimate the bearing azimuth offsets ({', '.join(KEY['flip_sensors'])}) per evaluation "
             "iteration; never tune on LJJX/QTVS/VKYY; treat cluster likeness as a prior, not as a class.")


def sec_limits(L: list[str], tr: Tracks) -> None:
    L.append("")
    L.append("## Limitations")
    L.append(f"- Labels exist only for the {len(set(tr.it[tr.testing].tolist()))} testing iterations and {KEY['n_flights']} tagged flights; an untagged real drone counts as "
             "'unmatched', so every drone share is a lower bound. 'drone' needs >= 50 % of a track's rows within the gate (50 m) of one flight.")
    L.append(f"- Only {int((tr.fit_test & tr.drone).sum())} drone tracks exist among {int(tr.fit_test.sum()):,} testing in_fit tracks, so per-sensor and per-cluster "
             "cells are small (see the intervals); clusters were selected on the same data (in-sample), LOIO is the honest figure.")
    L.append("- K-means silhouettes are low (continuum); K* is the silhouette argmax, K=6 is kept for finer structure. Distances are in robust-scaled "
             "log features; cluster ids are ordered by size, not meaning. The readings are mechanical rules on medians, not a verified class.")
    L.append("- Gates (30 m site, 50 m / 1 s co-observation, 5 % coverage, 15 km far) are chosen thresholds; gt_association.md section 7 shows the sensitivity. "
             "VKYY/QTVS/LJJX are label leakage against GT; the *_no_gt_feed variants and in_fit exclude them.")


# ---------------------------------------------------------------------------------------------
def write_track_csv(tr: Tracks, st_star: dict, st_6: dict) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["iter_key", "tree", "sensor_key", "object_id", "cluster_kstar", "drone_likeness_kstar",
                    "cluster_k6", "drone_likeness_k6", "gt_label", "in_fit"])
        for i, x in enumerate(tr.rows):
            ks, k6 = tr.k_star[i], tr.k6[i]
            ls = st_star[ks]["lik"] if ks in st_star else float("nan")
            l6 = st_6[k6]["lik"] if k6 in st_6 else float("nan")
            w.writerow([x["iter_key"], x["tree"], x["sensor_key"], x["object_id"], ks,
                        "" if ls != ls else f"{ls:.6f}", k6, "" if l6 != l6 else f"{l6:.6f}", tr.label[i], x["in_fit"]])


def main() -> None:
    tr = Tracks()
    inv = rd("sensor_inventory.csv")
    sem = rd("field_matrix_semantic.csv")
    fam = rd("sensor_families.csv")
    gt_md = (D / "gt_association.md").read_text(encoding="utf-8")
    prec = [{"sensor": sid(r["sensor"]), "precision@gate": r["precision@gate"]}
            for r in parse_md_table(gt_md, r"^## 2\. Per-sensor precision")]
    facts = detection_facts()

    L: list[str] = []
    per = sec_identification(L, inv, sem, fam)
    L.append("")
    L.append("## 2. Intersections")
    sec_field_intersections(L)
    sec_devices(L, per, facts)
    obs = sec_observation(L)
    stats = sec_kmeans(L, tr, fam)
    sec_sensor_class(L, tr)
    cls = sec_sensor_cluster(L, tr, stats)
    sec_evaluation(L, tr, stats, facts)
    sec_calibration(L, facts)
    sec_next(L, tr, obs, cls, prec, facts, stats)
    sec_limits(L, tr)

    av = cls["av"]
    ta = KEY["top_area"]
    head = [
        "# Pre-classification report: sensor data, intersections, K-means",
        "",
        "## Key numbers (all computed by tools/preclass_report.py from the derived tables)",
        f"- Data: {KEY['n_sensors']} sensors ({KEY['n_pos']} position, {KEY['n_sensors'] - KEY['n_pos']} bearing-only), {KEY['n_files']} files, "
        f"{num(KEY['n_rows'])} rows, {KEY['n_iters']} iterations ({KEY['n_test_it']} testing with Dronetag GT, {KEY['n_eval_it']} evaluation).",
        f"- Field intersections: the biggest discriminative set is {ta[0]} sensors x {ta[1]} attributes ({ta[2]}); "
        f"{KEY['n_mm_sites']} of {KEY['n_multi_sites']} multi-sensor sites are multi-modal; {KEY['n_serials_multi']} of {KEY['n_serials']} drone serials "
        f"are reported by 2-{KEY['serial_max']} sensors ({KEY['serial_sensors']}).",
        f"- Observation intersections: {KEY['flights_two']} of {KEY['n_flights']} GT flights are seen by >= 2 sensors without the GT feeds, "
        f"{KEY['flights_zero']} by none; strongest combinations {', '.join(KEY['top_combos'])}.",
        f"- Co-observation without GT: a group seen by >= 2 sensors lies on a GT drone {pct(KEY['p2'])} of the time versus {pct(KEY['p1'])} for one sensor "
        f"(lift x{KEY['lift']:.1f}); this indicator also exists on the evaluation iterations.",
        f"- K-means tracks: K* = {len(tr.k_star_ids)}; drone base rate {pct(av['base'], 2)}; best cluster lift {KEY['best_star']:.2f} (K*) / {KEY['best6']:.2f} (K=6); "
        f"no cluster reaches the x{LIFT_RULE:g} drone-like rule. The sensors' own 'Air Vehicle' class does: precision {pct(av['p'])}, recall {pct(av['r'], 0)} "
        f"(lift x{av['p'] / av['base']:.1f}).",
        "",
    ]
    L = head + L
    # collapse repeated blank lines, drop trailing blank
    out_l: list[str] = []
    for ln in L:
        if ln == "" and (not out_l or out_l[-1] == ""):
            continue
        out_l.append(ln)
    while out_l and out_l[-1] == "":
        out_l.pop()
    L = out_l
    text = "\n".join(L) + "\n"
    OUT_MD.write_text(text, encoding="utf-8")
    write_track_csv(tr, stats["K*"], stats["K=6"])

    # ----- console: acceptance table + self-checks
    st = stats["K*"]
    print(f"drone_likeness per cluster, K* = {len(tr.k_star_ids)} (testing in_fit tracks; base "
          f"{pct(tr.drone[tr.fit_test].mean(), 2)})")
    print(f"{'cluster':>7} {'testing':>8} {'eval':>6} {'in_fit_test':>11} {'drone':>6} {'likeness':>9} {'wilson95':>17} {'lift':>5}")
    for c in tr.k_star_ids:
        s = st[c]
        print(f"{c:>7} {s['n_test']:>8} {s['n_eval']:>6} {s['n_fit_test']:>11} {s['k']:>6} {pct(s['lik'], 2):>9} "
              f"{'[' + pct(s['lo'], 2) + ', ' + pct(s['hi'], 2) + ']':>17} {s['lift']:>5.2f}")
    out_rows = sum(1 for _ in open(OUT_CSV, encoding="utf-8")) - 1
    checks = []
    checks.append(("track_preclass.csv rows == track_clusters rows", out_rows == tr.n, f"{out_rows} == {tr.n}"))
    checks.append(("no duplicate track keys", tr.dup_keys == 0, f"duplicates {tr.dup_keys}"))
    checks.append(("every testing track has a label row", tr.unlabelled_testing == 0, f"unlabelled {tr.unlabelled_testing}"))
    for name, stt in stats.items():
        n_fit = sum(s["n_fit_test"] for s in stt.values())
        mix = sum(sum(s["mix"].values()) for s in stt.values())
        checks.append((f"{name}: cluster counts sum to testing in_fit tracks", n_fit == mix == int(tr.fit_test.sum()),
                       f"{n_fit} == {mix} == {int(tr.fit_test.sum())}"))
        n_all = sum(s["n_test"] + s["n_eval"] for s in stt.values())
        checks.append((f"{name}: clustered tracks + cluster -1 == all tracks",
                       n_all + int((~tr.in_fit).sum()) == tr.n, f"{n_all} + {int((~tr.in_fit).sum())} == {tr.n}"))
    checks.append((f"report <= {MAX_LINES} lines", len(L) <= MAX_LINES, f"{len(L)} lines"))
    ok = True
    for name, good, detail in checks:
        print(f"self-check {'OK  ' if good else 'FAIL'} {name} ({detail})")
        ok &= good
    print(f"wrote {OUT_MD.relative_to(C.REPO)} ({len(L)} lines) and {OUT_CSV.relative_to(C.REPO)} ({out_rows} rows)")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
