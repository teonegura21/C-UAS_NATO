#!/usr/bin/env python3
"""Sensor-to-sensor co-observation WITHOUT ground truth, on both trees (testing + evaluation).

Question: which position sensors report something at the same place and time? Everything is
computed from the detections only, so it also works for the evaluation datasets. Ground truth
(testing only) is used for VALIDATION of the "seen by >= 2 sensors" indicator, never for grouping.

Input rows: usable position detections (Q.usable_pos & ~Q.far), t_cal = t + the adopted POOLED
time offset of the sensor (data/derived/sensor_calibration.csv, param time_offset_s,
adopted == 1, applies_to == 'all'; azimuth offsets are ignored).

1. PAIRS (coobs_pairs.csv). For each iteration and ordered pair of distinct sensor_ids (A, B),
   n_a = A's detections, n_a_with_b = A's detections with >= 1 B detection within |dt| <= 1.0 s
   (on t_cal) and horizontal distance <= gate (25 / 50 / 100 m). The time window is exact
   (detections in neighbouring 1-s bins are compared too); the search uses a (time-bin, 100 m
   cell) hash so no 1-s bin edge splits a pair. Extra columns: n_a_in_span = A's detections
   inside B's [first, last] t_cal span in that iteration, frac_in_span = n_a_with_b / n_a_in_span
   (the fairer denominator when the two sensors were not recording over the same window).
2. GROUPS (coobs_groups.csv.gz). Within each 1-s bin (floor of t_cal) detections of ALL position
   sensors are single-linkage clustered. Two detections are linked when
     - they come from DIFFERENT sensor_ids and are <= 50 m apart (horizontal), or
     - they come from the SAME sensor_id and are <= 10 m apart,
   so one sensor's clutter carpet (e.g. an EGOI radar field) cannot chain a whole scene by itself:
   a same-sensor link only joins near-duplicates. Groups never span bins (a drone straddling a
   bin edge is cut in two; the pair statistics above are not affected by this). Every usable
   detection belongs to exactly one group. A group's `sensors` are its distinct sensor_ids
   ('|', sorted), `size` their number. Variants: 'all' and 'no_gt_feed' (VKYY / QTVS / LJJX, the
   relays of the Dronetag feed, removed BEFORE grouping; they would label-leak in any validation).
   gt_flight (testing): majority best_flight of the group's members that det_gt_match marks
   matched (50 m); '' if none. In variant 'all' that label is partly circular (the GT-feed sensors
   match GT trivially), so the validation uses 'no_gt_feed' only.
3. SETS (coobs_sets.csv): per tree and variant, every distinct sensor combination (groups with the
   same `sensors` string), its number of groups, number of iterations it occurs in and, for the
   testing tree, the share of its groups lying on a GT drone.
4. coobservation.md: pairwise matrices, biggest combinations, validation.

Run: python3 tools/coobservation.py     (self-checks run at the end; non-zero exit on failure)
"""

from __future__ import annotations

import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuas_clean as Q  # noqa: E402
import cuas_common as C  # noqa: E402

T0 = time.time()
OUT = C.DERIVED
OUT_PAIRS = OUT / "coobs_pairs.csv"
OUT_GROUPS = OUT / "coobs_groups.csv.gz"
OUT_SETS = OUT / "coobs_sets.csv"
OUT_MD = OUT / "coobservation.md"
CALIBRATION = OUT / "sensor_calibration.csv"
GT_MATCH = OUT / "det_gt_match.csv.gz"

GATES = (25.0, 50.0, 100.0)
GROUP_GATE_M = 50.0
SAME_SENSOR_LINK_M = 10.0
MAX_DT_S = 1.0
VARIANTS = ("all", "no_gt_feed")
WIDTH = 4096            # cell index range per axis (offset 2048; |e|,|n| <= 15 km / >= 25 m cells)
OFFSET = 2048
CHUNK = 6_000_000       # candidate pairs expanded at once


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Spatial hash: candidate pairs in neighbouring (time-bin, cell) buckets
# --------------------------------------------------------------------------- #


def _cell_index(x: np.ndarray, cell: float) -> np.ndarray:
    return np.floor(x / cell).astype(np.int64) + OFFSET


def candidate_chunks(tb: np.ndarray, ix: np.ndarray, iy: np.ndarray, dts: tuple[int, ...]):
    """Yield (i, j) index arrays of every ordered pair (i != j) whose (time-bin, cell) buckets are
    within `dts` time bins and +-1 cell on each axis. Each ordered pair appears exactly once."""
    key = (tb * WIDTH + ix) * WIDTH + iy
    order = np.argsort(key, kind="stable")
    ks = key[order]
    idx = np.arange(len(tb), dtype=np.int64)
    for dt in dts:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                q = ((tb + dt) * WIDTH + ix + dx) * WIDTH + iy + dy
                lo = np.searchsorted(ks, q, "left")
                cnt = np.searchsorted(ks, q, "right") - lo
                sel = np.flatnonzero(cnt)
                if sel.size == 0:
                    continue
                cum = np.cumsum(cnt[sel])
                start = 0
                while start < sel.size:
                    base = cum[start - 1] if start else 0
                    stop = int(np.searchsorted(cum, base + CHUNK, "right"))
                    stop = max(stop, start + 1)
                    s = sel[start:stop]
                    c = cnt[s]
                    tot = int(c.sum())
                    first = np.cumsum(c) - c
                    i = np.repeat(idx[s], c)
                    j = order[np.repeat(lo[s], c) + (np.arange(tot) - np.repeat(first, c))]
                    keep = i != j
                    yield i[keep], j[keep]
                    start = stop


# --------------------------------------------------------------------------- #
# Pair statistics
# --------------------------------------------------------------------------- #


def pair_hits(t: np.ndarray, e: np.ndarray, n: np.ndarray, sc: np.ndarray, n_sensors: int,
              gates=GATES):
    """For every gate: (hit[a, b], links[a, b]).

    hit[a, b]   = number of detections of sensor a with >= 1 detection of sensor b (a != b) within
                  |dt| <= MAX_DT_S and horizontal distance <= gate;
    links[a, b] = number of (detection of a, detection of b) pairs satisfying the same test.
    `sc` are integer sensor codes in [0, n_sensors)."""
    S = n_sensors
    base = np.floor(t.min())
    tb = (np.floor(t - base)).astype(np.int64)
    cell = max(gates)
    ix, iy = _cell_index(e, cell), _cell_index(n, cell)
    uniq: dict[float, list[np.ndarray]] = {g: [] for g in gates}
    links = {g: np.zeros(S * S, dtype=np.int64) for g in gates}
    for i, j in candidate_chunks(tb, ix, iy, (-1, 0, 1)):
        diff = sc[i] != sc[j]
        i, j = i[diff], j[diff]
        d = np.hypot(e[i] - e[j], n[i] - n[j])
        ok = (np.abs(t[i] - t[j]) <= MAX_DT_S) & (d <= max(gates))
        i, j, d = i[ok], j[ok], d[ok]
        for g in gates:
            m = d <= g
            if not m.any():
                continue
            ii, jj = i[m], j[m]
            links[g] += np.bincount(sc[ii] * S + sc[jj], minlength=S * S)
            uniq[g].append(np.unique(ii * S + sc[jj]))
    hit = {}
    for g in gates:
        h = np.zeros(S * S, dtype=np.int64)
        if uniq[g]:
            k = np.unique(np.concatenate(uniq[g]))        # unique (detection, other sensor)
            h = np.bincount(sc[k // S] * S + k % S, minlength=S * S)
        hit[g] = h.reshape(S, S)
        links[g] = links[g].reshape(S, S)
    return hit, links


# --------------------------------------------------------------------------- #
# Groups: single linkage inside 1-s bins
# --------------------------------------------------------------------------- #


def connected_components(n: int, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Label = smallest member index of each component (hook roots + pointer jumping)."""
    lab = np.arange(n, dtype=np.int64)
    if u.size == 0:
        return lab
    while True:
        lu, lv = lab[u], lab[v]
        if (lu == lv).all():
            return lab
        m = np.minimum(lu, lv)
        new = lab.copy()
        np.minimum.at(new, lu, m)
        np.minimum.at(new, lv, m)
        while True:
            nn = new[new]
            if np.array_equal(nn, new):
                break
            new = nn
        lab = new


def group_labels(tb: np.ndarray, e: np.ndarray, n: np.ndarray, sc: np.ndarray) -> np.ndarray:
    """Group id (0.. in order of first member) for detections that are SORTED by tb."""
    ix, iy = _cell_index(e, GROUP_GATE_M), _cell_index(n, GROUP_GATE_M)
    us, vs = [], []
    for i, j in candidate_chunks(tb, ix, iy, (0,)):
        k = i < j
        i, j = i[k], j[k]
        d = np.hypot(e[i] - e[j], n[i] - n[j])
        link = np.where(sc[i] != sc[j], d <= GROUP_GATE_M, d <= SAME_SENSOR_LINK_M)
        us.append(i[link])
        vs.append(j[link])
    u = np.concatenate(us) if us else np.zeros(0, np.int64)
    v = np.concatenate(vs) if vs else np.zeros(0, np.int64)
    lab = connected_components(len(tb), u, v)
    return np.unique(lab, return_inverse=True)[1].astype(np.int64)


def summarise_groups(gid: np.ndarray, sc: np.ndarray, names: list[str], e: np.ndarray, n: np.ndarray,
                     dz: np.ndarray, fcode: np.ndarray, flights: list[str]) -> dict:
    """Per-group aggregates. `fcode` = matched flight code per detection (-1 none)."""
    G = int(gid.max()) + 1 if len(gid) else 0
    S = len(names)
    n_det = np.bincount(gid, minlength=G)
    e_mean = np.bincount(gid, weights=e, minlength=G) / np.maximum(n_det, 1)
    n_mean = np.bincount(gid, weights=n, minlength=G) / np.maximum(n_det, 1)
    fin = np.isfinite(dz)
    dz_n = np.bincount(gid[fin], minlength=G)
    dz_sum = np.bincount(gid[fin], weights=dz[fin], minlength=G)
    dz_mean = np.where(dz_n > 0, dz_sum / np.maximum(dz_n, 1), np.nan)
    ks = np.unique(gid * S + sc)                      # (group, sensor) sorted by group then sensor code
    kg, ksc = ks // S, ks % S
    sensors = [""] * G
    size = np.bincount(kg, minlength=G)
    bounds = np.concatenate([[0], np.cumsum(size)])
    for g in range(G):
        sensors[g] = "|".join(names[c] for c in ksc[bounds[g]:bounds[g + 1]])
    gt = [""] * G
    m = fcode >= 0
    if m.any():
        F = len(flights)
        kk, cnt = np.unique(gid[m] * F + fcode[m], return_counts=True)
        kgg, kf = kk // F, kk % F
        o = np.lexsort((kf, -cnt, kgg))               # per group: highest count, then smallest flight
        first = np.ones(len(o), bool)
        first[1:] = kgg[o][1:] != kgg[o][:-1]
        for g, f in zip(kgg[o][first].tolist(), kf[o][first].tolist()):
            gt[g] = flights[f]
    return dict(n_det=n_det, size=size, sensors=sensors, e_mean=e_mean, n_mean=n_mean,
                dz_mean=dz_mean, gt_flight=gt)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_offsets() -> dict[str, float]:
    """{sensor_id: pooled time offset (s)} from rows adopted == 1, applies_to == 'all'."""
    out: dict[str, float] = {}
    with CALIBRATION.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["param"] == "time_offset_s" and r["adopted"] == "1" and r["applies_to"] == "all":
                out[r["sensor_key"].split("_")[0]] = float(r["value"])
    return out


def load_gt_labels() -> dict[tuple[str, int], str]:
    """{(file, row): best_flight} for rows det_gt_match marks matched (50 m)."""
    m = C.read_columns(GT_MATCH, {"row", "matched"},
                       usecols=["file", "row", "best_flight", "matched"])
    sel = np.flatnonzero(m["matched"] == 1)
    return {(m["file"][k], int(m["row"][k])): m["best_flight"][k] for k in sel.tolist()}


def fmt(x: float) -> str:
    return repr(float(x))


# --------------------------------------------------------------------------- #
# Main computation
# --------------------------------------------------------------------------- #


def main() -> int:
    offsets = load_offsets()
    log(f"adopted pooled time offsets: {offsets}")
    det = Q.load_clean(["iter_key", "file", "row", "sensor_id", "t", "e", "n"])
    log(f"loaded {len(det['t'])} detections")
    use = det["usable_pos"] & ~det["far"]
    off = np.array([offsets.get(s, 0.0) for s in det["sensor_id"]])
    t_cal = det["t"] + off
    sid_all = det["sensor_id"]
    sensors = sorted(set(sid_all[use].tolist()))
    S = len(sensors)
    scode_of = {s: k for k, s in enumerate(sensors)}
    is_feed = np.isin(np.array(sensors, dtype=object), list(Q.GT_FEED_SENSORS))
    log(f"{int(use.sum())} usable position detections, {S} sensors: {' '.join(sensors)}")
    gt_lab = load_gt_labels()
    log(f"{len(gt_lab)} det_gt_match rows with matched == 1")

    iters = sorted(set(det["iter_key"][use].tolist()))
    pair_rows: list[list] = []
    group_cols: dict[str, list] = defaultdict(list)
    counts_pair_check = []      # (iter, gate, hit, links, n_a)
    totals = {v: defaultdict(int) for v in VARIANTS}
    iter_n_use: dict[str, int] = {}

    for ik in iters:
        tree = ik.split("/")[0]
        idx = np.flatnonzero(use & (det["iter_key"] == ik))
        idx = idx[np.argsort(t_cal[idx], kind="stable")]          # time order, ties by file order
        t, e, n = t_cal[idx], det["e"][idx], det["n"][idx]
        sc = np.array([scode_of[s] for s in sid_all[idx]], dtype=np.int64)
        dz = det["dz_sensor"][idx]
        iter_n_use[ik] = len(idx)
        n_a = np.bincount(sc, minlength=S)
        t_lo = np.array([t[sc == k].min() if n_a[k] else np.nan for k in range(S)])
        t_hi = np.array([t[sc == k].max() if n_a[k] else np.nan for k in range(S)])
        log(f"{ik}: {len(idx)} detections, {int((n_a > 0).sum())} sensors")

        hit, links = pair_hits(t, e, n, sc, S)
        counts_pair_check.append((ik, hit, links, n_a))
        in_span = np.zeros((S, S), dtype=np.int64)       # A's detections inside B's [first, last] span
        for a in range(S):
            ta = t[sc == a]
            for b in range(S):
                if a != b and n_a[a] and n_a[b]:
                    in_span[a, b] = int(((ta >= t_lo[b]) & (ta <= t_hi[b])).sum())
        for g in GATES:
            for a in range(S):
                for b in range(S):
                    if a != b and n_a[a] and n_a[b]:
                        pair_rows.append([ik, tree, sensors[a], sensors[b], g, int(n_a[a]),
                                          int(hit[g][a, b]), int(in_span[a, b])])

        base = np.floor(t.min())
        tb = np.floor(t - base).astype(np.int64)
        fl = np.array([gt_lab.get((det["file"][k], int(det["row"][k])), "") for k in idx], dtype=object) \
            if tree == "testing" else np.full(len(idx), "", dtype=object)
        for variant in VARIANTS:
            keep = np.ones(len(idx), bool) if variant == "all" else ~is_feed[sc]
            sel = np.flatnonzero(keep)
            gid = group_labels(tb[sel], e[sel], n[sel], sc[sel])
            flights = sorted(set(fl[sel].tolist()) - {""})
            fmap = {f: k for k, f in enumerate(flights)}
            fcode = np.array([fmap.get(f, -1) for f in fl[sel]], dtype=np.int64)
            agg = summarise_groups(gid, sc[sel], sensors, e[sel], n[sel], dz[sel], fcode, flights)
            G = len(agg["n_det"])
            first = np.zeros(G, dtype=np.int64)
            first[gid[::-1]] = np.arange(len(sel))[::-1]               # first member of each group
            totals[variant]["det"] += len(sel)
            totals[variant]["groups"] += G
            assert agg["n_det"].sum() == len(sel)
            for name, vals in (
                    ("iter_key", [ik] * G), ("tree", [tree] * G), ("variant", [variant] * G),
                    ("bin_t", (base + tb[sel][first]).astype(np.int64)), ("group_id", np.arange(G)),
                    ("n_det", agg["n_det"]), ("sensors", agg["sensors"]), ("size", agg["size"]),
                    ("e_mean", agg["e_mean"]), ("n_mean", agg["n_mean"]), ("dz_mean", agg["dz_mean"]),
                    ("gt_flight", agg["gt_flight"])):
                group_cols[name].append(vals if isinstance(vals, list) else np.asarray(vals))
            log(f"   {variant}: {G} groups, max size {int(agg['size'].max())}")

    # ---- write tables ---- #
    pair_rows.sort(key=lambda r: (r[0], r[4], r[2], r[3]))
    with OUT_PAIRS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["iter_key", "tree", "sensor_a", "sensor_b", "gate_m", "n_a", "n_a_with_b", "frac",
                    "n_a_in_span", "frac_in_span"])
        for ik, tree, a, b, g, na, nh, nspan in pair_rows:
            w.writerow([ik, tree, a, b, int(g), na, nh, fmt(nh / na),
                        nspan, fmt(nh / nspan) if nspan else ""])
    log(f"wrote {OUT_PAIRS} ({len(pair_rows)} rows)")

    cols = {k: (np.concatenate(v) if not isinstance(v[0], list) else sum(v, []))
            for k, v in group_cols.items()}
    order = ["iter_key", "tree", "variant", "bin_t", "group_id", "n_det", "sensors", "size", "e_mean",
             "n_mean", "dz_mean", "gt_flight"]
    for k in ("e_mean", "n_mean", "dz_mean"):
        cols[k] = np.round(cols[k], 3)
    C.write_columns(OUT_GROUPS, cols, order)
    log(f"wrote {OUT_GROUPS} ({len(cols['n_det'])} rows)")

    sets = build_sets(cols)
    with OUT_SETS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tree", "variant", "combo", "size", "n_groups", "n_iterations",
                    "frac_groups_on_gt_drone", "n_groups_on_gt"])
        for r in sets:
            w.writerow(r)
    log(f"wrote {OUT_SETS} ({len(sets)} rows)")

    write_report(cols, pair_rows, sets, sensors, iter_n_use, offsets)
    log(f"wrote {OUT_MD}")

    self_checks(cols, counts_pair_check, pair_rows, totals, iter_n_use, sensors, t_cal, det, use, sid_all, scode_of)
    log("DONE")
    return 0


# --------------------------------------------------------------------------- #
# Sets
# --------------------------------------------------------------------------- #


def build_sets(cols: dict) -> list[list]:
    rows = []
    tree, variant = cols["tree"], cols["variant"]
    combo, size, itr, gt = cols["sensors"], cols["size"], cols["iter_key"], cols["gt_flight"]
    agg: dict[tuple, list] = {}
    for k in range(len(size)):
        key = (tree[k], variant[k], combo[k])
        a = agg.get(key)
        if a is None:
            a = agg[key] = [int(size[k]), 0, set(), 0]
        a[1] += 1
        a[2].add(itr[k])
        a[3] += gt[k] != ""
    for (tr, va, cb), (sz, ng, its, ong) in sorted(
            agg.items(), key=lambda kv: (kv[0][0], kv[0][1], -kv[1][1], kv[0][2])):
        rows.append([tr, va, cb, sz, ng, len(its), fmt(ong / ng) if tr == "testing" else "",
                     ong if tr == "testing" else ""])
    return rows


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pooled_matrices(pair_rows: list[list], tree: str, gate: float):
    """{(a, b): (n_a, hits, n_in_span)} summed over the iterations of `tree` where both sensors are present."""
    acc: dict[tuple, list[int]] = {}
    for ik, tr, a, b, g, na, nh, nspan in pair_rows:
        if tr != tree or g != gate:
            continue
        v = acc.setdefault((a, b), [0, 0, 0])
        v[0] += na
        v[1] += nh
        v[2] += nspan
    return acc


def write_report(cols: dict, pair_rows: list[list], sets: list[list], sensors: list[str],
                 iter_n_use: dict, offsets: dict) -> None:
    L: list[str] = []
    L.append("# Sensor co-observation (no ground truth)\n")
    L.append(f"Generated by `tools/coobservation.py`. Position detections only (usable_pos, not far): "
             f"{sum(iter_n_use.values()):,} rows in {len(iter_n_use)} iterations. "
             f"t_cal = t + pooled time offset ({', '.join(f'{k} {v:+g} s' for k, v in sorted(offsets.items()))}; "
             f"no other offset is adopted). Co-observation = |dt| <= {MAX_DT_S:g} s (exact, bin edges handled) and "
             f"horizontal distance <= gate. Groups = single linkage inside 1-s bins at {GROUP_GATE_M:g} m between "
             f"different sensors; two detections of the SAME sensor are linked only within {SAME_SENSOR_LINK_M:g} m "
             f"(so one sensor's clutter carpet cannot chain a scene). Groups do not cross bin edges.\n")
    trees = ["testing", "evaluation"]
    tree_arr, var_arr = np.asarray(cols["tree"]), np.asarray(cols["variant"])

    # --- validation first (the finding) --- #
    t_mask = (np.asarray(cols["tree"]) == "testing") & (np.asarray(cols["variant"]) == "no_gt_feed")
    size = np.asarray(cols["size"])[t_mask]
    on_gt = np.array([g != "" for g in np.asarray(cols["gt_flight"], dtype=object)[t_mask]])
    ndet = np.asarray(cols["n_det"])[t_mask]
    L.append("## 1. Validation on testing: is \"seen by >= 2 distinct sensors\" a drone indicator?\n")
    L.append("Groups of variant `no_gt_feed` (VKYY / QTVS / LJJX removed before grouping, so no label leakage). "
             "A group \"lies on a GT drone\" when at least one member is matched (<= 50 m after time calibration) "
             "to a Dronetag flight in det_gt_match. Caveat: only the 38 instrumented flights are GT; a real "
             "but un-tagged object seen by several sensors counts as NOT on a drone here, so every fraction is a "
             "lower bound on the true drone share.\n")
    rows = []
    for lab, m in (("1", size == 1), ("2", size == 2), ("3", size == 3), (">= 4", size >= 4),
                   (">= 2 (all)", size >= 2), ("all groups", size >= 1)):
        ng = int(m.sum())
        og = int(on_gt[m].sum())
        rows.append([lab, f"{ng:,}", f"{og:,}", f"{100 * og / ng:.1f} %" if ng else "-",
                     f"{int(ndet[m].sum()):,}"])
    L.append(md_table(["group size", "groups", "on GT drone", "fraction", "detections"], rows))
    ge2, e1 = size >= 2, size == 1
    p2 = on_gt[ge2].mean() if ge2.any() else float("nan")
    p1 = on_gt[e1].mean() if e1.any() else float("nan")
    rec = ge2[on_gt].mean() if on_gt.any() else float("nan")
    L.append(f"\nPrecision of the indicator: P(on GT | size >= 2) = {100 * p2:.1f} % versus "
             f"P(on GT | size 1) = {100 * p1:.1f} % (lift x{p2 / p1 if p1 else float('nan'):.1f}). "
             f"Coverage: only {100 * rec:.1f} % of the groups that lie on a GT drone have size >= 2 "
             f"(P(size >= 2 | on GT)).\n")
    # per iteration
    itr = np.asarray(cols["iter_key"])[t_mask]
    rows = []
    for ik in sorted(set(itr.tolist())):
        m = itr == ik
        a, b = m & ge2, m & e1
        rows.append([ik, f"{int(a.sum()):,}", f"{100 * on_gt[a].mean():.1f} %" if a.any() else "-",
                     f"{int(b.sum()):,}", f"{100 * on_gt[b].mean():.1f} %" if b.any() else "-"])
    L.append(md_table(["iteration", "groups size>=2", "on GT", "groups size 1", "on GT"], rows))
    # how often the indicator fires, per iteration, both trees (usable without GT)
    L.append("\nHow often the indicator fires (variant `no_gt_feed`, share of groups with size >= 2), "
             "available on every iteration including the evaluation ones:\n")
    tr_all = np.asarray(cols["iter_key"])[var_arr == "no_gt_feed"]
    sz_all = np.asarray(cols["size"])[var_arr == "no_gt_feed"]
    rows = []
    for ik in sorted(set(tr_all.tolist())):
        m = tr_all == ik
        rows.append([ik, f"{int(m.sum()):,}", f"{int((sz_all[m] >= 2).sum()):,}",
                     f"{100 * (sz_all[m] >= 2).mean():.1f} %", int(sz_all[m].max())])
    L.append(md_table(["iteration", "groups", "size >= 2", "share", "max size"], rows))
    # most precise combinations
    L.append("\nCombinations (testing, `no_gt_feed`, size >= 2, >= 100 groups) by share of groups on a GT drone:\n")
    cand = [r for r in sets if r[0] == "testing" and r[1] == "no_gt_feed" and r[3] >= 2 and r[4] >= 100]
    cand.sort(key=lambda r: (-float(r[6]), -r[4]))
    L.append(md_table(["combo", "size", "groups", "iterations", "on GT drone"],
                      [[r[2].replace("|", " + "), r[3], f"{r[4]:,}", r[5], f"{100 * float(r[6]):.1f} %"] for r in cand[:15]]))

    # --- pairwise matrices --- #
    L.append("\n## 2. Pairwise co-observation matrices (50 m, 1 s)\n")
    L.append("Cell (row A, column B) = share of A's detections that have at least one B detection within 1 s and "
             "50 m, pooled over the iterations of the tree in which both sensors have usable detections. "
             "`-` = never both present. First matrix: denominator all of A's detections in those iterations "
             "(radar clutter keeps radar rows near 0). Second matrix: denominator only A's detections inside "
             "B's time span (the fair read when recordings only partly overlap).\n")
    for tr in trees:
        acc = pooled_matrices(pair_rows, tr, 50.0)
        present = sorted({a for a, _ in acc} | {b for _, b in acc})
        if not present:
            continue
        for title, num in (("all of A's detections", 0), ("A's detections inside B's time span", 1)):
            L.append(f"\n### {tr}: {title}\n")
            hdr = ["A \\ B"] + present
            body = []
            for a in present:
                r = [f"**{a}**"]
                for b in present:
                    if a == b:
                        r.append("")
                        continue
                    v = acc.get((a, b))
                    den = None if v is None else (v[0] if num == 0 else v[2])
                    r.append("-" if not den else f"{100 * v[1] / den:.1f}")
                body.append(r)
            L.append(md_table(hdr, body))
            L.append("\n(percent)")
        # strongest unordered pairs
        pairs = []
        for (a, b), v in acc.items():
            if a < b and (b, a) in acc:
                w = acc[(b, a)]
                pairs.append((v[1] + w[1], a, b, v, w))
        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
        L.append(f"\n**{tr}: strongest sensor pairs by number of co-observed detections (both directions summed, 50 m)**\n")
        rows = []
        for tot, a, b, v, w in pairs[:12]:
            rows.append([f"{a}-{b}", f"{tot:,}", f"{v[1]:,}/{v[2]:,}", f"{100 * v[1] / v[2]:.1f} %" if v[2] else "-",
                         f"{w[1]:,}/{w[2]:,}", f"{100 * w[1] / w[2]:.1f} %" if w[2] else "-"])
        L.append(md_table(["pair", "co-observed dets", "A with B / A in span", "frac", "B with A / B in span", "frac"], rows))
        L.append("")

    # --- biggest combos --- #
    L.append("\n## 3. Biggest multi-sensor combinations (groups)\n")
    L.append("Ranked by number of 1-s groups seeing exactly this combination of sensors. `GT %` = share lying on "
             "a GT drone (testing only; for variant `all` it is inflated by the relay sensors).\n")
    for tr in trees:
        for va in VARIANTS:
            sub = [r for r in sets if r[0] == tr and r[1] == va]
            multi = sorted((r for r in sub if r[3] >= 2), key=lambda r: (-r[4], r[2]))
            L.append(f"\n### {tr} / `{va}` ({int(((tree_arr == tr) & (var_arr == va)).sum()):,} groups)\n")
            hist = defaultdict(int)
            for r in sub:
                hist[r[3]] += r[4]
            L.append("Groups by size: " + ", ".join(f"size {k}: {v:,}" for k, v in sorted(hist.items())) + "\n")
            fmt_row = lambda r: [r[2].replace("|", " + "), r[3], f"{r[4]:,}", r[5],  # noqa: E731
                                 f"{100 * float(r[6]):.1f} %" if r[6] != "" else "-"]
            L.append(f"Top combinations (size >= 2) by groups:\n")
            L.append(md_table(["combo", "size", "groups", "iterations", "GT %"], [fmt_row(r) for r in multi[:15]]))
            big = sorted(sub, key=lambda r: (-r[3], -r[4], r[2]))
            L.append(f"\nLargest sets achieved (size, then groups):\n")
            L.append(md_table(["combo", "size", "groups", "iterations", "GT %"], [fmt_row(r) for r in big[:8]]))
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Self-checks
# --------------------------------------------------------------------------- #


def self_checks(cols, pair_checks, pair_rows, totals, iter_n_use, sensors, t_cal, det, use, sid_all, scode_of) -> None:
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)

    print("SELF-CHECKS")
    tree = np.asarray(cols["tree"])
    variant = np.asarray(cols["variant"])
    ndet = np.asarray(cols["n_det"])
    size = np.asarray(cols["size"])
    n_use = sum(iter_n_use.values())
    feed = np.isin(sid_all, list(Q.GT_FEED_SENSORS))
    for va in VARIANTS:
        m = variant == va
        expected = n_use if va == "all" else int((use & ~feed).sum())
        check(f"every usable detection in exactly one group ({va})",
              int(ndet[m].sum()) == expected and totals[va]["det"] == expected,
              f"sum n_det={int(ndet[m].sum()):,} expected={expected:,}")
        check(f"group count matches ({va})", int(m.sum()) == totals[va]["groups"], f"{int(m.sum()):,}")
    # per-iteration partition and size = number of distinct sensors
    sens = cols["sensors"]
    check("size equals number of distinct sensors", all(
        size[k] == sens[k].count("|") + 1 for k in range(len(size))))
    check("n_det >= size", bool((ndet >= size).all()))
    check("group ids are 0..G-1 per (iteration, variant)", _group_ids_ok(cols))
    check("no_gt_feed groups contain no relay sensor", not any(
        any(s in Q.GT_FEED_SENSORS for s in sens[k].split("|")) for k in np.flatnonzero(variant == "no_gt_feed")))
    # symmetry / consistency of pair counts
    sym = cons = mono = True
    for ik, hit, links, n_a in pair_checks:
        for g, L_ in links.items():
            sym &= np.array_equal(L_, L_.T)
            h = hit[g]
            cons &= bool((h <= n_a[:, None]).all() and (h <= L_).all())
            cons &= bool(((h > 0) == (hit[g].T > 0)).all())       # A sees B <=> B sees A
        mono &= all((hit[GATES[i]] <= hit[GATES[i + 1]]).all() for i in range(len(GATES) - 1))
    check("pair link counts symmetric (links[a,b] == links[b,a]) for every gate/iteration", sym)
    check("n_a_with_b <= n_a and <= links; A sees B iff B sees A", cons)
    check("n_a_with_b non-decreasing with gate (25 <= 50 <= 100)", mono)
    # brute-force verification of the hash search on time slices of the busiest iterations
    bf_ok = True
    detail = []
    S = len(sensors)
    for ik in sorted(iter_n_use, key=lambda k: -iter_n_use[k])[:3]:
        idx = np.flatnonzero(use & (det["iter_key"] == ik))
        idx = idx[np.argsort(t_cal[idx], kind="stable")]
        mid = len(idx) // 2
        sl = idx[mid:mid + 2500]
        t, e, n = t_cal[sl], det["e"][sl], det["n"][sl]
        sc = np.array([scode_of[s] for s in sid_all[sl]], dtype=np.int64)
        hit, links = pair_hits(t, e, n, sc, S)
        dt = np.abs(t[:, None] - t[None, :])
        dist = np.hypot(e[:, None] - e[None, :], n[:, None] - n[None, :])
        diff = sc[:, None] != sc[None, :]
        for g in GATES:
            adj = diff & (dt <= MAX_DT_S) & (dist <= g)
            h_ref = np.zeros((S, S), dtype=np.int64)
            l_ref = np.zeros((S, S), dtype=np.int64)
            for b in range(S):
                seen = adj[:, sc == b].any(axis=1)
                h_ref[:, b] = np.bincount(sc[seen], minlength=S)
                l_ref[:, b] = np.bincount(sc, weights=adj[:, sc == b].sum(axis=1), minlength=S).astype(np.int64)
            bf_ok &= np.array_equal(h_ref, hit[g]) and np.array_equal(l_ref, links[g])
        # groups: brute-force connected components on the same slice
        base = np.floor(t.min())
        tb = np.floor(t - base).astype(np.int64)
        gid = group_labels(tb, e, n, sc)
        link = (tb[:, None] == tb[None, :]) & np.where(diff, dist <= GROUP_GATE_M, dist <= SAME_SENSOR_LINK_M)
        np.fill_diagonal(link, False)
        ref = _bfs_components(link)
        same = gid[:, None] == gid[None, :]
        bf_ok &= bool(np.array_equal(same, ref[:, None] == ref[None, :]))
        detail.append(f"{ik}:{len(sl)}dets/{int(gid.max()) + 1}groups/{int(hit[100.0].sum())}hits@100m")
    check("hash search == brute force (hits, links, 3 gates; groups == BFS components)", bf_ok, "; ".join(detail))
    # pairs table vs in-memory counts
    check("pairs table has gate rows for 25/50/100", {r[4] for r in pair_rows} == set(GATES))
    if not ok:
        print("SELF-CHECKS FAILED", flush=True)
        raise SystemExit(1)
    print("ALL SELF-CHECKS PASSED", flush=True)


def _group_ids_ok(cols) -> bool:
    key = list(zip(cols["iter_key"], cols["variant"]))
    gid = np.asarray(cols["group_id"])
    seen: dict = {}
    for k, g in zip(key, gid.tolist()):
        seen.setdefault(k, []).append(g)
    return all(v == list(range(len(v))) for v in seen.values())


def _bfs_components(adj: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    lab = -np.ones(n, dtype=np.int64)
    for s in range(n):
        if lab[s] >= 0:
            continue
        lab[s] = s
        stack = [s]
        while stack:
            x = stack.pop()
            for y in np.flatnonzero(adj[x] & (lab < 0)).tolist():
                lab[y] = s
                stack.append(y)
    return lab


if __name__ == "__main__":
    raise SystemExit(main())
