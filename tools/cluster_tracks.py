"""K-means pre-classification of position tracks, all sensors in one feature space.

UNSUPERVISED: ground truth is never loaded here. Whether a cluster is "drone-like" is the
job of a later step that compares clusters against Dronetag GT.

Pipeline
  1. Tracks = (iter_key, sensor_key, object_id) over position rows (pos3d / pos2d) that pass
     the quality layer (usable_pos, not far), sorted by t, exact duplicate timestamps dropped
     (first kept). Eligible: >= 5 rows and duration >= 2 s. Every drop is counted per sensor.
  2. Per-track features in raw units (finite differences use anchor points >= 1.0 s apart:
     starting at the first row, the next anchor is the first row at least 1.0 s later).
  3. Three clusterings (variants), each: fit set = eligible, all of the variant's fit features
     finite, not a ground-truth relay sensor (VKYY / QTVS / LJJX); relay tracks are still
     assigned to the nearest centroid (in_fit=0). K-means (K.scan_k, K = 2..12),
     K* = argmax silhouette over K = 3..12, labels for K in {K*, 4, 5, 6, 8}.
       A  8 kinematic + height-above-sensor features (original; only sensors with a known
          site altitude can enter, in practice the radars: columns cluster_kstar, cluster_k4.., in_fit)
       B  5 horizontal kinematic features, every position sensor can enter
          (cluster_b_kstar, cluster_b_k4.., in_fit_b)
       C  A's features + each track's share of rows the SENSOR itself classed as UAV / Air Vehicle /
          Animal-Bird / unknown (cluster_c_kstar, cluster_c_k4.., in_fit_c). These are sensor
          outputs, not ground truth, so the script stays unsupervised.
  4. Report: per-cluster discriminating names (features whose cluster median deviates most from the
     fit-set median), per-cluster sensor shares against the fit-set base rate, and per-sensor
     speed / vz method (reported vs finite difference).

Outputs (data/derived/): track_features.csv, track_clusters.csv, kmeans_tracks.md
Run: python3 tools/cluster_tracks.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuas_clean as Q  # noqa: E402
import cuas_common as C  # noqa: E402
import kmeans as K  # noqa: E402

OUT_FEATURES = C.DERIVED / "track_features.csv"
OUT_CLUSTERS = C.DERIVED / "track_clusters.csv"
OUT_REPORT = C.DERIVED / "kmeans_tracks.md"

MIN_ROWS = 5
MIN_DURATION_S = 2.0
SEG_MIN_DT_S = 1.0
MOVING_MPS = 1.0          # segments slower than this have no meaningful course
KS = list(range(2, 13))
K_CHOICE = range(3, 13)
K_EXTRA = (4, 5, 6, 8)
N_INIT = 10
SEED = 0
SIL_SAMPLE = 4000

FIT_FEATURES = ["log_duration", "slog_dz_med", "log_dz_iqr", "log_speed_med", "log_speed_p90",
                "log_vz_abs_med", "straightness", "log_turn_rate_med"]
# raw feature behind each fitted feature (None = already raw)
RAW_OF_FIT = ["duration_s", "dz_med", "dz_iqr", "speed_med", "speed_p90", "vz_abs_med",
              "straightness", "turn_rate_med"]
NUMERIC_FEATURES = ["duration_s", "n_rows", "rate_hz", "dz_med", "dz_iqr", "speed_med", "speed_p90",
                    "vz_abs_med", "straightness", "turn_rate_med", "range_med",
                    "det_conf_med", "rcs_med", "cls_top_frac"]
REPORT_FEATURES = ["n_rows", "duration_s", "rate_hz", "dz_med", "dz_iqr", "speed_med", "speed_p90",
                   "vz_abs_med", "straightness", "turn_rate_med", "range_med", "rcs_med",
                   "det_conf_med", "frac_uav", "frac_air_vehicle", "frac_bio", "frac_unknown"]

# the sensors' own class evidence per row (cls or cls_sub); sensor outputs, never ground truth
FRAC_FEATURES = ["frac_uav", "frac_air_vehicle", "frac_bio", "frac_unknown"]
UAV_CLASSES = ("UAV", "UAV Rotary Wing", "UAV Fixed Wing", "UAV Transmitter", "FIXED_WING")
BIO_CLASSES = ("Animal", "Bird")
UNKNOWN_CLASSES = ("Unknown", "Other", "")
B_FIT_FEATURES = ["log_duration", "log_speed_med", "log_speed_p90", "straightness", "log_turn_rate_med"]
B_RAW_OF_FIT = ["duration_s", "speed_med", "speed_p90", "straightness", "turn_rate_med"]
C_FIT_FEATURES = FIT_FEATURES + FRAC_FEATURES
C_RAW_OF_FIT = RAW_OF_FIT + FRAC_FEATURES
FLAG_RATIO = 1.5          # sensor share in a cluster vs its share of the fit set
MIN_FLAG_N = 1


# --------------------------------------------------------------------------- helpers

def _median(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def _slog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log10(1.0 + np.abs(x))


def _anchors(t: np.ndarray) -> np.ndarray:
    """Indices of anchor points spaced >= SEG_MIN_DT_S apart (first row, then the first
    row at least 1 s after the previous anchor, ...)."""
    idx = [0]
    i = 0
    while True:
        j = int(np.searchsorted(t, t[i] + SEG_MIN_DT_S, side="left"))
        if j >= len(t):
            break
        idx.append(j)
        i = j
    return np.array(idx)


def _top(values) -> tuple[str, float]:
    """Most frequent value (ties: smallest string) and its fraction."""
    cnt = Counter(values)
    best = min(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    return best[0], best[1] / len(values)


def track_features(t, e, n, alt_c, dz, ve, vn, vu, rng_m, det_conf, rcs, cls, serial, proto) -> dict:
    nr = len(t)
    f: dict = {"n_rows": float(nr), "duration_s": float(t[-1] - t[0])}
    f["rate_hz"] = (nr - 1) / f["duration_s"]
    dz_f = dz[np.isfinite(dz)]
    f["dz_med"] = float(np.median(dz_f)) if len(dz_f) else float("nan")
    f["dz_iqr"] = float(np.subtract(*np.percentile(dz_f, [75, 25]))) if len(dz_f) else float("nan")

    a = _anchors(t)
    seg_a, seg_b = a[:-1], a[1:]
    dt = t[seg_b] - t[seg_a]
    de, dn = e[seg_b] - e[seg_a], n[seg_b] - n[seg_a]
    dist = np.hypot(de, dn)
    seg_speed = dist / dt if len(dt) else np.empty(0)

    vh = np.hypot(ve, vn)
    if np.isfinite(vh).mean() >= 0.5:
        f["speed_med"] = _median(vh)
        f["speed_p90"] = float(np.percentile(vh[np.isfinite(vh)], 90))
        f["speed_src"] = "reported"
    elif len(seg_speed):
        f["speed_med"] = float(np.median(seg_speed))
        f["speed_p90"] = float(np.percentile(seg_speed, 90))
        f["speed_src"] = "finite_diff"
    else:
        f["speed_med"] = f["speed_p90"] = float("nan")
        f["speed_src"] = "none"

    if np.isfinite(vu).mean() >= 0.5:
        f["vz_abs_med"] = _median(np.abs(vu))
        f["vz_src"] = "reported"
    elif len(dt):
        dalt = np.abs(alt_c[seg_b] - alt_c[seg_a]) / dt
        f["vz_abs_med"] = _median(dalt)
        f["vz_src"] = "finite_diff" if np.isfinite(f["vz_abs_med"]) else "none"
    else:
        f["vz_abs_med"] = float("nan")
        f["vz_src"] = "none"

    if len(dt):
        path = float(dist.sum())
        net = float(np.hypot(e[a[-1]] - e[a[0]], n[a[-1]] - n[a[0]]))
        f["straightness"] = net / path if path > 0 else 0.0     # no movement at all -> 0
        mv = seg_speed > MOVING_MPS
        course = np.degrees(np.arctan2(de, dn))
        pair = mv[:-1] & mv[1:]
        if mv.sum() >= 2 and pair.any():
            dc = np.abs((course[1:][pair] - course[:-1][pair] + 180.0) % 360.0 - 180.0)
            dtm = 0.5 * (dt[:-1][pair] + dt[1:][pair])
            f["turn_rate_med"] = float(np.median(dc / dtm))
        else:
            f["turn_rate_med"] = 0.0
    else:
        f["straightness"] = float("nan")
        f["turn_rate_med"] = 0.0

    f["range_med"] = _median(rng_m)
    f["det_conf_med"] = _median(det_conf)
    f["rcs_med"] = _median(rcs)
    f["cls_top"], f["cls_top_frac"] = _top([c if c else "(none)" for c in cls])
    f["has_serial"] = 1 if any(s for s in serial) else 0
    protos = [p for p in proto if p]
    f["protocol_top"] = _top(protos)[0] if protos else ""
    return f


def fit_matrix(feat: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([
        np.log10(feat["duration_s"]),
        _slog(feat["dz_med"]),
        np.log10(1.0 + feat["dz_iqr"]),
        np.log10(1.0 + feat["speed_med"]),
        np.log10(1.0 + feat["speed_p90"]),
        np.log10(1.0 + feat["vz_abs_med"]),
        feat["straightness"],
        np.log10(1.0 + feat["turn_rate_med"]),
    ])


def fit_matrix_b(feat: dict[str, np.ndarray]) -> np.ndarray:
    """Variant B: horizontal kinematics only (no height-above-sensor, no vertical speed)."""
    return np.column_stack([
        np.log10(feat["duration_s"]),
        np.log10(1.0 + feat["speed_med"]),
        np.log10(1.0 + feat["speed_p90"]),
        feat["straightness"],
        np.log10(1.0 + feat["turn_rate_med"]),
    ])


def fit_matrix_c(feat: dict[str, np.ndarray]) -> np.ndarray:
    """Variant C: variant A's features + the sensors' own class-evidence fractions."""
    return np.column_stack([fit_matrix(feat)] + [feat[k] for k in FRAC_FEATURES])


def assign(z: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.sqrt(np.maximum(((z[:, None, :] - centers[None, :, :]) ** 2).sum(2), 0.0))
    lab = d.argmin(1)
    return lab, d[np.arange(len(z)), lab]


def elbow_k(ks, inertia) -> int:
    """Knee of the inertia curve: largest gap below the chord between first and last point."""
    x = (np.array(ks, float) - ks[0]) / (ks[-1] - ks[0])
    y = (np.array(inertia) - inertia[-1]) / (inertia[0] - inertia[-1])
    return int(ks[int(np.argmax((1.0 - x) - y))])


# --------------------------------------------------------------------------- stage 1: tracks

def build_tracks():
    cols = ["tree", "object_id", "t", "cls", "cls_sub", "det_conf", "rcs", "serial", "protocol"]
    d = Q.load_clean(cols)
    is_pos = np.isin(d["meas_kind"], Q.POSITION_KINDS)
    print(f"loaded {len(d['t'])} detections; {int(is_pos.sum())} position rows "
          f"({int((~is_pos).sum())} bearing/other rows are not tracks)")

    pos_idx = np.flatnonzero(is_pos)
    skey = d["sensor_key"][pos_idx]
    ikey = d["iter_key"][pos_idx]
    okey = d["object_id"][pos_idx]
    keys = np.array([f"{a}\x1f{b}\x1f{c}" for a, b, c in zip(ikey, skey, okey)], dtype=object)
    uniq, gid_all = np.unique(keys, return_inverse=True)

    usable = d["usable_pos"][pos_idx]
    far = d["far"][pos_idx]
    t = d["t"][pos_idx]

    # row rules, attributed in order: unusable -> far -> duplicate timestamp
    keep1 = usable
    keep2 = keep1 & ~far
    cand = np.flatnonzero(keep2)
    order = cand[np.lexsort((t[cand], gid_all[cand]))]       # stable: ties keep file order
    dup = np.zeros(len(order), bool)
    dup[1:] = (gid_all[order][1:] == gid_all[order][:-1]) & (t[order][1:] == t[order][:-1])
    kept = order[~dup]
    dup_rows = order[dup]

    sensors = sorted(set(skey.tolist()))
    log = {s: Counter() for s in sensors}
    for s in sensors:
        m = skey == s
        L = log[s]
        L["pos_rows"] = int(m.sum())
        L["tracks_all"] = len(set(gid_all[m].tolist()))
        L["rows_unusable"] = int((m & ~keep1).sum())
        L["rows_far"] = int((m & keep1 & far).sum())
        L["rows_dup_t"] = int(skey[dup_rows].tolist().count(s)) if len(dup_rows) else 0
    gk = gid_all[kept]
    starts = np.flatnonzero(np.r_[True, gk[1:] != gk[:-1]])
    ends = np.r_[starts[1:], len(kept)]
    tracks_after_rows = set(gk[starts].tolist())
    for s in sensors:
        gs = set(gid_all[skey == s].tolist())
        log[s]["tracks_lost_rowfilter"] = len(gs - tracks_after_rows)

    return d, pos_idx, uniq, kept, starts, ends, gk, log, sensors


# --------------------------------------------------------------------------- variants

def is_radar(sensor_key: str) -> bool:
    return sensor_key.endswith("_RADAR")


def nonfinite_reason(finite, X, raw_names, gt_feed) -> np.ndarray:
    out = []
    for i in range(len(finite)):
        if not finite[i]:
            r = "nonfinite:" + ",".join(raw_names[j] for j in np.flatnonzero(~np.isfinite(X[i])))
            out.append(r + (";gt_feed" if gt_feed[i] else ""))
        else:
            out.append("gt_feed_assigned_only" if gt_feed[i] else "")
    return np.array(out, dtype=object)


def run_variant(tag, title, X, fit_names, raw_names, gt_feed, sensor_key) -> dict:
    """Fit one variant: fit set, K scan, K*, labels for K in {K*, 4, 5, 6, 8} for every track."""
    nt = len(X)
    finite = np.isfinite(X).all(1)
    in_fit = finite & (gt_feed == 0)
    reason = nonfinite_reason(finite, X, raw_names, gt_feed)
    sensors = sorted(set(sensor_key.tolist()))
    print(f"\n{'=' * 100}\nVARIANT {tag}: {title}\nfit features ({len(fit_names)}): {', '.join(fit_names)}")
    print(f"fit set: {int(in_fit.sum())} tracks; assigned-only gt_feed: {int((finite & (gt_feed == 1)).sum())}; "
          f"non-finite: {int((~finite).sum())}")
    for r, c in sorted(Counter(x for x in reason if x.startswith("nonfinite")).items()):
        print(f"   {r}: {c}")
    comp_rows = []
    print(f"fit-set composition by sensor_key (variant {tag}):")
    print(f"  {'sensor_key':34s} {'eligible':>8s} {'fit':>7s} {'fit %':>7s} {'assigned-only':>13s} {'non-finite':>10s}")
    for s_ in sensors:
        m = sensor_key == s_
        row = [s_, int(m.sum()), int((m & in_fit).sum()), 100.0 * (m & in_fit).sum() / in_fit.sum(),
               int((m & finite & (gt_feed == 1)).sum()), int((m & ~finite).sum())]
        comp_rows.append(row)
        print(f"  {row[0]:34s} {row[1]:8d} {row[2]:7d} {row[3]:6.2f}% {row[4]:13d} {row[5]:10d}")
    radar = np.array([is_radar(x) for x in sensor_key])
    n_nonradar_fit = int((in_fit & ~radar).sum())
    print(f"  radar share of fit set: {100.0 * (in_fit & radar).sum() / in_fit.sum():.2f}%; "
          f"non-radar tracks in fit: {n_nonradar_fit} from {len({s_ for s_ in sensors if ((sensor_key == s_) & in_fit & ~radar).any()})} sensors")

    scaler = K.robust_scaler(X[in_fit])
    Z_fit = scaler.transform(X[in_fit])
    print(f"K-means {tag}: {len(Z_fit)} x {Z_fit.shape[1]}, K={KS[0]}..{KS[-1]}, n_init={N_INIT}, seed={SEED}")
    scan = K.scan_k(Z_fit, KS, n_init=N_INIT, seed=SEED, sample=SIL_SAMPLE)
    sil = {k: s_ for k, _, s_, _ in scan}
    inertia = {k: i for k, i, _, _ in scan}
    res = {k: r for k, _, _, r in scan}
    kstar = max(K_CHOICE, key=lambda k: (sil[k], -k))
    elbow = elbow_k(KS, [inertia[k] for k in KS])
    scan_lines = ["| K | inertia | drop vs K-1 | silhouette | smallest cluster |", "|---|---|---|---|---|"]
    print(f"\n  K    inertia   drop%   silhouette  smallest   [variant {tag}]")
    for k in KS:
        prev = inertia.get(k - 1)
        drop = f"{100 * (1 - inertia[k] / prev):.1f}%" if prev else ""
        small = int(np.bincount(res[k].labels, minlength=k).min())
        mark = " <- K*" if k == kstar else (" (elbow)" if k == elbow else "")
        print(f"{k:3d} {inertia[k]:10.1f} {drop:>7} {sil[k]:11.4f} {small:8d}{mark}")
        scan_lines.append(f"| {k} | {inertia[k]:.1f} | {drop} | {sil[k]:.4f} | {small} |")
    print(f"K* {tag} = {kstar} (argmax silhouette over K=3..12, silhouette {sil[kstar]:.4f}); elbow K = {elbow}")

    Z_all = np.full((nt, X.shape[1]), np.nan)
    Z_all[finite] = scaler.transform(X[finite])
    clus, dist = {}, {}
    for k in sorted(set(K_EXTRA) | {kstar}):
        lab = np.full(nt, -1, int)
        dd = np.full(nt, np.nan)
        lab[finite], dd[finite] = assign(Z_all[finite], res[k].centers)
        clus[k], dist[k] = lab, dd
        assert (lab[in_fit] == res[k].labels).all()      # fit tracks reproduce K-means' own labels
    return dict(tag=tag, title=title, X=X, fit_names=fit_names, raw_names=raw_names, finite=finite,
                in_fit=in_fit, reason=reason, scaler=scaler, scan_lines=scan_lines, kstar=kstar,
                elbow=elbow, sil=sil, res=res, Z_all=Z_all, clus=clus, dist=dist, comp_rows=comp_rows,
                radar_share=float((in_fit & radar).sum() / in_fit.sum()), n_nonradar_fit=n_nonradar_fit)


# --------------------------------------------------------------------------- main

def main() -> int:
    t_start = time.time()
    d, pos_idx, uniq, kept, starts, ends, gk, log, sensors = build_tracks()
    g = pos_idx[kept]                      # detection-row index for every kept row, sorted by (track, t)

    t_all, e_all, n_all = d["t"][g], d["e"][g], d["n"][g]
    alt_all, dz_all = d["alt_c"][g], d["dz_sensor"][g]
    ve_all, vn_all, vu_all = d["ve_c"][g], d["vn_c"][g], d["vu_c"][g]
    with np.errstate(invalid="ignore"):
        rng_all = np.where(d["sensor_pos_valid"][g] == 1,
                           np.hypot(e_all - d["sensor_e"][g], n_all - d["sensor_n"][g]), np.nan)
    conf_all, rcs_all = d["det_conf"][g], d["rcs"][g]
    cls_all, sub_all, ser_all, pro_all = d["cls"][g], d["cls_sub"][g], d["serial"][g], d["protocol"][g]
    # the sensors' own class evidence, per row (sensor outputs, not ground truth)
    row_flag = {
        "frac_uav": np.isin(cls_all, UAV_CLASSES) | np.isin(sub_all, UAV_CLASSES),
        "frac_air_vehicle": cls_all == "Air Vehicle",
        "frac_bio": np.isin(cls_all, BIO_CLASSES) | np.isin(sub_all, BIO_CLASSES),
        "frac_unknown": np.isin(cls_all, UNKNOWN_CLASSES) & np.isin(sub_all, UNKNOWN_CLASSES),
    }

    meta = []
    for s0, s1 in zip(starts, ends):
        sl = slice(s0, s1)
        parts = uniq[gk[s0]].split("\x1f")
        nr = s1 - s0
        dur = t_all[s1 - 1] - t_all[s0]
        sk = parts[1]
        if nr < MIN_ROWS:
            log[sk]["tracks_lt5_rows"] += 1
            log[sk]["rows_in_lt5"] += nr
            continue
        if dur < MIN_DURATION_S:
            log[sk]["tracks_lt2s"] += 1
            log[sk]["rows_in_lt2s"] += nr
            continue
        f = track_features(t_all[sl], e_all[sl], n_all[sl], alt_all[sl], dz_all[sl], ve_all[sl],
                           vn_all[sl], vu_all[sl], rng_all[sl], conf_all[sl], rcs_all[sl],
                           cls_all[sl], ser_all[sl], pro_all[sl])
        for k_, fl in row_flag.items():
            f[k_] = float(fl[sl].mean())
        first = g[s0]
        f.update(iter_key=parts[0], sensor_key=sk, object_id=parts[2], tree=str(d["tree"][first]),
                 sensor_id=str(d["sensor_id"][first]), gt_feed=int(d["gt_feed"][first]))
        log[sk]["tracks_eligible"] += 1
        log[sk]["rows_eligible"] += nr
        meta.append(f)

    # ---- drop-count table
    print("\nDROP COUNTS per sensor_key (rows are position rows; each row counted at its first failing rule)")
    hdr = ["sensor_key", "pos_rows", "unusable", "far", "dup_t", "in<5rows", "in<2s", "eligible_rows",
           "tracks_all", "lost_rowfilter", "<5rows", "<2s", "eligible"]
    drop_lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    tot = Counter()
    for s in sensors:
        L = log[s]
        vals = [L["pos_rows"], L["rows_unusable"], L["rows_far"], L["rows_dup_t"], L["rows_in_lt5"],
                L["rows_in_lt2s"], L["rows_eligible"], L["tracks_all"], L["tracks_lost_rowfilter"],
                L["tracks_lt5_rows"], L["tracks_lt2s"], L["tracks_eligible"]]
        assert vals[0] == sum(vals[1:7]), (s, vals)
        assert vals[7] == vals[8] + vals[9] + vals[10] + vals[11], (s, vals)
        for h, v in zip(hdr[1:], vals):
            tot[h] += v
        drop_lines.append("| " + " | ".join([s] + [str(v) for v in vals]) + " |")
    drop_lines.append("| **TOTAL** | " + " | ".join(str(tot[h]) for h in hdr[1:]) + " |")
    print("\n".join(drop_lines))

    nt = len(meta)
    cat = {k: np.array([m[k] for m in meta], dtype=object) for k in
           ["iter_key", "tree", "sensor_key", "sensor_id", "object_id", "cls_top", "protocol_top",
            "speed_src", "vz_src"]}
    num = {k: np.array([m[k] for m in meta], dtype=float) for k in NUMERIC_FEATURES + FRAC_FEATURES}
    has_serial = np.array([m["has_serial"] for m in meta], dtype=int)
    gt_feed = np.array([m["gt_feed"] for m in meta], dtype=int)
    order = np.lexsort((cat["object_id"], cat["sensor_key"], cat["iter_key"]))   # stable output order
    cat = {k: v[order] for k, v in cat.items()}
    num = {k: v[order] for k, v in num.items()}
    has_serial, gt_feed = has_serial[order], gt_feed[order]
    print(f"\n{nt} eligible tracks")

    # ---- three variants
    with np.errstate(invalid="ignore", divide="ignore"):
        XA, XB, XC = fit_matrix(num), fit_matrix_b(num), fit_matrix_c(num)
    sk = cat["sensor_key"]
    VA = run_variant("A", "kinematics + height above sensor (radar-only in practice)", XA, FIT_FEATURES,
                     RAW_OF_FIT, gt_feed, sk)
    VB = run_variant("B", "all position sensors, horizontal kinematics only", XB, B_FIT_FEATURES,
                     B_RAW_OF_FIT, gt_feed, sk)
    VC = run_variant("C", "variant A + the sensors' own class evidence", XC, C_FIT_FEATURES,
                     C_RAW_OF_FIT, gt_feed, sk)
    variants = [VA, VB, VC]

    print("\nSUMMARY")
    print(f"  {'variant':8s} {'fit n':>7s} {'radar %':>8s} {'non-radar fit':>13s} {'K*':>3s} {'sil(K*)':>8s} {'sil(K=2)':>9s}")
    for V in variants:
        print(f"  {V['tag']:8s} {int(V['in_fit'].sum()):7d} {100 * V['radar_share']:7.2f}% {V['n_nonradar_fit']:13d} "
              f"{V['kstar']:3d} {V['sil'][V['kstar']]:8.4f} {V['sil'][2]:9.4f}")

    # ---- write tables
    feat_cols = {**cat, **num, "has_serial": has_serial, "gt_feed": gt_feed}
    feat_order = ["iter_key", "tree", "sensor_key", "sensor_id", "object_id", "gt_feed", "duration_s",
                  "n_rows", "rate_hz", "dz_med", "dz_iqr", "speed_med", "speed_p90", "vz_abs_med",
                  "straightness", "turn_rate_med", "range_med", "cls_top", "cls_top_frac",
                  "det_conf_med", "rcs_med", "has_serial", "protocol_top",
                  *FRAC_FEATURES, "speed_src", "vz_src"]
    C.write_columns(OUT_FEATURES, feat_cols, feat_order)
    clu_cols = {"iter_key": cat["iter_key"], "tree": cat["tree"], "sensor_key": cat["sensor_key"],
                "object_id": cat["object_id"], "in_fit": VA["in_fit"].astype(int)}
    for V, sfx in ((VA, ""), (VB, "_b"), (VC, "_c")):
        if sfx:
            clu_cols[f"in_fit{sfx}"] = V["in_fit"].astype(int)
        clu_cols[f"cluster{sfx}_kstar"] = V["clus"][V["kstar"]]
        clu_cols[f"dist{sfx}_kstar"] = V["dist"][V["kstar"]]
        for k in (4, 5, 6, 8):
            clu_cols[f"cluster{sfx}_k{k}"] = V["clus"][k]
    clu_cols["reason"] = VA["reason"]
    clu_cols["reason_b"] = VB["reason"]
    clu_cols["reason_c"] = VC["reason"]
    C.write_columns(OUT_CLUSTERS, clu_cols, list(clu_cols))

    write_report(variants, drop_lines, num, cat, gt_feed)
    print(f"\nwrote {OUT_FEATURES.name}, {OUT_CLUSTERS.name}, {OUT_REPORT.name}")
    print(f"runtime {time.time() - t_start:.1f} s")
    return 0


# --------------------------------------------------------------------------- report

def _fmt(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.0f}" if abs(x) >= 1000 else f"{x:.3g}" if abs(x) < 10 else f"{x:.1f}"


def md_table(header, rows) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def cluster_names(V: dict, labels: np.ndarray, num: dict) -> dict:
    """Name = the 3 fitted features whose cluster median deviates most from the fit-set median
    (deviation in robust-scaled units: 1.0 = one fit-set IQR of that fitted feature), with the
    raw cluster median vs the raw fit-set median. Features are added to clashing names until
    every name is unique."""
    ids = [c for c in sorted(set(labels.tolist())) if c >= 0]
    raw_names, in_fit = V["raw_names"], V["in_fit"]
    fit_med = {r: _median(num[r][in_fit]) for r in raw_names}
    zmed, ordr, raw_med = {}, {}, {}
    for c in ids:
        m = labels == c
        zmed[c] = np.median(V["Z_all"][m], axis=0)
        ordr[c] = np.argsort(-np.abs(zmed[c]), kind="stable")
        raw_med[c] = {r: _median(num[r][m]) for r in raw_names}

    def text(c, n):
        bulk = "near fit-set median (all |dev| < 0.25), top: " if np.abs(zmed[c]).max() < 0.25 else ""
        return bulk + "; ".join(
            f"{'high' if zmed[c][j] > 0 else 'low'} {raw_names[j]} {_fmt(raw_med[c][raw_names[j]])} "
            f"vs {_fmt(fit_med[raw_names[j]])} ({zmed[c][j]:+.1f})" for j in ordr[c][:n])

    nfeat = {c: 3 for c in ids}
    for _ in range(len(raw_names)):
        names = {c: text(c, nfeat[c]) for c in ids}
        cnt = Counter(names.values())
        clash = [c for c in ids if cnt[names[c]] > 1]
        if not clash:
            break
        for c in clash:
            nfeat[c] = min(nfeat[c] + 1, len(raw_names))
    names = {c: text(c, nfeat[c]) for c in ids}
    assert len(set(names.values())) == len(names), "cluster names not unique"
    names[-1] = "non-finite fit features (not clustered)"
    return names


def _pct(x: float) -> str:
    return f"{100 * x:.2f}%" if x < 0.01 else f"{100 * x:.1f}%"


def share_table(labels, in_fit, sk, ids) -> tuple[list[str], list[str]]:
    """Per-cluster sensor shares among fit-set members next to the fit-set base rate; a cell is
    flagged (*) when the sensor's share in the cluster exceeds FLAG_RATIO x its base rate."""
    fit_sensors = sorted(set(sk[in_fit].tolist()))
    tot = int(in_fit.sum())
    base = {s_: ((sk == s_) & in_fit).sum() / tot for s_ in fit_sensors}
    rows = [["**fit-set base rate**", tot, *[f"{_pct(base[s_])} ({int(((sk == s_) & in_fit).sum())})"
                                              for s_ in fit_sensors]]]
    flags = []
    for c in ids:
        m = (labels == c) & in_fit
        n = int(m.sum())
        cells = []
        for s_ in fit_sensors:
            k = int((m & (sk == s_)).sum())
            share = k / n if n else 0.0
            flag = n > 0 and k >= MIN_FLAG_N and share > FLAG_RATIO * base[s_]
            cells.append(f"{_pct(share)} ({k}){' *' if flag else ''}")
            if flag:
                flags.append((c, s_, share, base[s_], share / base[s_], k, n))
        rows.append([c, n, *cells])
    out = md_table(["cluster", "fit n", *fit_sensors], rows)
    lines = []
    if flags:
        lines.append(f"Flagged (*): sensor share in the cluster > {FLAG_RATIO:g}x its fit-set base rate "
                     "(small counts are noisy; `k/n` is shown):")
        lines.append("")
        for c, s_, share, b, r, k, n in sorted(flags, key=lambda x: (x[0], -x[4])):
            lines.append(f"- cluster {c}: {s_} {_pct(share)} ({k}/{n}) vs base {_pct(b)} = {r:.1f}x")
    else:
        lines.append(f"No cluster has a sensor share above {FLAG_RATIO:g}x its fit-set base rate.")
    return out, lines


def cluster_section(V, title, k, num, cat, gt_feed) -> list[str]:
    labels, in_fit = V["clus"][k], V["in_fit"]
    sk = cat["sensor_key"]
    out = [f"### {title}", ""]
    ids = sorted(set(labels.tolist()))
    pos_ids = [c for c in ids if c >= 0]
    sizes = {c: int((labels == c).sum()) for c in ids}
    names = cluster_names(V, labels, num)
    rows = []
    for c in ids:
        m = labels == c
        med = {f: _median(num[f][m]) for f in REPORT_FEATURES}
        rows.append([c, sizes[c], int((m & in_fit).sum()), int((m & (cat["tree"] == "evaluation")).sum()),
                     *[_fmt(med[f]) for f in REPORT_FEATURES], names[c]])
    out += ["Centroid = median of members in ORIGINAL units. `n` = all assigned eligible tracks, "
            "`fit` = those used to fit K-means (the rest are ground-truth relay sensors or non-finite), "
            "`eval` = evaluation-tree tracks. Name = the 3 fitted features whose cluster median deviates most "
            "from the fit-set median, as `high|low feature cluster-median vs fit-set-median (deviation)`; "
            "deviation is in robust-scaled units of the FITTED (log-transformed) feature, 1.0 = one fit-set IQR "
            "(features are added to a name until it is unique). Names describe kinematics / sensor evidence only.", ""]
    out += md_table(["cluster", "n", "fit", "eval", *REPORT_FEATURES, "name (largest deviations from fit-set median)"],
                    rows)
    out.append("")

    sensors = sorted(set(sk.tolist()))
    out += ["Composition by sensor_key (tracks, all assigned):", ""]
    out += md_table(["cluster", *sensors, "total"],
                    [[c, *[int(((labels == c) & (sk == s_)).sum()) for s_ in sensors], sizes[c]] for c in ids])
    out += ["", f"Sensor share of each cluster's FIT members vs the fit-set base rate (n in brackets; * = share > "
                f"{FLAG_RATIO:g}x base rate, so the cluster may reflect a sensor rather than a behaviour):", ""]
    tbl, flag_lines = share_table(labels, in_fit, sk, pos_ids)
    out += tbl + [""] + flag_lines
    out += ["", "Composition by tree (tracks):", ""]
    trees = sorted(set(cat["tree"].tolist()))
    out += md_table(["cluster", *trees], [[c, *[int(((labels == c) & (cat["tree"] == t)).sum()) for t in trees]]
                                          for c in ids])
    out += ["", "Contingency vs the sensors' own classification (`cls_top` = most frequent cls in the track):", ""]
    classes = sorted(set(cat["cls_top"].tolist()))
    out += md_table(["cluster", *classes], [[c, *[int(((labels == c) & (cat["cls_top"] == k_)).sum())
                                                  for k_ in classes]] for c in ids])
    out.append("")
    return out


def method_section(cat, num) -> list[str]:
    sk = cat["sensor_key"]
    rows = []
    for s_ in sorted(set(sk.tolist())):
        m = sk == s_
        sp, vz = cat["speed_src"][m], cat["vz_src"][m]
        vzv = num["vz_abs_med"][m]
        fin = np.isfinite(vzv)
        rows.append([s_, int(m.sum()),
                     int((sp == "reported").sum()), int((sp == "finite_diff").sum()), int((sp == "none").sum()),
                     int((vz == "reported").sum()), int((vz == "finite_diff").sum()), int((vz == "none").sum()),
                     _fmt(_median(num["speed_med"][m])), _fmt(_median(num["speed_p90"][m])),
                     _fmt(_median(vzv)),
                     f"{100 * float((vzv[fin] == 0).mean()):.0f}%" if fin.any() else "-"])
    out = ["## Speed and vz method per sensor", "",
           "A track uses the sensor's REPORTED horizontal velocity (cleaned `ve`,`vn`) when >= 50 % of its rows have "
           "one, otherwise the speed of finite differences between anchors >= 1 s apart (`finite_diff`); likewise "
           "`vz_abs_med` is the reported `vu` (reported), or the finite difference of cleaned altitude. `none` = "
           "no finite value. Finite-difference speeds add position noise and inflate `speed_p90`; a finite-difference "
           "vz of a sensor with a constant altitude is exactly 0. Read a cluster's speed / vz together with its sensor "
           "shares so a behaviour cluster is not mistaken for a sensor artefact.", ""]
    out += md_table(["sensor_key", "tracks", "speed: reported", "speed: finite_diff", "speed: none",
                     "vz: reported", "vz: finite_diff", "vz: none", "median speed_med", "median speed_p90",
                     "median vz_abs_med", "tracks with vz_abs_med == 0"], rows)
    out.append("")
    return out


def variant_section(V, num, cat, gt_feed, first: bool) -> list[str]:
    tag, kstar = V["tag"], V["kstar"]
    sil = V["sil"]
    fin, in_fit = V["finite"], V["in_fit"]
    head = f"## K scan (K-means, n_init 10, seed 0, silhouette on a 4000-point sample)" if first else \
        f"## Variant {tag}: K scan (K-means, n_init 10, seed 0, silhouette on a 4000-point sample)"
    out = [f"## Variant {tag}: {V['title']}", "",
           f"- fit features ({len(V['fit_names'])}, robust-scaled median/IQR): {', '.join(V['fit_names'])}; "
           "slog(x)=sign(x)*log10(1+|x|)",
           f"- fit set: {int(in_fit.sum())} tracks; relay (gt_feed) tracks assigned only: "
           f"{int((fin & (gt_feed == 1)).sum())}; non-finite fit features (cluster -1): {int((~fin).sum())} "
           f"(of which {int(((~fin) & (gt_feed == 1)).sum())} relay)",
           "- non-finite reasons: " + (", ".join(f"{r} = {c}" for r, c in sorted(
               Counter(x for x in V["reason"] if x.startswith("nonfinite")).items())) or "none"),
           f"- fit set is {100 * V['radar_share']:.1f} % radar tracks; non-radar tracks in the fit: "
           f"{V['n_nonradar_fit']}", "", "Fit-set composition by sensor_key:", ""]
    out += md_table(["sensor_key", "eligible tracks", "in fit", "fit %", "assigned only (relay)", "non-finite"],
                    [[r[0], r[1], r[2], f"{r[3]:.2f}%", r[4], r[5]] for r in V["comp_rows"]])
    out += ["", head, ""]
    out += V["scan_lines"]
    out += ["", f"K* = **{kstar}** (argmax silhouette over K = 3..12, silhouette {sil[kstar]:.4f}). "
            f"K = 2 silhouette {sil[2]:.4f}. Elbow of the inertia curve (max gap below the chord, K = 2..12): K = "
            f"{V['elbow']}. Silhouette with low absolute values means the clusters are regions of a continuum "
            "rather than well-separated groups.", ""]
    out += cluster_section(V, f"Variant {tag}, K* = {kstar}", kstar, num, cat, gt_feed)
    if kstar != 6:
        out += cluster_section(V, f"Variant {tag}, K = 6", 6, num, cat, gt_feed)
    return out


def write_report(variants, drop_lines, num, cat, gt_feed) -> None:
    VA = variants[0]
    nt = len(gt_feed)
    finite, in_fit = VA["finite"], VA["in_fit"]
    L = ["# K-means pre-classification of sensor tracks", "",
         "Generated by `tools/cluster_tracks.py`. Unsupervised: no ground truth is read (variant C uses the "
         "sensors' own `cls` / `cls_sub` outputs, which are sensor data, not ground truth). Cluster ids "
         "are 0-based, ordered by descending size; -1 = a fit feature is non-finite. "
         "A cluster is a kinematic (/ sensor-evidence) grouping of tracks, NOT a verified object class; "
         "mapping clusters to drones is done separately against ground truth.", "",
         "Three clusterings are fitted on the same tracks:", "",
         "- **Variant A** (columns `cluster_kstar`, `cluster_k4`.., `in_fit`): 8 kinematic + height-above-sensor "
         "features. `dz_med` (height above sensor) exists only where the sensor's site altitude is known, so in "
         f"practice only the radars can enter: the fit set is {100 * VA['radar_share']:.1f} % radar.",
         "- **Variant B** (`cluster_b_*`, `in_fit_b`): 5 horizontal kinematic features "
         f"({', '.join(B_FIT_FEATURES)}); every eligible position track with these finite enters.",
         "- **Variant C** (`cluster_c_*`, `in_fit_c`): variant A's features + the track's share of rows the SENSOR "
         "classed as UAV (cls or cls_sub in UAV / UAV Rotary Wing / UAV Fixed Wing / UAV Transmitter / FIXED_WING), "
         "Air Vehicle (cls), Animal-or-Bird (cls or cls_sub), unknown (cls AND cls_sub in Unknown / Other / empty). "
         "The four fractions need not sum to 1 (Land Vehicle, Home, ... are none of them). Its fit set equals "
         "variant A's.", "",
         "## Tracks and fit set", "",
         f"- track key = (iter_key, sensor_key, object_id); rows = position rows with `usable_pos` and "
         f"not `far`, duplicate timestamps dropped (first kept); eligible = >= {MIN_ROWS} rows and "
         f">= {MIN_DURATION_S:g} s",
         f"- eligible tracks: {nt}",
         "- relay (gt_feed) sensors VKYY / QTVS / LJJX are excluded from every fit and assigned afterwards (`in_fit*` = 0)",
         "- variant summary:", ""]
    rows = []
    for V in variants:
        fin, itf = V["finite"], V["in_fit"]
        rows.append([V["tag"], V["title"], int(itf.sum()), f"{100 * V['radar_share']:.1f}%", V["n_nonradar_fit"],
                     int((fin & (gt_feed == 1)).sum()), int((~fin).sum()), V["kstar"],
                     f"{V['sil'][V['kstar']]:.4f}", f"{V['sil'][2]:.4f}"])
    L += md_table(["variant", "features", "fit n", "radar share of fit", "non-radar tracks in fit",
                   "relay assigned only", "non-finite (-1)", "K*", "silhouette K*", "silhouette K=2"], rows)
    L += ["", "### Drop counts per sensor_key", "",
          "Row columns partition `pos_rows` (each row counted at its first failing rule: unusable = "
          "null island / self report / non-finite e,n; far = > 15 km; dup_t = repeated timestamp; "
          "in<5rows / in<2s = rows of tracks failing the eligibility rule; eligible_rows = kept). "
          "Track columns partition `tracks_all` (lost_rowfilter = every row removed by row rules).", ""]
    L += drop_lines + [""]
    L += method_section(cat, num)
    for i, V in enumerate(variants):
        L += variant_section(V, num, cat, gt_feed, first=(i == 0))
    ev = cat["tree"] == "evaluation"
    L += ["## Evaluation-tree tracks per cluster", ""]
    rows = []
    for V in variants:
        for k in sorted({V["kstar"], 6}):
            cnt = Counter(V["clus"][k][ev].tolist())
            rows.append([f"{V['tag']}, K={k}", ", ".join(f"{c}: {cnt[c]}" for c in sorted(cnt)), int(ev.sum())])
    L += md_table(["labelling", "evaluation tracks per cluster", "total evaluation tracks"], rows)
    L.append("")
    OUT_REPORT.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
