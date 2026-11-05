#!/usr/bin/env python3
"""How the scores, and the best DECISIONS, change with the (unpublished) GOSPA parameters.

The rules do not publish the GOSPA parameters. Everything so far was scored at the PRIMARY
proxy c = 50 m, p = 2, 3-D. This tool re-scores the existing submissions on a grid and asks
three questions: (1) where do scores collapse as c shrinks or p changes, (2) does the best
report-threshold decision of F0 change with c / p, (3) how much would removing the slow
per-track sensor biases be worth if the official c is small.

Terms (also explained in sensitivity.md)
  c       the cut-off distance in metres: a truth point and a track row closer than c can be
          matched, anything farther counts as one miss plus one false track (c^p/2 each).
  p       the exponent on matched position errors (p = 1: errors add up linearly, p = 2: big
          errors hurt much more than small ones).
  dims    2 = horizontal distance only, 3 = also height.
  combined = 0.4 track + 0.2 classification + 0.1 identification + 0.3 metadata (score.py).

Grid and scope
  c in {5, 10, 15, 20, 25, 30, 40, 50, 75, 100} m, p in {1, 2}, dims in {2, 3}: 40
  configurations, each scored in both truth modes ('all' incl. parked drones, 'airborne') on
  the 6 testing iterations. For a configuration, that configuration is the PRIMARY one of
  score.score_rows, i.e. the label recalls use its assignment (score.score_rows(configs=[cfg],
  primary=cfg_name)). The c = 50, p = 2, 3-D cell is therefore identical to the default score.

  Methods (their written testing files; testing validation only): F0_fuse, O3_fuse_oracle,
  F1_fuse (if its files exist), B1_<sensor> for every deployable sensor that has files except
  QTVS / VKYY (they relay the Dronetag feed in testing, so a ground-truth score on them is
  leakage; LJJX is not in evaluation). B2_allpos only if all 6 files exist.

  Oracle rows (in memory, testing only, never written as submissions), built from O3's rows
  matched to their flight (tracks.csv column `flight`, truth of the scored mode):
    bias        per-track bias removed: from every row of a track subtract the track's median
                3-D error vector (componentwise median of row - flight truth at that step,
                taken over the rows that have a truth at their step).
    height      u replaced by the flight's truth u where the row has a truth at its step.
    horizontal  e, n replaced by the truth e, n likewise.
  Gain of an oracle = oracle score - O3 score, per configuration. Tracks without a hit and
  rows without a truth at their step are left unchanged.

  Decision sensitivity (IN-SAMPLE, uses the outer p of data/derived/fused/<slug>__tracks.csv
  and the fused rows of <slug>__grid.npz, whose geometry is each iteration's applied fold
  geometry): F0's rule "report a track iff p_max_<P_FIELD> >= TAU" for P_FIELD in {all, air}
  x TAU in {0.02, 0.05, 0.1, 0.2, 0.3, 0.5}, labels as F0 (classification from
  cls_fixed_share > 0.5, identity Friendly iff the serial is on the evaluation whitelist,
  model / protocol / serial from the track table, band Unknown). Objective, as in fuse.py:
  mean over iterations of (combined_all + combined_airborne) / 2. 'F0_actual' = the tracks
  F0 really reported (reported_f0, each iteration with its own fold's choice). The best choice
  is picked on the same 6 iterations it is scored on, so it is optimistic; it only shows
  whether the choice MOVES with the parameters.

Outputs (data/derived/gospa_sensitivity/)
  sensitivity.csv  one row per (kind, method, variant, iter_key, truth_mode, config): kind in
      method | oracle | decision; method (F0_fuse, ...), variant ('' | bias | height |
      horizontal | decision name), iter_key, truth_mode, config, c, p, dims, then combined,
      track, classification, identity, metadata, missed_rate, false_rate, loc_rms_m,
      n_truth_rows, n_est_rows, n_assigned, n_missed, n_false.
  sensitivity.md   plain-language summary (what c and p mean, per-iteration tables, where
      scores collapse, where the decision flips, the bias-oracle value per c, self-checks).

Self-checks (the run exits non-zero if one fails)
  1 the c = 50 / p = 2 / 3-D cells equal score.score_rows' defaults exactly (parts and the
    cell; the other default configurations' track cells too), F0 and O3, every iteration, mode;
  2 per-config track scores of F0 equal an independent recomputation with gospa.gospa_sequence
    on a freshly loaded truth, and the per-step costs equal gospa.gospa() on sampled steps;
  3 the bias-removed oracle reduces each track's median error to ~0 (after the lat/lon round
    trip, < 5 mm) after having been > 1 m somewhere (not vacuous), and never increases the
    missed rows at c = 100 (p = 2, 3-D) relative to O3;
  4 the reconstruction of F0's reported set from tracks.csv + grid.npz reproduces the F0
    submission's scores exactly;
  5 the oracle row sets match their flights (O3 tracks are all in tracks.csv with a flight).
  (compare.py has its own check that its win counts equal a loop over printed differences.)

Usage: python3 tools/gospa_sensitivity.py [--workers N]     (about 10-20 minutes)
Deterministic; Python 3 + numpy + openpyxl only. Never prints absolute coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import baselines as B  # noqa: E402
import compare as CMP  # noqa: E402
import cuas_common as C  # noqa: E402
import gospa as G  # noqa: E402
import score as SC  # noqa: E402
import submission as S  # noqa: E402
import truth as T  # noqa: E402

OUT = C.DERIVED / "gospa_sensitivity"
FUSED = C.DERIVED / "fused"
CS = (5, 10, 15, 20, 25, 30, 40, 50, 75, 100)
PS = (1, 2)
DIMS = (2, 3)
CONFIGS = tuple((CMP.cfg_name(c, p, d), float(c), float(p), d) for p in PS for d in DIMS for c in CS)
PRIMARY = SC.PRIMARY
MODES = ("all", "airborne")
P_FIELDS = ("all", "air")
TAUS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
DECISIONS = tuple(f"{f}@{t:g}" for f in P_FIELDS for t in TAUS)
EXCLUDED_SENSORS = ("QTVS", "VKYY", "LJJX")
ORACLES = ("bias", "bias_norelay", "height", "horizontal")
RELAY_SENSORS = ("VKYY", "QTVS", "LJJX")   # relay the Dronetag feed in testing
FIELDS = ["kind", "method", "variant", "iter_key", "truth_mode", "config", "c", "p", "dims",
          "combined", "track", "classification", "identity", "metadata", "missed_rate",
          "false_rate", "loc_rms_m", "n_truth_rows", "n_est_rows", "n_assigned", "n_missed",
          "n_false"]
NUMERIC = frozenset(FIELDS[6:])


# --------------------------------------------------------------------------- scoring helpers


def score_all(it: C.Iteration, rows: dict, mode: str, kind: str, method: str, variant: str,
              keep: bool = False) -> tuple[list[dict], dict]:
    """Score `rows` on every configuration (each as its own primary). -> (records, full)."""
    recs, full = [], {}
    for cfg in CONFIGS:
        res = SC.score_rows(it, rows, mode, configs=[cfg], primary=cfg[0])
        r, pa = res[cfg[0]], res["parts"]
        recs.append({
            "kind": kind, "method": method, "variant": variant, "iter_key": it.key,
            "truth_mode": mode, "config": cfg[0], "c": cfg[1], "p": cfg[2], "dims": cfg[3],
            "combined": pa["combined"], "track": pa["track"],
            "classification": pa["classification"], "identity": pa["identity"],
            "metadata": pa["metadata"], "missed_rate": r["missed_rate"],
            "false_rate": r["false_rate"], "loc_rms_m": r["loc_rms_m"],
            "n_truth_rows": res["n_truth_rows"], "n_est_rows": res["n_est_rows"],
            "n_assigned": r["n_assigned"], "n_missed": r["n_missed"], "n_false": r["n_false"]})
        if keep:
            full[cfg[0]] = res
    return recs, full


def same(a, b) -> bool:
    """Exact equality, NaN == NaN, through dicts / tuples / lists."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return a == b or (a != a and b != b)
    return a == b


def read_rows(method: str, it: C.Iteration):
    p = C.submission_path(method, it)
    if not p.exists():
        return None
    try:
        return S.read_submission(it, p)
    except Exception as exc:  # file being rewritten by another agent, or invalid
        print(f"  [skip] {method} {it.key}: {type(exc).__name__}: {exc}", flush=True)
        return None


def method_list() -> list[str]:
    its = SC.testing_iterations()
    out = ["F0_fuse", "O3_fuse_oracle", "F1_fuse"]
    out += [f"B1_{s}" for s in sorted(C.deployable_sensor_ids()) if s not in EXCLUDED_SENSORS]
    if all(C.submission_path("B2_allpos", it).exists() for it in its):
        out.append("B2_allpos")
    return [m for m in out if any(C.submission_path(m, it).exists() for it in its)]


# --------------------------------------------------------------------------- section: methods


def independent_track(it, rows, mode, cfg) -> tuple[float, dict]:
    """Track score by gospa.gospa_sequence on a freshly loaded truth (not score.cached_truth)
    and the iteration frame; also per-step costs by gospa.gospa() on sampled steps."""
    name, c, p, d = cfg
    tr = T.load_truth(it, mode)
    n_steps = len(tr.grid_t)
    e, n, u = C.iteration_frame(it).to_enu(np.asarray(rows["latitude"], float),
                                           np.asarray(rows["longitude"], float),
                                           np.asarray(rows["altitude_m"], float))
    y = np.column_stack([e, n, u])[:, :d]
    x = np.column_stack([tr.e, tr.n, tr.u])[:, :d]
    sx, sy = np.asarray(tr.step, np.int64), np.asarray(rows["step"], np.int64)
    seq = G.gospa_sequence(sx, x, sy, y, n_steps, c, p)
    worst = float((c ** p / 2.0 * (seq["n_truth"] + seq["n_est"])).sum())
    track = 1.0 - float(seq["cost"].sum()) / worst
    both = np.flatnonzero((seq["n_truth"] > 0) & (seq["n_est"] > 0))
    multi = np.flatnonzero((seq["n_truth"] >= 2) & (seq["n_est"] >= 2))
    pick = list(both[np.linspace(0, len(both) - 1, min(40, len(both))).astype(int)]) if len(both) else []
    if len(multi):
        pick += list(multi[np.linspace(0, len(multi) - 1, min(20, len(multi))).astype(int)])
    worst_gap = 0.0
    for s in pick:
        r = G.gospa(x[sx == s], y[sy == s], c, p)
        worst_gap = max(worst_gap, abs(r.cost - seq["cost"][s]) / (1.0 + abs(r.cost)))
    return track, {"steps": len(pick), "max_rel_gap": worst_gap}


def task_methods(it: C.Iteration) -> dict:
    recs, checks = [], []
    methods = method_list()
    for m in methods:
        rows = read_rows(m, it)
        if rows is None:
            continue
        for mode in MODES:
            keep = m in ("F0_fuse", "O3_fuse_oracle")
            rr, full = score_all(it, rows, mode, "method", m, "", keep=keep)
            recs += rr
            if keep:
                dflt = SC.score_rows(it, rows, mode)          # score.py defaults, untouched
                own = full[PRIMARY]
                ok = same(dflt[PRIMARY], own[PRIMARY]) and same(dflt["parts"], own["parts"]) \
                    and same(dflt["classification"], own["classification"]) \
                    and same(dflt["identity"], own["identity"]) and same(dflt["metadata"], own["metadata"])
                ok_other = all(same(dflt[c[0]], full[c[0]][c[0]]) for c in SC.CONFIGS[1:])
                checks.append((f"1 {m} {it.key} {mode}: c50/p2/3-D cell == score_rows defaults "
                               f"(+ the {len(SC.CONFIGS) - 1} other default track cells)",
                               bool(ok and ok_other), ""))
            if m == "F0_fuse":
                by = {r["config"]: r for r in rr}
                for cfg in (parse(PRIMARY), parse("c20_p2_3d"), parse("c100_p2_3d"), parse("c25_p1_2d")):
                    track, info = independent_track(it, rows, mode, cfg)
                    gap = abs(track - by[cfg[0]]["track"])
                    checks.append((f"2 F0 {it.key} {mode} {cfg[0]}: track score == independent "
                                   f"gospa_sequence recomputation; per-step gospa() on {info['steps']} "
                                   f"steps", gap < 1e-12 and info["max_rel_gap"] < 1e-9,
                                   f"gap {gap:.1e}, step gap {info['max_rel_gap']:.1e}"))
    return {"records": recs, "checks": checks}


def parse(name: str) -> tuple:
    return next(c for c in CONFIGS if c[0] == name)


# --------------------------------------------------------------------------- section: oracle


def read_tracks(it: C.Iteration) -> dict:
    return C.read_columns(FUSED / f"{C.iter_slug(it.key)}__tracks.csv",
                          numeric={"track", "p_max_all", "p_max_air", "cls_fixed_share", "reported_f0",
                                   "label_drone"})


def oracle_rows(it: C.Iteration, rows: dict, tid_flight: dict, mode: str,
                relay_tids: frozenset = frozenset()) -> tuple[dict, dict]:
    """The three oracle variants of `rows` plus diagnostics (bias vectors, hits)."""
    tr = SC.cached_truth(it, mode)
    fr = C.iteration_frame(it)
    e, n, u = fr.to_enu(np.asarray(rows["latitude"], float), np.asarray(rows["longitude"], float),
                        np.asarray(rows["altitude_m"], float))
    tids = np.asarray(rows["track_id"], dtype=str)
    uniq, inv = np.unique(tids, return_inverse=True)
    fidx = {f["flight_key"]: i for i, f in enumerate(tr.flights)}
    f_of_track = np.array([fidx.get(tid_flight.get(t, ""), -1) for t in uniq], dtype=np.int64)
    fi = f_of_track[inv]
    nf = max(len(tr.flights), 1)
    step = np.asarray(rows["step"], np.int64)
    tkey = tr.step.astype(np.int64) * nf + tr.flight.astype(np.int64)
    rkey = step * nf + np.maximum(fi, 0)
    pos = np.minimum(np.searchsorted(tkey, rkey), max(tkey.size - 1, 0))
    hit = (fi >= 0) & ((tkey[pos] == rkey) if tkey.size else np.zeros(rkey.size, bool))
    te, tn, tu = tr.e[pos], tr.n[pos], tr.u[pos]
    err = np.column_stack([e - te, n - tn, u - tu])
    med = np.zeros((len(uniq), 3))
    has = np.zeros(len(uniq), bool)
    for j in range(len(uniq)):
        m = hit & (inv == j)
        if m.any():
            med[j] = np.median(err[m], axis=0)
            has[j] = True
    shift = med[inv] * has[inv][:, None]
    is_relay = np.array([t in relay_tids for t in uniq], bool)[inv]
    shift_nr = shift * (~is_relay)[:, None]   # relay-bearing tracks left as they are

    def make(e2, n2, u2):
        lat, lon, alt = fr.to_geodetic(e2, n2, u2)
        new = dict(rows)
        new["latitude"], new["longitude"], new["altitude_m"] = lat, lon, alt
        return new

    out = {"bias": make(e - shift[:, 0], n - shift[:, 1], u - shift[:, 2]),
           "bias_norelay": make(e - shift_nr[:, 0], n - shift_nr[:, 1], u - shift_nr[:, 2]),
           "height": make(e, n, np.where(hit, tu, u)),
           "horizontal": make(np.where(hit, te, e), np.where(hit, tn, n), u)}
    diag = {"med_before": med[has], "hit_share": float(hit.mean()) if hit.size else 0.0,
            "n_tracks": int(len(uniq)), "n_no_flight": int((f_of_track < 0).sum()),
            "uniq": uniq, "med": med, "has": has,
            "inv": inv, "hit": hit, "pos": pos, "tr": tr, "fr": fr}
    return out, diag


def task_oracle(it: C.Iteration) -> dict:
    recs, checks = [], []
    rows = read_rows("O3_fuse_oracle", it)
    if rows is None:
        return {"records": recs, "checks": [(f"5 O3 rows exist for {it.key}", False, "missing file")]}
    tcols = read_tracks(it)
    tid_flight = {t: f for t, f, ld in zip(tcols["track_id"], tcols["flight"], tcols["label_drone"]) if ld == 1.0}
    unknown = set(np.unique(np.asarray(rows["track_id"], dtype=str)).tolist()) - set(tid_flight)
    checks.append((f"5 {it.key}: every O3 track is a label_drone track of tracks.csv with a flight",
                   not unknown and all(tid_flight[t] for t in set(rows["track_id"])), f"{len(unknown)} unknown"))
    has_relay = lambda sid: bool(set(str(sid).split("|")) & set(RELAY_SENSORS))
    relay_tids = frozenset(str(t) for t, sid in zip(tcols["track_id"], tcols["sensor_ids"]) if has_relay(sid))
    ld = np.asarray(tcols["label_drone"], float) == 1.0
    rel_all = np.array([str(t) in relay_tids for t in tcols["track_id"]], bool)
    rep = np.asarray(tcols["reported_f0"], float) == 1.0
    relay = {"n_drone_tracks": int(ld.sum()), "n_relay_drone_tracks": int((ld & rel_all).sum()),
             "n_f0_reported": int(rep.sum()), "n_relay_f0_reported": int((rep & rel_all).sum()),
             "n_o3_tracks": len(set(map(str, rows["track_id"]))),
             "n_relay_o3_tracks": len(set(map(str, rows["track_id"])) & relay_tids)}
    for mode in MODES:
        variants, diag = oracle_rows(it, rows, tid_flight, mode, relay_tids)
        if mode == "all":
            rl = np.array([str(t) in relay_tids for t in diag["uniq"]], bool) & diag["has"]
            ot = diag["has"] & ~rl
            nrm = np.linalg.norm(diag["med"], axis=1)
            relay["med_err_relay_m"] = float(np.median(nrm[rl])) if rl.any() else float("nan")
            relay["med_err_other_m"] = float(np.median(nrm[ot])) if ot.any() else float("nan")
        base = score_all(it, rows, mode, "oracle_base", "O3_fuse_oracle", "", keep=False)[0]
        by_base = {r["config"]: r for r in base}
        for v in ORACLES:
            rr, _ = score_all(it, variants[v], mode, "oracle", "O3_fuse_oracle", v)
            recs += rr
            if v == "bias":
                # check 3a: per-track median error after the round trip
                fr, tr, inv, hit, pos = diag["fr"], diag["tr"], diag["inv"], diag["hit"], diag["pos"]
                e, n, u = fr.to_enu(variants[v]["latitude"], variants[v]["longitude"], variants[v]["altitude_m"])
                err = np.column_stack([e - tr.e[pos], n - tr.n[pos], u - tr.u[pos]])
                worst = 0.0
                for j in range(int(inv.max()) + 1 if len(inv) else 0):
                    m = hit & (inv == j)
                    if m.any():
                        worst = max(worst, float(np.abs(np.median(err[m], axis=0)).max()))
                before = float(np.abs(diag["med_before"]).max()) if len(diag["med_before"]) else 0.0
                checks.append((f"3a {it.key} {mode}: bias oracle median error after {worst:.1e} m "
                               f"(< 5 mm), before up to {before:.1f} m (> 1 m); hit share "
                               f"{diag['hit_share']:.2f}", worst < 5e-3 and before > 1.0, ""))
                miss_o = next(r for r in rr if r["config"] == "c100_p2_3d")["n_missed"]
                miss_b = by_base["c100_p2_3d"]["n_missed"]
                checks.append((f"3b {it.key} {mode}: bias oracle missed rows at c=100 ({miss_o}) <= O3 "
                               f"({miss_b})", miss_o <= miss_b, ""))
    return {"records": recs, "checks": checks, "relay": relay}


# --------------------------------------------------------------------------- section: decision


def task_decision(it: C.Iteration) -> dict:
    recs, checks = [], []
    tc = read_tracks(it)
    z = np.load(FUSED / f"{C.iter_slug(it.key)}__grid.npz", allow_pickle=False)
    g = {k: z[k] for k in ("track", "step", "e", "n", "u")}
    nt = len(tc["track"])
    wl = B.load_whitelist()
    serial = np.array([s or "" for s in tc["serial"]], dtype=object)
    lab = {
        "classification": np.where(tc["cls_fixed_share"] > 0.5, "Fixed Wing UAS", "Rotary Wing UAS").astype(object),
        "identity": np.array(["Friendly" if s and s in wl else "Hostile" for s in serial], dtype=object),
        "uav_model/manufacturer": np.asarray(tc["model"], dtype=object),
        "protocol": np.asarray(tc["protocol"], dtype=object),
        "serial_number": np.array([s or "Unknown" for s in serial], dtype=object),
        "signal_frequency_band": np.full(nt, "Unknown", dtype=object),
    }
    tid = np.asarray(tc["track_id"], dtype=object)
    fr = C.iteration_frame(it)
    lat, lon, alt = fr.to_geodetic(g["e"], g["n"], g["u"])
    pmax = {"all": tc["p_max_all"], "air": tc["p_max_air"]}

    def rows_for(sel: np.ndarray) -> dict:
        idx = np.flatnonzero(sel[g["track"]])
        trk = g["track"][idx]
        out = {"step": g["step"][idx].astype(np.int64), "latitude": lat[idx], "longitude": lon[idx],
               "altitude_m": alt[idx], "track_id": tid[trk]}
        for c, v in lab.items():
            out[c] = v[trk]
        return out

    sels = {}
    for f in P_FIELDS:
        for t in TAUS:
            sels[f"{f}@{t:g}"] = pmax[f] >= t
    sels["F0_actual"] = tc["reported_f0"] == 1.0
    groups: dict = {}
    for name, sel in sels.items():
        groups.setdefault(hashlib.sha1(sel.tobytes()).hexdigest(), []).append(name)
    for names in groups.values():
        rows = rows_for(sels[names[0]])
        for mode in MODES:
            rr, _ = score_all(it, rows, mode, "decision", "F0_fuse", names[0])
            for nm in names:
                recs += [dict(r, variant=nm) for r in rr]
    # check 4: reconstruction == the F0 submission file
    f0 = read_rows("F0_fuse", it)
    if f0 is not None:
        for mode in MODES:
            ref = SC.score_rows(it, f0, mode, configs=[parse(PRIMARY)], primary=PRIMARY)
            mine = next(r for r in recs if r["variant"] == "F0_actual" and r["truth_mode"] == mode
                        and r["config"] == PRIMARY)
            ok = (ref["n_est_rows"] == mine["n_est_rows"]
                  and abs(ref["parts"]["combined"] - mine["combined"]) < 1e-9
                  and abs(ref["parts"]["track"] - mine["track"]) < 1e-9
                  and abs(ref["parts"]["metadata"] - mine["metadata"]) < 1e-9)
            checks.append((f"4 {it.key} {mode}: reconstructed F0 reported set == F0_fuse file "
                           f"(rows {mine['n_est_rows']}, combined {mine['combined']:.6f})", bool(ok), ""))
    return {"records": recs, "checks": checks}


# --------------------------------------------------------------------------- driver


TASKS = {"methods": task_methods, "oracle": task_oracle, "decision": task_decision}


def run_task(arg: tuple[str, str]) -> dict:
    key, section = arg
    CMP.memoise_frame()
    it = next(i for i in SC.testing_iterations() if i.key == key)
    t0 = time.time()
    out = TASKS[section](it)
    out["secs"] = time.time() - t0
    out["task"] = arg
    print(f"  done {section:<9} {key:<12} {len(out['records']):>5} records {out['secs']:6.0f} s", flush=True)
    return out


def write_csv(recs: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    order = {c[0]: i for i, c in enumerate(CONFIGS)}
    recs = sorted(recs, key=lambda r: (r["kind"], r["method"], r["variant"], r["iter_key"],
                                       r["truth_mode"], order[r["config"]]))
    cols = {f: [r[f] for r in recs] for f in FIELDS}
    for f in FIELDS:
        if f in NUMERIC:
            cols[f] = np.asarray(cols[f], dtype=float)
        else:
            cols[f] = np.asarray(cols[f], dtype=object)
    C.write_columns(OUT / "sensitivity.csv", cols, FIELDS)


class Table:
    """Query helper over the records."""

    def __init__(self, recs: list[dict]):
        self.idx: dict = {}
        for r in recs:
            self.idx[(r["kind"], r["method"], r["variant"], r["truth_mode"], r["config"], r["iter_key"])] = r

    def per_it(self, kind, method, variant, mode, cfg, field) -> dict:
        return {k[5]: r[field] for k, r in self.idx.items()
                if k[:5] == (kind, method, variant, mode, cfg)}

    def mean(self, kind, method, variant, mode, cfg, field) -> tuple[float, int]:
        v = [x for x in self.per_it(kind, method, variant, mode, cfg, field).values() if x == x]
        return (float(np.mean(v)) if v else float("nan"), len(v))


def fmt(x: float, nd: int = 3) -> str:
    return "n/a" if x != x else f"{x:.{nd}f}"


def md_table(head: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out + [""]


def cname(c: int, p: int = 2, d: int = 3) -> str:
    return CMP.cfg_name(c, p, d)


def decision_analysis(tab: Table, its: list[str]) -> dict:
    """In-sample best decision per config, objective, regret, per-iteration stability."""
    res = {}
    for cfg in CONFIGS:
        name = cfg[0]
        obj = {}
        per_it = {}
        for d in DECISIONS + ("F0_actual",):
            a = tab.per_it("decision", "F0_fuse", d, "all", name, "combined")
            b = tab.per_it("decision", "F0_fuse", d, "airborne", name, "combined")
            ks = [k for k in its if k in a and k in b]
            per_it[d] = {k: (a[k] + b[k]) / 2.0 for k in ks}
            obj[d] = float(np.mean([per_it[d][k] for k in ks])) if ks else float("nan")
        best = max(DECISIONS, key=lambda d: (obj[d], -DECISIONS.index(d)) if obj[d] == obj[d] else (-1, 0))
        res[name] = {"obj": obj, "per_it": per_it, "best": best}
    return res


def key_findings(tab: Table, its: list[str], dec: dict, b50: str, maxgain: tuple, relay: dict) -> list[str]:
    """Plain-language headline, every number computed from the records."""
    out = ["## Key findings\n"]
    nr = sum(r["n_relay_drone_tracks"] for r in relay.values())
    nd = sum(r["n_drone_tracks"] for r in relay.values())
    nrep = sum(r["n_relay_f0_reported"] for r in relay.values())
    with_relay = [k.split("/")[-1] for k in its if relay.get(k, {}).get("n_relay_o3_tracks", 0) > 0]
    out.append(f"- **Caveat: testing numbers are optimistic.** The F0, O3 and F1 rows rescored here contain fused "
               f"tracks that include VKYY / QTVS (or LJJX) members, which relay the Dronetag ground truth in testing "
               f"(the same caveat `fuse.py` carries). {nr} of {nd} drone tracks across the 6 iterations contain such a "
               f"member ({nrep} of the F0-reported tracks); O3 has relay-bearing tracks in: "
               f"{', '.join(with_relay) if with_relay else 'none'}. Per-iteration counts are in section 3. "
               f"The collapse-versus-c curves and the bias-oracle values below are therefore not clean validation "
               f"numbers on those iterations; the bias oracle also averages over relay-bearing tracks (section 3 splits the oracle and the "
               f"median bias into relay-bearing and other tracks, so the reader can see how much of the gain they carry).")
    m = lambda mode, c, p=2, d=3, meth="F0_fuse", f="combined": tab.mean("method", meth, "", mode, cname(c, p, d), f)[0]
    for mode in MODES:
        ref = m(mode, 50)
        half = [c for c in CS if m(mode, c) < 0.5 * ref]
        out.append(f"- **Scores collapse at small c** (`{mode}`, F0, p = 2, 3-D): combined {m(mode, 100):.3f} at c = 100, "
                   f"{ref:.3f} at c = 50, {m(mode, 20):.3f} at c = 20, {m(mode, 10):.3f} at c = 10, {m(mode, 5):.3f} at c = 5; "
                   + (f"it drops below half of its c = 50 value for c <= {max(half)} m." if half else
                      "it never drops below half of its c = 50 value."))
    out.append(f"- **Height costs about {m('all', 50, 2, 2) - m('all', 50):.3f} (`all`) / "
               f"{m('airborne', 50, 2, 2) - m('airborne', 50):.3f} (`airborne`) of F0's combined score at c = 50**: "
               f"the 2-D score ({m('all', 50, 2, 2):.3f} / {m('airborne', 50, 2, 2):.3f}) is higher than the 3-D one "
               f"({m('all', 50):.3f} / {m('airborne', 50):.3f}). p = 1 vs p = 2 (3-D, c = 50): "
               f"{m('all', 50, 1, 3):.3f} vs {m('all', 50):.3f} (`all`), {m('airborne', 50, 1, 3):.3f} vs "
               f"{m('airborne', 50):.3f} (`airborne`).")
    sp = []
    for mode in MODES:
        v = [x for x in tab.per_it("method", "F0_fuse", "", mode, cname(50), "combined").values() if x == x]
        if v:
            sp.append(f"{min(v):.2f} to {max(v):.2f} (`{mode}`)")
    spread_txt = ", ".join(sp)
    flips = [c for c in CS if dec[cname(c)]["best"] != b50]
    where = (f"differs from {b50} only at c = {', '.join(str(c) for c in flips)} (p = 2, 3-D; there: "
             + ", ".join(sorted({dec[cname(c)]['best'] for c in flips})) + ")") if flips else \
        f"is {b50} at every c (p = 2, 3-D)"
    out.append(f"- **Decision sensitivity (in-sample)**: the best report-threshold decision {where}. The largest "
               f"in-sample gain of switching to a configuration's own best is {maxgain[0]:.4f} combined points "
               f"({maxgain[1]}), against a spread of F0's per-iteration combined score at c = 50 of {spread_txt}; "
               f"{len({dec[c[0]]['best'] for c in CONFIGS})} distinct decisions are ever best across the 40 configurations.")
    for mode in MODES:
        g = {c: tab.mean("oracle", "O3_fuse_oracle", "bias", mode, cname(c), "combined")[0] - m(mode, c, meth="O3_fuse_oracle")
             for c in CS}
        h = {c: tab.mean("oracle", "O3_fuse_oracle", "horizontal", mode, cname(c), "combined")[0] - m(mode, c, meth="O3_fuse_oracle")
             for c in CS}
        z = {c: tab.mean("oracle", "O3_fuse_oracle", "height", mode, cname(c), "combined")[0] - m(mode, c, meth="O3_fuse_oracle")
             for c in CS}
        nrl = {c: tab.mean("oracle", "O3_fuse_oracle", "bias_norelay", mode, cname(c), "combined")[0] - m(mode, c, meth="O3_fuse_oracle")
               for c in CS}
        cb = max(g, key=g.get)
        o3 = m(mode, cb, meth="O3_fuse_oracle")
        out.append(f"- **Bias-removal oracle** (`{mode}`, p = 2, 3-D): +{g[10]:.3f} combined at c = 10, +{g[20]:.3f} at c = 20, "
                   f"+{g[50]:.3f} at c = 50, +{g[100]:.3f} at c = 100; peak +{g[cb]:.3f} at c = {cb} "
                   f"({g[cb] / o3:.0%} on top of O3's {o3:.3f}). Perfect height: +{z[cb]:.3f}; perfect horizontal position: "
                   f"+{h[cb]:.3f} (same c). Bias removed on the non-relay tracks only: "
                   f"+{nrl[10]:.3f} at c = 10, +{nrl[20]:.3f} at c = 20, +{nrl[50]:.3f} at c = 50. A real bias estimator recovers only part of the bias row.")
    out.append("")
    return out


def build_md(recs: list[dict], checks: list[tuple], secs: dict, methods: list[str], relay: dict) -> str:
    tab = Table(recs)
    its = sorted({r["iter_key"] for r in recs if r["kind"] == "method"})
    ids = [k.split("/")[-1] for k in its]
    L: list[str] = []
    A = L.append
    A("# GOSPA parameter sensitivity (testing iterations, proxy scorer)\n")
    A("Generated by `python3 tools/gospa_sensitivity.py`. Every number is a LOCAL proxy computed on the 6 "
      "testing iterations with the submissions already written; nothing is fitted on evaluation. "
      "Numbers come from `sensitivity.csv` (same folder).\n")
    KF = len(L)   # the key findings are inserted here once the tables are known
    A("## What the parameters mean\n")
    A("- **GOSPA** is the track-distance metric. At every 100 ms step the truth drones are matched to the "
      "reported rows; each matched pair costs its distance (raised to the power p), and every truth drone "
      "with no partner (a *miss*) and every reported row with no partner (a *false track*) costs c^p / 2.")
    A("- **c** (metres) is the cut-off: a pair farther apart than c is not a match at all (it becomes one "
      "miss plus one false track). Small c means a track that is 20 m off counts as nothing; large c "
      "forgives it. The organisers do not publish c.")
    A("- **p** is the exponent on the matched distances: p = 1 adds errors up linearly, p = 2 makes large "
      "errors cost much more than small ones.")
    A("- **dims**: 3 = horizontal and height error together, 2 = horizontal only.")
    A("- **track score** = 1 - (total cost) / (cost if everything were missed or false), so 1 is perfect and "
      "0 is nothing matched. **combined** = 0.4 track + 0.2 classification + 0.1 identification + 0.3 "
      "metadata; the three label parts are recalls over all eligible truth rows, using this configuration's "
      "matching, so a smaller c also lowers them (fewer matched rows to carry a label).")
    A("- **truth modes**: `all` includes the drones parked on the ground (about 55 % of truth rows), "
      "`airborne` only flying ones. **Oracle** = uses ground truth, testing only, never submitted.")
    A("- **Sign test**: wins/losses over iterations; with 6 iterations even 6-0 only gives p = 0.031.\n")

    # ---- methods
    A("## 1. Where scores collapse\n")
    A("Mean over the iterations available of the combined score (p = 2, 3-D). `n` = number of testing "
      "iterations with a file. The ratio table further down gives F0 relative to its own value at c = 50.\n")
    cshow = (5, 10, 20, 30, 50, 75, 100)
    for mode in MODES:
        rows = []
        for m in methods:
            vals = [tab.mean("method", m, "", mode, cname(c), "combined") for c in cshow]
            n = max(v[1] for v in vals)
            if n == 0:
                continue
            rows.append([m, str(n)] + [fmt(v[0]) for v in vals])
        A(f"Truth `{mode}`, combined\n")
        L.extend(md_table(["method", "n"] + [f"c={c}" for c in cshow], rows))
    f0 = {mode: [tab.mean("method", "F0_fuse", "", mode, cname(c), "combined")[0] for c in CS] for mode in MODES}
    A("F0_fuse, score relative to its c = 50 value (p = 2, 3-D):\n")
    rows = []
    for mode in MODES:
        ref = f0[mode][CS.index(50)]
        rows.append([mode] + [fmt(v / ref, 2) for v in f0[mode]])
    L.extend(md_table(["truth"] + [f"c={c}" for c in CS], rows))
    for mode in MODES:
        ref = f0[mode][CS.index(50)]
        below = [c for c, v in zip(CS, f0[mode]) if v < 0.5 * ref]
        A(f"- `{mode}`: F0's combined score " + (
            f"falls below half of its c = 50 value for c <= {max(below)} m." if below else
            f"never falls below half of its c = 50 value down to c = 5 m (at c = 5 it keeps "
            f"{f0[mode][0] / ref:.0%}; at c = 100 it is {f0[mode][-1] / ref:.0%} of it)."))
    # track-only + miss/false
    A("\nF0_fuse mean TRACK part, and missed / false rates, by c (p = 2, 3-D). Missed rate = share of truth "
      "rows with no partner; false rate = share of reported rows with no partner.\n")
    rows = []
    for mode in MODES:
        rows.append([f"{mode} track"] + [fmt(tab.mean("method", "F0_fuse", "", mode, cname(c), "track")[0]) for c in CS])
        rows.append([f"{mode} missed"] + [fmt(tab.mean("method", "F0_fuse", "", mode, cname(c), "missed_rate")[0]) for c in CS])
        rows.append([f"{mode} false"] + [fmt(tab.mean("method", "F0_fuse", "", mode, cname(c), "false_rate")[0]) for c in CS])
    L.extend(md_table([""] + [f"c={c}" for c in CS], rows))
    A("Other (p, dims) combinations, F0_fuse mean combined:\n")
    rows = []
    for mode in MODES:
        for p in PS:
            for d in DIMS:
                rows.append([f"{mode} p={p} {d}-D"] + [fmt(tab.mean("method", "F0_fuse", "", mode, cname(c, p, d), "combined")[0]) for c in CS])
    L.extend(md_table([""] + [f"c={c}" for c in CS], rows))
    for mode in MODES:
        A(f"F0_fuse PER ITERATION, combined, truth `{mode}` (p = 2, 3-D):\n")
        rows = [[i] + [fmt(tab.per_it("method", "F0_fuse", "", mode, cname(c), "combined").get(k, float("nan"))) for c in CS]
                for i, k in zip(ids, its)]
        L.extend(md_table(["iteration"] + [f"c={c}" for c in CS], rows))
    for mode in MODES:
        A(f"F0_fuse PER ITERATION, track part, truth `{mode}` (p = 2, 3-D):\n")
        rows = [[i] + [fmt(tab.per_it("method", "F0_fuse", "", mode, cname(c), "track").get(k, float("nan"))) for c in CS]
                for i, k in zip(ids, its)]
        L.extend(md_table(["iteration"] + [f"c={c}" for c in CS], rows))

    A("### Does the ranking of methods move with c?\n")
    A("Iterations (of n) on which the first method has the higher combined score (ties |diff| < 1e-4 "
      "excluded from wins and losses), p = 2, 3-D.\n")
    for other in ("F1_fuse", "O3_fuse_oracle"):
        if not any(r["method"] == other and r["kind"] == "method" for r in recs):
            continue
        rows = []
        for c in CS:
            cells = []
            for mode in MODES:
                a = tab.per_it("method", "F0_fuse", "", mode, cname(c), "combined")
                b = tab.per_it("method", other, "", mode, cname(c), "combined")
                ks = [k for k in its if k in a and k in b]
                w, l, t = CMP.count_wlt([a[k] - b[k] for k in ks])
                cells.append(f"F0 {w}-{l}-{t} of {len(ks)}")
            rows.append([f"c={c}"] + cells)
        A(f"F0_fuse vs {other} (wins-losses-ties):\n")
        L.extend(md_table(["c", "all", "airborne"], rows))

    # ---- decisions
    A("## 2. Does F0's best report-threshold decision change with c and p?\n")
    A("F0 reports a fused track when its best member's drone probability (`P_FIELD` = `all` or `air`, "
      "two probability versions) is at least `TAU`. 12 decisions are rescored for every configuration. "
      "**In-sample**: the best one is chosen on the same 6 iterations (mean of the `all` and `airborne` "
      "combined scores, as in fuse.py), so it is optimistic; this only shows whether the optimum MOVES. "
      "`F0 actual` = the tracks F0 really reported (each iteration with its own cross-validated choice).\n")
    dec = decision_analysis(tab, its)
    b50 = dec[PRIMARY]["best"]
    A(f"Best decision at the primary configuration (c50 p2 3-D): **{b50}**.\n")
    rows = []
    for p in PS:
        for d in DIMS:
            for c in CS:
                nm = cname(c, p, d)
                r = dec[nm]
                obj_best = r["obj"][r["best"]]
                obj_b50 = r["obj"][b50]
                wins = CMP.count_wlt([r["per_it"][r["best"]][k] - r["per_it"][b50][k] for k in its if k in r["per_it"][b50]])
                rows.append([f"p={p} {d}-D", f"{c}", r["best"], fmt(obj_best, 4), fmt(r["obj"]["F0_actual"], 4),
                             fmt(obj_b50, 4), fmt(obj_best - obj_b50, 4), f"{wins[0]}-{wins[1]}-{wins[2]}"])
    L.extend(md_table(["p, dims", "c", "best decision", "objective best", "F0 actual", f"objective of {b50}",
                       f"gain of best over {b50}", "per-iteration wins-losses-ties of best vs " + b50], rows))
    distinct = sorted({dec[c[0]]["best"] for c in CONFIGS})
    flips = [c for c in CS if dec[cname(c)]["best"] != b50]
    A(f"Distinct best decisions across the 40 configurations: {', '.join(distinct)}. "
      f"At p = 2, 3-D the best decision differs from {b50} at c = " +
      (", ".join(str(c) for c in flips) if flips else "(none)") + ".")
    maxgain = max(((dec[c[0]]["obj"][dec[c[0]]["best"]] - dec[c[0]]["obj"][b50]), c[0]) for c in CONFIGS)
    A(f"Largest in-sample gain (objective points) of switching from {b50} to the configuration's own "
      f"best decision: {maxgain[0]:.4f} ({maxgain[1]}).\n")
    for c in (10, 25, 50, 100):
        nm = cname(c)
        r = dec[nm]
        A(f"Per iteration at {nm} (mean of all and airborne combined): best-here `{r['best']}`, "
          f"`{b50}`, F0 actual:\n")
        rows = [[i, fmt(r["per_it"][r["best"]].get(k, float('nan')), 4), fmt(r["per_it"][b50].get(k, float('nan')), 4),
                 fmt(r["per_it"]["F0_actual"].get(k, float('nan')), 4)] for i, k in zip(ids, its)]
        L.extend(md_table(["iteration", r["best"], b50, "F0 actual"], rows))

    # ---- oracle
    A("## 3. What would removing the slow per-track biases be worth?\n")
    A("Oracle rows built from O3 (the fused tracks reported iff the track belongs to a real drone), testing "
      "only, using ground truth: **bias** = subtract each track's median 3-D error vector from all of its "
      "rows (what a perfect bias correction of that track would do), **height** = replace the height by the "
      "truth height, **horizontal** = replace east/north by the truth. Gain = oracle - O3 score, same "
      "configuration. A radar bias correction can at most approach the `bias` row; in practice it will "
      "recover only a part of it.\n")
    for metric in ("combined", "track"):
        for mode in MODES:
            A(f"Mean gain in {metric} (oracle - O3), truth `{mode}`, p = 2, 3-D; in brackets: iterations on "
              f"which the gain is positive (of n):\n")
            rows = []
            for v in ORACLES:
                cells = []
                for c in CS:
                    o = tab.per_it("oracle", "O3_fuse_oracle", v, mode, cname(c), metric)
                    b = tab.per_it("method", "O3_fuse_oracle", "", mode, cname(c), metric)
                    ks = [k for k in o if k in b]
                    g = [o[k] - b[k] for k in ks]
                    cells.append(f"{np.mean(g):+.3f} ({sum(1 for x in g if x > 1e-4)}/{len(g)})" if g else "n/a")
                rows.append([v] + cells)
            L.extend(md_table(["oracle"] + [f"c={c}" for c in CS], rows))
    A("**Relay caveat.** In testing, fused tracks that include a VKYY / QTVS / LJJX member carry the Dronetag "
      "feed itself, so their error is expected to be small, but they can also be merged with far or "
      "unrelated reports, so the table below gives the measured median bias of each group. "
      "`bias_norelay` applies the bias removal only to the other tracks (relay tracks untouched): it is the "
      "part of the `bias` gain that comes from radar / passive-sensor-only tracks. Counts per iteration "
      "(median size of each track's median 3-D error vector, truth `all`):\n")
    rows = []
    for k, i in zip(its, ids):
        r = relay.get(k)
        if r:
            rows.append([i, f"{r['n_relay_drone_tracks']} / {r['n_drone_tracks']}",
                         f"{r['n_relay_f0_reported']} / {r['n_f0_reported']}",
                         f"{r['n_relay_o3_tracks']} / {r['n_o3_tracks']}",
                         fmt(r["med_err_relay_m"], 1), fmt(r["med_err_other_m"], 1)])
    L.extend(md_table(["iteration", "drone tracks with relay", "F0-reported with relay", "O3 tracks with relay",
                       "median bias, relay tracks (m)", "median bias, other tracks (m)"], rows))
    A("O3 itself (the reference the gains are measured against), mean combined and track, p = 2, 3-D:\n")
    rows = []
    for mode in MODES:
        for metric in ("combined", "track"):
            rows.append([f"{mode} {metric}"] + [fmt(tab.mean("method", "O3_fuse_oracle", "", mode, cname(c), metric)[0]) for c in CS])
    L.extend(md_table([""] + [f"c={c}" for c in CS], rows))
    A("Mean bias-oracle gain in combined for the other (p, dims) combinations:\n")
    rows = []
    for mode in MODES:
        for p in PS:
            for d in DIMS:
                cells = []
                for c in CS:
                    nm = cname(c, p, d)
                    o = tab.per_it("oracle", "O3_fuse_oracle", "bias", mode, nm, "combined")
                    b = tab.per_it("method", "O3_fuse_oracle", "", mode, nm, "combined")
                    g = [o[k] - b[k] for k in o if k in b]
                    cells.append(f"{np.mean(g):+.3f}" if g else "n/a")
                rows.append([f"{mode} p={p} {d}-D"] + cells)
    L.extend(md_table([""] + [f"c={c}" for c in CS], rows))
    # summary sentences
    for mode in MODES:
        gs = {}
        for c in CS:
            o = tab.per_it("oracle", "O3_fuse_oracle", "bias", mode, cname(c), "combined")
            b = tab.per_it("method", "O3_fuse_oracle", "", mode, cname(c), "combined")
            gs[c] = float(np.mean([o[k] - b[k] for k in o if k in b])) if o else float("nan")
        cbest = max(gs, key=lambda c: gs[c] if gs[c] == gs[c] else -1)
        A(f"- `{mode}`: the bias oracle's mean combined gain is {gs[5]:+.3f} at c = 5, {gs[10]:+.3f} at c = 10, "
          f"{gs[25]:+.3f} at c = 25, {gs[50]:+.3f} at c = 50, {gs[100]:+.3f} at c = 100; largest at c = {cbest}.")
    A("")

    L[KF:KF] = key_findings(tab, its, dec, b50, maxgain, relay)

    # ---- checks
    A("## Self-checks\n")
    bad = [c for c in checks if not c[1]]
    A(f"{len(checks) - len(bad)} of {len(checks)} passed.\n")
    groups: dict = {}
    for name, ok, detail in checks:
        groups.setdefault(name.split(" ")[0], []).append(ok)
    for k in sorted(groups):
        A(f"- check {k}: {sum(groups[k])} of {len(groups[k])} passed")
    for name, ok, detail in bad:
        A(f"- FAILED: {name} {detail}")
    A("")
    A("## Notes and limits\n")
    A("- The O3 geometry is the tracker's, scored with ground-truth track selection: it is a reference, not a "
      "bound. The bias oracle uses the SAME ground truth that scores it, so it is the best case of a "
      "perfect per-track offset estimate; the value of a real estimator (reference tracks from the cyber / UTM "
      "sensors) is a fraction of it.")
    A("- F0, O3 and F1 rescored here contain VKYY / QTVS / LJJX members that relay the Dronetag ground truth "
      "in testing (section 3 counts them per iteration), so all F0 / O3 / F1 numbers above are optimistic on "
      "the iterations that have such tracks; the evaluation set has no such leak.")
    A("- Decision sensitivity is in-sample and uses F0's applied fold geometry with the outer p, not a fresh "
      "cross-validation per configuration.")
    A("- B1 single-sensor rows include every deployable sensor except QTVS / VKYY (they relay the Dronetag "
      "feed in testing); sensors with files for only some iterations show a smaller n.")
    A("- A frame memo (`compare.memoise_frame`) makes `C.iteration_frame` remember its per-iteration result "
      "within this process; it returns the same frame, so no score changes.")
    A("- Task times (s): " + ", ".join(f"{k[1]} {k[0].split('/')[-1]} {v:.0f}" for k, v in sorted(secs.items())))
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    its = SC.testing_iterations()
    tasks = [(it.key, s) for s in ("decision", "methods", "oracle") for it in its]
    methods = method_list()
    print(f"gospa_sensitivity: {len(CONFIGS)} configurations x 2 truth modes; methods {methods}")
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
        outs = pool.map(run_task, tasks, chunksize=1)
    recs, checks, secs, relay = [], [], {}, {}
    for o in sorted(outs, key=lambda o: o["task"]):
        recs += o["records"]
        checks += o["checks"]
        secs[o["task"]] = o["secs"]
        if "relay" in o:
            relay[o["task"][0]] = o["relay"]
    write_csv(recs)
    md = build_md(recs, checks, secs, methods, relay)
    (OUT / "sensitivity.md").write_text(md, encoding="utf-8")
    print(f"\nself-checks ({len(checks)})")
    bad = 0
    shown = set()
    for name, ok, detail in checks:
        if not ok:
            bad += 1
            print(f"  [FAIL] {name} {detail}")
        elif name.split(" ")[0] not in shown or "F0 testing/S1-I1 all" in name:
            shown.add(name.split(" ")[0])
            print(f"  [PASS] {name} {detail}")
    print(f"  {len(checks) - bad} of {len(checks)} self-checks passed")
    print(f"wrote {OUT.relative_to(C.REPO)}/sensitivity.csv ({len(recs)} rows) and sensitivity.md; "
          f"{time.time() - t0:.0f} s")
    if bad:
        print("SELF-CHECK FAILED")
        return 1
    print("ALL SELF-CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
