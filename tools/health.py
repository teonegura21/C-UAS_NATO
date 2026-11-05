#!/usr/bin/env python3
"""Health indicators of a method's submissions: VERIFICATION GATES, never optimisation targets.

WHY: the evaluation iterations have no ground truth, so a fix of an evaluation-only defect
(duplicate tracks, Friendly copies, impossible jumps) cannot be scored there. These indicators
count the defects directly in the submission files. They are PRE-REGISTERED here (definitions,
thresholds and the gate rule below are fixed before any use), calibrated on testing (where the
truth says whether a flagged row is really wrong) and guarded against the cheap way out:
deleting tracks makes every defect count smaller, so a coverage guard compares how much of the
air picture is still reported.

Terms. Grid = the template's 100 ms steps. Track = rows sharing a track_id. Hop = the 1 s step
pair (t, t+10 grid steps) of one track.

INDICATORS (per iteration, from the submission file via submission.read_submission; ENU =
C.iteration_frame(it).to_enu; ``tracks.csv`` member info from <tree>/fused/<slug>__tracks.csv
is only used to annotate the worst pairs):
  near_dup_step_share  Two reported tracks are a "near-duplicate pair" while they are both
                       present at consecutive grid steps and within 20 m horizontally (<= 20)
                       at every one of those steps. A run is flagged when it lasts MORE than
                       10 s (> 100 consecutive steps; exactly 100 steps = 10.0 s is not
                       flagged; a missing step breaks a run). Share = grid steps covered by
                       >= 1 flagged run / ALL grid steps of the template. Also: pairs (distinct
                       track pairs with a flagged run), pair_seconds (sum of flagged run
                       lengths), worst pairs.
  jump_row_share       A row is a jump row when the same track has a row 1 s later (step + 10)
                       and the horizontal displacement implies > 40 m/s. Share = jump rows /
                       ALL rows. Also: tracks with any jump row.
  friendly_per_serial  For every serial of the Friendly-copy whitelist (baselines.load_whitelist)
                       the rows with identity 'Friendly' carrying it; per grid step the number
                       of simultaneous tracks: max_simul, seconds with >= 2 (a second Friendly
                       track with the same serial is a copy). copy_seconds = sum over serials
                       of seconds_ge2; max_simul = max over serials.
  stationary_friendly  Friendly tracks whose hops are < 0.5 m/s for > 60 s (a run of
                       consecutive hop-start steps; duration = (hops + 10 steps) * 0.1 s).
                       INFORMATIONAL only: parked Blue drones are real (about half the truth
                       rows), so this is not in the gate.
  height_out_of_range  rows whose ENU up u is outside [-50, 1500] m (boundaries are inside).
  tracks_per_step      median / p90 / max number of reported tracks per grid step, over the
                       steps with >= 1 track (testing: next to the truth drones per step, both
                       truth modes).
  coverage             covered_step_share = grid steps with >= 1 track / all grid steps;
                       rows = reported rows.

GATE RULE (fixed here before any use): candidate B passes versus reference A on evaluation iff,
on ALL 3 evaluation iterations, every indicator is equal or better
  near_dup_step_share <=, jump rows (COUNT) <=, friendly copy_seconds <= and max_simul <=,
  height_out_of_range rows <=, tracks_per_step max <=   (float equality tolerance 1e-12)
The jump gate compares the COUNT of jump rows, not jump_row_share: the share has the reported rows
as denominator, so deleting clean rows would raise it and fail a harmless candidate (a measured
artefact of the first draft); near_dup already has a fixed denominator (all grid steps). The share
is still reported.
AND the COVERAGE GUARD holds on every evaluation iteration: B.covered_step_share >=
0.95 * A.covered_step_share and B.rows >= 0.95 * A.rows (B may not lose more than 5 % relative of
either). A missing evaluation file for A or B fails the gate. Testing deltas are printed as
context, never part of the verdict. Passing the gate does not mean B is better; it means B did
not hide defects by deleting tracks and did not add any.

CALIBRATION (testing, truth mode 'all', match radius 50 m): do the flags measure real defects?
  near-dup pair steps are classed same_drone (both tracks within 50 m of the SAME truth drone:
  a true duplicate), ghost (at least one track > 50 m from every truth drone), different_drones
  (both near different drones: a legitimate close pair). Only same_drone and different_drones steps
  are EVIDENCE (a ghost step is undecidable: two unrelated false tracks are ghosts too); the measure
  is the confirmed-duplicate share = same_drone / (same_drone + different_drones), with the ghost
  steps shown next to the count expected by chance. Jump rows are classed end_off (an
  endpoint > 50 m from every truth drone), swap (endpoints near different drones), same_drone
  (both near the same drone), truth_fast (the same drone really moved > 40 m in that second:
  not wrong); contrast: the share of NON-jump rows whose position is > 50 m from every truth
  drone. Caveat: a truth outage (Dronetag gap > 3 s) makes a real track look like a ghost.
  Written to <tree>/health/<METHOD>_calibration.md and into the JSON (key "calibration").

USAGE
  python3 tools/health.py METHOD [--iters KEY ...] [--tree DIR] [--no-calibration]
      -> table + <tree>/health/<METHOD>.json (default tree: data/derived or CUAS_DERIVED).
         --iters takes iteration keys (testing/S1-I1), 'testing' or 'evaluation'.
  python3 tools/health.py --compare A B [--tree DIR] [--tree-a DIR] [--tree-b DIR]
      -> per-iteration deltas (B - A) and the gate verdict (exit 1 if the gate fails).
  python3 tools/health.py --selftest
Every run executes the numbered self-checks (synthetic submissions with injected defects built
with track_rows.make_rows on the testing/S1-I1 frame, mutation tests, file round trip, a
plain-python recount of near_dup and jump on a real iteration, gate logic); exit 1 on failure.

JSON schema (<tree>/health/<METHOD>.json): {method, tree, code_stamp (hash of health.py + whitelist; a
cached JSON with another stamp is recomputed, so --compare never gates on stale definitions), file_stamps{key:[size,mtime_ns]},
iterations{key:{n_steps, n_rows, n_tracks, coverage{...}, near_dup{...}, jump{...},
friendly{...}, height{...}, tracks_per_step{...}, truth_drones_per_step{...}?}},
calibration{key:{...}}?}. Coordinates are never written (local metres only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuas_common as C  # noqa: E402
import submission as S  # noqa: E402

# --------------------------------------------------------------------------- pre-registered constants
NEAR_RADIUS_M = 20.0
NEAR_MIN_S = 10.0           # run must last MORE than this
HOP_S = 1.0
JUMP_SPEED_MS = 40.0
STATIONARY_SPEED_MS = 0.5
STATIONARY_MIN_S = 60.0
U_RANGE_M = (-50.0, 1500.0)
GUARD_REL = 0.95
EPS = 1e-12
MATCH_RADIUS_M = 50.0
WORST_N = 5

GATE_KEYS = ("near_dup_step_share", "jump_rows", "friendly_copy_seconds",
             "friendly_max_simul", "height_out_of_range_rows", "tracks_per_step_max")


def _health_dir(tree: Path) -> Path:
    return Path(tree) / "health"


def _py(x):
    """numpy -> plain python (JSON)."""
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return [_py(v) for v in x.tolist()]
    return x


def _load_whitelist() -> set[str]:
    last = None
    for _ in range(4):                   # sibling units may be editing imported modules
        try:
            import baselines as B
            return set(B.load_whitelist())
        except (SyntaxError, NameError, ImportError, AttributeError) as ex:  # pragma: no cover
            last = ex
            time.sleep(20)
    raise RuntimeError(f"baselines could not be imported: {last}")


# --------------------------------------------------------------------------- core indicators


def _grid_params(grid_t: np.ndarray) -> tuple[float, int, int, int, int]:
    n_steps = len(grid_t)
    dt = round(float((grid_t[-1] - grid_t[0]) / (n_steps - 1)), 6)
    near_min = int(round(NEAR_MIN_S / dt))
    hop = int(round(HOP_S / dt))
    stat_min = int(round(STATIONARY_MIN_S / dt))
    return dt, near_min, hop, stat_min, n_steps


def indicators(step, tid, e, n, u, identity, serial, grid_t, whitelist, detail: bool = False) -> dict:
    """All indicators of one iteration from track-row arrays (e, n, u in the local ENU frame).
    With detail=True the result also carries ``_detail`` (row-level arrays for calibration)."""
    step = np.asarray(step, dtype=np.int64)
    e, n, u = (np.asarray(a, dtype=float) for a in (e, n, u))
    identity = np.asarray(identity, dtype=object)
    serial = np.asarray(serial, dtype=object)
    R = len(step)
    dt, near_min, hop, stat_min, S_ = _grid_params(grid_t)
    uniq, code = np.unique(np.asarray(tid, dtype=object).astype(str), return_inverse=True)
    code = code.astype(np.int64)
    T = len(uniq)
    out: dict = {"n_steps": S_, "n_rows": int(R), "n_tracks": int(T)}
    det: dict = {}

    # ---- coverage / tracks per step
    cnt = np.bincount(step, minlength=S_) if R else np.zeros(S_, dtype=np.int64)
    covered = cnt > 0
    out["coverage"] = {"covered_step_share": float(covered.sum() / S_), "covered_steps": int(covered.sum()),
                       "rows": int(R)}
    if covered.any():
        c = cnt[covered]
        out["tracks_per_step"] = {"median": float(np.median(c)), "p90": float(np.percentile(c, 90)),
                                  "max": int(c.max())}
    else:
        out["tracks_per_step"] = {"median": 0.0, "p90": 0.0, "max": 0}

    # ---- near duplicates
    nd = {"step_share": 0.0, "steps": 0, "pairs": 0, "pair_seconds": 0.0, "worst": []}
    if R:
        o = np.lexsort((code, step))
        s_, c_, e_, n_ = step[o], code[o], e[o], n[o]
        ia, ib = [], []
        for d in range(1, int(cnt.max())):
            m = s_[:-d] == s_[d:]
            if not m.any():
                break
            i = np.nonzero(m)[0]
            j = i + d
            ok = (np.hypot(e_[i] - e_[j], n_[i] - n_[j]) <= NEAR_RADIUS_M) & (c_[i] != c_[j])
            ia.append(i[ok])
            ib.append(j[ok])
        ia = np.concatenate(ia) if ia else np.zeros(0, dtype=np.int64)
        ib = np.concatenate(ib) if ib else np.zeros(0, dtype=np.int64)
        if ia.size:
            key = c_[ia] * T + c_[ib]
            ps = s_[ia]
            q = np.lexsort((ps, key))
            key, ps, ia, ib = key[q], ps[q], ia[q], ib[q]
            brk = np.ones(len(key), dtype=bool)
            brk[1:] = (key[1:] != key[:-1]) | (ps[1:] != ps[:-1] + 1)
            run = np.cumsum(brk) - 1
            rlen = np.bincount(run)
            rflag = rlen > near_min
            row_flag = rflag[run]
            if rflag.any():
                fsteps = np.zeros(S_, dtype=bool)
                fsteps[ps[row_flag]] = True
                nd["steps"] = int(fsteps.sum())
                nd["step_share"] = float(fsteps.sum() / S_)
                rstart = np.nonzero(brk)[0]
                rkey = key[rstart]
                fk = rkey[rflag]
                fl = rlen[rflag]
                ukeys, inv = np.unique(fk, return_inverse=True)
                tot = np.bincount(inv, weights=fl).astype(np.int64)
                nd["pairs"] = int(len(ukeys))
                nd["pair_seconds"] = float(fl.sum() * dt)
                dist = np.hypot(e_[ia] - e_[ib], n_[ia] - n_[ib])
                worst = []
                for k in np.argsort(-tot, kind="stable")[:WORST_N]:
                    sel = row_flag & (key == ukeys[k])
                    worst.append({"track_a": str(uniq[int(ukeys[k] // T)]), "track_b": str(uniq[int(ukeys[k] % T)]),
                                  "seconds": float(tot[k] * dt), "first_step": int(ps[sel].min()),
                                  "min_dist_m": float(dist[sel].min()), "mean_dist_m": float(dist[sel].mean())})
                nd["worst"] = worst
            det["pairs"] = {"step": ps, "row_a": o[ia], "row_b": o[ib], "flag": row_flag, "run": run,
                            "key": key}
    out["near_dup"] = nd

    # ---- hops: jump rows and stationary Friendly runs
    jr = {"row_share": 0.0, "rows": 0, "evaluable_rows": 0, "tracks": 0}
    stf = {"tracks": 0, "seconds": 0.0, "worst": []}
    if R:
        o = np.lexsort((step, code))
        cs, ss = code[o], step[o]
        M = S_ + hop + 1
        key = cs * M + ss
        p = np.searchsorted(key, key + hop)
        has = p < R
        has[has] = key[p[has]] == key[has] + hop
        pp = np.where(has, p, 0)
        disp = np.where(has, np.hypot(e[o][pp] - e[o], n[o][pp] - n[o]), np.nan)
        hop_dt = np.where(has, grid_t[np.minimum(ss + hop, S_ - 1)] - grid_t[ss], np.nan)
        speed = disp / hop_dt
        jflag = has & (speed > JUMP_SPEED_MS)
        jr["rows"] = int(jflag.sum())
        jr["evaluable_rows"] = int(has.sum())
        jr["row_share"] = float(jflag.sum() / R)
        jr["tracks"] = int(len(np.unique(cs[jflag]))) if jflag.any() else 0
        det["jump"] = {"has": has, "order": o, "partner_all": o[pp], "speed": speed}
        # stationary Friendly runs
        fr_row = (identity[o] == "Friendly")
        sf = has & fr_row & (speed < STATIONARY_SPEED_MS)
        if sf.any():
            prev = np.zeros(R, dtype=bool)
            prev[1:] = sf[:-1] & (cs[1:] == cs[:-1]) & (ss[1:] == ss[:-1] + 1)
            start = sf & ~prev
            rid = np.cumsum(start) - 1
            nh = np.bincount(rid[sf], minlength=int(start.sum()))
            dur = nh + hop
            good = dur > stat_min
            if good.any():
                rtrack = cs[np.nonzero(start)[0]]
                stf["tracks"] = int(len(np.unique(rtrack[good])))
                stf["seconds"] = float(dur[good].sum() * dt)
                order = np.argsort(-dur, kind="stable")
                stf["worst"] = [{"track": str(uniq[int(rtrack[k])]), "seconds": float(dur[k] * dt)}
                                for k in order[:WORST_N] if good[k]]
    out["jump"] = jr

    # ---- Friendly copies per whitelisted serial
    fm = (identity == "Friendly")
    per: dict = {}
    for s in sorted(whitelist):
        m = fm & (serial == s)
        if not m.any():
            continue
        pair = np.unique(step[m] * T + code[m])
        c2 = np.bincount(pair // T, minlength=S_)
        per[s] = {"max_simul": int(c2.max()), "seconds_ge2": float((c2 >= 2).sum() * dt),
                  "tracks": int(len(np.unique(code[m])))}
    out["friendly"] = {"rows": int(fm.sum()), "tracks": int(len(np.unique(code[fm]))) if fm.any() else 0,
                       "per_serial": per,
                       "copy_seconds": float(sum(v["seconds_ge2"] for v in per.values())),
                       "max_simul": int(max([v["max_simul"] for v in per.values()], default=0)),
                       "stationary": stf}

    # ---- heights
    bad = (u < U_RANGE_M[0]) | (u > U_RANGE_M[1])
    out["height"] = {"out_of_range_rows": int(bad.sum()),
                     "tracks": int(len(np.unique(code[bad]))) if bad.any() else 0}
    if detail:
        out["_detail"] = det
    return out


def gate_values(m: dict) -> dict:
    """The scalar indicators the gate compares (one per GATE_KEYS entry)."""
    return {"near_dup_step_share": m["near_dup"]["step_share"], "jump_rows": m["jump"]["rows"],
            "friendly_copy_seconds": m["friendly"]["copy_seconds"], "friendly_max_simul": m["friendly"]["max_simul"],
            "height_out_of_range_rows": m["height"]["out_of_range_rows"],
            "tracks_per_step_max": m["tracks_per_step"]["max"]}


def coverage_guard(a: dict, b: dict) -> dict:
    """B may not lose more than 5 % (relative) of reference A's covered-step share or rows."""
    ca, cb = a["coverage"]["covered_step_share"], b["coverage"]["covered_step_share"]
    ra, rb = a["n_rows"], b["n_rows"]
    ok_c = cb >= GUARD_REL * ca - EPS
    ok_r = rb >= GUARD_REL * ra - EPS
    return {"ok": bool(ok_c and ok_r), "covered_ratio": (cb / ca) if ca > 0 else 1.0,
            "rows_ratio": (rb / ra) if ra > 0 else 1.0, "covered_ok": bool(ok_c), "rows_ok": bool(ok_r)}


def gate(a_by: dict, b_by: dict, eval_keys: list[str]) -> dict:
    """Gate verdict of candidate B versus reference A over the evaluation iterations."""
    per: dict = {}
    ok_all = True
    missing = [k for k in eval_keys if k not in a_by or k not in b_by]
    for k in eval_keys:
        if k in missing:
            continue
        ga, gb = gate_values(a_by[k]), gate_values(b_by[k])
        ind = {name: {"a": ga[name], "b": gb[name], "ok": bool(gb[name] <= ga[name] + EPS)} for name in GATE_KEYS}
        guard = coverage_guard(a_by[k], b_by[k])
        per[k] = {"indicators": ind, "guard": guard,
                  "ok": bool(all(v["ok"] for v in ind.values()) and guard["ok"])}
        ok_all &= per[k]["ok"]
    return {"pass": bool(ok_all and not missing and len(eval_keys) > 0), "missing": missing, "per_iter": per}


# --------------------------------------------------------------------------- per-iteration driver


def _tree_paths(tree: Path, method: str, it: C.Iteration) -> Path:
    return Path(tree) / "submissions" / method / C.iter_slug(it.key) / C.template_path(it).name


def _fused_info(tree: Path, it: C.Iteration) -> dict[str, dict]:
    for base in (Path(tree), C.BASE_DERIVED):
        p = base / "fused" / f"{C.iter_slug(it.key)}__tracks.csv"
        if p.exists():
            d = C.read_columns(p)
            return {tid: {"sensor_ids": sid, "n_members": nm}
                    for tid, sid, nm in zip(d["track_id"], d["sensor_ids"], d["n_members"])}
    return {}


def _rows_enu(it: C.Iteration, rows: dict):
    return C.iteration_frame(it).to_enu(rows["latitude"], rows["longitude"], rows["altitude_m"])


def health_of_rows(it: C.Iteration, rows: dict, whitelist: set[str], detail: bool = False):
    grid_t = C.template_grid(it)[1]
    e, n, u = _rows_enu(it, rows)
    m = indicators(rows["step"], rows["track_id"], e, n, u, rows["identity"], rows["serial_number"],
                   grid_t, whitelist, detail=detail)
    return (m, (e, n, u, grid_t)) if detail else m


def _truth_per_step(it: C.Iteration) -> dict:
    import truth as T
    out = {}
    S_ = len(C.template_grid(it)[1])
    for mode in ("all", "airborne"):
        tr = T.load_truth(it, mode)
        c = np.bincount(tr.step, minlength=S_)
        c = c[c > 0]
        out[mode] = ({"median": float(np.median(c)), "p90": float(np.percentile(c, 90)), "max": int(c.max())}
                     if c.size else {"median": 0.0, "p90": 0.0, "max": 0})
    return out


# --------------------------------------------------------------------------- calibration


def _truth_dense(it: C.Iteration, mode: str = "all"):
    import truth as T
    tr = T.load_truth(it, mode)
    S_ = len(tr.grid_t)
    F = len(tr.flights)
    E = np.full((max(F, 1), S_), np.nan)
    N = np.full((max(F, 1), S_), np.nan)
    E[tr.flight, tr.step] = tr.e
    N[tr.flight, tr.step] = tr.n
    return E, N


def _nearest_truth(E, N, q_step, q_e, q_n, chunk: int = 20000):
    """(flight index or -1, distance or inf) of the nearest truth drone at each query."""
    K = len(q_step)
    fl = np.full(K, -1, dtype=np.int64)
    dist = np.full(K, np.inf)
    for a in range(0, K, chunk):
        sl = slice(a, a + chunk)
        d = np.hypot(E[:, q_step[sl]] - q_e[sl], N[:, q_step[sl]] - q_n[sl])
        d = np.where(np.isnan(d), np.inf, d)
        am = np.argmin(d, axis=0)
        mn = d[am, np.arange(d.shape[1])]
        fl[sl] = np.where(np.isfinite(mn), am, -1)
        dist[sl] = mn
    return fl, dist


def calibrate(it: C.Iteration, det: dict, e, n, step, grid_t, truth=None) -> dict:
    """``truth`` = optional (E, N) dense truth arrays [flight, step] (self-checks); default: the real truth."""
    E, N = truth if truth is not None else _truth_dense(it, "all")
    res: dict = {"match_radius_m": MATCH_RADIUS_M}
    # near-dup pair steps
    P = det.get("pairs")
    nd = {"flagged_pair_steps": 0, "same_drone": 0, "ghost": 0, "different_drones": 0,
          "pairs": 0, "pairs_confirmed": 0, "pairs_legit_close": 0, "pairs_undetermined": 0,
          "pair_seconds": 0.0, "pair_seconds_confirmed": 0.0, "expected_ghost_steps": 0.0}
    if P is not None and P["flag"].any():
        f = P["flag"]
        ra, rb, ps = P["row_a"][f], P["row_b"][f], P["step"][f]
        fa, da = _nearest_truth(E, N, ps, e[ra], n[ra])
        fb, db = _nearest_truth(E, N, ps, e[rb], n[rb])
        near_a, near_b = da <= MATCH_RADIUS_M, db <= MATCH_RADIUS_M
        cat = np.where(~(near_a & near_b), 1, np.where(fa == fb, 0, 2))      # 0 same, 1 ghost, 2 different
        names = ("same_drone", "ghost", "different_drones")
        for ci, nm in enumerate(names):
            nd[nm] = int((cat == ci).sum())
        nd["flagged_pair_steps"] = int(len(cat))
        keys = P["key"][f]
        dt = _grid_params(grid_t)[0]
        for k in np.unique(keys):
            c = np.bincount(cat[keys == k], minlength=3)
            # ghost steps are UNDECIDABLE (a far track proves nothing about duplication): the pair verdict
            # uses only the decisive steps (both tracks near a truth drone)
            nd["pairs"] += 1
            nd["pair_seconds"] += float(c.sum() * dt)
            if c[0] > c[2]:
                nd["pairs_confirmed"] += 1
                nd["pair_seconds_confirmed"] += float(c.sum() * dt)
            elif c[2] > c[0]:
                nd["pairs_legit_close"] += 1
            else:
                nd["pairs_undetermined"] += 1
    res["near_dup"] = nd
    # jump rows
    J = det.get("jump")
    jr = {"jump_rows": 0, "end_off": 0, "swap": 0, "same_drone": 0, "truth_fast": 0,
          "nonjump_rows_evaluated": 0, "nonjump_end_off": 0}
    if J is not None:
        has, o, pa, speed = J["has"], J["order"], J["partner_all"], J["speed"]
        q_row0 = o[has]
        q_row1 = pa[has]
        q_step0 = step[q_row0]
        q_step1 = step[q_row1]
        f0, d0 = _nearest_truth(E, N, q_step0, e[q_row0], n[q_row0])
        f1, d1 = _nearest_truth(E, N, q_step1, e[q_row1], n[q_row1])
        flagged = speed[has] > JUMP_SPEED_MS
        off = (d0 > MATCH_RADIUS_M) | (d1 > MATCH_RADIUS_M)
        swap = ~off & (f0 != f1)
        same = ~off & (f0 == f1)
        tf = np.zeros(len(flagged), dtype=bool)                      # truth itself moved > 40 m/s
        idx = np.nonzero(same & flagged)[0]
        if idx.size:
            ff = f0[idx]
            td = np.hypot(E[ff, q_step1[idx]] - E[ff, q_step0[idx]], N[ff, q_step1[idx]] - N[ff, q_step0[idx]])
            tdt = grid_t[q_step1[idx]] - grid_t[q_step0[idx]]
            tf[idx] = np.nan_to_num(td / tdt, nan=0.0) > JUMP_SPEED_MS
        jr["jump_rows"] = int(flagged.sum())
        jr["end_off"] = int((off & flagged).sum())
        jr["swap"] = int((swap & flagged).sum())
        jr["truth_fast"] = int(tf.sum())
        jr["same_drone"] = int((same & flagged).sum() - tf.sum())
        nj = ~flagged
        jr["nonjump_rows_evaluated"] = int(nj.sum())
        jr["nonjump_end_off"] = int((off & nj).sum())
        if jr["nonjump_rows_evaluated"]:
            # chance level of a ghost pair step: >= 1 of two unrelated tracks far from truth
            g = jr["nonjump_end_off"] / jr["nonjump_rows_evaluated"]
            nd["expected_ghost_steps"] = float(nd["flagged_pair_steps"] * (1.0 - (1.0 - g) ** 2))
    res["jump"] = jr
    return res


def _pct(a, b) -> str:
    return "n/a (none flagged)" if not b else f"{100.0 * a / b:.1f} %"


def _calibration_md(method: str, cal: dict) -> str:
    L = [f"# Health indicator calibration on testing: {method}", "",
         "Do the flagged rows / pairs really point at wrong output? Truth mode 'all' (parked drones included), "
         f"match radius {MATCH_RADIUS_M:.0f} m. Terms: a *near-duplicate pair* = two reported tracks within "
         f"{NEAR_RADIUS_M:.0f} m of each other at every step of a run longer than {NEAR_MIN_S:.0f} s; a *jump row* = a "
         f"row whose position changes by more than {JUMP_SPEED_MS:.0f} m in the next second on the same track; "
         "*truth drone* = a Dronetag-logged flight (the ground truth); *ghost* = a track farther than "
         f"{MATCH_RADIUS_M:.0f} m from every truth drone. Caveat: while the Dronetag log has a gap (> 3 s) a real track "
         "looks like a ghost.", "",
         "## Near-duplicate pairs", "",
         "Per flagged pair step: same_drone = both tracks within the radius of the SAME truth drone (a true "
         "duplicate); ghost = at least one track far from every truth drone; different_drones = both near different "
         "drones (a legitimate close pair). ONLY same_drone and different_drones steps are evidence: a ghost step "
         "proves nothing about duplication (two unrelated false tracks would be a ghost too), so the real measure is "
         "the confirmed-duplicate share = same_drone / (same_drone + different_drones). A pair is *confirmed* when it "
         "has more same_drone than different_drones steps, *legit close* in the opposite case, *undetermined* when it "
         "has no decisive step or a tie. Contrast: *expected ghost* = the ghost steps that would be seen by chance, "
         "pair steps x (1 - (1 - g)^2) with g = the share of non-jump rows far from every truth drone; a ghost count "
         "far above it would mean flagged pairs sit preferentially on false tracks.", "",
         "| iteration | pair steps | same_drone | ghost (expected by chance) | different_drones | confirmed-duplicate share of decisive steps | pairs | confirmed | legit close | undetermined |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    tot = Counter()
    for k in sorted(cal):
        d = cal[k]["near_dup"]
        L.append(f"| {k} | {d['flagged_pair_steps']} | {d['same_drone']} | {d['ghost']} ({d['expected_ghost_steps']:.0f}) | "
                 f"{d['different_drones']} | {_pct(d['same_drone'], d['same_drone'] + d['different_drones'])} | "
                 f"{d['pairs']} | {d['pairs_confirmed']} | {d['pairs_legit_close']} | {d['pairs_undetermined']} |")
        for kk in ("flagged_pair_steps", "same_drone", "ghost", "different_drones", "pairs", "pairs_confirmed",
                   "pairs_legit_close", "pairs_undetermined", "expected_ghost_steps"):
            tot[kk] += d[kk]
    L.append(f"| **pooled** | {tot['flagged_pair_steps']} | {tot['same_drone']} | "
             f"{tot['ghost']} ({tot['expected_ghost_steps']:.0f}) | {tot['different_drones']} | "
             f"{_pct(tot['same_drone'], tot['same_drone'] + tot['different_drones'])} | {tot['pairs']} | "
             f"{tot['pairs_confirmed']} | {tot['pairs_legit_close']} | {tot['pairs_undetermined']} |")
    L += ["", "## Jump rows", "",
          "Per jump row (both endpoints, 1 s apart): end_off = an endpoint is far from every truth drone; swap = the "
          "endpoints are near different drones (an identity switch); same_drone = both near the same drone (an offset "
          "jump); truth_fast = that drone itself moved > 40 m in the second (not a defect). *wrong* = everything "
          "except truth_fast. Contrast: share of NON-jump rows with an endpoint far from every truth drone.", "",
          "| iteration | jump rows | end_off | swap | same_drone | truth_fast | wrong share | non-jump rows far from truth |",
          "|---|---|---|---|---|---|---|---|"]
    jt = Counter()
    for k in sorted(cal):
        d = cal[k]["jump"]
        wrong = d["jump_rows"] - d["truth_fast"]
        L.append(f"| {k} | {d['jump_rows']} | {d['end_off']} | {d['swap']} | {d['same_drone']} | {d['truth_fast']} | "
                 f"{_pct(wrong, d['jump_rows'])} | {_pct(d['nonjump_end_off'], d['nonjump_rows_evaluated'])} |")
        for kk in ("jump_rows", "end_off", "swap", "same_drone", "truth_fast", "nonjump_end_off",
                   "nonjump_rows_evaluated"):
            jt[kk] += d[kk]
    L.append(f"| **pooled** | {jt['jump_rows']} | {jt['end_off']} | {jt['swap']} | {jt['same_drone']} | "
             f"{jt['truth_fast']} | {_pct(jt['jump_rows'] - jt['truth_fast'], jt['jump_rows'])} | "
             f"{_pct(jt['nonjump_end_off'], jt['nonjump_rows_evaluated'])} |")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- run one method


def select_iterations(spec: list[str] | None) -> list[C.Iteration]:
    its = C.iterations()
    if not spec:
        return its
    out = []
    for it in its:
        for s in spec:
            if s == it.key or s == C.iter_slug(it.key) or s == it.key.split("/")[0]:
                out.append(it)
                break
    return out


def code_stamp(whitelist: set[str]) -> str:
    """sha256 over this file's source and the whitelist: a cached result is valid only for the same
    definitions (constants and code of health.py) and the same Friendly-copy whitelist."""
    h = hashlib.sha256()
    h.update(Path(__file__).read_bytes())
    h.update("\n".join(sorted(whitelist)).encode("utf-8"))
    return h.hexdigest()[:16]


def _stamp(p: Path) -> list[int]:
    st = p.stat()
    return [int(st.st_size), int(st.st_mtime_ns)]


def compute_method(method: str, its: list[C.Iteration], tree: Path, whitelist: set[str],
                   calibration: bool = True, verbose: bool = True, keep_first: list | None = None) -> dict:
    res = {"method": method, "tree": tree.name, "code_stamp": code_stamp(whitelist), "file_stamps": {},
           "iterations": {}, "calibration": {}}
    for it in its:
        p = _tree_paths(tree, method, it)
        if not p.exists():
            continue
        t0 = time.time()
        rows = S.read_submission(it, p)
        m, (e, n, u, grid_t) = health_of_rows(it, rows, whitelist, detail=True)
        det = m.pop("_detail")
        info = _fused_info(tree, it)
        for w in m["near_dup"]["worst"]:
            for k in ("a", "b"):
                fi = info.get(w["track_" + k], {})
                w["sensors_" + k] = fi.get("sensor_ids", "")
                w["members_" + k] = fi.get("n_members", "")
        if it.has_groundtruth:
            m["truth_drones_per_step"] = _truth_per_step(it)
            if calibration:
                res["calibration"][it.key] = calibrate(it, det, e, n, rows["step"], grid_t)
        res["iterations"][it.key] = m
        res["file_stamps"][it.key] = _stamp(p)
        if keep_first is not None and not keep_first:
            keep_first.append((it, rows, e, n, u, grid_t))
        if verbose:
            print(f"  {it.key}: {m['n_rows']} rows, {m['n_tracks']} tracks ({time.time() - t0:.1f} s)", flush=True)
    return res


def save(res: dict, tree: Path) -> Path:
    d = _health_dir(tree)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{res['method']}.json"
    p.write_text(json.dumps(_py(res), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    if res.get("calibration"):
        (d / f"{res['method']}_calibration.md").write_text(_calibration_md(res["method"], res["calibration"]),
                                                            encoding="utf-8")
    return p


def get_health(method: str, its: list[C.Iteration], tree: Path, whitelist: set[str]) -> dict:
    """Cached JSON if its file stamps match the current submission files, else recomputed."""
    p = _health_dir(tree) / f"{method}.json"
    if p.exists():
        try:
            cached = json.loads(p.read_text(encoding="utf-8"))
            if cached.get("code_stamp") != code_stamp(whitelist):
                raise ValueError("stale: health.py or whitelist changed since this JSON was written")
            fs = cached.get("file_stamps", {})
            need = [it for it in its if _tree_paths(tree, method, it).exists()]
            if need and all(it.key in fs and fs[it.key] == _stamp(_tree_paths(tree, method, it)) for it in need):
                return cached
        except (ValueError, KeyError):
            pass
    res = compute_method(method, its, tree, whitelist, verbose=False)
    save(res, tree)
    return json.loads(json.dumps(_py(res)))


# --------------------------------------------------------------------------- printing


def _f(x, nd=3):
    return f"{x:.{nd}f}"


def print_table(res: dict) -> None:
    print(f"\nHealth of {res['method']} (tree {res['tree']})")
    hdr = ("iteration", "rows", "trk", "cov%", "nd_share", "nd_pairs", "nd_sec", "jmp_share", "jmp_trk",
           "fr_copy_s", "fr_max", "fr_stat", "h_out", "tps med/p90/max", "truth/step all | airborne (med/p90/max)")
    print(" | ".join(hdr))
    for k, m in res["iterations"].items():
        tr = m.get("truth_drones_per_step")
        trs = (f"{tr['all']['median']:.0f}/{tr['all']['p90']:.0f}/{tr['all']['max']} | "
               f"{tr['airborne']['median']:.0f}/{tr['airborne']['p90']:.0f}/{tr['airborne']['max']}") if tr else "-"
        tps = m["tracks_per_step"]
        print(" | ".join([k, str(m["n_rows"]), str(m["n_tracks"]), _f(100 * m["coverage"]["covered_step_share"], 1),
                          _f(m["near_dup"]["step_share"], 4), str(m["near_dup"]["pairs"]),
                          _f(m["near_dup"]["pair_seconds"], 1), _f(m["jump"]["row_share"], 5),
                          str(m["jump"]["tracks"]), _f(m["friendly"]["copy_seconds"], 1),
                          str(m["friendly"]["max_simul"]), str(m["friendly"]["stationary"]["tracks"]),
                          str(m["height"]["out_of_range_rows"]),
                          f"{tps['median']:.0f}/{tps['p90']:.0f}/{tps['max']}", trs]))
    for k, m in res["iterations"].items():
        for w in m["near_dup"]["worst"][:3]:
            print(f"  worst near-dup {k}: {w['seconds']:.1f} s, min {w['min_dist_m']:.1f} m, mean "
                  f"{w['mean_dist_m']:.1f} m, tracks {w['track_a'][:8]}[{w.get('sensors_a', '')}] / "
                  f"{w['track_b'][:8]}[{w.get('sensors_b', '')}]")
    if res.get("calibration"):
        nd = Counter()
        jr = Counter()
        for c in res["calibration"].values():
            nd.update({k: v for k, v in c["near_dup"].items() if not isinstance(v, float)})
            jr.update(c["jump"])
        print(f"\nCalibration (testing, pooled over {len(res['calibration'])} iterations): near-dup pair steps "
              f"{nd['flagged_pair_steps']}: same_drone {nd['same_drone']}, ghost {nd['ghost']}, "
              f"different_drones {nd['different_drones']}; jump rows {jr['jump_rows']}: end_off {jr['end_off']}, "
              f"swap {jr['swap']}, same_drone {jr['same_drone']}, truth_fast {jr['truth_fast']}; non-jump rows far "
              f"from truth {_pct(jr['nonjump_end_off'], jr['nonjump_rows_evaluated'])}")


def print_compare(a: str, b: str, ra: dict, rb: dict, verdict: dict, eval_keys: list[str]) -> None:
    print(f"\nCompare candidate B={b} vs reference A={a}")
    print("iteration | " + " | ".join(f"{k} (A,B)" for k in GATE_KEYS) + " | guard cov/rows ratio | verdict")
    for k in sorted(set(ra["iterations"]) & set(rb["iterations"])):
        ga, gb = gate_values(ra["iterations"][k]), gate_values(rb["iterations"][k])
        g = coverage_guard(ra["iterations"][k], rb["iterations"][k])
        tag = "EVAL " if k in eval_keys else "test "
        if k in verdict["per_iter"]:
            v = "PASS" if verdict["per_iter"][k]["ok"] else "FAIL"
        else:
            v = "(context)"
        cells = []
        for name in GATE_KEYS:
            x, y = ga[name], gb[name]
            cells.append(f"{x:.6g},{y:.6g}" + ("" if y <= x + EPS else " WORSE"))
        print(f"{tag}{k} | " + " | ".join(cells) + f" | {g['covered_ratio']:.3f}/{g['rows_ratio']:.3f}"
              + ("" if g["ok"] else " GUARD") + f" | {v}")
    if verdict["missing"]:
        print(f"MISSING evaluation files for: {verdict['missing']}")
    print(f"GATE: {'PASS' if verdict['pass'] else 'FAIL'}")


# --------------------------------------------------------------------------- self-checks


class _Spec:
    """One synthetic track over grid steps [a, b]: start (e0, n0), velocity (ve, vn) m/s, labels;
    ``follow`` = copy another spec's path plus ``off``; ``tele`` = (step, de) east teleport."""

    def __init__(self, key, a, b, e0, n0, ve=0.0, vn=0.0, identity="Unknown", serial="Unknown",
                 follow=None, off=(0.0, 0.0), tele=None, heights=None):
        self.key, self.a, self.b, self.e0, self.n0, self.ve, self.vn = key, a, b, e0, n0, ve, vn
        self.identity, self.serial, self.follow, self.off, self.tele, self.heights = (
            identity, serial, follow, off, tele, heights)


def _base_specs(w1: str, w2: str) -> list[_Spec]:
    A = _Spec("A", 0, 4499, 0, 0, 8.0, 0.0)
    return [
        A,
        _Spec("dup_long", 500, 1499, 0, 0, 8.0, 0.0, follow=A, off=(0.0, 5.0)),       # 1000 steps = 100 s
        _Spec("dup_101", 2000, 2100, 0, 0, 8.0, 0.0, follow=A, off=(0.0, 10.0)),       # 101 steps = 10.1 s
        _Spec("dup_100", 3000, 3099, 0, 0, 8.0, 0.0, follow=A, off=(0.0, 10.0)),       # 100 steps = 10.0 s
        _Spec("B", 0, 4499, 0, 2000, 8.0, 0.0),
        _Spec("short_B", 1000, 1079, 0, 2000, 8.0, 0.0, off=(0.0, 15.0)),              # 8 s: not flagged
        _Spec("fast", 1000, 1100, 0, 4000, 60.0, 0.0),                                 # 91 jump rows
        _Spec("slow", 200, 300, 0, 5000, 39.5, 0.0),                                   # below the bar
        _Spec("tele", 0, 4499, 0, 6000, 8.0, 0.0, tele=(2500, 60.0)),                  # 10 jump rows
        _Spec("F1", 1000, 2999, 0, 8000, 5.0, 0.0, identity="Friendly", serial=w1),
        _Spec("F2", 2000, 2599, 0, 8100, 5.0, 0.0, identity="Friendly", serial=w1),    # 60 s overlap
        _Spec("F3", 3200, 3499, 0, 8200, 5.0, 0.0, identity="Friendly", serial=w1),
        _Spec("F4", 100, 400, 0, 8300, 5.0, 0.0, identity="Friendly", serial=w2),
        _Spec("S1", 500, 1199, 0, 9000, 0.0, 0.0, identity="Friendly"),                # 70 s stationary
        _Spec("S2", 1500, 2099, 0, 9100, 0.0, 0.0, identity="Friendly"),               # 60.0 s: not flagged
        _Spec("S3", 500, 1199, 0, 9200, 0.0, 0.0),                                     # not Friendly
        _Spec("H", 0, 61, 0, 10000, 0.0, 0.0, heights=lambda st: np.where(
            st < 50, 2000.0, np.where(st < 60, -100.0, np.where(st == 60, 1499.9, -49.9)))),
        _Spec("lonely", 4600, 5300, 0, 12000, 8.0, 0.0),
    ]


def _build(it: C.Iteration, specs: list[_Spec]) -> dict:
    cols = {k: [] for k in ("step", "key", "e", "n", "u", "identity", "serial")}
    for sp in specs:
        st = np.arange(sp.a, sp.b + 1)
        f = sp.follow if sp.follow is not None else sp
        e = f.e0 + f.ve * (st - f.a) * 0.1 + sp.off[0]
        n = f.n0 + f.vn * (st - f.a) * 0.1 + sp.off[1]
        if sp.tele is not None:
            e = e + np.where(st >= sp.tele[0], sp.tele[1], 0.0)
        u = sp.heights(st) if sp.heights else np.full(len(st), 100.0)
        for k, v in (("step", st), ("e", e), ("n", n), ("u", u)):
            cols[k].append(v)
        cols["key"].append(np.full(len(st), sp.key, dtype=object))
        cols["identity"].append(np.full(len(st), sp.identity, dtype=object))
        cols["serial"].append(np.full(len(st), sp.serial, dtype=object))
    cat = {k: np.concatenate(v) for k, v in cols.items()}
    import track_rows as TR
    return TR.make_rows(it, cat["step"], cat["key"], cat["e"], cat["n"], cat["u"], 25.0, 25.0, 25.0, 0.0,
                        {"identity": cat["identity"], "serial_number": cat["serial"]})


def _drop(rows: dict, ids: set[str]) -> dict:
    keep = np.array([t not in ids for t in rows["track_id"]])
    return {k: np.asarray(v)[keep] for k, v in rows.items()}


def _recount(step, tid, e, n, grid_t) -> tuple[float, float]:
    """Plain-python near_dup_step_share and jump_row_share (independent of the vectorised code)."""
    dt, near_min, hop, _, S_ = _grid_params(grid_t)
    by_step = defaultdict(list)
    pos = {}
    for s, t, x, y in zip(step.tolist(), tid.tolist(), e.tolist(), n.tolist()):
        by_step[s].append((t, x, y))
        pos[(t, s)] = (x, y)
    pair_steps = defaultdict(list)
    for s in sorted(by_step):
        lst = by_step[s]
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if lst[i][0] != lst[j][0] and math.hypot(lst[i][1] - lst[j][1], lst[i][2] - lst[j][2]) <= NEAR_RADIUS_M:
                    pair_steps[frozenset((lst[i][0], lst[j][0]))].append(s)
    flagged = set()
    for ps in pair_steps.values():
        ps.sort()
        run = [ps[0]]
        for s in ps[1:] + [None]:
            if s is not None and s == run[-1] + 1:
                run.append(s)
                continue
            if len(run) > near_min:
                flagged.update(run)
            run = [s]
    jumps = 0
    for (t, s), (x, y) in pos.items():
        q = pos.get((t, s + hop))
        if q is not None and math.hypot(q[0] - x, q[1] - y) / (grid_t[s + hop] - grid_t[s]) > JUMP_SPEED_MS:
            jumps += 1
    return len(flagged) / S_, jumps / len(step)


def selfchecks(real=None, full: bool = True) -> int:
    """Numbered self-checks; returns the number of failures."""
    results: list[tuple[str, bool, str]] = []

    def rec(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    # ---- boundary unit checks on raw arrays (no round trip)
    grid_t = 1000.0 + 0.1 * np.arange(2000)
    wl = {"W"}
    z = np.zeros
    ub = np.array([1500.0, -50.0, 1500.01, -50.01, 0.0])
    m = indicators(np.arange(5), ["t"] * 5, z(5), z(5), ub, ["Unknown"] * 5, ["Unknown"] * 5, grid_t, wl)
    rec("1 height boundaries: 1500 and -50 inside, 1500.01 and -50.01 outside", m["height"]["out_of_range_rows"] == 2)
    for n_steps, want in ((100, 0), (101, 1)):
        st = np.concatenate([np.arange(n_steps)] * 2)
        tid = ["a"] * n_steps + ["b"] * n_steps
        e = np.concatenate([z(n_steps), z(n_steps) + 20.0])           # exactly 20 m apart: inside
        mm = indicators(st, tid, e, z(2 * n_steps), z(2 * n_steps) + 50, ["Unknown"] * (2 * n_steps),
                        ["Unknown"] * (2 * n_steps), grid_t, wl)
        rec(f"2 near-dup run of {n_steps} steps (10.0 s boundary): flagged pairs == {want}",
            mm["near_dup"]["pairs"] == want)
    st = np.concatenate([np.r_[np.arange(0, 60), np.arange(61, 130)]] * 2)
    k = 129
    mm = indicators(st, ["a"] * k + ["b"] * k, np.r_[z(k), z(k) + 5.0], z(2 * k), z(2 * k) + 50,
                    ["Unknown"] * (2 * k), ["Unknown"] * (2 * k), grid_t, wl)
    rec("3 a one-step gap splits a run (60 + 69 steps, neither > 100): not flagged", mm["near_dup"]["pairs"] == 0)

    # ---- calibration classes on a synthetic truth (3 pair groups of 200 steps each)
    k3 = 200
    stp = np.concatenate([np.arange(k3)] * 6)
    tid6 = np.repeat(["a1", "a2", "b1", "b2", "c1", "c2"], k3)
    e6 = np.repeat([0.0, 5.0, 1000.0, 1010.0, 5000.0, 5005.0], k3)
    n6 = z(6 * k3)
    mm = indicators(stp, list(tid6), e6, n6, z(6 * k3) + 50, ["Unknown"] * (6 * k3), ["Unknown"] * (6 * k3),
                    grid_t, wl, detail=True)
    dtl = mm.pop("_detail")
    Et = np.tile(np.array([[0.0], [995.0], [1015.0]]), (1, 2000))     # truth drone 0 at the a-pair, 1 and 2 at b
    cal = calibrate(None, dtl, e6, n6, stp, grid_t, truth=(Et, np.zeros_like(Et)))["near_dup"]
    rec("3b calibration classes: a-pair = same_drone (confirmed), b-pair = different_drones (legit close), "
        "c-pair = ghost (undetermined, NOT counted as wrong)",
        cal["same_drone"] == k3 and cal["different_drones"] == k3 and cal["ghost"] == k3 and cal["pairs"] == 3
        and cal["pairs_confirmed"] == 1 and cal["pairs_legit_close"] == 1 and cal["pairs_undetermined"] == 1
        and abs(cal["pair_seconds_confirmed"] - k3 * 0.1) < 1e-9, str(cal))

    if full:
        import track_rows as TR
        it = next(i for i in C.iterations() if i.key == "testing/S1-I1")
        S_ = len(C.template_grid(it)[1])
        w1, w2 = sorted(_load_whitelist())[:2]
        specs = _base_specs(w1, w2)
        rows = _build(it, specs)
        wlist = {w1, w2}
        mA = health_of_rows(it, rows, wlist)
        tot_rows = sum(sp.b - sp.a + 1 for sp in specs)
        per_step = Counter(s for sp in specs for s in range(sp.a, sp.b + 1))
        rec("4 synthetic row count, covered steps, tracks and tracks-per-step max equal the construction",
            mA["n_rows"] == tot_rows and mA["coverage"]["covered_steps"] == len(per_step)
            and mA["tracks_per_step"]["max"] == max(per_step.values()) and mA["n_tracks"] == len(specs),
            f"rows {mA['n_rows']}/{tot_rows} cov {mA['coverage']['covered_steps']}/{len(per_step)} "
            f"max {mA['tracks_per_step']['max']}/{max(per_step.values())}")
        nd = mA["near_dup"]
        want_steps = 1000 + 101
        rec("5 injected duplicates: exactly the 100 s and the 10.1 s pairs (not 10.0 s, not 8 s)",
            nd["pairs"] == 2 and nd["steps"] == want_steps and abs(nd["pair_seconds"] - 110.1) < 1e-6
            and abs(nd["step_share"] - want_steps / S_) < 1e-12,
            str({k: nd[k] for k in ("pairs", "steps", "pair_seconds")}))
        jr = mA["jump"]
        rec("6 jumps: 60 m/s run (91 rows) + 60 m teleport (10 rows), 39.5 m/s not counted",
            jr["rows"] == 101 and jr["tracks"] == 2 and abs(jr["row_share"] - 101 / tot_rows) < 1e-12, str(jr))
        fr = mA["friendly"]
        rec("7 Friendly copies: serial 1 has max 2 simultaneous for 60.0 s over 3 tracks, serial 2 max 1",
            fr["per_serial"][w1]["max_simul"] == 2 and abs(fr["per_serial"][w1]["seconds_ge2"] - 60.0) < 1e-9
            and fr["per_serial"][w1]["tracks"] == 3 and fr["per_serial"][w2]["max_simul"] == 1
            and fr["per_serial"][w2]["seconds_ge2"] == 0.0 and abs(fr["copy_seconds"] - 60.0) < 1e-9
            and fr["max_simul"] == 2, str(fr["per_serial"]))
        rec("8 stationary Friendly: the 70 s track only (60.0 s and non-Friendly not counted)",
            fr["stationary"]["tracks"] == 1 and abs(fr["stationary"]["seconds"] - 70.0) < 1e-9, str(fr["stationary"]))
        rec("9 height out of range: 50 rows at 2000 m + 10 rows at -100 m, the two inside rows not counted",
            mA["height"]["out_of_range_rows"] == 60 and mA["height"]["tracks"] == 1, str(mA["height"]))

        def mut(edit):
            sp2 = _base_specs(w1, w2)
            edit({s.key: s for s in sp2})
            return health_of_rows(it, _build(it, sp2), wlist)

        m1 = mut(lambda d: setattr(d["dup_100"], "b", 3100))
        rec("10 mutation: dup_100 lengthened to 101 steps -> pairs 3", m1["near_dup"]["pairs"] == 3)
        m2 = mut(lambda d: setattr(d["dup_101"], "off", (0.0, 25.0)))
        rec("11 mutation: dup_101 moved to 25 m -> pairs 1", m2["near_dup"]["pairs"] == 1)
        m3 = mut(lambda d: setattr(d["fast"], "ve", 39.0))
        rec("12 mutation: fast track slowed to 39 m/s -> jump rows 10", m3["jump"]["rows"] == 10)
        m4 = mut(lambda d: setattr(d["F2"], "serial", "Unknown"))
        rec("13 mutation: second Friendly copy loses the serial -> copy seconds 0",
            m4["friendly"]["copy_seconds"] == 0.0)
        m5 = mut(lambda d: setattr(d["S1"], "ve", 1.0))
        rec("14 mutation: stationary Friendly moves at 1 m/s -> stationary tracks 0",
            m5["friendly"]["stationary"]["tracks"] == 0)

        tmp = Path(tempfile.mkdtemp(prefix="health_"))
        pth = tmp / C.template_path(it).name
        S.write_submission(it, rows, pth)
        mB = health_of_rows(it, S.read_submission(it, pth), wlist)
        rec("15 file round trip (xlsx write -> read) reproduces every indicator",
            _py(gate_values(mB)) == _py(gate_values(mA)) and mB["near_dup"]["pairs"] == nd["pairs"]
            and mB["jump"]["rows"] == jr["rows"] and mB["friendly"]["stationary"]["tracks"] == 1)
        pth.unlink()
        tmp.rmdir()

        # cache: a JSON written under another code stamp (changed definitions / whitelist) is not reused
        tdir = Path(tempfile.mkdtemp(prefix="health_cache_"))
        sub = tdir / "submissions" / "M" / C.iter_slug(it.key)
        sub.mkdir(parents=True)
        S.write_submission(it, rows, sub / C.template_path(it).name)
        r1 = get_health("M", [it], tdir, wlist)
        jp = tdir / "health" / "M.json"
        stale = json.loads(jp.read_text(encoding="utf-8"))
        stale["code_stamp"] = "stale"
        stale["iterations"][it.key]["near_dup"]["pairs"] = -1
        jp.write_text(json.dumps(stale), encoding="utf-8")
        r2 = get_health("M", [it], tdir, wlist)
        stale["code_stamp"] = r1["code_stamp"]
        jp.write_text(json.dumps(stale), encoding="utf-8")
        r3 = get_health("M", [it], tdir, wlist)
        rec("15b cache: a stale code stamp forces a recompute; a matching stamp is reused",
            r1["code_stamp"] == code_stamp(wlist) and r2["iterations"][it.key]["near_dup"]["pairs"] == nd["pairs"]
            and r3["iterations"][it.key]["near_dup"]["pairs"] == -1)
        for q in sorted(tdir.rglob("*"), reverse=True):
            q.unlink() if q.is_file() else q.rmdir()
        tdir.rmdir()

        tid = {sp.key: TR.track_id(it, sp.key) for sp in specs}
        c_del_a = health_of_rows(it, _drop(rows, {tid["A"]}), wlist)
        g = coverage_guard(mA, c_del_a)
        rec("16 coverage guard fires when track A (21 % of rows) is deleted (rows), covered steps unchanged",
            (not g["ok"]) and (not g["rows_ok"]) and g["covered_ok"], str(g))
        c_del_l = health_of_rows(it, _drop(rows, {tid["lonely"]}), wlist)
        g = coverage_guard(mA, c_del_l)
        rec("17 coverage guard fires when the lonely track is deleted (covered steps), rows within 5 %",
            (not g["ok"]) and (not g["covered_ok"]) and g["rows_ok"], str(g))
        c_del_h = health_of_rows(it, _drop(rows, {tid["H"]}), wlist)
        g = coverage_guard(mA, c_del_h)
        rec("18 coverage guard holds when only the 62-row height track is deleted", g["ok"], str(g))
        rec("19 coverage guard holds for A versus A", coverage_guard(mA, mA)["ok"])

        keys = ["e1", "e2", "e3"]
        a_by = {k: mA for k in keys}
        rec("20 gate: A versus A passes (all indicators equal)", gate(a_by, a_by, keys)["pass"])
        v_fix = gate(a_by, {k: c_del_h for k in keys}, keys)
        rec("21 gate: removing the out-of-range track (coverage held) passes", v_fix["pass"])
        rec("22 gate: deleting track A does not pass (coverage guard)",
            not gate(a_by, {k: c_del_a for k in keys}, keys)["pass"])
        sp3 = _base_specs(w1, w2)
        sp3.append(_Spec("dup_extra", 3600, 3800, 0, 0, 8.0, 0.0, follow=sp3[0], off=(0.0, 7.0)))
        c_worse = health_of_rows(it, _build(it, sp3), wlist)
        v_worse = gate(a_by, {"e1": mA, "e2": c_worse, "e3": mA}, keys)
        rec("23 gate: one extra duplicate on ONE of three iterations fails near_dup there",
            (not v_worse["pass"]) and not v_worse["per_iter"]["e2"]["indicators"]["near_dup_step_share"]["ok"]
            and v_worse["per_iter"]["e1"]["ok"], str(v_worse["per_iter"]["e2"]["indicators"]["near_dup_step_share"]))
        v_miss = gate(a_by, {"e1": mA, "e2": mA}, keys)
        rec("24 gate: a missing evaluation iteration fails", (not v_miss["pass"]) and v_miss["missing"] == ["e3"])

    if real is not None:
        it, rows, e, n, u, gt = real
        mm = indicators(rows["step"], rows["track_id"], e, n, u, rows["identity"], rows["serial_number"], gt, set())
        rs, rj = _recount(rows["step"], rows["track_id"], e, n, gt)
        rec(f"25 plain-python recount on {it.key}: near_dup_step_share and jump_row_share equal the vectorised values",
            abs(rs - mm["near_dup"]["step_share"]) < 1e-12 and abs(rj - mm["jump"]["row_share"]) < 1e-12,
            f"recount nd {rs:.6f} jump {rj:.6f}; vectorised nd {mm['near_dup']['step_share']:.6f} "
            f"jump {mm['jump']['row_share']:.6f}")
    bad = 0
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok or not detail else f"  <{detail}>"))
        bad += not ok
    print("ALL SELF-CHECKS PASSED" if not bad else f"{bad} SELF-CHECK(S) FAILED")
    return bad


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("method", nargs="?")
    ap.add_argument("--iters", nargs="*")
    ap.add_argument("--tree", default=None)
    ap.add_argument("--tree-a", default=None)
    ap.add_argument("--tree-b", default=None)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--no-calibration", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    tree = Path(args.tree).resolve() if args.tree else C.DERIVED
    if args.selftest:
        return 1 if selfchecks(None, True) else 0
    if args.compare:
        a, b = args.compare
        ta = Path(args.tree_a).resolve() if args.tree_a else tree
        tb = Path(args.tree_b).resolve() if args.tree_b else tree
        its = select_iterations(args.iters)
        wl = _load_whitelist()
        ra = get_health(a, its, ta, wl)
        rb = get_health(b, its, tb, wl)
        eval_keys = [it.key for it in its if not it.has_groundtruth]
        verdict = gate(ra["iterations"], rb["iterations"], eval_keys)
        print_compare(a, b, ra, rb, verdict, eval_keys)
        if a == b and ta == tb:
            ok = verdict["pass"] and all(
                gate_values(ra["iterations"][k]) == gate_values(rb["iterations"][k]) for k in eval_keys)
            print(f"  [{'PASS' if ok else 'FAIL'}] identity check: {a} versus itself is all-equal and passes the gate")
            if not ok:
                return 1
        print("\nSelf-checks (boundaries only; use --selftest or a METHOD run for the full set):")
        if selfchecks(None, False):
            return 1
        return 0 if verdict["pass"] else 1
    if not args.method:
        ap.error("METHOD, --compare A B or --selftest is required")
    its = select_iterations(args.iters)
    wl = _load_whitelist()
    first: list = []
    t0 = time.time()
    res = compute_method(args.method, its, tree, wl, calibration=not args.no_calibration, keep_first=first)
    if not res["iterations"]:
        print(f"no submission files for {args.method} in {tree}")
        return 1
    pth = save(res, tree)
    print_table(res)
    print(f"\nwritten {pth} ({time.time() - t0:.0f} s)")
    print("\nSelf-checks:")
    return 1 if selfchecks(first[0] if first else None, True) else 0


if __name__ == "__main__":
    sys.exit(main())
