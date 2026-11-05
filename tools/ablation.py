#!/usr/bin/env python3
"""Sensor ablation of F0_fuse: how much of the score rides on the UTM relays VKYY / QTVS (or any
other sensors), measured by removing them AT THE INPUT of the fusion tracker.

Why: in testing VKYY / QTVS (and LJJX) relay the Dronetag ground-truth feed itself, so every
testing number that contains them is optimistic. The evaluation tree has the same two sensor
ids, but nobody knows how much of the real air picture they carry there. This tool answers two
things for ANY comma-separated list of sensor ids (--drop, default VKYY,QTVS; handy when a
sensor fails in the live demo):

(a) PRESENCE: per iteration (testing and evaluation) which dropped sensors exist (files,
    segments, objects), how much of the template time grid they cover, how many of F0's rows
    come from tracks that contain them, how many F0 tracks consist only of them and, per
    iteration, how many Friendly / serial-bearing F0 tracks would lose their identity source.
(b) THE CHEAP NUMBER: F0 re-run with the dropped sensors made non-deployable before
    fuse.Prep is built (this process only: cuas_common.deployable_sensor_ids is wrapped; no
    other module is edited), so overlap pairs, clusters, fused rows and track tables never see
    them. What is NOT redone: the existence models (existence/*.csv), the sensor-noise table
    and the segment features were trained WITH the dropped sensors in the data. This is
    therefore "F0's trained models, input without the sensors" -- a fast approximation, not
    the honest number. The honest number retrains every upstream model without the sensors
    (tools/sensor_tracks.py drop_sensors option, a VARIANT tree rebuilt by the controller via
    tools/run.py --derived ...); pass --retrained METHOD to add its promoted submissions to the
    comparison. Differences between the two: (1) the retrained existence model sees no UTM
    segments as 'drone' training examples (here it still scores the remaining segments with a
    model that learnt from UTM-labelled positives), (2) the noise/offset tables and the cut of
    segments are refitted, (3) the retrained LOIO tuning sees the real remaining sensor set.

Variants scored on the 6 testing iterations, both truth modes (score.py, PRIMARY c = 50 m, p = 2,
3-D):
* F0_fuse             the shipped method (files scored as they are).
* <name>              the dropped sensors removed at the input; geometry (edge_gate,
                      cont_gap_s, bridge_s) of each fold = F0's fold geometry; (P_FIELD, TAU)
                      RE-CHOSEN on this sensor set exactly as fuse does (fuse.choose over the
                      other 5 iterations with p_inner of fold k; evaluation: all 6 testing
                      iterations with the outer p). Submissions for all 9 iterations.
* <name>@F0thr        the same tracks with F0's own (P_FIELD, TAU) of that fold (not re-tuned):
                      separates "the sensor is missing" from "the threshold was re-tuned".
                      In memory only (no file).

Outputs
-------
data/derived/ablation/<name>.md             plain-language report
data/derived/ablation/<name>.csv            scores: iter_key, truth_mode, variant, combined,
                                            track, classification, identity, metadata, n_tracks,
                                            n_rows, edge_gate, cont_gap_s, bridge_s, p_field, tau
data/derived/ablation/<name>_presence.csv   per (iteration, dropped sensor) presence figures
data/derived/ablation/<name>_dependence.csv per iteration: F0 rows / tracks that depend on them
data/derived/ablation/<name>_tuning.csv     fuse's tuning table schema, restricted to the fold
                                            geometry (k, j, ..., p_fold)
data/derived/ablation/<name>/<slug>__tracks.csv, __members.csv   the ablated fused tracks
data/derived/submissions/<name>/            9 validated submissions
experiments.csv                             headline scores appended (score.score_method(log=True))

Rules kept: leave-one-iteration-out (fold k's (P_FIELD, TAU) uses only j != k with p_inner rows of
outer_fold == k); nothing is fitted on evaluation; the testing-only sensors EUIG LJJX MMJJ NJQL
never feed an output; the dropped ids are removed from every member of every track.

Run: python3 tools/ablation.py [--drop VKYY,QTVS] [--name F0_noutm_input] [--workers 3]
                               [--retrained METHOD] [--skip-score-log]
Numbered self-checks at the end; exit status 1 on any failure.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuas_common as C  # noqa: E402

# --------------------------------------------------------------------------- the one patch
# Sensor ids that fuse.Prep must treat as non-deployable in THIS process. fuse reads
# C.deployable_sensor_ids() when it builds a Prep, so wrapping it here is enough; nothing else
# in the repository is touched. DROP is mutated in place (workers inherit it through fork).
DROP: set[str] = set()
_DEP0 = C.deployable_sensor_ids()          # the real deployable set, computed once


def _deployable_patched() -> frozenset:
    return frozenset(_DEP0 - DROP)


C.deployable_sensor_ids = _deployable_patched


@contextlib.contextmanager
def dropping(ids):
    """Temporarily set the dropped ids (a worker needs the empty set for the F0 reproduction)."""
    old = set(DROP)
    DROP.clear()
    DROP.update(ids)
    try:
        yield
    finally:
        DROP.clear()
        DROP.update(old)


import baselines as B  # noqa: E402
import compare as CMP  # noqa: E402
import fuse as FU  # noqa: E402
import metadata_map as MM  # noqa: E402
import score as SC  # noqa: E402
import submission as S  # noqa: E402

OUT = C.DERIVED / "ablation"
TRACKS_DIR = FU.TRACKS_DIR
FUSED = FU.OUT
ITERS = FU.ITERS
TESTING, EVALUATION = FU.TESTING, FU.EVALUATION
MODES = FU.TRUTH_MODES
DEFAULT_DROP = ("VKYY", "QTVS")
PROTECTED = ("F0_fuse",)
RUNTIME_LIMIT_S = 25 * 60
RT_TOL_M = 1e-3
PARTS = ("combined", "track", "classification", "identity", "metadata")
TUNING_COLS = ["k", "j", "edge_gate", "cont_gap_s", "bridge_s", "p_field", "tau", "combined_all",
               "track_all", "combined_airborne", "track_airborne", "n_tracks", "n_rows", "p_fold"]


def log(msg: str) -> None:
    print(msg, flush=True)


def _f(v, nd: int = 3) -> str:
    return "n/a" if v is None or v != v else f"{v:.{nd}f}"


def _sf(v, nd: int = 3) -> str:
    return "n/a" if v is None or v != v else f"{v:+.{nd}f}"


# --------------------------------------------------------------------------- F0 rules


def f0_rules() -> dict:
    """{fold key k: F0's (edge_gate, cont_gap_s, bridge_s, p_field, tau)} and 'outer' (the
    evaluation choice), re-derived from fused/tuning.csv with fuse.choose (the argmax fuse.py
    applied)."""
    tb = C.read_columns(FU.TUNING_CSV, numeric={"cont_gap_s", "bridge_s", "tau", "combined_all",
                                                "combined_airborne"})
    recs = [{"k": tb["k"][i], "j": tb["j"][i], "edge_gate": tb["edge_gate"][i],
             "cont_gap_s": float(tb["cont_gap_s"][i]), "bridge_s": float(tb["bridge_s"][i]),
             "p_field": tb["p_field"][i], "tau": float(tb["tau"][i]),
             "combined_all": float(tb["combined_all"][i]),
             "combined_airborne": float(tb["combined_airborne"][i])} for i in range(len(tb["k"]))]
    out = {k: FU.choose(recs, k, [j for j in TESTING if j != k])[0] for k in TESTING}
    out["outer"] = FU.choose(recs, "outer", TESTING)[0]
    return out


@contextlib.contextmanager
def geometry_grid(geo):
    """fuse.PARAMS restricted to one geometry (all p_field x tau values stay), so fuse.choose /
    fuse.all_combos enumerate only the combinations that were scored."""
    old = FU.PARAMS
    FU.PARAMS = dict(old, edge_gate=(geo[0],), cont_gap_s=(geo[1],), bridge_s=(geo[2],))
    try:
        yield
    finally:
        FU.PARAMS = old


def rule_key(k: str) -> str:
    return "outer" if k in EVALUATION else k


def p_src_of(k: str) -> str:
    """The p source of the final application: testing k and evaluation both use 'outer'."""
    return "outer"


# --------------------------------------------------------------------------- worker jobs


def _geometry(P: FU.Prep, geo: tuple, wl: set):
    Cl = FU.cluster(P, geo[0], geo[1])
    g = FU.fuse(P, Cl, geo[2])
    tt = FU.track_table(P, Cl, g, wl)
    return Cl, g, tt


def tune_job(job: tuple) -> dict:
    """All (src, geometry) scores of testing iteration j: for every fold k != j the fold-k
    geometry with p_inner of fold k, and the final geometry with the outer p. Same scoring as
    fuse.tune_task (shared _score_pair, selection-hash cache), restricted to the geometry each
    source needs."""
    key, srcgeo = job
    t0 = time.time()
    it = ITERS[key]
    P = FU.Prep(it)
    wl = B.load_whitelist()
    by_geo: dict = {}
    for src, geo in sorted(srcgeo.items()):
        by_geo.setdefault(tuple(geo), []).append(src)
    recs = []
    for geo, srcs in sorted(by_geo.items()):
        Cl, g, tt = _geometry(P, geo, wl)
        rows = FU.build_rows(it, g, tt, np.full(Cl.ntracks, 0.5))
        row_track = g["track"]
        cache: dict = {}
        for src in srcs:
            pa, pr, pfold = P.p[src]
            for pf in FU.PARAMS["p_field"]:
                pt = FU.track_p(Cl, pa if pf == "all" else pr, P.D.g_cnt)[0]
                for tau in FU.PARAMS["tau"]:
                    sel = pt >= tau
                    h = hashlib.sha1(np.packbits(sel).tobytes()).hexdigest()
                    if h not in cache:
                        rm = sel[row_track]
                        sub = {c: np.asarray(v)[rm] for c, v in rows.items()}
                        cache[h] = FU._score_pair(it, sub) + (int(sel.sum()), int(rm.sum()))
                    ca, ta, cr, tr_, ntk, nrw = cache[h]
                    recs.append(dict(k=src, j=key, edge_gate=geo[0], cont_gap_s=geo[1], bridge_s=geo[2],
                                     p_field=pf, tau=tau, combined_all=ca, track_all=ta,
                                     combined_airborne=cr, track_airborne=tr_, n_tracks=ntk,
                                     n_rows=nrw, p_fold=pfold))
        del rows, g, tt
    return {"kind": "tune", "key": key, "records": recs, "seconds": time.time() - t0,
            "n_dropped_segments": int(np.isin(P.D.sensor_id, sorted(DROP)).sum())}


def parts_of(it: C.Iteration, rows: dict) -> dict:
    return {m: SC.score_rows(it, rows, m, configs=(SC.PRIMARY,))["parts"] for m in MODES}


def members_sensor_check(members_seg: np.ndarray, seg_sensor: dict, drop) -> list:
    """Dropped sensor ids found among the members (seg ids) of tracks: [(seg_id, sensor)]."""
    return [(int(s), seg_sensor[int(s)]) for s in members_seg.tolist() if seg_sensor[int(s)] in drop]


def apply_job(job: tuple) -> dict:
    """One iteration with the dropped sensors removed: ablated fused tracks, selection (re-tuned
    (P_FIELD, TAU) -> submission file; F0's (P_FIELD, TAU) -> in-memory score), checks."""
    key, name, combo, combo_f0 = job
    t0 = time.time()
    it = ITERS[key]
    P = FU.Prep(it)
    D = P.D
    wl = B.load_whitelist()
    Cl, g, tt = _geometry(P, tuple(combo[:3]), wl)
    pa, pr, pfold = P.p["outer"]
    pmax_all, pmax_air = FU.track_p(Cl, pa, D.g_cnt)[0], FU.track_p(Cl, pr, D.g_cnt)[0]
    p_track = pmax_all if combo[3] == "all" else pmax_air
    sel = p_track >= combo[4]
    p_f0 = pmax_all if combo_f0[3] == "all" else pmax_air
    sel_f0 = p_f0 >= combo_f0[4]
    rows = FU.build_rows(it, g, tt, p_track, sel)
    out = {"kind": "apply", "key": key, "combo": tuple(combo), "combo_f0": tuple(combo_f0),
           "p_fold": pfold, "n_tracks_all": Cl.ntracks, "n_tracks": int(sel.sum()),
           "n_tracks_f0thr": int(sel_f0.sum()), "stats": Cl.stats,
           "n_rows": int(len(rows["step"])), "hash": FU.members_hash(Cl)}
    # in-memory scores before the file round trip (testing)
    if it.has_groundtruth:
        out["mem_parts"] = parts_of(it, rows)
        rows0 = FU.build_rows(it, g, tt, p_f0, sel_f0)
        out["f0thr_parts"] = parts_of(it, rows0)
        out["n_rows_f0thr"] = int(len(rows0["step"]))
        del rows0
    # file
    rm = np.flatnonzero(sel[g["track"]])
    path = C.submission_path(name, it)
    rt = B._file_roundtrip(it, rows, path, g["e"][rm], g["n"][rm], g["u"][rm])
    tb = np.asarray(rt["back"]["track_id"]).astype(str)
    uniq = True
    if len(tb):
        code = np.unique(tb, return_inverse=True)[1]
        pair = rt["back"]["step"].astype(np.int64) * (int(code.max()) + 1) + code
        uniq = len(np.unique(pair)) == len(pair)
    sens = set()
    for t in np.flatnonzero(sel).tolist():
        sens |= set(tt["sensor_ids"][t].split("|"))
    out["file"] = {"path": str(path.relative_to(C.REPO)), "rows": len(rows["step"]), "tracks": int(sel.sum()),
                   "problems": rt["problems"], "rt_err_m": rt["rt_err_m"], "rt_labels_ok": rt["rt_labels_ok"],
                   "n_back": rt["n_back"], "unique": uniq, "sensors": sorted(sens),
                   "friendly_tracks": int(sum(1 for t in np.flatnonzero(sel) if tt["lab"]["identity"][t] == "Friendly")),
                   "serial_tracks": int(sum(1 for t in np.flatnonzero(sel) if tt["serial"][t])),
                   "model_tracks": int(sum(1 for t in np.flatnonzero(sel) if tt["model"][t] != "Unknown"))}
    del rows, rt
    # ablated track tables (all tracks before the selection)
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    slug = C.iter_slug(key)
    ids = np.array([FU.TR.track_id(it, f"{key}/track{t}") for t in range(Cl.ntracks)], dtype=object)
    C.write_columns(d / f"{slug}__tracks.csv",
                    {"track": tt["track"], "track_id": ids, "n_members": tt["n_members"],
                     "n_sensors": tt["n_sensors"], "sensor_ids": tt["sensor_ids"], "n_steps": tt["n_steps"],
                     "p_max_all": pmax_all, "p_max_air": pmax_air, "serial": tt["serial"], "model": tt["model"],
                     "identity": tt["lab"]["identity"], "reported": sel.astype(np.int64),
                     "reported_f0thr": sel_f0.astype(np.int64)},
                    ["track", "track_id", "n_members", "n_sensors", "sensor_ids", "n_steps", "p_max_all",
                     "p_max_air", "serial", "model", "identity", "reported", "reported_f0thr"])
    C.write_columns(d / f"{slug}__members.csv", {"track": Cl.mem_track, "cluster": Cl.mem_cluster,
                                                 "seg_id": D.seg_id[Cl.mem_seg]}, ["track", "cluster", "seg_id"])
    # the tracks' own sensor lists from the clusters (check 2 re-reads the written files)
    out["c1"] = FU.check_constraints(P, Cl)
    out["c3"] = FU.check_singletons(P, Cl, g)
    out["n_dep_segments"] = int(P.dep.sum())
    out["n_dropped_segments"] = int(np.isin(D.sensor_id, sorted(DROP)).sum())
    out["n_dropped_in_dep"] = int((P.dep & np.isin(D.sensor_id, sorted(DROP))).sum())
    # presence by the second path (Iter arrays) for cross-checking the raw-file figures
    cov = {}
    for s in sorted(DROP):
        m = D.sensor_id[D.grow] == s
        cov[s] = int(np.unique(D.step[m]).size)
    out["cov_iter"] = cov
    out["seconds"] = time.time() - t0
    return out


def _mismatch(g: dict, z) -> int:
    """Number of fused-grid columns of g that differ from the saved F0 grid (exact)."""
    return sum(0 if np.array_equal(np.asarray(g[c]), z[c]) else 1 for c in z.files)


def repro_job(job: tuple) -> dict:
    """Self-check 1: with NO dropped sensor, the same pipeline reproduces F0_fuse's fused grid
    (fused/<slug>__grid.npz, exact), the reported-track set (fused/<slug>__tracks.csv
    reported_f0) and the rows of data/derived/submissions/F0_fuse (labels / ids exact,
    positions to 1e-3 m). A mutated grid must be detected."""
    key, combo = job
    t0 = time.time()
    it = ITERS[key]
    with dropping(()):
        P = FU.Prep(it)
        wl = B.load_whitelist()
        Cl, g, tt = _geometry(P, tuple(combo[:3]), wl)
    pa, pr, _ = P.p["outer"]
    pt = FU.track_p(Cl, pa if combo[3] == "all" else pr, P.D.g_cnt)[0]
    sel = pt >= combo[4]
    slug = C.iter_slug(key)
    z = np.load(FUSED / f"{slug}__grid.npz")
    n_col_bad = _mismatch(g, z)
    gm = {c: np.array(v, copy=True) for c, v in g.items() if isinstance(v, np.ndarray)}
    gm["e"] = gm["e"] + 1e-6 * (np.arange(gm["e"].size) == gm["e"].size // 2)
    mutation_detected = _mismatch(gm, z) > 0
    ft = C.read_columns(FUSED / f"{slug}__tracks.csv", numeric={"reported_f0"}, usecols=["track", "reported_f0"])
    sel_bad = int((sel.astype(np.int64) != ft["reported_f0"].astype(np.int64)).sum()) \
        if sel.size == ft["reported_f0"].size else -1
    rows = FU.build_rows(it, g, tt, pt, sel)
    back = S.read_submission(it, C.submission_path("F0_fuse", it))
    ee, nn, uu = C.iteration_frame(it).to_enu(back["latitude"], back["longitude"], back["altitude_m"])
    sa = np.asarray(rows["step"]).astype(np.int64)
    ka = np.asarray(rows["track_id"]).astype(str)
    oa = np.lexsort((ka, sa))
    sb = np.asarray(back["step"]).astype(np.int64)
    kb = np.asarray(back["track_id"]).astype(str)
    ob = np.lexsort((kb, sb))
    row_ok = (len(sa) == len(sb)) and np.array_equal(sa[oa], sb[ob]) and np.array_equal(ka[oa], kb[ob])
    lab_ok = row_ok and all(np.array_equal(np.asarray(rows[c]).astype(str)[oa], np.asarray(back[c]).astype(str)[ob])
                            for c in FU.TR.LABEL_COLUMNS)
    rt_err = float("nan")
    if row_ok and len(sa):
        rm = np.flatnonzero(sel[g["track"]])
        rt_err = float(max(np.abs(g["e"][rm][oa] - ee[ob]).max(), np.abs(g["n"][rm][oa] - nn[ob]).max(),
                           np.abs(g["u"][rm][oa] - uu[ob]).max()))
    return {"kind": "repro", "key": key, "n_cols_bad": n_col_bad, "mutation_detected": bool(mutation_detected),
            "sel_bad": sel_bad, "n_rows": int(len(sa)), "n_back": int(len(sb)), "row_ok": bool(row_ok),
            "lab_ok": bool(lab_ok), "rt_err_m": rt_err, "seconds": time.time() - t0}


def _job(job):
    return {"tune": tune_job, "apply": apply_job, "repro": repro_job}[job[0]](job[1:])


# --------------------------------------------------------------------------- presence (raw files)


def presence(key: str, drop: list[str]) -> list[dict]:
    """Per dropped sensor: files, segments, objects, grid coverage of the template, from the raw
    sensor files, sensor_tracks tracks.csv and the grid.npz (independent of fuse.Prep)."""
    it = ITERS[key]
    slug = C.iter_slug(key)
    iso, _ = C.template_grid(it)
    n_steps = len(iso)
    t = C.read_columns(TRACKS_DIR / f"{slug}__tracks.csv", numeric={"seg_id", "n_grid", "n_meas"},
                       usecols=["seg_id", "sensor_id", "object_id", "n_grid", "n_meas"])
    z = np.load(TRACKS_DIR / f"{slug}__grid.npz")
    gseg, gstep = z["seg_id"].astype(np.int64), z["step"].astype(np.int64)
    sid = t["seg_id"].astype(np.int64)
    sens_of = dict(zip(sid.tolist(), t["sensor_id"].tolist()))
    gsens = np.array([sens_of[s] for s in np.unique(gseg).tolist()], dtype=object)
    useg = np.unique(gseg)
    row_sens = gsens[np.searchsorted(useg, gseg)].astype(str)
    others = sorted(_DEP0 - set(drop))
    m_other = np.isin(row_sens, others)
    oth_steps = np.unique(gstep[m_other])
    out = []
    sf = C.sensor_files(it)
    for s in list(drop) + ["+".join(drop)] if len(drop) > 1 else list(drop):
        ids = s.split("+")
        msk_t = np.isin(t["sensor_id"], ids)
        msk_g = np.isin(row_sens, ids)
        steps = np.unique(gstep[msk_g])
        only = np.setdiff1d(steps, oth_steps)
        out.append({
            "iter_key": key, "sensor": s, "n_files": sum(1 for f in sf if f.sensor_id in ids),
            "n_segments": int(msk_t.sum()), "n_segments_grid": int((msk_t & (t["n_grid"] > 0)).sum()),
            "n_objects": len(set(t["object_id"][msk_t].tolist())), "n_meas": int(np.nansum(t["n_meas"][msk_t])),
            "n_template_steps": n_steps, "steps_covered": int(steps.size),
            "coverage": steps.size / n_steps if n_steps else float("nan"),
            "steps_only_dropped": int(only.size), "share_only_dropped": only.size / n_steps if n_steps else float("nan"),
            "steps_other_deployable": int(oth_steps.size)})
    return out


def dependence(key: str, drop: list[str], wl: set) -> dict:
    """What F0_fuse (the shipped files in data/derived/fused/) takes from the dropped sensors."""
    slug = C.iter_slug(key)
    testing = ITERS[key].has_groundtruth
    ft = C.read_columns(FUSED / f"{slug}__tracks.csv", numeric={"track", "n_steps", "reported_f0"},
                        usecols=["track", "sensor_ids", "n_steps", "reported_f0", "serial"]
                        + (["label_drone"] if testing else []))
    mem = C.read_columns(FUSED / f"{slug}__members.csv", numeric={"track", "seg_id"})
    st = C.read_columns(TRACKS_DIR / f"{slug}__tracks.csv", numeric={"seg_id"},
                        usecols=["seg_id", "sensor_id", "serial_top"])
    seg_sensor = dict(zip(st["seg_id"].astype(np.int64).tolist(), st["sensor_id"].tolist()))
    seg_serial = dict(zip(st["seg_id"].astype(np.int64).tolist(),
                          [MM.resolve_serial(s) or "" for s in st["serial_top"]]))
    mt, ms = mem["track"].astype(np.int64), mem["seg_id"].astype(np.int64)
    members: dict = {}
    for a, b in zip(mt.tolist(), ms.tolist()):
        members.setdefault(a, []).append(b)
    dset = set(drop)
    rep = ft["reported_f0"].astype(np.int64) == 1
    n_steps = ft["n_steps"].astype(np.int64)
    ids = [set(s.split("|")) for s in ft["sensor_ids"]]
    contains = np.array([bool(i & dset) for i in ids])
    only = np.array([bool(i) and i <= dset for i in ids])
    serial = np.asarray(ft["serial"]).astype(str)
    friendly = np.array([bool(s) and s in wl for s in serial])
    sbear = serial != ""
    depends = np.zeros(len(serial), dtype=bool)
    for t in np.flatnonzero(rep & sbear).tolist():
        car = [m for m in members.get(int(ft["track"][t]), []) if seg_serial[m] == serial[t]]
        depends[t] = bool(car) and all(seg_sensor[m] in dset for m in car)
    out = {"iter_key": key, "f0_tracks": int(rep.sum()), "f0_rows": int(n_steps[rep].sum()),
           "tracks_with": int((rep & contains).sum()), "rows_with": int(n_steps[rep & contains].sum()),
           "tracks_only": int((rep & only).sum()), "rows_only": int(n_steps[rep & only].sum()),
           "friendly_tracks": int((rep & friendly).sum()), "friendly_with": int((rep & friendly & contains).sum()),
           "friendly_depend": int((rep & friendly & depends).sum()),
           "serial_tracks": int((rep & sbear).sum()), "serial_with": int((rep & sbear & contains).sum()),
           "serial_depend": int((rep & sbear & depends).sum())}
    out["share_rows_with"] = out["rows_with"] / out["f0_rows"] if out["f0_rows"] else float("nan")
    out["share_rows_only"] = out["rows_only"] / out["f0_rows"] if out["f0_rows"] else float("nan")
    if testing:
        ld = ft["label_drone"].astype(np.int64) == 1
        out["drone_tracks"] = int((rep & ld).sum())
        out["drone_tracks_with"] = int((rep & ld & contains).sum())
        out["drone_tracks_only"] = int((rep & ld & only).sum())
    return out


# --------------------------------------------------------------------------- checks


def check_no_dropped(name: str, drop, keys, seg_sensor_by_key) -> dict:
    """Self-check 2 from the WRITTEN files: no member of any track, nor any track's sensor list,
    nor the track table of a reported track carries a dropped sensor."""
    bad_members, bad_lists, n_members, n_tracks = [], [], 0, 0
    for k in keys:
        slug = C.iter_slug(k)
        mem = C.read_columns(OUT / name / f"{slug}__members.csv", numeric={"track", "seg_id"})
        tr = C.read_columns(OUT / name / f"{slug}__tracks.csv", usecols=["sensor_ids"])
        bad_members += [(k,) + x for x in members_sensor_check(mem["seg_id"].astype(np.int64),
                                                              seg_sensor_by_key[k], set(drop))]
        bad_lists += [(k, s) for s in tr["sensor_ids"] if set(s.split("|")) & set(drop)]
        n_members += len(mem["seg_id"])
        n_tracks += len(tr["sensor_ids"])
    return {"n_members": n_members, "n_tracks": n_tracks, "bad_members": bad_members, "bad_lists": bad_lists}


def seg_sensor_map(key: str) -> dict:
    t = C.read_columns(TRACKS_DIR / f"{C.iter_slug(key)}__tracks.csv", numeric={"seg_id"},
                       usecols=["seg_id", "sensor_id"])
    return dict(zip(t["seg_id"].astype(np.int64).tolist(), t["sensor_id"].tolist()))


# --------------------------------------------------------------------------- report


def md_table(header, rows) -> list[str]:
    return FU.md_table(header, rows)


def write_report(path: Path, ctx: dict) -> None:
    name, drop = ctx["name"], ctx["drop"]
    dl = ", ".join(drop) if drop else "(none)"
    L: list[str] = []
    L.append(f"# Ablation `{name}`: F0_fuse without {dl}, removed at the input\n")
    L.append(f"Generated by `tools/ablation.py` in {ctx['secs'] / 60:.1f} min. Scores are the local PROXY of "
             f"`tools/score.py` (primary `{SC.PRIMARY}`: GOSPA cut-off c = 50 m, p = 2, 3-D) on the 6 testing "
             "iterations; the parameters of testing iteration k never saw k (leave-one-iteration-out).\n")
    L.append("## Plain-language glossary\n")
    for t_ in (
        "**ablation**: switching one ingredient off to see how much the result depended on it.",
        "**UTM relay (VKYY, QTVS)**: UTM (drone-traffic management) feeds; in the testing data they carry the "
        "Dronetag ground-truth positions themselves, so testing scores that include them are optimistic.",
        "**removed at the input**: the sensors' tracks are deleted before fusion; the fusion, the existence "
        "models and the noise tables are the ones trained WITH the sensors (the cheap number).",
        "**retrained without**: every upstream model is rebuilt on data without the sensors (the honest number; "
        "separate variant rebuild by the controller).",
        "**segment / object**: one sensor's track (one object id) / the id the sensor itself gave the target.",
        "**time coverage**: share of the template's 100 ms steps with at least one position sample of the sensor.",
        "**F0 thresholds kept (`@F0thr`)**: same tracks, but the existence cut-off TAU and its probability "
        "field are the ones F0 chose with the sensors present, not re-tuned for the smaller sensor set.",
        "**combined / track**: the proxy score = 0.4 track (GOSPA) + 0.2 classification + 0.1 identity + 0.3 "
        "metadata; 'track' alone is the GOSPA part.",
        "**wins / losses**: on how many of the 6 iterations the ablated method scores higher / lower (|diff| < "
        "1e-4 is a tie); the sign test says how unlikely such a split is by luck (6 iterations: best p = 0.031).",
        "**truth mode all / airborne**: 'all' keeps drones parked on the ground in the truth, 'airborne' drops them.",
    ):
        L.append(f"* {t_}")
    L.append("")
    L.append("## Reading\n")
    pk = "+".join(drop)
    pr_ = {p["iter_key"]: p for p in ctx["presence"] if p["sensor"] == pk}
    if drop:
        n_t = sum(1 for k in TESTING if pr_[k]["n_segments"] > 0)
        n_e = sum(1 for k in EVALUATION if pr_[k]["n_segments"] > 0)
        cov_e = [pr_[k]["coverage"] * 100 for k in EVALUATION]
        L.append(f"* {dl} have segments in {n_t} of {len(TESTING)} testing and {n_e} of {len(EVALUATION)} evaluation "
                 f"iterations; evaluation time coverage {min(cov_e):.0f} to {max(cov_e):.0f} % of the template steps.")
        sh = [ctx["dep"][k]["share_rows_with"] * 100 for k in EVALUATION]
        L.append(f"* Share of F0's rows that come from tracks containing them: evaluation {min(sh):.0f} to {max(sh):.0f} %, "
                 f"testing {min(ctx['dep'][k]['share_rows_with'] * 100 for k in TESTING):.0f} to "
                 f"{max(ctx['dep'][k]['share_rows_with'] * 100 for k in TESTING):.0f} %.")
        nt_eval0 = sum(ctx["dep"][k]["f0_tracks"] for k in EVALUATION)
        nt_eval1 = sum(ctx["applied"][k]["n_tracks"] for k in EVALUATION)
        fr0 = sum(ctx["dep"][k]["friendly_tracks"] for k in EVALUATION)
        fr1 = sum(ctx["applied"][k]["file"]["friendly_tracks"] for k in EVALUATION)
        L.append(f"* Evaluation without them: {nt_eval0} F0 tracks become {nt_eval1} (re-tuned) / "
                 f"{sum(ctx['applied'][k]['n_tracks_f0thr'] for k in EVALUATION)} (F0 thresholds); Friendly tracks {fr0} -> {fr1}.")
    for mode in MODES:
        dd = [ctx["scores"][(k, mode, name)]["combined"] - ctx["scores"][(k, mode, "F0_fuse")]["combined"] for k in TESTING]
        w, l_, t_ = CMP.count_wlt(dd)
        L.append(f"* Testing, truth `{mode}`: the ablated method beats F0 on {w}, loses on {l_}, ties on {t_} of {len(TESTING)} "
                 f"iterations (combined). A tie is a difference below 1e-4 (identical rows where no reported F0 track used a dropped sensor).")
    L.append("")
    L.append("## Headline: what removing the sensors costs (testing, per iteration)\n")
    L.append("Combined = the owner's weighted proxy. F0 is the shipped method; the ablated rows use "
             "F0's geometry per fold. Differences are ablated - F0 (negative = the sensors were helping).\n")
    var_order = [(name, "re-tuned (P_FIELD, TAU)"), (name + "@F0thr", "F0 thresholds kept")]
    cmp_rows = []
    for mode in MODES:
        L.append(f"### Truth mode `{mode}`\n")
        rows = []
        for k in TESTING:
            f0 = ctx["scores"][(k, mode, "F0_fuse")]
            a = ctx["scores"][(k, mode, name)]
            b = ctx["scores"][(k, mode, name + "@F0thr")]
            rows.append([FU.short(k), _f(f0["combined"]), _f(a["combined"]), _sf(a["combined"] - f0["combined"]),
                         _f(b["combined"]), _sf(b["combined"] - f0["combined"]),
                         _f(f0["track"]), _f(a["track"]), _sf(a["track"] - f0["track"]),
                         _f(b["track"]), _sf(b["track"] - f0["track"])])
        L += md_table(["iteration", "F0 comb.", f"{name} comb.", "diff", "@F0thr comb.", "diff",
                       "F0 track", f"{name} track", "diff", "@F0thr track", "diff"], rows)
        L.append("")
        for vname, desc in var_order:
            for part in PARTS:
                diffs = [ctx["scores"][(k, mode, vname)][part] - ctx["scores"][(k, mode, "F0_fuse")][part] for k in TESTING]
                w, l_, t_ = CMP.count_wlt(diffs)
                cmp_rows.append([mode, vname, part, f"{w}", f"{l_}", f"{t_}", f"{CMP.sign_test_p(w, l_):.3f}",
                                 _sf(float(np.nanmean([ctx['scores'][(k, mode, vname)][part] for k in TESTING])), 3),
                                 _sf(float(np.nanmean([ctx['scores'][(k, mode, 'F0_fuse')][part] for k in TESTING])), 3)])
        L.append("Parts per iteration (F0 -> ablated re-tuned -> ablated with F0 thresholds):\n")
        rows = []
        for k in TESTING:
            r = [FU.short(k)]
            for part in PARTS[2:]:
                vals = [ctx["scores"][(k, mode, v)][part] for v in ("F0_fuse", name, name + "@F0thr")]
                r.append(" / ".join(_f(x) for x in vals))
            rows.append(r)
        L += md_table(["iteration", "classification", "identity", "metadata"], rows)
        L.append("")
    L.append("### Wins and losses of the ablated method against F0 (per iteration, 6 testing iterations)\n")
    L += md_table(["truth", "variant", "part", "wins", "losses", "ties", "sign-test p", "mean ablated", "mean F0"], cmp_rows)
    L.append("\n(Means are context only; the count is the verdict, as in `tools/compare.py`.)\n")
    L.append("### What was chosen\n")
    rows = []
    for k in TESTING + EVALUATION:
        a, f0 = ctx["applied"][k], ctx["rules"][rule_key(k)]
        rows.append([FU.short(k), "/".join(str(x) for x in f0[:3]), f"{f0[3]} {f0[4]}",
                     f"{a['combo'][3]} {a['combo'][4]}", a["n_tracks"], a["n_tracks_f0thr"], a["file"]["rows"],
                     ctx["f0_counts"][k]["f0_tracks"], ctx["f0_counts"][k]["f0_rows"]])
    L += md_table(["iteration", "geometry (edge/cont/bridge) = F0's", "F0 (p_field tau)", "re-tuned (p_field tau)",
                   "tracks reported (re-tuned)", "tracks (F0 thr)", "rows (re-tuned)", "F0 tracks", "F0 rows"], rows)
    L.append("")
    L.append("## Presence of the dropped sensors, every iteration\n")
    L.append("From the raw sensor files and `sensor_tracks_v2` (not from the fusion). Coverage = share of the "
             "template's 100 ms steps with at least one grid sample of the sensor; 'only dropped' = steps where "
             "the dropped sensors are the only deployable sensors with a sample.\n")
    rows = [[FU.short(p["iter_key"]), ("testing" if ITERS[p["iter_key"]].has_groundtruth else "evaluation"),
             p["sensor"], p["n_files"], p["n_segments"], p["n_segments_grid"], p["n_objects"],
             f"{p['coverage'] * 100:.1f} %", f"{p['share_only_dropped'] * 100:.1f} %"] for p in ctx["presence"]]
    L += md_table(["iteration", "tree", "sensor", "files", "segments", "with grid", "objects", "time coverage",
                   "steps only dropped"], rows)
    L.append("")
    L.append("## What F0_fuse takes from them, every iteration\n")
    L.append("From the shipped F0 outputs (`fused/<iteration>__tracks.csv`, `__members.csv`). 'with' = the "
             "reported track has a dropped sensor among its members; 'only' = made of dropped sensors alone; "
             "'depend' = every member that carries the track's serial is a dropped sensor (the serial, hence "
             "'Friendly' and the serial/model metadata, would be lost).\n")
    rows = []
    for k in TESTING + EVALUATION:
        d = ctx["dep"][k]
        rows.append([FU.short(k), d["f0_tracks"], d["f0_rows"], f"{d['tracks_with']} ({d['share_rows_with'] * 100:.0f} % rows)",
                     f"{d['tracks_only']} ({d['share_rows_only'] * 100:.1f} % rows)",
                     f"{d['friendly_tracks']} / {d['friendly_with']} / {d['friendly_depend']}",
                     f"{d['serial_tracks']} / {d['serial_with']} / {d['serial_depend']}"])
    L += md_table(["iteration", "F0 tracks", "F0 rows", "tracks with them", "tracks only them",
                   "Friendly: total / with / depend", "serial-bearing: total / with / depend"], rows)
    L.append("")
    tdrone = [(FU.short(k), ctx["dep"][k]["drone_tracks"], ctx["dep"][k]["drone_tracks_with"],
               ctx["dep"][k]["drone_tracks_only"]) for k in TESTING]
    L.append("Testing, F0 tracks that are real drones (ground-truth label, diagnostics only): total / with them / only them:\n")
    L += md_table(["iteration", "drone tracks", "with", "only"], tdrone)
    L.append("")
    L.append("After the ablation (the re-tuned selection): Friendly / serial-bearing / model-bearing reported tracks, "
             "against F0's.\n")
    rows = []
    for k in TESTING + EVALUATION:
        f, d = ctx["applied"][k]["file"], ctx["dep"][k]
        rows.append([FU.short(k), f"{d['friendly_tracks']} -> {f['friendly_tracks']}",
                     f"{d['serial_tracks']} -> {f['serial_tracks']}", f["model_tracks"]])
    L += md_table(["iteration", "Friendly tracks F0 -> ablated", "serial tracks F0 -> ablated", "model tracks ablated"], rows)
    L.append("")
    L.append("## The honest retrained-without number (slot)\n")
    if ctx.get("retrained"):
        L.append(f"Method `{ctx['retrained']}` (promoted variant rebuild, retrained without the sensors) against F0 and the "
                 "input-removal rows above (combined, per iteration):\n")
        rows = []
        for mode in MODES:
            for k in TESTING:
                f0, a = ctx["scores"][(k, mode, "F0_fuse")], ctx["scores"][(k, mode, name)]
                r = ctx["scores"].get((k, mode, ctx["retrained"]))
                rows.append([mode, FU.short(k), _f(f0["combined"]), _f(a["combined"]),
                             _f(r["combined"]) if r else "n/a", _sf(r["combined"] - f0["combined"]) if r else "n/a"])
        L += md_table(["truth", "iteration", "F0", f"{name} (input)", f"{ctx['retrained']} (retrained)", "retrained - F0"], rows)
        L.append("")
    else:
        L.append("NOT FILLED YET. Build it with the sensor_tracks `drop_sensors` option in a variant tree "
                 "(`tools/run.py --derived data/derived_<tree> ...`, then `--promote <suffix>`), then either run "
                 "`python3 tools/compare.py F0_fuse <promoted method> --truth both` or re-run this tool with "
                 "`--retrained <promoted method>` to put it in the table above.\n")
    L.append("**Why the two numbers differ.** This report removes the sensors from the fusion input only. "
             "The existence models (which decide what is reported), the sensor-noise table, the height offsets and "
             "the segment cut were built WITH the sensors in the data; in testing the UTM relays carry the Dronetag "
             "feed, i.e. near-perfect positives, so they partly taught the models what a drone looks like and what "
             "the other sensors' errors are. Removing the sensors afterwards keeps those models. The retrained number "
             "rebuilds all of that without the sensors and re-runs the LOIO tuning on the real remaining sensor set; "
             "it is the one to quote as 'the system without UTM'. The direction of the gap between the two is not "
             "known in advance.\n")
    L.append("## Self-checks\n")
    for i, (nm, ok, detail) in enumerate(ctx["checks"], 1):
        L.append(f"{i}. {'PASS' if ok else 'FAIL'}: {nm}" + ("" if ok or not detail else f" -- {detail}"))
    L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--drop", default=",".join(DEFAULT_DROP),
                    help="comma-separated sensor ids to remove at the input ('' = none = the F0 reproduction)")
    ap.add_argument("--name", default=None, help="method name (default F0_noutm_input for VKYY,QTVS)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--retrained", default=None, help="promoted method of the retrained variant rebuild (adds a table)")
    ap.add_argument("--skip-score-log", action="store_true", help="do not append to experiments.csv")
    args = ap.parse_args()
    t0 = time.time()
    drop = sorted({s.strip() for s in args.drop.split(",") if s.strip()})
    args.workers = max(1, min(args.workers, 3))
    name = args.name or ("F0_noutm_input" if drop == sorted(DEFAULT_DROP)
                         else ("F0_drop_" + "_".join(drop) if drop else "F0_drop_none"))
    if name in PROTECTED or name.startswith(("B0_", "B1_", "B2_", "O1_", "O2_", "O3_")):
        print(f"refusing to overwrite protected method {name!r}")
        return 2
    exist = C.SUBMISSIONS / name
    if exist.exists() and not (OUT / f"{name}.md").exists():
        print(f"refusing: data/derived/submissions/{name} exists and was not written by this tool")
        return 2
    known = set()
    for it in ITERS.values():
        known |= {sf.sensor_id for sf in C.sensor_files(it)}
    unknown = [s for s in drop if s not in known]
    inactive = [s for s in drop if s in known and s not in _DEP0]
    if unknown:
        log(f"warning: no such sensor id in any iteration: {unknown}")
    if inactive:
        log(f"note: testing-only sensors are already excluded from every F0 output: {inactive}")
    DROP.update(drop)
    OUT.mkdir(parents=True, exist_ok=True)
    ctx0 = mp.get_context("fork")
    log(f"ablation {name}: drop {drop}; deployable now {sorted(C.deployable_sensor_ids())}")

    rules = f0_rules()
    wl = B.load_whitelist()
    all_keys = TESTING + EVALUATION

    # ---- presence + F0 dependence (cheap, main process)
    pres, dep_ = [], {}
    for k in all_keys:
        if drop:
            pres += presence(k, drop)
        dep_[k] = dependence(k, drop, wl)
    log(f"presence done ({len(pres)} rows) {time.time() - t0:.0f}s")

    # ---- phase 1: re-tuned (P_FIELD, TAU) on this sensor set + the F0 reproduction
    jobs = []
    for j in TESTING:
        srcgeo = {k: tuple(rules[k][:3]) for k in TESTING if k != j}
        srcgeo["outer"] = tuple(rules["outer"][:3])
        jobs.append(("tune", j, srcgeo))
    repro_keys = sorted(TESTING, key=lambda k: C.read_columns(TRACKS_DIR / f"{C.iter_slug(k)}__tracks.csv",
                                                              numeric={"n_grid"}, usecols=["n_grid"])["n_grid"].sum())[:2]
    jobs += [("repro", k, rules[rule_key(k)]) for k in repro_keys]
    log(f"phase 1: {len(jobs)} jobs ({len(TESTING)} tuning, {len(repro_keys)} F0 reproduction), {args.workers} workers")
    records, repro = [], []
    with ctx0.Pool(args.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(_job, jobs), 1):
            if r["kind"] == "tune":
                records += r["records"]
                log(f"  [{n}/{len(jobs)}] tune {r['key']}: {len(r['records'])} records {r['seconds']:.0f}s")
            else:
                repro.append(r)
                log(f"  [{n}/{len(jobs)}] repro {r['key']}: grid cols differing {r['n_cols_bad']}, sel_bad {r['sel_bad']}, "
                    f"rows {r['n_rows']}/{r['n_back']}, rt {r['rt_err_m']:.1e} m, {r['seconds']:.0f}s")
    records.sort(key=lambda r: (r["k"], r["j"]) + FU.combo_of(r))
    C.write_columns(OUT / f"{name}_tuning.csv", {c: (np.array([r[c] for r in records], dtype=float)
                                                     if isinstance(records[0][c], float) else [r[c] for r in records])
                                                 for c in TUNING_COLS}, TUNING_COLS)
    choices, used_by = {}, {}
    for k in TESTING:
        with geometry_grid(rules[k][:3]):
            cb, v, used = FU.choose(records, k, [j for j in TESTING if j != k])
        choices[k], used_by[k] = (cb, v), used
    with geometry_grid(rules["outer"][:3]):
        fcb, fv, fused_used = FU.choose(records, "outer", TESTING)
    final_choice = (fcb, fv)
    for k in TESTING:
        log(f"  fold {k}: F0 {rules[k][3:]} -> {choices[k][0][3:]} (objective {choices[k][1]:.4f})")
    log(f"  evaluation: F0 {rules['outer'][3:]} -> {fcb[3:]} (objective {fv:.4f})")

    # ---- phase 2: apply, write files
    t1 = time.time()
    ajobs = [("apply", k, name, choices[k][0], rules[k]) for k in TESTING] + \
            [("apply", k, name, fcb, rules["outer"]) for k in EVALUATION]
    applied = {}
    with ctx0.Pool(args.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(_job, ajobs), 1):
            applied[r["key"]] = r
            f = r["file"]
            log(f"  [{n}/{len(ajobs)}] apply {r['key']}: {r['n_tracks_all']} tracks, reported {f['tracks']} / {f['rows']:,} rows, "
                f"problems {len(f['problems'])}, {r['seconds']:.0f}s")

    # ---- scores
    scores: dict = {}
    for k in TESTING:
        it = ITERS[k]
        for mode in MODES:
            r0 = SC.score_file(it, C.submission_path("F0_fuse", it), mode)
            scores[(k, mode, "F0_fuse")] = dict(r0["parts"], n_tracks=r0["n_tracks"], n_rows=r0["n_est_rows"])
            sp = applied[k]["f0thr_parts"][mode]
            scores[(k, mode, name + "@F0thr")] = dict(sp, n_tracks=applied[k]["n_tracks_f0thr"],
                                                       n_rows=applied[k]["n_rows_f0thr"])
    note = f"ablation drop={','.join(drop) or 'none'}: F0 with the sensors removed at the input (trained models kept), P_FIELD/TAU re-tuned LOIO"
    results = {}
    for mode in MODES:
        results[mode] = SC.score_method(name, mode, iters=[ITERS[k] for k in TESTING], log=not args.skip_score_log,
                                        note=note)
        for r in results[mode]:
            scores[(r["iter_key"], mode, name)] = dict(r["parts"], n_tracks=r["n_tracks"], n_rows=r["n_est_rows"])
    if args.retrained:
        for mode in MODES:
            for k in TESTING:
                p = C.submission_path(args.retrained, ITERS[k])
                if p.exists():
                    r = SC.score_file(ITERS[k], p, mode)
                    scores[(k, mode, args.retrained)] = dict(r["parts"], n_tracks=r["n_tracks"], n_rows=r["n_est_rows"])

    # ---- csv outputs
    srows = []
    for (k, mode, v), s in sorted(scores.items()):
        a = applied[k]
        combo = a["combo"] if v == name else a["combo_f0"] if v in ("F0_fuse", name + "@F0thr") else ("", "", "", "", "")
        srows.append([k, mode, v] + [s[p] for p in PARTS] + [s["n_tracks"], s["n_rows"]] + list(combo))
    cols = ["iter_key", "truth_mode", "variant"] + list(PARTS) + ["n_tracks", "n_rows", "edge_gate", "cont_gap_s",
                                                                  "bridge_s", "p_field", "tau"]
    C.write_columns(OUT / f"{name}.csv", {c: [r[i] for r in srows] for i, c in enumerate(cols)}, cols)
    pc = ["iter_key", "sensor", "n_files", "n_segments", "n_segments_grid", "n_objects", "n_meas", "n_template_steps",
          "steps_covered", "coverage", "steps_only_dropped", "share_only_dropped", "steps_other_deployable"]
    if pres:
        C.write_columns(OUT / f"{name}_presence.csv", {c: [p[c] for p in pres] for c in pc}, pc)
    dc = list(dep_[all_keys[0]])
    for d in dep_.values():
        for c in d:
            if c not in dc:
                dc.append(c)
    C.write_columns(OUT / f"{name}_dependence.csv", {c: [dep_[k].get(c, "") for k in all_keys] for c in dc}, dc)

    # ---- checks
    checks: list[tuple[str, bool, str]] = []
    bad1 = [r for r in repro if r["n_cols_bad"] or r["sel_bad"] != 0 or not r["row_ok"] or not r["lab_ok"]
            or not (r["rt_err_m"] < RT_TOL_M) or not r["mutation_detected"]]
    checks.append((f"with NO dropped sensor the same pipeline reproduces F0_fuse on {len(repro)} iterations "
                   f"({', '.join(r['key'] for r in repro)}): fused grid exact (every saved column), reported-track set = "
                   f"reported_f0, {sum(r['n_rows'] for r in repro):,} rows = the F0_fuse submission (step, track_id, labels "
                   f"exact; positions max {max((r['rt_err_m'] for r in repro), default=float('nan')):.1e} m < {RT_TOL_M}); "
                   "a mutated grid is detected",
                   len(repro) >= 2 and not bad1, str(bad1)[:300]))
    seg_sensor_by_key = {k: seg_sensor_map(k) for k in all_keys}
    c2 = check_no_dropped(name, drop, all_keys, seg_sensor_by_key)
    # mutation: a track member from a dropped sensor must be caught (use VKYY if nothing is dropped)
    probe = set(drop) or {"VKYY"}
    mk = next(k for k in all_keys if any(s in probe for s in seg_sensor_by_key[k].values()))
    victim = next(s for s, v in seg_sensor_by_key[mk].items() if v in probe)
    caught = members_sensor_check(np.array([victim]), seg_sensor_by_key[mk], probe)
    n_dep_drop = sum(a["n_dropped_in_dep"] for a in applied.values())
    n_dropped_seg = sum(a["n_dropped_segments"] for a in applied.values())
    checks.append((f"no dropped sensor in any member of any track: {c2['n_members']:,} members of {c2['n_tracks']:,} tracks "
                   f"(written members.csv re-read and mapped to sensors through sensor_tracks_v2) and their sensor lists; "
                   f"{n_dropped_seg:,} dropped-sensor segments exist in the 9 iterations and {n_dep_drop} remained deployable "
                   f"in the Prep; non-deployable members found by fuse.check_constraints: "
                   f"{sum(a['c1']['non_deployable'] for a in applied.values())}; a planted member is caught: {bool(caught)}",
                   not c2["bad_members"] and not c2["bad_lists"] and n_dep_drop == 0 and bool(caught)
                   and all(a["c1"]["non_deployable"] == 0 for a in applied.values())
                   and (not drop or n_dropped_seg > 0),
                   str((c2["bad_members"][:3], c2["bad_lists"][:3]))[:300]))
    files = [(k, applied[k]["file"]) for k in all_keys]
    revalid = {k: S.validate_submission(ITERS[k], C.submission_path(name, ITERS[k])) for k in all_keys}
    bad3 = [(k, f["problems"][:2], f["rt_err_m"], f["unique"], f["rt_labels_ok"], f["n_back"], f["rows"])
            for k, f in files if f["problems"] or not f["unique"] or not (f["rt_err_m"] < RT_TOL_M)
            or f["rt_labels_ok"] is not True or f["n_back"] != f["rows"]]
    bad3 += [(k, p[:2]) for k, p in revalid.items() if p]
    fdiff = max(max(abs(applied[k]["mem_parts"][m]["combined"] - scores[(k, m, name)]["combined"]) for m in MODES)
                for k in TESTING)
    checks.append((f"{len(files)} files written (9 iterations), validate -> [] (twice: at write and re-validated from disk), "
                   f"(step, track_id) unique, round trip e/n/u max {max(f['rt_err_m'] for _, f in files):.1e} m < {RT_TOL_M}, "
                   f"labels identical; in-memory score before the write = score of the file (max diff {fdiff:.1e} < 1e-3)",
                   not bad3 and fdiff < 1e-3, str(bad3)[:300]))
    # leakage guard (as in fuse), plus the geometry being F0's
    leak = []
    for k in TESTING:
        js = {r["j"] for r in used_by[k]}
        if k in js or js != set(TESTING) - {k}:
            leak.append(f"{k}: objective used iterations {sorted(js)}")
        if any(r["k"] != k or r["p_fold"] != k for r in used_by[k]):
            leak.append(f"{k}: a score used p not from p_inner outer_fold == k")
        if any(tuple(r[c] for c in ("edge_gate", "cont_gap_s", "bridge_s")) != tuple(rules[k][:3]) for r in used_by[k]):
            leak.append(f"{k}: tuning used a geometry other than F0's fold-{k} geometry")
        if applied[k]["p_fold"] != k:
            leak.append(f"{k}: applied with p fold {applied[k]['p_fold']}")
        if tuple(applied[k]["combo"]) != tuple(choices[k][0]) or tuple(applied[k]["combo"][:3]) != tuple(rules[k][:3]):
            leak.append(f"{k}: applied combo differs from the fold choice")
    for k in EVALUATION:
        if applied[k]["p_fold"] != "final" or tuple(applied[k]["combo"]) != tuple(fcb):
            leak.append(f"{k}: evaluation p fold {applied[k]['p_fold']} / combo {applied[k]['combo']}")
    if tuple(fcb[:3]) != tuple(rules["outer"][:3]):
        leak.append("evaluation geometry is not F0's final geometry")
    if any(r["k"] != "outer" for r in fused_used) or {r["j"] for r in fused_used} != set(TESTING):
        leak.append("evaluation choice used non-outer scores")
    dep_now = C.deployable_sensor_ids()
    for k, a in applied.items():
        if not set(a["file"]["sensors"]) <= dep_now:
            leak.append(f"{k}: tracks use non-deployable sensors {sorted(set(a['file']['sensors']) - dep_now)}")
    tb = C.read_columns(OUT / f"{name}_tuning.csv", numeric={"cont_gap_s", "bridge_s", "tau", "combined_all",
                                                              "combined_airborne"})
    for k in TESTING + ["outer"]:
        acc: dict = {}
        for i in range(len(tb["k"])):
            src = tb["k"][i]
            if (k != "outer" and (src != k or tb["j"][i] == k)) or (k == "outer" and src != "outer"):
                continue
            cb = (tb["edge_gate"][i], float(tb["cont_gap_s"][i]), float(tb["bridge_s"][i]), tb["p_field"][i], float(tb["tau"][i]))
            acc.setdefault(cb, []).append((tb["combined_all"][i] + tb["combined_airborne"][i]) / 2)
        want = 5 if k != "outer" else 6
        bestv = max(sum(v) / len(v) for v in acc.values() if len(v) == want)
        mine = choices[k] if k != "outer" else final_choice
        cbm = tuple(float(x) if not isinstance(x, str) else x for x in mine[0])
        if abs(sum(acc[cbm]) / len(acc[cbm]) - bestv) > 1e-12:
            leak.append(f"{k}: choice is not the argmax of the written tuning table")
    checks.append(("leakage guard: fold k's (P_FIELD, TAU) uses only j != k scored with p_inner (outer_fold == k) on F0's fold-k "
                   "geometry and is applied with k's outer p; evaluation uses the all-6 outer-p choice on F0's final geometry "
                   "and the final model's p; no testing-only or dropped sensor in any track; choices re-derived from the "
                   "written tuning table", not leak, "; ".join(leak[:4])))
    # presence cross-check (raw files vs Prep arrays) + F0 dependence sanity
    cross = []
    for p in pres:
        if "+" in p["sensor"]:
            continue
        a = applied[p["iter_key"]]
        if a["cov_iter"][p["sensor"]] != p["steps_covered"]:
            cross.append((p["iter_key"], p["sensor"], a["cov_iter"][p["sensor"]], p["steps_covered"]))
    tot_with = sum(d["tracks_with"] for d in dep_.values())
    checks.append((f"presence: time coverage of every (iteration, sensor) from the raw grid files = coverage from fuse's Iter arrays "
                   f"({len(pres)} rows); F0 dependence counts consistent (only <= with <= total on all 9 iterations)",
                   not cross and all(d["tracks_only"] <= d["tracks_with"] <= d["f0_tracks"]
                                     and d["friendly_depend"] <= d["friendly_with"] <= d["friendly_tracks"]
                                     and d["serial_depend"] <= d["serial_with"] <= d["serial_tracks"] for d in dep_.values()),
                   str(cross[:3])))
    secs = time.time() - t0
    checks.append((f"runtime {secs / 60:.1f} min < {RUNTIME_LIMIT_S / 60:.0f} min", secs < RUNTIME_LIMIT_S, ""))

    ctx = {"name": name, "drop": drop, "scores": scores, "applied": applied, "rules": rules, "presence": pres, "dep": dep_,
           "f0_counts": {k: dep_[k] for k in all_keys}, "checks": checks, "secs": secs, "retrained": args.retrained}
    write_report(OUT / f"{name}.md", ctx)
    log(f"wrote {(OUT / f'{name}.md').relative_to(C.REPO)}")
    n_fail = 0
    for i, (nm, ok, detail) in enumerate(checks, 1):
        print(f"  [{'PASS' if ok else 'FAIL'}] ({i}) {nm}" + ("" if ok or not detail else f"\n         {detail}"))
        n_fail += not ok
    for mode in MODES:
        s0 = np.nanmean([scores[(k, mode, 'F0_fuse')]['combined'] for k in TESTING])
        s1 = np.nanmean([scores[(k, mode, name)]['combined'] for k in TESTING])
        s2 = np.nanmean([scores[(k, mode, name + '@F0thr')]['combined'] for k in TESTING])
        diffs = [scores[(k, mode, name)]['combined'] - scores[(k, mode, 'F0_fuse')]['combined'] for k in TESTING]
        w, l_, t_ = CMP.count_wlt(diffs)
        log(f"  [{mode}] mean combined F0 {s0:.4f} | {name} {s1:.4f} | @F0thr {s2:.4f} | ablated beats F0 on {w}, "
            f"loses on {l_}, ties {t_} of {len(TESTING)}")
    if n_fail:
        print("SELF-CHECK FAILED")
        return 1
    print("ALL SELF-CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
