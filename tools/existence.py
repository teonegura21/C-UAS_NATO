#!/usr/bin/env python3
"""Track existence: a calibrated probability that a sensor track segment is a real drone
(P_drone), validated leave-one-iteration-out (LOIO).

This is the 'is it a real drone?' filter in front of the tracker: every false track costs
GOSPA as much as a missed drone, and ~98 % of the testing segments are clutter (mostly radar).

Inputs and rules
----------------
* Segments: <DIR>/<slug>__tracks.csv of all 9 iterations (tools/sensor_tracks.py; schemas in
  its docstring), DIR = --tracks (default data/derived/sensor_tracks_v2).
* Targets (testing labels of sensor_tracks): y_all = label in {drone_air, drone_ground}
  (truth mode 'all'), y_air = label == drone_air (truth mode 'airborne'). Both are fitted.
* Features: FEATURES below = the numeric segment columns of tracks.csv minus identifiers
  (seg_id, part, object_id ...), times (t0, t1 and the grid-relative t0_rel_s / t1_rel_s, which
  only encode when in the scenario a segment happened), the string columns (noise_src,
  model_top, serial_top, protocol_top) and everything derived from ground truth (n_matched,
  frac_*, best_flight, label); plus a one-hot of sensor_id over every sensor id that has a
  segment in any iteration (sensor_<ID>). gt_feed and the has_* metadata flags are kept.
* Model: gbdt.GBDT (logistic loss), calibrated with gbdt.Isotonic.
  Search (tiny, fixed): CONFIGS (3 tree settings) x WEIGHTS ('unit' = 1 per segment,
  'grid' = n_grid, the segment's grid samples = its GOSPA exposure, normalised to mean 1 over
  the training rows so l2 keeps its scale) = 6 combinations. The selection metric is the
  step-weighted logloss (weights n_grid) of the raw GBDT output on the deployable sensors.
* Validation, nested LOIO over the 6 testing iterations. For held-out iteration k:
    - inner LOIO over the other 5: for every j != k a model fitted on testing \\ {k, j}
      predicts j (one 'pair' fit per unordered pair {k, j}, shared with fold j);
    - the combination with the lowest mean inner metric is chosen FOR FOLD k (the outer fold
      k is never looked at);
    - the chosen combination is refitted on testing \\ {k} and predicts k;
    - its isotonic calibrator is fitted on the pooled inner out-of-fold predictions of the
      chosen combination on the 5 training iterations (weights as the combination's), and
      applied to k.
  Final model (evaluation iterations): the combination with the lowest mean 6-fold LOIO metric
  (the outer folds above, which is legitimate here since no evaluation label exists), fitted
  on all 6 testing iterations, calibrated on its 6-fold OOF predictions. Nothing is fitted,
  tuned or calibrated on the evaluation tree.
* Naive baseline: P = the training share of drone segments of the segment's sensor id,
  (positives + 0.5) / (segments + 1) (Jeffreys, so a sensor without drones is not P = 0);
  a sensor with no training segment gets the overall training share.
* Headline metrics are on the DEPLOYABLE sensors (C.deployable_sensor_ids(): testing-only
  EUIG, LJJX, MMJJ stay in training but out of the numbers); 'all sensors' is a side note.
  'segment-weighted' = every segment counts 1; 'step-weighted' = weight n_grid, i.e. what a
  P > cut filter does to the scored grid samples.

Outputs (data/derived/existence/)
---------------------------------
<slug>__p.csv     one row per segment of the iteration (seg_id ascending):
                  seg_id, p_drone_all, p_drone_air (calibrated, [0, 1]),
                  fold = the held-out iteration key (testing: LOIO prediction) or 'final'
                  (evaluation: final model)
<slug>__p_inner.csv   testing only: the nested predictions for downstream LOIO tuning. For
                  every outer fold k != this iteration j: seg_id, outer_fold = k,
                  p_drone_all, p_drone_air = fold k's isotonic calibrator applied to the
                  model fitted on testing \\ {k, j}. No label of k or j entered it, and it is on
                  the same scale as k's own p, so a threshold for k can be tuned on these.
model_all.json, model_air.json   final model: {target, features, sensor_ids, combo, config,
                  weight, train_iters, gbdt (GBDT.to_dict), isotonic (Isotonic.to_dict)}
existence.md      per held-out iteration metrics model vs naive, per-sensor and within-sensor
                  AUC, calibration table + ECE / worst bin, feature importance, the P > 0.5 cut
                  (ties at the cut counted separately) and which sensors are above it,
                  plain-language glossary

Run: python3 tools/existence.py [--tracks DIR] [--jobs N]
Numbered self-checks at the end; exit status 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuas_common as C  # noqa: E402
import gbdt as GB  # noqa: E402

OUT = C.DERIVED / "existence"
TRACKS_DEFAULT = C.DERIVED / "sensor_tracks_v2"

FEATURES = ["n_meas", "n_rows", "n_dup_merged", "has_u", "n_grid", "duration_s", "meas_rate_hz",
            "path_len_m", "speed_mean_ms", "speed_p90_ms", "speed_max_ms", "accel_p90_ms2",
            "turn_p90_degs", "stationary_share", "dz_med_m", "dz_p10_m", "dz_p90_m", "dz_std_m",
            "u_med_m", "u_std_m", "range_med_m", "range_min_m", "range_max_m", "det_conf_mean",
            "cls_uav", "cls_uav_rotary", "cls_uav_fixed", "cls_uav_transmitter",
            "cls_air_vehicle", "cls_bird", "cls_manned", "cls_ground", "cls_unknown",
            "cls_blank", "cls_conf_mean", "has_model", "has_serial", "has_protocol", "has_freq",
            "gt_feed"]
EXCLUDED = ["seg_id", "iter_key", "sensor_key", "sensor_id", "object_id", "part", "t0", "t1",
            "t0_rel_s", "t1_rel_s", "noise_src", "model_top", "serial_top", "protocol_top",
            "n_matched", "frac_matched", "best_flight", "frac_best", "frac_airborne", "label"]
TARGETS = {"all": ("drone_air", "drone_ground"), "air": ("drone_air",)}
CONFIGS = {
    "d3": dict(n_trees=200, learning_rate=0.05, max_depth=3, min_samples_leaf=20, seed=0),
    "d5": dict(n_trees=200, learning_rate=0.05, max_depth=5, min_samples_leaf=20, seed=0),
    "d4s": dict(n_trees=300, learning_rate=0.03, max_depth=4, min_samples_leaf=40,
                subsample=0.8, colsample=0.8, seed=0),
}
WEIGHTS = ("unit", "grid")
COMBOS = [(c, w) for c in CONFIGS for w in WEIGHTS]
CUT = 0.5
CAL_TOL = 0.05
JEFFREYS = 0.5
TIE_EPS = 1e-9      # isotonic plateaus sit exactly on round values (0.5): |P - cut| <= this is a tie
BIN_MIN = 20        # calibration bins with fewer segments are not used for the per-bin error

# filled by load() before the worker pool forks
D: dict = {}


def log(msg: str) -> None:
    print(msg, flush=True)


def md_table(header, rows) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def combo_name(c: tuple[str, str]) -> str:
    return f"{c[0]}/{c[1]}"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def load(tracks_dir: Path) -> None:
    iters = C.iterations()
    keys, parts = [], []
    for it in iters:
        path = tracks_dir / f"{C.iter_slug(it.key)}__tracks.csv"
        if not path.exists():
            raise SystemExit(f"missing {path} (run tools/sensor_tracks.py)")
        d = C.read_columns(path, numeric=set(FEATURES) | {"seg_id"},
                           usecols=["seg_id", "iter_key", "sensor_id", "label"] + FEATURES)
        if not np.all(d["iter_key"] == it.key):
            raise SystemExit(f"{path}: iter_key column != {it.key}")
        if not np.all(np.diff(d["seg_id"]) > 0):
            raise SystemExit(f"{path}: seg_id not strictly ascending")
        keys.append(it.key)
        parts.append(d)
    cat = {c: np.concatenate([p[c] for p in parts]) for c in parts[0]}
    sensor_ids = sorted(set(cat["sensor_id"].tolist()))
    onehot = np.stack([(cat["sensor_id"] == s).astype(float) for s in sensor_ids], axis=1)
    X = np.column_stack([cat[c] for c in FEATURES] + [onehot])
    if np.isinf(X).any():
        raise SystemExit("infinite feature values")
    label = cat["label"].astype(str)
    is_test = np.char.startswith(cat["iter_key"].astype(str), "testing/")
    if np.any(label[~is_test] != "") or np.any(~np.isin(label[is_test],
                                                        ["drone_air", "drone_ground", "clutter"])):
        raise SystemExit("unexpected labels (evaluation must be blank, testing in the 3 classes)")
    D.update(
        keys=keys, test_keys=[k for k in keys if k.startswith("testing/")],
        eval_keys=[k for k in keys if k.startswith("evaluation/")],
        iter_key=cat["iter_key"].astype(str), seg_id=cat["seg_id"].astype(np.int64),
        sensor_id=cat["sensor_id"].astype(str), label=label, is_test=is_test, X=X,
        n_grid=cat["n_grid"].astype(float), sensor_ids=sensor_ids,
        names=FEATURES + [f"sensor_{s}" for s in sensor_ids],
        deploy=np.isin(cat["sensor_id"], sorted(C.deployable_sensor_ids())),
        y={t: np.isin(label, list(v)).astype(float) for t, v in TARGETS.items()},
    )
    D["rows"] = {k: np.flatnonzero(D["iter_key"] == k) for k in keys}


def train_rows(train_keys) -> np.ndarray:
    return np.concatenate([D["rows"][k] for k in sorted(train_keys)])


def weights(rows: np.ndarray, scheme: str) -> np.ndarray:
    if scheme == "unit":
        return np.ones(rows.size)
    g = D["n_grid"][rows]
    return g / g.mean()


# --------------------------------------------------------------------------- #
# Fits (run in worker processes; D is inherited through fork)
# --------------------------------------------------------------------------- #


def fit_task(task: tuple) -> dict:
    """task = (target, combo, train_keys, predict_keys). Returns raw predictions per predicted
    iteration and the iteration keys of the rows the model was actually fitted on."""
    target, combo, train_keys, pred_keys = task
    rows = train_rows(train_keys)
    m = GB.GBDT(**CONFIGS[combo[0]]).fit(D["X"][rows], D["y"][target][rows],
                                         sample_weight=weights(rows, combo[1]))
    return {"task": task, "fitted_on": tuple(sorted(set(D["iter_key"][rows].tolist()))),
            "pred": {k: m.predict_proba(D["X"][D["rows"][k]]) for k in pred_keys},
            "model": m.to_dict() if set(train_keys) == set(D["test_keys"]) else None}


def run_tasks(tasks: list[tuple], jobs: int) -> dict:
    if jobs <= 1:
        res = [fit_task(t) for t in tasks]
    else:
        with mp.get_context("fork").Pool(jobs) as pool:
            res = pool.map(fit_task, tasks, chunksize=1)
    return {r["task"]: r for r in res}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def metrics(y: np.ndarray, p: np.ndarray, g: np.ndarray) -> dict:
    out = {}
    for tag, w in (("seg", None), ("step", g)):
        ok = y.size > 0 and (w is None or w.sum() > 0)
        out[f"auc_{tag}"] = GB.auc(y, p, w) if ok else np.nan
        out[f"ll_{tag}"] = GB.logloss(y, p, w) if ok else np.nan
        out[f"brier_{tag}"] = GB.brier(y, p, w) if ok else np.nan
    return out


def naive_p(target: str, train_keys, rows: np.ndarray, step: bool = False) -> np.ndarray:
    tr = train_rows(train_keys)
    y, s = D["y"][target][tr], D["sensor_id"][tr]
    w = D["n_grid"][tr] if step else np.ones(tr.size)
    glob = (np.sum(w * y) + JEFFREYS) / (w.sum() + 1.0)
    p = np.full(rows.size, glob)
    for sid in np.unique(D["sensor_id"][rows]):
        m = s == sid
        if m.any():
            share = (np.sum(w[m] * y[m]) + JEFFREYS) / (w[m].sum() + 1.0)
            p[D["sensor_id"][rows] == sid] = share
    return p


def fmt(v, nd=3) -> str:
    return "n/a" if v is None or v != v else f"{v:.{nd}f}"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tracks", type=Path, default=TRACKS_DEFAULT)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    t_start = time.time()
    load(args.tracks)
    TK, EK = D["test_keys"], D["eval_keys"]
    log(f"segments: {D['X'].shape[0]} ({int(D['is_test'].sum())} testing), "
        f"{D['X'].shape[1]} features ({len(FEATURES)} numeric + {len(D['sensor_ids'])} sensor one-hot)")
    for t in TARGETS:
        log(f"  y_{t}: {int(D['y'][t][D['is_test']].sum())} drone segments in testing, "
            f"{int(D['y'][t][D['is_test'] & D['deploy']].sum())} on deployable sensors")
    assert len(TK) == 6, TK

    # ---- stage 1: pair fits (inner folds) + outer fits, every target x combo
    tasks = []
    for t in TARGETS:
        for c in COMBOS:
            for a, b in combinations(TK, 2):
                tasks.append((t, c, tuple(k for k in TK if k not in (a, b)), (a, b)))
            for k in TK:
                tasks.append((t, c, tuple(x for x in TK if x != k), (k,)))
    log(f"stage 1: {len(tasks)} GBDT fits on {args.jobs} processes ...")
    t1 = time.time()
    R = run_tasks(tasks, args.jobs)
    log(f"  done in {time.time() - t1:.0f} s")

    def pair(t, c, k, j):           # model fitted without {k, j}, prediction on j
        a, b = sorted((k, j))
        task = (t, c, tuple(x for x in TK if x not in (a, b)), (a, b))
        return R[task], R[task]["pred"][j]

    def outer(t, c, k):
        task = (t, c, tuple(x for x in TK if x != k), (k,))
        return R[task], R[task]["pred"][k]

    dep, g = D["deploy"], D["n_grid"]
    guard = []                      # (held-out key, keys the model / calibrator was fitted on)
    sel, results, inner_tab = {}, {}, {}
    p_inner: dict = {}              # (target, outer fold k, iteration j != k) -> calibrated p on j
    final_choice = {}
    for t in TARGETS:
        y = D["y"][t]
        p_cal = np.full(y.size, np.nan)
        p_raw = np.full(y.size, np.nan)
        p_naive = np.full(y.size, np.nan)
        p_naive_step = np.full(y.size, np.nan)
        fold_rows = []
        for k in TK:
            inner = {}
            for c in COMBOS:
                vals = []
                for j in TK:
                    if j == k:
                        continue
                    r, pj = pair(t, c, k, j)
                    assert k not in r["fitted_on"] and j not in r["fitted_on"]
                    rows = D["rows"][j]
                    mk = dep[rows]
                    vals.append(GB.logloss(y[rows][mk], pj[mk], g[rows][mk]))
                inner[c] = float(np.mean(vals))
            inner_tab[(t, k)] = inner
            best = min(COMBOS, key=lambda c: (inner[c], COMBOS.index(c)))
            sel[(t, k)] = best
            # calibrator: pooled inner OOF of the chosen combo on the 5 training iterations
            cp, cy, cw, cal_on = [], [], [], set()
            for j in TK:
                if j == k:
                    continue
                r, pj = pair(t, best, k, j)
                cal_on |= set(r["fitted_on"]) | {j}
                rows = D["rows"][j]
                cp.append(pj); cy.append(y[rows])
                cw.append(np.ones(rows.size) if best[1] == "unit" else g[rows])
            iso = GB.Isotonic().fit(np.concatenate(cp), np.concatenate(cy), np.concatenate(cw))
            for j in TK:            # nested predictions on fold k's scale, no label of k used
                if j != k:
                    p_inner[(t, k, j)] = iso.predict(pair(t, best, k, j)[1])
            r, pk = outer(t, best, k)
            rows = D["rows"][k]
            p_raw[rows] = pk
            p_cal[rows] = iso.predict(pk)
            train_k = tuple(x for x in TK if x != k)
            p_naive[rows] = naive_p(t, train_k, rows)
            p_naive_step[rows] = naive_p(t, train_k, rows, step=True)
            guard.append((k, r["fitted_on"], tuple(sorted(cal_on)), train_k))
        # final combination: lowest mean 6-fold LOIO metric (raw output, deployable)
        loio = {}
        for c in COMBOS:
            vals = []
            for k in TK:
                _, pk = outer(t, c, k)
                rows = D["rows"][k]
                mk = dep[rows]
                vals.append(GB.logloss(y[rows][mk], pk[mk], g[rows][mk]))
            loio[c] = float(np.mean(vals))
        final_choice[t] = (min(COMBOS, key=lambda c: (loio[c], COMBOS.index(c))), loio)
        results[t] = dict(p_cal=p_cal, p_raw=p_raw, p_naive=p_naive, p_naive_step=p_naive_step)

    # ---- stage 2: final models
    tasks = [(t, final_choice[t][0], tuple(TK), tuple(EK + TK)) for t in TARGETS]
    log("stage 2: final models ...")
    RF = run_tasks(tasks, min(args.jobs, len(tasks)))
    final = {}
    for t, task in zip(TARGETS, tasks):
        c = final_choice[t][0]
        cp, cy, cw = [], [], []
        for k in TK:
            _, pk = outer(t, c, k)
            rows = D["rows"][k]
            cp.append(pk); cy.append(D["y"][t][rows])
            cw.append(np.ones(rows.size) if c[1] == "unit" else g[rows])
        iso = GB.Isotonic().fit(np.concatenate(cp), np.concatenate(cy), np.concatenate(cw))
        r = RF[task]
        p_eval = {k: iso.predict(r["pred"][k]) for k in EK}
        p_insample = {k: iso.predict(r["pred"][k]) for k in TK}
        final[t] = dict(combo=c, iso=iso, model=r["model"], fitted_on=r["fitted_on"],
                        p_eval=p_eval, p_insample=p_insample)
        for k in EK:
            guard.append((k, r["fitted_on"], tuple(TK), tuple(TK)))
        doc = {"target": t, "positive_labels": list(TARGETS[t]), "features": D["names"],
               "sensor_ids": D["sensor_ids"], "combo": combo_name(c), "config": CONFIGS[c[0]],
               "weight": c[1], "train_iters": list(r["fitted_on"]),
               "calibration": "isotonic on the 6-fold LOIO out-of-fold predictions",
               "gbdt": r["model"], "isotonic": iso.to_dict()}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"model_{t}.json").write_text(json.dumps(doc))

    # ---- write per-iteration predictions
    for k in D["keys"]:
        rows = D["rows"][k]
        cols = {"seg_id": D["seg_id"][rows].tolist()}
        for t in TARGETS:
            cols[f"p_drone_{t}"] = (results[t]["p_cal"][rows] if k in TK
                                    else final[t]["p_eval"][k])
        cols["fold"] = [k if k in TK else "final"] * rows.size
        C.write_columns(OUT / f"{C.iter_slug(k)}__p.csv", cols,
                        ["seg_id", "p_drone_all", "p_drone_air", "fold"])
        if k in TK:                 # nested (inner) predictions for downstream LOIO tuning
            outs = [x for x in TK if x != k]
            cols = {"seg_id": np.tile(D["seg_id"][rows], len(outs)).tolist(),
                    "outer_fold": np.repeat(outs, rows.size).tolist()}
            for t in TARGETS:
                cols[f"p_drone_{t}"] = np.concatenate([p_inner[(t, o, k)] for o in outs]).tolist()
            C.write_columns(OUT / f"{C.iter_slug(k)}__p_inner.csv", cols,
                            ["seg_id", "outer_fold", "p_drone_all", "p_drone_air"])

    # ---- metrics
    test = D["is_test"]
    md_rep = {}
    for t in TARGETS:
        y = D["y"][t]
        res = results[t]
        per_fold = []
        for k in TK:
            rows = D["rows"][k]
            for scope, mk in (("deploy", dep[rows]), ("all", np.ones(rows.size, bool))):
                rr = rows[mk]
                per_fold.append(dict(k=k, scope=scope, n=rr.size, pos=int(y[rr].sum()),
                                     steps=float(g[rr].sum()), pos_steps=float((g[rr] * y[rr]).sum()),
                                     model=metrics(y[rr], res["p_cal"][rr], g[rr]),
                                     raw=metrics(y[rr], res["p_raw"][rr], g[rr]),
                                     naive=metrics(y[rr], res["p_naive"][rr], g[rr]),
                                     naive_step=metrics(y[rr], res["p_naive_step"][rr], g[rr])))
        md_rep[t] = per_fold

    def mean_of(t, scope, which, key):
        return float(np.nanmean([f[which][key] for f in md_rep[t] if f["scope"] == scope]))

    # ---- self-checks
    fails = 0

    def check(num, name, ok, detail):
        nonlocal fails
        fails += 0 if ok else 1
        log(f"[{'PASS' if ok else 'FAIL'}] {num}. {name}: {detail}")

    # 1 leakage guard
    bad = []
    for held, fitted_on, cal_on, naive_on in guard:
        if held in fitted_on or (held in TK and held in cal_on) or held in naive_on:
            bad.append(held)
        if any(x in EK for x in fitted_on + cal_on + naive_on):
            bad.append(f"eval in training ({held})")
    for r in list(R.values()) + list(RF.values()):
        if any(x in EK for x in r["fitted_on"]):
            bad.append(f"eval rows in fit {r['task'][:2]}")
        if set(r["fitted_on"]) != set(r["task"][2]):
            bad.append(f"fit rows {r['fitted_on']} != requested {r['task'][2]}")
        if r["task"] in R and set(r["fitted_on"]) & set(r["task"][3]):
            bad.append(f"fit {r['task'][:2]} predicts an iteration it was fitted on")
    pred_fold = {}
    for k in D["keys"]:
        pf = C.read_columns(OUT / f"{C.iter_slug(k)}__p.csv",
                            numeric={"seg_id", "p_drone_all", "p_drone_air"})
        pred_fold[k] = pf
    for k in TK:
        if not np.all(pred_fold[k]["fold"] == k):
            bad.append(f"{k}: fold column != own key")
    check(1, "leakage guard", not bad,
          f"{len(guard)} held-out predictions sets (6 testing x 2 targets + 3 eval x 2) and "
          f"{len(R) + len(RF)} fits: no held-out iteration in its model / calibrator / naive "
          f"training set, no evaluation row in any training set" + (f"; {bad[:5]}" if bad else ""))

    # 2 one prediction per segment in [0, 1]
    bad2 = []
    for k in D["keys"]:
        pf, rows = pred_fold[k], D["rows"][k]
        if not np.array_equal(pf["seg_id"].astype(np.int64), D["seg_id"][rows]) or \
                np.unique(pf["seg_id"]).size != rows.size:
            bad2.append(f"{k}: seg_id set")
        for t in TARGETS:
            p = pf[f"p_drone_{t}"]
            if not (np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))):
                bad2.append(f"{k} {t}: p outside [0, 1] or NaN")
        if k in EK and not np.all(pf["fold"] == "final"):
            bad2.append(f"{k}: fold != final")
    check(2, "exactly one prediction in [0, 1] per segment", not bad2,
          f"{sum(r.size for r in D['rows'].values())} segments over {len(D['keys'])} iterations"
          + (f"; {bad2[:5]}" if bad2 else ""))

    # 3 final model reproducible from JSON (+ an independent refit with the stored params)
    bad3 = []
    for t in TARGETS:
        doc = json.loads((OUT / f"model_{t}.json").read_text())
        m = GB.GBDT.from_dict(doc["gbdt"])
        iso = GB.Isotonic.from_dict(doc["isotonic"])
        if doc["features"] != D["names"]:
            bad3.append(f"{t}: feature list")
        for k in EK:
            p = iso.predict(m.predict_proba(D["X"][D["rows"][k]]))
            if not np.array_equal(p, pred_fold[k][f"p_drone_{t}"]):
                bad3.append(f"{t} {k}: JSON model != written predictions "
                            f"(max |d| {np.max(np.abs(p - pred_fold[k][f'p_drone_{t}'])):.2e})")
        rows = train_rows(doc["train_iters"])
        m2 = GB.GBDT(**doc["gbdt"]["params"]).fit(D["X"][rows], D["y"][t][rows],
                                                  sample_weight=weights(rows, doc["weight"]))
        if json.dumps(m2.to_dict()) != json.dumps(doc["gbdt"]):
            bad3.append(f"{t}: refit from stored params differs")
    check(3, "final model reproducible from model_*.json", not bad3,
          "JSON model + calibrator reproduce every written evaluation prediction bit for bit; a "
          "refit with the stored params on the stored training iterations gives identical trees"
          + (f"; {bad3}" if bad3 else ""))

    # 4 model beats naive in mean held-out step-weighted logloss (deployable)
    det4, ok4 = [], True
    for t in TARGETS:
        mm, nn = mean_of(t, "deploy", "model", "ll_step"), mean_of(t, "deploy", "naive", "ll_step")
        ok4 &= mm < nn
        det4.append(f"y_{t}: model {mm:.4f} vs naive {nn:.4f}")
    check(4, "model beats naive (mean held-out step-weighted logloss, deployable)", ok4,
          "; ".join(det4))

    # 5 calibration pooled over held-out folds
    det5, ok5 = [], True
    mk = test & dep
    for t in TARGETS:
        y, p = D["y"][t][mk], results[t]["p_cal"][mk]
        for tag, w in (("seg", np.ones(y.size)), ("step", g[mk])):
            d = abs(np.sum(w * p) / w.sum() - np.sum(w * y) / w.sum())
            ok5 &= d <= CAL_TOL
            det5.append(f"y_{t} {tag} {np.sum(w * p) / w.sum():.4f} vs {np.sum(w * y) / w.sum():.4f}")
        ce = calib_error(calib_bins(y, p, g[mk]))
        det5.append(f"y_{t} [not gated: ECE {ce['ece']:.4f}, worst bin {bin_name(ce['worst'])} "
                    f"P {ce['worst']['mp']:.3f} vs {ce['worst']['obs']:.3f}]")
    check(5, f"calibration |mean predicted - observed| <= {CAL_TOL} (pooled held-out, deployable)",
          ok5, "; ".join(det5) + " (mean only: isotonic preserves the mean by construction, "
          "per-bin errors are in existence.md)")

    runtime = time.time() - t_start
    write_report(md_rep, results, final, sel, inner_tab, final_choice, runtime, fails)
    log(f"runtime {runtime:.0f} s")
    if fails:
        log(f"SELF-CHECKS FAILED: {fails}")
        return 1
    log("ALL SELF-CHECKS PASSED")
    return 0


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def cut_stats(y, p, g, cut, inclusive: bool = False) -> dict:
    """What a 'P > cut' filter keeps. Isotonic output has plateaus whose value can land exactly
    on the cut (e.g. 0.5 up to float noise), so ties |P - cut| <= TIE_EPS are counted
    separately ('ties') and dropped by the strict cut, kept by the inclusive one (P >= cut)."""
    keep = p >= cut - TIE_EPS if inclusive else p > cut + TIE_EPS
    tp, fp = float(np.sum(keep * y)), float(np.sum(keep * (1 - y)))
    tps, fps = float(np.sum(keep * y * g)), float(np.sum(keep * (1 - y) * g))
    return dict(kept=int(keep.sum()), ties=int(np.sum(np.abs(p - cut) <= TIE_EPS)),
                prec=tp / (tp + fp) if tp + fp else np.nan, rec=tp / y.sum() if y.sum() else np.nan,
                prec_s=tps / (tps + fps) if tps + fps else np.nan,
                drone_surv=tps / np.sum(y * g) if np.sum(y * g) else np.nan,
                clutter_surv=fps / np.sum((1 - y) * g) if np.sum((1 - y) * g) else np.nan)


def calib_bins(y, p, g) -> list[dict]:
    """Bins of predicted P (width 0.1): segments, drones, mean P / observed rate (segment- and
    step-weighted)."""
    edges = np.linspace(0, 1, 11)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
    out = []
    for i in range(10):
        m = b == i
        w = g[m]
        out.append(dict(lo=edges[i], hi=edges[i + 1], n=int(m.sum()), pos=int(y[m].sum()),
                        mp=float(p[m].mean()) if m.any() else np.nan,
                        obs=float(y[m].mean()) if m.any() else np.nan,
                        mps=float(np.sum(w * p[m]) / w.sum()) if w.sum() else np.nan,
                        obss=float(np.sum(w * y[m]) / w.sum()) if w.sum() else np.nan))
    return out


def calib_error(bins: list[dict]) -> dict:
    """ECE = segment-weighted mean |mean P - observed| over the bins; 'worst' = the largest
    |mean P - observed| among bins with >= BIN_MIN segments; 'bad' = those bins off by > 0.1."""
    n = sum(b["n"] for b in bins)
    ece = sum(b["n"] * abs(b["mp"] - b["obs"]) for b in bins if b["n"]) / n
    big = [b for b in bins if b["n"] >= BIN_MIN]
    worst = max(big, key=lambda b: abs(b["mp"] - b["obs"]))
    return dict(ece=ece, worst=worst, worst_err=abs(worst["mp"] - worst["obs"]),
                bad=[b for b in big if abs(b["mp"] - b["obs"]) > 0.1])


def bin_name(b: dict) -> str:
    return f"{b['lo']:.1f}-{b['hi']:.1f}"


def within_sensor_auc(y, p, g, sid) -> dict:
    """AUC restricted to (drone, clutter) pairs from the SAME sensor: the per-sensor AUCs
    averaged with weight n_drone x n_clutter (segment) or their grid-sample products (step).
    It removes what the sensor id alone separates (e.g. EGOI/NYTF clutter vs relay feeds)."""
    num = {"seg": 0.0, "step": 0.0}
    den = {"seg": 0.0, "step": 0.0}
    for s in np.unique(sid):
        m = sid == s
        ys, ps, gs = y[m], p[m], g[m]
        if not 0 < ys.sum() < ys.size:
            continue
        for tag, w in (("seg", np.ones(ys.size)), ("step", gs)):
            pw = np.sum(w * ys) * np.sum(w * (1 - ys))
            if pw > 0:
                num[tag] += pw * GB.auc(ys, ps, w)
                den[tag] += pw
    return {t: num[t] / den[t] if den[t] else np.nan for t in num}


def write_report(md_rep, results, final, sel, inner_tab, final_choice, runtime, fails) -> None:
    TK = D["test_keys"]
    dep, g, test = D["deploy"], D["n_grid"], D["is_test"]
    L = ["# Track existence: P(segment is a real drone)", ""]
    L += [f"**Self-checks: {'ALL PASSED' if not fails else f'{fails} FAILED'}** "
          f"(run log of `python3 tools/existence.py`; runtime {runtime:.0f} s).", ""]
    L += ["Generated by `tools/existence.py` (rules in its docstring). One probability per sensor "
          "track segment of `tools/sensor_tracks.py`: `p_drone_all` (drone in the air or on the "
          "ground, truth mode 'all') and `p_drone_air` (airborne drone). Outputs: "
          "`<slug>__p.csv` per iteration, `model_all.json` / `model_air.json` (final models + "
          "isotonic calibrators).", ""]
    L += ["## Summary", ""]
    mk = test & dep
    cuts = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
    for t in TARGETS:
        fs = [f for f in md_rep[t] if f["scope"] == "deploy"]
        y, p = D["y"][t][mk], results[t]["p_cal"][mk]
        c5, c1 = cut_stats(y, p, g[mk], CUT), cut_stats(y, p, g[mk], 0.1)
        L += [f"- **y_{t}** ({int(y.sum())} drone / {int(mk.sum())} deployable testing segments): "
              f"held-out step-weighted AUC {np.nanmean([f['model']['auc_step'] for f in fs]):.3f} "
              f"(naive {np.nanmean([f['naive']['auc_step'] for f in fs]):.3f}), step-weighted "
              f"logloss {np.nanmean([f['model']['ll_step'] for f in fs]):.4f} (naive "
              f"{np.nanmean([f['naive']['ll_step'] for f in fs]):.4f}). A P > {CUT} cut keeps "
              f"{100 * c5['drone_surv']:.1f} % of the drone grid samples and "
              f"{100 * c5['clutter_surv']:.2f} % of the clutter ones; P > 0.1 keeps "
              f"{100 * c1['drone_surv']:.1f} % vs {100 * c1['clutter_surv']:.2f} %."]
    for t in TARGETS:
        y, p, gg = D["y"][t][mk], results[t]["p_cal"][mk], g[mk]
        bins = calib_bins(y, p, gg)
        ce = calib_error(bins)
        b0 = bins[0]
        cs = {c: cut_stats(y, p, gg, c) for c in cuts}
        best = max((c for c in cuts if cs[c]["kept"] >= BIN_MIN), key=lambda c: cs[c]["prec"])
        sid = D["sensor_id"][mk]
        wa = within_sensor_auc(y, p, gg, sid)
        ga = {"seg": GB.auc(y, p), "step": GB.auc(y, p, gg)}
        hi = (p > CUT + TIE_EPS) & (y == 0)
        u, n = np.unique(sid[hi], return_counts=True)
        o = np.lexsort((u, -n))[:4]
        top_worse = cs[best]["prec"] > cs[CUT]["prec"] and best < CUT
        L += [f"- **y_{t} calibration: per-bin worst error {ce['worst_err']:.2f} above P 0.1"
              + ("; the top of the ranking is worse than the middle" if top_worse else "")
              + f".** The 0-0.1 bin ({b0['n']} of {int(mk.sum())} segments) "
              f"matches (mean P {b0['mp']:.4f} vs observed {b0['obs']:.4f}). Above it the P "
              f"values are NOT probabilities: the worst bin with >= {BIN_MIN} segments is "
              f"{bin_name(ce['worst'])} (mean P {ce['worst']['mp']:.3f} vs observed "
              f"{ce['worst']['obs']:.3f}); bins off by > 0.1: "
              + (", ".join(f"{bin_name(b)} ({b['n']} seg, P {b['mp']:.2f} vs {b['obs']:.2f})"
                           for b in ce["bad"]) or "none")
              + f". Precision is highest at P > {best} ({fmt(cs[best]['prec'])}) and drops to "
              f"{fmt(cs[CUT]['prec'])} at P > {CUT}"
              + (": the segments given the highest P are mostly clutter, chiefly from "
                 + ", ".join(f"{u[i]} ({n[i]})" for i in o) + " (see 'who is above the cut')"
                 if cs[CUT]["prec"] < 0.5 and o.size else "")
              + f". So P = 0.6 does not mean a 60 % chance, "
              f"and P > {CUT} is not a usable operating point; read a cut off the P-cut table "
              f"instead. Ranking within a sensor: AUC over same-sensor (drone, clutter) pairs "
              f"{fmt(wa['seg'])} (step-weighted {fmt(wa['step'])}) vs {fmt(ga['seg'])} "
              f"({fmt(ga['step'])}) over all pairs (pooled folds). The gap is what the sensor "
              f"identity alone contributes; the same-sensor number is carried by the big radars "
              f"(EGOI, NYTF), and several small deployable sensors rank their own segments at "
              f"or below chance (per-sensor tables)."]
    L += ["- Check 5 (pooled mean P vs observed rate within 0.05) only tests the overall average. "
          "Isotonic regression preserves the weighted mean of its training data by construction, "
          "and with a base rate near 1 % even a constant P = 0 would pass it; it is necessary, "
          "not sufficient. The per-bin tables and the ECE below are the real calibration "
          "evidence. Likely cause: the calibrator is fitted on only ~300 drone segments from "
          "other iterations, and the high-P region is dominated by a few sensors with few "
          "segments or with a drone share that differs between iterations (listed above), so "
          "the score-to-rate map learnt on 5 iterations does not transfer to the 6th there.",
          ""]
    L += ["## Terms in one line each", "",
          "- **GOSPA**: the track-accuracy score; each grid step, a false track costs as much as a "
          "missed drone, so dropping clutter segments is worth as much as finding drones.",
          "- **segment**: one sensor's track of one object, cut at gaps > 30 s. **grid samples / "
          "steps**: the 100 ms template times a segment covers (`n_grid`): how often it would be "
          "scored.",
          "- **LOIO (leave-one-iteration-out)**: to judge iteration k, everything (model, "
          "calibration, choice of settings, baseline) is fitted on the other 5 testing iterations "
          "only; k is seen once, at the end. **Nested**: the settings and the calibrator for fold "
          "k come from a second LOIO inside those 5.",
          "- **AUC**: the chance that a random real-drone segment gets a higher P than a random "
          "clutter segment (0.5 = coin flip, 1 = perfect ranking). It ignores whether the P "
          "values are right in absolute terms.",
          "- **logloss**: the average of -log(P given to what actually happened); punishes "
          "confident mistakes hard. Lower is better. **Brier**: mean (P - outcome)^2; lower is better.",
          "- **segment-weighted** metrics count every segment once; **step-weighted** ones weight "
          "each segment by its grid samples, i.e. by how much it would move GOSPA.",
          "- **calibration**: P is calibrated if, among segments given P ~ 0.3, about 30 % are real "
          "drones. **Isotonic regression** is the calibrator: a non-decreasing map from the raw "
          "model score to the observed drone rate, fitted on out-of-fold predictions.",
          "- **precision** at a cut: of the segments kept (P > cut), the share that are real drones. "
          "**recall**: of the real-drone segments, the share kept.",
          "- **naive baseline**: P = the share of drone segments of the same sensor in the "
          "training iterations (Jeffreys-smoothed). It knows the sensor and nothing else; the gap "
          "to it is what the segment features add.",
          "- **deployable sensors**: sensors present in the evaluation data. EUIG, LJJX, MMJJ "
          "(testing only) are used for training but kept out of the headline numbers.", ""]

    names = D["names"]
    L += ["## Features", "",
          f"{len(names)} columns: {len(FEATURES)} numeric segment features of tracks.csv and a "
          f"one-hot of the {len(D['sensor_ids'])} sensor ids that have segments. Excluded: "
          + ", ".join(f"`{c}`" for c in EXCLUDED) + " (identifiers, times, strings, ground truth).",
          "", "`" + "`, `".join(names) + "`", ""]

    L += ["## Model selection", "",
          "Combinations: tree settings " + "; ".join(
              f"`{c}` = " + ", ".join(f"{k} {v}" for k, v in CONFIGS[c].items() if k != "seed")
              for c in CONFIGS) +
          " x weights `unit` (1 per segment) / `grid` (n_grid, normalised to mean 1). Metric: "
          "step-weighted logloss of the raw model output on the deployable sensors, mean over the "
          "inner LOIO folds (for each held-out iteration) or over the 6 LOIO folds (final model).", ""]
    for t in TARGETS:
        hdr = ["held-out iteration"] + [combo_name(c) for c in COMBOS] + ["chosen"]
        rows = [[k] + [fmt(inner_tab[(t, k)][c], 4) for c in COMBOS] + [combo_name(sel[(t, k)])]
                for k in TK]
        loio = final_choice[t][1]
        rows.append(["**6-fold LOIO (final)**"] + [fmt(loio[c], 4) for c in COMBOS]
                    + [combo_name(final_choice[t][0])])
        L += [f"### y_{t}: inner-LOIO step-weighted logloss per combination", ""]
        L += md_table(hdr, rows) + [""]
        uw = np.mean([inner_tab[(t, k)][c] for k in TK for c in COMBOS if c[1] == "unit"])
        gw = np.mean([inner_tab[(t, k)][c] for k in TK for c in COMBOS if c[1] == "grid"])
        L += [f"Weight 1 vs weight n_grid, mean inner metric over all configs and folds: "
              f"unit {uw:.4f}, grid {gw:.4f}.", ""]

    for t in TARGETS:
        L += [f"## y_{t}: held-out metrics per iteration (deployable sensors)", "",
              "Model = calibrated LOIO prediction; naive = sensor share. 'n_pos' = drone "
              "segments, 'pos steps' = their grid samples.", ""]
        for tag, title in (("seg", "segment-weighted"), ("step", "step-weighted")):
            hdr = ["iteration", "segments", "n_pos", "steps", "pos steps", "AUC model", "AUC naive",
                   "logloss model", "logloss naive", "Brier model", "Brier naive"]
            rows = []
            for f in md_rep[t]:
                if f["scope"] != "deploy":
                    continue
                rows.append([f["k"], f["n"], f["pos"], int(f["steps"]), int(f["pos_steps"]),
                             fmt(f["model"][f"auc_{tag}"]), fmt(f["naive"][f"auc_{tag}"]),
                             fmt(f["model"][f"ll_{tag}"], 4), fmt(f["naive"][f"ll_{tag}"], 4),
                             fmt(f["model"][f"brier_{tag}"], 4), fmt(f["naive"][f"brier_{tag}"], 4)])
            fs = [f for f in md_rep[t] if f["scope"] == "deploy"]
            rows.append(["**mean**", "", "", "", "",
                         *[fmt(np.nanmean([f[w][f"{m}_{tag}"] for f in fs]), 3 if m == "auc" else 4)
                           for m in ("auc", "ll", "brier") for w in ("model", "naive")]])
            L += [f"**{title}**", ""] + md_table(hdr, rows) + [""]
        fs = [f for f in md_rep[t] if f["scope"] == "deploy"]
        fa = [f for f in md_rep[t] if f["scope"] == "all"]
        L += ["Side notes (means over folds): raw (uncalibrated) model step logloss "
              f"{np.nanmean([f['raw']['ll_step'] for f in fs]):.4f}; naive with a step-weighted "
              f"sensor share {np.nanmean([f['naive_step']['ll_step'] for f in fs]):.4f} "
              f"(segment-share naive {np.nanmean([f['naive']['ll_step'] for f in fs]):.4f}). "
              f"All sensors (incl. testing-only): model step logloss "
              f"{np.nanmean([f['model']['ll_step'] for f in fa]):.4f} vs naive "
              f"{np.nanmean([f['naive']['ll_step'] for f in fa]):.4f}, AUC "
              f"{np.nanmean([f['model']['auc_seg'] for f in fa]):.3f} vs "
              f"{np.nanmean([f['naive']['auc_seg'] for f in fa]):.3f}.", ""]

        # per-sensor AUC
        y = D["y"][t]
        p, pn = results[t]["p_cal"], results[t]["p_naive"]
        rows = []
        for s in D["sensor_ids"]:
            mk = test & (D["sensor_id"] == s)
            if not mk.any():
                continue
            ys, ps, gs = y[mk], p[mk], g[mk]
            two = 0 < ys.sum() < ys.size
            rows.append([s, "yes" if s in C.deployable_sensor_ids() else "no (testing only)",
                         int(mk.sum()), int(ys.sum()),
                         fmt(GB.auc(ys, ps)) if two else "n/a",
                         fmt(GB.auc(ys, ps, gs)) if two and np.sum(gs * ys) > 0 and np.sum(gs * (1 - ys)) > 0 else "n/a",
                         fmt(float(np.mean(ps)), 4), fmt(float(np.mean(ys)), 4),
                         fmt(float(np.mean(pn[mk])), 4)])
        L += [f"### y_{t}: per-sensor AUC (pooled held-out folds)", "",
              "AUC within one sensor: how well the features rank that sensor's own segments "
              "(the naive baseline is constant per sensor, so its within-sensor AUC is 0.5).", ""]
        L += md_table(["sensor", "deployable", "segments", "drones", "AUC", "AUC step",
                       "mean P", "drone rate", "naive P (mean)"], rows) + [""]
        weak = [r for r in rows if r[1] == "yes" and r[3] > 0
                and any(v != "n/a" and float(v) < 0.6 for v in (r[4], r[5]))]
        mkd = test & dep
        wa = within_sensor_auc(y[mkd], p[mkd], g[mkd], D["sensor_id"][mkd])
        L += [f"Within-sensor ranking is weak or inverted on deployable sensors "
              + (", ".join(f"{r[0]} (AUC {r[4]}, step {r[5]}, {r[3]} drones / {r[2]} seg)"
                           for r in weak) or "none")
              + " (AUC or step AUC < 0.6). AUC < 0.5 means that sensor's clutter segments get "
              "HIGHER P than its drones; a step AUC near or below 0.5 means the ranking does not "
              "hold for the long segments that dominate GOSPA. Pooled over same-sensor pairs "
              f"only, the deployable AUC is {fmt(wa['seg'])} (step-weighted {fmt(wa['step'])}) "
              f"vs {fmt(GB.auc(y[mkd], p[mkd]))} ({fmt(GB.auc(y[mkd], p[mkd], g[mkd]))}) over "
              "all pairs: the difference is credit for the sensor identity, and the same-sensor "
              "number is carried by EGOI / NYTF, which hold most pairs. Sensors with a few dozen "
              "segments give noisy AUCs, but where they are below 0.5 the model should not be "
              "trusted to filter that sensor's segments.", ""]

        # calibration table
        mk = test & dep
        ys, ps, gs = y[mk], p[mk], g[mk]
        bins = calib_bins(ys, ps, gs)
        ce = calib_error(bins)
        rows = []
        for bb in bins:
            if not bb["n"]:
                rows.append([bin_name(bb), 0, 0, "", "", "", ""])
                continue
            rows.append([bin_name(bb), bb["n"], bb["pos"], fmt(bb["mp"], 4), fmt(bb["obs"], 4),
                         fmt(bb["mps"], 4), fmt(bb["obss"], 4)])
        rows.append(["**all**", int(mk.sum()), int(ys.sum()), fmt(float(ps.mean()), 4),
                     fmt(float(ys.mean()), 4), fmt(float(np.sum(gs * ps) / gs.sum()), 4),
                     fmt(float(np.sum(gs * ys) / gs.sum()), 4)])
        L += [f"### y_{t}: calibration (pooled held-out folds, deployable)", "",
              "Bins of predicted P (width 0.1). A calibrated P would have 'mean P' ~ 'observed "
              "rate' in every row; this one does only in the 0-0.1 bin. Expected calibration "
              f"error (ECE, segment-weighted mean |mean P - observed| over bins) {ce['ece']:.4f}, "
              "which the 0-0.1 bin dominates; the worst bin with >= "
              f"{BIN_MIN} segments is {bin_name(ce['worst'])} ({ce['worst']['n']} segments, "
              f"mean P {ce['worst']['mp']:.3f} vs observed {ce['worst']['obs']:.3f}). Above "
              "P 0.1 treat P as a rank, not as a probability.", ""]
        L += md_table(["P bin", "segments", "drones", "mean P", "observed rate",
                       "mean P (step-w.)", "observed (step-w.)"], rows) + [""]

        # cuts
        rows = []
        for cut in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
            cs = cut_stats(ys, ps, gs, cut)
            rows.append([f"{'**' if cut == CUT else ''}P > {cut}{'**' if cut == CUT else ''}",
                         cs["kept"], cs["ties"], fmt(cs["prec"]), fmt(cs["rec"]),
                         fmt(cs["prec_s"]), fmt(cs["drone_surv"]), fmt(cs["clutter_surv"])])
        cs = cut_stats(ys, ps, gs, CUT)
        ci = cut_stats(ys, ps, gs, CUT, inclusive=True)
        bc, bp = max(((c, cut_stats(ys, ps, gs, c)) for c in (0.05, 0.1, 0.2, 0.3)
                      if cut_stats(ys, ps, gs, c)["kept"] >= BIN_MIN),
                     key=lambda x: x[1]["prec"])
        bp = bp["prec"]
        L += [f"### y_{t}: what a P cut keeps (pooled held-out folds, deployable)", "",
              f"At P > {CUT}: precision {fmt(cs['prec'])}, recall {fmt(cs['rec'])} of the drone "
              f"segments; {fmt(100 * cs['drone_surv'], 1)} % of the drone grid samples survive vs "
              f"{fmt(100 * cs['clutter_surv'], 2)} % of the clutter grid samples. "
              f"{cs['ties']} segments sit exactly on P = {CUT} (an isotonic plateau; 'ties' "
              f"= |P - cut| <= {TIE_EPS:g}); the strict cut drops them, P >= {CUT} keeps them: "
              f"{ci['kept']} segments, precision {fmt(ci['prec'])}, recall {fmt(ci['rec'])}, "
              f"{fmt(100 * ci['drone_surv'], 1)} % of the drone vs "
              f"{fmt(100 * ci['clutter_surv'], 2)} % of the clutter grid samples."
              + (f" Precision at P > {CUT} is lower than at P > {bc} ({fmt(bp)}): see the "
                 "calibration section." if bp > cs["prec"] else ""), ""]
        L += md_table(["cut", "segments kept", "ties at cut", "precision", "recall",
                       "precision (step-w.)", "drone samples kept", "clutter samples kept"],
                      rows) + [""]
        # who is above the cut
        sid = D["sensor_id"][mk]
        rows = []
        for s in sorted(set(sid.tolist())):
            m = sid == s
            hi, mid = m & (ps > CUT + TIE_EPS), m & (ps > 0.1 + TIE_EPS) & (ps <= CUT + TIE_EPS)
            if not (hi.any() or mid.any()):
                continue
            rows.append([s, int(m.sum()), int(ys[hi].sum()), int((1 - ys[hi]).sum()),
                         int(ys[mid].sum()), int((1 - ys[mid]).sum())])
        rows.sort(key=lambda r: (-(r[2] + r[3]), r[0]))
        L += [f"**Who is above the cut** (deployable segments by sensor: drones / clutter with "
              f"P > {CUT}, and with 0.1 < P <= {CUT}):", ""]
        L += md_table(["sensor", "segments", f"drones P > {CUT}", f"clutter P > {CUT}",
                       f"drones 0.1-{CUT}", f"clutter 0.1-{CUT}"], rows) + [""]

    # feature importance
    L += ["## Feature importance of the final models (gain share)", "",
          "Gain = total loss reduction from the splits on a feature; shares sum to 1. Final model "
          "= fitted on all 6 testing iterations with the combination of the 6-fold LOIO row above.", ""]
    imp = {t: np.asarray(final[t]["model"]["gain"]) for t in TARGETS}
    imp = {t: v / v.sum() if v.sum() > 0 else v for t, v in imp.items()}
    order = np.argsort(-(imp["all"] + imp["air"]), kind="stable")
    rows = [[names[i], fmt(imp["all"][i]), fmt(imp["air"][i])] for i in order
            if imp["all"][i] + imp["air"][i] > 0]
    L += md_table(["feature", "y_all", "y_air"], rows)
    unused = [names[i] for i in order if imp["all"][i] + imp["air"][i] == 0]
    L += ["", "Never split on: " + (", ".join(f"`{u}`" for u in unused) or "none") + ".", ""]
    for t in TARGETS:
        L += [f"Final y_{t}: combination {combo_name(final[t]['combo'])}, "
              f"{len(final[t]['model']['trees'])} trees, fitted on {', '.join(final[t]['fitted_on'])}."]
    (OUT / "existence.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
