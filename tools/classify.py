#!/usr/bin/env python3
"""Fixed-wing vs rotary-wing: a per-segment probability P(fixed wing) for the submission's
'classification' column, validated leave-one-iteration-out (LOIO), plus the stake (how many
truth rows are fixed-wing at all).

Why this is small
-----------------
'classification' is 0.2 of the score and is a recall over all truth rows of known class
(score.py: Fixed Wing UAS / Rotary Wing UAS / UAS Transmitter; 'Unknown' rows are not eligible).
Only three drone names are fixed-wing (Heidrun, Vector, Quantum Vector), so the whole question
is how much of the 0.2 the fixed-wing rows are worth and whether anything beats 'call everything
Rotary'. The stake section answers the first part; the method comparison the second.

Inputs and rules
----------------
* Segments: data/derived/sensor_tracks_v2/<slug>__tracks.csv of all 9 iterations (columns used:
  seg_id, iter_key, sensor_id, n_grid, label, best_flight, FEATURES below).
* Targets (testing only): y_fixed = 1 for a segment labelled drone_air whose best_flight (the
  Dronetag file stem DroneTag_<SIDE>_<date>_<hhmm>_<hhmm>_<DroneName>) has catalog class
  'Fixed Wing UAS', 0 for 'Rotary Wing UAS'; flights of class 'Unknown' (RQ) are not used.
  The class comes from truth.catalog_row(DroneName, SIDE)['classification'].
* Training / headline rows: drone_air, class known, DEPLOYABLE sensors (C.deployable_sensor_ids();
  testing-only EUIG, LJJX, MMJJ never enter a fit and are reported separately). VKYY / QTVS are
  deployable UTM relays: on testing they relay the Dronetag feed itself, so their kinematics are
  the true ones (optimistic); they carry 1 and 0 training segments of this kind.
* Features (FEATURES): speed_mean/p10/p90/max_ms, stationary_share, turn_p90_degs,
  accel_p90_ms2, vu_abs_p90_ms, duration_s, path_len_m, det_conf_mean and the sensor class
  shares cls_uav_fixed, cls_uav_rotary, cls_uav, cls_air_vehicle, cls_bird. The height
  features (dz_*, u_ell_*) are left out: they encode where a drone flew, not what it is.
* Methods (every one LOIO: fitted on 5 testing iterations, predicts the 6th):
    all_rotary         p = the training n_grid-weighted fixed share (a constant < 0.5): always Rotary.
    rule_sensor_class  fixed if cls_uav_fixed > cls_uav_rotary (the sensors' own class string).
    rule_kinematic     fixed if stationary_share <= A and speed_p10_ms >= B (a segment that
                       never hovers and never slows down), A and B chosen on the 5 training
                       iterations by n_grid-weighted accuracy over the grid A_GRID x B_GRID
                       (B = inf = 'never fixed' is in the grid, so the rule cannot be worse
                       than all_rotary on its training rows). Ties: fewer rows called fixed,
                       then smaller A, then larger B.
    gbdt_d2 / gbdt_d3  gbdt.GBDT (logistic loss, depth 2 / 3, 150 trees, learning rate 0.05,
                       n_grid sample weights) + gbdt.Isotonic calibration fitted on the pooled
                       inner LOIO out-of-fold predictions of the 5 training iterations (nested,
                       as existence.py: for held-out k, every j != k is predicted by a model
                       fitted on testing minus {k, j}; the outer model is fitted on testing
                       minus {k}).
  For the two rules p_fixed is the training (Jeffreys-smoothed, n_grid-weighted) fixed share of
  the segments the rule calls fixed, respectively not fixed; the hard label of every method is
  p > 0.5 (the raw rule label is also printed for rule_sensor_class).
* Metric: accuracy of the hard label weighted by n_grid (the number of grid samples the
  segment can cover = the truth rows it can win), and AUC (weighted by n_grid), per held-out
  iteration and as mean over the 6; AUC is undefined for an iteration without fixed-wing
  segments (S2-I2). Reported on: deployable drone_air (headline); testing-only sensors; and
  drone_ground segments (a drone parked on the ground: no kinematic method can tell its class).
* Recommendation: the method with the highest mean held-out weighted accuracy, ties to the
  simpler one (order above). If that is not strictly better than all_rotary, all_rotary is
  recommended. NOTE: picking among 5 methods by the same 6 held-out folds is a mild selection
  (the numbers of the winner are slightly optimistic).
* Predictions for EVERY segment (clutter and ground included; the fusion step decides what to
  report): testing = the LOIO outer prediction of the recommended method (fold = held-out
  iteration); evaluation = the same method fitted on all 6 testing iterations (fold = 'final';
  for gbdt the calibrator is fitted on the 6-fold outer OOF predictions). Nothing is fitted on
  the evaluation tree.
* Known in-sample caveat inherited: the sensor_tracks variances and height offsets were fitted
  on all 6 testing iterations (no feature used here depends on them, except through the
  smoothing of speeds).

Outputs (data/derived/classify/)
--------------------------------
<slug>__pclass.csv          seg_id, p_fixed in [0, 1], method (the recommended one), fold
<slug>__pclass_methods.csv  seg_id, p_<method> for every method (same LOIO / final rule), fold
classify.md                 stake table, method comparison, side tables, glossary

Run: python3 tools/classify.py     Numbered self-checks at the end; exit status 1 on any failure.
"""

from __future__ import annotations

import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuas_common as C  # noqa: E402
import gbdt as GB  # noqa: E402
import truth as T  # noqa: E402

OUT = C.DERIVED / "classify"
TRACKS = C.DERIVED / "sensor_tracks_v2"

FEATURES = ["speed_mean_ms", "speed_p10_ms", "speed_p90_ms", "speed_max_ms", "stationary_share",
            "turn_p90_degs", "accel_p90_ms2", "vu_abs_p90_ms", "duration_s", "path_len_m",
            "det_conf_mean", "cls_uav_fixed", "cls_uav_rotary", "cls_uav", "cls_air_vehicle",
            "cls_bird"]
FIXED, ROTARY = "Fixed Wing UAS", "Rotary Wing UAS"
A_GRID = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
B_GRID = np.concatenate([np.arange(0.0, 31.0), [np.inf]])
GBDT_CFG = {
    "gbdt_d2": dict(n_trees=150, learning_rate=0.05, max_depth=2, min_samples_leaf=10, seed=0),
    "gbdt_d3": dict(n_trees=150, learning_rate=0.05, max_depth=3, min_samples_leaf=10, seed=0),
}
METHODS = ["all_rotary", "rule_sensor_class", "rule_kinematic", "gbdt_d2", "gbdt_d3"]
CUT = 0.5
JEFFREYS = 0.5

D: dict = {}
LEDGER: list[tuple] = []     # (method, held-out key, keys used by fit/threshold, keys used by calibrator)


def log(msg: str) -> None:
    print(msg, flush=True)


def md_table(header, rows) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def fmt(v, nd=3) -> str:
    return "n/a" if v is None or v != v else f"{v:.{nd}f}"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def parse_flight(stem: str) -> tuple[str, str]:
    """'DroneTag_RED_240916_0732_0735_Mavic 3' -> ('RED', 'Mavic 3')."""
    p = stem.split("_", 5)
    if len(p) != 6 or p[0] != "DroneTag" or p[1] not in ("RED", "BLUE"):
        raise SystemExit(f"unexpected best_flight {stem!r}")
    return p[1], p[5]


def load() -> None:
    iters = C.iterations()
    keys, parts = [], []
    for it in iters:
        path = TRACKS / f"{C.iter_slug(it.key)}__tracks.csv"
        if not path.exists():
            raise SystemExit(f"missing {path} (run tools/sensor_tracks.py)")
        d = C.read_columns(path, numeric=set(FEATURES) | {"seg_id", "n_grid"},
                           usecols=["seg_id", "iter_key", "sensor_id", "n_grid", "label",
                                    "best_flight"] + FEATURES)
        if not np.all(d["iter_key"] == it.key):
            raise SystemExit(f"{path}: iter_key column != {it.key}")
        if not np.all(np.diff(d["seg_id"]) > 0):
            raise SystemExit(f"{path}: seg_id not strictly ascending")
        keys.append(it.key)
        parts.append(d)
    cat = {c: np.concatenate([p[c] for p in parts]) for c in parts[0]}
    n = cat["seg_id"].size
    iter_key = cat["iter_key"].astype(str)
    is_test = np.char.startswith(iter_key, "testing/")
    label = cat["label"].astype(str)
    cls = np.full(n, "", dtype=object)
    for i in np.flatnonzero(is_test & np.isin(label, ["drone_air", "drone_ground"])):
        side, name = parse_flight(str(cat["best_flight"][i]))
        cls[i] = T.catalog_row(name, side)["classification"]
    known = np.isin(cls, [FIXED, ROTARY])
    y = np.where(cls == FIXED, 1.0, np.where(cls == ROTARY, 0.0, np.nan))
    X = np.column_stack([cat[f] for f in FEATURES])
    if np.isinf(X).any():
        raise SystemExit("infinite feature values")
    deploy = np.isin(cat["sensor_id"], sorted(C.deployable_sensor_ids()))
    air = label == "drone_air"
    D.update(
        keys=keys, test_keys=[k for k in keys if k.startswith("testing/")],
        eval_keys=[k for k in keys if k.startswith("evaluation/")],
        iter_key=iter_key, seg_id=cat["seg_id"].astype(np.int64), sensor_id=cat["sensor_id"].astype(str),
        label=label, cls=cls, y=y, known=known, is_test=is_test, deploy=deploy, air=air,
        ground=label == "drone_ground", w=cat["n_grid"].astype(float), X=X,
        head=is_test & air & known & deploy,           # headline / training rows
        only=is_test & air & known & ~deploy,          # testing-only sensors (side note)
        gnd=is_test & (label == "drone_ground") & known,
    )
    D["rows"] = {k: np.flatnonzero(iter_key == k) for k in keys}
    if np.any(D["y"][D["head"]] != D["y"][D["head"]]):
        raise SystemExit("NaN target in the headline rows")


def train_rows(train_keys, mask: str = "head") -> np.ndarray:
    return np.flatnonzero(np.isin(D["iter_key"], list(train_keys)) & D[mask])


def keys_of(rows: np.ndarray) -> tuple:
    return tuple(sorted(set(D["iter_key"][rows].tolist())))


# --------------------------------------------------------------------------- #
# Methods. Each fit returns a predictor over any row index array.
# --------------------------------------------------------------------------- #


def wnorm(rows: np.ndarray) -> np.ndarray:
    w = D["w"][rows]
    return w / w.mean()


def smoothed_share(rows: np.ndarray, mask: np.ndarray) -> float:
    """Jeffreys-smoothed n_grid-weighted fixed share of rows[mask] (weights mean 1 over rows)."""
    w, y = wnorm(rows)[mask], D["y"][rows][mask]
    return float((np.sum(w * y) + JEFFREYS) / (w.sum() + 1.0))


def rule_sensor_mask(rows: np.ndarray) -> np.ndarray:
    a, b = D["X"][rows, FEATURES.index("cls_uav_fixed")], D["X"][rows, FEATURES.index("cls_uav_rotary")]
    return np.nan_to_num(a, nan=0.0) > np.nan_to_num(b, nan=0.0)


def rule_kin_mask(rows: np.ndarray, A: float, B: float) -> np.ndarray:
    st = D["X"][rows, FEATURES.index("stationary_share")]
    p10 = D["X"][rows, FEATURES.index("speed_p10_ms")]
    with np.errstate(invalid="ignore"):
        return (st <= A) & (p10 >= B)          # NaN compares False: no evidence, not fixed


def search_thresholds(rows: np.ndarray) -> tuple[float, float, float]:
    """Vectorised search over A_GRID x B_GRID on `rows` (the training rows): max n_grid-weighted
    accuracy; ties: smaller weight called fixed, then smaller A, then larger B."""
    w, y = D["w"][rows], D["y"][rows] > 0.5
    st = D["X"][rows, FEATURES.index("stationary_share")]
    p10 = D["X"][rows, FEATURES.index("speed_p10_ms")]
    best = None
    with np.errstate(invalid="ignore"):
        for ia, A in enumerate(A_GRID):
            for ib, B in enumerate(B_GRID):
                m = (st <= A) & (p10 >= B)
                acc = float(np.sum(w[m == y]) / w.sum())
                key = (-round(acc, 12), round(float(np.sum(w[m])), 6), ia, -ib)
                if best is None or key < best[0]:
                    best = (key, float(A), float(B), acc)
    return best[1], best[2], best[3]


def fit_rotary(tr: np.ndarray):
    prior = smoothed_share(tr, np.ones(tr.size, bool))
    return (lambda rows: np.full(rows.size, prior)), keys_of(tr), (), {"prior": prior}


def fit_rule(tr: np.ndarray, kind: str):
    info = {}
    if kind == "rule_sensor_class":
        maskf = rule_sensor_mask
    else:
        A, B, acc = search_thresholds(tr)
        info = {"A": A, "B": B, "train_acc": acc}
        maskf = lambda rows: rule_kin_mask(rows, A, B)  # noqa: E731
    m = maskf(tr)
    p_pos, p_neg = smoothed_share(tr, m), smoothed_share(tr, ~m)
    info.update(p_pos=p_pos, p_neg=p_neg, n_pos=int(m.sum()))

    def predict(rows):
        return np.where(maskf(rows), p_pos, p_neg)
    return predict, keys_of(tr), (), info


def gbdt_fit_predict(cfg_name: str, fit_keys, pred_rows: np.ndarray) -> tuple[np.ndarray, tuple]:
    tr = train_rows(fit_keys)
    m = GB.GBDT(**GBDT_CFG[cfg_name]).fit(D["X"][tr], D["y"][tr], sample_weight=wnorm(tr))
    return m.predict_proba(D["X"][pred_rows]), keys_of(tr)


_PAIR_CACHE: dict = {}


def gbdt_pair(cfg_name: str, a: str, b: str):
    """Raw predictions on the headline rows of a and b from a model fitted on testing \\ {a, b}."""
    key = (cfg_name, a, b)
    if key not in _PAIR_CACHE:
        fit_keys = tuple(x for x in D["test_keys"] if x not in (a, b))
        tr = train_rows(fit_keys)
        m = GB.GBDT(**GBDT_CFG[cfg_name]).fit(D["X"][tr], D["y"][tr], sample_weight=wnorm(tr))
        used = keys_of(tr)
        out = {j: (train_rows((j,)), m.predict_proba(D["X"][train_rows((j,))])) for j in (a, b)}
        _PAIR_CACHE[key] = (out, used)
    return _PAIR_CACHE[key]


def gbdt_fold(cfg_name: str, k: str, pred_rows: np.ndarray):
    """Held-out k: nested isotonic calibration. Returns (p on pred_rows, fit keys, calibrator keys)."""
    TK = D["test_keys"]
    cp, cy, cw, cal_keys = [], [], [], set()
    for j in TK:
        if j == k:
            continue
        a, b = sorted((k, j))
        out, used = gbdt_pair(cfg_name, a, b)
        rows, p = out[j]
        assert k not in used and j not in used
        cp.append(p); cy.append(D["y"][rows]); cw.append(D["w"][rows])
        cal_keys |= set(used) | {j}
    iso = GB.Isotonic().fit(np.concatenate(cp), np.concatenate(cy), np.concatenate(cw))
    raw, fit_keys = gbdt_fit_predict(cfg_name, tuple(x for x in TK if x != k), pred_rows)
    return iso.predict(raw), fit_keys, tuple(sorted(cal_keys))


def gbdt_final(cfg_name: str, pred_rows: np.ndarray, oof_raw: dict):
    """Fit on all 6 testing iterations; calibrator on the pooled 6-fold outer OOF raw predictions."""
    TK = D["test_keys"]
    cp = np.concatenate([oof_raw[k][1] for k in TK])
    cy = np.concatenate([D["y"][oof_raw[k][0]] for k in TK])
    cw = np.concatenate([D["w"][oof_raw[k][0]] for k in TK])
    iso = GB.Isotonic().fit(cp, cy, cw)
    raw, fit_keys = gbdt_fit_predict(cfg_name, tuple(TK), pred_rows)
    return iso.predict(raw), fit_keys, tuple(TK)


def run_method(name: str, k: str, pred_rows: np.ndarray):
    """LOIO outer prediction of `name` for held-out testing iteration k on `pred_rows`.
    Returns (p, info). Records the leakage ledger."""
    TK = D["test_keys"]
    tr_keys = tuple(x for x in TK if x != k)
    if name.startswith("gbdt"):
        p, fit_keys, cal_keys = gbdt_fold(name, k, pred_rows)
        info = {}
    else:
        tr = train_rows(tr_keys)
        if name == "all_rotary":
            pred, fit_keys, cal_keys, info = fit_rotary(tr)
        else:
            pred, fit_keys, cal_keys, info = fit_rule(tr, name)
        p = pred(pred_rows)
    LEDGER.append((name, k, fit_keys, cal_keys))
    return p, info


def run_final(name: str, pred_rows: np.ndarray, oof_raw: dict | None = None):
    TK = D["test_keys"]
    if name.startswith("gbdt"):
        p, fit_keys, cal_keys = gbdt_final(name, pred_rows, oof_raw)
        info = {}
    else:
        tr = train_rows(TK)
        if name == "all_rotary":
            pred, fit_keys, cal_keys, info = fit_rotary(tr)
        else:
            pred, fit_keys, cal_keys, info = fit_rule(tr, name)
        p = pred(pred_rows)
    LEDGER.append((name, "final", fit_keys, cal_keys))
    return p, info


# --------------------------------------------------------------------------- #
# Metrics and stake
# --------------------------------------------------------------------------- #


def wacc(y: np.ndarray, hard: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * (hard == (y > 0.5))) / w.sum()) if w.sum() > 0 else float("nan")


def stake(mode: str) -> list[dict]:
    out = []
    for it in T.testing_iterations():
        tr = T.load_truth(it, mode)
        fcls = np.array([f["classification"] for f in tr.flights])
        cls = fcls[tr.flight]
        n = int(cls.size)
        nf, nr = int(np.sum(cls == FIXED)), int(np.sum(cls == ROTARY))
        nt = int(np.sum(cls == "UAS Transmitter"))
        out.append(dict(it=it.key, n=n, fixed=nf, rotary=nr, other=n - nf - nr - nt, transmitter=nt,
                        fixed_flights=sorted({f["drone_name"] for f in tr.flights
                                              if f["classification"] == FIXED})))
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    t_start = time.time()
    load()
    TK, EK, N = D["test_keys"], D["eval_keys"], D["seg_id"].size
    assert len(TK) == 6 and len(EK) == 3, (TK, EK)
    log(f"segments: {N} ({int(D['is_test'].sum())} testing); headline training rows "
        f"(drone_air, known class, deployable): {int(D['head'].sum())}, "
        f"of which fixed {int(D['y'][D['head']].sum())}")

    # ---- LOIO outer predictions on ALL segments of the held-out iteration
    P = {m: np.full(N, np.nan) for m in METHODS}
    infos = {m: {} for m in METHODS}
    oof_raw = {c: {} for c in GBDT_CFG}        # raw (uncalibrated) outer predictions on headline rows
    for k in TK:
        rows = D["rows"][k]
        for m in METHODS:
            t0 = time.time()
            P[m][rows], infos[m][k] = run_method(m, k, rows)
            log(f"  fold {k} {m:<18s} {time.time() - t0:5.1f} s")
        for c in GBDT_CFG:
            hr = train_rows((k,))
            raw, _ = gbdt_fit_predict(c, tuple(x for x in TK if x != k), hr)
            oof_raw[c][k] = (hr, raw)

    # ---- metrics (hard label p > 0.5)
    def fold_metrics(m: str, mask: np.ndarray) -> dict:
        res = {}
        for k in TK:
            rows = D["rows"][k]
            r = rows[mask[rows]]
            y, p, w = D["y"][r], P[m][r], D["w"][r]
            res[k] = dict(n=r.size, npos=int(y.sum()), w=float(w.sum()), wpos=float(np.sum(w * y)),
                          acc=wacc(y, p > CUT, w) if r.size else float("nan"),
                          auc=GB.auc(y, p, w) if r.size else float("nan"),
                          called=float(np.sum(w * (p > CUT))))
        return res

    M = {m: {"head": fold_metrics(m, D["head"]), "only": fold_metrics(m, D["only"]),
             "gnd": fold_metrics(m, D["gnd"])} for m in METHODS}

    def mean_acc(m, scope="head"):
        return float(np.nanmean([M[m][scope][k]["acc"] for k in TK]))

    def mean_auc(m, scope="head"):
        return float(np.nanmean([M[m][scope][k]["auc"] for k in TK]))

    def pooled_acc(m, scope="head"):
        mk = D[scope] if scope != "gnd" else D["gnd"]
        rows = np.flatnonzero(mk)
        return wacc(D["y"][rows], P[m][rows] > CUT, D["w"][rows])

    order = sorted(range(len(METHODS)), key=lambda i: (-round(mean_acc(METHODS[i]), 12), i))
    rec = METHODS[order[0]]
    if rec != "all_rotary" and not mean_acc(rec) > mean_acc("all_rotary"):
        rec = "all_rotary"
    log("mean held-out weighted accuracy (deployable drone_air): " + ", ".join(
        f"{m} {mean_acc(m):.4f}" for m in METHODS) + f" -> recommended {rec}")

    # ---- final predictions for evaluation + assembly
    rec_final_p = {}
    final_P = {m: {} for m in METHODS}
    final_info = {}
    for m in METHODS:
        for k in EK:
            p, info = run_final(m, D["rows"][k], oof_raw.get(m))
            final_P[m][k] = p
        final_info[m] = info
    for m in METHODS:
        for k in EK:
            P[m][D["rows"][k]] = final_P[m][k]

    # ---- write
    OUT.mkdir(parents=True, exist_ok=True)
    for k in D["keys"]:
        rows = D["rows"][k]
        fold = [k if k in TK else "final"] * rows.size
        C.write_columns(OUT / f"{C.iter_slug(k)}__pclass.csv",
                        {"seg_id": D["seg_id"][rows].tolist(), "p_fixed": P[rec][rows],
                         "method": [rec] * rows.size, "fold": fold},
                        ["seg_id", "p_fixed", "method", "fold"])
        cols = {"seg_id": D["seg_id"][rows].tolist(), "fold": fold}
        cols.update({f"p_{m}": P[m][rows] for m in METHODS})
        C.write_columns(OUT / f"{C.iter_slug(k)}__pclass_methods.csv", cols,
                        ["seg_id"] + [f"p_{m}" for m in METHODS] + ["fold"])

    # ---- stake
    st = {md: stake(md) for md in ("all", "airborne")}

    # ---- report
    lines = ["# Classification: Fixed Wing vs Rotary Wing", "",
             f"Generated by `tools/classify.py`. **Recommended method: `{rec}`.**", "",
             "## What was asked and the short answer", ""]
    st_all = st["all"]
    mean_fixed_known = float(np.mean([s["fixed"] / (s["fixed"] + s["rotary"]) for s in st_all
                                      if s["fixed"] + s["rotary"]]))
    lines += [
        f"* 'classification' is 0.2 of the score. It is a recall over the truth rows of known class "
        f"(Fixed or Rotary). Calling everything Rotary already gets the Rotary rows right; at best a "
        f"classifier can win back the Fixed rows: on average {100 * mean_fixed_known:.1f} % of the known-class "
        f"truth rows per testing iteration (truth mode 'all'), i.e. at most "
        f"{0.2 * mean_fixed_known:.3f} of the combined score at perfect coverage.",
        f"* Mean held-out weighted accuracy on deployable airborne drone segments: "
        + "; ".join(f"`{m}` {mean_acc(m):.3f}" for m in METHODS) + ".",
        f"* Held-out folds where `{rec}` is better / equal / worse than `all_rotary` (weighted accuracy, "
        f"tolerance 0.005): {sum(M[rec]['head'][k]['acc'] > M['all_rotary']['head'][k]['acc'] + 0.005 for k in TK)} / "
        f"{sum(abs(M[rec]['head'][k]['acc'] - M['all_rotary']['head'][k]['acc']) <= 0.005 for k in TK)} / "
        f"{sum(M[rec]['head'][k]['acc'] < M['all_rotary']['head'][k]['acc'] - 0.005 for k in TK)}.",
        f"* Of all Fixed truth rows, {100 * (1 - sum(s['fixed'] for s in st['airborne']) / sum(s['fixed'] for s in st['all'])):.1f} % "
        f"are on the ground (mode 'all' minus 'airborne'): no kinematic method can recover those.",
        f"* Recommended: `{rec}`" + ("" if rec != "all_rotary" else
                                     " (no method is strictly better than 'always Rotary' on the held-out iterations)")
        + ". See the tables for per-iteration numbers and the caveats.", ""]

    lines += ["## The stake: fixed-wing share of the truth rows", "",
              "Rows = (grid step, flight) pairs of the Dronetag truth on the template grid. 'all' includes "
              "drones parked on the ground (very likely part of the official truth), 'airborne' only rows "
              "above the airborne height. 'Unknown' = flights whose class the catalog cannot give (RQ). "
              "`all Rotary recall` = Rotary / (Fixed + Rotary) = the classification recall of the constant "
              "answer 'Rotary' at perfect coverage (every truth row matched by a track). `max gain` = "
              "0.2 x Fixed share of known rows = the most the classification part could add over it.", ""]
    for md in ("all", "airborne"):
        rows_t = []
        for s in st[md]:
            kn = s["fixed"] + s["rotary"]
            rows_t.append([s["it"], s["n"], s["fixed"], s["rotary"], s["other"] + s["transmitter"],
                           f"{100 * s['fixed'] / s['n']:.1f} %", f"{100 * s['rotary'] / s['n']:.1f} %",
                           f"{100 * s['other'] / s['n']:.1f} %",
                           fmt(s["rotary"] / kn if kn else float("nan")),
                           fmt(0.2 * s["fixed"] / kn if kn else float("nan")),
                           ", ".join(s["fixed_flights"]) or "-"])
        tot = {k: sum(s[k] for s in st[md]) for k in ("n", "fixed", "rotary", "other", "transmitter")}
        kn = tot["fixed"] + tot["rotary"]
        mrec = float(np.mean([s["rotary"] / (s["fixed"] + s["rotary"]) for s in st[md]]))
        rows_t.append(["**all 6 pooled / mean of per-iteration**", tot["n"], tot["fixed"], tot["rotary"],
                       tot["other"] + tot["transmitter"], f"{100 * tot['fixed'] / tot['n']:.1f} %",
                       f"{100 * tot['rotary'] / tot['n']:.1f} %", f"{100 * tot['other'] / tot['n']:.1f} %",
                       f"{fmt(tot['rotary'] / kn)} / {fmt(mrec)}",
                       f"{fmt(0.2 * tot['fixed'] / kn)} / {fmt(0.2 * (1 - mrec))}", ""])
        lines += [f"### truth mode '{md}'", ""]
        lines += md_table(["iteration", "truth rows", "Fixed", "Rotary", "Unknown", "Fixed %", "Rotary %",
                           "Unknown %", "all Rotary recall", "max gain (score pts)", "fixed-wing flights"], rows_t)
        lines.append("")
    lines += ["The evaluation iterations have no truth; their fixed-wing share is unknown (only the Blue "
              "whitelist and the sensors' class strings hint at it).", ""]

    lines += ["## Training / headline segments (testing, drone_air, class known)", "",
              "Weighted = by n_grid (grid samples the segment can cover).", ""]
    rows_t = []
    for k in TK:
        h, o, g = (M["all_rotary"][s][k] for s in ("head", "only", "gnd"))
        rows_t.append([k, f"{h['npos']} / {h['n']}", f"{100 * h['wpos'] / h['w']:.1f} %" if h["w"] else "n/a",
                       f"{o['npos']} / {o['n']}", f"{g['npos']} / {g['n']}"])
    hh = np.flatnonzero(D["head"])
    rows_t.append(["all 6", f"{int(D['y'][hh].sum())} / {hh.size}",
                   f"{100 * np.sum(D['w'][hh] * D['y'][hh]) / D['w'][hh].sum():.1f} %",
                   f"{int(D['y'][D['only']].sum())} / {int(D['only'].sum())}",
                   f"{int(D['y'][D['gnd']].sum())} / {int(D['gnd'].sum())}"])
    lines += md_table(["iteration", "fixed / all (deployable drone_air)", "fixed share, weighted",
                       "fixed / all (testing-only sensors, side note)", "fixed / all (drone_ground, all sensors)"],
                      rows_t)
    lines.append("")

    lines += ["## Method comparison (leave-one-iteration-out)", "",
              "Weighted accuracy of the hard label p > 0.5 on deployable drone_air segments of the held-out "
              "iteration. 'mean' = mean of the 6 iteration values (this is what the recommendation uses); "
              "'pooled' = all held-out segments in one pool. S2-I2 has no fixed-wing segment, so any "
              "'fixed' call there is an error and AUC is undefined (n/a).", ""]
    hdr = ["method"] + [k.split("/")[1] for k in TK] + ["mean", "pooled"]
    lines += ["### Accuracy", ""]
    lines += md_table(hdr, [[m] + [fmt(M[m]["head"][k]["acc"]) for k in TK] +
                            [fmt(mean_acc(m)), fmt(pooled_acc(m))] for m in METHODS])
    lines += ["", "### AUC (does the probability rank fixed above rotary? 0.5 = no skill)", ""]
    lines += md_table(["method"] + [k.split("/")[1] for k in TK] + ["mean"],
                      [[m] + [fmt(M[m]["head"][k]["auc"]) for k in TK] + [fmt(mean_auc(m))] for m in METHODS])
    lines += ["", "### Weight called Fixed / weight truly Fixed (deployable drone_air, held-out)", ""]
    lines += md_table(["method"] + [k.split("/")[1] for k in TK],
                      [[m] + [f"{M[m]['head'][k]['called']:.0f} / {M[m]['head'][k]['wpos']:.0f}" for k in TK]
                       for m in METHODS])
    raw_rule = []
    for k in TK:
        r = D["rows"][k]; r = r[D["head"][r]]
        raw_rule.append(wacc(D["y"][r], rule_sensor_mask(r), D["w"][r]))
    lines += ["", f"Raw `rule_sensor_class` label (before the training-share calibration) mean accuracy: "
              f"{fmt(float(np.nanmean(raw_rule)))}.", ""]

    lines += ["### Side: testing-only sensors (EUIG, LJJX, MMJJ), same held-out predictions", ""]
    lines += md_table(["method"] + [k.split("/")[1] for k in TK] + ["pooled"],
                      [[m] + [fmt(M[m]["only"][k]["acc"]) for k in TK] + [fmt(pooled_acc(m, "only"))]
                       for m in METHODS])
    lines += ["", "### Side: drone_ground segments (parked drones; Fixed ones cannot be seen by speed)", ""]
    lines += md_table(["method"] + [k.split("/")[1] for k in TK] + ["pooled"],
                      [[m] + [fmt(M[m]["gnd"][k]["acc"]) for k in TK] + [fmt(pooled_acc(m, "gnd"))]
                       for m in METHODS])
    lines.append("")

    lines += ["## Chosen thresholds and rule shares per fold", ""]
    rows_t = []
    for k in TK:
        a = infos["rule_kinematic"][k]
        rows_t.append([k, f"stationary_share <= {a['A']:g}", f"speed_p10_ms >= {a['B']:g}",
                       fmt(a["train_acc"]), a["n_pos"], fmt(a["p_pos"]), fmt(a["p_neg"]),
                       fmt(infos["rule_sensor_class"][k]["p_pos"]), fmt(infos["all_rotary"][k]["prior"])])
    lines += md_table(["held-out", "A", "B", "train acc", "train rows called fixed", "p if rule fires",
                       "p otherwise", "sensor-class rule: p if fires", "all_rotary prior"], rows_t)
    a = final_info["rule_kinematic"]
    lines += ["", f"Final rule for the evaluation iterations (fitted on all 6 testing): stationary_share <= "
              f"{a['A']:g} and speed_p10_ms >= {a['B']:g}.", ""]

    ev_rows = []
    for k in EK:
        p = P[rec][D["rows"][k]]
        ev_rows.append([k, p.size, int(np.sum(p > CUT)), fmt(float(p.max())), fmt(float(p.mean()))])
    lines += ["## Evaluation predictions (recommended method, fitted on all 6 testing iterations)", ""]
    lines += md_table(["iteration", "segments", "called Fixed (p > 0.5)", "max p", "mean p"], ev_rows)
    lines += ["", "## Caveats", "",
              "* Only 3 fixed-wing drone names and 4 iterations with them: every number rests on a handful "
              "of flights; segments of one flight are strongly correlated, so the effective sample is smaller "
              "than the segment counts.",
              "* The same airframes recur across iterations (Heidrun, Vector, Skydio X10D, FPV ...), so LOIO "
              "tests new flights of known drone types, not unseen types; the evaluation fleet is assumed to "
              "be the same.",
              "* Differences between methods often come from one or two long segments (n_grid weighting): "
              "e.g. gbdt_d2 vs gbdt_d3 differ mainly on one S1-I1 radar segment.",
              "* Radar speeds of rotary drones in forward flight (10-16 m/s, FPV 30-40 m/s) overlap the "
              "fixed-wing 11-20 m/s; hovering is the only clean separator, and a fixed-wing drone parked on "
              "the ground looks like a hovering rotary one.",
              "* VKYY / QTVS (UTM relays) relay the Dronetag feed in testing: optimistic if they appear in "
              "the headline rows. Testing-only sensors are not used to fit anything.",
              "* Selecting the best of 5 methods on the same 6 held-out folds is a mild selection; the "
              "winner's mean is slightly optimistic.",
              "* Inherited: sensor_noise.csv / sensor_tracks variances and height offsets were fitted on "
              "all 6 testing iterations.", "",
              "## Glossary", "",
              "* **Fixed wing / rotary wing**: an aeroplane-like drone that must keep moving / a multirotor "
              "that can hover.",
              "* **Segment**: one sensor's track of one object, cut at 30 s gaps (sensor_tracks.py).",
              "* **n_grid**: number of 100 ms template grid samples the segment covers = how many truth rows it "
              "can possibly win.",
              "* **Recall**: of the truth rows that should be labelled correctly, the share that are.",
              "* **LOIO (leave-one-iteration-out)**: fit on 5 testing iterations, test on the 6th, rotate.",
              "* **Nested**: thresholds / calibrators for held-out k are chosen only from iterations other than "
              "k, using models that never saw k either.",
              "* **Accuracy (weighted)**: share of n_grid weight whose hard label (p > 0.5) is right.",
              "* **AUC**: probability that a random fixed-wing segment gets a higher p than a random rotary one "
              "(0.5 = coin flip, 1 = perfect).",
              "* **Calibration (isotonic)**: monotone remapping of a model score so that p = 0.7 is right about "
              "70 % of the time.",
              "* **GBDT**: gradient-boosted decision trees, many small trees added one after the other.",
              "* **GOSPA**: the track-accuracy metric of the score (0.4 of it); not used here.",
              "* **Deployable sensors**: the sensor ids that also exist in the evaluation data.", ""]
    (OUT / "classify.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- self-checks
    fails = 0

    def check(num, name, ok, detail):
        nonlocal fails
        fails += 0 if ok else 1
        log(f"[{'PASS' if ok else 'FAIL'}] {num}. {name}: {detail}")

    # 1 leakage guard
    bad = []
    for name, held, fit_keys, cal_keys in LEDGER:
        if held in fit_keys or held in cal_keys:
            bad.append(f"{name}: {held} in its own fit/calibrator")
        if any(x in EK for x in fit_keys + cal_keys):
            bad.append(f"{name}: evaluation key in fit/calibrator ({held})")
        if held == "final" and (set(fit_keys) != set(TK)):
            bad.append(f"{name}: final fit not on exactly the 6 testing iterations")
        if held != "final" and not set(fit_keys) <= set(TK) - {held}:
            bad.append(f"{name}: fit keys {fit_keys} for {held}")
    for pk, (out, used) in _PAIR_CACHE.items():
        if set(used) & {pk[1], pk[2]}:
            bad.append(f"pair fit {pk} saw its own predicted iteration")
    for m in METHODS:
        for k in TK:                    # the train mask used for thresholds contains no row of k
            tr = train_rows(tuple(x for x in TK if x != k))
            if np.any(D["iter_key"][tr] == k) or np.any(~D["is_test"][tr]):
                bad.append(f"{m}: train rows of fold {k} contain k or evaluation rows")
    check(1, "leakage guard", not bad,
          f"{len(LEDGER)} prediction sets and {len(_PAIR_CACHE)} inner pair fits: no held-out iteration in its "
          f"own fit / threshold search / calibrator, no evaluation row in any fit" + (f"; {bad[:4]}" if bad else ""))

    # 2 one p_fixed in [0, 1] per segment, read back from the written files
    bad2 = []
    cat_all = {}
    for k in D["keys"]:
        pf = C.read_columns(OUT / f"{C.iter_slug(k)}__pclass.csv", numeric={"seg_id", "p_fixed"})
        pm = C.read_columns(OUT / f"{C.iter_slug(k)}__pclass_methods.csv",
                            numeric={"seg_id"} | {f"p_{m}" for m in METHODS})
        cat_all[k] = (pf, pm)
        rows = D["rows"][k]
        if not np.array_equal(pf["seg_id"].astype(np.int64), D["seg_id"][rows]) or \
                np.unique(pf["seg_id"]).size != rows.size:
            bad2.append(f"{k}: seg_id set")
        for arr, nm in [(pf["p_fixed"], "p_fixed")] + [(pm[f"p_{m}"], f"p_{m}") for m in METHODS]:
            if not (np.all(np.isfinite(arr)) and np.all((arr >= 0) & (arr <= 1))):
                bad2.append(f"{k} {nm}: outside [0, 1] or NaN")
        want = k if k in TK else "final"
        if not np.all(pf["fold"] == want) or not np.all(pf["method"] == rec):
            bad2.append(f"{k}: fold/method column")
    check(2, "exactly one p_fixed in [0, 1] per segment", not bad2,
          f"{N} segments over {len(D['keys'])} iterations (files re-read from disk)"
          + (f"; {bad2[:4]}" if bad2 else ""))

    # 3 hand-rule threshold search == brute force (plain python loops) on one fold
    k3 = TK[-1]
    tr = train_rows(tuple(x for x in TK if x != k3))
    A_v, B_v, acc_v = search_thresholds(tr)
    wv, yv = D["w"][tr].tolist(), (D["y"][tr] > 0.5).tolist()
    sv, pv = D["X"][tr, FEATURES.index("stationary_share")].tolist(), D["X"][tr, FEATURES.index("speed_p10_ms")].tolist()
    tot_w = sum(wv)
    cands = []
    for A in A_GRID.tolist():
        for B in B_GRID.tolist():
            ok_w = fired_w = 0.0
            for i in range(len(wv)):
                fire = (sv[i] == sv[i] and pv[i] == pv[i] and sv[i] <= A and pv[i] >= B)
                ok_w += wv[i] if fire == yv[i] else 0.0
                fired_w += wv[i] if fire else 0.0
            cands.append((round(ok_w / tot_w, 9), -round(fired_w, 3), -A, B, A, B, ok_w / tot_w))
    best_bf = max(cands)
    A_b, B_b, acc_b = best_bf[4], best_bf[5], best_bf[6]
    ok3 = (A_v == A_b and B_v == B_b and abs(acc_v - acc_b) < 1e-9)
    check(3, "hand-rule threshold search == brute force", ok3,
          f"fold {k3}: vectorised (A={A_v:g}, B={B_v:g}, acc {acc_v:.6f}) vs brute force "
          f"(A={A_b:g}, B={B_b:g}, acc {acc_b:.6f}) over {A_GRID.size * B_GRID.size} grid points x {len(wv)} rows")

    # 4 recommended method's mean held-out weighted accuracy >= all_rotary (recomputed from the written files)
    def acc_from_files(col_from_methods: str | None) -> float:
        accs = []
        for k in TK:
            pf, pm = cat_all[k]
            p = pf["p_fixed"] if col_from_methods is None else pm[col_from_methods]
            rows = D["rows"][k]
            mk = D["head"][rows]
            yy, ww = D["y"][rows][mk], D["w"][rows][mk]
            tot, good = 0.0, 0.0
            for yi, wi, pi in zip(yy.tolist(), ww.tolist(), p[mk].tolist()):
                tot += wi
                good += wi if (pi > CUT) == (yi > 0.5) else 0.0
            accs.append(good / tot)
        return float(np.mean(accs))
    a_rec, a_rot = acc_from_files(None), acc_from_files("p_all_rotary")
    check(4, "recommended method >= all Rotary (mean held-out weighted accuracy)",
          a_rec >= a_rot - 1e-12 and abs(a_rec - mean_acc(rec)) < 1e-9,
          f"recommended '{rec}' {a_rec:.4f} vs all_rotary {a_rot:.4f} (recomputed from the written csv; "
          + ", ".join(f"{m} {mean_acc(m):.4f}" for m in METHODS) + ")")

    # 5 determinism: a second, cache-free fit of one fold gives identical predictions
    _PAIR_CACHE.clear()
    k5 = TK[2]
    rows5 = D["rows"][k5]
    same = []
    for m in METHODS:
        p2, _ = run_method(m, k5, rows5)
        same.append(np.array_equal(p2, P[m][rows5]))
    check(5, "determinism", all(same),
          f"second fit of fold {k5} for {', '.join(METHODS)} bit-identical: {same}")

    # 6 metric sanity: all_rotary accuracy == 1 - weighted fixed share, from plain loops on the files
    bad6 = []
    for k in TK:
        d = C.read_columns(TRACKS / f"{C.iter_slug(k)}__tracks.csv", numeric={"n_grid"},
                           usecols=["n_grid", "label", "best_flight", "sensor_id"])
        tw = fw = 0.0
        for lab, bf, sid, ng in zip(d["label"], d["best_flight"], d["sensor_id"], d["n_grid"].tolist()):
            if lab != "drone_air" or sid not in C.deployable_sensor_ids():
                continue
            side, name = parse_flight(str(bf))
            c = T.catalog_row(name, side)["classification"]
            if c not in (FIXED, ROTARY):
                continue
            tw += ng
            fw += ng if c == FIXED else 0.0
        if abs((1 - fw / tw) - M["all_rotary"]["head"][k]["acc"]) > 1e-9:
            bad6.append((k, 1 - fw / tw, M["all_rotary"]["head"][k]["acc"]))
    check(6, "all_rotary accuracy == 1 - fixed weight share (independent loop)", not bad6,
          "6 folds" + (f"; {bad6}" if bad6 else ""))

    log(f"done in {time.time() - t_start:.0f} s; report {OUT / 'classify.md'}")
    log(f"{'ALL SELF-CHECKS PASSED' if not fails else str(fails) + ' SELF-CHECK(S) FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
