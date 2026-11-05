#!/usr/bin/env python3
"""Tier models: one existence model per CUMULATIVE attribute tier of the field ladder, the
importance ladder (what each tier of report fields adds), and a ROUTED model (each segment
scored by the richest tier its sensor supports) compared with the single model of
tools/existence.py, leave-one-iteration-out (LOIO) over the 6 testing iterations.

The professor's proposal: the field ladder (tools/eval_tiers.py) sorts the SAPIENT report
fields by the share of sensors that report them (tier 1 = every sensor ... tier 13 = one
sensor). Build one model on the fields every sensor has, a second one on those plus the next
most common group, and so on; the gain from one model to the next measures how much that
field group is worth, and each sensor can then be scored by the richest model its own
fields allow.

Feature -> raw field -> tier (written to tier_features.csv)
-----------------------------------------------------------
Every segment feature of data/derived/sensor_tracks_v2/<slug>__tracks.csv used by
existence.FEATURES or classify.FEATURES is mapped to the raw report fields it is computed from
(FEATURE_FIELDS below). A requirement is a list of groups; every group is needed, and within a
group ANY ONE alternative is enough (written 'a | b'). tier(feature) = max over the groups of
the min over the group's alternatives of the field's tier (data/derived/eval/field_ladder.csv,
column 'tier', or the content tier below). Support is tier-aware, so the min / any pair is
consistent: at tier t a group only counts as available through an alternative of tier <= t.
  tier 1  timestamp, objectId, nodeId: n_meas, n_rows, n_dup_merged, n_grid, duration_s,
          meas_rate_hz (how many reports, how long); gt_feed and the sensor one-hot
          (which sensor; nodeId)
  tier 2  classification: cls_* shares, cls_conf_mean
  tier 3  location_x / location_y (+ time): path_len_m, speed_*, accel_p90_ms2, turn_p90_degs,
          stationary_share; + sensor_location_x / y: range_*
  tier 4  location_z: has_u, u_med_m, u_std_m, vu_abs_p90_ms (classify); + sensor_location_z: dz_*
  tier 7  detectionConfidence: det_conf_mean
  content has_protocol <- protocol_value, has_serial <- serial_value, has_model <- model_value,
          has_freq <- freq_value (CONTENT below; tiers 9, 10, 12, 12 on the current data)
Content fields (fix of the 2026-10-29 review). The container columns id / objectInfo /
trackInfo / additionalInformation / signal are in many headers, but what matters for has_* is
whether they carry a serial / protocol / model / frequency: EGOI's objectInfo holds rcs / speed /
az, NYTF's and IUKL's trackInfo speed / az / trackId, so their has_* are 0 on every segment.
A header-presence rule granted them those features (constant 0) and routed 93 % of the
segments to a model whose extra 'fields' were empty. So each has_* feature needs a CONTENT
field (`<x>_value`): sensor s carries it iff its detections (data/derived/detections.csv.gz,
built by build_detections.py from exactly those containers) have a non-empty value in EVERY
tree where s has rows (IKJX has serials in evaluation only: not carried). The content field is
placed on the ladder by its own share, the way the ladder places a column: share = evaluation
sensors carrying it / evaluation sensors with detections; tier = the ladder tier whose pct_pdf
is nearest to that share (ties -> the higher, rarer tier), and never below the lowest tier of
its containers. Check 3 recomputes this independently.
F_t = the features with tier <= t (cumulative). Only tiers that change the feature set are
candidates. No feature uses the location errors, taskId, enuVelocity, rangeBearing, state,
behaviour or the associated / derived detections, so tiers 5, 6, 8, 11, 13 add nothing.

Which sensor supports which tier
--------------------------------
Sensor s supports tier t iff every feature of F_t is available from s's fields at tier t (each
group of each feature has an alternative of tier <= t that s reports / carries). Fields per
sensor: the 'sensors_data' column of field_ladder.csv (the evaluation CSV headers); sensors
absent from it (testing-only EUIG, LJJX, MMJJ) take their testing CSV headers; for sensors in
both trees the testing headers are compared with the ladder and differences are reported; plus
the content fields above. The same content principle holds for every field: field F is VOID for s
(removed from its effective fields) iff no feature computed from F (gt_feed, cls_blank and the
one-hot are identity, not content) has a finite non-zero value on ANY of s's segments, testing or
evaluation (void_fields: e.g. sensor_location feeds range_* / dz_*, and the UTM relays QTVS / VKYY
report their own position as 0,0,0; IKJX's classification is always empty). Field level, not
feature level: STUG's has_u (the vertical-filter flag) is 0 on every segment, but its location_z
carries heights (u_med_m, dz_*), so location_z counts. GT-free feature values only, never labels.
Support is CUMULATIVE (strict prefix): level0(s) = the highest
candidate tier t such that s supports every candidate tier <= t. Consequence, stated in the
report: a sensor that lacks one field of a lower tier never uses a rarer field it does carry
(CEQX / ILYS have serials but no detectionConfidence, so they stop below tier 7).
A candidate tier is MODELLED iff (i) some deployable segment sensor supports it and (ii) its
training population (deployable segments of the sensors supporting it) has at least one positive
and one negative segment for y_all and for y_air in EVERY testing iteration: nested LOIO needs
held-out rows of both classes in every fold (added after the review: on the current data the
protocol / serial tiers are STUG alone, 41 testing segments in 4 of 6 iterations). The routing
level(s) = the highest modelled tier <= level0(s).

Existence ladder (per tier t)
-----------------------------
Exactly existence.py's protocol and model family, reusing its data loader (existence.load:
same X matrix, targets y_all / y_air, n_grid, deployable mask), CONFIGS x WEIGHTS combos,
weights(), metrics and calibration helpers:
  * columns: F_t in existence.FEATURES order, then the sensor one-hot of the TRAINING sensors
    of the tier (a one-hot of a sensor without training rows would be a constant column);
  * segments predicted: those of sensors supporting t; segments fitted (training mask): those
    of DEPLOYABLE sensors supporting t. Testing-only sensors (EUIG, LJJX, MMJJ) never enter a
    fit, a combo choice or a calibrator here (RULES: they must never feed evaluation
    predictions); their segments are still predicted and reported separately. existence.py
    keeps them in training: see check 2 for the equivalence test on its configuration;
  * nested LOIO for held-out k: pair fits on testing \\ {k, j} predict j; the combination with
    the lowest mean inner step-weighted logloss (deployable training rows of j) is chosen for
    fold k; refit on testing \\ {k} predicts k; isotonic calibrator fitted on the pooled inner
    out-of-fold predictions of the chosen combo (training rows of the 5 other iterations);
  * p_inner (fold k's calibrator on the pair {k, j} model's prediction of j), exactly as
    existence.py writes it;
  * final model per tier: the combo with the lowest mean 6-fold LOIO metric, fitted on all 6
    testing iterations (training mask), calibrated on its 6-fold outer OOF predictions,
    predicts the evaluation segments. Nothing is fitted, tuned or calibrated on evaluation.
Importance ladder: per target, tier and testing iteration the held-out step-weighted logloss,
AUC, segments / drones, and two gains of tier t over the previous modelled tier t-, both on
the SAME held-out segments (tier t's; tier t-'s segment set contains them, support is a prefix):
  * FIELD gain (the importance of tier t's fields): the ablation model a<t> = F_{t-} fitted on
    tier t's OWN training population, minus the tier-t model. Same rows, same population, only
    the new features differ. When the populations are equal, a<t> IS the tier t- model and is
    not refitted.
  * ROUTING gain: the tier t- model (fitted on its larger population) minus the tier-t model;
    'what scoring these segments with the tier-t model changes', training population included.
Scopes: 'deploy' (headline: deployable sensors), 'deploy_norelay' (deployable minus the GT
relays VKYY, QTVS, whose testing feeds ARE the Dronetag truth), 'testing_only' (EUIG, LJJX,
MMJJ; predicted, never fitted) and 'all'.

Classifier ladder (fixed wing vs rotary)
----------------------------------------
classify.py's labels, rows and gbdt_d3 nested protocol (classify.gbdt_fold: pair fits, isotonic
calibration on the inner OOF), reused by restricting classify.D's columns to F_t and its
headline rows (testing, drone_air, known class, deployable) to sensors supporting t
(classify.FEATURES map to tiers 1, 2, 3, 4, 7). Metrics per held-out iteration: n_grid-weighted
accuracy of p > 0.5, AUC, logloss, the 'all Rotary' accuracy on the same rows, and the gain over
the previous tier on the same rows.

Routed model
------------
Each segment's p = the calibrated held-out prediction of the tier model of level(sensor)
(testing), resp. the final tier model (evaluation). Compared with existence.py's single model
(data/derived/existence/<slug>__p.csv) on the same testing segments per iteration: step-weighted
logloss, AUC, calibration (pooled ECE, mean P vs observed rate). Reported next to it, not part of
the decision: the per-sensor decomposition of each iteration's logloss difference; and the
reference 'sdep' = existence.py's single configuration (all features, NaN-aware) refitted with
the same protocol but WITHOUT the testing-only sensors in training, which separates the effect
of routing (routed vs sdep) from the effect of dropping EUIG / LJJX / MMJJ (sdep vs single).

PRE-REGISTERED DECISION (fixed before the first run, 2026-10-29)
----------------------------------------------------------------
The routed model is recommended to replace the single model iff BOTH
  (a) its held-out step-weighted logloss for y_all on the deployable sensors (scope 'deploy',
      VKYY / QTVS included) is lower than the single model's on >= 4 of the 6 testing
      iterations, AND
  (b) its pooled step-weighted AUC for y_all (deployable segments of the 6 testing iterations
      pooled) is not lower than the single model's by more than 0.01.
Everything else (y_air, without VKYY / QTVS, all sensors, segment-weighted numbers,
calibration) is reported but does not enter the decision. The end-to-end F0 test is a later
step (controller). The rule was not changed by the 2026-10-29 review fix (content fields,
modelled-tier rule, field gain, sdep); that fix changes which model each segment is routed to,
and it was made after seeing the first run's results, which the report states.

Outputs (data/derived/tier_models/)
-----------------------------------
tier_features.csv     feature, raw_fields, tier, field_tiers, used_by (existence / classify), note
sensor_tiers.csv      sensor_id, deployable, testing_only, segments, level_existence (routing
                      level), support_existence (level0), level_classify, fields_source,
                      missing_for_next (fields the sensor lacks for the next modelled tier),
                      unused_features (features the sensor genuinely has above its level)
ladder.csv            model (existence / classify), target, tier, iteration ('pooled' /
                      'mean' rows too), scope, n_seg, n_pos, steps, pos_steps, metrics of the
                      tier model and of the previous tier on the same segments
routed_<slug>__p.csv        seg_id, p_drone_all, p_drone_air, fold, tier (every segment;
                      testing: LOIO, evaluation: 'final')
routed_<slug>__p_inner.csv  testing only, existence.py's p_inner format
tier_models.md        plain-language report

Run: python3 tools/tier_models.py [--jobs 3] [--check2-fold testing/S3-I2]
Numbered self-checks at the end; exit status 1 on any failure.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classify as CL  # noqa: E402
import cuas_common as C  # noqa: E402
import existence as E  # noqa: E402
import gbdt as GB  # noqa: E402

OUT = C.DERIVED / "tier_models"
LADDER_CSV = C.DERIVED / "eval" / "field_ladder.csv"
EXIST_DIR = C.DERIVED / "existence"
RELAYS = ("QTVS", "VKYY")          # deployable GT relays (testing feeds = Dronetag truth)
AUC_TOL = 0.01                     # pre-registered decision (b)
WINS_NEEDED = 4                    # pre-registered decision (a)
CLS_CFG = "gbdt_d3"
CUT = 0.5
MARGIN = 0.01                      # descriptive robustness threshold (report only, not the rule)

TS, OID, NODE = "timestamp", "objectId", "nodeId"
LX, LY, LZ = "location_x", "location_y", "location_z"
SLX, SLY, SLZ = "sensor_location_x", "sensor_location_y", "sensor_location_z"
_T1 = ((TS,), (OID,))
_KIN = ((TS,), (OID,), (LX,), (LY,))
_CLS = (("classification",),)
DET_CSV = C.DERIVED / "detections.csv.gz"
# content fields: name -> (detections.csv.gz column, container report fields build_detections reads)
META_SRC = ("id", "objectInfo", "trackInfo", "additionalInformation")
CONTENT = {
    "protocol_value": ("protocol", META_SRC),
    "serial_value": ("serial", META_SRC),
    "model_value": ("model", ("objectInfo", "trackInfo", "additionalInformation")),
    "freq_value": ("freq_mhz", ("signal",)),
}
FEATURE_FIELDS: dict[str, tuple[tuple[str, ...], ...]] = {
    "n_meas": _T1, "n_rows": _T1, "n_dup_merged": _T1, "n_grid": _T1, "duration_s": _T1,
    "meas_rate_hz": _T1,
    "gt_feed": ((NODE,),), "sensor_<ID>": ((NODE,),),
    **{f: _CLS for f in ("cls_uav", "cls_uav_rotary", "cls_uav_fixed", "cls_uav_transmitter",
                         "cls_air_vehicle", "cls_bird", "cls_manned", "cls_ground", "cls_unknown",
                         "cls_blank", "cls_conf_mean")},
    **{f: _KIN for f in ("path_len_m", "speed_mean_ms", "speed_p10_ms", "speed_p90_ms",
                         "speed_max_ms", "accel_p90_ms2", "turn_p90_degs", "stationary_share")},
    **{f: ((LX,), (LY,), (SLX,), (SLY,)) for f in ("range_med_m", "range_min_m", "range_max_m")},
    "has_u": _T1 + ((LZ,),), "u_med_m": ((LZ,),), "u_std_m": ((LZ,),),
    "vu_abs_p90_ms": _KIN + ((LZ,),),
    **{f: ((LZ,), (SLZ,)) for f in ("dz_med_m", "dz_p10_m", "dz_p90_m", "dz_std_m")},
    "det_conf_mean": (("detectionConfidence",),),
    "has_serial": (("serial_value",),), "has_protocol": (("protocol_value",),),
    "has_model": (("model_value",),), "has_freq": (("freq_value",),),
}

# filled before the worker pool forks
SPECS: dict = {}
CFG: dict = {}


def log(msg: str) -> None:
    print(msg, flush=True)


def fmt(v, nd=3) -> str:
    return "n/a" if v is None or v != v else f"{v:.{nd}f}"


def sgn(v, nd=4) -> str:
    return "n/a" if v is None or v != v else f"{v:+.{nd}f}"


md_table = E.md_table


# --------------------------------------------------------------------------- #
# Field ladder, feature tiers, sensor support
# --------------------------------------------------------------------------- #


def read_ladder() -> tuple[dict, dict, dict]:
    with LADDER_CSV.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    tier = {r["field"]: int(r["tier"]) for r in rows}
    have = {r["field"]: set(r["sensors_data"].split()) for r in rows}
    tier_pct: dict = {}
    for r in rows:
        tier_pct.setdefault(int(r["tier"]), set()).add(float(r["pct_pdf"]))
    if any(len(v) != 1 for v in tier_pct.values()):
        raise SystemExit(f"field_ladder.csv: a tier with several pct_pdf values: {tier_pct}")
    return tier, have, {t: v.pop() for t, v in tier_pct.items()}


def content_counts() -> dict:
    """{(tree, sensor): {'rows': n, <detections column>: rows with a value}} for the CONTENT
    columns, streamed from detections.csv.gz."""
    cols = sorted({c for c, _ in CONTENT.values()})
    out: dict = {}
    with C._open_text(DET_CSV, "r") as fh:
        rd = csv.reader(fh)
        h = next(rd)
        it, il, ic = h.index("tree"), h.index("sensor_id"), [h.index(c) for c in cols]
        for row in rd:
            d = out.setdefault((row[it], row[il]), dict.fromkeys(["rows"] + cols, 0))
            d["rows"] += 1
            for c, i in zip(cols, ic):
                v = row[i]
                if v and v.lower() != "nan":
                    d[c] += 1
    return out


def content_fields(counts: dict, ftier: dict, tier_pct: dict) -> tuple[dict, dict, dict]:
    """Tier of every content field, carriers per sensor, and a description per field."""
    eval_s = sorted({s for (tr, s) in counts if tr == "evaluation"})
    trees = {}
    for (tr, s), d in counts.items():
        if d["rows"]:
            trees.setdefault(s, set()).add(tr)
    ctier, carriers, info = {}, {}, {}
    for name, (col, cont) in CONTENT.items():
        n = sum(1 for s in eval_s if counts[("evaluation", s)][col] > 0)
        share = 100.0 * n / len(eval_s)
        t_share = min(tier_pct, key=lambda t: (abs(tier_pct[t] - share), -t))
        t_cont = min(ftier[c] for c in cont if c in ftier)
        ctier[name] = max(t_share, t_cont)
        carriers[name] = {s for s in trees if all(counts[(tr, s)][col] > 0 for tr in trees[s])}
        partial = sorted(s for s in trees if s not in carriers[name]
                         and any(counts[(tr, s)][col] > 0 for tr in trees[s]))
        info[name] = dict(col=col, containers=cont, n_eval=n, n_eval_sensors=len(eval_s), share=share,
                          tier_share=t_share, tier_containers=t_cont,
                          eval_sensors=sorted(s for s in eval_s if counts[("evaluation", s)][col] > 0),
                          carriers=sorted(carriers[name]), partial=partial)
    return ctier, carriers, info


def feature_tier(groups, ftier: dict) -> tuple[int, list]:
    per = []
    for g in groups:
        ts = [ftier[a] for a in g if a in ftier]
        if not ts:
            raise SystemExit(f"requirement group {g} has no field in the ladder")
        per.append(min(ts))
    return max(per), per


def fields_str(groups) -> str:
    return " + ".join(" | ".join(g) for g in groups)


def parse_fields(s: str):
    return tuple(tuple(a.strip() for a in g.split("|")) for g in s.split("+"))


def available(groups, fields: set, ftier: dict | None = None, t: int | None = None) -> bool:
    """Every group has an alternative the sensor reports; with t, of tier <= t."""
    return all(any(a in fields and (t is None or ftier.get(a, 99) <= t) for a in g) for g in groups)


def sensor_fields(ftier: dict, have: dict, carriers: dict) -> tuple[dict, dict, list]:
    """{sensor: set of ladder fields} from the ladder's data-header column, testing headers
    for sensors missing there, plus the content fields the sensor carries; the disagreements
    between testing headers and the ladder."""
    headers = {"testing": {}, "evaluation": {}}
    for it in C.iterations():
        for sf in C.sensor_files(it):
            h, _ = C.iter_rows(sf.path)
            headers[it.tree].setdefault(sf.sensor_id, set()).update(x for x in h if x)
    from_ladder: dict = {}
    for f, ss in have.items():
        for s in ss:
            from_ladder.setdefault(s, set()).add(f)
    fields, source, diffs = {}, {}, []
    for s in sorted(set(headers["testing"]) | set(headers["evaluation"])):
        if s in from_ladder:
            fields[s] = set(from_ladder[s])
            source[s] = "field_ladder.csv (evaluation headers)"
            if s in headers["testing"]:
                t = {f for f in headers["testing"][s] if f in ftier}
                if t != fields[s]:
                    diffs.append(f"{s}: testing-only fields {sorted(t - fields[s])}, "
                                 f"ladder-only {sorted(fields[s] - t)}")
        else:
            fields[s] = {f for f in headers["testing"].get(s, set()) | headers["evaluation"].get(s, set())
                         if f in ftier}
            source[s] = "testing headers (not in the ladder)"
        fields[s] |= {name for name, ss in carriers.items() if s in ss}
    return fields, source, diffs


def feature_table(ftier: dict) -> list[dict]:
    missing = [f for f in list(E.FEATURES) + list(CL.FEATURES) if f not in FEATURE_FIELDS]
    if missing:
        raise SystemExit(f"features without a field mapping: {missing}")
    used = {f: [] for f in FEATURE_FIELDS}
    for f in E.FEATURES:
        used[f].append("existence")
    for f in CL.FEATURES:
        used[f].append("classify")
    used["sensor_<ID>"].append("existence (one-hot)")
    out = []
    for f, groups in FEATURE_FIELDS.items():
        t, per = feature_tier(groups, ftier)
        note = "; ".join(f"{a} = a value in {' | '.join(CONTENT[a][1])} (content field, tier from its "
                         f"share of sensors)" for g in groups for a in g if a in CONTENT)
        out.append(dict(feature=f, raw_fields=fields_str(groups), tier=t,
                        field_tiers=" + ".join("|".join(f"{a}:{ftier.get(a, 'n/a')}" for a in g)
                                               for g in groups),
                        used_by=" ".join(used[f]), note=note))
    return out


NOT_CONTENT = ("gt_feed", "cls_blank", "sensor_<ID>")   # identity flags / 'no classification' share


def void_fields(tracks_dir: Path) -> dict:
    """{sensor: report fields that carry no information for it}: field F is void for sensor s iff
    no feature computed from F (NOT_CONTENT excluded) has a finite non-zero value on any of s's
    segments (both trees). A field can be in the header and still be empty, or voided by the
    cleaning rules (a placeholder sensor position voids range / dz)."""
    feats = [f for f in FEATURE_FIELDS if f not in NOT_CONTENT]
    by_field: dict = {}
    for f in feats:
        for g in FEATURE_FIELDS[f]:
            for a in g:
                by_field.setdefault(a, []).append(f)
    info: dict = {}
    for it in C.iterations():
        d = C.read_columns(tracks_dir / f"{C.iter_slug(it.key)}__tracks.csv", numeric=set(feats),
                           usecols=["sensor_id"] + feats)
        sid = d["sensor_id"].astype(str)
        for s in np.unique(sid):
            m = sid == s
            for a, fs in by_field.items():
                v = np.column_stack([d[f][m] for f in fs])
                if np.any(np.isfinite(v) & (v != 0)):
                    info.setdefault(str(s), set()).add(a)
    return {s: set(by_field) - got for s, got in info.items()}


def support_levels(feats: list[str], ftab: dict, ftier: dict, sfields: dict, sensors) -> tuple[list, dict]:
    """Candidate tiers of a feature list and level0 of every sensor: the highest candidate
    tier t such that the sensor supports every candidate tier <= t (strict prefix). sfields:
    the EFFECTIVE fields (void fields removed)."""
    tiers = sorted({ftab[f] for f in feats})
    lev0 = {}
    for s in sensors:
        lv = 0
        for t in tiers:
            if not all(available(FEATURE_FIELDS[f], sfields[s], ftier, t) for f in feats if ftab[f] <= t):
                break
            lv = t
        lev0[s] = lv
    return tiers, lev0


def route_level(lev0: dict, modelled: list) -> dict:
    return {s: max([t for t in modelled if t <= lv], default=0) for s, lv in lev0.items()}


# --------------------------------------------------------------------------- #
# Generic nested-LOIO protocol (existence.py's), parameterised by columns and masks
# --------------------------------------------------------------------------- #


def make_spec(name, cols, seg, train, folds, final, targets=("all", "air")) -> dict:
    D = E.D
    return dict(name=name, cols=list(cols), seg=seg, train=train & seg, folds=list(folds),
                final=final, targets=tuple(targets), X=np.ascontiguousarray(D["X"][:, cols]),
                rows={k: D["rows"][k][seg[D["rows"][k]]] for k in D["keys"]},
                names=[D["names"][i] for i in cols])


def stage1_tasks(spec) -> list[tuple]:
    TK = E.D["test_keys"]
    pairs = sorted({tuple(sorted((k, j))) for k in spec["folds"] for j in TK if j != k})
    tasks = []
    for t in spec["targets"]:
        for c in E.COMBOS:
            for a, b in pairs:
                tasks.append((spec["name"], t, c, tuple(x for x in TK if x not in (a, b)), (a, b)))
            for k in spec["folds"]:
                tasks.append((spec["name"], t, c, tuple(x for x in TK if x != k), (k,)))
    return tasks


def task_cost(task) -> float:
    S = SPECS[task[0]]
    n = sum(int(S["train"][E.D["rows"][k]].sum()) for k in task[3])
    return n * len(S["cols"]) * {"d3": 1.0, "d5": 2.2, "d4s": 2.9}.get(task[2][0], 2.0)


def fit_task(task: tuple) -> dict:
    name, target, combo, train_keys, pred_keys = task
    S, D = SPECS[name], E.D
    rows = np.concatenate([D["rows"][k] for k in sorted(train_keys)])
    rows = rows[S["train"][rows]]
    m = GB.GBDT(**CFG[combo[0]]).fit(S["X"][rows], D["y"][target][rows],
                                     sample_weight=E.weights(rows, combo[1]))
    return {"task": task, "fitted_on": tuple(sorted(set(D["iter_key"][rows].tolist()))),
            "fitted_sensors": tuple(sorted(set(D["sensor_id"][rows].tolist()))),
            "n_fit": int(rows.size),
            "pred": {k: m.predict_proba(S["X"][S["rows"][k]]) for k in pred_keys},
            "gain": m.to_dict()["gain"] if set(train_keys) == set(D["test_keys"]) else None}


def run_tasks(tasks: list[tuple], jobs: int, label: str) -> dict:
    tasks = sorted(tasks, key=lambda t: -task_cost(t))
    out, t0, n = {}, time.time(), len(tasks)
    step = max(1, n // 10)
    if jobs <= 1:
        it = map(fit_task, tasks)
        pool = None
    else:
        pool = mp.get_context("fork").Pool(jobs)
        it = pool.imap_unordered(fit_task, tasks, chunksize=1)
    try:
        for i, r in enumerate(it, 1):
            out[r["task"]] = r
            if i % step == 0 or i == n:
                el = time.time() - t0
                log(f"  {label}: {i}/{n} fits, {el:.0f} s elapsed")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return out


def assemble(spec, R) -> dict:
    """existence.py's main loop for one spec: per target and fold the inner choice, the
    calibrator, the calibrated outer prediction, p_inner; final combo choice."""
    D, TK = E.D, E.D["test_keys"]
    dep, g, N = D["deploy"], D["n_grid"], D["X"].shape[0]
    name = spec["name"]
    res = {"p_cal": {}, "p_raw": {}, "p_inner": {}, "sel": {}, "inner": {}, "guard": [],
           "final_choice": {}}

    def pair(t, c, k, j):
        a, b = sorted((k, j))
        r = R[(name, t, c, tuple(x for x in TK if x not in (a, b)), (a, b))]
        return r, r["pred"][j]

    def outer(t, c, k):
        r = R[(name, t, c, tuple(x for x in TK if x != k), (k,))]
        return r, r["pred"][k]

    for t in spec["targets"]:
        y = D["y"][t]
        p_cal, p_raw = np.full(N, np.nan), np.full(N, np.nan)
        for k in spec["folds"]:
            inner = {}
            for c in E.COMBOS:
                vals = []
                for j in TK:
                    if j == k:
                        continue
                    r, pj = pair(t, c, k, j)
                    rows = spec["rows"][j]
                    mk = spec["train"][rows] & dep[rows]
                    vals.append(GB.logloss(y[rows][mk], pj[mk], g[rows][mk]))
                inner[c] = float(np.mean(vals))
            res["inner"][(t, k)] = inner
            best = min(E.COMBOS, key=lambda c: (inner[c], E.COMBOS.index(c)))
            res["sel"][(t, k)] = best
            cp, cy, cw, cal_on = [], [], [], set()
            for j in TK:
                if j == k:
                    continue
                r, pj = pair(t, best, k, j)
                rows = spec["rows"][j]
                mk = spec["train"][rows]
                cal_on |= set(r["fitted_on"]) | {j}
                cp.append(pj[mk]); cy.append(y[rows][mk])
                cw.append(np.ones(int(mk.sum())) if best[1] == "unit" else g[rows][mk])
            iso = GB.Isotonic().fit(np.concatenate(cp), np.concatenate(cy), np.concatenate(cw))
            for j in TK:
                if j != k:
                    res["p_inner"][(t, k, j)] = (spec["rows"][j], iso.predict(pair(t, best, k, j)[1]))
            r, pk = outer(t, best, k)
            rows = spec["rows"][k]
            p_raw[rows] = pk
            p_cal[rows] = iso.predict(pk)
            res["guard"].append((name, t, k, r["fitted_on"], tuple(sorted(cal_on))))
        res["p_cal"][t], res["p_raw"][t] = p_cal, p_raw
        if spec["final"]:
            loio = {}
            for c in E.COMBOS:
                vals = []
                for k in TK:
                    _, pk = outer(t, c, k)
                    rows = spec["rows"][k]
                    mk = spec["train"][rows] & dep[rows]
                    vals.append(GB.logloss(y[rows][mk], pk[mk], g[rows][mk]))
                loio[c] = float(np.mean(vals))
            res["final_choice"][t] = (min(E.COMBOS, key=lambda c: (loio[c], E.COMBOS.index(c))), loio)
    return res


def final_tasks(spec, res) -> list[tuple]:
    D = E.D
    return [(spec["name"], t, res["final_choice"][t][0], tuple(D["test_keys"]), tuple(D["eval_keys"]))
            for t in spec["targets"]]


def assemble_final(spec, res, R, RF) -> None:
    D, TK, EK = E.D, E.D["test_keys"], E.D["eval_keys"]
    g = D["n_grid"]
    res["final"] = {}
    for t in spec["targets"]:
        c = res["final_choice"][t][0]
        cp, cy, cw = [], [], []
        for k in TK:
            pk = R[(spec["name"], t, c, tuple(x for x in TK if x != k), (k,))]["pred"][k]
            rows = spec["rows"][k]
            mk = spec["train"][rows]
            cp.append(pk[mk]); cy.append(D["y"][t][rows][mk])
            cw.append(np.ones(int(mk.sum())) if c[1] == "unit" else g[rows][mk])
        iso = GB.Isotonic().fit(np.concatenate(cp), np.concatenate(cy), np.concatenate(cw))
        r = RF[(spec["name"], t, c, tuple(TK), tuple(EK))]
        res["final"][t] = dict(combo=c, fitted_on=r["fitted_on"], gain=r["gain"],
                               p_eval={k: (spec["rows"][k], iso.predict(r["pred"][k])) for k in EK})
        for k in EK:
            res["guard"].append((spec["name"], t, k, r["fitted_on"], tuple(TK)))


# --------------------------------------------------------------------------- #
# Classifier ladder (classify.py's gbdt_d3 nested protocol on restricted columns / rows)
# --------------------------------------------------------------------------- #


def classify_ladder(ftab: dict, ftier: dict, sfields: dict) -> dict:
    CL.load()
    D = CL.D
    TK = D["test_keys"]
    sensors = sorted(set(D["sensor_id"].tolist()))
    cand, lev0 = support_levels(CL.FEATURES, ftab, ftier, sfields, sensors)
    tiers = [t for t in cand if any(lev0[s] >= t for s in sensors)]
    lev = route_level(lev0, tiers)
    X0, head0 = D["X"].copy(), D["head"].copy()
    lev_row = np.array([lev[s] for s in D["sensor_id"]])
    out = {"tiers": tiers, "lev": lev, "p": {}, "head": {}, "guard": [], "cols": {}}
    try:
        for t in tiers:
            cols = [i for i, f in enumerate(CL.FEATURES) if ftab[f] <= t]
            D["X"] = X0[:, cols]
            D["head"] = head0 & (lev_row >= t)
            CL._PAIR_CACHE.clear()
            p = np.full(D["seg_id"].size, np.nan)
            for k in TK:
                rows = D["rows"][k]
                p[rows], fit_keys, cal_keys = CL.gbdt_fold(CLS_CFG, k, rows)
                out["guard"].append((f"classify t{t}", k, fit_keys, cal_keys))
            for (cfg, a, b), (_, used) in CL._PAIR_CACHE.items():
                out["guard"].append((f"classify t{t} pair", (a, b), used, ()))
            out["p"][t], out["head"][t], out["cols"][t] = p, D["head"].copy(), [CL.FEATURES[i] for i in cols]
        # determinism probe (check 5): refit one fold of the top tier without the pair cache
        t = tiers[-1]
        cols = [i for i, f in enumerate(CL.FEATURES) if ftab[f] <= t]
        D["X"], D["head"] = X0[:, cols], head0 & (lev_row >= t)
        CL._PAIR_CACHE.clear()
        k = TK[2]
        p2, _, _ = CL.gbdt_fold(CLS_CFG, k, D["rows"][k])
        out["det"] = (t, k, bool(np.array_equal(p2, out["p"][t][D["rows"][k]])))
    finally:
        D["X"], D["head"] = X0, head0
        CL._PAIR_CACHE.clear()
    return out


def classify_metrics(cl: dict) -> list[dict]:
    D = CL.D
    TK = D["test_keys"]
    rows_out = []
    prev = None
    for t in cl["tiers"]:
        p, head = cl["p"][t], cl["head"][t]
        pooled = {"y": [], "p": [], "w": [], "pp": []}
        for k in TK + ["pooled"]:
            if k == "pooled":
                y, pk, w = (np.concatenate(pooled[x]) for x in ("y", "p", "w"))
                pp = np.concatenate(pooled["pp"]) if prev is not None else None
            else:
                r = D["rows"][k]
                r = r[head[r]]
                y, pk, w = D["y"][r], p[r], D["w"][r]
                pp = cl["p"][prev][r] if prev is not None else None
                for x, v in (("y", y), ("p", pk), ("w", w)):
                    pooled[x].append(v)
                if pp is not None:
                    pooled["pp"].append(pp)
            m = cls_metric(y, pk, w)
            row = dict(model="classify", target="fixed_wing", tier=t, iteration=k, scope="deploy_air_known",
                       n_seg=int(y.size), n_pos=int(y.sum()), steps=float(w.sum()),
                       pos_steps=float(np.sum(w * y)), **m,
                       acc_all_rotary=float(np.sum(w * (y < 0.5)) / w.sum()) if w.sum() else np.nan)
            if pp is not None:
                mp_ = cls_metric(y, pp, w)
                row.update({f"prev_{a}": b for a, b in mp_.items()}, prev_tier=prev)
            rows_out.append(row)
        prev = t
    return rows_out


def cls_metric(y, p, w) -> dict:
    if y.size == 0:
        return dict(acc=np.nan, auc_step=np.nan, ll_step=np.nan)
    return dict(acc=float(np.sum(w * ((p > CUT) == (y > 0.5))) / w.sum()),
                auc_step=GB.auc(y, p, w), ll_step=GB.logloss(y, p, w))


# --------------------------------------------------------------------------- #
# Metrics helpers
# --------------------------------------------------------------------------- #


def scope_masks() -> dict:
    D = E.D
    sid = D["sensor_id"]
    dep = D["deploy"]
    return {"deploy": dep, "deploy_norelay": dep & ~np.isin(sid, RELAYS),
            "testing_only": ~dep, "all": np.ones(sid.size, bool)}


def ex_metric(y, p, g) -> dict:
    if y.size == 0:
        return dict(ll_step=np.nan, auc_step=np.nan, ll_seg=np.nan, auc_seg=np.nan, brier_step=np.nan)
    m = E.metrics(y, p, g)
    return dict(ll_step=m["ll_step"], auc_step=m["auc_step"], ll_seg=m["ll_seg"],
                auc_seg=m["auc_seg"], brier_step=m["brier_step"])


def calib(y, p, g) -> dict:
    ce = E.calib_error(E.calib_bins(y, p, g))
    return dict(ece=ce["ece"], worst=E.bin_name(ce["worst"]), worst_mp=ce["worst"]["mp"],
                worst_obs=ce["worst"]["obs"], mean_p_step=float(np.sum(g * p) / g.sum()),
                obs_step=float(np.sum(g * y) / g.sum()))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--check2-fold", default="testing/S3-I2",
                    help="held-out fold on which existence.py's configuration is re-run (check 2)")
    ap.add_argument("--smoke", action="store_true",
                    help="development only: 5 trees per model (check 2 then fails by design)")
    args = ap.parse_args()
    t_start = time.time()
    CFG.update({c: dict(v) for c, v in E.CONFIGS.items()})
    if args.smoke:
        for v in CFG.values():
            v["n_trees"] = 5

    # ---- field ladder -> feature tiers -> sensor levels
    ftier, have, tier_pct = read_ladder()
    log("content fields: scanning detections.csv.gz ...")
    counts = content_counts()
    ctier, carriers, cinfo = content_fields(counts, ftier, tier_pct)
    ftier = {**ftier, **ctier}
    ftab_rows = feature_table(ftier)
    ftab = {r["feature"]: r["tier"] for r in ftab_rows}
    sfields, ssource, hdr_diffs = sensor_fields(ftier, have, carriers)
    for name, ci in cinfo.items():
        log(f"  {name}: {ci['n_eval']}/{ci['n_eval_sensors']} evaluation sensors ({ci['share']:.1f} %) -> "
            f"tier {ftier[name]}; carried by {' '.join(ci['carriers']) or '-'}"
            + (f"; in one tree only: {' '.join(ci['partial'])}" if ci["partial"] else ""))
    E.load(E.TRACKS_DEFAULT)
    D = E.D
    TK, EK = D["test_keys"], D["eval_keys"]
    assert len(TK) == 6 and len(EK) == 3, (TK, EK)
    if args.check2_fold not in TK:
        raise SystemExit(f"--check2-fold must be one of {TK}")
    seg_sensors = D["sensor_ids"]
    deployable = C.deployable_sensor_ids()
    ex_feats = list(E.FEATURES) + ["sensor_<ID>"]
    void = void_fields(E.TRACKS_DEFAULT)
    sfe = {s: f - void.get(s, set()) for s, f in sfields.items()}     # effective fields
    log("fields present but without information: " + "; ".join(f"{s}: {' '.join(sorted(void[s] & sfields[s]))}" for s in sorted(void) if void[s] & sfields.get(s, set())))
    feat_tiers, lev0 = support_levels(ex_feats, ftab, ftier, sfe, seg_sensors)
    lev0_row = np.array([lev0[s] for s in D["sensor_id"]])
    tiers_all, not_modelled = [], {}
    for t in feat_tiers:
        train = (lev0_row >= t) & D["deploy"]
        if not train.any():
            not_modelled[t] = "no deployable sensor with segments supports it"
            continue
        short_k = [k.split("/")[-1] for k in TK for y in ("all", "air")
                   if len(set(D["y"][y][D["rows"][k]][train[D["rows"][k]]].tolist())) < 2]
        if short_k:
            pop = sorted(set(D["sensor_id"][train].tolist()))
            not_modelled[t] = (f"training population {' '.join(pop)} ({int((train & D['is_test']).sum())} testing "
                               f"segments) lacks a positive or a negative in {', '.join(sorted(set(short_k)))}")
            continue
        tiers_all.append(t)
    lev = route_level(lev0, tiers_all)
    unmodelled = sorted(not_modelled)
    lev_row = np.array([lev[s] for s in D["sensor_id"]])
    N = D["X"].shape[0]
    log(f"segments: {N} ({int(D['is_test'].sum())} testing); existence features by tier: "
        + ", ".join(f"t{t}: {sum(1 for f in ex_feats if ftab[f] == t)}" for t in feat_tiers))
    log(f"modelled tiers {tiers_all}; not modelled: "
        + ("; ".join(f"t{t}: {why}" for t, why in not_modelled.items()) or "none"))
    log("sensor levels (routing / support): " + ", ".join(f"{s} {lev[s]}/{lev0[s]}" for s in seg_sensors))
    log("testing vs ladder header differences: " + ("; ".join(hdr_diffs) or "none"))

    # ---- specs: one per modelled tier, the same-population ablations, the single model without
    #      testing-only sensors (sdep), and existence.py's configuration for check 2
    prev = None
    for t in tiers_all:
        seg = lev_row >= t
        train = seg & D["deploy"]
        train_sensors = sorted(set(D["sensor_id"][train].tolist()))
        onehot = [D["names"].index(f"sensor_{s}") for s in train_sensors]
        cols = [D["names"].index(f) for f in E.FEATURES if ftab[f] <= t] + onehot
        SPECS[f"t{t}"] = make_spec(f"t{t}", cols, seg, train, TK, True)
        if prev is not None and not np.array_equal(SPECS[f"t{t}"]["train"], SPECS[f"t{prev}"]["train"]):
            cols_a = [D["names"].index(f) for f in E.FEATURES if ftab[f] <= prev] + onehot
            SPECS[f"a{t}"] = make_spec(f"a{t}", cols_a, seg, train, TK, False)
        prev = t
    allm = np.ones(N, bool)
    dep_cols = [i for i, n in enumerate(D["names"])
                if not (n.startswith("sensor_") and n[len("sensor_"):] not in deployable)]
    SPECS["sdep"] = make_spec("sdep", dep_cols, allm, D["deploy"], TK, False)
    SPECS["c2"] = make_spec("c2", list(range(D["X"].shape[1])), allm, allm, [args.check2_fold], False)
    tasks = [x for s in SPECS.values() for x in stage1_tasks(s)]
    log(f"stage 1: {len(tasks)} GBDT fits ({', '.join(f'{n}: {len(stage1_tasks(s))}' for n, s in SPECS.items())}) "
        f"on {args.jobs} processes ...")
    R = run_tasks(tasks, args.jobs, "stage 1")
    RES = {n: assemble(s, R) for n, s in SPECS.items()}
    ftasks = [x for n, s in SPECS.items() if s["final"] for x in final_tasks(s, RES[n])]
    log(f"stage 2: {len(ftasks)} final fits ...")
    RF = run_tasks(ftasks, args.jobs, "stage 2")
    for n, s in SPECS.items():
        if s["final"]:
            assemble_final(s, RES[n], R, RF)

    # ---- classifier ladder (main process, small)
    log("classifier ladder ...")
    cl = classify_ladder(ftab, ftier, sfe)
    cl_rows = classify_metrics(cl)

    # ---- routed predictions
    routed = {t: np.full(N, np.nan) for t in ("all", "air")}
    routed_tier = np.zeros(N, int)
    for tt in tiers_all:
        m = (lev_row == tt) & D["is_test"]
        for t in ("all", "air"):
            routed[t][m] = RES[f"t{tt}"]["p_cal"][t][m]
            for k in EK:
                rows, p = RES[f"t{tt}"]["final"][t]["p_eval"][k]
                sel = lev_row[rows] == tt
                routed[t][rows[sel]] = p[sel]
        routed_tier[lev_row == tt] = tt
    routed_inner = {}
    for j in TK:
        for k in TK:
            if k == j:
                continue
            for t in ("all", "air"):
                arr = np.full(N, np.nan)
                for tt in tiers_all:
                    rows, p = RES[f"t{tt}"]["p_inner"][(t, k, j)]
                    sel = lev_row[rows] == tt
                    arr[rows[sel]] = p[sel]
                routed_inner[(t, k, j)] = arr[D["rows"][j]]

    # ---- write routed files
    OUT.mkdir(parents=True, exist_ok=True)
    for k in D["keys"]:
        rows = D["rows"][k]
        cols = {"seg_id": D["seg_id"][rows].tolist(), "p_drone_all": routed["all"][rows],
                "p_drone_air": routed["air"][rows], "fold": [k if k in TK else "final"] * rows.size,
                "tier": routed_tier[rows].tolist()}
        C.write_columns(OUT / f"routed_{C.iter_slug(k)}__p.csv", cols,
                        ["seg_id", "p_drone_all", "p_drone_air", "fold", "tier"])
        if k in TK:
            outs = [x for x in TK if x != k]
            cols = {"seg_id": np.tile(D["seg_id"][rows], len(outs)).tolist(),
                    "outer_fold": np.repeat(outs, rows.size).tolist()}
            for t in ("all", "air"):
                cols[f"p_drone_{t}"] = np.concatenate([routed_inner[(t, o, k)] for o in outs]).tolist()
            C.write_columns(OUT / f"routed_{C.iter_slug(k)}__p_inner.csv", cols,
                            ["seg_id", "outer_fold", "p_drone_all", "p_drone_air"])

    # ---- tier_features.csv / sensor_tiers.csv
    tf_cols = ["feature", "raw_fields", "tier", "field_tiers", "used_by", "note"]
    C.write_columns(OUT / "tier_features.csv", {c: [r[c] for r in ftab_rows] for c in tf_cols}, tf_cols)
    seg_count = {s: int(np.sum(D["sensor_id"] == s)) for s in seg_sensors}
    all_sensors = sorted(sfields)
    lev0_all = support_levels(ex_feats, ftab, ftier, sfe, all_sensors)[1]
    lev_all = route_level(lev0_all, tiers_all)
    cl_lev_all = route_level(support_levels(CL.FEATURES, ftab, ftier, sfe, all_sensors)[1], cl["tiers"])

    def missing_next(s):
        nxt = [t for t in feat_tiers if t > lev0_all[s]]
        if not nxt:
            return ""
        need = sorted({a for f in ex_feats if ftab[f] <= nxt[0] for g in FEATURE_FIELDS[f]
                       if not any(x in sfe[s] and ftier.get(x, 99) <= nxt[0] for x in g) for a in g})
        return f"t{nxt[0]}: " + ", ".join(a + (" (present, no information)" if a in sfields[s] else "")
                                          for a in need)

    def unused(s):
        return " ".join(f for f in ex_feats if ftab[f] > lev_all[s]
                        and available(FEATURE_FIELDS[f], sfe[s], ftier, ftab[f]))

    st_rows = [dict(sensor_id=s, deployable=int(s in deployable), testing_only=int(s not in deployable),
                    segments=seg_count.get(s, 0), level_existence=lev_all[s], support_existence=lev0_all[s],
                    level_classify=cl_lev_all[s], fields_source=ssource[s], missing_for_next=missing_next(s),
                    unused_features=unused(s)) for s in all_sensors]
    C.write_columns(OUT / "sensor_tiers.csv", {c: [r[c] for r in st_rows] for c in st_rows[0]},
                    list(st_rows[0]))

    # ---- ladder metrics
    g = D["n_grid"]
    scopes = scope_masks()
    lad = []
    for t in ("all", "air"):
        y = D["y"][t]
        prev = None
        for tt in tiers_all:
            spec, res = SPECS[f"t{tt}"], RES[f"t{tt}"]
            p = res["p_cal"][t]
            pp = RES[f"t{prev}"]["p_cal"][t] if prev is not None else None
            pa = RES[f"a{tt}"]["p_cal"][t] if f"a{tt}" in RES else pp
            for sc, smask in scopes.items():
                pool_rows = []
                for k in TK:
                    rows = spec["rows"][k]
                    rows = rows[smask[rows]]
                    pool_rows.append(rows)
                    lad.append(ladder_row("existence", t, tt, k, sc, rows, y, p, pp, prev, g, pa))
                rows = np.concatenate(pool_rows)
                lad.append(ladder_row("existence", t, tt, "pooled", sc, rows, y, p, pp, prev, g, pa))
                fold_rows = [r for r in lad if r["model"] == "existence" and r["target"] == t
                             and r["tier"] == tt and r["scope"] == sc and r["iteration"] in TK]
                mean = dict(model="existence", target=t, tier=tt, iteration="mean", scope=sc, prev_tier=prev)
                for key in fold_rows[0]:
                    if key not in mean and key != "iteration":
                        vals = [r[key] for r in fold_rows]
                        mean[key] = float(np.nanmean(vals)) if all(isinstance(v, (int, float)) for v in vals) \
                            and not all(v != v for v in vals) else np.nan
                mean["wins_vs_prev"] = sum(1 for r in fold_rows if r["ll_step"] < r.get("prev_ll_step", np.nan))
                mean["field_wins"] = sum(1 for r in fold_rows if r["ll_step"] < r.get("abl_ll_step", np.nan))
                lad.append(mean)
            prev = tt
    lad += cl_rows
    lad_cols = ["model", "target", "tier", "iteration", "scope", "n_seg", "n_pos", "steps", "pos_steps",
                "ll_step", "auc_step", "ll_seg", "auc_seg", "brier_step", "acc", "acc_all_rotary",
                "prev_tier", "prev_ll_step", "prev_auc_step", "prev_ll_seg", "prev_auc_seg", "prev_acc",
                "gain_ll_step", "gain_auc_step", "wins_vs_prev", "ablation", "abl_ll_step", "abl_auc_step",
                "field_gain_ll_step", "field_gain_auc_step", "field_wins"]
    for r in lad:
        if "prev_ll_step" in r and r.get("prev_ll_step") == r.get("prev_ll_step") and "gain_ll_step" not in r:
            r["gain_ll_step"] = r["prev_ll_step"] - r["ll_step"]
            r["gain_auc_step"] = r["auc_step"] - r["prev_auc_step"]
        if "abl_ll_step" in r and r["abl_ll_step"] == r["abl_ll_step"]:
            r["field_gain_ll_step"] = r["abl_ll_step"] - r["ll_step"]
            r["field_gain_auc_step"] = r["auc_step"] - r["abl_auc_step"]
    C.write_columns(OUT / "ladder.csv",
                    {c: [("" if r.get(c) is None else r.get(c, "")) if not isinstance(r.get(c), float)
                         else ("" if r[c] != r[c] else repr(r[c])) for r in lad] for c in lad_cols}, lad_cols)

    # ---- routed vs single
    single = {}
    for k in D["keys"]:
        pf = C.read_columns(EXIST_DIR / f"{C.iter_slug(k)}__p.csv", numeric={"seg_id", "p_drone_all", "p_drone_air"})
        if not np.array_equal(pf["seg_id"].astype(np.int64), D["seg_id"][D["rows"][k]]):
            raise SystemExit(f"{k}: existence p.csv seg_id does not match tracks.csv")
        single[k] = pf
    p_single = {t: np.concatenate([single[k][f"p_drone_{t}"] for k in D["keys"]]) for t in ("all", "air")}
    order = np.concatenate([D["rows"][k] for k in D["keys"]])
    ps = {t: np.empty(N) for t in p_single}
    for t in p_single:
        ps[t][order] = p_single[t]
    cmp_ = compare(routed, ps, scopes, g)
    dec = decision(cmp_)
    p_sdep = {t: RES["sdep"]["p_cal"][t] for t in ("all", "air")}
    cmp_rd = compare(routed, p_sdep, scopes, g)
    cmp_ds = compare(p_sdep, ps, scopes, g)
    contrib = sensor_contrib(routed["all"], ps["all"], scopes["deploy"], g)
    log(f"routed vs single (y_all, deploy, step logloss): routed wins {dec['wins']} of 6; pooled step AUC "
        f"routed {dec['auc_r']:.4f} vs single {dec['auc_s']:.4f} -> "
        f"{'RECOMMEND routed' if dec['recommend'] else 'keep the single model'}; reference sdep: "
        f"routed beats sdep on {cmp_rd[('all', 'deploy')]['wins']} of 6, sdep beats single on "
        f"{cmp_ds[('all', 'deploy')]['wins']} of 6")

    # ---- self-checks
    fails = 0

    def check(num, name, ok, detail):
        nonlocal fails
        fails += 0 if ok else 1
        log(f"[{'PASS' if ok else 'FAIL'}] {num}. {name}: {detail}")

    # 1 leakage guard
    bad = []
    tier_specs = [n for n in SPECS if n != "c2"]
    for n in SPECS:
        for (_, t, held, fitted_on, cal_on) in RES[n]["guard"]:
            if held in fitted_on or (held in TK and held in cal_on):
                bad.append(f"{n} {t}: {held} in its own fit / calibrator")
            if any(x in EK for x in fitted_on + cal_on):
                bad.append(f"{n} {t}: evaluation in training ({held})")
    for r in list(R.values()) + list(RF.values()):
        task = r["task"]
        if any(x in EK for x in r["fitted_on"]):
            bad.append(f"eval rows in fit {task[:3]}")
        if set(r["fitted_on"]) != set(task[3]):
            bad.append(f"fit {task[:3]} rows {r['fitted_on']} != requested {task[3]}")
        if task in R and set(r["fitted_on"]) & set(task[4]):
            bad.append(f"fit {task[:3]} predicts an iteration it was fitted on")
        if task[0] in tier_specs and set(r["fitted_sensors"]) - set(deployable):
            bad.append(f"fit {task[:3]} used testing-only sensors {set(r['fitted_sensors']) - set(deployable)}")
    for (name, held, fit_keys, cal_keys) in cl["guard"]:
        hk = set(held) if isinstance(held, tuple) else {held}
        if hk & set(fit_keys) or (not isinstance(held, tuple) and held in cal_keys):
            bad.append(f"{name}: {held} in its own fit / calibrator")
        if any(x in EK for x in tuple(fit_keys) + tuple(cal_keys)):
            bad.append(f"{name}: evaluation in training")
    for (t, k, j), arr in routed_inner.items():
        if k == j or not np.all(np.isfinite(arr)):
            bad.append(f"p_inner {t} outer {k} on {j}")
    n_guard = sum(len(RES[n]["guard"]) for n in SPECS) + len(cl["guard"])
    check(1, "leakage guard", not bad,
          f"{len(R) + len(RF)} existence-protocol fits, {n_guard} prediction / calibrator records "
          f"(tiers, check-2 configuration, classifier ladder): no held-out iteration in its own model / "
          f"calibrator / combo choice, no evaluation row in any fit, no testing-only sensor "
          f"({', '.join(sorted(set(seg_sensors) - set(deployable)))}) in any tier fit"
          + (f"; {bad[:5]}" if bad else ""))

    # 2 existence.py's configuration reproduced (one held-out fold, both targets, p and p_inner)
    k2 = args.check2_fold
    det2, worst = [], 0.0
    pf = single[k2]
    for t in ("all", "air"):
        rows = D["rows"][k2]
        d = float(np.max(np.abs(RES["c2"]["p_cal"][t][rows] - pf[f"p_drone_{t}"])))
        worst = max(worst, d)
        det2.append(f"y_{t} p.csv {k2} max|d| {d:.1e}")
    nin = 0
    for j in TK:
        if j == k2:
            continue
        q = C.read_columns(EXIST_DIR / f"{C.iter_slug(j)}__p_inner.csv",
                           numeric={"seg_id", "p_drone_all", "p_drone_air"})
        m = q["outer_fold"] == k2
        sid = q["seg_id"][m].astype(np.int64)
        o = np.argsort(sid, kind="stable")
        if not np.array_equal(sid[o], D["seg_id"][D["rows"][j]]):
            worst = np.inf
            continue
        for t in ("all", "air"):
            rows, p = RES["c2"]["p_inner"][(t, k2, j)]
            worst = max(worst, float(np.max(np.abs(p - q[f"p_drone_{t}"][m][o]))))
            nin += rows.size
    full_tier = max(feat_tiers)
    check(2, "existence.py's protocol reproduced", worst <= 1e-9,
          f"the tier holding every existence feature is t{full_tier} (not modelled: "
          f"{not_modelled.get(full_tier, 'modelled')}), and the tier models exclude testing-only sensors from training, "
          f"so the test runs this code on existence.py's configuration (all {D['X'].shape[1]} columns, "
          f"all segments, all sensors in training) for held-out fold {k2}: "
          + "; ".join(det2) + f"; p_inner (outer_fold {k2}, {nin} values on the other 5 iterations) "
          f"-> worst |d| {worst:.1e} (tolerance 1e-9); chosen combos "
          + ", ".join(f"y_{t} {E.combo_name(RES['c2']['sel'][(t, k2)])}" for t in ("all", "air")))

    # 3 feature -> tier mapping recomputed independently from field_ladder.csv
    bad3 = []
    with LADDER_CSV.open(newline="") as fh:
        lrows = list(csv.DictReader(fh))
    lt = {r["field"]: int(r["tier"]) for r in lrows}
    pct = {int(r["tier"]): float(r["pct_pdf"]) for r in lrows}
    # content tiers / carriers recomputed vectorised from detections.csv.gz (not content_counts)
    dd = C.read_columns(DET_CSV, numeric={"freq_mhz"},
                        usecols=["tree", "sensor_id", "serial", "protocol", "model", "freq_mhz"])
    dtree, dsid = dd["tree"].astype(str), dd["sensor_id"].astype(str)
    ev_s = np.unique(dsid[dtree == "evaluation"])
    car_tree = {}
    for name, (col, cont) in CONTENT.items():
        v = dd[col]
        has = np.isfinite(v) if col == "freq_mhz" else (v.astype(str) != "")
        n = np.unique(dsid[(dtree == "evaluation") & has]).size
        share = 100.0 * n / ev_s.size
        gap = {t: abs(pct[t] - share) for t in pct}
        best = max(t for t in pct if gap[t] == min(gap.values()))
        lt[name] = max(best, min(lt[c] for c in cont if c in lt))
        car = set()
        for s in np.unique(dsid):
            ms = dsid == s
            per_tree = {tr: bool(np.any(has[ms & (dtree == tr)])) for tr in np.unique(dtree[ms])}
            car_tree.update({(name, tr, s): x for tr, x in per_tree.items()})
            if all(per_tree.values()):
                car.add(str(s))
        if car != carriers[name]:
            bad3.append(f"{name}: carriers {sorted(car)} vs {sorted(carriers[name])}")
    del dd, dtree, dsid
    with (OUT / "tier_features.csv").open(newline="") as fh:
        back = list(csv.DictReader(fh))
    for r in back:
        worst_g = 0
        for grp in r["raw_fields"].split("+"):
            alts = [lt[a.strip()] for a in grp.split("|") if a.strip() in lt]
            worst_g = max(worst_g, min(alts) if alts else 99)
        if worst_g != int(r["tier"]):
            bad3.append(f"{r['feature']}: file {r['tier']} vs recomputed {worst_g}")
    if {r["feature"] for r in back} != set(FEATURE_FIELDS):
        bad3.append("feature set of tier_features.csv")
    # data cross-check (a): a sensor lacking a feature's fields (content fields: in that tree) never
    # has an informative (finite, non-zero) value in the tracks
    names = D["names"]
    n_cells = 0
    for f in E.FEATURES:
        col = D["X"][:, names.index(f)]
        groups = parse_fields(next(r["raw_fields"] for r in back if r["feature"] == f))
        for s in seg_sensors:
            for tr in ("testing", "evaluation"):
                ftr = {a for a in sfields[s] if a not in CONTENT} | \
                      {a for a in CONTENT if car_tree.get((a, tr, s), False)}
                if available(groups, ftr):
                    continue
                v = col[(D["sensor_id"] == s) & (D["is_test"] == (tr == "testing"))]
                n_cells += v.size
                if np.any(np.isfinite(v) & (v != 0)):
                    bad3.append(f"{f} informative for {s} ({tr}) without its fields")
    # (b) the converse, on the tier models' own matrices: every field a tier model's features use carries
    # information (some feature from it has a finite non-zero value) on at least one segment of EVERY
    # sensor the model scores (a field present in the header but always empty / voided, or a content
    # field whose has_* flag is always 0, fails here); identity flags are not content
    n_b = 0
    for n_, S in SPECS.items():
        if not n_.startswith("t"):
            continue
        grp_cols: dict = {}
        for ci, f in enumerate(S["names"]):
            if f in E.FEATURES and f not in NOT_CONTENT:
                for a in {x for g in FEATURE_FIELDS[f] for x in g}:
                    grp_cols.setdefault(a, []).append(ci)
        sid_seg = D["sensor_id"][S["seg"]]
        Xs = S["X"][S["seg"]]
        for s in sorted(set(sid_seg.tolist())):
            m = sid_seg == s
            for req, cis in grp_cols.items():
                n_b += 1
                v = Xs[m][:, cis]
                if not np.any(np.isfinite(v) & (v != 0)):
                    bad3.append(f"{n_} scores {s} but its field {req} carries no information "
                                f"({'/'.join(S['names'][c] for c in cis)} all NaN / 0 on its segments)")
    check(3, "feature -> tier mapping", not bad3,
          f"{len(back)} features: tier = max over required groups of the min tier of the alternatives, "
          f"recomputed with the csv module from field_ladder.csv and (content fields "
          f"{', '.join(f'{a} t{lt[a]}' for a in CONTENT)}) vectorised from detections.csv.gz == "
          f"tier_features.csv, carriers equal; (a) {n_cells} feature values of sensors lacking the "
          f"feature's fields (content: per tree) are all NaN / 0; (b) {n_b} (tier model, sensor, field) "
          f"triples: every field a tier model uses carries information on the segments of every sensor it "
          f"scores; testing vs ladder "
          f"headers: {'agree' if not hdr_diffs else '; '.join(hdr_diffs)}" + (f"; {bad3[:5]}" if bad3 else ""))

    # 4 every segment gets exactly one routed p in [0, 1] (files re-read)
    bad4 = []
    nseg = 0
    for k in D["keys"]:
        rows = D["rows"][k]
        pf4 = C.read_columns(OUT / f"routed_{C.iter_slug(k)}__p.csv",
                             numeric={"seg_id", "p_drone_all", "p_drone_air", "tier"})
        tr = C.read_columns(E.TRACKS_DEFAULT / f"{C.iter_slug(k)}__tracks.csv", numeric={"seg_id"},
                            usecols=["seg_id", "sensor_id"])
        nseg += rows.size
        if not np.array_equal(pf4["seg_id"].astype(np.int64), tr["seg_id"].astype(np.int64)) or \
                np.unique(pf4["seg_id"]).size != rows.size:
            bad4.append(f"{k}: seg_id set")
        for t in ("all", "air"):
            p = pf4[f"p_drone_{t}"]
            if not (np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))):
                bad4.append(f"{k} {t}: p outside [0, 1] or missing")
        want_tier = np.array([lev[s] for s in tr["sensor_id"]], float)
        if not np.array_equal(pf4["tier"], want_tier):
            bad4.append(f"{k}: tier column != level of the sensor")
        if not np.all(pf4["fold"] == (k if k in TK else "final")):
            bad4.append(f"{k}: fold column")
        if k in TK:
            q = C.read_columns(OUT / f"routed_{C.iter_slug(k)}__p_inner.csv",
                               numeric={"seg_id", "p_drone_all", "p_drone_air"})
            for o in TK:
                if o == k:
                    continue
                m = q["outer_fold"] == o
                if not np.array_equal(q["seg_id"][m].astype(np.int64), tr["seg_id"].astype(np.int64)):
                    bad4.append(f"{k}: p_inner fold {o} coverage")
                for t in ("all", "air"):
                    v = q[f"p_drone_{t}"][m]
                    if not (np.all(np.isfinite(v)) and np.all((v >= 0) & (v <= 1))):
                        bad4.append(f"{k}: p_inner fold {o} {t} outside [0, 1]")
            if set(q["outer_fold"].tolist()) != set(TK) - {k}:
                bad4.append(f"{k}: p_inner folds")
    check(4, "exactly one routed p in [0, 1] per segment", not bad4,
          f"{nseg} segments over {len(D['keys'])} iterations (p.csv re-read; tier column == level of the "
          f"segment's sensor; p_inner covers every segment once per outer fold)" + (f"; {bad4[:5]}" if bad4 else ""))

    # 5 determinism
    probe = []
    t1 = SPECS[f"t{tiers_all[0]}"]
    tk = TK[-1]
    probe.append(next(x for x in R if x[0] == t1["name"] and x[2] == ("d3", "unit") and x[4] == (tk,)))
    probe.append(next(x for x in R if x[0] == f"t{tiers_all[-1]}" and x[2] == ("d3", "grid") and len(x[4]) == 2))
    same = []
    for task in probe:
        r2 = fit_task(task)
        same.append(all(np.array_equal(r2["pred"][k], R[task]["pred"][k]) for k in task[4]))
    re_res = assemble(SPECS[f"t{tiers_all[-1]}"], R)
    same.append(all(np.array_equal(re_res["p_cal"][t], RES[f"t{tiers_all[-1]}"]["p_cal"][t], equal_nan=True)
                    for t in ("all", "air")))
    same.append(cl["det"][2])
    check(5, "determinism", all(same),
          f"serial refit of a tier-{tiers_all[0]} outer fit and a tier-{tiers_all[-1]} pair fit == pool results; "
          f"re-assembly of tier {tiers_all[-1]} identical; classifier tier {cl['det'][0]} fold {cl['det'][1]} "
          f"refit without the pair cache identical: {same}")

    # 6 decision recomputed from the written files
    wins6, yr, pr_, ps_, gg = 0, [], [], [], []
    for k in TK:
        a = C.read_columns(OUT / f"routed_{C.iter_slug(k)}__p.csv", numeric={"p_drone_all"})["p_drone_all"]
        b = single[k]["p_drone_all"]
        rows = D["rows"][k]
        mk = D["deploy"][rows]
        y, w = D["y"]["all"][rows][mk], g[rows][mk]
        wins6 += GB.logloss(y, a[mk], w) < GB.logloss(y, b[mk], w)
        yr.append(y); pr_.append(a[mk]); ps_.append(b[mk]); gg.append(w)
    yr, pr_, ps_, gg = (np.concatenate(v) for v in (yr, pr_, ps_, gg))
    rec6 = wins6 >= WINS_NEEDED and GB.auc(yr, pr_, gg) >= GB.auc(yr, ps_, gg) - AUC_TOL
    dsum = max(abs(sum(contrib[f["k"]].values()) - (f["r"]["ll_step"] - f["s"]["ll_step"]))
               for f in cmp_[("all", "deploy")]["per"])
    check(6, "pre-registered decision recomputed from the written files", rec6 == dec["recommend"]
          and wins6 == dec["wins"] and dsum < 1e-12,
          f"routed wins {wins6} of 6 (needs >= {WINS_NEEDED}), pooled step AUC {GB.auc(yr, pr_, gg):.4f} vs "
          f"{GB.auc(yr, ps_, gg):.4f} (tolerance {AUC_TOL}) -> recommend routed: {rec6}; per-sensor "
          f"decomposition sums to the iteration difference (worst |d| {dsum:.1e})")

    runtime = time.time() - t_start
    extra = dict(cmp_rd=cmp_rd, cmp_ds=cmp_ds, contrib=contrib, cinfo=cinfo, ftier=ftier,
                 not_modelled=not_modelled, lev0=lev0, sfields=sfields, have=have)
    L = write_report(ftab_rows, st_rows, tiers_all, unmodelled, lev, lad, cl, cmp_, dec, RES, hdr_diffs,
                     runtime, fails, args, extra)
    log("")
    for line in L["print"]:
        log(line)
    log(f"runtime {runtime:.0f} s; report {OUT / 'tier_models.md'}")
    if fails:
        log(f"SELF-CHECKS FAILED: {fails}")
        return 1
    log("ALL SELF-CHECKS PASSED")
    return 0


def ladder_row(model, t, tt, k, sc, rows, y, p, pp, prev, g, pa=None) -> dict:
    yy, gg = y[rows], g[rows]
    row = dict(model=model, target=t, tier=tt, iteration=k, scope=sc, n_seg=int(rows.size),
               n_pos=int(yy.sum()), steps=float(gg.sum()), pos_steps=float(np.sum(gg * yy)),
               **ex_metric(yy, p[rows], gg), prev_tier=prev)
    if pp is not None:
        m = ex_metric(yy, pp[rows], gg)
        row.update(prev_ll_step=m["ll_step"], prev_auc_step=m["auc_step"], prev_ll_seg=m["ll_seg"],
                   prev_auc_seg=m["auc_seg"])
    if pa is not None:
        m = ex_metric(yy, pa[rows], gg)
        row.update(ablation=f"a{tt}" if pa is not pp else f"t{prev}", abl_ll_step=m["ll_step"],
                   abl_auc_step=m["auc_step"])
    return row


def compare(routed, ps, scopes, g) -> dict:
    D, TK = E.D, E.D["test_keys"]
    out = {}
    for t in ("all", "air"):
        y = D["y"][t]
        for sc, smask in scopes.items():
            per, pool = [], []
            for k in TK:
                rows = D["rows"][k]
                rows = rows[smask[rows]]
                pool.append(rows)
                per.append(dict(k=k, n=int(rows.size), pos=int(y[rows].sum()),
                                r=ex_metric(y[rows], routed[t][rows], g[rows]),
                                s=ex_metric(y[rows], ps[t][rows], g[rows])))
            rows = np.concatenate(pool)
            out[(t, sc)] = dict(per=per, n=int(rows.size), pos=int(y[rows].sum()),
                                r=ex_metric(y[rows], routed[t][rows], g[rows]),
                                s=ex_metric(y[rows], ps[t][rows], g[rows]),
                                rc=calib(y[rows], routed[t][rows], g[rows]) if y[rows].sum() else None,
                                scal=calib(y[rows], ps[t][rows], g[rows]) if y[rows].sum() else None,
                                wins=sum(1 for f in per if f["r"]["ll_step"] < f["s"]["ll_step"]),
                                wins_auc=sum(1 for f in per if f["r"]["auc_step"] > f["s"]["auc_step"]))
    return out


def seg_ll(y, p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1.0 - 1e-15)     # GB.logloss's clip
    return -(y * np.log(p) + (1.0 - y) * np.log1p(-p))


def sensor_contrib(pa, pb, smask, g, t="all") -> dict:
    """{iteration: {sensor: share of (logloss a - logloss b)}}: step-weighted, sums over the
    sensors to the iteration's difference."""
    D, TK = E.D, E.D["test_keys"]
    y = D["y"][t]
    out = {}
    for k in TK:
        rows = D["rows"][k]
        rows = rows[smask[rows]]
        w = g[rows]
        d = w * (seg_ll(y[rows], pa[rows]) - seg_ll(y[rows], pb[rows])) / w.sum()
        sid = D["sensor_id"][rows]
        out[k] = {s: float(d[sid == s].sum()) for s in sorted(set(sid.tolist()))}
    return out


def decision(cmp_) -> dict:
    c = cmp_[("all", "deploy")]
    auc_r, auc_s = c["r"]["auc_step"], c["s"]["auc_step"]
    rec = c["wins"] >= WINS_NEEDED and auc_r >= auc_s - AUC_TOL
    return dict(wins=c["wins"], auc_r=auc_r, auc_s=auc_s, recommend=bool(rec),
                a=c["wins"] >= WINS_NEEDED, b=auc_r >= auc_s - AUC_TOL)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def write_report(ftab_rows, st_rows, tiers, unmodelled, lev, lad, cl, cmp_, dec, RES, hdr_diffs,
                 runtime, fails, args, extra) -> dict:
    D, TK = E.D, E.D["test_keys"]
    short = {k: k.split("/")[-1] for k in TK}
    ftab = {r["feature"]: r["tier"] for r in ftab_rows}
    cinfo, ftier, not_modelled = extra["cinfo"], extra["ftier"], extra["not_modelled"]
    cmp_rd, cmp_ds, contrib, sfields = extra["cmp_rd"], extra["cmp_ds"], extra["contrib"], extra["sfields"]
    ladder_fields = {}
    with LADDER_CSV.open(newline="") as fh:
        for r in csv.DictReader(fh):
            ladder_fields.setdefault(int(r["tier"]), []).append((r["field"], r["pct_data"]))
    for name, ci in cinfo.items():
        ladder_fields.setdefault(ftier[name], []).append(
            (f"{name} [a value in {' | '.join(ci['containers'])}]", f"{ci['share']:.1f}"))

    def exr(t, tt, sc, it):
        return next(r for r in lad if r["model"] == "existence" and r["target"] == t and r["tier"] == tt
                    and r["scope"] == sc and r["iteration"] == it)

    def train_sensors(tt):
        return " ".join(sorted(set(D["sensor_id"][SPECS[f"t{tt}"]["train"]].tolist())))

    L = ["# Tier models: one model per attribute tier, and a routed model", ""]
    L += [f"**Self-checks: {'ALL PASSED' if not fails else f'{fails} FAILED'}** (run log of "
          f"`python3 tools/tier_models.py --jobs {args.jobs}`; runtime {runtime:.0f} s). Rules and the "
          "pre-registered decision are in the script's docstring.", ""]

    # ---- answer in short
    c = cmp_[("all", "deploy")]
    margins = [(short[f["k"]], f["s"]["ll_step"] - f["r"]["ll_step"]) for f in c["per"]]
    wins_clear = sum(1 for _, m in margins if m >= MARGIN)
    pooled_better = c["r"]["ll_step"] < c["s"]["ll_step"]
    if dec["recommend"]:
        fragile = wins_clear < WINS_NEEDED or not pooled_better
    else:
        fragile = dec["b"] and dec["wins"] + sum(1 for _, m in margins if -MARGIN < m < 0) >= WINS_NEEDED
    top = []
    for k in TK:
        cs = contrib[k]
        s = max(cs, key=lambda x: (abs(cs[x]), x))
        top.append(f"{short[k]} {s} {sgn(cs[s])}")
    rd, ds = cmp_rd[("all", "deploy")], cmp_ds[("all", "deploy")]
    L += ["## Answer in short", "",
          f"- **Decision (pre-registered): {'replace the single model by the routed one' if dec['recommend'] else 'keep the single model'}.** "
          f"The routed model has the lower held-out step-weighted logloss (y_all, deployable sensors) on "
          f"**{dec['wins']} of 6** testing iterations (needed: >= {WINS_NEEDED}); pooled step-weighted AUC "
          f"{dec['auc_r']:.4f} routed vs {dec['auc_s']:.4f} single (allowed drop {AUC_TOL}): condition (a) "
          f"{'met' if dec['a'] else 'NOT met'}, (b) {'met' if dec['b'] else 'NOT met'}.",
          f"- **Robustness of the verdict (descriptive, not part of the rule): {'FRAGILE' if fragile else 'robust'}.** Margin "
          f"per iteration (single minus routed logloss; positive = routed better): "
          + ", ".join(f"{k} {sgn(m)}" for k, m in margins)
          + f"; {wins_clear} of 6 won by at least {MARGIN} (a descriptive threshold, not pre-registered). Pooled "
          f"logloss routed {c['r']['ll_step']:.4f} vs single {c['s']['ll_step']:.4f} "
          f"({'routed lower' if pooled_better else 'routed HIGHER'}); pooled ECE routed "
          f"{c['rc']['ece']:.4f} vs single {c['scal']['ece']:.4f}.",
          "- Largest single-sensor share of each iteration's difference (routed minus single logloss; "
          "positive = routed worse there): " + ", ".join(top) + ".",
          f"- Routing vs dropping the testing-only sensors: `sdep` (the single configuration refitted "
          f"without EUIG / LJJX / MMJJ in training) beats the single model on {ds['wins']} of 6 (pooled "
          f"{ds['r']['ll_step']:.4f} vs {ds['s']['ll_step']:.4f}); the routed model beats `sdep` on "
          f"{rd['wins']} of 6 (pooled {rd['r']['ll_step']:.4f} vs {rd['s']['ll_step']:.4f}). The routed-vs-"
          f"`sdep` comparison is the effect of routing alone.",
          f"- Without the relays VKYY / QTVS the routed model wins {cmp_[('all', 'deploy_norelay')]['wins']} of 6 "
          f"(pooled {cmp_[('all', 'deploy_norelay')]['r']['ll_step']:.4f} vs "
          f"{cmp_[('all', 'deploy_norelay')]['s']['ll_step']:.4f}); for y_air it wins "
          f"{cmp_[('air', 'deploy')]['wins']} of 6 (pooled {cmp_[('air', 'deploy')]['r']['ll_step']:.4f} vs "
          f"{cmp_[('air', 'deploy')]['s']['ll_step']:.4f}).",
          ]
    L += importance_reading(lad, tiers)
    L += [""]

    L += ["## What changed after the 2026-10-29 review", "",
          "- The first run decided which sensor 'has' a metadata feature from its CSV header. EGOI's "
          "`objectInfo` holds rcs / speed / azimuth and NYTF's and IUKL's `trackInfo` speed / azimuth / "
          "track id, never a serial, protocol or model, so `has_serial` / `has_protocol` / `has_model` "
          "were 0 on every one of their segments, yet those sensors were routed to the 'metadata' tiers. "
          "That run's tier-6 / 7 / 9 gains and its verdict (4 of 6, recommend) measured mostly a change of "
          "training population and are superseded by this report.",
          "- Now each `has_*` feature needs a content field (a value actually present, in every tree where "
          "the sensor reports), placed on the ladder by its own share of sensors (table below). The same "
          "rule is applied to every report field: a field from which no feature carries information on any "
          "of a sensor's segments does not count as available (the UTM relays QTVS / VKYY report their own "
          "position as 0,0,0, so their range is void; IKJX's classification is always empty; LJJX / MMJJ / "
          "IKJX have no usable height above their own sensor). The new self-check 3(b) fails if any tier "
          "model uses a field that is empty for a sensor it scores (the first run would fail it).",
          "- A tier is modelled only if its training population has drones and clutter in every testing "
          "iteration (nested LOIO needs both in every fold).",
          "- The importance of a tier's fields is now measured on the same training population (field "
          "gain); the old number (routing gain) is kept next to it.",
          "- The decision rule is unchanged, but this configuration was fixed after the first run's result "
          "was known: treat the verdict as exploratory until the end-to-end F0 test.", ""]

    L += ["## Terms in one line each", "",
          "- **field**: one column of a sensor's SAPIENT report (e.g. `classification`, `location_z`).",
          "- **content field**: a value inside a column (e.g. a serial number inside `id` / `objectInfo`); a "
          "column can be present but never hold it.",
          "- **tier**: a group of fields that the same share of sensors report (field ladder of "
          "`tools/eval_tiers.py`): tier 1 = every sensor, higher tiers = rarer fields.",
          "- **cumulative**: the tier-t model may use every field of tiers 1..t, not only tier t.",
          "- **feature**: a number computed per segment from fields (e.g. mean speed from `location_x/y` and time).",
          "- **supports tier t**: the sensor reports every field the tier-t features need, and every lower tier's.",
          "- **routed model**: each segment is scored by the model of the richest modelled tier its own sensor supports.",
          "- **single model**: `tools/existence.py`'s one model on all features; a missing field is just a "
          "missing value (NaN) that the trees route to one side.",
          "- **sdep**: the single model's configuration refitted without the testing-only sensors in training.",
          "- **segment**: one sensor's track of one object, cut at gaps > 30 s.",
          "- **LOIO**: to judge testing iteration k, everything is fitted on the other 5; **nested**: the "
          "settings and the calibrator for k come from a second LOIO inside those 5.",
          "- **logloss**: average of -log(probability given to what happened); lower is better. "
          "**step-weighted**: each segment weighted by its 100 ms grid samples, i.e. by how much it moves the score.",
          "- **AUC**: chance that a random real-drone segment gets a higher P than a random clutter one (0.5 = coin flip).",
          "- **calibration / ECE**: whether P = 0.3 means 30 % drones; ECE = average gap between mean P and "
          "the observed rate over P bins.",
          "- **field gain** of tier t: logloss of a model WITHOUT tier t's features, fitted on the same "
          "sensors as the tier-t model, minus the tier-t model's, on the same held-out segments "
          "(positive = the new fields help).",
          "- **routing gain** of tier t: logloss of the previous tier's model (fitted on more sensors) minus "
          "the tier-t model's, on the same segments.",
          "- **deployable**: sensors present in the evaluation data; **testing-only**: EUIG, LJJX, MMJJ "
          "(never fitted here); **relays**: VKYY, QTVS, which in testing forward the drones' own Dronetag positions.",
          "- **GBDT**: gradient-boosted decision trees; **isotonic**: the monotone map from model score to probability.",
          ""]

    L += ["## What each tier can use", "",
          "Fields per tier from `data/derived/eval/field_ladder.csv` (share of evaluation sensors that have "
          "the field in their CSV header) plus the content fields (share of evaluation sensors whose "
          "reports actually hold the value), and the segment features computed from them.", ""]
    rows = []
    for t in sorted(ladder_fields):
        fs = [r["feature"] for r in ftab_rows if r["tier"] == t]
        status = "yes" if t in tiers else (f"no: {not_modelled[t]}" if t in not_modelled else "no (adds no feature)")
        rows.append([t, ", ".join(f"{f} ({p} %)" for f, p in ladder_fields[t]),
                     ", ".join(f"`{f}`" for f in fs) or "-", status])
    L += md_table(["tier", "fields (share of sensors)", "features it adds", "modelled"], rows) + [""]
    L += ["Content fields (a value present, not just the column):", ""]
    rows = []
    for name, ci in cinfo.items():
        empty = sorted(s for s in sfields if set(ci["containers"]) & sfields[s]
                       and s not in ci["carriers"] and s not in ci["partial"])
        rows.append([f"`{name}`", " \\| ".join(ci["containers"]), f"{ci['n_eval']}/{ci['n_eval_sensors']} "
                     f"({ci['share']:.1f} %)", ftier[name], " ".join(ci["carriers"]) or "-",
                     " ".join(ci["partial"]) or "-", " ".join(empty) or "-"])
    L += md_table(["content field", "inside", "evaluation sensors with a value", "tier", "carried by (every tree)",
                   "in one tree only (not counted)", "column present but never holds it"], rows) + [""]
    L += ["Feature -> fields mapping (also `tier_features.csv`):", ""]
    L += md_table(["feature", "raw fields", "tier", "used by"],
                  [[f"`{r['feature']}`", r["raw_fields"], r["tier"], r["used_by"]] for r in ftab_rows]) + [""]

    L += ["## Which sensor supports which tier", "",
          "Support = the highest tier whose features (and every lower tier's) the sensor's fields cover. "
          "Routing level = the highest MODELLED tier up to that. Support is cumulative: a sensor missing one "
          "field of a lower tier stops there even if it reports rarer fields (last column). Bearing-only "
          "sensors have no segments in `sensor_tracks_v2` and are out of scope.", ""]
    L += md_table(["sensor", "deployable", "segments", "routing level", "support", "classifier level",
                   "fields lacking for the next tier", "has, but above its level"],
                  [[r["sensor_id"], "yes" if r["deployable"] else "no (testing only)", r["segments"],
                    r["level_existence"], r["support_existence"], r["level_classify"], r["missing_for_next"] or "-",
                    r["unused_features"] or "-"]
                   for r in st_rows if r["segments"] > 0]) + [""]
    L += ["Testing vs evaluation CSV headers per sensor: "
          + ("identical for every sensor present in both trees." if not hdr_diffs else "; ".join(hdr_diffs)), ""]

    # ---- importance ladder
    for t in ("all", "air"):
        L += [f"## Importance ladder, existence y_{t} (held-out, deployable sensors)", "",
              "Each row: the tier-t model, on the segments of the sensors that support tier t. 'field gain' = "
              "logloss of the same-population model without tier t's features minus this tier's (the value "
              "of the new fields); 'routing gain' = previous tier's model minus this tier's (includes the "
              "change of training sensors). Both pooled over the 6 held-out iterations on the same segments; "
              "positive = better; 'wins' = iterations where tier t has the lower step-weighted logloss.", ""]
        rows = []
        for tt in tiers:
            pr, mn = exr(t, tt, "deploy", "pooled"), exr(t, tt, "deploy", "mean")
            new = [f for f in E.FEATURES if ftab[f] == tt] + (["sensor one-hot"] if tt == 1 else [])
            first = pr.get("prev_tier") is None
            rows.append([tt, ", ".join(new), train_sensors(tt), pr["n_seg"], pr["n_pos"],
                         fmt(pr["ll_step"], 4), fmt(pr["auc_step"]), fmt(mn["ll_step"], 4),
                         "-" if first else sgn(pr.get("field_gain_ll_step")),
                         "-" if first else f"{mn['field_wins']} of 6",
                         "-" if first else sgn(pr.get("gain_ll_step")),
                         "-" if first else f"{mn['wins_vs_prev']} of 6"])
        L += md_table(["tier", "adds", "training sensors", "segments", "drones", "logloss (pooled)",
                       "AUC (pooled)", "logloss (mean of 6)", "field gain", "field wins", "routing gain",
                       "routing wins"], rows) + [""]
        if t == "all":
            L += ["Per testing iteration, step-weighted logloss of each tier (y_all, deployable; in "
                  "brackets the field gain on the same segments):", ""]
            rows = []
            for tt in tiers:
                cells = []
                for k in TK:
                    r = exr("all", tt, "deploy", k)
                    cells.append(f"{fmt(r['ll_step'], 4)}" + (f" ({sgn(r.get('field_gain_ll_step'))})"
                                                              if r.get("prev_tier") is not None else ""))
                rows.append([tt] + cells)
            L += md_table(["tier"] + [short[k] for k in TK], rows) + [""]
            L += ["Same ladder without the relays VKYY / QTVS (pooled):", ""]
            rows = []
            for tt in tiers:
                pr, mn = exr("all", tt, "deploy_norelay", "pooled"), exr("all", tt, "deploy_norelay", "mean")
                first = pr.get("prev_tier") is None
                rows.append([tt, pr["n_seg"], pr["n_pos"], fmt(pr["ll_step"], 4), fmt(pr["auc_step"]),
                             "-" if first else sgn(pr.get("field_gain_ll_step")),
                             "-" if first else f"{mn['field_wins']} of 6",
                             "-" if first else sgn(pr.get("gain_ll_step"))])
            L += md_table(["tier", "segments", "drones", "logloss", "AUC", "field gain", "field wins",
                           "routing gain"], rows) + [""]
    L += ["Combination chosen per held-out fold (inner LOIO) and for the final model:", ""]
    rows = []
    for tt in tiers:
        res = RES[f"t{tt}"]
        for t in ("all", "air"):
            rows.append([tt, f"y_{t}", ", ".join(E.combo_name(res["sel"][(t, k)]) for k in TK),
                         E.combo_name(res["final"][t]["combo"])])
    L += md_table(["tier", "target", "per fold (" + ", ".join(short[k] for k in TK) + ")", "final"], rows) + [""]

    # ---- classifier ladder
    L += ["## Importance ladder, fixed wing vs rotary (classifier)", "",
          f"`classify.py`'s `{CLS_CFG}` nested protocol per tier, on deployable airborne drone segments of "
          "known class whose sensor supports the tier. Accuracy = n_grid-weighted share of correct "
          "'fixed if p > 0.5'; 'all Rotary' = accuracy of calling everything rotary on the same rows. "
          "The gain here is the routing gain (previous tier's model on the same rows).", ""]
    rows = []
    for tt in cl["tiers"]:
        pr = next(r for r in lad if r["model"] == "classify" and r["tier"] == tt and r["iteration"] == "pooled")
        accs = [next(r for r in lad if r["model"] == "classify" and r["tier"] == tt and r["iteration"] == k) for k in TK]
        wins = sum(1 for r in accs if r.get("prev_ll_step") is not None and r["ll_step"] < r["prev_ll_step"])
        rows.append([tt, ", ".join(f for f in cl["cols"][tt] if ftab[f] == tt), pr["n_seg"], pr["n_pos"],
                     fmt(pr["acc"]), fmt(pr["acc_all_rotary"]), fmt(pr["auc_step"]), fmt(pr["ll_step"], 4),
                     sgn(pr["prev_ll_step"] - pr["ll_step"]) if pr.get("prev_ll_step") is not None else "-",
                     f"{wins} of 6" if pr.get("prev_ll_step") is not None else "-"])
    L += md_table(["tier", "adds", "segments", "fixed", "accuracy (pooled)", "all Rotary", "AUC (pooled)",
                   "logloss (pooled)", "gain logloss", "wins vs prev"], rows) + [""]
    rows = []
    for tt in cl["tiers"]:
        rows.append([tt] + [fmt(next(r for r in lad if r["model"] == "classify" and r["tier"] == tt
                                     and r["iteration"] == k)["acc"]) for k in TK])
    L += ["Accuracy per held-out iteration:", ""] + md_table(["tier"] + [short[k] for k in TK], rows) + [""]

    # ---- routed vs single
    L += ["## Routed vs single model, per testing iteration", "",
          "Same held-out segments; single = `data/derived/existence/<slug>__p.csv`, routed = "
          "`routed_<slug>__p.csv`. Step-weighted logloss and AUC; 'better' = lower logloss; margin = "
          "single minus routed logloss (positive = routed better).", ""]

    def cmp_table(cc, a, b):
        rows = []
        for f in cc["per"]:
            rows.append([short[f["k"]], f["n"], f["pos"], fmt(f["s"]["ll_step"], 4), fmt(f["r"]["ll_step"], 4),
                         sgn(f["s"]["ll_step"] - f["r"]["ll_step"]),
                         a if f["r"]["ll_step"] < f["s"]["ll_step"] else b,
                         fmt(f["s"]["auc_step"]), fmt(f["r"]["auc_step"])])
        rows.append(["**pooled**", cc["n"], cc["pos"], fmt(cc["s"]["ll_step"], 4), fmt(cc["r"]["ll_step"], 4),
                     sgn(cc["s"]["ll_step"] - cc["r"]["ll_step"]), f"{a} wins {cc['wins']} of 6",
                     fmt(cc["s"]["auc_step"]), fmt(cc["r"]["auc_step"])])
        return md_table(["iteration", "segments", "drones", f"logloss {b}", f"logloss {a}", "margin", "better",
                         f"AUC {b}", f"AUC {a}"], rows)

    tables_print = []
    for t, sc, title in (("all", "deploy", "y_all, deployable sensors (DECISION SCOPE)"),
                         ("all", "deploy_norelay", "y_all, deployable without VKYY / QTVS"),
                         ("all", "all", "y_all, all sensors incl. testing-only"),
                         ("air", "deploy", "y_air, deployable sensors")):
        cc = cmp_[(t, sc)]
        tab = cmp_table(cc, "routed", "single")
        L += [f"### {title}", ""] + tab + [""]
        if sc == "deploy" and t == "all":
            tables_print = [f"ROUTED vs SINGLE ({title})"] + tab
        if cc["rc"] is not None:
            L += [f"Calibration (pooled): single ECE {cc['scal']['ece']:.4f}, mean P {cc['scal']['mean_p_step']:.4f} "
                  f"vs observed {cc['scal']['obs_step']:.4f} (step-weighted), worst bin {cc['scal']['worst']} "
                  f"P {cc['scal']['worst_mp']:.3f} vs {cc['scal']['worst_obs']:.3f}; routed ECE {cc['rc']['ece']:.4f}, "
                  f"mean P {cc['rc']['mean_p_step']:.4f} vs {cc['rc']['obs_step']:.4f}, worst bin {cc['rc']['worst']} "
                  f"P {cc['rc']['worst_mp']:.3f} vs {cc['rc']['worst_obs']:.3f}.", ""]
    L += ["### Where the difference comes from (y_all, deployable)", "",
          "Each cell: the sensor's share of the iteration's logloss difference, routed minus single "
          "(step-weighted; positive = routed worse on that sensor). Columns sum to the iteration's "
          "difference (last row).", ""]
    sens = sorted({s for k in TK for s in contrib[k]})
    rows = [[s, lev.get(s, "-")] + [sgn(contrib[k].get(s, np.nan)) if s in contrib[k] else "-" for k in TK]
            for s in sens]
    rows.append(["**total**", ""] + [sgn(sum(contrib[k].values())) for k in TK])
    L += md_table(["sensor", "routed tier"] + [short[k] for k in TK], rows) + [""]
    L += ["### Routing alone: routed vs `sdep` (y_all, deployable)", "",
          "`sdep` = the single configuration (all features, NaN-aware) refitted with the same protocol "
          "without the testing-only sensors in training, i.e. the same training sensors as the tier models.", ""]
    L += cmp_table(cmp_rd[("all", "deploy")], "routed", "sdep") + [""]
    L += ["### Dropping the testing-only sensors: `sdep` vs single (y_all, deployable)", ""]
    L += cmp_table(cmp_ds[("all", "deploy")], "sdep", "single") + [""]

    L += ["## Decision", "",
          f"Pre-registered rule: routed replaces single iff (a) lower step-weighted logloss (y_all, deployable) "
          f"on >= {WINS_NEEDED} of 6 testing iterations AND (b) pooled step-weighted AUC not lower by more than "
          f"{AUC_TOL}. Result: (a) {dec['wins']} of 6 -> {'met' if dec['a'] else 'not met'}; (b) "
          f"{dec['auc_r']:.4f} vs {dec['auc_s']:.4f} -> {'met' if dec['b'] else 'not met'}. "
          f"**{'Recommend the routed model' if dec['recommend'] else 'Keep the single model'}** "
          f"(verdict {'fragile' if fragile else 'robust'}, see the margins above; the end-to-end F0 test "
          "is the controller's next step).", ""]

    top_t = tiers[-1]
    capped = [f"{r['sensor_id']} ({r['unused_features']})" for r in st_rows if r["segments"] > 0 and r["unused_features"]]
    L += ["## Caveats", "",
          f"- Few positives at the high tiers: the top modelled tier {top_t} is trained on {train_sensors(top_t)}; "
          "the drone counts per tier are in the ladder tables, and segments of one flight are strongly "
          "correlated, so the effective sample is smaller still.",
          "- Tiers that could not be modelled: " + ("; ".join(f"t{t}: {why}" for t, why in not_modelled.items())
                                                    or "none") + ". Their fields' value is therefore not measured.",
          "- Cumulative support caps sensors: a sensor stops at the first tier it lacks and never uses a "
          "rarer field it does carry; the single model has no such cap. Sensors with genuine features above "
          "their level: " + ("; ".join(capped) or "none") + ".",
          "- The tier models and `sdep` exclude the testing-only sensors (EUIG, LJJX, MMJJ) from training "
          "(they must not feed evaluation predictions); `existence.py`'s single model keeps them. Routed vs "
          "`sdep` isolates the routing; check 2 proves the protocol itself is identical.",
          "- Routing gain mixes the new fields with a different training population; the field gain does "
          "not (same sensors, same segments, only the features differ).",
          "- A tier is a share-of-sensors bucket, not a meaning bucket; whether a sensor reports a field "
          "says little about how informative it is.",
          "- The single model already sees every feature and handles a missing field as NaN, so the "
          "routed model can only win where a smaller model generalises better; with a model family "
          "that copes with missing values, the ladder is mainly a measurement of field importance.",
          "- Existence was found worth about +0.02 of the combined score end-to-end; a better logloss here "
          "is necessary, not sufficient, for an end-to-end gain.",
          "- p_inner is written exactly like `existence.py`: fold k's calibrator is fitted on the pooled "
          "inner predictions of the 5 other iterations, which include iteration j itself, so j's own labels "
          "enter the calibrator of its nested prediction (no label of the held-out fold k does).",
          "- Inherited: sensor_tracks variances and height offsets were fitted on all 6 testing iterations.", ""]
    (OUT / "tier_models.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    pr_lines = ["IMPORTANCE LADDER (existence y_all, deployable, pooled held-out, same segments; field gain = "
                "same-population model without the tier's features minus this tier; routing gain = previous "
                "tier's model minus this tier)"]
    rows = []
    for tt in tiers:
        pr, mn = exr("all", tt, "deploy", "pooled"), exr("all", tt, "deploy", "mean")
        first = pr.get("prev_tier") is None
        rows.append([tt, train_sensors(tt), pr["n_seg"], pr["n_pos"], fmt(pr["ll_step"], 4), fmt(pr["auc_step"]),
                     "-" if first else sgn(pr.get("field_gain_ll_step")),
                     "-" if first else f"{mn['field_wins']}/6",
                     "-" if first else sgn(pr.get("gain_ll_step")),
                     "-" if first else f"{mn['wins_vs_prev']}/6"])
    pr_lines += md_table(["tier", "training sensors", "segments", "drones", "logloss", "AUC", "field gain",
                          "field wins", "routing gain", "routing wins"], rows)
    pr_lines += ["not modelled: " + ("; ".join(f"t{t}: {why}" for t, why in not_modelled.items()) or "none")]
    pr_lines += ["", "CLASSIFIER LADDER (fixed wing, pooled held-out)"]
    rows = []
    for tt in cl["tiers"]:
        pr = next(r for r in lad if r["model"] == "classify" and r["tier"] == tt and r["iteration"] == "pooled")
        rows.append([tt, pr["n_seg"], pr["n_pos"], fmt(pr["acc"]), fmt(pr["acc_all_rotary"]), fmt(pr["auc_step"]),
                     fmt(pr["ll_step"], 4)])
    pr_lines += md_table(["tier", "segments", "fixed", "accuracy", "all Rotary", "AUC", "logloss"], rows)
    pr_lines += [""] + tables_print
    pr_lines += ["", "ROUTED vs SDEP (routing alone, y_all, deployable)"] + cmp_table(cmp_rd[("all", "deploy")], "routed", "sdep")
    pr_lines += ["", f"verdict robustness: {'FRAGILE' if fragile else 'robust'} ({wins_clear} of 6 won by >= {MARGIN}; "
                 f"pooled routed {c['r']['ll_step']:.4f} vs single {c['s']['ll_step']:.4f}); largest sensor share: "
                 + ", ".join(top)]
    return {"print": pr_lines}


def importance_reading(lad, tiers) -> list[str]:
    out = []
    pooled = {tt: next(r for r in lad if r["model"] == "existence" and r["target"] == "all" and r["tier"] == tt
                       and r["scope"] == "deploy" and r["iteration"] == "pooled") for tt in tiers}
    mean = {tt: next(r for r in lad if r["model"] == "existence" and r["target"] == "all" and r["tier"] == tt
                     and r["scope"] == "deploy" and r["iteration"] == "mean") for tt in tiers}
    gains = [(tt, pooled[tt].get("field_gain_ll_step", np.nan), mean[tt]["field_wins"]) for tt in tiers[1:]]
    if not gains:
        return out
    best = max(gains, key=lambda x: (x[1] if x[1] == x[1] else -np.inf))
    solid = [f"t{tt} ({sgn(gn)}, {w}/6)" for tt, gn, w in gains if gn == gn and gn > 0 and w >= WINS_NEEDED]
    weak = [f"t{tt} ({sgn(gn)}, {w}/6)" for tt, gn, w in gains if not (gn == gn and gn > 0 and w >= WINS_NEEDED)]
    out.append(f"- Importance ladder (y_all, deployable, field gain = same training sensors with vs without the "
               f"tier's fields): tier 1 alone (report counts, duration, sensor id) reaches pooled logloss "
               f"{pooled[tiers[0]]['ll_step']:.4f} / AUC {fmt(pooled[tiers[0]]['auc_step'])}. The largest pooled "
               f"field gain comes from tier {best[0]} ({sgn(best[1])}). Gains that hold on >= {WINS_NEEDED} of 6 "
               f"iterations: {', '.join(solid) or 'none'}; gains that do not: {', '.join(weak) or 'none'} "
               f"(pooled field gain, iterations won).")
    return out


if __name__ == "__main__":
    sys.exit(main())
