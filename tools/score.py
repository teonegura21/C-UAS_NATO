#!/usr/bin/env python3
"""Local scorer: our estimate of the competition score on the testing iterations.

The rules (01_Competition_Instructions.pdf section 2.5) name the four parts and their weights
but not the GOSPA parameters or how GOSPA becomes a score, so this scorer is a PROXY:

* Track (0.4): GOSPA (alpha = 2) per 100 ms grid step between the truth (tools/truth.py) and
  the submission's rows at that step, in the iteration's local ENU frame. Several (c, p, dims)
  configurations are scored because the official ones are unknown; PRIMARY is the one the
  experiments are compared on. The track score is 1 - sum_t cost_t / sum_t c^p/2 (|X_t| + |Y_t|):
  1 for a perfect submission, 0 when nothing is within c of a truth (an empty submission, or
  only false tracks).
* Classification (0.2): recall over truth rows (one per drone per step) whose catalog class is
  known: the truth's GOSPA partner (d < c, PRIMARY) must carry the same classification string.
  A missed truth row counts as wrong.
* Identification (0.1): the same with identity (BLUE -> Friendly, RED -> Hostile).
* Fusion completeness (0.3): "reported class metadata correctness". For each of the four
  metadata columns, the recall as above over truth rows whose catalog value is known (not
  'Unknown'); the part score is the mean over columns that have any known value. The catalog
  (tools/drone_catalog.csv) is our own mapping, so this part is only as right as the catalog.

combined = 0.4 track + 0.2 classification + 0.1 identification + 0.3 metadata, per iteration;
the headline is the mean over iterations (the leaderboard scores each data set separately).

The submission is read back from the .xlsx (tools/submission.py), so the score is the score of
the file that would be uploaded. Coordinates in it are in the anonymized longitude frame, the
same frame as the truth, so the ENU geometry is exact.

Usage:
  python3 tools/score.py --method NAME [--truth all|airborne] [--log --note TEXT]
  python3 tools/score.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import cuas_common as C  # noqa: E402
import gospa as G  # noqa: E402
import submission as S  # noqa: E402
import truth as T  # noqa: E402

# (name, c metres, p, dims). The first is PRIMARY.
CONFIGS = (
    ("c50_p2_3d", 50.0, 2.0, 3),
    ("c10_p2_3d", 10.0, 2.0, 3),
    ("c25_p2_3d", 25.0, 2.0, 3),
    ("c100_p2_3d", 100.0, 2.0, 3),
    ("c50_p1_3d", 50.0, 1.0, 3),
    ("c50_p2_2d", 50.0, 2.0, 2),
)
PRIMARY = CONFIGS[0][0]
WEIGHTS = {"track": 0.4, "classification": 0.2, "identity": 0.1, "metadata": 0.3}
KNOWN_CLASSES = ("Fixed Wing UAS", "Rotary Wing UAS", "UAS Transmitter")
EXPERIMENTS = C.DERIVED / "experiments.csv"
EXPERIMENT_COLUMNS = [
    "run", "method", "note", "code_hash", "iter_key", "truth_mode", "combined", "track",
    "classification", "identity", "metadata", "gospa_mean", "loc_rms_m", "missed_rate",
    "false_rate", "cost_share_loc", "cost_share_missed", "cost_share_false",
    "track_c10", "track_c25", "track_c100", "track_p1", "track_2d", "n_est_rows", "n_tracks",
    "logged_utc",       # when the block was appended (blank for rows logged before 2026-10-29)
]


def testing_iterations() -> list[C.Iteration]:
    return [it for it in C.iterations() if it.has_groundtruth]


def _recall(assigned_est: np.ndarray, truth_vals: np.ndarray, est_vals: np.ndarray,
            eligible: np.ndarray) -> tuple[int, int]:
    """(correct, eligible) over truth rows. assigned_est[j] = est row of truth j or -1."""
    ok = eligible & (assigned_est >= 0)
    hit = np.zeros(len(truth_vals), dtype=bool)
    hit[ok] = est_vals[assigned_est[ok]] == truth_vals[ok]
    return int(hit.sum()), int(eligible.sum())


_TRUTH_CACHE: dict = {}


def cached_truth(it: C.Iteration, truth_mode: str):
    """T.load_truth memoised per (iteration, mode): tuning loops score many variants."""
    k = (it.key, truth_mode)
    if k not in _TRUTH_CACHE:
        _TRUTH_CACHE[k] = T.load_truth(it, truth_mode)
    return _TRUTH_CACHE[k]


def score_rows(it: C.Iteration, rows: dict, truth_mode: str = "all",
               configs: tuple | None = None, primary: str | None = None) -> dict:
    """Score one iteration's track rows (the submission.read_submission dict).

    `configs` restricts the GOSPA configurations computed: names from CONFIGS and/or custom
    (name, c, p, dims) tuples (sensitivity studies); None = every CONFIGS entry. `primary`
    (default PRIMARY) names the configuration whose assignment the label recalls use and
    whose track score enters 'combined'; it must be among the computed ones. Headline
    numbers always use the defaults."""
    primary = PRIMARY if primary is None else primary
    if configs is None:
        cfgs = list(CONFIGS)
    else:
        named = {c[0]: c for c in CONFIGS}
        cfgs = [named[c] if isinstance(c, str) else tuple(c) for c in configs]
    if primary not in [c[0] for c in cfgs]:
        raise ValueError(f"configs must include the primary configuration {primary}")
    tr = cached_truth(it, truth_mode)
    n_steps = len(tr.grid_t)
    step_y = np.asarray(rows["step"], dtype=np.int64)
    if len(step_y):
        e, n, u = C.iteration_frame(it).to_enu(np.asarray(rows["latitude"], float),
                                               np.asarray(rows["longitude"], float),
                                               np.asarray(rows["altitude_m"], float))
        y = np.column_stack([e, n, u])
    else:
        y = np.zeros((0, 3))
    x = np.column_stack([tr.e, tr.n, tr.u])
    step_x = np.asarray(tr.step, dtype=np.int64)

    out: dict = {"iter_key": it.key, "truth_mode": truth_mode, "n_steps": n_steps,
                 "n_truth_rows": int(len(step_x)), "n_est_rows": int(len(step_y)),
                 "n_tracks": int(len(set(rows["track_id"])))}
    pairs_primary = None
    for name, c, p, dims in cfgs:
        seq = G.gospa_sequence(step_x, x[:, :dims], step_y, y[:, :dims], n_steps, c, p)
        half = c ** p / 2.0
        cost = float(seq["cost"].sum())
        worst = float((half * (seq["n_truth"] + seq["n_est"])).sum())
        n_missed = int(seq["n_missed"].sum())
        n_false = int(seq["n_false"].sum())
        n_assigned = int(len(seq["pairs"]))
        loc = float(seq["loc"].sum())
        out[name] = {
            "track": 1.0 - cost / worst if worst > 0 else 1.0,
            "gospa_mean": float(seq["distance"].mean()) if n_steps else 0.0,
            "loc_rms_m": (loc / n_assigned) ** (1.0 / p) if n_assigned else float("nan"),
            "missed_rate": n_missed / len(step_x) if len(step_x) else 0.0,
            "false_rate": n_false / len(step_y) if len(step_y) else 0.0,
            "cost_share_loc": loc / cost if cost else 0.0,
            "cost_share_missed": half * n_missed / cost if cost else 0.0,
            "cost_share_false": half * n_false / cost if cost else 0.0,
            "n_assigned": n_assigned, "n_missed": n_missed, "n_false": n_false,
            "cost": cost, "worst": worst,
        }
        if name == primary:
            pairs_primary = seq["pairs"]

    # Label recalls on the primary assignment (truth row -> est row).
    assigned = np.full(len(step_x), -1, dtype=np.int64)
    if pairs_primary is not None and len(pairs_primary):
        assigned[pairs_primary[:, 1]] = pairs_primary[:, 2]
    flights = tr.flights

    def truth_col(col: str) -> np.ndarray:
        return np.array([flights[f][col] for f in tr.flight], dtype=object)

    def est_col(col: str) -> np.ndarray:
        return np.asarray(rows[col], dtype=object)

    t_cls = truth_col("classification")
    c_ok, c_n = _recall(assigned, t_cls, est_col("classification"), np.isin(t_cls, KNOWN_CLASSES))
    t_id = truth_col("identity")
    i_ok, i_n = _recall(assigned, t_id, est_col("identity"), np.isin(t_id, ("Friendly", "Hostile")))
    meta = {}
    for col in C.METADATA_COLUMNS:
        t_v = truth_col(col)
        ok, n_el = _recall(assigned, t_v, est_col(col), t_v != "Unknown")
        meta[col] = (ok, n_el)
    out["classification"] = (c_ok, c_n)
    out["identity"] = (i_ok, i_n)
    out["metadata"] = meta
    out["primary"] = primary
    out["parts"] = _parts(out)
    return out


def _ratio(ok_n: tuple[int, int]) -> float:
    ok, n = ok_n
    return ok / n if n else float("nan")


def _parts(res: dict) -> dict:
    meta_vals = [_ratio(v) for v in res["metadata"].values() if v[1]]
    parts = {
        "track": res[res.get("primary", PRIMARY)]["track"],
        "classification": _ratio(res["classification"]),
        "identity": _ratio(res["identity"]),
        "metadata": float(np.mean(meta_vals)) if meta_vals else float("nan"),
    }
    # A part with no eligible truth rows does not count (weights renormalised).
    have = {k: v for k, v in parts.items() if v == v}
    w = sum(WEIGHTS[k] for k in have)
    parts["combined"] = sum(WEIGHTS[k] * v for k, v in have.items()) / w if w else float("nan")
    return parts


def score_file(it: C.Iteration, path: Path, truth_mode: str = "all") -> dict:
    problems = S.validate_submission(it, path)
    if problems:
        raise ValueError(f"{path}: invalid submission:\n  " + "\n  ".join(problems[:20]))
    return score_rows(it, S.read_submission(it, path), truth_mode)


def code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(Path(__file__).resolve().parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def summarize(results: list[dict]) -> dict:
    """Mean over iterations of the per-iteration parts (the headline)."""
    keys = ("combined", "track", "classification", "identity", "metadata")
    return {k: float(np.nanmean([r["parts"][k] for r in results])) for k in keys}


def print_table(method: str, results: list[dict]) -> None:
    print(f"\n{method}: PRIMARY = {PRIMARY} (truth mode {results[0]['truth_mode']})")
    head = (f"{'iteration':<16}{'combined':>9}{'track':>8}{'class':>8}{'ident':>8}{'meta':>8}"
            f"{'gospa':>8}{'loc_m':>7}{'miss':>7}{'false':>7}{'c10':>7}{'c100':>7}"
            f"{'tracks':>7}{'rows':>8}")
    print(head)
    for r in results:
        pr, pa = r[PRIMARY], r["parts"]
        print(f"{r['iter_key'].split('/')[-1]:<16}{pa['combined']:>9.4f}{pa['track']:>8.4f}"
              f"{pa['classification']:>8.3f}{pa['identity']:>8.3f}{pa['metadata']:>8.3f}"
              f"{pr['gospa_mean']:>8.2f}{pr['loc_rms_m']:>7.1f}{pr['missed_rate']:>7.3f}"
              f"{pr['false_rate']:>7.3f}{r['c10_p2_3d']['track']:>7.3f}"
              f"{r['c100_p2_3d']['track']:>7.3f}{r['n_tracks']:>7d}{r['n_est_rows']:>8d}")
    s = summarize(results)
    print(f"{'MEAN':<16}{s['combined']:>9.4f}{s['track']:>8.4f}{s['classification']:>8.3f}"
          f"{s['identity']:>8.3f}{s['metadata']:>8.3f}")


def _migrate_experiments_header() -> None:
    """Rewrite an experiments.csv whose header predates a column (e.g. logged_utc): old rows get ''."""
    with EXPERIMENTS.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows or rows[0] == EXPERIMENT_COLUMNS:
        return
    head = rows[0]
    if head != EXPERIMENT_COLUMNS[:len(head)]:
        raise ValueError(f"{EXPERIMENTS}: unexpected header {head[:6]}...; not migrating")
    tmp = EXPERIMENTS.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(EXPERIMENT_COLUMNS)
        for r in rows[1:]:
            w.writerow(r + [""] * (len(EXPERIMENT_COLUMNS) - len(r)))
    tmp.replace(EXPERIMENTS)


def log_results(method: str, note: str, results: list[dict], run: str) -> None:
    new = not EXPERIMENTS.exists()
    EXPERIMENTS.parent.mkdir(parents=True, exist_ok=True)
    if not new:
        _migrate_experiments_header()
    h = code_hash()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with EXPERIMENTS.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(EXPERIMENT_COLUMNS)
        rows = []
        for r in results:
            pr, pa = r[PRIMARY], r["parts"]
            rows.append([run, method, note, h, r["iter_key"], r["truth_mode"],
                         pa["combined"], pa["track"], pa["classification"], pa["identity"],
                         pa["metadata"], pr["gospa_mean"], pr["loc_rms_m"], pr["missed_rate"],
                         pr["false_rate"], pr["cost_share_loc"], pr["cost_share_missed"],
                         pr["cost_share_false"], r["c10_p2_3d"]["track"],
                         r["c25_p2_3d"]["track"], r["c100_p2_3d"]["track"],
                         r["c50_p1_3d"]["track"], r["c50_p2_2d"]["track"], r["n_est_rows"],
                         r["n_tracks"], stamp])
        s = summarize(results)
        rows.append([run, method, note, h, "testing/MEAN", results[0]["truth_mode"],
                     s["combined"], s["track"], s["classification"], s["identity"],
                     s["metadata"]] + [""] * (len(EXPERIMENT_COLUMNS) - 12) + [stamp])
        for row in rows:
            w.writerow(["" if isinstance(v, float) and v != v else
                        (f"{v:.6g}" if isinstance(v, float) else v) for v in row])


def score_method(method: str, truth_mode: str = "all", iters: list[C.Iteration] | None = None,
                 log: bool = False, note: str = "", run: str = "") -> list[dict]:
    results = []
    for it in iters or testing_iterations():
        path = C.submission_path(method, it)
        if not path.exists():
            print(f"  [skip] {it.key}: no file at {path.relative_to(C.REPO)}")
            continue
        results.append(score_file(it, path, truth_mode))
    if not results:
        raise SystemExit(f"no submissions found for method {method!r}")
    print_table(method, results)
    if log:
        log_results(method, note, results, run or method)
    return results


# --------------------------------------------------------------------------- self-test


def _rows_from_truth(tr: T.Truth, shift_e: float = 0.0, duplicate: bool = False,
                     it: C.Iteration | None = None) -> dict:
    """A submission made from the truth itself (optionally shifted east / duplicated)."""
    lat, lon, h = tr.lat, tr.lon, tr.h
    if shift_e:
        fr = C.iteration_frame(it)
        lat, lon, h = fr.to_geodetic(tr.e + shift_e, tr.n, tr.u)
    reps = 2 if duplicate else 1
    rows: dict = {k: [] for k in ("step", "track_id") + C.METADATA_COLUMNS
                  + ("classification", "identity")}
    num = {k: [] for k in ("latitude", "longitude", "altitude_m", "C_nn", "C_ne", "C_nu",
                           "C_ee", "C_eu", "C_uu", "classification_confidence")}
    for r in range(reps):
        for j in range(len(tr.step)):
            f = tr.flights[tr.flight[j]]
            rows["step"].append(int(tr.step[j]))
            rows["track_id"].append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{f['flight_key']}/{r}")))
            cls = f["classification"]
            rows["classification"].append(cls)
            rows["identity"].append(f["identity"])
            for col in C.METADATA_COLUMNS:
                rows[col].append(f[col])
            num["latitude"].append(lat[j])
            num["longitude"].append(lon[j])
            num["altitude_m"].append(h[j])
            for k, v in (("C_nn", 25.0), ("C_ne", 0.0), ("C_nu", 0.0), ("C_ee", 25.0),
                         ("C_eu", 0.0), ("C_uu", 100.0), ("classification_confidence", 0.9)):
                num[k].append(v)
    out = {k: np.asarray(v, dtype=float) for k, v in num.items()}
    out["step"] = np.asarray(rows.pop("step"), dtype=np.int64)
    out.update({k: np.asarray(v, dtype=object) for k, v in rows.items()})
    return out


def selftest() -> None:
    t0 = time.time()
    it = next(i for i in testing_iterations() if i.key == "testing/S1-I1")
    tr = T.load_truth(it, "all")
    method = "_selftest_score"
    checks = []

    def run(rows: dict) -> dict:
        path = C.submission_path(method, it)
        S.write_submission(it, rows, path)
        return score_file(it, path, "all")

    perfect = run(_rows_from_truth(tr))
    pa, pr = perfect["parts"], perfect[PRIMARY]
    checks.append(("perfect submission scores 1 on every part, GOSPA ~ 0",
                   all(abs(pa[k] - 1) < 1e-9 for k in ("track", "classification", "identity"))
                   and (pa["metadata"] != pa["metadata"] or abs(pa["metadata"] - 1) < 1e-9)
                   and pr["gospa_mean"] < 1e-3 and pr["n_missed"] == 0 and pr["n_false"] == 0))

    empty = score_rows(it, S.empty_rows(), "all")
    checks.append(("empty submission: track 0, every truth missed, recalls 0",
                   empty[PRIMARY]["track"] == 0 and empty[PRIMARY]["missed_rate"] == 1
                   and empty["parts"]["identity"] == 0))

    shifted = run(_rows_from_truth(tr, shift_e=3.0, it=it))
    sp = shifted[PRIMARY]
    checks.append(("3 m east shift: loc RMS 3 m, nothing missed or false, labels intact",
                   abs(sp["loc_rms_m"] - 3.0) < 1e-3 and sp["n_missed"] == 0
                   and sp["n_false"] == 0 and abs(shifted["parts"]["identity"] - 1) < 1e-9))
    # For p = 2 one truth per step at 3 m gives GOSPA 3 at that step.
    one = np.bincount(tr.step, minlength=len(tr.grid_t)) == 1
    checks.append(("3 m shift: track score = 1 - 9 / (c^2/2 * 2)",
                   abs(sp["track"] - (1 - 9.0 / (50.0 ** 2))) < 1e-6 and one.any()))

    dup = run(_rows_from_truth(tr, duplicate=True))
    dp = dup[PRIMARY]
    checks.append(("every track duplicated: false rate 0.5, no misses, track = 1 - 1/3",
                   abs(dp["false_rate"] - 0.5) < 1e-12 and dp["n_missed"] == 0
                   and abs(dp["track"] - (1 - 1.0 / 3.0)) < 1e-9))

    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"selftest {time.time() - t0:.1f} s")
    if not all(ok for _, ok in checks):
        raise SystemExit("SELF-CHECK FAILED")
    print("ALL SELF-CHECKS PASSED")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--method")
    ap.add_argument("--truth", default="all", choices=("all", "airborne"))
    ap.add_argument("--log", action="store_true", help="append to data/derived/experiments.csv")
    ap.add_argument("--note", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.method:
        ap.error("--method is required")
    score_method(a.method, a.truth, log=a.log, note=a.note)


if __name__ == "__main__":
    main()
