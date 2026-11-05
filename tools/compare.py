#!/usr/bin/env python3
"""Per-iteration method comparison: "A beats B on k of n iterations".

The owner's standard for every comparison (CLAUDE.md, rules for methods): never judge two
methods by a mean over the 6 testing iterations alone (0.166 vs 0.165 is noise); count on how
many iterations A is better than B, and give the sign-test p-value of that count.

Usage:
  python3 tools/compare.py A B [--truth all|airborne|both] [--config NAME|c,p,dims]
  python3 tools/compare.py --selftest

  A, B      method names; their testing files are C.submission_path(method, it) for every
            testing iteration. A testing iteration where either file is missing is skipped
            (listed). Methods whose rows exist only in memory are out of scope.
  --truth   truth mode (default both): 'all' (includes the parked-drone rows) or 'airborne'.
  --config  the GOSPA configuration whose assignment scores everything (track part and the
            label recalls): a name from score.CONFIGS (default c50_p2_3d, the PRIMARY proxy) or
            custom 'c,p,dims' such as 20,2,3 (c metres, p exponent, dims 2 or 3).

Output per truth mode: a table with the combined score of both methods and the difference
(A - B) per iteration, the four parts (track, classification, identification, metadata) of
both methods with their differences, then "A beats B on k, loses on l, ties on t of n
iterations" for the combined score and for each part, and the two-sided exact sign-test
p-value of the combined win/loss counts. A tie is |A - B| < 1e-4. Means are printed only for
context, last, and are never the verdict.

Terms: GOSPA = the track-distance metric (matched positions + a penalty c^p/2 for each
missed truth point and each false estimate); c = the distance beyond which a match no longer
counts; p = the exponent applied to position errors; sign test = a coin-flip test: with
wins w and losses l (ties dropped), how likely is a split at least this lopsided if the two
methods were equally good (with 6 iterations the best possible p is 0.031).

Exit status is non-zero if a self-check fails. Self-check each run: the win/loss/tie counts
equal a plain loop over the printed combined differences.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import cuas_common as C  # noqa: E402
import score as SC  # noqa: E402
import submission as S  # noqa: E402

TIE = 1e-4
PARTS = ("track", "classification", "identity", "metadata")


# --------------------------------------------------------------------------- helpers


def cfg_name(c: float, p: float, dims: int) -> str:
    """c50_p2_3d style name; identical to score.CONFIGS' names where they coincide."""
    return f"c{c:g}_p{p:g}_{int(dims)}d"


def parse_config(spec: str | None) -> tuple:
    """A score.CONFIGS name, or 'c,p,dims' -> (name, c, p, dims)."""
    if spec is None:
        return SC.CONFIGS[0]
    named = {c[0]: c for c in SC.CONFIGS}
    if spec in named:
        return named[spec]
    try:
        c, p, d = (s.strip() for s in spec.split(","))
        c, p, d = float(c), float(p), int(d)
    except ValueError:
        raise SystemExit(f"--config {spec!r}: expected a name from {sorted(named)} or 'c,p,dims'")
    if c <= 0 or p < 1 or d not in (2, 3):
        raise SystemExit("--config needs c > 0, p >= 1, dims in {2, 3}")
    return (cfg_name(c, p, d), c, p, d)


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact binomial (p = 1/2) p-value of the win/loss split, ties excluded."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def count_wlt(diffs, tie: float = TIE) -> tuple[int, int, int]:
    """(wins, losses, ties) of A over B from the differences A - B (NaN ignored)."""
    w = l = t = 0
    for d in diffs:
        if d != d:
            continue
        if abs(d) < tie:
            t += 1
        elif d > 0:
            w += 1
        else:
            l += 1
    return w, l, t


_FRAME_MEMO: dict = {}


def memoise_frame() -> None:
    """Process-local: make C.iteration_frame remember its result per iteration (it re-reads
    every sensor CSV on each call, ~1 s; score.score_rows calls it on every scoring). The
    returned Frame is the same object C.iteration_frame would build, so no result changes."""
    if getattr(C.iteration_frame, "_memo", False):
        return
    orig = C.iteration_frame

    def cached(it):
        if it.key not in _FRAME_MEMO:
            _FRAME_MEMO[it.key] = orig(it)
        return _FRAME_MEMO[it.key]

    cached._memo = True  # type: ignore[attr-defined]
    C.iteration_frame = cached


# --------------------------------------------------------------------------- comparison


def compare(a: str, b: str, modes: list[str], cfg: tuple) -> dict:
    """Score both methods on every testing iteration where both files exist."""
    memoise_frame()
    its, skipped = [], []
    for it in SC.testing_iterations():
        pa, pb = C.submission_path(a, it), C.submission_path(b, it)
        (its if pa.exists() and pb.exists() else skipped).append(it)
    if not its:
        raise SystemExit(f"no testing iteration has files for both {a!r} and {b!r}")
    res: dict = {m: {a: {}, b: {}} for m in modes}
    for it in its:
        for meth in dict.fromkeys((a, b)):
            rows = S.read_submission(it, C.submission_path(meth, it))
            for m in modes:
                res[m][meth][it.key] = SC.score_rows(it, rows, m, configs=[cfg],
                                                     primary=cfg[0])["parts"]
    return {"a": a, "b": b, "cfg": cfg, "its": [i.key for i in its],
            "skipped": [i.key for i in skipped], "res": res}


def _f(x: float, nd: int = 4) -> str:
    return "nan" if x != x else f"{x:.{nd}f}"


def report(out: dict, mode: str) -> dict:
    """Print the tables for one truth mode; returns the counts (and the printed diffs)."""
    a, b, cfg = out["a"], out["b"], out["cfg"]
    ra, rb = out["res"][mode][a], out["res"][mode][b]
    keys = out["its"]
    print(f"\n=== {a} (A) vs {b} (B): truth '{mode}', GOSPA config {cfg[0]} "
          f"(c={cfg[1]:g} m, p={cfg[2]:g}, {cfg[3]}-D) ===")
    if out["skipped"]:
        print(f"    skipped (a file is missing): {', '.join(k.split('/')[-1] for k in out['skipped'])}")
    print(f"\n{'iteration':<10}{'A comb':>9}{'B comb':>9}{'A-B':>11}")
    printed = []
    for k in keys:
        d = ra[k]["combined"] - rb[k]["combined"]
        s = f"{d:+.6f}"
        printed.append(s)
        print(f"{k.split('/')[-1]:<10}{_f(ra[k]['combined']):>9}{_f(rb[k]['combined']):>9}{s:>11}")
    print("\nparts, A / B / A-B per iteration")
    print(f"{'iteration':<10}" + "".join(f"{p:>28}" for p in PARTS))
    for k in keys:
        cells = []
        for p in PARTS:
            va, vb = ra[k][p], rb[k][p]
            cells.append(f"{_f(va, 3)}/{_f(vb, 3)}/{(va - vb):+.3f}".rjust(28))
        print(f"{k.split('/')[-1]:<10}" + "".join(cells))
    n = len(keys)
    summary = {}
    print(f"\nverdict (tie = |difference| < {TIE:g}), n = {n} iterations")
    for label in ("combined",) + PARTS:
        diffs = [ra[k][label] - rb[k][label] for k in keys]
        w, l, t = count_wlt(diffs)
        pv = sign_test_p(w, l)
        summary[label] = (w, l, t, pv)
        print(f"  {label:<15} A beats B on {w} of {n}, loses on {l}, ties on {t}; "
              f"sign test p = {pv:.3f}")
    comb = [ra[k]["combined"] - rb[k]["combined"] for k in keys]
    print(f"  (context only, not the verdict) mean combined: A {np.nanmean([ra[k]['combined'] for k in keys]):.4f}"
          f", B {np.nanmean([rb[k]['combined'] for k in keys]):.4f}, mean A-B {np.nanmean(comb):+.4f}")
    return {"summary": summary, "printed": printed, "diffs": comb}


def selfcheck_counts(rep: dict, mode: str) -> bool:
    """Check 4: the combined win count equals a plain loop over the differences (the unrounded
    ones, so the tie rule is applied to the true value, not to a six-decimal string), and every
    printed difference is that same difference formatted."""
    w = l = t = 0
    for d in rep["diffs"]:
        if d != d:
            continue
        if abs(d) < TIE:
            t += 1
        elif d > 0:
            w += 1
        else:
            l += 1
    got = rep["summary"]["combined"][:3]
    printed_ok = (len(rep["printed"]) == len(rep["diffs"])
                  and all(s == f"{d:+.6f}" for s, d in zip(rep["printed"], rep["diffs"])))
    ok = (w, l, t) == got and printed_ok
    print(f"  [{'PASS' if ok else 'FAIL'}] ({mode}) win/loss/tie {got} equals a plain loop over the "
          f"differences {(w, l, t)}; printed differences consistent: {printed_ok}")
    return ok


# --------------------------------------------------------------------------- selftest


def selftest() -> int:
    checks = []
    checks.append(("sign test: 6-0 -> 0.03125, 5-1 -> 0.21875, 3-3 -> 1, 0-0 -> 1, 8-2 -> 0.109375",
                   abs(sign_test_p(6, 0) - 0.03125) < 1e-12 and abs(sign_test_p(1, 5) - 0.21875) < 1e-12
                   and sign_test_p(3, 3) == 1.0 and sign_test_p(0, 0) == 1.0
                   and abs(sign_test_p(8, 2) - 0.109375) < 1e-12))
    checks.append(("win/loss/tie with the 1e-4 tie rule and NaN ignored",
                   count_wlt([0.5, -0.2, 0.00009, -0.00009, 0.0001, float("nan")]) == (2, 1, 2)))
    cfg = parse_config("20,2,3")
    checks.append(("parse_config: custom 20,2,3 and named; names agree with score.CONFIGS",
                   cfg == ("c20_p2_3d", 20.0, 2.0, 3) and parse_config("c10_p2_3d")[1] == 10.0
                   and parse_config(None) == SC.CONFIGS[0]
                   and cfg_name(50, 2, 3) == SC.PRIMARY and cfg_name(50, 1, 3) == "c50_p1_3d"
                   and cfg_name(50, 2, 2) == "c50_p2_2d"))
    out = compare("F0_fuse", "F0_fuse", ["all"], SC.CONFIGS[0])
    rep = report(out, "all")
    w, l, t, pv = rep["summary"]["combined"]
    checks.append(("a method compared with itself: all ties, p = 1",
                   (w, l, t) == (0, 0, len(out["its"])) and pv == 1.0))
    checks.append(("self-check 4 on that run", selfcheck_counts(rep, "all")))
    edge = [9.996e-5, -9.996e-5, 1.0e-4, 0.3]
    checks.append(("self-check 4 does not misclassify a difference that prints as +-0.000100",
                   selfcheck_counts({"summary": {"combined": (*count_wlt(edge)[:3], 1.0)},
                                     "printed": [f"{d:+.6f}" for d in edge], "diffs": edge}, "edge")))
    print()
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not all(ok for _, ok in checks):
        print("SELF-CHECK FAILED")
        return 1
    print("ALL SELF-CHECKS PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("methods", nargs="*", help="A B")
    ap.add_argument("--truth", default="both", choices=("all", "airborne", "both"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if len(a.methods) != 2:
        ap.error("give exactly two methods: A B")
    cfg = parse_config(a.config)
    modes = ["all", "airborne"] if a.truth == "both" else [a.truth]
    out = compare(a.methods[0], a.methods[1], modes, cfg)
    reps = {m: report(out, m) for m in modes}
    print("\nself-checks")
    ok = all([selfcheck_counts(reps[m], m) for m in modes])
    print("ALL SELF-CHECKS PASSED" if ok else "SELF-CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
