#!/usr/bin/env python3
"""De-anonymise submission files: anonymized longitude frame -> real frame, for upload.

Every longitude in data/ was shifted by a secret constant (the owner's anonymizer; see
data/Anon_*/ANONYMIZATION_MANIFEST.json, format cuas-anon-manifest/1). Every submission
written by tools/submission.py is therefore in the ANONYMIZED frame. This tool rewrites ONLY
the ``longitude`` column of such a file into the real frame and leaves every other cell
identical (value and type), so the owner can upload it.

Owner's commands, in this order, before uploading (run from the repo root):

  # 1. put the shift in the environment without echoing it or saving it in the shell history
  read -rs CUAS_LON_SHIFT_DEG && export CUAS_LON_SHIFT_DEG
  # 2. self-test with synthetic shifts (needs no key; about 3-5 min on a loaded machine)
  python3 tools/deanonymize.py --selftest
  # 3. OPTIONAL: prove the key and the sign on real data: ORIGINAL (non-anonymized) testing root,
  #    same layout as data/Anon_Testing (only the FriendDronesList/DroneTag_*.csv files are needed).
  #    Without the original data, check the sign instead by looking at one de-anonymized track
  #    position on a map of the test site (a wrong sign puts it far away).
  python3 tools/deanonymize.py --verify-original /path/to/original/Testing --iter testing/S1-I1
  #    -> must end with "VERIFY PASSED". If check (a) fails by a large margin, rerun step 3
  #       with --sign -1, and use the same --sign in step 4.
  # 4. write the upload files (default: the 3 evaluation iterations)
  python3 tools/deanonymize.py --method F0_fuse
  # 5. upload data/derived/upload/F0_fuse/<dataset slug>/scenario_<N>_iteration_<M>.xlsx
  #    (file names are the template names), then:  unset CUAS_LON_SHIFT_DEG

Rules:

* The shift is supplied ONLY at runtime, in the environment variable ``CUAS_LON_SHIFT_DEG``
  (degrees). Convention: anonymized_lon = real_lon + SIGN * SHIFT, so
  real_lon = wrap180(anonymized_lon - SIGN * SHIFT) with wrap180 into [-180, 180).
  ``--sign`` (+1 default, or -1) flips the convention if --verify-original says so.
* The shift and every absolute longitude are never printed, logged or written; error messages
  never echo them. Validation messages that could carry a coordinate are reduced to their tag.
  What is printed instead is a FINGERPRINT: 8 hex characters of PBKDF2-HMAC-SHA256 (200,000
  rounds) of the canonical (sign, shift) with the manifest's key_id as salt, so the owner can
  tell runs with different keys apart. Treat it as local information anyway: a 32-bit tag of a
  short-decimal shift can in principle be brute-forced (the KDF only makes that slow).
* Latitudes, heights, covariances (local NEU, invariant under the rotation), times and labels
  are untouched. A constant longitude shift is a rotation about the Earth's axis, so every
  local ENU quantity, and therefore every GOSPA/recall score, is the same in both frames if the
  frame origin moves with the data (the --selftest checks 'real-frame score == anonymized-frame
  score' prove it numerically, and their mutations show a wrong sign or key fails visibly).
* Without ``CUAS_LON_SHIFT_DEG`` the de-anonymising and verify commands exit with status 2
  before writing anything. Writes go only under data/derived/upload/ (``--selftest`` uses
  data/derived/upload/_selftest/ and deletes it at the end).

Outputs (de-anonymising run):

* ``data/derived/upload/<method>/<iteration slug>/<template file name>``: the upload file, e.g.
  ``upload/F0_fuse/evaluation__DII__S2-I1/scenario_2_iteration_1.xlsx``. Files are staged in
  ``upload/<method>/.staging/`` and moved into place only after every file of the run passed:
  (1) ``submission.validate_submission`` (frame-agnostic: header, grid, time cells, ranges incl.
  longitude in [-180, 180], PSD covariances, UUIDs, dropdowns), (2) every non-longitude cell
  identical to the anonymized input (value and type), (3) re-anonymising the written longitudes
  gives the input's to <= 1e-9 deg, (4) real-frame geometry: the rows' ENU in the real frame
  C.Frame(lat0, wrap180(lon0 - SIGN*SHIFT), h0) equal their ENU in the anonymized
  C.iteration_frame to <= 1e-6 m.
* ``data/derived/upload/<method>/MANIFEST.csv``: columns ``file`` (path under upload/<method>),
  ``iter_key``, ``rows`` (track rows), ``sha256`` (of the upload file), ``fingerprint``. No
  coordinates. Every de-anonymising run first DELETES data/derived/upload/<method>/, so the
  directory never mixes files of runs with a different key or sign (a failed run leaves it empty).

--verify-original DIR (owner, real key): (a) every DroneTag_*.csv of every testing iteration
present in DIR has the same rows, header and cells as the anonymized copy except longitude, and
original_lon == wrap180(anon_lon - SIGN*SHIFT) to <= 1e-4 m east (max |diff| printed in metres,
never values); (b) for each --iter, the de-anonymized submission scored against truth REBUILT
from the ORIGINAL Dronetag files in a real-frame ENU frame equals score.score_rows on the
anonymized file against the anonymized truth (combined, track per GOSPA config, every part,
assignment counts; <= 1e-9), for truth modes 'all' and 'airborne'. Both scores are printed.
The de-anonymized testing file is written to upload/<method>/_verify/ for the comparison and deleted
again when the verification ends (it holds real-frame coordinates).

Usage:
  python3 tools/deanonymize.py --selftest
  python3 tools/deanonymize.py --method NAME [--iters evaluation|all|KEY ...] [--sign 1|-1]
  python3 tools/deanonymize.py --verify-original DIR [--iter KEY ...] [--method NAME] [--sign 1|-1]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuas_common as C  # noqa: E402
import gospa as G  # noqa: E402
import score as SC  # noqa: E402
import submission as S  # noqa: E402
import truth as T  # noqa: E402

ENV_VAR = "CUAS_LON_SHIFT_DEG"
UPLOAD = C.DERIVED / "upload"
SELFTEST_DIR = UPLOAD / "_selftest"
LON_COL = S.COLUMNS.index("longitude")            # 0-based column of longitude (3)
LAT_COL = S.COLUMNS.index("latitude")
ALT_COL = S.COLUMNS.index("altitude_m")
LON_TOL_DEG = 1e-9                                 # file round trip
ENU_TOL_M = 1e-6                                   # real-frame geometry check
SCORE_TOL = 1e-9                                   # score invariance
DRONETAG_TOL_M = 1e-4                              # --verify-original (a), float formatting
KDF_ROUNDS = 200_000
MANIFEST_COLUMNS = ["file", "iter_key", "rows", "sha256", "fingerprint"]
SEED = 20261028
# Validation tags whose messages may quote a coordinate value: printed as the tag only.
COORD_TAGS = {"latitude", "longitude", "numeric"}


class ShiftError(SystemExit):
    """Exit (status 2) for a missing / malformed shift; the message never contains the value."""


# --------------------------------------------------------------------------- #
# The shift
# --------------------------------------------------------------------------- #


def wrap180(x):
    """Angle(s) in degrees -> [-180, 180)."""
    return (np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0


def lon_diff_deg(a, b) -> np.ndarray:
    """Smallest |a - b| between longitudes, in degrees (handles the +-180 seam)."""
    return np.abs(wrap180(np.asarray(a, float) - np.asarray(b, float)))


def to_real(lon_anon, shift: float, sign: int):
    return wrap180(np.asarray(lon_anon, float) - sign * shift)


def to_anon(lon_real, shift: float, sign: int):
    return wrap180(np.asarray(lon_real, float) + sign * shift)


def shift_from_env() -> float:
    raw = os.environ.get(ENV_VAR)
    if raw is None or raw.strip() == "":
        raise ShiftError(f"{ENV_VAR} is not set. Set it in this shell first "
                        f"(read -rs {ENV_VAR} && export {ENV_VAR}); nothing was written.")
    try:
        s = float(raw.strip())
    except ValueError:
        raise ShiftError(f"{ENV_VAR} is not a decimal number of degrees (value not shown); "
                        "nothing was written.") from None
    if not math.isfinite(s) or abs(s) > 360.0:
        raise ShiftError(f"{ENV_VAR} must be a finite number of degrees with |shift| <= 360 "
                        "(value not shown); nothing was written.")
    return s


def key_id() -> str:
    ids = set()
    for root in C.TREES.values():
        man = json.loads((root / "ANONYMIZATION_MANIFEST.json").read_text(encoding="utf-8"))
        if man.get("format") != "cuas-anon-manifest/1":
            raise SystemExit(f"unexpected manifest format in {root.name}")
        ids.add(str(man["key_id"]))
    if len(ids) != 1:
        raise SystemExit("the testing and evaluation manifests name different key_ids")
    return ids.pop()


def fingerprint(shift: float, sign: int) -> str:
    """8 hex chars of a slow salted KDF of the canonical (sign, shift): tells runs apart,
    does not reveal the shift."""
    msg = f"{int(sign):+d}|{float(shift)!r}".encode()
    return hashlib.pbkdf2_hmac("sha256", msg, key_id().encode(), KDF_ROUNDS)[:4].hex()


# --------------------------------------------------------------------------- #
# File rewrite + checks
# --------------------------------------------------------------------------- #


def _cells(path: Path) -> list[tuple]:
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True)
    try:
        ws = wb.worksheets[0]
        ws.reset_dimensions()
        return [tuple(r) for r in ws.iter_rows(values_only=True)]
    finally:
        wb.close()


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def rewrite_longitudes(src: Path, dst: Path, fn) -> int:
    """Copy the submission `src` to `dst` with fn(lon) applied to every numeric longitude cell
    of the data rows (header, other cells, sheet, styles untouched). Returns the cells changed.
    A non-numeric, non-empty longitude cell raises ValueError (input must be valid)."""
    import openpyxl
    wb = openpyxl.load_workbook(src)
    ws = wb.worksheets[0]
    header = tuple(c.value for c in ws[1][:S.N_COLS])
    if header != S.COLUMNS:
        raise ValueError(f"{src.name}: unexpected header")
    cells = [row[0] for row in ws.iter_rows(min_row=2, min_col=LON_COL + 1, max_col=LON_COL + 1)]
    idx = [k for k, c in enumerate(cells) if _is_num(c.value)]
    bad = [k for k, c in enumerate(cells) if not _is_num(c.value) and c.value not in (None, "")]
    if bad:
        raise ValueError(f"{src.name}: {len(bad)} non-numeric longitude cells (first at row {bad[0] + 2})")
    vals = np.array([float(cells[k].value) for k in idx], dtype=float)
    new = np.asarray(fn(vals), dtype=float)
    if not np.all(np.isfinite(new)):
        raise ValueError(f"{src.name}: non-finite longitude after the transform")
    for k, v in zip(idx, new.tolist()):
        cells[k].value = float(v)
    dst.parent.mkdir(parents=True, exist_ok=True)
    wb.save(dst)
    return len(idx)


_SRC: dict[tuple, dict] = {}


def source_info(it: C.Iteration, src: Path) -> dict:
    """Validation problems, track rows and raw cells of an (unchanged) anonymized input file,
    memoised by path, size and mtime (the self-test de-anonymises one file many times)."""
    st = src.stat()
    k = (str(src.resolve()), st.st_size, st.st_mtime_ns)
    if k not in _SRC:
        probs = S.validate_submission(it, src)
        _SRC[k] = {"problems": probs, "rows": None if probs else S.read_submission(it, src),
                   "cells": None if probs else _cells(src)}
    return _SRC[k]


def compare_files(anon: Path, real: Path, shift: float, sign: int, anon_cells: list | None = None) -> dict:
    """Cell-by-cell comparison of an anonymized file and its de-anonymized copy (no values in
    the result): row counts, non-longitude cells that differ in value or type, longitude cells
    whose emptiness differs, and the max re-anonymisation error in degrees."""
    a = _cells(anon) if anon_cells is None else anon_cells
    b = _cells(real)
    out = {"rows_a": len(a), "rows_b": len(b), "other_diff": 0, "lon_shape_diff": 0,
           "lon_err_deg": 0.0, "n_lon": 0}
    if len(a) != len(b):
        out["other_diff"] = -1
        return out
    la, lb = [], []
    for i, (ra, rb) in enumerate(zip(a, b)):
        if len(ra) != len(rb):
            out["other_diff"] += 1
            continue
        for j, (x, y) in enumerate(zip(ra, rb)):
            if j == LON_COL and i > 0:                # data rows; the header is compared below
                if _is_num(x) and _is_num(y):
                    la.append(float(x))
                    lb.append(float(y))
                elif not (x is None and y is None) and x != y:
                    out["lon_shape_diff"] += 1
            elif x != y or type(x) is not type(y):
                out["other_diff"] += 1
    if la:
        out["n_lon"] = len(la)
        out["lon_err_deg"] = float(lon_diff_deg(to_anon(np.array(lb), shift, sign), np.array(la)).max())
    return out


_FRAMES: dict[str, C.Frame] = {}


def anon_frame(it: C.Iteration) -> C.Frame:
    """C.iteration_frame memoised (it reads every sensor file)."""
    if it.key not in _FRAMES:
        _FRAMES[it.key] = C.iteration_frame(it)
    return _FRAMES[it.key]


def real_frame(it: C.Iteration, shift: float, sign: int) -> C.Frame:
    """The iteration frame expressed in the real longitude frame (same tangent plane)."""
    fr = anon_frame(it)
    return C.Frame(fr.lat0, float(to_real(fr.lon0, shift, sign)), fr.h0)


def enu_check(it: C.Iteration, rows_anon: dict, rows_real: dict, shift: float, sign: int) -> float:
    """Max ENU difference (m) between the anonymized rows in the anonymized frame and the
    real rows in the real frame. Inf if the rows do not line up."""
    if len(rows_anon["step"]) != len(rows_real["step"]):
        return float("inf")
    if len(rows_anon["step"]) == 0:
        return 0.0
    if not (np.array_equal(rows_anon["step"], rows_real["step"])
            and np.array_equal(rows_anon["latitude"], rows_real["latitude"])
            and np.array_equal(rows_anon["altitude_m"], rows_real["altitude_m"])):
        return float("inf")
    a = anon_frame(it).to_enu(rows_anon["latitude"], rows_anon["longitude"], rows_anon["altitude_m"])
    b = real_frame(it, shift, sign).to_enu(rows_real["latitude"], rows_real["longitude"], rows_real["altitude_m"])
    return float(max(np.abs(x - y).max() for x, y in zip(a, b)))


def safe_problem(p: str) -> str:
    tag = p.split(":", 1)[0]
    return f"{tag}: (details withheld: may contain a coordinate)" if tag in COORD_TAGS else p


def deanonymize_file(it: C.Iteration, src: Path, dst: Path, shift: float, sign: int) -> dict:
    """Write the real-frame copy of `src` at `dst` (name = template name) and run the four
    post-write checks. Returns a report dict (no coordinates); report['ok'] is the verdict."""
    rep: dict = {"iter_key": it.key, "dst": dst}
    info = source_info(it, src)
    probs_in = info["problems"]
    if probs_in:
        rep.update(ok=False, problems=["input: " + safe_problem(p) for p in probs_in])
        return rep
    n = rewrite_longitudes(src, dst, lambda lon: to_real(lon, shift, sign))
    probs = [safe_problem(p) for p in S.validate_submission(it, dst)]
    cmp = compare_files(src, dst, shift, sign, info["cells"])
    rows_a = info["rows"]
    rows_r = S.read_submission(it, dst)
    enu = enu_check(it, rows_a, rows_r, shift, sign)
    checks = {
        "validate_submission": not probs,
        "other cells identical": cmp["rows_a"] == cmp["rows_b"] and cmp["other_diff"] == 0
                                 and cmp["lon_shape_diff"] == 0,
        "re-anonymised longitude <= 1e-9 deg": cmp["n_lon"] == n and cmp["lon_err_deg"] <= LON_TOL_DEG,
        "real-frame ENU == anonymized-frame ENU (<= 1e-6 m)": enu <= ENU_TOL_M,
    }
    rep.update(ok=all(checks.values()), checks=checks, problems=probs, rows=int(len(rows_r["step"])),
               lon_cells=n, lon_err_deg=cmp["lon_err_deg"], enu_err_m=enu, cmp=cmp)
    return rep


# --------------------------------------------------------------------------- #
# Independent real-frame truth + scorer
# --------------------------------------------------------------------------- #


def _num(cells, n) -> np.ndarray:
    return C.to_float(cells, n)


def build_truth(it: C.Iteration, files: list[tuple[Path, str, str, str]], lon_fn, frame: C.Frame,
                mode: str, max_gap_s: float = T.MAX_GAP_S) -> dict:
    """Truth on the template grid rebuilt from Dronetag CSVs, independently of
    groundtruth.csv.gz / truth.load_truth but with the same rules (truth.py docstring):
    ENU = frame.to_enu(lat, lon_fn(lon), geo_alt), duplicate times averaged, linear
    interpolation between samples <= max_gap_s apart, airborne = height > 5 m, rows sorted
    by (step, flight), flights sorted by flight_key. `files`: (path, flight_key, side, drone_name)."""
    grid = C.template_grid(it)[1]
    steps, fls, cols = [], [], {c: [] for c in ("e", "n", "u", "height")}
    flights = []
    for fi, (path, fkey, side, name) in enumerate(sorted(files, key=lambda f: f[1])):
        cat = T.catalog_row(name, side)
        flights.append(dict(flight_key=fkey, side=side, drone_name=name, identity=T.IDENTITY[side],
                            **{c: cat[c] for c in T.LABEL_COLS}))
        tb = C.read_table(path)
        n = tb.n_rows
        if n == 0:
            continue
        t = C.parse_times(tb.columns["timestamp"])
        lat, lon = _num(tb.columns["latitude"], n), _num(tb.columns["longitude"], n)
        ga, hgt = _num(tb.columns.get("geo_altitude"), n), _num(tb.columns.get("height"), n)
        with np.errstate(invalid="ignore"):
            e, nn, u = frame.to_enu(lat, lon_fn(lon), np.nan_to_num(ga))
        u = np.where(np.isfinite(ga), u, np.nan)
        ok = np.isfinite(t) & np.isfinite(e) & np.isfinite(nn) & np.isfinite(u)
        t, v = t[ok], {"e": e[ok], "n": nn[ok], "u": u[ok], "height": hgt[ok]}
        tu, inv = np.unique(t, return_inverse=True)
        if tu.size != t.size:
            cnt = np.bincount(inv, minlength=tu.size)
            v = {c: np.bincount(inv, weights=np.nan_to_num(x), minlength=tu.size) / cnt for c, x in v.items()}
        else:
            o = np.argsort(t, kind="stable")
            v = {c: x[o] for c, x in v.items()}
        if tu.size < 2:
            continue
        gap_ok = np.diff(tu) <= max_gap_s
        g_idx = np.arange(np.searchsorted(grid, tu[0], "left"), np.searchsorted(grid, tu[-1], "right"))
        tg = grid[g_idx]
        nseg = tu.size - 1
        kr = np.clip(np.searchsorted(tu, tg, "right") - 1, 0, nseg - 1)
        kl = np.clip(np.searchsorted(tu, tg, "left") - 1, 0, nseg - 1)
        in_r = (tu[kr] <= tg) & (tg <= tu[kr + 1]) & gap_ok[kr]
        in_l = (tu[kl] <= tg) & (tg <= tu[kl + 1]) & gap_ok[kl]
        k = np.where(in_r, kr, kl)
        use = in_r | in_l
        g_idx, k, tg = g_idx[use], k[use], tg[use]
        w = (tg - tu[k]) / (tu[k + 1] - tu[k])
        for c in cols:
            cols[c].append(v[c][k] + w * (v[c][k + 1] - v[c][k]))
        steps.append(g_idx)
        fls.append(np.full(g_idx.size, fi, dtype=np.int64))
    if steps:
        step, flight = np.concatenate(steps).astype(np.int64), np.concatenate(fls)
        cc = {c: np.concatenate(x) for c, x in cols.items()}
    else:
        step = flight = np.zeros(0, dtype=np.int64)
        cc = {c: np.zeros(0) for c in cols}
    o = np.lexsort((flight, step))
    step, flight, cc = step[o], flight[o], {c: x[o] for c, x in cc.items()}
    keep = cc["height"] > T.AIRBORNE_HEIGHT_M if mode == "airborne" else np.ones(step.size, bool)
    return dict(n_steps=int(grid.size), step=step[keep], flight=flight[keep], e=cc["e"][keep],
                n=cc["n"][keep], u=cc["u"][keep], flights=flights)


def anon_dronetag_files(it: C.Iteration) -> list[tuple[Path, str, str, str]]:
    return [(g.path, g.flight_key, g.side, g.drone_name) for g in C.groundtruth_files(it)]


def real_score(rows: dict, tr: dict, frame: C.Frame, configs=SC.CONFIGS, primary=SC.PRIMARY) -> dict:
    """Score `rows` (real-frame lat/lon) against the real-frame truth `tr` in `frame`, with
    the definitions of score.py's docstring re-implemented here (GOSPA via gospa.py)."""
    sy = np.asarray(rows["step"], dtype=np.int64)
    if len(sy):
        y = np.column_stack(frame.to_enu(rows["latitude"], rows["longitude"], rows["altitude_m"]))
    else:
        y = np.zeros((0, 3))
    x = np.column_stack([tr["e"], tr["n"], tr["u"]])
    sx = tr["step"]
    out: dict = {}
    pairs = None
    for name, c, p, dims in configs:
        seq = G.gospa_sequence(sx, x[:, :dims], sy, y[:, :dims], tr["n_steps"], c, p)
        worst = float((c ** p / 2.0 * (seq["n_truth"] + seq["n_est"])).sum())
        out[f"track[{name}]"] = 1.0 - float(seq["cost"].sum()) / worst if worst > 0 else 1.0
        out[f"n_assigned[{name}]"] = int(len(seq["pairs"]))
        out[f"n_missed[{name}]"] = int(seq["n_missed"].sum())
        out[f"n_false[{name}]"] = int(seq["n_false"].sum())
        if name == primary:
            pairs = seq["pairs"]
    assigned = np.full(len(sx), -1, dtype=np.int64)
    if pairs is not None and len(pairs):
        assigned[pairs[:, 1]] = pairs[:, 2]
    fl = tr["flights"]

    def recall(col: str, eligible_fn) -> tuple[int, int]:
        tv = np.array([fl[f][col] for f in tr["flight"]], dtype=object)
        ev = np.asarray(rows[col], dtype=object)
        elig = eligible_fn(tv)
        hit = 0
        for j in np.flatnonzero(elig & (assigned >= 0)):
            hit += ev[assigned[j]] == tv[j]
        return int(hit), int(elig.sum())

    def ratio(ok_n):
        return ok_n[0] / ok_n[1] if ok_n[1] else float("nan")

    cls = ratio(recall("classification", lambda v: np.isin(v, SC.KNOWN_CLASSES)))
    ide = ratio(recall("identity", lambda v: np.isin(v, ("Friendly", "Hostile"))))
    meta = {}
    for col in C.METADATA_COLUMNS:
        ok, n_el = recall(col, lambda v: v != "Unknown")
        meta[col] = ok / n_el if n_el else float("nan")
        out[f"meta[{col}]"] = meta[col]
    mv = [v for v in meta.values() if v == v]
    parts = {"track": out[f"track[{primary}]"], "classification": cls, "identity": ide,
             "metadata": sum(mv) / len(mv) if mv else float("nan")}
    have = {k: v for k, v in parts.items() if v == v}
    wsum = sum(SC.WEIGHTS[k] for k in have)
    parts["combined"] = sum(SC.WEIGHTS[k] * v for k, v in have.items()) / wsum if wsum else float("nan")
    out.update(parts)
    return out


def flatten_score(res: dict) -> dict:
    """score.score_rows result -> the same flat keys as real_score."""
    out = dict(res["parts"])
    for name, *_ in SC.CONFIGS:
        if name in res:
            out[f"track[{name}]"] = res[name]["track"]
            out[f"n_assigned[{name}]"] = res[name]["n_assigned"]
            out[f"n_missed[{name}]"] = res[name]["n_missed"]
            out[f"n_false[{name}]"] = res[name]["n_false"]
    for col, (ok, n_el) in res["metadata"].items():
        out[f"meta[{col}]"] = ok / n_el if n_el else float("nan")
    return out


def score_diff(a: dict, b: dict) -> float:
    """Max |a - b| over the shared keys (counts included; NaN == NaN); inf if keys differ."""
    if set(a) != set(b):
        return float("inf")
    worst = 0.0
    for k in a:
        x, y = float(a[k]), float(b[k])
        if x != x and y != y:
            continue
        if (x != x) != (y != y):
            return float("inf")
        worst = max(worst, abs(x - y))
    return worst


def with_lon(rows: dict, lon: np.ndarray) -> dict:
    out = dict(rows)
    out["longitude"] = np.asarray(lon, dtype=float)
    return out


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_iters(spec: list[str]) -> list[C.Iteration]:
    its = C.iterations()
    by = {i.key: i for i in its}
    if spec == ["evaluation"]:
        return [i for i in its if i.tree == "evaluation"]
    if spec == ["all"]:
        return its
    bad = [k for k in spec if k not in by]
    if bad:
        raise SystemExit(f"unknown iteration key(s) {bad}; known: {sorted(by)}")
    return [by[k] for k in spec]


def _print_report(rep: dict) -> None:
    for name, ok in rep.get("checks", {}).items():
        print(f"    [{'ok' if ok else 'FAIL'}] {name}")
    for p in rep.get("problems", []):
        print(f"    PROBLEM {p}")


def run_deanonymize(method: str, iter_spec: list[str], sign: int, out_root: Path = UPLOAD) -> int:
    shift = shift_from_env()                          # exits before anything is written
    its = resolve_iters(iter_spec)
    srcs = [(it, C.submission_path(method, it)) for it in its]
    missing = [str(p.relative_to(C.REPO)) for _, p in srcs if not p.exists()]
    if missing:
        print(f"missing submission file(s), nothing written: {missing}")
        return 1
    fp = fingerprint(shift, sign)
    print(f"key fingerprint {fp} (sign {sign:+d}); {len(its)} file(s) of method {method!r}")
    base = out_root / method
    shutil.rmtree(base, ignore_errors=True)          # never mix runs with different keys / signs
    staging = base / ".staging"
    reports = []
    try:
        for it, src in srcs:
            dst = staging / C.iter_slug(it.key) / src.name
            t0 = time.time()
            rep = deanonymize_file(it, src, dst, shift, sign)
            reports.append(rep)
            print(f"  {it.key}: {'ok' if rep['ok'] else 'FAILED'} ({rep.get('rows', 0)} track rows, "
                  f"{time.time() - t0:.1f} s)")
            _print_report(rep)
        if not all(r["ok"] for r in reports):
            print("DE-ANONYMISATION FAILED: no file was moved into place")
            return 1
        man_path = base / "MANIFEST.csv"
        man = {}
        if man_path.exists():
            old = C.read_columns(man_path)
            for i in range(len(old["file"])):
                man[str(old["file"][i])] = {k: str(old[k][i]) for k in MANIFEST_COLUMNS}
        for rep in reports:
            rel = rep["dst"].relative_to(staging)
            final = base / rel
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(rep["dst"], final)
            man[str(rel)] = {"file": str(rel), "iter_key": rep["iter_key"], "rows": str(rep["rows"]),
                             "sha256": _sha256(final), "fingerprint": fp}
            print(f"  wrote {final.relative_to(C.REPO)}")
        keys = sorted(man)
        C.write_columns(man_path, {c: [man[k][c] for k in keys] for c in MANIFEST_COLUMNS}, MANIFEST_COLUMNS)
        print(f"  wrote {man_path.relative_to(C.REPO)}")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print("DE-ANONYMISATION PASSED: files are in the REAL longitude frame (upload them as they are)")
    return 0


def _fmt_score(d: dict) -> str:
    return "  ".join(f"{k} {d[k]:.6f}" for k in ("combined", "track", "classification", "identity", "metadata"))


def verify_dronetags(orig_root: Path, shift: float, sign: int) -> tuple[bool, list[str]]:
    """Check (a) over every testing iteration whose FriendDronesList exists under orig_root."""
    lines, ok_all, n_files, n_iters = [], True, 0, 0
    worst_m = 0.0
    for it in [i for i in C.iterations() if i.has_groundtruth]:
        rel = it.path.relative_to(C.TREES["testing"])
        odir = orig_root / rel / "FriendDronesList"
        if not odir.is_dir():
            continue
        n_iters += 1
        for g in C.groundtruth_files(it):
            op = odir / g.path.name
            if not op.exists():
                lines.append(f"  {it.key} {g.path.name}: MISSING in the original tree")
                ok_all = False
                continue
            a, o = C.read_table(g.path), C.read_table(op)
            n_files += 1
            same_shape = list(a.columns) == list(o.columns) and a.n_rows == o.n_rows
            if not same_shape:
                lines.append(f"  {it.key} {g.path.name}: header or row count differs")
                ok_all = False
                continue
            other = sum(1 for c in a.columns if c != "longitude"
                        for x, y in zip(a.columns[c], o.columns[c]) if x != y)
            la, lo = _num(a.columns["longitude"], a.n_rows), _num(o.columns["longitude"], o.n_rows)
            lat = _num(a.columns["latitude"], a.n_rows)
            fin = np.isfinite(la) & np.isfinite(lo)
            blank_mismatch = int((np.isfinite(la) != np.isfinite(lo)).sum())
            d_deg = lon_diff_deg(to_real(la[fin], shift, sign), lo[fin])
            d_m = np.radians(d_deg) * C.WGS84_A * np.cos(np.radians(np.nan_to_num(lat[fin])))
            mx = float(d_m.max()) if d_m.size else 0.0
            worst_m = max(worst_m, mx)
            good = other == 0 and blank_mismatch == 0 and mx <= DRONETAG_TOL_M
            ok_all &= good
            if not good:
                lines.append(f"  {it.key} {g.path.name}: other cells differing {other}, blank-longitude "
                             f"mismatches {blank_mismatch}, max |lon diff| {mx:.3g} m east")
    if n_files == 0:
        lines.append("  no Dronetag file of any testing iteration found under the original root")
        ok_all = False
    lines.insert(0, f"(a) {n_files} Dronetag files in {n_iters} testing iteration(s): max |original - "
                    f"de-anonymized| longitude {worst_m:.3g} m east (tolerance {DRONETAG_TOL_M:g} m), "
                    f"{'PASS' if ok_all else 'FAIL'}")
    return ok_all, lines


def verify_scores(it: C.Iteration, orig_root: Path, method: str, shift: float, sign: int,
                  out_dir: Path) -> tuple[bool, list[str], float]:
    """Check (b) on one testing iteration; returns (ok, lines, max diff)."""
    src = C.submission_path(method, it)
    dst = out_dir / C.iter_slug(it.key) / src.name
    rep = deanonymize_file(it, src, dst, shift, sign)
    lines = []
    if not rep["ok"]:
        return False, [f"(b) {it.key}: de-anonymising the {method} file failed"] + [
            f"    {k}: {v}" for k, v in rep.get("checks", {}).items()] + rep.get("problems", []), float("inf")
    rows_anon = source_info(it, src)["rows"]
    rows_real = S.read_submission(it, dst)
    odir = orig_root / it.path.relative_to(C.TREES["testing"]) / "FriendDronesList"
    files = [(odir / g.path.name, g.flight_key, g.side, g.drone_name) for g in C.groundtruth_files(it)]
    if not all(f[0].exists() for f in files):
        return False, [f"(b) {it.key}: original Dronetag files missing"], float("inf")
    fr = real_frame(it, shift, sign)
    ok_all, worst = True, 0.0
    for mode in ("all", "airborne"):
        a = flatten_score(SC.score_rows(it, rows_anon, mode))
        b = real_score(rows_real, build_truth(it, files, lambda lon: lon, fr, mode), fr)
        d = score_diff(a, b)
        worst = max(worst, d)
        ok = d <= SCORE_TOL
        ok_all &= ok
        lines += [f"(b) {it.key} truth={mode}: {'PASS' if ok else 'FAIL'} (max |diff| {d:.3g})",
                  f"    anonymized frame: {_fmt_score(a)}",
                  f"    real frame      : {_fmt_score(b)}"]
    return ok_all, lines, worst


def run_verify(orig_root: Path, iter_keys: list[str], method: str, sign: int) -> int:
    shift = shift_from_env()
    orig_root = Path(orig_root)
    if not orig_root.is_dir():
        print(f"original root {orig_root} is not a directory; nothing was written")
        return 1
    its = resolve_iters(iter_keys)
    if any(not i.has_groundtruth for i in its):
        print("--iter must name testing iterations (they have ground truth)")
        return 1
    print(f"key fingerprint {fingerprint(shift, sign)} (sign {sign:+d})")
    ok_a, lines = verify_dronetags(orig_root, shift, sign)
    print("\n".join(lines), flush=True)
    ok_b = True
    vdir = UPLOAD / method / "_verify"
    try:
        for it in its:
            ok, ls, _ = verify_scores(it, orig_root, method, shift, sign, vdir)
            ok_b &= ok
            print("\n".join(ls), flush=True)
    finally:
        shutil.rmtree(vdir, ignore_errors=True)       # real-frame coordinates: do not leave them on disk
    if ok_a and ok_b:
        print("VERIFY PASSED: the key and sign are right; the upload files will score as the anonymized ones")
        return 0
    if not ok_a:
        print("VERIFY FAILED in (a). If the longitude differences are large, try the other --sign; "
              "if they stay large, the key is wrong.")
    else:
        print("VERIFY FAILED in (b)")
    return 1


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #


def _synthetic_original(it: C.Iteration, root: Path, shift: float, sign: int) -> None:
    """A synthetic 'original' tree: the anonymized Dronetag CSVs of `it` with longitude
    replaced by wrap180(anon - sign*shift) (repr), every other cell and the dialect kept."""
    odir = root / it.path.relative_to(C.TREES["testing"]) / "FriendDronesList"
    odir.mkdir(parents=True, exist_ok=True)
    for g in C.groundtruth_files(it):
        with g.path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        j = rows[0].index("longitude")
        for r in rows[1:]:
            if len(r) == len(rows[0]) and r[j]:
                r[j] = repr(float(to_real(float(r[j]), shift, sign)))
        with (odir / g.path.name).open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh, lineterminator="\n").writerows(rows)


def selftest() -> int:
    t_start = time.time()
    fails: list[str] = []
    count = [0]

    def check(name: str, cond: bool, detail: str = "") -> None:
        count[0] += 1
        print(f"[{count[0]:2d}] {'ok  ' if cond else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""),
              flush=True)
        if not cond:
            fails.append(name)

    shutil.rmtree(SELFTEST_DIR, ignore_errors=True)
    SELFTEST_DIR.mkdir(parents=True, exist_ok=True)
    by = {i.key: i for i in C.iterations()}
    it1, it2 = by["testing/S1-I1"], by["testing/S4-I4"]
    method = "F0_fuse"
    rng = np.random.default_rng(SEED)
    lon0 = anon_frame(it1).lon0
    # Synthetic shifts (never the real key): two random, one tiny, and two that put the real
    # longitudes of the iteration on both sides of the +-180 seam.
    shifts = {"random-a": float(rng.uniform(-180, 180)), "random-b": float(rng.uniform(-180, 180)),
              "tiny": 1e-6, "seam-a": lon0 + 180.0, "seam-b": lon0 - 180.0 + 0.003}

    try:
        # ---- (1) file round trip ------------------------------------------------------- #
        src = C.submission_path(method, it1)
        rows_anon1 = source_info(it1, src)["rows"]
        lon1 = rows_anon1["longitude"]
        for label, s in shifts.items():               # in memory: all shifts, both signs
            for sign in (1, -1):
                err = float(lon_diff_deg(to_anon(to_real(lon1, s, sign), s, sign), lon1).max())
                check(f"shift {label} sign {sign:+d}: to_anon(to_real(lon)) == lon (<= 1e-9 deg, in memory)",
                      err <= LON_TOL_DEG, f"{len(lon1)} longitudes, max err {err:.2e} deg")
        for label in ("random-b", "seam-a", "seam-b"):  # through the .xlsx files
            s = shifts[label]
            sign = -1 if label == "random-b" else 1
            d = SELFTEST_DIR / "rt" / label
            real = d / "real" / src.name
            rep = deanonymize_file(it1, src, real, s, sign)
            back = d / "back" / src.name
            rewrite_longitudes(real, back, lambda lon, s=s, sign=sign: to_anon(lon, s, sign))
            cb = compare_files(src, back, 0.0, 1, source_info(it1, src)["cells"])   # back == original
            seam = ""
            if label.startswith("seam"):
                rl = S.read_submission(it1, real)["longitude"]
                n_hi, n_lo = int((rl > 179.0).sum()), int((rl < -179.0).sum())
                seam = f", rows east of the seam {n_lo}, west {n_hi}"
                check(f"shift {label}: the real longitudes straddle +-180 (wrap exercised)",
                      n_hi > 0 and n_lo > 0 and bool(np.all((rl >= -180) & (rl < 180))), seam[2:])
            check(f"shift {label} (sign {sign:+d}): de-anonymized file passes the 4 upload checks",
                  rep["ok"], f"{rep.get('checks')} {rep.get('problems', [])[:2]}")
            check(f"shift {label}: de-anonymize then re-anonymize == original (lon <= 1e-9 deg, "
                  f"every other cell identical)",
                  cb["rows_a"] == cb["rows_b"] and cb["other_diff"] == 0 and cb["lon_shape_diff"] == 0
                  and cb["n_lon"] == len(rows_anon1["step"]) and cb["lon_err_deg"] <= LON_TOL_DEG,
                  f"{cb['n_lon']} longitudes, max err {cb['lon_err_deg']:.2e} deg, other diffs "
                  f"{cb['other_diff']}")
            shutil.rmtree(d / "back", ignore_errors=True)
            if label != "seam-a":
                shutil.rmtree(d, ignore_errors=True)
        # mutation of the round-trip comparator: a 1e-4 deg error must be caught
        s = shifts["random-a"]
        m = SELFTEST_DIR / "rt" / "mut" / src.name
        rewrite_longitudes(src, m, lambda lon: to_real(lon, s + 1e-4, 1))
        cm = compare_files(src, m, s, 1)
        check("mutation: a 1e-4 deg shift error fails the round-trip comparator",
              cm["lon_err_deg"] > 1e3 * LON_TOL_DEG, f"max err {cm['lon_err_deg']:.2e} deg")
        rep_m = deanonymize_file(it1, src, m, s, 1)   # correct file -> fine; then tamper a cell
        import openpyxl
        wb = openpyxl.load_workbook(m)
        wb.worksheets[0].cell(row=2, column=LAT_COL + 1).value = 1.0
        wb.save(m)
        cm2 = compare_files(src, m, s, 1)
        check("mutation: one changed non-longitude cell is caught by the comparator",
              rep_m["ok"] and cm2["other_diff"] == 1, f"other diffs {cm2['other_diff']}")
        shutil.rmtree(SELFTEST_DIR / "rt" / "mut", ignore_errors=True)

        # ---- (2) independent truth rebuild == truth.load_truth (shift 0) --------------- #
        for it in (it1, it2):
            for mode in ("all", "airborne"):
                tr = T.load_truth(it, mode)
                mine = build_truth(it, anon_dronetag_files(it), lambda lon: lon, anon_frame(it), mode)
                same = (np.array_equal(mine["step"], tr.step) and np.array_equal(mine["flight"], tr.flight)
                        and [f["flight_key"] for f in mine["flights"]] == [f["flight_key"] for f in tr.flights])
                err = max(float(np.abs(mine[c] - getattr(tr, c)).max()) for c in ("e", "n", "u")) \
                    if same and len(tr.step) else (0.0 if same else float("inf"))
                labels = all(mine["flights"][k][c] == tr.flights[k][c] for k in range(len(tr.flights))
                             for c in T.LABEL_COLS + ("identity",)) if same else False
                check(f"rebuilt truth == truth.load_truth on {it.key} ({mode}): same rows, labels, "
                      f"e/n/u <= 1e-6 m", same and labels and err <= 1e-6,
                      f"{len(tr.step)} rows, max {err:.2e} m")

        # ---- (3) score invariance under synthetic shifts ------------------------------- #
        inv_shifts = {k: shifts[k] for k in ("random-a", "seam-a", "random-b")}
        worst_inv = 0.0
        mut_margin = {}
        for it in (it1, it2):
            rows_anon = source_info(it, C.submission_path(method, it))["rows"]
            for mode in ("all", "airborne"):
                a = flatten_score(SC.score_rows(it, rows_anon, mode))
                for label, s in inv_shifts.items():
                    sign = -1 if label == "random-b" else 1
                    if it is it1 and label == "seam-a":       # through the written file
                        rows_real = S.read_submission(it, SELFTEST_DIR / "rt" / "seam-a" / "real" / src.name)
                    else:
                        rows_real = with_lon(rows_anon, to_real(rows_anon["longitude"], s, sign))
                    fr = real_frame(it, s, sign)
                    truth_real = build_truth(it, anon_dronetag_files(it),
                                             lambda lon, s=s, sign=sign: to_real(lon, s, sign), fr, mode)
                    c = real_score(rows_real, truth_real, fr)
                    b = flatten_score(SC.score_rows(
                        it, with_lon(rows_real, to_anon(rows_real["longitude"], s, sign)), mode))
                    dc, db = score_diff(a, c), score_diff(a, b)
                    worst_inv = max(worst_inv, dc, db)
                    check(f"{it.key} {mode} shift {label}: real-frame score == anonymized-frame score "
                          f"(independent scorer and score_rows on re-anonymized rows, <= 1e-9)",
                          dc <= SCORE_TOL and db <= SCORE_TOL,
                          f"combined {a['combined']:.6f} vs {c['combined']:.6f}; max diff {dc:.2e} / {db:.2e}")
                    if it is it1 and mode == "all" and label == "random-a":
                        for mlabel, (ms, msign) in {"wrong sign": (s, -sign),
                                                    "shift + 1e-4 deg": (s + 1e-4, sign)}.items():
                            rm = with_lon(rows_anon, to_real(rows_anon["longitude"], ms, msign))
                            cm_ = real_score(rm, truth_real, fr)
                            mut_margin[mlabel] = (score_diff(a, cm_), abs(cm_["combined"] - a["combined"]))
        print(f"     worst invariance diff {worst_inv:.2e}", flush=True)
        for mlabel, (dm, dcomb) in mut_margin.items():
            check(f"mutation: submission de-anonymized with {mlabel} makes the invariance check FAIL "
                  f"(combined moves by > 1e-4 = 1e5 x tolerance)", dm > SCORE_TOL and dcomb > 1e-4,
                  f"|d combined| {dcomb:.3g}, max diff incl. counts {dm:.3g}")

        # ---- (4) --verify-original on a synthetic original tree ------------------------ #
        s_v, sign_v = shifts["random-b"], 1
        orig = SELFTEST_DIR / "original"
        _synthetic_original(it1, orig, s_v, sign_v)
        vdir = SELFTEST_DIR / "verify"
        ok_a, la = verify_dronetags(orig, s_v, sign_v)
        ok_b, lb, wb_ = verify_scores(it1, orig, method, s_v, sign_v, vdir)
        print("     " + "\n     ".join(la + lb), flush=True)
        check("verify-original with the right shift: (a) Dronetag files match and (b) scores match",
              ok_a and ok_b, f"(b) max diff {wb_:.2e}")
        ok_a2, la2 = verify_dronetags(orig, s_v + 1e-4, sign_v)
        ok_b2, _, wb2 = verify_scores(it1, orig, method, s_v + 1e-4, sign_v, vdir)
        check("mutation: verify-original with shift + 1e-4 deg FAILS (a) and (b)", not ok_a2 and not ok_b2,
              f"{la2[0][:110]}...; (b) max diff {wb2:.3g}")
        ok_a3, la3 = verify_dronetags(orig, s_v, -sign_v)
        check("mutation: verify-original with the wrong sign FAILS (a)", not ok_a3, la3[0][:110])

        # ---- (5) CLI: no key -> exit 2, nothing written; with a key -> no leak ---------- #
        def snapshot() -> list[tuple[str, int]]:
            return sorted((str(p.relative_to(UPLOAD)), p.stat().st_mtime_ns) for p in UPLOAD.rglob("*")
                          if "_selftest" not in p.parts)
        env = {k: v for k, v in os.environ.items() if k != ENV_VAR}
        before = snapshot()
        res = subprocess.run([sys.executable, __file__, "--method", method], env=env,
                             capture_output=True, text=True, timeout=300)
        check("no key: '--method F0_fuse' exits 2 with a clear message and writes nothing",
              res.returncode == 2 and ENV_VAR in (res.stdout + res.stderr) and snapshot() == before,
              (res.stdout + res.stderr).strip()[:120])
        res = subprocess.run([sys.executable, __file__, "--method", method], capture_output=True, text=True,
                             env=dict(env, **{ENV_VAR: "not-a-number-12.5"}), timeout=300)
        check("malformed key: exits 2 without echoing the value",
              res.returncode == 2 and "12.5" not in res.stdout + res.stderr and snapshot() == before)
        s_cli = shifts["random-a"]
        cli_out = SELFTEST_DIR / "cli"
        res = subprocess.run([sys.executable, __file__, "--method", method, "--iters", it2.key,
                              "--out", str(cli_out)], env=dict(env, **{ENV_VAR: repr(s_cli)}),
                             capture_output=True, text=True, timeout=600)
        up = cli_out / method / C.iter_slug(it2.key) / C.template_path(it2).name
        man = cli_out / method / "MANIFEST.csv"
        text = res.stdout + res.stderr + (man.read_text(encoding="utf-8") if man.exists() else "")
        fr2 = anon_frame(it2)
        secrets = [s_cli % 360.0, float(wrap180(s_cli)), float(to_real(fr2.lon0, s_cli, 1)), fr2.lon0]

        def leaks(txt: str) -> tuple[int, list[float]]:
            found = [float(x) for x in re.findall(r"-?\d+\.\d{3,}", txt)]
            return len(found), [x for x in found for sv in secrets if abs(float(wrap180(x - sv))) < 1e-3]
        n_nums, leak = leaks(text)
        planted = leaks(text + f" lon {float(to_real(fr2.lon0 + 0.0004, s_cli, 1)):.5f}")[1]
        check("CLI with a synthetic key: exit 0, upload file + MANIFEST written, valid",
              res.returncode == 0 and up.exists() and man.exists() and S.validate_submission(it2, up) == []
              and fingerprint(s_cli, 1) in text, res.stdout.strip().splitlines()[-1][:120] if res.stdout else "")
        check("CLI output and MANIFEST.csv contain neither the shift nor any absolute longitude "
              "(detector catches a planted real longitude)",
              not leak and repr(s_cli) not in text and len(planted) == 1,
              f"{n_nums} decimal numbers scanned, {len(leak)} suspicious, planted caught {len(planted)}")
        mrow = C.read_columns(man) if man.exists() else {"sha256": [""]}
        check("MANIFEST sha256 matches the upload file", man.exists() and mrow["sha256"][0] == _sha256(up))
        check("fingerprint changes with the shift and with the sign",
              len({fingerprint(s_cli, 1), fingerprint(s_cli + 1e-9, 1), fingerprint(s_cli, -1)}) == 3)
    finally:
        shutil.rmtree(SELFTEST_DIR, ignore_errors=True)

    print(f"\nselftest {time.time() - t_start:.0f} s")
    if fails:
        print(f"{len(fails)} SELF-CHECK(S) FAILED: {fails}")
        return 1
    print("ALL SELF-CHECKS PASSED")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--method", help="submission method folder under data/derived/submissions/")
    ap.add_argument("--iters", nargs="+", default=["evaluation"],
                    help="'evaluation' (default), 'all', or iteration keys")
    ap.add_argument("--sign", type=int, choices=(1, -1), default=1,
                    help=f"anonymized = real + SIGN * {ENV_VAR} (default +1)")
    ap.add_argument("--verify-original", metavar="DIR", type=Path,
                    help="ORIGINAL (non-anonymized) testing root, same layout as data/Anon_Testing")
    ap.add_argument("--iter", nargs="+", default=["testing/S1-I1"],
                    help="testing iteration(s) scored by --verify-original (default testing/S1-I1)")
    ap.add_argument("--out", type=Path, default=UPLOAD, help=argparse.SUPPRESS)
    a = ap.parse_args()
    try:
        if a.selftest:
            return selftest()
        out = a.out.resolve()
        if out != UPLOAD.resolve() and UPLOAD.resolve() not in out.parents:
            ap.error("--out must lie under data/derived/upload/")
        if a.verify_original:
            return run_verify(a.verify_original, a.iter, a.method or "F0_fuse", a.sign)
        if a.method:
            return run_deanonymize(a.method, a.iters, a.sign, out)
    except ShiftError as e:
        print(e.code, file=sys.stderr)
        return 2
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
