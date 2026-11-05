#!/usr/bin/env python3
"""Writer, reader and validator for the per-iteration .xlsx submission.

Rules (README/README_SubmissionTemplate.pdf, CLAUDE.md "Submission template"):

* The file is built FROM the official template (openpyxl keeps styles / properties) and
  keeps the template's file name ``scenario_<N>_iteration_<M>.xlsx``.
* ``timestamp`` / ``unix_timestamp`` are copied verbatim from the template (value AND type;
  the first unix value is an int, the rest floats). They are never recomputed.
* One row per tracked target per timestamp. A timestamp with K tracks has K contiguous
  rows; a timestamp with no track keeps its single row with only the two time cells.
  Rows are written in grid order, then by ``track_id`` within a step.
* Position is WGS84 lat/lon + ellipsoid height; covariance is the upper triangle of a local
  North-East-Up matrix in m^2 (C_nn, C_ne, C_nu, C_ee, C_eu, C_uu). Floats are numeric cells.
* Categorical columns use the exact strings in ``C.DROPDOWNS`` (the templates carry no data
  validations, so the validator is the only guard; en-dash bands, 'DJI Ocusyinc' typo).

ANONYMIZED LONGITUDES: every coordinate this module writes is in the ANONYMIZED longitude
frame (the secret constant shift is still applied). De-anonymising is the owner's own step
before upload; this module never tries to undo or infer the shift.

Track-rows contract (writer input, reader output): a dict with ``step`` (int grid index),
float arrays ``latitude longitude altitude_m C_nn C_ne C_nu C_ee C_eu C_uu
classification_confidence`` and string arrays ``track_id classification identity
uav_model/manufacturer protocol serial_number signal_frequency_band``.

``validate_submission`` returns the list of problems (``[]`` = valid); every message starts
with a short tag (``header``, ``grid-missing``, ...) followed by a colon. Non-fatal warnings
go to the optional ``warnings`` list argument.

Usage:
  python3 tools/submission.py --selftest
  python3 tools/submission.py --validate <iteration-key> <file.xlsx>
  python3 tools/submission.py --write-empty <iteration-key> <out-dir>
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuas_common as C  # noqa: E402

COLUMNS = C.SUBMISSION_COLUMNS
N_COLS = len(COLUMNS)                      # 19
FLOAT_FIELDS = ("latitude", "longitude", "altitude_m", "C_nn", "C_ne", "C_nu", "C_ee",
                "C_eu", "C_uu", "classification_confidence")
STR_FIELDS = ("track_id", "classification", "identity", "uav_model/manufacturer",
              "protocol", "serial_number", "signal_frequency_band")
BODY = COLUMNS[2:]                                             # the 17 non-time columns
TRACK_FIELDS = ("step",) + BODY                                # contract keys, column order
LAT_RANGE, LON_RANGE, ALT_RANGE = (-90.0, 90.0), (-180.0, 180.0), (-500.0, 10000.0)
PSD_TOL = 1e-6
MAX_ROWS_PER_STEP_WARN = 50
ANON_REMINDER = ("NOTE: coordinates are in the ANONYMIZED longitude frame; de-anonymising "
                 "is the owner's own step before upload.")


def empty_rows() -> dict[str, np.ndarray]:
    """Zero-length track-rows dict."""
    out: dict[str, np.ndarray] = {"step": np.zeros(0, dtype=np.int64)}
    for c in FLOAT_FIELDS:
        out[c] = np.zeros(0, dtype=float)
    for c in STR_FIELDS:
        out[c] = np.empty(0, dtype=object)
    return out


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #


def _canon_uuid(v: str) -> str:
    """Canonical lowercase hyphenated spelling of a parseable UUID, else `v` unchanged
    (unparseable ids are left for the validator to report)."""
    try:
        return str(uuid.UUID(v))
    except ValueError:
        return v


def _check_rows(rows: dict, n_steps: int) -> tuple[int, dict[str, list]]:
    """Structural checks of the writer input; returns (n_rows, columns as python lists)."""
    missing = [k for k in TRACK_FIELDS if k not in rows]
    if missing:
        raise ValueError(f"rows is missing keys {missing}")
    step = np.asarray(rows["step"])
    n = len(step)
    if n and step.dtype.kind not in "iu":
        raise ValueError("rows['step'] must be integer grid indices")
    if n and (step.min() < 0 or step.max() >= n_steps):
        raise ValueError(f"rows['step'] outside the grid [0, {n_steps})")
    cols: dict[str, list] = {"step": step.astype(np.int64).tolist()}
    for k in FLOAT_FIELDS:
        a = np.asarray(rows[k], dtype=float)
        if a.shape != (n,):
            raise ValueError(f"rows[{k!r}] has shape {a.shape}, expected ({n},)")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"rows[{k!r}] contains non-finite values")
        cols[k] = a.tolist()
    for k in STR_FIELDS:
        v = list(rows[k])
        if len(v) != n:
            raise ValueError(f"rows[{k!r}] has {len(v)} entries, expected {n}")
        bad = next((x for x in v if not isinstance(x, str) or x == ""), None)
        if bad is not None:
            raise ValueError(f"rows[{k!r}] must hold non-empty str, found {bad!r}")
        if k == "track_id":                  # one spelling per UUID (sorting, duplicate checks)
            v = [_canon_uuid(x) for x in v]
        cols[k] = v
    return n, cols


def write_submission(it: C.Iteration, rows: dict, out_path) -> Path:
    """Write `rows` onto the iteration's template and save as `out_path` (its name must
    equal the template's). Structure is checked here; content (dropdowns, ranges, PSD) is
    left to `validate_submission`, which callers should run on the result."""
    import openpyxl
    out_path = Path(out_path)
    tpl = C.template_path(it)
    if out_path.name != tpl.name:
        raise ValueError(f"out_path name {out_path.name!r} must equal template name {tpl.name!r}")
    if out_path.resolve() == tpl.resolve() or (out_path.exists() and os.path.samefile(out_path, tpl)):
        raise ValueError(f"refusing to overwrite the official template {tpl}")
    wb = openpyxl.load_workbook(tpl)
    ws = wb.worksheets[0]
    header = tuple(c.value for c in ws[1][:N_COLS])
    if header != COLUMNS:
        raise ValueError(f"unexpected template header: {header}")
    # Verbatim time cells (value and type) from the template.
    grid = [(r[0].value, r[1].value) for r in ws.iter_rows(min_row=2, max_col=2)
            if r[0].value is not None]
    n_steps = len(grid)
    n, cols = _check_rows(rows, n_steps)

    step = np.asarray(cols["step"], dtype=np.int64)
    if n:
        _, tid_code = np.unique(np.asarray(cols["track_id"], dtype=object), return_inverse=True)
        order = np.lexsort((tid_code, step)).tolist()
    else:
        order = []
    ordered = {k: [cols[k][i] for i in order] for k in TRACK_FIELDS}
    ostep = ordered["step"]

    ws.delete_rows(2, max(ws.max_row, 2))
    ws._current_row = 1                      # delete_rows leaves the append cursor behind
    body_cols = [ordered[c] for c in BODY]
    pos = 0                                  # next unwritten track row
    for s in range(n_steps):
        ts, unix = grid[s]
        if pos >= n or ostep[pos] != s:
            ws.append([ts, unix])
            continue
        while pos < n and ostep[pos] == s:
            ws.append([ts, unix] + [col[pos] for col in body_cols])
            pos += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    return out_path


# --------------------------------------------------------------------------- #
# Reader
# --------------------------------------------------------------------------- #


def _is_empty(v) -> bool:
    return v is None or (isinstance(v, str) and v == "")


def _load_raw(path: Path) -> tuple[list[str], tuple, list[tuple], list[int], list[bool], list[int]]:
    """(sheet names, header row, body rows padded/truncated to N_COLS, their spreadsheet row
    numbers, per-row "has data beyond column S" flags, spreadsheet rows of interior blank
    rows). Fully blank rows are dropped from the body; trailing ones (artefacts of some
    spreadsheet tools) are ignored, interior ones are returned so the validator can flag them."""
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True)
    try:
        names = list(wb.sheetnames)
        ws = wb.worksheets[0]
        ws.reset_dimensions()
        it = ws.iter_rows(values_only=True)
        header = tuple(next(it, ()))
        while header and header[-1] is None:
            header = header[:-1]
        rows: list[tuple] = []
        rownums: list[int] = []
        extra: list[bool] = []
        blanks: list[int] = []
        for rn, r in enumerate(it, start=2):
            r = tuple(r)
            has_extra = any(not _is_empty(v) for v in r[N_COLS:])
            r = r[:N_COLS] + (None,) * (N_COLS - len(r))
            if all(_is_empty(v) for v in r) and not has_extra:
                blanks.append(rn)
                continue
            rows.append(r)
            rownums.append(rn)
            extra.append(has_extra)
    finally:
        wb.close()
    last = rownums[-1] if rownums else 1
    return names, header, rows, rownums, extra, [b for b in blanks if b < last]


def _num_array(col) -> np.ndarray:
    """Cells -> float array; anything that is not a real int/float cell (text, None, bool)
    becomes NaN."""
    out = np.full(len(col), np.nan)
    for i, v in enumerate(col):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[i] = v
    return out


def read_submission(it: C.Iteration, path) -> dict[str, np.ndarray]:
    """Track-rows dict of a submission file (empty rows are not returned). Raises
    ValueError for timestamps outside the template grid or non-numeric float cells."""
    iso, _ = C.template_grid(it)
    index = {t: i for i, t in enumerate(iso)}
    _, _, rows, _, _, _ = _load_raw(Path(path))
    rows = [r for r in rows if any(not _is_empty(v) for v in r[2:N_COLS])]
    out = empty_rows()
    if not rows:
        return out
    cols = list(zip(*rows))
    try:
        out["step"] = np.array([index[t] for t in cols[0]], dtype=np.int64)
    except KeyError as e:
        raise ValueError(f"timestamp {e.args[0]!r} is not on the template grid") from None
    for name in FLOAT_FIELDS:
        a = _num_array(cols[COLUMNS.index(name)])
        if not np.all(np.isfinite(a)):
            raise ValueError(f"column {name!r} holds non-numeric or non-finite cells")
        out[name] = a
    for name in STR_FIELDS:
        arr = np.empty(len(rows), dtype=object)
        arr[:] = [v if isinstance(v, str) else str(v) for v in cols[COLUMNS.index(name)]]
        out[name] = arr
    return out


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #


def validate_submission(it: C.Iteration, path, warnings: list[str] | None = None) -> list[str]:
    """Problems of the file (empty list = valid). Never raises for content problems.
    Row numbers in messages are spreadsheet rows (header = 1)."""
    import openpyxl
    path = Path(path)
    probs: list[str] = []
    warns = warnings if warnings is not None else []
    tpl = C.template_path(it)
    if path.name != tpl.name:
        probs.append(f"name: file name {path.name!r} != template name {tpl.name!r}")
    try:
        names, header, rows, rownums, extra, blanks = _load_raw(path)
    except Exception as e:                                # unreadable / not an xlsx
        return probs + [f"unreadable: cannot read {path.name!r} as xlsx ({type(e).__name__}: {e})"]
    twb = openpyxl.load_workbook(tpl, read_only=True)
    try:
        tpl_names = list(twb.sheetnames)
    finally:
        twb.close()
    if names != tpl_names:
        probs.append(f"sheets: sheet names {names} != template {tpl_names}")
    if header != COLUMNS:
        probs.append(f"header: {header!r} != {COLUMNS!r}")
    iso, unix = C.template_grid(it)
    n_steps = len(iso)
    index = {t: i for i, t in enumerate(iso)}
    if blanks:
        probs.append(f"blank-row: {len(blanks)} fully blank rows inside the sheet "
                     f"(first at row {blanks[0]})")
    n = len(rows)
    if n == 0:
        return probs + ["grid-missing: file has no data rows"]
    cols = list(zip(*rows))
    rownum = np.array(rownums)
    if any(extra):
        xi = [i for i, f in enumerate(extra) if f]
        probs.append(f"extra-cell: {len(xi)} rows have data beyond column "
                     f"{openpyxl.utils.get_column_letter(N_COLS)} (first at row {rownums[xi[0]]})")

    def first_rows(idx: np.ndarray, k: int = 3) -> str:
        return ", ".join(str(int(rownum[i])) for i in idx[:k])

    # --- time columns --------------------------------------------------------
    ts = cols[0]
    steps = np.array([index.get(t, -1) if isinstance(t, str) else -1 for t in ts], dtype=np.int64)
    no_ts = np.array([_is_empty(t) for t in ts])
    if no_ts.any():
        probs.append(f"grid-empty-timestamp: {int(no_ts.sum())} rows have an empty timestamp "
                     f"(first at row {first_rows(np.flatnonzero(no_ts))})")
    unk = (steps < 0) & ~no_ts
    if unk.any():
        i = np.flatnonzero(unk)
        probs.append(f"grid-extra: {len({repr(ts[k]) for k in i})} timestamps are "
                     f"not on the template grid, e.g. {ts[i[0]]!r} at row {int(rownum[i[0]])}")
    known = steps >= 0
    present = np.zeros(n_steps, dtype=bool)
    present[steps[known]] = True
    if not present.all():
        miss = np.flatnonzero(~present)
        probs.append(f"grid-missing: {len(miss)} of {n_steps} template timestamps are absent, "
                     f"first {iso[int(miss[0])]!r}")
    ks = steps[known]
    if len(ks):
        run_start = np.ones(len(ks), dtype=bool)
        run_start[1:] = ks[1:] != ks[:-1]
        runs = ks[run_start]
        runs_per = np.bincount(runs, minlength=n_steps)
        if (runs_per > 1).any():
            bad = np.flatnonzero(runs_per > 1)
            probs.append(f"grid-contiguous: rows of {len(bad)} timestamps are not contiguous, "
                         f"first {iso[int(bad[0])]!r}")
        first_seen = runs[np.sort(np.unique(runs, return_index=True)[1])]
        if len(first_seen) > 1 and np.any(np.diff(first_seen) < 0):
            j = int(np.flatnonzero(np.diff(first_seen) < 0)[0])
            probs.append(f"grid-order: timestamps are not in template order "
                         f"({iso[int(first_seen[j])]!r} listed before {iso[int(first_seen[j + 1])]!r})")
    uv = _num_array(cols[1])
    bad_u = known & ~(uv == unix[np.where(known, steps, 0)])
    if bad_u.any():
        i = np.flatnonzero(bad_u)
        probs.append(f"unix: {len(i)} unix_timestamp cells differ from the template "
                     f"(first at row {int(rownum[i[0]])}: {cols[1][i[0]]!r})")

    # --- row completeness ------------------------------------------------------
    filled = np.array([[not _is_empty(v) for v in c] for c in cols[2:N_COLS]])   # (17, n)
    nf = filled.sum(axis=0)
    empty = nf == 0
    complete = nf == len(BODY)
    partial = ~empty & ~complete
    if partial.any():
        i = np.flatnonzero(partial)
        probs.append(f"partial-row: {len(i)} rows are neither empty nor complete "
                     f"(first at row {int(rownum[i[0]])}, {int(nf[i[0]])} of {len(BODY)} cells filled)")
    cnt_at_step = np.bincount(steps[known], minlength=n_steps)
    clash = empty & known & (cnt_at_step[np.where(known, steps, 0)] > 1)
    if clash.any():
        i = np.flatnonzero(clash)
        probs.append(f"empty-row-with-tracks: {len(i)} empty rows share a timestamp with other rows "
                     f"(first at row {int(rownum[i[0]])})")
    big = np.flatnonzero(cnt_at_step > MAX_ROWS_PER_STEP_WARN)
    if len(big):
        warns.append(f"warning: {len(big)} timestamps have more than {MAX_ROWS_PER_STEP_WARN} rows "
                     f"(max {int(cnt_at_step.max())}, first {iso[int(big[0])]!r})")

    # --- content of complete rows -----------------------------------------------
    cx = np.flatnonzero(complete)
    if len(cx) == 0:
        return probs
    sub = {c: [cols[COLUMNS.index(c)][i] for i in cx] for c in BODY}
    rn = rownum[cx]
    num: dict[str, np.ndarray] = {}
    for c in FLOAT_FIELDS:
        a = _num_array(sub[c])
        num[c] = a
        bad = ~np.isfinite(a)
        if bad.any():
            i = np.flatnonzero(bad)
            probs.append(f"numeric: column {c}: {len(i)} cells are not finite numbers "
                         f"(first at row {int(rn[i[0]])}: {sub[c][i[0]]!r})")

    def rng(col: str, lo: float, hi: float) -> None:
        a = num[col]
        bad = np.isfinite(a) & ((a < lo) | (a > hi))
        if bad.any():
            i = np.flatnonzero(bad)
            probs.append(f"{col}: {len(i)} values outside [{lo:g}, {hi:g}] "
                         f"(first at row {int(rn[i[0]])}: {float(a[i[0]])!r})")

    rng("latitude", *LAT_RANGE)
    rng("longitude", *LON_RANGE)
    rng("altitude_m", *ALT_RANGE)
    rng("classification_confidence", 0.0, 1.0)

    cov_names = ("C_nn", "C_ne", "C_nu", "C_ee", "C_eu", "C_uu")
    ok = np.all([np.isfinite(num[c]) for c in cov_names], axis=0)
    neg = ok & ((num["C_nn"] < 0) | (num["C_ee"] < 0) | (num["C_uu"] < 0))
    if neg.any():
        i = np.flatnonzero(neg)
        probs.append(f"covariance-variance: {len(i)} rows have a negative variance "
                     f"(C_nn, C_ee or C_uu; first at row {int(rn[i[0]])})")
    chk = np.flatnonzero(ok & ~neg)
    if len(chk):
        m = np.empty((len(chk), 3, 3))
        nn, ne, nu = num["C_nn"][chk], num["C_ne"][chk], num["C_nu"][chk]
        ee, eu, uu = num["C_ee"][chk], num["C_eu"][chk], num["C_uu"][chk]
        m[:, 0, 0], m[:, 0, 1], m[:, 0, 2] = nn, ne, nu
        m[:, 1, 0], m[:, 1, 1], m[:, 1, 2] = ne, ee, eu
        m[:, 2, 0], m[:, 2, 1], m[:, 2, 2] = nu, eu, uu
        lam = np.linalg.eigvalsh(m)[:, 0]
        bad = lam < -PSD_TOL * np.maximum(1.0, nn + ee + uu)
        if bad.any():
            i = chk[np.flatnonzero(bad)]
            probs.append(f"covariance-psd: {len(i)} NEU covariance matrices are not positive "
                         f"semi-definite (first at row {int(rn[i[0]])}, min eigenvalue "
                         f"{float(lam[bad][0]):.6g})")

    tid = sub["track_id"]
    bad_i = []
    for k, v in enumerate(tid):
        try:
            if not isinstance(v, str):
                raise ValueError
            if str(uuid.UUID(v)) != v:          # only lowercase hyphenated spelling
                raise ValueError
        except ValueError:
            bad_i.append(k)
    if bad_i:
        probs.append(f"track_id-uuid: {len(bad_i)} track_id values are not a canonical lowercase hyphenated UUID "
                     f"(first at row {int(rn[bad_i[0]])}: {tid[bad_i[0]]!r})")
    kn = steps[cx] >= 0
    if kn.any():
        codes = np.unique(np.array([str(v) for v in tid], dtype=object), return_inverse=True)[1]
        key = steps[cx][kn].astype(np.int64) * (int(codes.max()) + 1) + codes[kn]
        _, first, cnt = np.unique(key, return_index=True, return_counts=True)
        if (cnt > 1).any():
            j = first[np.flatnonzero(cnt > 1)[0]]
            probs.append(f"track_id-duplicate: {int((cnt > 1).sum())} (timestamp, track_id) pairs "
                         f"occur more than once (first {tid[np.flatnonzero(kn)[j]]!r} at "
                         f"{iso[int(steps[cx][kn][j])]!r})")

    for c, allowed in C.DROPDOWNS.items():
        ok_set = set(allowed)
        vals = sub[c]
        bad = [k for k, v in enumerate(vals) if not (isinstance(v, str) and v in ok_set)]
        if bad:
            shown: list = []
            for k in bad:
                if vals[k] not in shown:
                    shown.append(vals[k])
                if len(shown) == 3:
                    break
            probs.append(f"{c}: {len(bad)} values not in the dropdown, e.g. "
                         f"{', '.join(repr(v) for v in shown)} (first at row {int(rn[bad[0]])})")
    return probs


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

SELFTEST_DIR = C.DERIVED / "submissions" / "_selftest"
LO_TIMEOUT_S = 300


def _synthetic_rows(tracks: list[dict], spans: list[tuple[int, int]], seed: int = 0) -> dict:
    """Rows for `tracks` (dicts of the string fields) active on the half-open step spans;
    floats are smooth functions of the step plus noise, covariances valid by construction."""
    rng = np.random.default_rng(seed)
    steps = np.concatenate([np.arange(a, b) for a, b in spans]).astype(np.int64)
    tr = np.concatenate([np.full(b - a, k) for k, (a, b) in enumerate(spans)]).astype(np.int64)
    n = len(steps)
    out = empty_rows()
    out["step"] = steps
    out["latitude"] = 45.0 + 1e-5 * steps + 1e-3 * tr + rng.normal(0, 1e-7, n)
    out["longitude"] = 10.0 + 2e-5 * steps + 1e-3 * tr + rng.normal(0, 1e-7, n)
    out["altitude_m"] = 120.0 + 0.01 * steps + 10.0 * tr + rng.normal(0, 0.01, n)
    a = rng.uniform(1.0, 4.0, (n, 3))
    out["C_nn"], out["C_ee"], out["C_uu"] = a[:, 0] ** 2, a[:, 1] ** 2, a[:, 2] ** 2
    out["C_ne"] = 0.3 * a[:, 0] * a[:, 1]
    out["C_nu"] = 0.1 * a[:, 0] * a[:, 2]
    out["C_eu"] = -0.2 * a[:, 1] * a[:, 2]
    out["classification_confidence"] = rng.uniform(0, 1, n)
    for f in STR_FIELDS:
        arr = np.empty(n, dtype=object)
        arr[:] = [tracks[k][f] for k in tr]
        out[f] = arr
    return out


def _tracks3() -> list[dict]:
    base = {"classification": "Rotary Wing UAS", "identity": "Hostile",
            "uav_model/manufacturer": "DJI Mavic 3", "protocol": "DJI Ocusyinc",
            "serial_number": "F45TB21BR1AE03FP", "signal_frequency_band": "2400 – 2483.5"}
    t = [dict(base, track_id=str(uuid.UUID(int=0xA0 + k))) for k in range(3)]
    t[1].update(classification="Fixed Wing UAS", identity="Friendly",
                **{"uav_model/manufacturer": "Parrot Anafi", "protocol": "Wifi",
                   "serial_number": "Unknown", "signal_frequency_band": "5725 – 5850"})
    t[2].update(classification="Unknown", identity="Unknown",
                **{"uav_model/manufacturer": "Not Applicable", "protocol": "Not Applicable",
                   "serial_number": "Not Applicable", "signal_frequency_band": "Not Applicable"})
    return t


def _with_first(rows: dict, **chg) -> dict:
    """Copy of rows with entry 0 of the given fields replaced."""
    out = {k: np.array(v, copy=True) for k, v in rows.items()}
    for k, v in chg.items():
        out[k][0] = v
    return out


def _edit(src: Path, dst: Path, fn) -> Path:
    """Copy `src` to `dst` via openpyxl with `fn(ws)` applied (corrupted variants)."""
    import openpyxl
    wb = openpyxl.load_workbook(src)
    fn(wb.worksheets[0])
    wb.save(dst)
    return dst


def _tags(problems: list[str]) -> set[str]:
    return {p.split(":", 1)[0] for p in problems}


def selftest() -> int:
    import openpyxl
    fails: list[str] = []
    count = [0]

    def check(name: str, cond: bool, detail: str = "") -> None:
        count[0] += 1
        print(f"[{count[0]:2d}] {'ok  ' if cond else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""),
              flush=True)
        if not cond:
            fails.append(name)

    SELFTEST_DIR.mkdir(parents=True, exist_ok=True)
    its = C.iterations()
    by_key = {i.key: i for i in its}

    # (1) empty submissions for all 9 iterations
    for it in its:
        d = SELFTEST_DIR / "empty" / C.iter_slug(it.key)
        out = write_submission(it, empty_rows(), d / C.template_path(it).name)
        probs = validate_submission(it, out)
        wb = openpyxl.load_workbook(out, read_only=True)
        n_rows = sum(1 for _ in wb.worksheets[0].iter_rows(values_only=True))
        wb.close()
        n_grid = len(C.template_grid(it)[0])
        check(f"empty submission {it.key}: valid and {n_grid + 1} rows",
              probs == [] and n_rows == n_grid + 1, f"problems={probs[:2]} rows={n_rows}")
        shutil.rmtree(d)

    # (2) synthetic 3-track round trip on testing/S1-I1
    it = by_key["testing/S1-I1"]
    tpl_iso, tpl_unix = C.template_grid(it)
    n_steps = len(tpl_iso)
    tracks = _tracks3()
    spans = [(100, 160), (130, 190), (140, 142)]      # steps 140,141 carry all three tracks
    rows = _synthetic_rows(tracks, spans)
    good = write_submission(it, rows, SELFTEST_DIR / "roundtrip" / C.template_path(it).name)
    probs = validate_submission(it, good)
    check("synthetic 3-track file validates", probs == [], str(probs[:3]))
    back = read_submission(it, good)
    code = np.unique(rows["track_id"].astype(str), return_inverse=True)[1]
    order = np.lexsort((code, rows["step"]))
    same_s = all(list(back[f]) == list(rows[f][order]) for f in STR_FIELDS)
    same_i = np.array_equal(back["step"], rows["step"][order])
    err = max(float(np.abs(back[f] - rows[f][order]).max()) for f in FLOAT_FIELDS)
    check("round trip: strings and steps exact", same_s and same_i)
    check("round trip: floats within 1e-9", err <= 1e-9, f"max err {err:.3g}")
    check("round trip: en-dash bands preserved",
          {"2400 – 2483.5", "5725 – 5850"} <= set(back["signal_frequency_band"]))
    steps_b, cnts = np.unique(back["step"], return_counts=True)
    check("3 rows at some steps, none at others",
          cnts.max() == 3 and len(steps_b) < n_steps and int((cnts == 3).sum()) == 2,
          f"steps with rows={len(steps_b)} of {n_steps}")
    wb = openpyxl.load_workbook(good, read_only=True)
    first = list(wb.worksheets[0].iter_rows(min_row=2, max_row=3, max_col=2, values_only=True))
    wb.close()
    twb = openpyxl.load_workbook(C.template_path(it), read_only=True)
    tfirst = list(twb.worksheets[0].iter_rows(min_row=2, max_row=3, max_col=2, values_only=True))
    twb.close()
    check("time cells copied verbatim (value and type)",
          first == tfirst and [type(v) for v in first[0]] == [type(v) for v in tfirst[0]],
          f"{[type(v).__name__ for v in first[0]]}")
    check("empty_rows() is zero-length with all keys",
          set(empty_rows()) == set(TRACK_FIELDS) and all(len(v) == 0 for v in empty_rows().values()))

    # (3) corrupted variants, one per rule: the tag set must be exactly the expected one
    bdir = SELFTEST_DIR / "bad"
    bdir.mkdir(parents=True, exist_ok=True)
    name = C.template_path(it).name
    row_of_step: dict[str, int] = {}
    wb = openpyxl.load_workbook(good, read_only=True)
    for r, vals in enumerate(wb.worksheets[0].iter_rows(min_row=2, max_col=1, values_only=True), start=2):
        row_of_step.setdefault(vals[0], r)
    wb.close()
    r10 = row_of_step[tpl_iso[10]]                   # an empty row
    r100 = row_of_step[tpl_iso[100]]                 # the single track row of step 100
    r140 = row_of_step[tpl_iso[140]]                 # first of 3 rows at step 140
    r150 = row_of_step[tpl_iso[150]]                 # first of 2 rows at step 150

    def variant(label: str, expect: set[str], path: Path) -> list[str]:
        p = validate_submission(it, path)
        check(f"corrupt {label}: tags {sorted(expect)}", _tags(p) == expect,
              f"got {[x[:100] for x in p]}")
        return p

    def set_cell(r: int, c: int, v):
        def fn(ws):
            ws.cell(row=r, column=c).value = v      # (value=None would not clear the cell)
        return fn

    def mk(label: str, fn) -> Path:
        (bdir / label).mkdir(exist_ok=True)
        return _edit(good, bdir / label / name, fn)

    def mkrows(label: str, expect: set[str], **chg) -> list[str]:
        return variant(label, expect, write_submission(it, _with_first(rows, **chg), bdir / label / name))

    wrong = bdir / "wrong_name.xlsx"
    shutil.copy(good, wrong)
    variant("file name", {"name"}, wrong)
    variant("header", {"header"}, mk("header", set_cell(1, 3, "lat")))
    variant("unix_timestamp changed", {"unix"}, mk("unix", set_cell(r10, 2, 12345.5)))
    variant("partial row", {"partial-row"}, mk("partial", set_cell(r100, 3, None)))
    variant("missing timestamp", {"grid-missing"}, mk("missing", lambda ws: ws.delete_rows(r10)))
    variant("timestamp off the grid", {"grid-extra", "grid-missing"},
            mk("unknown", set_cell(r10, 1, "2000-01-01T00:00:00.000000")))

    def fn_swap(ws):                                  # two empty steps in swapped order
        for c in range(1, N_COLS + 1):
            a, b = ws.cell(row=r10, column=c), ws.cell(row=r10 + 1, column=c)
            a.value, b.value = b.value, a.value
    variant("timestamp order", {"grid-order"}, mk("order", fn_swap))

    def fn_noncontig(ws):                             # 3rd row of step 140 moved into step 150's block
        r_last = r140 + 2
        vals = [ws.cell(row=r_last, column=c).value for c in range(1, N_COLS + 1)]
        ws.delete_rows(r_last)
        ws.insert_rows(r150)                          # step 150's first row is now at r150 - 1
        for c, v in enumerate(vals, start=1):
            ws.cell(row=r150, column=c, value=v)
    variant("non-contiguous rows", {"grid-contiguous"}, mk("noncontig", fn_noncontig))

    def fn_clash(ws):                                 # an empty row next to a track row
        ws.insert_rows(r100 + 1)
        ws.cell(row=r100 + 1, column=1, value=tpl_iso[100])
        ws.cell(row=r100 + 1, column=2, value=float(tpl_unix[100]))
    variant("empty row beside a track", {"empty-row-with-tracks"}, mk("clash", fn_clash))
    variant("non-numeric cell", {"numeric"}, mk("numeric", set_cell(r100, 6, "abc")))
    mkrows("latitude 91", {"latitude"}, latitude=91.0)
    mkrows("longitude 181", {"longitude"}, longitude=181.0)
    mkrows("altitude 20000", {"altitude_m"}, altitude_m=20000.0)
    mkrows("covariance not PSD", {"covariance-psd"}, C_ne=1e3)
    mkrows("negative variance", {"covariance-variance"}, C_nn=-1.0)
    mkrows("track_id not a UUID", {"track_id-uuid"}, track_id="not-a-uuid")
    tid0 = str(rows["track_id"][np.lexsort((np.unique(rows["track_id"].astype(str), return_inverse=True)[1],
                                           rows["step"]))[0]])
    variant("track_id without hyphens", {"track_id-uuid"},
            mk("tid_nohyphen", set_cell(r100, 12, tid0.replace("-", ""))))
    variant("track_id upper case", {"track_id-uuid"}, mk("tid_upper", set_cell(r100, 12, tid0.upper())))
    variant("data beyond column S", {"extra-cell"}, mk("extra", set_cell(r100, 20, "junk")))

    def fn_blank(ws):                                 # blank row inserted before a track row
        ws.insert_rows(5)
    p = variant("interior blank row", {"blank-row"}, mk("blank", fn_blank))
    pb2 = validate_submission(it, mk("blank_lat", lambda ws: (ws.insert_rows(5), ws.cell(row=r100 + 1, column=3, value=95.0))[0]))
    check("row numbers stay correct after a blank row",
          any(f"row {r100 + 1}:" in x and x.startswith("latitude") for x in pb2), str(pb2))
    try:
        write_submission(it, rows, C.template_path(it))
        check("writer refuses to overwrite the template", False)
    except ValueError:
        check("writer refuses to overwrite the template", True)
    spell = _with_first(rows, track_id=str(uuid.UUID(int=0xA0)).upper().replace("-", ""))
    sp_back = read_submission(it, write_submission(it, spell, bdir / "spell" / name))
    check("writer canonicalises track_id spelling", all(str(uuid.UUID(v)) == v for v in sp_back["track_id"]))
    mkrows("confidence 1.5", {"classification_confidence"}, classification_confidence=1.5)
    mkrows("classification case", {"classification"}, classification="rotary wing UAS")
    mkrows("identity", {"identity"}, identity="Neutral")
    p = mkrows("band with hyphen-minus", {"signal_frequency_band"}, signal_frequency_band="2400 - 2483.5")
    check("bad band message shows repr", any("'2400 - 2483.5'" in x for x in p))
    mkrows("model typo", {"uav_model/manufacturer"}, **{"uav_model/manufacturer": "DJI Mavic3"})
    mkrows("protocol", {"protocol"}, protocol="DJI Ocusync")        # the dropdown keeps 'Ocusyinc'
    mkrows("serial number", {"serial_number"}, serial_number="XYZ")
    dup = {k: np.concatenate([v, v[:1]]) for k, v in rows.items()}
    variant("duplicate track_id at a step", {"track_id-duplicate"},
            write_submission(it, dup, bdir / "dup" / name))
    warns: list[str] = []
    many = _synthetic_rows([dict(tracks[0], track_id=str(uuid.UUID(int=0x1000 + k))) for k in range(51)],
                           [(200, 201)] * 51)
    pr = validate_submission(it, write_submission(it, many, bdir / "many" / name), warns)
    check("51 rows at one timestamp: warning, no error", pr == [] and len(warns) == 1, f"{pr[:1]} {warns}")

    # (4) LibreOffice opens the file
    prof = SELFTEST_DIR / "lo_profile"
    csv_dir = SELFTEST_DIR / "lo_csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Some environments lose the $ORIGIN runpath of soffice.bin: put its own dir on the path.
        prog = Path(shutil.which("soffice")).resolve().parent
        env = dict(os.environ, LD_LIBRARY_PATH=f"{prog}:{os.environ.get('LD_LIBRARY_PATH', '')}")
        res = subprocess.run(["soffice", f"-env:UserInstallation=file://{prof}", "--headless",
                              "--convert-to", "csv", "--outdir", str(csv_dir), str(good)],
                             capture_output=True, text=True, timeout=LO_TIMEOUT_S, env=env)
        with (csv_dir / (good.stem + ".csv")).open(newline="", encoding="utf-8") as fh:
            lo_rows = list(csv.reader(fh))
        expect_rows = 1 + n_steps + len(rows["step"]) - len(np.unique(rows["step"]))
        check("LibreOffice converts the file to CSV with the expected rows",
              res.returncode == 0 and len(lo_rows) == expect_rows and len(lo_rows[0]) == N_COLS,
              f"rc={res.returncode} rows={len(lo_rows)} expected={expect_rows}")
        hit = [r for r in lo_rows if len(r) == N_COLS and r[11] == tracks[1]["track_id"]]
        check("LibreOffice CSV keeps track data (en-dash band, numeric latitude)",
              bool(hit) and hit[0][18] == "5725 – 5850" and 44.0 < float(hit[0][2]) < 46.0,
              f"{hit[0][18] if hit else None}")
    except Exception as e:                                          # noqa: BLE001
        check("LibreOffice converts the file to CSV with the expected rows", False, repr(e))
    shutil.rmtree(prof, ignore_errors=True)

    # (5) timing: 60,000 track rows on the 18,601-step grid
    it_big = by_key["testing/S3-I1"]
    n_big = len(C.template_grid(it_big)[0])
    t4 = [dict(tracks[k % 3], track_id=str(uuid.UUID(int=0xB00 + k))) for k in range(4)]
    big = _synthetic_rows(t4, [(i * 900, i * 900 + 15000) for i in range(4)], seed=1)
    n_tr = len(big["step"])
    t0 = time.perf_counter()
    bp = write_submission(it_big, big, SELFTEST_DIR / "timing" / C.template_path(it_big).name)
    t_w = time.perf_counter() - t0
    t0 = time.perf_counter()
    pb = validate_submission(it_big, bp)
    t_v = time.perf_counter() - t0
    t0 = time.perf_counter()
    rb = read_submission(it_big, bp)
    t_r = time.perf_counter() - t0
    check(f"timing: {n_tr} rows on {n_big} steps written in < 60 s", t_w < 60, f"{t_w:.1f} s")
    check("timing: validated in < 60 s with no problems", t_v < 60 and pb == [], f"{t_v:.1f} s {pb[:1]}")
    check("timing file reads back all rows", len(rb["step"]) == n_tr, f"read {t_r:.1f} s")
    for d in ("timing", "bad", "empty"):
        shutil.rmtree(SELFTEST_DIR / d, ignore_errors=True)

    if fails:
        print(f"\n{len(fails)} SELF-CHECK(S) FAILED: {fails}")
        return 1
    print("\nALL SELF-CHECKS PASSED")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--validate", nargs=2, metavar=("ITERATION_KEY", "FILE"))
    ap.add_argument("--write-empty", nargs=2, metavar=("ITERATION_KEY", "OUT_DIR"))
    args = ap.parse_args()
    print(ANON_REMINDER, flush=True)
    by_key = {i.key: i for i in C.iterations()}
    if args.selftest:
        return selftest()
    if args.validate:
        key, path = args.validate
        warns: list[str] = []
        probs = validate_submission(by_key[key], path, warns)
        for w in warns:
            print(w)
        for p in probs:
            print("PROBLEM", p)
        print("valid" if not probs else f"{len(probs)} problem(s)")
        return 1 if probs else 0
    if args.write_empty:
        key, out_dir = args.write_empty
        it = by_key[key]
        out = write_submission(it, empty_rows(), Path(out_dir) / C.template_path(it).name)
        print("wrote", out)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
