#!/usr/bin/env python3
"""Shared builder: tracks on the 100 ms grid -> the submission writer's track-rows dict.

Every method (the baselines now, the fusion tracker later) turns its tracks into submission
rows through `make_rows`, so the geometry, covariance and label conventions live in ONE place.

Contract
--------
make_rows(it, step, track_key, e, n, u, var_e, var_n, var_u, cov_en, labels, u_fallback=None)
  it          C.Iteration (only used for its key and its local frame).
  step        (R,) int grid index (position in C.template_grid(it)).
  track_key   (R,) str per row. track_id = str(uuid.uuid5(uuid.NAMESPACE_URL,
              f"cuas/{it.key}/{track_key}")), canonical lowercase hyphenated: the same key
              gives the same id in every run, a different key a different id. The pair
              (step, track_key) must be unique (two rows of one track at one step are not
              allowed).
  e, n, u     (R,) metres in C.iteration_frame(it) (the frame the tracks were built in).
              u NaN -> `u_fallback` (per-row array chosen by the caller from sensor data
              only, never from ground truth); a NaN left after the fallback raises.
  var_e, var_n, var_u, cov_en
              (R,) m^2, local East-North-Up. Written as the North-East-Up upper triangle:
              C_nn = var_n, C_ne = cov_en, C_nu = 0, C_ee = var_e, C_eu = 0, C_uu = var_u.
              NaN variance -> 400 m^2 (horizontal: 20 m sd) / 900 m^2 (vertical: 30 m sd);
              NaN cov_en -> 0. A cov_en that would make the horizontal block not positive
              semi-definite raises (it never does for the decoupled sensor_tracks output).
  labels      dict column -> (R,) strings (or a scalar string) for 'classification',
              'identity', 'uav_model/manufacturer', 'protocol', 'serial_number',
              'signal_frequency_band', and a float (R,) / scalar 'classification_confidence'.
              A missing column is 'Unknown' (confidence 0.5). Every string must be an exact
              entry of C.DROPDOWNS.
  returns     dict for submission.write_submission: step, latitude, longitude, altitude_m (from
              frame.to_geodetic(e, n, u), anonymized-longitude frame, ellipsoid height), the six
              covariance cells, classification_confidence, track_id and the label columns.

The module never prints or stores absolute coordinates.

Usage: python3 tools/track_rows.py        (numbered self-checks, exit 1 on a failure)
"""

from __future__ import annotations

import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuas_common as C  # noqa: E402
import submission as S  # noqa: E402

DEFAULT_VAR_H_M2 = 400.0        # 20 m sd, when a track has no variance
DEFAULT_VAR_U_M2 = 900.0        # 30 m sd
DEFAULT_CONFIDENCE = 0.5
LABEL_COLUMNS = ("classification", "identity") + C.METADATA_COLUMNS


def track_id(it: C.Iteration, key: str) -> str:
    """The track UUID of `key` in iteration `it` (canonical lowercase)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"cuas/{it.key}/{key}"))


def _col(v, n: int, name: str, dtype=float) -> np.ndarray:
    a = np.asarray(v, dtype=dtype)
    if a.ndim == 0:
        a = np.full(n, a, dtype=dtype)
    if a.shape != (n,):
        raise ValueError(f"{name} has shape {a.shape}, expected ({n},)")
    return a


def make_rows(it: C.Iteration, step, track_key, e, n, u, var_e, var_n, var_u, cov_en,
              labels: dict, u_fallback=None) -> dict:
    step = np.asarray(step)
    R = len(step)
    if R and step.dtype.kind not in "iu":
        raise ValueError("step must be integer grid indices")
    step = step.astype(np.int64)
    keys = np.empty(R, dtype=object)
    keys[:] = [str(k) for k in track_key] if np.ndim(track_key) else [str(track_key)] * R
    if len(keys) != R:
        raise ValueError(f"track_key has {len(keys)} entries, expected {R}")
    e, n = _col(e, R, "e"), _col(n, R, "n")
    u = _col(u, R, "u").copy()
    if not (np.isfinite(e).all() and np.isfinite(n).all()):
        raise ValueError("e / n contain non-finite values")
    nan_u = ~np.isfinite(u)
    if nan_u.any():
        if u_fallback is None:
            raise ValueError(f"{int(nan_u.sum())} rows have no height and no u_fallback")
        fb = _col(u_fallback, R, "u_fallback")
        u[nan_u] = fb[nan_u]
        if not np.isfinite(u).all():
            raise ValueError(f"u_fallback leaves {int((~np.isfinite(u)).sum())} heights non-finite")

    def var(v, default, name):
        a = _col(v, R, name).copy()
        a[~np.isfinite(a)] = default
        if (a < 0).any():
            raise ValueError(f"{name} has negative variances")
        return a

    ve, vn = var(var_e, DEFAULT_VAR_H_M2, "var_e"), var(var_n, DEFAULT_VAR_H_M2, "var_n")
    vu = var(var_u, DEFAULT_VAR_U_M2, "var_u")
    cen = _col(cov_en, R, "cov_en").copy()
    cen[~np.isfinite(cen)] = 0.0
    if (cen ** 2 > ve * vn * (1 + 1e-9)).any():
        raise ValueError("cov_en makes the horizontal covariance not positive semi-definite")

    # (step, track_key) unique
    if R:
        _, code = np.unique(keys.astype(str), return_inverse=True)
        pair = step * (int(code.max()) + 1) + code
        if len(np.unique(pair)) != R:
            raise ValueError("(step, track_key) is not unique: two rows of one track at one step")
        uk = np.unique(keys.astype(str))
        id_of = {k: track_id(it, k) for k in uk}
        tid = np.empty(R, dtype=object)
        tid[:] = [id_of[k] for k in keys]
        lat, lon, alt = C.iteration_frame(it).to_geodetic(e, n, u)
    else:
        tid = np.empty(0, dtype=object)
        lat = lon = alt = np.zeros(0)

    out = S.empty_rows()
    out.update({"step": step, "latitude": np.asarray(lat, float),
                "longitude": np.asarray(lon, float), "altitude_m": np.asarray(alt, float),
                "C_nn": vn, "C_ne": cen, "C_nu": np.zeros(R), "C_ee": ve, "C_eu": np.zeros(R),
                "C_uu": vu, "track_id": tid})
    unknown = set(labels) - set(LABEL_COLUMNS) - {"classification_confidence"}
    if unknown:
        raise ValueError(f"unknown label columns {sorted(unknown)}")
    for c in LABEL_COLUMNS:
        if c in labels:
            v = np.asarray(labels[c], dtype=object)
            if v.ndim == 0:
                v = np.full(R, str(v), dtype=object)
            if v.shape != (R,):
                raise ValueError(f"labels[{c!r}] has shape {v.shape}, expected ({R},)")
        else:
            v = np.full(R, "Unknown", dtype=object)
        allowed = set(C.DROPDOWNS[c])
        bad = {x for x in set(v.tolist()) if x not in allowed}
        if bad:
            raise ValueError(f"labels[{c!r}] has strings outside C.DROPDOWNS: {sorted(bad)[:3]}")
        out[c] = v
    conf = labels.get("classification_confidence", DEFAULT_CONFIDENCE)
    conf = _col(conf, R, "classification_confidence")
    if R and ((conf < 0).any() or (conf > 1).any() or not np.isfinite(conf).all()):
        raise ValueError("classification_confidence outside [0, 1]")
    out["classification_confidence"] = conf
    return out


# --------------------------------------------------------------------------- self-test


def selftest() -> None:
    it = next(i for i in C.iterations() if i.key == "testing/S1-I1")
    fr = C.iteration_frame(it)
    rng = np.random.default_rng(0)
    R = 200
    step = np.repeat(np.arange(100), 2)
    key = np.where(np.arange(R) % 2 == 0, "a", "b")
    e, n, u = rng.normal(0, 300, R), rng.normal(0, 300, R), rng.uniform(20, 120, R)
    u_in = u.copy()
    u_in[::7] = np.nan
    fb = np.full(R, 77.0)
    ve = np.where(np.arange(R) % 5 == 0, np.nan, 25.0)
    rows = make_rows(it, step, key, e, n, u_in, ve, 16.0, np.nan, 0.0,
                     {"classification": "Fixed Wing UAS"}, u_fallback=fb)
    checks = []
    e2, n2, u2 = fr.to_enu(rows["latitude"], rows["longitude"], rows["altitude_m"])
    u_exp = np.where(np.isnan(u_in), 77.0, u_in)
    err = max(np.abs(e2 - e).max(), np.abs(n2 - n).max(), np.abs(u2 - u_exp).max())
    checks.append(("geodetic round trip within 1e-3 m, NaN u -> fallback", err < 1e-3))
    checks.append(("covariance: NaN -> 400 / 900, C_nn is var_n, C_ee is var_e",
                   np.all(rows["C_uu"] == 900.0) and np.all(rows["C_nn"] == 16.0)
                   and np.all(rows["C_ee"][ve == 25.0] == 25.0)
                   and np.all(rows["C_ee"][np.isnan(ve)] == 400.0)
                   and np.all(rows["C_nu"] == 0) and np.all(rows["C_eu"] == 0)))
    checks.append(("labels: given / missing / confidence",
                   set(rows["classification"]) == {"Fixed Wing UAS"}
                   and set(rows["identity"]) == {"Unknown"}
                   and np.all(rows["classification_confidence"] == 0.5)))
    tid = rows["track_id"]
    checks.append(("track_id: canonical uuid5, stable, one per key",
                   all(str(uuid.UUID(t)) == t for t in tid) and len(set(tid)) == 2
                   and tid[0] == track_id(it, "a")
                   and track_id(it, "a") != track_id(C.iterations()[0], "a")))

    def raises(fn) -> bool:
        try:
            fn()
        except ValueError:
            return True
        return False

    checks.append(("raises: NaN u without fallback, bad dropdown string, duplicate (step, key)",
                   raises(lambda: make_rows(it, step, key, e, n, u_in, 1.0, 1.0, 1.0, 0.0, {}))
                   and raises(lambda: make_rows(it, step, key, e, n, u, 1.0, 1.0, 1.0, 0.0,
                                                {"identity": "Blue"}))
                   and raises(lambda: make_rows(it, np.zeros(R, int), key, e, n, u, 1.0, 1.0,
                                                1.0, 0.0, {}))))
    path = Path(tempfile.mkdtemp(prefix="track_rows_")) / C.template_path(it).name
    S.write_submission(it, rows, path)
    probs = S.validate_submission(it, path)
    back = S.read_submission(it, path)
    checks.append(("written file validates and reads back with the same row count",
                   probs == [] and len(back["step"]) == R))
    empty = make_rows(it, [], [], [], [], [], [], [], [], [], {})
    checks.append(("zero rows are accepted", len(empty["step"]) == 0))
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if probs:
        print("  problems:", probs[:3])
    path.unlink()
    path.parent.rmdir()
    if not all(ok for _, ok in checks):
        raise SystemExit("SELF-CHECK FAILED")
    print("ALL SELF-CHECKS PASSED")


if __name__ == "__main__":
    selftest()
