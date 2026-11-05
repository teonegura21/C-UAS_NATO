#!/usr/bin/env python3
"""Scoring ground truth on the template's 100 ms grid (testing iterations only).

One shared definition used by the scorer and every trainer: ``load_truth(it)``.

Source and conventions (all verified by the self-checks below)
* Dronetag files: ``C.groundtruth_files(it)``; one flight per file (``flight_key`` = file
  stem). Samples come from ``data/derived/groundtruth.csv.gz`` (``Q.load_groundtruth``),
  whose e/n/u were built by ``build_detections.py`` as
  ``C.iteration_frame(it).to_enu(lat, lon, geo_alt)``, i.e. ``u`` = height above the WGS84
  ellipsoid from ``geo_altitude`` (what the submission's ``altitude_m`` uses), h0 = 0.
  Self-check (1) recomputes this from the RAW CSVs; self-check (6) compares every
  groundtruth.csv.gz row with the raw files.
* Grid: ``C.template_grid(it)[1]`` (unix seconds, 100 ms).
* Interpolation: each flight is linearly interpolated at the grid times that lie between
  two consecutive samples whose gap is <= ``max_gap_s``. No extrapolation outside the
  flight's first/last sample; a grid time inside a longer gap gets no row for that flight.
  Samples with identical timestamps are averaged first (counted per flight; zero today).
  e/n/u, lat/lon/h, height and velocities are each interpolated linearly (h = geo_alt).
* ``airborne`` = interpolated height above take-off > ``AIRBORNE_HEIGHT_M`` (same rule as
  gt_associate.py). ``mode='airborne'`` keeps only those rows.
* Rows are sorted by (step, flight). Flights are indexed in sorted flight_key order and
  ``Truth.flights`` always lists every flight of the iteration, also in airborne mode.
* ``load_truth`` raises ValueError for an evaluation iteration (no ground truth there).

MAX_GAP_S: the Dronetag logs 1 s samples; the sampling-interval distribution is printed
in the report. Nearly all intervals are 1 s, a few are 2 s, and the rest are longer
outages (3 s up to 750 s) where the aircraft is not tracked at all; interpolating over
them would invent a straight flight. 3 s lets one missed sample through and refuses the
rest.

CLI: ``python3 tools/truth.py`` prints and writes data/derived/truth_summary.md (per-iteration
coverage, per-flight kinematic evidence for tools/drone_catalog.csv, proposed catalog edits)
and runs the numbered self-checks (exit 1 if any fails).
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuas_clean as Q  # noqa: E402
import cuas_common as C  # noqa: E402

MAX_GAP_S = 3.0
AIRBORNE_HEIGHT_M = 5.0
CATALOG = Path(__file__).resolve().parent / "drone_catalog.csv"
OUT = C.DERIVED / "truth_summary.md"
SEED = 0
INTERP_COLS = ("e", "n", "u", "lat", "lon", "geo_alt", "height", "ve", "vn", "vu")
LABEL_COLS = ("classification", "uav_model/manufacturer", "protocol", "serial_number",
              "signal_frequency_band")
IDENTITY = {"BLUE": "Friendly", "RED": "Hostile"}
WATCH = ("Skyranger", "RQ", "Heidrun", "Vector", "Quantum Vector")


@dataclass
class Truth:
    it_key: str
    grid_t: np.ndarray          # (S,) unix seconds of the 100 ms grid
    step: np.ndarray            # (R,) int grid index
    flight: np.ndarray          # (R,) int index into `flights`
    e: np.ndarray               # (R,) ENU metres in C.iteration_frame(it)
    n: np.ndarray
    u: np.ndarray               # height above the WGS84 ellipsoid (geo_altitude)
    lat: np.ndarray
    lon: np.ndarray
    h: np.ndarray               # = geo_alt, interpolated
    height: np.ndarray          # above take-off, interpolated
    speed: np.ndarray           # horizontal m/s, from interpolated ve, vn
    airborne: np.ndarray        # bool
    flights: list[dict]         # flight_key, side, drone_name, identity, + LABEL_COLS
    vu: np.ndarray | None = None  # extra: vertical speed m/s (interpolated)


# --------------------------------------------------------------------------- #
# catalog + samples
# --------------------------------------------------------------------------- #

_CATALOG: dict[str, dict[str, str]] | None = None
_GT: dict[str, np.ndarray] | None = None


def load_catalog() -> dict[str, dict[str, str]]:
    """drone_name -> the side-independent ('*') catalog row. Use catalog_row() for a flight:
    a (name, side) row overrides it (e.g. only the BLUE 'Autel Evo2' carries the whitelist
    serial; the RED ones of the same model must not inherit it)."""
    global _CATALOG
    if _CATALOG is None:
        cols = C.read_columns(CATALOG)
        _CATALOG = {}
        for i in range(len(cols["drone_name"])):
            row = {k: str(v[i]) for k, v in cols.items()}
            _CATALOG[(row["drone_name"], row["side"])] = row
    return {name: row for (name, side), row in _CATALOG.items() if side == "*"}


def catalog_row(drone_name: str, side: str) -> dict[str, str]:
    load_catalog()
    return _CATALOG.get((drone_name, side)) or _CATALOG[(drone_name, "*")]


def _groundtruth() -> dict[str, np.ndarray]:
    global _GT
    if _GT is None:
        _GT = Q.load_groundtruth()
    return _GT


def _flight_samples(it: C.Iteration) -> list[dict]:
    """Per flight (sorted by flight_key): metadata + unique-time sorted samples, duplicates
    averaged. Rows with non-finite time / position are dropped (counted)."""
    gt = _groundtruth()
    sel = gt["iter_key"] == it.key
    catalog = load_catalog()
    out = []
    for gf in sorted(C.groundtruth_files(it), key=lambda g: g.flight_key):
        m = sel & (gt["flight_key"] == gf.flight_key)
        t = gt["t"][m]
        cols = {c: gt[c][m] for c in INTERP_COLS}
        ok = np.isfinite(t) & np.isfinite(cols["e"]) & np.isfinite(cols["n"]) & np.isfinite(cols["u"])
        n_bad = int((~ok).sum())
        t = t[ok]
        cols = {c: v[ok] for c, v in cols.items()}
        uniq, inv = np.unique(t, return_inverse=True)
        n_dup = int(t.size - uniq.size)
        if n_dup:
            cnt = np.bincount(inv, minlength=uniq.size)
            cols = {c: np.bincount(inv, weights=np.nan_to_num(v), minlength=uniq.size) / cnt
                    for c, v in cols.items()}
        else:
            order = np.argsort(t, kind="stable")
            cols = {c: v[order] for c, v in cols.items()}
        t = uniq
        cat = catalog_row(gf.drone_name, gf.side)   # KeyError if the catalog lacks the name (self-check 4)
        meta = dict(flight_key=gf.flight_key, side=gf.side, drone_name=gf.drone_name,
                    identity=IDENTITY[gf.side], **{c: cat[c] for c in LABEL_COLS})
        out.append(dict(meta=meta, t=t, cols=cols, n_dup=n_dup, n_bad=n_bad))
    return out


def _segment_index(tg: np.ndarray, t: np.ndarray, gap_ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For grid times `t` -> (segment k, valid). A segment k spans tg[k]..tg[k+1] (closed) and
    is usable when gap_ok[k]. A time that is exactly a sample can lie in two segments; either
    usable one is taken (both interpolate to the sample)."""
    nseg = tg.size - 1
    kr = np.clip(np.searchsorted(tg, t, "right") - 1, 0, nseg - 1)
    kl = np.clip(np.searchsorted(tg, t, "left") - 1, 0, nseg - 1)
    in_r = (tg[kr] <= t) & (t <= tg[kr + 1]) & gap_ok[kr]
    in_l = (tg[kl] <= t) & (t <= tg[kl + 1]) & gap_ok[kl]
    return np.where(in_r, kr, kl), in_r | in_l


def load_truth(it: C.Iteration, mode: str = "all", max_gap_s: float = MAX_GAP_S) -> Truth:
    if not it.has_groundtruth:
        raise ValueError(f"{it.key} is an evaluation iteration: it has no ground truth")
    if mode not in ("all", "airborne"):
        raise ValueError(f"mode must be 'all' or 'airborne', got {mode!r}")
    grid_t = C.template_grid(it)[1]
    flights = _flight_samples(it)
    parts: dict[str, list[np.ndarray]] = {c: [] for c in INTERP_COLS}
    steps, fl = [], []
    for fi, f in enumerate(flights):
        tg = f["t"]
        if tg.size < 2:
            continue
        gap_ok = np.diff(tg) <= max_gap_s
        a = np.searchsorted(grid_t, tg[0], "left")
        b = np.searchsorted(grid_t, tg[-1], "right")
        idx = np.arange(a, b)
        k, valid = _segment_index(tg, grid_t[idx], gap_ok)
        idx, k = idx[valid], k[valid]
        w = (grid_t[idx] - tg[k]) / (tg[k + 1] - tg[k])
        for c in INTERP_COLS:
            v = f["cols"][c]
            parts[c].append(v[k] + w * (v[k + 1] - v[k]))
        steps.append(idx)
        fl.append(np.full(idx.size, fi, dtype=np.int64))
    if steps:
        step, flight = np.concatenate(steps), np.concatenate(fl)
        cols = {c: np.concatenate(parts[c]) for c in INTERP_COLS}
    else:
        step = flight = np.zeros(0, dtype=np.int64)
        cols = {c: np.zeros(0) for c in INTERP_COLS}
    order = np.lexsort((flight, step))
    step, flight = step[order], flight[order]
    cols = {c: v[order] for c, v in cols.items()}
    airborne = cols["height"] > AIRBORNE_HEIGHT_M
    keep = airborne if mode == "airborne" else np.ones(step.size, dtype=bool)
    return Truth(
        it_key=it.key, grid_t=grid_t, step=step[keep], flight=flight[keep],
        e=cols["e"][keep], n=cols["n"][keep], u=cols["u"][keep],
        lat=cols["lat"][keep], lon=cols["lon"][keep], h=cols["geo_alt"][keep],
        height=cols["height"][keep], speed=np.hypot(cols["ve"], cols["vn"])[keep],
        airborne=airborne[keep], flights=[f["meta"] for f in flights], vu=cols["vu"][keep])


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #

PCTS = (50, 90, 95, 99, 99.9)


def _fmt(x, nd=2):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def testing_iterations() -> list[C.Iteration]:
    return [it for it in C.iterations() if it.has_groundtruth]


def interval_stats(its: list[C.Iteration]) -> list[str]:
    """Sampling intervals (consecutive unique-time samples within a flight) as md lines."""
    per_it = {}
    for it in its:
        dts = [np.diff(f["t"]) for f in _flight_samples(it) if f["t"].size > 1]
        per_it[it.key] = np.concatenate(dts) if dts else np.zeros(0)
    d = np.concatenate(list(per_it.values()))
    lines = ["| scope | n | " + " | ".join(f"p{p:g}" for p in PCTS) + " | max |",
             "|---|---|" + "---|" * (len(PCTS) + 1)]
    for name, x in [*per_it.items(), ("all testing", d)]:
        lines.append(f"| {name} | {x.size} | " + " | ".join(_fmt(float(np.percentile(x, p))) for p in PCTS)
                     + f" | {_fmt(float(x.max()))} |")
    bins = [(0.0, 1.0), (1.0, 2.0), (2.0, MAX_GAP_S), (MAX_GAP_S, 10.0), (10.0, 60.0), (60.0, np.inf)]
    lines += ["", "Interval histogram (all testing):", "",
              "| interval (s) | count | share of intervals | seconds spanned |", "|---|---|---|---|"]
    for lo, hi in bins:
        m = (d > lo) & (d <= hi)
        lbl = f"> {lo:g}" if np.isinf(hi) else f"({lo:g}, {hi:g}]"
        lines.append(f"| {lbl} | {int(m.sum())} | {100 * m.mean():.3f}% | {d[m].sum():.0f} |")
    return lines


def flight_kinematics(tr: Truth) -> list[dict]:
    """Per-flight evidence from the interpolated truth rows (each row = 0.1 s)."""
    out = []
    dt = 0.1
    for fi, meta in enumerate(tr.flights):
        m = tr.flight == fi
        air = m & tr.airborne
        n_all, n_air = int(m.sum()), int(air.sum())
        sp_air = tr.speed[air]
        gmove = m & (tr.height <= AIRBORNE_HEIGHT_M) & (tr.speed > 2.0)
        nan = float("nan")
        out.append(dict(
            iter_key=tr.it_key, flight_key=meta["flight_key"], side=meta["side"],
            drone_name=meta["drone_name"], total_s=n_all * dt, airborne_s=n_air * dt,
            hover_share=float((sp_air < 1.0).mean()) if n_air else nan,
            p10=float(np.percentile(sp_air, 10)) if n_air else nan,
            p50=float(np.percentile(sp_air, 50)) if n_air else nan,
            p90=float(np.percentile(sp_air, 90)) if n_air else nan,
            max_vu=float(np.abs(tr.vu[m]).max()) if n_all else nan,
            max_height=float(tr.height[m].max()) if n_all else nan,
            ground_move_share=float(gmove.sum() / n_all) if n_all else nan,
            ground_move_s=float(gmove.sum() * dt)))
    return out


def _wmean(rs: list[dict], key: str) -> float:
    w = [(r[key], r["airborne_s"]) for r in rs if np.isfinite(r[key]) and r["airborne_s"] > 0]
    den = sum(x for _, x in w)
    return sum(v * x for v, x in w) / den if den else float("nan")


def per_drone(rows: list[dict]) -> list[dict]:
    """Pool the per-flight rows by drone_name (airborne-seconds weighted means of the
    per-flight shares / speed percentiles, so the pooled percentiles are only indicative)."""
    out = []
    for name in sorted({r["drone_name"] for r in rows}):
        rs = [r for r in rows if r["drone_name"] == name]
        tot = sum(r["total_s"] for r in rs)
        out.append(dict(
            drone_name=name, n_flights=len(rs), sides="/".join(sorted({r["side"] for r in rs})),
            total_s=tot, airborne_s=sum(r["airborne_s"] for r in rs),
            hover_share=_wmean(rs, "hover_share"), p10=_wmean(rs, "p10"),
            p50=_wmean(rs, "p50"), p90=_wmean(rs, "p90"),
            max_vu=max(r["max_vu"] for r in rs), max_height=max(r["max_height"] for r in rs),
            ground_move_share=sum(r["ground_move_s"] for r in rs) / tot if tot else float("nan"),
            ground_move_s=sum(r["ground_move_s"] for r in rs)))
    return out


def verdict(d: dict, cat: dict[str, str]) -> str:
    """Rule-based reading of the pooled kinematics against the catalog's classification."""
    cls, basis = cat["classification"], cat["basis"]
    air_s = d["airborne_s"]
    never_flies = d["max_height"] <= AIRBORNE_HEIGHT_M + 2 or air_s < 5
    hovers = d["hover_share"] >= 0.10
    fast_steady = d["p10"] >= 6.0 and d["hover_share"] < 0.02
    speeds = f"airborne speed p10/p50/p90 = {d['p10']:.1f}/{d['p50']:.1f}/{d['p90']:.1f} m/s"
    if never_flies and d["ground_move_s"] >= 30:
        return (f"CONTRADICTED: never leaves the ground (max height {d['max_height']:.1f} m, {air_s:.0f} s airborne) "
                f"yet moves {d['ground_move_s']:.0f} s at > 2 m/s on the ground: a vehicle, not a UAS. "
                f"Catalog says '{cls}' ({basis}).")
    if never_flies:
        return (f"UNSETTLED: never airborne (max height {d['max_height']:.1f} m, {d['ground_move_s']:.0f} s of ground "
                f"movement); kinematics cannot separate a parked drone from a vehicle. Catalog says '{cls}'.")
    if cls == "Rotary Wing UAS":
        if hovers:
            return (f"SUPPORTS '{cls}': hovers {100 * d['hover_share']:.0f} % of airborne time, {speeds}, "
                    f"max |vu| {d['max_vu']:.1f} m/s.")
        if fast_steady and air_s >= 600 and d["p10"] >= 9.0 and d["max_height"] >= 150:
            return (f"CONTRADICTED: '{cls}' but over {air_s:.0f} s airborne it hovers {100 * d['hover_share']:.1f} % of the "
                    f"time, {speeds}, reaching {d['max_height']:.0f} m: the kinematics of a fixed wing / VTOL cruising "
                    f"(propose 'Fixed Wing UAS'). Basis: {basis}.")
        if fast_steady:
            return (f"QUESTIONS '{cls}': hover share {100 * d['hover_share']:.1f} %, {speeds}; consistent with a fast "
                    f"multirotor (FPV) or a fixed wing.")
        return f"NEUTRAL for '{cls}': hover share {100 * d['hover_share']:.1f} %, {speeds}."
    if cls == "Fixed Wing UAS":
        if hovers:
            return (f"CONTRADICTED: '{cls}' but hovers {100 * d['hover_share']:.0f} % of airborne time ({speeds}); "
                    f"a pure fixed wing cannot hover (VTOL phases only). Basis: {basis}.")
        return f"SUPPORTS '{cls}': hover share {100 * d['hover_share']:.1f} %, {speeds}."
    return (f"INFO for '{cls}': airborne {air_s:.0f} s, hover {100 * d['hover_share']:.0f} %, {speeds}, "
            f"max height {d['max_height']:.0f} m.")


def write_report(its: list[C.Iteration]) -> str:
    catalog = load_catalog()
    L = ["# Truth summary (testing iterations)", "",
         f"Generated by `python3 tools/truth.py`. `MAX_GAP_S = {MAX_GAP_S:g}` s, `AIRBORNE_HEIGHT_M = {AIRBORNE_HEIGHT_M:g}` m. "
         "Truth is linearly interpolated per flight at the template's 100 ms grid between consecutive "
         "Dronetag samples whose gap is <= MAX_GAP_S (no extrapolation, no row inside longer gaps). "
         "e/n/u are in `C.iteration_frame(it)`; u = height above the WGS84 ellipsoid from `geo_altitude` "
         "(groundtruth.csv.gz e/n/u were built by `build_detections.py` as exactly "
         "`iteration_frame(it).to_enu(lat, lon, geo_alt)`; self-check 6 re-verifies it on the raw files).", ""]
    L += ["## Coverage per iteration", "",
          "| iteration | grid steps | grid span (s) | Dronetag first - grid first (s) | Dronetag last - grid last (s) | "
          "samples outside grid | rows all | rows airborne | max simult. all | mean simult. all (all steps) | "
          "mean simult. all (steps >= 1) | max simult. airborne | mean simult. airborne (all steps) | "
          "steps with 0 truths (all) | steps with 0 truths (airborne) | duplicate-time samples | dropped non-finite |",
          "|" + "---|" * 17]
    kin_rows: list[dict] = []
    for it in its:
        tr = load_truth(it, "all")
        ta = load_truth(it, "airborne")
        fs = _flight_samples(it)
        g = tr.grid_t
        first = min(f["t"][0] for f in fs)
        last = max(f["t"][-1] for f in fs)
        outside = sum(int(((f["t"] < g[0]) | (f["t"] > g[-1])).sum()) for f in fs)
        c_all = np.bincount(tr.step, minlength=g.size)
        c_air = np.bincount(ta.step, minlength=g.size)
        L.append(f"| {it.key} | {g.size} | {g[-1] - g[0]:.1f} | {first - g[0]:+.1f} | {last - g[-1]:+.1f} | {outside} | "
                 f"{tr.step.size} | {ta.step.size} | {int(c_all.max())} | {c_all.mean():.2f} | "
                 f"{c_all[c_all > 0].mean():.2f} | {int(c_air.max())} | {c_air.mean():.2f} | "
                 f"{int((c_all == 0).sum())} | {int((c_air == 0).sum())} | {sum(f['n_dup'] for f in fs)} | "
                 f"{sum(f['n_bad'] for f in fs)} |")
        kin_rows.extend(flight_kinematics(tr))
    L += ["", "'Dronetag first - grid first' is the earliest sample over all flights of the iteration minus the first "
          "grid time (negative = flights start before the scored window; those samples are never used). "
          "'last - grid last' positive = samples after the grid end.", ""]

    L += ["## Dronetag sampling intervals", "", "Consecutive samples within a flight (seconds).", ""]
    L += interval_stats(its)
    L += ["", f"Justification of MAX_GAP_S = {MAX_GAP_S:g} s: the logger samples at 1 s; an interval of 2 s is a single "
          "missed sample and is bridged. Longer intervals are rare logger outages; interpolating across them "
          "would fabricate a straight-line track, so those steps carry no truth for that flight (a tracker is "
          "neither rewarded nor penalised there).", ""]

    L += ["## Per-flight kinematics (interpolated truth rows, 0.1 s each)", "",
          "airborne = height above take-off > 5 m; hover = airborne and horizontal speed < 1 m/s; "
          "ground-move = height <= 5 m and speed > 2 m/s (share of all of the flight's rows).", "",
          "| iteration | flight | side | total s | airborne s | hover share | v p10 | v p50 | v p90 | max abs vu | max height | ground-move share |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in kin_rows:
        L.append(f"| {r['iter_key']} | {r['flight_key'].replace('DroneTag_', '')} | {r['side']} | {r['total_s']:.0f} | "
                 f"{r['airborne_s']:.0f} | {_fmt(r['hover_share'])} | {_fmt(r['p10'], 1)} | {_fmt(r['p50'], 1)} | "
                 f"{_fmt(r['p90'], 1)} | {_fmt(r['max_vu'], 1)} | {_fmt(r['max_height'], 1)} | {_fmt(r['ground_move_share'])} |")
    pd = per_drone(kin_rows)
    L += ["", "## Per drone_name (pooled over iterations)", "",
          "Hover share and speed percentiles are airborne-seconds weighted means of the per-flight values.", "",
          "| drone_name | catalog classification | sides | flights | airborne s | hover share | v p10 | v p50 | v p90 | max abs vu | max height | ground-move s |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in pd:
        L.append(f"| {d['drone_name']} | {catalog[d['drone_name']]['classification']} | {d['sides']} | {d['n_flights']} | "
                 f"{d['airborne_s']:.0f} | {_fmt(d['hover_share'])} | {_fmt(d['p10'], 1)} | {_fmt(d['p50'], 1)} | "
                 f"{_fmt(d['p90'], 1)} | {_fmt(d['max_vu'], 1)} | {_fmt(d['max_height'], 1)} | {d['ground_move_s']:.0f} |")
    L += ["", "## Catalog check against the kinematics", "",
          "Rule-based reading of the pooled table above (`tools/drone_catalog.csv` is controller-owned and is NOT edited "
          "here; edits are proposals). First the rows the catalog marks UNCERTAIN or that need a check:", ""]
    by = {d["drone_name"]: d for d in pd}
    for name in WATCH:
        if name in by:
            d = by[name]
            L.append(f"- **{name}** ({d['sides']}, {d['n_flights']} flights, {d['airborne_s']:.0f} s airborne, "
                     f"max height {d['max_height']:.0f} m): " + verdict(d, catalog[name]))
    grounded = [r for r in kin_rows if r["airborne_s"] < 5]
    L += ["", "Flights that never get airborne (height <= 5 m throughout; their truth rows exist in mode 'all' only):", ""]
    for r in grounded:
        L.append(f"- {r['iter_key']} {r['side']} {r['drone_name']}: {r['total_s']:.0f} s, max height {r['max_height']:.1f} m, "
                 f"ground movement {r['ground_move_s']:.0f} s")
    sky = [r for r in kin_rows if r["drone_name"] == "Skyranger"]
    L += ["", "Proposed catalog edits (for the controller to apply; nothing here edits the catalog):", ""]
    if sky:
        flying = [r for r in sky if r["airborne_s"] >= 5]
        L.append(f"- Skyranger: keep 'Rotary Wing UAS'. {len(flying)} of {len(sky)} flights are airborne "
                 f"(max {max(r['max_height'] for r in sky):.0f} m, hovers, no sustained ground movement), so it is not a "
                 "ground vehicle; the other flight(s) are a Dronetag sitting on the ground. Drop the UNCERTAIN note.")
    if "Heidrun" in by and verdict(by["Heidrun"], catalog["Heidrun"]).startswith("CONTRADICTED"):
        L.append("- Heidrun: change classification from 'Rotary Wing UAS' to 'Fixed Wing UAS' (never hovers, cruises "
                 "10-20 m/s up to ~300 m, same speed profile as Vector / Quantum Vector).")
    if "RQ" in by:
        L.append("- RQ: leave 'Unknown'; the single BLUE flight never leaves the ground, so no kinematic evidence exists.")
    for nm in ("Vector", "Quantum Vector"):
        if nm in by and verdict(by[nm], catalog[nm]).startswith("SUPPORTS"):
            L.append(f"- {nm}: keep 'Fixed Wing UAS' (kinematics support it).")
    L += ["", "All other drone names (same rules):", ""]
    for d in pd:
        if d["drone_name"] not in WATCH:
            L.append(f"- {d['drone_name']}: " + verdict(d, catalog[d["drone_name"]]))
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# self-checks
# --------------------------------------------------------------------------- #


def _raw_flights(it: C.Iteration) -> dict[str, dict]:
    """RAW Dronetag CSVs (C.read_table), ENU from the iteration frame, unique times averaged
    with plain Python (independent of load_truth and of groundtruth.csv.gz)."""
    fr = C.iteration_frame(it)
    out = {}
    for gf in C.groundtruth_files(it):
        tb = C.read_table(gf.path)

        def col(name):
            return np.array([float(c) if c not in ("", None) else np.nan for c in tb.columns[name]], dtype=float)

        t = C.parse_times(tb.columns["timestamp"])
        lat, lon, ga = col("latitude"), col("longitude"), col("geo_altitude")
        e, nn, u = fr.to_enu(lat, lon, np.nan_to_num(ga))
        acc: dict[float, list[tuple[float, float, float]]] = {}
        for i in range(tb.n_rows):
            if np.isfinite(t[i]) and np.isfinite(ga[i]) and np.isfinite(lat[i]) and np.isfinite(lon[i]):
                acc.setdefault(float(t[i]), []).append((float(e[i]), float(nn[i]), float(u[i])))
        times = sorted(acc)
        pts = {tt: tuple(sum(p[j] for p in acc[tt]) / len(acc[tt]) for j in range(3)) for tt in times}
        out[gf.flight_key] = dict(times=times, pts=pts, t=t, enu=(e, nn, u), ok=np.isfinite(ga))
    return out


def run_checks() -> int:
    its = testing_iterations()
    results: list[tuple[str, bool, str]] = []

    def rec(name, ok, detail):
        results.append((name, bool(ok), detail))

    rng = random.Random(SEED)
    truths = {it.key: load_truth(it, "all") for it in its}
    raws = {it.key: _raw_flights(it) for it in its}

    # (1) brute force vs RAW files
    pool = [(it, i) for it in its for i in range(truths[it.key].step.size)]
    pick = rng.sample(pool, min(2000, len(pool)))
    worst, bad = 0.0, 0
    for it, i in pick:
        tr = truths[it.key]
        rf = raws[it.key][tr.flights[int(tr.flight[i])]["flight_key"]]
        t = float(tr.grid_t[int(tr.step[i])])
        times, pts = rf["times"], rf["pts"]
        got = None
        for a, b in zip(times[:-1], times[1:]):
            if a <= t <= b and b - a <= MAX_GAP_S:
                w = (t - a) / (b - a)
                got = tuple(pts[a][j] + w * (pts[b][j] - pts[a][j]) for j in range(3))
                break
        if got is None:
            bad += 1
            continue
        err = max(abs(got[0] - tr.e[i]), abs(got[1] - tr.n[i]), abs(got[2] - tr.u[i]))
        worst = max(worst, err)
        bad += err > 1e-6
    rec("(1) 2000 random rows == brute force from raw CSVs (<= 1e-6 m in e/n/u)", bad == 0 and len(pick) == 2000,
        f"{len(pick)} rows, {bad} failing, worst {worst:.2e} m")

    # (2) no row outside a span / inside a gap; no usable step missing; no duplicate (step, flight)
    viol = missing = dup = 0
    for it in its:
        tr = truths[it.key]
        g = tr.grid_t
        for fi, meta in enumerate(tr.flights):
            times = np.array(raws[it.key][meta["flight_key"]]["times"])
            steps = tr.step[tr.flight == fi]
            expect = np.zeros(g.size, dtype=bool)
            for a, b in zip(times[:-1], times[1:]):
                if b - a <= MAX_GAP_S:
                    expect |= (g >= a) & (g <= b)
            got = np.zeros(g.size, dtype=bool)
            got[steps] = True
            viol += int((got & ~expect).sum())
            missing += int((expect & ~got).sum())
            dup += int(steps.size - np.unique(steps).size)
    rec("(2) no row outside a flight span or inside a gap > MAX_GAP_S; no usable step missing; no duplicates",
        viol == 0 and missing == 0 and dup == 0, f"outside/gap rows {viol}, missing rows {missing}, duplicates {dup}")

    # (3) labels in DROPDOWNS
    badlab = [(tr.it_key, f["flight_key"], c, f[c]) for tr in truths.values() for f in tr.flights
              for c in LABEL_COLS + ("identity",) if f[c] not in C.DROPDOWNS[c]]
    load_catalog()
    cat_bad = [(n, c, r[c]) for n, r in _CATALOG.items() for c in LABEL_COLS if r[c] not in C.DROPDOWNS[c]]
    cat_bad += [(k, 'side', k[1]) for k in _CATALOG if k[1] not in ('*', 'BLUE', 'RED')]
    rec("(3) every label string of every flight and catalog row is in C.DROPDOWNS", not badlab and not cat_bad,
        f"{len(badlab)} bad flight labels, {len(cat_bad)} bad catalog cells {(badlab + cat_bad)[:3]}")

    # (4) every drone_name has a catalog row
    names = {gf.drone_name for it in its for gf in C.groundtruth_files(it)}
    miss = sorted(n for n in names if n not in load_catalog())
    rec("(4) every Dronetag drone_name has a catalog row", not miss, f"{len(names)} names, missing {miss}")

    # (5) ValueError on an evaluation iteration
    ev = [it for it in C.iterations() if not it.has_groundtruth]
    raised = 0
    for it in ev:
        try:
            load_truth(it)
        except ValueError:
            raised += 1
    rec("(5) load_truth raises ValueError on every evaluation iteration", ev and raised == len(ev),
        f"{raised}/{len(ev)} raised")

    # (6) groundtruth.csv.gz e/n/u == to_enu(raw lat, lon, geo_alt) with the iteration frame
    gt = _groundtruth()
    worst6, n6 = 0.0, 0
    for it in its:
        for fk, rf in raws[it.key].items():
            m = (gt["iter_key"] == it.key) & (gt["flight_key"] == fk)
            if m.sum() != rf["t"].size or not np.array_equal(gt["t"][m], rf["t"]):
                worst6 = float("inf")
                continue
            ok = rf["ok"]
            for gcol, rcol in zip(("e", "n", "u"), rf["enu"]):
                worst6 = max(worst6, float(np.abs(gt[gcol][m][ok] - rcol[ok]).max()))
            n6 += int(m.sum())
    rec("(6) groundtruth.csv.gz e/n/u == C.iteration_frame(it).to_enu(lat, lon, geo_alt) on raw files (< 1e-6 m)",
        worst6 < 1e-6, f"{n6} rows, worst {worst6:.2e} m")

    # (7) structure
    ok7, detail7 = True, "all iterations"
    for it in its:
        tr = truths[it.key]
        ta = load_truth(it, "airborne")
        order_ok = bool(np.all(np.diff(tr.step) >= 0)) and bool(
            np.all((np.diff(tr.step) > 0) | (np.diff(tr.flight) > 0)))
        sub = int(ta.step.size) == int(tr.airborne.sum()) and bool(ta.airborne.all())
        rng_ok = tr.step.size == 0 or (tr.step.min() >= 0 and tr.step.max() < tr.grid_t.size)
        if not (order_ok and sub and rng_ok):
            ok7, detail7 = False, f"{it.key}: order {order_ok} subset {sub} range {rng_ok}"
    rec("(7) rows sorted by (step, flight); airborne mode is the airborne subset; steps inside the grid", ok7, detail7)

    print("\nSELF-CHECKS")
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if all(ok for _, ok, _ in results):
        print("ALL SELF-CHECKS PASSED")
        return 0
    print("SELF-CHECKS FAILED")
    return 1


def main() -> int:
    text = write_report(testing_iterations())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {OUT}")
    return run_checks()


if __name__ == "__main__":
    sys.exit(main())
