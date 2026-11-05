#!/usr/bin/env python3
"""End-to-end baselines and ORACLE diagnostics: sensor tracks -> .xlsx -> local score.

This is the score floor and a test of the whole chain (segments on the grid -> track_rows.make_rows
-> submission.write_submission -> validate_submission -> read back -> score). The oracles show
where the score is lost (false tracks vs duplicates vs coverage / localization).

Methods (all built from data/derived/sensor_tracks_v2/, the default output of sensor_tracks.py: every
segment of the per-sensor Kalman / RTS tracks on the 100 ms grid, heights above the WGS84 ellipsoid;
a segment is one sensor's own track cut at gaps > 30 s. The version-1 directory data/derived/
sensor_tracks/ still works through --tracks but its heights are not ellipsoidal, so its 3-D numbers
are not comparable. Caveat: the v2 calibration (sensor_noise.csv sigmas, height offsets, clock
offsets) was fitted on all six testing iterations, so testing scores are in-sample w.r.t. it)
  B0_radar        every segment of the deployable RADAR sensors is a track.
  B1_<ID>         one method per deployable position sensor id (all its segments), incl. VKYY and
                  QTVS; B1_LJJX is reported too but is TESTING-ONLY (LJJX relays the Dronetag feed
                  in testing and does not exist in evaluation). Testing iterations only for LJJX.
  B2_allpos       every segment of every deployable position sensor.
  O1_oracle_exist ORACLE (testing only, uses GT labels, never deployable): B2 restricted to
                  segments labelled drone_air or drone_ground (a perfect existence filter).
  O2_oracle_dedup ORACLE: O1, but at each grid step only one row per best_flight: the segment of
                  the sensor with the smallest sigma_h_m in sensor_noise.csv (blank -> 20 m; ties:
                  smallest seg_id). track_id is per best_flight (one track per drone).
Deployable = C.deployable_sensor_ids() (sensor ids present in evaluation). The headline local
score therefore never uses EUIG, LJJX, MMJJ, NJQL (testing-only). Oracles run on testing only and
must have no evaluation file.

Rows and labels (GT-free for the B methods, identical for the oracles)
* position / covariance: the segment's smoothed grid sample (e, n, u, var_e, var_n, var_u, cov_en)
  via track_rows.make_rows; track_key = '<sensor_key>/seg<seg_id>' (O2: the flight).
* height: the grid u when finite and inside the validator's altitude range S.ALT_RANGE
  (-500 .. 10000 m; the Kalman height of some radar segments is nonsense, up to 13000 m, and
  such a value would make the file invalid, so it counts as missing); otherwise the median of the
  in-range finite u of the same segment, else of the same sensor in the iteration, else of all
  segments in the iteration (these medians use every segment of the iteration, not only the
  selected ones; never ground truth).
* classification 'Fixed Wing UAS' if cls_uav_fixed > 0.5 else 'Rotary Wing UAS' (confidence 0.5);
  identity 'Friendly' when metadata_map.resolve_serial(serial_top) is in the evaluation
  Friendly(Blue)UAS whitelist (data/derived/friendly_whitelist.csv serial_mapped, union over the
  evaluation iterations: the only friendly list provided), else 'Hostile'; uav_model/manufacturer
  = map_value(model_top) or the model of the resolved serial; protocol = map_value(protocol_top);
  serial_number = the resolved dropdown serial; unmappable -> 'Unknown'; band 'Unknown'.

File row cap (DEVIATION from "write every file", see --file-row-cap)
  submission.write_submission builds the whole sheet in memory (about 7.5 KB per row: 300 000
  rows peak at 2.3 GB) and writing + validating + reading back costs about 350 us per row; the
  raw-union methods (B0, B2, B1_EGOI) have up to 2.5 million rows in one iteration (and Excel
  itself stops at 1 048 576 rows). Above the cap (default 250 000 rows) the file is NOT written:
  the rows are built by the same code, put in the writer's row order (step, track_id) and scored in
  memory with score.score_rows (the identical scorer, minus a < 1e-3 m coordinate round trip). So
  that the writer, validator and reader are still exercised on these rows, a contiguous window of
  steps (about SAMPLE_ROWS rows, from the middle of the iteration) of every capped job is written to
  a scratch file, validated, read back and compared (e/n/u, ids, labels; on testing also the score
  of the sample in memory vs from the file), then deleted. The plan table and baselines.md list
  which (method, iteration) pairs were written. --file-row-cap 0 writes everything.

Scoring: testing files with score.score_rows for truth modes 'all' and 'airborne' (the same code
as score.score_method; one read of the file serves both modes), logged with score.log_results to
data/derived/experiments.csv (one block per method x truth mode). Evaluation files are only
written and validated. Output: data/derived/submissions/<method>/..., data/derived/baselines.md.

Usage: python3 tools/baselines.py [--tracks DIR] [--only METHOD ...] [--file-row-cap N]
                                  [--workers N] [--plan] [--no-log]
Numbered self-checks at the end; exit status 1 on any failure.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuas_common as C  # noqa: E402
import metadata_map as MM  # noqa: E402
import score as SC  # noqa: E402
import submission as S  # noqa: E402
import track_rows as TR  # noqa: E402

TRACKS_DIR = C.DERIVED / "sensor_tracks_v2"
NOISE_CSV = C.DERIVED / "sensor_noise.csv"
WHITELIST_CSV = C.DERIVED / "friendly_whitelist.csv"
REPORT = C.DERIVED / "baselines.md"
DEFAULT_SIGMA_H_M = 20.0
DEFAULT_CAP = 250_000
SAMPLE_ROWS = 20_000
SAMPLE_DIR = C.SUBMISSIONS / "_samplecheck"
TRUTH_MODES = ("all", "airborne")
ORACLE_LABELS = ("drone_air", "drone_ground")
RT_TOL_M = 1e-3
NUM_TRACK_COLS = {"seg_id", "cls_uav_fixed", "n_grid"}
TRACK_COLS = ["seg_id", "sensor_key", "sensor_id", "n_grid", "cls_uav_fixed", "model_top",
              "serial_top", "protocol_top", "best_flight", "label"]


# --------------------------------------------------------------------------- iteration data


class IterData:
    """One iteration's segments (tracks.csv) and grid samples (grid.npz), plus the GT-free
    derived per-segment labels and the height fallback."""

    def __init__(self, it: C.Iteration, tracks_dir: Path, whitelist: set[str]):
        self.it = it
        slug = C.iter_slug(it.key)
        t = C.read_columns(tracks_dir / f"{slug}__tracks.csv", numeric=NUM_TRACK_COLS,
                           usecols=TRACK_COLS)
        self.seg_id = t["seg_id"].astype(np.int64)
        if not np.array_equal(self.seg_id, np.arange(len(self.seg_id))):
            raise ValueError(f"{it.key}: seg_id is not 0..N-1 in file order")
        self.sensor_key = t["sensor_key"]
        self.sensor_id = t["sensor_id"]
        self.n_grid = t["n_grid"].astype(np.int64)
        self.label = t["label"]
        self.best_flight = t["best_flight"]
        z = np.load(tracks_dir / f"{slug}__grid.npz")
        self.g = {k: z[k] for k in z.files}
        if len(self.g["seg_id"]) != int(self.n_grid.sum()):
            raise ValueError(f"{it.key}: grid rows != sum of n_grid")
        self._labels(t, whitelist)
        self._fallback()

    def _labels(self, t: dict, whitelist: set[str]) -> None:
        n = len(self.seg_id)
        cls = np.where(t["cls_uav_fixed"] > 0.5, "Fixed Wing UAS", "Rotary Wing UAS")
        serial = np.array([MM.resolve_serial(s) for s in t["serial_top"]], dtype=object)
        self.lab = {
            "classification": cls.astype(object),
            "identity": np.array(["Friendly" if s and s in whitelist else "Hostile"
                                  for s in serial], dtype=object),
            "uav_model/manufacturer": np.array(
                [MM.map_value(MM.MODEL, m) or MM.model_for_serial(s) or "Unknown"
                 for m, s in zip(t["model_top"], t["serial_top"])], dtype=object),
            "protocol": np.array([MM.map_value(MM.PROTOCOL, p) or "Unknown"
                                  for p in t["protocol_top"]], dtype=object),
            "serial_number": np.array([s or "Unknown" for s in serial], dtype=object),
            "signal_frequency_band": np.full(n, "Unknown", dtype=object),
        }

    def _fallback(self) -> None:
        """u used (NaN where missing or outside the validator's range) and the GT-free
        fallback per grid row: segment median, else sensor median, else iteration median."""
        g = self.g
        u = g["u"].astype(float)
        lo, hi = S.ALT_RANGE
        u[~np.isfinite(u) | (u < lo) | (u > hi)] = np.nan
        self.u_use = u
        seg = g["seg_id"]
        nseg = len(self.seg_id)
        bounds = np.searchsorted(seg, np.arange(nseg + 1))        # grid is sorted by seg_id
        med = np.full(nseg, np.nan)
        for i in range(nseg):
            x = u[bounds[i]:bounds[i + 1]]
            x = x[np.isfinite(x)]
            if len(x):
                med[i] = np.median(x)
        sensor_med: dict[str, float] = {}
        row_sensor = self.sensor_id[seg]
        for sid in sorted(set(self.sensor_id.tolist())):
            x = u[row_sensor == sid]
            x = x[np.isfinite(x)]
            sensor_med[sid] = float(np.median(x)) if len(x) else np.nan
        allx = u[np.isfinite(u)]
        all_med = float(np.median(allx)) if len(allx) else np.nan
        seg_sensor_med = np.array([sensor_med[s] for s in self.sensor_id])
        seg_fb = np.where(np.isfinite(med), med,
                          np.where(np.isfinite(seg_sensor_med), seg_sensor_med, all_med))
        self.seg_fallback = seg_fb                                # (nseg,)


# --------------------------------------------------------------------------- methods


def load_whitelist() -> set[str]:
    out = set()
    with WHITELIST_CSV.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["iter_key"].startswith("evaluation") and r["serial_mapped"]:
                out.add(r["serial_mapped"])
    return out


def load_sigma_h() -> dict[str, float]:
    out: dict[str, float] = {}
    with NOISE_CSV.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["sensor_key"], float(r["sigma_h_m"]) if r["sigma_h_m"] else DEFAULT_SIGMA_H_M)
    return out


class Method:
    def __init__(self, name: str, desc: str, kind: str, sensor: str = "", oracle: bool = False,
                 testing_only: bool = False):
        self.name, self.desc, self.kind, self.sensor = name, desc, kind, sensor
        self.oracle, self.testing_only = oracle, testing_only

    def iterations(self, all_its: list[C.Iteration]) -> list[C.Iteration]:
        return [i for i in all_its if i.has_groundtruth or not (self.oracle or self.testing_only)]

    def seg_mask(self, d: IterData, deployable: frozenset[str]) -> np.ndarray:
        dep = np.isin(d.sensor_id, sorted(deployable))
        if self.kind == "radar":
            radar = np.array([k.split("_", 1)[1] == "RADAR" for k in d.sensor_key])
            return dep & radar
        if self.kind == "sensor":
            return d.sensor_id == self.sensor
        if self.kind == "allpos":
            return dep
        if self.kind in ("oracle_exist", "oracle_dedup"):
            return dep & np.isin(d.label, ORACLE_LABELS)
        raise ValueError(self.kind)


def build_methods(tracks_dir: Path) -> list[Method]:
    dep = C.deployable_sensor_ids()
    ids = set()
    for p in sorted(tracks_dir.glob("*__tracks.csv")):
        ids |= set(C.read_columns(p, usecols=["sensor_id"])["sensor_id"].tolist())
    ms = [Method("B0_radar", "every segment of the deployable RADAR sensors", "radar")]
    for sid in sorted(ids & dep):
        ms.append(Method(f"B1_{sid}", f"every segment of sensor {sid}", "sensor", sid))
    if "LJJX" in ids:
        ms.append(Method("B1_LJJX", "every segment of LJJX (TESTING-ONLY: relays the Dronetag feed, "
                         "not deployable)", "sensor", "LJJX", testing_only=True))
    ms.append(Method("B2_allpos", "every segment of every deployable position sensor", "allpos"))
    ms.append(Method("O1_oracle_exist", "ORACLE: B2 restricted to GT-labelled drone segments "
                     "(perfect existence filter)", "oracle_exist", oracle=True))
    ms.append(Method("O2_oracle_dedup", "ORACLE: O1 with one row per (step, best_flight), from the "
                     "lowest-sigma sensor (perfect de-duplication)", "oracle_dedup", oracle=True))
    return ms


def select(d: IterData, m: Method, deployable: frozenset[str], sigma_h: dict[str, float]):
    """(row indices into the grid, track keys per row, independent row count, segment mask)."""
    smask = m.seg_mask(d, deployable)
    idx = np.flatnonzero(smask[d.g["seg_id"]])
    seg = d.g["seg_id"][idx]
    keys = np.array([f"{k}/seg{i}" for k, i in zip(d.sensor_key, d.seg_id)], dtype=object)[seg]
    expected = int(d.n_grid[smask].sum())                         # from tracks.csv, not the grid
    if m.kind == "oracle_dedup":
        step = d.g["step"][idx].astype(np.int64)
        flights = d.best_flight[seg]
        _, fcode = np.unique(flights.astype(str), return_inverse=True)
        sig = np.array([sigma_h.get(k, DEFAULT_SIGMA_H_M) for k in d.sensor_key])[seg]
        order = np.lexsort((seg, sig, fcode, step))               # step, flight, sigma, seg_id
        pair = step[order] * (int(fcode.max()) + 1 if len(fcode) else 1) + fcode[order]
        first = np.ones(len(order), dtype=bool)
        first[1:] = pair[1:] != pair[:-1]
        keep = np.sort(order[first])
        idx, seg, keys = idx[keep], seg[keep], flights[keep]
        # independent count: distinct (step, flight) pairs by python set
        expected = len(set(zip(step.tolist(), flights.tolist())))
    return idx, seg, keys, expected


def build_rows(d: IterData, idx: np.ndarray, seg: np.ndarray, keys: np.ndarray) -> dict:
    g = d.g
    labels = {c: v[seg] for c, v in d.lab.items()}
    labels["classification_confidence"] = 0.5
    return TR.make_rows(
        d.it, g["step"][idx], keys, g["e"][idx].astype(float), g["n"][idx].astype(float),
        d.u_use[idx], g["var_e"][idx], g["var_n"][idx], g["var_u"][idx], g["cov_en"][idx],
        labels, u_fallback=d.seg_fallback[seg])


# --------------------------------------------------------------------------- worker

_CACHE: dict[str, IterData] = {}
_CTX: dict = {}


def _init(tracks_dir: str) -> None:
    _CTX["tracks_dir"] = Path(tracks_dir)
    _CTX["whitelist"] = load_whitelist()
    _CTX["sigma_h"] = load_sigma_h()
    _CTX["deployable"] = C.deployable_sensor_ids()
    _CTX["methods"] = {m.name: m for m in build_methods(Path(tracks_dir))}
    _CTX["iters"] = {i.key: i for i in C.iterations()}


def _iterdata(key: str) -> IterData:
    if key not in _CACHE:
        if len(_CACHE) >= 2:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = IterData(_CTX["iters"][key], _CTX["tracks_dir"], _CTX["whitelist"])
    return _CACHE[key]


def _sub(rows: dict, sel) -> dict:
    """Subset / reorder every column of a rows dict by an index array or boolean mask."""
    return {k: np.asarray(v)[sel] for k, v in rows.items()}


def _file_roundtrip(it: C.Iteration, rows: dict, path: Path, e_exp, n_exp, u_exp) -> dict:
    """write -> validate -> read back -> to_enu, compared with the expected e/n/u (aligned with
    `rows` in its given order) and with the labels / track ids of `rows`."""
    S.write_submission(it, rows, path)
    out = {"problems": S.validate_submission(it, path)}
    back = S.read_submission(it, path)
    R = len(rows["step"])
    out["back"] = back
    out["n_back"] = len(back["step"])
    out["rt_err_m"], out["rt_labels_ok"] = float("nan"), None
    if R == 0:
        out["rt_err_m"], out["rt_labels_ok"] = 0.0, out["n_back"] == 0
    elif out["n_back"] == R:
        tid = np.asarray(rows["track_id"]).astype(str)
        _, tcode = np.unique(tid, return_inverse=True)
        order = np.lexsort((tcode, np.asarray(rows["step"])))         # the writer's row order
        ee, nn, uu = C.iteration_frame(it).to_enu(back["latitude"], back["longitude"],
                                                  back["altitude_m"])
        out["rt_err_m"] = float(max(np.abs(ee - e_exp[order]).max(), np.abs(nn - n_exp[order]).max(),
                                    np.abs(uu - u_exp[order]).max()))
        out["rt_labels_ok"] = bool(
            np.array_equal(np.asarray(back["track_id"]).astype(str), tid[order])
            and all(np.array_equal(np.asarray(back[c]).astype(str),
                                   np.asarray(rows[c]).astype(str)[order])
                    for c in TR.LABEL_COLUMNS))
    return out


def _sample_window(step: np.ndarray, max_rows: int) -> np.ndarray:
    """Boolean mask of a contiguous window of whole steps (about max_rows rows) starting at the
    median distinct step."""
    us, cnt = np.unique(step, return_counts=True)
    i0 = len(us) // 2
    cum = np.cumsum(cnt[i0:])
    last = i0 + max(int(np.searchsorted(cum, max_rows, side="right")), 1)
    return np.isin(step, us[i0:last])


def _coverage(it: C.Iteration, d: IterData, idx: np.ndarray, seg: np.ndarray, u_exp: np.ndarray,
              mode: str) -> dict:
    """How much of the truth the selected (oracle) rows can reach at all: share of truth rows
    with any selected row of the right flight at the same step (temporal coverage), and with one
    within 50 m (3-D and 2-D) of that truth row. Uses best_flight, i.e. ground-truth labels."""
    from truth import load_truth
    tr = load_truth(it, mode)
    fidx = {f["flight_key"]: i for i, f in enumerate(tr.flights)}
    nf = len(tr.flights)
    tkey = tr.step.astype(np.int64) * nf + tr.flight.astype(np.int64)         # sorted (step, flight)
    fl = np.array([fidx.get(str(f), -1) for f in d.best_flight[seg]], dtype=np.int64)
    ok = fl >= 0
    rkey = d.g["step"][idx].astype(np.int64)[ok] * nf + fl[ok]
    pos = np.searchsorted(tkey, rkey)
    pos_c = np.minimum(pos, max(len(tkey) - 1, 0))
    hit = (pos < len(tkey)) & (tkey[pos_c] == rkey) if len(tkey) else np.zeros(len(rkey), bool)
    t = pos_c[hit]
    de = d.g["e"][idx][ok][hit].astype(float) - tr.e[t]
    dn = d.g["n"][idx][ok][hit].astype(float) - tr.n[t]
    du = u_exp[ok][hit] - tr.u[t]
    d2, d3 = np.hypot(de, dn), np.sqrt(de ** 2 + dn ** 2 + du ** 2)
    n_t = max(len(tkey), 1)
    out = {"n_truth": len(tkey)}
    cov = np.zeros(len(tkey), bool)
    cov[t] = True
    out["time"] = float(cov.sum() / n_t)
    for name, dd in (("within50_3d", d3), ("within50_2d", d2)):
        m = np.full(len(tkey), np.inf)
        np.minimum.at(m, t, dd)
        out[name] = float((m <= 50.0).sum() / n_t)
    return out


def run_job(job: tuple[str, str, int]) -> dict:
    mname, key, cap = job
    t0 = time.time()
    m, it = _CTX["methods"][mname], _CTX["iters"][key]
    d = _iterdata(key)
    idx, seg, tkeys, expected = select(d, m, _CTX["deployable"], _CTX["sigma_h"])
    rows = build_rows(d, idx, seg, tkeys)
    n_rows = len(rows["step"])
    out = {"method": mname, "iter_key": key, "n_rows": n_rows, "expected": expected,
           "written": False, "problems": [], "rt_err_m": float("nan"), "rt_labels_ok": None,
           "sensors": sorted(set(d.sensor_id[seg].tolist())), "n_segments": len(set(seg.tolist())),
           "n_friendly": int((np.asarray(rows["identity"]) == "Friendly").sum()),
           "o2_ok": None, "scores": {}, "coverage": {}, "sample": None}
    path = C.submission_path(mname, it)
    # expected u / order for the independent round-trip comparison
    u_exp = np.where(np.isfinite(d.u_use[idx]), d.u_use[idx], d.seg_fallback[seg])
    e_exp, n_exp = d.g["e"][idx].astype(float), d.g["n"][idx].astype(float)
    tid_str = np.asarray(rows["track_id"]).astype(str)
    _, tcode = np.unique(tid_str, return_inverse=True)
    order = np.lexsort((tcode, rows["step"]))                     # the writer's row order
    if cap and n_rows > cap:
        if path.exists():
            path.unlink()
        ee, nn, uu = C.iteration_frame(it).to_enu(rows["latitude"], rows["longitude"],
                                                  rows["altitude_m"])
        out["rt_err_m"] = float(max(np.abs(ee - e_exp).max(), np.abs(nn - n_exp).max(),
                                    np.abs(uu - u_exp).max())) if n_rows else 0.0
        back = _sub(rows, order)                  # scored in the order the file would have
        # exercise writer / validator / reader on a window of the real rows (scratch file)
        win = _sample_window(np.asarray(rows["step"]), SAMPLE_ROWS)
        spath = SAMPLE_DIR / mname / C.iter_slug(key) / path.name
        try:
            rt = _file_roundtrip(it, _sub(rows, win), spath, e_exp[win], n_exp[win], u_exp[win])
        finally:
            if spath.exists():
                spath.unlink()
            for q in (spath.parent, spath.parent.parent):
                try:
                    q.rmdir()
                except OSError:
                    pass
        sm = {"rows": int(win.sum()), "problems": rt["problems"], "rt_err_m": rt["rt_err_m"],
              "labels_ok": rt["rt_labels_ok"], "n_back": rt["n_back"], "score_diff": float("nan")}
        if it.has_groundtruth:
            a = SC.score_rows(it, _sub(rows, win), "all")["parts"]["combined"]
            b = SC.score_rows(it, rt["back"], "all")["parts"]["combined"]
            sm["score_diff"] = abs(a - b)
        out["sample"] = sm
    else:
        rt = _file_roundtrip(it, rows, path, e_exp, n_exp, u_exp)
        back = rt["back"]
        out.update(written=True, problems=rt["problems"], rt_err_m=rt["rt_err_m"],
                   rt_labels_ok=rt["rt_labels_ok"], n_back=rt["n_back"])
        if m.kind == "oracle_dedup":
            flights = {TR.track_id(it, f) for f in set(d.best_flight[seg].tolist())}
            tb = np.asarray(back["track_id"]).astype(str)
            code = np.unique(tb, return_inverse=True)[1] if len(tb) else tb
            pair = back["step"].astype(np.int64) * (int(code.max()) + 1 if len(tb) else 1) + code
            out["o2_ok"] = bool(set(tb.tolist()) <= flights and len(np.unique(pair)) == len(pair))
    if it.has_groundtruth:
        for mode in TRUTH_MODES:
            out["scores"][mode] = SC.score_rows(it, back, mode)
            if m.oracle:
                out["coverage"][mode] = _coverage(it, d, idx, seg, u_exp, mode)
    out["seconds"] = time.time() - t0
    return out


# --------------------------------------------------------------------------- report


def _mean_row(results: list[dict]) -> dict:
    s = SC.summarize(results)
    pr = [r[SC.PRIMARY] for r in results]
    for k in ("gospa_mean", "loc_rms_m", "missed_rate", "false_rate", "cost_share_loc",
              "cost_share_missed", "cost_share_false"):
        v = np.array([p[k] for p in pr], dtype=float)
        s[k] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
    s["track_2d"] = float(np.mean([r["c50_p2_2d"]["track"] for r in results]))
    s["n_tracks"] = float(np.mean([r["n_tracks"] for r in results]))
    s["n_loc_its"] = int(np.isfinite([p["loc_rms_m"] for p in pr]).sum())
    return s


def _f(v: float, nd: int = 3) -> str:
    return "n/a" if v != v else f"{v:.{nd}f}"


def _cov_mean(jobs_info: dict, method: str, mode: str, key: str = "time") -> float:
    v = [j["coverage"][mode][key] for (m, _), j in jobs_info.items()
         if m == method and mode in j.get("coverage", {})]
    return float(np.mean(v)) if v else float("nan")


def write_report(methods: list[Method], res: dict, jobs_info: dict, secs: float,
                 cap: int, path: Path) -> None:
    """res[method][mode] = list of per-iteration score dicts (testing order)."""
    L: list[str] = []
    names = [m.name for m in methods]
    show = [n for n in names if n in res]
    meta = {m.name: m for m in methods}

    def mark(n: str) -> str:
        return n + (" (testing-only)" if meta[n].testing_only else "") + \
            (" (ORACLE)" if meta[n].oracle else "")

    L.append("# Baselines and oracle diagnostics (local score floor)\n")
    L.append(f"Generated by `tools/baselines.py` in {secs / 60:.1f} min. Everything here is a PROXY "
             "of the competition score computed by `tools/score.py` on the 6 testing iterations "
             f"(primary configuration `{SC.PRIMARY}`: GOSPA with cut-off c = 50 m, p = 2, 3-D). "
             "Numbers are means over the testing iterations; the leaderboard scores each data set "
             "separately. The inputs are the version-2 sensor tracks (`data/derived/sensor_tracks_v2/`, "
             "heights above the WGS84 ellipsoid, like the scorer's truth); the version-1 heights were "
             "not ellipsoidal and gave much worse 3-D numbers. Caveat: the version-2 calibration "
             "(per-sensor noise, height offsets, clock offsets) was fitted on all six testing "
             "iterations including the one scored, so the testing numbers are optimistic (in-sample) "
             "for every method that uses it. The headline never uses testing-only sensors (EUIG, LJJX, MMJJ, NJQL); "
             "`B1_LJJX` is shown only to document why: LJJX relays the Dronetag feed in testing "
             "and does not exist in evaluation. VKYY and QTVS (UTM services) are also GT-feed "
             "relays in testing, so their testing numbers are leakage-boosted even though the "
             "sensors exist in evaluation.\n")
    L.append("## Plain-language glossary\n")
    for t, dsc in (
        ("Track / segment", "a sensor's own track: one object id of one sensor, cut where two "
         "reports are more than 30 s apart. Every baseline simply declares segments to be tracks."),
        ("GOSPA", "Generalized Optimal Sub-Pattern Assignment: the official-style track metric. At "
         "each 100 ms step, truth drones are matched to submitted rows; a matched pair costs its "
         "distance squared, an unmatched truth (missed) or unmatched row (false) costs c^2/2 "
         "(c = 50 m). Lower is better; here it is turned into a 0..1 score (1 = perfect)."),
        ("track (score)", "1 minus the GOSPA cost divided by the worst possible cost; weight 0.4."),
        ("classification / identity / metadata", "recall: the share of truth rows whose matched "
         "submitted row carries the right label (Fixed/Rotary wing; Friendly/Hostile; model, "
         "protocol, serial, band). A missed drone counts as wrong. Weights 0.2 / 0.1 / 0.3."),
        ("combined", "0.4 track + 0.2 classification + 0.1 identity + 0.3 metadata."),
        ("identity in testing (caveat)", "the only friendly list provided is the evaluation "
         "whitelist, and no testing drone carries one of its serials, so every testing segment is "
         "labelled Hostile. The testing 'ident.' column is therefore only the share of RED (hostile) "
         "truth rows that were matched; it measures coverage, not identification skill, and a BLUE "
         "drone can never be identified correctly here. It still carries 0.1 of every combined "
         "number."),
        ("localization RMS", "root-mean-square 3-D distance (m) of the pairs the GOSPA assignment "
         "matched (each within 50 m): how accurate the rows that do match a drone are. It is "
         "undefined for an iteration without any matched pair, so the mean over iterations only "
         "uses iterations with matches (column 'loc its' = how many of the 6)."),
        ("missed rate", "share of truth rows that the optimal GOSPA assignment leaves without a "
         "submitted row (no row within 50 m, or the only close row was assigned to another truth "
         "drone). Mostly a coverage problem."),
        ("false rate", "share of submitted rows that match no truth row: false tracks, "
         "duplicates of an already matched drone, and clutter."),
        ("cost shares", "how the GOSPA cost splits into localization (loc), misses and "
         "false rows; they sum to 1."),
        ("truth mode 'all' / 'airborne'", "the template grid equals the Dronetag span exactly, so "
         "the official truth very likely includes drones parked on the ground (about half of the "
         "truth rows); 'airborne' keeps only flying drones. Both are reported."),
        ("oracle", "a method that peeks at ground-truth labels to show an upper bound for one "
         "ability. Never deployable."),
    ):
        L.append(f"* **{t}**: {dsc}")
    L.append("")

    cols = ["combined", "track", "classification", "identity", "metadata", "gospa_mean",
            "loc_rms_m", "missed_rate", "false_rate", "cost_share_loc", "cost_share_missed",
            "cost_share_false"]
    heads = ["combined", "track", "class.", "ident.", "metadata", "mean GOSPA (m)", "loc RMS (m)",
             "missed", "false", "loc share", "missed share", "false share"]
    for mode in TRUTH_MODES:
        L.append(f"## All methods, truth mode '{mode}' (mean over the 6 testing iterations)\n")
        L.append("Every column is the mean over the 6 iterations, except 'loc RMS', which is "
                 "averaged over the iterations that have at least one matched pair ('loc its'). "
                 "'ident.' in testing is RED coverage only (see the glossary).\n")
        L.append("| method | " + " | ".join(heads) + " | loc its | rows/iter |")
        L.append("|---|" + "---|" * (len(heads) + 2))
        for n in show:
            r = _mean_row(res[n][mode])
            rows_it = np.mean([x["n_est_rows"] for x in res[n][mode]])
            L.append(f"| {mark(n)} | " + " | ".join(_f(r[c]) for c in cols) +
                     f" | {r['n_loc_its']}/{len(res[n][mode])} | {rows_it:,.0f} |")
        L.append("")

    for n in ("B0_radar", "B2_allpos", "O1_oracle_exist", "O2_oracle_dedup"):
        if n not in res:
            continue
        for mode in TRUTH_MODES:
            L.append(f"### {mark(n)} per iteration, truth mode '{mode}'\n")
            L.append("| iteration | " + " | ".join(heads[:9]) + " | tracks | rows |")
            L.append("|---|" + "---|" * 11)
            for r in res[n][mode]:
                pa, pr = r["parts"], r[SC.PRIMARY]
                v = [pa["combined"], pa["track"], pa["classification"], pa["identity"],
                     pa["metadata"], pr["gospa_mean"], pr["loc_rms_m"], pr["missed_rate"],
                     pr["false_rate"]]
                L.append(f"| {r['iter_key']} | " + " | ".join(_f(x) for x in v) +
                         f" | {r['n_tracks']} | {r['n_est_rows']:,} |")
            L.append("")

    L.append("## Where the score is lost: the oracle ladder\n")
    L.append("Each step gives the best-case improvement from fixing ONE problem, so it bounds what "
             "the real component can win. B2 (every sensor segment is a track) -> O1 (only real "
             "drone segments: the value of a perfect existence / false-track filter) -> O2 "
             "(additionally one row per drone and step: the value of perfect de-duplication across "
             "sensors) -> 1.0 (what remains: coverage, i.e. drones no sensor saw, plus "
             "localization error and wrong labels). Note that the testing B2 and the oracles "
             "contain VKYY / QTVS, which relay the ground truth in testing (leakage); the O1/O2 "
             "numbers are therefore optimistic and only diagnostic.\n")
    for mode in TRUTH_MODES:
        if not all(n in res for n in ("B2_allpos", "O1_oracle_exist", "O2_oracle_dedup")):
            continue
        L.append(f"Truth mode '{mode}':\n")
        L.append("| step | combined | gain | track | gain | track, 2-D only | missed | false | "
                 "loc RMS (m) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        prev = None
        for lab, n in (("B2 all segments", "B2_allpos"), ("O1 + perfect existence filter",
                                                           "O1_oracle_exist"),
                       ("O2 + perfect de-duplication", "O2_oracle_dedup")):
            r = _mean_row(res[n][mode])
            gc = "" if prev is None else f"{r['combined'] - prev['combined']:+.3f}"
            gt = "" if prev is None else f"{r['track'] - prev['track']:+.3f}"
            L.append(f"| {lab} | {_f(r['combined'])} | {gc} | {_f(r['track'])} | {gt} | "
                     f"{_f(r['track_2d'])} | {_f(r['missed_rate'])} | {_f(r['false_rate'])} | "
                     f"{_f(r['loc_rms_m'], 1)} |")
            prev = r
        o2 = _mean_row(res["O2_oracle_dedup"][mode])
        L.append(f"| remaining to 1.0 | {1 - o2['combined']:.3f} | | {1 - o2['track']:.3f} | | "
                 f"{1 - o2['track_2d']:.3f} | | | |")
        L.append("")
        L.append(f"What remains after O2 is {1 - o2['combined']:.3f} of the combined score "
                 f"(of which track part {0.4 * (1 - o2['track']):.3f}, classification "
                 f"{0.2 * (1 - o2['classification']):.3f}, identity "
                 f"{0.1 * (1 - o2['identity']):.3f}, metadata {0.3 * (1 - o2['metadata']):.3f} "
                 "in combined units). The GOSPA cost of O2 splits into "
                 f"{100 * o2['cost_share_loc']:.0f} % localization, "
                 f"{100 * o2['cost_share_missed']:.0f} % missed drones and "
                 f"{100 * o2['cost_share_false']:.0f} % false rows.\n")
        c1, w1, c2 = (_cov_mean(jobs_info, "O1_oracle_exist", mode),
                      _cov_mean(jobs_info, "O1_oracle_exist", mode, "within50_3d"),
                      _cov_mean(jobs_info, "O2_oracle_dedup", mode, "within50_3d"))
        if all(x == x for x in (c1, w1, c2)):
            L.append("Why truth rows are missed, split by cause (share of all truth rows, mean over "
                     "the testing iterations; ground-truth labels are used to find the right "
                     "flight, so this is a diagnostic ceiling):\n")
            L.append("| cause | share of truth rows |")
            L.append("|---|---|")
            L.append(f"| no drone-labelled segment of that flight exists at that step (true coverage "
                     f"gap: no sensor tracked it) | {1 - c1:.3f} |")
            L.append(f"| a labelled segment exists but none is within 50 m (3-D) of the drone, even "
                     f"the best duplicate (localization / height error) | {c1 - w1:.3f} |")
            L.append(f"| O1's best duplicate is within 50 m but O2's single chosen sensor is not "
                     f"(cost of choosing one sensor by nominal sigma) | {w1 - c2:.3f} |")
            L.append(f"| reachable by O2's chosen row (within 50 m) | {c2:.3f} |")
            L.append("")
    L.append("How to read the ladder. 'track, 2-D only' scores the same rows ignoring height; the gap "
             "to the 3-D track score is what height error and the fallback heights cost (the 50 m "
             "cut-off applies to the 3-D distance). A negative O2 gain is not a bug: O1 keeps all "
             "duplicates, so GOSPA's optimal assignment can pick, for every drone and step, the "
             "duplicate that happens to be closest (a hidden advantage of keeping duplicates), whereas "
             "O2 must commit to the sensor with the lowest NOMINAL sigma (sensor_noise.csv; VKYY, QTVS "
             "and CEQX have no sigma and count as 20 m). A real fusion that merges the duplicates "
             "should do better than O2's fixed choice, so the O1 -> O2 step mixes the value of "
             "de-duplication with the loss from a poor sensor choice (the split above). Drones parked "
             "on the ground are not simply invisible to radars: the sensor tracks label radar "
             "segments on parked drones as drone_ground, and O1/O2 contain them. What is missed is "
             "mostly time when no sensor tracked the drone at all, plus rows farther than 50 m from "
             "it. 'remaining to 1.0' therefore bundles true coverage gaps, localization error and "
             "wrong labels; use the cause table above to separate them.\n")

    L.append("## Top 5 single sensors (B1, deployable sensors only, mean combined)\n")
    singles = [n for n in show if n.startswith("B1_") and not meta[n].testing_only]
    for mode in TRUTH_MODES:
        rk = sorted(singles, key=lambda n: -_mean_row(res[n][mode])["combined"])[:5]
        L.append(f"Truth mode '{mode}':\n")
        L.append("| rank | sensor | combined | track | class. | ident. | metadata | missed | false | "
                 "loc RMS (m) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for i, n in enumerate(rk, 1):
            r = _mean_row(res[n][mode])
            leak = " (GT-feed relay in testing: leakage)" if n[3:] in ("VKYY", "QTVS") else ""
            L.append(f"| {i} | {n[3:]}{leak} | " + " | ".join(
                _f(r[c]) for c in ("combined", "track", "classification", "identity", "metadata",
                                   "missed_rate", "false_rate")) + f" | {_f(r['loc_rms_m'], 1)} |")
        L.append("")

    L.append("## Which files were written\n")
    L.append(f"Files above the row cap ({cap:,} rows; `--file-row-cap`) are not written: "
             "`submission.write_submission` holds the whole sheet in memory and Excel itself caps "
             "a sheet at 1,048,576 rows. Those (method, iteration) pairs are scored in memory "
             "with the same scorer, on the rows in the order the file would have. To still "
             f"exercise the writer, validator and reader on them, a window of about {SAMPLE_ROWS:,} "
             "rows (whole steps from the middle of the iteration) of each such job is written to a "
             "scratch file, validated, read back and compared, then deleted (self-checks 1, 2 and 8). "
             "Rows per (method, iteration); `*` = in memory only.\n")
    its = sorted({k[1] for k in jobs_info})
    L.append("| method | " + " | ".join(i.replace("testing/", "t/").replace("evaluation/", "e/")
                                       for i in its) + " |")
    L.append("|---|" + "---|" * len(its))
    for n in show:
        cells = []
        for i in its:
            j = jobs_info.get((n, i))
            cells.append("-" if j is None else f"{j['n_rows']:,}" + ("" if j["written"] else "*"))
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- main


def run_selfchecks(methods: list[Method], jobs: dict, res: dict, exp_before: int, args) -> int:
    checks: list[tuple[str, bool, str]] = []
    written = {k: j for k, j in jobs.items() if j["written"]}
    bad = {k: j["problems"] for k, j in written.items() if j["problems"]}
    capped = {k: j for k, j in jobs.items() if not j["written"]}
    sbad = {k: j["sample"]["problems"] for k, j in capped.items()
            if j["sample"] is None or j["sample"]["problems"]}
    checks.append((f"every written file validates ([]): {len(written)} files; the {len(capped)} jobs "
                   f"above the cap are scored in memory and a {SAMPLE_ROWS:,}-row window of each also "
                   "validates as a file", not bad and not sbad and bool(written),
                   "; ".join(f"{k}: {v[:2]}" for k, v in list((bad | sbad).items())[:3])))
    rt = [j["rt_err_m"] for j in written.values() if j["n_rows"]]
    srt = [j["sample"]["rt_err_m"] for j in capped.values()]
    lab_bad = [k for k, j in written.items() if j["rt_labels_ok"] is False]
    slab_bad = [k for k, j in capped.items() if j["sample"]["labels_ok"] is not True
                or j["sample"]["n_back"] != j["sample"]["rows"]]
    rt_all = [j["rt_err_m"] for j in jobs.values() if j["n_rows"]]
    checks.append(("make_rows round trip (write -> read_submission -> to_enu) reproduces e/n/u within "
                   f"{RT_TOL_M} m on every row of every written file (max {max(rt, default=0):.2e} m) and "
                   f"of each capped job's sample file (max {max(srt, default=0):.2e} m), track_id and "
                   "labels identical; in-memory rows of capped jobs also within tolerance "
                   f"(max {max(rt_all, default=0):.2e} m)",
                   bool(rt) and all(np.isfinite(x) and x < RT_TOL_M for x in rt + srt + rt_all)
                   and not lab_bad and not slab_bad,
                   f"label mismatch {lab_bad[:3]} {slab_bad[:3]}"))
    cnt_bad = [(k, j["n_rows"], j["expected"], j.get("n_back")) for k, j in jobs.items()
               if j["n_rows"] != j["expected"] or (j["written"] and j.get("n_back") != j["n_rows"])]
    checks.append(("row count == independently counted grid samples (tracks.csv n_grid / python "
                   "set for O2) and == rows read back from the file", not cnt_bad, str(cnt_bad[:3])))
    o2 = [(k, j["o2_ok"]) for k, j in jobs.items() if k[0] == "O2_oracle_dedup"]
    checks.append(("O2 has <= 1 row per (step, best_flight) (file rows, track_id of known flights)",
                   bool(o2) == ("O2_oracle_dedup" in [m.name for m in methods])
                   and all(ok is True for _, ok in o2), str([k for k, ok in o2 if ok is not True][:3])))
    ev_oracle = [str(C.submission_path(m.name, it)) for m in methods if m.oracle
                 for it in C.iterations() if it.tree == "evaluation"
                 if C.submission_path(m.name, it).exists()]
    ev_jobs = [k for k in jobs if meta_is_oracle(methods, k[0]) and k[1].startswith("evaluation")]
    checks.append(("no oracle method has an evaluation file (none written, none on disk)",
                   not ev_oracle and not ev_jobs, str(ev_oracle[:2])))
    dep = C.deployable_sensor_ids()
    leak = [(k, j["sensors"]) for k, j in jobs.items() if not k[0].startswith("B1_LJJX")
            and not set(j["sensors"]) <= dep]
    checks.append(("no testing-only sensor (EUIG, LJJX, MMJJ, NJQL) feeds any headline method",
                   not leak, str(leak[:2])))
    # equivalence with the official file scorer on one small written method
    eq_name = next((m.name for m in methods if m.name in ("O2_oracle_dedup", "O1_oracle_exist")
                    and all(jobs.get((m.name, i.key), {}).get("written") for i in SC.testing_iterations())),
                   None)
    if eq_name:
        with contextlib.redirect_stdout(io.StringIO()):
            ref = SC.score_method(eq_name, "all")
        mine = res[eq_name]["all"]
        diff = max(abs(a["parts"]["combined"] - b["parts"]["combined"]) for a, b in zip(ref, mine))
        checks.append((f"scores from the one-read path equal score.score_method on {eq_name} "
                       f"(max combined diff {diff:.1e})", len(ref) == len(mine) and diff < 1e-12, ""))
    sd = [j["sample"]["score_diff"] for j in capped.values()
          if j["sample"] and j["sample"]["score_diff"] == j["sample"]["score_diff"]]
    if capped and sd:
        checks.append((f"scoring a capped job's sample window in memory equals scoring it from the "
                       f"read-back file ({len(sd)} samples, max combined diff {max(sd):.1e} < 1e-6)",
                       max(sd) < 1e-6, ""))
    if args.no_log:
        print("  [SKIP] experiments.csv block check (--no-log)")
    else:
        try:
            lines = EXPERIMENTS_LINES()[exp_before:]
        except OSError:
            lines = []
        rows = [r for r in csv.reader(lines) if r and r[0] != "run"]
        grp: dict[tuple[str, str], list[str]] = {}
        for r in rows:
            grp.setdefault((r[1], r[5]), []).append(r[4])
        want = {(n, md) for n in res for md in TRUTH_MODES}
        n_its = len(SC.testing_iterations())
        ok = set(grp) == want and all(v.count("testing/MEAN") == 1 and len(v) == n_its + 1
                                      for v in grp.values())
        checks.append((f"experiments.csv gained one block per (method, truth mode): "
                       f"{len(grp)} blocks of {n_its} iterations + MEAN", ok,
                       f"missing {sorted(want - set(grp))[:3]}"))
    n_fail = 0
    for i, (name, ok, detail) in enumerate(checks, 1):
        print(f"  [{'PASS' if ok else 'FAIL'}] ({i}) {name}" + ("" if ok or not detail else f"\n         {detail}"))
        n_fail += not ok
    if n_fail:
        print("SELF-CHECK FAILED")
        return 1
    print("ALL SELF-CHECKS PASSED")
    return 0


def meta_is_oracle(methods: list[Method], name: str) -> bool:
    return any(m.name == name and m.oracle for m in methods)


def EXPERIMENTS_LINES() -> list[str]:
    with SC.EXPERIMENTS.open(encoding="utf-8", newline="") as fh:
        return fh.read().splitlines()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tracks", default=str(TRACKS_DIR), help="sensor_tracks directory")
    ap.add_argument("--only", nargs="+", metavar="METHOD", help="run only these methods")
    ap.add_argument("--file-row-cap", type=int, default=DEFAULT_CAP,
                    help="above this many rows a file is not written, rows are scored in memory "
                         "(0 = always write)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--plan", action="store_true", help="print the plan and exit")
    ap.add_argument("--no-log", action="store_true", help="do not append to experiments.csv")
    args = ap.parse_args()
    t0 = time.time()
    tracks_dir = Path(args.tracks).resolve()
    all_methods = build_methods(tracks_dir)
    methods = [m for m in all_methods if not args.only or m.name in args.only]
    unknown = set(args.only or []) - {m.name for m in all_methods}
    if unknown:
        ap.error(f"unknown methods {sorted(unknown)}; known: {[m.name for m in all_methods]}")
    its = C.iterations()
    # job sizes from tracks.csv only (cheap), biggest first
    wl, deployable = load_whitelist(), C.deployable_sensor_ids()
    sizes: dict[tuple[str, str], int] = {}
    for it in its:
        t = C.read_columns(tracks_dir / f"{C.iter_slug(it.key)}__tracks.csv",
                           numeric={"n_grid"}, usecols=["sensor_key", "sensor_id", "n_grid", "label"])
        sid = t["sensor_id"]
        dep = np.isin(sid, sorted(deployable))
        for m in methods:
            if it not in m.iterations(its):
                continue
            if m.kind == "radar":
                sel = dep & np.array([k.split("_", 1)[1] == "RADAR" for k in t["sensor_key"]])
            elif m.kind == "sensor":
                sel = sid == m.sensor
            elif m.kind == "allpos":
                sel = dep
            else:
                sel = dep & np.isin(t["label"], ORACLE_LABELS)
            sizes[(m.name, it.key)] = int(t["n_grid"][sel].sum())
    jobs = sorted(((m, k, s) for (m, k), s in sizes.items()), key=lambda x: (-x[2], x[0], x[1]))
    print(f"plan: {len(jobs)} (method, iteration) jobs, {len(methods)} methods; rows above "
          f"{args.file_row_cap:,} are scored in memory, not written")
    est = sum(s * 350e-6 for _, _, s in jobs if not (args.file_row_cap and s > args.file_row_cap))
    print(f"  rows to write: {sum(s for _, _, s in jobs if not (args.file_row_cap and s > args.file_row_cap)):,} "
          f"(~{est / 60:.0f} cpu-min at 350 us/row); in memory: "
          f"{sum(s for _, _, s in jobs if args.file_row_cap and s > args.file_row_cap):,}")
    if args.plan:
        for m, k, s in jobs[:40]:
            print(f"  {m:<18}{k:<22}{s:>10,}")
        return 0
    exp_before = len(EXPERIMENTS_LINES()) if SC.EXPERIMENTS.exists() else 0

    ctx = mp.get_context("fork")
    done: dict[tuple[str, str], dict] = {}
    with ctx.Pool(args.workers, initializer=_init, initargs=(str(tracks_dir),)) as pool:
        for n, r in enumerate(pool.imap_unordered(
                run_job, [(m, k, args.file_row_cap) for m, k, _ in jobs]), 1):
            done[(r["method"], r["iter_key"])] = r
            print(f"  [{n}/{len(jobs)}] {r['method']:<16}{r['iter_key']:<22}rows {r['n_rows']:>9,} "
                  f"{'written' if r['written'] else 'memory '} problems {len(r['problems'])} "
                  f"{r['seconds']:.0f}s", flush=True)

    testing = SC.testing_iterations()
    res: dict[str, dict[str, list[dict]]] = {}
    for m in methods:
        if not all((m.name, i.key) in done for i in testing):
            continue
        res[m.name] = {md: [done[(m.name, i.key)]["scores"][md] for i in testing]
                       for md in TRUTH_MODES}
        for md in TRUTH_MODES:
            n_mem = sum(not done[(m.name, i.key)]["written"] for i in testing)
            note = (f"baselines.py: {m.desc}; headline without testing-only sensors; "
                    f"{n_mem} of {len(testing)} iterations scored in memory (rows above file cap)")
            SC.print_table(f"{m.name} [truth {md}]", res[m.name][md])
            if not args.no_log:
                SC.log_results(m.name, note, res[m.name][md], m.name)
    write_report(methods, res, done, time.time() - t0, args.file_row_cap, REPORT)
    print(f"wrote {REPORT.relative_to(C.REPO)}  ({(time.time() - t0) / 60:.1f} min)")
    return run_selfchecks(methods, done, res, exp_before, args)


if __name__ == "__main__":
    sys.exit(main())
