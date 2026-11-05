#!/usr/bin/env python3
"""Triangulation oracle (TESTING ONLY): of the truth rows that no position sensor covers, how many
could be recovered by crossing the azimuths of two or more differently placed direction finders?

Question behind it.  About 39 % of airborne ground-truth rows (more in 'all', which includes the
parked drones) have no position-sensor track within 50 m.  The only way to put a track there is to
start it from bearings: two sensors at different places each give a line of sight, the two lines
cross at the drone.  This tool measures, with the ground truth as an ORACLE (we know where the drone
is, so we know which bearing reports point at it), how often that would be possible and how accurate
the crossing would be.  If the answer is "rarely", track initiation from triangulation is not worth
building.  Nothing here is fitted or tuned; every constant below is stated in the task brief.

Definitions
* truth rows: truth.load_truth(it, 'all') (every flight on the template's 100 ms grid) and its
  airborne subset (Truth.airborne).  Testing iterations only (evaluation has no ground truth).
* 'unseen' (main definition): NO grid sample of any sensor_tracks_v2 segment (every segment, drone
  or clutter, of a DEPLOYABLE sensor, C.deployable_sensor_ids(), excluding the ground-truth relays
  VKYY / QTVS) at that grid step lies within UNSEEN_M = 50 m (3-D) of the truth position.
  Side notes: 'main_2d' (same, but horizontal distance only: this is the definition behind the
  "about 39 %" figure in the task brief; the 3-D version is higher because radar heights are 16-35 m off),
  'with_relay' (VKYY / QTVS samples also count; they relay the Dronetag feed in testing,
  so this is the leakage-contaminated variant) and the stricter 'f0' (no row of a REPORTED F0_fuse
  track, fused/<slug>__grid.npz with tracks.csv reported_f0 = 1, within 50 m 3-D).
* bearing reports: cuas_clean.load_clean rows with meas_kind in Q.BEARING_KINDS, not 'duplicate',
  valid sensor position, finite azimuth, sensor id in {WVFX, FGIX (cameras), EMXG (acoustic),
  ORFH, YXAB, OEPS, GTYT (passive RF)} AND deployable (NJQL, the testing-only camera, never counts).
  az_deg is clockwise from north, degrees.
* offsets (value ADDED to the reported azimuth; time offset added to the report time):
    oracle     sensor_calibration.csv rows with status 'applied' (GT-based; param az_offset_deg and
               time_offset_s), the iteration's own row preferred over the '*' row, else 0.
    realistic  bearing_support/registration.csv offset_deg of (iteration, sensor_key) (GT-free
               registration, mostly 0), else 0.
    control    reported azimuth + CONTROL_ROT_DEG (120): a deliberately wrong pointing, i.e. the
               chance level of the whole procedure (a wide gate hits something by luck).
* sites: bearing-report sensor positions (local ENU) merged greedily when within SITE_M = 50 m, so
  two sensors on the same mast are ONE site (they cannot triangulate each other).
* hit: a report within +-WIN_S = 0.5 s of the grid step whose (offset-corrected) azimuth is within
  gate of the azimuth to the truth position, gate = max(2 x az_err_deg, family floor), floors
  camera 3, acoustic 10, rf 15 degrees (blank az_err -> the floor), sensor within RANGE_MAX_M = 3000 m
  (same limit as bearing_support).  A row has k sites 'hitting' when reports of k distinct sites hit.
* azimuth sigma of a report: max(az_err_deg, floor / 2) (the gate is about 2 sigma; the nominal
  az_err of the cameras, 0.01 degrees, is not an honest accuracy and the RF sectors are coarse).
* triangulation geometry for two sites i, j with horizontal ranges r to the drone: crossing angle
  theta = |wrap180(az_i - az_j)| (angle between the two lines of sight at the drone, 0..180), and the
  expected RMS horizontal error of the crossing point is
      sqrt((r_i sigma_i)^2 + (r_j sigma_j)^2) / |sin theta|        (sigma in radians)
  (each line is displaced sideways by r sigma; the crossing point moves by displacement / sin theta
  along the other line).  A row is RECOVERABLE when >= 2 sites hit and the best site pair has expected
  error < 50 m (variant: < 25 m).  rec50_emp is the measured version: the actual intersection of the
  two corrected reported rays lies within 50 m of the truth (the best of the pairs).
* clutter side (every 10th grid step that holds an unseen row, per truth mode): the reports of the
  +-0.5 s window; a report is a GHOST when it hits no truth drone of that mode; a pair of reports of
  different sites is a FALSE intersection when it is not (both hit the same truth drone), the rays
  cross in front of both sensors, within RANGE_MAX_M of both, inside the scenario box (bounding box of
  the iteration's truth e/n + 300 m); 'usable' false intersections also have expected error < 50 m,
  i.e. they would pass the same test as a real pair.

Outputs (data/derived/triangulation/)
  triangulation.md           plain-language report with the verdict
  triangulation_rows.csv     iter_key, truth_mode, offsets, unseen_def, n_truth, n_unseen, share_unseen,
                             n_ge1_site, n_ge2_site, n_rec50, n_rec25, n_rec50_emp, shares of unseen,
                             rec50_of_all (n_rec50 / n_truth); plus iter_key 'POOLED' rows
  triangulation_pairs.csv    iter_key, truth_mode, offsets, pair, n_rec50, median crossing angle (folded
                             to 0..90), median ranges, median expected error
  triangulation_clutter.csv  iter_key, truth_mode, offsets, sensor_key, ghost / real report rates per
                             sampled step, and false-intersection densities (sensor_key 'ALL')
  triangulation_sites.csv    iter_key, site, sensors, n_reports, distance to the nearest other site

Run: python3 tools/triangulation_oracle.py     (about 2 minutes; numbered self-checks, exit 1 on failure)
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuas_clean as Q  # noqa: E402
import cuas_common as C  # noqa: E402
import truth as T  # noqa: E402

OUT = C.DERIVED / "triangulation"
TRACKS = C.DERIVED / "sensor_tracks_v2"
FUSED = C.DERIVED / "fused"
CALIBRATION = C.DERIVED / "sensor_calibration.csv"
REGISTRATION = C.DERIVED / "bearing_support" / "registration.csv"

FAMILY_OF_ID = {"WVFX": "camera", "FGIX": "camera", "NJQL": "camera", "EMXG": "acoustic",
                "ORFH": "rf", "YXAB": "rf", "OEPS": "rf", "GTYT": "rf"}
FAMILY_FLOOR = {"camera": 3.0, "acoustic": 10.0, "rf": 15.0}
RELAY_IDS = frozenset(Q.GT_FEED_SENSORS | {"VKYY", "QTVS"})   # ground-truth relays (LJJX is not deployable anyway)
TESTING_ONLY_IDS = frozenset({"EUIG", "LJJX", "MMJJ", "NJQL"})

UNSEEN_M = 50.0
SITE_M = 50.0
WIN_S = 0.5
RANGE_MAX_M = 3000.0
GATE_K = 2.0
ERR_LIMITS = (50.0, 25.0)
CONTROL_ROT_DEG = 120.0
BOX_PAD_M = 300.0
CLUTTER_STRIDE = 10
SUBSTANTIAL_REPORTS = 100
SCHEMES = ("oracle", "realistic", "control")
MODES = ("airborne", "all")
DEFS = ("main", "main_2d", "f0", "with_relay")
PAIR_CHUNK = 2_000_000
SEED = 0


def log(msg: str) -> None:
    print(msg, flush=True)


def wrap180(x):
    return C.angle_diff_deg(x, 0.0)


# --------------------------------------------------------------------------- #
# geometry (pure functions, closed-form checked by self-check 1)
# --------------------------------------------------------------------------- #

def az_to(de, dn):
    """Azimuth (degrees clockwise from north, (-180, 180]) of the ENU offset (de, dn)."""
    return np.degrees(np.arctan2(de, dn))


def crossing_angle_deg(az1, az2):
    """Angle between two lines of sight at the drone, 0..180 (inputs: azimuths from the sensors)."""
    return np.abs(wrap180(np.asarray(az1, float) - np.asarray(az2, float)))


def expected_error_m(r1, sig1_deg, r2, sig2_deg, theta_deg):
    """Expected RMS horizontal error of the crossing point of two bearing lines (inf if parallel)."""
    s = np.abs(np.sin(np.radians(np.asarray(theta_deg, float))))
    num = np.hypot(np.asarray(r1, float) * np.radians(sig1_deg), np.asarray(r2, float) * np.radians(sig2_deg))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(s > 1e-6, num / s, np.inf)


def intersect_rays(e1, n1, az1, e2, n2, az2):
    """Intersection of two rays (origin, azimuth deg clockwise from north).  Returns (e, n, t1, t2)
    with t the distance along each ray (negative = behind the sensor); NaN where parallel."""
    a1, a2 = np.radians(np.asarray(az1, float)), np.radians(np.asarray(az2, float))
    d1x, d1y, d2x, d2y = np.sin(a1), np.cos(a1), np.sin(a2), np.cos(a2)
    wx, wy = np.asarray(e2, float) - e1, np.asarray(n2, float) - n1
    cross = d1x * d2y - d1y * d2x
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = np.abs(cross) > 1e-9
        t1 = np.where(ok, (wx * d2y - wy * d2x) / np.where(ok, cross, 1.0), np.nan)
        t2 = np.where(ok, (wx * d1y - wy * d1x) / np.where(ok, cross, 1.0), np.nan)
    return np.asarray(e1, float) + t1 * d1x, np.asarray(n1, float) + t1 * d1y, t1, t2


def assign_sites(e: np.ndarray, n: np.ndarray, tol: float = SITE_M) -> np.ndarray:
    """Site id per point: positions (rounded to 1 m) merged greedily, most frequent first, a point
    joining the first existing site centre within `tol` metres, else founding a new site."""
    e, n = np.asarray(e, float), np.asarray(n, float)
    if e.size == 0:
        return np.zeros(0, np.int64)
    key = np.round(np.c_[e, n])
    uniq, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    order = sorted(range(len(uniq)), key=lambda i: (-cnt[i], uniq[i, 0], uniq[i, 1]))
    centres: list[tuple[float, float]] = []
    site_u = np.zeros(len(uniq), np.int64)
    for i in order:
        for s, (ce, cn) in enumerate(centres):
            if math.hypot(uniq[i, 0] - ce, uniq[i, 1] - cn) <= tol:
                site_u[i] = s
                break
        else:
            centres.append((uniq[i, 0], uniq[i, 1]))
            site_u[i] = len(centres) - 1
    return site_u[inv]


def gate_deg(az_err, floor):
    ae = np.asarray(az_err, float)
    return np.maximum(np.where(np.isfinite(ae), GATE_K * ae, 0.0), floor)


def sigma_deg(az_err, floor):
    ae = np.asarray(az_err, float)
    return np.maximum(np.where(np.isfinite(ae), ae, 0.0), np.asarray(floor, float) / 2.0)


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #

def select_reports(det: dict, it_key: str, deployable: frozenset[str]) -> dict | None:
    """Bearing reports of one iteration usable by this tool (see module docstring), sorted by time."""
    ids = det["sensor_id"]
    m = ((det["iter_key"] == it_key) & np.isin(det["meas_kind"], Q.BEARING_KINDS) & ~det["duplicate"]
         & (det["sensor_pos_valid"] == 1) & np.isfinite(det["az_deg"]) & np.isfinite(det["sensor_e"])
         & np.isfinite(det["sensor_n"]))
    m &= np.array([(s in FAMILY_OF_ID) and (s in deployable) for s in ids])
    if not m.any():
        return None
    o = np.argsort(det["t"][m], kind="stable")
    R = {k: det[k][m][o] for k in ("sensor_key", "sensor_id", "t", "az_deg", "az_err_deg", "sensor_e", "sensor_n")}
    R["fam"] = np.array([FAMILY_OF_ID[s] for s in R["sensor_id"]], dtype=object)
    R["floor"] = np.array([FAMILY_FLOOR[f] for f in R["fam"]])
    R["gate"] = gate_deg(R["az_err_deg"], R["floor"])
    R["sigma"] = sigma_deg(R["az_err_deg"], R["floor"])
    R["site"] = assign_sites(R["sensor_e"], R["sensor_n"])
    return R


def calibration_offsets(cal: dict, it_key: str, sensor_keys, param: str) -> np.ndarray:
    """Applied offsets of `param` per report sensor: the iteration's own 'applied' row, else the
    '*' 'applied' row, else 0 (sensor_calibration.csv; GT-based, testing only)."""
    own, star = {}, {}
    for sk, ik, p, st, v in zip(cal["sensor_key"], cal["iter_key"], cal["param"], cal["status"], cal["value"]):
        if p == param and st == "applied" and np.isfinite(v):
            if ik == it_key:
                own[sk] = float(v)
            elif ik == "*":
                star[sk] = float(v)
    return np.array([own.get(s, star.get(s, 0.0)) for s in sensor_keys])


def registration_offsets(reg: dict, it_key: str, sensor_keys) -> np.ndarray:
    d = {sk: float(v) for ik, sk, v in zip(reg["iter_key"], reg["sensor_key"], reg["offset_deg"]) if ik == it_key}
    return np.array([d.get(s, 0.0) for s in sensor_keys])


def schemed(R: dict, scheme: str, cal: dict, reg: dict, it_key: str) -> dict:
    """Reports with corrected time `tc` and azimuth `azc` for the offset scheme, sorted by tc."""
    sk = R["sensor_key"]
    if scheme == "oracle":
        az_off = calibration_offsets(cal, it_key, sk, "az_offset_deg")
        t_off = calibration_offsets(cal, it_key, sk, "time_offset_s")
    elif scheme == "realistic":
        az_off, t_off = registration_offsets(reg, it_key, sk), np.zeros(sk.size)
    elif scheme == "control":
        az_off, t_off = np.full(sk.size, CONTROL_ROT_DEG), np.zeros(sk.size)
    else:
        raise ValueError(scheme)
    tc = R["t"] + t_off
    o = np.argsort(tc, kind="stable")
    S = {k: v[o] for k, v in R.items()}
    S["tc"] = tc[o]
    S["azc"] = wrap180(R["az_deg"] + az_off)[o]
    S["az_off"] = az_off[o]
    S["t_off"] = t_off[o]
    return S


class Pool:
    """Grid samples (step-sorted) of a set of tracks, with a per-sample selector."""

    def __init__(self, step, e, n, u, n_steps: int):
        o = np.argsort(step, kind="stable")
        self.step = np.asarray(step)[o].astype(np.int64)
        self.e, self.n, self.u = (np.asarray(x, float)[o] for x in (e, n, u))
        self.order = o
        self.starts = np.searchsorted(self.step, np.arange(n_steps + 1))


def _expand(lo: np.ndarray, cnt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For rows with a candidate range [lo, lo + cnt): (row index repeated, flat candidate index)."""
    tot = int(cnt.sum())
    r = np.repeat(np.arange(lo.size), cnt)
    off = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    return r, np.repeat(lo, cnt) + off


def _chunks(cnt: np.ndarray, limit: int):
    n = cnt.size
    a = 0
    cs = np.cumsum(cnt)
    while a < n:
        base = cs[a - 1] if a else 0
        b = int(np.searchsorted(cs, base + limit, side="right"))
        b = max(b, a + 1)
        yield a, min(b, n)
        a = min(b, n)


def covered(pool: Pool, masks: dict[str, np.ndarray], step, e, n, u, radius: float = UNSEEN_M,
            chunk: int = PAIR_CHUNK, horizontal: bool = False) -> dict[str, np.ndarray]:
    """Per truth row (step, e, n, u): is there a pool sample at that step, selected by mask name,
    within `radius` metres (3-D, or horizontal only when `horizontal`)?  `masks` are per POOL
    sample (step-sorted) booleans."""
    step = np.asarray(step, np.int64)
    lo = pool.starts[step]
    cnt = pool.starts[step + 1] - lo
    out = {k: np.zeros(step.size, bool) for k in masks}
    for a, b in _chunks(cnt, chunk):
        r, gi = _expand(lo[a:b], cnt[a:b])
        if r.size == 0:
            continue
        d = np.sqrt((pool.e[gi] - e[a:b][r]) ** 2 + (pool.n[gi] - n[a:b][r]) ** 2
                    + (0.0 if horizontal else (pool.u[gi] - u[a:b][r]) ** 2))
        near = d <= radius
        for k, m in masks.items():
            hit = near & m[gi]
            out[k][a:b] |= np.bincount(r[hit], minlength=b - a) > 0
    return out


def load_position_pool(it: C.Iteration, n_steps: int, deployable: frozenset[str]):
    """sensor_tracks_v2 grid samples with the 'main' and 'with_relay' eligibility masks."""
    slug = C.iter_slug(it.key)
    tr = C.read_columns(TRACKS / f"{slug}__tracks.csv", {"seg_id"}, usecols=["seg_id", "sensor_id"])
    seg_id = tr["seg_id"].astype(np.int64)
    if not np.array_equal(seg_id, np.arange(seg_id.size)):
        raise SystemExit(f"{it.key}: seg_id is not 0..S-1")
    z = np.load(TRACKS / f"{slug}__grid.npz")
    pool = Pool(z["step"], z["e"], z["n"], z["u"], n_steps)
    sid = tr["sensor_id"]
    elig_all = np.array([s in deployable for s in sid])
    elig_main = np.array([(s in deployable) and (s not in RELAY_IDS) for s in sid])
    gseg = z["seg_id"].astype(np.int64)[pool.order]
    return pool, elig_main[gseg], elig_all[gseg], sid


def load_f0_pool(it: C.Iteration, n_steps: int):
    slug = C.iter_slug(it.key)
    tr = C.read_columns(FUSED / f"{slug}__tracks.csv", {"track", "reported_f0"}, usecols=["track", "reported_f0"])
    if not np.array_equal(tr["track"].astype(np.int64), np.arange(tr["track"].size)):
        raise SystemExit(f"{it.key}: fused track ids are not 0..K-1")
    z = np.load(FUSED / f"{slug}__grid.npz")
    pool = Pool(z["step"], z["e"], z["n"], z["u"], n_steps)
    rep = tr["reported_f0"] > 0
    return pool, rep[z["track"].astype(np.int64)[pool.order]]


# --------------------------------------------------------------------------- #
# hits
# --------------------------------------------------------------------------- #

def hit_pairs(S: dict, tg: np.ndarray, te: np.ndarray, tn: np.ndarray, chunk: int = PAIR_CHUNK) -> dict:
    """All (row, report) pairs with the report inside the +-WIN_S window of the row's grid time that
    point at the truth position (rows: tg, te, tn).  Returns row, rep (index into S), res (degrees),
    rng (m), azp (azimuth sensor -> truth position)."""
    lo = np.searchsorted(S["tc"], tg - WIN_S, side="left")
    hi = np.searchsorted(S["tc"], tg + WIN_S, side="right")
    cnt = np.maximum(hi - lo, 0)
    acc = {k: [] for k in ("row", "rep", "res", "rng", "azp")}
    for a, b in _chunks(cnt, chunk):
        r, k = _expand(lo[a:b], cnt[a:b])
        if r.size == 0:
            continue
        row = r + a
        de, dn = te[row] - S["sensor_e"][k], tn[row] - S["sensor_n"][k]
        rng = np.hypot(de, dn)
        azp = az_to(de, dn)
        res = wrap180(S["azc"][k] - azp)
        ok = (np.abs(res) <= S["gate"][k]) & (rng <= RANGE_MAX_M) & (rng > 0)
        for name, v in (("row", row), ("rep", k), ("res", res), ("rng", rng), ("azp", azp)):
            acc[name].append(v[ok])
    return {k: (np.concatenate(v) if v else np.zeros(0, np.int64 if k in ("row", "rep") else float))
            for k, v in acc.items()}


def row_stats(H: dict, S: dict, n_rows: int, te: np.ndarray, tn: np.ndarray) -> dict:
    """Per-row aggregation of hit pairs: distinct sites, best site pair (expected error), measured error."""
    n_rep = np.bincount(H["row"], minlength=n_rows)
    out = dict(n_hit_reports=n_rep, n_sites=np.zeros(n_rows, np.int64), best_err=np.full(n_rows, np.inf),
               best_theta_fold=np.full(n_rows, np.nan), best_r1=np.full(n_rows, np.nan),
               best_r2=np.full(n_rows, np.nan), best_pair=np.array([""] * n_rows, dtype=object),
               best_emp=np.full(n_rows, np.inf))
    if H["row"].size == 0:
        return out
    site = S["site"][H["rep"]]
    o = np.lexsort((np.abs(H["res"]), site, H["row"]))
    row, site, rep = H["row"][o], site[o], H["rep"][o]
    rng, azp = H["rng"][o], H["azp"][o]
    first = np.ones(row.size, bool)
    first[1:] = (row[1:] != row[:-1]) | (site[1:] != site[:-1])
    row, rep, rng, azp = row[first], rep[first], rng[first], azp[first]    # one report per (row, site)
    out["n_sites"] = np.bincount(row, minlength=n_rows)
    start = np.searchsorted(row, row)                                       # first entry of each row
    rank = np.arange(row.size) - start
    kmax = int(rank.max()) + 1
    if kmax < 2:
        return out
    pa, pb = [], []
    for a in range(kmax - 1):
        ia = np.flatnonzero(rank == a)
        for b in range(a + 1, kmax):
            ib = np.flatnonzero(rank == b)
            if ib.size == 0:
                continue
            pos = np.searchsorted(row[ib], row[ia])
            pos_c = np.minimum(pos, ib.size - 1)
            m = (pos < ib.size) & (row[ib][pos_c] == row[ia])
            pa.append(ia[m])
            pb.append(ib[pos_c[m]])
    if not pa:
        return out
    pa, pb = np.concatenate(pa), np.concatenate(pb)
    prow = row[pa]
    theta = crossing_angle_deg(azp[pa], azp[pb])
    err = expected_error_m(rng[pa], S["sigma"][rep[pa]], rng[pb], S["sigma"][rep[pb]], theta)
    ie, in_, t1, t2 = intersect_rays(S["sensor_e"][rep[pa]], S["sensor_n"][rep[pa]], S["azc"][rep[pa]],
                                     S["sensor_e"][rep[pb]], S["sensor_n"][rep[pb]], S["azc"][rep[pb]])
    emp = np.where((t1 > 0) & (t2 > 0), np.hypot(ie - te[prow], in_ - tn[prow]), np.inf)
    emp = np.where(np.isfinite(emp), emp, np.inf)
    # best expected-error pair per row
    o2 = np.lexsort((err, prow))
    prow_s = prow[o2]
    f2 = np.ones(prow_s.size, bool)
    f2[1:] = prow_s[1:] != prow_s[:-1]
    best = o2[f2]
    br = prow[best]
    out["best_err"][br] = err[best]
    fold = np.minimum(theta[best], 180.0 - theta[best])
    out["best_theta_fold"][br] = fold
    out["best_r1"][br] = rng[pa][best]
    out["best_r2"][br] = rng[pb][best]
    sid_a, sid_b = S["sensor_id"][rep[pa]][best], S["sensor_id"][rep[pb]][best]
    out["best_pair"][br] = np.array(["+".join(sorted((x, y))) for x, y in zip(sid_a, sid_b)], dtype=object)
    np.minimum.at(out["best_emp"], prow, emp)
    return out


# --------------------------------------------------------------------------- #
# clutter side
# --------------------------------------------------------------------------- #

def clutter_step_stats(S: dict, grid_t: np.ndarray, steps: np.ndarray, tstep: np.ndarray, te: np.ndarray,
                       tn: np.ndarray, box: tuple[float, float, float, float]) -> dict:
    """Ghost reports and false pairwise intersections on the sampled grid `steps`.
    (tstep, te, tn): the truth rows of the mode (sorted by step)."""
    t_lo = np.searchsorted(tstep, steps, side="left")
    t_hi = np.searchsorted(tstep, steps, side="right")
    skeys = sorted(set(S["sensor_key"].tolist()))
    n_steps = int(steps.size)
    rep_cnt = {k: 0 for k in skeys}
    ghost_cnt = {k: 0 for k in skeys}
    tot = dict(n_pairs=0, n_true=0, n_false=0, n_false_usable50=0, n_false_usable25=0, n_true_usable50=0, n_win=0)
    for s, a, b in zip(steps, t_lo, t_hi):
        tg = grid_t[s]
        i0 = int(np.searchsorted(S["tc"], tg - WIN_S, side="left"))
        i1 = int(np.searchsorted(S["tc"], tg + WIN_S, side="right"))
        if i1 <= i0:
            continue
        k = np.arange(i0, i1)
        tot["n_win"] += k.size
        de = te[a:b][None, :] - S["sensor_e"][k][:, None]
        dn = tn[a:b][None, :] - S["sensor_n"][k][:, None]
        rng = np.hypot(de, dn)
        res = wrap180(S["azc"][k][:, None] - az_to(de, dn))
        hit = (np.abs(res) <= S["gate"][k][:, None]) & (rng <= RANGE_MAX_M) & (rng > 0)
        ghost = ~hit.any(axis=1) if hit.shape[1] else np.ones(k.size, bool)
        for sk, g in zip(S["sensor_key"][k], ghost):
            rep_cnt[sk] += 1
            ghost_cnt[sk] += int(g)
        if k.size < 2:
            continue
        ia, ib = np.triu_indices(k.size, 1)
        diff = S["site"][k][ia] != S["site"][k][ib]
        ia, ib = ia[diff], ib[diff]
        if ia.size == 0:
            continue
        ka, kb = k[ia], k[ib]
        pe, pn, t1, t2 = intersect_rays(S["sensor_e"][ka], S["sensor_n"][ka], S["azc"][ka],
                                        S["sensor_e"][kb], S["sensor_n"][kb], S["azc"][kb])
        ok = ((t1 > 0) & (t2 > 0) & (t1 <= RANGE_MAX_M) & (t2 <= RANGE_MAX_M)
              & (pe >= box[0]) & (pe <= box[1]) & (pn >= box[2]) & (pn <= box[3]))
        if not ok.any():
            continue
        ia, ib, ka, kb, t1, t2 = ia[ok], ib[ok], ka[ok], kb[ok], t1[ok], t2[ok]
        true_pair = (hit[ia] & hit[ib]).any(axis=1) if hit.shape[1] else np.zeros(ia.size, bool)
        theta = crossing_angle_deg(S["azc"][ka], S["azc"][kb])
        err = expected_error_m(t1, S["sigma"][ka], t2, S["sigma"][kb], theta)
        tot["n_pairs"] += ia.size
        tot["n_true"] += int(true_pair.sum())
        tot["n_false"] += int((~true_pair).sum())
        tot["n_false_usable50"] += int(((~true_pair) & (err < ERR_LIMITS[0])).sum())
        tot["n_false_usable25"] += int(((~true_pair) & (err < ERR_LIMITS[1])).sum())
        tot["n_true_usable50"] += int((true_pair & (err < ERR_LIMITS[0])).sum())
    return dict(n_steps=n_steps, rep_cnt=rep_cnt, ghost_cnt=ghost_cnt, **tot)


# --------------------------------------------------------------------------- #
# per-iteration driver
# --------------------------------------------------------------------------- #

def process_iteration(it: C.Iteration, det: dict, cal: dict, reg: dict, deployable: frozenset[str]) -> dict:
    t0 = time.time()
    tr = T.load_truth(it, "all")
    n_steps = tr.grid_t.size
    pool, m_main, m_all, seg_sensor = load_position_pool(it, n_steps, deployable)
    fpool, f_rep = load_f0_pool(it, n_steps)
    cov = covered(pool, {"main": m_main, "with_relay": m_all}, tr.step, tr.e, tr.n, tr.u)
    cov_f0 = covered(fpool, {"f0": f_rep}, tr.step, tr.e, tr.n, tr.u)
    cov_2d = covered(pool, {"main_2d": m_main}, tr.step, tr.e, tr.n, tr.u, horizontal=True)
    seen = {"main": cov["main"], "with_relay": cov["with_relay"], "f0": cov_f0["f0"], "main_2d": cov_2d["main_2d"]}
    R = select_reports(det, it.key, deployable)
    res = dict(it=it, truth=tr, seen=seen, R=R, schemes={}, clutter={})
    unseen_any = ~seen["main"] | ~seen["f0"]
    need = np.flatnonzero(unseen_any)
    box = (float(tr.e.min()) - BOX_PAD_M, float(tr.e.max()) + BOX_PAD_M,
           float(tr.n.min()) - BOX_PAD_M, float(tr.n.max()) + BOX_PAD_M)
    res["box"] = box
    if R is not None:
        for scheme in SCHEMES:
            S = schemed(R, scheme, cal, reg, it.key)
            H = hit_pairs(S, tr.grid_t[tr.step[need]], tr.e[need], tr.n[need])
            rs = row_stats(H, S, need.size, tr.e[need], tr.n[need])
            full = {}
            for k, v in rs.items():
                if k == "best_pair":
                    arr = np.array([""] * tr.step.size, dtype=object)
                elif k in ("best_err", "best_emp"):
                    arr = np.full(tr.step.size, np.inf)
                elif k.startswith("best_"):
                    arr = np.full(tr.step.size, np.nan)
                else:
                    arr = np.zeros(tr.step.size, np.int64)
                arr[need] = v
                full[k] = arr
            res["schemes"][scheme] = full
            # clutter on every CLUTTER_STRIDE-th step holding an unseen (main) row, per truth mode
            for mode in MODES:
                sel = ~seen["main"] & (tr.airborne if mode == "airborne" else np.ones(tr.step.size, bool))
                steps = np.unique(tr.step[sel])
                steps = steps[steps % CLUTTER_STRIDE == 0]
                tm = tr.airborne if mode == "airborne" else np.ones(tr.step.size, bool)
                res["clutter"][(scheme, mode)] = clutter_step_stats(
                    S, tr.grid_t, steps, tr.step[tm], tr.e[tm], tr.n[tm], box)
    log(f"  {it.key}: {tr.step.size} truth rows, {int((~seen['main']).sum())} unseen (main), "
        f"{len(R['t']) if R is not None else 0} bearing reports, {time.time() - t0:.1f} s")
    return res


def summarize(res: dict) -> list[dict]:
    tr = res["truth"]
    rows = []
    for mode in MODES:
        mm = tr.airborne if mode == "airborne" else np.ones(tr.step.size, bool)
        for scheme in SCHEMES:
            st = res["schemes"].get(scheme)
            for d in DEFS:
                un = mm & ~res["seen"][d]
                n_un = int(un.sum())
                rec = dict(iter_key=res["it"].key, truth_mode=mode, offsets=scheme, unseen_def=d,
                           n_truth=int(mm.sum()), n_unseen=n_un)
                if st is None:
                    rec.update(n_ge1_site=0, n_ge2_site=0, n_rec50=0, n_rec25=0, n_rec50_emp=0)
                else:
                    rec.update(n_ge1_site=int((un & (st["n_sites"] >= 1)).sum()),
                               n_ge2_site=int((un & (st["n_sites"] >= 2)).sum()),
                               n_rec50=int((un & (st["best_err"] < ERR_LIMITS[0])).sum()),
                               n_rec25=int((un & (st["best_err"] < ERR_LIMITS[1])).sum()),
                               n_rec50_emp=int((un & (st["best_emp"] < ERR_LIMITS[0])).sum()))
                rows.append(rec)
    return rows


def add_shares(rec: dict) -> dict:
    n, u = rec["n_truth"], rec["n_unseen"]
    rec["share_unseen"] = u / n if n else float("nan")
    for k in ("ge1_site", "ge2_site", "rec50", "rec25", "rec50_emp"):
        rec[f"share_{k}"] = rec[f"n_{k}"] / u if u else float("nan")
    rec["rec50_of_all"] = rec["n_rec50"] / n if n else float("nan")
    return rec


def pool_records(rows: list[dict]) -> list[dict]:
    out = []
    for mode in MODES:
        for scheme in SCHEMES:
            for d in DEFS:
                sel = [r for r in rows if r["truth_mode"] == mode and r["offsets"] == scheme and r["unseen_def"] == d]
                p = dict(iter_key="POOLED", truth_mode=mode, offsets=scheme, unseen_def=d)
                for k in ("n_truth", "n_unseen", "n_ge1_site", "n_ge2_site", "n_rec50", "n_rec25", "n_rec50_emp"):
                    p[k] = sum(r[k] for r in sel)
                out.append(add_shares(p))
    return out


REC_COLS = ["iter_key", "truth_mode", "offsets", "unseen_def", "n_truth", "n_unseen", "share_unseen", "n_ge1_site",
            "n_ge2_site", "n_rec50", "n_rec25", "n_rec50_emp", "share_ge1_site", "share_ge2_site", "share_rec50",
            "share_rec25", "share_rec50_emp", "rec50_of_all"]


def pair_records(results: list[dict]) -> list[dict]:
    out = []
    for mode in MODES:
        for scheme in SCHEMES:
            per = {}
            for res in results:
                st = res["schemes"].get(scheme)
                if st is None:
                    continue
                tr = res["truth"]
                mm = tr.airborne if mode == "airborne" else np.ones(tr.step.size, bool)
                sel = mm & ~res["seen"]["main"] & (st["best_err"] < ERR_LIMITS[0])
                for i in np.flatnonzero(sel):
                    per.setdefault((res["it"].key, st["best_pair"][i]), []).append(
                        (st["best_theta_fold"][i], st["best_r1"][i], st["best_r2"][i], st["best_err"][i]))
            for scope in ["POOLED"] + sorted({k[0] for k in per}):
                agg: dict[str, list] = {}
                for (ik, pr), v in per.items():
                    if scope in ("POOLED", ik):
                        agg.setdefault(pr, []).extend(v)
                for pr, v in sorted(agg.items(), key=lambda kv: -len(kv[1])):
                    a = np.array(v)
                    out.append(dict(iter_key=scope, truth_mode=mode, offsets=scheme, pair=pr, n_rec50=len(v),
                                    med_theta_fold_deg=float(np.median(a[:, 0])), med_r1_m=float(np.median(a[:, 1])),
                                    med_r2_m=float(np.median(a[:, 2])), med_expected_err_m=float(np.median(a[:, 3]))))
    return out


def clutter_records(results: list[dict]) -> list[dict]:
    out = []
    for res in results:
        for (scheme, mode), c in res["clutter"].items():
            ns = max(c["n_steps"], 1)
            for sk in sorted(c["rep_cnt"]):
                out.append(dict(iter_key=res["it"].key, truth_mode=mode, offsets=scheme, sensor_key=sk,
                                n_steps=c["n_steps"], reports_per_step=c["rep_cnt"][sk] / ns,
                                ghost_per_step=c["ghost_cnt"][sk] / ns,
                                ghost_share=(c["ghost_cnt"][sk] / c["rep_cnt"][sk]) if c["rep_cnt"][sk] else float("nan"),
                                intersections_per_step=float("nan"), false_per_step=float("nan"),
                                false_usable50_per_step=float("nan"), false_usable25_per_step=float("nan"),
                                true_per_step=float("nan"), true_usable50_per_step=float("nan")))
            out.append(dict(iter_key=res["it"].key, truth_mode=mode, offsets=scheme, sensor_key="ALL",
                            n_steps=c["n_steps"], reports_per_step=c["n_win"] / ns,
                            ghost_per_step=sum(c["ghost_cnt"].values()) / ns,
                            ghost_share=(sum(c["ghost_cnt"].values()) / max(sum(c["rep_cnt"].values()), 1)),
                            intersections_per_step=c["n_pairs"] / ns, false_per_step=c["n_false"] / ns,
                            false_usable50_per_step=c["n_false_usable50"] / ns,
                            false_usable25_per_step=c["n_false_usable25"] / ns,
                            true_per_step=c["n_true"] / ns, true_usable50_per_step=c["n_true_usable50"] / ns))
    return out


def nearmiss_records(results: list[dict]) -> list[dict]:
    out = []
    for mode in MODES:
        for scheme in SCHEMES:
            errs, thetas, iters = [], [], set()
            for res in results:
                st = res["schemes"].get(scheme)
                if st is None:
                    continue
                tr = res["truth"]
                mm = tr.airborne if mode == "airborne" else np.ones(tr.step.size, bool)
                sel = mm & ~res["seen"]["main"] & (st["n_sites"] >= 2)
                if sel.any():
                    iters.add(res["it"].key.split("/")[1])
                errs.append(st["best_err"][sel])
                thetas.append(st["best_theta_fold"][sel])
            e = np.concatenate(errs) if errs else np.zeros(0)
            th = np.concatenate(thetas) if thetas else np.zeros(0)
            out.append(dict(mode=mode, scheme=scheme, n=int(e.size), iters=", ".join(sorted(iters)) or "-",
                            min=float(e.min()) if e.size else float("nan"), med=float(np.median(e)) if e.size else float("nan"),
                            theta=float(np.nanmedian(th)) if e.size else float("nan"),
                            lt100=int((e < 100).sum()), lt50=int((e < ERR_LIMITS[0]).sum())))
    return out


def site_records(results: list[dict]) -> list[dict]:
    out = []
    for res in results:
        R = res["R"]
        if R is None:
            continue
        sites = sorted(set(R["site"].tolist()))
        cent = {}
        for s in sites:
            m = R["site"] == s
            cent[s] = (float(np.median(R["sensor_e"][m])), float(np.median(R["sensor_n"][m])))
        for s in sites:
            m = R["site"] == s
            names = {}
            for sid in R["sensor_id"][m]:
                names[sid] = names.get(sid, 0) + 1
            others = [math.hypot(cent[s][0] - cent[o][0], cent[s][1] - cent[o][1]) for o in sites if o != s]
            out.append(dict(iter_key=res["it"].key, site=s, sensors=";".join(f"{k}:{v}" for k, v in sorted(names.items())),
                            n_reports=int(m.sum()), n_families=len({FAMILY_OF_ID[x] for x in names}),
                            nearest_other_site_m=min(others) if others else float("nan")))
    return out


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def pct(x) -> str:
    return "n/a" if x != x else f"{100 * x:.1f} %"


def md_table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def write_report(results, recs, pooled, pairs, clut, sites, nearmiss, checks_ok: bool | None) -> str:
    byk = {(r["iter_key"], r["truth_mode"], r["offsets"], r["unseen_def"]): r for r in recs + pooled}
    keys = [r["it"].key for r in results]
    L: list[str] = []
    L.append("# Triangulation oracle: can the unseen truth points be recovered from bearings?")
    L.append("")
    L.append(f"**Self-checks: {'all passed' if checks_ok else ('not run' if checks_ok is None else 'FAILED')}.**  "
             "Generated by `tools/triangulation_oracle.py` (definitions and file schemas in its docstring). "
             "Testing iterations only (evaluation has no ground truth). Nothing here is fitted.")
    L.append("")
    L.append("## Terms in one line each")
    L += ["* **Truth row**: one drone at one 100 ms step of the scoring grid; 'all' includes parked drones, "
          "'airborne' only drones more than 5 m above their take-off point.",
          "* **Unseen**: no position-sensor track sample within 50 m (3-D) of the truth row at that step "
          "(radars, passive-RF/cyber position feeds; clutter segments count as 'seeing'; the UTM relays VKYY/QTVS are excluded). "
          "'f0' = no reported F0_fuse track row within 50 m. 'with_relay' = relays allowed to see (leakage side note).",
          "* **Bearing sensor / direction finder**: reports only an azimuth (compass direction, degrees clockwise "
          "from north), no distance. Cameras WVFX/FGIX, acoustic EMXG, passive-RF ORFH/YXAB/OEPS/GTYT.",
          "* **Site**: a place where bearing sensors stand; two sensors within 50 m are one site and cannot triangulate each other.",
          "* **Hit**: a bearing report within +-0.5 s whose azimuth points at the true drone position within a gate "
          "(max of 2 x the report's own angular error and 3 / 10 / 15 degrees for camera / acoustic / RF).",
          "* **Crossing angle**: the angle between the two lines of sight at the drone. 90 degrees is ideal, near 0 or 180 "
          "the lines are almost parallel and the crossing point is poorly defined.",
          "* **Expected error**: sqrt((r1 s1)^2 + (r2 s2)^2) / sin(crossing angle), r the distance sensor-drone, s the angular error (radians).",
          "* **Recoverable (<50 m / <25 m)**: at least 2 sites hit and the best site pair has expected error below the limit. "
          "**rec50_emp** is the measured version: the actual crossing of the two reported rays is within 50 m of the truth.",
          "* **Oracle offsets**: azimuth/clock corrections measured against the ground truth (not available on evaluation). "
          "**Realistic**: the GT-free registration (mostly 0 = no correction). **Control**: all azimuths rotated by 120 degrees, "
          "i.e. what pure chance gives with this procedure.",
          "* **Ghost report / false intersection**: a bearing report pointing at no real drone / two rays from different sites "
          "that cross at a place where no drone is."]
    L.append("")
    # ---- sites
    L.append("## Where the bearing sensors stand")
    L.append("")
    L.append("Without two sites that both report, triangulation is impossible. Sites per iteration (sensor:report count):")
    L.append("")
    rows = []
    for s in sites:
        rows.append([s["iter_key"], s["site"], s["sensors"], s["n_reports"],
                     "-" if s["nearest_other_site_m"] != s["nearest_other_site_m"] else f"{s['nearest_other_site_m']:.0f} m"])
    L += md_table(["iteration", "site", "sensors:reports", "reports", "nearest other site"], rows)
    L.append("")
    # ---- main tables
    for mode in MODES:
        L.append(f"## Main result, truth mode '{mode}' (unseen = no position-sensor sample within 50 m)")
        L.append("")
        L.append("Shares are of the UNSEEN rows except the first two columns and the last. "
                 "`rec/all` = recoverable rows as a share of ALL truth rows of the mode (the most triangulation could add to coverage).")
        L.append("")
        for scheme in SCHEMES:
            L.append(f"**Offsets: {scheme}**")
            L.append("")
            rows = []
            for k in keys + ["POOLED"]:
                r = byk[(k, mode, scheme, "main")]
                rows.append([k, r["n_truth"], pct(r["share_unseen"]), pct(r["share_ge1_site"]),
                             f"{r['n_ge2_site']} ({pct(r['share_ge2_site'])})",
                             pct(r["share_rec50"]), pct(r["share_rec25"]), pct(r["share_rec50_emp"]), pct(r["rec50_of_all"])])
            L += md_table(["iteration", "truth rows", "unseen", ">=1 site hits", ">=2 sites hit", "recoverable <50 m",
                           "recoverable <25 m", "rec50_emp (measured)", "rec/all"], rows)
            L.append("")
        # per-iteration count of positive iterations
        for scheme in ("oracle", "realistic"):
            pos = [k for k in keys if byk[(k, mode, scheme, "main")]["n_rec50"] > 0]
            ge10 = [k for k in keys if byk[(k, mode, scheme, "main")]["share_rec50"] >= 0.10]
            L.append(f"* {scheme}: recoverable <50 m exists in {len(pos)} of {len(keys)} iterations, "
                     f"and covers at least 10 % of the unseen rows in {len(ge10)} of {len(keys)}.")
        L.append("")
    # ---- near misses
    L.append("## Closest misses: unseen rows that two sites DO point at")
    L.append("")
    L.append("Rows (main unseen definition) with >= 2 sites hitting, and how good their best crossing would be. "
             "Pooled over the six testing iterations.")
    L.append("")
    L += md_table(["truth mode", "offsets", "rows with >=2 sites", "in iterations", "best expected error",
                   "median expected error", "median crossing angle (folded)", "< 100 m", "< 50 m"],
                  [[x["mode"], x["scheme"], x["n"], x["iters"], "-" if x["n"] == 0 else f"{x['min']:.0f} m",
                    "-" if x["n"] == 0 else f"{x['med']:.0f} m", "-" if x["n"] == 0 else f"{x['theta']:.0f} deg",
                    x["lt100"], x["lt50"]] for x in nearmiss])
    L.append("")
    # ---- other unseen definitions
    L.append("## Other 'unseen' definitions (side notes)")
    L.append("")
    for d, desc in (("main_2d", "horizontal distance only (the definition behind the 'about 39 %' figure)"),
                    ("f0", "stricter: no row of a reported F0_fuse track within 50 m"),
                    ("with_relay", "VKYY / QTVS relay samples also count as seeing (leakage-contaminated)")):
        L.append(f"**{d}**: {desc}. Pooled over the testing iterations.")
        L.append("")
        rows = []
        for mode in MODES:
            for scheme in SCHEMES:
                r = byk[("POOLED", mode, scheme, d)]
                rows.append([mode, scheme, r["n_unseen"], pct(r["share_unseen"]), pct(r["share_ge2_site"]),
                             pct(r["share_rec50"]), pct(r["share_rec25"]), pct(r["rec50_of_all"])])
        L += md_table(["truth mode", "offsets", "unseen rows", "unseen share", ">=2 sites hit", "recoverable <50 m",
                       "recoverable <25 m", "rec/all"], rows)
        L.append("")
    # ---- pairs
    L.append("## Which sensor pairs recover rows (recoverable <50 m, main unseen, pooled over iterations)")
    L.append("")
    L.append("Median crossing angle is folded to 0-90 degrees (90 = perpendicular lines of sight).")
    L.append("")
    for mode in MODES:
        for scheme in ("oracle", "realistic", "control"):
            sel = [p for p in pairs if p["iter_key"] == "POOLED" and p["truth_mode"] == mode and p["offsets"] == scheme]
            L.append(f"**{mode} / {scheme}**")
            L.append("")
            if not sel:
                L.append("No recoverable rows.")
            else:
                L += md_table(["pair", "recoverable rows", "median crossing angle", "median range 1", "median range 2",
                               "median expected error"],
                              [[p["pair"], p["n_rec50"], f"{p['med_theta_fold_deg']:.0f} deg", f"{p['med_r1_m']:.0f} m",
                                f"{p['med_r2_m']:.0f} m", f"{p['med_expected_err_m']:.1f} m"] for p in sel])
            L.append("")
    # ---- clutter
    L.append("## Clutter side: ghost reports and false intersections")
    L.append("")
    L.append("Counted on every 10th grid step that holds an unseen row, over the +-0.5 s report window; "
             "values are per sampled step (about one second of report time). 'usable' = a false intersection whose "
             "expected error is below the limit, i.e. it would look like a real pair.")
    L.append("")
    for mode in MODES:
        rows = []
        for res in results:
            k = res["it"].key
            for scheme in ("oracle", "realistic"):
                c = [x for x in clut if x["iter_key"] == k and x["truth_mode"] == mode and x["offsets"] == scheme
                     and x["sensor_key"] == "ALL"]
                if not c:
                    continue
                c = c[0]
                rows.append([k, scheme, c["n_steps"], f"{c['reports_per_step']:.2f}", f"{c['ghost_per_step']:.2f}",
                             pct(c["ghost_share"]), f"{c['false_per_step']:.2f}", f"{c['false_usable50_per_step']:.3f}",
                             f"{c['false_usable25_per_step']:.3f}", f"{c['true_usable50_per_step']:.3f}"])
        L.append(f"**Truth mode '{mode}'**")
        L.append("")
        L += md_table(["iteration", "offsets", "sampled steps", "reports/step", "ghost reports/step", "ghost share",
                       "false intersections/step", "false usable <50 m /step", "false usable <25 m /step",
                       "true usable <50 m /step"], rows)
        L.append("")
    L.append("Ghost reports per sensor (oracle offsets, truth mode 'airborne'):")
    L.append("")
    rows = []
    for x in clut:
        if x["truth_mode"] == "airborne" and x["offsets"] == "oracle" and x["sensor_key"] != "ALL" and x["reports_per_step"] > 0:
            rows.append([x["iter_key"], x["sensor_key"], f"{x['reports_per_step']:.2f}", f"{x['ghost_per_step']:.2f}",
                         pct(x["ghost_share"])])
    L += md_table(["iteration", "sensor", "reports/step", "ghost/step", "ghost share"], rows)
    L.append("")
    # ---- verdict
    L.append("## Verdict")
    L.append("")
    L += verdict(byk, keys, clut, sites)
    return "\n".join(L) + "\n"


def verdict(byk, keys, clut, sites) -> list[str]:
    a = byk[("POOLED", "airborne", "oracle", "main")]
    ar = byk[("POOLED", "airborne", "realistic", "main")]
    ac = byk[("POOLED", "airborne", "control", "main")]
    al = byk[("POOLED", "all", "oracle", "main")]
    alr = byk[("POOLED", "all", "realistic", "main")]
    fr = byk[("POOLED", "airborne", "realistic", "f0")]
    fo = byk[("POOLED", "airborne", "oracle", "f0")]
    a2 = byk[("POOLED", "airborne", "oracle", "main_2d")]

    def cl(mode, scheme, field):
        v = [x[field] for x in clut if x["truth_mode"] == mode and x["offsets"] == scheme and x["sensor_key"] == "ALL"
             and x["n_steps"] > 0]
        return float(np.mean(v)) if v else float("nan")

    fu = cl("airborne", "realistic", "false_usable50_per_step")
    tu = cl("airborne", "realistic", "true_usable50_per_step")
    L = [f"* Unseen share (no position sensor within 50 m): airborne {pct(a['share_unseen'])}, all {pct(al['share_unseen'])}; "
         f"horizontal-only distance (the brief's ~39 %) airborne {pct(a2['share_unseen'])}; "
         f"with the stricter F0 definition airborne {pct(fo['share_unseen'])}.",
         f"* Of the unseen airborne rows, {pct(a['share_ge1_site'])} have at least one bearing report pointing at the drone and "
         f"{pct(a['share_ge2_site'])} have two or more DIFFERENT sites pointing at it (oracle offsets); realistic offsets: "
         f"{pct(ar['share_ge1_site'])} and {pct(ar['share_ge2_site'])}; chance level (control): {pct(ac['share_ge1_site'])} and "
         f"{pct(ac['share_ge2_site'])}.",
         f"* Recoverable (>=2 sites AND expected error < 50 m): oracle {pct(a['share_rec50'])} of unseen airborne rows "
         f"({pct(a['rec50_of_all'])} of all airborne rows), realistic {pct(ar['share_rec50'])} ({pct(ar['rec50_of_all'])}), "
         f"control {pct(ac['share_rec50'])}; strict < 25 m: oracle {pct(a['share_rec25'])}, realistic {pct(ar['share_rec25'])}. "
         f"Truth mode 'all': oracle {pct(al['share_rec50'])}, realistic {pct(alr['share_rec50'])} of unseen rows.",
         f"* Measured (not just expected) crossing error < 50 m: oracle {pct(a['share_rec50_emp'])}, realistic "
         f"{pct(ar['share_rec50_emp'])} of the unseen airborne rows.",
         f"* Clutter: with realistic offsets there are on average {fu:.3f} false intersections per second that look as good as a real "
         f"pair (< 50 m expected error) against {tu:.3f} real ones in the same windows"
         + (f" (false:real ratio {fu / tu:.1f})." if tu > 0 else " (no real pair at all, so no ratio)."),
         f"* Why: the bearing sensors stand at few places. Sites with >= {SUBSTANTIAL_REPORTS} reports per iteration: "
         + ", ".join(f"{k.split('/')[1]} {sum(1 for x in sites if x['iter_key'] == k and x['n_reports'] >= SUBSTANTIAL_REPORTS)}"
                     for k in keys)
         + ". The cameras and the acoustic sensor usually share one site (WVFX and EMXG); the RF direction finders at the "
           "other sites send a few dozen reports (presence-only ORFH / YXAB) per iteration."]
    per_it = [k for k in keys if byk[(k, "airborne", "realistic", "main")]["share_rec50"] >= 0.10]
    worth = ar["share_rec50"] >= 0.20 and ar["share_rec50"] > 3 * max(ac["share_rec50"], 1e-9) and (tu > 0 and fu <= tu)
    marginal = (not worth) and a["share_rec50"] >= 0.10
    if worth:
        L.append("* **Worth building.** A large share of the unseen rows can be crossed by two sites with realistic offsets, "
                 "far above the chance level, and real pairs are not drowned by false ones. Initiate tracks from "
                 "triangulation, gated by cross-sensor agreement.")
    elif marginal:
        L.append("* **Marginal.** The oracle shows a usable share, but with realistic (GT-free) offsets, the chance level or "
                 "the false-intersection density it is not clearly separable. Only worth building if registration can be made "
                 "reliable on evaluation; start with the sensor pairs in the table above.")
    else:
        L.append("* **Not worth building as a general track initiator.** Even with oracle (ground-truth) offsets the share of "
                 "unseen rows that two differently placed direction finders can both point at, with a crossing accurate enough "
                 "to score, is small, and with GT-free offsets it is smaller still. The cap on what it could add is the "
                 "rec/all figures above. Bearing evidence is more useful as support for existing position tracks "
                 "(the bearing_support features) than as a source of new tracks.")
    L.append(f"* Iterations with >= 10 % of unseen airborne rows recoverable under realistic offsets: "
             f"{', '.join(per_it) if per_it else 'none'}.")
    return L


# --------------------------------------------------------------------------- #
# self-checks
# --------------------------------------------------------------------------- #

def plain_row_hits(S: dict, tg: float, te: float, tn: float) -> dict[int, tuple[float, float]]:
    """Pure-python recomputation of the hits of one truth row: {report index: (residual, range)}."""
    out = {}
    for i in range(len(S["tc"])):
        tc = float(S["t"][i]) + float(S["t_off"][i])
        if not (tc >= tg - WIN_S and tc <= tg + WIN_S):
            continue
        de, dn = te - float(S["sensor_e"][i]), tn - float(S["sensor_n"][i])
        rng = math.sqrt(de * de + dn * dn)
        if not (0 < rng <= RANGE_MAX_M):
            continue
        azp = math.degrees(math.atan2(de, dn))
        raw = float(S["az_deg"][i]) + float(S["az_off"][i])
        res = ((raw - azp + 180.0) % 360.0) - 180.0
        ae = float(S["az_err_deg"][i])
        gate = max(GATE_K * ae if ae == ae else 0.0, FAMILY_FLOOR[FAMILY_OF_ID[S["sensor_id"][i]]])
        if abs(res) <= gate:
            out[i] = (res, rng)
    return out


def plain_best_pair_err(S: dict, hits: dict[int, tuple[float, float]], te: float, tn: float) -> tuple[int, float]:
    """(number of distinct sites, best expected pair error) from a plain hit dict."""
    best_per_site: dict[int, int] = {}
    for i, (res, _) in hits.items():
        s = int(S["site"][i])
        if s not in best_per_site or abs(res) < abs(hits[best_per_site[s]][0]):
            best_per_site[s] = i
    reps = list(best_per_site.values())
    best = math.inf
    for x in range(len(reps)):
        for y in range(x + 1, len(reps)):
            i, j = reps[x], reps[y]
            az_i = math.degrees(math.atan2(te - float(S["sensor_e"][i]), tn - float(S["sensor_n"][i])))
            az_j = math.degrees(math.atan2(te - float(S["sensor_e"][j]), tn - float(S["sensor_n"][j])))
            th = abs(((az_i - az_j + 180.0) % 360.0) - 180.0)
            s = abs(math.sin(math.radians(th)))
            if s <= 1e-6:
                continue
            fi, fj = FAMILY_FLOOR[FAMILY_OF_ID[S["sensor_id"][i]]], FAMILY_FLOOR[FAMILY_OF_ID[S["sensor_id"][j]]]
            ai, aj = float(S["az_err_deg"][i]), float(S["az_err_deg"][j])
            si = max(ai if ai == ai else 0.0, fi / 2)
            sj = max(aj if aj == aj else 0.0, fj / 2)
            err = math.hypot(hits[i][1] * math.radians(si), hits[j][1] * math.radians(sj)) / s
            best = min(best, err)
    return len(best_per_site), best


def run_checks(results: list[dict], det: dict, cal: dict, reg: dict, deployable: frozenset[str]) -> int:
    fails = 0
    rng_ = random.Random(SEED)

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal fails
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        if not ok:
            fails += 1

    log("self-checks")
    # ---- 1. geometry vs closed-form -------------------------------------------------
    ok = (abs(float(az_to(0, 100))) < 1e-12 and abs(float(az_to(100, 0)) - 90) < 1e-12
          and abs(float(az_to(0, -100)) - 180) < 1e-12 and abs(float(az_to(-100, 0)) + 90) < 1e-12)
    check("1a azimuth of the four compass directions", ok)
    ok = (abs(float(wrap180(179 - (-179))) - (-2.0)) < 1e-12 and abs(float(wrap180(-179 - 179)) - 2.0) < 1e-12
          and abs(float(crossing_angle_deg(179.0, -179.0)) - 2.0) < 1e-12
          and abs(float(crossing_angle_deg(180.0, -180.0))) < 1e-12 and abs(float(wrap180(190.0)) + 170.0) < 1e-12)
    check("1b wrap at +-180 (179 vs -179 -> 2 deg, 180 vs -180 -> 0)", ok)
    # right angle: sensors (0,0), (1000,0), drone (500,500); r = 500 sqrt 2, sigma 1 deg each -> 1000 sigma exactly
    a1, a2 = float(az_to(500, 500)), float(az_to(-500, 500))
    th = float(crossing_angle_deg(a1, a2))
    r = 500 * math.sqrt(2)
    err = float(expected_error_m(r, 1.0, r, 1.0, th))
    pe, pn, t1, t2 = intersect_rays(0, 0, a1, 1000, 0, a2)
    ok = (abs(th - 90) < 1e-9 and abs(err - 1000 * math.radians(1.0)) < 1e-9 and abs(float(pe) - 500) < 1e-9
          and abs(float(pn) - 500) < 1e-9 and abs(float(t1) - r) < 1e-9 and abs(float(t2) - r) < 1e-9)
    check("1c right-angle case: crossing 90 deg, error 1000 sigma, intersection (500,500)", ok,
          f"theta={th:.6f} err={err:.6f}")
    # wrap case: sensors (0,0), (100,0), drone (50,-2000): azimuths 178.57 / -178.57, crossing 2 atan(50/2000)
    a1, a2 = float(az_to(50, -2000)), float(az_to(-50, -2000))
    th = float(crossing_angle_deg(a1, a2))
    th_cf = 2 * math.degrees(math.atan(50 / 2000))
    r = math.hypot(50, 2000)
    err = float(expected_error_m(r, 1.0, r, 1.0, th))
    err_cf = math.sqrt(2) * r * math.radians(1.0) / math.sin(math.radians(th_cf))
    pe, pn, _, _ = intersect_rays(0, 0, a1, 100, 0, a2)
    ok = (a1 > 178 and a2 < -178 and abs(th - th_cf) < 1e-9 and abs(err - err_cf) < 1e-6
          and abs(float(pe) - 50) < 1e-6 and abs(float(pn) + 2000) < 1e-6)
    check("1d crossing across the +-180 wrap, near-parallel (2.86 deg): angle, error, intersection", ok,
          f"theta={th:.6f} (cf {th_cf:.6f}) err={err:.2f} (cf {err_cf:.2f})")
    pe, pn, t1, t2 = intersect_rays(0, 0, 10.0, 100, 0, 10.0)
    pe2, pn2, t1b, t2b = intersect_rays(0, 0, 0.0, 100, 0, 180.0)
    inf_ok = (math.isinf(float(expected_error_m(1000, 1.0, 1000, 1.0, 0.0)))
              and math.isinf(float(expected_error_m(1000, 1.0, 1000, 1.0, 180.0))))
    check("1e parallel lines: no intersection, infinite expected error (theta 0 and 180)",
          bool(np.isnan(pe)) and bool(np.isnan(pe2)) and inf_ok)
    # near-parallel but not parallel: error grows like 1/sin(theta)
    e_a = float(expected_error_m(1000, 1.0, 1000, 1.0, 1.0))
    e_b = float(expected_error_m(1000, 1.0, 1000, 1.0, 0.5))
    check("1f near-parallel: halving the crossing angle doubles the error (small-angle limit)",
          abs(e_b / e_a - 2.0) < 1e-3, f"ratio {e_b / e_a:.5f}")
    # finite difference: moving az1 by delta moves the crossing by r1 delta / sin(theta)
    s1, s2, X = (0.0, 0.0), (1000.0, 0.0), (300.0, 800.0)
    a1 = float(az_to(X[0] - s1[0], X[1] - s1[1]))
    a2 = float(az_to(X[0] - s2[0], X[1] - s2[1]))
    th = float(crossing_angle_deg(a1, a2))
    r1 = math.hypot(X[0] - s1[0], X[1] - s1[1])
    d = 1e-5
    pe, pn, _, _ = intersect_rays(s1[0], s1[1], a1 + math.degrees(d), s2[0], s2[1], a2)
    shift = math.hypot(float(pe) - X[0], float(pn) - X[1])
    cf = r1 * d / math.sin(math.radians(th))
    check("1g sideways line displacement r*delta moves the crossing by r*delta/sin(theta)",
          abs(shift / cf - 1) < 1e-3, f"shift {shift:.6f} cf {cf:.6f} theta {th:.2f}")
    # Monte Carlo of the RMS formula
    g = np.random.default_rng(SEED)
    N = 40000
    sg = 0.5
    r2 = math.hypot(X[0] - s2[0], X[1] - s2[1])
    pe, pn, t1, t2 = intersect_rays(s1[0], s1[1], a1 + g.normal(0, sg, N), s2[0], s2[1], a2 + g.normal(0, sg, N))
    rms = float(np.sqrt(np.mean((pe - X[0]) ** 2 + (pn - X[1]) ** 2)))
    f = float(expected_error_m(r1, sg, r2, sg, th))
    check("1h Monte-Carlo RMS error of the crossing within 3 % of the formula", abs(rms / f - 1) < 0.03,
          f"MC {rms:.2f} m, formula {f:.2f} m")
    # site clustering
    sites = assign_sites(np.array([0.0, 49.0, 51.0 + 49.0, 400.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
    sites2 = assign_sites(np.array([0.0, 51.0]), np.array([0.0, 0.0]))
    check("1i sites: 49 m apart merge, 51 m apart do not", sites[0] == sites[1] == sites[4] and sites[2] != sites[0]
          and sites[3] != sites[0] and sites2[0] != sites2[1])
    # gate / sigma rule
    ok = (float(gate_deg(0.01, 3.0)) == 3.0 and float(gate_deg(25.0, 15.0)) == 50.0 and float(gate_deg(np.nan, 10.0)) == 10.0
          and float(sigma_deg(0.01, 3.0)) == 1.5 and float(sigma_deg(np.nan, 10.0)) == 5.0 and float(sigma_deg(25.0, 15.0)) == 25.0)
    check("1j gate = max(2 az_err, floor), sigma = max(az_err, floor / 2), blank az_err -> floor", ok)
    # offset lookup against known rows of sensor_calibration.csv
    off = {(sk, ik): float(calibration_offsets(cal, ik, np.array([sk]), "az_offset_deg")[0]) for sk, ik in
           (("EMXG_ACOUSTIC", "testing/S3-I3"), ("EMXG_ACOUSTIC", "testing/S3-I2"), ("FGIX_CAMERA", "testing/S3-I2"),
            ("WVFX_CAMERA", "testing/S1-I1"), ("WVFX_CAMERA", "testing/S3-I2"), ("EMXG_ACOUSTIC", "testing/S1-I1"),
            ("ORFH_PASSIVE-RF", "testing/S3-I3"))}
    ok = (off[("EMXG_ACOUSTIC", "testing/S3-I3")] == -172.0 and off[("EMXG_ACOUSTIC", "testing/S3-I2")] == -156.5
          and off[("FGIX_CAMERA", "testing/S3-I2")] == -83.5 and off[("WVFX_CAMERA", "testing/S1-I1")] == 5.0
          and off[("WVFX_CAMERA", "testing/S3-I2")] == 0.0 and off[("EMXG_ACOUSTIC", "testing/S1-I1")] == 0.0
          and off[("ORFH_PASSIVE-RF", "testing/S3-I3")] == 0.0)
    check("1k oracle offsets read from sensor_calibration.csv ('applied' rows only; 'rejected' / absent -> 0)", ok)

    # ---- 2. unseen mask vs plain loop -------------------------------------------------
    n_checked = n_boundary = n_unseen_checked = 0
    bad = 0
    for res in results:
        it = res["it"]
        tr = res["truth"]
        slug = C.iter_slug(it.key)
        trk = C.read_columns(TRACKS / f"{slug}__tracks.csv", {"seg_id"}, usecols=["seg_id", "sensor_id"])
        z = np.load(TRACKS / f"{slug}__grid.npz")
        zs, ze, zn, zu, zseg = z["step"], z["e"].astype(float), z["n"].astype(float), z["u"].astype(float), z["seg_id"]
        sid = trk["sensor_id"]
        ft = C.read_columns(FUSED / f"{slug}__tracks.csv", {"track", "reported_f0"}, usecols=["track", "reported_f0"])
        fz = np.load(FUSED / f"{slug}__grid.npz")
        fs, fe, fn, fu_, ftr = fz["step"], fz["e"], fz["n"], fz["u"], fz["track"]
        rep_f0 = ft["reported_f0"] > 0
        idx = rng_.sample(range(tr.step.size), 60)
        # add rows close to the 50 m boundary of the main mask to make the test discriminating
        for i in rng_.sample(range(tr.step.size), 400):
            if len(idx) >= 80:
                break
            s = int(tr.step[i])
            sel = np.flatnonzero(zs == s)
            if sel.size:
                dd = np.sqrt((ze[sel] - tr.e[i]) ** 2 + (zn[sel] - tr.n[i]) ** 2 + (zu[sel] - tr.u[i]) ** 2)
                if np.any(np.abs(dd - UNSEEN_M) < 15):
                    idx.append(i)
                    n_boundary += 1
        for i in idx:
            s = int(tr.step[i])
            exp = {"main": False, "with_relay": False, "f0": False, "main_2d": False}
            tmin = math.inf
            for j in np.flatnonzero(zs == s):
                d = math.sqrt((float(ze[j]) - float(tr.e[i])) ** 2 + (float(zn[j]) - float(tr.n[i])) ** 2
                              + (float(zu[j]) - float(tr.u[i])) ** 2)
                if abs(d - UNSEEN_M) < 1e-9:
                    tmin = 0.0
                sidj = sid[int(zseg[j])]
                d2 = math.sqrt((float(ze[j]) - float(tr.e[i])) ** 2 + (float(zn[j]) - float(tr.n[i])) ** 2)
                if d <= UNSEEN_M and sidj in deployable:
                    exp["with_relay"] = True
                    if sidj not in RELAY_IDS:
                        exp["main"] = True
                if d2 <= UNSEEN_M and sidj in deployable and sidj not in RELAY_IDS:
                    exp["main_2d"] = True
            for j in np.flatnonzero(fs == s):
                d = math.sqrt((float(fe[j]) - float(tr.e[i])) ** 2 + (float(fn[j]) - float(tr.n[i])) ** 2
                              + (float(fu_[j]) - float(tr.u[i])) ** 2)
                if d <= UNSEEN_M and rep_f0[int(ftr[j])]:
                    exp["f0"] = True
            if tmin == 0.0:
                continue
            n_checked += 1
            n_unseen_checked += int(not exp["main"])
            for dname in DEFS:
                if bool(res["seen"][dname][i]) != exp[dname]:
                    bad += 1
    check("2 'unseen' masks (main / main_2d / with_relay / f0) equal a plain python loop", bad == 0 and n_checked >= 300,
          f"{n_checked} random truth rows over {len(results)} iterations ({n_boundary} near the 50 m boundary, "
          f"{n_unseen_checked} unseen), {bad} disagreements")

    # ---- 3. hit counting vs plain loop --------------------------------------------------
    n_rows = n_pairs = n_bad = n_sitebad = n_errbad = 0
    n_hitrows = 0
    for res in results:
        if res["R"] is None:
            continue
        tr = res["truth"]
        need = np.flatnonzero(~res["seen"]["main"] | ~res["seen"]["f0"])
        if need.size == 0:
            continue
        for scheme in ("oracle", "realistic"):
            S = schemed(res["R"], scheme, cal, reg, res["it"].key)
            H = hit_pairs(S, tr.grid_t[tr.step[need]], tr.e[need], tr.n[need])
            rs = row_stats(H, S, need.size, tr.e[need], tr.n[need])
            by_row: dict[int, set] = {}
            for rw, rp in zip(H["row"].tolist(), H["rep"].tolist()):
                by_row.setdefault(rw, set()).add(rp)
            with_hits = sorted(by_row)
            pick = rng_.sample(with_hits, min(40, len(with_hits))) + rng_.sample(range(need.size), 40)
            for rw in pick:
                i = need[rw]
                ph = plain_row_hits(S, float(tr.grid_t[tr.step[i]]), float(tr.e[i]), float(tr.n[i]))
                n_rows += 1
                n_hitrows += int(len(ph) > 0)
                n_pairs += len(ph) + len(S["tc"]) // 50
                if set(ph) != by_row.get(rw, set()):
                    n_bad += 1
                ns, be = plain_best_pair_err(S, ph, float(tr.e[i]), float(tr.n[i]))
                if ns != int(rs["n_sites"][rw]):
                    n_sitebad += 1
                v = float(rs["best_err"][rw])
                if not ((math.isinf(be) and math.isinf(v)) or abs(be - v) <= 1e-6 * max(1.0, be)):
                    n_errbad += 1
    check("3 hit sets, distinct-site counts and best pair error equal a plain python loop",
          n_rows >= 200 and n_hitrows >= 100 and n_bad == 0 and n_sitebad == 0 and n_errbad == 0,
          f"{n_rows} (row, all-reports) comparisons, {n_hitrows} with >=1 hit; disagreements: hit sets {n_bad}, "
          f"site counts {n_sitebad}, pair errors {n_errbad}")
    # mutation: ignoring the oracle offsets must change the hit counts (the check can see offsets)
    changed = 0
    for res in results:
        if res["R"] is None:
            continue
        tr = res["truth"]
        need = np.flatnonzero(~res["seen"]["main"])
        S1 = schemed(res["R"], "oracle", cal, reg, res["it"].key)
        if not np.any(S1["az_off"] != 0):
            continue
        S0 = dict(S1)
        S0["azc"] = wrap180(S1["az_deg"])
        h1 = hit_pairs(S1, tr.grid_t[tr.step[need]], tr.e[need], tr.n[need])["row"].size
        h0 = hit_pairs(S0, tr.grid_t[tr.step[need]], tr.e[need], tr.n[need])["row"].size
        changed += int(h0 != h1)
    check("3m mutation: dropping the oracle azimuth offsets changes the hit pair count", changed >= 1,
          f"{changed} iterations changed")

    # ---- 4. testing-only sensors never counted -------------------------------------------
    used_ids = set()
    for res in results:
        if res["R"] is not None:
            used_ids |= set(res["R"]["sensor_id"].tolist())
    check("4a report sensors of every iteration are deployable, none testing-only",
          used_ids <= deployable and not (used_ids & TESTING_ONLY_IDS), f"ids: {sorted(used_ids)}")
    # injection: a perfectly pointing NJQL (testing-only) report and a fake non-deployable id must be dropped
    res0 = next(r for r in results if r["R"] is not None)
    it_key = res0["it"].key
    inj = {k: np.repeat(np.asarray(v)[:1], 2) for k, v in det.items()}
    inj["iter_key"][:] = it_key
    inj["meas_kind"][:] = "bearing"
    inj["duplicate"][:] = False
    inj["sensor_pos_valid"][:] = 1.0
    inj["az_deg"][:] = 45.0
    inj["sensor_e"][:] = 0.0
    inj["sensor_n"][:] = 0.0
    inj["t"][:] = 1.0
    inj["sensor_id"][0], inj["sensor_key"][0] = "NJQL", "NJQL_CAMERA"
    inj["sensor_id"][1], inj["sensor_key"][1] = "WVFX", "WVFX_CAMERA"
    kept = select_reports(inj, it_key, deployable)
    kept_mut = select_reports(inj, it_key, deployable | {"NJQL"})
    check("4b injected testing-only NJQL report is dropped by select_reports (and kept when the filter is mutated)",
          kept is not None and kept["sensor_id"].tolist() == ["WVFX"]
          and kept_mut is not None and sorted(kept_mut["sensor_id"].tolist()) == ["NJQL", "WVFX"])
    # position pool: testing-only sensors (EUIG, LJJX, MMJJ) and relays must not make a row 'seen' in main
    leak = 0
    for res in results:
        slug = C.iter_slug(res["it"].key)
        trk = C.read_columns(TRACKS / f"{slug}__tracks.csv", {"seg_id"}, usecols=["seg_id", "sensor_id"])
        pool, m_main, m_all, sid = load_position_pool(res["it"], res["truth"].grid_t.size, deployable)
        z = np.load(TRACKS / f"{slug}__grid.npz")
        used = set(sid[np.unique(z["seg_id"][pool.order][m_main])].tolist())
        leak += len(used & (TESTING_ONLY_IDS | RELAY_IDS)) + len(used - deployable)
    check("4c position pool of the 'main' unseen mask has no testing-only sensor and no VKYY / QTVS segment", leak == 0,
          f"{leak} offending sensor ids")
    log(f"self-checks: {'ALL PASSED' if fails == 0 else str(fails) + ' FAILED'}")
    return fails


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    t0 = time.time()
    deployable = C.deployable_sensor_ids()
    log("loading detections (clean) ...")
    det = Q.load_clean(["az_err_deg", "sensor_id"])
    keep = np.isin(det["meas_kind"], Q.BEARING_KINDS)
    det = {k: v[keep] for k, v in det.items()}
    cal = C.read_columns(CALIBRATION, {"value"})
    reg = C.read_columns(REGISTRATION, {"offset_deg"})
    its = [i for i in C.iterations() if i.tree == "testing"]
    log(f"{len(its)} testing iterations; deployable sensor ids: {', '.join(sorted(deployable))}")
    results = [process_iteration(it, det, cal, reg, deployable) for it in its]
    recs: list[dict] = []
    for res in results:
        recs += [add_shares(r) for r in summarize(res)]
    pooled = pool_records(recs)
    pairs = pair_records(results)
    clut = clutter_records(results)
    sites = site_records(results)
    nearmiss = nearmiss_records(results)

    OUT.mkdir(parents=True, exist_ok=True)
    C.write_columns(OUT / "triangulation_rows.csv",
                    {c: [r.get(c, "") for r in recs + pooled] for c in REC_COLS}, REC_COLS)
    pc = ["iter_key", "truth_mode", "offsets", "pair", "n_rec50", "med_theta_fold_deg", "med_r1_m", "med_r2_m",
          "med_expected_err_m"]
    C.write_columns(OUT / "triangulation_pairs.csv", {c: [r[c] for r in pairs] for c in pc}, pc)
    cc = ["iter_key", "truth_mode", "offsets", "sensor_key", "n_steps", "reports_per_step", "ghost_per_step",
          "ghost_share", "intersections_per_step", "false_per_step", "false_usable50_per_step",
          "false_usable25_per_step", "true_per_step", "true_usable50_per_step"]
    C.write_columns(OUT / "triangulation_clutter.csv", {c: [r[c] for r in clut] for c in cc}, cc)
    sc = ["iter_key", "site", "sensors", "n_reports", "n_families", "nearest_other_site_m"]
    C.write_columns(OUT / "triangulation_sites.csv", {c: [r[c] for r in sites] for c in sc}, sc)

    fails = run_checks(results, det, cal, reg, deployable)
    md = write_report(results, recs, pooled, pairs, clut, sites, nearmiss, fails == 0)
    (OUT / "triangulation.md").write_text(md, encoding="utf-8")
    log(f"wrote {OUT / 'triangulation.md'} and 4 csv files ({time.time() - t0:.0f} s)")
    pa = {(p["truth_mode"], p["offsets"]): p for p in pooled if p["unseen_def"] == "main"}
    for mode in MODES:
        for scheme in SCHEMES:
            p = pa[(mode, scheme)]
            log(f"POOLED {mode:8s} {scheme:9s} unseen {pct(p['share_unseen'])}  >=1 site {pct(p['share_ge1_site'])}  "
                f">=2 sites {pct(p['share_ge2_site'])}  rec<50 {pct(p['share_rec50'])}  rec<25 {pct(p['share_rec25'])}  "
                f"emp<50 {pct(p['share_rec50_emp'])}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
