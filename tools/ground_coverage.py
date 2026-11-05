#!/usr/bin/env python3
"""How much of the ground truth is a drone parked on the ground, which sensors see it, and would a
tracker gain by reporting parked drones? (testing iterations only; evaluation has no truth.)

Why: the template grid equals the Dronetag span exactly, so the official ground truth very likely
contains drones sitting on the ground (about half of all truth rows). In GOSPA every truth row
nobody reports costs c^p / 2 = 1250 (c = 50, p = 2), and every reported row that matches nothing
costs the same, so the question is which side of that trade a parked-drone rule lands on.

Definitions (all from the shared modules, nothing new)
* Truth: ``truth.load_truth(it, 'all')`` (every interpolated 100 ms row, one row = 0.1 drone-second)
  and ``'airborne'`` (height above take-off > ``truth.AIRBORNE_HEIGHT_M`` = 5 m). Ground row =
  an 'all' row that is not airborne.
* Phase of a ground row, per flight: ``before first take-off`` (before the flight's first
  airborne row), ``between flights`` (a ground row between its first and last airborne row),
  ``after last landing``, ``never airborne`` (flights without any airborne row).
* Detections: ``cuas_clean.load_clean``; position rows only (pos3d / pos2d, finite e, n), minus
  null_island and duplicate rows. Bearing-only sensors (WVFX EMXG FGIX NJQL ORFH YXAB OEPS GTYT)
  have no position and are NOT assessed here (they need an association step, not a distance).
  Each position row falls in exactly one group, in this priority: ``self_report`` (within 5 m of
  its own sensor, dropped by cuas_clean from usable_pos), ``home`` / ``uav_tx`` / ``em_tx`` (class
  'Home' / 'UAV Transmitter' / 'EM Transmitter'), ``far`` (> 15 km from the iteration origin),
  else ``regular``. "Usable" = every group except self_report (= usable_pos).
* Coverage: a truth row is covered by a (sensor, group) when that sensor has at least one row of
  the group within ``GATE_M`` = 30 m horizontally and ``WIN_S`` = 2 s of the truth row's grid time
  (raw report time; no clock calibration). Headline = deployable sensors only
  (``C.deployable_sensor_ids()``); also shown: deployable without the two Dronetag relays that
  exist in evaluation (QTVS, VKYY: in testing they report the Dronetag feed itself, so their
  coverage of BLUE flights is leakage), and all sensors (adds testing-only LJJX EUIG MMJJ).
* Parked spots: ground segments of >= ``MIN_SEG_S`` = 5 s clustered at ``SPOT_M`` = 10 m (single
  linkage on each segment's median position).
* Ground reference for the hold rule: ``u_ref`` = GT-free sensor-site ellipsoidal height. Two
  candidates are scored against the truth's ground height (median ``u`` of each flight's
  pre-take-off rows) and the better RMS one is used: ``median`` = median ``sensor_u`` of the
  iteration's deployable located sensors with a non-zero ``sensor_alt``, ``nearest`` = the
  ``sensor_u`` of the located deployable sensor nearest to the track end (or start). The choice
  between the two candidates is made on testing; evaluation iterations use the same code.
* Hold rule (GT-free; the "track" is a stand-in): the perfect track = the flight's airborne truth
  rows. A track *segment* = a maximal run of airborne rows of one flight of >= ``MIN_AIR_S`` = 5 s
  (shorter runs are height flicker, not a confirmed track; their rows stay in the perfect track).
  * ``hold``: when a segment ends with ``u_end - u_ref <= X`` keep reporting its last position
    (u included) until the iteration end, or ``T`` seconds, or until a new segment (any flight,
    including the same one) starts within ``R_M`` = 30 m horizontally of the held position.
  * ``start``: when a segment starts with ``u_start - u_ref <= X`` report its start position from
    the iteration start (or ``T`` seconds before the start). If an earlier low segment ended
    within R_M of that start, the report begins at that segment's end (the gap is a continuation,
    not an early start); with ``both`` the hold already covers the gap and ``start`` only adds
    what the (T-limited) hold leaves uncovered.
  Cost = full GOSPA^p summed over the 100 ms steps, c = 50, p = 2, 3-D (score.PRIMARY), of the
  truth against (perfect track + added rows). 'all' truth: the gain of a variant = 1250 x ground
  rows (cost of the perfect track alone) minus the cost with the added rows, i.e. it is already
  net of the false rows the rule adds (an added row may also win an airborne drone's partner
  slot in the optimal assignment: counted as 'swap'). 'airborne' truth: an added row can only be
  a false track, cost 1250 each (self-check 6). The perfect track is an UPPER BOUND on what a
  real tracker would gain.

CLI: ``python3 tools/ground_coverage.py`` writes data/derived/ground_coverage.md and prints the
numbered self-checks (exit 1 if one fails). Deterministic (check 2 is a seeded uniform sample of
500 truth rows against a brute force rebuilt from the raw cleaned columns; check 2b adds a hit-enriched sample).
Rule selection is leave-one-iteration-out (section 4): the in-sample best rule is reported but is not the decision basis.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuas_clean as Q  # noqa: E402
import cuas_common as C  # noqa: E402
import gospa as G  # noqa: E402
import truth as T  # noqa: E402

OUT = C.DERIVED / "ground_coverage.md"
WIN_S = 2.0
GATE_M = 30.0
DT = 0.1
MIN_SEG_S = 5.0
SPOT_M = 10.0
MIN_AIR_S = 5.0
R_M = 30.0
SELF_M = Q.SELF_REPORT_M
GOSPA_C = 50.0
GOSPA_P = 2.0
HALF = GOSPA_C ** GOSPA_P / 2.0          # 1250: cost of one miss or one false row
X_GRID = (5.0, 10.0, 20.0, 40.0)         # "low" = within X m of the sensor-site ellipsoidal height
T_GRID = (10.0, 30.0, 60.0, 120.0, 300.0, float("inf"))   # hold / early-start duration limit, seconds
Q_GRID = (0.5, 0.8, 0.9, 1.0)            # assumed probability that the official truth includes parked drones
Q_HEAD = 0.8                             # used to pick the headline rule and in the leave-one-out check
HORIZON_BINS_S = (0.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, float("inf"))
SHORT_GAP_S = 5.0                        # ground runs shorter than this sit between airborne runs
GROUPS = ("regular", "home", "uav_tx", "em_tx", "self_report", "far")
SPECIAL = {"Home": "home", "UAV Transmitter": "uav_tx", "EM Transmitter": "em_tx"}
GT_RELAYS = frozenset({"QTVS", "VKYY"})
PHASES = ("before first take-off", "between flights", "after last landing", "never airborne")
VEHICLE_NAMES = frozenset({"Skyranger"})  # catalog: ground vehicle (its 'airborne' rows are GPS height noise)
SEED = 20241028


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def pct(x: float, nd: int = 1) -> str:
    return "-" if not np.isfinite(x) else f"{100 * x:.{nd}f} %"


def table(head: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out + [""]


def pctiles(x: np.ndarray, ps=(5, 25, 50, 75, 95, 100)) -> list[str]:
    if len(x) == 0:
        return ["-"] * len(ps)
    return [f"{v:.1f}" for v in np.percentile(x, ps)]


def tl(t_: float) -> str:
    return "no limit" if not np.isfinite(t_) else f"{t_:g} s"


# --------------------------------------------------------------------------- #
# truth: phases, flights, segments
# --------------------------------------------------------------------------- #

def flight_phases(tr: T.Truth) -> np.ndarray:
    """Per truth row of the 'all' set: 0 airborne, else 1..4 = PHASES index + 1 (ground rows)."""
    phase = np.zeros(len(tr.step), dtype=np.int8)
    for fi in range(len(tr.flights)):
        idx = np.flatnonzero(tr.flight == fi)        # ascending step (load_truth sorts by step, flight)
        if idx.size == 0:
            continue
        air = np.flatnonzero(tr.airborne[idx])
        if air.size == 0:
            phase[idx] = 4
            continue
        k = np.arange(idx.size)
        ph = np.where(k < air[0], 1, np.where(k > air[-1], 3, 2)).astype(np.int8)
        ph[tr.airborne[idx]] = 0
        phase[idx] = ph
    return phase


def run_lengths(tr: T.Truth) -> np.ndarray:
    """Per truth row: number of rows in the maximal same-state (ground / airborne) run of its flight."""
    out = np.zeros(len(tr.step), dtype=np.int64)
    for fi in range(len(tr.flights)):
        idx = np.flatnonzero(tr.flight == fi)
        if idx.size == 0:
            continue
        a = tr.airborne[idx]
        cut = np.flatnonzero(a[1:] != a[:-1]) + 1
        starts = np.r_[0, cut]
        ends = np.r_[cut, idx.size]
        out[idx] = np.repeat(ends - starts, ends - starts)
    return out


def ground_segments(tr: T.Truth, phase: np.ndarray) -> list[dict]:
    """Maximal runs of ground rows of one flight inside one phase."""
    segs = []
    for fi in range(len(tr.flights)):
        idx = np.flatnonzero(tr.flight == fi)
        if idx.size == 0:
            continue
        ph = phase[idx]
        cut = np.flatnonzero(ph[1:] != ph[:-1]) + 1
        starts = np.r_[0, cut]
        ends = np.r_[cut, idx.size]
        for s, e in zip(starts, ends):
            if ph[s] == 0:
                continue
            r = idx[s:e]
            segs.append(dict(flight=fi, phase=int(ph[s]), rows=r, n=int(r.size),
                             e=float(np.median(tr.e[r])), nn=float(np.median(tr.n[r])),
                             u=float(np.median(tr.u[r]))))
    return segs


def air_segments(tr: T.Truth) -> list[dict]:
    """Track segments of the stand-in perfect track: maximal airborne runs of one flight with
    >= MIN_AIR_S seconds of rows."""
    segs = []
    min_rows = int(round(MIN_AIR_S / DT))
    for fi in range(len(tr.flights)):
        idx = np.flatnonzero(tr.flight == fi)
        if idx.size == 0:
            continue
        a = tr.airborne[idx]
        cut = np.flatnonzero(a[1:] != a[:-1]) + 1
        starts = np.r_[0, cut]
        ends = np.r_[cut, idx.size]
        for s, e in zip(starts, ends):
            if not a[s] or e - s < min_rows:
                continue
            r0, r1 = idx[s], idx[e - 1]
            segs.append(dict(flight=fi, step0=int(tr.step[r0]), step1=int(tr.step[r1]),
                             p0=(float(tr.e[r0]), float(tr.n[r0]), float(tr.u[r0])),
                             p1=(float(tr.e[r1]), float(tr.n[r1]), float(tr.u[r1]))))
    return segs


# --------------------------------------------------------------------------- #
# detections: grouping, coverage
# --------------------------------------------------------------------------- #

def group_codes(d: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    """Group index (into GROUPS) of every row in `mask`; -1 elsewhere."""
    g = np.full(len(d["t"]), -1, dtype=np.int8)
    cls = d["cls"]
    cat = np.zeros(len(g), dtype=np.int8)           # 0 = regular
    for name, grp in SPECIAL.items():
        cat[cls == name] = GROUPS.index(grp)
    cat[(cat == 0) & d["far"]] = GROUPS.index("far")
    cat[d["self_report"]] = GROUPS.index("self_report")
    g[mask] = cat[mask]
    return g


def cover_matrix(tr: T.Truth, t, e, n, combo, n_combo: int) -> np.ndarray:
    """(rows, n_combo) bool: truth row r is covered by combo k when some detection of combo k lies
    within WIN_S seconds and GATE_M metres horizontally. Searches around each detection's grid
    step, so the cost is (detections x steps in window x flights)."""
    cov = np.zeros((len(tr.step), n_combo), dtype=bool)
    gt = tr.grid_t
    S = len(gt)
    nf = len(tr.flights)
    row_of = np.full((nf, S), -1, dtype=np.int64)
    row_of[tr.flight, tr.step] = np.arange(len(tr.step))
    keep = (t >= gt[0] - WIN_S) & (t <= gt[-1] + WIN_S)
    keep &= (e >= tr.e.min() - GATE_M) & (e <= tr.e.max() + GATE_M)
    keep &= (n >= tr.n.min() - GATE_M) & (n <= tr.n.max() + GATE_M)
    t, e, n, combo = t[keep], e[keep], n[keep], combo[keep]
    s0 = np.rint((t - gt[0]) / DT).astype(np.int64)
    half = int(WIN_S / DT) + 1
    for dstep in range(-half, half + 1):
        s = s0 + dstep
        ok = (s >= 0) & (s < S)
        sc = np.clip(s, 0, S - 1)
        ok &= np.abs(t - gt[sc]) <= WIN_S
        for f in range(nf):
            r = row_of[f, sc]
            v = ok & (r >= 0)
            if not v.any():
                continue
            rv = r[v]
            near = np.hypot(e[v] - tr.e[rv], n[v] - tr.n[rv]) <= GATE_M
            cov[rv[near], combo[v][near]] = True
    return cov


# --------------------------------------------------------------------------- #
# per-iteration container
# --------------------------------------------------------------------------- #

class IterData:
    pass


def sensor_sites(d: dict[str, np.ndarray], it_mask: np.ndarray, deployable: frozenset[str]) -> list[dict]:
    """Located sensors of one iteration (valid site position): median e, n, u, alt per sensor_key."""
    m = it_mask & (d["sensor_pos_valid"] == 1) & np.isfinite(d["sensor_e"]) & np.isfinite(d["sensor_n"])
    out = []
    for key in sorted(set(d["sensor_key"][m])):
        mm = m & (d["sensor_key"] == key)
        sid = str(d["sensor_id"][mm][0])
        out.append(dict(key=key, sid=sid, deployable=sid in deployable,
                        e=float(np.median(d["sensor_e"][mm])), n=float(np.median(d["sensor_n"][mm])),
                        u=float(np.nanmedian(d["sensor_u"][mm])), alt=float(np.nanmedian(d["sensor_alt"][mm]))))
    return out


def nearest_dist(e, n, sites: list[dict]) -> np.ndarray:
    if not sites:
        return np.full(len(e), np.inf)
    se = np.array([s["e"] for s in sites])
    sn = np.array([s["n"] for s in sites])
    return np.sqrt((e[:, None] - se[None, :]) ** 2 + (n[:, None] - sn[None, :]) ** 2).min(axis=1)


def build_iteration(it: C.Iteration, d: dict, dmask: np.ndarray, gcode: np.ndarray, n_combo: int,
                    combo_index: dict, deployable: frozenset[str]) -> IterData:
    X = IterData()
    X.it = it
    X.tr = T.load_truth(it, "all")
    X.tr_air = T.load_truth(it, "airborne")
    tr = X.tr
    X.ground = ~tr.airborne
    X.flight_gu = np.array([float(np.median(tr.u[(tr.flight == fi) & X.ground])) if ((tr.flight == fi) & X.ground).any()
                            else np.inf for fi in range(len(tr.flights))])
    X.phase = flight_phases(tr)
    X.runlen = run_lengths(tr)
    sel = dmask & (d["iter_key"] == it.key)
    X.sel = sel
    X.n_det = int(sel.sum())
    X.ci = np.array([combo_index[(k, int(g))] for k, g in zip(d["sensor_key"][sel], gcode[sel])], dtype=np.int64)
    X.cov = cover_matrix(tr, d["t"][sel], d["e"][sel], d["n"][sel], X.ci, n_combo)
    X.sites = sensor_sites(d, d["iter_key"] == it.key, deployable)
    X.sites_dep = [s for s in X.sites if s["deployable"]]
    X.dist_dep = nearest_dist(tr.e, tr.n, X.sites_dep)
    return X


# --------------------------------------------------------------------------- #
# hold / start-early rule
# --------------------------------------------------------------------------- #

def u_reference(X: IterData, p: tuple, mode: str) -> float:
    """GT-free ground height near position p = (e, n, u): sensor-site ellipsoidal heights."""
    if mode == "oracle":                       # NOT GT-free: the flight's own median ground height (sensitivity only)
        return p[3]
    cand = [s for s in X.sites_dep if s["alt"] != 0 and np.isfinite(s["u"])]
    if not cand:
        return np.nan
    if mode == "median":
        return float(np.median([s["u"] for s in cand]))
    dist = [np.hypot(s["e"] - p[0], s["n"] - p[1]) for s in cand]
    return float(cand[int(np.argmin(dist))]["u"])


EMPTY_ADDED = (np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int8),
               np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))


def build_added(X: IterData, segs: list[dict], variant: str, xlim: float, tmax: float, ref_mode: str):
    """Rows the rule adds: (step, e, n, u, kind, age) arrays. kind 0 = hold, 1 = early start; age =
    steps since the hold began / steps until the track starts (>= 0)."""
    n_steps = len(X.tr.grid_t)
    tsteps = n_steps if not np.isfinite(tmax) else int(round(tmax / DT))
    steps, es, ns, us, kinds, ages, fls = [], [], [], [], [], [], []

    def add(a, b, p, kind, ref, fl):
        if b > a:
            steps.append(np.arange(a, b, dtype=np.int64))
            ages.append(np.abs(steps[-1] - ref))
            fls.append(np.full(b - a, fl, dtype=np.int64))
            k = b - a
            es.append(np.full(k, p[0]))
            ns.append(np.full(k, p[1]))
            us.append(np.full(k, p[2]))
            kinds.append(np.full(k, kind, dtype=np.int8))

    low_end = [s["p1"][2] - u_reference(X, (*s["p1"], X.flight_gu[s["flight"]]), ref_mode) <= xlim for s in segs]
    low_start = [s["p0"][2] - u_reference(X, (*s["p0"], X.flight_gu[s["flight"]]), ref_mode) <= xlim for s in segs]
    use_hold = variant in ("hold", "both")
    use_start = variant in ("start", "both")
    for k, s in enumerate(segs):
        if use_hold and low_end[k]:
            stop = n_steps
            for o in segs:
                if o["step0"] > s["step1"] and np.hypot(o["p0"][0] - s["p1"][0], o["p0"][1] - s["p1"][1]) <= R_M:
                    stop = min(stop, o["step0"])
            add(s["step1"] + 1, min(stop, s["step1"] + 1 + tsteps), s["p1"], 0, s["step1"] + 1, s["flight"])
        if use_start and low_start[k]:
            prev_end = -1
            for j, o in enumerate(segs):
                if (o["step1"] < s["step0"] and low_end[j]
                        and np.hypot(o["p1"][0] - s["p0"][0], o["p1"][1] - s["p0"][1]) <= R_M):
                    prev_end = max(prev_end, o["step1"])
            lo = max(0, s["step0"] - tsteps)
            if prev_end >= 0:                                   # continuation of an earlier low segment
                lo = max(lo, prev_end + 1)
                if use_hold:
                    lo = max(lo, prev_end + 1 + tsteps)         # the T-limited hold covers the first part
            add(lo, s["step0"], s["p0"], 1, s["step0"], s["flight"])
    if not steps:
        return EMPTY_ADDED
    return (np.concatenate(steps), np.concatenate(es), np.concatenate(ns), np.concatenate(us),
            np.concatenate(kinds), np.concatenate(ages), np.concatenate(fls))


def eval_added(X: IterData, added, detail: bool = False) -> dict:
    """Full GOSPA (c = 50, p = 2, 3-D) of 'all' truth against the perfect airborne track plus the
    added rows. gain_all = cost without added rows (1250 x ground rows) minus cost with them.
    Against 'airborne' truth the perfect track matches at distance 0 and an added row can only be
    false, so its cost is exactly 1250 per row (self-check 6)."""
    tr = X.tr
    air = tr.airborne
    nb = int(air.sum())
    st = np.concatenate([tr.step[air], added[0]])
    y = np.vstack([np.column_stack([tr.e[air], tr.n[air], tr.u[air]]),
                   np.column_stack([added[1], added[2], added[3]])])
    x = np.column_stack([tr.e, tr.n, tr.u])
    seq = G.gospa_sequence(tr.step, x, st, y, len(tr.grid_t), GOSPA_C, GOSPA_P)
    pairs = seq["pairs"]
    ap = pairs[pairs[:, 2] >= nb] if len(pairs) else pairs
    on_ground = X.ground[ap[:, 1]] if len(ap) else np.zeros(0, dtype=bool)
    ap_g = ap[on_ground]
    run_s = X.runlen[ap_g[:, 1]] * DT if len(ap_g) else np.zeros(0)
    kd = added[4]
    n_gr = int(X.ground.sum())
    extra = {}
    if detail:
        mm = np.zeros(len(added[0]), dtype=bool)
        mm[ap_g[:, 2] - nb] = True
        extra = dict(matched_mask=mm, age=added[5], flight=added[6], step=added[0])
    return dict(
        **extra, n_added=len(added[0]), n_matched=int(len(ap_g)), n_false=len(added[0]) - int(len(ap_g)),
        n_swap=int(len(ap) - len(ap_g)),
        gain_all=HALF * n_gr - float(seq["cost"].sum()), loss_air=HALF * len(added[0]), n_ground=n_gr,
        matched_long=int((run_s >= SHORT_GAP_S).sum()),
        matched_kind=np.bincount(kd[ap_g[:, 2] - nb], minlength=2) if len(ap_g) else np.zeros(2, dtype=np.int64),
        added_kind=np.bincount(kd, minlength=2) if len(kd) else np.zeros(2, dtype=np.int64),
        step_cost=seq["cost"])


def full_gospa(X: IterData, added, mode: str) -> float:
    """Full GOSPA cost sum for truth = 'all' or 'airborne' (used by the self-checks)."""
    tr = X.tr if mode == "all" else X.tr_air
    air = X.tr.airborne
    st = np.concatenate([X.tr.step[air], added[0]])
    y = np.vstack([np.column_stack([X.tr.e[air], X.tr.n[air], X.tr.u[air]]),
                   np.column_stack([added[1], added[2], added[3]])])
    x = np.column_stack([tr.e, tr.n, tr.u])
    return float(G.gospa_sequence(tr.step, x, st, y, len(tr.grid_t), GOSPA_C, GOSPA_P)["cost"].sum())


def aggregate(results: dict, keys: list[str], variant, xlim, tmax, which=None) -> dict:
    """Sum the per-iteration results of one (variant, X, T) over `which` (default: all keys)."""
    out = dict(n_added=0, n_matched=0, n_false=0, n_swap=0, gain_all=0.0, loss_air=0.0, n_ground=0, matched_long=0,
               matched_kind=np.zeros(2, dtype=np.int64), added_kind=np.zeros(2, dtype=np.int64))
    for kk in (which or keys):
        r = results[(kk, variant, xlim, tmax)]
        for f in out:
            out[f] = out[f] + r[f]
    return out


def cluster_spots(pos: np.ndarray, radius: float) -> np.ndarray:
    """Single-linkage labels (union-find) for points within `radius` of each other."""
    n = len(pos)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]) <= radius:
                parent[find(j)] = find(i)
    roots = [find(i) for i in range(n)]
    uniq = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.array([uniq[r] for r in roots], dtype=np.int64)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    t_start = time.time()
    fails: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    its = T.testing_iterations()
    keys = [it.key for it in its]
    deployable = C.deployable_sensor_ids()
    print("loading detections ...")
    d = Q.load_clean(["sensor_id", "cls", "sensor_u"])
    test = np.array([k.startswith("testing/") for k in d["iter_key"]])
    is_pos = np.isin(d["meas_kind"], Q.POSITION_KINDS)
    dmask = test & is_pos & ~d["null_island"] & ~d["duplicate"] & np.isfinite(d["e"]) & np.isfinite(d["n"])
    gcode = group_codes(d, dmask)
    pairs_present = sorted({(k, int(g)) for k, g in zip(d["sensor_key"][dmask], gcode[dmask])})
    combos = [(k, GROUPS[g]) for k, g in pairs_present]
    combo_index = {kg: i for i, kg in enumerate(pairs_present)}
    sensor_keys = sorted({k for k, _ in combos})
    sid_of = {k: k.split("_")[0] for k in sensor_keys}
    print(f"  {int(dmask.sum()):,} testing position rows, {len(combos)} (sensor, group) combos")

    data: dict[str, IterData] = {}
    for it in its:
        data[it.key] = build_iteration(it, d, dmask, gcode, len(combos), combo_index, deployable)
        print(f"  {it.key}: {len(data[it.key].tr.step):,} truth rows, {data[it.key].n_det:,} detections")

    def cmask(pred) -> np.ndarray:
        return np.array([pred(k, g) for k, g in combos], dtype=bool)

    is_dep = cmask(lambda k, g: sid_of[k] in deployable)
    is_dep_norelay = cmask(lambda k, g: sid_of[k] in deployable and sid_of[k] not in GT_RELAYS)
    is_usable = cmask(lambda k, g: g != "self_report")
    is_regular = cmask(lambda k, g: g == "regular")
    is_special = cmask(lambda k, g: g in ("home", "uav_tx", "em_tx"))
    is_far = cmask(lambda k, g: g == "far")

    def covered(X: IterData, *masks) -> np.ndarray:
        m = np.ones(len(combos), dtype=bool)
        for k in masks:
            m &= k
        return X.cov[:, m].any(axis=1)

    def share_gr(mask_combo, which="ground", phases=None) -> float:
        num_, den_ = 0, 0
        for X in data.values():
            rows = X.ground if which == "ground" else ~X.ground
            if phases is not None:
                rows = rows & np.isin(X.phase, phases)
            num_ += int(covered(X, mask_combo)[rows].sum())
            den_ += int(rows.sum())
        return num_ / den_ if den_ else float("nan")

    L: list[str] = []
    A = L.append

    # ----------------------------------------------------------------------- Q1
    tot_rows = sum(len(data[k].tr.step) for k in keys)
    phase_all = np.concatenate([data[k].phase for k in keys])
    rows_by_phase = [int((phase_all == i).sum()) for i in range(5)]
    q1_rows = []
    for k in keys + ["pooled"]:
        ph = phase_all if k == "pooled" else data[k].phase
        n = len(ph)
        r = [k, f"{n:,}", f"{n * DT:,.0f}", pct((ph == 0).mean()), pct((ph > 0).mean())]
        for i in range(1, 5):
            r.append(f"{pct((ph == i).mean())} ({(ph == i).sum() * DT:,.0f} s)")
        q1_rows.append(r)
    n_ground = int(sum(X.ground.sum() for X in data.values()))
    ground_share = n_ground / tot_rows

    flight_rows = []
    never_flights = []
    for k in keys:
        X = data[k]
        for fi, m in enumerate(X.tr.flights):
            idx = X.tr.flight == fi
            ph = X.phase[idx]
            gmove = float((X.tr.speed[idx][ph > 0] > 1.0).mean()) if (ph > 0).any() else float("nan")
            if (ph == 4).any():
                never_flights.append((k, m["drone_name"], (ph == 4).sum() * DT))
            flight_rows.append([k, m["drone_name"], m["side"], f"{idx.sum() * DT:,.0f}", f"{(ph == 0).sum() * DT:,.0f}",
                                f"{(ph == 1).sum() * DT:,.0f}", f"{(ph == 2).sum() * DT:,.0f}",
                                f"{(ph == 3).sum() * DT:,.0f}", f"{(ph == 4).sum() * DT:,.0f}",
                                pct(gmove, 0), f"{X.tr.height[idx].max():.0f}"])
    step_rows = []
    for k in keys:
        X = data[k]
        S = len(X.tr.grid_t)
        n_g = np.bincount(X.tr.step[X.ground], minlength=S)
        n_a = np.bincount(X.tr.step[~X.ground], minlength=S)
        step_rows.append([k, f"{S:,}", pct((n_g > 0).mean()), pct(((n_g > 0) & (n_a == 0)).mean()),
                          pct(((n_g == 0) & (n_a == 0)).mean()), f"{n_g.max()}", f"{n_g.mean():.2f}"])
    short_rows = sum(int((((X.runlen * DT) < SHORT_GAP_S) & X.ground).sum()) for X in data.values())

    # ----------------------------------------------------------------------- Q2
    q2_sensor = []
    for key in sensor_keys:
        sid = sid_of[key]
        tag = ("deployable" if sid in deployable else "testing-only") + (", gt relay" if sid in GT_RELAYS or sid == "LJJX" else "")
        cells = [pct(share_gr(cmask(lambda k, gg, key=key, g=g: k == key and gg == g))) for g in GROUPS]
        anyc = pct(share_gr(cmask(lambda k, gg, key=key: k == key)))
        air_c = pct(share_gr(cmask(lambda k, gg, key=key: k == key and gg != "self_report"), "air"))
        q2_sensor.append([key, tag, *cells, anyc, air_c])
    q2_sens_it = []
    for key in sensor_keys:
        row = [key]
        mk_ = cmask(lambda k, gg, key=key: k == key and gg != "self_report")
        for k in keys:
            X = data[k]
            n_rows_ = int(np.bincount(X.ci, minlength=len(combos))[cmask(lambda kk, gg, key=key: kk == key)].sum())
            if n_rows_ == 0:
                row.append("no reports")
            else:
                c_ = covered(X, mk_)
                row.append(f"{c_[X.ground].mean() * 100:.1f} % / {c_[~X.ground].mean() * 100:.1f} %")
        q2_sens_it.append(row)
    unions = [
        ("deployable, usable (regular + Home/transmitter classes) [headline per brief]", is_dep & is_usable),
        ("deployable, regular class only", is_dep & is_regular),
        ("deployable, Home / UAV Transmitter / EM Transmitter classes only", is_dep & is_special),
        ("deployable, usable + self_report rows", is_dep),
        ("deployable without QTVS/VKYY relays, usable [honest headline]", is_dep_norelay & is_usable),
        ("deployable without relays, regular class only", is_dep_norelay & is_regular),
        ("deployable without relays, usable + self_report rows", is_dep_norelay),
        ("all sensors incl. testing-only (side note), usable", is_usable),
    ]
    q2_union = [[name] + [pct(share_gr(m, "ground", [p])) for p in range(1, 5)]
                + [pct(share_gr(m, "ground")), pct(share_gr(m, "air"))] for name, m in unions]
    q2_iter = []
    for k in keys:
        X = data[k]
        row = [k]
        for _, m in (unions[0], unions[4], unions[6], unions[7]):
            c = covered(X, m)
            row.append(f"{c[X.ground].mean() * 100:.1f} % / {c[~X.ground].mean() * 100:.1f} %")
        q2_iter.append(row)
    m_h = is_dep_norelay & is_usable
    q2_flight = []
    for k in keys:
        X = data[k]
        c = covered(X, m_h)
        for fi, m in enumerate(X.tr.flights):
            idx = X.tr.flight == fi
            g, a = idx & X.ground, idx & ~X.ground
            q2_flight.append([k, m["drone_name"], m["side"], f"{g.sum() * DT:,.0f}",
                              pct(c[g].mean()) if g.any() else "-", pct(c[a].mean()) if a.any() else "-"])
    bins = [0, 5, 50, 200, 500, 1000, 1e9]
    dist_all = np.concatenate([data[k].dist_dep[data[k].ground] for k in keys])
    cov_h = np.concatenate([covered(data[k], m_h)[data[k].ground] for k in keys])
    cov_self = np.concatenate([covered(data[k], is_dep_norelay)[data[k].ground] for k in keys])
    q2_dist = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (dist_all >= lo) & (dist_all < hi)
        lbl = f"{lo:g} - {hi:g} m" if hi < 1e8 else f"> {lo:g} m"
        q2_dist.append([lbl, f"{m.sum() * DT:,.0f}", pct(m.mean()), pct(cov_h[m].mean()) if m.any() else "-",
                        pct(cov_self[m].mean()) if m.any() else "-"])
    far_hits = int(sum(X.cov[:, is_far].sum() for X in data.values()))

    # ----------------------------------------------------------------------- Q3
    seg_all = []
    for k in keys:
        X = data[k]
        X.gsegs = ground_segments(X.tr, X.phase)
        seg_all += X.gsegs
    q3_dist_rows = []
    for k in keys + ["pooled"]:
        dd = (np.concatenate([data[x].dist_dep[data[x].ground] for x in keys]) if k == "pooled"
              else data[k].dist_dep[data[k].ground])
        dd = dd[np.isfinite(dd)]
        q3_dist_rows.append([k, f"{len(dd) * DT:,.0f}"] + pctiles(dd) + [pct(float((dd < SELF_M).mean()) if len(dd) else np.nan)])
    spot_rows, spots_pooled = [], []
    for k in keys:
        X = data[k]
        long_segs = [s for s in X.gsegs if s["n"] * DT >= MIN_SEG_S]
        if not long_segs:
            continue
        pos = np.array([[s["e"], s["nn"]] for s in long_segs])
        lab = cluster_spots(pos, SPOT_M)
        n_spots = int(lab.max()) + 1
        flights_per = [len({s["flight"] for s, l in zip(long_segs, lab) if l == sp}) for sp in range(n_spots)]
        S = len(X.tr.grid_t)
        conc = 0
        for sp in range(n_spots):
            cnt = np.zeros(S, dtype=np.int64)
            for s, l in zip(long_segs, lab):
                if l == sp:
                    cnt[X.tr.step[s["rows"]]] += 1
            conc = max(conc, int(cnt.max()))
        multi_rows = sum(s["n"] for s, l in zip(long_segs, lab) if flights_per[l] >= 2)
        tot_ = sum(s["n"] for s in long_segs)
        dsite = nearest_dist(pos[:, 0], pos[:, 1], X.sites_dep)
        spot_rows.append([k, len(long_segs), len({s["flight"] for s in long_segs}), n_spots, max(flights_per),
                          conc, pct(multi_rows / tot_), f"{np.median(dsite):.0f}"])
        spots_pooled.append((len(long_segs), n_spots, multi_rows, tot_))
    to_rows, to_stats = [], {"first": [], "last": [], "land": [], "dz": []}
    ret = []
    for k in keys:
        X = data[k]
        tr = X.tr
        for fi, m in enumerate(tr.flights):
            idx = np.flatnonzero(tr.flight == fi)
            ph = X.phase[idx]
            if not (ph == 0).any():
                continue
            air_idx = idx[ph == 0]
            if len(air_idx) * DT < MIN_AIR_S:
                continue
            vehicle = m["drone_name"] in VEHICLE_NAMES
            pre, post = idx[ph == 1], idx[ph == 3]
            a0, a1 = air_idx[0], air_idx[-1]
            row = [k, m["drone_name"], m["side"], f"{len(pre) * DT:,.0f}"]
            if len(pre) * DT >= MIN_SEG_S:
                d_med = float(np.hypot(tr.e[a0] - np.median(tr.e[pre]), tr.n[a0] - np.median(tr.n[pre])))
                d_last = float(np.hypot(tr.e[a0] - tr.e[pre[-1]], tr.n[a0] - tr.n[pre[-1]]))
                du = float(tr.u[a0] - np.median(tr.u[pre]))
                row += [f"{d_med:.1f}", f"{d_last:.1f}", f"{du:.1f}"]
                if not vehicle:
                    to_stats["first"].append(d_med)
                    to_stats["last"].append(d_last)
                    to_stats["dz"].append(abs(du))
            else:
                row += ["-", "-", "-"]
            if len(post) * DT >= MIN_SEG_S:
                d_land = float(np.hypot(tr.e[a1] - np.median(tr.e[post]), tr.n[a1] - np.median(tr.n[post])))
                row += [f"{len(post) * DT:,.0f}", f"{d_land:.1f}"]
                if not vehicle:
                    to_stats["land"].append(d_land)
            else:
                row += ["-", "-"]
            row.append("vehicle" if vehicle else "")
            to_rows.append(row)
            if (not vehicle) and len(pre) * DT >= MIN_SEG_S and len(post) * DT >= MIN_SEG_S:
                ret.append(float(np.hypot(np.median(tr.e[pre]) - np.median(tr.e[post]),
                                          np.median(tr.n[pre]) - np.median(tr.n[post]))))

    # ----------------------------------------------------------------------- Q4
    ref_err = {"median": [], "nearest": []}
    ref_err_it = {}
    ref_rows = []
    for k in keys:
        X = data[k]
        tr = X.tr
        errs = {"median": [], "nearest": []}
        for fi, m in enumerate(tr.flights):
            idx = np.flatnonzero(tr.flight == fi)
            pre = idx[X.phase[idx] == 1]
            if len(pre) * DT < MIN_SEG_S:
                continue
            p = (float(np.median(tr.e[pre])), float(np.median(tr.n[pre])), float(np.median(tr.u[pre])))
            for mode in errs:
                errs[mode].append(u_reference(X, p, mode) - p[2])
        for mode in errs:
            ref_err[mode] += errs[mode]
        ref_err_it[k] = {mm: list(v) for mm, v in errs.items()}
        ref_rows.append([k, str(len(X.sites_dep)), f"{u_reference(X, (0, 0, 0), 'median'):.1f}"] +
                        [f"{np.median(errs[mm]):+.1f} / {np.max(np.abs(errs[mm])):.1f}" if errs[mm] else "-"
                         for mm in ("median", "nearest")])
    rms = {mm: float(np.sqrt(np.mean(np.square(v)))) for mm, v in ref_err.items()}
    ref_mode = min(rms, key=rms.get)
    # leave-one-out: the reference mode for held-out iteration k is chosen on the other five only
    fold_mode = {}
    for k in keys:
        rr = {mm: float(np.sqrt(np.mean(np.square([e_ for kk in keys if kk != k for e_ in ref_err_it[kk][mm]]))))
              for mm in ref_err}
        fold_mode[k] = min(rr, key=rr.get)

    for k in keys:
        data[k].asegs = air_segments(data[k].tr)
    seg_end_rows = []
    for k in keys:
        X = data[k]
        ends = [s["p1"][2] - u_reference(X, (*s["p1"], X.flight_gu[s["flight"]]), ref_mode) for s in X.asegs]
        starts = [s["p0"][2] - u_reference(X, (*s["p0"], X.flight_gu[s["flight"]]), ref_mode) for s in X.asegs]
        seg_end_rows.append([k, len(X.asegs)] + [f"{sum(v <= x for v in ends)} / {sum(v <= x for v in starts)}" for x in X_GRID])

    variants = ("hold", "start", "both")
    results: dict[tuple, dict] = {}
    t_eval = time.time()
    for k in keys:
        X = data[k]
        for variant in variants:
            for xlim in X_GRID:
                for tmax in T_GRID:
                    results[(k, variant, xlim, tmax)] = eval_added(X, build_added(X, X.asegs, variant, xlim, tmax, ref_mode))
    print(f"  evaluated {len(results)} rule variants in {time.time() - t_eval:.0f} s")

    def agg(variant, xlim, tmax, which=None):
        return aggregate(results, keys, variant, xlim, tmax, which)

    cand = [(v, x, t_) for v in variants for x in X_GRID for t_ in T_GRID]

    def objective(z, which=None, q=Q_HEAD):
        """Expected cost saving if the official truth includes parked drones with probability q."""
        a = agg(*z, which=which)
        return (q * a["gain_all"] - (1 - q) * a["loss_air"], -z[1], -T_GRID.index(z[2]))

    q4_rows = []
    for z in cand:
        a = agg(*z)
        q4_rows.append([z[0], f"{z[1]:g}", tl(z[2]), f"{a['n_added']:,}", f"{a['n_matched']:,}", f"{a['n_false']:,}",
                        pct(a["n_matched"] / a["n_ground"]), f"{a['gain_all']:,.0f}", f"{-a['loss_air']:,.0f}",
                        pct(a["loss_air"] / (a["gain_all"] + a["loss_air"])) if a["gain_all"] > 0 else "never"])
    max_gain = HALF * n_ground
    best_by_q = {q: max(cand, key=lambda z, q=q: objective(z, None, q)) for q in Q_GRID}
    best_q_rows = []
    for q in Q_GRID:
        z = best_by_q[q]
        a = agg(*z)
        ev = q * a["gain_all"] - (1 - q) * a["loss_air"]
        best_q_rows.append([f"{q:g}", z[0], f"{z[1]:g}", tl(z[2]), f"{a['n_added']:,}", f"{a['n_matched']:,}", f"{a['n_false']:,}",
                            f"{a['gain_all']:,.0f}", f"{-a['loss_air']:,.0f}", f"{ev:,.0f}",
                            pct(ev / max_gain) if ev > 0 else "none (rule does not pay)"])
    best = best_by_q[Q_HEAD]
    ab = agg(*best)
    # sensitivity: the same grid with the TRUE ground height of each flight as u_ref (not GT-free)
    results_or = {}
    for k in keys:
        X = data[k]
        for z in cand:
            results_or[(k, *z)] = eval_added(X, build_added(X, X.asegs, *z, "oracle"))

    def objective_or(z, q):
        a = aggregate(results_or, keys, *z)
        return (q * a["gain_all"] - (1 - q) * a["loss_air"], -z[1], -T_GRID.index(z[2]))

    best_or_rows = []
    for q in Q_GRID:
        z = max(cand, key=lambda z, q=q: objective_or(z, q))
        a = aggregate(results_or, keys, *z)
        ev = q * a["gain_all"] - (1 - q) * a["loss_air"]
        best_or_rows.append([f"{q:g}", z[0], f"{z[1]:g}", tl(z[2]), f"{a['n_added']:,}", f"{a['n_matched']:,}", f"{a['n_false']:,}",
                             f"{a['gain_all']:,.0f}", f"{-a['loss_air']:,.0f}", f"{ev:,.0f}",
                             pct(ev / max_gain) if ev > 0 else "none (rule does not pay)"])
    # leave-one-iteration-out: reference mode and (rule, X, T) both chosen on the other five iterations
    results_alt = {}
    other_mode = [mm for mm in ref_err if mm != ref_mode][0]
    for k in keys:
        X = data[k]
        for z in cand:
            results_alt[(k, *z)] = eval_added(X, build_added(X, X.asegs, *z, other_mode))
    res_mode = {ref_mode: results, other_mode: results_alt}

    def loio_pick(k: str, q: float):
        res = res_mode[fold_mode[k]]
        others = [kk for kk in keys if kk != k]

        def obj(z):
            a = aggregate(res, keys, *z, which=others)
            return (q * a["gain_all"] - (1 - q) * a["loss_air"], -z[1], -T_GRID.index(z[2]))
        return max(cand, key=obj), res

    loio_rows, loio_tot = [], np.zeros(2)
    for k in keys:
        pick, res = loio_pick(k, Q_HEAD)
        r = res[(k, *pick)]
        loio_tot += (r["gain_all"], r["loss_air"])
        loio_rows.append([k, fold_mode[k], f"{pick[0]}, X={pick[1]:g}, T={tl(pick[2])}", f"{r['gain_all']:,.0f}", f"{-r['loss_air']:,.0f}",
                          f"{Q_HEAD * r['gain_all'] - (1 - Q_HEAD) * r['loss_air']:,.0f}", f"{r['n_matched']:,} / {r['n_false']:,}"])
    loio_rows.append(["pooled held-out", "", "", f"{loio_tot[0]:,.0f}", f"{-loio_tot[1]:,.0f}",
                      f"{Q_HEAD * loio_tot[0] - (1 - Q_HEAD) * loio_tot[1]:,.0f}", ""])
    # pooled held-out expected saving for every assumed probability q (rule re-picked per q)
    loio_q = {}
    loio_q_rows = []
    for q in Q_GRID:
        tot = np.zeros(2)
        n_neg = 0
        for k in keys:
            pick, res = loio_pick(k, q)
            r = res[(k, *pick)]
            tot += (r["gain_all"], r["loss_air"])
            n_neg += int(q * r["gain_all"] - (1 - q) * r["loss_air"] < 0)
        ev = q * tot[0] - (1 - q) * tot[1]
        loio_q[q] = ev
        loio_q_rows.append([f"{q:g}", f"{tot[0]:,.0f}", f"{-tot[1]:,.0f}", f"{ev:,.0f}", pct(ev / max_gain), f"{n_neg} of {len(keys)}"])
    per_iter_best = []
    for k in keys:
        r = results[(k, *best)]
        per_iter_best.append([k, f"{r['n_ground']:,}", f"{r['n_added']:,}", f"{r['n_matched']:,}", f"{r['n_false']:,}",
                              f"{r['gain_all']:,.0f}", f"{-r['loss_air']:,.0f}",
                              f"{Q_HEAD * r['gain_all'] - (1 - Q_HEAD) * r['loss_air']:,.0f}", f"{HALF * r['n_ground']:,.0f}"])

    # hit rate of an added row against how far it is from the track (time since a track ended / until one starts)
    hz_ref_x = 20.0
    hz = {}
    for variant in ("hold", "start"):
        ages, hits, absent, iidx = [], [], [], []
        for ki_, k in enumerate(keys):
            X = data[k]
            r = eval_added(X, build_added(X, X.asegs, variant, hz_ref_x, float("inf"), ref_mode), detail=True)
            present = np.zeros((len(X.tr.flights), len(X.tr.grid_t)), dtype=bool)
            present[X.tr.flight, X.tr.step] = True
            ages.append(r["age"] * DT)
            hits.append(r["matched_mask"])
            absent.append(~present[r["flight"], r["step"]])
            iidx.append(np.full(len(r["step"]), ki_, dtype=np.int64))
        hz[variant] = (np.concatenate(ages), np.concatenate(hits), np.concatenate(absent), np.concatenate(iidx))
    why_rows = []
    for variant in ("hold", "start"):
        age, hit, absent, _ = hz[variant]
        nn = len(age)
        why_rows.append([variant, f"{nn:,}", pct(hit.mean()), pct((~hit & absent).mean()), pct((~hit & ~absent).mean())])
    def horizon_end(age, hit, th):
        """Largest bin edge up to which every bin's hit rate is >= th (0 = even the first bin fails)."""
        end = 0.0
        for lo, hi in zip(HORIZON_BINS_S[:-1], HORIZON_BINS_S[1:]):
            m = (age >= lo) & (age < hi)
            if m.any() and hit[m].mean() >= th:
                end = hi
            else:
                break
        return end

    hz_rows, horizon, horizon_loo = [], {}, {}
    margin_rows = []
    for variant in ("hold", "start"):
        age, hit, _, iidx = hz[variant]
        for lo, hi in zip(HORIZON_BINS_S[:-1], HORIZON_BINS_S[1:]):
            m = (age >= lo) & (age < hi)
            lbl = f"{lo:g} - {hi:g} s" if np.isfinite(hi) else f"> {lo:g} s"
            h = float(hit[m].mean()) if m.any() else float("nan")
            per_it = [float(hit[m & (iidx == i_)].mean()) if (m & (iidx == i_)).any() else float("nan") for i_ in range(len(keys))]
            loo = [float(hit[m & (iidx != i_)].mean()) if (m & (iidx != i_)).any() else float("nan") for i_ in range(len(keys))]
            hz_rows.append([variant, lbl, f"{m.sum():,}", f"{hit[m].sum():,}", pct(h), f"{(2 * hit[m].sum() - m.sum()) * HALF:,.0f}"])
            margin_rows.append([variant, lbl, pct(h), f"{pct(np.nanmin(loo))} - {pct(np.nanmax(loo))}" if np.isfinite(loo).any() else "-",
                                " / ".join(pct(v, 0) for v in per_it)])
        for q in (0.8, 0.9, 1.0):
            th = 1.0 / (2 * q)
            horizon[(variant, q)] = horizon_end(age, hit, th)
            # conservative: the horizon must also hold with any one iteration left out
            horizon_loo[(variant, q)] = min([horizon[(variant, q)]] + [horizon_end(age[iidx != i_], hit[iidx != i_], th)
                                                                         for i_ in range(len(keys))])

    # ----------------------------------------------------------------------- self-checks
    print("self-checks")
    ok1, detail1 = True, []
    for k in keys:
        X = data[k]
        n_all = len(X.tr.step)
        n_gr = int((X.tr.height <= T.AIRBORNE_HEIGHT_M).sum())
        n_ai = len(X.tr_air.step)
        ok1 &= (n_gr + n_ai == n_all) and (int(X.ground.sum()) == n_gr)
        detail1.append(f"{n_gr}+{n_ai}={n_all}")
    check("1. ground + airborne truth rows == 'all' rows per iteration", ok1, "; ".join(detail1))

    # Independent reference: rebuilt from the raw cleaned columns (no X.sel / X.ci / gcode / dmask), with a scalar
    # per-row rule for the group, so that grouping, row filtering and the sensor/group join are all exercised.
    is_pos_names = set(Q.POSITION_KINDS)

    def ref_combos(X, r) -> set:
        tt = X.tr.grid_t[X.tr.step[r]]
        cand_ = np.flatnonzero((d["iter_key"] == X.it.key) & (np.abs(d["t"] - tt) <= WIN_S))
        out_ = set()
        for j in cand_:
            if d["meas_kind"][j] not in is_pos_names or d["null_island"][j] or d["duplicate"][j]:
                continue
            if not (np.isfinite(d["e"][j]) and np.isfinite(d["n"][j])):
                continue
            if np.hypot(d["e"][j] - X.tr.e[r], d["n"][j] - X.tr.n[r]) > GATE_M:
                continue
            if d["self_report"][j]:
                g_ = "self_report"
            elif d["cls"][j] in SPECIAL:
                g_ = SPECIAL[d["cls"][j]]
            elif d["far"][j]:
                g_ = "far"
            else:
                g_ = "regular"
            out_.add((d["sensor_key"][j], g_))
        return out_

    def run_sample_check(label: str, hit_enriched: bool) -> None:
        rng = np.random.default_rng(SEED + (7 if hit_enriched else 0))
        bad = n_chk = n_hit_rows = 0
        for ki, k in enumerate(keys):
            X = data[k]
            n_rows = len(X.tr.step)
            per = 500 // len(keys) + (1 if ki < 500 % len(keys) else 0)
            if hit_enriched:
                hit_idx = np.flatnonzero(X.cov.any(axis=1))
                n_hit = min(per // 2, len(hit_idx))
                pick = np.concatenate([rng.choice(n_rows, per - n_hit, replace=False),
                                       rng.choice(hit_idx, n_hit, replace=False) if n_hit else np.zeros(0, dtype=np.int64)])
            else:
                pick = rng.choice(n_rows, per, replace=False)
            for r in pick:
                got = {combos[i] for i in np.flatnonzero(X.cov[r])}
                ref = ref_combos(X, r)
                bad += int(got != ref)
                n_hit_rows += int(len(ref) > 0)
                n_chk += 1
        check(label, bad == 0 and n_chk == 500, f"{n_chk} rows checked, {n_hit_rows} with a hit, {bad} mismatches")

    run_sample_check("2. coverage == independent brute-force loop on a uniform random sample of 500 truth rows", False)
    run_sample_check("2b. same, on a sample half drawn from covered rows (so hits are exercised)", True)

    ph_ok = True
    for k in keys:
        X = data[k]
        ph_ok &= bool(((X.phase == 0) == X.tr.airborne).all())
    seg_rows = sum(s["n"] for s in seg_all)
    check("3. phases partition the rows; ground segments cover the ground rows exactly",
          ph_ok and seg_rows == n_ground and sum(rows_by_phase[1:]) == n_ground and rows_by_phase[0] + n_ground == tot_rows,
          f"{seg_rows} segment rows == {n_ground} ground rows")

    base_ok = ideal_ok = True
    for k in keys:
        X = data[k]
        base_ok &= abs(full_gospa(X, EMPTY_ADDED, "all") - HALF * int(X.ground.sum())) < 1e-6
        base_ok &= abs(full_gospa(X, EMPTY_ADDED, "airborne")) < 1e-6
        xx = np.column_stack([X.tr.e, X.tr.n, X.tr.u])
        ideal_ok &= abs(G.gospa_sequence(X.tr.step, xx, X.tr.step, xx, len(X.tr.grid_t), GOSPA_C, GOSPA_P)["cost"].sum()) < 1e-9
    check("4. perfect airborne track costs 1250 x ground rows ('all') and 0 ('airborne'); all-truth track costs 0",
          base_ok and ideal_ok)

    ok5 = ok6 = True
    n5 = 0
    worst5 = 0.0
    rng5 = np.random.default_rng(SEED + 1)
    for k in keys:
        X = data[k]
        for z in (best, ("both", 20.0, float("inf")), ("hold", 10.0, 120.0), ("start", 40.0, 30.0)):
            added = build_added(X, X.asegs, *z, ref_mode)
            r = results[(k, *z)]
            ok6 &= abs(full_gospa(X, added, "airborne") - r["loss_air"]) < 1e-6
            # brute force: per-step GOSPA of random steps with the exact small solver
            air = X.tr.airborne
            est_step = np.concatenate([X.tr.step[air], added[0]])
            est = np.vstack([np.column_stack([X.tr.e[air], X.tr.n[air], X.tr.u[air]]),
                             np.column_stack([added[1], added[2], added[3]])])
            for stp in rng5.choice(len(X.tr.grid_t), 60, replace=False):
                xt = np.column_stack([X.tr.e, X.tr.n, X.tr.u])[X.tr.step == stp]
                ye = est[est_step == stp]
                ref = G.gospa(xt, ye, GOSPA_C, GOSPA_P).cost
                worst5 = max(worst5, abs(ref - r["step_cost"][stp]))
                n5 += 1
    ok5 = worst5 < 1e-6
    check("5. per-step GOSPA cost == gospa.gospa on 60 random steps per iteration (4 variants each)",
          ok5, f"max abs diff {worst5:.2e} over {n5} steps")
    check("6. the full GOSPA against 'airborne' truth charges exactly 1250 per added row", ok6)
    check("7. no far-group row (> 15 km from the origin) lies within the gate of a truth row", far_hits == 0, f"{far_hits} hits")
    check("8. a variant result exists for every (iteration, variant, X, T)",
          len(results) == len(keys) * len(cand), f"{len(results)} entries")

    # ----------------------------------------------------------------------- report text
    cov_head = share_gr(unions[0][1])
    cov_honest = share_gr(unions[4][1])
    cov_honest_reg = share_gr(unions[5][1])
    cov_self_gain = share_gr(unions[6][1]) - cov_honest
    cov_air_honest = share_gr(unions[4][1], "air")
    cov_never = share_gr(unions[4][1], "ground", [4])
    pre_share = rows_by_phase[1] / tot_rows
    ntake = len(to_stats["first"])
    fa = np.array(to_stats["first"])
    p50_first = float(np.median(fa)) if ntake else float("nan")
    share_first_50 = float((fa < GOSPA_C).mean()) if ntake else float("nan")
    share_first_10 = float((fa < 10).mean()) if ntake else float("nan")
    tp = np.array(spots_pooled)

    A("# Parked drones: how much of the ground truth is on the ground, who sees it, and should we report it?")
    A("")
    A("Generated by `python3 tools/ground_coverage.py` (testing iterations only; the evaluation tree has no truth). "
      "Every number below is recomputed on each run. Terms in plain language:")
    A("")
    A("* **Truth row**: one drone at one 100 ms time step; 10 rows = 1 drone-second. 'All' truth keeps every row; "
      "'airborne' truth keeps only rows more than 5 m above the drone's take-off height.")
    A("* **GOSPA** (the track-accuracy score): at every time step, match our reported points to the true points; "
      "each match costs (distance in metres)^2, each true drone we do not report costs 1250 (a 'miss') and each reported "
      "point that matches nothing costs 1250 (a 'false track'). Matches farther than 50 m do not count. The numbers "
      "below are GOSPA cost units summed over all steps: lower is better, and 'gain' means cost saved.")
    A("* **Covered**: a truth row has a sensor report within 30 m horizontally and 2 s of it.")
    A("* **Deployable sensors**: sensor types that also exist in the evaluation data (all except LJJX, EUIG, MMJJ, NJQL). "
      "QTVS and VKYY are deployable but in testing they re-broadcast the Dronetag truth feed itself; they are shown "
      "separately so that the honest headline excludes them.")
    A("")
    A("## Summary and recommendation")
    A("")
    A(f"1. **{pct(ground_share, 0)} of all truth rows are on the ground** ({n_ground:,} of {tot_rows:,} rows = "
      f"{n_ground * DT:,.0f} of {tot_rows * DT:,.0f} drone-seconds, six testing iterations). "
      f"{pct(pre_share, 0)} of all rows are before the drone's first take-off, {pct(rows_by_phase[2] / tot_rows, 0)} between flights, "
      f"{pct(rows_by_phase[3] / tot_rows, 0)} after the last landing and {pct(rows_by_phase[4] / tot_rows, 0)} belong to the "
      f"{len(never_flights)} flights that never take off ({', '.join(f'{n} {int(s)} s' for _, n, s in never_flights)}). "
      f"If the official GOSPA keeps these rows, reporting nothing for them costs {max_gain:,.0f} cost units on testing alone.")
    A(f"2. **Sensors see few parked drones.** The honest union of deployable position sensors (without the two "
      f"Dronetag relays) has a report within 30 m / 2 s of {pct(cov_honest)} of the ground rows (regular class only: "
      f"{pct(cov_honest_reg)}), against {pct(cov_air_honest)} of the airborne rows. With the QTVS/VKYY relays included it is "
      f"{pct(cov_head)}. Rows that cleaning drops as `self_report` (within 5 m of their own sensor) add {pct(cov_self_gain)} of ground rows. "
      f"Only {int((dist_all < SELF_M).sum())} ground rows ({(dist_all < SELF_M).sum() * DT:.1f} s) lie within the 5 m self_report zone of a deployable sensor, "
      f"so that cleaning rule hides no parked drone. "
      f"Rows labelled Home / UAV Transmitter / EM Transmitter by themselves cover {pct(share_gr(unions[2][1]))} of ground rows. "
      f"The never-airborne flights are covered {pct(cov_never)} of their time.")
    A(f"3. **A take-off spot is predictable from where the airborne track starts**: over {ntake} flights the median distance "
      f"between the parked position and the first airborne position is {p50_first:.1f} m; {pct(share_first_10, 0)} are within 10 m "
      f"and {pct(share_first_50, 0)} within the 50 m GOSPA cut-off.")
    z1, zq = best_by_q[1.0], best
    a1, aq = agg(*z1), agg(*zq)
    ev_in = Q_HEAD * aq["gain_all"] - (1 - Q_HEAD) * aq["loss_air"]      # in-sample (optimistic)
    ev_q = loio_q[Q_HEAD]                                                 # held-out (decision basis)
    n_neg_head = sum(1 for r_ in loio_rows[:-1] if float(r_[5].replace(",", "")) < 0)
    A(f"4. **A GT-free parked-drone rule does not pay once it is tested honestly on held-out iterations.** The stand-in is a perfect "
      f"airborne track (an upper bound). *Held-out result* (the reference mode and the rule `hold` / `start` / `both`, X, T are each chosen on the "
      f"other five iterations and applied to the sixth, section 4): pooled expected saving at p = {Q_HEAD:g} is {ev_q:,.0f} units "
      f"({pct(ev_q / max_gain)} of the {max_gain:,.0f} recoverable); {n_neg_head} of {len(keys)} held-out iterations lose; even if parked drones are "
      f"certain to be scored (p = 1) the held-out saving is {loio_q[1.0]:,.0f}. *In-sample result* (optimistic, because the best of "
      f"{len(cand)} variants is picked on the same six iterations it is reported on): best rule at p = {Q_HEAD:g} is `{zq[0]}`, X = {zq[1]:g} m, "
      f"T = {tl(zq[2])}, expected saving {ev_in:,.0f} units ({pct(ev_in / max_gain)}); at p = 1 `{z1[0]}`, X = {z1[1]:g} m, T = {tl(z1[2])} saves "
      f"{a1['gain_all']:,.0f} units with {a1['n_added']:,} added rows ({a1['n_matched']:,} hit a parked drone, {a1['n_false']:,} false), and against "
      f"airborne-only truth the same rule costs {a1['loss_air']:,.0f}. The in-sample gain is selection bias, not an expected saving. "
      f"Unlimited holds / early starts are strongly negative even if parked drones are scored (e.g. `hold`, X = 20 m, no limit: "
      f"{agg('hold', 20.0, float('inf'))['gain_all']:,.0f} units), because most added rows are false.")
    A("")
    if ev_q <= 0:
        rec = ("**Do not build a parked-drone hold / early-start rule.** Chosen without peeking at the iteration it is applied to, it loses "
               f"on the pooled held-out data at every assumed probability that parked drones are scored ({', '.join(f'p = {q:g}: {loio_q[q]:,.0f}' for q in Q_GRID)} units), "
               "and a perfect track is the best case for it. The in-sample 'best rule' above is a selection artefact. "
               "Report a parked drone only when a sensor actually sees it (never-airborne and pre-take-off flights are covered "
               f"{pct(cov_never, 0)} and {pct(share_gr(unions[4][1], 'ground', [1]), 0)} of the time by the honest sensor union). "
               "If the owner later learns that parked drones are scored, revisit with the real tracker's output instead of a perfect track.")
    elif ev_q / max_gain < 0.05:
        rec = (f"The parked-drone rule is a small side gain at best on held-out data ({pct(ev_q / max_gain)} of the recoverable cost at probability "
               f"{Q_HEAD:g}) and only for short horizons (in-sample pick: `{zq[0]}` with X = {zq[1]:g} m and T = {tl(zq[2])}, never unlimited). "
               "Build it last, behind a switch, and only after the airborne tracker is good; do not spend effort on 'report every parked drone'. "
               "The bulk of the ground rows (before first take-off and never-airborne flights) cannot be reached from airborne tracks, only from "
               "sensors that see the parked drone (see section 2).")
    else:
        rec = (f"Implement `{zq[0]}` (X = {zq[1]:g} m, T = {tl(zq[2])}, R = {R_M:g} m) behind a switch; held-out expected saving "
               f"{pct(ev_q / max_gain)} of the recoverable cost at probability {Q_HEAD:g}. Treat as an upper bound (perfect track).")
    A("**Recommendation.** " + rec)
    A("")
    hz_h, hz_s = horizon[("hold", 0.8)], horizon[("start", 0.8)]
    hl_h, hl_s = horizon_loo[("hold", 0.8)], horizon_loo[("start", 0.8)]
    A(f"Row-by-row rule of thumb (section 4, descriptive only; it does not make the rule pay, see the held-out result): an added row is worth keeping "
      f"while its chance of hitting a parked drone is above 1 / (2 x p), where p is the probability that the official truth contains parked drones "
      f"({1 / (2 * 0.8):.3f} for p = 0.8, {1 / 2:.3f} for p = 1). Measured on testing (X = {hz_ref_x:g} m) on the pooled data that holds for `hold` rows up to "
      f"{tl(hz_h) if hz_h else 'no horizon'} after the track ended and for `start` rows up to {tl(hz_s) if hz_s else 'no horizon'} before the track starts "
      f"(p = 0.8). The margins are thin (table 'margin of the horizon claim' in section 4): requiring the bin to clear the threshold also with any one "
      f"iteration left out, the horizons are {tl(hl_h) if hl_h else 'none'} for `hold` and {tl(hl_s) if hl_s else 'none'} for `start`. "
      f"Treat the conservative values as the usable ones.")
    A("")
    A("## 1. Share of the truth on the ground")
    A("")
    A("Columns 4-9 are shares of ALL truth rows of the iteration (drone-seconds in brackets).")
    A("")
    L += table(["iteration", "truth rows", "drone-seconds", "airborne", "on the ground", *PHASES], q1_rows)
    A(f"Ground rows inside short ground runs (< {SHORT_GAP_S:g} s between rows of the other state in the same flight, mostly "
      f"height flicker around the 5 m threshold): {short_rows:,} ({pct(short_rows / n_ground)} of ground rows).")
    A("")
    A("Time view (a step is one 100 ms instant): how often the picture contains a ground drone at all.")
    A("")
    L += table(["iteration", "steps", "steps with >= 1 ground drone", "steps with ground drones only",
                "steps with no truth drone", "max ground drones", "mean ground drones"], step_rows)
    A("Per flight (seconds). 'ground moving' = share of the flight's ground rows with horizontal speed above 1 m/s "
      "(a vehicle, a hand-carried drone or taxiing).")
    A("")
    L += table(["iteration", "drone", "side", "total s", "airborne s", "before 1st take-off s", "between s", "after landing s",
                "never airborne s", "ground moving", "max height m"], flight_rows)
    A("## 2. Which sensors have a report near the parked drones")
    A("")
    A(f"Window: {GATE_M:g} m horizontally and +-{WIN_S:g} s, raw report time. Bearing-only sensors "
      "(WVFX EMXG FGIX NJQL ORFH YXAB OEPS GTYT) cannot be tested by a distance and are excluded. Each cell is the share of the pooled "
      "ground rows (all six iterations) covered by that sensor in that report group; the last column shows the same sensor's "
      "usable groups against AIRBORNE rows for comparison. Groups are disjoint: self_report (within 5 m of its own sensor) first, "
      "then the class labels Home / UAV Transmitter / EM Transmitter, then far (> 15 km), else regular.")
    A("")
    L += table(["sensor", "status", *GROUPS, "any group", "usable vs airborne rows"], q2_sensor)
    A("Per sensor and per iteration (usable groups; cells: ground rows covered / airborne rows covered; 'no reports' = the sensor has no position "
      "rows in that iteration).")
    A("")
    L += table(["sensor", *keys], q2_sens_it)
    A("Unions of sensors: share of ground rows by phase, then all ground rows and, for comparison, all airborne rows.")
    A("")
    L += table(["union", *PHASES, "all ground rows", "all airborne rows"], q2_union)
    A("Per iteration, ground rows / airborne rows covered.")
    A("")
    L += table(["iteration", "deployable, usable [per brief]", "deployable w/o relays, usable [honest]",
                "deployable w/o relays + self_report", "all sensors, usable"], q2_iter)
    A("Per flight, honest union (deployable without relays, usable groups).")
    A("")
    L += table(["iteration", "drone", "side", "ground s", "ground rows covered", "airborne rows covered"], q2_flight)
    A("Coverage against the distance from the parked drone to the nearest deployable sensor site (honest union; last column adds self_report rows).")
    A("")
    L += table(["distance to nearest sensor", "ground seconds", "share of ground rows", "covered (usable)", "covered (+ self_report)"], q2_dist)
    A(f"Check: far-group rows within the gate of any truth row: {far_hits} (must be 0).")
    A("")
    A("## 3. Where the parked drones are")
    A("")
    A(f"Distance from each ground row to the nearest deployable sensor with a known site (metres; percentiles over ground rows; "
      f"last column = share within {SELF_M:g} m, the self_report zone that cleaning removes).")
    A("")
    L += table(["iteration", "ground s", "p5", "p25", "p50", "p75", "p95", "max", f"< {SELF_M:g} m"], q3_dist_rows)
    A(f"Parking spots (ground segments of >= {MIN_SEG_S:g} s, single-linkage clusters at {SPOT_M:g} m).")
    A("")
    L += table(["iteration", "segments", "flights", "spots", "max flights per spot", "max parked at once at one spot",
                "segment rows at multi-flight spots", "median m to nearest sensor"], spot_rows)
    A(f"Pooled: {int(tp[:, 0].sum())} segments form {int(tp[:, 1].sum())} spots; {pct(tp[:, 2].sum() / tp[:, 3].sum(), 0)} of the "
      "long-segment rows sit at a spot shared by 2 or more flights.")
    A("")
    A("Take-off predictability. 'first-air vs parked' = horizontal distance from the flight's first airborne truth position to the median "
      "position of its pre-take-off ground rows; 'vs last pre row' = to the last ground row before it; 'dz' = first airborne u minus median "
      "parked u; 'landing vs parked' = last airborne position to the median of the post-landing rows. Only flights with >= 5 s airborne are "
      "listed ('-' = fewer than 5 s on the ground in that phase); the ground vehicle (Skyranger) is left out of the percentiles.")
    A("")
    L += table(["iteration", "drone", "side", "pre-take-off s", "first-air vs parked m", "vs last pre row m", "dz m", "post-landing s",
                "landing vs parked m", "note"], to_rows)
    A("Percentiles over flights (metres).")
    A("")
    L += table(["measure", "n", "p5", "p25", "p50", "p75", "p95", "max"],
               [["first airborne vs parked position", len(to_stats["first"]), *pctiles(np.array(to_stats["first"]))],
                ["first airborne vs last pre-take-off row", len(to_stats["last"]), *pctiles(np.array(to_stats["last"]))],
                ["abs height difference, first airborne vs parked", len(to_stats["dz"]), *pctiles(np.array(to_stats["dz"]))],
                ["last airborne vs parked position after landing", len(to_stats["land"]), *pctiles(np.array(to_stats["land"]))],
                ["parked before take-off vs parked after landing (same flight)", len(ret), *pctiles(np.array(ret))]])
    A("## 4. Parked-drone rules, scored with GOSPA")
    A("")
    A("Ground height reference (GT-free): the median `sensor_u` of the deployable located sensors with a non-zero site height, or the "
      "`sensor_u` of the nearest such sensor. Error against the truth's ground height (reference minus the median pre-take-off `u` of "
      f"each flight): RMS median-reference {rms['median']:.1f} m, nearest-sensor {rms['nearest']:.1f} m. **Used: `{ref_mode}`.** "
      "Table: number of deployable located sensors, the median-reference height (m, ellipsoidal), and 'median error / worst error' (m) of the two references.")
    A("")
    L += table(["iteration", "located sensors", "median-reference u", "median ref: median / worst abs error",
                "nearest ref: median / worst abs error"], ref_rows)
    A(f"Track segments of the stand-in track (airborne runs >= {MIN_AIR_S:g} s) and how many end / start 'low' (u - u_ref <= X) for X = "
      + ", ".join(f"{x:g}" for x in X_GRID) + " m (cells: ends / starts).")
    A("")
    L += table(["iteration", "segments", *[f"X = {x:g} m" for x in X_GRID]], seg_end_rows)
    A("How to read the numbers. **gain if truth has parked drones** = cost units saved against 'all' truth (already net of the false rows the "
      "rule adds); **cost if truth is airborne only** = extra cost against 'airborne' truth (every added row is false there); **expected saving** "
      "= p x gain - (1 - p) x cost for an assumed probability p that the official truth contains parked drones. "
      f"Recoverable maximum: {max_gain:,.0f} units ({n_ground:,} ground rows x 1250). X = how close to the ground a track must end / start, "
      "T = how long a hold / early start may last. The full list of variants is in Appendix A.")
    A("")
    A(f"Best rule for each assumed probability p (all {len(cand)} variants searched, pooled over the six iterations). **In-sample, optimistic**: the "
      "rule is picked on the same data it is scored on, so these gains are inflated by selection; the decision basis is the held-out table below.")
    A("")
    L += table(["p", "rule", "X m", "T", "rows added", "on a truth ground row", "false rows", "gain if truth has parked drones",
                "cost if truth is airborne only", "expected saving", "share of recoverable"], best_q_rows)
    A("Sensitivity: the same search with the TRUE ground height of each flight as the reference (not available to a real tracker). This shows how "
      "much the poor sensor-site heights (see the reference table) cost:")
    A("")
    L += table(["p", "rule", "X m", "T", "rows added", "on a truth ground row", "false rows", "gain if truth has parked drones",
                "cost if truth is airborne only", "expected saving", "share of recoverable"], best_or_rows)
    A(f"Hit rate of an added row by its distance in time from the track (pooled, X = {hz_ref_x:g} m, no time limit). `hold` rows are counted "
      "from the moment the track ended, `start` rows backwards from the moment the track starts. 'net if p = 1' = (hits - false rows) x 1250.")
    A("")
    L += table(["rule", "time from track", "rows", "hit a parked drone", "hit rate", "net if p = 1"], hz_rows)
    A("Why added rows are false (same variants, no time limit): the rule's own drone has no Dronetag record at that time (not yet switched on, "
      "or already switched off), or it has one but is not within 50 m of the reported position (it was moved, or another drone took the slot).")
    A("")
    L += table(["rule", "rows added", "hit a parked drone", "false: own drone not in the truth yet / any more",
                "false: own drone in the truth but elsewhere"], why_rows)
    A(f"Per iteration for the headline rule (`{best[0]}`, X = {best[1]:g}, T = {tl(best[2])}, p = {Q_HEAD:g}):")
    A("")
    L += table(["iteration", "ground rows", "rows added", "on truth ground", "false", "gain (all truth)", "cost (airborne truth)",
                f"expected saving (p = {Q_HEAD:g})", "recoverable max"], per_iter_best)
    mk, ak = ab["matched_kind"], ab["added_kind"]
    A(f"Of the headline rule's rows: hold matched {mk[0]:,} of {ak[0]:,} added, early start matched {mk[1]:,} of {ak[1]:,}; "
      f"{ab['n_swap']:,} added rows won the partner slot of an airborne truth row instead. {ab['matched_long']:,} of the {ab['n_matched']:,} covered "
      f"rows are in ground runs of >= {SHORT_GAP_S:g} s (the rest sit in short gaps that a tracker coasting through a brief height flicker would cover anyway).")
    A("")
    A(f"**Leave-one-iteration-out (held-out) check, the decision basis.** For each held-out iteration the ground-height reference (`median` or `nearest`) "
      f"and the (rule, X, T) maximising the expected saving at p = {Q_HEAD:g} are chosen on the other five iterations only, then applied to the held-out one "
      f"(the in-sample sections above choose the reference `{ref_mode}` on all six; the per-fold choice is listed). "
      "'expected saving' = p x gain - (1 - p) x cost for that iteration.")
    A("")
    L += table(["held-out iteration", "reference mode", "choice", "gain (all truth)", "cost (airborne truth)", f"expected saving (p = {Q_HEAD:g})",
                "matched / false rows"], loio_rows)
    A("Pooled held-out result for each assumed probability p (rule re-picked on the other five for every p; the last column counts held-out iterations that lose):")
    A("")
    L += table(["p", "pooled gain (all truth)", "pooled cost (airborne truth)", "expected saving", "share of recoverable", "iterations losing"], loio_q_rows)
    A(f"Margin of the horizon claim: hit rate of added rows per distance-in-time bin (X = {hz_ref_x:g} m, no time limit); a bin pays at p = 0.8 only above "
      f"{1 / 1.6:.3f} and at p = 1 only above 0.500. 'leave-one-out range' = min - max of the pooled hit rate when one iteration is dropped; "
      "'per iteration' = the hit rate in each iteration in table order.")
    A("")
    L += table(["rule", "time from track", "pooled hit rate", "leave-one-out range", "per iteration (" + ", ".join(k.split("/")[1] for k in keys) + ")"], margin_rows)
    A("Caveats: the stand-in perfect track overstates the gain (a real tracker has gaps, identity switches and detection delays, and may not "
      "follow the drone down to the last seconds before landing); the cost units assume the official GOSPA uses c = 50, p = 2 in 3-D "
      "(score.PRIMARY; other c / p only rescale); ground reference heights come from sensor-site metadata of mixed quality (see the reference table).")
    A("")
    A("## Appendix A. All rule variants")
    A("")
    A("Pooled over the six testing iterations; break-even = the probability p above which the rule pays off in expectation "
      "(never = it loses even when parked drones are certainly scored).")
    A("")
    L += table(["rule", "X m", "T", "rows added", "on a truth ground row", "false rows", "ground rows covered",
                "gain if truth has parked drones", "cost if truth is airborne only", "break-even probability"], q4_rows)

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    runtime = time.time() - t_start
    check("9. runtime < 10 min", runtime < 600, f"{runtime:.0f} s")
    if fails:
        print("SELF-CHECKS FAILED: " + "; ".join(fails))
        return 1
    print("ALL SELF-CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
