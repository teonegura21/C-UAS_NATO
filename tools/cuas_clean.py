"""Quality layer over data/derived/detections.csv.gz, shared by association and clustering.

detections.csv.gz is a faithful copy of the raw reports (verified field-by-field). The raw
reports contain placeholders and non-target rows; this module encodes, in ONE place, the
rules for what downstream analysis may treat as a measurement. Each rule was found by the
physics verification against Dronetag ground truth (2026-10-26):

* null_island     target lat = lon = 0 (2 ILYS rows) -> not a position.
* self_report     position rows within 5 m (horizontal) of the reporting sensor: IKJX 86 %,
                  MMJJ 73 %, STUG 13 % of rows are the device's own / home location, never
                  near a drone.
* velocity zeros  a sensor whose finite (ve, vn) is exactly (0, 0) on > 50 % of rows uses 0
                  as "missing" (VKYY, QTVS 100 %, STUG 83 %); those zeros -> NaN. A sensor
                  whose finite vu is always 0 (UXYJ) -> vu NaN.
* alt sentinel    alt == 0 exactly on a position row is "unknown altitude" (IKJX 91 %,
                  MMJJ 97 %, STUG 16 %, CEQX 13 %) -> alt / u NaN.
* height ref      NYTF reports height above ground in some iterations and above the
                  ellipsoid in others (sensor_z 2 m vs 79 m). `dz_sensor` = alt - sensor_alt
                  is consistent in both, so it is the height feature to pool across
                  iterations. NaN where sensor_alt is 0 (unknown) or the sensor has no site.
* gt_feed         VKYY, QTVS (UTM services) and LJJX (Blue-force relay) report the Dronetag
                  feed itself (sub-metre agreement). Legitimate inputs, but evaluating
                  against ground truth on them is label leakage.
* far             > 15 km horizontally from the iteration origin (EGOI airliners out to
                  ~200 km). Not a Class I UAS in this scenario; kept, but flagged.
* duplicate       an exact copy of an earlier row (same iteration, sensor, object, time,
                  position, bearing): 91,118 rows = 12 % (NYTF / LJJX streams recorded in two
                  teams' files, EGOI / NYTF repeating rows in one file). Flagged, excluded from
                  usable_pos (2026-10-28).

Sensor class labels are NOT cleaning rules: on testing 60 % of UXYJ 'Bird' rows and 47 % of
STUG 'Home' rows (drones parked on the ground) sit on a real drone, so labels are features.

Known per-sensor biases (time lag, azimuth offset, EMXG 170-degree flip in testing S3-I2/I3,
FGIX ~80 degrees off, YXAB / ORFH bearings uncorrelated with drones) are calibration, not
cleaning: they are estimated by tools/gt_associate.py and written to
data/derived/sensor_calibration.csv.
"""

from __future__ import annotations

import numpy as np

import cuas_common as C

DETECTIONS = C.DERIVED / "detections.csv.gz"
GROUNDTRUTH = C.DERIVED / "groundtruth.csv.gz"

DET_NUMERIC = frozenset({
    "row", "t", "sensor_lat", "sensor_lon", "sensor_alt", "sensor_pos_valid", "sensor_e",
    "sensor_n", "sensor_u", "lat", "lon", "alt", "e", "n", "u", "az_deg", "el_deg", "range_m",
    "az_err_deg", "el_err_deg", "err_x_m", "err_y_m", "err_z_m", "ve", "vn", "vu", "det_conf",
    "cls_conf", "cls_sub_conf", "rcs", "freq_mhz", "signal_amp", "n_assoc",
})
GT_NUMERIC = frozenset({
    "t", "lat", "lon", "geo_alt", "baro_alt", "height", "e", "n", "u", "ve", "vn", "vu",
    "heading", "h_acc", "v_acc", "speed_acc",
})

GT_FEED_SENSORS = frozenset({"VKYY", "QTVS", "LJJX"})
POSITION_KINDS = ("pos3d", "pos2d")
BEARING_KINDS = ("bearing", "bearing_el")
SELF_REPORT_M = 5.0
FAR_M = 15_000.0

# Columns `clean` needs on top of whatever the caller asks for.
_CLEAN_NEEDS = ["iter_key", "sensor_id", "sensor_key", "meas_kind", "lat", "lon", "alt", "e", "n", "u",
                "sensor_e", "sensor_n", "sensor_alt", "sensor_pos_valid", "ve", "vn", "vu",
                "object_id", "t", "az_deg", "el_deg", "file", "row"]


def load_detections(usecols: list[str] | None = None) -> dict[str, np.ndarray]:
    """Detections with numeric columns as float64. `usecols=None` loads everything (~3 GB)."""
    cols = None if usecols is None else list(dict.fromkeys(usecols))
    return C.read_columns(DETECTIONS, DET_NUMERIC, usecols=cols)


def load_groundtruth() -> dict[str, np.ndarray]:
    return C.read_columns(GROUNDTRUTH, GT_NUMERIC)


def load_clean(usecols: list[str] | None = None) -> dict[str, np.ndarray]:
    """load_detections(usecols + what cleaning needs), then clean() in place."""
    det = load_detections(None if usecols is None else list(usecols) + _CLEAN_NEEDS)
    clean(det)
    return det


def clean(det: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Add quality flags and cleaned columns to `det` (in place) and return it.

    Added: null_island, self_report, far, gt_feed, duplicate, usable_pos (bool); ve_c, vn_c,
    vu_c, alt_c, u_c, dz_sensor (float). Raw columns are left untouched. Pass whole iterations
    (duplicates are found among the rows passed).
    """
    duplicate = flag_duplicates(det)
    kind = det["meas_kind"]
    is_pos = np.isin(kind, POSITION_KINDS)
    lat, lon = det["lat"], det["lon"]
    null_island = is_pos & (lat == 0) & (lon == 0)

    sensor_ok = det["sensor_pos_valid"] == 1
    with np.errstate(invalid="ignore"):
        d_sensor = np.hypot(det["e"] - det["sensor_e"], det["n"] - det["sensor_n"])
        self_report = is_pos & sensor_ok & (d_sensor < SELF_REPORT_M)
        far = is_pos & (np.hypot(det["e"], det["n"]) > FAR_M)

    ve, vn, vu = det["ve"].copy(), det["vn"].copy(), det["vu"].copy()
    skey = det["sensor_key"]
    for key in np.unique(skey):
        m = skey == key
        fin = m & np.isfinite(ve) & np.isfinite(vn)
        zero = fin & (ve == 0) & (vn == 0)
        if fin.any() and zero.sum() > 0.5 * fin.sum():
            ve[zero] = np.nan
            vn[zero] = np.nan
        fin_u = m & np.isfinite(vu)
        if fin_u.any() and (vu[fin_u] == 0).all():
            vu[fin_u] = np.nan

    alt = det["alt"].copy()
    u = det["u"].copy()
    sentinel = is_pos & (alt == 0)
    alt[sentinel] = np.nan
    u[sentinel] = np.nan
    s_alt = det["sensor_alt"]
    with np.errstate(invalid="ignore"):
        dz = np.where(sensor_ok & (s_alt != 0) & np.isfinite(s_alt), alt - s_alt, np.nan)

    det.update(
        null_island=null_island, self_report=self_report, far=far, duplicate=duplicate,
        gt_feed=np.isin(det["sensor_id"], list(GT_FEED_SENSORS)),
        usable_pos=is_pos & ~null_island & ~self_report & ~duplicate
        & np.isfinite(det["e"]) & np.isfinite(det["n"]),
        ve_c=ve, vn_c=vn, vu_c=vu, alt_c=alt, u_c=u, dz_sensor=dz,
    )
    return det


DUP_NEEDS = ["iter_key", "sensor_key", "object_id", "t", "lat", "lon", "alt", "az_deg",
             "el_deg", "file", "row"]


def flag_duplicates(det: dict[str, np.ndarray]) -> np.ndarray:
    """True on every extra copy of an identical measurement: same iteration, sensor, object,
    time (1 ms), position (1e-7 deg, 1 cm) and bearing (1e-3 deg). The first copy in
    (file, row) order is kept.

    12 % of all rows (2026-10-28 audit): NYTF and LJJX streams recorded by two teams'
    files of the same iteration (cross-file copies), and EGOI / NYTF repeating rows inside
    one file. A copy adds no information but would double a measurement's weight in a filter,
    a count feature or a co-observation tally.
    """
    order = sorted(range(len(det["t"])), key=lambda i: (det["file"][i], det["row"][i]))

    def r(col: str, nd: int) -> list:
        return np.round(np.nan_to_num(det[col], nan=-9e9), nd).tolist()

    t, lat, lon, alt = r("t", 3), r("lat", 7), r("lon", 7), r("alt", 2)
    az, el = r("az_deg", 3), r("el_deg", 3)
    it, sk, ob = det["iter_key"], det["sensor_key"], det["object_id"]
    seen: set = set()
    dup = np.zeros(len(order), dtype=bool)
    for i in order:
        k = (it[i], sk[i], ob[i], t[i], lat[i], lon[i], alt[i], az[i], el[i])
        if k in seen:
            dup[i] = True
        else:
            seen.add(k)
    return dup


def interp_flight(gt: dict[str, np.ndarray], flight_mask: np.ndarray, t: np.ndarray,
                  cols=("e", "n", "u")) -> tuple[np.ndarray, ...]:
    """Ground-truth columns of one flight linearly interpolated at times `t`; NaN outside
    the flight's time span (no extrapolation)."""
    tg = gt["t"][flight_mask]
    order = np.argsort(tg, kind="stable")
    tg = tg[order]
    out = []
    for c in cols:
        v = gt[c][flight_mask][order]
        vi = np.interp(t, tg, v)
        vi[(t < tg[0]) | (t > tg[-1])] = np.nan
        out.append(vi)
    return tuple(out)
