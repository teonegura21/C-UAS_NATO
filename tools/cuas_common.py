"""Shared discovery, parsing and geometry for the C-UAS challenge data.

Every tool in ``tools/`` imports this module so that all of them agree on:

* how an iteration folder is named (``Iteration.key``),
* how sensor / Dronetag file names are parsed,
* how raw cells become numbers (timestamps, radians vs degrees, Python literals),
* the local East-North-Up frame used for all geometry.

Conventions verified against the data (see CLAUDE.md "Sensor CSV gotchas"):

* ``*_coordinateSystem`` is per row: ``LAT_LNG_DEG_M`` (degrees) or ``LAT_LNG_RAD_M``
  (radians, the EGOI radar). ``x`` is longitude and ``y`` latitude in both.
* ``RANGE_BEARING_COORDINATE_SYSTEM_DEGREES_M``: azimuth / elevation are DEGREES
  (README_Sensor.pdf says radians; the data and the enum say degrees). Azimuth is
  clockwise from north; ``RANGE_BEARING_DATUM_MAGNETIC`` (FGIX only) is from
  magnetic north, which we cannot correct without the real location.
* ``*_datum``: ``WGS84_E`` = height above the ellipsoid, ``WGS84_G`` = above the
  geoid (MSL). The geoid offset is unknown here (longitudes are anonymized), so
  heights are kept as reported and the datum is carried alongside.
* Longitudes are shifted by a secret constant. A constant longitude shift is a
  rotation about the polar axis, so every local ENU quantity below is exact.
"""

from __future__ import annotations

import ast
import csv
import math
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
BASE_DERIVED = DATA / "derived"     # the main derived tree (F0 and every baseline live here)
# CUAS_DERIVED=<dir> redirects every tool's derived outputs to a VARIANT tree (run.py --derived),
# so a rebuild with corrections never overwrites the main tree. Unset = the main tree.
DERIVED = Path(os.environ["CUAS_DERIVED"]).resolve() if os.environ.get("CUAS_DERIVED") else BASE_DERIVED
TREES = {"testing": DATA / "Anon_Testing", "evaluation": DATA / "Anon_Evaluation"}

csv.field_size_limit(1 << 30)

# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Iteration:
    tree: str        # 'testing' | 'evaluation'
    dataset: str     # '' for testing, 'I' | 'II' | 'III' for evaluation
    scenario: int
    iteration: int
    path: Path

    @property
    def key(self) -> str:
        """Stable id used in every derived table, e.g. 'testing/S1-I1', 'evaluation/DIII/S2-I3'."""
        middle = f"D{self.dataset}/" if self.dataset else ""
        return f"{self.tree}/{middle}S{self.scenario}-I{self.iteration}"

    @property
    def has_groundtruth(self) -> bool:
        return self.tree == "testing"


_SCEN = re.compile(r"Scenario (\d+)$")
_ITER = re.compile(r"Iteration (\d+)$")
_DSET = re.compile(r"Dataset (III|II|I)$")


def iterations() -> list[Iteration]:
    """All scenario/iteration folders of both trees, sorted by key."""
    found: list[Iteration] = []
    for tree, root in TREES.items():
        for it_dir in root.rglob("Iteration *"):
            if not it_dir.is_dir() or not (it_dir / "SensorData").is_dir():
                continue
            scen = _SCEN.match(it_dir.parent.name)
            num = _ITER.match(it_dir.name)
            dset = _DSET.match(it_dir.parent.parent.name)
            found.append(Iteration(tree, dset.group(1) if dset else "",
                                   int(scen.group(1)), int(num.group(1)), it_dir))
    return sorted(found, key=lambda it: it.key)


# <Team>_<Id>_<Type>_<YYMMDD>_<HHMM>_<HHMM>.csv ; <Type> may contain '-' but not '_'.
_SENSOR_FILE = re.compile(
    r"^(?P<team>[^_]+)_(?P<id>[^_]+)_(?P<type>[^_]+)_(?P<date>\d{6})_(?P<start>\d{4})_(?P<end>\d{4})$")
# DroneTag_<BLUE|RED>_<YYMMDD>_<HHMM>_<HHMM>_<DroneName>.csv ; DroneName may contain spaces.
_GT_FILE = re.compile(
    r"^DroneTag_(?P<side>BLUE|RED)_(?P<date>\d{6})_(?P<start>\d{4})_(?P<end>\d{4})_(?P<name>.+)$")


@dataclass(frozen=True)
class SensorFile:
    path: Path
    team: str          # operator: Alpha..Echo (NOT the sensor)
    sensor_id: str     # 4-letter anonymized sensor id, e.g. 'EGOI'
    sensor_type: str   # e.g. 'RADAR', 'PASSIVE-RF--JAMMER--CYBER'

    @property
    def sensor_key(self) -> str:
        return f"{self.sensor_id}_{self.sensor_type}"


@dataclass(frozen=True)
class GroundTruthFile:
    path: Path
    side: str          # 'BLUE' (friendly) | 'RED' (hostile)
    drone_name: str    # e.g. 'Skydio X10D'

    @property
    def flight_key(self) -> str:
        """Unique per file inside an iteration (several flights can share a drone name)."""
        return self.path.stem


# Browser re-downloads ("name (1).csv"). Both such files in the data are byte-identical
# to their sibling and would double-count every detection.
_COPY_SUFFIX = re.compile(r" \(\d+\)$")


def sensor_files(it: Iteration) -> list[SensorFile]:
    out = []
    for p in sorted((it.path / "SensorData").glob("*.csv")):
        stem = _COPY_SUFFIX.sub("", p.stem)
        if stem != p.stem:
            original = p.with_name(stem + p.suffix)
            if original.exists() and original.read_bytes() == p.read_bytes():
                continue
        m = _SENSOR_FILE.match(stem)
        if not m:
            raise ValueError(f"unexpected sensor file name: {p}")
        out.append(SensorFile(p, m["team"], m["id"], m["type"]))
    return out


def groundtruth_files(it: Iteration) -> list[GroundTruthFile]:
    """Dronetag files (testing tree only; evaluation has only Friendly(Blue)UAS.xlsx)."""
    out = []
    for p in sorted((it.path / "FriendDronesList").glob("DroneTag_*.csv")):
        m = _GT_FILE.match(p.stem)
        if not m:
            raise ValueError(f"unexpected Dronetag file name: {p}")
        out.append(GroundTruthFile(p, m["side"], m["name"]))
    return out


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


@dataclass
class Table:
    """A CSV as column-name -> list[str]. The unnamed index column is kept as 'row'."""

    path: Path
    columns: dict[str, list[str]]
    n_rows: int
    ragged_rows: int   # rows whose field count != header; skipped (a few CEQX rows)

    def get(self, name: str) -> list[str] | None:
        return self.columns.get(name)


def _rows(reader, width: int) -> Iterator[list[str] | None]:
    """Data rows of a sensor CSV; None for a ragged row (field count != header)."""
    for row in reader:
        if len(row) == 1 and width > 1:
            # One CEQX file (evaluation DII) wraps every data line whole in a pair of
            # quotes; the unquoted text is a normal CSV row.
            inner = next(csv.reader([row[0]]), [])
            if len(inner) == width:
                row = inner
        yield row if len(row) == width else None


def iter_rows(path: Path) -> tuple[list[str], Iterator[list[str] | None]]:
    """(header, rows) for scripts that stream a sensor CSV themselves; a ragged row is None.
    Use this rather than a bare csv.reader so every tool keeps the same rows."""
    fh = path.open(newline="", encoding="utf-8")
    reader = csv.reader(fh)
    header = next(reader)

    def gen():
        with fh:
            yield from _rows(reader, len(header))
    return header, gen()


def read_table(path: Path) -> Table:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        header = ["row" if (i == 0 and not h) else h for i, h in enumerate(header)]
        cols: dict[str, list[str]] = {h: [] for h in header}
        lists = [cols[h] for h in header]
        n = ragged = 0
        for row in _rows(reader, len(header)):
            if row is None:
                ragged += 1
                continue
            for lst, cell in zip(lists, row):
                lst.append(cell)
            n += 1
    return Table(path, cols, n, ragged)


def to_float(cells: list[str] | None, n: int) -> np.ndarray:
    """Column -> float64 array; blanks / non-numbers -> NaN. Missing column -> all NaN."""
    out = np.full(n, np.nan)
    if cells is None:
        return out
    for i, c in enumerate(cells):
        if c:
            try:
                out[i] = float(c)
            except ValueError:
                pass
    return out


def parse_times(cells: list[str]) -> np.ndarray:
    """ISO-8601 strings (naive UTC, optional trailing 'Z') -> float unix seconds."""
    arr = np.array([c.rstrip("Z") for c in cells], dtype="datetime64[us]")
    return arr.astype("int64") / 1e6


def parse_literal(cell: str):
    """Nested fields are Python literals ("[{'type': ...}]"), not JSON. None if blank/bad."""
    if not cell or cell[0] not in "[{(":
        return None
    try:
        return ast.literal_eval(cell)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None


def top_class(cell: str) -> tuple[str, float, str, float]:
    """(type, confidence, subtype, sub_confidence) of the highest-confidence entry.

    Missing parts are '' / NaN. Case is normalized to the spec's capitalisation by
    `norm_class` (data mixes 'Air vehicle' / 'Air Vehicle', 'UAV rotary wing' / ...).
    """
    obj = parse_literal(cell)
    if not isinstance(obj, list) or not obj:
        return "", math.nan, "", math.nan
    entries = [d for d in obj if isinstance(d, dict)]
    if not entries:
        return "", math.nan, "", math.nan
    # Rank by the entry's own confidence, else its best subclass confidence (EGOI / IUKL
    # often give confidence only on the subclass). max() with NaN keys would silently
    # return the first entry, so missing scores rank lowest instead.
    best = max(entries, key=lambda d: _rank(d))
    sub = [d for d in (best.get("subClass") or []) if isinstance(d, dict)]
    sub_best = max(sub, key=lambda d: _rank(d), default={})
    return (norm_class(best.get("type", "")), _conf(best.get("confidence")),
            norm_class(sub_best.get("type", "")), _conf(sub_best.get("confidence")))


def _conf(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _rank(entry: dict) -> float:
    own = _conf(entry.get("confidence"))
    if own == own:
        return own
    subs = [_conf(s.get("confidence")) for s in (entry.get("subClass") or []) if isinstance(s, dict)]
    subs = [s for s in subs if s == s]
    return max(subs) if subs else -1.0


_CLASS_CANON = {c.lower(): c for c in (
    "Air Vehicle", "UAV", "UAV Fixed Wing", "UAV Rotary Wing", "UAV Transmitter",
    "Manned Fixed Wing", "Manned Rotary Wing", "Animal", "Bird", "EM Transmitter",
    "FIXED_WING", "Fixed Object", "Home", "Land Vehicle", "Sea Vessel", "Other", "Unknown")}


def norm_class(name: str) -> str:
    name = str(name).strip()
    return _CLASS_CANON.get(name.lower(), name)


# --------------------------------------------------------------------------- #
# Geometry (WGS84). All angles in degrees unless the name says _rad.
# --------------------------------------------------------------------------- #

WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)


def latlon_deg(x, y, coordinate_system) -> tuple[np.ndarray, np.ndarray]:
    """SAPIENT (x=lon, y=lat) -> (lat_deg, lon_deg), honouring a per-row RAD/DEG system.

    `coordinate_system` is a scalar string or a sequence of strings (one per row).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cs = np.asarray(coordinate_system, dtype=object)
    is_rad = np.vectorize(lambda s: "_RAD_" in str(s), otypes=[bool])(cs)
    lat = np.where(is_rad, np.degrees(y), y)
    lon = np.where(is_rad, np.degrees(x), x)
    return lat, lon


def geodetic_to_ecef(lat_deg, lon_deg, h) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    h = np.asarray(h, dtype=float)
    sin_lat = np.sin(lat)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat**2)
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1 - WGS84_E2) + h) * sin_lat
    return x, y, z


@dataclass(frozen=True)
class Frame:
    """Local tangent ENU frame at (lat0, lon0, h0)."""

    lat0: float
    lon0: float
    h0: float = 0.0

    def _rot(self) -> np.ndarray:
        la, lo = math.radians(self.lat0), math.radians(self.lon0)
        return np.array([
            [-math.sin(lo), math.cos(lo), 0.0],
            [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
            [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)],
        ])

    def to_enu(self, lat_deg, lon_deg, h) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, z = geodetic_to_ecef(lat_deg, lon_deg, h)
        x0, y0, z0 = geodetic_to_ecef(self.lat0, self.lon0, self.h0)
        d = np.stack([np.asarray(x) - x0, np.asarray(y) - y0, np.asarray(z) - z0])
        e, n, u = np.tensordot(self._rot(), d, axes=1)
        return e, n, u

    def to_geodetic(self, e, n, u) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Inverse of to_enu (Bowring, sub-millimetre at these ranges)."""
        x0, y0, z0 = geodetic_to_ecef(self.lat0, self.lon0, self.h0)
        enu = np.stack([np.asarray(e, float), np.asarray(n, float), np.asarray(u, float)])
        x, y, z = np.tensordot(self._rot().T, enu, axes=1)
        x, y, z = x + x0, y + y0, z + z0
        b = WGS84_A * (1 - WGS84_F)
        ep2 = (WGS84_A**2 - b**2) / b**2
        p = np.hypot(x, y)
        th = np.arctan2(z * WGS84_A, p * b)
        lat = np.arctan2(z + ep2 * b * np.sin(th) ** 3, p - WGS84_E2 * WGS84_A * np.cos(th) ** 3)
        lon = np.arctan2(y, x)
        nrad = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - nrad
        return np.degrees(lat), np.degrees(lon), h


def bearing_offset(az_deg, el_deg, range_m) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Range/azimuth/elevation (az clockwise from north) -> ENU offset from the sensor.

    NaN elevation is treated as 0 (horizontal). Pass range_m=1 for a unit direction.
    """
    az = np.radians(np.asarray(az_deg, float))
    el = np.radians(np.nan_to_num(np.asarray(el_deg, float), nan=0.0))
    r = np.asarray(range_m, float)
    return r * np.cos(el) * np.sin(az), r * np.cos(el) * np.cos(az), r * np.sin(el)


def azimuth_deg(de, dn) -> np.ndarray:
    """Clockwise-from-north azimuth in [0, 360) of an ENU offset."""
    return np.degrees(np.arctan2(de, dn)) % 360.0


def angle_diff_deg(a, b) -> np.ndarray:
    """Smallest signed difference a-b in degrees, in (-180, 180]."""
    d = (np.asarray(a, float) - np.asarray(b, float) + 180.0) % 360.0 - 180.0
    return np.where(d == -180.0, 180.0, d)


def sensor_position_valid(lat, lon) -> np.ndarray:
    """False for the 0,0 placeholder (QTVS / VKYY are UTM services with no sensor site)."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    return np.isfinite(lat) & np.isfinite(lon) & ~((lat == 0) & (lon == 0))


def iteration_frame(it: Iteration) -> Frame:
    """THE local frame of an iteration: median valid sensor lat / lon over every detection
    row of every sensor file in it, at h0 = 0 (so `u` is height above the ellipsoid, up
    to curvature: < 1 m within 3.5 km)."""
    lats: list[float] = []
    lons: list[float] = []
    for sf in sensor_files(it):
        with sf.path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            ix, iy, ic = (header.index(c) for c in (
                "sensor_location_x", "sensor_location_y", "sensor_location_coordinateSystem"))
            xs, ys, cs = [], [], []
            for row in _rows(reader, len(header)):
                if row is None:
                    continue
                try:
                    xs.append(float(row[ix]))
                    ys.append(float(row[iy]))
                except ValueError:
                    continue
                cs.append(row[ic])
        lat, lon = latlon_deg(xs, ys, cs)
        ok = sensor_position_valid(lat, lon)
        lats.extend(lat[ok].tolist())
        lons.extend(lon[ok].tolist())
    return Frame(float(np.median(lats)), float(np.median(lons)), 0.0)


def iter_tables(it: Iteration) -> Iterator[tuple[SensorFile, Table]]:
    for sf in sensor_files(it):
        yield sf, read_table(sf.path)


# --------------------------------------------------------------------------- #
# Submission template: the 100 ms scoring grid and the dropdown vocabularies
# --------------------------------------------------------------------------- #

SUBMISSION_COLUMNS = (
    "timestamp", "unix_timestamp", "latitude", "longitude", "altitude_m",
    "C_nn", "C_ne", "C_nu", "C_ee", "C_eu", "C_uu", "track_id", "classification",
    "classification_confidence", "identity", "uav_model/manufacturer", "protocol",
    "serial_number", "signal_frequency_band",
)

# README_SubmissionTemplate.pdf Tables C.2-C.4, copied exactly: the templates carry no
# data validations, so these are the only source. Keep the typos ('DJI Ocusyinc') and
# the EN DASH (U+2013) in the bands; the rules text writes 'Not applicable', the
# dropdowns 'Not Applicable' (use the dropdown spelling).
DROPDOWNS: dict[str, tuple[str, ...]] = {
    "classification": ("Fixed Wing UAS", "Rotary Wing UAS", "UAS Transmitter",
                       "Unknown", "Not Applicable"),
    "identity": ("Hostile", "Friendly", "Unknown", "Not Applicable"),
    "uav_model/manufacturer": ("DJI Mavic 3", "DJI Mavic 3 Enterprise", "MTRC 300R",
                               "Autel Drone", "Parrot Anafi", "FPV", "FPV Chimera",
                               "Unknown", "Not Applicable"),
    "protocol": ("DJI Ocusyinc", "Autel Skylink", "Wifi", "Video",
                 "Digital video link type 2", "DIY Crossfire", "Unknown", "Not Applicable"),
    "serial_number": ("F45TB21BR1AE03FP", "163DG1700154R3", "F5FHD233M00DEYG6",
                      "1ZNBJAB00C00MQ", "1748FEV2HL7824051124", "Unknown", "Not Applicable"),
    "signal_frequency_band": ("2400 – 2483.5", "5725 – 5850", "902 – 928",
                              "5150 – 5250", "2200 – 2500", "Unknown",
                              "Not Applicable"),
}
METADATA_COLUMNS = ("uav_model/manufacturer", "protocol", "serial_number",
                    "signal_frequency_band")


def iter_slug(key: str) -> str:
    """Iteration key -> file-name stem, e.g. 'evaluation/DII/S2-I1' -> 'evaluation__DII__S2-I1'."""
    return key.replace("/", "__")


def deployable_sensor_ids() -> frozenset[str]:
    """Sensor ids present in at least one evaluation iteration (from file names). Testing-only
    sensors (EUIG, LJJX, MMJJ, NJQL as of 2026-10-28) must not carry a method's local score:
    LJJX relays the Blue drones' own positions in testing and does not exist in evaluation."""
    return frozenset(sf.sensor_id for it in iterations() if it.tree == "evaluation"
                     for sf in sensor_files(it))


SUBMISSIONS = DERIVED / "submissions"


def submission_path(method: str, it: Iteration) -> Path:
    """Where a method's file for an iteration lives: submissions/<method>/<slug>/<template name>
    (the file name must stay the template's; the folder keeps iterations apart)."""
    return SUBMISSIONS / method / iter_slug(it.key) / template_path(it).name


def template_path(it: Iteration) -> Path:
    """The iteration's ResultTemplate/scenario_<N>_iteration_<M>.xlsx (the submission name)."""
    return it.path / "ResultTemplate" / f"scenario_{it.scenario}_iteration_{it.iteration}.xlsx"


def template_grid(it: Iteration) -> tuple[list[str], np.ndarray]:
    """(timestamp strings, unix seconds) of the template's pre-filled 100 ms grid, in file
    order. The strings are returned exactly as stored; `unix_timestamp` is SECONDS (the
    PDF says ms) and is checked against the strings to 1 ms."""
    import openpyxl
    wb = openpyxl.load_workbook(template_path(it), read_only=True)
    try:
        rows = wb.active.iter_rows(values_only=True)
        header = next(rows)
        if tuple(header[:len(SUBMISSION_COLUMNS)]) != SUBMISSION_COLUMNS:
            raise ValueError(f"unexpected template header: {header}")
        iso, unix = [], []
        for r in rows:
            if r[0] is None:
                continue
            iso.append(str(r[0]))
            unix.append(float(r[1]))
    finally:
        wb.close()
    unix_arr = np.array(unix)
    if np.abs(parse_times(iso) - unix_arr).max() > 1e-3:
        raise ValueError("template unix_timestamp disagrees with timestamp")
    return iso, unix_arr


# --------------------------------------------------------------------------- #
# Derived tables (data/derived/*.csv[.gz]): column dict <-> file
# --------------------------------------------------------------------------- #


def _open_text(path: Path, mode: str):
    import gzip
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", newline="", encoding="utf-8")
    return path.open(mode, newline="", encoding="utf-8")


def write_columns(path: Path, columns: dict[str, np.ndarray | list], order: list[str]) -> None:
    """Write equal-length columns in `order`. NaN floats are written as empty cells."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(columns[order[0]])
    cols = []
    for name in order:
        col = columns[name]
        if len(col) != n:
            raise ValueError(f"column {name!r} has {len(col)} rows, expected {n}")
        if isinstance(col, np.ndarray) and col.dtype.kind == "f":
            col = ["" if v != v else repr(float(v)) for v in col.tolist()]
        cols.append(col)
    with _open_text(path, "w") as fh:
        w = csv.writer(fh)
        w.writerow(order)
        w.writerows(zip(*cols))


def read_columns(path: Path, numeric: set[str] | frozenset[str] = frozenset(),
                 usecols: list[str] | None = None) -> dict[str, np.ndarray]:
    """Read a derived table. Columns in `numeric` -> float64 (blank = NaN); others -> object.
    `usecols` limits which columns are kept (detections.csv.gz is 758k x 58)."""
    with _open_text(path, "r") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        keep = list(range(len(header))) if usecols is None else [header.index(c) for c in usecols]
        raw: list[list[str]] = [[] for _ in keep]
        for row in reader:
            for lst, i in zip(raw, keep):
                lst.append(row[i])
    out: dict[str, np.ndarray] = {}
    for name, cells in zip([header[i] for i in keep], raw):
        if name in numeric:
            out[name] = np.array([float(c) if c else np.nan for c in cells], dtype=float)
        else:
            out[name] = np.array(cells, dtype=object)
    return out
