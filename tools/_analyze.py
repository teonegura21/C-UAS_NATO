#!/usr/bin/env python3
"""Summarise the numeric columns of every CSV/XLSX file in a dataset tree.

Usage
-----
    python analyze_columns.py                          # uses ./"Evaluation datasets"
    python analyze_columns.py "path/to/data" -o out.txt

What it does
------------
1. Walks the dataset folder and finds every ``.csv`` and ``.xlsx`` file.
2. Reads each file into a *table* (one per CSV file, one per XLSX worksheet).
3. For every column that contains numbers it computes min / max / mean / median /
   count / number of distinct values, per table.
4. Merges those results into cross-dataset statistics (per column, per dataset
   group, per sensor type).
5. Writes everything to a plain-text report.

Design notes (the *why*)
------------------------
* Memory: numbers are kept in ``array('d')`` (8 bytes per value) instead of a list
  of 4-tuples of Python objects (~200+ bytes per value). Global statistics are
  merged incrementally from per-table summaries, so no value is stored twice.
* Robustness: one unreadable file must not abort a long batch run, so each file is
  processed inside an isolation boundary and failures are listed in the report.
* Correctness: every statistic is computed in the way its definition requires
  (true median, compensated summation, finite values only).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import math
import re
import sys
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

try:  # XLSX support is optional: CSV-only users should not need it installed.
    import openpyxl
except ImportError:  # pragma: no cover - depends on the environment
    openpyxl = None

LOG = logging.getLogger("analyze_columns")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_ROOT = "Evaluation datasets"
DEFAULT_OUTPUT = "csv_xlsx_columns.txt"

# utf-8-sig transparently strips the BOM that Excel puts at the start of "CSV UTF-8"
# files (otherwise the first header becomes '\ufeffname'). latin-1 can decode any
# byte sequence, so it is a safe last resort for legacy exports.
CSV_ENCODINGS = ("utf-8-sig", "latin-1")
CSV_DELIMITERS = ",;\t|"

# File names that are never real data: Excel lock files ("~$book.xlsx", which are not
# valid zip archives) and macOS AppleDouble resource forks ("._book.csv").
IGNORED_PREFIXES = ("~$", "._")

MIN_VALUES_FOR_STATS = 3      # below this, mean/median are not meaningful
MAX_LISTED_UNIQUES = 30       # list the distinct values when there are at most this many
SMALL_SENSOR_LIMIT = 50       # "small" = a sensor type with fewer values than this
SENSOR_ROUNDING = 3           # decimals used when listing a small sensor's values
DEVIATION_EPSILON = 1e-6      # ranges narrower than this get no percentage deviation

# Order matters: the first matching rule wins (e.g. "sensor_location_x_error" is an
# error value, not a position).
CATEGORY_RULES = (
    ("Error values", "error"),
    ("Sensor position", "sensor_location"),
    ("Object position", "location_"),
    ("Velocity metrics", "velocity"),
    ("Range/Bearing", "rangebearing"),
    ("Detection confidence", "detectionconfidence"),
)

# --------------------------------------------------------------------------- #
# Classification helpers
# --------------------------------------------------------------------------- #

# "Dataset I", "Dataset II" and "Dataset III" share a prefix, so a plain substring
# test ("Dataset I" in path) is TRUE for all three. The regex lists the longest
# numeral first and forbids a trailing alphanumeric, so each folder maps to exactly
# one group.
_GROUP_PATTERN = re.compile(r"Dataset[\s_-]*(III|II|I)(?![A-Za-z0-9])", re.IGNORECASE)
_GROUP_NAMES = {"I": "Dataset_I", "II": "Dataset_II", "III": "Dataset_III"}


def classify_group(relative_path: str) -> str:
    """Return the dataset group ('Dataset_I', 'Dataset_II', ...) of a relative path."""
    match = _GROUP_PATTERN.search(relative_path)
    return _GROUP_NAMES[match.group(1).upper()] if match else "Unknown"


# README_Sensor §1.1.1: <Team>_<Id>_<Type>_<YYMMDD>_<HHMM>_<HHMM>.csv, e.g.
# "Delta_IKJX_PASSIVE-RF--JAMMER--CYBER_240916_1129_1141". <Type> may itself contain
# '-' but not '_', and the three trailing numeric fields anchor the match.
_SENSOR_FILE = re.compile(
    r"^(?P<team>[^_]+)_(?P<id>[^_]+)_(?P<type>[^_]+)_\d{6}_\d{4}_\d{4}$")


def sensor_name(path: Path) -> str:
    """Sensor = '<Id>_<Type>' (e.g. 'STUG_CYBER'), not the team that operated it.

    The text before the first underscore is the TEAM. Grouping by team mixes physically
    different sensors (Delta runs EGOI, EMXG, IKJX, ORFH, ...) and splits one sensor
    across teams (STUG appears under Alpha, Delta and Echo). Files that do not follow
    the convention (template, friendly list) fall back to the first token.
    """
    match = _SENSOR_FILE.match(path.stem)
    if match:
        return f"{match['id']}_{match['type']}"
    prefix = path.name.split("_", 1)[0] if "_" in path.name else ""
    return prefix or "Other"


# --------------------------------------------------------------------------- #
# Cell parsing
# --------------------------------------------------------------------------- #


def is_blank(cell: object) -> bool:
    """True for None and for strings that are empty or whitespace-only."""
    return cell is None or (isinstance(cell, str) and not cell.strip())


def to_number(cell: object) -> float | None:
    """Convert a cell to a *finite* float, or return None if it is not one.

    * ``bool`` is rejected explicitly: in Python ``True`` is an ``int``, so an Excel
      TRUE/FALSE column would otherwise be summarised as 1/0.
    * ``nan`` and ``inf`` are rejected: ``float('inf')`` parses fine and equals itself,
      so the old ``fval == fval`` test only removed NaN and let infinities poison
      min/max/mean and the percentage arithmetic.
    """
    if isinstance(cell, bool):
        return None
    try:
        value = float(cell) if isinstance(cell, (int, float)) else float(str(cell).strip())
    except (ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def clean_header(raw_header: Iterable[object]) -> list[str]:
    """Strip header cells and make duplicate names unique (``x``, ``x__2``, ...).

    Two columns with the same name would otherwise silently overwrite each other in
    every dictionary keyed by column name.
    """
    seen: Counter = Counter()
    header: list[str] = []
    for cell in raw_header:
        name = "" if cell is None else str(cell).strip()
        if name:
            seen[name] += 1
            if seen[name] > 1:
                name = f"{name}__{seen[name]}"
        header.append(name)
    return header


# --------------------------------------------------------------------------- #
# Reading files
# --------------------------------------------------------------------------- #


@dataclass
class RawTable:
    """The numeric content of one table (CSV file or XLSX worksheet)."""

    name: str
    header: list[str]
    row_count: int
    values: dict[str, array]        # column -> finite numbers found in it
    rejected: dict[str, int]        # column -> non-blank cells that were not numbers


def ingest(name: str, rows: Iterable[Sequence[object]]) -> RawTable:
    """Consume an iterable of rows (first non-blank row = header) into a RawTable.

    Works for csv.reader (lists of str) and openpyxl (tuples of typed cells) alike.
    """
    row_iter = iter(rows)
    # Skip leading blank rows. Doing next() on an *empty* file used to raise
    # StopIteration and kill the whole script.
    header_row = next((r for r in row_iter if not all(is_blank(c) for c in r)), None)
    if header_row is None:
        return RawTable(name, [], 0, {}, {})

    header = clean_header(header_row)
    named = [(index, column) for index, column in enumerate(header) if column]
    values = {column: array("d") for _, column in named}
    rejected: Counter = Counter()
    row_count = 0

    for row in row_iter:
        if all(is_blank(cell) for cell in row):
            continue                      # blank lines are not data rows
        row_count += 1
        width = len(row)
        for index, column in named:
            if index >= width:            # short row: all later columns are missing too
                break
            cell = row[index]
            if is_blank(cell):
                continue
            number = to_number(cell)
            if number is None:
                rejected[column] += 1
            else:
                values[column].append(number)

    return RawTable(name, header, row_count, values, dict(rejected))


def guess_delimiter(header_line: str) -> str:
    """Pick the delimiter that occurs most often in the header line.

    Excel with a Romanian/European locale writes ';' instead of ','. With the default
    ',' such a file parses as ONE column and every statistic is silently lost.
    """
    best = max(CSV_DELIMITERS, key=header_line.count)   # ties -> first, i.e. ','
    return best if header_line.count(best) else ","


def read_csv(path: Path, display_name: str) -> RawTable:
    """Read one CSV file, trying each encoding in CSV_ENCODINGS in turn."""
    last_error: UnicodeDecodeError | None = None
    for encoding in CSV_ENCODINGS:
        try:
            # newline='' is what the csv module requires so that line endings inside
            # quoted fields are preserved and parsed correctly.
            with path.open("r", encoding=encoding, newline="") as handle:
                first_line = handle.readline()
                reader = csv.reader(
                    itertools.chain([first_line], handle),
                    delimiter=guess_delimiter(first_line),
                )
                table = ingest(display_name, reader)
        except UnicodeDecodeError as error:
            last_error = error
            continue
        if encoding != CSV_ENCODINGS[0]:
            LOG.warning("%s: not valid %s, decoded as %s", display_name, CSV_ENCODINGS[0], encoding)
        return table
    assert last_error is not None  # latin-1 never fails, so this is unreachable
    raise last_error


def read_xlsx(path: Path, display_name: str) -> Iterator[RawTable]:
    """Yield one RawTable per worksheet. Sheets are processed one at a time."""
    if openpyxl is None:
        raise RuntimeError("openpyxl is not installed (pip install openpyxl)")
    # read_only streams the sheet instead of loading it all; data_only returns the
    # last *calculated* value of formula cells instead of the formula text ('=A1+B1'),
    # which would otherwise be counted as non-numeric.
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        for sheet in workbook.worksheets:
            yield ingest(f"{display_name} [{sheet.title}]", sheet.iter_rows(values_only=True))
    finally:
        workbook.close()


def read_tables(path: Path, display_name: str) -> Iterator[RawTable]:
    if path.suffix.lower() == ".xlsx":
        yield from read_xlsx(path, display_name)
    else:
        yield read_csv(path, display_name)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


@dataclass
class ColumnStats:
    """Summary of one numeric column inside one table."""

    count: int
    rejected: int
    minimum: float
    maximum: float
    mean: float
    median: float
    nunique: int
    unique_values: list[float]      # filled only when nunique <= MAX_LISTED_UNIQUES


def summarise_column(values: array, rejected: int) -> ColumnStats:
    """Compute exact statistics. ``values`` must contain at least one number."""
    ordered = sorted(values)                       # one sort serves min, max, median, uniques
    count = len(ordered)
    middle = count // 2
    # True median: the old code returned ordered[n // 2], which for an even count is
    # the UPPER of the two middle values instead of their average.
    median = ordered[middle] if count % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
    # groupby on sorted data yields the distinct values in order, in O(n) and without
    # building a set of every float in the column.
    distinct = [value for value, _ in itertools.groupby(ordered)]
    return ColumnStats(
        count=count,
        rejected=rejected,
        minimum=ordered[0],
        maximum=ordered[-1],
        mean=math.fsum(ordered) / count,           # fsum avoids accumulated rounding error
        median=median,
        nunique=len(distinct),
        unique_values=distinct if len(distinct) <= MAX_LISTED_UNIQUES else [],
    )


@dataclass
class RangeAccumulator:
    """Running min / max / count that can absorb per-table summaries."""

    minimum: float = math.inf
    maximum: float = -math.inf
    count: int = 0

    def update(self, minimum: float, maximum: float, count: int) -> None:
        self.minimum = min(self.minimum, minimum)
        self.maximum = max(self.maximum, maximum)
        self.count += count


@dataclass
class SensorAccumulator:
    """Per-sensor count, plus the distinct values *only while the sensor is small*."""

    count: int = 0
    groups: set[str] = field(default_factory=set)
    uniques: set[float] = field(default_factory=set)

    def add(self, values: Sequence[float], group: str) -> None:
        # Once `count` reaches SMALL_SENSOR_LIMIT the sensor can never be "small"
        # again, so we stop collecting. Memory stays bounded by the limit.
        if self.count < SMALL_SENSOR_LIMIT:
            room = SMALL_SENSOR_LIMIT - self.count
            # "+ 0.0" turns -0.0 (from rounding a tiny negative) into 0.0.
            self.uniques.update(round(v, SENSOR_ROUNDING) + 0.0 for v in itertools.islice(values, room))
        self.count += len(values)
        self.groups.add(group)

    @property
    def is_small(self) -> bool:
        return self.count < SMALL_SENSOR_LIMIT


@dataclass
class ColumnAccumulator:
    """Everything the cross-dataset section needs to know about one column name."""

    overall: RangeAccumulator = field(default_factory=RangeAccumulator)
    by_group: dict[str, RangeAccumulator] = field(
        default_factory=lambda: defaultdict(RangeAccumulator))
    by_sensor: dict[str, SensorAccumulator] = field(
        default_factory=lambda: defaultdict(SensorAccumulator))
    tables: int = 0


@dataclass
class TableSummary:
    name: str
    group: str
    sensor: str
    row_count: int
    column_count: int
    columns: dict[str, ColumnStats]


class DatasetAnalysis:
    """Collects per-table summaries and the global accumulators."""

    def __init__(self) -> None:
        self.tables: list[TableSummary] = []
        self.columns: dict[str, ColumnAccumulator] = defaultdict(ColumnAccumulator)
        self.skipped: list[tuple[str, str]] = []

    def process_file(self, path: Path, root: Path) -> None:
        display_name = path.relative_to(root).as_posix()   # '/' on every OS
        group = classify_group(display_name)
        sensor = sensor_name(path)
        LOG.debug("reading %s (%s, %s)", display_name, group, sensor)
        try:
            for raw in read_tables(path, display_name):
                self._add_table(raw, group, sensor)
        except Exception as error:  # noqa: BLE001 - deliberate isolation boundary
            # One corrupt file (bad zip, bad encoding, permission error, ...) must not
            # throw away the results of every other file.
            LOG.warning("skipping %s: %s: %s", display_name, type(error).__name__, error)
            self.skipped.append((display_name, f"{type(error).__name__}: {error}"))

    def _add_table(self, raw: RawTable, group: str, sensor: str) -> None:
        stats_by_column: dict[str, ColumnStats] = {}
        for column, values in raw.values.items():
            if not values:                     # text-only column: not numeric
                continue
            stats = summarise_column(values, raw.rejected.get(column, 0))
            stats_by_column[column] = stats
            accumulator = self.columns[column]
            accumulator.overall.update(stats.minimum, stats.maximum, stats.count)
            accumulator.by_group[group].update(stats.minimum, stats.maximum, stats.count)
            accumulator.by_sensor[sensor].add(values, group)
            accumulator.tables += 1
        self.tables.append(TableSummary(
            raw.name, group, sensor, raw.row_count, len(raw.header), stats_by_column))


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _fmt_values(values: Iterable[float]) -> str:
    return "[" + ", ".join(f"{v:.6g}" for v in values) + "]"


def _column_lines(name: str, stats: ColumnStats) -> list[str]:
    """Two-line description of one column inside one table."""
    extra = f", non-numeric={stats.rejected:,d}" if stats.rejected else ""
    if stats.count < MIN_VALUES_FOR_STATS:
        return [f"  {name:42s} | only {stats.count} numeric value(s): "
                f"{_fmt_values(stats.unique_values)}{extra}"]
    head = f"  {name:42s} | min={stats.minimum:>14.6g}, max={stats.maximum:>14.6g}"
    detail = (f"mean={stats.mean:>14.6g}, median={stats.median:>14.6g}, "
              f"count={stats.count:,d}{extra}")
    # `nunique` is now THIS table's distinct count. The old code printed the *global*
    # distinct count here and tested it against a per-file sample, so the last
    # branch was unreachable and the "(N unique)" figure belonged to a different scope.
    if stats.nunique <= MAX_LISTED_UNIQUES:
        detail += ", values=" + _fmt_values(stats.unique_values)
    else:
        detail += f" ({stats.nunique:,d} unique)"
    return [head, " " * 14 + detail]


def _section_title(title: str) -> list[str]:
    return ["=" * 80, title, "=" * 80]


def build_report(analysis: DatasetAnalysis, csv_count: int, xlsx_count: int) -> list[str]:
    lines: list[str] = _section_title("NUMERICAL COLUMN ANALYSIS")
    lines += [
        "",
        f"Files: {csv_count} CSV | {xlsx_count} XLSX",
        f"Tables analysed: {len(analysis.tables)} (one per CSV file, one per XLSX sheet)",
        f"Numeric columns: {len(analysis.columns)}",
        "",
    ]
    lines += [f"  [{i}] {name}" for i, name in enumerate(sorted(analysis.columns), 1)]
    lines.append("")

    # ---- per-table section -------------------------------------------------
    for table in sorted(analysis.tables, key=lambda t: t.name):
        lines += ["-" * 65, f"FILE: {table.name}",
                  f"  Rows={table.row_count:<8} Cols={table.column_count}  Group={table.group}",
                  "-" * 65]
        if not table.columns:
            lines += ["  [No numeric columns]", ""]
            continue
        # Sort on the name only: sorting (name, stats) tuples would fall back to
        # comparing the stats objects if two names were ever equal.
        for name in sorted(table.columns):
            lines += _column_lines(name, table.columns[name])
        lines.append("")

    # ---- cross-dataset section --------------------------------------------
    lines += _section_title("CROSS-DATASET GLOBAL SAMPLE SPACE") + [""]
    for name in sorted(analysis.columns):
        acc = analysis.columns[name]
        lines += [f"COLUMN: {name}",
                  f"  GLOBAL RANGE: [{acc.overall.minimum:>12.4g} .. {acc.overall.maximum:>12.4g}]",
                  f"  Across {len(acc.by_group)} group(s):"]
        for group in sorted(acc.by_group):
            rng = acc.by_group[group]
            lines.append(f"    {group:15s}: [{rng.minimum:>10.4f} .. {rng.maximum:>10.4f}] "
                         f"({rng.count:>8,d})")
        span = acc.overall.maximum - acc.overall.minimum
        if len(acc.by_group) > 1 and span > DEVIATION_EPSILON:
            for group in sorted(acc.by_group):
                rng = acc.by_group[group]
                # How far, as % of the global span, a group stops short of each end.
                # Group min/max are taken from the same values as the global ones, so
                # both differences are exactly >= 0 and need no clamping.
                low = (rng.minimum - acc.overall.minimum) / span * 100
                high = (acc.overall.maximum - rng.maximum) / span * 100
                lines.append(f"    {group:15s}: min_dev={low:6.1f}%  max_dev={high:6.1f}%")
        small = [(s, a) for s, a in sorted(acc.by_sensor.items()) if a.is_small]
        if small:
            lines.append(f"  Small sensors (< {SMALL_SENSOR_LIMIT} values; "
                         f"unique values, {SENSOR_ROUNDING} dp):")
            for sensor, sensor_acc in small:
                label = f"sensor_{sensor}"
                lines.append(f"    {label:25s}: {_fmt_values(sorted(sensor_acc.uniques))}")
        lines += [f"  Present in {acc.tables} file(s)/sheet(s)", ""]

    # ---- key findings ------------------------------------------------------
    lines += _section_title("KEY FINDINGS") + ["", "SENSOR TYPES BY DATASET:"]
    # Case-insensitive lookup; the old code hard-coded 'location_x' and printed an
    # empty section without any hint when the column was spelled differently.
    location_column = next((n for n in analysis.columns if n.lower() == "location_x"), None)
    if location_column is None:
        lines.append("  (no 'location_x' column found)")
    else:
        sensors = analysis.columns[location_column].by_sensor
        for sensor in sorted(sensors):
            lines.append(f"  {sensor:20s}: {', '.join(sorted(sensors[sensor].groups))}")

    categories: dict[str, list[str]] = defaultdict(list)
    for name in analysis.columns:
        lowered = name.lower()          # ONE case rule for every test (was mixed before)
        for label, keyword in CATEGORY_RULES:
            if keyword in lowered:
                categories[label].append(name)
                break
    for label in sorted(categories):
        lines += ["", label + ":"]
        for name in sorted(categories[label]):
            rng = analysis.columns[name].overall
            lines.append(f"  - {name:42s} [{rng.minimum:.6g} .. {rng.maximum:.6g}]")

    if analysis.skipped:
        lines += ["", "SKIPPED FILES (could not be read):"]
        lines += [f"  - {name}: {reason}" for name, reason in analysis.skipped]

    lines += ["", f"TOTAL NUMERIC COLUMNS = {len(analysis.columns)}", "=" * 80]
    return lines


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def discover_files(root: Path) -> tuple[list[Path], list[Path]]:
    """Return (csv_files, xlsx_files), sorted, matching extensions case-insensitively."""
    csv_files: list[Path] = []
    xlsx_files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith(IGNORED_PREFIXES):
            continue
        suffix = path.suffix.lower()      # also finds DATA.CSV, which a '*.csv' glob misses on Linux
        if suffix == ".csv":
            csv_files.append(path)
        elif suffix == ".xlsx":
            xlsx_files.append(path)
    return csv_files, xlsx_files


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", nargs="?", type=Path, default=Path(DEFAULT_ROOT),
                        help=f"dataset folder (default: {DEFAULT_ROOT!r})")
    parser.add_argument("-o", "--output", type=Path, default=Path(DEFAULT_OUTPUT),
                        help=f"report file (default: {DEFAULT_OUTPUT})")
    parser.add_argument("-v", "--verbose", action="store_true", help="log every file read")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")

    if not args.root.is_dir():
        # A relative default path silently depends on the current directory; say so.
        LOG.error("dataset folder not found: %s (current directory: %s)", args.root, Path.cwd())
        return 2

    csv_files, xlsx_files = discover_files(args.root)
    LOG.info("Found %d CSV files, %d XLSX files", len(csv_files), len(xlsx_files))
    if not (csv_files or xlsx_files):
        LOG.error("no .csv or .xlsx files under %s", args.root)
        return 1
    if xlsx_files and openpyxl is None:
        LOG.warning("openpyxl is not installed: XLSX files will be skipped "
                    "(pip install openpyxl)")

    analysis = DatasetAnalysis()
    for path in csv_files + xlsx_files:
        analysis.process_file(path, args.root)

    lines = build_report(analysis, len(csv_files), len(xlsx_files))
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"DONE - {len(lines)} lines written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())