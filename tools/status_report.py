#!/usr/bin/env python3
"""The project report: ONE local page in three parts.

  python3 tools/status_report.py   -> data/derived/REPORT.html
    part 1  the one-screen summary for the coordinator (built, honest scores, approvals, questions, probe plan)
    part 2  the story: eight short chapters, one explanatory chart each, and a glossary
    part 3  the technical appendix (collapsed): scoreboards, sensitivity, calibration, health, probes, provenance
  options: --allow-stale (a source older than the tool that produces it does not fail check 4)
           --findings PATH (read another findings.json, e.g. a proposed edit, instead of the
           controller's data/derived/status/findings.json; the page is then written next to PATH)

The page is a LOCAL, self-contained file in the house style of tools/eval_charts.py (inline
SVG + inline CSS / JS, no external URL, font or script). Never publish them: they describe the
competition data, which must not leave the machine.

Where the numbers come from (nothing is typed into the narrative by hand):
* data/derived/status/findings.json: the controller's verified findings; each is rendered with
  its values substituted, its source and, for status 'journal' / 'unverified', a badge.
* Scores are RE-SCORED from the current submission files with score.score_rows (every default
  configuration, truth 'all' and 'airborne') for the methods that have all 6 testing files.
  Methods without files (B0_radar, B2_allpos: their files are capped to one testing iteration)
  come from their LAST block in experiments.csv and are labelled so.
* Per-iteration comparisons use compare.count_wlt / compare.sign_test_p ("wins k of 6"); means
  are context only. S1-I1 and S2-I2 are marked: they come from the scenarios the evaluation
  iterations use. The three S3 iterations are correlated, so per-scenario counts are given too.
* Other inputs: gospa_sensitivity/sensitivity.csv, probes/probes.csv + probes.md,
  sensor_calibration.csv (radar_bias rows), radar_registration/offsets.csv,
  triangulation/triangulation_rows.csv, fused/<slug>__tracks.csv, sensor_inventory.csv and the
  sensor file names, the plan ~/.claude/plans/wiggly-napping-melody.md, and the optional
  outputs of tools still being built (radar_model/, health/, compact/, promoted variants
  F0_fuse_v3h / F0_fuse_v3a / F0_fuse_noutm / F0_noutm_input), which render as 'pending' cards.

Self-checks (numbered; any failure exits 1):
 1 F0's per-iteration combined score (c50_p2_3d, both truth modes) agrees to 1e-6 across the
   re-score, sensitivity.csv's method rows and probes.csv's P0_f0 'proxy' rows.
 2 every rendered win / loss / tie count equals a plain loop over the differences.
 3 GOSPA cost shares (localisation + missed + false) sum to 1 per iteration and truth mode.
 4 staleness: every source file read is newer than the tool that produces it (or --allow-stale).
 5 every findings.json placeholder resolves, every source_path exists or is a journal
   reference, every finding is rendered, and no narrative text contains a digit that did not come
   from a value or a computed field: neither the templates written in this tool nor the
   controller's text (findings[].text and the coordinator lines). Excepted: identifiers (F0_fuse,
   S1-I1, 3-D, file names), product names (Matrice 300R) and definition parameters attached to a
   comparison ('within 50 m', '> 40 m/s', 'c = 20 m', '1-s speed'); a number such as 'about 55 %'
   or '3-4 times' fails. Coordinator lines may cite a finding value as {finding_id.key} or a
   computed field ({n_iterations}, {n_testing}, {n_evaluation}, {grid_ms}); every one must resolve.
 6 the HTML parses with html.parser with balanced tags, every <svg> parses with xml.etree,
   element ids are unique, and there is no 'http', 'src=', 'url(' or '@import'.
 7 no number near an iteration origin: no number with >= 2 decimals anywhere in the page (text,
   tooltips, SVG geometry) lies within 0.5 deg of any iteration's origin latitude or anonymized
   longitude (integers and 1-decimal numbers locate nothing and are not scanned; planted values
   in text and in a tooltip prove the scan finds them); no anonymization key_id and no
   stand-alone 8-hex token (deanonymize fingerprint; UUIDs in tool outputs are redacted); no file
   under data/derived/upload/ is read.
 8 every figure has a data table; no 'nan' / 'inf' in the page; every plotted value lies inside
   its axes.
 9 rendering is deterministic (two renders give the same sha256 apart from the as-of line; the
   page on disk from the previous run must match when no source changed since); page < 3 MB.
10 file-backed findings this tool can recompute (F0 / B2 means, F0 versus c, bias oracle,
   unseen shares, radar azimuth biases, the F1-versus-F0 counts at their source's three-decimal
   precision) agree with the files at their printed precision.
The story re-scores F0_fuse, F0_fuse_v3h and F0_fuse_v3a itself at c = 10 / 20 / 30 / 50 / 75 /
100 m, p = 2, 3-D, both truth modes, with score.score_rows custom configurations (the '<method>@c<c>'
score blocks); the appendix and the summary use the default-configuration re-score.
11 story budget (part 2 only): at most 8 charts; at most 120 words per chapter outside table cells,
   collapsed data blocks, charts and the source line (no other exemption; words in an open table
   are printed); part 2 < 400 KB; every chart has a takeaway title (>= 5 words), an axis label on every panel and a
   collapsible data table; chapters 1-8 present, none pending.
12 the story's numbers agree with the findings they stand next to: weights vs score.WEIGHTS, F0
   versus c re-scored here, v3a / v3h win counts and means, the S1-I1 honesty drop, truth shares
   (per-iteration rows sum to POOLED, shares sum to 1), the v3a rotation table vs
   sensor_calibration.csv, health_eval vs health/F0_fuse.json, the tier ladder vs tier_models.md
   (wins recounted per iteration, tier names backed by the tier's fields). A rounded finding the
   page does not rely on (it prints the file value) is reported as INFO.
A 'journal' finding whose source_path also names a file is rendered with a warning when that
file does not contain the finding's numbers (the file is context, not the source of the numbers).

Coordinates: only local ENU metres and angles are shown; absolute latitudes / longitudes
(anonymized) are never printed. Testing-only sensors (EUIG, LJJX, MMJJ, NJQL) are labelled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from html.parser import HTMLParser
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare as CP  # noqa: E402
import cuas_common as C  # noqa: E402
import eval_charts as EC  # noqa: E402
import score as SC  # noqa: E402
import submission as S  # noqa: E402

TOOLS = Path(__file__).resolve().parent
DER = C.DERIVED
OUT_PAGE = DER / "REPORT.html"
OLD_PAGES = ("STORY.html", "STATUS_COORDINATOR.html", "STATUS_REPORT.html")   # the three pages REPORT.html replaced
FINDINGS = DER / "status" / "findings.json"
PLAN = Path.home() / ".claude" / "plans" / "wiggly-napping-melody.md"
UPLOAD = (DER / "upload").resolve()
MAX_BYTES = 3 * 1024 * 1024
W = 880
esc = html.escape

MODES = ("all", "airborne")
PARTS = ("track", "classification", "identity", "metadata")
VARIANTS = (("v3h", ("F0_fuse_v3h", "F0h_fuse")), ("v3a", ("F0_fuse_v3a", "F0a_fuse")),
            ("noutm", ("F0_fuse_noutm",)), ("noutm_input", ("F0_noutm_input",)))
VARIANT_PRODUCER = {"v3h": "run.py --derived data/derived_v3h, then --promote v3h",
                    "v3a": "run.py --derived data/derived_v3a, then --promote v3a",
                    "noutm": "ablation.py (F0 retrained without the UTM relays)",
                    "noutm_input": "ablation.py (UTM relays removed at the input)"}
BASE_METHODS = (  # name, role, producer, short label
    ("F0_fuse", "deployable", "fuse.py", "F0 fusion tracker, the upload candidate"),
    ("F1_fuse", "deployable", "track_existence.py", "F1 track-level existence model (shelved)"),
    ("B2_allpos", "baseline", "baselines.py", "B2 every deployable position-sensor track"),
    ("B0_radar", "baseline", "baselines.py", "B0 every deployable radar track"),
    ("O3_fuse_oracle", "reference", "fuse.py", "O3 F0's fused tracks, chosen with the truth (reference)"),
    ("O1_oracle_exist", "reference", "baselines.py", "O1 B2 restricted to real-drone segments (reference)"),
    ("O2_oracle_dedup", "reference", "baselines.py", "O2 O1 with one row per drone and step (reference)"),
)
RADARS = ("EGOI", "NYTF", "UXYJ", "IUKL", "EUIG")


# --------------------------------------------------------------------------- #
# Source registry (provenance + staleness + the upload/ guard)
# --------------------------------------------------------------------------- #

SOURCES: dict[str, dict] = {}


def rel(p: Path) -> str:
    p = Path(p).resolve()
    for base, lab in ((C.REPO, ""), (Path.home(), "~/")):
        try:
            return lab + str(p.relative_to(base))
        except ValueError:
            pass
    return "(outside the repo) " + p.name   # e.g. a trial --findings file in a scratch folder


def src(path: Path, producer: str | None = None) -> Path:
    """Register a file this tool reads (provenance, check 4, check 7's upload/ guard)."""
    path = Path(path)
    rp = path.resolve()
    SOURCES[rel(rp)] = {"path": rp, "producer": producer}
    return path


def read_csv(path: Path, producer: str | None = None) -> list[dict]:
    with src(path, producer).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def mtime_str(p: Path) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))


# --------------------------------------------------------------------------- #
# Narrative discipline (check 5) and plot bookkeeping (check 8)
# --------------------------------------------------------------------------- #

IDENT_RE = re.compile(r"""
    [\w./~-]+\.(?:py|csv|md|json|html|npz|gz|xlsx)\b     # file names
  | [A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+              # snake identifiers: F0_fuse, c50_p2_3d
  | \bS\d(?:-I\d)?\b                                   # scenario / iteration names
  | \b[FOBP]\d[a-z]?\b                                 # short method names F0, O3, B2, P0
  | \bv\d[a-z]\b                                       # variant names v3h, v3a
  | \b[23]-D\b                                         # GOSPA dimension modes
""", re.X)
PLACEHOLDER_RE = re.compile(r"\{[a-z_0-9]+\}")
REF_RE = re.compile(r"\{(\w+)(?:\.(\w+))?\}")   # {computed_field} or {finding_id.key} in controller text
MODEL_RE = re.compile(r"\b(?:Matrice|Mavic|Evo|Anafi|Chimera|MTRC)\s+\d+\w*", re.I)  # product names
DEF_RE = re.compile(r"""
    (?:\bwithin|\bmore\ than|\bless\ than|\bat\ least|\bat\ most|(?<!-)[<>]=?|(?<![<>])=)\s*   # a comparison or '=' (not an arrow '->') ...
    [-+]?\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)*                               # ... a number or 'a / b / c' list ...
    (?:\s*(?:m/s|ms|km|m|s|deg|%))?                                           # ... and its unit
  | \b\d+-s\b                                                                 # duration adjective: '1-s speed'
""", re.X)


class Raw(str):
    """A value that is already HTML."""


class State:
    def __init__(self):
        self.narr_bad: list[str] = []
        self.out_axes: list[tuple] = []
        self.fig_without_table: list[str] = []
        self.pending: list[str] = []
        self.rendered_findings: set[str] = set()
        self.wlt_rendered: dict = {}
        self.ctl_unresolved: list[str] = []
        self.journal_gaps: list[str] = []
        self.story: list[dict] = []


ST = State()


def literal_numbers(text: str) -> list[str]:
    """Digit-bearing tokens of controller text that come from no value / computed field and are no
    identifier, product name or definition parameter (check 5)."""
    bare = REF_RE.sub(" ", text)
    for rx in (MODEL_RE, DEF_RE, IDENT_RE):
        bare = rx.sub(" ", bare)
    return re.findall(r"\S*\d\S*", bare)


def N(tpl: str, **kw) -> str:
    """Render a narrative template: the template itself may hold no digit (numbers arrive as
    values); values are escaped unless Raw."""
    bare = IDENT_RE.sub("", PLACEHOLDER_RE.sub("", tpl))
    if re.search(r"\d", bare):
        ST.narr_bad.append(tpl)
    return tpl.format(**{k: (v if isinstance(v, Raw) else esc(str(v))) for k, v in kw.items()})


class Plot(EC.Plot):
    """eval_charts.Plot that records every value mapped outside its axes (check 8)."""

    def X(self, v):
        lo, hi = sorted((self.x0, self.x1))
        if not (v == v and lo - 1e-9 <= v <= hi + 1e-9):
            ST.out_axes.append(("x", v, lo, hi))
        return super().X(v)

    def Y(self, v):
        lo, hi = sorted((self.y0, self.y1))
        if not (v == v and lo - 1e-9 <= v <= hi + 1e-9):
            ST.out_axes.append(("y", v, lo, hi))
        return super().Y(v)


def axis_hi(vmax: float, lo: float = 0.0) -> tuple[float, list]:
    vmax = max(vmax, lo + 1e-6)
    t = EC.nice_ticks(lo, vmax * 1.04 if vmax > 0 else vmax + 1e-6)
    step = t[1] - t[0] if len(t) > 1 else vmax
    hi = t[-1] if t[-1] >= vmax else t[-1] + step
    return hi, [x for x in t if x <= hi] + ([hi] if t[-1] < hi else [])


def sym_lim(vmax_abs: float) -> tuple[float, list]:
    hi, ticks = axis_hi(max(vmax_abs, 1e-6))
    return hi, sorted({-t for t in ticks} | set(ticks))


def card(title: str, what: str, fig: str = "", extra: str = "", how: str = "", cls: str = "") -> str:
    if fig and "<table" not in extra:
        ST.fig_without_table.append(title)
    how_html = f'<p class="how"><b>How to read it:</b> {how}</p>' if how else ""
    fig_html = f'<div class="fig">{fig}</div>' if fig else ""
    return (f'<section class="card{(" " + cls) if cls else ""}"><h3>{esc(title)}</h3><p class="what">{what}</p>'
            f'{how_html}{fig_html}{extra}</section>')


def pending_card(title: str, producer: str, extra: str = "") -> str:
    ST.pending.append(f"{title} (pending: produced by {producer})")
    return (f'<section class="card pending"><h3>{esc(title)}</h3><p class="what"><span class="badge warn">pending</span> '
            f'produced by <code>{esc(producer)}</code>; not available yet.</p>{extra}</section>')


def table(header, rows, caption="Data table", open_=False) -> str:
    h = "".join(f"<th>{esc(str(x))}</th>" for x in header)
    b = "".join("<tr>" + "".join(f"<td>{x if isinstance(x, Raw) else esc(str(x))}</td>" for x in r) + "</tr>" for r in rows)
    op = " open" if open_ else ""
    return (f'<details{op}><summary>{esc(caption)}</summary><div class="tw"><table><thead><tr>{h}</tr></thead>'
            f'<tbody>{b}</tbody></table></div></details>')


def plain_table(header, rows) -> str:
    h = "".join(f"<th>{esc(str(x))}</th>" for x in header)
    b = "".join("<tr>" + "".join(f"<td>{x if isinstance(x, Raw) else esc(str(x))}</td>" for x in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def f4(v, nd=4) -> str:
    if v is None or (isinstance(v, float) and (v != v or math.isinf(v))):
        return "–"
    return f"{v:.{nd}f}"


def sgn(v, nd=4) -> str:
    if v is None or v != v:
        return "–"
    return f"{v:+.{nd}f}"


def pct(v, nd=1) -> str:
    if v is None or v != v:
        return "–"
    return f"{100 * v:.{nd}f} %"


def short(key: str) -> str:
    return key.split("/")[-1]


# --------------------------------------------------------------------------- #
# Scoring (re-score submission files; <= 3 worker processes)
# --------------------------------------------------------------------------- #

_IT_BY_KEY: dict = {}
UNREADABLE: dict = {}   # method -> why its files could not be read (being written by another tool)


def _it(key: str) -> C.Iteration:
    if not _IT_BY_KEY:
        _IT_BY_KEY.update({i.key: i for i in C.iterations()})
    return _IT_BY_KEY[key]


def _score_task(task):
    method, key = task
    CP.memoise_frame()
    it = _it(key)
    try:
        rows = S.read_submission(it, C.submission_path(method, it))
    except Exception as e:  # noqa: BLE001  a file another tool is writing right now
        return method, key, {"error": f"{type(e).__name__}: {e}"[:160]}
    out = {}
    for mode in MODES:
        r = SC.score_rows(it, rows, mode)
        pr = r[SC.PRIMARY]
        d = dict(r["parts"])
        for k in ("cost_share_loc", "cost_share_missed", "cost_share_false", "loc_rms_m", "missed_rate",
                  "false_rate", "cost", "worst", "n_missed", "n_false", "n_assigned"):
            d[k] = pr[k]
        d["n_tracks"], d["n_est_rows"], d["n_truth_rows"] = r["n_tracks"], r["n_est_rows"], r["n_truth_rows"]
        d["track_by_cfg"] = {c[0]: r[c[0]]["track"] for c in SC.CONFIGS}
        out[mode] = d
    return method, key, out


def cfg_label(cfg: tuple) -> str:
    return f"c = {cfg[1]:g} m, p = {cfg[2]:g}, {cfg[3]}-D"


def testing_keys() -> list[str]:
    return [it.key for it in SC.testing_iterations()]


def has_all_files(method: str, keys: list[str]) -> bool:
    return all(C.submission_path(method, _it(k)).exists() for k in keys)


def experiments_block(method: str, keys: list[str]) -> tuple[dict, str] | None:
    """The LAST block of `method` in experiments.csv: {mode: {key: parts}} and its code hash."""
    path = DER / "experiments.csv"
    if not path.exists():
        return None
    rows = [r for r in read_csv(path, "baselines.py") if r["method"] == method]
    if not rows:
        return None
    h = rows[-1]["code_hash"]
    out: dict = {m: {} for m in MODES}
    for r in rows:
        if r["code_hash"] != h or r["iter_key"] not in keys or r["truth_mode"] not in MODES:
            continue
        d = {}
        for k in ("combined",) + PARTS + ("cost_share_loc", "cost_share_missed", "cost_share_false",
                                          "loc_rms_m", "missed_rate", "false_rate"):
            v = r.get(k, "")
            d[k] = float(v) if v not in ("", None) else float("nan")
        d["n_tracks"] = int(r["n_tracks"]) if r.get("n_tracks") else 0
        d["n_est_rows"] = int(r["n_est_rows"]) if r.get("n_est_rows") else 0
        out[r["truth_mode"]][r["iter_key"]] = d
    if not all(len(out[m]) == len(keys) for m in MODES):
        return None
    return out, h


def gather_scores(coordinator: bool, keys: list[str]) -> tuple[dict, dict, dict]:
    """SCORES[method][mode][key] = parts dict; META[method] = info; VAR[tag] = method name."""
    meta: dict = {}
    var: dict = {}
    wanted = [m for m in BASE_METHODS if not coordinator or m[0] == "F0_fuse"]
    for name, role, prod, label in wanted:
        meta[name] = {"role": role, "producer": prod, "label": label}
    for tag, names in VARIANTS:
        if coordinator and tag not in ("v3h", "v3a"):
            continue
        for nm in names:
            if (C.SUBMISSIONS / nm).is_dir() and has_all_files(nm, keys):
                var[tag] = nm
                meta[nm] = {"role": "variant", "producer": VARIANT_PRODUCER[tag],
                            "label": f"{nm} (variant {tag} of F0)"}
                break
    tasks, scores = [], {}
    for name in meta:
        if has_all_files(name, keys):
            meta[name]["source"] = "re-scored from the current submission files"
            for k in keys:
                p = C.submission_path(name, _it(k))
                src(p, meta[name]["producer"].split()[0])
                tasks.append((name, k))
        else:
            blk = experiments_block(name, keys)
            if blk is None:
                meta[name]["source"] = None
                continue
            scores[name] = blk[0]
            meta[name]["source"] = f"experiments.csv, last block (code {blk[1]}): no complete set of files"
    meta = {k: v for k, v in meta.items() if v.get("source")}
    if tasks:
        gt = DER / "groundtruth.csv.gz"
        if gt.exists():
            src(gt, "build_detections.py")
        if not coordinator:
            print(f"  re-scoring {len(tasks)} files with up to 3 worker processes ...", flush=True)
        with Pool(min(3, len(tasks))) as pool:
            res = pool.map(_score_task, tasks, chunksize=1)
        broken = {}
        for method, key, out in res:
            if "error" in out:
                broken.setdefault(method, f"{short(key)}: {out['error']}")
                continue
            for mode in MODES:
                scores.setdefault(method, {m: {} for m in MODES})[mode][key] = out[mode]
        for method, why in sorted(broken.items()):
            print(f"  [skip] {method}: unreadable file ({why}); shown as pending")
            scores.pop(method, None)
            meta.pop(method, None)
            for tag in [t for t, m in var.items() if m == method]:
                var.pop(tag)
            UNREADABLE[method] = why
    return scores, meta, var


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def load_findings() -> dict:
    return json.loads(src(FINDINGS).read_text(encoding="utf-8"))


def load_sensitivity() -> list[dict]:
    p = DER / "gospa_sensitivity" / "sensitivity.csv"
    return read_csv(p, "gospa_sensitivity.py") if p.exists() else []


def load_probes() -> tuple[list[dict], str]:
    p, md = DER / "probes" / "probes.csv", DER / "probes" / "probes.md"
    rows = read_csv(p, "probes.py") if p.exists() else []
    text = src(md, "probes.py").read_text(encoding="utf-8") if md.exists() else ""
    return rows, text


def frames() -> dict:
    """Iteration origins (lat0, lon0) for check 7 only; never printed."""
    CP.memoise_frame()
    return {it.key: C.iteration_frame(it) for it in C.iterations()}


# --------------------------------------------------------------------------- #
# Findings rendering
# --------------------------------------------------------------------------- #


def finding_text(f: dict) -> str:
    vals = f.get("values", {})
    return re.sub(r"\{(\w+)\}", lambda m: f"<b>{esc(str(vals[m.group(1)]))}</b>" if m.group(1) in vals
                  else "<b>[missing value]</b>", esc(f["text"]))


def badge(status: str) -> str:
    if status == "journal":
        return '<span class="badge warn" title="verified in a workflow journal only">journal only</span>'
    if status == "unverified":
        return '<span class="badge crit" title="not independently verified">unverified</span>'
    return ""


def cited_files(f: dict) -> list[str]:
    return re.findall(r"(?:data|tools|README|docs)/[\w./-]+\.\w+", f.get("source_path", ""))


def journal_gap(f: dict) -> str:
    """For a 'journal' finding that also cites a text file: a warning when that file does not
    contain the finding's decimal numbers (the reader would look there and not find them)."""
    if f.get("status") != "journal":
        return ""
    # one pattern per value: its decimal numbers in order, close together ('27.1 -> 8.4' must occur as such)
    pats = {k: r"(?<![\d.])" + r"\D{1,15}".join(re.escape(t) for t in re.findall(r"\d+\.\d+", str(v))) + r"(?![\d])"
            for k, v in f.get("values", {}).items() if re.search(r"\d+\.\d+", str(v))}
    out = []
    for rp in cited_files(f):
        path = C.REPO / rp
        if not pats or not path.exists() or path.suffix not in (".md", ".csv", ".json", ".txt"):
            continue
        text = src(path).read_text(encoding="utf-8", errors="replace")
        missing = [k for k, rx in pats.items() if not re.search(rx, text)]
        if missing:
            ST.journal_gaps.append(f"{f['id']}: {len(missing)} of {len(pats)} values not in {rp}")
            out.append(N(' <span class="badge warn">numbers not in {p}</span> <span class="src">{n} of the {k} printed values '
                         'do not occur in {p}; they come from the journal. That file is related context and may use another statistic.</span>',
                         p=rp, n=len(missing), k=len(pats)))
    return "".join(out)


def finding_li(f: dict) -> str:
    ST.rendered_findings.add(f["id"])
    return (f'<li>{finding_text(f)} {badge(f.get("status", ""))}<br><span class="src">source: <code>{esc(f["source_path"])}</code>'
            f' · {esc(f.get("locator", ""))} · verified by: {esc(f.get("verifier", ""))}</span>{journal_gap(f)}</li>')


COMPUTED = {  # computed fields controller text may cite as {name}
    "n_iterations": lambda: len(C.iterations()),
    "n_testing": lambda: sum(1 for it in C.iterations() if it.tree == "testing"),
    "n_evaluation": lambda: sum(1 for it in C.iterations() if it.tree == "evaluation"),
    "grid_ms": lambda: _grid_ms(),
}


def _grid_ms() -> str:
    it = C.iterations()[0]
    src(C.template_path(it))
    return f"{1000 * float(np.median(np.diff(C.template_grid(it)[1]))):.0f}"
_COMPUTED_CACHE: dict = {}


def ctext(x: str, fs: list[dict]) -> str:
    """Render one controller-written line (coordinator lists): escaped, with {finding_id.key}
    replaced by that finding's value and {name} by a computed field; unresolved ones are marked
    and fail check 5."""
    def sub(m):
        a, b = m.group(1), m.group(2)
        if b is not None:
            v = fval(fs, a, b)
        elif a in COMPUTED:
            if a not in _COMPUTED_CACHE:
                _COMPUTED_CACHE[a] = COMPUTED[a]()
            v = _COMPUTED_CACHE[a]
        else:
            v = None
        if v is None:
            ST.ctl_unresolved.append(m.group(0))
            return "<b>[missing value]</b>"
        st = next((f.get("status", "") for f in fs if f["id"] == a), "") if b is not None else ""
        return f"<b>{esc(str(v))}</b>" + (" " + badge(st) if badge(st) else "")
    return REF_RE.sub(sub, esc(x))


def ctl_list(items, fs, tag="ul", cls="plain") -> str:
    return f'<{tag} class="{cls}">' + "".join(f"<li>{ctext(x, fs)}</li>" for x in items) + f"</{tag}>"


def findings_list(fs: list[dict], ids=None, sections=None) -> str:
    sel = [f for f in fs if (ids and f["id"] in ids) or (sections and f["section"] in sections)]
    if ids:
        sel.sort(key=lambda f: list(ids).index(f["id"]) if f["id"] in ids else 999)
    return '<ul class="find">' + "".join(finding_li(f) for f in sel) + "</ul>" if sel else ""


def fval(fs: list[dict], fid: str, key: str):
    for f in fs:
        if f["id"] == fid:
            return f.get("values", {}).get(key)
    return None


def num_in(s) -> float | None:
    m = re.search(r"[-+]?\d+(?:\.\d+)?", str(s or ""))
    return float(m.group(0)) if m else None


# --------------------------------------------------------------------------- #
# Comparisons ("wins k of n")
# --------------------------------------------------------------------------- #


def wlt(scores, a: str, b: str, mode: str, keys: list[str], part: str = "combined") -> dict:
    diffs = [scores[a][mode][k][part] - scores[b][mode][k][part] for k in keys]
    w, l, t = CP.count_wlt(diffs)
    per_scen: dict = defaultdict(lambda: [0, 0, 0])
    for k, d in zip(keys, diffs):
        sc = short(k).split("-")[0]
        ww, ll, tt = CP.count_wlt([d])
        per_scen[sc][0] += ww
        per_scen[sc][1] += ll
        per_scen[sc][2] += tt
    res = {"w": w, "l": l, "t": t, "n": len(keys), "p": CP.sign_test_p(w, l), "diffs": diffs,
           "scen": {k: tuple(v) for k, v in sorted(per_scen.items())}}
    ST.wlt_rendered[(a, b, mode, part)] = res
    return res


def wlt_text(r: dict) -> str:
    return f"{r['w']} win / {r['l']} loss / {r['t']} tie of {r['n']} (sign test p = {r['p']:.3f})"


def scen_text(r: dict) -> str:
    return " · ".join(f"{s}: {v[0]}–{v[1]}" + (f" ({v[2]} tie)" if v[2] else "") for s, v in r["scen"].items())


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #

SHAPES = {  # method role in the dot plot -> (shape, css class)
    "F0_fuse": ("circle", "s1"), "v3h": ("circle", "s2"), "v3a": ("circle", "s3"),
    "F1_fuse": ("diamond", "mk-ink"), "B2_allpos": ("square", "mk-ink"), "B0_radar": ("triangle", "mk-ink"),
    "O3_fuse_oracle": ("ring", "mk-ring"),
}


def mark(shape: str, cls: str, x: float, y: float, tip: str = "", r: float = 5) -> str:
    t = f' data-tip="{esc(tip)}" tabindex="0"' if tip else ""
    if shape == "circle":
        return f'<circle class="{cls} ring" cx="{x:.1f}" cy="{y:.1f}" r="{r}"{t}/>'
    if shape == "ring":
        return f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="{r + 0.5}"{t}/>'
    if shape == "diamond":
        return (f'<path class="{cls} ring" d="M{x:.1f},{y - r - 1:.1f}L{x + r + 1:.1f},{y:.1f}L{x:.1f},{y + r + 1:.1f}'
                f'L{x - r - 1:.1f},{y:.1f}Z"{t}/>')
    if shape == "square":
        return f'<rect class="{cls} ring" x="{x - r + 0.5:.1f}" y="{y - r + 0.5:.1f}" width="{2 * r - 1}" height="{2 * r - 1}"{t}/>'
    return (f'<path class="{cls} ring" d="M{x:.1f},{y - r - 1:.1f}L{x + r + 1:.1f},{y + r:.1f}L{x - r - 1:.1f},'
            f'{y + r:.1f}Z"{t}/>')


def shape_legend(items) -> str:
    out = []
    for shape, cls, lab in items:
        out.append(f'<span><svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">{mark(shape, cls, 8, 8, r=5)}</svg>'
                   f'{esc(lab)}</span>')
    return '<div class="legend">' + "".join(out) + "</div>"


def plot_methods(scores, var) -> list[tuple[str, str]]:
    """(method name, SHAPES key) in drawing order."""
    out = []
    for nm, key in (("B0_radar", "B0_radar"), ("B2_allpos", "B2_allpos"), ("F1_fuse", "F1_fuse"),
                    ("O3_fuse_oracle", "O3_fuse_oracle"), ("F0_fuse", "F0_fuse")):
        if nm in scores:
            out.append((nm, key))
    for tag in ("v3h", "v3a"):
        if tag in var and var[tag] in scores:
            out.append((var[tag], tag))
    return out


def dotplot(scores, meta, var, keys, eval_scen) -> tuple[str, str, list]:
    pm = plot_methods(scores, var)
    vmax = max(scores[m][mode][k]["combined"] for m, _ in pm for mode in MODES for k in keys)
    hi, ticks = axis_hi(vmax)
    svgs = []
    rows_tab = []
    for mode in MODES:
        n = len(keys)
        row_h = 30
        h = 16 + n * row_h + 44
        p = Plot(430, h, (0, hi), (0, n), left=86, right=14, top=16, bottom=44, xticks=ticks, yticks=[],
                 xlabel=f"combined score, truth '{mode}'")
        for i, k in enumerate(keys):
            yc = p.Y(n - i - 0.5)
            mk = " ◆" if int(short(k)[1]) in eval_scen else ""
            p.add(f'<text class="tick" x="{p.l - 8}" y="{yc + 4:.1f}" text-anchor="end">{esc(short(k) + mk)}</text>')
            p.add(f'<line class="grid" x1="{p.l}" x2="{p.w - p.r}" y1="{yc:.1f}" y2="{yc:.1f}"/>')
            for j, (m, sk) in enumerate(pm):
                v = scores[m][mode][k]["combined"]
                off = (j - (len(pm) - 1) / 2) * 3.0
                shape, cls = SHAPES[sk]
                p.add(mark(shape, cls, p.X(v), yc + off, tip=f"{m} · {short(k)} · truth '{mode}': combined {v:.4f}"))
        svgs.append(f"<div>{p.svg(f'Combined score per testing iteration, truth {mode}')}</div>")
    for m, _ in pm:
        for mode in MODES:
            rows_tab.append([m, mode] + [f4(scores[m][mode][k]["combined"]) for k in keys]
                            + [f4(float(np.mean([scores[m][mode][k]["combined"] for k in keys])))])
    leg = shape_legend([SHAPES[sk] + (m,) for m, sk in pm])
    return '<div class="two">' + "".join(svgs) + "</div>", leg, rows_tab


def cost_share_chart(sc_f0, keys) -> tuple[str, list]:
    segs = (("cost_share_loc", "localisation error", "s1"), ("cost_share_missed", "missed drones", "s2"),
            ("cost_share_false", "false rows", "s3"))
    row_h, left = 22, 150
    h = 12 + (len(keys) + 1) * 2 * row_h + 44
    p = Plot(W, h, (0, 100), (0, 1), left=left, right=20, top=10, bottom=44, xticks=[0, 25, 50, 75, 100], yticks=[],
             xfmt=lambda v: f"{v:g}%", xlabel=f"share of F0's GOSPA cost ({cfg_label(SC.CONFIGS[0])})")
    y, rows = 10, []
    for mode in MODES:
        p.add(f'<text class="tier" x="8" y="{y + 15}">truth \'{esc(mode)}\'</text>')
        y += row_h
        for k in keys:
            d = sc_f0[mode][k]
            p.add(f'<text class="tick" x="{left - 8}" y="{y + 15}" text-anchor="end">{esc(short(k))}</text>')
            x = p.X(0)
            for i, (col, lab, cls) in enumerate(segs):
                v = 100 * d[col]
                x1 = p.X(min(100.0, (100 * sum(d[c] for c, _, _ in segs[:i + 1]))))
                wpx = x1 - x
                gap = 2 if wpx > 3 else 0
                if wpx > 0.2:
                    p.add(f'<rect class="{cls}" x="{x:.1f}" y="{y + 4}" width="{max(wpx - gap, 0.6):.1f}" height="{row_h - 8}" '
                          f'data-tip="{esc(short(k))}, truth \'{esc(mode)}\': {esc(lab)} {v:.1f}% of the cost" tabindex="0"/>')
                x = x1
            rows.append([mode, short(k)] + [pct(d[c]) for c, _, _ in segs] + [pct(sum(d[c] for c, _, _ in segs), 2)])
            y += row_h
    return p.svg("F0 GOSPA cost shares per iteration"), rows, EC.legend([(c, l) for _, l, c in segs])


def recall_minis(sc_f0, keys) -> tuple[str, list]:
    minis, rows = [], []
    labels = {"track": "track part", "classification": "classification recall", "identity": "identification recall",
              "metadata": "metadata part"}
    for part in PARTS:
        vals = [sc_f0[m][k][part] for m in MODES for k in keys]
        hi, ticks = axis_hi(max(v for v in vals if v == v) if any(v == v for v in vals) else 1.0)
        hi = min(max(hi, 0.1), 1.0) if hi <= 1 else hi
        ticks = [t for t in ticks if t <= hi]
        p = Plot(280, 200, (-0.5, len(keys) - 0.5), (0, hi), left=40, right=10, top=14, bottom=40,
                 xticks=list(range(len(keys))), yticks=ticks, xfmt=lambda i: short(keys[int(round(i))]).replace("-", " "))
        for mi, (mode, cls) in enumerate((("all", "s1"), ("airborne", "s2"))):
            for i, k in enumerate(keys):
                v = sc_f0[mode][k][part]
                if v == v:
                    p.add(EC.dot(p.X(i + (mi - 0.5) * 0.18), p.Y(v), cls, r=4, tip=f"{short(k)}, truth '{mode}': {labels[part]} {v:.3f}"))
        minis.append(f'<div class="mini"><div class="minititle">{esc(labels[part])}</div>{p.svg(labels[part])}</div>')
    for mode in MODES:
        for k in keys:
            rows.append([mode, short(k)] + [f4(sc_f0[mode][k][pp], 3) for pp in PARTS] + [f4(sc_f0[mode][k]["combined"])])
    return '<div class="minis">' + "".join(minis) + "</div>", rows


def hbar_chart(items, xlabel) -> str:
    """items: [(label, value, tip)] -> horizontal bars from 0 (single series)."""
    row_h, left = 26, 330
    vals = [v for _, v, _ in items]
    lo = min(0.0, min(vals))
    hi, ticks = axis_hi(max(vals + [1e-6]))
    if lo < 0:
        lo = -axis_hi(-lo)[0]
        ticks = sorted(set(EC.nice_ticks(lo, hi)))
    h = 12 + len(items) * row_h + 44
    p = Plot(W, h, (lo, hi), (0, 1), left=left, right=60, top=10, bottom=44, xticks=ticks, yticks=[], xlabel=xlabel,
             xfmt=lambda v: f"{v:+.2f}" if v else "0")
    y = 10
    for lab, v, tip in items:
        p.add(f'<text class="tick" x="{left - 8}" y="{y + 17}" text-anchor="end">{esc(lab)}</text>')
        x0, x1 = p.X(0), p.X(v)
        if v > 0:
            p.add(f'<path class="s1" d="{EC.bar_path(x0, y + 5, x1 - x0, row_h - 10)}" data-tip="{esc(tip)}" tabindex="0"/>')
        elif v < 0:
            p.add(f'<rect class="s1" x="{x1:.1f}" y="{y + 5}" width="{x0 - x1:.1f}" height="{row_h - 10}" data-tip="{esc(tip)}" tabindex="0"/>')
        p.add(f'<text class="val" x="{max(x0, x1) + 6:.1f}" y="{y + 17}">{v:+.3f}</text>')
        y += row_h
    p.add(f'<line class="zero" x1="{p.X(0):.1f}" x2="{p.X(0):.1f}" y1="{p.t}" y2="{h - p.b}"/>')
    return p.svg(xlabel)


def score_vs_c(sens, keys) -> tuple[str, str, list] | None:
    series = (("F0_fuse", "method", "", "F0", "l1", "s1"), ("O3_fuse_oracle", "method", "", "O3 (reference)", "l2", "s2"),
              ("F1_fuse", "method", "", "F1", "l3", "s3"), ("O3_fuse_oracle", "oracle", "bias", "O3 + bias removed (oracle)", "l4", "s4"))
    data: dict = {}
    for r in sens:
        if r["p"] in ("2", "2.0") and r["dims"] in ("3", "3.0") and r["iter_key"] in keys:
            for m, kind, varn, lab, _, _ in series:
                if r["method"] == m and r["kind"] == kind and r["variant"] == varn:
                    data.setdefault((lab, r["truth_mode"]), defaultdict(list))[float(r["c"])].append(float(r["combined"]))
    if not data:
        return None
    vmax = max(np.mean(v) for d in data.values() for v in d.values())
    hi, ticks = axis_hi(vmax)
    cs = sorted({c for d in data.values() for c in d})
    svgs, rows = [], []
    for mode in MODES:
        p = Plot(430, 280, (0, max(cs)), (0, hi), left=46, right=96, top=14, bottom=44, yticks=ticks,
                 xlabel=f"cut-off c (m), truth '{mode}'", xticks=[c for c in (0, 20, 40, 60, 80, 100) if c <= max(cs)])
        for m, kind, varn, lab, lcls, dcls in series:
            d = data.get((lab, mode))
            if not d:
                continue
            pts = [(c, float(np.mean(d[c]))) for c in sorted(d) if len(d[c]) == len(keys)]
            p.add(f'<polyline class="ln {lcls}" points="{" ".join(f"{p.X(c):.1f},{p.Y(v):.1f}" for c, v in pts)}"/>')
            for c, v in pts:
                p.add(EC.dot(p.X(c), p.Y(v), dcls, r=3.5, tip=f"{lab}, truth '{mode}', c = {c:g} m: mean combined {v:.3f}"))
            c, v = pts[-1]
            p.add(f'<text class="ann" x="{p.X(c) + 8:.1f}" y="{p.Y(v) + 4:.1f}">{esc(lab.split(" (")[0])}</text>')
            for c, v in pts:
                rows.append([mode, lab, f"{c:g}", f4(v, 4)] + [f4(x, 4) for x in d[c]])
        svgs.append(f"<div>{p.svg(f'Mean combined score versus GOSPA cut-off, truth {mode}')}</div>")
    leg = EC.legend([(dc, lab) for _, _, _, lab, _, dc in series])
    return '<div class="two">' + "".join(svgs) + "</div>", leg, rows


def radar_minis(cal, offs, keys, deployable) -> tuple[str, list, dict]:
    az: dict = defaultdict(dict)
    pooled: dict = {}
    nrows: dict = defaultdict(dict)
    for r in cal:
        if r["param"] != "az_offset_deg" or r["source"] != "radar_bias":
            continue
        sid = r["sensor_key"].split("_")[0]
        if r["iter_key"] == "*":
            pooled[sid] = float(r["value"])
        elif r["iter_key"] in keys:
            az[sid][r["iter_key"]] = float(r["value"])
            nrows[sid][r["iter_key"]] = r["n"]
    en: dict = defaultdict(dict)
    for r in sorted(offs, key=lambda r: (r["iter_key"], r["sensor_key"], r["method"])):
        if r["method"] == "gt_free_all" and r["iter_key"] in keys and r["gt_d_e"] not in ("", None):
            en[r["sensor_key"]][r["iter_key"]] = (float(r["gt_d_e"]), float(r["gt_d_n"]))
    azmax = max([abs(v) for d in az.values() for v in d.values()] + [abs(v) for v in pooled.values()] + [1.0])
    az_hi, az_t = sym_lim(azmax)
    enmax = max([abs(x) for d in en.values() for v in d.values() for x in v] + [1.0])
    en_hi, en_t = sym_lim(enmax)
    blocks, rows = [], []
    xt = list(range(len(keys)))
    xf = lambda i: short(keys[int(round(i))]).replace("-", " ")  # noqa: E731
    for rd in RADARS:
        if rd not in az and rd not in en:
            continue
        tag = "" if rd in deployable else " (testing-only)"
        p = Plot(430, 200, (-0.5, len(keys) - 0.5), (-az_hi, az_hi), left=46, right=60, top=14, bottom=40, xticks=xt,
                 yticks=az_t, xfmt=xf)
        p.add(f'<line class="zero" x1="{p.l}" x2="{p.w - p.r}" y1="{p.Y(0):.1f}" y2="{p.Y(0):.1f}"/>')
        if rd in pooled:
            yv = p.Y(pooled[rd])
            p.add(f'<line class="guide" x1="{p.l}" x2="{p.w - p.r}" y1="{yv:.1f}" y2="{yv:.1f}"/>'
                  f'<text class="ann" x="{p.w - p.r + 4}" y="{yv + 4:.1f}">pooled</text>')
        for i, k in enumerate(keys):
            if k in az[rd]:
                p.add(EC.dot(p.X(i), p.Y(az[rd][k]), "s1", r=4.5, tip=f"{rd} {short(k)}: azimuth bias {az[rd][k]:+.2f} deg (add to the measured azimuth)"))
        q = Plot(430, 200, (-0.5, len(keys) - 0.5), (-en_hi, en_hi), left=46, right=60, top=14, bottom=40, xticks=xt,
                 yticks=en_t, xfmt=xf)
        q.add(f'<line class="zero" x1="{q.l}" x2="{q.w - q.r}" y1="{q.Y(0):.1f}" y2="{q.Y(0):.1f}"/>')
        for i, k in enumerate(keys):
            if k in en[rd]:
                e, n = en[rd][k]
                q.add(EC.dot(q.X(i - 0.08), q.Y(e), "s1", r=4, tip=f"{rd} {short(k)}: east offset {e:+.1f} m (radar minus truth)"))
                q.add(EC.dot(q.X(i + 0.08), q.Y(n), "s2", r=4, tip=f"{rd} {short(k)}: north offset {n:+.1f} m (radar minus truth)"))
        blocks.append(f'<div class="minititle"><b>{esc(rd + tag)}</b></div><div class="two"><div class="mini">'
                      f'<div class="minititle">azimuth bias (deg, to add)</div>{p.svg(rd + " azimuth bias per iteration")}</div>'
                      f'<div class="mini"><div class="minititle">east / north offset (m)</div>{q.svg(rd + " east and north offset per iteration")}</div></div>')
        for k in keys:
            if k in az[rd] or k in en[rd]:
                e, n = en[rd].get(k, (None, None))
                rows.append([rd + tag, short(k), sgn(az[rd].get(k), 2), nrows[rd].get(k, "–"),
                             sgn(e, 1) if e is not None else "–", sgn(n, 1) if n is not None else "–"])
        if rd in pooled:
            rows.append([rd + tag, "pooled", sgn(pooled[rd], 2), "–", "–", "–"])
    return "".join(blocks), rows, {"az": az, "pooled": pooled}


def presence_grid(deployable) -> tuple[str, list]:
    its = C.iterations()
    inv = read_csv(DER / "sensor_inventory.csv", "sensor_inventory.py") if (DER / "sensor_inventory.csv").exists() else []
    rows_by: dict = defaultdict(int)
    pos_valid: dict = defaultdict(set)
    kinds: dict = defaultdict(set)
    for r in inv:
        rows_by[(r["iter_key"], r["sensor_id"])] += int(r["rows"] or 0)
        pos_valid[r["sensor_id"]].add(r["sensor_pos_valid"])
        kinds[r["sensor_id"]].add(r["meas_kind"])
    present: dict = defaultdict(lambda: defaultdict(int))
    types: dict = {}
    for it in its:
        for sf in C.sensor_files(it):
            present[sf.sensor_id][it.key] += 1
            types[sf.sensor_id] = sf.sensor_type
    def tags(s):
        t = []
        if s not in deployable:
            t.append("testing-only")
        if pos_valid.get(s) == {"0"}:
            t.append("UTM relay")
        if kinds.get(s) and all(k.startswith("bearing") for k in kinds[s]):
            t.append("bearing-only")
        return t
    order = sorted(present, key=lambda s: (s not in deployable, "UTM relay" in tags(s), "bearing-only" in tags(s), s))
    cell_w, row_h, left, top = 66, 20, 250, 44
    h = top + len(order) * row_h + 16
    parts = []
    for j, it in enumerate(its):
        x = left + j * cell_w + cell_w / 2
        parts.append(f'<text class="tick" x="{x:.1f}" y="{top - 22}" text-anchor="middle">{esc(it.tree)}</text>'
                     f'<text class="tick" x="{x:.1f}" y="{top - 8}" text-anchor="middle">{esc(short(it.key) if it.tree == "testing" else "D" + it.dataset + " " + short(it.key))}</text>')
    rows = []
    for i, s in enumerate(order):
        y = top + i * row_h
        tg = tags(s)
        lab = f"{s} · {types[s].lower()}" + (f" [{', '.join(tg)}]" if tg else "")
        parts.append(f'<text class="tick" x="{left - 8}" y="{y + 14}" text-anchor="end">{esc(lab)}</text>')
        rr = [s, types[s], ", ".join(tg) or "deployable"]
        for j, it in enumerate(its):
            x = left + j * cell_w
            nf = present[s].get(it.key, 0)
            if nf:
                nr = rows_by.get((it.key, s), 0)
                parts.append(f'<rect class="s1" x="{x + 2}" y="{y + 2}" width="{cell_w - 4}" height="{row_h - 4}" rx="3" '
                             f'data-tip="{esc(s)} in {esc(it.key)}: {nf} file(s), {nr:,} rows" tabindex="0"/>')
                rr.append(f"{nr:,}")
            else:
                parts.append(f'<rect class="empty" x="{x + 2}" y="{y + 2}" width="{cell_w - 4}" height="{row_h - 4}" rx="3"/>')
                rr.append("–")
        rows.append(rr)
    w = left + len(its) * cell_w + 8
    svg = (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Sensor presence per iteration"><title>Sensor presence per iteration</title>'
           + "".join(parts) + "</svg>")
    hdr = ["sensor", "type", "category"] + [short(it.key) if it.tree == "testing" else f"D{it.dataset} {short(it.key)}" for it in its]
    return svg, [hdr] + rows


def triangulation_chart(rows_csv, keys) -> tuple[str, list]:
    pick = {}
    for r in rows_csv:
        if r["offsets"] == "oracle" and r["unseen_def"] == "main":
            pick[(r["iter_key"], r["truth_mode"])] = r
    segs = (("share_unseen", "airborne truth rows no position sensor sees", "s1"),
            ("share_ge1_site", "of those, hit by bearings from one site", "s2"),
            ("share_ge2_site", "of those, hit from two or more sites", "s3"))
    order = [k for k in keys if (k, "airborne") in pick] + (["POOLED"] if ("POOLED", "airborne") in pick else [])
    row_h, left = 16, 120
    h = 10 + len(order) * (3 * row_h + 10) + 44
    p = Plot(W, h, (0, 100), (0, 1), left=left, right=60, top=10, bottom=44, xticks=[0, 25, 50, 75, 100], yticks=[],
             xfmt=lambda v: f"{v:g}%", xlabel="share (%), truth 'airborne', oracle azimuth offsets")
    y, tab = 10, []
    for k in order:
        r = pick[(k, "airborne")]
        p.add(f'<text class="tick" x="{left - 8}" y="{y + 2 * row_h - 2}" text-anchor="end">{esc(short(k) if k != "POOLED" else "pooled")}</text>')
        for col, lab, cls in segs:
            v = 100 * float(r[col])
            x0, x1 = p.X(0), p.X(v)
            if x1 - x0 > 0.3:
                p.add(f'<path class="{cls}" d="{EC.bar_path(x0, y + 1, x1 - x0, row_h - 3)}" data-tip="{esc(short(k))}: {esc(lab)} {v:.2f}%" tabindex="0"/>')
            p.add(f'<text class="val" x="{x1 + 5:.1f}" y="{y + row_h - 4}">{v:.1f}%</text>')
            y += row_h
        y += 10
    for (k, mode), r in sorted(pick.items()):
        tab.append([short(k) if k != "POOLED" else "pooled", mode, r["n_truth"], r["n_unseen"], pct(float(r["share_unseen"])),
                    pct(float(r["share_ge1_site"])), pct(float(r["share_ge2_site"]), 2), pct(float(r["share_rec50"]), 2)])
    return p.svg("Triangulation reach per iteration") + EC.legend([(c, l) for _, l, c in segs]), tab


def probe_strips(prows, keys) -> tuple[str, list, list]:
    probes = sorted({r["probe"] for r in prows} - {"P0_f0"})
    rules = sorted({r["label_rule"] for r in prows}, key=lambda s: ("proxy", "unk", "unk_red").index(s) if s in ("proxy", "unk", "unk_red") else 9)
    cfgs = [c for c in ("c50_p2_3d", "c10_p2_3d") if any(r["config"] == c for r in prows)]
    idx = {(r["probe"], r["truth_mode"], r["label_rule"], r["config"], r["iter_key"]): float(r["d_combined"]) for r in prows}
    dmax = max([abs(v) for v in idx.values()] + [1e-3])
    hi, ticks = sym_lim(dmax)
    out, tab = [], []
    for pr in probes:
        hyps = [(m, rl, c) for m in MODES for rl in rules for c in cfgs]
        row_h, left = 18, 230
        h = 12 + len(hyps) * row_h + 44
        p = Plot(W, h, (-hi, hi), (0, 1), left=left, right=20, top=10, bottom=44, xticks=ticks, yticks=[],
                 xfmt=lambda v: f"{v:+.2f}" if v else "0", xlabel=f"{pr}: predicted change of the combined score versus P0 (one dot per testing iteration)")
        p.add(f'<line class="zero" x1="{p.X(0):.1f}" x2="{p.X(0):.1f}" y1="{p.t}" y2="{h - p.b}"/>')
        y = 10
        for m, rl, c in hyps:
            ds = [idx.get((pr, m, rl, c, k)) for k in keys]
            p.add(f'<text class="tick" x="{left - 8}" y="{y + 13}" text-anchor="end">{esc(f"truth {m} · labels {rl} · {c}")}</text>')
            got = [d for d in ds if d is not None]
            for k, d in zip(keys, ds):
                if d is not None:
                    p.add(EC.dot(p.X(d), y + row_h / 2, "s1 faint", r=3.5, tip=f"{pr} · {short(k)} · truth {m} · labels {rl} · {c}: {d:+.4f}"))
            if got:
                med = float(np.median(got))
                xm = p.X(med)
                p.add(f'<line class="tick-med" x1="{xm:.1f}" x2="{xm:.1f}" y1="{y + 3}" y2="{y + row_h - 3}"/>')
                up = sum(d > 0 for d in got)
                tab.append([pr, m, rl, c] + [sgn(d) for d in ds] + [f"{up} up / {len(got) - up} down or equal"])
            y += row_h
        out.append(f'<div class="mini">{p.svg(pr + " predicted deltas")}</div>')
    return "".join(out), tab, probes


def fused_counts(keys_all) -> tuple[str, list] | None:
    rows = []
    for key in keys_all:
        p = DER / "fused" / f"{C.iter_slug(key)}__tracks.csv"
        if not p.exists():
            continue
        tr = read_csv(p, "fuse.py")
        rep = [r for r in tr if r["reported_f0"] == "1"]
        lab = sum(1 for r in rep if r.get("label_drone") == "1") if key.startswith("testing") else None
        med = float(np.median([float(r["duration_s"]) for r in rep])) if rep else float("nan")
        rows.append((key, len(tr), len(rep), lab, med))
    if not rows:
        return None
    hi, ticks = axis_hi(max(r[2] for r in rows))
    row_h, left = 24, 150
    h = 10 + len(rows) * row_h + 44
    p = Plot(W, h, (0, hi), (0, 1), left=left, right=60, top=10, bottom=44, xticks=ticks, yticks=[],
             xlabel="F0-reported fused tracks per iteration")
    y = 10
    for key, n, nr, lab, med in rows:
        p.add(f'<text class="tick" x="{left - 8}" y="{y + 16}" text-anchor="end">{esc(("eval " if key.startswith("evaluation") else "test ") + short(key))}</text>')
        x0, x1 = p.X(0), p.X(nr)
        if nr:
            p.add(f'<path class="s1" d="{EC.bar_path(x0, y + 5, x1 - x0, row_h - 10)}" data-tip="{esc(key)}: {nr} reported of {n} fused tracks" tabindex="0"/>')
        p.add(f'<text class="val" x="{x1 + 6:.1f}" y="{y + 16}">{nr}</text>')
        y += row_h
    tab = [[k, n, nr, "–" if lab is None else lab, f4(med, 1)] for k, n, nr, lab, med in rows]
    return p.svg("Reported fused tracks per iteration"), tab


# --------------------------------------------------------------------------- #
# Markdown bits (probes.md, the plan, deanonymize.py's docstring)
# --------------------------------------------------------------------------- #


def md_inline(s: str) -> str:
    s = esc(re.sub(r"\binf\b", "∞", s))          # quoted 'hold = inf' -> the symbol (check 8 scans for 'inf')
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)


def md_section(text: str, heading: str) -> str:
    m = re.search(rf"^#+ {re.escape(heading)}\s*$(.*?)(?=^#+ |\Z)", text, re.M | re.S)
    return m.group(1).strip() if m else ""


def md_render(block: str) -> str:
    out, tbl = [], []
    def flush():
        if tbl:
            rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in tbl if not re.match(r"^\|[\s|:-]+\|$", ln.strip())]
            if rows:
                hdr = "".join(f"<th>{md_inline(c)}</th>" for c in rows[0])
                body = "".join("<tr>" + "".join(f"<td>{md_inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
                out.append(f'<div class="tw"><table><thead><tr>{hdr}</tr></thead><tbody>{body}</tbody></table></div>')
            tbl.clear()
    for ln in block.splitlines():
        if ln.strip().startswith("|"):
            tbl.append(ln)
            continue
        flush()
        if ln.strip():
            out.append(f'<p class="mdp">{md_inline(ln.strip())}</p>')
    flush()
    return "".join(out)


def deanon_commands() -> str:
    p = TOOLS / "deanonymize.py"
    text = src(p).read_text(encoding="utf-8")
    m = re.search(r"Owner's commands.*?\n(.*?)\nRules:", text, re.S)
    return m.group(1).rstrip() if m else ""


def plan_phases(var, probes) -> list[list]:
    if not PLAN.exists():
        return []
    text = src(PLAN).read_text(encoding="utf-8")
    secs = re.findall(r"^## (Phase [0-9a-z]+): (.*?)$(.*?)(?=^## |\Z)", text, re.M | re.S)
    verif = md_section(text, "Verification")
    def status(ph: str) -> str:
        n = ph.split()[1]
        exists = lambda p: (C.REPO / p).exists()  # noqa: E731
        if n == "0":
            return "done" if "CUAS_DERIVED" in (TOOLS / "cuas_common.py").read_text() else "not started"
        if n == "1":
            if "v3h" in var and "v3a" in var:
                return "measured (variants promoted)"
            return "in progress" if "--corrections" in (TOOLS / "sensor_tracks.py").read_text() else "in progress (code not landed)"
        if n == "2":
            return "this coordinator page (first honest numbers wait for Phase 1)"
        if n == "3":
            have = [t for t in ("radar_model", "ablation", "health") if exists(f"tools/{t}.py")]
            outs = [t for t in ("radar_model", "health") if (DER / t).exists()]
            return f"in progress: tools present {', '.join(have) or 'none'}; outputs present {', '.join(outs) or 'none'}"
        if n == "4":
            return "this report (sections marked pending fill in as Phase 1 / Phase 3 outputs appear)"
        if n == "4b":
            return ("outputs present in data/derived/compact/ (" + ("tools/compact.py present" if (TOOLS / "compact.py").exists() else "tool not found") + ")"
                    if (DER / "compact").exists() else "not started")
        if n == "6":
            return "started" if (C.SUBMISSIONS / "F2_fuse").exists() else "not started"
        if n == "7":
            return ("probe files built, P_shift15 present" if "P_shift15" in probes else "probe files built (P_shift15 not yet)") + "; upload waits for approval"
        return "not started"
    rows = []
    for ph, title, body in secs:
        lines = [ln.strip() for ln in body.strip().splitlines() if ln.strip()]
        prose = [ln for ln in lines if not ln.startswith("|")]
        goal = re.sub(r"^(?:\d+\.|-)\s*", "", prose[0]) if prose else "a table of units: see the plan"
        acc = []
        for i, ln in enumerate(lines):
            if re.search(r"\bAcceptance\b|\bAccepted\b", ln):
                acc.append(re.sub(r"^(?:\d+\.|-)\s*", "", ln))
                if ln.rstrip("*").endswith(":"):
                    acc += [x.lstrip("- ").strip() for x in lines[i + 1:i + 6] if x.startswith("- ")]
        if not acc:
            vm = re.search(rf"\*\*{re.escape(ph)}[^*]*\*\*(.*?)(?=^- \*\*Phase|\Z)", verif, re.S | re.M)
            if vm:
                acc = [ln.strip().lstrip("-* ").strip() for ln in vm.group(1).splitlines() if ln.strip().lstrip("-* ").strip()]
        acc_s = " ".join(acc)
        rows.append([ph, Raw(md_inline(title)), Raw(md_inline(goal[:260] + ("…" if len(goal) > 260 else ""))),
                     Raw(md_inline(acc_s[:420] + ("…" if len(acc_s) > 420 else "")) or "see the plan"), status(ph)])
    return rows


# --------------------------------------------------------------------------- #
# Page shell
# --------------------------------------------------------------------------- #

CSS_EXTRA = """
.kpis{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px;margin:12px 0}
.kpi{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:12px 14px}
.kpi .v{font-size:28px;font-weight:600;color:var(--text-primary);line-height:1.2}.kpi .l{font-size:13px;color:var(--text-secondary)}
.badge{display:inline-flex;align-items:center;gap:4px;font-size:12px;border:1px solid var(--border);border-radius:6px;padding:0 6px;margin-left:4px;color:var(--text-primary);background:var(--surface-1)}
.badge::before{content:"";display:inline-block;width:8px;height:8px;border-radius:2px}
.badge.warn::before{background:#fab219}.badge.crit::before{background:#d03b3b}.badge.ok::before{background:#0ca30c}
.src{font-size:12px;color:var(--muted)} ul.find li{margin:8px 0;color:var(--text-primary)} ol.q li,ul.plain li{margin:6px 0;color:var(--text-primary)}
.pending{border-style:dashed} p.mdp{margin:4px 0}
pre{white-space:pre-wrap;font-size:12.5px;background:var(--page);border:1px solid var(--border);border-radius:8px;padding:10px;overflow-x:auto;color:var(--text-primary)}
.ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}.l1{stroke:var(--series-1)}.l2{stroke:var(--series-2)}.l3{stroke:var(--series-3)}.l4{stroke:var(--series-4)}
.mk-ink{fill:var(--text-secondary)}.mk-ring{fill:none;stroke:var(--text-primary);stroke-width:1.5}
.tick-med{stroke:var(--text-primary);stroke-width:2}
.note{border-left:3px solid #fab219;padding:6px 12px;background:var(--surface-1);border-radius:6px;margin:10px 0}
#asof{font-size:13px;color:var(--muted)} a{color:var(--series-1)}
"""


def page(title: str, body: list[str], asof: str, css: str = "") -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{esc(title)}</title><style>{EC.CSS}{CSS_EXTRA}{css}</style></head><body><div class="viz-root"><main>'
            f'<header><button class="toggle" id="theme" type="button">Light / dark</button><h1>{esc(title)}</h1>'
            f'\n<p id="asof">{esc(asof)}</p>\n</header>{"".join(body)}</main></div>'
            f'<div id="tip" role="status"></div><script>{EC.JS}</script></body></html>')


def kpi(value: str, label: str) -> str:
    return f'<div class="kpi"><div class="v">{esc(value)}</div><div class="l">{label}</div></div>'


# --------------------------------------------------------------------------- #
# Coordinator page
# --------------------------------------------------------------------------- #


def iteration_note(k: str, eval_scen: set) -> str:
    sc = int(short(k)[1])
    if sc in eval_scen:
        return "◆ same scenario as evaluation"
    if sc == 3:
        return "S3 group (correlated)"
    return ""


def coordinator_body(D: dict) -> list[str]:
    fs, co = D["findings"]["findings"], D["findings"]["coordinator"]
    sc, var, keys, es = D["scores"], D["var"], D["keys"], D["eval_scen"]
    f0 = sc["F0_fuse"]
    honest = "v3h" in var
    opt = "" if honest else ' <span class="badge warn">optimistic, correction in progress</span>'
    B = ['<div class="note">' + N(
        "Every number below is a <b>local estimate</b> on the labelled testing iterations, made with our own copy of the scoring rules (the official scorer is not published). "
        "Numbers that still rest on calibration fitted on all testing iterations (in-sample) carry the badge {b}.",
        b=Raw('<span class="badge warn">optimistic, correction in progress</span>')) + "</div>"]
    B.append("<h2>" + N("What we built") + "</h2>" + '<section class="card">' + ctl_list(co.get("built", []), fs) + "</section>")
    mean_all = float(np.mean([f0["all"][k]["combined"] for k in keys]))
    mean_air = float(np.mean([f0["airborne"][k]["combined"] for k in keys]))
    tiles = [kpi(f"{mean_all:.3f}", N("F0 held-out mean, truth 'all' (parked drones count)") + opt),
             kpi(f"{mean_air:.3f}", N("F0 held-out mean, truth 'airborne'") + opt)]
    for tag in ("v3h", "v3a"):
        if tag in var:
            m = var[tag]
            r = wlt(sc, m, "F0_fuse", "all", keys)
            tiles.append(kpi(f"{np.mean([sc[m]['all'][k]['combined'] for k in keys]):.3f}",
                             N("{m} mean, truth 'all'; beats F0 on {w} of {n}", m=m, w=r["w"], n=r["n"])))
        else:
            tiles.append(kpi("pending", N("{m}: produced by {p}", m="F0_fuse_" + tag, p=VARIANT_PRODUCER[tag])))
            ST.pending.append(f"coordinator KPI F0_fuse_{tag} (pending: produced by {VARIANT_PRODUCER[tag]})")
    B.append("<h2>" + N("Held-out scores per testing iteration") + "</h2>" + '<div class="kpis">' + "".join(tiles) + "</div>")
    hdr = ["iteration", "note", "F0 all", "F0 airborne"]
    present = [t for t in ("v3h", "v3a") if t in var]
    for t in present:
        hdr += [f"{t} all", f"{t} − F0 all", f"{t} airborne", f"{t} − F0 airborne"]
    rows = []
    for k in keys:
        r = [short(k), iteration_note(k, es), f4(f0["all"][k]["combined"]), f4(f0["airborne"][k]["combined"])]
        for t in present:
            m = var[t]
            for mode in MODES:
                r += [f4(sc[m][mode][k]["combined"]), sgn(sc[m][mode][k]["combined"] - f0[mode][k]["combined"])]
        rows.append(r)
    mean_row = ["mean (context only)", "", f4(mean_all), f4(mean_air)]
    for t in present:
        m = var[t]
        for mode in MODES:
            mv = float(np.mean([sc[m][mode][k]["combined"] for k in keys]))
            mean_row += [f4(mv), sgn(mv - float(np.mean([f0[mode][k]["combined"] for k in keys])))]
    rows.append(mean_row)
    wl = []
    for t in present:
        for mode in MODES:
            r = wlt(sc, var[t], "F0_fuse", mode, keys)
            wl.append(f"<li>{esc(var[t])} vs F0, truth '{esc(mode)}': {esc(wlt_text(r))}; per scenario {esc(scen_text(r))}</li>")
    wl_html = ('<ul class="plain">' + "".join(wl) + "</ul>") if wl else (
        '<p class="what">' + N("The honest variants {a} and {b} are not promoted yet, so there is no comparison with F0 yet.",
                               a="F0_fuse_v3h", b="F0_fuse_v3a") + "</p>")
    B.append('<section class="card">' + plain_table(hdr, rows) + wl_html +
             '<p class="how">' + N("<b>How to read it:</b> one row per labelled testing iteration; the score is between zero and one, higher is better. "
                                   "{m} marks iterations from the same scenarios as the evaluation data sets; the three S3 iterations are correlated, "
                                   "so the scenario counts matter more than the iteration count.", m="◆") + "</p></section>")
    B.append("<h2>" + N("In progress") + "</h2>" + '<section class="card">' + ctl_list(co.get("in_progress", []), fs) + "</section>")
    B.append("<h2>" + N("Needs approval") + "</h2>" + '<section class="card">' + ctl_list(co.get("approvals", []), fs) +
             (f'<p class="what">{finding_text(next(f for f in fs if f["id"] == "uploads_blocked"))}</p>'
              if any(f["id"] == "uploads_blocked" for f in fs) else "") + "</section>")
    B.append("<h2>" + N("Questions for the organisers, in priority order") + "</h2>" + '<section class="card">' +
             ctl_list(co.get("questions", []), fs, "ol", "q") + "</section>")
    B.append("<h2>" + N("Leaderboard probe plan") + "</h2>" + '<section class="card">' + probe_plan_lines(D) +
             '<p class="src">' + N("Details: {p} and the technical report.", p="data/derived/probes/probes.md") + "</p></section>")
    return B


def probe_plan_lines(D: dict) -> str:
    probes, text = D["probe_names"], D["probes_md"]
    shift = next((p for p in sorted(probes) if p.startswith("P_shift")), None)
    off = re.search(r"\d+", shift or "P_shift15").group(0)
    # P_park at the smallest c in probes.csv (proxy labels, p = 2, 3-D): on how many testing
    # iterations does it go up under 'all' (the separation it is read for)?
    small = sorted({float(r["c"]) for r in D["probes"] if r["probe"] == "P_park" and r["p"] in ("2", "2.0") and r["dims"] in ("3", "3.0")})
    park_txt = ""
    if small:
        c0 = small[0]
        ds = [float(r["d_combined"]) for r in D["probes"] if r["probe"] == "P_park" and r["label_rule"] == "proxy"
              and r["truth_mode"] == "all" and float(r["c"]) == c0 and r["p"] in ("2", "2.0") and r["dims"] in ("3", "3.0")]
        park_txt = N(" At c = {c} m it goes up under truth 'all' on only {k} of {n} testing iterations.",
                     c=f"{c0:g}", k=sum(d > 0 for d in ds), n=len(ds))
    l1 = N("<b>First</b>, on each evaluation data set: {p0} (F0 unchanged, the reference), then {meta} (best-guess metadata instead of 'Unknown') "
           "and {ident} ('Unknown' instead of 'Hostile' for non-whitelisted drones), one factor per upload.",
           p0="P0_f0", meta="P_meta", ident="P_ident")
    l2 = N("<b>Then</b> {s}: F0 with every position shifted by a known {o} m; the size of the drop brackets the unknown cut-off c. Status: {st}.",
           s=shift or "P_shift15", o=off, st="built" if shift else "planned, not built yet")
    l3 = N("<b>Last</b> {p}: adds parked-drone rows; it separates 'parked drones scored' from 'airborne only' only for a large c, so it is read after {s} has bracketed c.{t}",
           p="P_park", s=shift or "P_shift15", t=Raw(park_txt))
    return f'<ol class="q"><li>{l1}</li><li>{l2}</li><li>{l3}</li></ol>'


# --------------------------------------------------------------------------- #
# Technical report
# --------------------------------------------------------------------------- #


def report_body(D: dict) -> list[str]:
    fs = D["findings"]["findings"]
    sc, meta, var, keys, es = D["scores"], D["meta"], D["var"], D["keys"], D["eval_scen"]
    f0 = sc["F0_fuse"]
    honest = "v3h" in var
    B: list[str] = []
    sec = [0]

    def h2(title: str) -> str:
        sec[0] += 1
        return f"<h2>{sec[0]}. {N(title)}</h2>"

    B.append('<div class="note">' + N(
        "The technical appendix. Scores are our proxy of the official scoring on the labelled testing iterations, re-scored from the current files. "
        "{b} marks a claim verified only in a workflow journal; {o} marks numbers that rest on in-sample calibration.",
        b=Raw('<span class="badge warn">journal only</span>'), o=Raw('<span class="badge warn">optimistic, correction in progress</span>')) + "</div>")

    # 1 executive summary
    opt = "" if honest else ' <span class="badge warn">optimistic, correction in progress</span>'
    mean = lambda m, mode: float(np.mean([sc[m][mode][k]["combined"] for k in keys]))  # noqa: E731
    tiles = [kpi(f"{mean('F0_fuse', 'all'):.3f}", N("F0 mean combined, truth 'all'") + opt),
             kpi(f"{mean('F0_fuse', 'airborne'):.3f}", N("F0 mean combined, truth 'airborne'") + opt)]
    if "O3_fuse_oracle" in sc:
        tiles.append(kpi(f"{mean('O3_fuse_oracle', 'all'):.3f}", N("O3 reference (F0's tracks chosen with the truth), truth 'all'")))
    if "F1_fuse" in sc:
        r = wlt(sc, "F1_fuse", "F0_fuse", "all", keys)
        tiles.append(kpi(f"{r['w']} of {r['n']}", N("iterations where F1 beats F0 (truth 'all')")))
    for tag in ("v3h", "v3a"):
        if tag in var:
            tiles.append(kpi(f"{mean(var[tag], 'all'):.3f}", N("{m} mean, truth 'all'", m=var[tag])))
        else:
            tiles.append(kpi("pending", N("{m}: produced by {p}", m="F0_fuse_" + tag, p=VARIANT_PRODUCER[tag])))
            ST.pending.append(f"report KPI F0_fuse_{tag} (pending: produced by {VARIANT_PRODUCER[tag]})")
    story_ids = ["f0_mean", "f0_cost_air", "unseen_airborne", "az_rotation_gain", "rules_weights_only"]
    story = " ".join(finding_text(f) + badge(f.get("status", "")) for i in story_ids for f in fs if f["id"] == i)
    for f in fs:
        if f["id"] in story_ids:
            ST.rendered_findings.add(f["id"])
    B.append(h2("Executive summary") + '<div class="kpis">' + "".join(tiles) + "</div>" +
             card(N("The story so far, from the verified findings"), story,
                  how=N("each sentence is one finding of {f}; its source is listed in the sections below.", f="findings.json")))

    # 2 per-iteration scores
    svg, leg, rows = dotplot(sc, meta, var, keys, es)
    B.append(h2("Scores per testing iteration"))
    B.append(card(N("Combined score of every method on every testing iteration, both truth modes"),
                  N("One row per labelled testing iteration ({m} = same scenario as an evaluation data set). Coloured dots are F0 and its corrected variants; "
                    "grey shapes are the other methods; the open ring is the O3 reference, which uses the truth to choose tracks and is not a ceiling.", m="◆"),
                  svg, leg + table(["method", "truth", *[short(k) for k in keys], "mean (context)"], rows) +
                  table(["method", "role", "where its numbers come from"], [[m, meta[m]["role"], meta[m]["source"]] for m in meta], "Method sources"),
                  how=N("further right is better. Compare a method with F0 within one row: the verdict is how many rows it is to the right, not the mean.")))
    for tag in ("v3h", "v3a"):
        if tag not in var and not any(m in UNREADABLE for m in dict(VARIANTS)[tag]):
            B.append(pending_card(N("Corrected variant {m} (not promoted yet)", m="F0_fuse_" + tag), VARIANT_PRODUCER[tag]))
    mat = []
    for m in meta:
        if m == "F0_fuse":
            continue
        cells = [m, meta[m]["role"]]
        for mode in MODES:
            r = wlt(sc, m, "F0_fuse", mode, keys)
            cells += [wlt_text(r), scen_text(r)]
        mat.append(cells)
    B.append(card(N("Win / loss matrix against F0 (combined score)"),
                  N("Counts over the testing iterations: a win is a combined score above F0's on that iteration (a tie is a difference below {t}).", t=f"{CP.TIE:g}"),
                  extra=plain_table(["method", "role", "truth 'all'", "per scenario ('all')", "truth 'airborne'", "per scenario ('airborne')"], mat) +
                  table(["method", "truth", *[short(k) for k in keys], "mean (context)"],
                        [[m, mode] + [f4(sc[m][mode][k]["combined"]) for k in keys] + [f4(mean(m, mode))] for m in meta for mode in MODES],
                        "Data table: combined score of every method in the matrix"),
                  how=N("references and baselines are listed for orientation; only deployable methods compete with F0. With {n} iterations the smallest possible sign-test p is {p}.",
                        n=len(keys), p=f"{CP.sign_test_p(len(keys), 0):.3f}")))
    B.append(findings_list(fs, sections=("scores",)) + f1_note(sc, keys))

    # 3 where the points go
    svg3, rows3, leg3 = cost_share_chart(f0, keys)
    B.append(h2("Where F0 loses its points"))
    B.append(card(N("What F0's GOSPA cost is made of, per iteration"),
                  N("The track part's cost splits into localisation error of matched rows, drones we miss and rows that match no drone."),
                  svg3, leg3 + table(["truth", "iteration", "localisation", "missed", "false", "sum"], rows3),
                  how=N("a long orange part means coverage is the problem; a long aqua part means false tracks; a long blue part means positions are off.")))
    svg3b, rows3b = recall_minis(f0, keys)
    B.append(card(N("F0's four score parts per iteration"),
                  N("Track part and the three label recalls (a missed drone counts as wrong in every recall)."),
                  svg3b, EC.legend([("s1", "truth 'all'"), ("s2", "truth 'airborne'")]) +
                  table(["truth", "iteration", *PARTS, "combined"], rows3b),
                  how=N("each small chart is one part, zero to one; blue = truth 'all', orange = truth 'airborne'.")))
    B.append(findings_list(fs, ids=("f0_cost_air", "unseen_airborne", "offset_rows", "under_merge")))

    # 4 value of each fix
    items = []
    if "O3_fuse_oracle" in sc:
        r = wlt(sc, "O3_fuse_oracle", "F0_fuse", "all", keys)
        g = mean("O3_fuse_oracle", "all") - mean("F0_fuse", "all")
        items.append((N("report exactly the real-drone tracks (O3 minus F0)"), g,
                      f"computed from the re-score; O3 beats F0 on {r['w']} of {r['n']}"))
    lab = {("oracle_gains", "en"): "true horizontal position", ("oracle_gains", "u"): "true height",
           ("oracle_gains", "both"): "true horizontal position and height", ("oracle_gains", "best"): "best member height only",
           **{("bias_oracle", k): f"remove each track's slow bias (on O3, c = {k[1:]} m)"
              for k in sorted((next((f.get("values", {}) for f in fs if f["id"] == "bias_oracle"), {})), reverse=True)}}
    fx_rows = [[it[0], sgn(it[1], 3), "re-score", ""] for it in items]
    for (fid, key), l in lab.items():
        v = fval(fs, fid, key)
        if v is None:
            continue
        st = next(f["status"] for f in fs if f["id"] == fid)
        tag = {"journal": " [journal only]", "unverified": " [unverified]"}.get(st, "")
        items.append((l + tag, num_in(v), f"{l}: {v} ({fid}, status {st})"))
        fx_rows.append([l, v, fid, st])
    B.append(h2("Value of each fix (oracle gains)"))
    B.append(card(N("How much the combined score would gain if one error source were removed perfectly"),
                  N("Oracle rows use the truth, so a real correction recovers only a part of each bar. Bars with a journal tag rest on a workflow journal, not on a file."),
                  hbar_chart(items, f"gain in combined score (truth 'all', c = {SC.CONFIGS[0][1]:g} m unless stated)"),
                  table(["fix", "gain", "source", "evidence"], fx_rows),
                  how=N("longer bar = more points at stake. Compare bars, not their sum: the fixes overlap.")))
    B.append(findings_list(fs, ids=("oracle_gains", "bias_oracle")))

    # 5 score vs c
    B.append(h2("How the score depends on the unknown cut-off c"))
    svc = score_vs_c(D["sens"], keys)
    if svc:
        svg5, leg5, rows5 = svc
        B.append(card(N("Mean combined score versus the GOSPA cut-off c (p = {p}, 3-D)", p=f"{SC.CONFIGS[0][2]:g}"),
                      N("The organisers do not publish c. F0, the O3 reference, F1 and the bias-removal oracle, each averaged over the testing iterations."),
                      svg5, leg5 + table(["truth", "series", "c (m)", "mean", *[short(k) for k in keys]], rows5),
                      how=N("read along the x-axis: if the official c is small, every method collapses and accuracy (radar bias) dominates; the gap between the O3 line and the oracle line is what bias removal could add.")))
    else:
        B.append(pending_card(N("Score versus c"), "gospa_sensitivity.py"))
    B.append(findings_list(fs, ids=("gospa_c",)))

    # 6 radar
    B.append(h2("Radar calibration: the azimuth bias is stable, the east / north offsets are not"))
    rsvg, rrows, _ = radar_minis(D["cal"], D["offs"], keys, D["deployable"])
    B.append(card(N("Per radar: azimuth bias per iteration (left) next to the east / north offset (right)"),
                  N("Left: the angle to add to the measured azimuth, from matched drone rows (sign: truth minus measured). "
                    "Right: the radar's median position offset (radar minus truth). Grey line: pooled azimuth."),
                  rsvg, EC.legend([("s1", "east offset"), ("s2", "north offset")]) +
                  table(["radar", "iteration", "azimuth bias (deg, add)", "rows", "east (m)", "north (m)"], rrows),
                  how=N("left dots on one level = a stable rotation that one pooled correction can remove; right dots jumping across zero = offsets that do not transfer between iterations.")))
    B.append(findings_list(fs, sections=("calibration",)))
    B.append(registration_card())
    rm = DER / "radar_model"
    if rm.exists() and any(rm.glob("*.csv")):
        B.append(generic_csv_card(N("Radar error model (latency, azimuth, range terms)"), rm, "radar_model.py"))
    else:
        B.append(pending_card(N("Radar error model in sensor coordinates"), "radar_model.py"))

    # 7 health
    B.append(h2("Testing versus evaluation health"))
    hd = DER / "health"
    if hd.exists() and any(hd.glob("*.json")):
        B.append(generic_json_card(N("Evaluation health indicators"), hd, "health.py"))
        B.append(card(N("Earlier evaluation sanity findings"), N("Measured before the health indicators existed; kept for comparison."),
                      extra=findings_list(fs, sections=("health",))))
    else:
        B.append(pending_card(N("Pre-registered health indicators (duplicates, jumps, friendly copies)"), "health.py",
                              extra='<p class="what">' + N("Until then, the earlier evaluation sanity findings:") + "</p>" + findings_list(fs, sections=("health",))))
    fc = fused_counts([it.key for it in C.iterations()])
    if fc:
        B.append(card(N("Reported fused tracks per iteration, testing and evaluation"),
                      N("A ground-truth-free size check: how many fused tracks F0 reports in each iteration."),
                      fc[0], table(["iteration", "fused tracks", "reported by F0", "reported and drone-labelled (testing)", "median duration of reported (s)"], fc[1]),
                      how=N("evaluation iterations with many more reported tracks than testing ones point at duplicates or clutter (see the findings above).")))

    # 8 sensor presence
    svg8, tab8 = presence_grid(D["deployable"])
    B.append(h2("Which sensors exist in which iteration"))
    n_eval = sum(1 for it in C.iterations() if it.tree == "evaluation")
    B.append(card(N("Sensor presence grid: sensors by the {n} iterations", n=len(C.iterations())),
                  N("A filled cell means the sensor has at least one file in that iteration. Testing-only sensors cannot help on evaluation; UTM relays repeat the drones' own positions; bearing-only sensors give a direction, no distance."),
                  svg8, table(tab8[0], tab8[1:]),
                  how=N("read a row to see where a sensor is available; the last {n} columns are the evaluation data sets that get submitted.", n=n_eval)))

    cd = DER / "compact"
    if cd.exists() and any(cd.iterdir()):
        crow = []
        for p in sorted(cd.iterdir()):
            if p.is_file():
                if p.suffix == ".csv":
                    with src(p, "compact.py").open(newline="", encoding="utf-8") as fh:
                        rd = csv.reader(fh)
                        ncol = len(next(rd, []))
                        nrow = sum(1 for _ in rd)
                    crow.append([p.name, f"{nrow:,}", ncol, mtime_str(p)])
                else:
                    src(p, "compact.py")
                    crow.append([p.name, "–", "–", mtime_str(p)])
        B.append(card(N("Compact dataset"), N("The documented tables of {d} (built by {t}); column meanings are in its README.", d="data/derived/compact/", t="tools/compact.py")
                      + stale_badge(cd, "compact.py"), extra=plain_table(["file", "rows", "columns", "modified"], crow)))
    else:
        B.append(pending_card(N("Compact dataset and data atlas"), "compact.py"))

    # 9 triangulation
    B.append(h2("Triangulation of the drones no position sensor sees"))
    if D["tri"]:
        svg9, tab9 = triangulation_chart(D["tri"], keys)
        B.append(card(N("How much unseen airborne truth bearing sensors could recover"),
                      N("Triangulation needs bearings from two different places. Per iteration: the unseen share, and the parts of it hit by bearings from one site and from two or more sites."),
                      svg9, table(["iteration", "truth", "truth rows", "unseen", "unseen share", "hit from one site", "hit from two or more sites", "recoverable (share_rec50)"], tab9),
                      how=N("the aqua bar (two sites) is what triangulation could use; it is essentially zero because the direction finders stand together.")))
    else:
        B.append(pending_card(N("Triangulation"), "triangulation_oracle.py"))
    B.append(findings_list(fs, sections=("triangulation",)))

    # 10 probes
    B.append(h2("Leaderboard probes"))
    if D["probes"]:
        s10, t10, _ = probe_strips(D["probes"], keys)
        B.append(card(N("Predicted change of each probe against P0, per testing iteration and scoring hypothesis"),
                      N("Each probe is F0 with one factor changed. A probe is informative when its direction differs between hypotheses and agrees across iterations."),
                      s10, table(["probe", "truth", "labels", "GOSPA", *[short(k) for k in keys], "direction"], t10),
                      how=N("dots right of the zero line = the probe should score higher than P0 under that hypothesis; the dark tick is the median iteration.")))
    else:
        B.append(pending_card(N("Probe predictions"), "probes.py"))
    dt, ck = md_section(D["probes_md"], "Decision table"), md_section(D["probes_md"], "Upload checklist")
    B.append(card(N("Decision table and upload checklist"), N("Copied from {p}.", p="data/derived/probes/probes.md"),
                  extra=md_render(dt) + md_render(ck) + "<h3>" + N("De-anonymisation, the owner's step before uploading") + "</h3>" +
                  f"<pre>{esc(deanon_commands())}</pre>" + '<p class="src">' + N("Commands from the docstring of {t}.", t="tools/deanonymize.py") + "</p>"))
    B.append(probe_plan_lines(D))
    B.append(findings_list(fs, sections=("probes",)))

    # 11 scoring semantics
    B.append(h2("Scoring semantics, labels and metadata"))
    wt = [[k, f"{v:g}"] for k, v in SC.WEIGHTS.items()]
    cf = [[n, f"{c:g}", f"{p:g}", f"{d}-D", "primary" if n == SC.PRIMARY else ""] for n, c, p, d in SC.CONFIGS]
    B.append(card(N("How the local proxy scores a submission"),
                  N("Weights of the four parts and the GOSPA configurations that every re-score above computes (from {s}).", s="tools/score.py"),
                  extra=plain_table(["part", "weight"], wt) + plain_table(["configuration", "c (m)", "p", "dims", ""], cf)))
    B.append(findings_list(fs, sections=("scoring", "labels", "data")))

    # 12 decisions and questions
    co = D["findings"]["coordinator"]
    B.append(h2("Decisions and open questions"))
    B.append(card(N("Decisions taken"), N("What was shelved or not applied, and why."),
                  extra=findings_list(fs, ids=("f1_lost", "triangulation_zero", "enu_offsets_fail", "uploads_blocked")) + f1_note(sc, keys)))
    B.append(card(N("Needs approval"), "", extra=ctl_list(co.get("approvals", []), fs)))
    B.append(card(N("Questions for the organisers, in priority order"), "", extra=ctl_list(co.get("questions", []), fs, "ol", "q")))
    B.append(card(N("In progress"), "", extra=ctl_list(co.get("in_progress", []), fs)))
    for m, why in sorted(UNREADABLE.items()):
        B.append(pending_card(N("{m}: its files could not be read ({w})", m=m, w=why), "the tool writing them; re-run this report"))
    for tag in ("noutm", "noutm_input"):
        if tag not in var and not any(m in UNREADABLE for m in dict(VARIANTS)[tag]):
            B.append(pending_card(N("F0 without the UTM relays ({t})", t=tag), VARIANT_PRODUCER[tag]))
    rest = [f for f in fs if f["id"] not in ST.rendered_findings]
    if rest:
        B.append(card(N("Other findings"), "", extra='<ul class="find">' + "".join(finding_li(f) for f in rest) + "</ul>"))

    # 13 future actions
    B.append(h2("Proposed future actions (the approved plan)"))
    ph = plan_phases(var, D["probe_names"])
    if ph:
        B.append(card(N("Plan phases with goal, acceptance rule and status"),
                      N("From the approved plan {p}; the status column is computed from what exists on disk now.", p="~/.claude/plans/wiggly-napping-melody.md"),
                      extra=plain_table(["phase", "title", "goal (first line)", "acceptance rule", "status now"], ph)))
    else:
        B.append(pending_card(N("Plan phases"), "~/.claude/plans/wiggly-napping-melody.md"))

    # 14 wave log
    B.append(h2("Wave log"))
    B.append(card(N("What each wave built and learned"),
                  N("Grouping of the tools into waves reconstructed from the order in {c}; approximate. Learned = the findings of that wave.", c="CLAUDE.md"),
                  extra=plain_table(["wave", "tools (present on disk)", "learned"], wave_rows(fs, var))))

    # 15 glossary
    B.append(h2("Glossary"))
    B.append('<section class="card"><dl class="gloss">' + "".join(f"<dt>{esc(a)}</dt><dd>{b}</dd>" for a, b in glossary()) + "</dl></section>")

    # 16 provenance
    B.append(h2("Provenance"))
    B.append(provenance_card(D))
    return B


F1_ND = 3   # decimals of the score table the f1_lost finding was counted from (track_existence.md)


def rounded_wlt(sc, a: str, b: str, mode: str, keys: list[str], nd: int = F1_ND) -> tuple[int, int, int]:
    """Win / loss / tie of a vs b with both scores rounded to nd decimals (ties = equal when printed)."""
    w = l = t = 0
    for k in keys:
        x, y = float(f"{sc[a][mode][k]['combined']:.{nd}f}"), float(f"{sc[b][mode][k]['combined']:.{nd}f}")
        if x == y:
            t += 1
        elif x > y:
            w += 1
        else:
            l += 1
    return w, l, t


def f1_note(sc, keys) -> str:
    """Why the f1_lost finding's counts and the matrix's counts can differ (tie rule)."""
    if "F1_fuse" not in sc:
        return ""
    parts, diff = [], []
    for mode in MODES:
        r = wlt(sc, "F1_fuse", "F0_fuse", mode, keys)
        w3, l3, t3 = rounded_wlt(sc, "F1_fuse", "F0_fuse", mode, keys)
        parts.append(N("truth '{m}': {a} rounded, {b} unrounded", m=mode, a=f"{w3} win / {l3} loss / {t3} tie",
                       b=f"{r['w']} win / {r['l']} loss / {r['t']} tie"))
        for k in keys:
            x, y = sc["F1_fuse"][mode][k]["combined"], sc["F0_fuse"][mode][k]["combined"]
            tie_r = f"{x:.{F1_ND}f}" == f"{y:.{F1_ND}f}"
            if tie_r != (abs(x - y) < CP.TIE):
                diff.append(f"{short(k)} '{mode}' (F1 {x:.4f}, F0 {y:.4f})")
    return ('<p class="src">' + N("Two counts of F1 against F0: the finding above counts its source's table, where scores are printed with {nd} decimals "
                                  "and equal printed scores are a tie; the matrix counts the unrounded current files with the tie rule |difference| below {t}. "
                                  "From the current files: {p}. Iterations where the two rules disagree: {d}.",
                                  nd=F1_ND, t=f"{CP.TIE:g}", p="; ".join(parts), d=", ".join(diff) or "none") + "</p>")


def registration_card() -> str:
    """The azimuth-rotation gain as radar_registration.py writes it (file-backed)."""
    path = DER / "radar_registration" / "registration.md"
    if not path.exists():
        return pending_card(N("Azimuth rotation gain (file)"), "radar_registration.py")
    text = src(path, "radar_registration.py").read_text(encoding="utf-8")
    rx = re.compile(r"\b([A-Z]{4}) (\d+(?:\.\d+)?) -> (\d+(?:\.\d+)?) m \((\d+)% removed, (\d+) iterations")
    rows = [[m.group(1), m.group(2), m.group(3), m.group(4) + " %", m.group(5)] for m in rx.finditer(text)]
    if not rows:
        return pending_card(N("Azimuth rotation gain (file): the summary line of {p} was not found", p=rel(path)), "radar_registration.py")
    return card(N("Azimuth rotation gain as {p} prints it (file-backed)", p=rel(path)),
                N("Median over the testing iterations of each radar's median horizontal error: no correction, then rotated back by the leave-one-out "
                  "mean azimuth bias (the bias of the other testing iterations). This is the reproducible version; the journal finding above used another statistic."),
                extra=plain_table(["radar", "no correction (m)", "rotated (m)", "removed", "iterations"], rows),
                how=N("a large drop means the error is mostly a stable rotation, which one pooled angle removes; a small drop means something else dominates."))


def stale_badge(d: Path, producer: str) -> str:
    tool = TOOLS / producer
    old = [p for p in d.iterdir() if p.is_file() and tool.exists() and p.stat().st_mtime < tool.stat().st_mtime]
    return (' <span class="badge warn">stale: older than the tool, which is still being changed</span>' if old else "")


UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def generic_csv_card(title, d: Path, producer) -> str:
    parts = []
    for p in sorted(d.glob("*.csv")):
        rows = read_csv(p, producer)
        if rows:
            parts.append(table(list(rows[0].keys()), [[UUID_RE.sub("[uuid]", str(v)) for v in r.values()] for r in rows[:80]],
                               f"{p.name} ({len(rows)} rows, first 80)"))
    return card(title, N("Raw outputs of {t}, shown as tables until a chart is designed for them.", t=producer) + stale_badge(d, producer),
                extra="".join(parts))


def generic_json_card(title, d: Path, producer) -> str:
    parts = []
    for p in sorted(d.glob("*.json")):
        obj = json.loads(src(p, producer).read_text(encoding="utf-8"))
        flat = []
        def walk(o, pre=""):
            if isinstance(o, dict):
                for k in sorted(o):
                    walk(o[k], f"{pre}.{k}" if pre else str(k))
            elif isinstance(o, list) and len(o) <= 40:
                for i, v in enumerate(o):
                    walk(v, f"{pre}[{i}]")
            else:
                v = json.dumps(o)[:200] if not isinstance(o, str) else o[:200]
                flat.append([pre, UUID_RE.sub("[uuid]", v)])
        walk(obj)
        parts.append(table(["key", "value"], flat[:300], f"{p.name} ({len(flat)} values)"))
    return card(title, N("Raw outputs of {t}, shown as tables until a chart is designed for them.", t=producer) + stale_badge(d, producer),
                extra="".join(parts))


WAVES = (
    ("Wave 1 · data preparation", ("build_detections", "sensor_inventory", "field_intersection", "gt_associate", "coobservation",
                                   "cluster_tracks", "preclass_report", "eval_tiers", "eval_charts"), ("duplicates",)),
    ("Wave 2 · measurement layer", ("gospa", "truth", "submission", "score", "sensor_noise", "metadata_scan"), ("height_leak", "clock")),
    ("Wave 3 · tracks and baselines", ("sensor_tracks", "gbdt", "track_rows", "baselines", "existence"), ()),
    ("Wave 4 · fusion", ("pair_features", "ground_coverage", "classify", "bearing_support", "fuse", "track_existence"),
     ("truth_ground_share", "classifier", "f0_mean", "f0_cost_air", "f1_lost", "eval_dups", "eval_friendly", "eval_jumps")),
    ("Wave 5 · owner priorities", ("deanonymize", "probes", "radar_registration", "gospa_sensitivity", "compare", "triangulation_oracle", "run"),
     ("az_bias_stable", "enu_offsets_fail", "gospa_c", "bias_oracle", "triangulation_zero", "unseen_airborne", "probe_park", "uploads_blocked", "ref_latency")),
    ("Research · scoring and metadata", (), ("rules_weights_only", "metadata_ceiling", "whitelist_seen")),
    ("Investigations · localisation, calibration", (), ("offset_rows", "oracle_gains", "under_merge", "az_rotation_gain")),
    ("Wave 6a · diagnosis (in progress)", ("radar_model", "ablation", "health"), ()),
    ("Phase 1 · verified corrections (in progress)", ("sensor_noise", "sensor_tracks"), ()),
)


def wave_rows(fs, var) -> list:
    out = []
    byid = {f["id"]: f for f in fs}
    for name, tools, learned in WAVES:
        tl = ", ".join(f"{t}.py" + ("" if (TOOLS / f"{t}.py").exists() else " (not yet)") for t in tools) or "–"
        if name.startswith("Phase 1"):
            tl += "; variants: " + (", ".join(f"{t} = {m}" for t, m in sorted(var.items())) or "none promoted yet")
        ln = Raw("<br>".join(finding_text(byid[i]) + badge(byid[i].get("status", "")) for i in learned if i in byid) or "–")
        out.append([name, tl, ln])
    return out


def glossary() -> list[tuple[str, str]]:
    w = SC.WEIGHTS
    grid = re.search(r"per (\d+) ms grid", SC.__doc__ or "")
    g = grid.group(1) if grid else "fixed"
    return [
        (N("GOSPA"), N("The track-accuracy metric: at each grid step our rows are paired with the true drones; a pair costs its distance (raised to the power p); an unpaired drone or row costs half of c to the power p.")),
        (N("cut-off c"), N("The distance beyond which a row no longer counts as the drone: it becomes a false row and the drone a missed one. Not published; our primary guess is {c} m.", c=f"{SC.CONFIGS[0][1]:g}")),
        (N("exponent p"), N("Larger p punishes large position errors more.")),
        (N("2-D / 3-D"), N("Whether height errors count (3-D) or only the horizontal position (2-D).")),
        (N("grid step"), N("The template's time grid: one row per drone every {g} ms.", g=g)),
        (N("truth 'all' / 'airborne'"), N("Whether drones parked on the ground are part of the truth ('all') or only flying drones ('airborne'). Unknown which the organisers use, so both are scored.")),
        (N("combined score"), N("{a} × track part + {b} × classification recall + {c} × identification recall + {d} × metadata part, per iteration.",
                                 a=f"{w['track']:g}", b=f"{w['classification']:g}", c=f"{w['identity']:g}", d=f"{w['metadata']:g}")),
        (N("recall"), N("The share of true drone rows whose paired row carries the right label; a missed drone counts as wrong.")),
        (N("held-out / LOIO"), N("Leave-one-iteration-out: whatever is tuned for testing iteration k is chosen on the other iterations only.")),
        (N("in-sample calibration"), N("A correction fitted on the same iteration that is then scored; it makes local numbers look better than they will be on evaluation.")),
        (N("oracle"), N("A diagnostic that uses the truth to remove one error source; it shows what is at stake, not what is reachable.")),
        (N("reference (O3)"), N("F0's own fused tracks, reported exactly when they belong to a real drone. A reference point, not a ceiling: better tracks could beat it.")),
        (N("baseline"), N("A simple method to beat, e.g. submitting every sensor track.")),
        (N("fusion"), N("Merging the tracks that different sensors report for the same drone into one track.")),
        (N("segment"), N("One sensor's track of one object, cut where it has gaps.")),
        (N("azimuth bias"), N("A radar turns every bearing by a small fixed angle; rotating its positions back about the radar site removes it.")),
        (N("east / north offset"), N("A constant shift of a radar's positions. It does not transfer between iterations, because a rotation's east and north parts depend on where the drone flies.")),
        (N("UTM relay"), N("A service (VKYY, QTVS) that repeats the drones' own positions; on testing it is the ground truth itself, so it inflates local scores.")),
        (N("testing-only sensor"), N("A sensor that exists only in the labelled testing data (EUIG, LJJX, MMJJ, NJQL); it cannot help on evaluation and does not count in headline scores.")),
        (N("bearing-only sensor"), N("A camera, acoustic or radio direction finder: it reports a direction, no distance.")),
        (N("triangulation"), N("Locating a drone by crossing bearings from two different places.")),
        (N("probe"), N("A submission built only to learn a scoring rule from the leaderboard: F0 with one factor changed.")),
        (N("de-anonymisation"), N("Every longitude in the data is shifted by a secret constant; before uploading, the owner shifts the submitted longitudes back with his key.")),
        (N("wins k of n"), N("On how many testing iterations a method scores above another; a difference below {t} is a tie.", t=f"{CP.TIE:g}")),
        (N("sign test"), N("A coin-flip test of a win / loss split: how likely a split at least this lopsided is if both methods were equally good.")),
        (N("cost share"), N("The part of the GOSPA cost from localisation error, missed drones or false rows.")),
        (N("journal only"), N("A claim verified by an independent reviewer in a workflow journal, not re-extractable from a data file.")),
    ]


def provenance_card(D: dict) -> str:
    rows = []
    for name in sorted(SOURCES):
        info = SOURCES[name]
        p = info["path"]
        prod = info["producer"]
        pm = mtime_str(TOOLS / prod) if prod and (TOOLS / prod).exists() else "–"
        rows.append([name, mtime_str(p), f"{p.stat().st_size:,}", prod or "–", pm, "STALE" if name in D["stale"] else ""])
    regen = ("python3 tools/status_report.py                 # -> data/derived/REPORT.html\n"
             "python3 tools/status_report.py --allow-stale   # if a source is older than its tool for a documented reason")
    stale_note = ('<p class="what"><span class="badge warn">stale sources allowed</span> ' + esc(", ".join(sorted(D["stale"]))) + "</p>") if D["stale"] else ""
    return card(N("Every source file of this page"),
                N("Modification times, sizes and the tool that produces each file. Scorer code hash ({h}) covers every tools file.", h=D["code_hash"]),
                extra=stale_note + plain_table(["file", "modified", "bytes", "produced by", "tool modified", ""], rows) +
                f"<pre>{esc(regen)}</pre>")


# --------------------------------------------------------------------------- #
# Part 2, the story: eight short chapters for the owner and the coordinator
# --------------------------------------------------------------------------- #

STORY_MAX_BYTES = 400 * 1024      # check 11
STORY_MAX_FIGS = 8                # check 11
STORY_MAX_WORDS = 120             # check 11: words per chapter outside tables, charts and the source line
STORY_C = (10.0, 20.0, 30.0, 50.0, 75.0, 100.0)   # cut-offs re-scored for chapters 3 and 5 (p = 2, 3-D)
STORY_PRIMARY_C = 30.0            # the pre-registered primary condition: c = 30 m, truth 'all'
UNSEEN_RADIUS_M = 50              # triangulation_oracle.py's 'main_2d' unseen definition (horizontal)
V3A_ROOT = C.REPO / "data" / "derived_v3a"
STORY_CSS = """
.kicker{font-size:13px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin:0}
.chapter h2{font-size:21px;margin:4px 0 8px;line-height:1.3} .ctitle{font-size:15px;margin:16px 0 2px}
.gry{fill:var(--muted)} .lg{stroke:var(--muted)} .dash{stroke-dasharray:5 4} .faintln{opacity:.55}
.toc{columns:2 280px;font-size:14px;margin:8px 0 0;padding-left:20px} .toc li{margin:2px 0}
.intro{font-size:15px;color:var(--text-primary)}
"""


def story_cfg(c: float) -> str:
    return f"c{c:g}_p2_3d"


def _story_task(task):
    """Re-score one submission file at every STORY_C cut-off (p = 2, 3-D), both truth modes. The
    label recalls use each cut-off's own assignment (primary = that configuration)."""
    method, key = task
    CP.memoise_frame()
    it = _it(key)
    try:
        rows = S.read_submission(it, C.submission_path(method, it))
    except Exception as e:  # noqa: BLE001  a file another tool is writing right now
        return method, key, {"error": f"{type(e).__name__}: {e}"[:160]}
    out = {}
    for mode in MODES:
        for c in STORY_C:
            name = story_cfg(c)
            r = SC.score_rows(it, rows, mode, configs=[(name, c, 2.0, 3)], primary=name)
            d = dict(r["parts"])
            for k in ("cost_share_loc", "cost_share_missed", "cost_share_false"):
                d[k] = r[name][k]
            out[f"{mode}|{c:g}"] = d
    return method, key, out


def gather_story_scores(keys: list[str]) -> tuple[dict, dict, dict, dict]:
    """SS[method][(mode, c)][key] = parts; SCORES[method] = the c = 50 m block (checks 1 and 3)
    and SCORES['method@c<c>'] = the block at cut-off c (wins / losses, check 2)."""
    meta = {"F0_fuse": {"role": "deployable", "producer": "fuse.py", "label": "F0_fuse"}}
    var: dict = {}
    for tag in ("v3h", "v3a"):
        for nm in dict(VARIANTS)[tag]:
            if (C.SUBMISSIONS / nm).is_dir() and has_all_files(nm, keys):
                var[tag] = nm
                meta[nm] = {"role": "variant", "producer": VARIANT_PRODUCER[tag], "label": nm}
                break
    if has_all_files("B2_allpos", keys):
        meta["B2_allpos"] = {"role": "baseline", "producer": "baselines.py", "label": "B2_allpos"}
    tasks = []
    for name in meta:
        for k in keys:
            src(C.submission_path(name, _it(k)), meta[name]["producer"].split()[0])
            tasks.append((name, k))
    gt = DER / "groundtruth.csv.gz"
    if gt.exists():
        src(gt, "build_detections.py")
    print(f"  re-scoring {len(tasks)} files at c = {', '.join(f'{c:g}' for c in STORY_C)} m with up to 3 worker processes ...", flush=True)
    with Pool(min(3, len(tasks))) as pool:
        res = pool.map(_story_task, tasks, chunksize=1)
    ss: dict = {}
    broken: dict = {}
    for method, key, out in res:
        if "error" in out:
            broken.setdefault(method, f"{short(key)}: {out['error']}")
            continue
        for mk, d in out.items():
            mode, c = mk.split("|")
            ss.setdefault(method, {}).setdefault((mode, float(c)), {})[key] = d
    for method, why in sorted(broken.items()):
        print(f"  [skip] {method}: unreadable file ({why})")
        ss.pop(method, None)
        meta.pop(method, None)
        for tag in [t for t, m in var.items() if m == method]:
            var.pop(tag)
        UNREADABLE[method] = why
    scores: dict = {}
    for m in ss:
        scores[m] = {mode: ss[m][(mode, 50.0)] for mode in MODES}
        for c in STORY_C:
            scores[f"{m}@c{c:g}"] = {mode: ss[m][(mode, c)] for mode in MODES}
    return scores, meta, var, ss


def plain_words(s: str) -> list[str]:
    return re.findall(r"\S+", html.unescape(re.sub(r"<[^>]+>", " ", s)))


def story_table(header, rows, caption="Data table", note: str = "") -> str:
    """table() with an optional note paragraph inside the collapsible block (win / loss lines)."""
    t = table(header, rows, caption)
    return t.replace("</details>", (f'<p class="src">{note}</p>' if note else "") + "</details>", 1)


def chapter(no: int, title: str, headline: str, text: str, fig: str = "", ctitle: str = "", legend: str = "",
            how: str = "", tables: str = "", source: str = "", extra: str = "") -> str:
    """One story chapter. The word budget (check 11) counts every visible word except table cells,
    collapsed data blocks, the chart itself and the source line (anything in `tables` outside a
    <table> or a <details> block counts too)."""
    kicker = N("Chapter {n} · {t}", n=no, t=title)
    how_html = f'<p class="how"><b>{N("How to read it:")}</b> {how}</p>' if how else ""
    tab_vis = re.sub(r"<table\b.*?</table>", " ", re.sub(r"<details\b.*?</details>", " ", tables, flags=re.S), flags=re.S)
    vis = " ".join([kicker, headline, text, ctitle, legend, how_html, extra, tab_vis])
    tab_words = len(plain_words(re.sub(r"<details\b.*?</details>", " ", tables, flags=re.S))) - len(plain_words(tab_vis))
    rec = {"no": no, "title": title, "headline": " ".join(plain_words(headline)), "words": len(plain_words(vis)),
           "table_words": tab_words, "ctitle": " ".join(plain_words(ctitle)), "fig": bool(fig), "bad": []}
    if fig:
        if "<details" not in tables:
            ST.fig_without_table.append(f"chapter {no}")
            rec["bad"].append("no collapsible data table")
        if len(rec["ctitle"].split()) < 5:
            rec["bad"].append("chart title is not a takeaway sentence")
        for s in re.findall(r"<svg\b.*?</svg>", fig, re.S):
            if 'class="lab"' not in s:
                rec["bad"].append("a chart panel without an axis label")
    ST.story.append(rec)
    fig_html = (f'<h3 class="ctitle">{ctitle}</h3>{legend}<div class="fig">{fig}</div>{how_html}' if fig else how_html)
    src_html = f'<p class="src">{N("Source:")} {source}</p>' if source else ""
    return (f'<section class="card chapter" id="ch{no}"><p class="kicker">{kicker}</p><h2>{headline}</h2>'
            f'<p class="what">{text}</p>{extra}{fig_html}{tables}{src_html}</section>')


def fbadge(fs, fid) -> str:
    """Badge of a finding (journal / unverified) and mark it as rendered."""
    f = next((f for f in fs if f["id"] == fid), None)
    if f is None:
        return ""
    ST.rendered_findings.add(fid)
    b = badge(f.get("status", ""))
    return Raw(" " + b) if b else Raw("")


def fv(fs, fid, key):
    """A finding value for a story template; a missing one fails check 5."""
    v = fval(fs, fid, key)
    if v is None:
        ST.ctl_unresolved.append("{" + f"{fid}.{key}" + "}")
        return "[missing value]"
    ST.rendered_findings.add(fid)
    return v


def fsrc(fs, *ids) -> str:
    out = []
    for fid in ids:
        f = next((f for f in fs if f["id"] == fid), None)
        if f:
            out.append(f"<code>{esc(f['source_path'])}</code> ({esc(fid)})")
    return "; ".join(out)


def label_dodge(items: list[tuple[float, str]], min_gap: float = 14.0) -> list[tuple[float, str]]:
    """Vertical positions for end-of-line labels: sorted, at least min_gap px apart."""
    out, last = [], -1e9
    for y, t in sorted(items):
        y = max(y, last + min_gap)
        out.append((y, t))
        last = y
    return out


# ---- chapter charts ------------------------------------------------------- #


def story_weights_chart() -> tuple[str, list]:
    lab = {"track": "track accuracy (GOSPA)", "classification": "classification recall",
           "identity": "identification recall", "metadata": "metadata"}
    items = [(lab[k], SC.WEIGHTS[k]) for k in PARTS]
    hi, ticks = axis_hi(max(v for _, v in items))
    row_h, left = 30, 190
    h = 10 + len(items) * row_h + 44
    p = Plot(640, h, (0, hi), (0, 1), left=left, right=40, top=10, bottom=44, xticks=ticks, yticks=[],
             xlabel="weight in the combined score")
    y = 10
    for name, v in items:
        p.add(f'<text class="tick" x="{left - 8}" y="{y + 19}" text-anchor="end">{esc(name)}</text>')
        x0, x1 = p.X(0), p.X(v)
        p.add(f'<path class="s1" d="{EC.bar_path(x0, y + 7, x1 - x0, row_h - 14)}" data-tip="{esc(name)}: weight {v:g}" tabindex="0"/>')
        p.add(f'<text class="val" x="{x1 + 6:.1f}" y="{y + 19}">{v:g}</text>')
        y += row_h
    return p.svg("Weights of the four score parts"), [[n, f"{v:g}"] for n, v in items]


def story_truth_data(tri, keys) -> list[dict]:
    pick = {(r["iter_key"], r["truth_mode"], r["unseen_def"]): r for r in tri if r["offsets"] == "oracle"}
    out = []
    for k in list(keys) + ["POOLED"]:
        a, b = pick.get((k, "all", "main_2d")), pick.get((k, "airborne", "main_2d"))
        if not a or not b:
            continue
        n_all, n_air, uns = int(a["n_truth"]), int(b["n_truth"]), int(b["n_unseen"])
        out.append({"key": k, "label": "pooled" if k == "POOLED" else short(k), "n_all": n_all, "n_air": n_air, "n_unseen": uns,
                    "seen": (n_air - uns) / n_all, "unseen": uns / n_all, "parked": (n_all - n_air) / n_all,
                    "unseen_of_air": uns / n_air})
    return out


def story_truth_chart(td) -> tuple[str, str, list]:
    segs = (("seen", "airborne, seen", "s1"), ("unseen", "airborne, unseen (lost)", "s2"), ("parked", "parked", "gry"))
    row_h, left = 26, 90
    h = 10 + len(td) * row_h + 34 + 44
    p = Plot(W, h, (0, 100), (0, 1), left=left, right=20, top=10, bottom=44, xticks=[0, 25, 50, 75, 100], yticks=[],
             xfmt=lambda v: f"{v:g} %", xlabel="share of all truth rows (truth 'all')")
    y = 10
    for d in td:
        if d["key"] == "POOLED":
            y += 34    # room for the direct labels above the pooled bar
        p.add(f'<text class="tick{" tier" if d["key"] == "POOLED" else ""}" x="{left - 8}" y="{y + 17}" text-anchor="end">{esc(d["label"])}</text>')
        acc = 0.0
        for col, lab, cls in segs:
            v = 100 * d[col]
            x0, x1 = p.X(acc), p.X(min(100.0, acc + v))
            wpx = x1 - x0
            if wpx > 0.3:
                gap = 2 if wpx > 3 else 0
                p.add(f'<rect class="{cls}" x="{x0:.1f}" y="{y + 4}" width="{max(wpx - gap, 0.6):.1f}" height="{row_h - 8}" '
                      f'data-tip="{esc(d["label"])}: {esc(lab)} {v:.1f} % of {d["n_all"]:,} truth rows" tabindex="0"/>')
            if d["key"] == "POOLED":
                p.add(f'<text class="ann strong" x="{(x0 + x1) / 2:.1f}" y="{y - 4}" text-anchor="middle">{v:.1f} %</text>')
            acc += v
        y += row_h
    rows = [[d["label"], f"{d['n_all']:,}", f"{d['n_air']:,}", f"{d['n_unseen']:,}", pct(d["seen"]), pct(d["unseen"]),
             pct(d["parked"]), pct(d["unseen_of_air"])] for d in td]
    return p.svg("Truth rows by visibility per testing iteration"), EC.legend([(c, l) for _, l, c in segs]), rows


def story_dot_chart(ss, series, keys, c, mode="all") -> tuple[str, str]:
    """series: [(method, label, shape, css)] -> one row per testing iteration at cut-off c."""
    vals = [ss[m][(mode, c)][k]["combined"] for m, _, _, _ in series for k in keys]
    hi, ticks = axis_hi(max(vals))
    n, row_h, top = len(keys), 30, 40
    h = top + n * row_h + 44
    p = Plot(W, h, (0, hi), (0, n), left=70, right=24, top=top, bottom=44, xticks=ticks, yticks=[],
             xlabel=f"combined score at c = {c:g} m, truth '{mode}' (higher is better)")
    for i, k in enumerate(keys):
        yc = top + i * row_h + row_h / 2
        p.add(f'<text class="tick" x="{p.l - 8}" y="{yc + 4:.1f}" text-anchor="end">{esc(short(k))}</text>')
        p.add(f'<line class="grid" x1="{p.l}" x2="{p.w - p.r}" y1="{yc:.1f}" y2="{yc:.1f}"/>')
        xs = {m: p.X(ss[m][(mode, c)][k]["combined"]) for m, _, _, _ in series}
        if len(series) >= 2:   # a thin connector from the honest rebuild to the radar fix
            a, b = series[-2][0], series[-1][0]
            p.add(f'<line class="arrow" x1="{xs[a]:.1f}" x2="{xs[b]:.1f}" y1="{yc:.1f}" y2="{yc:.1f}"/>')
        for m, lab, shape, cls in series:
            v = ss[m][(mode, c)][k]["combined"]
            p.add(mark(shape, cls, xs[m], yc, tip=f"{lab} · {short(k)} · c = {c:g} m, truth '{mode}': combined {v:.3f}"))
        if i == 0:   # direct labels on the top row, dodged sideways onto two lines
            placed = []
            for m, lab, _, _ in sorted(series, key=lambda s: xs[s[0]]):
                nm = lab.split(" (")[0]
                lvl = 0
                while any(lv == lvl and abs(xs[m] - xx) < 8 + 3.4 * max(len(nm), ln) for xx, lv, ln in placed):
                    lvl += 1
                placed.append((xs[m], lvl, len(nm)))
                p.add(f'<text class="ann" x="{xs[m]:.1f}" y="{yc - 10 - 13 * lvl:.1f}" text-anchor="middle">{esc(nm)}</text>')
    return p.svg(f"Combined score per testing iteration at c = {c:g} m"), shape_legend([(sh, cls, lab) for _, lab, sh, cls in series])


def story_radar_chart(az, rot, deployable) -> tuple[str, list]:
    radars = [r for r in RADARS if r in az and r in deployable and az[r]]
    vmax = max(abs(v) for r in radars for v in az[r].values())
    hi, ticks = sym_lim(vmax)
    row_h, top = 34, 14
    h = max(top + len(radars) * row_h + 44, 250)
    row_h = (h - top - 44) / max(len(radars), 1)
    p = Plot(430, h, (-hi, hi), (0, 1), left=56, right=16, top=top, bottom=44, xticks=ticks, yticks=[],
             xfmt=lambda v: f"{v:+g}" if v else "0", xlabel="azimuth bias per testing iteration (deg to add)")
    p.add(f'<line class="zero" x1="{p.X(0):.1f}" x2="{p.X(0):.1f}" y1="{top}" y2="{h - 44}"/>')
    for i, r in enumerate(radars):
        yc = top + i * row_h + row_h / 2
        p.add(f'<text class="tick" x="{p.l - 8}" y="{yc + 4:.1f}" text-anchor="end">{esc(r)}</text>')
        vs = [az[r][k] for k in sorted(az[r])]
        mn = float(np.mean(vs))
        p.add(f'<line class="tick-med" x1="{p.X(mn):.1f}" x2="{p.X(mn):.1f}" y1="{yc - 11:.1f}" y2="{yc + 11:.1f}"/>')
        for j, k in enumerate(sorted(az[r])):   # colour = the radar's identity on the right panel; grey = not rotated
            p.add(EC.dot(p.X(az[r][k]), yc + (j % 3 - 1) * 4, {"EGOI": "s1", "NYTF": "s2"}.get(r, "gry"), r=4.5,
                         tip=f"{r} {short(k)}: azimuth bias {az[r][k]:+.2f} deg"))
    left_svg = p.svg("Azimuth bias per radar and testing iteration")
    # right: slope chart of the median horizontal error, as reported -> rotated back
    cls = {"EGOI": "1", "NYTF": "2"}
    ymax = max(max(x["raw"], x["after"]) for x in rot)
    yhi, yt = axis_hi(ymax)
    q = Plot(430, h, (-0.15, 1.15), (0, yhi), left=50, right=124, top=top, bottom=44, xticks=[0, 1], yticks=yt,
             xfmt=lambda v: "as reported" if v < 0.5 else "rotated back", ylabel="median horizontal error (m)",
             xlabel="one line per radar and testing iteration")
    for x in rot:
        c = cls.get(x["radar"], "1")
        q.add(f'<line class="ln l{c} faintln" x1="{q.X(0):.1f}" x2="{q.X(1):.1f}" y1="{q.Y(x["raw"]):.1f}" y2="{q.Y(x["after"]):.1f}"/>')
        for xv, yv in ((0, x["raw"]), (1, x["after"])):
            q.add(EC.dot(q.X(xv), q.Y(yv), f"s{c}", r=3.5, tip=f"{x['radar']} {x['iter']}: {x['raw']:.1f} m as reported, {x['after']:.1f} m rotated back"))
    lab = []
    for r in sorted({x["radar"] for x in rot}):
        med = float(np.median([x["after"] for x in rot if x["radar"] == r]))
        lab.append((q.Y(med) + 4, f"{r} median {med:.1f} m"))
    for yv, t in label_dodge(lab):
        q.add(f'<text class="ann" x="{q.X(1) + 10:.1f}" y="{yv:.1f}">{esc(t)}</text>')
    rows = [[x["radar"], x["iter"], sgn(az.get(x["radar"], {}).get("testing/" + x["iter"]), 2), f"{x['delta']:+.3f}",
             f"{x['raw']:.1f}", f"{x['after']:.1f}", f"{x['raw'] - x['after']:.1f}"] for x in rot]
    for r in radars:
        if not any(x["radar"] == r for x in rot):
            rows += [[r, short(k), sgn(az[r][k], 2), "not applied", "–", "–", "–"] for k in sorted(az[r])]
    return ('<div class="two"><div>' + left_svg + '</div><div>' + q.svg("Median horizontal error before and after the rotation")
            + "</div></div>"), rows


def load_rotation_table() -> list[dict]:
    """The 'Radar azimuth check' table of the v3a variant's sensor_tracks_v2/README.md."""
    path = V3A_ROOT / "sensor_tracks_v2" / "README.md"
    if not path.exists():
        return []
    text = src(path, "sensor_tracks.py").read_text(encoding="utf-8")
    m = re.search(r"^\| iteration \| radar \| delta applied.*?$\n\|[-| ]+\|\n((?:\|.*\|\n)+)", text, re.M)
    out = []
    for ln in (m.group(1).splitlines() if m else []):
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(c) < 11:
            continue
        out.append({"iter": c[0], "radar": c[1].split("_")[0], "delta": float(c[2]), "cal": float(c[7]),
                    "raw": float(c[9]), "after": float(c[10])})
    return out


def story_c_chart(ss, series, keys, mode="all") -> tuple[str, str, list]:
    pts = {m: [(c, float(np.mean([ss[m][(mode, c)][k]["combined"] for k in keys]))) for c in STORY_C] for m, _, _ in series}
    hi, ticks = axis_hi(max(v for s in pts.values() for _, v in s))
    p = Plot(W, 300, (0, max(STORY_C)), (0, hi), left=56, right=150, top=14, bottom=44, yticks=ticks,
             xticks=[0, 20, 40, 60, 80, 100], xlabel="GOSPA cut-off c (m)", ylabel=f"mean combined score, truth '{mode}'")
    xg = p.X(STORY_PRIMARY_C)
    p.add(f'<line class="guide dash" x1="{xg:.1f}" x2="{xg:.1f}" y1="{p.t}" y2="{p.h - p.b}"/>'
          f'<text class="ann" x="{xg + 5:.1f}" y="{p.t + 12}">pre-registered c</text>')
    ends = []
    for m, lab, k in series:
        cl = "lg" if k == "gry" else "l" + k[1:]
        p.add(f'<polyline class="ln {cl}" points="{" ".join(f"{p.X(c):.1f},{p.Y(v):.1f}" for c, v in pts[m])}"/>')
        for c, v in pts[m]:
            p.add(EC.dot(p.X(c), p.Y(v), k, r=3.5, tip=f"{lab}, c = {c:g} m: mean combined {v:.3f}"))
        ends.append((p.Y(pts[m][-1][1]) + 4, lab.split(" (")[0]))
    for yv, t in label_dodge(ends):
        p.add(f'<text class="ann" x="{p.X(max(STORY_C)) + 8:.1f}" y="{yv:.1f}">{esc(t)}</text>')
    rows = []
    for m, lab, _ in series:
        for c in STORY_C:
            for md in MODES:
                vs = [ss[m][(md, c)][k]["combined"] for k in keys]
                rows.append([m, md, f"{c:g}", f4(float(np.mean(vs)), 3)] + [f4(v, 3) for v in vs])
    return p.svg("Mean combined score versus the GOSPA cut-off"), EC.legend([(k, lab) for _, lab, k in series]), rows


def story_health_chart(hj) -> tuple[str, str, list]:
    its = [it for it in C.iterations() if it.key in hj["iterations"]]
    groups = (("testing", [it for it in its if it.tree == "testing"], "s1"),
              ("evaluation", [it for it in its if it.tree == "evaluation"], "s2"))
    vals = [100 * hj["iterations"][it.key]["near_dup"]["step_share"] for it in its]
    hi, ticks = axis_hi(max(vals))
    row_h, left = 24, 110
    h = 10 + sum(len(g[1]) + 1 for g in groups) * row_h + 44
    p = Plot(W, h, (0, hi), (0, 1), left=left, right=50, top=10, bottom=44, xticks=ticks, yticks=[],
             xfmt=lambda v: f"{v:g} %", xlabel="time steps with a near-duplicate pair of F0 tracks (share)")
    y, rows = 10, []
    for gname, g, cls in groups:
        p.add(f'<text class="tier" x="8" y="{y + 16}">{esc(gname)}</text>')
        y += row_h
        for it in g:
            d = hj["iterations"][it.key]
            v = 100 * d["near_dup"]["step_share"]
            lab = short(it.key) if it.tree == "testing" else f"D{it.dataset} {short(it.key)}"
            p.add(f'<text class="tick" x="{left - 8}" y="{y + 16}" text-anchor="end">{esc(lab)}</text>')
            x0, x1 = p.X(0), p.X(v)
            if x1 - x0 > 0.3:
                p.add(f'<path class="{cls}" d="{EC.bar_path(x0, y + 5, x1 - x0, row_h - 10)}" data-tip="{esc(lab)}: {v:.1f} % of time steps" tabindex="0"/>')
            p.add(f'<text class="val" x="{x1 + 6:.1f}" y="{y + 16}">{v:.0f} %</text>')
            rows.append([gname, lab, pct(d["near_dup"]["step_share"]), d["jump"]["rows"], f"{d['friendly']['copy_seconds']:.0f}"])
            y += row_h
    return (p.svg("Near-duplicate time steps per iteration"), EC.legend([("s1", "testing"), ("s2", "evaluation (submitted)")]), rows)


TIER_NAMES = (  # field that names a tier -> plain name (check 12: the field must be in that tier's row)
    ("classification", "classification"), ("location_x", "horizontal position"), ("location_z", "height"),
    ("id", "serial / protocol id"), ("detectionConfidence", "detection confidence"), ("objectInfo", "model name"),
    ("messageType", "basic report fields"))


def load_tiers() -> tuple[list[dict], dict, dict]:
    """Tier ladder of the existence model (target y_all, deployable sensors) and the tier field lists."""
    lp, mp = DER / "tier_models" / "ladder.csv", DER / "tier_models" / "tier_models.md"
    if not lp.exists() or not mp.exists():
        return [], {}, {}
    rows = [r for r in read_csv(lp, "tier_models.py") if r["model"] == "existence" and r["target"] == "all" and r["scope"] == "deploy"]
    text = src(mp, "tier_models.py").read_text(encoding="utf-8")
    fields = {}
    for m in re.finditer(r"^\| (\d+) \| ([^|]+) \|", md_section(text, "What each tier can use"), re.M):
        fields[int(m.group(1))] = re.findall(r"([A-Za-z_]+) \(\d", m.group(2))
    rule = re.search(r"hold on >= (\d+) of (\d+) iterations", text)
    dec = re.search(r"on \*\*(\d+) of (\d+)\*\* testing iterations \(needed: >= (\d+)\)", text)
    ll = (re.search(r"Pooled logloss \(step-weighted, deployable, y_all\): routed (\d+\.\d+), single (\d+\.\d+)", text)
          or re.search(r"Pooled logloss routed (\d+\.\d+) vs single (\d+\.\d+)", text))
    big = re.search(r"(\d+) of (\d+) won by at least 0\.01", text)
    fragile = "FRAGILE" in text
    meta = {"need": int(rule.group(1)), "of": int(rule.group(2))} if rule else {}
    if meta and dec and ll:
        meta.update({"routed_w": int(dec.group(1)), "routed_n": int(dec.group(2)), "routed_need": int(dec.group(3)),
                     "ll_routed": ll.group(1), "ll_single": ll.group(2), "fragile": fragile,
                     "big_w": int(big.group(1)) if big else None})
    else:
        meta = {}
    out = []
    for t in sorted({int(r["tier"]) for r in rows}):
        pooled = next((r for r in rows if int(r["tier"]) == t and r["iteration"] == "pooled"), None)
        mean = next((r for r in rows if int(r["tier"]) == t and r["iteration"] == "mean"), None)
        per = [r for r in rows if int(r["tier"]) == t and r["iteration"].startswith("testing/")]
        # field gain (same training sensors with vs without the tier's fields) when tier_models.py
        # wrote it; the older routing gain mixed the fields' value with a change of training sensors
        fg = "field_gain_ll_step" in pooled and pooled["field_gain_ll_step"] not in ("", None)
        gk, wk = ("field_gain_ll_step", "field_wins") if fg else ("gain_ll_step", "wins_vs_prev")
        base = "abl_ll_step" if fg else "prev_ll_step"
        if not pooled or not mean or pooled[gk] in ("", None):
            continue
        fl = fields.get(t, [])
        name = next((nm for f, nm in TIER_NAMES if f in fl), fl[0] if fl else "")
        key = next((f for f, nm in TIER_NAMES if nm == name), None)
        out.append({"tier": t, "name": name, "key_field": key, "fields": fl, "gain": float(pooled[gk]),
                    "wins": int(mean[wk]), "n": len(per),
                    "wins_recount": sum(1 for r in per if r["ll_step"] not in ("", None) and r[base] not in ("", None)
                                        and float(r["ll_step"]) < float(r[base]))})
    return out, meta, fields


def load_tiers_retry(tries: int = 6, wait_s: float = 20.0) -> tuple[list[dict], dict]:
    """load_tiers, retried while tools/tier_models.py may be rewriting its outputs (a half-written
    file parses to no tiers or no decision lines, or the two files disagree in time)."""
    for i in range(tries):
        try:
            lp, mp = DER / "tier_models" / "ladder.csv", DER / "tier_models" / "tier_models.md"
            t0 = (lp.stat().st_mtime, mp.stat().st_mtime) if lp.exists() and mp.exists() else None
            tiers, meta, _ = load_tiers()
            t1 = (lp.stat().st_mtime, mp.stat().st_mtime) if lp.exists() and mp.exists() else None
            if t0 is None or (tiers and meta and t0 == t1):
                return tiers, meta
        except (ValueError, KeyError, OSError) as e:  # a row cut in half
            print(f"  tier_models outputs unreadable ({type(e).__name__}); retrying")
        if i + 1 < tries:
            print(f"  tier_models outputs incomplete or changing; waiting {wait_s:g} s")
            time.sleep(wait_s)
    return [], {}


def story_tier_chart(tiers, need) -> tuple[str, str, list]:
    vals = [t["gain"] for t in tiers]
    lo = min(0.0, min(vals))
    hi, ticks = axis_hi(max(vals + [1e-6]))
    if lo < 0:
        lo = -axis_hi(-lo)[0]
        ticks = sorted(set(EC.nice_ticks(lo, hi)))
    row_h, left = 30, 250
    h = 10 + len(tiers) * row_h + 44
    p = Plot(W, h, (lo, hi), (0, 1), left=left, right=130, top=10, bottom=44, xticks=ticks, yticks=[],
             xfmt=lambda v: f"{v:+.3f}" if v else "0", xlabel="pooled held-out log-loss gain over the tiers before it")
    y = 10
    for t in tiers:
        cls = "s1" if t["wins"] >= need else "gry"
        lab = f"tier {t['tier']} · {t['name']}"
        p.add(f'<text class="tick" x="{left - 8}" y="{y + 19}" text-anchor="end">{esc(lab)}</text>')
        x0, x1 = p.X(0), p.X(t["gain"])
        tip = f"{lab}: pooled gain {t['gain']:+.4f}, lower log-loss on {t['wins']} of {t['n']} testing iterations"
        if t["gain"] > 0:
            p.add(f'<path class="{cls}" d="{EC.bar_path(x0, y + 7, x1 - x0, row_h - 14)}" data-tip="{esc(tip)}" tabindex="0"/>')
        elif t["gain"] < 0:
            p.add(f'<rect class="{cls}" x="{x1:.1f}" y="{y + 7}" width="{x0 - x1:.1f}" height="{row_h - 14}" data-tip="{esc(tip)}" tabindex="0"/>')
        p.add(f'<text class="val" x="{max(x0, x1) + 6:.1f}" y="{y + 19}">{t["gain"]:+.3f} · wins {t["wins"]} of {t["n"]}</text>')
        y += row_h
    p.add(f'<line class="zero" x1="{p.X(0):.1f}" x2="{p.X(0):.1f}" y1="{p.t}" y2="{h - p.b}"/>')
    rows = [[t["tier"], t["name"], ", ".join(t["fields"]), f"{t['gain']:+.4f}", f"{t['wins']} of {t['n']}"] for t in tiers]
    leg = EC.legend([("s1", f"holds on at least {need} of {tiers[0]['n']}"), ("gry", "does not hold")])
    return p.svg("Marginal gain of each attribute tier"), leg, rows


def story_glossary() -> list[tuple[str, str]]:
    base = dict(glossary())
    pick = ["GOSPA", "cut-off c", "truth 'all' / 'airborne'", "combined score", "recall", "held-out / LOIO",
            "in-sample calibration", "fusion", "azimuth bias", "UTM relay", "probe", "journal only"]
    out = [(k, base[k]) for k in pick if k in base]
    out += [
        (N("testing / evaluation"), N("Testing iterations come with ground truth and are used to measure; the evaluation data sets have none and are what we submit.")),
        (N("latency"), N("How late a sensor reports: its position shows where the drone was a moment ago.")),
        (N("direction finder"), N("A camera, acoustic or radio sensor that reports only a direction, no distance.")),
        (N("near-duplicate"), N("Two of our tracks that stay close together for a while: probably one drone reported twice.")),
        (N("tier"), N("A group of report fields that the same share of sensors report; tier one is what every sensor reports.")),
        (N("log-loss"), N("How wrong the drone-or-clutter probabilities are on held-out data; lower is better.")),
        (N("routed model"), N("One model per tier; each sensor track is scored by the model of the richest tier its sensor reports.")),
    ]
    return out


# ---- the page --------------------------------------------------------------- #


def story_body(D: dict) -> list[str]:
    fs, co = D["findings"]["findings"], D["findings"]["coordinator"]
    ss, var, keys = D["ss"], D["var"], D["keys"]
    n = len(keys)
    B: list[str] = []
    titles = ("What we are scored on", "Where the drones are, and which we can see", "From raw sensor tracks to one track per drone",
              "The key discovery: the radars are rotated", "Why the unknown distance limit c matters",
              "What still breaks on the evaluation data", "The professor's idea: one model per attribute tier", "What we need now")
    jb = Raw('<span class="badge warn">journal only</span>')
    B.append('<p class="intro">' + N(
        "A ten-minute read in eight chapters. Every number is a local estimate on the {n} labelled testing iterations "
        "(the official scorer is not published); it is computed from the files or taken from the verified findings, and {b} "
        "marks a claim verified only in a workflow journal. The one-screen summary is <a href=\"#summary\">above</a>; "
        "the full detail is in the <a href=\"#appendix\">technical appendix</a> at the end of this page.",
        n=n, b=jb) + "</p>")
    B.append('<ol class="toc">' + "".join(f'<li><a href="#ch{i + 1}">{N("{t}", t=t)}</a></li>' for i, t in enumerate(titles))
             + f'<li><a href="#gloss">{N("Glossary")}</a></li></ol>')

    # 1 scoring
    w = SC.WEIGHTS
    svg1, rows1 = story_weights_chart()
    B.append(chapter(
        1, titles[0],
        N("Four parts make the score; track accuracy weighs most, and its exact settings are not published"),
        N("Each iteration scores {a} × track accuracy (GOSPA) + {b} × classification recall + {c} × identification recall + {d} × metadata. "
          "The rules give only these weights; GOSPA's distance limit c, its exponent p and whether height counts are not stated, "
          "so every number here is our own estimate on the {n} labelled testing iterations.",
          a=f"{w['track']:g}", b=f"{w['classification']:g}", c=f"{w['identity']:g}", d=f"{w['metadata']:g}", n=n),
        svg1, N("Track accuracy and metadata together carry {x} of the score", x=pct(w["track"] + w["metadata"], 0)), "",
        N("longer bar = larger weight; the weights sum to one."),
        story_table(["part", "weight"], rows1),
        Raw("<code>tools/score.py</code> WEIGHTS; " + fsrc(fs, "rules_weights_only"))))
    ST.rendered_findings.add("rules_weights_only")

    # 2 truth composition
    td = story_truth_data(D["tri"], keys)
    pooled = next((d for d in td if d["key"] == "POOLED"), None)
    if pooled and len(td) > 1:
        svg2, leg2, rows2 = story_truth_chart(td)
        B.append(chapter(
            2, titles[1],
            N("{g} of the truth rows are parked drones, and {u} of the airborne rows no position sensor sees",
              g=pct(pooled["parked"]), u=pct(pooled["unseen_of_air"])),
            N("The truth lists every drone every tenth of a second, even parked. Airborne rows with no position sensor within {r} m "
              "are lost for every team, and bearings cannot rescue them ({t} are seen from two places: all direction finders stand at one site).",
              r=UNSEEN_RADIUS_M, t=fv(fs, "triangulation_zero", "v")),
            svg2, N("Our points are in the blue part: {s} of all truth rows, pooled", s=pct(pooled["seen"])),
            leg2, N("one bar per testing iteration; all truth rows = {h}.", h="100 %"),
            story_table(["iteration", "truth rows ('all')", "airborne rows", "airborne, unseen", "airborne seen (share of all)",
                         "airborne unseen (share of all)", "parked (share of all)", "unseen share of airborne"], rows2),
            Raw("<code>data/derived/triangulation/triangulation_rows.csv</code> (unseen definition main_2d, oracle offsets); "
                + fsrc(fs, "truth_ground_share", "unseen_airborne", "triangulation_zero"))))
        for fid in ("truth_ground_share", "unseen_airborne"):
            ST.rendered_findings.add(fid)
    else:
        B.append(pending_card(N("Chapter {n} · {t}", n=2, t=titles[1]), "triangulation_oracle.py"))

    # 3 scores at c = 30 m
    c0 = STORY_PRIMARY_C
    if "v3h" in var and "v3a" in var:
        mh, ma = var["v3h"], var["v3a"]
        series = [("F0_fuse", "F0_fuse (in-sample calibration)", "circle", "gry"),
                  (mh, "v3h (honest)", "circle", "s2"), (ma, "v3a (honest + radar fix)", "circle", "s1")]
        if "B2_allpos" in ss:
            series.insert(0, ("B2_allpos", "B2_allpos (every raw track)", "square", "mk-ink"))
        svg3, leg3 = story_dot_chart(ss, series, keys, c0)
        tag = f"@c{c0:g}"
        r_all = wlt(D["scores"], ma + tag, mh + tag, "all", keys)
        r_air = wlt(D["scores"], ma + tag, mh + tag, "airborne", keys)
        r_f0 = wlt(D["scores"], ma + tag, "F0_fuse" + tag, "all", keys)
        r_h = wlt(D["scores"], mh + tag, "F0_fuse" + tag, "all", keys)
        s11k = next((k for k in keys if short(k) == "S1-I1"), keys[0])
        d11 = ss[mh][("all", c0)][s11k]["combined"] - ss["F0_fuse"][("all", c0)][s11k]["combined"]
        rows3 = []
        for md in MODES:
            for k in keys:
                v = {m: ss[m][(md, c0)][k]["combined"] for m, _, _, _ in series}
                rows3.append([md, short(k)] + [f4(v[m], 3) for m, _, _, _ in series] + [sgn(v[ma] - v[mh], 3), sgn(v[mh] - v["F0_fuse"], 3)])
        note = N("{a} vs {b}, truth 'all': {x}; truth 'airborne': {y}. {a} vs F0_fuse, truth 'all': {z}. {b} vs F0_fuse, truth 'all': {q}.",
                 a=ma, b=mh, x=wlt_text(r_all), y=wlt_text(r_air), z=wlt_text(r_f0), q=wlt_text(r_h))
        b2txt = (N("Raw tracks (B2_allpos) are shown as grey squares.") if "B2_allpos" in ss else
                 N("Raw sensor tracks alone score {b} against F0's {f} (c = {c} m; one iteration's files only, not plotted).",
                   c=f"{SC.CONFIGS[0][1]:g}", b=fv(fs, "f0_mean", "b2all"), f=fv(fs, "f0_mean", "all")))
        every = "in every row" if r_all["w"] == n else f"in {r_all['w']} of {n} rows"
        B.append(chapter(
            3, titles[2],
            N("{a} beats the honest rebuild on {w} of {n} testing iterations and the in-sample F0 on {w0} of {n}",
              a="v3a", w=r_all["w"], n=n, w0=r_f0["w"]),
            N("F0_fuse merges each drone's sensor tracks into one; part of its calibration was fitted on testing data (a leak), "
              "which {h} removes; {a} adds the radar fix. Honesty costs mostly S1-I1 ({d}, one sensor's height). {b2}",
              h="v3h", a="v3a", d=f"{d11:+.3f}", b2=Raw(b2txt)),
            svg3, N("At c = {c} m blue lies right of orange {e}", c=f"{c0:g}", e=every), leg3,
            N("right is better; compare within a row."),
            story_table(["truth", "iteration"] + [m for m, _, _, _ in series] + ["v3a − v3h", "v3h − F0"], rows3,
                        f"Data table (re-scored at c = {c0:g} m, both truth modes)", note),
            Raw("re-scored from the submission files with <code>score.score_rows</code> (custom configuration "
                + esc(story_cfg(c0)) + ", the label recalls on its assignment); " + fsrc(fs, "v3_azimuth_score", "v3_honesty_drop", "f0_mean"))))
        for fid in ("v3_azimuth_score", "v3_honesty_drop"):
            ST.rendered_findings.add(fid)
    else:
        B.append(pending_card(N("Chapter {n} · {t}", n=3, t=titles[2]), "run.py --promote v3h / v3a"))

    # 4 radar rotation
    rot = load_rotation_table()
    az = D["radar_az"]["az"]
    if rot and az:
        svg4, rows4 = story_radar_chart(az, rot, D["deployable"])
        med = {r: (float(np.median([x["raw"] for x in rot if x["radar"] == r])), float(np.median([x["after"] for x in rot if x["radar"] == r])))
               for r in ("EGOI", "NYTF") if any(x["radar"] == r for x in rot)}
        better = sum(1 for x in rot if x["after"] < x["raw"])
        rng = 1500.0
        spread = {r: max(az[r].values()) - min(az[r].values()) for r in ("EGOI", "NYTF") if az.get(r)}
        sp_txt = lambda r: f"{spread[r]:.1f}"  # noqa: E731
        B.append(chapter(
            4, titles[3],
            N("The radars are rotated: turning them back cuts EGOI's error from {a} to {b} m, NYTF's from {c} to {d} m",
              a=f"{med['EGOI'][0]:.0f}", b=f"{med['EGOI'][1]:.0f}", c=f"{med['NYTF'][0]:.0f}", d=f"{med['NYTF'][1]:.0f}"),
            N("A direction {x} degree off misplaces a drone at {r} km by {e} m. It looked like random east/north offsets flipping sign "
              "between iterations; it is mostly one angle per radar, corrected leave-one-out. Radar latency is about zero (EGOI {l1}, NYTF {l2}).",
              x="1", r=f"{rng / 1000:g}", e=f"{rng * math.radians(1.0):.0f}",
              l1=fv(fs, "radar_latency_zero", "egoi"), l2=fv(fs, "radar_latency_zero", "nytf")),
            svg4, N("Across iterations EGOI's angle varies by {se} degrees, NYTF's by {sn}; turning it back helps in {k} of {m} cases",
                    se=sp_txt("EGOI"), sn=sp_txt("NYTF"), k=better, m=len(rot)),
            EC.legend([("s1", "EGOI"), ("s2", "NYTF"), ("gry", "other radars (not rotated)")]),
            N("left: one dot per iteration, tick = mean; right: one line per radar and iteration."),
            story_table(["radar", "iteration", "azimuth bias (deg, add)", "rotation applied (deg, leave-one-out)",
                         "median error as reported (m)", "rotated back (m)", "reduction (m)"], rows4),
            Raw("<code>data/derived/sensor_calibration.csv</code> (radar_bias rows); "
                "<code>data/derived_v3a/sensor_tracks_v2/README.md</code> (radar azimuth check, matched airborne rows); "
                + fsrc(fs, "radar_latency_zero", "az_bias_stable"))))
        ST.rendered_findings.add("az_bias_stable")
    else:
        B.append(pending_card(N("Chapter {n} · {t}", n=4, t=titles[3]), "run.py --derived data/derived_v3a"))

    # 5 score versus c
    if "v3h" in var and "v3a" in var:
        mh, ma = var["v3h"], var["v3a"]
        series5 = [("F0_fuse", "F0_fuse (in-sample calibration)", "gry"), (mh, "v3h (honest)", "s2"), (ma, "v3a (honest + radar fix)", "s1")]
        svg5, leg5, rows5 = story_c_chart(ss, series5, keys)
        mean = lambda m, c: float(np.mean([ss[m][("all", c)][k]["combined"] for k in keys]))  # noqa: E731
        clo, chi = min(STORY_C), max(STORY_C)
        gain = {c: mean(ma, c) - mean(mh, c) for c in STORY_C}
        c_best = max(STORY_C, key=lambda c: gain[c])
        c_worst = min(STORY_C, key=lambda c: gain[c])
        trend = N("The radar fix gains most at c = {a} m ({ga}), least at c = {b} m ({gb})",
                  a=f"{c_best:g}", ga=f"{gain[c_best]:+.3f}", b=f"{c_worst:g}", gb=f"{gain[c_worst]:+.3f}")
        shift = next((p for p in sorted(D["probe_names"]) if p.startswith("P_shift")), "P_shift15")
        pm = re.search(r"predicted BELOW P0 in (\d+) of (\d+)", D.get("probes_md") or "")
        if pm is None:
            ST.ctl_unresolved.append("{probes.md: P_shift 'predicted BELOW P0 in k of n'}")
        below = f"{pm.group(1)} of {pm.group(2)}" if pm else "[missing value]"
        B.append(chapter(
            5, titles[4],
            N("The unknown cut-off c changes our score {k}-fold; the smaller c, the more accuracy matters",
              k=f"{mean('F0_fuse', chi) / mean('F0_fuse', clo):.0f}"),
            N("GOSPA counts a track as the drone only within c metres. F0's mean falls from {hi} at c = {chi} m to {lo} at c = {clo} m. "
              "The probe {p} (positions moved {s} m) can bracket c, but it is predicted below P0 in {bl} cases, "
              "so a best-only board never shows it.",
              hi=f"{mean('F0_fuse', chi):.3f}", chi=f"{chi:g}", lo=f"{mean('F0_fuse', clo):.3f}", clo=f"{clo:g}",
              p=shift, s=re.search(r"\d+", shift).group(0), bl=below),
            svg5, N("{t}", t=Raw(trend)), leg5,
            N("the dashed line is our pre-registered c."),
            story_table(["method", "truth", "c (m)", "mean"] + [short(k) for k in keys], rows5),
            Raw("re-scored from the submission files (p = 2, 3-D; the label recalls on each cut-off's assignment); "
                "<code>data/derived/probes/probes.md</code> (P_shift section); " + fsrc(fs, "gospa_c"))))
        ST.rendered_findings.add("gospa_c")
    else:
        B.append(pending_card(N("Chapter {n} · {t}", n=5, t=titles[4]), "run.py --promote v3h / v3a"))

    # 6 evaluation health
    hp = DER / "health" / "F0_fuse.json"
    if hp.exists():
        hj = json.loads(src(hp, "health.py").read_text(encoding="utf-8"))
        D["health"] = hj
        svg6, leg6, rows6 = story_health_chart(hj)
        ev = [hj["iterations"][it.key]["near_dup"]["step_share"] for it in C.iterations() if it.tree == "evaluation" and it.key in hj["iterations"]]
        te = [hj["iterations"][k]["near_dup"]["step_share"] for k in keys if k in hj["iterations"]]
        jr = [hj["iterations"][it.key]["jump"]["rows"] for it in C.iterations() if it.tree == "evaluation" and it.key in hj["iterations"]]
        t6 = (N("Every evaluation iteration has more duplicates than any testing iteration") if min(ev) > max(te) else
              N("Duplicate time steps on evaluation range {a} to {b}, on testing {c} to {d}",
                a=pct(min(ev), 0), b=pct(max(ev), 0), c=pct(min(te), 0), d=pct(max(te), 0)))
        B.append(chapter(
            6, titles[5],
            N("On evaluation, {a} to {b} of time steps carry duplicate tracks; on testing at most {t}",
              a=pct(min(ev), 0), b=pct(max(ev), 0), t=pct(max(te), 0)),
            N("Why tracks duplicate is unverified. Impossible jumps ({j1} to {j2} rows per evaluation "
              "iteration) have two known causes: STUG's two streams {gap} apart under one id{b1}, and UXYJ drifting away ({grow}) "
              "before leaving a fused track{b2}. Friendly drones appear as {fr} copies{b3}; UTM relays, within {il} of ILYS, get a radar's "
              "weight ({v}).",
              j1=f"{min(jr):,}", j2=f"{max(jr):,}",
              gap=fv(fs, "stug_streams", "gap"), b1=fbadge(fs, "stug_streams"), grow=fv(fs, "uxyj_leaves", "grow"),
              b2=fbadge(fs, "uxyj_leaves"), fr=fv(fs, "eval_friendly", "v"), b3=fbadge(fs, "eval_friendly"),
              v=fv(fs, "utm_weight_default", "v"), il=fv(fs, "utm_eval_precise", "ilys")),
            svg6, t6, leg6,
            N("one bar per iteration."),
            story_table(["tree", "iteration", "near-duplicate step share", "impossible-jump rows", "seconds with two Friendly copies"], rows6),
            Raw("<code>data/derived/health/F0_fuse.json</code> (near_dup.step_share, jump.rows, friendly.copy_seconds); "
                + fsrc(fs, "health_eval", "stug_streams", "uxyj_leaves", "eval_friendly", "utm_weight_default", "utm_eval_precise"))))
        ST.rendered_findings.add("health_eval")
    else:
        B.append(pending_card(N("Chapter {n} · {t}", n=6, t=titles[5]), "health.py"))

    # 7 tier models
    tiers, tmeta = D.get("tiers") or [], D.get("tier_meta") or {}
    show = [t for t in tiers if t["tier"] > 1]
    if show and tmeta:
        svg7, leg7, rows7 = story_tier_chart(show, tmeta["need"])
        hold = [t for t in show if t["wins"] >= tmeta["need"]]
        def join(xs):
            return (", ".join(xs[:-1]) + " and " + xs[-1]) if len(xs) > 1 else (xs[0] if xs else "no tier")
        gmax = max((t["gain"] for t in hold), default=0.0)
        strong = [t["name"] for t in hold if t["gain"] >= 0.5 * gmax]   # within a factor two of the best tier
        weak = [t["name"] for t in hold if t["gain"] < 0.5 * gmax]
        names_txt = join(strong) + (" carry" if len(strong) > 1 else " carries") + " the signal" + (
            f", {join(weak)} {'add' if len(weak) > 1 else 'adds'} a little" if weak else "")
        passed = tmeta["routed_w"] >= tmeta["routed_need"]
        verdict = (("passes only at the minimum, not adopted" if tmeta["routed_w"] == tmeta["routed_need"] else
                    "passes its test") if passed else "does not beat the single model")
        dll = float(tmeta["ll_routed"]) - float(tmeta["ll_single"])
        cmp = "practically equal" if abs(dll) < 0.0005 else ("worse" if dll > 0 else "better")
        frag = (N(" Fragile: only {b} of {n} won by a clear margin.", b=tmeta["big_w"], n=tmeta["routed_n"])
                if tmeta.get("fragile") and tmeta.get("big_w") is not None else "")
        B.append(chapter(
            7, titles[6],
            N("The professor's idea: {nm}; one model per tier {v}", nm=names_txt, v=verdict),
            N("Field tiers added one at a time to the drone-or-clutter model. "
              "One model per tier wins {w} of {n} iterations (needed: {need}), pooled log-loss {c} "
              "({lr} vs {ls}); {nxt}",
              w=tmeta["routed_w"], n=tmeta["routed_n"], need=tmeta["routed_need"], c=cmp, lr=tmeta["ll_routed"], ls=tmeta["ll_single"],
              nxt="the end-to-end score test decides next." if passed else "the single model stays.") + frag,
            svg7, N("{k} of {m} tiers help on at least {need} of {n} iterations", k=len(hold), m=len(show), need=tmeta["need"], n=tmeta["of"]),
            leg7, N("bar = gain from the tier's own fields; right of zero helps."),
            story_table(["tier", "name", "fields", "pooled gain", "iterations won"], rows7),
            Raw("<code>data/derived/tier_models/ladder.csv</code> (existence, y_all, deployable); "
                "<code>data/derived/tier_models/tier_models.md</code> (answer in short); " + fsrc(fs, "tier_ladder"))))
        ST.rendered_findings.add("tier_ladder")
    else:
        B.append(pending_card(N("Chapter {n} · {t}", n=7, t=titles[6]), "tier_models.py"))

    # 8 what we need now
    appr, qs = co.get("approvals", []), co.get("questions", [])
    steps = [N("<b>Merge physics</b>: find why tracks duplicate, split STUG's double streams, drop UXYJ before it drifts, one Friendly track per serial."),
             N("<b>Radar model and UTM</b>: the remaining radar terms; UTM relays weighted by their measured precision."),
             N("<b>Uploads</b>, once approved: P0, then the one-factor probes; the c probe only if every score is shown.")]
    asks = ([[N("approval"), Raw(ctext(x, fs))] for x in appr] +
            [[N("question for the organisers"), Raw(ctext(x, fs))] for x in qs])
    asks_html = "<h3 class=\"ctitle\">" + N("What we ask") + "</h3>" + plain_table([N("kind"), N("verbatim")], asks)
    extra = ('<p class="what">' + N("Metadata: we keep 'Unknown' where there is no evidence; 'Not Applicable' loses on {na} iterations.{b}",
                                     na=fv(fs, "meta_verdict", "na"), b=fbadge(fs, "meta_verdict")) + "</p>" +
             "<h3 class=\"ctitle\">" + N("The next three steps") + "</h3>" + '<ol class="q">' + "".join(f"<li>{s}</li>" for s in steps) + "</ol>" +
             '<p class="what">' + N("Details: <a href=\"#appendix\">appendix</a>, <a href=\"#summary\">summary</a>.") + "</p>")
    B.append(chapter(
        8, titles[7],
        N("We need {a} approvals and {q} answers; then three fixes, one at a time", a=len(appr), q=len(qs)),
        N("Nothing has been uploaded yet: uploads wait for the coordinator's approval."),
        extra=extra, tables=asks_html,
        source=Raw("<code>data/derived/status/findings.json</code> (coordinator approvals / questions); " + fsrc(fs, "meta_verdict"))))

    B.append('<section class="card" id="gloss"><h2>' + N("Glossary") + '</h2><dl class="gloss">' +
             "".join(f"<dt>{esc(a)}</dt><dd>{b}</dd>" for a, b in story_glossary()) + "</dl></section>")
    return B


STORY_START, STORY_END = "<!-- part 2: story -->", "<!-- end of part 2 -->"


def story_part(html_a: str) -> str:
    """Part 2 of the page (check 11 budgets the story, not the appendix)."""
    i, j = html_a.find(STORY_START), html_a.find(STORY_END)
    return html_a[i:j] if 0 <= i < j else ""


def story_budget_check(html_a: str) -> tuple[str, bool, str]:
    figs = html_a.count('<div class="fig">')
    size = len(html_a.encode())
    over = [f"chapter {r['no']}: {r['words']} words" for r in ST.story if r["words"] > STORY_MAX_WORDS]
    bad = [f"chapter {r['no']}: {b}" for r in ST.story for b in r["bad"]]
    nos = [r["no"] for r in ST.story]
    pend = [p for p in ST.pending if p.startswith("Chapter")]
    ok = figs <= STORY_MAX_FIGS and size < STORY_MAX_BYTES and not over and not bad and nos == list(range(1, 9)) and not pend
    words = ", ".join(f"{r['no']}: {r['words']}" + (f" (+{r['table_words']} in an open table)" if r["table_words"] else "") for r in ST.story)
    return (f"11 story budget: <= {STORY_MAX_FIGS} charts, <= {STORY_MAX_WORDS} words per chapter (outside table cells, collapsed data "
            f"blocks, charts and source lines; no other exemption), page < {STORY_MAX_BYTES // 1024} KB, every chart with a "
            "takeaway title, axis labels and a collapsible table, eight chapters, none pending", ok and bool(html_a),
            f"{figs} charts; {size / 1024:.0f} KB; words per chapter {words}; over budget {over}; chart defects {bad}; "
            f"chapters {nos}; pending {pend}")


def story_recompute(D: dict) -> tuple[str, bool, str]:
    """Findings and cross-file facts the story rests on, recomputed from the files."""
    fs, ss, var, keys = D["findings"]["findings"], D["ss"], D["var"], D["keys"]
    bad, n, info = [], [0], []

    def cmp(label, want, got, nd, scale=1.0):
        if want is None or got is None or got != got:
            bad.append(f"{label}: missing ({want!r} / {got!r})")
            return
        n[0] += 1
        w = num_in(want) if not isinstance(want, (int, float)) else want
        if w is None or abs(w - round(got * scale, nd)) > 10 ** -nd * 0.5001 + 1e-12:
            bad.append(f"{label}: findings {want} vs files {got * scale:.{nd + 1}f}")

    def k_of(v):
        m = re.findall(r"(\d+) of (\d+)", str(v or ""))
        return [(int(a), int(b)) for a, b in m]

    mean = lambda m, md, c: float(np.mean([ss[m][(md, c)][k]["combined"] for k in keys]))  # noqa: E731
    # weights
    wv = fval(fs, "rules_weights_only", "w")
    got = " / ".join(f"{SC.WEIGHTS[p]:g}" for p in PARTS)
    n[0] += 1
    if wv != got:
        bad.append(f"rules_weights_only.w {wv} vs score.WEIGHTS {got}")
    # F0 versus c, re-scored here (independent of sensitivity.csv)
    gc = fval(fs, "gospa_c", "v")
    for c, v in zip((100, 50, 30, 20, 10), (gc or "").split("/")):
        cmp(f"gospa_c c={c}", v.strip(), mean("F0_fuse", "all", float(c)), 3)
    if "v3h" in var and "v3a" in var:
        mh, ma = var["v3h"], var["v3a"]
        cmp("v3_azimuth_score.m30h", fval(fs, "v3_azimuth_score", "m30h"), mean(mh, "all", 30.0), 3)
        cmp("v3_azimuth_score.m30a", fval(fs, "v3_azimuth_score", "m30a"), mean(ma, "all", 30.0), 3)
        def wins(c, md):
            return CP.count_wlt([ss[ma][(md, c)][k]["combined"] - ss[mh][(md, c)][k]["combined"] for k in keys])[0]
        for fk, cs in (("c2030", (20.0, 30.0)), ("c50", (50.0,)), ("c100", (100.0,))):
            want = k_of(fval(fs, "v3_azimuth_score", fk))
            gotl = [(wins(c, md), len(keys)) for c in cs for md in MODES]
            if fk == "c2030":
                want = want * len(gotl)
            n[0] += 1
            if want != gotl:
                bad.append(f"v3_azimuth_score.{fk}: findings {want} vs files {gotl}")
        s11k = next((k for k in keys if short(k) == "S1-I1"), None)
        if s11k:
            cmp("v3_honesty_drop.s11", fval(fs, "v3_honesty_drop", "s11"),
                ss[mh][("all", 30.0)][s11k]["combined"] - ss["F0_fuse"][("all", 30.0)][s11k]["combined"], 3)
    # truth composition
    td = story_truth_data(D["tri"], keys)
    pooled = next((d for d in td if d["key"] == "POOLED"), None)
    if pooled:
        # the page prints its own one-decimal share; the finding's rounded value is compared as INFO
        gv = num_in(fval(fs, "truth_ground_share", "v"))
        if gv is None or abs(gv - round(100 * pooled["parked"])) > 0.5001:
            info.append(f"truth_ground_share.v {fval(fs, 'truth_ground_share', 'v')} vs files {100 * pooled['parked']:.2f} %")
        cmp("unseen_airborne.v", fval(fs, "unseen_airborne", "v"), pooled["unseen_of_air"], 1, 100)
        parts = [d for d in td if d["key"] != "POOLED"]
        n[0] += 1
        if sum(d["n_all"] for d in parts) != pooled["n_all"] or sum(d["n_air"] for d in parts) != pooled["n_air"]:
            bad.append("triangulation_rows: per-iteration truth rows do not sum to POOLED")
        worst = max(abs(d["seen"] + d["unseen"] + d["parked"] - 1) for d in td)
        n[0] += 1
        if worst > 1e-9:
            bad.append(f"truth shares do not sum to 1 (max {worst:.1e})")
    # rotation table vs sensor_calibration.csv
    rot, az = load_rotation_table(), D["radar_az"]["az"]
    for x in rot:
        n[0] += 1
        v = az.get(x["radar"], {}).get("testing/" + x["iter"])
        if v is None or abs(v - x["cal"]) > 0.00005001:
            bad.append(f"rotation table {x['radar']} {x['iter']}: {x['cal']} vs sensor_calibration {v}")
    # health
    hj = D.get("health")
    if hj:
        ev = [it.key for it in C.iterations() if it.tree == "evaluation"]
        for fk, path, nd in (("dup", ("near_dup", "step_share"), 3), ("jump", ("jump", "rows"), 0), ("fr", ("friendly", "copy_seconds"), 0)):
            want = (fval(fs, "health_eval", fk) or "").split("/")
            for k, wv in zip(ev, want):
                d = hj["iterations"][k]
                cmp(f"health_eval.{fk} {short(k)}", wv.strip(), float(d[path[0]][path[1]]), nd)
        td_max = max(hj["iterations"][k]["near_dup"]["step_share"] for k in keys)
        cmp("health_eval.tdup (max)", (fval(fs, "health_eval", "tdup") or "").split("-")[-1], td_max, 2)
    # tiers: ladder.csv vs tier_models.md vs the tier_ladder finding (t<k>wins / t<k>gain / routed / ll)
    tiers = D.get("tiers") or []
    tm = D.get("tier_meta") or {}
    n[0] += 1
    if not tiers or not tm:
        bad.append("tier ladder or the decision lines of tier_models.md not found")
    for t in tiers:
        n[0] += 1
        if t["wins"] != t["wins_recount"]:
            bad.append(f"tier {t['tier']}: wins_vs_prev {t['wins']} vs recount {t['wins_recount']}")
        if t["tier"] > 1 and (t["key_field"] is None or t["key_field"] not in t["fields"]):
            bad.append(f"tier {t['tier']}: name '{t['name']}' not backed by its fields {t['fields']}")
        wv, gv = fval(fs, "tier_ladder", f"t{t['tier']}wins"), fval(fs, "tier_ladder", f"t{t['tier']}gain")
        if wv is not None:
            n[0] += 1
            if k_of(wv) != [(t["wins"], t["n"])]:
                bad.append(f"tier_ladder.t{t['tier']}wins {wv} vs ladder {t['wins']} of {t['n']}")
        if gv is not None:
            cmp(f"tier_ladder.t{t['tier']}gain", gv, t["gain"], 3)
    if tm:
        rv = fval(fs, "tier_ladder", "routed")
        n[0] += 1
        if rv is not None and k_of(rv) != [(tm["routed_w"], tm["routed_n"])]:
            bad.append(f"tier_ladder.routed {rv} vs tier_models.md {tm['routed_w']} of {tm['routed_n']}")
        lv = fval(fs, "tier_ladder", "ll")
        n[0] += 1
        if lv is not None and re.findall(r"\d+\.\d+", lv) != [tm["ll_routed"], tm["ll_single"]]:
            bad.append(f"tier_ladder.ll {lv} vs tier_models.md routed {tm['ll_routed']}, single {tm['ll_single']}")
    return ("12 story numbers recomputed from the files agree with the findings (weights, F0 versus c, v3a / v3h counts and means, "
            "honesty drop, truth shares, rotation table vs sensor_calibration.csv, health, tier ladder)", not bad,
            f"{n[0]} values compared" + (f"; disagree: {bad}" if bad else "") +
            f"; INFO rounded findings that differ from the files' value (the page prints the file value): {info or 'none'}")


# --------------------------------------------------------------------------- #
# Self-checks
# --------------------------------------------------------------------------- #


class TagCheck(HTMLParser):
    VOID = {"meta", "br", "hr", "img", "input", "link", "col", "area", "base", "wbr", "source"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.errors, self.ids = [], [], []

    def _ids(self, attrs):
        self.ids += [v for k, v in attrs if k == "id"]

    def handle_starttag(self, tag, attrs):
        self._ids(attrs)
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self._ids(attrs)

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        else:
            self.errors.append(f"</{tag}> with open {self.stack[-3:]}")
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass


def strip_asof(s: str) -> str:
    return re.sub(r'<p id="asof">.*?</p>', "", s)


NUM_ANY = re.compile(r"(?<![\w.])-?\d+\.\d{2,}")


def coord_hits(page: str, fr: dict) -> list[str]:
    """Numbers with >= 2 decimals anywhere in the page (text, tooltips, SVG geometry, CSS) that lie
    within 0.5 deg of an origin. Integers and 1-decimal numbers are not scanned: at that precision
    (>= ~10 km) a value locates nothing, and percentages / pixel coordinates such as 51.8 would
    otherwise collide with the latitude band."""
    origins = [(f.lat0, f.lon0) for f in fr.values()]
    cands = {float(m) for m in NUM_ANY.findall(page)}
    hits = []
    for v in cands:
        for la, lo in origins:
            if abs(v - la) < 0.5 or abs(v - lo) < 0.5:
                hits.append("value near an origin (not printed)")
                break
    return hits


def run_checks(D: dict, html_a: str, html_b: str, out: Path, allow_stale: bool) -> list[tuple[str, bool, str]]:
    res = []
    keys, sc = D["keys"], D["scores"]
    # 1
    f0 = sc["F0_fuse"]
    worst, n_cmp, msg = 0.0, 0, []
    sens = {(r["truth_mode"], r["iter_key"]): float(r["combined"]) for r in D["sens"]
            if r["kind"] == "method" and r["method"] == "F0_fuse" and r["variant"] == "" and r["config"] == "c50_p2_3d"}
    prb = {(r["truth_mode"], r["iter_key"]): float(r["combined"]) for r in D["probes"]
           if r["probe"] == "P0_f0" and r["label_rule"] == "proxy" and r["config"] == "c50_p2_3d"}
    for mode in MODES:
        for k in keys:
            v = f0[mode][k]["combined"]
            for nm, srcd in (("sensitivity.csv", sens), ("probes.csv", prb)):
                if (mode, k) not in srcd:
                    msg.append(f"{nm} lacks {mode} {short(k)}")
                    continue
                d = abs(srcd[(mode, k)] - v)
                worst = max(worst, d)
                n_cmp += 1
    ok1 = not msg and worst <= 1e-6 and n_cmp == 4 * len(keys)
    res.append(("1 F0 combined per iteration agrees across re-score / sensitivity.csv / probes.csv P0 (c50_p2_3d, both modes)",
                ok1, f"{n_cmp} comparisons, max |diff| {worst:.2e}" + (f"; {'; '.join(msg[:4])}" if msg else "")))
    # 2
    bad = []
    for (a, b, mode, part), r in ST.wlt_rendered.items():
        w = l = t = 0
        for k in keys:
            d = sc[a][mode][k][part] - sc[b][mode][k][part]
            if d != d:
                continue
            if abs(d) < CP.TIE:
                t += 1
            elif d > 0:
                w += 1
            else:
                l += 1
        if (w, l, t) != (r["w"], r["l"], r["t"]) or sum(v[0] for v in r["scen"].values()) != w:
            bad.append(f"{a} vs {b} {mode}")
        if f"{r['w']} win / {r['l']} loss / {r['t']} tie of {r['n']}" not in html_a and (a, b, mode, part) in ST.wlt_rendered:
            bad.append(f"{a} vs {b} {mode} text missing")
    res.append(("2 rendered win/loss/tie counts equal a plain loop over the differences", not bad and bool(ST.wlt_rendered),
                f"{len(ST.wlt_rendered)} comparisons" + (f"; bad: {bad[:4]}" if bad else "")))
    # 3
    worst3 = max(abs(f0[m][k]["cost_share_loc"] + f0[m][k]["cost_share_missed"] + f0[m][k]["cost_share_false"] - 1)
                 for m in MODES for k in keys)
    res.append(("3 F0 cost shares sum to 1 per iteration and truth mode", worst3 < 1e-9, f"max |sum - 1| {worst3:.1e}"))
    # 4
    stale_list = sorted(D["stale"])
    res.append(("4 every source is newer than the tool that produces it", not stale_list or allow_stale,
                (f"{len(stale_list)} stale{' (allowed by --allow-stale)' if allow_stale else ''}: "
                 + "; ".join(f"{s} (older than {SOURCES[s]['producer']})" for s in stale_list)) if stale_list else
                f"{len(SOURCES)} sources fresh"))
    # 5
    fs = D["findings"]["findings"]
    p_bad, s_bad, lit = [], [], []
    for f in fs:
        ph = set(re.findall(r"\{(\w+)\}", f["text"]))
        if not ph <= set(f.get("values", {})):
            p_bad.append(f["id"])
        sp = f.get("source_path", "")
        paths = re.findall(r"(?:data|tools|README|docs)/[\w./-]+\.\w+", sp)
        if f.get("status") == "journal" and "journal" in sp:
            pass
        elif not paths:
            s_bad.append(f["id"])
        for p in paths:
            if not (C.REPO / p).exists():
                s_bad.append(f"{f['id']}:{p}")
        ln = literal_numbers(f["text"])
        if ln:
            lit.append(f"{f['id']} {ln}")
    co = D["findings"].get("coordinator", {})
    for sect, lines in co.items():
        for i, x in enumerate(lines):
            ln = literal_numbers(x)
            if ln:
                lit.append(f"coordinator.{sect}[{i}] {ln}")
            for m in REF_RE.finditer(x):   # references must resolve even where this page does not render the line
                a, b = m.group(1), m.group(2)
                if (b is not None and fval(fs, a, b) is None) or (b is None and a not in COMPUTED):
                    ST.ctl_unresolved.append(m.group(0))
    unres = sorted(set(ST.ctl_unresolved))
    unrendered = sorted({f["id"] for f in fs} - ST.rendered_findings)
    ok5 = not p_bad and not s_bad and not ST.narr_bad and not unrendered and not lit and not unres
    res.append(("5 findings placeholders resolve, sources exist, all findings rendered, no digit in any narrative text "
                "(tool templates and controller text) that did not come from a value or computed field",
                ok5, f"{len(fs)} findings, {sum(len(v) for v in co.values())} coordinator lines; unresolved {p_bad + unres}; "
                     f"bad sources {s_bad}; unrendered {unrendered}; "
                     f"templates with digits {len(ST.narr_bad)}{(': ' + repr(ST.narr_bad[:3])) if ST.narr_bad else ''}; "
                     f"controller text with literal numbers {len(lit)}{(': ' + ' | '.join(lit)) if lit else ''}; "
                     f"INFO journal findings whose cited file lacks their numbers: {sorted(set(ST.journal_gaps)) or 'none'}"))
    # 6
    tc = TagCheck()
    tc.feed(html_a)
    tc.close()
    svgs = re.findall(r"<svg\b.*?</svg>", html_a, re.S)
    svg_bad = 0
    for s in svgs:
        try:
            ET.fromstring(s)
        except ET.ParseError:
            svg_bad += 1
    dup = sorted({i for i in tc.ids if tc.ids.count(i) > 1})
    low = html_a.lower()
    ext = [t for t in ("http", "src=", "url(", "@import") if t in low]
    ok6 = not tc.errors and not tc.stack and not svg_bad and not dup and not ext
    res.append(("6 HTML tags balanced, every <svg> parses as XML, ids unique, offline", ok6,
                f"{len(svgs)} svg ({svg_bad} bad); tag errors {tc.errors[:3]}; unclosed {tc.stack[-3:]}; dup ids {dup}; external {ext}"))
    # 7
    fr = D["frames"]
    hits = coord_hits(html_a, fr)
    any_fr = next(iter(fr.values()))
    plant1 = html_a + f'<p>{any_fr.lat0 + 0.1234:.6f}</p>'
    plant2 = html_a.replace("</main>", f'<p data-tip="x {any_fr.lon0 - 0.2:.2f} deg">-</p></main>', 1)
    planted_ok = bool(coord_hits(plant1, fr)) and bool(coord_hits(plant2, fr))
    key_ids = D["key_ids"]
    key_hit = [1 for kid in key_ids if kid and kid in html_a]
    hex8 = re.findall(r"(?<![0-9A-Za-z_#-])[0-9a-f]{8}(?![0-9A-Za-z_])", strip_asof(html_a))
    up = [s for s, i in SOURCES.items() if str(i["path"]).startswith(str(UPLOAD))]
    ok7 = not hits and planted_ok and not key_hit and not hex8 and not up
    res.append(("7 no number within 0.5 deg of an iteration origin, planted values found, no key_id / fingerprint, upload/ never read",
                ok7, f"{len(fr)} origins; hits {len(hits)}; planted control {'found' if planted_ok else 'MISSED'}; key_id {len(key_hit)}; "
                     f"8-hex tokens {len(hex8)}; upload files read {len(up)}"))
    # 8
    nan = re.findall(r"\b(?:nan|inf|NaN|Infinity)\b", re.sub(r"<(style|script)\b.*?</\1>", "", html_a, flags=re.S))
    ok8 = not ST.fig_without_table and not nan and not ST.out_axes
    res.append(("8 every figure has a data table, no nan / inf, plotted values inside their axes", ok8,
                f"figures without table {ST.fig_without_table}; nan/inf {len(nan)}; outside axes {len(ST.out_axes)} {ST.out_axes[:3]}"))
    # 9
    ha, hb = (hashlib.sha256(strip_asof(x).encode()).hexdigest() for x in (html_a, html_b))
    size = len(html_a.encode())
    prev_note = "no previous page"
    prev_ok = True
    if out.exists():
        newest = max([i["path"].stat().st_mtime for i in SOURCES.values()] + [Path(__file__).stat().st_mtime])
        if newest < out.stat().st_mtime:
            hp = hashlib.sha256(strip_asof(out.read_text(encoding="utf-8")).encode()).hexdigest()
            prev_ok = hp == ha
            prev_note = "previous page " + ("identical" if prev_ok else "DIFFERS although no source changed")
        else:
            prev_note = "previous page not comparable (a source or this tool changed since)"
    res.append(("9 deterministic render (sha256 apart from the as-of line) and page < 3 MB", ha == hb and prev_ok and size < MAX_BYTES,
                f"two renders {'identical' if ha == hb else 'DIFFER'} ({ha[:12]}); {prev_note}; {size / 1024:.0f} KB"))
    # 10
    res.append(recompute_findings(D))
    res.append(story_budget_check(story_part(html_a)))
    res.append(story_recompute(D))
    return res


def recompute_findings(D: dict) -> tuple[str, bool, str]:
    fs, sc, keys = D["findings"]["findings"], D["scores"], D["keys"]
    bad, n = [], 0

    def cmp(fid, key, got, nd, scale=1.0):
        nonlocal n
        v = fval(fs, fid, key)
        if v is None or got is None or got != got:
            return
        n += 1
        want = num_in(v)
        if want is None or abs(want - round(got * scale, nd)) > 10 ** -nd * 0.5001 + 1e-12:
            bad.append(f"{fid}.{key}: findings {v} vs files {got * scale:.{nd + 1}f}")

    mean = lambda m, mode: float(np.mean([sc[m][mode][k]["combined"] for k in keys])) if m in sc else None  # noqa: E731
    cmp("f0_mean", "all", mean("F0_fuse", "all"), 3)
    cmp("f0_mean", "air", mean("F0_fuse", "airborne"), 3)
    cmp("f0_mean", "b2all", mean("B2_allpos", "all"), 3)
    cmp("f0_mean", "b2air", mean("B2_allpos", "airborne"), 3)
    sens = defaultdict(list)
    for r in D["sens"]:
        if r["p"] in ("2", "2.0") and r["dims"] in ("3", "3.0") and r["truth_mode"] == "all" and r["iter_key"] in keys:
            sens[(r["kind"], r["method"], r["variant"], float(r["c"]))].append(float(r["combined"]))
    gc = fval(fs, "gospa_c", "v")
    if gc:
        for c, v in zip((100, 50, 30, 20, 10), gc.split("/")):
            got = sens.get(("method", "F0_fuse", "", float(c)))
            if got:
                n += 1
                if abs(float(v) - round(float(np.mean(got)), 3)) > 0.0005001:
                    bad.append(f"gospa_c c={c}: {v.strip()} vs {np.mean(got):.4f}")
    for key, c in (("c20", 20.0), ("c50", 50.0)):
        a, b = sens.get(("oracle", "O3_fuse_oracle", "bias", c)), sens.get(("method", "O3_fuse_oracle", "", c))
        if a and b:
            cmp("bias_oracle", key, float(np.mean(a)) - float(np.mean(b)), 3)
    pooled = {(r["truth_mode"], r["unseen_def"]): r for r in D["tri"] if r["iter_key"] == "POOLED" and r["offsets"] == "oracle"}
    if ("airborne", "main") in pooled:
        cmp("unseen_airborne", "v3d", float(pooled[("airborne", "main")]["share_unseen"]), 1, 100)
        cmp("unseen_airborne", "v", float(pooled[("airborne", "main_2d")]["share_unseen"]), 1, 100)
        cmp("triangulation_zero", "v", float(pooled[("airborne", "main")]["share_ge2_site"]), 1, 100)
    f1v = {"all": fval(fs, "f1_lost", "all"), "airborne": fval(fs, "f1_lost", "air")}
    if "F1_fuse" in sc:
        for mode, v in f1v.items():
            if not v:
                continue
            want = [0, 0, 0]
            for num, word in re.findall(r"(\d+)\s+(win|loss|tie)", str(v)):
                want[("win", "loss", "tie").index(word)] = int(num)
            got = list(rounded_wlt(sc, "F1_fuse", "F0_fuse", mode, keys))
            n += 1
            if want != got:
                bad.append(f"f1_lost.{mode}: findings {v} vs files (rounded to {F1_ND} decimals) {got[0]} win / {got[1]} loss / {got[2]} tie")
    az = D["radar_az"]["az"]
    for rd in ("EGOI", "NYTF", "UXYJ", "IUKL"):
        v = fval(fs, "az_bias_stable", rd.lower())
        if not v:
            continue
        want = [float(x) for x in v.split("/")]
        got = [round(az[rd][k], 2) for k in keys if k in az.get(rd, {})]
        n += 1
        if len(want) != len(got) or any(abs(a - b) > 0.0050001 for a, b in zip(want, got)):
            bad.append(f"az_bias_stable.{rd.lower()}: {v} vs {' / '.join(f'{x:+.2f}' for x in got)}")
    return ("10 file-backed findings recomputed from the files agree at their printed precision", not bad,
            f"{n} values compared" + (f"; disagree: {bad}" if bad else ""))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def staleness() -> set[str]:
    out = set()
    for name, info in SOURCES.items():
        prod = info["producer"]
        if prod and (TOOLS / prod).exists() and info["path"].stat().st_mtime < (TOOLS / prod).stat().st_mtime:
            out.add(name)
    return out


PAGE_CSS = """
.parts{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 4px;padding:0;list-style:none;font-size:14px}
.parts li a{display:inline-block;border:1px solid var(--border);border-radius:8px;padding:4px 10px;background:var(--surface-1);text-decoration:none}
.part{margin:34px 0 6px;border-top:2px solid var(--border);padding-top:10px} .part h2{font-size:24px;margin:2px 0 4px}
details.appendix>summary{cursor:pointer;font-size:16px;font-weight:600;color:var(--text-primary);padding:8px 0}
"""


def part_head(no: int, pid: str, title: str, what: str) -> str:
    return (f'<div class="part" id="{pid}"><p class="kicker">{N("Part {n}", n=no)}</p><h2>{N(title)}</h2>'
            f'<p class="what">{N(what)}</p></div>')


def page_body(D: dict) -> list[str]:
    """Part 1 summary, part 2 story, part 3 the appendix folded shut."""
    nav = ('<ul class="parts">' + "".join(f'<li><a href="#{a}">{N(t)}</a></li>' for a, t in (
        ("summary", "Part one · the summary in one screen"), ("story", "Part two · the story in eight chapters"),
        ("appendix", "Part three · technical appendix"))) + "</ul>")
    B = [nav]
    B.append(part_head(1, "summary", "The summary in one screen",
                       "For the coordinator: what exists, the honest scores, what needs approval and what we ask the organisers."))
    B += coordinator_body(D)
    B.append(STORY_START)
    B.append(part_head(2, "story", "The story in eight chapters",
                       "What we found, in order, with one explanatory chart per chapter."))
    B += story_body(D)
    B.append(STORY_END)
    B.append(part_head(3, "appendix-head", "Technical appendix",
                       "Every table and chart behind the story, for checking a number. Folded: click to open."))
    B.append('<details class="appendix" id="appendix"><summary>' + N("Open the technical appendix") + "</summary>")
    B += report_body(D)
    B.append("</details>")
    B.append('<p class="src">' + N("Generated by {t}; local file, do not publish.", t="tools/status_report.py") + "</p>")
    return B


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--allow-stale", action="store_true", help="a source older than its producing tool does not fail check 4")
    ap.add_argument("--debug-out", default="", help="also write the page here when a self-check fails (inspection only)")
    ap.add_argument("--findings", default="", help="read this findings.json instead of data/derived/status/findings.json")
    a = ap.parse_args()
    if a.findings:
        global FINDINGS
        FINDINGS = Path(a.findings).resolve()
    t0 = time.time()
    out = OUT_PAGE
    if a.findings:   # a trial findings file never overwrites the page built from the controller's file
        out = FINDINGS.parent / out.name
    keys = testing_keys()
    D: dict = {"keys": keys}
    D["eval_scen"] = {it.scenario for it in C.iterations() if it.tree == "evaluation"}
    D["deployable"] = C.deployable_sensor_ids()
    D["findings"] = load_findings()
    D["scores"], D["meta"], D["var"] = gather_scores(False, keys)
    story_scores, _, _, D["ss"] = gather_story_scores(keys)
    D["scores"].update({k: v for k, v in story_scores.items() if "@" in k})   # the story's per-cut-off blocks
    if "F0_fuse" not in D["scores"]:
        print("F0_fuse has no complete set of testing files: cannot build the page")
        return 1
    D["sens"] = load_sensitivity()
    D["probes"], D["probes_md"] = load_probes()
    D["probe_names"] = sorted({r["probe"] for r in D["probes"]} | {p.name for p in C.SUBMISSIONS.glob("P_*") if p.is_dir()})
    D["cal"] = read_csv(DER / "sensor_calibration.csv", "gt_associate.py")
    op = DER / "radar_registration" / "offsets.csv"
    D["offs"] = read_csv(op, "radar_registration.py") if op.exists() else []
    tp = DER / "triangulation" / "triangulation_rows.csv"
    D["tri"] = read_csv(tp, "triangulation_oracle.py") if tp.exists() else []
    D["frames"] = frames()
    D["key_ids"] = []
    for root in C.TREES.values():
        mp = root / "ANONYMIZATION_MANIFEST.json"
        if mp.exists():
            D["key_ids"].append(json.loads(src(mp).read_text(encoding="utf-8")).get("key_id", ""))
    D["code_hash"] = SC.code_hash()
    _, _, D["radar_az"] = radar_minis(D["cal"], D["offs"], keys, D["deployable"])
    if PLAN.exists():
        src(PLAN)
    deanon_commands()
    D["tiers"], D["tier_meta"] = load_tiers_retry()
    # sources are all registered by the first render too; render once to register, then fix staleness
    def render() -> str:
        global ST
        ST = State()
        body = page_body(D)
        newest = max(i["path"].stat().st_mtime for i in SOURCES.values())
        asof = (f"As of {time.strftime('%Y-%m-%d %H:%M')} local time · newest source "
                f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(newest))} · local file, do not publish")
        return page("C-UAS fusion: where we stand", body, asof, STORY_CSS + PAGE_CSS)
    D["stale"] = set()
    render()
    D["stale"] = staleness()
    html_b = render()
    st_b = ST
    html_a = render()
    checks = run_checks(D, html_a, html_b, out, a.allow_stale)
    print(f"\nself-checks ({out.name})")
    for name, ok, info in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {info}")
    pend = list(dict.fromkeys(ST.pending))
    print("\npending cards:" + ("".join(f"\n  - {p}" for p in pend) if pend else " none"))
    del st_b
    if all(ok for _, ok, _ in checks):
        out.write_text(html_a, encoding="utf-8")
        print(f"\nwrote {rel(out)} ({len(html_a.encode()) / 1024:.0f} KB) in {time.time() - t0:.0f} s")
        if not a.findings:
            for nm in OLD_PAGES:   # the separate pages this one replaced, so no stale copy is opened by mistake
                old = DER / nm
                if old.exists():
                    old.unlink()
                    print(f"removed the replaced page {rel(old)}")
        print("ALL SELF-CHECKS PASSED")
        return 0
    if a.debug_out:
        Path(a.debug_out).write_text(html_a, encoding="utf-8")
    print(f"\nSELF-CHECK FAILED: {rel(out)} not written ({time.time() - t0:.0f} s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
