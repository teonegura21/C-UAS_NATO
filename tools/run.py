#!/usr/bin/env python3
"""ONE command that rebuilds everything from the raw data to validated submissions.

Why: every script reads the previous scripts' outputs in data/derived/, so a script that is
re-run alone, or run in the wrong order, silently works on stale inputs and the results diverge.
This runner executes the whole chain in dependency order, one subprocess per step, with a log per
step, a summary table and a machine-readable manifest. It is also the single command for the live
demo. It contains no analysis of its own.

Usage (run from the repo root; numpy + openpyxl only)
    python3 tools/run.py                       full pipeline (no analysis steps)
    python3 tools/run.py --plan                print the ordered plan (inputs, outputs, runtime), run nothing
    python3 tools/run.py --from fuse           that step and everything after it
    python3 tools/run.py --only gospa gbdt     just these steps (always executed in plan order)
    python3 tools/run.py --skip bearing_support  everything except these steps
    python3 tools/run.py --with-analysis       also run the analysis / diagnostic steps
    python3 tools/run.py --selftest            test the scheduler itself on dummy commands (no data)
    --from / --skip / --with-analysis combine; --only cannot be combined with --from.

Steps (plan order; `--plan` prints inputs, outputs and the expected runtime of each)
    data prep      build_detections, sensor_inventory
    calibration    gt_associate (testing only), truth, sensor_noise, metadata_scan
    tracks/models  sensor_tracks, existence, classify, bearing_support, fuse (F0_fuse),
                   track_existence (F1_fuse; only if the file exists), probes (only if it exists)
    self-tests     gospa, gbdt, track_rows, submission_selftest, score_selftest
    analysis       baselines, pair_features, ground_coverage, gospa_sensitivity, triangulation_oracle,
                   radar_registration  (only with --with-analysis; each only if the file exists)
    final          validate_submissions: every submission of the deployable methods, all 9
                   iterations, through submission.validate_submission (plus the probe methods
                   P0_f0 / P_park / P_meta / P_ident when their directories exist); prints the
                   evaluation files the owner would upload and flags STALE ones.
`sensor_inventory` is not in the original brief but is a hard dependency of sensor_noise
(sensor_sites.csv); without it a clean data/derived/ cannot be rebuilt. Analysis runs before the
final step so that the validation summary is the last thing on the screen.

Behaviour
* Stops at the first UNEXPECTED failure (non-zero exit). The steps after it are reported 'not run'.
* KNOWN_NONZERO is an explicit allowlist: a step that exits 1 BY DESIGN because named self-checks
  fail on purpose (bearing_support check 3, pair_features checks 4 and 5). The failing check numbers
  are parsed from the step's log and must be a non-empty subset of the allowlisted ones, otherwise
  the failure is unexpected. Allowlisted failures are printed loudly in the summary and in the
  manifest (`status: known_nonzero`, the reason) but do not stop the run. A step on the list that
  now exits 0 is reported ('now passes: remove it from KNOWN_NONZERO').
* Logs: data/derived/run_logs/<UTC timestamp>/<nn>_<step>.log (stdout + stderr; nn = position in
  the full plan, so it is stable across --only / --from), run_manifest.json and, after the final
  step, validation.json.
* run_manifest.json: UTC start / end, argv, the selection, the code hash (score.code_hash(): sha256
  of every tools/*.py, taken at the start and again at the end), per step (status, exit code,
  seconds, failed checks, last self-check line, log file) and, for every submission file of the
  produced and validated methods: sha256, byte size, non-empty row count, `fresh` (modified since
  the run started). Longitudes in those files are in the anonymized frame; the manifest holds no
  coordinates.
* A step that declares `produces` (fuse, track_existence, probes, baselines) must leave at least one
  .xlsx modified since the step started in a matching method directory of data/derived/submissions/;
  otherwise it FAILS even with exit 0 (a script with no entry point, or one that wrote nothing, is not
  a silent success).
* Freshness: the manifest marks each submission file `fresh` (modified since the run started) and lists
  the others under `stale_submissions`; the console summary and the final validation print a loud
  warning for them (e.g. `--skip track_existence` leaves old F1_fuse files). Stale files do not fail the
  run: `--only validate_submissions` on existing files is a legitimate use.
* Exit status: 0 when every executed step passed or is a known-nonzero; 1 on an unexpected failure;
  2 on a usage error (unknown step name, missing script of a non-optional step).

Rules this runner does NOT bend
* De-anonymising the longitudes is the owner's separate step (tools/deanonymize.py). The submissions
  written by this pipeline are in the anonymized frame.
* The final step validates the deployable methods (F0_fuse, F1_fuse), which must exist for all 9
  iterations, and the leaderboard probes (PROBE_METHODS), which are validated when their directory
  exists. Baselines and oracles are diagnostics (some exist only for part of the iterations).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
BASE_DERIVED = REPO / "data" / "derived"
# The derived tree this process works on: the main tree, or a VARIANT tree (--derived, or CUAS_DERIVED in the
# environment of a step subprocess; tools/cuas_common.py honours the same variable).
DERIVED = Path(os.environ["CUAS_DERIVED"]).resolve() if os.environ.get("CUAS_DERIVED") else BASE_DERIVED
SUBMISSIONS = DERIVED / "submissions"
LOG_ROOT = DERIVED / "run_logs"
sys.path.insert(0, str(TOOLS))

PY = sys.executable or "python3"
D = "data/derived/"                      # repo-relative prefix used in the plan tables


# --------------------------------------------------------------------------- the plan


@dataclasses.dataclass(frozen=True)
class Step:
    name: str
    script: Optional[str] = None         # file name in tools/ (None for dummy steps)
    args: tuple = ()
    group: str = ""
    desc: str = ""
    inputs: tuple = ()                   # repo-relative paths read (files or directories)
    outputs: tuple = ()                  # repo-relative paths written
    est_s: Optional[float] = None        # last recorded runtime in seconds, None = unmeasured
    optional: bool = False               # silently absent when the script does not exist
    analysis: bool = False               # only with --with-analysis
    produces: tuple = ()                 # submission method names (fnmatch patterns)
    argv_override: Optional[tuple] = None  # dummy steps in the self-test

    def command(self, log_dir: Optional[Path] = None, since: Optional[float] = None) -> list[str]:
        if self.argv_override is not None:
            return list(self.argv_override)
        args = [a.replace("{log_dir}", str(log_dir) if log_dir else "<log_dir>")
                 .replace("{since}", f"{since:.3f}" if since is not None else "<run start>") for a in self.args]
        return [PY, str(TOOLS / self.script), *args]

    def command_text(self) -> str:
        if self.argv_override is not None:
            return " ".join(self.argv_override)
        return "python3 tools/" + " ".join([self.script, *self.args])

    def available(self) -> bool:
        return self.script is None or (TOOLS / self.script).is_file()


# Exit-non-zero-by-design allowlist: step name -> (exit codes, check numbers allowed to fail, reason).
KNOWN_NONZERO = {
    "bearing_support": dict(
        exit_codes=(1,), failed_checks=frozenset({"3"}),
        reason="self-check 3 (self-calibrated camera / acoustic offsets agree with the ground-truth reference "
               "offsets) fails by design: the GT-free registration is conservative and the GT reference "
               "offsets are themselves disputed (kept failing on purpose, see bearing_support.md)."),
    "pair_features": dict(
        exit_codes=(1,), failed_checks=frozenset({"4", "5"}),
        reason="self-check 4 (the learned pair model beats the hand rule in mean held-out F1 at p > 0.5) fails "
               "only for continuation pairs (model 0.000 vs hand rule 0.002: ~90 true pairs among 1.7 million "
               "candidates, almost none reach p > 0.5; for overlap pairs the model wins, 0.254 vs 0.161). "
               "Self-check 5 (per-bin calibration of p_same) fails because the highest raw scores overstate the "
               "same-drone share on a new iteration (overlap bin 0.8-0.95: mean p 0.849 vs observed 0.381): "
               "use p_same as a ranking, not as a probability (pairs.md, Caveats)."),
}

# Methods the owner may upload: written for all 9 iterations, validated by the final step.
DEPLOYABLE_METHODS = ("F0_fuse", "F1_fuse")
# Leaderboard probes (tools/probes.py, written for all 9 iterations, meant for upload): produced by the
# probes step, validated when present on disk (a probe directory that exists must be complete and valid).
PROBE_METHODS = ("P0_f0", "P_park", "P_meta", "P_ident")


def make_steps() -> list[Step]:
    RAW = ("data/Anon_Testing", "data/Anon_Evaluation")
    s: list[Step] = []
    add = s.append
    add(Step("build_detections", "build_detections.py", group="data prep",
             desc="raw sensor CSVs -> one normalised row per report, local ENU, Dronetag ground truth",
             inputs=RAW, outputs=(D + "detections.csv.gz", D + "groundtruth.csv.gz", D + "iteration_refs.csv",
                                  D + "build_log.md"), est_s=35))
    add(Step("sensor_inventory", "sensor_inventory.py", group="data prep",
             desc="per-file inventory and co-located sensor sites (sensor_noise needs sensor_sites.csv)",
             inputs=RAW, outputs=(D + "sensor_inventory.csv", D + "sensor_sites.csv", D + "fusion_links.csv",
                                  D + "sensor_devices.md")))
    add(Step("gt_associate", "gt_associate.py", group="calibration",
             desc="TESTING ONLY: clock / azimuth calibration, detection<->Dronetag matching",
             inputs=(D + "detections.csv.gz", D + "groundtruth.csv.gz"),
             outputs=(D + "sensor_calibration.csv", D + "det_gt_match.csv.gz", D + "gt_association.md")))
    add(Step("truth", "truth.py", group="calibration",
             desc="ground truth on the template's 100 ms grid (testing) + its self-checks",
             inputs=(D + "groundtruth.csv.gz", D + "iteration_refs.csv"), outputs=(D + "truth_summary.md",)))
    add(Step("sensor_noise", "sensor_noise.py", group="calibration",
             desc="per-sensor error model (testing) + u_ellipsoid height rule",
             inputs=(D + "det_gt_match.csv.gz", D + "detections.csv.gz", D + "groundtruth.csv.gz",
                     D + "sensor_calibration.csv", D + "sensor_sites.csv"),
             outputs=(D + "sensor_noise.csv", D + "sensor_noise_by_iteration.csv", D + "sensor_noise.md",
                      D + "sensor_noise_loio.csv")))
    add(Step("metadata_scan", "metadata_scan.py", group="calibration",
             desc="serial / model / protocol / band values -> dropdown mapping evidence, friendly whitelist",
             inputs=(D + "detections.csv.gz", D + "groundtruth.csv.gz", D + "track_labels.csv"),
             outputs=(D + "metadata_values.csv", D + "drone_metadata_testing.csv",
                      D + "friendly_whitelist.csv", D + "metadata_report.md")))
    add(Step("sensor_tracks", "sensor_tracks.py", group="tracks/models",
             desc="per-sensor track segments, Kalman/RTS on the 100 ms grid, features, testing labels",
             inputs=(D + "detections.csv.gz", D + "sensor_calibration.csv", D + "sensor_noise.csv",
                     D + "det_gt_match.csv.gz"), outputs=(D + "sensor_tracks_v2",)))
    add(Step("existence", "existence.py", group="tracks/models",
             desc="P(segment is a real drone), nested leave-one-iteration-out",
             inputs=(D + "sensor_tracks_v2",), outputs=(D + "existence",), est_s=157))
    add(Step("classify", "classify.py", group="tracks/models",
             desc="P(fixed wing) per segment, leave-one-iteration-out",
             inputs=(D + "sensor_tracks_v2",), outputs=(D + "classify",)))
    add(Step("bearing_support", "bearing_support.py", group="tracks/models",
             desc="do camera / acoustic / RF direction-finders point at a track; GT-free azimuth registration",
             inputs=(D + "sensor_tracks_v2", D + "existence", D + "detections.csv.gz",
                     D + "sensor_calibration.csv"), outputs=(D + "bearing_support",), est_s=89))
    add(Step("fuse", "fuse.py", group="tracks/models",
             desc="multi-sensor fusion tracker, method F0_fuse (LOIO-tuned); also the O3 oracle",
             inputs=(D + "sensor_tracks_v2", D + "existence", D + "sensor_noise_by_iteration.csv",
                     D + "friendly_whitelist.csv"),
             outputs=(D + "fused", D + "submissions/F0_fuse", D + "submissions/O3_fuse_oracle"),
             est_s=522, produces=("F0_fuse", "O3_fuse_oracle")))
    add(Step("track_existence", "track_existence.py", group="tracks/models", optional=True,
             desc="track-level existence model on the fused tracks, method F1_fuse",
             inputs=(D + "fused", D + "existence", D + "bearing_support", D + "classify", D + "sensor_tracks_v2"),
             outputs=(D + "track_existence", D + "submissions/F1_fuse"), est_s=378, produces=("F1_fuse",)))
    add(Step("probes", "probes.py", group="tracks/models", optional=True,
             desc="leaderboard probes P0_f0 / P_park / P_meta / P_ident (one-factor variants of F0_fuse); "
                  "CLI assumed to take no arguments; a run that writes no fresh probe submission FAILS",
             inputs=(D + "submissions/F0_fuse", D + "fused"),
             outputs=tuple(D + "submissions/" + m for m in PROBE_METHODS) + (D + "probes",),
             produces=PROBE_METHODS))
    add(Step("gospa", "gospa.py", group="self-tests", desc="GOSPA metric + rectangular assignment solver (no data)", est_s=2))
    add(Step("gbdt", "gbdt.py", group="self-tests", desc="numpy gradient-boosting classifier (no data)", est_s=11))
    add(Step("track_rows", "track_rows.py", group="self-tests",
             desc="tracks -> submission rows builder (selftest when run as a script)", est_s=2))
    add(Step("submission_selftest", "submission.py", ("--selftest",), group="self-tests",
             desc=".xlsx writer / reader / validator on the official templates", est_s=70))
    add(Step("score_selftest", "score.py", ("--selftest",), group="self-tests",
             desc="local proxy scorer on synthetic submissions", est_s=32))
    add(Step("baselines", "baselines.py", ("--no-log",), group="analysis", analysis=True,
             desc="end-to-end baselines and oracle diagnostics (--no-log: experiments.csv stays the owner's log)",
             inputs=(D + "sensor_tracks_v2", D + "sensor_noise.csv", D + "friendly_whitelist.csv"),
             outputs=(D + "baselines.md", D + "submissions/B*", D + "submissions/O1_oracle_exist",
                      D + "submissions/O2_oracle_dedup"),
             est_s=582, produces=("B0_radar", "B1_*", "B2_allpos", "O1_oracle_exist", "O2_oracle_dedup")))
    add(Step("pair_features", "pair_features.py", group="analysis", analysis=True,
             desc="learned P(two segments are one object); checks 4 and 5 fail by design (KNOWN_NONZERO)",
             inputs=(D + "sensor_tracks_v2", D + "sensor_noise_by_iteration.csv"),
             outputs=(D + "pairs",), est_s=684))
    add(Step("ground_coverage", "ground_coverage.py", group="analysis", analysis=True,
             desc="how much of the truth is parked drones, who sees them, should we report them",
             inputs=(D + "detections.csv.gz", D + "groundtruth.csv.gz"), outputs=(D + "ground_coverage.md",)))
    for nm, d in (("gospa_sensitivity", "re-score decisions on a grid of GOSPA c and p"),
                  ("triangulation_oracle", "share of unseen truth with azimuths from >= 2 direction-finders"),
                  ("radar_registration", "GT-free radar offset estimation against the cyber / UTM tracks")):
        add(Step(nm, nm + ".py", group="analysis", analysis=True, optional=True,
                 desc=d + " (CLI assumed to take no arguments)"))
    methods = list(DEPLOYABLE_METHODS)
    if not (TOOLS / "track_existence.py").is_file():
        methods.remove("F1_fuse")
    optional_methods = list(PROBE_METHODS) if (TOOLS / "probes.py").is_file() else []
    add(Step("validate_submissions", "run.py",
             ("--final-validate", "--methods", *methods,
              *(("--optional-methods", *optional_methods) if optional_methods else ()),
              "--since", "{since}", "--json", "{log_dir}/validation.json"),
             group="final", desc="validate every submission of the deployable methods for all 9 iterations; "
                                 "list the evaluation files to upload",
             inputs=tuple(D + "submissions/" + m for m in methods), est_s=173))
    return s


# --------------------------------------------------------------------------- selection


def select_steps(steps: list[Step], *, from_: Optional[str] = None, only: Optional[list[str]] = None,
                 skip: Optional[list[str]] = None, with_analysis: bool = False) -> dict[str, str]:
    """{step name: 'run' | reason it is not run}. Raises ValueError on a usage error."""
    names = [s.name for s in steps]
    for label, group in (("--from", [from_] if from_ else []), ("--only", only or []), ("--skip", skip or [])):
        for g in group:
            if g not in names:
                raise ValueError(f"{label}: unknown step {g!r}; steps: {', '.join(names)}")
    if only and from_:
        raise ValueError("--only cannot be combined with --from")
    by = {s.name: s for s in steps}
    if from_ and by[from_].analysis and not with_analysis:
        raise ValueError(f"--from {from_}: an analysis step; add --with-analysis")
    out: dict[str, str] = {}
    seen_from = from_ is None
    for s in steps:
        if s.name == from_:
            seen_from = True
        if only:
            reason = "run" if s.name in only else "not in --only"
        elif not seen_from:
            reason = "before --from"
        elif s.analysis and not with_analysis:
            reason = "off (analysis; --with-analysis)"
        else:
            reason = "run"
        if reason == "run" and skip and s.name in skip:
            reason = "--skip"
        if reason == "run" and not s.available():
            reason = "absent (optional script missing)" if s.optional else "MISSING-SCRIPT"
        out[s.name] = reason
    return out


def missing_inputs(chosen: list[Step], root: Path = REPO) -> list[str]:
    """Warnings for inputs that do not exist and are not produced by an earlier chosen step."""
    warns = []
    produced: list[str] = []
    for s in chosen:
        for p in s.inputs:
            if (root / p).exists():
                continue
            if any(p == o or p.startswith(o.rstrip("/") + "/") or fnmatch.fnmatch(p, o) for o in produced):
                continue
            warns.append(f"{s.name}: input {p} does not exist and no earlier selected step writes it")
        produced += list(s.outputs)
    return warns


# --------------------------------------------------------------------------- execution

_FAIL_LINE = re.compile(r"self-?checks?\s+fail\w*[:\s]*(.*)$", re.I)
_CHECK_LINE = re.compile(r"self-?checks?|\[check\]|\ball ok\b|\bchecks? (passed|failed)|all .*passed", re.I)


def log_facts(text: str) -> tuple[str, Optional[set]]:
    """(last self-check line, set of failing check numbers parsed from the last 'SELF-CHECK(S) FAIL...' line)."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    last = next((ln for ln in reversed(lines) if _CHECK_LINE.search(ln)), lines[-1] if lines else "")
    failed = None
    for ln in reversed(lines):
        m = _FAIL_LINE.search(ln)
        if m:
            failed = set(re.findall(r"\d+[a-z]?", m.group(1)))
            break
    return last.strip()[:110], failed


def judge(name: str, rc: int, failed: Optional[set], known: dict) -> tuple[str, str]:
    """(status, note). status in ok | known_nonzero | failed."""
    k = known.get(name)
    if rc == 0:
        return "ok", ("now passes: remove it from KNOWN_NONZERO" if k else "")
    if k and rc in k["exit_codes"] and failed and failed <= set(k["failed_checks"]):
        return "known_nonzero", k["reason"]
    why = f"exit {rc}"
    if k:
        why += f"; allowlisted only for exit {sorted(k['exit_codes'])} with failing checks within " \
               f"{sorted(k['failed_checks'])}, log shows {sorted(failed) if failed else 'no parsable failing checks'}"
    return "failed", why


def unproduced(produces: tuple, root: Path, since: float) -> list[str]:
    """Patterns of `produces` that match no method directory holding a .xlsx modified at or after `since`.
    A step that exits 0 without writing what it promises (e.g. a script with no entry point) is a failure."""
    missing = []
    for pat in produces:
        have = [p for p in root.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, pat)] if root.is_dir() else []
        if not any(f.stat().st_mtime >= since for d in have for f in d.glob("*/*.xlsx")):
            missing.append(pat)
    return missing


def run_plan(full: list[Step], chosen: list[Step], log_dir: Path, *, known: dict = KNOWN_NONZERO,
             echo: Callable[[str], None] = print, env: Optional[dict] = None,
             since: Optional[float] = None, sub_root: Path = SUBMISSIONS) -> list[dict]:
    """Run `chosen` (already in plan order) one subprocess each; stop at the first unexpected failure.
    `since` (epoch seconds, default now) is the run start handed to steps that take {since}."""
    since = time.time() if since is None else since
    log_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    stopped = False
    penv = dict(os.environ if env is None else env)
    penv.setdefault("PYTHONUNBUFFERED", "1")
    for i, st in enumerate(chosen, 1):
        pos = full.index(st) + 1
        rec = dict(name=st.name, plan_pos=pos, command=st.command_text(), status="not_run", exit_code=None,
                   seconds=None, failed_checks=None, last_line="", note="", log=None)
        if stopped:
            results.append(rec)
            continue
        logp = log_dir / f"{pos:02d}_{st.name}.log"
        rec["log"] = logp.name
        echo(f"[{i:>2}/{len(chosen)}] {st.name:<22} running ...")
        t0 = time.monotonic()
        t_wall = time.time() - 2.0           # file-system timestamp granularity
        with logp.open("w", encoding="utf-8") as fh:
            fh.write(f"$ {st.command_text()}\n")
            fh.flush()
            try:
                rc = subprocess.run(st.command(log_dir, since), cwd=REPO, stdout=fh, stderr=subprocess.STDOUT,
                                    env=penv, stdin=subprocess.DEVNULL).returncode
            except OSError as e:
                fh.write(f"cannot start: {e}\n")
                rc = 127
        rec["seconds"] = round(time.monotonic() - t0, 2)
        rec["exit_code"] = rc
        text = logp.read_text(encoding="utf-8", errors="replace")
        rec["last_line"], failed = log_facts(text)
        rec["failed_checks"] = sorted(failed) if failed is not None else None
        rec["status"], rec["note"] = judge(st.name, rc, failed, known)
        if rec["status"] in ("ok", "known_nonzero") and st.produces:
            gone = unproduced(st.produces, sub_root, t_wall)
            if gone:
                rec["status"] = "failed"
                rec["note"] = (f"exit {rc} but no fresh submission .xlsx was written for: {', '.join(gone)} "
                               f"(a step that promises submissions must write them; does the script have an entry point?)")
        tag = {"ok": "ok", "known_nonzero": "KNOWN-NONZERO", "failed": "FAILED"}[rec["status"]]
        echo(f"       {st.name:<22} {tag}  exit {rc}  {rec['seconds']:.1f} s")
        if rec["status"] == "known_nonzero":
            echo(f"       !! {st.name} exited {rc} BY DESIGN (failing checks {rec['failed_checks']}): {rec['note']}")
        if rec["status"] == "ok" and rec["note"]:
            echo(f"       !! {st.name}: {rec['note']}")
        if rec["status"] == "failed":
            echo(f"       !! UNEXPECTED FAILURE ({rec['note']}); last log lines ({logp}):")
            for ln in text.splitlines()[-12:]:
                echo("          | " + ln)
            stopped = True
        results.append(rec)
    return results


def format_summary(results: list[dict]) -> str:
    rows = [("step", "status", "exit", "seconds", "last self-check line")]
    for r in results:
        rows.append((r["name"], r["status"].upper().replace("_", "-"),
                     "-" if r["exit_code"] is None else str(r["exit_code"]),
                     "-" if r["seconds"] is None else f"{r['seconds']:.1f}", r["last_line"]))
    w = [max(len(r[i]) for r in rows) for i in range(4)]
    out = []
    for n, r in enumerate(rows):
        out.append("  ".join(r[i].ljust(w[i]) for i in range(4)) + "  " + r[4])
        if n == 0:
            out.append("  ".join("-" * w[i] for i in range(4)) + "  " + "-" * 20)
    return "\n".join(out)


def overall_status(results: list[dict]) -> str:
    return "failed" if any(r["status"] == "failed" for r in results) else "ok"


# --------------------------------------------------------------------------- manifest


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def resolve_methods(root: Path, patterns: list[str]) -> list[str]:
    have = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    return sorted({m for m in have if any(fnmatch.fnmatch(m, pat) for pat in patterns)})


def submission_records(root: Path, methods: list[str], since: float, count_rows: Callable[[Path], Optional[int]]) -> dict:
    """{method: {slug: {file, sha256, bytes, rows, fresh}}} for every .xlsx under root/<method>/<slug>/."""
    out: dict = {}
    for m in methods:
        for f in sorted((root / m).glob("*/*.xlsx")):
            rec = dict(file=str(f.relative_to(root.parents[2])), sha256=sha256_file(f), bytes=f.stat().st_size,
                       rows=count_rows(f), fresh=f.stat().st_mtime >= since)
            out.setdefault(m, {})[f.parent.name] = rec
    return out


def stale_files(submissions: dict) -> list[str]:
    """Submission files NOT rewritten by this run (mtime before its start): the producer step was skipped or
    did not touch them, so they may come from older code or older inputs."""
    return [r["file"] for m in sorted(submissions) for _, r in sorted(submissions[m].items()) if not r["fresh"]]


def build_manifest(*, started: dt.datetime, ended: dt.datetime, argv: list[str], selection: dict, results: list[dict],
                   code_hash_start: str, code_hash_end: str, submissions: dict, log_dir: Path) -> dict:
    return dict(
        started_utc=started.strftime("%Y-%m-%dT%H:%M:%SZ"), ended_utc=ended.strftime("%Y-%m-%dT%H:%M:%SZ"),
        argv=argv, python=sys.version.split()[0], log_dir=str(log_dir),
        code_hash=code_hash_start, code_hash_end=code_hash_end,
        code_changed_during_run=code_hash_start != code_hash_end,
        selection=selection, overall=overall_status(results),
        stopped_at=next((r["name"] for r in results if r["status"] == "failed"), None),
        known_nonzero=[dict(step=r["name"], exit_code=r["exit_code"], failed_checks=r["failed_checks"],
                            reason=r["note"]) for r in results if r["status"] == "known_nonzero"],
        stale_submissions=stale_files(submissions), steps=results, submissions=submissions)


def make_row_counter(cache: dict) -> Callable[[Path], Optional[int]]:
    """Row counter for real submission files: reuses validation.json (matched by sha256) else reads the file."""
    def count(path: Path) -> Optional[int]:
        sha = sha256_file(path)
        if sha in cache:
            return cache[sha]
        try:
            import cuas_common as C
            import submission as S
            it = next(i for i in C.iterations() if C.iter_slug(i.key) == path.parent.name)
            return int(len(S.read_submission(it, path)["step"]))
        except Exception:
            return None
    return count


# --------------------------------------------------------------------------- final validation step


def freshness_tag(valid: Optional[bool], fresh: Optional[bool]) -> str:
    """Suffix of an upload-list line: validity, plus STALE when the file predates this run."""
    if not valid:
        return "(NOT VALID)"
    return "(valid, STALE: not rewritten by this run)" if fresh is False else "(valid)"


def final_validate(methods: list[str], json_path: Optional[str], optional: Optional[list[str]] = None,
                   since: Optional[float] = None, root: Optional[Path] = None) -> int:
    """Validate `methods` (all must exist for all iterations) plus those of `optional` whose directory exists
    (probes: present means complete). `since` = run start (epoch s): files older than it are flagged STALE,
    loudly but without failing (--only validate_submissions, or a skipped producer, is a legitimate use)."""
    import cuas_common as C
    import submission as S
    print(S.ANON_REMINDER)
    root = SUBMISSIONS if root is None else root
    present = [m for m in (optional or []) if (root / m).is_dir()]
    absent = [m for m in (optional or []) if m not in present]
    methods = list(methods) + present
    its = C.iterations()
    problems_total = 0
    report: dict = {}
    print(f"\nvalidating {len(methods)} method(s) x {len(its)} iterations: {', '.join(methods)}")
    if absent:
        print(f"not validated (no directory yet, run the step that produces them): {', '.join(absent)}")
    for m in methods:
        print(f"\n{m}")
        print(f"  {'iteration':<24}{'status':<9}{'rows':>9}{'tracks':>8}{'MB':>7}")
        for it in its:
            p = C.submission_path(m, it)
            slug = C.iter_slug(it.key)
            if not p.is_file():
                print(f"  {it.key:<24}MISSING  {p.relative_to(C.REPO)}")
                problems_total += 1
                report.setdefault(m, {})[slug] = dict(valid=False, problems=["missing file"], rows=None)
                continue
            warns: list[str] = []
            probs = S.validate_submission(it, p, warns)
            rows = ntr = None
            if not probs:
                rr = S.read_submission(it, p)
                rows, ntr = int(len(rr["step"])), int(len(set(rr["track_id"].tolist())))
            fresh = None if since is None else p.stat().st_mtime >= since
            print(f"  {it.key:<24}{'ok' if not probs else 'INVALID':<9}"
                  f"{'-' if rows is None else rows:>9}{'-' if ntr is None else ntr:>8}{p.stat().st_size / 1e6:>7.1f}")
            for pr in probs[:5]:
                print(f"      PROBLEM {pr}")
            for w in warns[:3]:
                print(f"      warning {w}")
            problems_total += len(probs)
            report.setdefault(m, {})[slug] = dict(valid=not probs, problems=probs[:20], warnings=warns[:20],
                                                  rows=rows, tracks=ntr, sha256=sha256_file(p), fresh=fresh)
    if json_path:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\nFILES THE OWNER WOULD UPLOAD (evaluation iterations; anonymized longitudes, NOT yet de-anonymised):")
    for m in methods:
        for it in its:
            if it.tree != "evaluation":
                continue
            rec = report.get(m, {}).get(C.iter_slug(it.key), {})
            print(f"  [{m}] {C.submission_path(m, it).relative_to(C.REPO)}  "
                  f"{freshness_tag(rec.get('valid'), rec.get('fresh'))}")
    print("\nNEXT (owner's separate step): de-anonymise the longitudes before upload with tools/deanonymize.py\n"
          "  (python3 tools/deanonymize.py --help for its arguments). Which method to upload is the owner's choice,\n"
          "  made on the per-iteration local scores (experiments.csv, score.py) and the leaderboard.")
    if problems_total:
        print(f"FINAL VALIDATION FAILED: {problems_total} problem(s)")
        return 1
    stale = sorted(f"{m}/{slug}" for m, d in report.items() for slug, r in d.items() if r.get("fresh") is False)
    if stale:
        print(f"!! {len(stale)} validated file(s) were NOT rewritten by this run (their producer step was skipped or "
              f"did not touch them; they may come from older code or inputs): {', '.join(stale[:6])}"
              + (" ..." if len(stale) > 6 else ""))
    print("FINAL VALIDATION PASSED: every submission of " + ", ".join(methods) + " is valid for all iterations"
          + (f" ({len(stale)} STALE, see above)" if stale else ""))
    return 0


# --------------------------------------------------------------------------- variant trees
#
# A variant tree (--derived data/derived_<name>) is a second derived/ directory: every tool writes there
# (CUAS_DERIVED), so a rebuild with corrections never overwrites the main tree. The outputs of the steps BEFORE
# the first executed step are symlinked from the main tree (they are unchanged inputs); a guard refuses a run in
# which an executed step would write through such a link, and an mtime fingerprint of every linked target is
# compared before / after the run. `variant.json` in the tree (copied from --variant-config) is read by the
# tools that support corrections (sensor_tracks.py). --promote NAME copies the tree's submissions into the main
# tree as <method>_<NAME> so compare.py / health.py can compare them with the main methods.

_REL_OK = re.compile(r"^[A-Za-z0-9_.\-/]+$")


def _rel(out: str) -> Optional[str]:
    """data/derived-relative path of a Step output, None for outputs outside it or glob patterns."""
    if not out.startswith(D):
        return None
    rel = out[len(D):]
    return rel if rel and _REL_OK.match(rel) and not any(ch in rel for ch in "*?[") else None


# Inputs read by plan steps but produced by no plan step (older analysis scripts whose other outputs must not be
# regenerated, e.g. tools/cluster_tracks.py writes track_labels.csv, read by metadata_scan). Always seeded.
EXTRA_SEED = (D + "track_labels.csv", D + "track_features.csv")


def seed_variant(base: Path, dest: Path, steps: list[Step], chosen: set) -> list[str]:
    """Symlink into `dest` the existing main-tree (`base`) outputs that the variant run needs but does not rebuild:
    every output of the steps BEFORE the first step in `chosen`, the outputs of skipped steps that an executed
    step declares as an input (e.g. metadata_scan's friendly_whitelist.csv read by fuse), and EXTRA_SEED. Outputs
    of skipped DOWNSTREAM steps (F1, probes, baselines) are never linked: they would be stale for the variant.
    Entries already present in `dest` are kept (a re-run of the same variant). Returns the linked names."""
    names = [s.name for s in steps]
    first = min((names.index(n) for n in chosen if n in names), default=len(names))
    need = {_rel(p) for s in steps if s.name in chosen and s.name != "validate_submissions" for p in s.inputs}
    need.discard(None)

    def needed(rel: str) -> bool:
        return any(n == rel or n.startswith(rel.rstrip("/") + "/") for n in need)

    outs = [o for i, s in enumerate(steps) if s.name not in chosen for o in s.outputs
            if i < first or (_rel(o) is not None and needed(_rel(o)))] + list(EXTRA_SEED)
    linked = []
    for o in outs:
        if True:
            rel = _rel(o)
            if rel is None:
                continue
            src, dst = base / rel, dest / rel
            if src.exists() and not dst.exists() and not dst.is_symlink():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
                linked.append(rel)
    return linked


def write_through_risks(chosen: list[Step], dest: Path) -> list[str]:
    """Executed steps whose declared outputs sit at or under a symlink of the variant tree: running them would
    overwrite the main tree's files through the link."""
    risks = []
    for s in chosen:
        for o in s.outputs:
            rel = _rel(o)
            if rel is None:
                continue
            q = dest / rel
            while q != dest and q != q.parent:
                if q.is_symlink():
                    risks.append(f"{s.name} writes {rel} through the seeded link {q.relative_to(dest)}")
                    break
                q = q.parent
    return risks


def link_fingerprint(dest: Path) -> dict[str, tuple]:
    """(size, mtime_ns) of every file reached through a top-level symlink of `dest` (the seeded inputs)."""
    fp: dict[str, tuple] = {}
    if not dest.is_dir():
        return fp
    for top in sorted(dest.iterdir()):
        if not top.is_symlink():
            continue
        target = top.resolve()
        if target.is_file():
            st = target.stat()
            fp[top.name] = (st.st_size, st.st_mtime_ns)
        elif target.is_dir():
            for d, _, files in os.walk(target):
                for f in sorted(files):
                    p = Path(d) / f
                    st = p.stat()
                    fp[str(Path(top.name) / p.relative_to(target))] = (st.st_size, st.st_mtime_ns)
    return fp


def set_derived(dest: Path) -> None:
    """Point this process and every step subprocess at the derived tree `dest`."""
    global DERIVED, SUBMISSIONS, LOG_ROOT
    os.environ["CUAS_DERIVED"] = str(dest)
    DERIVED, SUBMISSIONS, LOG_ROOT = dest, dest / "submissions", dest / "run_logs"


def promote(src_root: Path, dst_root: Path, name: str, methods: Optional[list[str]] = None) -> list[tuple]:
    """Copy src_root/<method>/ to dst_root/<method>_<name>/ (replacing an earlier promotion). Returns
    (method, promoted name, files) rows."""
    import shutil
    if not re.fullmatch(r"[A-Za-z0-9]+", name):
        raise ValueError(f"--promote: name must be letters / digits only, got {name!r}")
    have = sorted(p.name for p in src_root.iterdir() if p.is_dir() and not p.name.startswith("_")) \
        if src_root.is_dir() else []
    methods = have if not methods else methods
    rows = []
    for m in methods:
        if m not in have:
            raise ValueError(f"--promote: no method directory {m!r} in {src_root}")
        dst = dst_root / f"{m}_{name}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src_root / m, dst)
        rows.append((m, dst.name, len(list(dst.glob("*/*.xlsx")))))
    return rows


# --------------------------------------------------------------------------- plan printing


def fmt_est(s: Optional[float]) -> str:
    if s is None:
        return "unmeasured"
    return f"{s:.0f} s" if s < 90 else f"{s / 60:.1f} min"


def print_plan(steps: list[Step], sel: dict[str, str]) -> None:
    print(f"Pipeline plan ({len(steps)} steps, dependency order). Nothing is run by --plan.")
    print("Runtimes are the last recorded ones (build_log / step reports); 'unmeasured' = not recorded yet.\n")
    for n, s in enumerate(steps, 1):
        r = sel[s.name]
        tag = "RUN" if r == "run" else r
        print(f"{n:>2}. {s.name}  [{s.group}]  -> {tag}")
        print(f"      cmd : {s.command_text()}")
        print(f"      what: {s.desc}")
        if s.inputs:
            print(f"      in  : {', '.join(s.inputs)}")
        if s.outputs:
            print(f"      out : {', '.join(s.outputs)}")
        extra = []
        if s.name in KNOWN_NONZERO:
            k = KNOWN_NONZERO[s.name]
            extra.append(f"KNOWN-NONZERO: exit 1 allowed with failing checks {sorted(k['failed_checks'])} only")
        if s.optional:
            extra.append("optional: " + ("script present" if s.available() else "script ABSENT, skipped"))
        print(f"      time: {fmt_est(s.est_s)}" + ("   " + "; ".join(extra) if extra else ""))
    run = [s for s in steps if sel[s.name] == "run"]
    known = sum(s.est_s or 0 for s in run)
    unk = [s.name for s in run if s.est_s is None]
    print(f"\nselected: {len(run)} step(s); recorded runtime of the measured ones ~{known / 60:.1f} min"
          + (f"; unmeasured: {', '.join(unk)}" if unk else ""))
    print("Logs would go to data/derived/run_logs/<UTC timestamp>/<nn>_<step>.log, plus run_manifest.json.")


# --------------------------------------------------------------------------- main


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", action="store_true", help="print the ordered plan and run nothing")
    ap.add_argument("--from", dest="from_", metavar="STEP", help="run STEP and every step after it")
    ap.add_argument("--only", nargs="+", metavar="STEP", help="run just these steps (in plan order)")
    ap.add_argument("--skip", nargs="+", metavar="STEP", help="do not run these steps")
    ap.add_argument("--with-analysis", action="store_true", help="also run the analysis / diagnostic steps")
    ap.add_argument("--selftest", action="store_true", help="self-test of the scheduler on dummy commands")
    ap.add_argument("--log-root", type=Path, default=LOG_ROOT, help=argparse.SUPPRESS)
    ap.add_argument("--final-validate", action="store_true", help=argparse.SUPPRESS)   # internal: the last step
    ap.add_argument("--methods", nargs="+", help=argparse.SUPPRESS)
    ap.add_argument("--optional-methods", nargs="+", help=argparse.SUPPRESS)
    ap.add_argument("--since", type=float, help=argparse.SUPPRESS)
    ap.add_argument("--json", help=argparse.SUPPRESS)
    ap.add_argument("--derived", type=Path, metavar="DIR",
                    help="work on the VARIANT tree DIR (under data/, not data/derived itself): every step writes there; "
                         "outputs of the steps before the first executed one are symlinked from data/derived")
    ap.add_argument("--variant-config", type=Path, metavar="JSON",
                    help="copy this file to <DIR>/variant.json (read by tools that support corrections)")
    ap.add_argument("--promote", metavar="NAME",
                    help="with --derived: copy the variant's submissions to data/derived/submissions/<method>_<NAME> "
                         "(no step is run)")
    ap.add_argument("--promote-methods", nargs="+", metavar="METHOD", help="restrict --promote to these methods")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()
    if a.derived is not None:
        dest = a.derived.resolve()
        if dest == BASE_DERIVED.resolve() or not dest.is_relative_to((REPO / "data").resolve()):
            print(f"usage error: --derived must be a directory under data/ other than data/derived (got {dest})",
                  file=sys.stderr)
            return 2
        default_logs = a.log_root == LOG_ROOT
        set_derived(dest)
        if default_logs:
            a.log_root = LOG_ROOT
    if a.promote:
        if a.derived is None:
            print("usage error: --promote needs --derived", file=sys.stderr)
            return 2
        try:
            rows = promote(SUBMISSIONS, BASE_DERIVED / "submissions", a.promote, a.promote_methods)
        except ValueError as e:
            print(f"usage error: {e}", file=sys.stderr)
            return 2
        for m, name, n in rows:
            print(f"promoted {m:<20} -> data/derived/submissions/{name}  ({n} files)")
        return 0
    if a.final_validate:
        return final_validate(a.methods or list(DEPLOYABLE_METHODS), a.json, a.optional_methods, a.since)

    steps = make_steps()
    try:
        sel = select_steps(steps, from_=a.from_, only=a.only, skip=a.skip, with_analysis=a.with_analysis)
    except ValueError as e:
        print(f"usage error: {e}", file=sys.stderr)
        return 2
    # the final step validates only the deployable methods that this run produces or that exist on disk
    prod_of = {m: s.name for s in steps for m in s.produces}
    vi = next(i for i, s in enumerate(steps) if s.name == "validate_submissions")
    va = list(steps[vi].args)
    mi, mj = va.index("--methods") + 1, va.index("--since") if "--optional-methods" not in va else va.index("--optional-methods")
    keep = [m for m in va[mi:mj] if sel.get(prod_of.get(m, "")) == "run" or (SUBMISSIONS / m).is_dir()]
    steps[vi] = dataclasses.replace(steps[vi], args=tuple(va[:mi] + (keep or va[mi:mi + 1]) + va[mj:]))
    if a.plan:
        print_plan(steps, sel)
        return 0
    missing = [n for n, r in sel.items() if r == "MISSING-SCRIPT"]
    if missing:
        print(f"usage error: script missing for non-optional step(s): {', '.join(missing)}", file=sys.stderr)
        return 2
    chosen = [s for s in steps if sel[s.name] == "run"]
    if not chosen:
        print("nothing to run")
        return 0
    fp0: dict = {}
    if a.derived is not None:
        DERIVED.mkdir(parents=True, exist_ok=True)
        if a.variant_config is not None:
            (DERIVED / "variant.json").write_text(a.variant_config.read_text(encoding="utf-8"), encoding="utf-8")
        linked = seed_variant(BASE_DERIVED, DERIVED, steps, {s.name for s in chosen})
        risks = write_through_risks(chosen, DERIVED)
        if risks:
            print("usage error: refusing to run, it would overwrite the main tree:\n  " + "\n  ".join(risks),
                  file=sys.stderr)
            return 2
        fp0 = link_fingerprint(DERIVED)
        vj = DERIVED / "variant.json"
        print(f"VARIANT tree {DERIVED.relative_to(REPO)}: {len(linked)} new link(s) to data/derived"
              + (f"; variant.json = {vj.read_text(encoding='utf-8').strip()}" if vj.is_file() else "; no variant.json"))
    for w in missing_inputs(chosen):
        print(f"WARNING {w}")

    import score as SC          # code_hash only; imported late so --plan / --selftest stay light
    started = dt.datetime.now(dt.timezone.utc)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    log_dir = a.log_root / stamp
    h0 = SC.code_hash()
    print(f"run {stamp}  code hash {h0}  logs {log_dir.relative_to(REPO) if log_dir.is_relative_to(REPO) else log_dir}")
    print(f"{len(chosen)} step(s): {', '.join(s.name for s in chosen)}\n")
    t_start = time.time()
    results = run_plan(steps, chosen, log_dir, since=t_start, sub_root=SUBMISSIONS)
    ended = dt.datetime.now(dt.timezone.utc)
    if a.derived is not None:
        fp1 = link_fingerprint(DERIVED)
        changed = sorted(k for k in fp0 if fp1.get(k) != fp0[k])
        if changed:
            results.append(dict(name="variant_input_guard", plan_pos=0, command="(seeded inputs unchanged)",
                                status="failed", exit_code=None, seconds=0.0, failed_checks=None,
                                last_line=f"{len(changed)} seeded main-tree file(s) changed: {', '.join(changed[:5])}",
                                note="a step wrote through a symlink into data/derived", log=None))

    cache = {}
    vj = log_dir / "validation.json"
    if vj.is_file():
        try:
            for m in json.loads(vj.read_text()).values():
                for rec in m.values():
                    if rec.get("sha256") and rec.get("rows") is not None:
                        cache[rec["sha256"]] = rec["rows"]
        except (OSError, ValueError):
            pass
    ran = [s for s in chosen if next(r for r in results if r["name"] == s.name)["status"] in ("ok", "known_nonzero")]
    patterns = [p for s in ran for p in s.produces]
    if any(s.name == "validate_submissions" for s in ran):
        patterns += list(DEPLOYABLE_METHODS) + list(PROBE_METHODS)
    methods = resolve_methods(SUBMISSIONS, patterns)
    subs = submission_records(SUBMISSIONS, methods, t_start, make_row_counter(cache))
    man = build_manifest(started=started, ended=ended, argv=sys.argv[1:] if argv is None else argv,
                         selection=sel, results=results, code_hash_start=h0, code_hash_end=SC.code_hash(),
                         submissions=subs, log_dir=log_dir)
    (log_dir / "run_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")

    print("\n" + "=" * 100 + "\nSUMMARY\n" + "=" * 100)
    print(format_summary(results))
    for r in results:
        if r["status"] == "known_nonzero":
            print(f"\n!! KNOWN NONZERO  {r['name']}: exit {r['exit_code']}, failing checks {r['failed_checks']} (by design)\n"
                  f"   reason: {r['note']}")
    if man["code_changed_during_run"]:
        print(f"\n!! tools/*.py changed while the pipeline ran (code hash {h0} -> {man['code_hash_end']}).")
    n_files = sum(len(v) for v in subs.values())
    if man["stale_submissions"]:
        st_ = man["stale_submissions"]
        print(f"\n!! {len(st_)} submission file(s) were NOT rewritten by this run (producer step skipped or untouched; "
              f"may come from older code or inputs): {', '.join(st_[:5])}" + (" ..." if len(st_) > 5 else ""))
    print(f"\nmanifest: {(log_dir / 'run_manifest.json')}  ({n_files} submission files hashed)")
    bad = [r["name"] for r in results if r["status"] == "failed"]
    if bad:
        left = [r["name"] for r in results if r["status"] == "not_run"]
        print(f"\nPIPELINE FAILED at {bad[0]}" + (f"; not run: {', '.join(left)}" if left else ""))
        return 1
    notes = (["known-nonzero steps listed above"] if man["known_nonzero"] else []) + \
            ([f"{len(man['stale_submissions'])} stale submission file(s), see above"] if man["stale_submissions"] else [])
    print("\nPIPELINE OK" + (f" (with {'; '.join(notes)})" if notes else ""))
    return 0


# --------------------------------------------------------------------------- self-test


def selftest() -> int:
    fails: list[str] = []

    def check(num: str, name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] ({num}) {name}" + ("" if ok or not detail else f"\n         {detail}"))
        if not ok:
            fails.append(num)

    with tempfile.TemporaryDirectory(prefix="run_selftest_") as td:
        tmp = Path(td)
        trace = tmp / "trace.txt"

        def dummy(name, body="", rc=0, **kw) -> Step:
            code = (f"import sys\nopen({str(trace)!r},'a').write({name!r}+'\\n')\n{body}\nsys.exit({rc})")
            return Step(name, argv_override=(PY, "-c", code), **kw)

        def order() -> list[str]:
            return trace.read_text().split() if trace.exists() else []

        def reset() -> None:
            if trace.exists():
                trace.unlink()

        def plan() -> list[Step]:
            return [dummy("a"), dummy("b"), dummy("c"), Step("d", script="no_such_script_xyz.py", optional=True),
                    dummy("e"), dummy("f", analysis=True), dummy("g"), dummy("h")]

        P = plan()

        def sel_run(**kw) -> list[str]:
            return [n for n, r in select_steps(P, **kw).items() if r == "run"]

        # 1 selection semantics
        cases = {
            "default": (sel_run(), list("abcegh")),
            "with-analysis": (sel_run(with_analysis=True), list("abcefgh")),
            "from c": (sel_run(from_="c"), list("cegh")),
            "from c + analysis": (sel_run(from_="c", with_analysis=True), list("cefgh")),
            "only (reversed input)": (sel_run(only=["h", "a"]), list("ah")),
            "only an analysis step": (sel_run(only=["f"]), ["f"]),
            "skip": (sel_run(skip=["b", "g"]), list("aceh")),
            "from + skip": (sel_run(from_="c", skip=["e"]), list("cgh")),
            "only + skip": (sel_run(only=["a", "b", "c"], skip=["b"]), list("ac")),
        }
        bad = [k for k, (got, want) in cases.items() if got != want]
        check("1", "selection: default / --with-analysis / --from / --only / --skip combinations", not bad,
              "; ".join(f"{k}: got {cases[k][0]} want {cases[k][1]}" for k in bad))
        d_state = select_steps(P)["d"]
        errs = []
        for kw in (dict(only=["a"], from_="b"), dict(from_="zzz"), dict(only=["nope"]), dict(skip=["nope"]),
                   dict(from_="f")):
            try:
                select_steps(P, **kw)
                errs.append(f"no error for {kw}")
            except ValueError:
                pass
        check("2", "usage errors: unknown names, --only with --from, --from an analysis step without --with-analysis; "
                   "absent optional script is reported 'absent', not run", not errs and d_state.startswith("absent"),
              "; ".join(errs) + f" d={d_state}")

        # 3 order of execution + logs
        reset()
        chosen = [s for s in P if select_steps(P, only=["g", "a", "c"])[s.name] == "run"]
        res = run_plan(P, chosen, tmp / "logs1", echo=lambda _: None)
        logs = sorted(p.name for p in (tmp / "logs1").glob("*.log"))
        check("3", "steps execute in plan order whatever the order of --only, one log per step named <nn>_<step>.log",
              order() == ["a", "c", "g"] and logs == ["01_a.log", "03_c.log", "07_g.log"]
              and [r["status"] for r in res] == ["ok"] * 3, f"order {order()} logs {logs}")

        # 4 stdout + stderr in the log, exit code, last self-check line, seconds
        reset()
        noisy = dummy("noisy", "import sys as _s\nprint('to-stdout')\nprint('to-stderr', file=_s.stderr)\n"
                              "print('ALL SELF-CHECKS PASSED')\nprint('trailer')")
        r = run_plan([noisy], [noisy], tmp / "logs2", echo=lambda _: None)[0]
        txt = (tmp / "logs2" / "01_noisy.log").read_text()
        check("4", "log holds stdout and stderr; summary keeps the last self-check line, exit code, seconds",
              "to-stdout" in txt and "to-stderr" in txt and r["last_line"] == "ALL SELF-CHECKS PASSED"
              and r["exit_code"] == 0 and r["seconds"] is not None and r["seconds"] >= 0
              and "ALL SELF-CHECKS PASSED" in format_summary([r]), f"{r}")

        # 5 stop at the first unexpected failure
        reset()
        steps5 = [dummy("s1"), dummy("s2", rc=3), dummy("s3"), dummy("s4")]
        out5: list[str] = []
        res5 = run_plan(steps5, steps5, tmp / "logs3", echo=out5.append)
        check("5", "stop at the first unexpected failure: later steps are 'not_run' and never executed; "
                   "overall status failed; the failing log tail is echoed",
              order() == ["s1", "s2"] and [r["status"] for r in res5] == ["ok", "failed", "not_run", "not_run"]
              and overall_status(res5) == "failed" and any("UNEXPECTED FAILURE" in x for x in out5)
              and res5[2]["exit_code"] is None and not (tmp / "logs3" / "03_s3.log").exists(), f"{order()} {res5}")

        # 6 KNOWN_NONZERO
        kn = {"kn": dict(exit_codes=(1,), failed_checks=frozenset({"4", "5"}), reason="by design")}
        reset()
        steps6 = [dummy("kn", "print('[ 12.0s] SELF-CHECK FAILURE: 4, 5')", rc=1), dummy("after")]
        out6: list[str] = []
        res6 = run_plan(steps6, steps6, tmp / "logs4", known=kn, echo=out6.append)
        loud = any("BY DESIGN" in x for x in out6)
        check("6a", "known-nonzero: exit 1 with exactly the allowlisted failing checks is reported loudly and the run continues",
              [r["status"] for r in res6] == ["known_nonzero", "ok"] and order() == ["kn", "after"] and loud
              and res6[0]["failed_checks"] == ["4", "5"] and overall_status(res6) == "ok", f"{res6}")
        mut = {}
        for label, body, rc in (("extra failing check", "print('SELF-CHECK FAILURE: 4, 6')", 1),
                                ("exit code 2", "print('SELF-CHECK FAILURE: 4, 5')", 2),
                                ("no parsable check list", "print('boom')", 1),
                                ("other step name", "print('SELF-CHECK FAILURE: 4')", 1)):
            reset()
            name = "other" if label == "other step name" else "kn"
            sts = [dummy(name, body, rc=rc), dummy("after")]
            rr = run_plan(sts, sts, tmp / "logs5", known=kn, echo=lambda _: None)
            mut[label] = (rr[0]["status"], order())
        check("6b", "known-nonzero mutations are UNEXPECTED and stop the run (extra failing check, other exit code, "
                    "unparsable log, step not on the list)",
              all(v == ("failed", ["other" if k == "other step name" else "kn"]) for k, v in mut.items()), str(mut))
        reset()
        st_ok = dummy("kn", "print('ALL SELF-CHECKS PASSED')")
        r7 = run_plan([st_ok], [st_ok], tmp / "logs6", known=kn, echo=lambda _: None)[0]
        sub = dummy("kn", "print('SELF-CHECK FAILURE: 5')", rc=1)
        r8 = run_plan([sub], [sub], tmp / "logs7", known=kn, echo=lambda _: None)[0]
        check("6c", "a listed step that now exits 0 is 'ok' with a 'remove it' note; a strict subset of the failing "
                    "checks is still known-nonzero",
              r7["status"] == "ok" and "remove it" in r7["note"] and r8["status"] == "known_nonzero", f"{r7} {r8}")

        # 7 the real allowlist and plan table are consistent with the scripts on disk
        real = make_steps()
        names = [s.name for s in real]
        probs = []
        if len(set(names)) != len(names):
            probs.append("duplicate step names")
        probs += [f"KNOWN_NONZERO key {k} is not a step" for k in KNOWN_NONZERO if k not in names]
        for s in real:
            if not s.optional and not s.available():
                probs.append(f"script missing for {s.name}")
            if s.available() and s.script:
                src = (TOOLS / s.script).read_text(encoding="utf-8")
                for arg in s.args:
                    if arg.startswith("--") and f'"{arg}"' not in src:
                        probs.append(f"{s.name}: flag {arg} not found in {s.script}")
        by_name = {s.name: s for s in real}
        if (TOOLS / "probes.py").is_file():
            psrc = (TOOLS / "probes.py").read_text(encoding="utf-8")
            probs += [f"probe {m} not named in probes.py" for m in PROBE_METHODS if f'"{m}"' not in psrc]
            if tuple(by_name["probes"].produces) != PROBE_METHODS:
                probs.append("probes step does not declare PROBE_METHODS as produced")
            if "--optional-methods" not in by_name["validate_submissions"].args:
                probs.append("validate_submissions does not validate the probe methods")
        want_order = ["build_detections", "sensor_inventory", "gt_associate", "truth", "sensor_noise", "metadata_scan",
                      "sensor_tracks", "existence", "classify", "bearing_support", "fuse"]
        if names[:len(want_order)] != want_order or names[-1] != "validate_submissions":
            probs.append("plan order changed")
        i = lambda n: names.index(n)  # noqa: E731
        prod = {}
        for s in real:
            for o in s.outputs:
                prod.setdefault(o, i(s.name))
        for s in real:
            for p in s.inputs:
                src_i = [j for o, j in prod.items() if p == o or p.startswith(o.rstrip("/") + "/") or fnmatch.fnmatch(p, o)]
                if src_i and min(src_i) >= i(s.name):
                    probs.append(f"{s.name} reads {p} before its producer runs")
        check("7", "real plan: unique names, scripts exist, flags exist in the scripts' source, order matches the "
                   "dependency chain, every input is produced earlier (when produced at all)", not probs, "; ".join(probs))

        # 8 input warnings
        a1 = Step("x", argv_override=(PY, "-c", "pass"), inputs=("nonexistent/in",), outputs=("nonexistent/out",))
        a2 = Step("y", argv_override=(PY, "-c", "pass"), inputs=("nonexistent/out/file",))
        a3 = Step("z", argv_override=(PY, "-c", "pass"), inputs=("nonexistent/in",))
        w_ok = missing_inputs([a1, a2], root=tmp)
        w_bad = missing_inputs([a3], root=tmp)
        check("8", "input warnings: a missing input is flagged unless an earlier selected step writes it",
              len(w_ok) == 1 and "x: input nonexistent/in" in w_ok[0] and len(w_bad) == 1, f"{w_ok} {w_bad}")

        # 9 manifest
        root = tmp / "data" / "derived" / "submissions"
        old, new = root / "M_old" / "testing__S1-I1", root / "M_new" / "evaluation__DI__S1-I3"
        old.mkdir(parents=True)
        new.mkdir(parents=True)
        (old / "scenario_1_iteration_1.xlsx").write_bytes(b"old content")
        (new / "scenario_1_iteration_3.xlsx").write_bytes(b"new content")
        long_ago = time.time() - 10_000
        os.utime(old / "scenario_1_iteration_1.xlsx", (long_ago, long_ago))
        (root / "_selftest").mkdir()
        started = dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc)
        meths = resolve_methods(root, ["M_*"])
        recs = submission_records(root, meths, time.time() - 5, lambda p: 17 if "new" in p.read_bytes().decode() else None)
        man = build_manifest(started=started, ended=started, argv=["--only", "x"], selection={"a": "run"},
                             results=res5, code_hash_start="aaa", code_hash_end="bbb", submissions=recs,
                             log_dir=tmp / "logs3")
        man2 = json.loads(json.dumps(man))
        h_new = hashlib.sha256(b"new content").hexdigest()
        rn = man2["submissions"]["M_new"]["evaluation__DI__S1-I3"]
        ro = man2["submissions"]["M_old"]["testing__S1-I1"]
        check("9", "manifest: code hash (start/end/changed), per-step exit codes and timings, submission sha256 "
                   "(independent), row counts, fresh flag, stop point; survives a JSON round trip",
              meths == ["M_new", "M_old"] and man2["code_hash"] == "aaa" and man2["code_changed_during_run"] is True
              and man2["overall"] == "failed" and man2["stopped_at"] == "s2"
              and [s["exit_code"] for s in man2["steps"]] == [0, 3, None, None]
              and all(s["seconds"] is not None for s in man2["steps"][:2])
              and rn["sha256"] == h_new and rn["rows"] == 17 and rn["fresh"] is True and rn["bytes"] == 11
              and ro["fresh"] is False and ro["rows"] is None
              and man2["stale_submissions"] == [ro["file"]]
              and man2["started_utc"] == "2026-01-02T03:04:05Z", json.dumps(man2["submissions"])[:300])
        kman = build_manifest(started=started, ended=started, argv=[], selection={}, results=res6, code_hash_start="a",
                              code_hash_end="a", submissions={}, log_dir=tmp)
        check("10", "manifest lists known-nonzero steps with their reason and reports overall ok",
              kman["overall"] == "ok" and kman["known_nonzero"] == [dict(step="kn", exit_code=1, failed_checks=["4", "5"],
                                                                         reason="by design")]
              and kman["code_changed_during_run"] is False, str(kman["known_nonzero"]))

        # 11 log_facts parsing
        t1 = "x\n[ 612.3s] SELF-CHECK FAILURE: 4, 5\n"
        t2 = "...\nSELF-CHECKS FAILED: ['3']\n"
        t3 = "[check] runtime 3.0s: ALL OK\n"
        t4 = "nothing relevant\nlast line\n"
        f1, f2, f3, f4 = log_facts(t1), log_facts(t2), log_facts(t3), log_facts(t4)
        check("11", "log parsing: failing check numbers (not the timestamp) and the last self-check line, for the formats "
                    "of pair_features, bearing_support and sensor_inventory",
              f1[1] == {"4", "5"} and f2[1] == {"3"} and f3 == ("[check] runtime 3.0s: ALL OK", None)
              and f4 == ("last line", None), f"{f1} {f2} {f3} {f4}")

        # 12 --plan prints every real step and runs nothing
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--plan"])
        text = buf.getvalue()
        check("12", "--plan prints all steps in order with cmd / in / out / time and exits 0 without running anything",
              rc == 0 and all(f". {n}  [" in text for n in names)
              and text.index("build_detections") < text.index("fuse") < text.index("validate_submissions")
              and "off (analysis; --with-analysis)" in text, text[:200])

        # 13 a step that promises submissions but writes none fails loudly (silent no-op, stale leftovers)
        sroot = tmp / "subs"
        (sroot / "PX" / "testing__S1-I1").mkdir(parents=True)
        (sroot / "PY" / "testing__S1-I1").mkdir(parents=True)
        stale_f = sroot / "PY" / "testing__S1-I1" / "scenario_1_iteration_1.xlsx"
        stale_f.write_bytes(b"leftover")
        os.utime(stale_f, (time.time() - 10_000, time.time() - 10_000))
        writer = dummy("writer", f"open({str(sroot / 'PX' / 'testing__S1-I1' / 'scenario_1_iteration_1.xlsx')!r},'wb')"
                                 f".write(b'x')", produces=("PX",))
        noop_stale = dummy("noop_stale", produces=("PY",))        # exits 0; only an old leftover file exists
        noop_nodir = dummy("noop_nodir", produces=("PZ",))        # exits 0; no such method directory at all
        reset()
        seq = [writer, noop_stale, dummy("after13")]
        r13 = run_plan(seq, seq, tmp / "logs8", echo=lambda _: None, sub_root=sroot)
        order13 = order()
        r13b = run_plan([noop_nodir], [noop_nodir], tmp / "logs9", echo=lambda _: None, sub_root=sroot)
        check("13", "produces: a step that exits 0 but writes no fresh submission is FAILED and stops the run "
                    "(leftover old file, or no directory); a step that writes one passes",
              [r["status"] for r in r13] == ["ok", "failed", "not_run"] and order13 == ["writer", "noop_stale"]
              and "no fresh submission" in r13[1]["note"] and r13[1]["exit_code"] == 0
              and r13b[0]["status"] == "failed" and "PZ" in r13b[0]["note"]
              and unproduced(("PX", "PY"), sroot, time.time() + 60) == ["PX", "PY"],
              f"{[(r['name'], r['status'], r['note']) for r in r13]} {r13b}")

        # 15 variant trees: seeding, write-through guard, fingerprint, promotion
        base, dest = tmp / "data" / "derived_base", tmp / "data" / "derived_var"
        (base / "up_dir").mkdir(parents=True)
        (base / "up_file.csv").write_text("u")
        (base / "up_dir" / "x.npz").write_bytes(b"x")
        (base / "mid.csv").write_text("m")
        vs = [Step("up", argv_override=(PY, "-c", "pass"), outputs=(D + "up_file.csv", D + "up_dir", D + "absent.csv",
                                                                       D + "submissions/B*")),
              Step("mid", argv_override=(PY, "-c", "pass"), outputs=(D + "mid.csv",)),
              Step("down", argv_override=(PY, "-c", "pass"), outputs=(D + "down",))]
        dest.mkdir(parents=True)
        (dest / "mid.csv").write_text("variant's own")             # kept, never replaced by a link
        lk = seed_variant(base, dest, vs, {"mid", "down"})
        lk_again = seed_variant(base, dest, vs, {"mid", "down"})
        risk_ok = write_through_risks(vs[1:], dest)
        risk_bad = write_through_risks([Step("bad", argv_override=(PY, "-c", "pass"), outputs=(D + "up_dir/new.csv",))],
                                       dest)
        fpa = link_fingerprint(dest)
        (base / "up_dir" / "x.npz").write_bytes(b"changed!")
        fpb = link_fingerprint(dest)
        src_sub = dest / "submissions"
        (src_sub / "F0_fuse" / "testing__S1-I1").mkdir(parents=True)
        (src_sub / "F0_fuse" / "testing__S1-I1" / "scenario_1_iteration_1.xlsx").write_bytes(b"v")
        (src_sub / "_samplecheck").mkdir()
        dst_sub = tmp / "main_subs"
        pr = promote(src_sub, dst_sub, "v3h")
        pr_again = promote(src_sub, dst_sub, "v3h")
        try:
            promote(src_sub, dst_sub, "bad name")
            bad_name = False
        except ValueError:
            bad_name = True
        check("15", "variant trees: upstream outputs are symlinked once (globs / absent / existing entries skipped), a "
                    "step writing under a link is refused, a changed seeded file is detected, --promote copies "
                    "<method> to <method>_<NAME> (idempotent, private dirs skipped, names checked)",
              lk == ["up_file.csv", "up_dir"] and lk_again == [] and (dest / "up_dir").is_symlink()
              and (dest / "mid.csv").read_text() == "variant's own" and not risk_ok and len(risk_bad) == 1
              and set(fpa) == {"up_dir/x.npz", "up_file.csv"} and fpa != fpb
              and pr == [("F0_fuse", "F0_fuse_v3h", 1)] and pr_again == pr and bad_name
              and (dst_sub / "F0_fuse_v3h" / "testing__S1-I1" / "scenario_1_iteration_1.xlsx").read_bytes() == b"v",
              f"lk={lk} again={lk_again} risks={risk_ok}/{risk_bad} fp={fpa}/{fpb} pr={pr}")

        # 14 freshness reporting of the final validation
        check("14", "final validation flags stale files without failing: tag text and manifest list",
              freshness_tag(True, True) == "(valid)" and freshness_tag(True, None) == "(valid)"
              and "STALE" in freshness_tag(True, False) and freshness_tag(False, True) == "(NOT VALID)"
              and freshness_tag(False, False) == "(NOT VALID)"
              and stale_files({"M": {"a": dict(file="f1", fresh=True), "b": dict(file="f2", fresh=False)}}) == ["f2"],
              "")

    if fails:
        print(f"SELF-CHECK FAILED: {', '.join(fails)}")
        return 1
    print("ALL SELF-CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
