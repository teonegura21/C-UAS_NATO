# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Working repo for the NATO NCIA C-UAS Data Challenge. The task is to fuse SAPIENT detection reports from radar, passive RF, EO/IR, acoustic, cyber and UTM sensors into one recognised air picture. That means deduplicated tracks of Class I UAS, each classified, identified as friendly or hostile, and tagged with class metadata. The results are written into the per-iteration `.xlsx` templates. Scoring (`README/01_Competition_Instructions.pdf` §2.5): track accuracy by GOSPA ×0.4, classification recall ×0.2, identification recall ×0.1, fusion completeness (class-metadata correctness) ×0.3.

A fusion/tracking solution is being built on the plan in the repo-root roadmap ("Roadmap C-UAS de la date curate la scor competitiv.md", Romanian): measurement first, then tracking and fusion, then ML. `tools/` holds the data-preparation, analysis and solution pipeline below. There is no build, lint, test suite or CI: each script prints its own self-checks. Python 3 with `numpy` and `openpyxl` only (no `pandas`/`scipy`/`sklearn`). The datasets may only be used for this competition (rules §2.3), so do not send them to external services.

## Commands

Analysis pipeline. Each step reads the previous steps' outputs in `data/derived/` (gitignored); all are deterministic and take under a minute:

```bash
python3 tools/build_detections.py    # raw CSVs -> detections.csv.gz (one row per report, normalized units, local ENU), groundtruth.csv.gz, iteration_refs.csv
python3 tools/sensor_inventory.py    # per-file inventory, co-located sensor sites, fusion links
python3 tools/field_intersection.py  # README_Sensor.pdf vs data field matrices, field-set intersections, sensor-family K-means
python3 tools/gt_associate.py        # testing only: clock/azimuth calibration, detection<->Dronetag matching, co-detection sets
python3 tools/coobservation.py       # GT-free cross-sensor co-observation on both trees (needs gt_associate outputs)
python3 tools/cluster_tracks.py      # per-track features + K-means variants (unsupervised)
python3 tools/preclass_report.py     # synthesis -> data/derived/PRECLASS_REPORT.md
python3 tools/eval_tiers.py          # EVALUATION ONLY: field ladder (README fields by share of sensors) + K-means per tier + same-time intersections -> data/derived/eval/
python3 tools/eval_charts.py         # charts + glossary for the above -> data/derived/eval/EVAL_CLUSTERS.html (self-contained, open locally)
```

The owner wants analysis results on the **evaluation** tree only; testing is for labelled validation, never pooled into evaluation results.

Solution pipeline (measurement layer, built 2026-10-28; each script runs its own self-checks):

```bash
python3 tools/gospa.py               # self-test of the GOSPA metric + rectangular assignment solver (library, no data)
python3 tools/truth.py               # ground truth on the template's 100 ms grid (testing) -> data/derived/truth_summary.md
python3 tools/submission.py --selftest   # .xlsx writer / reader / validator on the official templates
python3 tools/score.py --method NAME [--truth all|airborne] [--log --note TEXT]   # local proxy score; --selftest
python3 tools/sensor_noise.py        # per-sensor error model (testing) + u_ellipsoid height rule
python3 tools/metadata_scan.py       # serial / model / protocol / band values -> dropdown mapping (tools/metadata_map.py)
python3 tools/sensor_tracks.py       # per-sensor track segments, Kalman/RTS on the grid, features, testing labels -> data/derived/sensor_tracks_v2/ (canonical; sensor_tracks/ is a stale v1 with non-ellipsoidal heights)
python3 tools/gbdt.py                # self-test of the numpy gradient-boosting classifier
python3 tools/track_rows.py          # self-test of make_rows: the one grid-tracks -> submission-rows builder
python3 tools/baselines.py           # B0/B1/B2 baselines + O1/O2 oracles on testing -> baselines.md, experiments.csv (~10 min)
python3 tools/existence.py           # per-segment P(drone), nested LOIO -> data/derived/existence/ (p.csv; p_inner.csv for tuning) (~3 min)
python3 tools/pair_features.py       # overlap / continuation segment pairs + hand rules (the learned p_same is not a usable merge rule) (~11 min)
python3 tools/ground_coverage.py     # parked-drone coverage; the hold-last-position rule loses held-out, so it is not built
python3 tools/classify.py            # per-segment P(fixed wing), nested LOIO -> data/derived/classify/ (~15 s)
python3 tools/bearing_support.py     # camera / acoustic / RF bearing hits per segment + GT-free azimuth registration (check 3 fails by design: conservative registration, disputed GT offsets) (~1.5 min)
python3 tools/fuse.py                # fusion tracker (constrained cross-sensor merge, fused state, gap bridging) + method F0_fuse, LOIO-tuned -> data/derived/fused/, submissions (~9 min)
python3 tools/track_existence.py     # track-level existence model, method F1_fuse (SHELVED: lost to F0, 1 win / 1 tie / 4 losses) (~6 min)
python3 tools/probes.py              # one-factor leaderboard probe variants of F0 (P0_f0, P_park, P_meta, P_ident) + predicted deltas -> data/derived/probes/ (~6 min)
python3 tools/gospa_sensitivity.py   # scores and the F0 threshold decision on a grid of GOSPA c / p / 2-D-3-D, bias oracles (~5 min)
python3 tools/compare.py A B --truth both   # per-iteration method comparison: "A beats B on k of 6" + sign test (the standard for every comparison)
python3 tools/radar_registration.py  # GT-free radar offsets vs cyber / UTM tracks; finding: ENU offsets do not transfer, the AZIMUTH rotation does (~1 min)
python3 tools/triangulation_oracle.py   # unseen airborne truth reachable by >= 2 direction finders: 0 % (all bearing sensors at one site), shelved
python3 tools/deanonymize.py --selftest # owner's upload step: anonymized -> real longitudes with the owner's key at runtime (CUAS_LON_SHIFT_DEG, never stored)
python3 tools/run.py [--plan|--from STEP|--only ...|--with-analysis|--selftest]   # ONE command: the whole chain in dependency order, logs + run_manifest.json
python3 tools/run.py --derived data/derived_<name> --variant-config cfg.json --from sensor_tracks --skip track_existence probes
                                     # VARIANT tree: every tool writes there (CUAS_DERIVED); upstream outputs are symlinked from data/derived
python3 tools/run.py --derived data/derived_<name> --promote <name>   # copy the variant's submissions to data/derived/submissions/<method>_<name>
```

Rules for every method and score:

- Validation is leave-one-iteration-out over the 6 testing iterations. Evaluation is never used to fit, tune or calibrate anything.
- Headline local scores use only `C.deployable_sensor_ids()`, the sensors present in evaluation. EUIG, LJJX, MMJJ and NJQL are testing-only, and LJJX relays the Blue drones' own positions.
- The template grid equals the Dronetag span exactly, so the official truth very likely includes drones parked on the ground (about half the truth rows). Score both truth modes (`all`, `airborne`).
- The scorer's GOSPA parameters are guesses (PRIMARY c = 50 m, p = 2, 3D, plus a grid of alternatives): the rules do not publish them.
- Submissions are written in the anonymized longitude frame. De-anonymising before upload is the owner's step.
- Coverage drives most of the score: `score.py`'s classification, identity and metadata parts are recalls over all eligible truth rows, so a missed drone loses them too, not only the GOSPA part.
- Thresholds tuned for held-out testing iteration k must be chosen on the other iterations' nested predictions (`existence/<slug>__p_inner.csv`, `outer_fold == k`), not on their `p.csv` outer predictions, which came from models that saw k.
- Experiments with corrections run in a VARIANT tree (`run.py --derived`), never in `data/derived/` itself, so F0 and the baselines stay reproducible; compare with `compare.py` after `--promote`.
- Acceptance rules (owner, 2026-10-29): compare per iteration ("wins k of 6"), never by means alone. A fix of an evaluation-only defect (duplicates, Friendly copies, impossible jumps) is accepted if it does not lose on testing AND improves the evaluation health indicators, which are checks, not optimisation targets. Each decision has ONE pre-registered primary condition (e.g. c = 30 m, truth 'all', with UTM); the other truth modes / c values / without-UTM are non-regression checks with a fixed tolerance. Honesty fixes (removing in-sample calibration) are accepted unconditionally and their drop is reported.
- Calibration facts (verified 2026-10-28): radar ENU offsets flip sign across iterations and do not transfer; the radar AZIMUTH bias is stable (EGOI -1.04, NYTF -1.6, UXYJ -0.5 deg to ADD to the measured azimuth; IUKL unstable) and leave-one-out rotation about the radar site cuts EGOI / NYTF median horizontal error 27 -> 8 m / 25 -> 10 m. The height offsets (UXYJ +23.9, IUKL +40.4, ILYS +41.6 m) and the sigma tables are fitted on all testing iterations, and `sensor_noise.load_rules` ignores `u_applies_to`, so they reach evaluation too. Only the UXYJ clock offset (-3.5 s, stable) is applied. STUG / ILYS / CEQX report 0.5-1.5 s late on moving drones: correct that before using them as a reference.
- `tools/drone_catalog.csv` (controller-owned) maps Dronetag drone names (and the side, where it matters) to the dropdown labels. Its class and metadata columns are evidence-based (`truth_summary.md`, `metadata_report.md`); 'Heidrun' is fixed-wing by its kinematics.

`tools/_analyze.py` is an older standalone column-stats script: `python3 tools/_analyze.py data/Anon_Evaluation -o <report.txt>` (always pass the root; its default and docstring predate the current layout).

## Architecture of tools/

- `cuas_common.py` is the shared contract: iteration discovery (`Iteration.key` such as `testing/S1-I1` or `evaluation/DII/S2-I1`), file-name parsing, the one raw-row reader (`iter_rows`/`read_table`: always use it, never a bare `csv.reader`), parsers, WGS84 geometry and the per-iteration local ENU frame (`iteration_frame`, h0 = 0), and derived-table IO. `sensor_files()` skips two byte-identical `name (1).csv` duplicates.
- `cuas_clean.py` is the single place for data-quality rules on top of the faithful `detections.csv.gz`: self-reports, exact duplicate rows (12 % of all rows: two teams' copies of one stream, repeated rows), placeholder zero velocities, `alt == 0` sentinels, the cross-iteration height feature `dz_sensor`, ground-truth relay feeds, far targets. Sensor class labels are evidence, not cleaning rules: on testing, 60 % of UXYJ 'Bird' rows and 47 % of STUG 'Home' rows sit on real drones. Read its docstring before using detections.
- `kmeans.py` is a deterministic numpy K-means with silhouette-based K selection.

## Layout

- `README/`: the four spec PDFs (rules, sensor formats, ground truth, submission template). `docs/` holds byte-identical copies.
- `data/` (gitignored, ~650 MB) has two trees. Both use `Scenario N/Iteration M/{SensorData,FriendDronesList,ResultTemplate}/`, but `FriendDronesList/` holds different things in each:
  - `data/Anon_Testing/`: S1-I1, S2-I2, S3-I1/I2/I3, S4-I4. These **have ground truth**. `FriendDronesList/` holds one Dronetag CSV per flight, named `DroneTag_<BLUE|RED>_<YYMMDD>_<HHMM>_<HHMM>_<DroneName>.csv`. BLUE means friendly and RED means hostile. Dronetag `geo_altitude` is height above the WGS84 ellipsoid, which is what the submission's `altitude_m` uses; `altitude` is barometric. Use this tree for training and validation.
  - `data/Anon_Evaluation/Dataset {I,II,III}/`: S1-I3, S2-I1, S2-I3. These have **no ground truth**. `FriendDronesList/Friendly(Blue)UAS.xlsx` is only the Blue whitelist (manufacturer and serial number). Results for these iterations are what gets submitted. The Phase 2 set (S1-I2) is not here yet.
  - Each tree has an `ANONYMIZATION_MANIFEST.json` listing per-file row counts, sha256 and which columns were shifted.

## Anonymized longitudes: read before doing any geometry

Every longitude in `data/` has been shifted by a secret constant. That covers `sensor_location_x`, `location_x`, Dronetag `longitude`, and longitudes embedded in CEQX string fields. Latitudes, altitudes and timestamps are untouched. The `.xlsx` files were copied unchanged.

A constant longitude shift is a rotation about Earth's axis. Anything computed in a local frame (ENU/NEU offsets, ranges, azimuths, velocities, covariances) is therefore exactly preserved; only the absolute longitude value is fake. Never hardcode real-world coordinates, and don't try to infer or undo the shift. The anonymizer and its key are not in this repo.

## Sensor CSV gotchas

File names follow `<Team>_<Id>_<Type>_<YYMMDD>_<HHMM>_<HHMM>.csv`. The Team (Alpha…Echo) is the operator, not the sensor. The same `<Id>` appears under several teams, and one sensor can have several files per iteration. `<Type>` may contain `-` (e.g. `PASSIVE-RF--JAMMER--CYBER`) but never `_`. The parser is `sensor_files` in `tools/cuas_common.py`.

- The first column is an unnamed pandas index. Timestamps are naive ISO 8601 with microseconds, in UTC with no `Z`.
- Nested fields (`classification`, `trackInfo`, `objectInfo`, `signal`, `behaviour`, `associatedDetection`, `derivedDetection`, `associatedFile`) are Python-literal strings in single quotes. Parse them with `ast.literal_eval`; `json.loads` fails on them.
- Columns vary by sensor Id; see `README/README_Sensor.pdf` §1.1.2, which is wrong in several places. What the data actually holds:
  - 14 sensors report absolute `location_x/y/z` (x is longitude, y latitude): CEQX EGOI EUIG IKJX ILYS IUKL LJJX MMJJ NYTF QTVS SMFS (no z) STUG UXYJ VKYY. EGOI uses `LAT_LNG_RAD_M`, i.e. **radians**; honour `*_coordinateSystem` per row.
  - 8 are bearing-only: azimuth plus elevation for WVFX EMXG FGIX NJQL, azimuth alone for ORFH YXAB OEPS GTYT. Angles are **degrees** (`RANGE_BEARING_COORDINATE_SYSTEM_DEGREES_M`), clockwise from north, not the radians the PDF says. No sensor has a real range: FGIX's is always 200 and NJQL's always 1.
  - QTVS and VKYY (UTM services) report sensor_location 0,0,0, and with LJJX they relay the Dronetag feed itself, so scoring against ground truth on them is leakage.
- `location_xError` and `location_yError` are in metres, except for CEQX, where they are degrees.
- One evaluation CEQX file wraps each line whole in quotes. `cuas_common` unwraps it; a bare `csv.reader` sees 1-field rows.
- Measured sensor quirks (testing, against Dronetag):
  - Clock lag: LJJX about −2.25 s, UXYJ about −3.5 s.
  - Radar azimuth bias of 1–2°.
  - EMXG azimuth flipped by about 165–172° in some iterations; FGIX about 80° off.
  - YXAB and ORFH bearings don't track drones.
  - Dronetag files include long ground periods (only 19 of 38 records are airborne for most of their duration; five flights never take off), so drone labels must be time-resolved. "Skyranger" hovers in 2 of its 3 flights, so it is a multirotor; its third Dronetag sits on the ground.
  - Position errors are slow biases, not white noise: the feeds are already tracker outputs, and lag-1 error autocorrelation is 0.83–0.97. Radars are biased 10–25 m and unstable across iterations; ILYS and STUG are 3–10 m.
  - NYTF reports height above ground in some iterations (sensor_alt about 2 m). `sensor_noise.u_ellipsoid` converts reported heights to ellipsoidal ones.
  - Serials sometimes carry vendor prefixes or suffixes, e.g. STUG `1581F1ZNBJAB00C00MQ0`. `metadata_map.resolve_serial` maps them to the dropdown serials, and `metadata_map.SERIAL_MODEL` maps serial to model.
- Some names differ between the spec and the files. The spec's `WCFX` is `WVFX` in the files, and `EMXGA` is `EMXG`. `LJJX_FUSION-NODE` appears only in testing data and isn't documented. It has an extra `colour` column, which is always `Blue`.
- Classification strings follow `README_Sensor.pdf` §1.1.3, but their case doesn't always match it. For example, the data has `'Air vehicle'` where the spec has `'Air Vehicle'`.

## Submission template

Each iteration has `ResultTemplate/scenario_<N>_iteration_<M>.xlsx`, a single sheet with 19 columns. Keep the file name when submitting.

- `timestamp` and `unix_timestamp` are pre-filled every 100 ms from the first to the last ground-truth time. Never modify them.
- Write one row per tracked target per timestamp. K targets means K rows sharing a timestamp. Leave timestamps with no track empty.
- Position is WGS84 latitude, longitude and ellipsoid height.
- Covariance is the upper triangle of a local North-East-Up (NEU) matrix in m², in `C_nn`, `C_ne`, `C_nu`, `C_ee`, `C_eu` and `C_uu`.
- `track_id` is a UUID that stays the same for a target across the whole iteration.
- Categorical columns must use the exact dropdown strings from `README_SubmissionTemplate.pdf` Tables C.2–C.4. That includes the distinction between `Unknown` (cannot determine) and `Not Applicable` (no data).
- `unix_timestamp` is actually in **seconds** (a float), not the milliseconds the PDF says.
- The PDF's example rows have `node_id` and `object-id` columns that the template lacks. Follow the template.
