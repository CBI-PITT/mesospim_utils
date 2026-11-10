# BigStitcher Optimization — What Was Done

This document explains the work on the `bigstitcher-optimization` branch: why it
was built, what changed, what the validation showed, and what to know when using
or extending it. For session-to-session handoff notes see `PROJECT_CONTEXT.md`.

## Motivation

The work started from an analysis of 19 production MesoSPIM datasets that found
three systematic problems with the existing alignment pipeline:

1. **Zero-detection views** — 57% of surveyed datasets had at least one view
   with near-zero dynamic range (illumination dropout on the scope). Views that
   never fired cannot contribute overlaps, so STAGE 2 (ICP) on them is
   untrustworthy.
2. **Systematic zoom/ETL miscalibration** — the tile grid spacing recorded in
   the acquisition metadata disagrees with the spacing the images actually show
   by 2-7% (measured per zoom/ETL-CFG entry). Every fused volume was produced at
   the *nominal* pixel size, so physical measurements from it were
   systematically mis-scaled.
3. **Unmeasurable alignment quality** — whether an alignment was good was
   judged by eye. There was no quantitative, comparable metric.

## What was built

### Fiji installation and distribution choice

- Rewrote the Fiji auto-install so a fresh machine gets a working Fiji +
  BigStitcher.
- Made the Fiji distribution config-selectable: `stable` (BigStitcher 2.5.3,
  reads OME-Zarr 0.4 / zarr v2) vs `latest` (BigStitcher 3.x + the new N5
  stack, adds OME-Zarr 0.5 sharded reading). Every macro key the templates send
  was verified against both jars.

### Compatibility and the align/fuse split

- Pinned multiview-reconstruction 9.0.12 — required for BigStitcher 3.0.8 to
  run under the latest Fiji build.
- Split the macro into configurable stages (`--stages align`, `fuse`,
  `align+fuse`). Align-only runs produce registrations without fusing, which
  makes tuning cheap and risk-free.
- Moved the STAGE 1 link thresholds, ICP parameters, and detection settings
  into the config. Alignment parameters went from hardcoded macro strings to
  machine-readable knobs — the prerequisite for everything after.

### Phase 1 — the alignment scorer (`bigstitcher_scoring.py`)

Parses an align log (pairwise r-values, optimization rounds, residuals, dropped
links) plus the stitched XML (fitted grid scale offset, residual scatter) and
produces a composite 0-100 score with actionable recommendations. Every
recommendation is tied to a measured quantity (dropped links, zero-dynamic-range
views, miscalibration, phase-correlation precision limits, starvation).

### Phase 2 — adaptive knobs and the self-calibration guard

- Adaptive overlap-driven downsampling: targets a pixel count per overlap strip
  instead of fixed downsampling factors (`downsample_in_x/y/z: 'overlap'` with
  `min_overlap_strip_px`).
- DoG detection intensity gating modes (`off` / `common` / `fixed`).
- The **self-calibration guard**: measures the fitted grid scale offset from the
  stitched XML and, when it exceeds `self_calibration_threshold_pct` (2%),
  rescales the fused output's voxel size by the fitted scale — correcting the
  systematic mis-scaling. Beyond 10% it warns that the registration itself is
  likely unreliable and the zoom/ETL lookup should be recalibrated instead.
  The correction is config-gated (`self_calibration_correction`) and is
  **not idempotent**: each fusion produces a fresh nominal-metadata volume so
  the chained guard corrects exactly once per fusion; manually re-running the
  guard CLI on an already-corrected volume double-corrects.

### Phase 3 — the auto-tune search driver

- `big-stitcher-search <dir>`: runs a grid of link-threshold candidates
  headlessly (align-only runs, chained as SLURM dependencies because parallel
  runs would clobber the shared stitched XML), snapshots each candidate's
  registrations, then `big-stitcher-search-score <dir>` scores every candidate
  with the Phase 1 scorer, restores the winner's registrations, and submits
  exactly one fusion. Budget: one align-only run per candidate plus one fusion.
- `--dry-run` prints the whole chain without submitting.

## Validation results

### Config tuning (delima search, then multi-dataset)

The first auto-tune search on delima found the most relaxed pair
(link_relative 7.0 / link_absolute 5.0) best: 68.0/100 vs 54.0 for the previous
defaults, with zero dropped links. `link_absolute` is the dominant axis.
The tuned values plus `self_calibration_correction: true` are now the config
defaults.

Cross-dataset validation with the tuned config (scores from the Phase 1
scorer; align-only runs):

| dataset | tiles | score | dropped | grid offset | flags |
|---|---|---|---|---|---|
| delima | 18 | 60.2 (control 54.0) | 6 → 3 | +5.58% | — |
| dutta | 12 | 82.9 (baseline 76.1) | 5 → 2 | +6.69% | — |
| 102226_2 | 4 | 56.9 (= warm baseline) | 1 | +7.16% | — |
| green_embryo_3x3 | 27 | 58.2 | 7 | +2.66% | 1 zero-DR view |
| almarza_knee | 28 | 34.4 | 39 | +5.95% | 5 zero-DR views |
| zhang_070726 | 36 | 49.7 | 11 | +2.18% | 6 zero-DR views |
| klimstra | 40 | 44.5 | 21 | +5.45% | — |
| alder_mcdyer_lung | 48 | 51.2 | 11 | +5.45% | — |
| exosomes_2 | 2 | n/a (too few pairs) | 0 | — | seam check passed |

Reading of the results:

- The tuned thresholds clearly improve the datasets they were derived from
  (delima, dutta) and are neutral on well-stitched ones (102226_2).
- Four datasets still drop links at 7.0/5.0. These have colder, more wildly
  miscalibrated grids (first-round residuals 83-686px) and several have
  zero-detection views. They are the per-dataset-search candidates (relax
  beyond 7.0/5.0) plus scope-side attention: illumination for the
  zero-detection views, zoom/ETL recalibration for the offset.
- The zoom/ETL offset clusters by instrument configuration, not by sample:
  Mag4x sessions measure +5.45-5.95%, one Mag8x session +2.18-2.66%, another
  Mag8x session +7.16%. The offset is a property of the scope's calibration;
  a well-calibrated scope measures ~0% and the guard does nothing.
- Visual ground truth: the fused 2-tile exosomes volume was MAX-projected and
  inspected — continuous field, no brightness step at the tile boundary, no
  ghosting. The quantitative seam check found no step anywhere.

### Pipeline bugs found and fixed during validation

1. **Fuse false-success on a stale container** — BigStitcher refuses to create
   an existing zarr container, but the success check (`&& [ -e ... ]`) passed on
   the stale container from a previous run, silently reporting success without
   writing anything. Fixed with `-nt` (the container must be newer than the
   macro).
2. **Zarr v3 metadata crash** — the chained metadata-fix step read zarr v2
   `.zattrs`, but multiview-reconstruction 9.0.12 writes zarr v3 `zarr.json`
   (multiscales nested under `attributes.ome`). Fixed to support v3; the fused
   metadata turns out to be already correct for v3 (the v3 writer transfers the
   voxel size), so the step is a no-op rather than a correction for v3 output.
3. **Wrong chained command name** — the chained guard step called
   `apply-self-calibration-guard`, but the registered typer command is
   `apply-self-calibration-guard-cli`.
4. **Missing import** — `SLURM_PARAMETERS_FOR_DEPENDENCIES` was not imported in
   the search driver's scope.

## Known caveats

- **The recommender quirk**: the scorer's recommendation engine suggests
  "try the RELAXED preset (7.0/5.0)" even when that preset is already the
  active config. Read it as "relax further or fix the calibration". A small
  fix (pass the current thresholds into the recommendation) is a worthwhile
  follow-up.
- **Cross-lab generality**: the measurement and the guard are generic by
  construction (computed from any BigStitcher XML; no instrument-specific
  assumptions; no hardcoded machine paths; every knob is config with
  conservative defaults). The config *values* are tuned on this lab's scopes —
  a laboratory optimizing for its own samples should run the search once on a
  representative dataset.
- **Store permissions gotcha**: datasets owned by other users have read-only
  `logs_mesospim_utils/` directories and beegfs refuses hardlinks to their
  files, so an align run fails in ~1s on the sbatch log redirect. Workarounds:
  rename the old logs dir aside (requires a lab-writable parent) or run on a
  wrapper directory with real-copied XMLs and a symlinked zarr container.
- **The correction is not idempotent** (see above) — relevant only for manual
  guard re-runs, not for the chained flow.
