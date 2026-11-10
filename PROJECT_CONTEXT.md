## Purpose Of This File

- This file is for session handoff only.
- Keep stable repository guidance in `AGENTS.md`.
- Use this document for recent changes, active debugging context, validation gaps, and open questions that matter to the next agent.

## Current Branch And Working State

- Expected branch when this note was updated: `bigstitcher-optimization` (branched off `dev` at `966259d`; Phase 3 committed as `206e778`+`5c56a52`, pushed at `8266dd9`)
- Uncommitted from the delima search verification round: `mesospim_utils/automated.py` (SLURM_PARAMETERS_FOR_DEPENDENCIES import fix in big_stitcher_search; `-nt` fresh-container success check replacing `-e` in both fuse branches; `apply-self-calibration-guard-cli` command name fix in the chained fix step) and `mesospim_utils/bigstitcher.py` (zarr v3 support in adjust_scale_in_bigstitcher_produced_ome_zarr - reads root zarr.json attributes.ome.multiscales when .zattrs is absent).
- Recent work has touched BigStitcher, metadata/config, Fiji install helpers, and packaging metadata.

## Current Focus Areas

- BigStitcher alignment and fusion behavior
- BigStitcher parameter automation roadmap (planned; Phase 0 spec ready)
- Chromatic aberration and per-channel alignment tuning
- Memory pressure during BigStitcher fusion/export on SLURM
- Objective-aware deconvolution configuration and CLI plumbing
- BaSiCPy pooled-fit refactor (planned, not yet implemented)
- Keeping agent documentation split cleanly between `AGENTS.md` and this file

## Planned Follow-Up: BaSiCPy Pooled Fit Refactor (discussed, not yet implemented)

- Current behavior in `mesospim_utils/basicpy_worker.py` + `mesospim_utils/preprocess.py::process_basicpy_group`: `basic.fit()` re-runs on the same fit tile once per target tile (~N fits per channel/filter group), fields are estimated from a single representative tile, and `transform()` applies one 2D flatfield + one 2D darkfield to every z-plane.
- Verified in the installed BaSiCPy 2.0.0 source: fitting runs internally at `working_size=128` and upsamples fields to full res with bilinear `align_corners=True`; `transform()` broadcasts a single field pair across all z-planes, so the correction has **no z-dependence** at all (per-z intensity differences are absorbed into fit-time baseline terms and dropped). Light-sheet attenuation along z is currently uncaptured.
- Stage 1 (pooled single-field fit, one fit per group): add a `basicpy-fit-fields` command that lazily pools z-planes across ALL tiles in the (channel, filter) group (per-plane zarr reads resized to working size immediately, optional z subsampling to bound image count), fits once, and saves fields as a small `.npz` at working size in a hidden dir inside the output collection (same pattern as `.basicpy_tmp/`), e.g. `.basicpy_fields/<channel>_<filter>.npz`. The existing per-target-tile worker invocation would accept `--fields <npz>`, skip the fit, and transform as today, keeping the per-tile queue shape, tile-level resume, and `.basicpy_tmp` flow unchanged.
- Stage 2 (z-mapped fields pooled across all tiles): bin the z-depth (e.g. 10-20 bins), fit one field pair per bin from all tiles' planes in that bin, store fields at working size and upsample at transform time (bilinear `align_corners=True` to match BaSiCPy), apply by indexing an `(n_bins, H, W)` field stack by plane depth, and smooth across bins with a moving average to stabilize the depth profile. Stage 2 slots into Stage 1's fit/apply structure.
- Main new Stage 2 failure mode to design for: empty/low-signal bins. The BigStitcher analysis showed many near-zero-signal planes; bins above/below the sample will fit garbage. Skip low-signal planes per bin and fall back to the nearest good bin's fields, or weight planes by signal.
- Channel scope decision: flatfield stays per (channel, filter) group since vignetting differs per excitation/filter. Optional cross-channel darkfield sharing (average per-channel darkfields and reassign before transform) is a refinement only — darkfield interacts with per-plane exposure/signal level and may bias dim channels, so default off and validate before enabling.
- User is not concerned about illumination drift over acquisition time, so pooling tiles acquired at different times within a scan is acceptable.
- Config knobs for this change: `basicpy.fit_z_subsample`, optional `working_size` passthrough, optional `share_darkfield` in `config/example.yaml` (+ `main.yaml`).
- Files that would change: `mesospim_utils/basicpy_worker.py` (new fit-fields command, `--fields` on apply path, binning later), `mesospim_utils/preprocess.py` (fit job + per-tile apply orchestration), `mesospim_utils/automated.py` (CLI plumbing if options are surfaced), `mesospim_utils/config/example.yaml` + `main.yaml`, `mesospim_utils/constants.py` if needed.
- Validation plan: run one channel of a real dataset old-vs-new and compare flatfield/darkfield and corrected tiles; for Stage 2 specifically compare shallow vs deep planes; time and peak-RAM the fit job on the largest recent grid.
- Full plan text is also saved at `/h20/home/lab/.opencode/plan/basicpy-pooled-fit.md` (this machine only).

## Planned Follow-Up: BigStitcher Parameter Automation (discussed, not yet implemented)

- Goal: automate optimization of BigStitcher parameters to fine-tune alignment accuracy, building directly on the findings in `BIGSTITCHER_STITCHING_ANALYSIS.md` (residual ceiling ~2.9px from 4x-downsampled narrow strips; STAGE 2 signal starvation; 3-7% zoom/ETL pixel-size miscalibration).
- Core architecture idea: keep fusion OUT of the optimization loop. Alignment only writes registrations into the XML, so parameter sets are evaluated with cheap align-only macro runs (minutes for 16-48 views) and fusion happens exactly once at the end with the winner. BigStitcher caches interest-point detections and pairwise comparisons, so sweeping thresholds (relative/absolute, ICP params) only re-runs global optimization.
- Objective function comes from existing logs + XML, no fusion needed: first-round Avg/Max Error, dropped links, pairwise r values, DoG peaks per view, ICP correspondence counts, plus a scaled-grid fit of final tile positions in the stitched XML (residual scatter + fitted scale ratio, which doubles as the self-calibration estimate). IMPORTANT: the score must combine residual + connectivity + dropped links - Avg Error alone is gameable by dropping links.
- Most gains are deterministic, no search needed: STAGE 1 downsample from overlap % (target strip >= ~150-300px, i.e. 2x/2x/1x for 10% overlap), DoG min/max intensity or empty-view exclusion from tile percentile stats, and a self-calibration guard when fitted |s-1| > ~2%. Remaining search space is small: relative/absolute thresholds, icp_maximum_error, lambda, STAGE 2 scoping - coordinate descent over 3-4 knobs = ~10-20 align-only runs per dataset.
- Two caveats every phase must respect: (1) headless run() silently ignores macro keys that do not match the installed BigStitcher version's dialog labels (2.5.3 as of 2026-10-23), so any automated loop must verify effective params from logs (e.g. "Correcting coordinates for downsampling") before trusting scores; (2) the zoom/ETL pixel-size miscalibration cannot be tuned away - it needs the calibration fix or the self-calibration guard (analysis §9 revised priorities 1-2).
- Phased plan: Phase 0 = split macro templates into align-only + fuse-only and parameterize relative/absolute/icp_maximum_error/lambda via config (defaults = current hardcoded values, byte-identical output, no production behavior change); Phase 1 = log/XML scorer module + report-only mode grading past runs; Phase 2 = deterministic adaptive selection; Phase 3 = optional search driver (align-only SLURM array -> score -> single fusion).
- Phase 0 safety property: it only parameterizes macro keys already exercised verbatim in the current templates AND in the real production log (alder-j/072326/.../1642738_align_fuse_bigstitcher.log) - no new macro keys, so no macro-key-drift risk in that phase. Custom link-filtering numbers (today a baked-in RELAXED (5.0x/7.0px) dialog preset label) and explicit DoG min/max intensity are deferred because both need dialog-label verification against BigStitcher 2.5.3 before being sent headlessly.
- Full roadmap including the complete Phase 0 implementation spec (exact files, config keys, backward-compat notes, validation plan, risks) is saved at `/h20/home/lab/.opencode/plan/bigstitcher-parameter-automation.md` (this machine only).

## Recent Change: BigStitcher Alignment Optimization — Phase 0 (align/fuse split + knob config)

- `bigstitcher_macro_templates.py` now has `BIGSTITCHER_ALIGN_ONLY_OUT` (STAGE 1+2, no fusion) and `BIGSTITCHER_FUSE_ONLY_OMEZARR_OUT` / `BIGSTITCHER_FUSE_ONLY_HDF5_OUT` (STAGE 3 only), composed from slices of the combined templates so the stage text is byte-identical across all three. Hardcoded knobs parameterized as keyword placeholders (`{link_relative}`, `{link_absolute}`, `{icp_maximum_error}`, `{icp_lambda}`) plus a `BIGSTITCHER_PARAMS:` print line at the top of every template (self-describing logs; the params block is the ONLY addition to the generated macros).
- `constants.py`: `BIGSTITCHER_LINK_RELATIVE` (2.5), `BIGSTITCHER_LINK_ABSOLUTE` (3.5), `BIGSTITCHER_ICP_MAXIMUM_ERROR` (5), `BIGSTITCHER_ICP_LAMBDA` (0.10) — defaults equal the previously hardcoded values; `bigstitcher.py` formats relative/absolute with 3 decimals and lambda with 2 so the macros stay byte-identical. Config keys under `bigstitcher:` in both yamls.
- `bigstitcher.py`: `_resolve_automated_sampling(...)` helper (extracted from the twice-duplicated automated-sampling block) used by the two combined getters and the new `get_bigstitcher_align_only_macro` / `get_bigstitcher_omezarr_fuse_only_macro` / `get_bigstitcher_hdf5_fuse_only_macro`. The align-only macro writes registrations into the input XML; the fuse-only macro takes the same XML after an align run. `make_bigstitcher_slurm_dir_and_macro(path, format, stages='align+fuse')` writes `macro.ijm` / `macro_align.ijm` / `macro_fuse.ijm`.
- `automated.py big-stitcher-align --stages align|fuse|align+fuse` (default unchanged, byte-identical): `align` submits the align-only macro with a `BIGSTITCHER_ALIGN_SUCCESS:` gate and log suffix `align_only_bigstitcher` (no fused-output check, no metadata-fix follow-up, no downstream steps); `fuse` keeps the fused-output gate + follow-ups with suffix `fuse_only_bigstitcher`; `is_bigstitcher_log_successful` globs are unaffected.
- Two runtime fixes required by the Fiji-Latest upgrade (found during baseline runs): `constants.py` now picks the launcher that exists (`ImageJ-linux64` for stable, jaunch-based `fiji-linux-x64` for latest) instead of hardcoding `ImageJ-linux64`; and `BIGSTITCHER_ALIGN_TEMPLATE` prepends `_JAVA_OPTIONS=-Djava.awt.headless=true` because the jaunch launcher shows a splash screen that needs X11 and dies with `NoClassDefFoundError sun.awt.X11.XToolkit` on SLURM compute nodes (harmless for the stable launcher).
- Validation: py_compile clean; byte-identical macro regression (diff = only the 3-line params block); `--stages` in CLI help; align-only macro has STAGE 1+2 with no fusion banner; fuse-only has no STAGE 1 with resolved subsampling; one align-only SLURM smoke run.

## Baseline Results: BigStitcher 3.0.8 vs 2.5.3 (SLURM runs, production pipeline as-is)

- Test data: `/CBI_FastStore/test_data/mesospim/omezarr/` (`delima_071326_1channel_3x1tiles`, `dutta_brain_2channels_4x3tiles`, `102226_2_green_embryo_2x2tiles_2channels/102226_2`). Job logs under each dataset's `logs_mesospim_utils/`; parse with the metrics parser (grep-based; section-7 fields). Reference numbers from the 2.5.3-era log `head30/.../2039520_align_fuse_bigstitcher.log`.
- delima (Mag4x, 18 tiles, 1ch, job 2049507, cold start): BS 3.0.8 first-round Avg 105.0px / Max 322.0px, 13 dropped links, 14 opt rounds, final Avg 1.75px / Max 3.74px, 0 zero-detection views, ICP clean (0 not-enough / 0 no-corresponding), success marker + fresh fused OME-Zarr. Shape mirrors the alder-j 2.5.3 reference run (13 link-dropping rounds from a miscalibrated initial grid).
- 102226_2 (Mag8x, 2x2 grid, 8 tile-channels): COLD-start run 2049505 first-round Avg 83.62px / Max 110.67px -> final Avg 1.46px, 2 dropped links, ICP not-enough 1 / no-corresponding 15 - directly comparable to the 2.5.3 reference (first-round Avg 83.7px / Max 110.7px -> final Avg 1.26px): alignment quality carried over from 2.5.3 to 3.0.8 on the same XML. The shipped montage is from the clean re-run 2049512 (fresh metadata, final Avg 1.03px); that re-run starts from the already-stitched XML so its first-round Avg 5.03px is a warm-start number, not comparable to the 2.5.3 baseline.
- dutta (Mag4x, 4x3 grid, 2ch, 24 tiles, job 2049509): first-round Avg 88.69px / Max 195.43px, 5 dropped links, 6 opt rounds, final Avg 2.39px / Max 3.31px, 0 zero-detection views, ICP clean, success marker + valid fused OME-Zarr. Heavy heap pressure during fusion: ~186 `block error (will be re-tried): OutOfMemoryError: Java heap space` events during pyramid writing (s0->s1); all blocks recovered via retry (each level ends `remaining for retry: 0`) but this confirms the fusion-OOM note below on a 2ch 4x3 grid.
- Exceptions counted in these logs are benign except the dutta fusion block retries noted above: optional-export `VerifyError` (OpenSeaDragon path, caught), `HeadlessException` from the macro's final "Wrapping ImagePlus around fused image" display step (caught), and `Could not find X11 library, not running XInitThreads` (headless startup).
- Macro-key drift on 3.0.8: all keys the templates send (alignment, DoG detector, ICP, fusion, downsampling) still take effect under the jaunch-based latest launcher - logs show resolved parameters and `Downsampling: (4, 4, 1, 1, 1) with relative downsampling of (2, 2, 1, 1, 1)` style lines, and every macro-controlled stage banner appears. No silent key drops observed.

## Recent Change: Delima Auto-Tune Search Result + Verification Fixes

- The first auto-tune search ran on delima (6 align-only candidates, lr 2.5-7.0 x la 3.5-5.0, scored with the Phase 1 scorer). WINNER lr7.0/la5.0 at 68.0/100 vs in-search control (config thresholds) 54.0 vs original baseline composite 45.6 - zero dropped links (control dropped 6). link_absolute is the dominant axis; the fitted grid-fit numbers (scale offset +5.55-5.59%, residual scatter 38-39px) are FLAT across all candidates, confirming the zoom/ETL miscalibration is not tunable by link thresholds - it needs the self-calibration correction (guard measured x +4.65%, y +5.58% vs 2% threshold; correction disabled by default).
- Two pipeline bugs found and fixed during verification: (1) the fuse success marker passed on a STALE container from a previous run (BigStitcher refuses to create an existing container, but `&& [ -e ]` passed) - replaced with `-nt` against the macro; (2) the chained metadata-fix step crashed with FileNotFoundError on `.zattrs` since the mvrecon 9.0.12 pin (it writes zarr v3 `zarr.json`) - both fixed and proven on the real re-run fusion (fresh container written, adjust-scale v3 read/write in the fix log).
- The fused volume metadata is already correct for v3 (the v3 writer transfers voxel size from the tile XML), so adjust-scale is a no-op for v3 rather than a correction.
- The stale baseline fused container was preserved as `pig_ON_Mag4x_Ch488_montage_baseline_nov9.ome.zarr` (non-destructive rename) before the re-fuse.

## Recent Change: Config Changes Applied + Multi-Dataset Validation (approved)

- Config changes applied to `config/main.yaml` and `config/example.yaml` (person-approved): `link_relative: 7.0`, `link_absolute: 5.0` (the search winner), `self_calibration_correction: true`. Constants verified loading 7.0/5.0/True in a fresh process.
- Multi-dataset validation (align-only runs with the new config, scored with the Phase 1 scorer): delima 54.0 -> 60.2 (dropped 6 -> 3), dutta 76.1 -> 82.9 (dropped 5 -> 2), 102226_2 56.9 -> 56.9 (already well-stitched, unchanged, no regression). All runs warm-started from existing registrations, so the dropped-links comparison is the clean signal.
- Self-calibration correction tested end-to-end on the delima winner fusion: guard measured x +4.67%, y +5.58% (> 2% threshold) and rescaled the fused voxel size across all 6 pyramid levels (y 1.2 -> 1.137um, x 1.2 -> 1.146um; z untouched - no resolvable z fit). CAVEAT: the correction is not idempotent - each fuse produces a fresh (nominal) volume so the chained guard corrects once, but manually re-running the guard on an already-corrected volume double-corrects.
- Uncommitted work now spans: `automated.py` (import fix, `-nt` fuse checks, guard-cli name), `bigstitcher.py` (zarr v3 in adjust-scale), `config/main.yaml`, `config/example.yaml`, `PROJECT_CONTEXT.md`.

## Recent Change: Six-Dataset Config Validation (new thresholds, align-only + scorer)

- Ran the new config (7.0/5.0 + correction) align-only on six more datasets and scored all with the Phase 1 scorer. Tiles/score/dropped/grid-offset/flags: exosomes_2 2t/100-vacuous (2 tiles = no scoreable metrics; fused + MAX-rendered, seam check PASSED - continuous field, no step, no ghosting), green_embryo_3x3 27t/58.2 (7 dropped, +2.66%, 1 zero-DR view), almarza_knee 28t/34.4 (39 dropped, +5.95%, 5 zero-DR views - WORST; ran on a lab wrapper `almarza_knee_lab/` with real-copied XMLs + symlinked 590G data because the original is iana-owned read-only), zhang_070726 36t/49.7 (11 dropped, +2.18%, 6 zero-DR views), klimstra 40t/44.5 (21 dropped, +5.45%), alder_mcdyer_lung 48t/51.2 (11 dropped, +5.45%).
- Conclusion: the tuned config clearly improves the previously-validated datasets (delima/dutta) and is neutral on well-stitched ones (102226_2), but 4 of 5 scorable new datasets still drop links at 7.0/5.0 - these have colder/wilder grids (first-round 83-686px) and several have zero-detection views (illumination problems). Per the agreed workflow these are the per-dataset-search candidates (relax beyond 7.0/5.0) plus scope-side attention (illumination, calibration).
- Known recommender quirk: it suggests "try the RELAXED preset (7.0/5.0)" even when that preset is already active - read it as "relax further or fix calibration". Worth a small fix (pass the current thresholds) later.
- Pre-existing logs-dir permission gotcha: datasets owned by other users (iana) have read-only `logs_mesospim_utils/` - sbatch fails in 1s. Workaround: rename the old dir aside (parent dir must be lab-writable) or use a wrapper dir with real-copied XMLs.

## Recent Change: Auto-tune Search Driver (Phase 3 of the parameter-automation roadmap)

- New CLI in `mesospim_utils/automated.py`: `big-stitcher-search <dir> [--link-relative ... --link-absolute ...] [--fuse/--no-fuse] [--dry-run]` and `big-stitcher-search-score <dir> [--fuse/--no-fuse]`.
- Driver: builds a coarse grid over the STAGE 1 link thresholds (config baseline plus 5.0/7.0 relative, 5.0 absolute by default; explicit comma lists override), patches the module-level constants per candidate, generates a per-candidate tagged macro (`bigstitcher/macro_align_lr2.5_la3.5.ijm`), and submits the align-only runs as a SLURM dependency chain (parallel runs would clobber the shared stitched XML). Each candidate's command copies the stitched XML to a per-candidate snapshot (`bigstitcher/snapshot_<tag>.xml`) right after its run. A final chained step runs the scoring subcommand.
- Scoring subcommand: globs the `*search_*_align_only_bigstitcher.log` files, attributes each to its candidate via the `BIGSTITCHER_PARAMS:` line, scores each with the Phase 1 scorer (log metrics + the per-candidate snapshot's XML grid fit), prints the ranked report, restores the winner's snapshot into the stitched XML, and submits exactly one fuse-only run (the fuse uses the restored registrations). Candidates with no scoreable metrics (single-tile datasets emit no error lines) are flagged and cannot win vacuously over real measurements; a missing winner snapshot skips fusion with a manual-restore warning instead of fusing the last candidate's registrations.
- Budget: one align-only run per candidate (6 by default) plus exactly one fusion, per the plan's ~10-20 runs / one fusion. `--dry-run` prints the candidate commands without submitting.
- Validation: py_compile clean; CLI help; dry run on delima produced the 6-candidate chain with correct per-candidate macros (each carries its own `BIGSTITCHER_PARAMS` values) and snapshot steps; the scoring subcommand smoke-tested against a real align-only log (params attribution + no-scoreable-metrics warning + safe no-snapshot path).

## Recent Change: Adaptive Selection + Self-Calibration Guard (Phase 2 of the parameter-automation roadmap)

- **Adaptive STAGE 1 downsampling from overlap**: `downsample_in_x/y: 'overlap'` (a new value alongside `'automated'` and numbers) computes the largest lateral downsample keeping the overlap strip >= `bigstitcher.min_overlap_strip_px` (default 150px): `strip_px = tile_px * overlap / downsample`. Reads overlap + tile_shape from the annotated metadata JSON next to the XML; z stays 1. For 10% overlap / 3000px tiles -> 2x/2x/1x (verified on delima). The hdf5 getter now shares `_resolve_automated_sampling` with the omezarr getter (behavior-identical for the existing modes).
- **DoG detection intensity gating** (config `dog_intensity_mode: 'off'|'common'|'fixed'`, default off = byte-identical macros): 'common' sends `use_same_min_&_max_intensity_for_all_views=true` (BigStitcher computes a common min/max across views, making the DoG threshold comparable instead of per-image relative); 'fixed' sends `set_minimal_and_maximal_intensity=true` plus explicit `minimal_intensity`/`maximal_intensity`. Macro keys verified against the installed multiview-reconstruction 9.0.12 DifferenceOfGUI.class AND upstream DifferenceOfGUI.java + Interest_Point_Detection.java: the first dialog checkbox is `Set_minimal_and_maximal_intensity` (default false); when true the sub-dialog shows the numeric fields; when false the sub-dialog shows the `Use_same_min & max intensity for all views` checkbox. All 6 detection sites (3 levels x omezarr+hdf5) parameterized with trailing-space placeholders so 'off' keeps the macros byte-identical. The `BIGSTITCHER_PARAMS:` log line now also carries the dog keys.
- **Self-calibration guard** (`bigstitcher.py apply-self-calibration-guard <stitched_xml_or_dir> <fused_omezarr>`): scores the stitched XML with the Phase 1 scorer, prints every fitted axis offset, and when |s-1| > `self_calibration_threshold_pct` (2%) warns loudly; with `self_calibration_correction: true` it divides the fused output's scale array by s per fitted axis (true px = nominal px / s). Handles Zarr v2 `.zattrs` (multiscales top-level) and Zarr v3 `zarr.json` (multiscales/axes under `attributes.ome`, axes as dicts with name keys). Wired as a SLURM dependency after the adjust-scale step in the omezarr final path of big-stitcher-align. Verified on delima: x 0.9557 / y 0.9472 / z unchanged -> corrected px 1.137um/1.147um matching the analysis-doc implied true px (~1.138um for Mag4x photometrics).
- Validation: py_compile clean; byte-identical combined-macro regression (only the params line differs, documented); CLI help; functional runs of all three pieces on the baselines.

## Recent Change: BigStitcher Scoring Module (Phase 1 — log/XML scorer + report-only mode)

- New `mesospim_utils/bigstitcher_scoring.py` (Phase 1 of the parameter-automation roadmap). Stdlib only (re/json/math/ElementTree/typer-lazy), runnable with any Python without the mesospim_utils environment.
- `parse_align_log(log_path)` parses all analysis-doc §7 metrics from a SLURM log: first-round/final Avg+Max errors (one pair per optimization round), dropped links + their errors, `added strong link between` count (wording matches both single-view and group_tiles `between 0-11, 0-23 and 0-6, 0-18` forms), DoG peaks when present, zero-detection views, ICP not-enough/no-corresponding failures, per-view `Min=X max=Y` intensity lines (STAGE 2; a view is starved only when EVERY line for it has dynamic range < 10 counts — lines repeat per STAGE 2 level), effective-downsampling lines, the `BIGSTITCHER_PARAMS:` line (parsed into a dict), the last `Saved xml 'file:...'` path (locates the stitched XML), versions, success markers.
- `score_from_xml(stitched_xml_path)` fits a scaled grid to the final tile positions: final position = sum of ALL affine ViewTransform translations per ViewRegistration (BigStitcher composes in document order); nominal grid = sum of the non-`Stitching Transform` translations. Per axis with >= 2 distinct nominal offsets, `final = s * nominal + t` is fit by least squares — the fitted scale ratio reproduces the analysis doc's 3-7% zoom/ETL miscalibration (delima +5.57%/+4.63%, dutta +6.75%/+6.66%, 102226_2 +7.16%) and the residual scatter around the fit is the tile-grid misalignment signal. Single-row grids skip the degenerate axis (z is always undefined). Per-setup fitting mixes channel (chromatic) offsets into the residual on multi-channel datasets — a known Phase 1 limitation.
- `alignment_quality_score(metrics, xml_metrics)` returns (score 0-100, components): weighted 0.4 residual (XML fit scatter when available, else log final Avg Error; exp(-residual/25)) + 0.3 connectivity (min(1, added_links/(n_views-1)) * exp(-dropped/10)) + 0.2 signal (exp(-zero_views) * non-starved fraction) + 0.1 ICP (exp(-fails/20)). Missing inputs are neutral; dropped links and ICP failures default to zero events. Never residual alone — dropping links kills the connectivity term.
- `verify_params_effective(metrics, intended_params)` drift guard against the log's `BIGSTITCHER_PARAMS:` values (False when the line is absent). `recommend_params(metrics, xml_metrics)` emits the deterministic recommendations (RELAXED thresholds on large first-round error + many drops, DoG intensity gating on zero-detection views, self-calibration warning when |s-1| > 2%, downsampling reduction on high grid scatter, ICP failures, final-error above the ~3px ceiling).
- CLI (single-command typer, lazy import): `python mesospim_utils/bigstitcher_scoring.py <dataset_dir or log_path> [...] [--json]` grades past runs — newest `*_align_*bigstitcher.log` per dataset dir, the stitched XML from the log's last `Saved xml` line, score + recommendations; `--json` for machine-readable output. Executes nothing, writes nothing. Older logs (2.5.3) degrade gracefully: no intensity lines -> `n/None`, XML on another machine -> log-based residual fallback.
- Validation: py_compile clean; CLI help verified in the mesospim_utils env; report-only run over the three 3.0.8 baselines + the 2.5.3 reference log reproduced all previously parsed numbers exactly and the grid-fit scale offsets landed in the documented 3-7% band; scores rank sensibly (delima 45.6 < dutta 76.1 < 102226_2 56.9 warm / 79.1 ref — delima is the known worst dataset).

## Recent Change: Config-Selectable Fiji Distribution + OME-Zarr 0.5 Sharded Read (verified)

- `mesospim_utils/fiji.py` now supports two install distributions selected by config `fiji.distribution: stable|latest` (new key in `config/example.yaml` + `main.yaml`; loaded as `FIJI_DISTRIBUTION` in `constants.py` with env override `FIJI_DISTRIBUTION`). Default `stable` preserves the prior behavior exactly (Java 8 + pinned classic BigStitcher 2.5.3).
- `latest` installs the Fiji-Latest archive, pinned to the exact verified build (`https://downloads.imagej.net/fiji/archive/latest/20261030-1817/fiji-latest-linux64-jdk.zip`, ~706 MB, sha256 `a1c7996c...` - imagej's own sidecar confirms the snapshot is the build that was verified; the `fiji/latest/` alias is a moving target). The download's sha256 is checked against `FIJI_LATEST_ZIP_SHA256` in fiji.py and the install refuses on mismatch (override the URL with the `FIJI_LATEST_ZIP_URL` env var to install a different build). Self-contained with Java 21, BigStitcher 3.0.8 (in `jars/`, not `plugins/`), multiview-reconstruction 9.0.11, and the new N5 stack (n5 4.0.1, n5-zarr 2.0.1, n5-ij 5.0.0) all bundled. The latest flow skips the update site + updater + pinned jars entirely; enabling the BigStitcher update site on top would pull conflicting 2.x-era jars. Launcher is jaunch-based `Fiji/fiji-linux-x64`; `fiji_exe` and `bigstitcher_jar` handle both layouts.
- Extraction fix: `_extract_zip` now preserves zip mode bits (custom zipfile loop, not `extractall`). Bare extractall drops the exec bit on `config/jaunch/jaunch-linux-x64` and the latest launcher dies silently before printing anything.
- Latest-only pin: bundled multiview-reconstruction 9.0.11's `bdv.multimg.zarr` loader cannot open OME-Zarr v0.5 (zarr v3) collections (opens the root with legacy `N5ZarrReader`, fails `No container exists`). fiji.py now pins multiview-reconstruction **9.0.12** from Maven (Java 21, v3-aware `XmlIoAllenOMEZarrLoader`) via `MVRECON_PINNED_JAR`, disabling the bundled 9.0.11. 9.0.12 is the best-of-both version: the BigStitcher update site pairs it with BigStitcher 3.0.8, and it keeps the `GlobalOptTwoRound.computeTiles` signature BigStitcher 3.0.8 calls. **9.1.x+ is a trap**: 9.1.4/10.0.0 have the v3 loader but changed that API — BigStitcher 3.0.8 alignment crashes with `NoSuchMethodError: GlobalOptTwoRound.computeTiles` plus `VerifyError: Bad type on operand stack` in the ICP path (found during the baseline runs).
- VERIFIED end-to-end on a fresh from-scratch `--force` install WITH the pin (exit 0): a 1-tile 4x4 OME-Zarr 0.5 SHARDED collection loads through the real BigStitcher XML loader path (`XmlIoSpimData2().load()`) with correct pixel reads (SHARDED_LOAD_OK=True). Requirements for the read: (1) `zarr.version="ZarrV3"` attribute on the ImageLoader element in the XML (found in 9.1.4 bytecode; without it the loader defaults to the legacy v2 reader), and (2) the tile group's zarr.json `attributes.ome` must contain a `multiscales` ARRAY (axes/datasets directly on `ome` without it fail "Could not parse OME-ZARR multiscales object").
- Hardening found during final verification: `bigstitcher_jar`'s glob (`Big_[Ss]titcher*.jar`) never matched the 3.x core jar naming `BigStitcher-3.0.8.jar` (no underscore), so `is_fiji_ready` was always False on latest installs and every plain re-run took the slow refresh path; the glob now also matches `Big[Ss]titcher*.jar` and the fast "already installed" path triggers. Distribution switches are also hardened: `ensure_fiji_and_bigstitcher` reads the stamp's recorded distribution and forces a clean reinstall when it differs from the selected one (a plain ensure with the other distribution used to build a broken hybrid: stable pins shadowing the latest archive, `ome/ngff/v04` NoClassDefFoundError).
- STABLE CANNOT read 0.5 sharded by design (Java 8, n5-zarr 1.5.1 is pure zarr v2) - sharded support requires `fiji.distribution: latest`.
- NOTE for future pipeline work: the repo's XML generator (`mesospim_metadata_to_bigstitcher_xml`) does NOT yet emit `zarr.version="ZarrV3"`; pointing the automated pipeline at 0.5 sharded collections needs that attribute added (harmless with 9.0.11 too, but the metadata must also carry the multiscales array).
- Both from-scratch `--force` installs (stable and latest) exit 0 with stamps and headless import checks passing; the latest install with the mvrecon pin additionally passes the 0.5 sharded read end-to-end and the fast idempotent re-run.

## Recent Change: Fiji Auto-Install Hardening (resolved and verified)

- `mesospim_utils/fiji.py` now delivers a working Fiji + BigStitcher install. Verified: a full from-scratch `--force` run exits 0, writes the `.fiji_bigstitcher_ready` stamp, and an independent headless import check (`from net.preibisch.stitcher.plugin import BigStitcher`) passes.
- Root cause of the long debugging saga was path resolution, not the filesystem: `resolve_install_root()` returns `<repo>/fiji-linux` (the dir CONTAINING `Fiji.app`), but `_install_pinned_bigstitcher` and `bigstitcher_jar` built `root / "plugins"` paths that landed jars OUTSIDE `Fiji.app`, where the classloader never sees them. Script-side checks on the same wrong path reported "present" while diagnostics on `Fiji.app/plugins/` reported "missing", which masqueraded as phantom beegfs views.
- Fixed: all jar paths now resolve app-relative (`app = root / "Fiji.app"`); `bigstitcher_jar` globs `root / "Fiji.app" / "plugins"`.
- Also fixed: `--update add-update-site` exits non-zero with "Site ... was already added!" on re-runs - the script now checks `Fiji.app/db.xml.gz` for the site before registering; the Fiji 544MB download is skipped when `fiji_exe(root)` already exists (re-runs are fast); pinned jars are always downloaded fresh to `/tmp`, zip-validated, copied via `_copy_jar_into_place` (with fresh-subprocess `unzip -t` verification and retries), and the headless verify retries 3x for FS metadata lag.
- Working pinned Maven set (from `https://maven.scijava.org/content/groups/public`): BigStitcher 2.5.3 -> `plugins/Big_Stitcher-2.5.3.jar`, multiview-reconstruction 7.0.7 -> `plugins/`, multiview-simulation 0.2.0 -> `jars/`, with any Java-21 `jars/multiview-reconstruction*.jar` disabled in place. Base Fiji's own mpicbg + bigdataviewer-core are sufficient - no extra pins needed.
- Genuine beegfs quirk worth remembering: per-process negative-dentry caching sticks a stale "No such file" in one process while fresh processes see the file. Verify filesystem state from fresh subprocesses, not one long-lived process's view.

## Recent Work Snapshot

- Strong signs of recent edits in:
  - `mesospim_utils/bigstitcher.py`
  - `mesospim_utils/bigstitcher_macro_templates.py`
  - `mesospim_utils/config/`
  - `mesospim_utils/metadata.py`
  - `mesospim_utils/constants.py`
  - `mesospim_utils/fiji.py`
  - `setup.cfg`
- Recent debugging edit also touched:
  - `mesospim_utils/preprocess.py`
  - `mesospim_utils/basicpy_worker.py`
- Last visible commit message draft seen previously: `Relaxed ICP refinement for chromatic abberation`
- Likely recent intent:
  - relax ICP refinement behavior
  - tune channel-alignment/chromatic-aberration handling
  - adjust BigStitcher refinement downsampling and fusion memory settings
  - move deconvolution PSF/objective parameters out of hardcoded `rl.py` values and into config

## Recent Change: BigStitcher Stitching Review (read-only)

- Added `BIGSTITCHER_STITCHING_ANALYSIS.md` at the repo root: a detailed read-only review of the BigStitcher alignment path explaining why stitched montages can come out poorly aligned, based on code inspection plus a full real run log (`alder-j/072326/.../1642738_align_fuse_bigstitcher.log`). No code was changed for this review.
- Key findings in that document: the real run's initial metadata grid disagreed with measured tile offsets by a constant ~5.6% per step (corrections grew ~150px/step to ~550px at the grid corner, forcing 13 link-dropping rounds from Avg Error 417px to 2.73px); STAGE 2 DoG/ICP ran on near-zero signal (2-9823 peaks per view due to per-image relative thresholds, 5 views with zero detections, ICP pairs with 1-149 matches); STAGE 2 with `group_tiles` only applies one rigid transform per (channel, illumination) group and never refines tile geometry.
- Top recommendations (details and verification greps in the document): derive the initial grid from actual stage positions / re-check nominal `Pixelsize in um` against measured offsets; lower STAGE 1 downsampling to 2x/2x/1x for 10% overlap; set explicit min/max intensity in the DoG stage; re-scope STAGE 2 to lateral-only or drop it for samples without common per-channel structure.

## Recent Change: IMS Export From Fused OME-Zarr

- Follow-up robustness fix for direct IMS export: the SLURM-launched `omezarr_to_ims.py` command in `mesospim_utils/automated.py` now derives the target env root and `PyImarisWriter` package location from `general.location_pyimariswriter_environment` itself, then exports `LD_LIBRARY_PATH` before starting the dedicated interpreter. This avoids relying on conda activation hooks, the submitter's `CONDA_PREFIX`, or the submitter's Python environment when IMS jobs are submitted from a different env.
- The `--final-file-type ims` path now builds IMS directly from the fused montage OME-Zarr via `mesospim_utils/omezarr_to_ims.py` and no longer depends on intermediate TIFF extraction.
- Added `general.location_pyimariswriter_environment` in config so the direct IMS conversion can run under `/h20/home/iana/.conda/envs/pyimariswriter/bin/python` instead of the standard mesospim_utils environment.
- The existing `--ims-resolution-level` option still controls the exported OME-Zarr multiscale level for both direct `ims` output and `tiff` output, with a default of `0`.
- The `--final-file-type ims` path in `mesospim_utils/automated.py` no longer uses one bulk `extract-tiff-series` job.
- Added `--ims-resolution-level` to `automated-method-slurm` and `big-stitcher-align`; default is `0`.
- The fused OME-Zarr to IMS path now submits one SLURM array per channel, with one task per z-plane TIFF.
- Added `extract-single-tiff-plane` in `mesospim_utils/omezarr.py` to read one plane from `(t, c, z, y, x)` and write one tiled zlib-compressed TIFF.
- TIFF output is flat in the `*_tiffstack` directory and uses names like `composite_r00_t00_c00_z0000.tif` so the current Imaris TIFF-series ingestion can group by channel and z order.
- `mesospim_utils/imaris.py make_ims_from_tiff_series()` now names the final IMS file from the TIFF-series directory name instead of the first TIFF stem.
- While validating this change, a pre-existing invalid nested f-string in `mesospim_utils/imaris.py` was simplified so the touched files now compile under Python 3.12.
- Follow-up fix after a branch merge: the IMS path now checks TIFF completion before queueing any extraction jobs, the old bulk `extract-tiff-series` fallback was removed from the automated IMS branch, and TIFF success is now based on one explicit log marker per extracted z-plane.
- TIFF completeness for skip logic now uses the requested `--ims-resolution-level` shape from the source OME-Zarr instead of acquisition tile metadata, so nonzero multiscale levels can skip correctly.
- Follow-up follow-up: the BigStitcher XML points to an OME-Zarr collection root that contains per-tile child OME-Zarrs, not a root multiscale image. IMS skip/extraction logic now resolves one representative child tile OME-Zarr for pre-fusion multiscale shape and scale-factor reads, while still extracting TIFFs from the fused montage OME-Zarr.
- Follow-up follow-up follow-up: representative source tiles can be 3D `(z, y, x)` rather than 5D. The OME-Zarr reader now normalizes source-level z-depth and scale extraction across both 3D per-tile and 5D datasets via `get_level_zyx_info()`, and the IMS skip/extraction path uses that normalized information instead of assuming 5D source arrays.
- Follow-up follow-up follow-up follow-up: the IMS TIFF extraction path now reuses previously extracted planes on a per-plane basis. A plane is treated as complete only when the TIFF exists, has nonzero size, and has a matching `OMEZARR_TO_TIFF_SUCCESS:` log marker. When the TIFF stack is partial, the pipeline now submits compact SLURM arrays containing only the missing z-plane extraction commands for each affected channel instead of requeueing every plane.
- Follow-up follow-up follow-up follow-up follow-up follow-up: IMS TIFF extraction from fused OME-Zarr now batches work in 10-plane chunks per SLURM task. Array jobs are submitted at batch starts `0, 10, 20, ...` per channel, each task reads up to 10 full z-planes into memory in one OME-Zarr slice, and then skips rewriting any TIFFs that already exist with nonzero size while still emitting per-plane success markers for completion tracking.
- Follow-up follow-up follow-up follow-up follow-up follow-up follow-up: the new per-channel batched TIFF extraction exposed a pre-existing `submit_array()` script-path collision. Separate channel arrays submitted into the same directory were both wrapping `sbatch.sh`, so the second submission could overwrite the first before execution and make channel-0 jobs run channel-1 commands. `mesospim_utils/slurm.py::submit_array()` now names the wrapped script from `log_prefix`/`log_suffix` (for example `sbatch_omezarr_to_tiff_stack_c00.sh`) so concurrent arrays in one directory do not clobber each other.
- Follow-up follow-up follow-up follow-up follow-up: the downstream TIFF-to-IMS dependency path exposed a pre-existing bug in `slurm.py::sbatch_depends()` where multiple `afterok` job IDs were formatted as `--depend=afterok:job1 --kill-on-invalid-dep=yes :job2`, which `sbatch` rejects. This is now fixed to emit one valid colon-joined dependency list. While validating that fix, two additional pre-existing Python 3.12 parser issues in `slurm.py` were also normalized: command escaping in `submit_array()` and log filename formatting in `format_sbatch_wrap()`.

## Recent Change: TIFF Series As A Final Automated Output

- `mesospim_utils/automated.py automated-method-slurm` now accepts `--final-file-type tiff` in addition to `omezarr`, `hdf5`, and `ims`.
- The new `tiff` final-output path reuses the existing fused OME-Zarr to TIFF-series extraction flow that was previously only used as the intermediate step for `ims`.
- Behavior is intentionally minimal: BigStitcher still fuses to montage OME-Zarr first, TIFF extraction still supports partial resume via per-plane success markers, and the workflow now stops after TIFF extraction instead of queueing Imaris conversion.
- The existing option name `--ims-resolution-level` was retained and now also controls the exported multiscale level for `--final-file-type tiff`.

## Recent Change: BigStitcher Rerun Skip Logic

- Added a skip gate to `mesospim_utils/automated.py::big_stitcher_align()` so BigStitcher is not requeued when prior work is already complete.
- Added helpers in `mesospim_utils/bigstitcher.py` to:
  - derive expected fused montage and TIFF-stack output paths
  - detect successful prior BigStitcher logs
  - validate the IMS cleanup case where the montage OME-Zarr was removed after TIFF extraction
- Added `validate_ome_zarr_multiscale()` in `mesospim_utils/omezarr.py` for lightweight fused OME-Zarr validation.
- Current skip policy is conservative and only applies to OME-Zarr fusion paths:
  - skip if a prior BigStitcher success log exists and `_montage.ome.zarr` validates as complete
  - for `final_file_type=ims`, also skip if the montage is absent but the `_tiffstack` directory exists with exactly `channels * z_planes` TIFFs and all TIFFs have the same nonzero size
- New BigStitcher submissions now append an explicit `BIGSTITCHER_SUCCESS:` marker to the SLURM log after Fiji exits cleanly.

## Recent Change: Objective Profiles For Deconvolution

- Added objective-profile support under `decon.objectives` in `mesospim_utils/config/example.yaml`.
- Added `decon.default_objective` support in config and surfaced it in `mesospim_utils/constants.py`.
- Added `--objective` to `automated-method-slurm`, which now passes through `mesospim_utils/slurm.py` into `mesospim_utils/rl.py`.
- `mesospim_utils/rl.py` now resolves PSF optics from the selected objective profile instead of hardcoded values.
- Current precedence is: CLI `--objective` -> metadata objective name if present in future -> config `decon.default_objective`.
- Sample refractive index still comes from runtime metadata/CLI `refractive_index`; objective optics stay in config.
- `psf_model` is back to a hardcoded default of `gaussian` in `mesospim_utils/rl.py` and is no longer configured per objective.
- `oversample_factor` now uses the `psf.py` default and is no longer configured per objective.
- If `objective_immersion_ri_design` or `objective_immersion_ri_actual` is set to `'auto'`, `rl.py` now substitutes the actual sample RI used for deconvolution.

## Recent Change: OME-Zarr Preprocessing And Workflow Order

- Added optional `--basicpy` and `--gain-correction` to `automated-method-slurm` in `mesospim_utils/automated.py`.
- Processing order is now fixed as: `decon -> basicpy -> gain correction` when those stages are enabled.
- `.btf` input is now normalized to tile OME-Zarr before any downstream processing, even when preprocessing is not requested.
- For `.ome.zarr` deconvolution, pre-DECON BigStitcher XML generation remains required so `rl.py` can clone collection metadata into the decon output.
- Added `mesospim_utils/preprocess.py` with low-level commands:
  - `basicpy-apply`
  - `gain-correction-apply`
- Both preprocessing commands now operate only on OME-Zarr level `0` and then regenerate multiscales, matching the decon-style output pattern more closely than the original reference scripts.
- Rows/cols now come from metadata `grid_size`; gain correction uses metadata `overlap` by default, with only the low-level command exposing an override.
- Added SLURM config plumbing for `slurm.basicpy` and `slurm.gain_correction`, plus `general.location_basicpy_environment` in `config/example.yaml`.
- Added XML regeneration before BigStitcher alignment for any new tile OME-Zarr collection produced by preprocessing or deconvolution.
- Preprocess-triggered BigStitcher alignment submission now reuses the older working queue shape: `queue_bigstitcher_alignment()` submits `automated.py big-stitcher-align` with `SLURM_PARAMETERS_FOR_BIGSTITCHER` instead of the lightweight dependency profile.
- Added conservative output-based skip logic for preprocess group submission in `mesospim_utils/slurm.py`: unless `--overwrite` is requested, completed `basicpy-apply` and `gain-correction-apply` channel/filter groups are skipped when all expected output tiles are present and each expected tile contains valid tile-level multiscale metadata with all listed dataset paths present.

## Recent Change: BigStitcher XML Tile Path Fix

- Fixed `modify_file_names_in_annotated_metadata()` in `mesospim_utils/metadata.py` so the default `.ome.zarr` rewrite is idempotent.
- Before this fix, metadata entries that already ended in `.ome.zarr` were rewritten to `.ome.zarr.ome.zarr`, which produced invalid `<zgroup path="...">` entries in generated BigStitcher XML.
- The failure was reproduced on `/h20/Acquire/MesoSPIM/dutta-p/4CL94_donotdelete/060826_movedtopublic/basicpy/gain_correction/MI_3_Mag4x_Ch488_Ch561_BASICPY_GCORR.ome.zarr.xml` and matched the user-reported BigStitcher `AllenOMEZarrProperties.getDataType(...)` NPE.
- Regenerated that XML after the patch and confirmed the tile paths now end in a single `.ome.zarr` suffix.

## Recent Change: BigStitcher XML Rewrite Plumbing And Success Marker Guard

- Fixed `mesospim_utils/bigstitcher.py::mesospim_metadata_to_bigstitcher_xml()` so it now passes the caller-provided `modify_filename_in_xml` argument into `modify_file_names_in_annotated_metadata(...)` instead of silently ignoring it.
- Added `normalize_ome_zarr_suffix()` in `mesospim_utils/metadata.py` to collapse repeated `.ome.zarr` suffixes and ensure the final tile name ends in exactly one `.ome.zarr`.
- Tightened `mesospim_utils/automated.py::queue_bigstitcher_alignment()` so the `BIGSTITCHER_SUCCESS:` marker is printed only if the expected fused output path actually exists after Fiji exits.
- Tightened `mesospim_utils/bigstitcher.py::is_bigstitcher_log_successful()` so explicit success markers only count when the fused output exists, and fallback log parsing now treats `exception` / `nullpointer` as failure markers.
- Follow-up robustness fix: `mesospim_utils/bigstitcher.py::make_bigstitcher_slurm_dir_and_macro()` now avoids rerun failures when an existing `*.xml.backup` file in `bigstitcher/` is protected by another owner. If the default backup path is not writable it chooses a numbered backup filename, and it now uses `shutil.copyfile(...)` instead of `shutil.copy(...)` so shared filesystems that reject `chmod` on the destination do not abort setup.

## Recent Change: Decon Queueing After Preprocess

- Fixed `mesospim_utils/slurm.py::decon_dir()` so it no longer crashes when preprocess output collections are scheduled but not populated yet.
- Added `after_slurm_jobs` support to `decon_dir()` so decon now waits for upstream `basicpy` / `gain_correction` jobs.
- Added `ram_estimate_dir` fallback support so decon can size SLURM RAM from the original input collection when the future preprocess output collection is still empty.
- For `.ome.zarr` preprocess workflows, RAM estimation now falls back to the original root OME-Zarr collection.
- For `.btf` preprocess workflows, RAM estimation and future tile-name templating now fall back to the raw `.btf` inputs so the older workflow remains operational.

## Recent Change: Pre-Decon XML Generation

- Fixed an OME-Zarr decon workflow ordering bug where `rl.py` decon workers expected the input collection XML to exist so they could clone it into the decon output collection.
- `automated_method_slurm()` now queues `queue_bigstitcher_xml(dir_loc, out_dir, ...)` before decon starts when the current decon input is `.ome.zarr`.
- The downstream post-decon `queue_bigstitcher_xml(..., out_dir)` and `queue_bigstitcher_alignment(..., out_dir)` steps remain unchanged, so stitching still targets the decon collection rather than the pre-decon collection.

## Recent Change: Metadata-First Preprocess Grouping

- Fixed preprocess compatibility for older OME-Zarr datasets whose tile names omit the filter segment, e.g. `Mag4_Tile0_Ch405_Sh1_Rot0.12.ome.zarr`.
- `mesospim_utils/preprocess.py` now derives preprocess grouping from metadata first rather than reparsing channel/filter pairs from tile filenames.
- The tile-name regex used by preprocess and `basicpy_worker.py` now accepts both modern names with explicit filter tokens and older names without them.
- When filename filter text is absent, grouping falls back to metadata `CFG.Filter`; only if metadata is missing does it fall back to a channel-only sentinel.
- This keeps newer channel+filter workflows working while allowing older datasets to be grouped effectively by channel via metadata.

## Recent Change: Filter Tokens With Underscores

- Fixed preprocess tile-name parsing for older OME-Zarr datasets whose filename filter token contains underscores, e.g. `Flt525_50_(GFP)`.
- The preprocess and `basicpy_worker.py` tile regex now capture the optional filter segment lazily up to `_Sh`, instead of assuming the filter token contains no underscores.
- This preserves support for all three observed naming styles:
  - no filter in tile name: `..._Ch405_Sh1_...`
  - simple filter token: `..._Ch488_FltGFP_Sh1_...`
  - underscored filter token: `..._Ch488_Flt525_50_(GFP)_Sh1_...`

## Recent Change: Remove `/tmp` Staging From BaSiCPy Apply

- `process_basicpy_group()` no longer stages corrected tiles as `.npy` files in `tempfile.TemporaryDirectory()`.
- `basicpy_worker.py` now processes one `--target-tile` per invocation and writes one temporary `.npy` into a hidden `.basicpy_tmp/` directory inside the final output collection.
- `preprocess.py` immediately reads that per-tile `.npy`, writes the final OME-Zarr tile, and deletes the temporary file.
- This was changed to avoid failures where `/tmp` filled up even though the final destination filesystem had enough free space.
- Tradeoff: the current implementation re-fits BaSiCPy for each target tile, so it should use less temporary disk but may be slower than the prior batch-worker design.

## Recent Change: BaSiCPy Multi-Channel Temp Dir Race

- Fixed a multi-channel BaSiCPy race in `mesospim_utils/preprocess.py` where separate SLURM channel/filter jobs shared one `.basicpy_tmp/` directory under the output collection.
- Each channel/filter group now writes temporary `.npy` files into its own sanitized subdirectory inside `.basicpy_tmp/`.
- This prevents one group from deleting an empty temp directory used by another in-flight group, which had caused intermittent `FileNotFoundError` failures at `np.save(...)` while the second channel was finishing.
- `mesospim_utils/basicpy_worker.py` is still owned by another user on this machine and was not edited in this session; the race was fixed from the orchestrating preprocess side instead.

## Recent Change: BaSiCPy Tile-Level Resume

- Added tile-level resume logic to `mesospim_utils/preprocess.py::process_basicpy_group()`.
- When `--overwrite` is not set, BaSiCPy now skips tiles whose existing output tile directory already has valid tile-level multiscales metadata and all referenced dataset paths present.
- This allows reruns after a partial failure to continue from the remaining incomplete tiles rather than recomputing the whole channel/filter group.

## Recent Change: BaSiCPy Fit Tile Uses Weighted Center And Low-Res Size Score

- Updated `mesospim_utils/preprocess.py` so the default BaSiCPy fit tile for tile OME-Zarr collections is no longer always the middle of the field of view.
- When `--fit-tile` is not passed, preprocess now inspects each tile's lowest-resolution multiscale dataset using the last dataset entry from tile `multiscales` metadata.
- If that lowest-resolution dataset is compressed, preprocess computes the on-disk size of that dataset directory only, using a recursive `os.scandir(...)` size walk.
- It then assigns each tile a weighted score of `0.6 * center_proximity + 0.4 * normalized_low_res_size` and chooses the tile with the highest total score.
- `center_proximity` now uses a squared-distance falloff, `1 - (distance / max_distance) ** 2`, so near-center tiles are penalized less sharply than the earlier linear distance score.
- This applies both to native `.ome.zarr` inputs and to `.btf` datasets that were converted to tile OME-Zarr before preprocessing.
- If compression is not enabled for the lowest-resolution dataset, preprocess falls back to the previous middle-of-field-of-view fit-tile behavior.
- The `basicpy-apply --help` text was updated to describe the new default fit-tile rule.

## Active Debugging Note: BigStitcher OOM During Fusion

- Observed failure mode: SLURM OOM kill during BigStitcher jobs launched from `automated_method_slurm()` through the BigStitcher alignment path.
- Failures were reported late in fusion, often 1-2 channels into fusion, and more often on HDF5 output than OME-Zarr output.
- Most likely interpretation from prior review:
  - HDF5 export in `mesospim_utils/bigstitcher_macro_templates.py` is probably the highest-memory stage.
  - Fiji/BigStitcher receives most allocated memory as Java heap via `bigstitcher.ram_fraction`.
  - JVM off-heap/native overhead plus fusion/export buffers can still exceed the SLURM cgroup limit.
  - Higher CPU counts may increase working-memory pressure further.

## Relevant Files For The OOM Issue

- `mesospim_utils/automated.py` - queues the BigStitcher path
- `mesospim_utils/string_templates.py` - Fiji `--mem` sizing
- `mesospim_utils/bigstitcher_macro_templates.py` - Stage 3 fusion/export, especially HDF5 export
- `mesospim_utils/config/main.yaml` - active runtime knobs on the local machine
- `mesospim_utils/config/example.yaml` - portable reference for any config changes that should be documented

## Config Changes Already Made During OOM Debugging

- Reduced `bigstitcher.blocksize_factor_x/y/z` from `2` to `1`
- Reduced `bigstitcher.ram_fraction` from `0.95` to `0.85`

## Why Those Changes Were Chosen

- Lower block-size factors should reduce per-block fusion/export working-set size.
- Lower RAM fraction leaves more non-heap headroom for Fiji/JVM/native buffers and compression overhead.

## If OOM Persists In Future Testing

- First inspect the BigStitcher SLURM log and confirm whether the last printed stage is `Creating Fused Dataset to HDF5` or the OME-Zarr equivalent.
- If failure is at fused export, treat fusion/export as the bottleneck rather than alignment.
- Next knobs to try:
  - lower `slurm.bigstitcher.CPUS`
  - lower `bigstitcher.blocksize_x/y/z`
  - lower or disable HDF5 compression in the HDF5 fusion macro path
  - prefer OME-Zarr fusion and convert later if another format is still required
- If failure happens before fusion starts, revisit Stage 1/Stage 2 alignment and ICP settings instead.

## Validation Gaps

- The repo still has no checked-in automated test suite in regular use.
- Validation has been mostly smoke tests and code inspection.
- BigStitcher/Fiji behavior, SLURM resource behavior, and config-driven workflows still need real environment verification after changes.
- Edited Python files were checked with `python -m py_compile` successfully.
- `automated.py automated-method-slurm --help` succeeded in the configured `mesospim_utils` environment.
- `basicpy_worker.py --help` in the `basicpy-cuda` environment still hung past the local timeout during this session, so runtime validation of the new per-tile `.basicpy_tmp` path is still needed on a real dataset.
- `mesospim_utils/automated.py`, `mesospim_utils/slurm.py`, and `mesospim_utils/preprocess.py` compiled successfully after restoring BigStitcher queueing and adding preprocess completion checks.
- `python -m py_compile mesospim_utils/preprocess.py mesospim_utils/basicpy_worker.py` succeeded after the BaSiCPy fit-tile selection change.
- `python -m py_compile mesospim_utils/preprocess.py` succeeded after the per-group `.basicpy_tmp` change for concurrent multi-channel BaSiCPy runs.
- `python -m py_compile mesospim_utils/preprocess.py` succeeded after adding tile-level BaSiCPy resume checks.
- `python mesospim_utils/preprocess.py --help` succeeded after the BaSiCPy fit-tile selection change.
- `python mesospim_utils/automated.py automated-method-slurm --help` could not be re-run in this environment during this session because `typer` is not installed in the current local Python.
- `python -m py_compile mesospim_utils/automated.py mesospim_utils/slurm.py mesospim_utils/preprocess.py` succeeded after reordering the workflow to `.btf -> .ome.zarr`, then `decon -> basicpy -> gain correction`.
- `python -m py_compile mesospim_utils/rl.py` succeeded after the workflow reorder, confirming the decon worker still compiles against the updated orchestration path.
- For the IMS export change, `python3.12 -m py_compile mesospim_utils/automated.py mesospim_utils/omezarr.py mesospim_utils/imaris.py` succeeded in this environment.
- CLI help still could not be exercised here because runtime packages such as `typer` are not installed in the available Python 3.12 environment.
- `python3 mesospim_utils/automated.py automated-method-slurm --help` still fails in this shell with `ModuleNotFoundError: No module named 'typer'`, so the new `tiff` CLI option was syntax-checked but not help-validated here.
- The new BigStitcher skip logic was syntax-checked here with `python3 -m py_compile`, but still needs runtime verification against real SLURM logs and a real `_tiffstack` directory.
- The partial IMS TIFF-resume change was syntax-checked here with `PYTHONPYCACHEPREFIX=/tmp/opencode/pycache python3 -m py_compile mesospim_utils/automated.py mesospim_utils/bigstitcher.py`, but still needs runtime verification against a partially populated `_tiffstack` plus real SLURM logs.
- The 10-plane batched IMS TIFF extraction change was syntax-checked here with `PYTHONPYCACHEPREFIX=/tmp/opencode/pycache python3 -m py_compile mesospim_utils/automated.py mesospim_utils/omezarr.py`, but still needs runtime verification against real fused OME-Zarr chunking, partial `_tiffstack` reuse, and SLURM logs.
- The `submit_array()` script-collision fix was syntax-checked here with `PYTHONPYCACHEPREFIX=/tmp/opencode/pycache python3 -m py_compile mesospim_utils/slurm.py`, but still needs runtime verification by re-running a two-channel IMS TIFF extraction and confirming channel-specific arrays execute the intended commands.
- The SLURM dependency/log-format follow-up was syntax-checked here with `PYTHONPYCACHEPREFIX=/tmp/opencode/pycache python3 -m py_compile mesospim_utils/slurm.py`.
- `python3 -m py_compile mesospim_utils/metadata.py mesospim_utils/bigstitcher.py mesospim_utils/automated.py` succeeded after the `.ome.zarr` normalization, BigStitcher success-marker guard, and backup-copy follow-up.
- `PYTHONPYCACHEPREFIX=/tmp/opencode/pycache python3 -m py_compile mesospim_utils/automated.py mesospim_utils/omezarr.py mesospim_utils/bigstitcher.py` succeeded after adding `--final-file-type tiff` support to the automated BigStitcher workflow.
- `PYTHONPYCACHEPREFIX=/tmp/opencode/pycache python3 -m py_compile mesospim_utils/automated.py mesospim_utils/constants.py mesospim_utils/omezarr_to_ims.py` should pass for the new direct IMS path, and `"/h20/home/iana/.conda/envs/pyimariswriter/bin/python" mesospim_utils/omezarr_to_ims.py --help` is the narrowest environment check for the dedicated converter.
- `PYTHONPYCACHEPREFIX=/tmp/opencode/pycache python3 -m py_compile mesospim_utils/automated.py` succeeded after the direct IMS launcher was updated to export `LD_LIBRARY_PATH` from `location_pyimariswriter_environment`, and a shell-level check confirmed the derived paths resolve to `/h20/home/lab/miniconda3/envs/pyiw/lib` and `/h20/home/lab/miniconda3/envs/pyiw/lib/python3.10/site-packages/PyImarisWriter` without relying on the submitter environment.

## Open Questions

- Is the remaining memory issue specific to HDF5 export settings, or mostly driven by heap sizing plus CPU count?
- Are current ICP/channel-alignment relaxations sufficient for chromatic aberration cases, or still too aggressive?
- Should any dependency notes or missing runtime packages be documented more explicitly in packaging files later?
- When metadata eventually includes objective information, should it provide only an objective name or full numeric PSF parameters?

## Suggested Resume Path For The Next Agent

1. Read `AGENTS.md` for stable repo guidance.
2. Run `git status` to understand in-progress local changes.
3. If resuming alignment/OOM work, inspect `mesospim_utils/bigstitcher.py` and `mesospim_utils/bigstitcher_macro_templates.py` first.
4. Check active values in `mesospim_utils/config/main.yaml` before assuming behavior from `example.yaml`.
5. After finishing a work session, update this file with only the new handoff context.
