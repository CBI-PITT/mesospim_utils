## Purpose Of This File

- This file is for session handoff only.
- Keep stable repository guidance in `AGENTS.md`.
- Use this document for recent changes, active debugging context, validation gaps, and open questions that matter to the next agent.

## Current Branch And Working State

- Expected branch when this note was updated: `main`
- The repository may be dirty from ongoing local work; inspect `git status` before editing.
- Recent work has touched BigStitcher, metadata/config, Fiji install helpers, and packaging metadata.

## Current Focus Areas

- BigStitcher alignment and fusion behavior
- Chromatic aberration and per-channel alignment tuning
- Memory pressure during BigStitcher fusion/export on SLURM
- Objective-aware deconvolution configuration and CLI plumbing
- Keeping agent documentation split cleanly between `AGENTS.md` and this file

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

## Recent Change: IMS Export From Fused OME-Zarr

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
- CLI `--help` smoke tests could not run in this environment because required runtime packages such as `psutil` and `tifffile` are not installed here.
- Edited Python files were checked with `python -m py_compile` successfully.
- `automated.py automated-method-slurm --help` succeeded in the configured `mesospim_utils` environment.
- `preprocess.py --help` succeeded in `/h20/home/lab/miniconda3/envs/basicpy-cuda/bin/python` after removing the Typer dependency from that script.
- After the `/tmp`-staging removal, `mesospim_utils/preprocess.py` compiled successfully in `/h20/home/lab/miniconda3/envs/mesospim_utils_v0.1/bin/python` and `mesospim_utils/basicpy_worker.py` compiled successfully in `/h20/home/lab/miniconda3/envs/basicpy-cuda/bin/python`.
- `basicpy_worker.py --help` in the `basicpy-cuda` environment still hung past the local timeout during this session, so runtime validation of the new per-tile `.basicpy_tmp` path is still needed on a real dataset.
- `mesospim_utils/automated.py`, `mesospim_utils/slurm.py`, and `mesospim_utils/preprocess.py` compiled successfully after restoring BigStitcher queueing and adding preprocess completion checks.
- Direct validation against `/h20/Acquire/MesoSPIM/dutta-p/4CL94_donotdelete/060826_movedtopublic` confirmed `is_preprocess_group_complete(...) == True` for both existing BasicPy channel/filter groups and both gain-correction channel/filter groups once the check was corrected to validate tile-level multiscales rather than the collection root.
- `mesospim_utils/metadata.py` compiled successfully in `/h20/home/lab/miniconda3/envs/mesospim_utils_v0.1/bin/python` after making the `.ome.zarr` filename rewrite idempotent.
- Regenerated `/h20/Acquire/MesoSPIM/dutta-p/4CL94_donotdelete/060826_movedtopublic/basicpy/gain_correction/MI_3_Mag4x_Ch488_Ch561_BASICPY_GCORR.ome.zarr.xml` and verified there were no remaining `.ome.zarr.ome.zarr` paths.
- `mesospim_utils/automated.py` and `mesospim_utils/slurm.py` compiled successfully in `/h20/home/lab/miniconda3/envs/mesospim_utils_v0.1/bin/python` after adding decon dependency chaining and RAM-estimation fallback behavior for preprocess workflows.
- `mesospim_utils/automated.py` compiled successfully in `/h20/home/lab/miniconda3/envs/mesospim_utils_v0.1/bin/python` after adding pre-decon OME-Zarr XML generation for the decon worker XML-cloning step.
- `mesospim_utils/preprocess.py` and `mesospim_utils/basicpy_worker.py` compiled successfully in `/h20/home/lab/miniconda3/envs/mesospim_utils_v0.1/bin/python` after switching preprocess grouping to metadata-first discovery.
- Validation on `/CBI_FastStore/test_data/mesospim/omezarr/eye/Mag4x_Ch488_Ch405.ome.zarr` now reports preprocess groups `[('405', 'Dapi'), ('488', '525/50 (GFP)')]` even though the tile directory names do not include filter tokens.
- Validation on `/h20/Acquire/MesoSPIM/dutta-p/4CL94_donotdelete/060826_movedtopublic/basicpy/gain_correction/MI_3_Mag4x_Ch488_Ch561_BASICPY_GCORR.ome.zarr` still reports the expected modern groups `[('488', 'GFP'), ('561', 'RFP')]`.
- Validation on `/CBI_FastStore/test_data/mesospim/omezarr/012926_omezarr_exosomes_2/exosomes_test2_Mag16x_Ch488_Ch561.ome.zarr` now reports preprocess groups `[('488', '525/50 (GFP)'), ('561', '595/44 (RFP)')]` and correctly recognizes tile names containing `Flt525_50_(GFP)` / `Flt595_44_(RFP)`.
- `python -m py_compile mesospim_utils/preprocess.py mesospim_utils/basicpy_worker.py` succeeded after the BaSiCPy fit-tile selection change.
- `python -m py_compile mesospim_utils/preprocess.py` succeeded after the per-group `.basicpy_tmp` change for concurrent multi-channel BaSiCPy runs.
- `python -m py_compile mesospim_utils/preprocess.py` succeeded after adding tile-level BaSiCPy resume checks.
- `python mesospim_utils/preprocess.py --help` succeeded after the BaSiCPy fit-tile selection change.
- `python mesospim_utils/automated.py automated-method-slurm --help` could not be re-run in this environment during this session because `typer` is not installed in the current local Python.
- `python -m py_compile mesospim_utils/automated.py mesospim_utils/slurm.py mesospim_utils/preprocess.py` succeeded after reordering the workflow to `.btf -> .ome.zarr`, then `decon -> basicpy -> gain correction`.
- `python -m py_compile mesospim_utils/rl.py` succeeded after the workflow reorder, confirming the decon worker still compiles against the updated orchestration path.
- For the IMS export change, `python3.12 -m py_compile mesospim_utils/automated.py mesospim_utils/omezarr.py mesospim_utils/imaris.py` succeeded in this environment.
- CLI help still could not be exercised here because runtime packages such as `typer` are not installed in the available Python 3.12 environment.
- The new BigStitcher skip logic was syntax-checked here with `python3 -m py_compile`, but still needs runtime verification against real SLURM logs and a real `_tiffstack` directory.
- The partial IMS TIFF-resume change was syntax-checked here with `PYTHONPYCACHEPREFIX=/tmp/opencode/pycache python3 -m py_compile mesospim_utils/automated.py mesospim_utils/bigstitcher.py`, but still needs runtime verification against a partially populated `_tiffstack` plus real SLURM logs.
- The 10-plane batched IMS TIFF extraction change was syntax-checked here with `PYTHONPYCACHEPREFIX=/tmp/opencode/pycache python3 -m py_compile mesospim_utils/automated.py mesospim_utils/omezarr.py`, but still needs runtime verification against real fused OME-Zarr chunking, partial `_tiffstack` reuse, and SLURM logs.
- The `submit_array()` script-collision fix was syntax-checked here with `PYTHONPYCACHEPREFIX=/tmp/opencode/pycache python3 -m py_compile mesospim_utils/slurm.py`, but still needs runtime verification by re-running a two-channel IMS TIFF extraction and confirming channel-specific arrays execute the intended commands.
- The SLURM dependency/log-format follow-up was syntax-checked here with `PYTHONPYCACHEPREFIX=/tmp/opencode/pycache python3 -m py_compile mesospim_utils/slurm.py`.

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
