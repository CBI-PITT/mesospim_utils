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
