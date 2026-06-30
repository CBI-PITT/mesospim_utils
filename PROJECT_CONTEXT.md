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

## Recent Change: Pre-Deconvolution OME-Zarr Preprocessing

- Added optional `--basicpy` and `--gain-correction` to `automated-method-slurm` in `mesospim_utils/automated.py`.
- Preprocessing order is now fixed as: `basicpy -> gain correction -> decon`.
- When preprocessing is enabled for `.btf` input, the workflow now converts tiles to OME-Zarr before deconvolution so the preprocessing stages can run on tile OME-Zarr data.
- Added `mesospim_utils/preprocess.py` with low-level commands:
  - `basicpy-apply`
  - `gain-correction-apply`
- Both preprocessing commands now operate only on OME-Zarr level `0` and then regenerate multiscales, matching the decon-style output pattern more closely than the original reference scripts.
- Rows/cols now come from metadata `grid_size`; gain correction uses metadata `overlap` by default, with only the low-level command exposing an override.
- Added SLURM config plumbing for `slurm.basicpy` and `slurm.gain_correction`, plus `general.location_basicpy_environment` in `config/example.yaml`.
- Added XML regeneration before BigStitcher alignment for any new tile OME-Zarr collection produced by preprocessing or deconvolution.

## Recent Change: Remove `/tmp` Staging From BaSiCPy Apply

- `process_basicpy_group()` no longer stages corrected tiles as `.npy` files in `tempfile.TemporaryDirectory()`.
- `basicpy_worker.py` now processes one `--target-tile` per invocation and writes one temporary `.npy` into a hidden `.basicpy_tmp/` directory inside the final output collection.
- `preprocess.py` immediately reads that per-tile `.npy`, writes the final OME-Zarr tile, and deletes the temporary file.
- This was changed to avoid failures where `/tmp` filled up even though the final destination filesystem had enough free space.
- Tradeoff: the current implementation re-fits BaSiCPy for each target tile, so it should use less temporary disk but may be slower than the prior batch-worker design.

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
