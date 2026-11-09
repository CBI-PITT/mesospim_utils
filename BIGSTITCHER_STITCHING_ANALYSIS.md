# BigStitcher Stitching Review: Findings and Recommendations

**Scope:** Read-only review of the Fiji BigStitcher alignment path in `mesospim_utils` - why stitched
montages can come out poorly aligned, and what to change. No code was modified for this review.

**Date:** 2026-10-23
**Reviewed by:** Agent code review + one real production run log analyzed end-to-end.

---

## 1. Sources reviewed

| Source | What it provided |
|---|---|
| `mesospim_utils/bigstitcher.py` | XML generation, sampling-factor derivation, skip/validation logic |
| `mesospim_utils/bigstitcher_macro_templates.py` | The 3-stage alignment macro (STAGE 1 pairwise+global opt, STAGE 2 DoG+ICP x3, STAGE 3 fusion) |
| `mesospim_utils/metadata.py` | `determine_overlap()`, `get_grid_location()`, `get_affine_transform()` - the initial tile grid |
| `mesospim_utils/constants.py`, `config/main.yaml` | Active runtime knobs (`downsample_*: automated`, `ram_fraction: 0.85`, block sizes) |
| `mesospim_utils/automated.py`, `string_templates.py` | SLURM queueing, Fiji `--mem` sizing, success-marker flow |
| `ome_zarr_multiscale_writer` (installed package) | Pyramid schedule (`level_factors`, `compute_xy_only_levels`) - verified the automated sampling math |
| Real acquisition: `/h20/Acquire/MesoSPIM/iyer-s/5CL21/090426/` (16 tiles, 2ch, Mag4x) | Verified sampling factors and initial-grid math on real metadata |
| Real run log: `/h20/Acquire/MesoSPIM/alder-j/072326/logs_mesospim_utils/1642738_align_fuse_bigstitcher.log` | **Direct evidence** of what the alignment actually did (BigStitcher 2.5.3, multiview-reconstruction 7.0.7, 16 tiles x 3ch = 48 view setups, stitched the decon collection) |
| BigStitcher documentation (imagej.net) | Exact semantics of link filtering, global optimization thresholds, ICP grouping |

---

## 2. What the pipeline actually runs (as built)

### 2.1 Initial tile positions (the XML)

`mesospim_metadata_to_bigstitcher_xml()` writes a translation-only "Translation to Regular Grid"
per view, computed in `get_affine_transform()`:

```
x_offset = gx_rel * nx * (1 - overlap)      # grid_location * tile_width * (1 - overlap)
y_offset = gy_rel * ny * (1 - overlap)
z_offset = (z_start - z0) / z_step          # actual stage z positions
```

where `overlap` comes from `determine_overlap()` (tiles 0 and 1 only, y direction, one value for
every step in both axes), and `nx/ny` come from `CAMERA PARAMETERS`. A per-view "calibration"
transform adds the z anisotropy ratio (z_res/x_res; 4.17 for 5um/1.2um data, 12.5 for 15um/1.2um).

Notes:
- The initial registration is **translation-only** - rotation in `affine_voxel` is discarded.
  Fine for standard axis-aligned grid scans; wrong if tiles are ever acquired with per-tile rotation
  (metadata carries `rot`, hardcoded to angle `0` in the XML).
- The scan-direction sign convention is baked into `get_grid_location()` (y-fastest, positive
  direction). `get_stage_direction()` detects direction but the affine does not use it.

### 2.2 Sampling factors (all `automated` in config)

Verified by running `determine_sampling_factors_for_bigstitcher()` against the real iyer-s dataset
(voxels z=5.0um, xy=1.2um; pyramid built with `xy_levels = floor(log2(dz/dxy)) = 2`):

- Cumulative pyramid factors (z,y,x): `[(1,1,1), (1,2,2), (1,4,4), (2,8,8), (4,16,16), (8,32,32)]`
- **Stage 1 phase correlation downsample: x=4, y=4, z=1** (pyramid level 2) - effective voxels
  4.8um xy / 5.0um z. Sensible for light-sheet data; within the BigStitcher docs' 2-4x advice.
- **Stage 2 DoG detection levels (lowest res first):**
  - Level 1: `downsample_xy=16, downsample_z=4`  -> effective voxels **19.2um / 20um**
  - Level 2: `downsample_xy=8, downsample_z=2`   -> 9.6um / 10um
  - Level 3: `downsample_xy=4, downsample_z=1`   -> 4.8um / 5um
  (Confirmed by the real log: `Correcting coordinates for downsampling ... (16.0, ..., 16.0, ..., 2.0, 0.5)`.)

### 2.3 The macro stages

- **STAGE 1:** `Calculate pairwise shifts` with `method=[Phase Correlation]`,
  `channels=[Average Channels]`, grouped channels/illuminations; then `Optimize globally and apply
  shifts` with hardcoded `relative=2.500 absolute=3.500` (BigStitcher defaults = "STRICT"),
  strategy `Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad
  links`, `fix_group_0-0`.
- **STAGE 2 (x3 levels):** `Detect Interest Points for Registration` (Difference-of-Gaussian,
  `interest_point_specification=[Comparable to Sample & small (beads)]`, 3D quadratic subpixel
  fit, `compute_on=[CPU (Java)]`) + `ICP Refinement` (Expert, `interest_points=beads`,
  `icp_maximum_error=5`, `transformation=Rigid`, `regularize_model` with Rigid `lambda=0.10`,
  link strategy `... RELAXED (5.0x / 7.0px)`, `group_channels=[Do not group]`, `group_tiles`).
- **STAGE 3:** `Image Fusion` (linear interpolation, `Avg, Blending`, 16-bit, multiscale from the
  tile-derived `subsampling_factors` string).

Important semantic consequence (verified against the BigStitcher ICP docs): `group_tiles` +
`group_channels=[Do not group]` means STAGE 2 computes **one rigid transform per (channel,
illumination) group with tile geometry locked** - i.e. it is a chromatic-aberration correction
only. It never refines tile-to-tile alignment. Tile alignment quality rests entirely on STAGE 1.
(The example GUI-recorded macro at the top of `bigstitcher_macro_templates.py` used
`Simple (all together)`, which does both - the generated macro does not.)

---

## 3. What the real run actually did (alder-j log, 16 tiles x 3 channels)

### 3.1 STAGE 1: the initial grid was wrong, and phase correlation had to rescue it

The metadata grid for alder-j is internally consistent (stage steps exactly 3240um = 2700px at the
nominal 1.2um/px; `determine_overlap()` = 0.1; affine steps = 2700px exactly). But the applied
corrections grow **linearly at ~150px per grid step in both x and y**:

```
tile 1: ( +16.7, +152.0, -14.4) px      tile 4:  (+151.4,  -11.7, -31.9) px
tile 2: ( +37.7, +302.6, -17.8) px      tile 8:  (+304.6,  -26.1, -23.9) px
tile 3: ( +58.2, +451.3, -18.5) px      tile 12: (+451.2,  -43.4,   3.7) px
corner tile: (+544.9, -121.7, -64.2) px
```

150px of a 2700px step = **5.6% per step, accumulating to ~550px (~660um) at the grid corner.**
A linear-per-step error of constant relative size is the signature of a **voxel-scale mismatch**:
the grid step in voxels (stage um / nominal pixel size) disagrees with the true voxel offsets by a
constant ratio. The most likely cause is that the nominal `Pixelsize in um: 1.2` in the acquisition
metadata (ETL zoom lookup) does not match the actual magnification (true ~1.13-1.27um depending on
sign convention), or equivalently that the recorded stage positions differ from the real scan step.
Tile cropping is ruled out (decon tiles are exactly 3000x3000x500). Whichever it is, **the pipeline
inherits it and amplifies it**: the initial grid is systematically stretched.

Consequences visible in the same log:

- Many pairwise correlations were **negative or ~0-0.3** (vs 0.88-0.99 for good pairs) because the
  compared overlap regions were misaligned by the accumulated grid error.
- One kept link had a **z-shift of 900 planes**; another -450 planes.
- First optimization round: **Avg Error 417.8px, Max Error 1617.6px**, followed by **13 consecutive
  link-dropping rounds** (`Removed link ... error=2849/465/166/...`) before converging to
  Avg 2.73px / Max 5.89px with all views connected ("Not more than one group left after first
  round ... we are already done").
- `Removed 78 non-overlapping view-pairs for computing. Total number of comparisons: 42` - pair
  selection itself runs on the wrong grid, so truly-overlapping pairs can be skipped and
  non-overlapping ones compared. This is exactly the failure mode the BigStitcher docs warn about
  ("might lead to incorrect results if the initial overlap ... of the images is too small").

**Interpretation:** STAGE 1 did converge here, but only after discarding roughly half the link
graph. That outcome is fragile: on a larger grid, or with slightly worse content, the two-round
strategy ends at a minimal spanning tree (quality capped at stage repeatability) or splits into
unconnected components patched by metadata (the stretched grid again).

### 3.2 STAGE 2: interest-point/ICP ran on almost no usable signal

- Per-view DoG detections ranged from **2 to 9823 peaks**. Because `Set minimal and maximal
  intensity` is never enabled, the preset threshold (`sigma=1.8, threshold=0.03`,
  sigma1/2 ~ 1.73/2.08 px) is applied **relative to each image's own min/max** - dark/empty tiles
  go silent. **5 views had zero interest points** (`WARNING: no interestpoints available ...`).
- The detection preset operates at effective sigma ~29um in xy at Level 1 (16x downsample) - these
  are large tissue blobs, not beads. On samples without beads/autofluorescent structures visible in
  *all* channels, cross-channel correspondences are between *different* biological structures.
- ICP correspondences per view pair ranged from **1 to ~149**; several group pairs failed the
  12-match minimum outright:
  - `[Group 0-32>->0-39 >>> Group 0-40>->0-47]: Not enough corresponding points found (only 3/12)`
  - `[Group 0-16>->0-31 >>> Group 0-32>->0-39]: No corresponding points.`
  - `[Group 0-0>->0-15 >>> Group 0-40>->0-47]: Not enough corresponding points found (only 4/12)`
  - others converged with 24-649 matches at avg error ~3.2px, max ~5px (i.e. at the
    `icp_maximum_error=5` cap - consistent with partly-noise-limited matching).
- **What it applied:** final ICP transforms were near-identity laterally (~2px) but included a
  **z-shift of ~7.7 units for the ch1 block and ~6.7 for one illumination-half of ch2**
  (0.6 planes if units are calibrated, ~7.7 planes = 115um if raw voxels - worth verifying which).
  Since `groupIllums: false`, the ch2 block split into two groups at the illumination flip and
  received *different* corrections - so the two illumination halves of the same channel are now
  z-offset from each other by construction.
- Regularizing `Rigid` with `Rigid` (lambda=0.1) is a mathematical no-op - it adds no protection.

### 3.3 STAGE 3: fusion

Fusion ran to completion (full resolution ~13 min, then pyramid levels). The fused zattrs
after `adjust_scale_in_bigstitcher_produced_ome_zarr` matched the expected tile-derived scales for
this dataset, so the scale-metadata rewrite path works here.

---

## 4. Root causes, ranked

1. **The initial grid disagrees with the measured tile offsets by a constant ~5.6% per step**
   (alder-j). Not a code bug in isolation - the code faithfully renders the metadata - but the
   pipeline trusts it completely: overlap regions for pairwise comparison, link filtering, and the
   two-round metadata fallback all inherit the stretch. `determine_overlap()` is structurally
   fragile (tiles 0-1 only, y only, one value for all steps both axes; silently returns overlap=1.0
   for single-row scans because the x-fallback can never trigger), so nothing catches it.
2. **Narrow overlap strips at 4x downsample.** 10% overlap = ~75px strip at 4x. Phase correlation
   along the narrow axis is noise-sensitive on repetitive tissue, producing the r<0.3 links.
3. **STRICT `absolute=3.5` (~4.2um calibrated)** drops links whenever placement error exceeds it;
   with the stretched grid, real links get dropped and the two-round strategy degenerates.
4. **STAGE 2 is mis-specified for this data**: per-image relative DoG threshold, sub-voxel "beads"
   at 16x downsample, rigid-per-(channel,illumination) model, no min/max intensity, and `group_tiles`
   meaning tile geometry is never refined. Its main visible effect was a per-channel-block z-shift.

---

## 5. Recommendations (ranked by expected impact)

1. **Fix the grid at the source.** Derive per-axis steps from the actual consecutive stage
   positions (unique deltas of x_pos/y_pos), or write measured stage offsets directly into the XML
   translations, instead of `grid_location * FOV * (1-overlap)` from a single tile-pair overlap.
   Also make `determine_overlap()` detect the single-row/x-neighbor case (currently returns 1.0)
   and compute x-overlap independently when both axes have >1 tile. Then re-check the nominal
   `Pixelsize in um` against the measured offsets: if adjacent-tile phase correlation consistently
   disagrees with the stage-recorded step by a fixed ratio, the magnification calibration is off -
   that single number explains the entire alder-j failure pattern.
2. **Lower STAGE 1 downsampling to 2x/2x/1x** for 10%-overlap data (doubles the strip width; the
   compute cost for ~16-48 views is trivial). Alternatively increase acquisition overlap.
3. **Loosen the global-opt threshold when the grid is suspect** - e.g. RELAXED (`absolute=7`,
   `relative=5`) - but treat this as a mitigations, not a fix; with a correct grid STRICT is fine.
4. **Set explicit min/max intensity in the DoG stage** (or exclude empty tiles from STAGE 2) so the
   per-image relative threshold stops silencing dark tiles.
5. **Re-scope STAGE 2 to what it can actually measure.** At 5um z-steps an axial chromatic shift is
   not resolvable; at 16x xy downsample "beads" are ~29um blobs. Either restrict ICP to lateral
   corrections on channels with genuinely common structure, or drop STAGE 2 and get per-tile
   lateral channel offsets from the repo's own `align.py` path (per-tile offsets at resolution
   level 3). A single rigid transform per channel cannot correct field-dependent lateral aberration
   anyway.
6. **If bead-based tile refinement is wanted**, note `group_tiles` in the ICP macro locks tile
   geometry - the opposite of tile refinement. `Simple (all together)` (as in the recorded example
   macro) refines both, at the risk of per-tile chromatic inconsistency.
7. **Monitor instead of guessing:** the log already contains everything needed -
   `grep -E "Avg Error|Removed link|Compute pairwise|no interestpoints|Not enough corresponding"`.
   Healthy signature: first-round Avg Error of a few px, few/no dropped links, thousands of peaks
   per view, ICP matches >> 100 per pair.

---

## 6. Latent reliability issues (not stitching accuracy, but reliability)

- `bigstitcher.py::is_bigstitcher_omezarr_scale_metadata_complete()` - the `return False` on scale
  mismatch is commented out (mismatches only print). Verified currently passing on real data, but a
  wrong fused pyramid would silently validate, and downstream IMS export would inherit it.
- `bigstitcher.py::adjust_scale_in_bigstitcher_produced_ome_zarr()` - will IndexError if BigStitcher
  ever emits more pyramid levels than the tile-derived scale list (`scales_list_zyx[idx]` for every
  actual dataset, no bound check).
- `bigstitcher.py::is_bigstitcher_log_successful()` - fallback treats "fusion banner + no failure
  markers + output dir exists" as success. An SLURM OOM kill often prints nothing to the log, so a
  partially-written montage can pass; the skip gate then validates a truncated zarr
  (`validate_ome_zarr_multiscale` checks level shapes, not readability, by default).
- Macro-parameter drift: headless `run(...)` silently ignores keys that do not match the installed
  BigStitcher version's dialog labels (the update site auto-installs the latest). If a dialog label
  changes between versions, a stage silently reverts to GUI defaults. Current install verified as
  BigStitcher 2.5.3.
- Minor: `downsample_refinement[0][1]` (the y component) is never used - harmless while x/y pyramid
  factors are equal, but the adjacent comment "First y" next to `[0][2]` (which is X) is wrong.

---

## 7. Quick verification checklist for the next debugging session

```bash
# How bad was the initial grid? (big corrections / many drops = stretched grid)
grep -E "Avg Error|Max Error|Removed link" <log>

# Were pairwise shifts trustworthy? (r near 0 or negative = bad comparisons)
grep "Compute pairwise" <log>

# Did STAGE 2 have any signal?
grep -cE "WARNING: no interestpoints" <log>
grep -E "Found [0-9]+ initial peaks" <log | sort -t' ' -k7 -n | head    # fewest detections
grep -E "Not enough corresponding|No corresponding" <log

# Which sampling was actually used?
grep "Correcting coordinates for downsampling" <log

# What did ICP move?
grep -A50 "Final transformation models" <log
```

Cross-check one adjacent tile pair by hand: metadata step (stage um / pixel size) vs the
correction applied to that tile. A constant ratio between them = magnification/step calibration
error (root cause #1); scattered errors = stage repeatability.

---

## 9. Cross-dataset validation (2026-10-23 survey)

A follow-up survey tested these findings across all stitching runs in
`/h20/Acquire/MesoSPIM` and `/CBI_FastStore/Acquire/MesoSPIM` (169 run logs, 162 datasets,
111 with recoverable XML + metadata). Method: per dataset, parse the stitching XML (initial
metadata translation + 'Stitching Transform'), compute measured inter-tile steps over adjacent
stage pairs, and compare against the metadata step (stage um / nominal pixel size).

### CORRECTION (2026-10-23): the first pass of this survey used the wrong XMLs

The align/fuse run saves its stitching result into the XML of the collection it stitches (usually
`<dataset>/decon/<collection>.ome.zarr.xml`); the root conversion XML stays unstitched. The first
pass compared the metadata grid against unstitched root XMLs for 105 of 111 datasets, so those
ratios were the metadata grid compared against itself (trivially 1.0000). The "95% of grids
correct" claim and the "5 outlier datasets" table below were artifacts of that bug. The corrected
re-scan locates each dataset's stitched XML (the one carrying 'Stitching Transform' entries,
referenced by the newest align log) and recomputes every ratio (script: /tmp/rescan_grid_ratios.py;
results: /tmp/rescan_results.jsonl; 112 datasets with a stitched XML + metadata). All
log-based statistics (STAGE 2, residuals, link drops) remain valid - they come from the logs.

### Corrected result: the metadata grid is ~3-7% wrong for most zoom settings

Over 112 real measurements (content step / metadata step):

- y-axis median **1.0490**, x-axis median **1.0483** (range 0.933-1.098) - nearly isotropic.
- Only **26/112 (23%)** have ratio <= 1.01; 77/112 sit in 1.03-1.08.
- The ratio is a near-constant per (nominal zoom x ETL calibration file), not per dataset:

| nominal px | ETL CFG file family | n | median ratio | implied true px |
|---|---|---|---|---|
| 1.2 (Mag4x) | photometrics variants | 34 | 1.054 | ~1.138um |
| 1.2 (Mag4x) | bsi variants | 14 | ~1.066 | ~1.125um |
| 0.6 (Mag8x) | photometrics variants | 23 | 1.031 | ~0.582um |
| 0.6 (Mag8x) | bsi variants | 14 | ~1.066 | ~0.563um |
| 0.24 (20X) | 20X variants | 15 | 0.937-0.947 | ~0.254um |
| 0.46 / 0.25 (20X) | 20X variants | 8 | ~1.00 | correct |

Evidence this is a zoom/pixel-size calibration error rather than stage error or drift:

1. **Tight constants**: within a (zoom, ETL file) group the spread is ~0.3% across different
   users, samples, and months (klimstra-w spans 08/26-10/22 at ratio 1.0530-1.0541).
2. **Sign flips with zoom**: 4x/8x/1.25um ratios > 1 (true pixel SMALLER than nominal); the 20X
   0.24um zoom shows 0.937-0.947 (true pixel LARGER than nominal); 0.46um and 0.25um are correct.
   A single stage-step error cannot flip sign with zoom; a per-zoom pixel-size lookup can.
3. **Isotropic**: y and x medians differ by 0.1% - a magnification property, not a stage axis.
4. **It depends on the ETL calibration file (camera/RI variant), which the pixel-size lookup
   ignores**: the lookup maps '4x' -> 1.2um regardless of camera, but the true magnification
   differs between the bsi and photometrics ETL configurations.

### The correlation with stitching quality is INVERTED (important)

- Datasets with ratio ~1.0 have **high** first-round Avg Error (median 336px, n=24): these are the
  poorly-stitched datasets (weak DoG signal, dropped links) where the optimization stays near the
  grid initialization instead of converging to the true offsets.
- Datasets showing the offset have **low** first-round error (median 84px, n=67): a constant grid
  error does not hurt phase correlation. BigStitcher measures content-based offsets and
  self-corrects, so good stitchings converge to the TRUE offsets (ratio ~1.05) with low residual.

Consequences:

- Stitching accuracy itself is largely unaffected by the wrong grid - the good stitchings prove
  it. The ~5.6% grid disagreement found for alder-j (section 3) is the NORM for Mag4x, not an
  anomaly.
- But every fused volume carries the NOMINAL pixel size in its metadata while its true content
  spacing is 3-7% different: physical measurements taken from stitched/fused data are
  systematically mis-scaled by that factor.
- STAGE 2 and residual statistics below remain valid (log-based).

### STAGE 2 and residual statistics (corrected)

- **STAGE 2 (ICP) is NOT a no-op**: it moved at least one setup in **90/112 (80%)** of measured
  datasets (the earlier "5/111" was an artifact of parsing unstitched XMLs). ICP translations run
  ~1-135 voxel units (up to ~160um at 1.2um/px), median per-dataset maximum ~10-25 units. The
  largest moves (delima-s/5CL15/081426: 134.6, vasudevan-d/5CL16/090426: 109.3,
  klimstra-w/5CL02/082626: 60.5) deserve case-by-case review: with the Stage-1 grid ~5% wrong, ICP
  is refining on mis-scaled initial geometry.
- **Signal starvation is real**: median minimum DoG detections per view = 6; 64/112 (57%) datasets
  contain at least one zero-detection view; median 2.5 ICP correspondence failures per dataset.
- **Final residual ceiling ~2.9px**: median final Avg Error 2.88px; worst 36.8px
  (delima-s/5CL18/091526), 31.9px (klimstra-w/5CL02/082626). At 1.2um/px this is a ~3px seam - if
  "poor stitching" means visible seams, this ceiling (phase-correlation precision on
  4x-downsampled narrow overlap strips) is the likely cause, not the grid.

### Hardware/pipeline localization: superseded

The earlier "stretched vs clean" instrument comparison is void - the "clean" group was unstitched
XMLs. The corrected data localizes the error to the zoom/ETL configuration, not to an instrument:
both microscopes and all ETL files show offsets, with the magnitude set by the zoom x ETL-file
combination. What still holds from that check: the processing pipelines (decon, BASICPY,
gain-correction) carry identical tile geometry to their raw sources, so they did not alter the
grid.

### Revised priorities (corrected)

1. **Re-measure the true pixel size per (zoom, ETL CFG file) with a calibration slide** and update
   the zoom->pixel-size lookup; or record a measured pixel size per acquisition. The current
   lookup is wrong by 3-7% for most settings and ignores the camera/RI variant.
2. **Self-calibration guard at stitch time** (section 5.1/5.3): measure the scale from
   adjacent-pair offsets; when |s-1| > ~2%, correct the voxel-size metadata of the stitched output
   (or warn loudly). This also fixes the mis-scaled fused volumes being produced now.
3. Lowering STAGE 1 downsampling to 2x/2x/1x remains the main fix for stitching residuals - the
   poor-stitching datasets (ratio ~1.0) have 300-600px first-round errors and are the real
   accuracy problem.
4. STAGE 2's no-op status across the population confirms it can be dropped for samples without
   beads.

---

## 10. References

- BigStitcher pairwise shifts / overlap warning: https://imagej.net/plugins/bigstitcher/pairwise-shift
- Global optimization (thresholds, two-round strategy, fixing views): https://imagej.net/plugins/bigstitcher/global-optimization
- ICP refinement (grouping semantics, simple vs expert): https://imagej.net/plugins/bigstitcher/icp-refinement
- Interest point detection (relative thresholds, downsampling guidance): https://imagej.net/plugins/bigstitcher/interest-points
