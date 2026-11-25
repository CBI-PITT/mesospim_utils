# mesospim_utils

## This repository offers tools to deal with mesospim data at CBI

#### Installing:

[uv](https://docs.astral.sh/uv/) manages the Python environment from the repo's `pyproject.toml` and `uv.lock`:

```bash
# Clone the repo
cd ~/src
git clone https://github.com/CBI-PITT/mesospim_utils.git

# Change into the repo and create the environment from uv.lock
cd mesospim_utils
uv sync

# If installing on linux / SLURM nodes that will do deconvolution (pulls in the optional psfmodels extra):
uv sync --extra psfmodels

# Run any tool through the managed environment
uv run python -u mesospim_utils/automated.py automated-method-slurm <location_of_mesospim_acquisition_directory>

# Optional but recommended: pre-install Fiji with BigStitcher now
# (otherwise this happens automatically on the first run that needs it)
uv run python -u mesospim_utils/fiji.py

# Optional: pre-install the BaSiCPy CUDA environment
# (needed only for --basicpy / --gain-correction)
uv sync --project basicpy_env
```

Notes:

- `uv sync` creates `.venv` in the repo root from `uv.lock`. There is no activate step: every command below runs through `uv run` from the repository root and uses that environment.
- `psfmodels` ships as the optional extra in `pyproject.toml`; it is only needed on nodes that run CUDA deconvolution.
- The BaSiCPy CUDA preprocessing environment is a separate uv sub-project at `basicpy_env/` with its own `pyproject.toml` and `uv.lock`; create it with `uv sync --project basicpy_env` before running with `--basicpy` or `--gain-correction`.
- The main tools run as `uv run --project <repo root> python -u <script>`; the SLURM stages build this same command from `general.location_module` in the config file, so that key must point to `<repo root>/mesospim_utils`.
- Fiji with BigStitcher is downloaded and installed automatically on the first run that needs it; the `fiji.py` command above pre-installs it at install time so the first run starts faster. The distribution follows the config key `fiji.distribution` (`stable` or `latest`); the install location defaults to `<repo root>/fiji-linux` and can be overridden with the `FIJI_ROOT` environment variable. Pass `--force` (`uv run python -u mesospim_utils/fiji.py --force`) to reinstall.
- `requirements.txt` contains only `-e .` and is retained for pip-based development installs; `uv sync` is the documented path.



##### MesoSPIM metadata module:

This module underlies the whole project by collecting and 'annotating' the MesoSPIM metadata. A JSON file is written to disk in the data acquisition directory which contains a single representation of the original metadata. An *annotated json file is also stored and sorted by channel and then by tile. The annotated json file is only for reference and is never read off of disk but it is an exact representation of what mesospim_utils uses during processing. 

*Annotated data include information that can be derived from the original mesospim *_meta.txt files but are not necessarily stored explicitly in those files. Annotated data is calculated de novo on each run which allows for it to adapt to changes in mesospim_utils configuration parameters or when data has been moved to a new location.



Examples of annotated metadata are shown in lines 1 and 2 below: 

<u>Original Metadata:</u>

1) "Metadata for file": KC34_overview_Mag2x_Tile0_Ch488_Sh0_Rot0.btf
2) "ETL CFG File": ETL-parametersp_photometrics BE_RI_1.561_ BSI.csv

<u>Annotated Fields:</u>

tile_number = 0  # calculated from "Metadata for file" {Tile0_}

channel = 488 # calculated from "Metadata for file" {\_Ch488_}

refractive_index = 1.561 # calculated from "ETL CFG File" {\_RI_1.561_ }



###### Using the metadata module:

```python
from mesospim_utils.metadata import collect_all_metadata, get_first_entry

data_acquisition_directory = '/dir/where/data/were/acquired'
metadata_dict_stored_by_channel = collect_all_metadata(data_acquisition_directory)

# Show the channel names
channels = list(metadata_dict_stored_by_channel.keys())
print(channels)

# How many tiles in a channel?
num_tiles = len(metadata_dict_stored_by_channel[channels[0]])
print(num_tiles)

# Let's get a sample metadata entry (channel 0, tile 0):
first_metadata_entry = get_first_entry(metadata_dict_stored_by_channel)
first_metadata_entry.keys()

'''
## Each annotated key will be present for every tile entry.
## Annotation keys added by the module include:
"tile_number":          # From file name _Tile{#}_
"channel":              # From file name _Ch{#}_
"emission_wavelength"   # From CFG/Filter: nm, common name mapped according to config
"rgb_representation"    # emission_wavelength mapped to RGB based on config
"grid_size"             # Y,X dimensions of the imaging grid
"grid_location"         # The specific Y,X grid coordinate of this tile_number
"stage_direction"       # Does Y,X position increse/decrease (1/-1) from tile-to-tile
"overlap"               # Tile overlap determined by stage coordinates of multipe files
"resolution"            # (z,y,x): xy = "Pixelsize in um", z = "z_stepsize"
"tile_shape"            # (z,y,x): z="z_planes", y="y_pixels", x="x_pixels" 
"tile_size_um"          # resolution * tile_shape
"file_name"             # Name only of the image file
"file_path"             # Full path of the image file in the current location
"refractive_index"      # From "ETL CFG File" pattern "_RI_{float}_", None if missing
"sheet"                 # From "Shutter", left/right
“username”              # Derived from file_path by regex pattern defined in config
“affine_voxel”          # Affine transform with units in voxels
“affine_microns”        # Affine transform with units in microns
'''

# Iterate through each metadata entry:
for ch in metadata_dict_stored_by_channel:
    for entry in metadata_dict_stored_by_channel.get(ch):
        tile_num = entry.get('tile_number')
        channel = entry.get('channel')
        loc = entry.get('grid_location')
        sheet = entry.get('sheet')
        
        message = f'Tile #{tile_num} from channel {channel} is located at position {(loc.x, loc.y)} and was imaged using the {sheet} light sheet'
        
        print(message)



```

###### There is a command line tool that produces the metadata json:

```bash
# From the repository root
cd <location_of_install>/mesospim_utils

uv run python -u mesospim_utils/metadata.py --help

uv run python -u mesospim_utils/metadata.py <location_of_mesospim_acquisition_directory>

# The metadata json and annotated json will be saved into the acquisition directory
```



##### **<u>Automated method to process MesoSPIM data using SLURM:</u>**

Dependencies:

1) SLURM: Used to manage processing tasks
2) CUDA-capable GPU nodes: Required only when deconvolution runs
3) Fiji with BigStitcher: Used for alignment and fusion of OME-Zarr datasets - downloaded and installed automatically on the first run if not present (see `mesospim_utils/fiji.py`)
4) Linux x86-64 for `.ims` output: the PyImarisWriter runtime is included under `mesospim_utils/_vendor/PyImarisWriter/`. The default native library targets glibc 2.17+ and `GLIBCXX_3.4.30` (compatible with the cluster's glibc 2.35 / GCC 11 compute nodes); the newer login-node build is retained alongside it. Build instructions and licenses are in that directory. The converter uses the project's uv Python and reads both Zarr v2 and v3.

Wine and ImarisFileConverter_wine are no longer used by `automated-method-slurm`: the `.ims` conversion reads the fused OME-Zarr directly through PyImarisWriter. They are retained only for legacy workflows (`imaris.py`, `resample_ims.py`).

###### **Primary supported inputs for `automated-method-slurm`:**

- MesoSPIM BigTIFF tile directories (`.btf`)
- Existing BigStitcher OME-Zarr datasets referenced by `*.ome.zarr.xml`

All commands below run from the repository root (`cd <location_of_install>/mesospim_utils`) through `uv run`:

```bash
uv run python -u mesospim_utils/automated.py --help
uv run python -u mesospim_utils/automated.py automated-method-slurm --help
```

###### **Options for `automated-method-slurm`:**

The only required argument is the directory location of the MesoSPIM data. Every other option is resolved from the metadata or the config file when not passed.

| Option | Default | Description |
|---|---|---|
| `DIR_LOC` (positional) | required | Directory of the MesoSPIM dataset (acquisition directory for `.btf`, or the directory holding an OME-Zarr collection) |
| `--file-type` | auto-detect | Input type: `.ome.zarr` or `.btf`; auto-detected from the directory contents when omitted |
| `--final-file-type` | `omezarr` | Final output format: `omezarr`, `hdf5`, `ims`, `tiff` |
| `--decon` / `--no-decon` | on | Run deconvolution. Proceeds only when a refractive index is available from metadata or `--refractive-index`; skipped entirely with `--no-decon` |
| `--objective` | from metadata/config | Name of the objective profile (from `decon.objectives` in the config) used for PSF parameters. See objective precedence below |
| `--refractive-index` | from metadata | Sample refractive index; a manually supplied value overrides the metadata |
| `--iterations` | `20` | Deconvolution iterations |
| `--frames-per-chunk` | auto | How many z-planes are deconvolved at once; best left automatic |
| `--num-parallel` | config `slurm.decon.PARALLEL_JOBS` | How many tiles are deconvolved in parallel in the SLURM array |
| `--basicpy` / `--no-basicpy` | off | Run BaSiCPy flat-field correction after deconvolution and before gain correction |
| `--gain-correction` / `--no-gain-correction` | off | Run gain correction after deconvolution and BaSiCPy |
| `--ims-resolution-level` | `0` | OME-Zarr multiscale level converted when `--final-file-type ims` or `tiff` |
| `--supernice` | off | Submit all downstream SLURM jobs with an elevated nice value |

Boolean options are enabled by passing the flag (e.g. `--basicpy`) and explicitly disabled with the `--no-` prefix (e.g. `--no-decon`).

###### **Examples:**

```bash
# Deconvolve + stitch + fuse to OME-Zarr (the default; decon runs when a refractive index is in the metadata)
uv run python -u mesospim_utils/automated.py automated-method-slurm <location_of_mesospim_acquisition_directory>

# Deconvolution + BaSiCPy + gain correction, then stitch/fuse
uv run python -u mesospim_utils/automated.py automated-method-slurm <location_of_mesospim_acquisition_directory> --basicpy --gain-correction

# Skip deconvolution entirely (metadata, stitching and fusion only)
uv run python -u mesospim_utils/automated.py automated-method-slurm <location_of_mesospim_acquisition_directory> --no-decon

# Trigger deconvolution when the metadata has no refractive index
uv run python -u mesospim_utils/automated.py automated-method-slurm <location_of_mesospim_acquisition_directory> --refractive-index 1.561

# Pick a specific objective profile for PSF generation
uv run python -u mesospim_utils/automated.py automated-method-slurm <location_of_mesospim_acquisition_directory> --objective Nikon_A2_Plan_Apo_4x

# Re-stitch an existing OME-Zarr collection (e.g. a deconved output)
uv run python -u mesospim_utils/automated.py automated-method-slurm <location_of_omezarr_collection> --file-type .ome.zarr

# Final output as Imaris .ims (converted directly from the fused OME-Zarr via the bundled PyImarisWriter; no Wine)
uv run python -u mesospim_utils/automated.py automated-method-slurm <location_of_mesospim_acquisition_directory> --final-file-type ims

# Final output as a TIFF series
uv run python -u mesospim_utils/automated.py automated-method-slurm <location_of_mesospim_acquisition_directory> --final-file-type tiff

# Limit decon array concurrency to 8 tiles at a time
uv run python -u mesospim_utils/automated.py automated-method-slurm <location_of_mesospim_acquisition_directory> --num-parallel 8
```

###### **Outputs (relative to `DIR_LOC`):**

- `decon/<collection>.ome.zarr/` - deconvolved tile collection (when deconvolution runs)
- `decon/<collection>_montage.ome.zarr/` - fused output (default `--final-file-type omezarr`); `tiff` and `ims` outputs are derived from this fused OME-Zarr
- `logs_mesospim_utils/` - SLURM job logs; `BIGSTITCHER_SUCCESS: <montage name>` marks a completed alignment/fusion

Re-running the chain on the same dataset skips work that is already complete: tiles with existing deconvolution or BaSiCPy outputs are skipped, and the BigStitcher XML/alignment stages run on the current collection.

###### **Processing workflow:**

```bash
# Processing workflow:
# 1) Metadata collection: mesospim metadata json files are generated or refreshed in the acquisition directory.
# 2) If the input is `.btf`, tiles are first converted to OME-Zarr before any downstream processing.
# 3) Optional deconvolution: runs when `--decon` is enabled and a refractive index is available from metadata or `--refractive-index`.
# 4) Optional preprocessing: if `--basicpy` is enabled, it runs on the current OME-Zarr tiles and rebuilds multiscales.
# 5) Optional preprocessing: if `--gain-correction` is enabled, it runs after BaSiCPy on the current OME-Zarr tiles and then rebuilds multiscales.
# 6) A BigStitcher XML is generated for the final OME-Zarr tile collection.
# 7) BigStitcher alignment and fusion run on SLURM using Fiji/BigStitcher.
# 8) Final output is produced as OME-Zarr, HDF5, TIFF series, or IMS depending on `--final-file-type`.
```

When `--final-file-type tiff`, BigStitcher still fuses to a montage OME-Zarr first, then the fused dataset is extracted as a TIFF series and processing stops there.

When `--final-file-type ims`, BigStitcher fuses to a montage OME-Zarr first and then builds the final IMS directly from that fused OME-Zarr. The project's uv Python reads Zarr v2 or v3 and loads the bundled Linux PyImarisWriter library. TIFF extraction is only used for the `tiff` final-output path.

If `--decon`, `--basicpy`, and `--gain-correction` are all enabled, execution order is always:

1. `--decon`
2. `--basicpy`
3. `--gain-correction`

Cluster resources for every stage are configured per stage in the `slurm:` section of the config file: `dependencies`, `omezarr`, `decon`, `basicpy`, `gain_correction`, `align`, `ims_convert`, and `bigstitcher` - each with `PARTITION`, `CPUS`, `JOB_LABEL`, `RAM_GB`, `GRES`, `PARALLEL_JOBS`, `NICE`, and `TIME_LIMIT` keys. Defaults live in `mesospim_utils/config/example.yaml`, e.g.:

- `decon`: partition `gpu`, `1300 GB` RAM, `gpu:1`, 32 parallel array jobs
- `basicpy`: partition `ai`, `512 GB` RAM, `gpu:1`, 8 parallel jobs
- `gain_correction`: partition `gpu`, `512 GB` RAM, no GPU request, 8 parallel jobs
- `bigstitcher`: partition `gpu,ai`, `1000 GB` RAM, 4 parallel jobs

If deconvolution is enabled, objective parameters will be discovered from the MesoSPIM metadata if it includes the  `[OBJECTIVE PARAMETERS]`
section. These metadata objective parameters are used by `automated-method-slurm` only when `--objective` is not passed.
If the section is present, it is authoritative for deconvolution and must be complete.

The required objective metadata fields must match the canonical decon objective fields used in
`mesospim_utils/config/example.yaml` which defines objectives(s) that may be used for 
deconvolution:

- `name`
- `numerical_aperture`
- `objective_immersion_ri_design`
- `objective_immersion_ri_actual`
- `objective_working_distance_mm`
- `coverslip_ri_design`
- `coverslip_ri_actual`
- `coverslip_thickness_actual_mm`
- `coverslip_thickness_design_mm`

Objective precedence for deconvolution is:

1. `--objective` passed to `automated-method-slurm`
2. metadata `[OBJECTIVE PARAMETERS]`
3. metadata objective name fields such as `objective`, `objective_name`, or `microscope_objective`
4. `decon.default_objective` from the config file

If `[OBJECTIVE PARAMETERS]` is present but missing any required field, `automated-method-slurm` will fail before
submitting downstream SLURM jobs.

##### 



##### <u>Configuration File:</u>

Default reference: `mesospim_utils/config/example.yaml`

Make a copy of this file to `mesospim_utils/config/main.yaml`

Edit `mesospim_utils/config/main.yaml`

At runtime, `mesospim_utils` loads `mesospim_utils/config/main.yaml` if present and otherwise falls back to
`mesospim_utils/config/example.yaml`.

SLURM parameters for each subprocess can be edited in the config file.

If using only the metadata module, edit the general parameters: location_module, emission_map, emission_to_rgb, and username_pattern.

```yaml
general:
  # Interpreter paths derive from this: main tools run as `uv run --project <repo root> python -u <script>`
  # (uv sync in the repo creates .venv); basicpy runs as `uv run --project <repo root>/basicpy_env python`
  location_module: '<location_of_install>/mesospim_utils/mesospim_utils'
  metadata_filename: 'mesospim_metadata.json'
  metadata_annotated_filename: 'mesospim_annotated_metadata.json'
  montage_name: 'auto_montage.ims'
  verbose: 1 # true/false or 0-2. 0=off, 1-2 are increasing levels of verbosity. true==1
  overide_stage_direction: true # true, don't use stage direction for stitch/resample

  # Legacy: retained for older Windows-based workflows; not used by the primary automated-method-slurm path
  windows_mappings: #linux_path:windows_mapped_drive_letter:
    '/<path_to_mount_1>': 'i:'
    '/<path_to_mount_2>': 'z:'

  # Legacy: retained for older Wine-based workflows; the current .ims conversion runs
  # directly on Linux through PyImarisWriter and does not use Wine
  wine_mappings: #linux_path:wine_drive_letter:
    '/<path_to_mount_1>': 'h:'
    '/<path_to_mount_2>': 'f:'

  emission_map:
    #Mapping common names to emission wavelengths, only used if wavelength is not explicitly stated in metadata file
    "emp": 405
    "dapi": 450
    "hoechst": 450
    "gfp": 525
    "yfp": 525
    "green": 525
    "rfp": 595
    "red": 595
    "cy5": 665
    "far_red": 665
    "cy7": 790

  emission_to_rgb:
    # Mapping emission range to RGB colors
    # Keys are wavelength ranges in nm
    # Values are (R,G,B)
    'default': [ 0.5, 0.5, 0.5 ]
    '300-479': [ 0, 0, 1 ]
    '480-540': [ 0, 1, 0 ]
    '541-625': [ 1, 0, 0 ]
    '627-730': [ 1, 0, 0.75 ]
    '731-2000': [ 0.75, 0, 1 ]

  username_pattern: '[\\/]([a-z]+-[a-z])[\\/]'
  # example: holmes-s
  # In environments where user data is processed, the system parses path names to extract appropriate user.

## EXAMPLE SLURM CONFIG 
# Configs available for dependencies, decon, align, ims_convert
slurm:
  dependencies: # Helper jobs that facilitate the workflow - very low resource.
    'PARTITION': 'compute,gpu' #multiple partitions, comma separation part1,part2
    'CPUS': 1
    'JOB_LABEL': 'ms_depends'
    'RAM_GB': 4
    'GRES': # Specified exactly as it would be in slurm eg. "gpu:1" or None
    'PARALLEL_JOBS': 1
    'NICE': 0
    'TIME_LIMIT': '0-00:02:00' # Specify a time limit for the job. This can kill jobs that get stuck but short times can also increase priority
```

The full `slurm:` section covers every stage (`dependencies`, `omezarr`, `decon`, `basicpy`, `gain_correction`, `align`, `ims_convert`, `bigstitcher`); see `mesospim_utils/config/example.yaml` for all defaults.

###### **Format, stitching, and dev options:**

These keys control what the automated method produces. The output format must be readable by the Fiji distribution used for alignment, so `omezarr.ome_version` and `fiji.distribution` should agree (see `example.yaml` for the full annotated set):

```yaml
fiji:
  # 'stable': Fiji stable archive + pinned classic BigStitcher 2.5.3 (reads OME-Zarr v0.4 / zarr v2 only)
  # 'latest': pinned Fiji-Latest archive build; BigStitcher 3.x (adds OME-Zarr v0.5 sharded / zarr v3 reading)
  distribution: latest

omezarr:
  # OME-Zarr version for the raw-tile converter. Defaults to what the configured fiji.distribution
  # can read ('latest' -> 0.5 sharded zarr v3, 'stable' -> 0.4 zarr v2); this key overrides.
  ome_version: '0.5'
  # Shard shape (z,y,x) for source tiles (0.5 only): each shard file holds this many voxels,
  # ~8x fewer files than per-chunk writing. Ignored for 0.4.
  shard_shape: [512, 512, 512]

dev:
  slurm_top_priority: false # If true, runs "scontrol top" to boost the priority of this run's pending jobs
  # Only applies to jobs from the user running the pipeline; requires SLURM to allow users to change job priority.
```

BigStitcher alignment and fused-output parameters (link-filtering thresholds `link_relative`/`link_absolute`, ICP refinement, adaptive downsampling, fused-output sharding with `enable_sharding` and the `blocksize_*` keys) are configured in the `bigstitcher:` section; see `mesospim_utils/config/example.yaml` for the annotated defaults.
