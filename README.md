# mesospim_utils

## This repository offers tools to deal with mesospim data at CBI

#### Installing:

```bash
# Clone the repo
cd ~/src
git clone https://github.com/CBI-PITT/mesospim_utils.git

# Create a virtual environment
# This assumes that you have miniconda or anaconda installed
conda create -n mesospim_utils python=3.12 -y

# Activate environment and install mesospim_utils
conda activate mesospim_utils

# If installing on linux / SLURM nodes that will do deconvolution (pulls in the optional psfmodels extra):
pip install -e "~/src/mesospim_utils[psfmodels]"

# If installing on systems that will be used only for metadata inspection or non-decon utilities:
pip install -e ~/src/mesospim_utils
```

Notes:

- `requirements.txt` contains only `-e .` - the editable install is the supported mode, because the CLI tools are run as scripts from the package tree (see below).
- `psfmodels` ships as the optional extra `[psfmodels]` in `setup.cfg`; it is only needed on nodes that run CUDA deconvolution.

Alternative with [uv](https://docs.astral.sh/uv/) (uses the repo's `uv.lock`):

```bash
cd ~/src/mesospim_utils
uv sync
uv run python -u mesospim_utils/automated.py automated-method-slurm <location_of_mesospim_acquisition_directory>
```



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
conda activate mesospim_utils

python <location_of_install>/mesospim_utils/mesospim_utils/metadata.py --help

python <location_of_install>/mesospim_utils/mesospim_utils/metadata.py <location_of_mesospim_acquisition_directory>

# The metadata json and annotated json will be saved into the acquisition directory

```



##### **<u>Automated method to process MesoSPIM data using SLURM:</u>**

Dependencies:

1) SLURM: Used to manage processing tasks
2) CUDA-capable GPU nodes: Required only when deconvolution runs
3) Fiji with BigStitcher: Used for alignment and fusion of OME-Zarr datasets
4) ImarisFileConverter_wine: (https://github.com/CBI-PITT/ImarisFileConverter_wine), only needed when `--final-file-type ims`
5) Wine: Only needed when `--final-file-type ims`

###### **Primary supported inputs for `automated-method-slurm`:**

- MesoSPIM BigTIFF tile directories (`.btf`)
- Existing BigStitcher OME-Zarr datasets referenced by `*.ome.zarr.xml`

```bash
 python <location_of_install>/mesospim_utils/mesospim_utils/automated.py --help
 python <location_of_install>/mesospim_utils/mesospim_utils/automated.py automated-method-slurm --help
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
python <location_of_install>/mesospim_utils/mesospim_utils/automated.py automated-method-slurm <location_of_mesospim_acquisition_directory>

# Deconvolution + BaSiCPy + gain correction, then stitch/fuse
python <location_of_install>/mesospim_utils/mesospim_utils/automated.py automated-method-slurm <location_of_mesospim_acquisition_directory> --basicpy --gain-correction

# Skip deconvolution entirely (metadata, stitching and fusion only)
python <location_of_install>/mesospim_utils/mesospim_utils/automated.py automated-method-slurm <location_of_mesospim_acquisition_directory> --no-decon

# Trigger deconvolution when the metadata has no refractive index
python <location_of_install>/mesospim_utils/mesospim_utils/automated.py automated-method-slurm <location_of_mesospim_acquisition_directory> --refractive-index 1.561

# Pick a specific objective profile for PSF generation
python <location_of_install>/mesospim_utils/mesospim_utils/automated.py automated-method-slurm <location_of_mesospim_acquisition_directory> --objective Nikon_A2_Plan_Apo_4x

# Re-stitch an existing OME-Zarr collection (e.g. a deconved output)
python <location_of_install>/mesospim_utils/mesospim_utils/automated.py automated-method-slurm <location_of_omezarr_collection> --file-type .ome.zarr

# Final output as Imaris .ims (requires ImarisFileConverter_wine and Wine)
python <location_of_install>/mesospim_utils/mesospim_utils/automated.py automated-method-slurm <location_of_mesospim_acquisition_directory> --final-file-type ims

# Final output as a TIFF series
python <location_of_install>/mesospim_utils/mesospim_utils/automated.py automated-method-slurm <location_of_mesospim_acquisition_directory> --final-file-type tiff

# Limit decon array concurrency to 8 tiles at a time
python <location_of_install>/mesospim_utils/mesospim_utils/automated.py automated-method-slurm <location_of_mesospim_acquisition_directory> --num-parallel 8
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

When `--final-file-type ims`, BigStitcher now fuses to a montage OME-Zarr first and then builds the final IMS directly from that fused OME-Zarr using the configured `general.location_pyimariswriter_environment` Python interpreter. TIFF extraction is only used for the `tiff` final-output path.

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

If using only the metadata module, edit the general parameters: location_module, location_environment, emission_map, emission_to_rgb, username_pattern, 

```yaml
general:
  location_module: '<location_of_install>/mesospim_utils/mesospim_utils'
  location_environment: '<python_install_location>/envs/mesospim_utils/bin/python'
  location_pyimariswriter_environment: '<python_install_location>/envs/pyimariswriter/bin/python'
  metadata_filename: 'mesospim_metadata.json'
  metadata_annotated_filename: 'mesospim_annotated_metadata.json'
  montage_name: 'auto_montage.ims'
  verbose: 1 # true/false or 0-2. 0=off, 1-2 are increasing levels of verbosity. true==1
  overide_stage_direction: true # true, don't use stage direction for stitch/resample

  # Drive mappings for linux directories in WINDOWS OS.
  # Retained for older Windows-based workflows; not used by the primary automated-method-slurm path.
  windows_mappings: #linux_path:windows_mapped_drive_letter:
    '/<path_to_mount_1>': 'i:'
    '/<path_to_mount_2>': 'z:'

  # Drive mappings for linux directories in WINE 
  # This is used only when converting a final fused output to IMS.
  wine_mappings: #linux_path:wine_drive_letter:
    '/<path_to_mount_1>': 'h:'
    '/<path_to_mount_2>': 'f:'

  emission_map:
    #Mapping common names to emission wavelengths, only used if wavelength is not explicitly stated in metadata file (e.g. 594/44)
    "gfp": 525
    "yfp": 525
    "green": 525
    "rfp": 595
    "red": 595
    "cy5": 665
    "far_red": 665

  emission_to_rgb:
    # Mapping emission range to RGB colors
    # Keys are wavelength ranges in nm
    # Values are [R,G,B]
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
  distribution: stable

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
