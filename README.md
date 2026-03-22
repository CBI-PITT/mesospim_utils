# mesospim_utils

## This repository offers tools to deal with mesospim data at CBI

## Docker on Windows

The repository now includes a Windows-oriented Docker appliance under `docker/` for running a full single-node
SLURM-based `mesospim_utils` workflow inside Docker Desktop with the WSL2 backend.

- The base container runs without GPUs, and an optional GPU compose override is available for deconvolution hosts.
- SLURM and Wine are compiled from the pinned versions in `docker/.env`.
- The container expects stable in-container mount roots under `/data`.
- `C:` is bind-mounted directly, while network shares for `/data/z` and `/data/h` are intended to be mounted by Docker as CIFS volumes.
- The default Wine mappings are `/data/z -> z:` and `/data/share -> y:`.

Basic usage:

```bash
cd docker
mkdir -p config work disabled-share
docker compose build
docker compose up -d
docker compose exec mesospim_utils bash
```

By default the compose file mounts Windows-host-visible sources into:

- `MESO_C_SRC -> /data/c`
- `MESO_Z_SRC -> /data/z`
- `MESO_H_SRC -> /data/h`
- `MESO_CONFIG_SRC -> /data/config`
- `MESO_WORK_SRC -> /data/work`
- optional `MESO_SHARE_SRC -> /data/share`

If `/data/config/main.yaml` does not exist at startup, the container copies
`mesospim_utils/config/docker-example.yaml` into place automatically.

See `docker/README.md` for Windows path examples using both drive letters and UNC-backed shares.

For GPU-enabled deconvolution hosts, start with the GPU override file:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

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

# If installing on linux / SLURM nodes that will do deconvolution:
pip install -e ~/src/mesospim_utils psfmodels

# If installing on systems that will be used only for metadata inspection or non-decon utilities:
pip install -e ~/src/mesospim_utils
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
```

```bash
python <location_of_install>/mesospim_utils/mesospim_utils/automated.py automated-method-slurm --help

# Kick off a fully automated processing of a dataset
python <location_of_install>/mesospim_utils/mesospim_utils/automated.py automated-method-slurm <location_of_mesospim_acquisition_directory>

# Processing workflow:
# 1) Metadata collection: mesospim metadata json files are generated or refreshed in the acquisition directory.
# 2) Optional deconvolution: runs when `--decon` is enabled and a refractive index is available from metadata or `--refractive-index`.
# 3) If the input is `.btf`, tiles are converted to OME-Zarr and a BigStitcher XML is generated.
# 4) BigStitcher alignment and fusion run on SLURM using Fiji/BigStitcher.
# 5) Final output is produced as OME-Zarr, HDF5, or IMS depending on `--final-file-type`.
```

If deconvolution is enabled, objective parameters will be discovered from the MesoSPIM metadata if it includes the  `[OBJECTIVE PARAMETERS]`
section. These metadata objective parameters are used by `automated-method-slurm` only when `--objective` is not passed.
If the section is present, it is authoritative for deconvolution and must be complete.

The required objective metadata fields must match the canonical decon objective fields used in
`mesospim_utils/config/example.yaml` which defines objectives(s) that may be used for 
deconvolution:

- `name`
- `na`
- `objective_immersion_ri_design`
- `objective_immersion_ri_actual`
- `objective_working_distance_um`
- `coverslip_ri_design`
- `coverslip_ri_actual`
- `coverslip_thickness_actual_um`
- `coverslip_thickness_design_um`

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
