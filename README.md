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

# Activate environment and install zarr_stores
conda activate mesospim_utils

# If installing on linux / SLURM nodes that will do deconvolution:
pip install -e ~/src/mesospim_utils psfmodels

# If installing on windows that will be used for ImarisStitcher align/resampling:
pip install -e ~/src/mesospim_utils
```



##### MesoSPIM metadata module:

This module underlies the whole project by collecting and 'annotating' the MesoSPIM metadata. A JSON file is written to disk in the data acquisition directory which contains a single representation of the original metadata. A annotated json file is also stored which adds fields to each metadata entry based on information that can be derived from the primary metadata. The annotated json file is only for reference and is never read off of disk. *Annotated data are abstractions of information contained within each metadata file(s) and is calculated with each run.



For example information derived from original metadata lines 1, 2 below: 

1) "Metadata for file": KC34_overview_Mag2x_Tile0_Ch488_Sh0_Rot0.btf
2) "ETL CFG File": ETL-parametersp_photometrics BE_RI_1.561_ BSI.csv

tile_number = 0  # calculated from "Metadata for file" {Tile0_}

channel = 488 # calculated from "Metadata for file" {\_Ch488_}

refractive_index = 1.561 # calculated from "ETL CFG File" {\_RI_1.561_ }



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

# Let's get a sample metadata entry:
first_metadata_entry = get_first_entry(metadata_dict_stored_by_channel)
first_metadata_entry.keys()

'''
## Keys for each metadata category will be present mirroring the origional file.
## Annotation keys added by the module include:
"tile_number":			# From file name _Tile{#}_
"channel": 				# From file name _Ch{#}_
"emission_wavelength"	# From CFG/Filter: nm, common name mapped according to config
"rgb_representation"	# emission_wavelength mapped to RGB based on config
"grid_size"				# Y,X dimensions of the imaging grid
"grid_location"			# The specific Y,X grid coordinate of this tile_number
"stage_direction"		# Does Y,X position increse/decrease (1/-1) from tile-to-tile
"overlap"				# Tile overlap determined by stage coordinates of multipe files
"resolution"			# (z,y,x): xy = "Pixelsize in um", z = "z_stepsize"
"tile_shape"			# (z,y,x): z="z_planes", y="y_pixels", x="x_pixels" 
"tile_size_um"			# resolution * tile_shape
"file_name"				# Name only of the image file
"file_path"				# Full path of the image file in the current location
"refractive_index"		# From "ETL CFG File" pattern "_RI_{float}_", None if missing
"sheet"					# From "Shutter", left/right 
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

There is a command line tool that produces the metadata json:

```bash
conda activate mesospim_utils

python <location_of_install>/mesospim_utils/mesospim_utils/metadata.py --help

python <location_of_install>/mesospim_utils/mesospim_utils/metadata.py <location_of_mesospim_acquisition_directory>

# The metadata json and annotated json will be saved into the acquisition directory

```



##### **<u>Automated method to process MesoSPIM data on SLURM:</u>**

Dependencies:

1) SLURM: Used to manage processing tasks
2) CUDA Capable GPU Nodes: Used for DECON
3) ImarisFileConverter: Installed using wine (https://github.com/CBI-PITT/ImarisFileConverter_wine)
4) ImarisStitcher: Licenced and installed on a windows machine. Used primarily for resampling data into a final 3D montage.

**MesoSPIM data must be acquired using BigTIFF files (.btf).** 

```bash
 python /CBI_FastStore/cbiPythonTools/mesospim_utils/mesospim_utils/automated.py --help
```

```bash
python /CBI_FastStore/cbiPythonTools/mesospim_utils/mesospim_utils/automated.py automated-method-slurm --help

# Kick off a fully automated processing of a dataset
python /CBI_FastStore/cbiPythonTools/mesospim_utils/mesospim_utils/automated.py automated-method-slurm /CBI_FastStore/tmp/mesospim/knee

# Processing workflow:
# 1) Deconvolution: done automatically if the ETL configuration file name contains the refractive index - pattern "_RI_{float_RI}_".
# 2) IMS Conversion: BTF files are converted to .ims using imaris converter installed in wine.
# 3) Alignment: Alignment of tiles using multiscales from ims files
# 4) Stitching: 3D data is assembled in windows using ImarisStitcher.
```

##### 



##### <u>Configuration File:</u>

Default: ./config/example.yaml

Make a copy of this file to ./config/main.yaml

Edit ./config/main.yaml

SLURM parameters for each subprocess can be edited in the config file.

If using only the metadata module, edit the general parameters: location_module, location_environment, emission_map, emission_to_rgb, username_pattern, 

```yaml
general:
  location_module: '/CBI_FastStore/cbiPythonTools/mesospim_utils/mesospim_utils'
  location_environment: '/h20/home/lab/miniconda3/envs/mesospim_dev/bin/python'
  metadata_filename: 'mesospim_metadata.json'
  metadata_annotated_filename: 'mesospim_annotated_metadata.json'
  montage_name: 'auto_montage.ims'
  verbose: 1 # true/false or 0-2. 0=off, 1-2 are increasing levels of verbosity. true==1
  overide_stage_direction: true # true, don't use stage direction for stitch/resample

  # Drive mappings for linux directories in WINDOWS for ims file conversions
  windows_mappings: #linux_path:windows_mapped_drive_letter:
    '/h20': 'i:'
    '/CBI_FastStore': 'z:'

  # Drive mappings for linux directories in WINE for ims file conversions
  wine_mappings: #linux_path:wine_drive_letter:
    '/h20': 'h:'
    '/CBI_FastStore': 'f:'

  emission_map:
    #Mapping common names to emission wavelengths, only used if wavelength is not explicitly stated in metadata file
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
    # Values are (R,G,B)
    '300-479': [ 0, 0, 1 ]
    '480-540': [ 0, 1, 0 ]
    '541-625': [ 1, 0, 0 ]
    '627-730': [ 1, 0, 0.75 ]
    '731-2000': [ 0.75, 0, 1 ]

  username_pattern: '[\\/]([a-z]+-[a-z])[\\/]'
  # example: holmes-s
  # In environments where user data is processed, the system parses path names to extract appropriate user.

## EXAMPLE SLURM CONFIG 
# Config available for dependencies, decon, align, ims_convert
slurm:
  dependencies:
    'PARTITION': 'compute,gpu' #multiple partitions, comma separation part1,part2
    'CPUS': 1
    'JOB_LABEL': 'ms_depends'
    'RAM_GB': 4
    'GRES': # Specified exactly as it would be in slurm eg. "gpu:1" or None
    'PARALLEL_JOBS': 1
    'NICE': 0
    'TIME_LIMIT': '0-00:02:00' # Specify a time limit for the job. This can kill jobs that get stuck but short times can also increase priority
```
