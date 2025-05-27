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

# Deconvolution will be done automatically IF the ETL configuration file contains the pattern "_RI_{float_RI}_". If this pattern is not found, decon does not proceed.
```

##### MesoSPIM metadata module:

This module underlies the whole project by collecting and 'annotating' the MesoSPIM metadata. A JSON file is written to disk in the data acquisition directory the is a single representation of the original metadata. A annotated json file is also stored which adds fields to each metadata entry for easy use. The annotated json file is only for reference and is never read off of disk. All annotated data are abstractions of information contained within each metadata file(s). 



For example: 

1) "Metadata for file": KC34_overview_Mag2x_Tile0_Ch488_Sh0_Rot0.btf
2) "ETL CFG File": ETL-parametersp_photometrics BE_RI_1.561_ BSI.csv

tile_number = 0 {\_Tile0_}

channel = 488 {\_Ch488_}

refractive_index = 1.561 {\_RI_1.561_ }



```python
from mesospim_utils.metadata import collect_all_metadata, get_first_entry

data_acquisition_directory = '/dir/where/data/were/acquired'
metadata_dict_stored_by_channel = collect_all_metadata(data_acquisition_directory)

# Show the channel names
channels = list(metadata_dict_stored_by_channel.keys())
print(channels)

# How many files in a channel?
num_files = len(metadata_dict_stored_by_channel[channels[0]])
print(num_files)

# Let's get a sample metadata entry:
first_metadata_entry = get_first_entry(metadata_dict_stored_by_channel)
first_metadata_entry.keys()

'''
## Keys for each metadata category will be present mirroring the origional file.
## Annotation keys added by the module include:
"tile_number": 			# From file name _Tile{#}_
"channel": 				# From file name _Ch{#}_
"emission_wavelength"	# From CFG/Filter: nm, common name mapped according to config
"rgb_representation"	# emission_wavelength mapped to RGB based on config
"grid_size"				# X,Y dimensions of the imaging grid
"grid_location"			# The specific Y,X coordinate of this tile_number
"overlap"				# Tile overlap determined by stage coordinates of multipe files
"resolution"			# (z,y,z): xy = "Pixelsize in um", z = "z_stepsize"
"tile_shape"			# (z,y,x): z="z_planes", y="y_pixels", x="x_pixels" 
"tile_size_um"			# resolution * tile_shape
"file_name"				# Name only of the image file
"file_path"				# Full path of the image file in the current location
"refractive_index"		# From "ETL CFG File" pattern "_RI_{float}_"
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



##### <u>Configuration File (INCOMPLETE):</u>

Default: ./config/example.yaml

./config/main.yaml is present, it will be used.

SLURM parameters for each subprocess can be edited in the config file:

```yaml
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
