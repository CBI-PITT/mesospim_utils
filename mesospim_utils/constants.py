from pathlib import Path

LOCATION_OF_MESOSPIM_UTILS_INSTALL = '/CBI_FastStore/cbiPythonTools/mesospim_utils/mesospim_utils'
ENV_PYTHON_LOC = '/h20/home/lab/miniconda3/envs/mesospim_dev/bin/python'

#######################################################################################################################
####  DECON rl.py constants ###
#######################################################################################################################

DECON_SCRIPT = '/CBI_FastStore/cbiPythonTools/mesospim_utils/mesospim_utils/rl3.py'
DECON_SLURM_PARTITION = 'gpu' #multiple partitions can be specified with comma separation part1,par2
DECON_SLURM_CPUS = None  # Number of CPUs (int), None=SLURM partition Default
DECON_SLURM_JOB_LABEL = 'decon'
DECON_SLURM_RAM_MB = None # Value in gigabytes, None will be SLURM partition default
DECON_GRES = 'gpu:1' # Specific exactly as it would be in slurm eg. "gpu:1" or None
DECON_PARALLEL_JOBS = 16 # Number of jobs that will run in parallel, only applicable to array submissions

DECON_DEFAULT_OUTPUT_DIR = 'decon'
SLURM_PARAMETERS_DECON = {
    'PARTITION': 'gpu', #multiple partitions can be specified with comma separation part1,par2
    'CPUS': None,
    'JOB_LABEL': 'decon',
    'RAM_GB': None,
    'GRES': 'gpu:1', # Specific exactly as it would be in slurm eg. "gpu:1" or None
    'PARALLEL_JOBS': 16,
    'NICE': 0,
}

#######################################################################################################################
####  Imaris file converter constants ###
#######################################################################################################################

WINE_INSTALL_LOC = '/h20/home/lab/src/wine/wine64'
IMARIS_CONVERTER_LOC = '/h20/home/lab/src/ImarisFileConverter 10.2.0/ImarisConvert.exe'

# Drive mappings for linux directories in wine for ims file conversions
WINE_MAPPINGS = { #linux_path:wine_drive_letter:
'/h20':'h:',
'/CBI_FastStore':'f:',
}

IMS_CONVERTER_COMPRESSION_LEVEL = 6 #GZIP 0-9

SLURM_PARTITION = 'compute,gpu' #multiple partitions can be specified with comma separation part1,par2
SLURM_CPUS = 24 # Number of CPUs (int), None=SLURM partition Default
SLURM_JOB_LABEL = 'ims_test_conv'
SLURM_RAM_MB = 64 # Value in gigabytes, None will be SLURM partition default
SLURM_GRES = None # Specific exactly as it would be in slurm eg. "gpu:1" or None
SLURM_PARALLEL_JOBS = 8 # Number of jobs that will run in parallel, only applicable to array submissions

SLURM_PARAMETERS_IMARIS_CONVERTER = {
    'PARTITION': 'compute', #multiple partitions can be specified with comma separation part1,par2
    'CPUS': 24,
    'JOB_LABEL': 'ims_test_conv',
    'RAM_GB': 64,
    'GRES': None, # Specific exactly as it would be in slurm eg. "gpu:1" or None
    'PARALLEL_JOBS': 50,
    'NICE': 0,
}


#######################################################################################################################
####  Imaris Stitcher constants ###
#######################################################################################################################


## Change this path for any specific installation of ImarisStitcher
PATH_TO_IMARIS_STITCHER_FOLDER = r"C:\Program Files\Bitplane\ImarisStitcher 10.2.0"

SHARED_WINDOWS_PATH_WHERE_WIN_CLIENT_JOB_FILES_ARE_STORED = r"Z:\tmp\stitch_jobs"
SHARED_LINUX_PATH_WHERE_WIN_CLIENT_JOB_FILES_ARE_STORED = r"/CBI_FastStore/tmp/stitch_jobs"

METADATA_FILENAME = 'mesospim_metadata.json'

CORRELATION_THRESHOLD_FOR_ALIGNMENT = 0.85

FRACTION_OF_RAM_FOR_PROCESSING = 0.5

IMS_STITCHER_COMPRESSION_LEVEL = 6 #GZIP 0-9

EMISSION_TO_RGB = {
    #Mapping emission range to RGB colors
    # Keys are wavelength ranges in nm
    # Values are (R,G,B)
    '300-479': (0, 0, 1),
    '480-540': (0, 1, 0),
    '541-625': (1, 0, 0),
    '627-730': (1, 0, 0.75),
    '731-2000': (.75, 0, 1),
}


#######################################################################################################################
####  MesoSPIM constants ###
#######################################################################################################################


EMISSION_MAP = {
    #Mapping common names to emission wavelengths, only used if wavelength is not explicitly stated in metadata file
    "gfp": 525,
    "green": 525,
    "rfp": 595,
    "red": 595,
    "cy5": 665,
    "far_red": 665,
}



### Format constants ###  DO NOT CHANGE
LOCATION_OF_MESOSPIM_UTILS_INSTALL = Path(LOCATION_OF_MESOSPIM_UTILS_INSTALL)
ENV_PYTHON_LOC = Path(ENV_PYTHON_LOC)
WINE_INSTALL_LOC = Path(WINE_INSTALL_LOC)
IMARIS_CONVERTER_LOC = Path(IMARIS_CONVERTER_LOC)