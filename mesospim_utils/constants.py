'''
This module will handle the reading and organization of configuration
and any necessary hardcoded constants
'''

from pathlib import Path
import math, os
import yaml

# Define umask for files.  This only works if constants in imported early in other modules.
os.umask(0o006)

def read_config():

    # Define relative path
    config_path = Path(__file__).parent / 'config' / 'main.yaml'
    # Resolve it to an absolute path
    config_path = config_path.resolve()

    if not config_path.is_file():
        # Define relative path
        config_path = Path(__file__).parent / 'config' / 'example.yaml'
        # Resolve it to an absolute path
        config_path = config_path.resolve()

    # Load YAML config
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = read_config()

general = config.get('general')
slurm = config.get('slurm')

LOCATION_OF_MESOSPIM_UTILS_INSTALL = general.get('location_module')
ENV_PYTHON_LOC = general.get('location_environment')
METADATA_FILENAME = general.get('metadata_filename')
METADATA_ANNOTATED_FILENAME = general.get('metadata_annotated_filename')
MONTAGE_NAME = general.get('montage_name')

WINDOWS_MAPPINGS = general.get('windows_mappings')
WINE_MAPPINGS = general.get('wine_mappings')

EMISSION_MAP = general.get('emission_map')
EMISSION_TO_RGB = general.get('emission_to_rgb')

VERBOSE = general.get('verbose')

USERNAME_PATTERN = general.get('username_pattern')

OVERIDE_STAGE_DIRECTION = general.get('overide_stage_direction')

# Dependencies are tasks that run lightweight commands that spin off other processes
# Very little resources are required
# Used to coordinate between DECON, IMARIS conversion
SLURM_PARAMETERS_FOR_DEPENDENCIES = slurm.get('dependencies')

#######################################################################################################################
####  Alignment align_py.py ###
#######################################################################################################################
align = config.get('align')
ALIGNMENT_DIRECTORY = align.get('directory')
RESOLUTION_LEVEL_FOR_ALIGN = align.get('resolution_level_for_align')
REMOVE_OUTLIERS = align.get('remove_outliers')
OFFSET_METRIC = align.get('offset_metric')
ALIGN_ALL_OUTPUT_FILE_NAME = f'all_{align.get("align_output_file_name")}'
ALIGN_METRIC_OUTPUT_FILE_NAME = f'{OFFSET_METRIC}_{align.get("align_output_file_name")}'
CORRELATION_THRESHOLD_FOR_ALIGN = align.get('correlation_threshold_for_alignment')

SLURM_PARAMETERS_FOR_MESOSPIM_ALIGN = slurm.get('align')

#######################################################################################################################
####  Resample settings ###
#######################################################################################################################
resample = config.get('resample')
USE_SEPARATE_ALIGN_DATA_PER_SHEET = resample.get('use_separate_align_data_per_sheet')

#######################################################################################################################
####  DECON rl.py constants ###
#######################################################################################################################
decon = config.get('decon')

DECON_SCRIPT = LOCATION_OF_MESOSPIM_UTILS_INSTALL + '/' + decon.get('script')
DECON_DEFAULT_OUTPUT_DIR = decon.get('output_dir')

PSF_THRESHOLD = decon.get('psf_threshold')
MAX_VRAM = decon.get('max_vram') * decon.get('margin_vram')
VRAM_PER_VOXEL = decon.get('vram_per_voxel') # Approximation based on real data

SLURM_PARAMETERS_DECON = slurm.get('decon')

#######################################################################################################################
####  Imaris file converter constants ###
#######################################################################################################################
ims_converter = config.get('ims_converter')

WINE_INSTALL_LOC = ims_converter.get('wine_install_location')
IMARIS_CONVERTER_LOC = ims_converter.get('imaris_installer_location')
IMS_CONVERTER_COMPRESSION_LEVEL = ims_converter.get('compression_level')

SLURM_PARAMETERS_IMARIS_CONVERTER = slurm.get('ims_convert')

#######################################################################################################################
####  Imaris Stitcher AND Imaris Resample constants ###
#######################################################################################################################
imaris_stitcher = config.get('imaris_stitcher')

## Change this path for any specific installation of ImarisStitcher
PATH_TO_IMARIS_STITCHER_FOLDER = imaris_stitcher.get('path_to_imaris_stitcher_folder')
PATH_TO_IMARIS_STITCHER_TEMP_FOLDER = imaris_stitcher.get('path_to_imaris_stitcher_temp_folder')

SHARED_WINDOWS_PATH_WHERE_WIN_CLIENT_JOB_FILES_ARE_STORED = (
    imaris_stitcher.get('shared_windows_path_where_win_client_job_files_are_stored'))
SHARED_LINUX_PATH_WHERE_WIN_CLIENT_JOB_FILES_ARE_STORED = (
    imaris_stitcher.get('shared_linux_path_where_win_client_job_files_are_stored'))

CORRELATION_THRESHOLD_FOR_IMS_STITCHER = imaris_stitcher.get('correlation_threshold_for_alignment')


PERCENTAGE_OF_MACHINE_RESOURCES_TO_USE_FOR_STITCHING = (
    imaris_stitcher.get('percentage_of_machine_resources_to_use_for_stitching'))
FRACTION_OF_RAM_FOR_PROCESSING = imaris_stitcher.get('percentage_of_machine_resources_to_use_for_stitching')
NUM_CPUS_FOR_STITCH = math.floor(os.cpu_count() * FRACTION_OF_RAM_FOR_PROCESSING)
IMS_STITCHER_COMPRESSION_LEVEL = imaris_stitcher.get('compression_level')


### Format constants ###  DO NOT CHANGE
LOCATION_OF_MESOSPIM_UTILS_INSTALL = Path(LOCATION_OF_MESOSPIM_UTILS_INSTALL)
ENV_PYTHON_LOC = Path(ENV_PYTHON_LOC)
WINE_INSTALL_LOC = Path(WINE_INSTALL_LOC)
IMARIS_CONVERTER_LOC = Path(IMARIS_CONVERTER_LOC)

########################################################################################################################
####  BigStitcher constants ###
########################################################################################################################
BIGSTITCHER = config.get('bigstitcher')

FIJI_INSTALL_LOCATION = BIGSTITCHER.get('fiji_install_folder')
if FIJI_INSTALL_LOCATION is None:
    FIJI_EXECUTABLE = None
else:
    FIJI_EXECUTABLE = Path(FIJI_INSTALL_LOCATION) / "ImageJ-linux64"

DOWNSAMPLE_IN_X = BIGSTITCHER.get('downsample_in_x')
DOWNSAMPLE_IN_Y = BIGSTITCHER.get('downsample_in_y')
DOWNSAMPLE_IN_Z = BIGSTITCHER.get('downsample_in_z')
BLOCKSIZE_X = BIGSTITCHER.get('blocksize_x')
BLOCKSIZE_Y = BIGSTITCHER.get('blocksize_y')
BLOCKSIZE_Z = BIGSTITCHER.get('blocksize_z')
BLOCKSIZE_FACTOR_X = BIGSTITCHER.get('blocksize_factor_x')
BLOCKSIZE_FACTOR_Y = BIGSTITCHER.get('blocksize_factor_y')
BLOCKSIZE_FACTOR_Z = BIGSTITCHER.get('blocksize_factor_z')
SUBSAMPLING_FACTORS = BIGSTITCHER.get('subsampling_factors')

SLURM_PARAMETERS_FOR_BIGSTITCHER = slurm.get('bigstitcher')

########################################################################################################################