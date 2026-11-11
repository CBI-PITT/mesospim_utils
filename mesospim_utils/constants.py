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
REPO_ROOT = Path(LOCATION_OF_MESOSPIM_UTILS_INSTALL).parent
UV_PYTHON_PREFIX = f'uv run --project {REPO_ROOT} python'
BASICPY_PROJECT_DIR = REPO_ROOT / 'basicpy_env'
LOCATION_PYIMARISWRITER_ENV = general.get('location_pyimariswriter_environment')
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
####  DEV Options ###
#######################################################################################################################
dev = config.get('dev')
DEV_SLURM_TOP_PRIORITY = dev.get('slurm_top_priority')

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
DECON_DEFAULT_OBJECTIVE = decon.get('default_objective')
DECON_OBJECTIVES = decon.get('objectives', {})

SLURM_PARAMETERS_DECON = slurm.get('decon')
SLURM_PARAMETERS_BASICPY = slurm.get('basicpy', {
    'PARTITION': 'ai',
    'CPUS': 1,
    'JOB_LABEL': 'dev_basicpy',
    'RAM_GB': 512,
    'GRES': 'gpu:1',
    'PARALLEL_JOBS': 8,
    'NICE': 0,
    'TIME_LIMIT': None,
})
SLURM_PARAMETERS_GAIN_CORRECTION = slurm.get('gain_correction', {
    'PARTITION': 'gpu',
    'CPUS': 1,
    'JOB_LABEL': 'dev_gaincorr',
    'RAM_GB': 512,
    'GRES': None,
    'PARALLEL_JOBS': 8,
    'NICE': 0,
    'TIME_LIMIT': None,
})

#######################################################################################################################
####  OME-Zarr file converter constants ###
#######################################################################################################################
SLURM_PARAMETERS_OMEZARR = slurm.get('omezarr')

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
LOCATION_PYIMARISWRITER_ENV = Path(LOCATION_PYIMARISWRITER_ENV)
WINE_INSTALL_LOC = Path(WINE_INSTALL_LOC)
IMARIS_CONVERTER_LOC = Path(IMARIS_CONVERTER_LOC)

########################################################################################################################
####  BigStitcher constants ###
########################################################################################################################
BIGSTITCHER = config.get('bigstitcher')

FIJI = config.get('fiji', {})
# 'stable': Fiji stable archive + pinned classic BigStitcher 2.5.3 (OME-Zarr v0.4 / zarr v2 reading)
# 'latest': pinned Fiji-Latest archive build; BigStitcher 3.x + new N5 stack ship with it (adds OME-Zarr v0.5 sharded / zarr v3 reading)
FIJI_DISTRIBUTION = FIJI.get('distribution', 'stable')

# OME-Zarr raw-tile converter (omezarr.py / ome_zarr_multiscale_writer): the source
# version defaults to what the configured fiji.distribution can read ('latest' ->
# 0.5 sharded zarr v3, 'stable' -> 0.4 zarr v2); an explicit ome_version key overrides.
OMEZARR = config.get('omezarr', {})
OMEZARR_OME_VERSION = OMEZARR.get('ome_version', '0.5' if FIJI_DISTRIBUTION == 'latest' else '0.4')
# Shard shape (z,y,x) for the source tiles: each shard file holds this many voxels
# (~8x fewer files than per-chunk writing). Only applies to 0.5 / zarr v3.
OMEZARR_SHARD_SHAPE = OMEZARR.get('shard_shape')

# BigStitcher alignment knobs, mapped to the macro placeholders {link_relative},
# {link_absolute}, {icp_maximum_error}, {icp_lambda}. Defaults are the values
# previously hardcoded in the macro templates; bigstitcher.py formats relative/
# absolute with 3 decimals and lambda with 2 so the generated macros stay
# byte-identical to the pre-parameterization output.
BIGSTITCHER_LINK_RELATIVE = float(BIGSTITCHER.get('link_relative', 2.5))
BIGSTITCHER_LINK_ABSOLUTE = float(BIGSTITCHER.get('link_absolute', 3.5))
BIGSTITCHER_ICP_MAXIMUM_ERROR = int(BIGSTITCHER.get('icp_maximum_error', 5))
BIGSTITCHER_ICP_LAMBDA = float(BIGSTITCHER.get('icp_lambda', 0.10))

# Adaptive STAGE 1 downsampling from the metadata overlap: 'overlap' as a
# downsample_in_x/y/z value computes the largest lateral downsample keeping the
# overlap strip >= BIGSTITCHER_MIN_OVERLAP_STRIP_PX (10% overlap, 3000px tiles
# -> 2x/2x/1x). Numeric values keep the fixed per-axis behavior; 'automated'
# keeps the multiscale-derived behavior.
BIGSTITCHER_MIN_OVERLAP_STRIP_PX = float(BIGSTITCHER.get('min_overlap_strip_px', 150))

# DoG detection intensity gating (macro keys verified against the installed
# BigStitcher 3.0.8 / multiview-reconstruction 9.0.12 dialogs and upstream
# DifferenceOfGUI.java): 'off' sends nothing (prior behavior), 'common' sends
# use_same_min_&_max_intensity_for_all_views=true so the DoG threshold is
# comparable across views, 'fixed' sends set_minimal_and_maximal_intensity=true
# with explicit minimal/maximal values.
BIGSTITCHER_DOG_INTENSITY_MODE = BIGSTITCHER.get('dog_intensity_mode', 'off')
BIGSTITCHER_DOG_MIN_INTENSITY = float(BIGSTITCHER.get('dog_min_intensity', 0.0))
BIGSTITCHER_DOG_MAX_INTENSITY = float(BIGSTITCHER.get('dog_max_intensity', 65535.0))

# Self-calibration guard: after fusion, the stitched XML's fitted grid scale
# estimates the zoom/ETL pixel-size miscalibration; when |s - 1| exceeds
# BIGSTITCHER_SELF_CALIBRATION_THRESHOLD_PCT the pipeline warns loudly and,
# when BIGSTITCHER_SELF_CALIBRATION_CORRECTION is true, rescales the fused
# output voxel size by 1/s (true px = nominal px / s).
BIGSTITCHER_SELF_CALIBRATION_THRESHOLD_PCT = float(BIGSTITCHER.get('self_calibration_threshold_pct', 2.0))
BIGSTITCHER_SELF_CALIBRATION_CORRECTION = bool(BIGSTITCHER.get('self_calibration_correction', False))

# Sharded OME-Zarr v0.5 fused output: the N5-API export groups the compute
# blocks (blocksize * blocksize_factor per axis, e.g. 256^3 * 2 = 512^3 voxels)
# into one shard file each, cutting the per-chunk file count by ~8x. Shard size
# is a multiple of block size and defined by the compute block size factor
# (macro keys verified against the multiview-reconstruction 9.0.12 dialog:
# 'Enable_sharding' checkbox in the OME-Zarr v3 & Sharding Options section).
BIGSTITCHER_ENABLE_SHARDING = bool(BIGSTITCHER.get('enable_sharding', False))

FIJI_INSTALL_LOCATION = BIGSTITCHER.get('fiji_install_folder')
if FIJI_INSTALL_LOCATION:
    FIJI_INSTALL_LOCATION = Path(FIJI_INSTALL_LOCATION)
else:
    FIJI_INSTALL_LOCATION = Path(__file__).resolve().parent.parent / 'fiji-linux' / 'Fiji.app'
FIJI_EXECUTABLE = FIJI_INSTALL_LOCATION / "ImageJ-linux64"
if not FIJI_EXECUTABLE.exists() and (FIJI_INSTALL_LOCATION / "fiji-linux-x64").exists():
    # The Fiji-Latest distribution launches via the jaunch-based
    # fiji-linux-x64 launcher; the stable archive ships ImageJ-linux64.
    FIJI_EXECUTABLE = FIJI_INSTALL_LOCATION / "fiji-linux-x64"

DOWNSAMPLE_IN_X = str(BIGSTITCHER.get('downsample_in_x'))
DOWNSAMPLE_IN_Y = str(BIGSTITCHER.get('downsample_in_y'))
DOWNSAMPLE_IN_Z = str(BIGSTITCHER.get('downsample_in_z'))
DOWNSAMPLE_REFINEMENT = str(BIGSTITCHER.get('downsample_refinement'))
BLOCKSIZE_X = BIGSTITCHER.get('blocksize_x')
BLOCKSIZE_Y = BIGSTITCHER.get('blocksize_y')
BLOCKSIZE_Z = BIGSTITCHER.get('blocksize_z')
BLOCKSIZE_FACTOR_X = BIGSTITCHER.get('blocksize_factor_x')
BLOCKSIZE_FACTOR_Y = BIGSTITCHER.get('blocksize_factor_y')
BLOCKSIZE_FACTOR_Z = BIGSTITCHER.get('blocksize_factor_z')
SUBSAMPLING_FACTORS = BIGSTITCHER.get('subsampling_factors')
BS_RAM_FRACTION = BIGSTITCHER.get('ram_fraction')

SLURM_PARAMETERS_FOR_BIGSTITCHER = slurm.get('bigstitcher')


########################################################################################################################
