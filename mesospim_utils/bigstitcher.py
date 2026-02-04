from typing import Any

import typer

# STD library imports
from pathlib import Path
import shutil
import json

# Installed Package imports
import zarr

# Local imports
from metadata import collect_all_metadata, get_first_entry, get_number_of_sheets, get_rotations
from utils import ensure_path, sort_list_of_paths_by_tile_number, make_directories, dict_to_json_file
from constants import (
    ALIGNMENT_DIRECTORY,
    CORRELATION_THRESHOLD_FOR_ALIGN,
    RESOLUTION_LEVEL_FOR_ALIGN,
    VERBOSE,
    OFFSET_METRIC,
    REMOVE_OUTLIERS,
    ALIGN_ALL_OUTPUT_FILE_NAME,
    ALIGN_METRIC_OUTPUT_FILE_NAME,
    OVERIDE_STAGE_DIRECTION
)

from constants import (
DOWNSAMPLE_IN_X,
DOWNSAMPLE_IN_Y,
DOWNSAMPLE_IN_Z,
BLOCKSIZE_X,
BLOCKSIZE_Y,
BLOCKSIZE_Z,
BLOCKSIZE_FACTOR_X,
BLOCKSIZE_FACTOR_Y,
BLOCKSIZE_FACTOR_Z,
SUBSAMPLING_FACTORS
)

# INIT typer cmdline interface
app = typer.Typer()

'''
This module has functions to orchestrate BigStitcher for mesospim data.
Assumptions are the data were acquired using the omezarr writer plugin for mesospim
'''


BIGSTITCHER_ALIGN_OMEZARR_OUT = '''
// This is a ImageJ macro file for running BigStitcher alignment
// It can be run headlessly in Fiji with BigStitcher installed

// --------------------------------------------------------------------
// STAGE 1: Tile alignment using channels=[Average Channels]
// --------------------------------------------------------------------

// select= xml file path to the ome.zarr.xml file
// downsample_in_x/y/z = downsampling factors for alignment (1 = no downsampling)

print("");
print("----------------------------------------------------------------------------------------");
print("Stage 1: Calculate pairwise shifts (Average Channels)");
print("----------------------------------------------------------------------------------------");
print("");

run("Calculate pairwise shifts ...",
"select='{0}' \
process_angle=[All angles] \
process_channel=[All channels] \
process_illumination=[All illuminations] \
process_tile=[All tiles] \
process_timepoint=[All Timepoints] \
method=[Phase Correlation] \
show_expert_grouping_options \
how_to_treat_timepoints=[treat individually] \
how_to_treat_channels=group \
how_to_treat_illuminations=group \
how_to_treat_angles=[treat individually] \
how_to_treat_tiles=compare \
channels=[Average Channels] \
downsample_in_x={1} \
downsample_in_y={2} \
downsample_in_z={3}");

// select= xml file path to the ome.zarr.xml file

print("");
print("----------------------------------------------------------------------------------------");
print("Stage 1: Optimize globally and apply shifts (tile geometry)");
print("----------------------------------------------------------------------------------------");
print("");

run("Optimize globally and apply shifts ...",
"select='{0}' \
process_angle=[All angles] \
process_channel=[All channels] \
process_illumination=[All illuminations] \
process_tile=[All tiles] \
process_timepoint=[All Timepoints] \
relative=2.500 \
absolute=3.500 \
global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad links] \
show_expert_grouping_options \
how_to_treat_timepoints=[treat individually] \
how_to_treat_channels=group \
how_to_treat_illuminations=group \
how_to_treat_angles=[treat individually] \
how_to_treat_tiles=compare \
fix_group_0-0");

// --------------------------------------------------------------------
// STAGE 2: Channel alignment with fixed tile positions
// --------------------------------------------------------------------

// select= xml file path to the ome.zarr.xml file
// downsample_in_x/y/z = downsampling factors for alignment (1 = no downsampling)

print("");
print("----------------------------------------------------------------------------------------");
print("Stage 2: Optimize per channel with fixed tile positions (Channel alignment)");
print("----------------------------------------------------------------------------------------");
print("");

run("Calculate pairwise shifts ...",
"select='{0}' \
process_angle=[All angles] \
process_channel=[All channels] \
process_illumination=[All illuminations] \
process_tile=[All tiles] \
process_timepoint=[All Timepoints] \
method=[Phase Correlation] \
show_expert_grouping_options \
how_to_treat_timepoints=[treat individually] \
how_to_treat_channels=compare \
how_to_treat_illuminations=group \
how_to_treat_angles=[treat individually] \
how_to_treat_tiles=group \
illuminations=[Average Illuminations] \
tiles=[Average Tiles] \
downsample_in_x={1} \
downsample_in_y={2} \
downsample_in_z={3}");


// select= xml file path to the ome.zarr.xml file

print("");
print("----------------------------------------------------------------------------------------");
print("Stage 2: Optimize globally and apply shifts (Channel geometry)");
print("----------------------------------------------------------------------------------------");
print("");

run("Optimize globally and apply shifts ...",
"select='{0}' \
process_angle=[All angles] \
process_channel=[All channels] \
process_illumination=[All illuminations] \
process_tile=[All tiles] \
process_timepoint=[All Timepoints] \
relative=2.500 \
absolute=3.500 \
global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad links] \
show_expert_grouping_options \
how_to_treat_timepoints=[treat individually] \
how_to_treat_channels=compare \
how_to_treat_illuminations=group \
how_to_treat_angles=[treat individually] \
how_to_treat_tiles=group \
fix_group_0-0");


// --------------------------------------------------------------------
// STAGE 3: Image Fusion into OME-Zarr
// --------------------------------------------------------------------


// select= xml file path to the ome.zarr.xml file
// zarr_dataset_path= output path for fused ome.zarr file
// block_size_x/y/z = block size for fused output
// block_size_factor_x/y/z = block size factors for fused output
// subsampling_factors= downsampling factors for multiscale output.

print("");
print("----------------------------------------------------------------------------------------");
print("Creating Fused Dataset to OME-Zarr");
print("----------------------------------------------------------------------------------------");
print("");

run("Image Fusion",
"select='{0}' \
process_angle=[All angles] \
process_channel=[All channels] \
process_illumination=[All illuminations] \
process_tile=[All tiles] \
process_timepoint=[All Timepoints] \
bounding_box=[Currently Selected Views] \
downsampling=1 \
interpolation=[Linear Interpolation] \
fusion_type=[Avg, Blending] \
pixel_type=[16-bit unsigned integer] \
interest_points_for_non_rigid=[-= Disable Non-Rigid =-] \
preserve_original \
produce=[Each timepoint & channel] \
fused_image=[OME-ZARR/N5/HDF5 export using N5-API] \
define_input=[Auto-load from input data (values shown below)] \
export=OME-ZARR \
compression=Zstandard compression_level=5 \
create_multi-resolution \
store \
zarr_dataset_path='{4}' \
show_advanced_block_size_options \
block_size_x={5} \
block_size_y={6} \
block_size_z={7} \
block_size_factor_x={8} \
block_size_factor_y={9} \
block_size_factor_z={10} \
subsampling_factors=[{11}]");

// hard-exit JVM so Slurm releases resources
call("java.lang.System.exit", "0");
'''

'''
file:
To use BIGSTITCHER_ALIGN_OMEZARR_OUT
- format in order with:
    - input ome.zarr.xml path
    - downsample_in_x for alignment
    - downsample_in_y for alignment
    - downsample_in_z for alignment
    - output fused ome.zarr path
    - block_size_x for fused output
    - block_size_y for fused output
    - block_size_z for fused output
    - block_size_factor_x for processing fused output (block_size multiplier)
    - block_size_factor_y for processing fused output (block_size multiplier)
    - block_size_factor_z for processing fused output (block_size multiplier)
    - subsampling_factors string, e.g. { {1,1,1}, {2,2,2}, {4,4,4}, {8,8,8} }

BIGSTITCHER_ALIGN_OMEZARR_OUT.format(0,1,2,3,4,5,6,7,8,9,10,11,12)
'''


BIGSTITCHER_ALIGN_HDF5_OUT = '''
// This is a ImageJ macro file for running BigStitcher alignment
// It can be run headlessly in Fiji with BigStitcher installed

// --------------------------------------------------------------------
// STAGE 1: Tile alignment using channels=[Average Channels]
// --------------------------------------------------------------------

// select= xml file path to the ome.zarr.xml file
// downsample_in_x/y/z = downsampling factors for alignment (1 = no downsampling)

print("");
print("----------------------------------------------------------------------------------------");
print("Stage 1: Calculate pairwise shifts (Average Channels)");
print("----------------------------------------------------------------------------------------");
print("");

run("Calculate pairwise shifts ...",
"select='{0}' \
process_angle=[All angles] \
process_channel=[All channels] \
process_illumination=[All illuminations] \
process_tile=[All tiles] \
process_timepoint=[All Timepoints] \
method=[Phase Correlation] \
show_expert_grouping_options \
how_to_treat_timepoints=[treat individually] \
how_to_treat_channels=group \
how_to_treat_illuminations=group \
how_to_treat_angles=[treat individually] \
how_to_treat_tiles=compare \
channels=[Average Channels] \
downsample_in_x={1} \
downsample_in_y={2} \
downsample_in_z={3}");

// select= xml file path to the ome.zarr.xml file

print("");
print("----------------------------------------------------------------------------------------");
print("Stage 1: Optimize globally and apply shifts (tile geometry)");
print("----------------------------------------------------------------------------------------");
print("");

run("Optimize globally and apply shifts ...",
"select='{0}' \
process_angle=[All angles] \
process_channel=[All channels] \
process_illumination=[All illuminations] \
process_tile=[All tiles] \
process_timepoint=[All Timepoints] \
relative=2.500 \
absolute=3.500 \
global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad links] \
show_expert_grouping_options \
how_to_treat_timepoints=[treat individually] \
how_to_treat_channels=group \
how_to_treat_illuminations=group \
how_to_treat_angles=[treat individually] \
how_to_treat_tiles=compare \
fix_group_0-0");

// --------------------------------------------------------------------
// STAGE 2: Channel alignment with fixed tile positions
// --------------------------------------------------------------------

// select= xml file path to the ome.zarr.xml file
// downsample_in_x/y/z = downsampling factors for alignment (1 = no downsampling)

print("");
print("----------------------------------------------------------------------------------------");
print("Stage 2: Optimize per channel with fixed tile positions (Channel alignment)");
print("----------------------------------------------------------------------------------------");
print("");

run("Calculate pairwise shifts ...",
"select='{0}' \
process_angle=[All angles] \
process_channel=[All channels] \
process_illumination=[All illuminations] \
process_tile=[All tiles] \
process_timepoint=[All Timepoints] \
method=[Phase Correlation] \
show_expert_grouping_options \
how_to_treat_timepoints=[treat individually] \
how_to_treat_channels=compare \
how_to_treat_illuminations=group \
how_to_treat_angles=[treat individually] \
how_to_treat_tiles=group \
illuminations=[Average Illuminations] \
tiles=[Average Tiles] \
downsample_in_x={1} \
downsample_in_y={2} \
downsample_in_z={3}");


// select= xml file path to the ome.zarr.xml file

print("");
print("----------------------------------------------------------------------------------------");
print("Stage 2: Optimize globally and apply shifts (Channel geometry)");
print("----------------------------------------------------------------------------------------");
print("");

run("Optimize globally and apply shifts ...",
"select='{0}' \
process_angle=[All angles] \
process_channel=[All channels] \
process_illumination=[All illuminations] \
process_tile=[All tiles] \
process_timepoint=[All Timepoints] \
relative=2.500 \
absolute=3.500 \
global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad links] \
show_expert_grouping_options \
how_to_treat_timepoints=[treat individually] \
how_to_treat_channels=compare \
how_to_treat_illuminations=group \
how_to_treat_angles=[treat individually] \
how_to_treat_tiles=group \
fix_group_0-0");


// --------------------------------------------------------------------
// STAGE 3: Image Fusion into HDF5
// --------------------------------------------------------------------


// select= xml file path to the ome.zarr.xml file
// zarr_dataset_path= output path for fused ome.zarr file
// block_size_x/y/z = block size for fused output
// block_size_factor_x/y/z = block size factors for fused output
// subsampling_factors= downsampling factors for multiscale output.

print("");
print("----------------------------------------------------------------------------------------");
print("Creating Fused Dataset to HDF5");
print("----------------------------------------------------------------------------------------");
print("");

run("Image Fusion",
"select='{0}' \
process_angle=[All angles] \
process_channel=[All channels] \
process_illumination=[All illuminations] \
process_tile=[All tiles] \
process_timepoint=[All Timepoints] \
bounding_box=[Currently Selected Views] \
downsampling=1 \
interpolation=[Linear Interpolation] \
fusion_type=[Avg, Blending] \
pixel_type=[16-bit unsigned integer] \
interest_points_for_non_rigid=[-= Disable Non-Rigid =-] \
preserve_original \
produce=[Each timepoint & channel] \
fused_image=[OME-ZARR/N5/HDF5 export using N5-API] \
define_input=[Auto-load from input data (values shown below)] \
export=HDF5 \
compression=Zstandard compression_level=5 \
create_a_bdv/bigstitcher \
hdf5_file='{4}' \
xml_output_file={4}.xml \
show_advanced_block_size_options \
block_size_x={5} \
block_size_y={6} \
block_size_z={7} \
block_size_factor_x={8} \
block_size_factor_y={9} \
block_size_factor_z={10} \
subsampling_factors=[{11}]");

// hard-exit JVM so Slurm releases resources
call("java.lang.System.exit", "0");
'''

def get_bigstitcher_omezarr_alignment_marco(
    input_omezarr_xml_path: Path,
    output_omezarr_path: Path,
    path_to_write_macro: Path=None,
    downsample_in_x: int=DOWNSAMPLE_IN_X,
    downsample_in_y: int=DOWNSAMPLE_IN_Y,
    downsample_in_z: int=DOWNSAMPLE_IN_Z,
    block_size_x: int=BLOCKSIZE_X,
    block_size_y: int=BLOCKSIZE_Y,
    block_size_z: int=BLOCKSIZE_Z,
    block_size_factor_x: int=BLOCKSIZE_FACTOR_X,
    block_size_factor_y: int=BLOCKSIZE_FACTOR_Y,
    block_size_factor_z: int=BLOCKSIZE_FACTOR_Z,
    subsampling_factors: str=SUBSAMPLING_FACTORS
):
    '''
    Generate BigStitcher macro for aligning omezarr data
    Return the macro string
    If given path_to_write_macro, also write the macro to that path
    '''

    automated_downsample = any([x.lower()=='automated' for x in [downsample_in_x, downsample_in_y, downsample_in_z]])
    automated_subsampling = subsampling_factors.lower() == 'automated'

    if automated_downsample or automated_subsampling:
        _, scale_factors_list_zyx, subsampling_str = determine_sampling_factors_for_bigstitcher(input_omezarr_xml_path)

    if automated_subsampling:
        subsampling_factors = subsampling_str

    if automated_subsampling:
        scale_for_downsample_zyx = scale_factors_list_zyx[2]
        downsample_in_x = scale_for_downsample_zyx[2]
        downsample_in_y = scale_for_downsample_zyx[1]
        downsample_in_z = scale_for_downsample_zyx[0]


    macro = BIGSTITCHER_ALIGN_OMEZARR_OUT.format(
        ensure_path(input_omezarr_xml_path).as_posix(),
        downsample_in_x,
        downsample_in_y,
        downsample_in_z,
        ensure_path(output_omezarr_path).as_posix(),
        block_size_x,
        block_size_y,
        block_size_z,
        block_size_factor_x,
        block_size_factor_y,
        block_size_factor_z,
        subsampling_factors
    )
    if path_to_write_macro:
        with open(path_to_write_macro, 'w') as f:
            f.write(macro)
    return macro

def get_bigstitcher_hdf5_alignment_marco(
    input_omezarr_xml_path: Path,
    output_omezarr_path: Path,
    path_to_write_macro: Path=None,
    downsample_in_x: int=DOWNSAMPLE_IN_X,
    downsample_in_y: int=DOWNSAMPLE_IN_Y,
    downsample_in_z: int=DOWNSAMPLE_IN_Z,
    block_size_x: int=BLOCKSIZE_X,
    block_size_y: int=BLOCKSIZE_Y,
    block_size_z: int=BLOCKSIZE_Z,
    block_size_factor_x: int=BLOCKSIZE_FACTOR_X,
    block_size_factor_y: int=BLOCKSIZE_FACTOR_Y,
    block_size_factor_z: int=BLOCKSIZE_FACTOR_Z,
    subsampling_factors: str=SUBSAMPLING_FACTORS
):
    '''
    Generate BigStitcher macro for aligning omezarr data
    Return the macro string
    If given path_to_write_macro, also write the macro to that path
    '''

    automated_downsample = any([x.lower()=='automated' for x in [downsample_in_x, downsample_in_y, downsample_in_z]])
    automated_subsampling = subsampling_factors.lower() == 'automated'

    if automated_downsample or automated_subsampling:
        _, scale_factors_list_zyx, subsampling_str = determine_sampling_factors_for_bigstitcher(input_omezarr_xml_path)

    if automated_subsampling:
        subsampling_factors = subsampling_str

    if automated_subsampling:
        scale_for_downsample_zyx = scale_factors_list_zyx[2]
        downsample_in_x = scale_for_downsample_zyx[2]
        downsample_in_y = scale_for_downsample_zyx[1]
        downsample_in_z = scale_for_downsample_zyx[0]


    macro = BIGSTITCHER_ALIGN_HDF5_OUT.format(
        ensure_path(input_omezarr_xml_path).as_posix(),
        downsample_in_x,
        downsample_in_y,
        downsample_in_z,
        ensure_path(output_omezarr_path).as_posix(),
        block_size_x,
        block_size_y,
        block_size_z,
        block_size_factor_x,
        block_size_factor_y,
        block_size_factor_z,
        subsampling_factors
    )
    if path_to_write_macro:
        with open(path_to_write_macro, 'w') as f:
            f.write(macro)
    return macro

def does_dir_contain_bigstitcher_metadata(path):
    '''
    Check if directory contains a .ome.zarr.xml file indicating BigStitcher metadata presence
    Return None if not found, else return path to the xml file
    '''
    path = ensure_path(path)
    zarr_xml_files = list(path.glob('*.ome.zarr.xml'))
    if len(zarr_xml_files) == 0:
        return None
    return zarr_xml_files[0]

@app.command()
def make_bigstitcher_slurm_dir_and_macro(path: Path, format: str='omezarr'):
    '''
    Takes path where bigstitcher metadata xml and
    makes a bigsitcher dir, backup of xml file,
    macrofile, and is the dir is used for SLURM logfiles
    '''

    # Ensure Fiji with BigStitcher is available, install if not present
    from fiji import ensure_fiji_and_bigstitcher
    ensure_fiji_and_bigstitcher()

    omezarr_xml = does_dir_contain_bigstitcher_metadata(path)
    if not omezarr_xml:
        return None
    omezarr_xml = ensure_path(omezarr_xml)
    bigstitcher_dir = path / 'bigstitcher'
    bigstitcher_dir.mkdir(parents=True, exist_ok=True)
    backup_xml = bigstitcher_dir / (omezarr_xml.name + '.backup')
    shutil.copy(omezarr_xml, backup_xml)

    macro_file = bigstitcher_dir / 'macro.ijm'

    # Writes macro file
    if format == 'omezarr':
        fused_out_dir_or_file = str(omezarr_xml).removesuffix('.ome.zarr.xml')
        fused_out_dir_or_file = fused_out_dir_or_file + '_montage.ome.zarr'
        _ = get_bigstitcher_omezarr_alignment_marco(omezarr_xml, fused_out_dir_or_file, macro_file)
    elif format == 'hdf5':
        fused_out_dir_or_file = str(omezarr_xml).removesuffix('.ome.zarr.xml')
        fused_out_dir_or_file = fused_out_dir_or_file + '_montage.h5'
        _ = get_bigstitcher_hdf5_alignment_marco(omezarr_xml, fused_out_dir_or_file, macro_file)

    return bigstitcher_dir, fused_out_dir_or_file, macro_file

def list_mesospim_ome_zarr_tile_dirs(path_to_mesospim_omezarr:Path):
    '''
    Given the path to a ome-zarr directory produce by mesospim,
    return a list of directories for each tile
    '''
    path_to_mesospim_omezarr = ensure_path(path_to_mesospim_omezarr)
    tile_dir_list = path_to_mesospim_omezarr.glob('*')
    tile_dir_list = [p for p in tile_dir_list if p.is_dir()]
    return tile_dir_list

def list_mesospim_ome_zarr_zattrs(path_to_mesospim_omezarr:Path):
    '''
    Given the path to a ome-zarr directory produce by mesospim,
    return a list of .zattrs files for each tile
    '''
    path_to_mesospim_omezarr = ensure_path(path_to_mesospim_omezarr)
    tile_dir_list = list_mesospim_ome_zarr_tile_dirs(path_to_mesospim_omezarr)
    zattrs_list = [x / '.zattrs' for x in tile_dir_list]
    zattrs_list = [x for x in zattrs_list if x.is_file()]
    return zattrs_list

@app.command()
def determine_sampling_factors_for_bigstitcher(omezarr_xml_path: Path) -> tuple[list[Any], str]:
    '''
    Determine subsampling factors string for BigStitcher from ome-zarr zattrs
    Return string in format: {{1,1,1},{2,2,2},{4,4,4},{8,8,8}}
    '''
    omezarr_xml_path = ensure_path(omezarr_xml_path)
    zarr_dir = str(omezarr_xml_path).removesuffix('.xml')
    zattrs_list = list_mesospim_ome_zarr_zattrs(zarr_dir)
    zattr_file = zattrs_list[0]
    zattr_data = json.loads(zattr_file.read_text())
    multiscales_dict = zattr_data.get('multiscales',[])[0]
    datasets_list = multiscales_dict.get('datasets',[])

    # Extract a list of scales for each multiscale
    scales_list = []
    for scale in datasets_list:
        for coord_transform in scale.get('coordinateTransformations'):
            if coord_transform.get('type') == 'scale':
                scales = coord_transform.get('scale')
                scales_list.append(scales)


    scale_factors_list = []
    for idx, _ in enumerate(scales_list):
        if idx == 0:
            scale_factors_list.append([1,1,1])
        else:
            factors = [
                round(scales_list[idx][0] / scales_list[idx-1][0]),
                round(scales_list[idx][1] / scales_list[idx-1][1]),
                round(scales_list[idx][2] / scales_list[idx-1][2])
            ]
            scale_factors_list.append(factors)

    print(f'''
        ========================================================================================
        {scale_factors_list=}
        ========================================================================================
        ''')

    for idx, _ in enumerate(scale_factors_list):
        if idx == 0:
            scale_factors_list[idx] = (scale_factors_list[0])
        else:
            scale_factors_list[idx] = [x*y for x,y in zip(scale_factors_list[idx-1], scale_factors_list[idx])]


    # Format to BigStitcher string
    subsampling_factors_str = '{'
    for factors in scale_factors_list:
        subsampling_factors_str += '{' + f'{factors[2]},{factors[1]},{factors[0]}' + '},' # XYZ order
    subsampling_factors_str = subsampling_factors_str.rstrip(',') + '}'

    print(f'''
            ========================================================================================
            {subsampling_factors_str=}
            ========================================================================================
            ''')

    return scales_list, scale_factors_list, subsampling_factors_str

@app.command()
def adjust_scale_in_bigstitcher_produced_ome_zarr(omezarr_xml_or_acquisition_path: Path,
                                                  omezarr_produced_by_bigstitcher_path: Path):
    '''
    Given the path to a bigstitcher xml OR a mesospim acquisition directory,
    Extract scale information from the origional .ome.zarr and transfer the
    scale to the bigstitcher-produced ome-zarr
    This is because bigstitcher does not embed scale information in ome-zarr.
    '''
    omezarr_xml_or_acquisition_path = ensure_path(omezarr_xml_or_acquisition_path)
    omezarr_produced_by_bigstitcher_path = ensure_path(omezarr_produced_by_bigstitcher_path)

    if omezarr_xml_or_acquisition_path.as_posix().endswith('.ome.zarr.xml'):
        omezarr_xml = omezarr_xml_or_acquisition_path
    else:
        omezarr_xml = does_dir_contain_bigstitcher_metadata(omezarr_xml_or_acquisition_path)
        omezarr_xml = ensure_path(omezarr_xml)

    scales_list_zyx, _, _ = determine_sampling_factors_for_bigstitcher(omezarr_xml)
    target_zattr_path = omezarr_produced_by_bigstitcher_path / '.zattrs'
    target_zattr = json.loads(target_zattr_path.read_text())
    print(f'{target_zattr=}')

    for dataset in target_zattr.get('multiscales',[])[0].get('datasets',[]):
        idx = dataset.get('path')
        idx = int(idx)
        dataset['coordinateTransformations'][0]['scale'] = [1,1] + scales_list_zyx[idx]

    print(f'{target_zattr=}')

    with open(target_zattr_path, "w") as json_file:
        json.dump(target_zattr, json_file, indent=4)


from xml.etree import ElementTree as ET
from pathlib import Path


def get_ome_zarr_directory_from_xml(xml_path):
    """
    Extract the OME-Zarr directory path from an XML file.

    Parameters
    ----------
    xml_path : str or Path
        Path to the input XML file

    Returns
    -------
    str
        The OME-Zarr directory path
    None
        If xml_path is None or no relative zarr path is found
    """
    if not xml_path:
        return None

    xml_path = ensure_path(xml_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for zarr in root.iter("zarr"):
        if zarr.attrib.get("type") == "relative":
            return xml_path.parent / zarr.text

    return None

def replace_xml_zarr_relative_group_name(
    xml_path,
    old_name,
    new_name,
    output_path=None
):
    """
    Replace the text of a <zarr type="relative"> element in an XML file.

    Parameters
    ----------
    xml_path : str or Path
        Path to the input XML file
    old_name : str
        Existing zarr filename to replace
    new_name : str
        New zarr filename
    output_path : str or Path, optional
        If provided, write to this path; otherwise overwrite input file
    """
    xml_path = Path(xml_path)
    output_path = Path(output_path) if output_path else xml_path

    tree = ET.parse(xml_path)
    root = tree.getroot()

    replaced = False

    for zarr in root.iter("zarr"):
        if zarr.attrib.get("type") == "relative" and zarr.text == old_name:
            zarr.text = new_name
            replaced = True

    if not replaced:
        raise ValueError("No matching <zarr type='relative'> element found")

    tree.write(output_path, encoding="utf-8", xml_declaration=True)

###############################################################################
##  Functions to facilitate making bigstitcher xml files from mesospim data  ##
###############################################################################

@app.command()
def mesospim_metadata_to_bigstitcher_xml(
    output_xml_path: Path
):

    import xml.etree.ElementTree as ET
    metadata_by_channel = collect_all_metadata(output_xml_path)
    first_metadata_entry = get_first_entry(metadata_by_channel)

    # Build bigstitcher xml structure
    spimdata = ET.Element("SpimData", version="0.2")

    # spimdata/BasePath:
    basepath = ET.SubElement(spimdata, 'BasePath')
    basepath.set('type', 'relative')
    basepath.text = '.'

    # spimdata/GeneratedBy:
    generated_by = ET.SubElement(spimdata, 'generatedBy')

    # spimdata/GeneratedBy/library:
    library = ET.SubElement(generated_by, 'library')
    library.set('version', '0.1.0')
    library.text = 'BigStitcher XML generated by mesospm_utils'

    # spimdata/GeneratedBy/microscope:
    microscope = ET.SubElement(generated_by, 'microscope')
    name = ET.SubElement(microscope, 'name')
    name.text = 'mesospim'
    version = ET.SubElement(microscope, 'version')
    version.text = '0.0'
    user = ET.SubElement(microscope, 'user')
    user.text = first_metadata_entry.get('username', "")

    # spimdata/GeneratedBy COMPLETE

    # spimdata/SequenceDescription:
    sequence_description = ET.SubElement(spimdata, 'SequenceDescription')

    # spimdata/SequenceDescription/ImageLoader:
    imageloader = ET.SubElement(sequence_description, 'ImageLoader')
    imageloader.set('format', 'bdv.multimg.zarr')
    imageloader.set('version', '3.0')

    # spimdata/SequenceDescription/ViewSetups:
    viewsetups = ET.SubElement(sequence_description, 'ViewSetups')

    # spimdata/SequenceDescription/ImageLoader/zarr
    zarr = ET.SubElement(imageloader, 'zarr')
    zarr.set('type', 'relative')
    zarr.text = f'{"108561_m1_PEG_Mag4x_Ch488_Ch561_Ch640.ome.zarr"}'

    # spimdata/SequenceDescription/ImageLoader/zgroups
    zgroups = ET.SubElement(imageloader, 'zgroups')

    setupid = 0
    ch_idx = 0
    for channel_wavelength, channel_metadata in metadata_by_channel.items():
        for tile_entry in channel_metadata:
            zgroup = ET.SubElement(zgroups, 'zgroup')
            zgroup.set('setup', str(setupid))
            zgroup.set('tp', '0') # Future-proofing for timepoints
            zgroup.set('path', str(tile_entry.get('file_name'))) # File path relative to basepath may need to adjust
            zgroup.set('indicies', '0 0')

            view = ET.SubElement(viewsetups, 'view') # Self closing element with no data.

            viewsetup = ET.SubElement(viewsetups, 'ViewSetup')

            id = ET.SubElement(viewsetup, 'id')
            id.text = str(setupid)

            name = ET.SubElement(viewsetup, 'name')
            # name.text = f'setup {setupid}'
            name.text = f'{tile_entry.get("file_name")}'

            size = ET.SubElement(viewsetup, 'size')
            size.text = f'{tile_entry.get("tile_shape")[2]} {tile_entry.get("tile_shape")[1]} {tile_entry.get("tile_shape")[0]}'  # X Y Z order

            voxelsize = ET.SubElement(viewsetup, 'voxelSize')

            unit = ET.SubElement(voxelsize, 'unit')
            unit.text = 'um'

            size = ET.SubElement(voxelsize, 'size')
            size.text = f'{tile_entry.get("resolution")[2]} {tile_entry.get("resolution")[1]} {tile_entry.get("resolution")[0]}'  # X Y Z order

            camera = ET.SubElement(viewsetup, 'camera')

            name = ET.SubElement(camera, 'name')
            name.text = 'default'

            exposuretime = ET.SubElement(camera, 'exposureTime')
            exposuretime.text = str( tile_entry.get("CAMERA PARAMETERS").get("camera_exposure", 0) )

            exposureunits = ET.SubElement(camera, 'exposureUnits')
            exposureunits.text = 's'

            attributes = ET.SubElement(viewsetup, 'attributes')

            illumination = ET.SubElement(attributes, 'illumination')
            shutter = tile_entry.get('sheet')
            if shutter is None:
                shutter = 0
            else:
                shutter = 0 if shutter.lower() == 'right' else 1 # 'right'->0, 'left'->1
            illumination.text = str(shutter)

            channel = ET.SubElement(attributes, 'channel')
            channel.text = str(ch_idx)

            tile = ET.SubElement(attributes, 'tile')
            tile.text = str( tile_entry.get('tile_number', 0) )

            angle = ET.SubElement(attributes, 'angle')
            angle.text = '0' # Placeholder for angle ID. Need to figure out how to get multiple angles from mesospim data
            # angle.text = str( tile_entry.get("POSITION").get("rot", 0) )

            setupid += 1

        ch_idx += 1

    # Attribute description for ViewSetups

    attributes = ET.SubElement(viewsetups, 'Attributes')
    attributes.set('name', "illumination")
    # get_number_of_sheets_used(metadata_by_channel) # returns number of sheets used (if 'left' and 'right' both used, returns 2)
    # Do we need to customize this to number of sheets used in the data?
    for idx in range(2):
        illumination = ET.SubElement(attributes, "Illumination")

        id = ET.SubElement(illumination, 'id')
        id.text = str(idx)

        name = ET.SubElement(illumination, 'name')
        name.text = 'Right' if idx == 0 else 'Left'

    attributes = ET.SubElement(viewsetups, 'Attributes')
    attributes.set('name', "channel")
    for idx, channel_wavelength in enumerate(metadata_by_channel.keys()):
        channel = ET.SubElement(attributes, "Channel")

        id = ET.SubElement(channel, 'id')
        id.text = str(idx)

        name = ET.SubElement(channel, 'name')
        name.text = f'{tuple(metadata_by_channel.keys())[idx]} nm'

        color = ET.SubElement(channel, 'color')
        rgb = metadata_by_channel[channel_wavelength][0].get('rgb_representation', [1,1,1]) # default white
        rgb = [int(x*255) for x in rgb] # convert 0-1 to 0-255 range
        color.text = f'{rgb[0]} {rgb[1]} {rgb[2]} {180}' # RGBA format A:180 slightly transparent to help with visual overlays


    attributes = ET.SubElement(viewsetups, 'Attributes')
    attributes.set('name', "angle")
    for idx, rot in enumerate(get_rotations(metadata_by_channel)):
        angle = ET.SubElement(attributes, "Angle")

        id = ET.SubElement(angle, 'id')
        id.text = str(idx)

        name = ET.SubElement(angle, 'name')
        name.text = str(rot)

    attributes = ET.SubElement(viewsetups, 'Attributes')
    attributes.set('name', "tile")
    for channel_wavelength, channel_metadata in metadata_by_channel.items():
        for tile_entry in channel_metadata:
            tile_number = str( tile_entry.get('tile_number', 0) )

            tile = ET.SubElement(attributes, "Tile")

            id = ET.SubElement(tile, 'id')
            id.text = tile_number

            name = ET.SubElement(tile, 'name')
            name.text = tile_number
        break

    # attributes = ET.SubElement(viewsetups, 'Timepoints')
    # attributes.set('name', "tile")
    timepoints = ET.SubElement(sequence_description, 'Timepoints')
    timepoints.set('type', 'range')
    first = ET.SubElement(timepoints, 'first')
    first.text = '0'
    last = ET.SubElement(timepoints, 'last')
    last.text = '0'  # Placeholder for timepoints. Mesospim data is typically single timepoint

    # spimdata/SequenceDescription COMPLETE

    # spimdata/ViewRegistrations:
    viewregistrations = ET.SubElement(spimdata, 'ViewRegistrations')

    setupid = 0
    for channel_wavelength, channel_metadata in metadata_by_channel.items():
        for tile_entry in channel_metadata:
            viewregistration = ET.SubElement(viewregistrations, 'ViewRegistration')
            viewregistration.set('timepoint', '0')
            viewregistration.set('setup', str(setupid))

            viewtransform = ET.SubElement(viewregistration, 'ViewTransform')
            viewtransform.set('type', 'affine')

            name = ET.SubElement(viewtransform, 'name')
            name.text = 'Translation to Regular Grid'

            affine = ET.SubElement(viewtransform, 'affine')

            affine_voxel = tile_entry.get("affine_voxel") # affine matrix in voxel coordinates (zyx)
            z_shift = affine_voxel[0][-1]
            y_shift = affine_voxel[1][-1]
            x_shift = affine_voxel[2][-1]
            affine_voxel_bigstitcher = f'{1.0} {0.0} {0.0} {x_shift} {0.0} {1.0} {0.0} {y_shift} {0.0} {0.0} {1.0} {z_shift}'

            affine.text = affine_voxel_bigstitcher

            viewtransform = ET.SubElement(viewregistration, 'ViewTransform')
            viewtransform.set('type', 'affine')

            name = ET.SubElement(viewtransform, 'name')
            name.text = 'calibration'

            affine = ET.SubElement(viewtransform, 'affine')

            resolution = tile_entry.get('resolution')  # z,y,x
            z_ratio_of_xy = resolution[0] / resolution[2]
            affine_calibration_bigstitcher = f'{1.0} {0.0} {0.0} {0.0} {0.0} {1.0} {0.0} {0.0} {0.0} {0.0} {z_ratio_of_xy} {0.0}'

            affine.text = affine_calibration_bigstitcher

            setupid += 1



    # Write xml to file with indentation
    tree = ET.ElementTree(spimdata)
    ET.indent(tree, space="  ")

    tree.write(
        output_xml_path,
        encoding="utf-8",
        xml_declaration=True
    )






if __name__ == '__main__':
    app()


