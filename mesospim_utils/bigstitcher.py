from typing import Any

import typer

# STD library imports
from pathlib import Path
import shutil
import json

# Installed Package imports
import zarr

# Local imports
from metadata import collect_all_metadata, get_first_entry, get_all_tile_entries
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


if __name__ == '__main__':
    app()
















def write_big_stitcher_config_file(path):
    path = ensure_path(path)
    tile_cfg_path = path / 'big_stitcher_config.txt'
    macro_path = path / 'big_stitcher_macro.ijm'
    # BigStitcher Configuration file

    # Write tile_configuration.txt
    lines = [
        "# Define the number of dimensions we are working on",
        "dim = 3",
        "",
        "# Define the image coordinates (in microns)",
        "# Format: image_path; ; ; x y z",
    ]

    metadata_by_channel = collect_all_metadata(path)
    for ch in metadata_by_channel:
        for tile in metadata_by_channel[ch]:
            path = tile.get("file_path")

            # Extract tile origin in um from affine_microns
            am = tile.get("affine_microns")
            z = float(am[0][3])
            y = float(am[1][3])
            x = float(am[2][3])
            lines.append(f"{path}; ; ; {x:.6f} {y:.6f} {z:.6f}")

            tile_cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    pixel_z_um, pixel_y_um, pixel_x_um = get_first_entry(metadata_by_channel)['resolution']
    # Create a minimal headless macro that embeds pixel sizes and runs a safe pipeline
    macro = """// BigStitcher headless macro (args parsed manually)

    // Read options passed after --run "...":
    args = getArgument();
    config_path = call("ij.Macro.getValue", args, "config_path", "");
    output_dir  = call("ij.Macro.getValue", args, "output_dir", "");

    if (config_path == "" || output_dir == "") {
      exit("Missing required args: config_path and/or output_dir");
    }

    // If your paths may contain spaces, wrap them in [] for IJ command arguments
    file_arg = "file=[" + config_path + "]";

    setBatchMode(true);

    // Import tiles via configuration (positions in microns)
    run("BigStitcher - Import Tiles from Configuration",
        file_arg + " pixel_size_x=1 pixel_size_y=1 pixel_size_z=5.0 unit=micron");

    // Detect & match interest points (SIFT)
    run("BigStitcher - Detect Interest Points (SIFT)",
        "initialSigma=1.6 steps=3 minOctaveSize=64 maxOctaveSize=1024 featureDescriptorSize=8 featureDescriptorOrientationBins=8 threshold=0.01 findSmallerScale=true");

    // Match
    run("BigStitcher - Match Interest Points",
        "ratioOfDistance=0.92 maxAllowedError=10.0 inlierRatio=0.05 model=Rigid");

    // Optimize globally
    run("BigStitcher - Optimize Globally", "regularize=false");

    // Export to BDV N5 (multiscale)
    run("BigStitcher - Export to BDV N5",
        "path=[" + output_dir + "] export_type=FUSED multires=true blending=LINEAR downsamplingFactors=\\"[1,1,1];[2,2,2];[4,4,4];[8,8,8]\\" compression=RAW");

    setBatchMode(false);
    print("BigStitcher headless run completed.");
    """

    macro_path.write_text(macro, encoding="utf-8")

    tile_cfg_path.as_posix(), macro_path.as_posix(), {"pixel_size_x_um": pixel_x_um, "pixel_size_y_um": pixel_y_um, "pixel_size_z_um": pixel_z_um}#, "tiles": len(tiles)}




def _make_fiji_install(script_path):
    # Retry creating the installer script since the previous Python session was reset.
    from pathlib import Path
    script_path = ensure_path(script_path)

    script = r"""#!/usr/bin/env bash
# install_fiji_bigstitcher.sh
# Headless installer for Fiji (ImageJ2) + BigStitcher on Linux.
# Usage:
#   bash install_fiji_bigstitcher.sh [/path/to/install/dir]
#
# If no path is provided, installs to $HOME/apps/fiji
#
# After installation, Fiji with BigStitcher can be run headlessly:
#   $INSTALL_DIR/ImageJ-linux64 --ij2 --headless --run "BigStitcher - Import Tiles from Configuration"

set -euo pipefail

RED=$'\e[31m'; GRN=$'\e[32m'; YLW=$'\e[33m'; BLU=$'\e[34m'; RST=$'\e[0m'

die() { echo "${RED}ERROR:${RST} $*" >&2; exit 1; }
log() { echo "${BLU}==>${RST} $*"; }
ok()  { echo "${GRN}✔${RST} $*"; }
warn(){ echo "${YLW}⚠${RST} $*"; }

INSTALL_DIR="${1:-$HOME/apps/fiji}"
FIJI_ZIP_URL="https://downloads.imagej.net/fiji/archive/stable/20250808-2217/fiji-stable-linux64-jdk.zip"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

# 1) Prepare install directory
log "Installing to: $INSTALL_DIR"
mkdir -p "$(dirname "$INSTALL_DIR")"

# 2) Download Fiji
DL="$WORKDIR/fiji-linux64.zip"
if command -v wget >/dev/null 2>&1; then
  log "Downloading Fiji (wget) ..."
  wget -q -O "$DL" "$FIJI_ZIP_URL" || die "wget failed to download Fiji."
elif command -v curl >/dev/null 2>&1; then
  log "Downloading Fiji (curl) ..."
  curl -fsSL -o "$DL" "$FIJI_ZIP_URL" || die "curl failed to download Fiji."
else
  die "Need wget or curl installed to download Fiji."
fi
ok "Downloaded Fiji zip"

# 3) Unpack
log "Unpacking Fiji ..."
if command -v unzip >/dev/null 2>&1; then
    unzip -"$DL" -d "$WORKDIR"
elif command -v python3 >/dev/null 2>&1; then
    python3 - "$DL" "$WORKDIR" <<'PY'
import sys, zipfile, os
zip_path, dest = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(dest)
PY
else
    echo "No extractor available (install unzip/bsdtar/7z or ensure Java/Python present)" >&2

fi

[ -d "$WORKDIR/Fiji.app" ] || die "Fiji.app not found after unzip."
rm -rf "$INSTALL_DIR"
mv "$WORKDIR/Fiji.app" "$INSTALL_DIR"
ok "Fiji unpacked at $INSTALL_DIR"

FIJI_BIN="$INSTALL_DIR/ImageJ-linux64"
chmod 755 "$FIJI_BIN"

# 4) First headless run to initialize update system
log "Initializing Fiji update system (headless) ..."
"$FIJI_BIN" --ij2 --headless --eval 'System.exit(0);' >/dev/null 2>&1 || true
ok "Initialization complete"
log "Updating Fiji ... This may take awhile ... (headless) ..."
"$FIJI_BIN" --headless --update update >/dev/null 2>&1 || true
ok "Fiji update complete"

# 5) Enable BigStitcher
log "Installing BigStitcher (headless) ..."
"$FIJI_BIN" --headless --update edit-update-site BigStitcher https://sites.imagej.net/BigStitcher/ >/dev/null 2>&1 || true
ok "BigStitcher Installed"

# 6): add other useful sites (uncomment if desired)
log "Installing BigDataProcessor (headless) ..."
"$FIJI_BIN" --headless --update edit-update-site BigDataProcessor2 https://sites.imagej.net/BigDataProcessor/ >/dev/null 2>&1 || true
ok "BigDataProcessor Installed"

# 7) Verify BigStitcher command is available
log "Verifying BigStitcher availability (headless) ..."
set +e
OUT="$("$FIJI_BIN" --ij2 --headless --run "BigStitcher - Import Tiles from Configuration" 2>&1)"
echo "$OUT"
RET=$?
set -e
if echo "$OUT" | grep -qiE "No such command|Unknown command"; then
  echo "$OUT"
  die "BigStitcher command not found. Update site may not have installed correctly."
fi
ok "BigStitcher detected"

# 8) Print usage tips
cat <<EOF

${GRN}Installation successful!${RST}

Fiji with BigStitcher is installed at:
  $INSTALL_DIR

Headless usage example:
  $INSTALL_DIR/ImageJ-linux64 --ij2 --headless \\
    --run /path/to/bigstitcher_headless_auto.ijm \\
    "config_path='/path/to/tile_configuration.txt',output_dir='/scratch/bs_out'"

# To add Fiji to your PATH:
#   echo 'export PATH=$INSTALL_DIR:\$PATH' >> ~/.bashrc
#   source ~/.bashrc

EOF
"""

    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o755)

    str(script_path)

