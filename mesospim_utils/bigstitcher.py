'''
This module has functions to orchestrate BigStitcher alignment for mesospim data.
Assumptions are that data were acquired using the omezarr writer plugin for mesospim
OR data have been converted to ome-zarr format using the convert_mesospim_btf_to_omezarr function from the omezarr module.
The mesospim_metadata_to_bigstitcher_xml function in this module can be used to create a bigstitcher xml file for use here.
'''

from typing import Any, Optional

import typer

# STD library imports
from pathlib import Path
import shutil
import json
import re

# Installed Package imports
import zarr

# Local imports
from metadata import collect_all_metadata, get_first_entry, get_number_of_sheets, get_rotations, modify_file_names_in_annotated_metadata
from utils import ensure_path

from constants import (
DOWNSAMPLE_IN_X,
DOWNSAMPLE_IN_Y,
DOWNSAMPLE_IN_Z,
DOWNSAMPLE_REFINEMENT,
BLOCKSIZE_X,
BLOCKSIZE_Y,
BLOCKSIZE_Z,
BLOCKSIZE_FACTOR_X,
BLOCKSIZE_FACTOR_Y,
BLOCKSIZE_FACTOR_Z,
SUBSAMPLING_FACTORS
)

from bigstitcher_macro_templates import (
    BIGSTITCHER_ALIGN_OMEZARR_OUT,
    BIGSTITCHER_ALIGN_HDF5_OUT
)

# INIT typer cmdline interface
app = typer.Typer()


def get_bigstitcher_omezarr_alignment_marco(
    input_omezarr_xml_path: Path,
    output_omezarr_path: Path,
    path_to_write_macro: Path=None,
    downsample_in_x: str=DOWNSAMPLE_IN_X,
    downsample_in_y: str=DOWNSAMPLE_IN_Y,
    downsample_in_z: str=DOWNSAMPLE_IN_Z,
    block_size_x: int=BLOCKSIZE_X,
    block_size_y: int=BLOCKSIZE_Y,
    block_size_z: int=BLOCKSIZE_Z,
    block_size_factor_x: int=BLOCKSIZE_FACTOR_X,
    block_size_factor_y: int=BLOCKSIZE_FACTOR_Y,
    block_size_factor_z: int=BLOCKSIZE_FACTOR_Z,
    subsampling_factors: str=SUBSAMPLING_FACTORS,
    downsample_refinement: str=DOWNSAMPLE_REFINEMENT
):
    '''
    Generate BigStitcher macro for aligning omezarr data
    Return the macro string
    If given path_to_write_macro, also write the macro to that path
    '''

    automated_downsample = any([x.lower() == 'automated' for x in [downsample_in_x, downsample_in_y, downsample_in_z]])
    automated_subsampling = subsampling_factors.lower() == 'automated'
    automated_refinement_sampling = downsample_refinement.lower() == 'automated'

    if automated_downsample or automated_subsampling or automated_refinement_sampling:
        _, scale_factors_list_zyx, subsampling_str = determine_sampling_factors_for_bigstitcher(input_omezarr_xml_path)
        sampling_num_max_idx = len(scale_factors_list_zyx) - 1

    if automated_subsampling:
        subsampling_factors = subsampling_str

    if automated_downsample:
        scale_for_downsample_zyx = scale_factors_list_zyx[2 if sampling_num_max_idx >= 2 else sampling_num_max_idx]  # 3rd downsample factor (1,1,1),(2,2,1),(4,4,2),(8,8,4)
        downsample_in_x = scale_for_downsample_zyx[2]
        downsample_in_y = scale_for_downsample_zyx[1]
        downsample_in_z = scale_for_downsample_zyx[0]

    if automated_refinement_sampling:
        # Setup 3 stage refinement for channel alignment
        # Start on level 5 (~16x in xy, if it exists) for initial coarse alignment, then level 4 for medium refinement, then level 3 for fine refinement
        starting_level = 4 if sampling_num_max_idx >= 4 else sampling_num_max_idx
        starting_level += 1
        downsample_refinement = list(reversed(scale_factors_list_zyx[starting_level-3:starting_level]))  # 5th downsample factor (1,1,1),(2,2,1),(4,4,2),(8,8,4)
        # downsample_refinement list of tuples lowest res to highest res [(8,8,4),(4,4,2),(2,2,1)]
        print(f'{downsample_refinement=}')


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
        subsampling_factors,
        downsample_refinement[0][2], # First y
        downsample_refinement[0][0], # First z
        downsample_refinement[1][2], # Second y
        downsample_refinement[1][0], # Second z
        downsample_refinement[2][2], # Third y
        downsample_refinement[2][0]  # Third z
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
    subsampling_factors: str=SUBSAMPLING_FACTORS,
    downsample_refinement: str=DOWNSAMPLE_REFINEMENT
):
    '''
    Generate BigStitcher macro for aligning omezarr data
    Return the macro string
    If given path_to_write_macro, also write the macro to that path
    '''

    automated_downsample = any([x.lower() == 'automated' for x in [downsample_in_x, downsample_in_y, downsample_in_z]])
    automated_subsampling = subsampling_factors.lower() == 'automated'
    automated_refinement_sampling = downsample_refinement.lower() == 'automated'

    if automated_downsample or automated_subsampling or automated_refinement_sampling:
        _, scale_factors_list_zyx, subsampling_str = determine_sampling_factors_for_bigstitcher(input_omezarr_xml_path)
        sampling_num_max_idx = len(scale_factors_list_zyx) - 1

    if automated_subsampling:
        subsampling_factors = subsampling_str

    if automated_downsample:
        scale_for_downsample_zyx = scale_factors_list_zyx[2 if sampling_num_max_idx>=2 else sampling_num_max_idx]  # 3rd downsample factor (1,1,1),(2,2,1),(4,4,2),(8,8,4)
        downsample_in_x = scale_for_downsample_zyx[2]
        downsample_in_y = scale_for_downsample_zyx[1]
        downsample_in_z = scale_for_downsample_zyx[0]

    if automated_refinement_sampling:
        # Setup 3 stage refinement for channel alignment
        # Start on level 5 (~16x in xy, if it exists) for initial coarse alignment, then level 4 for medium refinement, then level 3 for fine refinement
        starting_level = 4 if sampling_num_max_idx >= 4 else sampling_num_max_idx
        starting_level += 1
        downsample_refinement = list(reversed(scale_factors_list_zyx[
                                                  starting_level - 3:starting_level]))  # 5th downsample factor (1,1,1),(2,2,1),(4,4,2),(8,8,4)
        # downsample_refinement list of tuples lowest res to highest res [(8,8,4),(4,4,2),(2,2,1)]
        print(f'{downsample_refinement=}')


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
        subsampling_factors,
        downsample_refinement[0][2],  # First y
        downsample_refinement[0][0],  # First z
        downsample_refinement[1][2],  # Second y
        downsample_refinement[1][0],  # Second z
        downsample_refinement[2][2],  # Third y
        downsample_refinement[2][0]  # Third z
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


def get_bigstitcher_fused_output_path(path: Path, format: str='omezarr') -> Path:
    """
    Return the expected fused BigStitcher output path for a directory.
    """
    omezarr_xml = does_dir_contain_bigstitcher_metadata(path)
    if not omezarr_xml:
        raise FileNotFoundError(f'No BigStitcher XML found in {path}')

    omezarr_xml = ensure_path(omezarr_xml)
    fused_out_dir_or_file = str(omezarr_xml).removesuffix('.ome.zarr.xml')
    if format == 'omezarr':
        fused_out_dir_or_file = fused_out_dir_or_file + '_montage.ome.zarr'
    elif format == 'hdf5':
        fused_out_dir_or_file = fused_out_dir_or_file + '_montage.h5'
    else:
        raise ValueError(f'Unsupported BigStitcher fused output format: {format}')

    return ensure_path(fused_out_dir_or_file)


def get_bigstitcher_tiff_series_output_path(fused_omezarr_path: Path) -> Path:
    fused_omezarr_path = ensure_path(fused_omezarr_path)
    tiff_series_dir_name = str(fused_omezarr_path.name[:-9]) + '_tiffstack'
    return fused_omezarr_path.parent / tiff_series_dir_name


def is_bigstitcher_log_successful(log_dir: Path, fused_output_path: Path=None) -> bool:
    """
    Detect whether a prior BigStitcher SLURM log indicates successful fusion.

    Prefer the explicit success marker added by this pipeline. For older logs,
    fall back to the fusion-stage banner as long as the log does not contain
    obvious failure markers.
    """
    log_dir = ensure_path(log_dir)
    if not log_dir.exists():
        return False

    fused_output_name = ensure_path(fused_output_path).name if fused_output_path else None
    explicit_marker = 'BIGSTITCHER_SUCCESS:'
    failure_pattern = re.compile(r'(outofmemory|killed|cancelled|terminated|traceback|failed)', re.IGNORECASE)

    for log_file in sorted(log_dir.glob('*_align_fuse_bigstitcher.log')):
        try:
            content = log_file.read_text(errors='ignore')
        except (OSError, IOError):
            continue

        if explicit_marker in content:
            if fused_output_name is None or fused_output_name in content:
                return True

        if fused_output_name and fused_output_name not in content:
            continue

        if 'Creating Fused Dataset to OME-Zarr' in content or 'Creating Fused Dataset to HDF5' in content:
            if not failure_pattern.search(content):
                return True

    return False


def _get_omezarr_level_zyx_info(omezarr_path: Path, resolution_level: int) -> dict[str, Any]:
    from omezarr import OmeZarrV2Multiscale

    omezarr_path = ensure_path(omezarr_path)
    multiscale = OmeZarrV2Multiscale(omezarr_path)
    return multiscale.get_level_zyx_info(resolution_level)


def _get_expected_tiff_plane_pairs(reference_tile_omezarr_path: Path, num_channels: int, resolution_level: int) -> set[tuple[int, int]]:
    z_layers = _get_omezarr_level_zyx_info(reference_tile_omezarr_path, resolution_level)['z_layers']
    return {
        (channel, z)
        for channel in range(num_channels)
        for z in range(z_layers)
    }


def _get_logged_tiff_success_pairs(log_dir: Path, tiff_series_name: str, resolution_level: int) -> set[tuple[int, int]]:
    log_dir = ensure_path(log_dir)
    if not log_dir.exists():
        return set()

    explicit_marker = 'OMEZARR_TO_TIFF_SUCCESS:'
    marker_pattern = re.compile(
        rf'{explicit_marker}\s+{re.escape(tiff_series_name)}\s+'
        rf'r(?P<resolution>\d+)\s+c(?P<channel>\d+)\s+z(?P<z>\d+)',
        re.IGNORECASE,
    )
    seen_markers = set()

    for log_file in sorted(log_dir.glob('*_omezarr_to_tiff_stack_c*.log')):
        try:
            content = log_file.read_text(errors='ignore')
        except (OSError, IOError):
            continue

        for match in marker_pattern.finditer(content):
            if int(match.group('resolution')) != resolution_level:
                continue
            seen_markers.add((int(match.group('channel')), int(match.group('z'))))

    return seen_markers


def get_completed_tiff_planes(log_dir: Path, tiff_series_dir: Path, resolution_level: int = 0) -> set[tuple[int, int]]:
    tiff_series_dir = ensure_path(tiff_series_dir)
    if not tiff_series_dir.is_dir():
        return set()

    successful_log_pairs = _get_logged_tiff_success_pairs(log_dir, tiff_series_dir.name, resolution_level)
    completed_pairs = set()
    tiff_files = sorted(tiff_series_dir.glob('*.tif')) + sorted(tiff_series_dir.glob('*.tiff'))
    file_name_pattern = re.compile(
        r'_r(?P<resolution>\d+)_t\d+_c(?P<channel>\d+)_z(?P<z>\d+)\.tif{1,2}$',
        re.IGNORECASE,
    )

    for tiff_file in tiff_files:
        if not tiff_file.is_file() or tiff_file.stat().st_size <= 0:
            continue

        match = file_name_pattern.search(tiff_file.name)
        if not match:
            continue
        if int(match.group('resolution')) != resolution_level:
            continue

        pair = (int(match.group('channel')), int(match.group('z')))
        if pair in successful_log_pairs:
            completed_pairs.add(pair)

    return completed_pairs


def get_missing_tiff_planes_by_channel(log_dir: Path, tiff_series_dir: Path, reference_tile_omezarr_path: Path, num_channels: int, resolution_level: int = 0) -> dict[int, list[int]]:
    expected_pairs = _get_expected_tiff_plane_pairs(reference_tile_omezarr_path, num_channels, resolution_level)
    completed_pairs = get_completed_tiff_planes(log_dir, tiff_series_dir, resolution_level=resolution_level)
    missing_pairs = expected_pairs - completed_pairs

    missing_by_channel = {}
    for channel in range(num_channels):
        missing_z = sorted(z for current_channel, z in missing_pairs if current_channel == channel)
        if missing_z:
            missing_by_channel[channel] = missing_z

    return missing_by_channel


def is_tiff_generation_log_successful(log_dir: Path, tiff_series_dir: Path, reference_tile_omezarr_path: Path, num_channels: int, resolution_level: int = 0) -> bool:
    """
    Detect whether per-plane OME-Zarr to TIFF extraction completed successfully.

    Success requires one explicit log marker for every expected `(channel, z)`
    plane at the requested multiscale level.
    """
    log_dir = ensure_path(log_dir)
    if not log_dir.exists():
        return False

    tiff_series_dir = ensure_path(tiff_series_dir)
    expected_markers = _get_expected_tiff_plane_pairs(reference_tile_omezarr_path, num_channels, resolution_level)
    seen_markers = _get_logged_tiff_success_pairs(log_dir, tiff_series_dir.name, resolution_level)
    return seen_markers == expected_markers


def is_bigstitcher_omezarr_scale_metadata_complete(source_xml_or_dir: Path, fused_omezarr_path: Path) -> bool:
    source_xml_or_dir = ensure_path(source_xml_or_dir)
    fused_omezarr_path = ensure_path(fused_omezarr_path)

    if source_xml_or_dir.as_posix().endswith('.ome.zarr.xml'):
        omezarr_xml = source_xml_or_dir
    else:
        omezarr_xml = does_dir_contain_bigstitcher_metadata(source_xml_or_dir)
        if not omezarr_xml:
            return False
        omezarr_xml = ensure_path(omezarr_xml)

    target_zattr_path = fused_omezarr_path / '.zattrs'
    if not target_zattr_path.is_file():
        return False

    try:
        scales_list_zyx, _, _ = determine_sampling_factors_for_bigstitcher(omezarr_xml)
        target_zattr = json.loads(target_zattr_path.read_text())
        datasets = target_zattr.get('multiscales', [])[0].get('datasets', [])
    except Exception:
        return False

    if len(datasets) != len(scales_list_zyx):
        return False

    for dataset in datasets:
        try:
            idx = int(dataset.get('path'))
            expected_scale = [1, 1] + scales_list_zyx[idx]
            actual_scale = dataset['coordinateTransformations'][0]['scale']
        except (IndexError, KeyError, TypeError, ValueError):
            return False

        if list(actual_scale) != list(expected_scale):
            print("============= actual_scale in zattrs != expected_scale from determine_sampling_factors_for_bigstitcher ===========")
            print("actual_scale", actual_scale)
            print("expected_scale", expected_scale)
            # return False

    return True


def is_bigstitcher_omezarr_montage_valid_and_complete(source_xml_or_dir: Path, fused_omezarr_path: Path) -> bool:
    from omezarr import validate_ome_zarr_multiscale

    fused_omezarr_path = ensure_path(fused_omezarr_path)
    if not fused_omezarr_path.is_dir():
        return False
    print("=========================validate_ome_zarr_multiscale", validate_ome_zarr_multiscale(fused_omezarr_path))
    print("=========================is_bigstitcher_omezarr_scale_metadata_complete", is_bigstitcher_omezarr_scale_metadata_complete(source_xml_or_dir, fused_omezarr_path))
    return (
        validate_ome_zarr_multiscale(fused_omezarr_path)
        and is_bigstitcher_omezarr_scale_metadata_complete(source_xml_or_dir, fused_omezarr_path)
    )


def is_bigstitcher_tiff_series_complete(tiff_series_dir: Path, reference_tile_omezarr_path: Path, num_channels: int, resolution_level: int = 0) -> bool:
    tiff_series_dir = ensure_path(tiff_series_dir)
    if not tiff_series_dir.is_dir():
        return False

    z_layers = _get_omezarr_level_zyx_info(reference_tile_omezarr_path, resolution_level)['z_layers']
    expected_tiff_count = num_channels * z_layers
    tiff_files = sorted(tiff_series_dir.glob('*.tif')) + sorted(tiff_series_dir.glob('*.tiff'))
    if len(tiff_files) != expected_tiff_count:
        return False

    expected_pairs = {
        (channel, z)
        for channel in range(num_channels)
        for z in range(z_layers)
    }
    seen_pairs = set()
    file_name_pattern = re.compile(
        r'_r(?P<resolution>\d+)_t\d+_c(?P<channel>\d+)_z(?P<z>\d+)\.tif{1,2}$',
        re.IGNORECASE,
    )

    for tiff_file in tiff_files:
        if not tiff_file.is_file() or tiff_file.stat().st_size <= 0:
            return False

        match = file_name_pattern.search(tiff_file.name)
        if not match:
            return False
        if int(match.group('resolution')) != resolution_level:
            return False

        seen_pairs.add((int(match.group('channel')), int(match.group('z'))))

    return seen_pairs == expected_pairs


def should_skip_bigstitcher_run(dir_loc: Path, fused_file_type: str, final_file_type: str, log_dir: Path, metadata_by_channel: dict, resolution_level: int = 0) -> tuple[bool, str]:
    if fused_file_type.lower() != 'omezarr':
        return False, ''

    fused_output_path = get_bigstitcher_fused_output_path(dir_loc, format=fused_file_type)
    has_successful_log = is_bigstitcher_log_successful(log_dir, fused_output_path=fused_output_path)
    print("==================has_successful_log", has_successful_log)
    if not has_successful_log:
        return False, ''

    if is_bigstitcher_omezarr_montage_valid_and_complete(dir_loc, fused_output_path):
        return True, 'existing valid montage ome.zarr and successful BigStitcher log'

    return False, ''

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
    fused_out_dir_or_file = get_bigstitcher_fused_output_path(path, format=format)

    # Writes macro file
    if format == 'omezarr':
        _ = get_bigstitcher_omezarr_alignment_marco(omezarr_xml, fused_out_dir_or_file, macro_file)
    elif format == 'hdf5':
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


def get_reference_multiscale_tile_path(path_to_mesospim_omezarr: Path) -> Path:
    """
    Return a representative child tile OME-Zarr that contains multiscales.

    The collection root referenced by the BigStitcher XML is a container of per-tile
    OME-Zarr datasets, not usually a multiscale image itself.
    """
    from omezarr import validate_ome_zarr_multiscale

    path_to_mesospim_omezarr = ensure_path(path_to_mesospim_omezarr)
    tile_dir_list = sorted(list_mesospim_ome_zarr_tile_dirs(path_to_mesospim_omezarr))

    for tile_dir in tile_dir_list:
        if validate_ome_zarr_multiscale(tile_dir):
            return tile_dir

    raise FileNotFoundError(
        f'No child multiscale OME-Zarr tile was found in collection root {path_to_mesospim_omezarr}'
    )

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
    output_xml_path: Path,
    different_relative_zarr_path: str = None,
    modify_filename_in_xml: str = None
):

    '''
    Convert mesospim metadata to bigstitcher xml format

    output_xml_path: Will be in the same directory or at a lower level from as the mesospim metadata files
    different_relative_zarr_path:
        If provided: set the relative zarr path in the bigstitcher xml to this value
        if None: relative zarr path will be set to the parent directory name of output_xml_path with .ome.zarr suffix

    ** Currently hardcoded to work with single timepoint, multiple channels, single angle data **
    ** Only ome-zarr format supported currently **
    '''

    import xml.etree.ElementTree as ET
    metadata_by_channel = collect_all_metadata(output_xml_path)

    if modify_filename_in_xml:
        print(f'Modifying file names in bigstitcher xml to be {modify_filename_in_xml} for all tiles')
        metadata_by_channel = modify_file_names_in_annotated_metadata(metadata_by_channel)

    first_metadata_entry = get_first_entry(metadata_by_channel)

    print(f'Building bigstitcher xml')
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

    if different_relative_zarr_path:
        zarr.text = different_relative_zarr_path
    else:
        parent_dir_name = output_xml_path.parent.name
        zarr.text = f'{parent_dir_name}.ome.zarr'

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
    print(f'Writing bigstitcher xml to {output_xml_path}')
    tree = ET.ElementTree(spimdata)
    ET.indent(tree, space="  ")

    tree.write(
        output_xml_path,
        encoding="utf-8",
        xml_declaration=True
    )
    print(f'Completed writing bigstitcher xml to {output_xml_path}')


@app.command()
def test_func():
    print('Test function in bigstitcher.py')



if __name__ == '__main__':
    app()
