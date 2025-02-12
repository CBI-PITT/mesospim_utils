from pathlib import Path
from collections import defaultdict
from pprint import pprint as print
import re

from utils import dict_to_json_file, read_json_file_to_dict

from constants import EMISSION_MAP

def collect_all_metadata(location: Path, save_json_path:Path = None):
    print("Verifying that metadata files are present for each .btf file")
    verify_metadata_file_for_each_btf(location)
    print("Grouping BTF files by channel and storing by tile order")
    files_by_channel = group_btf_files_by_channel_sort_by_tile(location)
    print("Collecting metadata for each tile")
    metadata_by_channel = collect_metadata_for_each_btf(files_by_channel)
    print(metadata_by_channel)
    if save_json_path:
        if not isinstance(save_json_path, Path):
            save_json_path = Path(save_json_path)
        dict_to_json_file(metadata_by_channel, save_json_path)
    return metadata_by_channel

def mesospim_meta_data(meta_file: Path):
    # Extract imaging metadata from mesospim metadata file

    if not isinstance(meta_file, Path):
        meta_file = Path(meta_file)

    assert meta_file.is_file(), 'Path is not a file'
    #print(meta_file)

    meta_dict = {}
    meta_dict['file'] = meta_file
    with meta_file.open('r') as f:
        metadata = f.readlines()

    emission_wavelength = [x for x in metadata if '[Filter]' in x][0][9:12]
    try:
        emission_wavelength = int(emission_wavelength)
    except Exception:
        assert emission_wavelength.lower() in EMISSION_MAP, f"No emission wavelength found for: {meta_file}"
        emission_wavelength = EMISSION_MAP.get(emission_wavelength.lower())

    meta_dict['emission_wavelength'] = emission_wavelength

    xy_res = [x for x in metadata if '[Pixelsize in um]' in x][0][18:-1]
    xy_res = float(xy_res)
    y_res = x_res = xy_res
    meta_dict['x_res'] = x_res
    meta_dict['y_res'] = y_res

    z_res = [x for x in metadata if '[z_stepsize]' in x][0][13:-1]
    z_res = float(z_res)
    meta_dict['z_res'] = z_res

    z_planes = [x for x in metadata if '[z_planes]' in x][0][11:-1]
    z_planes = int(z_planes)
    meta_dict['z_planes'] = z_planes

    x_pixels = [x for x in metadata if '[x_pixels]' in x][0][11:-1]
    x_pixels = int(x_pixels)
    meta_dict['x_pixels'] = x_pixels

    y_pixels = [x for x in metadata if '[y_pixels]' in x][0][11:-1]
    y_pixels = int(y_pixels)
    meta_dict['y_pixels'] = y_pixels


    ''' EXAMPLE
    [POSITION] 
    [x_pos] 1773.4
    [y_pos] -8526.3
    [f_start] 3301.0
    [f_end] 3401.0
    [z_start] 279.4
    [z_end] 6779.3
    [z_stepsize] 2.0
    [z_planes] 3249
    [rot] 0
    '''

    x_pos = [x for x in metadata if '[x_pos]' in x][0][8:-1]
    x_pos = float(x_pos)
    meta_dict['x_pos'] = x_pos

    y_pos = [x for x in metadata if '[y_pos]' in x][0][8:-1]
    y_pos = float(y_pos)
    meta_dict['y_pos'] = y_pos

    #print(meta_dict)
    return meta_dict


def get_metadata_file_name(btf_file: Path):
    return btf_file.with_name(btf_file.name + "_meta.txt")

def group_btf_files_by_channel_sort_by_tile(location: Path):

    if not isinstance(location, Path):
        location = Path(location)

    assert location.is_dir(), 'Path is not a directory'

    btf_files = list(location.glob("*.btf"))
    btf_files = [str(x) for x in btf_files]

    # Dictionary to hold lists of files for each number
    files_by_channel = defaultdict(list)

    # Regular expressions for extracting channel and tile numbers
    channel_pattern = r"_Ch(\d+)_"
    tile_pattern = r"_Tile(\d+)_"

    # Group files by channel
    for file_name in btf_files:
        channel_match = re.search(channel_pattern, file_name)
        if channel_match:
            channel_number = channel_match.group(1)
            files_by_channel[channel_number].append(file_name)

    # Sort files in each channel by tile number
    for channel, files in files_by_channel.items():
        files.sort(key=lambda x: int(re.search(tile_pattern, x).group(1)))
        files_by_channel[channel] = [Path(file) for file in files]

    # # Convert defaultdict to a regular dictionary for better display
    # files_by_channel = dict(files_by_channel)

    # Print grouped files
    #print(files_by_channel)
    return files_by_channel


def collect_metadata_for_each_btf(files_by_channel):

    for channel, files in files_by_channel.items():
        meta_files = [get_metadata_file_name(x) for x in files]
        files_by_channel[channel] = [mesospim_meta_data(x) for x in meta_files]

    return files_by_channel



def verify_metadata_file_for_each_btf(location: Path):

    if not isinstance(location, Path):
        location = Path(location)

    assert location.is_dir(), 'Path is not a directory'

    btf_files = list(location.glob("*.btf"))

    # Verify a metadata for for each .btf
    for btf in btf_files:
        meta_file = get_metadata_file_name(btf)
        #print(f"Checking {meta_file}")
        assert meta_file.is_file(), f"{meta_file} is missing"

    print('Verified a metadata file is present for each .btf file')

    # # Collect metadata for each .btf file
    # master_meta_data_dict
    # meta_data_dict = {}
