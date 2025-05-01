from pathlib import Path
from collections import defaultdict
from pprint import pprint as print
import re
import os
import json

from collections import namedtuple

from utils import ensure_path, dict_to_json_file, json_file_to_dict, convert_paths_to_posix, convert_paths, convert_str_to_nums
from utils import map_wavelength_to_RGB

from constants import EMISSION_MAP, METADATA_FILENAME, VERBOSE


def collect_all_metadata(location: Path, prepare=True):
    """
    Collect all relevant metadata from mesospim Tile metadata files, sort by channel

    prepare=True will sort and annotate the data. This is the primary format for all downstream uses
    """

    location = ensure_path(location)
    location = find_metadata_dir(location)

    # If the default file is present, read it, sort it by channel, by tile
    save_json_path = location / METADATA_FILENAME
    if location.suffix.lower() == ".json" or save_json_path.is_file():

        if VERBOSE: print("Reading metadata json")
        metadata_by_channel = json_file_to_dict(save_json_path)

        if prepare:
            metadata_by_channel = annotate_metadata(sort_meta_list(metadata_by_channel), location=location)

        return metadata_by_channel


    # Otherwise, read all raw metadata entries and create the json
    if VERBOSE: print("Verifying that metadata files are present for each .btf file")
    verify_metadata_file_for_each_btf(location)

    if VERBOSE: print("Collecting all metadata into json and saving the json")
    meta_list = save_dir_of_meta_to_json(location, save_json_path)

    if VERBOSE: print("Sorting metadata by channel and tile")
    metadata_by_channel = meta_list
    if prepare:
        metadata_by_channel = convert_paths(metadata_by_channel)
        metadata_by_channel = convert_str_to_nums(metadata_by_channel)
        metadata_by_channel = annotate_metadata(sort_meta_list(metadata_by_channel), location=location)

        print("Saving annotated metadata json")
        annotated_name = save_json_path.with_name("mesospim_annotated_metadata.json")
        dict_to_json_file(metadata_by_channel, annotated_name)

    if VERBOSE: print(metadata_by_channel)

    return metadata_by_channel


def sort_meta_list(meta_list):

    '''
    take a list of dictionaries where each dictionary represents a metadata file from the mesospim.
    The meta_list is the output of function: save_dir_of_meta_to_json and read by function: json_file_to_dict(save_json_path)
    '''

    # Dictionary to store sorted data dynamically
    sorted_data = {}

    # Regex patterns to extract Tile number and Channel
    tile_pattern = re.compile(r"Tile(\d+)")
    # channel_pattern = re.compile(r"_Ch(\d+)_")
    channel_pattern = re.compile(r"_Ch(\d+)([a-zA-Z]*)_")

    # Process each entry
    for entry in meta_list:
        file_path = Path(entry["Metadata for file"])  # Convert to Path object
        file_name = file_path.name

        # Extract tile number
        tile_match = tile_pattern.search(file_name) #Extract tile number from filename
        tile_number = int(tile_match.group(1)) if tile_match else None

        # Extract channel number
        channel_match = channel_pattern.search(file_name) #Extract channel wavelength from filename
        if channel_match:
            channel_number = channel_match.group(1) + channel_match.group(2)  # e.g., '561b'
        else:
            channel_number = None
        # channel_number = channel_match.group(1) if channel_match else None
        # channel_number = int(channel_number)

        if tile_number is not None and channel_number is not None:
            if channel_number not in sorted_data:
                sorted_data[channel_number] = []  # Initialize list for new channel

            entry["tile_number"] = tile_number  # Add tile number to dictionary
            sorted_data[channel_number].append(entry)

    # Sort lists by tile number
    for channel in sorted_data:
        sorted_data[channel].sort(key=lambda x: x["tile_number"])

    # Ensure keys of metadata dictionary are always sorted in order of channel excitation
    sorted_data = dict(sorted(sorted_data.items(), key=lambda item: sort_key(item[0])))

    return sorted_data


def sort_key(key):
    match = re.match(r"(\d+)([a-zA-Z]*)", str(key))
    if match:
        num_part = int(match.group(1))
        letter_part = match.group(2)
        return (num_part, letter_part)
    else:
        return (float('inf'), str(key))  # fallback for unexpected format

def annotate_metadata(metadata_by_channel, location=None):

    if location:
        location = ensure_path(location)

    for color, data in metadata_by_channel.items():

        # Collect grid size information to be appended to each metadata parameter
        grid_size = determine_grid_size(data)

        # Find Overlap
        overlap = determine_overlap(data)

        # Add 'emission_wavelength' to each metadata parameter
        for entry in data:

            emission_wavelength = determine_emission_wavelength(entry)

            # Append info
            entry['channel'] = color
            entry['emission_wavelength'] = emission_wavelength
            entry['rgb_representation'] = map_wavelength_to_RGB(emission_wavelength)
            entry['grid_size'] = grid_size
            entry['grid_location'] = get_grid_location(entry['grid_size'], entry['tile_number'])
            entry['overlap'] = overlap
            entry['resolution'] = determine_xyz_resolution(entry)
            entry['tile_shape'] = determine_tile_shape(entry)
            entry['tile_size_um'] = determine_tile_size_um(entry)
            entry['file_name'] = entry.get('Metadata for file').name
            entry['refractive_index'] = determine_refractive_index_from_ETL_file_name(entry)
            entry['sheet'] = determine_sheet_direction(entry)

            ### NOTES ###
            # entry['tile_number'] is added by the sort_meta_list() function


            if location:
                entry['file_path'] = location / entry['file_name']

    return metadata_by_channel

def get_grid_location(grid_size, tile_number):

    GridLocation = namedtuple('GridLocation', ['x', 'y'])

    current_tile = 0
    for x in range(grid_size.x):
        for y in range(grid_size.y):
            if tile_number == current_tile:
                return GridLocation(x=x, y=y)
            current_tile += 1

def determine_sheet_direction_from_tile_number(metadata_by_channel, tile_num):
    for ch in metadata_by_channel:
        for tile in metadata_by_channel[ch]:
            if int(tile.get('tile_number')) == tile_num:
                return tile.get("sheet")
def determine_sheet_direction(metadata_entry):
    return metadata_entry.get("CFG").get("Shutter").lower()

def determine_refractive_index_from_ETL_file_name(metadata_entry):
    '''
    Assumes that refractive index is embedded in the etl cgf filename as:
    *_RI_{float_RI}_*.csv
    This function extracts the 'float_RI' value
    '''
    etl_file_name = metadata_entry["ETL PARAMETERS"]["ETL CFG File"]
    etl_file_name = str(etl_file_name.name)
    split_ri = etl_file_name.split('_RI_')[-1]
    split_ri = split_ri.split('_')[0]
    try:
        ri = float(split_ri)
        return ri
    except Exception:
        if VERBOSE: print('No RI found in the ETL cfg file')
        return None

def determine_tile_size_um(metadata_entry):
    shape = determine_tile_shape(metadata_entry)
    res = determine_xyz_resolution(metadata_entry)

    TileSizeUm = namedtuple('TileSizeUm', ['x', 'y', 'z'])

    return TileSizeUm(x=shape.x * res.x, y=shape.y * res.y, z=shape.z * res.z)


def determine_tile_shape(metadata_entry):
    TileShape = namedtuple('TileShape', ['x', 'y', 'z'])

    x = metadata_entry['CAMERA PARAMETERS']['x_pixels']
    y = metadata_entry['CAMERA PARAMETERS']['y_pixels']
    z = metadata_entry['POSITION']['z_planes']

    return TileShape(x=x, y=y, z=z)


def determine_emission_wavelength(metadata_entry):
    # Extract 3 digit wavelength, if it doesn't exist reference EMISSION_MAP in the constants module
    emission_wavelength = metadata_entry['CFG']['Filter'][0:3]
    try:
        emission_wavelength = int(emission_wavelength)
    except Exception:
        assert emission_wavelength.lower() in EMISSION_MAP, f"No emission wavelength found for channel: {color}"
        emission_wavelength = EMISSION_MAP.get(emission_wavelength.lower())
    return emission_wavelength


def determine_xyz_resolution(metadata_entry):
    xy = metadata_entry["CFG"]["Pixelsize in um"]
    z = metadata_entry["POSITION"]["z_stepsize"]

    Resolution = namedtuple('Resolution', ['x', 'y', 'z'])
    return Resolution(x=xy, y=xy, z=z)

def determine_overlap(single_channel_metadata):

    for metadata in single_channel_metadata:
        if metadata['tile_number'] == 0:
            loc0 = metadata['POSITION']["y_pos"]
        if metadata['tile_number'] == 1:
            loc1 = metadata['POSITION']["y_pos"]

    resolution = single_channel_metadata[0]["CFG"]["Pixelsize in um"]
    cam_pixels = single_channel_metadata[0]["CAMERA PARAMETERS"]["y_pixels"]

    distance_moved = abs(loc1 - loc0)
    distance_fov = resolution * cam_pixels
    overlap_percent = 1 - (distance_moved / distance_fov)
    overlap_percent = round(overlap_percent, 2)
    return overlap_percent

def determine_grid_size(single_channel_metadata):
    """
    Determines the grid size based on unique x_pos and y_pos values.
    Args:
        data (list of dict): List of dictionaries containing 'x_pos' and 'y_pos'.

    Returns:
        tuple: (number of unique x_pos, number of unique y_pos)
    """
    data =single_channel_metadata

    # Extract unique x_pos and y_pos values
    x_positions = {entry['POSITION']["x_pos"] for entry in data}
    y_positions = {entry['POSITION']["y_pos"] for entry in data}

    # Calculate the grid size
    grid_size_x = len(x_positions)
    grid_size_y = len(y_positions)

    #print(f'{grid_size_x}X, {grid_size_y}Y')
    Grid_size = namedtuple('Grid_size', ['x', 'y'])

    return Grid_size(x=grid_size_x, y=grid_size_y)


def get_metadata_file_name(btf_file: Path):
    '''
    Takes: a mesospim tile file name
    returns: metadata file name
    '''
    btf_file = ensure_path(btf_file)

    return btf_file.with_name(btf_file.name + "_meta.txt")

def find_metadata_dir(start_location):
    '''
    Given a path, search backwards to find mesospim metadata
    do not ascend all the way to root
    '''
    start_location = ensure_path(start_location)
    paths_to_test = [start_location] + list(start_location.parents)
    for current_location in paths_to_test[:-1]: # Cut out that last entry which is root
        if VERBOSE: print(f'Searching for Metadata at {current_location}')
        if (current_location / METADATA_FILENAME).is_file():
            return current_location
        meta_files = list(current_location.glob("*_meta.txt"))
        if len(meta_files) > 0:
            return current_location
    return None


def verify_metadata_file_for_each_btf(location: Path):
    """
    Verify that a metadata file exist for each .btf tile file
    """

    if not isinstance(location, Path):
        location = Path(location)

    assert location.is_dir(), 'Path is not a directory'

    btf_files = list(location.glob("*.btf"))

    # Verify a metadata for for each .btf
    for btf in btf_files:
        meta_file = get_metadata_file_name(btf)
        #print(f"Checking {meta_file}")
        assert meta_file.is_file(), f"{meta_file} is missing"

    if VERBOSE: print('Verified a metadata file is present for each .btf file')

#####################################################################################################################
#####################################################################################################################

def parse_btf_meta_file(filepath):
    """Parses a .btf_meta.txt file into a structured dictionary."""
    data = {}
    section = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            elif line.startswith("[Metadata for file]"):
                data["Metadata for file"] = line.split('] ')[-1]
            elif line.startswith("[") and line.endswith("]"):  # Section headers
                section = line.strip("[]")
                data[section] = {}
            else:  # Key-value pairs within a section
                key, value = line.split('] ')
                key = key.strip("[")  # Remove leading [
                data[section][key] = value

    return data

def save_dir_of_meta_to_json(directory, output_json):
    """Reads all .btf_meta.txt files, parses them, and saves to JSON."""
    meta_list = []

    for filename in os.listdir(directory):
        if filename.endswith(".btf_meta.txt"):
            filepath = os.path.join(directory, filename)
            meta_list.append(parse_btf_meta_file(filepath))

    dict_to_json_file(meta_list, output_json)
    return meta_list

def recreate_files_from_meta_dict(meta_list_from_json, output_directory):
    """Writes dictionary data back to .btf_meta.txt files.
    Takes metadata gathered from raw files using the following method:
    collect_all_metadata(input_directory, prepare=False)
    """
    os.makedirs(output_directory, exist_ok=True)

    meta_list_from_json = convert_paths_to_posix(meta_list_from_json)

    meta_file_name = ''
    for entry in meta_list_from_json:
        file_content = ''
        for key, value in entry.items():
            if key == "Metadata for file":
                file_content = f'[Metadata for file] {value}\n\n'
                meta_file_name = os.path.split(value)[-1] + '_meta.txt'
                meta_file_name = os.path.join(output_directory, meta_file_name)
            else:
                file_content = f'{file_content}[{key}] \n'
                for inner_key, inner_value in value.items():
                    file_content = f'{file_content}[{inner_key}] {inner_value}\n'
                file_content = f'{file_content}\n'


        with open(meta_file_name, 'w') as f:
            f.write(file_content[:-1]) # Write contents after stripping the final \n

def get_first_entry(meta_dict):
    '''
    Given the meta_dict return the first entry from the first channel key
    '''
    for ch in meta_dict:
        for entry in meta_dict[ch]:
            return entry

def get_ch_entry_for_file_name(meta_dict, file_name):
    '''
    Given the meta_dict created by fn collect_all_metadata()
    And the file_name: /{dir1}/{dir2}/.../{file_name}

    return the channel key and index of the specific metadata record

    If the file_name is not found, then return (None,None)
    '''
    for ch in meta_dict:
        for idx,entry in enumerate(meta_dict[ch]):
            if file_name.lower() == entry.get('file_name').lower():
                return ch,idx
    return None, None

def get_entry_for_file_name(meta_dict, file_name):
    '''
    Given the meta_dict created by fn collect_all_metadata()
    And the file_name: /{dir1}/{dir2}/.../{file_name}

    return specific meta_data dict record for the given file_name

    If the file_name is not found, then return (None,None)
    '''
    ch, idx = get_ch_entry_for_file_name(meta_dict, file_name)
    if ch is not None and idx is not None:
        return meta_dict[ch][idx]
    return None

def get_all_tile_entries(meta_dict, tile_num):
    tile_num = int(tile_num)
    tile_entries = []
    for ch in meta_dict:
        for entry in meta_dict[ch]:
            if int(entry.get('tile_number')) == tile_num:
                tile_entries.append(entry)
    return tile_entries


if __name__ == "__main__":
    pass
