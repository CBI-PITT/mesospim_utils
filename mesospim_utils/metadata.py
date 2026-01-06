from pathlib import Path
from collections import defaultdict
from pprint import pprint as print
import re
import os
import json

from collections import namedtuple

from utils import ensure_path, dict_to_json_file, json_file_to_dict, convert_paths_to_posix, convert_paths, \
    convert_str_to_nums, get_user
from utils import map_wavelength_to_RGB

from constants import EMISSION_MAP, METADATA_FILENAME, METADATA_ANNOTATED_FILENAME, VERBOSE

def collect_all_metadata(location: Path, prepare=True):
    """
    Collect all relevant metadata from mesospim Tile metadata files, sort by channel

    prepare=True will sort and annotate the data. This is the primary format used for all downstream applications
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
    if VERBOSE: print("Verifying that metadata files are present for each image file")
    acquisition_format = determine_acquisition_format(location)
    verify_metadata_file_for_each_image(location, acquisition_format)

    if VERBOSE: print("Collecting all metadata into json and saving the json")
    meta_list = save_dir_of_meta_to_json(location, save_json_path)

    if VERBOSE: print("Sorting metadata by channel and tile")
    metadata_by_channel = meta_list
    if prepare:
        metadata_by_channel = convert_paths(metadata_by_channel)
        metadata_by_channel = convert_str_to_nums(metadata_by_channel)
        metadata_by_channel = annotate_metadata(sort_meta_list(metadata_by_channel), location=location)

        print("Saving annotated metadata json")
        annotated_name = save_json_path.with_name(METADATA_ANNOTATED_FILENAME)
        dict_to_json_file(metadata_by_channel, annotated_name)

    if VERBOSE > 1: print(metadata_by_channel)

    return metadata_by_channel

def remove_max_projections(list_of_paths: list[Path]) -> list[Path]:
    list_of_paths = [ensure_path(x) for x in list_of_paths]
    return [entry for entry in list_of_paths if 'MAX_' not in entry.name]


def determine_acquisition_format(location):
    location = ensure_path(location)
    files = [entry for entry in location.iterdir() if entry.is_file()]  # List all files
    files = [entry for entry in files if '_meta.txt' not in entry.name]  # Remove _meta.txt files
    files = remove_max_projections(files)
    extensions = [x.suffix for x in files]
    if '.btf' in extensions:
        return '.btf'
    if '.h5' in extensions and '.xml' in extensions:
        return '.h5'
    if '.tiff' in extensions:
        return '.tiff'


def sort_meta_list(meta_list):
    '''
    take a list of dictionaries where each dictionary represents a metadata file from the mesospim.
    The meta_list is the output of function: save_dir_of_meta_to_json and read by function: json_file_to_dict(save_json_path)
    '''

    # Dictionary to store sorted data dynamically
    sorted_data = {}

    # Regex patterns to extract Tile number and Channel
    tile_pattern = re.compile(r"tile(\d+)")
    # channel_pattern = re.compile(r"_Ch(\d+)_")
    channel_pattern = re.compile(r"_ch(\d+)([a-zA-Z]*)_")

    # Process each entry
    for entry in meta_list:
        file_path = Path(entry["Metadata for file"])  # Convert to Path object
        file_name = file_path.name.lower()
        if VERBOSE > 1: print(f'{file_name=}')

        # Extract tile number
        tile_match = tile_pattern.search(file_name)  # Extract tile number from filename
        tile_number = int(tile_match.group(1)) if tile_match else None
        if VERBOSE > 1: print(f'{tile_number=}')

        # Extract channel number
        channel_match = channel_pattern.search(file_name)  # Extract channel wavelength from filename
        if VERBOSE > 1: print(f'{channel_match=}')
        if channel_match:
            channel_number = channel_match.group(1) + channel_match.group(2)  # e.g., '561b'
        else:
            channel_number = None
        if VERBOSE > 1: print(f'{channel_number=}')
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

    if VERBOSE > 1: print(sorted_data)

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

    ch = -1
    for color, data in metadata_by_channel.items():
        ch += 1

        # Collect grid size information to be appended to each metadata parameter
        grid_size = determine_grid_size(data)

        # Collect direction of stage movement to be appended to each metadata parameter
        stage_direction = get_stage_direction(data, grid_size)

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
            entry['stage_direction'] = stage_direction
            entry['overlap'] = overlap
            entry['resolution'] = determine_xyz_resolution(entry)
            entry['tile_shape'] = determine_tile_shape(entry)
            entry['tile_size_um'] = determine_tile_size_um(entry)
            entry['file_name'] = entry.get('Metadata for file').name
            entry['refractive_index'] = determine_refractive_index_from_ETL_file_name(entry)
            entry['sheet'] = determine_sheet_direction(entry)
            if tuple(entry.get('grid_location')) == (0,0) and ch == 0:
                entry['anchor_tile'] = True
            else:
                entry['anchor_tile'] = False

            ### NOTES ###
            # entry['tile_number'] is added by the sort_meta_list() function

            if location:
                entry['file_path'] = location / entry['file_name']

            entry['username'] = get_user(entry.get('file_path',"")) # Default "" if no entry['file_path']

    metadata_by_channel = get_affine_transform(metadata_by_channel)
    return metadata_by_channel


def get_stage_direction(channel_data, grid_size):
    # Outputs tuple (y,x) where y and x are 1 or -1,
    # 1 is that the stage positions are advancing in the positive direction
    # -1 is that the stage positions are advancing in the negative direction

    y, x = 1, 1
    if grid_size[0] > 1:
        t0 = channel_data[0]["POSITION"]["y_pos"]
        t1 = channel_data[1]["POSITION"]["y_pos"]
        if t1 < t0:
            y = -1
    if grid_size[1] > 1:
        t0 = channel_data[0]["POSITION"]["x_pos"]
        t1 = channel_data[grid_size[1]]["POSITION"]["x_pos"]
        if t1 < t0:
            x = -1
    StageDirection = namedtuple('StageDirection', ['y', 'x'])
    return StageDirection(y=y, x=x)


def get_anchor_tile_entry(metadata_by_channel):
    for _, data in metadata_by_channel.items():
        for entry in data:
            if entry.get('anchor_tile'):
                return entry


def get_affine_transform(metadata_by_channel):
    anchor_entry = get_anchor_tile_entry(metadata_by_channel)
    gy0, gx0 = anchor_entry.get('grid_location')
    z0 = anchor_entry["POSITION"]["z_start"]
    z_step = anchor_entry["POSITION"]["z_stepsize"]

    for _, data in metadata_by_channel.items():
        for entry in data:
            overlap = entry.get('overlap')
            gy, gx = entry.get('grid_location')
            z_start = entry["POSITION"]["z_start"]
            gx_rel = gx - gx0
            gy_rel = gy - gy0

            nz, ny, nx = entry.get('tile_shape')

            x_offset = gx_rel * nx * (1 - overlap)
            y_offset = gy_rel * ny * (1 - overlap)
            z_offset = (z_start - z0) / z_step

            # TODO confirm that maps to numpy (z,y,x) indexing
            affine_voxel_global = [
                [1, 0, 0, z_offset],
                [0, 1, 0, y_offset],
                [0, 0, 1, x_offset],
                [0, 0, 0, 1]
            ]
            entry['affine_voxel'] = affine_voxel_global

            rz, ry, rx = entry.get('resolution')

            #TODO confirm that maps to numpy (z,y,x) indexing
            affine_microns_global = [
                [1, 0, 0, z_offset * rz],
                [0, 1, 0, y_offset * ry],
                [0, 0, 1, x_offset * rx],
                [0, 0, 0, 1]
            ]
            entry['affine_microns'] = affine_microns_global
    return metadata_by_channel


def get_grid_location(grid_size, tile_number):
    GridLocation = namedtuple('GridLocation', ['y', 'x'])

    current_tile = 0
    for x in range(grid_size.x):
        for y in range(grid_size.y):
            if tile_number == current_tile:
                return GridLocation(y=y, x=x)
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
    etl_file_name = etl_file_name.lower()
    split_ri = etl_file_name.split('_ri_')[-1]
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

    TileSizeUm = namedtuple('TileSizeUm', ['z', 'y', 'x'])

    return TileSizeUm(z=shape.z * res.z, y=shape.y * res.y, x=shape.x * res.x)


def determine_tile_shape(metadata_entry):
    TileShape = namedtuple('TileShape', ['z', 'y', 'x'])

    x = metadata_entry['CAMERA PARAMETERS']['x_pixels']
    y = metadata_entry['CAMERA PARAMETERS']['y_pixels']
    z = metadata_entry['POSITION']['z_planes']

    return TileShape(z=z, y=y, x=x)


def determine_emission_wavelength(metadata_entry):
    # Extract 3 digit wavelength, if it doesn't exist reference EMISSION_MAP in the constants module
    emission_wavelength = metadata_entry['CFG']['Filter'][0:3]
    try:
        if emission_wavelength.lower() == 'emp': # function grabs first 3 characters, 'Empty' filter case
            emission_wavelength = None
        else:
            emission_wavelength = int(emission_wavelength)
    except Exception:
        assert emission_wavelength.lower() in EMISSION_MAP, f"No emission wavelength found for channel: {color}"
        emission_wavelength = EMISSION_MAP.get(emission_wavelength.lower())
    return emission_wavelength


def determine_xyz_resolution(metadata_entry):
    xy = metadata_entry["CFG"]["Pixelsize in um"]
    z = metadata_entry["POSITION"]["z_stepsize"]

    Resolution = namedtuple('Resolution', ['z', 'y', 'x'])
    return Resolution(z=z, y=xy, x=xy)


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
    data = single_channel_metadata

    # Extract unique x_pos and y_pos values
    x_positions = {entry['POSITION']["x_pos"] for entry in data}
    y_positions = {entry['POSITION']["y_pos"] for entry in data}

    # Calculate the grid size
    grid_size_x = len(x_positions)
    grid_size_y = len(y_positions)

    # print(f'{grid_size_x}X, {grid_size_y}Y')
    Grid_size = namedtuple('Grid_size', ['y', 'x'])

    return Grid_size(y=grid_size_y, x=grid_size_x)


def get_metadata_file_name(file: Path):
    '''
    Takes: a mesospim tile file name
    returns: metadata file name
    '''
    file = ensure_path(file)

    return file.with_name(file.name + "_meta.txt")


def find_metadata_dir(start_location):
    '''
    Given a path, search backwards to find mesospim metadata
    do not ascend all the way to root
    '''
    start_location = ensure_path(start_location)
    paths_to_test = [start_location] + list(start_location.parents)
    for current_location in paths_to_test[:-1]:  # Cut out that last entry which is root
        if VERBOSE: print(f'Searching for Metadata at {current_location}')
        if (current_location / METADATA_FILENAME).is_file():
            return current_location
        meta_files = list(current_location.glob("*_meta.txt"))
        if len(meta_files) > 0:
            return current_location
    return None


def verify_metadata_file_for_each_image(location: Path, file_ext: str):
    """
    Verify that a metadata file exist for each image tile file
    """

    location = ensure_path(location)

    assert location.is_dir(), 'Path is not a directory'

    files = list(location.glob(f"*{file_ext}"))
    files = remove_max_projections(files)

    # Verify a metadata for for each image
    for file in files:
        meta_file = get_metadata_file_name(file)
        # print(f"Checking {meta_file}")
        assert meta_file.is_file(), f"{meta_file} is missing"

    if VERBOSE: print('Verified a metadata file is present for each image file')


#####################################################################################################################
#####################################################################################################################

def parse_meta_file(filepath):
    """Parses a _meta.txt file into a structured dictionary."""
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
    """Reads all _meta.txt files, parses them, and saves to JSON."""
    meta_list = []

    for filename in os.listdir(directory):
        if filename.endswith("_meta.txt"):
            filepath = os.path.join(directory, filename)
            meta_list.append(parse_meta_file(filepath))

    dict_to_json_file(meta_list, output_json)
    return meta_list


def recreate_files_from_meta_dict(meta_list_from_json, output_directory):
    """Writes dictionary data back to _meta.txt files.
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
            f.write(file_content[:-1])  # Write contents after stripping the final \n


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
        for idx, entry in enumerate(meta_dict[ch]):
            if file_name.lower() == entry.get('file_name').lower():
                return ch, idx
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
    import typer
    app = typer.Typer()
    collect_all_metadata = app.command()(collect_all_metadata)
    app()

