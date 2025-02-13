from pathlib import Path
from collections import defaultdict
from pprint import pprint as print
import re
import os

from utils import dict_to_json_file, json_file_to_dict, convert_paths_to_posix

from constants import EMISSION_MAP, METADATA_FILENAME

def collect_all_metadata(location: Path, save_json_path:Path = None):
    """
    Collect all relevant metadata from mesospim Tile metadata files, sort by channel

    location: a directory where metadata files for mesospim are located
    save_json_path <optional>: a path where all collected metadata are saved as a single json
    """

    if not isinstance(location, Path):
        location = Path(location)

    if location.suffix.lower() == ".json":
        return json_file_to_dict(location)

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
    """
    This function collects the relevant metadata from a specific file

    return: dict
    """
    # Extract imaging metadata from mesospim metadata file

    if not isinstance(meta_file, Path):
        meta_file = Path(meta_file)

    assert meta_file.is_file(), 'Path is not a file'
    #print(meta_file)

    meta_dict = {}
    meta_dict['file'] = meta_file
    with meta_file.open('r') as f:
        metadata = f.readlines()

    # Extract 3 digit wavelength, if it doesn't exist reference EMISSION_MAP in the constants module
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
    '''
    Takes: a mesospim tile file name
    returns: metadata file name
    '''
    if not isinstance(btf_file, Path):
        btf_file = Path(btf_file)

    return btf_file.with_name(btf_file.name + "_meta.txt")

def group_btf_files_by_channel_sort_by_tile(location: Path):
    """
    Given path to mesospim btf files
    returns: dict with channel number as key formatted as a string, list of files in tile order 0-?? as value
    """

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

    return files_by_channel


def collect_metadata_for_each_btf(files_by_channel):

    for channel, files in files_by_channel.items():
        meta_files = [get_metadata_file_name(x) for x in files]
        files_by_channel[channel] = [mesospim_meta_data(x) for x in meta_files]

    return files_by_channel



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

    print('Verified a metadata file is present for each .btf file')

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

def recreate_files_from_meta_dict(meta_list_from_json, output_directory):
    """Writes dictionary data back to .btf_meta.txt files."""
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


if __name__ == "__main__":

    input_directory = r"Z:\tmp\mesospim\meta_test_data"  # Change to the folder where the files are stored
    output_json = fr"Z:\tmp\mesospim\meta_test_data\{METADATA_FILENAME}"
    output_directory = r"Z:\tmp\mesospim\meta_test_data\recreated_files"

    # Step 1: Save parsed data to JSON
    save_dir_of_meta_to_json(input_directory, output_json)

    # Step 2: Load the JSON back
    meta_list_from_json = json_file_to_dict(output_json)
    print(meta_list_from_json)


    # Step 3: Recreate the original files from the dictionary
    recreate_files_from_meta_dict(meta_list_from_json, output_directory)
