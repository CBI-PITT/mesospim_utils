import re
from pathlib import Path
import json
import os
import psutil

from constants import EMISSION_TO_RGB

def map_wavelength_to_RGB(wavelength):
    '''
    Given a wavelength in nm, return a tuple containing (R,G,B) values.
    If the specific wavelength is not found, return white

    Ranges are color maps are defined in the EMISSION_TO_RGB object
    '''

    if wavelength is None:
        # Default to white
        return (0.5, 0.5, 0.5)

    for key, item in EMISSION_TO_RGB.items():
        low, high = [int(x) for x in key.split('-')]
        if wavelength >= low and wavelength < high:
            return item

    # Default to white
    return (0.5, 0.5, 0.5)

def sort_list_of_paths_by_tile_number(list_of_paths, pattern=r"_Tile(\d+)_"):
    files = [str(x) for x in list_of_paths]
    files.sort(key=lambda x: int(re.search(pattern, x).group(1)))
    return [Path(x) for x in files]

def dict_to_json_file(my_dict:dict, file_name:Path):

    class PathEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Path):
                return str(obj)
            return super().default(obj)

    if not isinstance(file_name, Path):
        file_name = Path(file_name)

    # Write dictionary to JSON file
    with open(file_name, "w") as json_file:
        json.dump(my_dict, json_file, indent=4, cls=PathEncoder)  # indent=4 makes it more readable

def read_json_file_to_dict(file_name:Path):
    # Read JSON file
    with open(file_name, "r") as json_file:
        data = json.load(json_file)
    data = convert_paths(data)
    return data


# Function to recursively convert path-like strings to Path obj in a nested dictionary
def convert_paths(obj):
    if isinstance(obj, dict):
        return {k: convert_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths(v) for v in obj]
    elif isinstance(obj, str) and ("/" in obj or "\\" in obj):  # Heuristic check for paths
        return Path(obj)
    return obj

def get_num_processors():
    return os.cpu_count()

def get_ram_mb():
    return psutil.virtual_memory().total // 1024 // 1024
