import re
from pathlib import Path
import json
import os
import psutil
from collections import namedtuple
from datetime import datetime

from constants import EMISSION_TO_RGB

def map_wavelength_to_RGB(wavelength):
    '''
    Given a wavelength in nm, return a tuple containing (R,G,B) values.
    If the specific wavelength is not found, return white

    Ranges are color maps are defined in the EMISSION_TO_RGB object
    '''

    RGB = namedtuple('RGB', ['R', 'G', 'B'])
    default = RGB(R=0.5, G=0.5, B=0.5)

    if wavelength is None:
        # Default to white
        return default

    for key, item in EMISSION_TO_RGB.items():
        low, high = [int(x) for x in key.split('-')]
        if wavelength >= low and wavelength < high:
            return RGB(*item)

    # Default to white
    return default

def sort_list_of_paths_by_tile_number(list_of_paths, pattern=r"_Tile(\d+)_"):
    files = [str(x) for x in list_of_paths]
    files.sort(key=lambda x: int(re.search(pattern, x).group(1)))
    return [Path(x) for x in files]

def dict_to_json_file(my_dict:dict, file_name:Path):
    '''
    Write dictionary to JSON ensure all Path objects are converted to posix style string
    '''

    class PathEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Path):
                return obj.as_posix()
            if isinstance(obj, datetime):
                return obj.isoformat()  # Or use str(obj) or a custom format with obj.strftime(...)
            return super().default(obj)

    if not isinstance(file_name, Path):
        file_name = Path(file_name)

    # Write dictionary to JSON file
    with open(file_name, "w") as json_file:
        json.dump(my_dict, json_file, indent=4, cls=PathEncoder)  # indent=4 makes it more readable

def json_file_to_dict(file_name:Path):
    '''
    Read JSON and convert all path like strings to Path obj
    '''
    # Read JSON file
    with open(file_name, "r") as json_file:
        data = json.load(json_file)
    # data = convert_paths(data)
    # data = convert_str_to_nums(data)
    data = recursive_convert_to_useable_objects(data)
    return data

############################################################################
###### Recursive functions to convert strings to useful objects ############
############################################################################

def convert_paths(obj):
    '''
    Recursively convert path-like strings to Path obj in a nested dictionary
    '''
    if isinstance(obj, dict):
        return {k: convert_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths(v) for v in obj]
    elif isinstance(obj, str) and (':' in obj) and ("/" in obj or "\\" in obj):  # Heuristic check for paths
        return Path(obj)
    return obj

def convert_datetimes(obj):
    '''
    Function to recursively convert Path objects to posix path strings in a nested dictionary
    '''
    if isinstance(obj, dict):
        return {k: convert_datetimes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_datetimes(v) for v in obj]
    elif isinstance(obj, str):
        try:
            return datetime.fromisoformat(obj)
        except ValueError:
            pass
    return obj

def convert_str_to_nums(obj):
    '''
    Function to recursively convert string objects to int or float
    '''
    if isinstance(obj, dict):
        return {k: convert_str_to_nums(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_str_to_nums(v) for v in obj]
    elif isinstance(obj, str):  # Heuristic check for paths
        try:
            return int(obj)
        except ValueError:
            try:
                return float(obj)
            except ValueError:
                pass
    return obj

def recursive_convert_to_useable_objects(obj):
    '''
    Function to recursively convert json strings to relevant objects
    i.e. int, float, Path, datetime

    It is inefficient at is serially calls separate recursive operations, but it works
    '''
    obj = convert_paths(obj)
    obj = convert_datetimes(obj)
    obj = convert_str_to_nums(obj)
    return obj




def convert_paths_to_posix(obj):
    '''
    Function to recursively convert Path objects to posix path strings in a nested dictionary
    Mostly used when
    '''
    if isinstance(obj, dict):
        return {k: convert_paths_to_posix(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths_to_posix(v) for v in obj]
    elif isinstance(obj, Path):  # Heuristic check for paths
        return obj.as_posix()
    return obj

def get_num_processors():
    '''
    Returns the number of cpus on system
    '''
    return os.cpu_count()

def get_ram_mb():
    '''
    Returns total RAM in MB on system
    '''
    return psutil.virtual_memory().total // 1024 // 1024

def ensure_path(file_name: Path):
    '''
    Returns a Path object of the input file_name
    '''
    if not isinstance(file_name, Path):
        return Path(file_name)
    return file_name

def path_to_wine_mappings(path: Path) -> Path:
    '''Convert a linux path to wine (windows) paths using MAPPINGS defined at top of script'''
    from constants import WINE_MAPPINGS
    if isinstance(path, Path):
        path = path.as_posix()
    for key in WINE_MAPPINGS:
        path = path.replace(key,WINE_MAPPINGS[key])
    path = path.replace('/','\\')
    print(path)
    return Path(path)