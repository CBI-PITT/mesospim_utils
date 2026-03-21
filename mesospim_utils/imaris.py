#!/h20/home/lab/miniconda3/envs/decon/bin/python -u

'''
This script encodes several commandline utilities for richardson-lucy deconvolution
of mesospim datasets and then conversion to ims files.

The conversion are automatically queued via CBI slurm cluster.

This script assumes that you will be running it from the lab account
wine must be installed and configured with the MAPPING below.

conda create -n decon python=3.12 -y
conda activate decon

pip install -e /location/of/library
'''

import typer
from pathlib import Path
import subprocess
import os
from typing import Union, Tuple

import numpy as np
import tifffile
import skimage
from skimage import img_as_float32, img_as_uint

from psf import get_psf
from metadata import collect_all_metadata

from utils import path_to_wine_mappings, rgb_to_hex
from constants import LOCATION_OF_MESOSPIM_UTILS_INSTALL, ENV_PYTHON_LOC
from constants import WINE_INSTALL_LOC, IMARIS_CONVERTER_LOC
from constants import SLURM_PARAMETERS_IMARIS_CONVERTER
from constants import IMS_CONVERTER_COMPRESSION_LEVEL

SLURM_PARTITION = SLURM_PARAMETERS_IMARIS_CONVERTER.get('PARTITION')
SLURM_CPUS = SLURM_PARAMETERS_IMARIS_CONVERTER.get('CPUS')
SLURM_JOB_LABEL = SLURM_PARAMETERS_IMARIS_CONVERTER.get('JOB_LABEL')
SLURM_RAM_MB = SLURM_PARAMETERS_IMARIS_CONVERTER.get('RAM_GB')

LOC_OF_THIS_SCRIPT = LOCATION_OF_MESOSPIM_UTILS_INSTALL / 'imaris.py'

# Append -u so unbuffered outputs scroll in realtime to slurm out files
ENV_PYTHON_LOC = f'{ENV_PYTHON_LOC} -u'

mesospim_root_application = f'{ENV_PYTHON_LOC} {LOCATION_OF_MESOSPIM_UTILS_INSTALL}'

app = typer.Typer()


@app.command()
def convert_ims(file: list[str], res: tuple[float, float, float] = (1, 1, 1), out_dir: Path=None,
                            after_slurm_jobs: list[str]=None, run_conversion: bool=True):
    '''
    Convert a single imaris file using a subprocess OR 
    if return_cmd=True, return a string command without running the subprocess
    
    Files will be written to disk including the FileSeriesLayout files and a .bat file to execute the conversions
    
    file = [file1.tif, file2.tif]

    output.bat:
    convert: file1.tif, file2.tif --> out.part
    rename_output: out.part --> out.ims
    '''

    res_z, res_y, res_x = res

    if not isinstance(file, list):
        file = [file]

    file = [Path(x) for x in file if not isinstance(x,Path)]

    ext = file[0].suffix
    inputformat = None # For .h5 files this stays None
    if ext == '.tif' or ext == '.tiff' or ext == '.btf':
        inputformat = 'TiffSeries'

    if not out_dir:
        out_dir = file[0].parent / 'ims_files'
    out_file = out_dir / (file[0].stem + '.ims')
    out_file.parent.mkdir(parents=True, exist_ok=True)

    log_location = out_file.parent / 'ims_convert_logs' / (out_file.stem + '.txt')
    log_location.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists():
        return None, log_location, out_dir

    if inputformat == 'TiffSeries':

        layout_path = out_file.parent / 'ims_convert_layouts' / (out_file.stem + '_layout.txt')
        layout_path.parent.mkdir(parents=True, exist_ok=True)

        line = f'<FileSeriesLayout>'
        for c, f in enumerate(file):
            line += f'\n<ImageIndex name="{path_to_wine_mappings(f)}" x="0" y="0" z="0" c="{c}" t="0"/>'
        line += '</FileSeriesLayout>'

        with layout_path.open("w") as f:
            f.write(line)

    # Add channel colors using mesospim metadata if available
    metadata_by_channel = collect_all_metadata(file[0])
    rgb_colors = []
    if metadata_by_channel:
        for channel in metadata_by_channel:
            channel_data = metadata_by_channel[channel]
            first_tile = channel_data[0]
            rgb = first_tile.get('rgb_representation', (0.5,0.5,0.5)) # Default to gray if no color info
            rgb_colors.append(rgb)

    # Main ims converter command
    lines = f'{WINE_INSTALL_LOC} "{IMARIS_CONVERTER_LOC}" --voxelsizex {res_x} --voxelsizey {res_y} --voxelsizez {res_z} -i "{path_to_wine_mappings(file[0])}" -o "{path_to_wine_mappings(out_file).as_posix() + ".part"}" {f' -il "{path_to_wine_mappings(layout_path)}"' if inputformat else ""} --logprogress --nthreads {SLURM_CPUS} --compression {IMS_CONVERTER_COMPRESSION_LEVEL} -ps {SLURM_RAM_MB * 1024} -of Imaris5 -a{f" --inputformat {inputformat}" if inputformat else ""}'

    # Change ims-file channel colors to match those from mesospim metadata
    # This runs the cmd-line utility defined in this same script that uses h5py to edit the .ims file after conversion.
    # It is important to run this after conversion because the converter will overwrite any pre-set colors.
    if len(rgb_colors) > 0:
        for c, rgb in enumerate(rgb_colors):
            lines += f'\n\n{mesospim_root_application}/imaris.py set-ims-channel-color "{out_file}.part" {c} {rgb[0]} {rgb[1]} {rgb[2]}'

    # BASH script if/then statements to rename the .ims.part file to .ims
    lines = lines + f'\n\nif [ -f "{out_file}.part" ]; then\n  mv "{out_file}.part" "{out_file}"\n  echo "File renamed to {out_file}"\nelse\n  echo "File {out_file} does not exist."\nfi'

    if run_conversion:
        print(lines)
        print('Running Conversion')
        subprocess.run(f'#!/bin/bash\n\n{lines}', shell=True, capture_output=True)
    else:
        print(lines)
        return lines, log_location, out_dir #This will be a str bash script that can be executed separately to do the conversion

@app.command()
def convert_ims_dir_mesospim_tiles(dir_loc: Path, file_type: str='.btf', res: tuple[float,float,float]=(1,1,1),
                                after_slurm_jobs: list[str]=None, run_conversion=True):
    '''
    Convert all files in a directory to Imaris Files [executed on SLURM]
    The function assumes this is mesospim data.
    Tiles are grouped and sorted by color using the function
    nested_list_tile_files_sorted_by_color

    names are expected to be as follows: _Tile{NUMBER}_ with a number representing laser line

    Currently, resolution must be provided manually.
    '''

    tiles = nested_list_tile_files_sorted_by_color(dir_loc=dir_loc, file_type=file_type)
    for ii in tiles:
        convert_ims(ii, res=res, run_conversion=run_conversion)


def nested_list_tile_files_sorted_by_color(dir_loc: Path, file_type: str='.tif'):
    ''' Given a directory, return a list of lists. Where each nested list is
    a represents 1 tile with all channel files, sorted by channel'''
    from natsort import natsorted
    print('In nested')
    if not isinstance(dir_loc, Path):
        path = Path(dir_loc)
    else:
        path = dir_loc
    file_list = path.glob('*' + file_type)
    file_list = [x.as_posix() for x in file_list]
    print(file_list)

    idx = 0
    tiles = []
    while idx is not None:
        to_find = f'_Tile{idx}_'
        current_tile = []
        for i in file_list:
            if to_find in i:
                current_tile.append(i)
        if len(current_tile) > 0:
            tiles.append(current_tile)
            idx += 1
        else:
            idx = None
    #print(tiles)
    tiles = [natsorted(x) for x in tiles]
    return tiles

@app.command()
def set_ims_channel_color(ims_path:Path, channel:int, rgb:Tuple[float,float,float], mode="BaseColor"):
    """
    ims_path: path to .ims
    channel: int (0-based)
    rgb: tuple/list of 3 floats in [0, 1] (r,g,b)
    mode: "BaseColor" or "TableColor"
    """

    import h5py

    def _as_char_array(s: str) -> np.ndarray:
        """
        Convert Python string to a numpy array of 1-byte chars matching the style
        Imaris uses: array([b'1', b'.', b'0', ...], dtype='|S1')
        """
        # Turn string into bytes, then into S1 array
        arr = np.frombuffer(s.encode("ascii"), dtype="S1")
        # Match shape if needed (usually 1-D exact length)
        return arr

    rgb = tuple([float(x) for x in rgb])
    r, g, b = rgb
    if not all(0.0 <= x <= 1.0 for x in (r, g, b)):
        raise ValueError("rgb values must be floats in [0,1]")

    ch_group = f"/DataSetInfo/Channel {channel}"

    # Imaris stores these as strings like: "1 0 0"
    color_str = f"{r:.3f} {g:.3f} {b:.3f}"
    mode_str = str(mode)

    with h5py.File(ims_path, "r+") as f:
        if ch_group not in f:
            raise KeyError(f"Channel group not found: {ch_group}")

        g = f[ch_group]
        g.attrs["Color"] = _as_char_array(color_str)           # e.g. "1 0 0" for red :contentReference[oaicite:1]{index=1}
        g.attrs["ColorMode"] = _as_char_array(mode_str)        # "BaseColor" or "TableColor" :contentReference[oaicite:2]{index=2}


if __name__ == "__main__":
    app()



