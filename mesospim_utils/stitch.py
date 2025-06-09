
'''
Stitch a mesospim dataset after conversion of all tiles to ims file.
Uses the ImarisStitcher and currently requires it to be installed on a Windows computer
'''

# Std lib imports
from pathlib import Path
#from pprint import pprint as print
from datetime import datetime
from multiprocessing import Process
import subprocess
import xml.etree.ElementTree as ET
import traceback
from time import sleep
import os
from collections import namedtuple
import statistics
import numpy as np


# external package imports
from imaris_ims_file_reader.ims import ims
import typer
import json

## Local imports
from metadata import collect_all_metadata, get_first_entry
from utils import sort_list_of_paths_by_tile_number
from utils import map_wavelength_to_RGB
from utils import ensure_path
from utils import dict_to_json_file, json_file_to_dict
from utils import path_to_windows_mappings
from string_templates import WIN_ALIGN_BAT, WIN_RESAMPLE_BAT, COLOR_RECORD_TEMPLATE

from constants import CORRELATION_THRESHOLD_FOR_IMS_STITCHER
from constants import SHARED_WINDOWS_PATH_WHERE_WIN_CLIENT_JOB_FILES_ARE_STORED, SHARED_LINUX_PATH_WHERE_WIN_CLIENT_JOB_FILES_ARE_STORED

if os.name == 'nt':
    from constants import SHARED_WINDOWS_PATH_WHERE_WIN_CLIENT_JOB_FILES_ARE_STORED as listen_path
elif os.name == 'posix':
    from constants import SHARED_LINUX_PATH_WHERE_WIN_CLIENT_JOB_FILES_ARE_STORED as listen_path



def build_align_inputs(metadata_by_channel: dict, dir_with_ims_files: Path, overlap=0.10):
    if not isinstance(dir_with_ims_files, Path):
        dir_with_ims_files = Path(dir_with_ims_files)

    assert dir_with_ims_files.is_dir(), 'Path is not a directory'
    ims_files = list(dir_with_ims_files.glob('*Tile*.ims'))
    assert len(ims_files) >= 1, 'There are no Imaris files in this path'

    ims_files = sort_list_of_paths_by_tile_number(ims_files)
    print(ims_files)

    align_bat_file_list = []
    align_output_file_list = []
    for ch_idx in range(len(metadata_by_channel)):
        channel_data = list(metadata_by_channel.keys())[ch_idx]
        sample_tile = metadata_by_channel[channel_data][0]

        resolution = sample_tile['tile_size_um']

        grid_y, grid_x  = sample_tile['grid_size']
        stage_direction = sample_tile['stage_direction']


        ## Build input file for alignment:
        input = f'<ImageList>\n'

        MinX = 0.0
        MinZ = 0.0
        MaxX = float(resolution.x)
        MaxZ = float(resolution.z)
        # file_idx = 0
        # for x in range(grid_x):
        #     MinY = 0.0
        #     MaxY = float(resolution.y)
        #     for y in range(grid_y):
        #         new = f'<Image MinX="{MinX:.6f}" MinY="{MinY:.6f}" MinZ="{MinZ:.6f}" MaxX="{MaxX:.6f}" MaxY="{MaxY:.6f}" MaxZ="{MaxZ:.6f}">{ims_files[file_idx]}</Image>\n'
        #         MinY = MinY + (resolution.y * (1-overlap))
        #         MaxY = MinY + resolution.y
        #         input = input + new
        #         file_idx += 1
        #     MinX = MinX + (resolution.x * (1-overlap))
        #     MaxX = MinX + resolution.x
        # input = input + '</ImageList>'

        for x in range(grid_x)[::stage_direction[1]]:
            MinY = 0.0
            MaxY = float(resolution.y)
            for y in range(grid_y)[::stage_direction[0]]:
                file_idx = (x*grid_x) + y
                new = f'<Image MinX="{MinX:.6f}" MinY="{MinY:.6f}" MinZ="{MinZ:.6f}" MaxX="{MaxX:.6f}" MaxY="{MaxY:.6f}" MaxZ="{MaxZ:.6f}">{ims_files[file_idx]}</Image>\n'
                MinY = MinY + (resolution.y * (1 - overlap))
                MaxY = MinY + resolution.y
                input = input + new
            MinX = MinX + (resolution.x * (1 - overlap))
            MaxX = MinX + resolution.x
        input = input + '</ImageList>'

        print(input)

        align_input_file = dir_with_ims_files / f'align_input_ch{ch_idx}.xml'
        align_output_file = dir_with_ims_files / f'align_output_ch{ch_idx}.xml'
        align_bat_file = dir_with_ims_files / f'align_ch{ch_idx}.bat'
        with open(align_input_file,'w') as f:
            f.write(input)

        align_bat = WIN_ALIGN_BAT.format(align_input_file,align_output_file, ch_idx)
        with open(align_bat_file, 'w') as f:
            f.write(align_bat)

        align_bat_file_list.append(align_bat_file)
        align_output_file_list.append(align_output_file)

    return align_bat_file_list, align_output_file_list

def run_bat(bat_file):
    IDLE_PRIORITY_CLASS = subprocess.IDLE_PRIORITY_CLASS # 0x00000040
    BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
    subprocess.run([bat_file], shell=True, check=True, creationflags=BELOW_NORMAL_PRIORITY_CLASS)
    return True


def parse_align_outputs(align_output_file_list):

    image_extends_list_final = []
    pairwise_alignment_list_final = []

    for align_output_file in align_output_file_list:
        align_output_file = ensure_path(align_output_file)

        if not align_output_file.exists():
            print(f'File does not exist: {align_output_file}')
            continue

        tree = ET.parse(align_output_file)
        root = tree.getroot()

        # Extract ImageExtendsList
        image_extends_list = []

        for image_extends in root.findall(".//ImageExtendsList/ImageExtends"):
            image_data = {
                "Image": image_extends.attrib["Image"],
                "ExtendMin": dict(zip(["x", "y", "z"], map(float, image_extends.attrib["ExtendMin"].split()))),
                "ExtendMax": dict(zip(["x", "y", "z"], map(float, image_extends.attrib["ExtendMax"].split()))),
            }
            image_extends_list.append(image_data)

        print(image_extends_list)

        # Extract PairwiseAlignmentList
        pairwise_alignment_list = []

        for pairwise_alignment in root.findall(".//PairwiseAlignmentList/PairwiseAlignment"):
            alignment_data = {
                "Correlation": float(pairwise_alignment.attrib["Correlation"]),
                "ImageB": pairwise_alignment.attrib["ImageB"],
                "ImageA": pairwise_alignment.attrib["ImageA"],
                "Translation": dict(zip(["x", "y", "z"], map(float, pairwise_alignment.attrib["Translation"].split()))),
            }
            pairwise_alignment_list.append(alignment_data)
        print(pairwise_alignment_list)

        image_extends_list_final.append(image_extends_list)
        pairwise_alignment_list_final.append(pairwise_alignment_list)

    return image_extends_list_final, pairwise_alignment_list_final


def get_moving_tile(pairwise_alignment_list, tile_num, return_highest_correlation=False, get_average_translation=False):
    moving_tiles = []

    for pairwise_alignment in pairwise_alignment_list:
        for entry in pairwise_alignment:
            tile_name = f'Tile{int(tile_num)}_'
            if tile_name in entry.get('ImageB'):
                moving_tiles.append(entry)
        print(moving_tiles)

    # If no matching tiles for ImageB, Extract ImageA name to return file_name
    if len(moving_tiles) == 0:
        print("No Alignments for this Tile")
        for entry in pairwise_alignment:
            tile_name = f'Tile{int(tile_num)}_'
            if tile_name in entry.get('ImageA'):
                moving_tiles.append(entry)
        print(moving_tiles[0].get('ImageA'))
        return [], moving_tiles[0].get('ImageA')

    return moving_tiles, moving_tiles[0].get('ImageB')

def get_extends_lines(image_extends_list):
    '''
    Generator that takes image_extends_list and yields data for lines in resample input file
    '''
    image = namedtuple('Extends', ['channel','file','MinX','MinY','MinZ','MaxX','MaxY','MaxZ'])
    for ch_num, channel in enumerate(image_extends_list):
        for ims_file in channel:
            extends_min = ims_file.get('ExtendMin')
            extends_max = ims_file.get('ExtendMax')

            current = image(ch_num,
                ims_file.get('Image'),
                extends_min.get('x'),
                extends_min.get('y'),
                extends_min.get('z'),
                extends_max.get('x'),
                extends_max.get('y'),
                extends_max.get('z'),
            )
            yield current

def build_resample_input(image_extends_list, pairwise_alignment_list,
                         metadata_by_channel, montage_name='montage.ims'):

    anchor_image = image_extends_list[0][0]
    ExtendMax = anchor_image.get('ExtendMax')

    x_pixels = ExtendMax.get('x')
    y_pixels = ExtendMax.get('y')
    z_pixels = ExtendMax.get('z')

    print(f'Tile Size: {x_pixels}x, {y_pixels}y, {z_pixels}z')

    grid_y, grid_x  = get_grid_size_from_metadata_by_channel(metadata_by_channel)
    print(f'Grid Size 273: {grid_x}x, {grid_y}y')

    # Collect down alignment
    downs = {'x':[],'y':[],'z':[]}
    overs = {'x':[],'y':[],'z':[]}
    all_downs = {'x': [], 'y': [], 'z': []}
    all_overs = {'x': [], 'y': [], 'z': []}
    for tile_num in range(grid_y * grid_x): # Go through all potential tile numbers
        # Find any moving tiles associated with each tile
        moving_tiles, file_name = get_moving_tile(pairwise_alignment_list, tile_num, return_highest_correlation=False,
                                                  get_average_translation=False)

        tile_name_previous = f'Tile{int(tile_num - 1)}_'
        print(f'{tile_name_previous=}')
        for tile in moving_tiles:
            if tile_name_previous in tile.get('ImageA'): # Find alignment for the previous Tile i.e. Tile5 aligned with Tile4
                # if tile.get('Correlation') >= 0:
                translation = tile.get('Translation')
                all_downs['x'].append(translation.get('x'))
                all_downs['y'].append(translation.get('y'))
                all_downs['z'].append(translation.get('z'))
                if tile.get('Correlation') >= CORRELATION_THRESHOLD_FOR_IMS_STITCHER:
                    downs['x'].append(translation.get('x'))
                    downs['y'].append(translation.get('y'))
                    downs['z'].append(translation.get('z'))
            else: # If not previous Tile then it must be a sideways alignment
                # if tile.get('Correlation') >= 0:
                translation = tile.get('Translation')
                all_overs['x'].append(translation.get('x'))
                all_overs['y'].append(translation.get('y'))
                all_overs['z'].append(translation.get('z'))
                if tile.get('Correlation') >= CORRELATION_THRESHOLD_FOR_IMS_STITCHER:
                    overs['x'].append(translation.get('x'))
                    overs['y'].append(translation.get('y'))
                    overs['z'].append(translation.get('z'))

    print(f'{overs=}')
    print(f'{downs=}')

    any_overs = [True if len(y) > 0 else False for _, y in overs.items()]
    overs = {x:y if len(y) > 0 else [0] for x,y in overs.items()} # replace empty lists with 0
    overs = {x:statistics.median(y) for x, y in overs.items()} # replace lists with median of values

    any_downs = [True if len(y) > 0 else False for _, y in downs.items()]
    downs = {x: y if len(y) > 0 else [0] for x, y in downs.items()}  # replace empty lists with 0
    downs = {x: statistics.median(y) for x, y in downs.items()}  # replace lists with median of values

    # Add Calculated offsets only to be saved to a file
    # If True, the values were derived by the ImarisStitcher
    # If False, it implies that the values defaulted to 0,0,0 because no data was retrieved from Stitcher
    # If False, most likely no values had a Correlation >= the CORRELATION_THRESHOLD_FOR_IMS_STITCHER
    if any(any_overs):
        overs['Calculated offsets'] = True
    else:
        overs['Calculated offsets'] = False

    if any(any_downs):
        downs['Calculated offsets'] = True
    else:
        downs['Calculated offsets'] = False

    ## Build x,y,z coordinate grids
    metadata_entry = get_first_entry(metadata_by_channel)
    overlap = metadata_entry.get('overlap') # Proportion i.e. 0.1
    tile_size_um = metadata_entry.get('tile_size_um') # namedtuple i.e TileSizeUm(x=3200, y=3200, z=9500.0)

    x_min = np.zeros((grid_x, grid_y))
    y_min = np.zeros((grid_x, grid_y))
    z_min = np.zeros((grid_x, grid_y))

    for x in range(grid_x):
        for y in range(grid_y):
            print(f'Coords for: {(x,y)}')
            x_min[x, y] = (x * (tile_size_um.x * (1-overlap))) + (y * downs.get('x')) + (x * overs.get('x'))
            y_min[x, y] = (y * (tile_size_um.y * (1-overlap))) + (y * downs.get('y')) + (x * overs.get('y'))

            z_min[x, y] = 0 + (x * overs.get('z')) + (y * downs.get('z'))

    x_max = x_min + tile_size_um.x
    y_max = y_min + tile_size_um.y
    z_max = z_min + tile_size_um.z


    input = f'<ImageList>\n'
    tile_num = 0
    for x in range(grid_x):
        for y in range(grid_y):
            _ , file_name = get_moving_tile(pairwise_alignment_list, tile_num,
                                                      return_highest_correlation=False,
                                                      get_average_translation=False)
            new = f'<Image MinX="{x_min[x, y]:.6f}" MinY="{y_min[x, y]:.6f}" MinZ="{z_min[x, y]:.6f}" MaxX="{x_max[x, y]:.6f}" MaxY="{y_max[x, y]:.6f}" MaxZ="{z_max[x, y]:.6f}">{file_name}</Image>\n'
            input += new
            tile_num += 1
    input += '</ImageList>'
    print(input)

    output_folder = Path(file_name).parent
    resample_input_file_name = output_folder / 'resample_input.xml'

    with open(resample_input_file_name, 'w') as f:
        f.write(input)

    # Record the offsets in downs and overs.
    resample_tile_offsets_file = output_folder / 'resample_median_tile_offsets_microns.txt'
    offsets = {
        'overs': overs,
        'downs': downs,
        'app': 'ImarisStitcher'
    }
    # Write to a JSON file
    with open(resample_tile_offsets_file, 'w') as f:
        json.dump(offsets, f, indent=4)

    # Record the all offsets in all_downs and all_overs.
    resample_all_tile_offsets_file = output_folder / 'resample_all_tile_offsets_microns.txt'
    all_offsets = {
        'overs': all_overs,
        'downs': all_downs,
        'app': 'ImarisStitcher'
    }
    # Write to a JSON file
    with open(resample_all_tile_offsets_file, 'w') as f:
        json.dump(all_offsets, f, indent=4)

    resample_bat_file = output_folder / 'resample.bat'
    output_resample_file = output_folder / montage_name

    montage_name_tmp = montage_name + '.part'
    output_resample_file_tmp = output_folder / montage_name_tmp

    color_file = create_color_file(image_extends_list, metadata_by_channel)
    resample_bat = WIN_RESAMPLE_BAT.format(resample_input_file_name, output_resample_file_tmp, color_file)

    # resample_bat += f'\n\nif [ -f "{output_resample_file_tmp}" ]; then\n  mv "{output_resample_file_tmp}" "{output_resample_file}"\n  echo "File renamed to {output_resample_file}"\nelse\n  echo "File {output_resample_file_tmp} does not exist."\nfi'

    with open(resample_bat_file, 'w') as f:
        f.write(resample_bat)

    print(input)
    print(resample_bat)
    return resample_bat_file, output_resample_file_tmp, output_resample_file


def create_color_file(image_extends_list, metadata_by_channel):

    COLOR_RECORD_TEMPLATE = '''<Channel ChannelIndex="Channel {}" Selection="true" RangeMax="{}" RangeMin="{}" GammaCorrection="1" Opacity="1" ColorMode="BaseColor" RangeMinB="3.40282e+38" RangeMaxB="3.40282e+38">
    <ColorTable/>
    <BaseColor>
    <Color Red="{}" Green="{}" Blue="{}"/>
    </BaseColor>
    </Channel>\n'''

    color_file = image_extends_list[0][0].get('Image')
    color_file = Path(color_file).parent / 'resample_color_file.xml'

    # Open all ims files and gran the hist_max and hist_min, take an average and use it for the colors
    histo_max = [0]
    histo_min = [0]
    num_ims_files = len(image_extends_list[0])
    for entry in image_extends_list[0]:
        imaris_obj = ims(entry.get('Image'))
        metadata_dict = imaris_obj.metaData
        channels = imaris_obj.Channels
        resolution_levels = imaris_obj.ResolutionLevels
        if len(histo_max) != imaris_obj.Channels:
            histo_max = histo_max * channels
            histo_min = histo_min * channels
        del(imaris_obj)

        for ch in range(channels):
            sum_hist_max = sum([value for key,value in metadata_dict.items() if (key[-1] == 'HistogramMax' and key[2] == ch)])
            mean_hist_max = sum_hist_max / resolution_levels
            histo_max[ch] += mean_hist_max

            sum_hist_min = sum([value for key, value in metadata_dict.items() if (key[-1] == 'HistogramMin' and key[2] == ch)])
            mean_hist_max = sum_hist_min / resolution_levels
            histo_min[ch] += mean_hist_max

    for ch in range(channels):
        histo_max[ch] /= num_ims_files
        histo_min[ch] /= num_ims_files

    # Reduce the histo_max to 75% to increase the contrast for better presentation
    histo_max = [x*0.75 for x in histo_max]

    histo_max = [int(x) for x in histo_max]
    histo_min = [int(x) for x in histo_min]
    print(f'HistogrmaMax: {histo_max}')
    print(f'HistogrmaMin: {histo_min}')

    wavelengths = list(metadata_by_channel.keys())
    wavelengths = [metadata_by_channel[x][0]["emission_wavelength"] for x in wavelengths]
    wavelengths = [int(x) for x in wavelengths]

    # Build the channel xml
    text = '<Channels>\n'
    for ch in range(channels):
        rgb = map_wavelength_to_RGB(wavelengths[ch])
        text = text + COLOR_RECORD_TEMPLATE.format(ch, histo_max[ch], histo_min[ch], rgb[0], rgb[1], rgb[2])
    text = text + '</Channels>'
    print(text)

    with open(color_file, 'w') as f:
        f.write(text)

    return color_file

def get_grid_size_from_metadata_by_channel(metadata_by_channel):
    first_channel = list(metadata_by_channel.keys())[0]
    sample_tile = metadata_by_channel[first_channel][0]
    return sample_tile['grid_size']



app = typer.Typer()
@app.command()
def stitch_and_assemble(directory_with_mesospim_metadata: Path, directory_with_ims_files_to_stitch: Path = None,
                       name_of_montage_file: str = 'montage.ims',
                       skip_align: bool = False, skip_resample: bool = False, build_scripts_only: bool = False):
    '''
    Align and assemble MesoSPIM data
    directory_with_ims_files_to_stitch: should be provided otherwise it will default to the relative path /ims_files
    '''

    directory_with_mesospim_metadata = ensure_path(directory_with_mesospim_metadata)
    if directory_with_ims_files_to_stitch:
        directory_with_ims_files_to_stitch = ensure_path(directory_with_ims_files_to_stitch)
    else:
        directory_with_mesospim_metadata = directory_with_mesospim_metadata.parent / 'ims_files'

    # Collect all metadata from MesoSPIM acquisition directory and save to mesospim_metadata.json in the ims file dir
    metadata_by_channel = collect_all_metadata(directory_with_mesospim_metadata)
    overlap = get_first_entry(metadata_by_channel).get('overlap')

    # Build files for calculating the alignment
    align_bat_file_list, align_output_file_list = build_align_inputs(metadata_by_channel,
                                                                     directory_with_ims_files_to_stitch,
                                                                     overlap=overlap)

    #Run alignment
    if not skip_align or not build_scripts_only:
        for align_bat_file in align_bat_file_list:
            print(f'Running alignment: {align_bat_file}')
            run_bat(align_bat_file)

    # Build files for resample
    image_extends_list, pairwise_alignment_list = parse_align_outputs(align_output_file_list)

    resample_bat_file, output_resample_file_tmp, output_resample_file = build_resample_input(image_extends_list, pairwise_alignment_list, metadata_by_channel,
                                             montage_name=name_of_montage_file)

    # Run resample to build montage
    if not skip_resample or not build_scripts_only:
        run_bat(resample_bat_file)
        # tmp_name = name_of_montage_file + '.part'
        # tmp_name = directory_with_ims_files_to_stitch / tmp_name
        # new_name = directory_with_ims_files_to_stitch / name_of_montage_file
        os.rename(output_resample_file_tmp, output_resample_file)


auto_stitch_json_message = {
    'directory_with_mesospim_metadata': None,
    'directory_with_ims_files_to_stitch': None,
    'name_of_montage_file': None,
    'skip_align': False,
    'skip_resample': False,
    'build_scripts_only': False,
    'job_info': {
        'number': None,
        'started': None,
        'ended': None,
        'errored': None,
        'error_message': None,
        'traceback': None,
    }
}

@app.command()
def write_auto_stitch_message(metadata_location: Path, ims_files_location: Path, job_number: int, name_of_montage_file: str=None,
                              skip_align: bool=False, skip_resample: bool=False, build_scripts_only: bool=False):

    if name_of_montage_file is None:
        from constants import MONTAGE_NAME as name_of_montage_file

    message = auto_stitch_json_message
    message['directory_with_mesospim_metadata'] = path_to_windows_mappings(metadata_location) if not os.name == 'nt' else metadata_location
    message['directory_with_ims_files_to_stitch'] = path_to_windows_mappings(ims_files_location) if not os.name == 'nt' else ims_files_location
    message['name_of_montage_file'] = name_of_montage_file
    message['skip_align'] = skip_align
    message['skip_resample'] = skip_resample
    message['build_scripts_only'] = build_scripts_only
    message['job_info']['number'] = job_number
    dict_to_json_file(message, Path(listen_path) / f'{job_number}.json')


@app.command()
def run_windows_auto_stitch_client(listen_path: Path=listen_path, seconds_between_checks: int=60,
               running_path: str='running', complete_path: str='complete', error_path: str='error'):
    '''
    A program that runs in windows with ImarisStitcher installed and looks for jobs files to run stitching
    '''

    ## For testing - write a message to disk at start.
    # dict_to_json_file(auto_stitch_json_message, listen_path / 'test.json')

    import gc

    running_path = listen_path / running_path
    complete_path = listen_path / complete_path
    error_path = listen_path / error_path

    # Create the subdirectories if they don't exist
    error_path.mkdir(parents=True, exist_ok=True)
    running_path.mkdir(parents=True, exist_ok=True)
    complete_path.mkdir(parents=True, exist_ok=True)


    while True:
        print(f'Look for stitch jobs at: {listen_path}')
        list_of_job_files = list(listen_path.glob('*.json'))

        if len(list_of_job_files) == 0:
            print(f'No jobs found')

        else:
            try:
                error = False
                print(list_of_job_files)
                current_location_of_message = list_of_job_files[0]
                message = json_file_to_dict(current_location_of_message)
                print(f'Running the following stitch job: {message}')

                # Make names of files
                err_location = error_path / current_location_of_message.name
                running_location = running_path / current_location_of_message.name
                complete_location = complete_path / current_location_of_message.name
                data_dir_file_name = message['directory_with_ims_files_to_stitch'] / current_location_of_message.name


                # Copy json to 'running' folder
                current_location_of_message.replace(running_location)
                current_location_of_message = running_location

                # Add start time stamp and rewrite to disk
                message['job_info']['started'] = datetime.now()
                dict_to_json_file(message, current_location_of_message)
                dict_to_json_file(message, data_dir_file_name)

                # Prepare parameters to be passed to alignment program
                to_run = message.copy()
                del to_run['job_info']

                ## RUN
                # Create the process
                # Perform garbage collection to clean up memory
                gc.collect()
                p = Process(target=stitch_and_assemble, kwargs=to_run)
                p.start()
                print(f'Processing: {data_dir_file_name}')
                p.join()
                gc.collect()
                if p.exitcode != 0:
                    raise Exception("This process exited with a non-zero code")

                # Fill in processing details
                message['job_info']['ended'] = datetime.now()
                message['job_info']['errored'] = False

            except Exception as e:
                error = True
                message['job_info']['errored'] = datetime.now()
                message['job_info']['error_message'] = str(e)
                message['job_info']['traceback'] = traceback.format_exc()
                print(traceback.format_exc())

            # Save to disk in completed directory and in directory_with_ims_files_to_stitch
            dict_to_json_file(message, current_location_of_message)

            try:
                dict_to_json_file(message, data_dir_file_name)
            except:
                pass

            if error:
                current_location_of_message.replace(err_location)
            else:
                current_location_of_message.replace(complete_location)

        print(f'Waiting {seconds_between_checks} seconds before checking again')
        sleep(seconds_between_checks)

if __name__ == "__main__":
    app()

    # align_output = r'/CBI_FastStore/tmp/mesospim/kidney/ims_files/align_output_ch0.xml'
    # align_output_file_list = [align_output]
    #
    # metadata_dir = r'/CBI_FastStore/tmp/mesospim/kidney'
    # metadata_by_channel = collect_all_metadata(metadata_dir)
    #
    # # Build files for resample
    # image_extends_list, pairwise_alignment_list = parse_align_outputs(align_output_file_list)
    #
    # moving_tiles, file_name = get_moving_tile(pairwise_alignment_list, 1, return_highest_correlation=False,
    #                                           get_average_translation=False)
    #
    # x_cord, y_cord, z_cord = build_resample_input(image_extends_list, pairwise_alignment_list, metadata_by_channel, montage_name='montage.ims')

    # app()
    #
    # locations = [
    #     r'Z:\Acquire\MesoSPIM\zhao-y\4CL24\021225',
    #     r'Z:\Acquire\MesoSPIM\zhao-y\4CL24\020725',
    #     r'Z:\Acquire\MesoSPIM\zhao-y\4CL24\020625',
    #
    # ]
    #
    # ims_locations = [
    #     r'Z:\Acquire\MesoSPIM\zhao-y\4CL24\021225\ims_files',
    #     r'Z:\Acquire\MesoSPIM\zhao-y\4CL24\020725\ims_files',
    #     r'Z:\Acquire\MesoSPIM\zhao-y\4CL24\020625\ims_files',
    #
    # ]
    #
    # name_of_montage_file = 'montage_auto_2_channel.ims'
    #
    # for l,i in zip(locations, ims_locations):
    #     stitch_as_assemble(l, i, name_of_montage_file=name_of_montage_file)


    # metadata_by_channel = collect_all_metadata(location, save_json_path=dir_with_ims_files + r'\mesospim_metadata.json')
    # print(metadata_by_channel)
    # grid_y, grid_x = determine_grid_size(metadata_by_channel)
    # print(f'Grid Size 348: {grid_x}x, {grid_y}y')
    # align_bat_file, align_output_file = build_align_input(metadata_by_channel, dir_with_ims_files)
    # run_bat(align_bat_file)
    # image_extends_list, pairwise_alignment_list = parse_align_output(align_output_file)
    # moving_tiles, file_name = get_moving_tile(pairwise_alignment_list, 5, return_highest_correlation=False, get_average_translation=True)
    # resample_bat_file = build_resample_input(image_extends_list, pairwise_alignment_list, metadata_by_channel, grid_x=grid_x, grid_y=grid_y, overlap=0.10, montage_name='TEST_OUT2.ims')
    # run_bat(resample_bat_file)

