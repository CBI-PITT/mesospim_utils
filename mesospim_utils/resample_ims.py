
import numpy as np
import os
from time import sleep
from datetime import datetime
import traceback
from pathlib import Path
import subprocess

import typer
app = typer.Typer()

from imaris_ims_file_reader.ims import ims

from metadata import get_first_entry, determine_sheet_direction_from_tile_number, collect_all_metadata
from utils import (
    json_file_to_dict,
    ensure_path,
    path_to_windows_mappings,
    path_to_wine_mappings,
    map_wavelength_to_RGB,
    write_file,
    dict_to_json_file
    )
from constants import (
    ALIGNMENT_DIRECTORY,
    USE_SEPARATE_ALIGN_DATA_PER_SHEET,
    VERBOSE,
    ALIGN_ALL_OUTPUT_FILE_NAME,
    ALIGN_METRIC_OUTPUT_FILE_NAME
)
from string_templates import WIN_RESAMPLE_BAT


if os.name == 'nt':
    from constants import SHARED_WINDOWS_PATH_WHERE_WIN_CLIENT_JOB_FILES_ARE_STORED as listen_path
elif os.name == 'posix':
    from constants import SHARED_LINUX_PATH_WHERE_WIN_CLIENT_JOB_FILES_ARE_STORED as listen_path

listen_path = ensure_path(listen_path)

'''
Given a median tile offsets from align_utils
This module will handle resampling using imaris stitcher resampler
'''


def build_resample_offsets(directory_with_align_data: Path, metadata_by_channel: dict,
                           separate_sheet_stitch=USE_SEPARATE_ALIGN_DATA_PER_SHEET):

    median_tile_offsets = get_median_tile_offsets(directory_with_align_data)

    ## Build x,y,z coordinate grids
    metadata_entry = get_first_entry(metadata_by_channel)
    overlap = metadata_entry.get('overlap') # Proportion i.e. 0.1
    tile_size_um = metadata_entry.get('tile_size_um') # namedtuple i.e TileSizeUm(x=3200, y=3200, z=9500.0)
    grid_y, grid_x = metadata_entry.get('grid_size')

    x_min = np.zeros((grid_x, grid_y))
    y_min = np.zeros((grid_x, grid_y))
    z_min = np.zeros((grid_x, grid_y))

    tile = 0
    for x in range(grid_x):
        for y in range(grid_y):
            if VERBOSE == 2: print(f'Coords for: {(x,y)}')

            if separate_sheet_stitch:
                light_sheet_direction = determine_sheet_direction_from_tile_number(metadata_by_channel, tile)
            else:
                light_sheet_direction = 'both'

            if light_sheet_direction == 'left':
                overs = median_tile_offsets.get('overs').get('left')
                downs = median_tile_offsets.get('downs').get('left')

            elif light_sheet_direction == 'right':
                overs = median_tile_offsets.get('overs').get('right')
                downs = median_tile_offsets.get('downs').get('right')

            else:
                overs = median_tile_offsets.get('overs').get('both')
                downs = median_tile_offsets.get('downs').get('both')

            # Axes for overs/downs are inverted in x/y compared to imaris resampler.  The all x/y values are -x and -y
            x_min[x, y] = (x * (tile_size_um.x * (1-overlap))) + (y * -downs.get('x')) + (x * -overs.get('x'))
            y_min[x, y] = (y * (tile_size_um.y * (1-overlap))) + (y * -downs.get('y')) + (x * -overs.get('y'))
            z_min[x, y] = 0 + (x * overs.get('z')) + (y * downs.get('z'))
            tile += 1

    x_max = x_min + tile_size_um.x
    y_max = y_min + tile_size_um.y
    z_max = z_min + tile_size_um.z

    return x_min, x_max, y_min, y_max, z_min, z_max

def list_file_names_in_tile_order(all_tile_offsets: dict, format: str=None):
    tile_file_name_list = []

    if format == None:
        format = lambda x: ensure_path(x)
    elif format == 'wine':
        from utils import path_to_wine_mappings
        format = path_to_wine_mappings
    elif format == 'windows':
        from utils import path_to_windows_mappings
        format = path_to_windows_mappings

    combined_list_of_tile_alignments = all_tile_offsets['overs'] + all_tile_offsets['downs']

    tile_num = 0
    tile_template = '_Tile{}_'
    complete = False
    while not complete:

        current_tile = tile_template.format(tile_num)
        found = False

        for entry in combined_list_of_tile_alignments:

            if not found and current_tile in entry.get('fixed'):
                tile_file_name_list.append( format(entry.get('fixed')) )
                found = True
            elif not found and current_tile in entry.get('moving'):
                tile_file_name_list.append( format(entry.get('moving')) )
                found = True

        if not found:
            complete = True

        tile_num += 1

    return tile_file_name_list


def get_all_tile_offsets(directory_with_align_data):
    return json_file_to_dict(directory_with_align_data / ALIGN_ALL_OUTPUT_FILE_NAME)

def get_median_tile_offsets(directory_with_align_data):
    return json_file_to_dict(directory_with_align_data / ALIGN_METRIC_OUTPUT_FILE_NAME)


def build_ims_resample_input(x_min, x_max, y_min, y_max, z_min, z_max, directory_with_align_data, metadata_by_channel):
    all_tile_offsets = get_all_tile_offsets(directory_with_align_data)
    tile_file_name_list = list_file_names_in_tile_order(all_tile_offsets, format='windows')

    metadata_entry = get_first_entry(metadata_by_channel)
    grid_y, grid_x  = metadata_entry.get('grid_size')
    stage_direction = metadata_entry.get('stage_direction')

    input = f'<ImageList>\n'
    tile_num = 0
    for x in range(grid_x)[::stage_direction[1]]:
        for y in range(grid_y)[::stage_direction[0]]:
            file_name = tile_file_name_list[tile_num]
            new = f'<Image MinX="{x_min[x, y]:.6f}" MinY="{y_min[x, y]:.6f}" MinZ="{z_min[x, y]:.6f}" MaxX="{x_max[x, y]:.6f}" MaxY="{y_max[x, y]:.6f}" MaxZ="{z_max[x, y]:.6f}">{file_name}</Image>\n'
            input += new
            tile_num += 1
    input += '</ImageList>'
    return input

def create_color_file(directory_with_align_data, metadata_by_channel):

    if VERBOSE: print('Creating Color File')

    COLOR_RECORD_TEMPLATE = '''<Channel ChannelIndex="Channel {}" Selection="true" RangeMax="{}" RangeMin="{}" GammaCorrection="1" Opacity="1" ColorMode="BaseColor" RangeMinB="3.40282e+38" RangeMaxB="3.40282e+38">
    <ColorTable/>
    <BaseColor>
    <Color Red="{}" Green="{}" Blue="{}"/>
    </BaseColor>
    </Channel>\n'''

    color_file = ensure_path(directory_with_align_data) / 'resample_color_file.xml'

    all_tile_offsets = get_all_tile_offsets(directory_with_align_data)
    tile_file_name_list = list_file_names_in_tile_order(all_tile_offsets, format=None)
    # Ensure windows paths
    tile_file_name_list = [path_to_windows_mappings(x) for x in tile_file_name_list]

    # Open all ims files and gran the hist_max and hist_min, take an average and use it for the colors
    histo_max = [0]
    histo_min = [0]
    num_ims_files = len(tile_file_name_list)
    for entry in tile_file_name_list:
        imaris_obj = ims(entry)
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
    if VERBOSE == 2: print(f'HistogrmaMax: {histo_max}')
    if VERBOSE == 2: print(f'HistogrmaMin: {histo_min}')

    wavelengths = list(metadata_by_channel.keys())
    wavelengths = [metadata_by_channel[x][0]["emission_wavelength"] for x in wavelengths]
    wavelengths = [int(x) for x in wavelengths]

    # Build the channel xml
    text = '<Channels>\n'
    for ch in range(channels):
        rgb = map_wavelength_to_RGB(wavelengths[ch])
        text = text + COLOR_RECORD_TEMPLATE.format(ch, histo_max[ch], histo_min[ch], rgb[0], rgb[1], rgb[2])
    text = text + '</Channels>'
    if VERBOSE == 2: print(text)

    # with open(color_file, 'w') as f:
    #     f.write(text)

    return text


def generate_files_for_resample(directory_ims_tiles, directory_with_mesospim_metadata:Path=None, name_of_montage_file:str = None):

    directory_ims_tiles = ensure_path(directory_ims_tiles)
    directory_with_align_data = directory_ims_tiles / ALIGNMENT_DIRECTORY
    assert directory_with_align_data.is_dir(), f"Align directory {directory_with_align_data} does not exist"

    if not directory_with_mesospim_metadata:
        directory_with_mesospim_metadata = directory_ims_tiles
    directory_with_mesospim_metadata = ensure_path(directory_with_mesospim_metadata)

    mesospim_metadata = collect_all_metadata(directory_with_mesospim_metadata)

    resample_offsets = build_resample_offsets(directory_with_align_data, mesospim_metadata)

    resample_input = build_ims_resample_input(*resample_offsets, directory_with_align_data, mesospim_metadata)
    resample_input_file_name = directory_with_align_data / 'resample_input.xml'

    color_file = create_color_file(directory_with_align_data, mesospim_metadata)
    color_file_file_name = directory_with_align_data / 'resample_color_file.xml'

    if not name_of_montage_file:
        from constants import MONTAGE_NAME as name_of_montage_file
    output_resample_file_tmp = directory_ims_tiles / (name_of_montage_file + '.part')

    resample_bat_file_name = directory_with_align_data / 'resample.bat'

    # Same resample input and color files
    write_file(resample_input_file_name, resample_input)
    write_file(color_file_file_name, color_file)

    resample_bat = WIN_RESAMPLE_BAT.format(path_to_windows_mappings(resample_input_file_name),
                                           path_to_windows_mappings(output_resample_file_tmp),
                                           path_to_windows_mappings(color_file_file_name))

    write_file(resample_bat_file_name, resample_bat)

    return resample_bat_file_name, output_resample_file_tmp

def run_bat(bat_file):
    IDLE_PRIORITY_CLASS = subprocess.IDLE_PRIORITY_CLASS # 0x00000040
    BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
    subprocess.run([bat_file], shell=True, check=True, creationflags=BELOW_NORMAL_PRIORITY_CLASS)
    return True

def resample(directory_with_ims_files:Path=None, directory_with_mesospim_metadata:Path=None, name_of_montage_file=None):
    '''
    Intended to be run in windows to kick off the whole resampling process
    '''

    resample_bat_file_name, output_resample_file_tmp = generate_files_for_resample(directory_with_ims_files,
                                                         directory_with_mesospim_metadata=directory_with_mesospim_metadata, name_of_montage_file=name_of_montage_file)
    resample_bat_file_name = ensure_path(resample_bat_file_name)


    run_bat(resample_bat_file_name)

    if resample_bat_file_name.is_file():
        if not name_of_montage_file:
            from constants import MONTAGE_NAME as name_of_montage_file
        montage_file_name = directory_with_ims_files / name_of_montage_file
        os.rename(output_resample_file_tmp, montage_file_name)


auto_stitch_json_message = {
    'directory_with_mesospim_metadata': None,
    'directory_with_ims_files': None,
    'name_of_montage_file': None,
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
def write_auto_resample_message(metadata_location: Path, ims_files_location: Path,
                                job_number: int, name_of_montage_file: str=None):

    if name_of_montage_file is None:
        from constants import MONTAGE_NAME as name_of_montage_file

    message = auto_stitch_json_message
    message['directory_with_mesospim_metadata'] = path_to_windows_mappings(metadata_location) if not os.name == 'nt' else metadata_location
    message['directory_with_ims_files'] = path_to_windows_mappings(ims_files_location) if not os.name == 'nt' else ims_files_location
    message['name_of_montage_file'] = name_of_montage_file
    message['job_info']['number'] = job_number
    dict_to_json_file(message, listen_path / f'{job_number}.json')


@app.command()
def run_windows_auto_resample_client(listen_path: Path=listen_path, seconds_between_checks: int=60,
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
        print(f'Looking for stitch jobs at: {listen_path}')
        list_of_job_files = list(listen_path.glob('*.json'))

        if len(list_of_job_files) == 0:
            print(f'No jobs found')

        else:
            try:
                error = False
                if VERBOSE: print(list_of_job_files)
                current_location_of_message = list_of_job_files[0]
                message = json_file_to_dict(current_location_of_message)
                print(f'Running the following resample job: {message}')

                # Make names of files
                err_location = error_path / current_location_of_message.name
                running_location = running_path / current_location_of_message.name
                complete_location = complete_path / current_location_of_message.name
                data_dir_file_name = message['directory_with_ims_files'] / current_location_of_message.name


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
                print(f'Processing: {data_dir_file_name}')
                resample(**to_run)

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


if __name__ == '__main__':
    app()

    # directory_ims_tiles = f'/CBI_FastStore/tmp/mesospim/kidney/ims_files'
    # directory_ims_tiles = ensure_path(directory_ims_tiles)
    #
    # resample(directory_ims_tiles, directory_with_mesospim_metadata = None)

    # directory_with_align_data = directory_ims_tiles / ALIGNMENT_DIRECTORY
    #
    # directory_with_mesospim_metadata = f'/CBI_FastStore/tmp/mesospim/kidney'
    # directory_with_mesospim_metadata = ensure_path(directory_with_mesospim_metadata)
    #
    # mesospim_metadata = collect_all_metadata(directory_ims_tiles)
    #
    # resample_offsets = build_resample_offsets(directory_with_align_data, mesospim_metadata)
    #
    # resample_input = build_ims_resample_input(*resample_offsets, directory_with_align_data, mesospim_metadata)
    # resample_input_file_name = directory_with_align_data / 'resample_input.xml'
    # # print(resample_input)
    #
    # color_file = create_color_file(directory_with_align_data, mesospim_metadata)
    # color_file_file_name = directory_with_align_data / 'resample_color_file.xml'
    # # print(color_file)
    #
    # from constants import MONTAGE_NAME as name_of_montage_file
    # output_resample_file_tmp = directory_ims_tiles / (name_of_montage_file + '.part')
    #
    # resample_bat_file_name = directory_with_align_data / 'resample.bat'
    #
    #
    # # Same resample input and color files
    # write_file(resample_input_file_name, resample_input)
    # write_file(color_file_file_name, color_file)
    #
    # resample_bat = WIN_RESAMPLE_BAT.format(path_to_windows_mappings(resample_input_file_name), path_to_windows_mappings(output_resample_file_tmp), path_to_windows_mappings(color_file_file_name))
    # write_file(resample_bat_file_name, resample_bat)




