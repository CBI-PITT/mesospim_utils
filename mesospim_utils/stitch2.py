
'''
Stitch a mesospim dataset after conversion of all tiles to ims file.
Uses the ImarisStitcher and currently requires it to be install on a Windows computer
'''

location = r"I:\Acquire\MesoSPIM\zhang-l\4CL19\012425"
dir_with_ims_files = r"I:\Acquire\MesoSPIM\zhang-l\4CL19\012425\ims_files"

location = r"Z:\Acquire\MesoSPIM\cakir-i\4CL16\020425"
dir_with_ims_files = r"Z:\Acquire\MesoSPIM\cakir-i\4CL16\020425\ims_files"

location = r"Z:\tmp\mesospim\kidney"
dir_with_ims_files = r"Z:\tmp\mesospim\kidney\decon\ims_files"


# Std lib imports
from pathlib import Path
#from pprint import pprint as print
from datetime import datetime
import subprocess
import xml.etree.ElementTree as ET
import traceback
from time import sleep

# external package imports
from imaris_ims_file_reader.ims import ims
import typer

## Local imports
from metadata2 import collect_all_metadata
from utils import sort_list_of_paths_by_tile_number
from utils import map_wavelength_to_RGB
from utils import get_num_processors
from utils import ensure_path
from utils import dict_to_json_file, json_file_to_dict
from string_templates import WIN_ALIGN_BAT, WIN_RESAMPLE_BAT, COLOR_RECORD_TEMPLATE

from constants import CORRELATION_THRESHOLD_FOR_ALIGNMENT
from constants import SHARED_WINDOWS_PATH_WHERE_WIN_CLIENT_JOB_FILES_ARE_STORED as listen_path



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
        # x_pixels = sample_tile['CAMERA PARAMETERS']['x_pixels'] * sample_tile['POSITION']['x_res']
        # y_pixels = sample_tile['CAMERA PARAMETERS']['y_pixels'] * sample_tile['POSITION']['y_res']
        # z_pixels = sample_tile['POSITION']['z_planes'] * sample_tile['POSITION']['z_res']

        grid_x, grid_y = sample_tile['grid_size']

        # print(f'Tile Size: {resolution.x}x, {resolution.y}y, {resolution.z}z')
        # print(f'Grid Size: {grid_x}x, {grid_y}y')

        ## Build input file for alignment:
        input = f'<ImageList>\n'

        MinX = 0.0
        MinZ = 0.0
        MaxX = float(resolution.x)
        MaxZ = float(resolution.z)
        file_idx = 0
        for x in range(grid_x):
            MinY = 0.0
            MaxY = float(resolution.y)
            for y in range(grid_y):
                new = f'<Image MinX="{MinX:.6f}" MinY="{MinY:.6f}" MinZ="{MinZ:.6f}" MaxX="{MaxX:.6f}" MaxY="{MaxY:.6f}" MaxZ="{MaxZ:.6f}">{ims_files[file_idx]}</Image>\n'
                MinY = MinY + (resolution.y * (1-overlap))
                MaxY = MinY + resolution.y
                input = input + new
                file_idx += 1
            MinX = MinX + (resolution.x * (1-overlap))
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
    subprocess.run([bat_file], shell=True, check=True)
    return True


def parse_align_outputs(align_output_file_list):

    image_extends_list_final = []
    pairwise_alignment_list_final = []

    for align_output_file in align_output_file_list:
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


def get_moving_tile(pairwise_alignment_list, tile_num, return_highest_correlation=False, get_average_translation=True):
    moving_tiles_list = []

    for pairwise_alignment in pairwise_alignment_list:
        moving_tiles = []
        for entry in pairwise_alignment:
            tile_name = f'Tile{int(tile_num)}_'
            if tile_name in entry.get('ImageB'):
                moving_tiles.append(entry)
        print(moving_tiles)

        # Short circuit, if no matching tiles for ImageB, Extract ImageA name to return file_name
        if len(moving_tiles) == 0:
            print("No Alignments for this Tile")
            moving_tiles = []
            for entry in pairwise_alignment:
                tile_name = f'Tile{int(tile_num)}_'
                if tile_name in entry.get('ImageA'):
                    moving_tiles.append(entry)
            print(moving_tiles[0].get('ImageA'))
            return None, moving_tiles[0].get('ImageA')


        if return_highest_correlation and not get_average_translation:
            highest_index = 0
            for idx, tile in enumerate(moving_tiles):
                if tile.get('Correlation') > moving_tiles[highest_index].get('Correlation'):
                    highest_index = idx
            moving_tiles = [moving_tiles[highest_index]]
            #print(moving_tiles)

        # Build a weighted average translation based on correlation
        if get_average_translation:
            num_aligns = len(moving_tiles)
            sum_correlation = sum([x.get('Correlation') for x in moving_tiles])
            avg_coorelation = sum([x.get('Correlation')*(x.get('Correlation')/sum_correlation) for x in moving_tiles])
            print(f'Sum: {sum_correlation}, avg: {avg_coorelation}')

            x, y, z = 0,0,0
            for data in moving_tiles:
                translation = data.get('Translation')
                x += translation.get('x') * (data.get('Correlation') / sum_correlation)
                y += translation.get('y') * (data.get('Correlation') / sum_correlation)
                z += translation.get('z') * (data.get('Correlation') / sum_correlation)

            alignment_data = {
                "Correlation": avg_coorelation,
                "AverageAlign": True,
                "ImageB": data["ImageB"],
                "ImageA": [x["ImageA"] for x in moving_tiles],
                "Translation": {'x': x, 'y': y, 'z': z},
            }
            print(alignment_data)
            moving_tiles = alignment_data
            moving_tiles_list.append(moving_tiles)

    # Take the highest correlation alignment regardless of channel
    for tiles in moving_tiles_list:
        if tiles["Correlation"] > moving_tiles["Correlation"]:
            moving_tiles = tiles

    return moving_tiles, moving_tiles.get('ImageB')

def build_resample_input(image_extends_list, pairwise_alignment_list, metadata_by_channel, montage_name='montage.ims'):

    anchor_image = image_extends_list[0][0]
    ExtendMax = anchor_image.get('ExtendMax')

    x_pixels = ExtendMax.get('x')
    y_pixels = ExtendMax.get('y')
    z_pixels = ExtendMax.get('z')

    print(f'Tile Size: {x_pixels}x, {y_pixels}y, {z_pixels}z')

    grid_x, grid_y = get_grid_size_from_metadata_by_channel(metadata_by_channel)
    print(f'Grid Size 273: {grid_x}x, {grid_y}y')

    overlap = metadata_by_channel[list(metadata_by_channel.keys())[0]][0]['overlap']

    #moving_tiles = get_moving_tile(pairwise_alignment_list, 16, return_highest_correlation=True)

    ## Build input file for alignment:
    template = '<Image MinX="{}" MinY="{}" MinZ="{}" MaxX="{}" MaxY="{}" MaxZ="{}">{}</Image>\n'
    input = f'<ImageList>\n'

    MinX = 0.0
    MinZ = 0.0
    MaxX = float(x_pixels)
    MaxZ = float(z_pixels)
    file_idx = 0
    for _ in range(grid_x):
        MinY = 0.0
        MaxY = float(y_pixels)
        for _ in range(grid_y):

            moving_tiles, file_name = get_moving_tile(pairwise_alignment_list, file_idx, return_highest_correlation=False, get_average_translation=True)
            print(f'{moving_tiles=}  : {file_name=}')

            if moving_tiles and moving_tiles.get('Correlation') > CORRELATION_THRESHOLD_FOR_ALIGNMENT:
                translation = moving_tiles.get('Translation')
                MinX += translation.get('x')
                MaxX += translation.get('x')
                MinY += translation.get('y')
                MaxY += translation.get('y')
                MinZ += translation.get('z')
                MaxZ += translation.get('z')

            new = f'<Image MinX="{MinX:.6f}" MinY="{MinY:.6f}" MinZ="{MinZ:.6f}" MaxX="{MaxX:.6f}" MaxY="{MaxY:.6f}" MaxZ="{MaxZ:.6f}">{file_name}</Image>\n'
            print(new)
            MinY = MinY + (y_pixels * (1-overlap))
            MaxY = MinY + y_pixels
            input = input + new
            file_idx += 1
        MinX = MinX + (x_pixels * (1-overlap))
        MaxX = MinX + x_pixels
    input = input + '</ImageList>'

    output_folder = Path(file_name).parent
    resample_input_file_name = output_folder / 'resample_input.xml'

    with open(resample_input_file_name, 'w') as f:
        f.write(input)

    resample_bat_file = output_folder / 'resample.bat'
    output_resample_file = output_folder / montage_name

    color_file = create_color_file(image_extends_list, metadata_by_channel)
    resample_bat = WIN_RESAMPLE_BAT.format(resample_input_file_name, output_resample_file, color_file, get_num_processors())
    with open(resample_bat_file, 'w') as f:
        f.write(resample_bat)

    print(input)
    print(resample_bat)
    return resample_bat_file


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

    # Build files for calculating the alignment
    align_bat_file_list, align_output_file_list = build_align_inputs(metadata_by_channel, directory_with_ims_files_to_stitch)

    #Run alignment
    if not skip_align or not build_scripts_only:
        for align_bat_file in align_bat_file_list:
            print(f'Running alignment: {align_bat_file}')
            run_bat(align_bat_file)

    # Build files for resample
    image_extends_list, pairwise_alignment_list = parse_align_outputs(align_output_file_list)

    resample_bat_file = build_resample_input(image_extends_list, pairwise_alignment_list, metadata_by_channel,
                                             montage_name=name_of_montage_file)

    # Run resample to build montage
    if not skip_resample or not build_scripts_only:
        run_bat(resample_bat_file)


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

# auto_stitch_json_message = {
#     'directory_with_mesospim_metadata': Path(r'Z:\tmp\mesospim\kidney'),
#     'directory_with_ims_files_to_stitch': Path(r'Z:\tmp\mesospim\kidney\decon\ims_files'),
#     'name_of_montage_file': 'TEST_AUTO_MONTAGE.ims',
#     'skip_align': False,
#     'skip_resample': False,
#     'build_scripts_only': False,
#     'job_info': {
#         'number': None,
#         'started': None,
#         'ended': None,
#         'errored': None,
#         'error_message': None,
#         'traceback': None,
#     }
# }
@app.command()
def write_auto_stitch_message(metadata_location: Path, ims_files_location: Path, job_number: int, name_of_montage_file: str='auto_test_montage.ims',
                              skip_align: bool=False, skip_resample: bool=False, build_scripts_only: bool=False):

    message = auto_stitch_json_message
    message['directory_with_mesospim_metadata'] = metadata_location
    message['directory_with_ims_files_to_stitch'] = ims_files_location
    message['name_of_montage_file'] = name_of_montage_file
    message['skip_align'] = skip_align
    message['skip_resample'] = skip_resample
    message['build_scripts_only'] = build_scripts_only
    message['job_info']['number'] = job_number
    dict_to_json_file(message, Path(listen_path) / f'{job_number}.json')


@app.command()
def run_windows_auto_stitch_client(listen_path: Path=listen_path, seconds_between_checks: int=30,
               running_path: str='running', complete_path: str='complete', error_path: str='error'):
    '''
    A program that runs in windows with ImarisStitcher installed and looks for jobs files to run stitching
    '''

    ## For testing - write a message to disk at start.
    dict_to_json_file(auto_stitch_json_message, listen_path / 'test.json')

    running_path = listen_path / running_path
    complete_path = listen_path / complete_path
    error_path = listen_path / error_path

    # Create the subdirectories if they don't exist
    error_path.mkdir(parents=True, exist_ok=True)
    running_path.mkdir(parents=True, exist_ok=True)
    complete_path.mkdir(parents=True, exist_ok=True)


    while True:
        print(listen_path)
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
                stitch_and_assemble(**to_run)
                # sleep(5)
                #float('abc')
                # float('123')

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
    # grid_x, grid_y = determine_grid_size(metadata_by_channel)
    # print(f'Grid Size 348: {grid_x}x, {grid_y}y')
    # align_bat_file, align_output_file = build_align_input(metadata_by_channel, dir_with_ims_files)
    # run_bat(align_bat_file)
    # image_extends_list, pairwise_alignment_list = parse_align_output(align_output_file)
    # moving_tiles, file_name = get_moving_tile(pairwise_alignment_list, 5, return_highest_correlation=False, get_average_translation=True)
    # resample_bat_file = build_resample_input(image_extends_list, pairwise_alignment_list, metadata_by_channel, grid_x=grid_x, grid_y=grid_y, overlap=0.10, montage_name='TEST_OUT2.ims')
    # run_bat(resample_bat_file)

