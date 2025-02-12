

'''

Build stitcher image list

image_template = "<Image MinX="0.000000" MinY="0.000000" MinZ="0.000000" MaxX="3200.000000" MaxY="3200.000000" MaxZ="9500.000000">Z:/mesospim/081924/decon/ims_files/KIDNEY_Tile0_Ch561_Sh0.ims</Image>"

<ImageList>
<Image MinX="0.000000" MinY="0.000000" MinZ="0.000000" MaxX="3200.000000" MaxY="3200.000000" MaxZ="9500.000000">Z:/mesospim/081924/decon/ims_files/KIDNEY_Tile0_Ch561_Sh0.ims</Image>

pixel_size_x X and Y
percent_overlap

MinX: Origin 0.0000
Each new tile = pixel_size_x - (pixel_size_x*percent_overlap)
'''

location = r"Z:\Acquire\MesoSPIM\zhang-l\4CL19\011525"
location = r"Z:\tmp\mesospim"

from pathlib import Path
#from pprint import pprint as print

from metadata import collect_all_metadata

# Sort function for serpentine order with negative values
def serpentine_sort(entry):
    x_pos, y_pos = entry["x_pos"], entry["y_pos"]
    # Define row parity based on x_pos rounded to the nearest integer
    serpentine_y = y_pos if round(x_pos) % 2 == 0 else -y_pos
    return (x_pos, serpentine_y)

# Sort the list in-place
#data.sort(key=serpentine_sort)

def determine_grid_size(data):
    """
    Determines the grid size based on unique x_pos and y_pos values.
    Args:
        data (list of dict): List of dictionaries containing 'x_pos' and 'y_pos'.

    Returns:
        tuple: (number of unique x_pos, number of unique y_pos)
    """
    # Extract unique x_pos and y_pos values
    x_positions = {entry["x_pos"] for entry in data}
    y_positions = {entry["y_pos"] for entry in data}

    # Calculate the grid size
    grid_size_x = len(x_positions)
    grid_size_y = len(y_positions)

    print(f'{grid_size_x}X, {grid_size_y}Y')
    return grid_size_x, grid_size_y


OVERLAP = 0.1
def build_stitching_input_file(metadata_by_channel):

    # Determine grid shape
    for channel, metadata in metadata_by_channel.items():
        x, y = determine_grid_size(metadata)
        x_pixels = metadata[0]['x_pixels']
        y_pixels = metadata[0]['y_pixels']
        z_planes = metadata[0]['z_planes']
        break

    input_file = f"<ImageList>\n"

    idx = 0

    min_z, max_z = 0.0, z_planes

    x_increase = x_pixels - (x_pixels * OVERLAP)
    y_increase = y_pixels - (y_pixels * OVERLAP)

    min_x, max_x = 0.0, x_pixels  # Set origin for x
    for _ in range(x):
        min_y, max_y = 0.0, y_pixels  # Set origin for y
        for _ in range(y):
            file = metadata[idx]['file']
            file = file.parent / RELATIVE_PATH_TO_IMS / f"{file.stem + '.ims'}"
            image_info = f'<Image MinX="{min_x:.4f}" MinY="{min_y:.4f}" MinZ="{min_z:.4f}" MaxX="{max_x:.4f}" MaxY="{max_y:.4f}" MaxZ="{max_z:.4f}">{file}</Image>\n'
            input_file = f'{input_file}{image_info}'
            min_y += y_increase
            max_y += y_increase
            idx+=1
        min_x += x_increase
        max_x += x_increase

    input_file = f'{input_file}</ImageList>'
    return input_file


STITCHING_BIN = "C:/Program Files/Bitplane/ImarisStitcher 10.2.0/ImarisStitchAlignPairwise.exe"
RELATIVE_PATH_TO_IMS = r"decon/ims_files"

if __name__ == "__main__":
    # Get all metadata
    metadata_by_channel = collect_all_metadata(location)
    print(metadata_by_channel)
    layout = build_stitching_input_file(metadata_by_channel)
    print(layout)

    with open(f'{location}\input', 'w') as f:
        f.write(layout)


