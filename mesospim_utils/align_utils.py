from pathlib import Path
import statistics
import re

from utils import ensure_path
from metadata import collect_all_metadata, get_all_tile_entries

'''
Any alignment in mesospim_utils is assumed to output a dictionary that represents offsets in microns
for tiles as they increase in the x-axis (overs), and y-axis (downs).  It is these median values that are used to 
create the arrangement for the final montage.

# median tile offsets
{
    "overs": {
        "x": <median_offset_microns>,
        "y": <median_offset_microns>,
        "z": <median_offset_microns>
    },
    "downs": {
        "x": <median_offset_microns>,
        "y": <median_offset_microns>,
        "z": <median_offset_microns>
    }
}

# Each individual tile alignment is expected to produce a dictionary. Multiple alignments are stored in a list[dict]
# Whether the list represents "overs" or "downs" is indicated by the name or can be inferred by the fixed/moving images:
[
    {
        "fixed": <path_to_fixed_tile_file_name>, 
        "moving": <path_to_moving_tile_file_name>,
        "x": <offset_microns>, 
        "y": <offset_microns>, 
        "z": <offset_microns>, 
        "corr": <alignment_cross_correlation>, 
        "sheet": <sheet_used_to_acquire_tile> # ["left", "right"]
    },
    {...},
    {...}
]

The functions in this module enable us to work with this 
'''

def filer_coorelation(align_list, correlation=0.75):
    '''
    align_list: A list of alignments,
    correlation: correlation threshold to retain
    return: A list of alignments where the correlation >= to correlation
    '''
    return [x for x in align_list if x.get('corr') >= correlation]


def calculate_offsets(aligns_list, correlation=0.75):
    '''
    align_list: A list of alignments,
    correlation: correlation threshold to retain
    return: a dict where 'x','y','z' is the median value of alignment with a correlation >= correlation
    '''
    aligns_list = filer_coorelation(aligns_list, correlation=correlation)
    x = statistics.median([x.get('x') for x in aligns_list])
    y = statistics.median([x.get('y') for x in aligns_list])
    z = statistics.median([x.get('z') for x in aligns_list])
    medians = {
        'x':x,
        'y':y,
        'z':z
    }
    return medians

def annotate_with_sheet_direction(directory_with_mesospim_metadata: Path, align_list: list):
    '''
    directory_with_mesospim_metadata: Directory where mesospim_metadata can be found
    align_list: A list of alignments
    return: align_list with key 'sheet' added to each alignment indicating 'left' or 'right'
    '''
    directory_with_mesospim_metadata = ensure_path(directory_with_mesospim_metadata)
    mesospim_metadata = collect_all_metadata(directory_with_mesospim_metadata)
    for current_align in align_list:
        # Extract moving image name
        moving_fn = current_align.get('moving').name

        # Get tile number
        match = re.search(r'Tile(\d+)', moving_fn)
        tile_num = match.group(1)

        # Grab the first entry for this tile number
        meta_entry_current_align = get_all_tile_entries(mesospim_metadata, tile_num)[0]

        # Determine direction left/right light sheet
        sheet_direction = meta_entry_current_align.get("CFG").get("Shutter")
        sheet_direction = sheet_direction.lower()

        # Append direction to entry
        current_align['sheet'] = sheet_direction
    return align_list

def separate_by_sheet_direction(align_list):
    '''
    align_list: A list of alignments
    return: 2 align_lists where each list only includes tiles acquired with the 'left' or 'right' sheet
    '''
    left = [x for x in align_list if x.get('sheet') == 'left']
    right = [x for x in align_list if x.get('sheet') == 'right']
    return left, right