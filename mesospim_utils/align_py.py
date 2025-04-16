from pathlib import Path
import statistics
# from pprint import pprint as print

import numpy as np
import SimpleITK as sitk
from skimage import img_as_float32, img_as_uint
from skimage.registration import phase_cross_correlation
from scipy.ndimage import gaussian_filter
from imaris_ims_file_reader.ims import ims

from metadata import collect_all_metadata, get_first_entry, get_all_tile_entries
from utils import ensure_path, sort_list_of_paths_by_tile_number
from rl import mesospim_btf_helper
from utils import dict_to_json_file
from align_utils import calculate_offsets, annotate_with_sheet_direction, separate_by_sheet_direction

'''
This module has functions to deal with calculating alignment of mesospim data using python-based tools.
Currently it is only compatible with tiles converted to Imaris (.ims) files and SimpleITK for alignment 
'''

def compute_shift_np_microns_multiscale(fixed, moving, spacing, initial_transform=None, shrink_factors=[2, 1],
                                        smoothing_sigmas=[1, 0]):
    """
    Multi-scale registration to compute translation (in microns) between 3D numpy arrays.

    Parameters:
        fixed_np (np.ndarray): Fixed image, shape (Z, Y, X)
        moving_np (np.ndarray): Moving image, shape (Z, Y, X)
        spacing (tuple): (x, y, z) voxel spacing in microns
        shrink_factors (list): Pyramid downsampling factors per level
        smoothing_sigmas (list): Gaussian smoothing sigmas per level

    Returns:
        np.ndarray: [shift_x, shift_y, shift_z] in microns
    """

    if fixed.dtype != float:
        fixed = img_as_float32(fixed)
    if moving.dtype != float:
        moving = img_as_float32(moving)

    fixed = sitk.GetImageFromArray(fixed)
    moving = sitk.GetImageFromArray(moving)

    fixed.SetSpacing(spacing)
    moving.SetSpacing(spacing)

    fixed.SetOrigin((0, 0, 0))
    moving.SetOrigin((0, 0, 0))

    method = sitk.ImageRegistrationMethod()
    # method.SetMetricAsMattesMutualInformation()
    method.SetMetricAsMeanSquares()
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=0.01,
        numberOfIterations=500,
        gradientMagnitudeTolerance=1e-6
    )
    transform = sitk.TranslationTransform(3)
    method.SetInitialTransform(transform)
    method.SetInterpolator(sitk.sitkLinear)

    # Multi-resolution pyramid settings
    method.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    method.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing_sigmas)
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    if initial_transform:
        method.SetInitialTransform(initial_transform, inPlace=False)

    # Add observer to print at each iteration
    def iteration_update(method):
        print(
            f"Level: {method.GetCurrentLevel()}, Iter: {method.GetOptimizerIteration()}, Metric: {method.GetMetricValue()}, Shift: {transform.GetOffset()}")

    method.AddCommand(sitk.sitkIterationEvent, lambda: iteration_update(method))

    final_transform = method.Execute(fixed, moving)
    offsets = final_transform.GetParameters()
    print(offsets)
    offsets = offsets[::-1] # Reverse axes for numpy i.e. (x,y,z --> z,y,x)
    print(offsets)

    # Resample moving to fixed space using final transform
    resampled_moving = sitk.Resample(moving, fixed, final_transform, sitk.sitkLinear, 0.0,
                                     moving.GetPixelID())

    # Convert both to NumPy arrays for NCC
    fixed_np = sitk.GetArrayFromImage(fixed)
    moving_np = sitk.GetArrayFromImage(resampled_moving)

    ncc_score = compute_ncc(fixed_np, moving_np)
    print("NCC after alignment:", ncc_score)


    return np.array(offsets), final_transform, ncc_score



def compute_ncc(fixed_np, moving_np):
    fixed_flat = fixed_np.flatten()
    moving_flat = moving_np.flatten()

    fixed_mean = np.mean(fixed_flat)
    moving_mean = np.mean(moving_flat)

    numerator = np.sum((fixed_flat - fixed_mean) * (moving_flat - moving_mean))
    denominator = np.sqrt(np.sum((fixed_flat - fixed_mean) ** 2) * np.sum((moving_flat - moving_mean) ** 2))

    return numerator / denominator if denominator != 0 else 0.0


def align_ims_files(directory_with_mesospim_metadata, directory_with_ims_tiles, channel: list=None, res_overide=None):


    directory_with_mesospim_metadata = ensure_path(directory_with_mesospim_metadata)
    directory_with_ims_tiles = ensure_path(directory_with_ims_tiles)

    # Collect all metadata from MesoSPIM acquisition directory and save to mesospim_metadata.json in the ims file dir
    metadata_by_channel = collect_all_metadata(directory_with_mesospim_metadata)
    grid_x, grid_y = get_first_entry(metadata_by_channel).get('grid_size')
    resolution_um = get_first_entry(metadata_by_channel).get('resolution')  # (x,y,z)
    overlap = get_first_entry(metadata_by_channel).get('overlap')
    tile_shape = get_first_entry(metadata_by_channel).get('tile_shape')  # (x,y,z)

    # List IMS files in tile order
    ims_files = list(directory_with_ims_tiles.glob('*Tile*.ims'))
    ims_files = sort_list_of_paths_by_tile_number(ims_files)
    print(ims_files)

    if not channel:
        sample = ims(ims_files[0])
        channel = sample.Channels
        channel = list(range(channel))
    elif isinstance(channel,int):
        channel = [channel]


    # Place files in a grid shape, as nested lists. Each list represents a row of files in the x-axis,
    # the list index is the y-axis location
    file_name_grid_layout = [[] for x in range(grid_y)]
    idx = 0
    for x in range(grid_x):
        for y in range(grid_y):
            current_file = ims_files[idx]
            file_name_grid_layout[y].append(current_file)
            idx += 1

    # Align 1 row (overs)
    overs = []
    for ch in channel:
        for row, current_row in enumerate(file_name_grid_layout):
            # current_row = file_name_grid_layout[5]
            for idx in range(len(current_row)):

                if idx == 0:
                    continue

                print(f'Aligning overs: row {row}, Column {idx}')
                fixed_fn = current_row[idx - 1]
                moving_fn = current_row[idx]

                fixed_tile = ims(fixed_fn)
                moving_tile = ims(moving_fn)

                initial_transform = None
                for res in range(3)[::-1]:
                    # if res == 0:
                    #     continue

                    if res_overide is not None:
                        res = res_overide

                    print(f'Aligning resolution level {res}')
                    fixed_tile.change_resolution_lock(res)
                    moving_tile.change_resolution_lock(res)

                    spacing = [x * y for x, y in zip(moving_tile.resolution[::-1], resolution_um)]  # (x,y,z)
                    print(f'XYZ Spacing {spacing}')

                    moving_start_idx = int(moving_tile.shape[-1] * overlap)
                    fixed_start_idx = -moving_start_idx

                    print(f'Reading fixed image: {fixed_fn}')
                    if res > 0:
                        fixed = fixed_tile[res, 0, ch, :, :, :]
                        fixed = fixed[:, :, fixed_start_idx:]
                    else:
                        fixed = fixed_tile[0, 0, ch, :, :, fixed_start_idx:]

                    print(f'Reading moving image: {moving_fn}')
                    if res > 0:
                        moving = moving_tile[res, 0, ch, :, :, :]
                        moving = moving[:, :, 0:moving_start_idx]
                    else:
                        moving = moving_tile[0, 0, ch, :, :, 0:moving_start_idx]

                    fixed = img_as_float32(fixed)
                    moving = img_as_float32(moving)

                    print(f'Compute Alignment')
                    # Spacing is delivered in sitk space (x,y,x)
                    # Shift_um is returned in numpy space (z,y,x)
                    shift_um, initial_transform, correlation = compute_shift_np_microns_multiscale(fixed, moving, spacing,
                                                                                                   initial_transform=initial_transform,
                                                                                                   shrink_factors=[4, 2, 1],
                                                                                                   smoothing_sigmas=[2, 1,
                                                                                                                     0]
                                                                                                   )

                    pair = {
                        'fixed': fixed_fn,
                        'moving': moving_fn,
                        'x': float(shift_um[2]),
                        'y': float(shift_um[1]),
                        'z': float(shift_um[0]),
                        'corr': float(correlation)
                    }

                    print(pair)
                    overs.append(pair)

                    if res_overide is not None:
                        break

    # Align columns (downs)
    downs = []
    for ch in channel:
        for row, current_row in enumerate(file_name_grid_layout):
            # current_row = file_name_grid_layout[5]
            if row + 1 == len(file_name_grid_layout):
                continue
            next_row = file_name_grid_layout[row + 1]

            for idx in range(len(current_row)):

                print(f'Aligning downs: row {row}, Column {idx}')
                fixed_fn = current_row[idx]
                moving_fn = next_row[idx]

                fixed_tile = ims(fixed_fn)
                moving_tile = ims(moving_fn)

                initial_transform = None
                for res in range(3)[::-1]:
                    # if res == 0:
                    #     continue

                    if res_overide is not None:
                        res = res_overide

                    print(f'Aligning resolution level {res}')
                    fixed_tile.change_resolution_lock(res)
                    moving_tile.change_resolution_lock(res)

                    spacing = [x * y for x, y in zip(moving_tile.resolution[::-1], resolution_um)]  # (x,y,z)
                    print(f'XYZ Spacing {spacing}')

                    moving_start_idx = int(moving_tile.shape[-1] * overlap)
                    fixed_start_idx = -moving_start_idx

                    print(f'Reading fixed image: {fixed_fn}')
                    if res > 0:
                        fixed = fixed_tile[res, 0, ch, :, :, :]
                        fixed = fixed[:, fixed_start_idx:, :]
                    else:
                        fixed = fixed_tile[0, 0, ch, :, fixed_start_idx:, :]

                    print(f'Reading moving image: {moving_fn}')
                    if res > 0:
                        moving = moving_tile[res, 0, ch, :, :, :]
                        moving = moving[:, 0:moving_start_idx, :]
                    else:
                        moving = moving_tile[0, 0, ch, :, 0:moving_start_idx, :]

                    fixed = img_as_float32(fixed)
                    moving = img_as_float32(moving)

                    print(f'Compute Alignment')
                    # Spacing is delivered in sitk space (x,y,x)
                    # Shift_um is returned in numpy space (z,y,x)
                    shift_um, initial_transform, correlation = compute_shift_np_microns_multiscale(fixed, moving, spacing,
                                                                                                   initial_transform=initial_transform,
                                                                                                   shrink_factors=[4, 2, 1],
                                                                                                   smoothing_sigmas=[2, 1,
                                                                                                                     0]
                                                                                                   )

                    pair = {
                        'fixed': fixed_fn,
                        'moving': moving_fn,
                        'x': float(shift_um[2]),
                        'y': float(shift_um[1]),
                        'z': float(shift_um[0]),
                        'corr': float(correlation)
                    }
                    print(pair)
                    downs.append(pair)

                    if res_overide is not None:
                        break

    return overs, downs


def align(directory_with_mesospim_metadata, directory_with_ims_tiles, res_overide=3):
    overs, downs = align_ims_files(directory_with_mesospim_metadata, directory_with_ims_tiles, res_overide=res_overide)

    # Extract sheet directionality information
    overs = annotate_with_sheet_direction(directory_with_mesospim_metadata, overs)
    downs = annotate_with_sheet_direction(directory_with_mesospim_metadata, downs)

    # Calculate median offsets for overs and downs
    overs_med = calculate_offsets(overs, correlation=0.75)
    downs_med = calculate_offsets(downs, correlation=0.75)

    # Separate offsets by sheet directionality
    over_left, over_right = separate_by_sheet_direction(overs)
    down_left, down_right = separate_by_sheet_direction(downs)

    # Calculate median offsets for L/R Overs/Downs
    o_l = calculate_offsets(over_left, correlation=0.75)
    o_r = calculate_offsets(over_right, correlation=0.75)
    d_l = calculate_offsets(down_left, correlation=0.75)
    d_r = calculate_offsets(down_right, correlation=0.75)

    ## Save offset info
    file_output = Path(directory_with_ims_tiles) / 'resample_median_tile_offsets_microns.json'
    median_offsets = {
        'overs': overs_med,
        'downs': downs_med,
        'over_left_sheet': o_l,
        'over_right_sheet': o_r,
        'down_left_sheet': d_l,
        'down_right_sheet': d_r,
    }

    # Write to a JSON file
    dict_to_json_file(median_offsets, file_output)

    file_output = Path(directory_with_ims_tiles) / 'resample_all_tile_offsets_microns.json'
    all_offsets = {
        'overs': overs,
        'downs': downs,
        'over_left_sheet': over_left,
        'over_right_sheet': over_right,
        'down_left_sheet': down_left,
        'down_right_sheet': down_right,
    }

    # Write to a JSON file
    dict_to_json_file(all_offsets, file_output)

    return overs, downs, over_left, over_right, down_left, down_right, overs_med, downs_med, o_l, o_r, d_l, d_r


if __name__ == '__main__':

    directory_with_mesospim_metadata = f'/CBI_FastStore/tmp/mesospim/brain'
    directory_with_mesospim_metadata = f'/CBI_FastStore/tmp/mesospim/kidney'
    directory_with_ims_tiles = f'/CBI_FastStore/tmp/mesospim/kidney/ims_files'

    overs, downs, over_left, over_right, down_left, down_right, overs_med, downs_med, o_l, o_r, d_l, d_r = \
        align(directory_with_mesospim_metadata, directory_with_ims_tiles, res_overide=2)









