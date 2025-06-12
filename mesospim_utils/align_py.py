# STD library imports
from pathlib import Path
import statistics

# Installed Packages imports
import numpy as np
import SimpleITK as sitk
from skimage import img_as_float32, img_as_uint
from skimage.registration import phase_cross_correlation
from skimage.filters import gaussian
from scipy.ndimage import gaussian_filter
from scipy.ndimage import fourier_shift
from scipy.fft import fftn, ifftn
from imaris_ims_file_reader.ims import ims
import typer

# Local imports
from metadata import collect_all_metadata, get_first_entry, get_all_tile_entries
from utils import ensure_path, sort_list_of_paths_by_tile_number, make_directories, dict_to_json_file
from rl import mesospim_btf_helper
from align_utils import calculate_offsets, annotate_with_sheet_direction, separate_by_sheet_direction
from constants import (
    ALIGNMENT_DIRECTORY,
    CORRELATION_THRESHOLD_FOR_ALIGN,
    RESOLUTION_LEVEL_FOR_ALIGN,
    VERBOSE,
    OFFSET_METRIC,
    REMOVE_OUTLIERS,
    ALIGN_ALL_OUTPUT_FILE_NAME,
    ALIGN_METRIC_OUTPUT_FILE_NAME
)

# INIT typer cmdline interface
app = typer.Typer()


'''
This module has functions to deal with calculating alignment of mesospim data using python-based tools.
Currently it is only compatible with tiles converted to Imaris (.ims) files and SimpleITK for alignment 
'''

def gaussian_blur(image, pixel_size_um, sigma_microns=4):

    if image.dtype != float:
        image = img_as_float32(image)

    if isinstance(pixel_size_um,int):
        pixel_size_um = [pixel_size_um] * 3
    pixel_size_um = np.array(pixel_size_um)

    if isinstance(sigma_microns,int):
        sigma_microns = [sigma_microns] * 3
    sigma_microns = np.array(sigma_microns)

    # Convert sigma to pixel units
    sigma_pixels = sigma_microns / pixel_size_um

    # Apply Gaussian blur
    blurred_image = gaussian(image, sigma=sigma_pixels.tolist(), preserve_range=True)
    return blurred_image

def shift_image(moving, pixel_shift):

    # Shift the moving image in Fourier space
    shifted_fft = fourier_shift(fftn(moving), pixel_shift)
    aligned = np.real(ifftn(shifted_fft))
    return aligned

def compute_shift_np_microns_multiscale2(fixed, moving, spacing, initial_transform=None, shrink_factors=[2, 1],
                                        smoothing_sigmas=[1, 0], verbose=VERBOSE):
    """
    Multi-scale, two-stage registration to compute translation (in microns) between 3D numpy arrays.

    Parameters:
        fixed (np.ndarray): Fixed image, shape (Z, Y, X)
        moving (np.ndarray): Moving image, shape (Z, Y, X)
        spacing (tuple): (z, y, x) voxel spacing in microns
        initial_transform: Optional initial sitk.Transform
        shrink_factors (list): Pyramid downsampling factors
        smoothing_sigmas (list): Gaussian smoothing sigmas
        verbose (bool): If True, prints registration progress

    Returns:
        np.ndarray: [shift_z, shift_y, shift_x] in microns
        sitk.Transform: Final SimpleITK transform
        float: NCC score between aligned images
    """

    if fixed.dtype != np.float32:
        fixed = img_as_float32(fixed)
    if moving.dtype != np.float32:
        moving = img_as_float32(moving)

    fixed_sitk = sitk.GetImageFromArray(fixed)
    moving_sitk = sitk.GetImageFromArray(moving)

    fixed_sitk.SetSpacing(spacing[::-1])  # (x,y,z) for SITK
    moving_sitk.SetSpacing(spacing[::-1])

    fixed_sitk.SetOrigin((0, 0, 0))
    moving_sitk.SetOrigin((0, 0, 0))

    # If no initial transform, use center-aligned Euler3D
    if initial_transform is None:
        initial_transform = sitk.TranslationTransform(fixed_sitk.GetDimension())

    ### ---- Stage 1: Coarse Registration with MeanSquares ---- ###
    coarse_reg = sitk.ImageRegistrationMethod()
    coarse_reg.SetMetricAsMeanSquares()
    coarse_reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=4.0,
        minStep=0.1,
        numberOfIterations=200,
        gradientMagnitudeTolerance=1e-6
    )
    coarse_reg.SetInterpolator(sitk.sitkLinear)
    coarse_reg.SetInitialTransform(initial_transform, inPlace=False)
    coarse_reg.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    coarse_reg.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing_sigmas)
    coarse_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    if verbose:
        def coarse_iteration():
            print(f"[Coarse] Level: {coarse_reg.GetCurrentLevel()} Iter: {coarse_reg.GetOptimizerIteration()} Metric: {coarse_reg.GetMetricValue():.5f}")
        coarse_reg.AddCommand(sitk.sitkIterationEvent, coarse_iteration)

    try:
        coarse_success = False
        coarse_transform = coarse_reg.Execute(fixed_sitk, moving_sitk)
        coarse_success = True
    except Exception as e:
        if verbose: print(f"[Error] Coarse registration failed: {e}")
        coarse_transform = initial_transform

    ### ---- Stage 2: Fine Registration with Mattes Mutual Information ---- ###
    fine_reg = sitk.ImageRegistrationMethod()
    fine_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    fine_reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=0.01,
        numberOfIterations=300,
        gradientMagnitudeTolerance=1e-6
    )
    fine_reg.SetInterpolator(sitk.sitkLinear)
    fine_reg.SetInitialTransform(coarse_transform, inPlace=False)
    fine_reg.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    fine_reg.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing_sigmas)
    fine_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()


    if verbose:
        def fine_iteration():
            print(f"[Fine] Level: {fine_reg.GetCurrentLevel()} Iter: {fine_reg.GetOptimizerIteration()} Metric: {fine_reg.GetMetricValue():.5f}")
        fine_reg.AddCommand(sitk.sitkIterationEvent, fine_iteration)

    try:
        fine_success = False
        final_transform = fine_reg.Execute(fixed_sitk, moving_sitk)
        fine_success = True
    except Exception as e:
        if verbose: print(f"[Error] Fine registration failed: {e}")
        final_transform = coarse_transform

    if not coarse_success and not fine_success:
        return np.array([0.0, 0.0, 0.0]), None, 0.0

    # Extract final offsets
    try:
        offsets = final_transform.GetParameters()
        offsets = np.array(offsets[::-1])  # Reverse (x,y,z) -> (z,y,x)
    except Exception as e:
        if verbose: print(f"[Warning] Transform parameters extraction failed: {e}")
        return np.array([0.0, 0.0, 0.0]), None, 0.0

    # Resample moving to fixed space
    resampled_moving = sitk.Resample(moving_sitk, fixed_sitk, final_transform, sitk.sitkLinear, 0.0,
                                     moving_sitk.GetPixelID())

    # Compute NCC score
    fixed_np = sitk.GetArrayFromImage(fixed_sitk)
    moving_np = sitk.GetArrayFromImage(resampled_moving)

    ncc_score = compute_ncc(fixed_np, moving_np)
    if verbose:
        print(f"NCC after alignment: {ncc_score:.5f}")

    return offsets, final_transform, ncc_score


def compute_shift_np_microns_phase(fixed, moving, spacing):

    if fixed.dtype != float:
        fixed = img_as_float32(fixed)
    if moving.dtype != float:
        moving = img_as_float32(moving)

    spacing = np.array(spacing)

    # calculate upsample for 0.5 micron subpixel accuracy in x,y
    upsample_factor = max((spacing / 0.5)[-2:]) # Use only x,y values
    if upsample_factor < 1:
        upsample_factor = 1
    if VERBOSE == 2: print(f'Upsampling factor {upsample_factor}')

    # Normalize to mean 0, std 1 to improve correlation
    # print('Normalizing fixed image')
    # fixed = (fixed - np.mean(fixed)) / np.std(fixed)
    # print('Normalizing moving image')
    # moving = (moving - np.mean(moving)) / np.std(moving)

    if VERBOSE: print('Applying Gaussian Blur Moving')
    moving = gaussian_blur(moving, spacing.tolist(), sigma_microns=(spacing*2).tolist())

    if VERBOSE: print('Applying Gaussian Blur Fixed')
    fixed = gaussian_blur(fixed, spacing.tolist(), sigma_microns=(spacing*2).tolist())

    if VERBOSE: print('Matching Histograms')
    moving = match_hist(moving, fixed)

    # Compute the shift between the reference and moving volumes
    if VERBOSE: print('Aligning')
    shift, error, diffphase = phase_cross_correlation(fixed, moving,
                                                      upsample_factor=upsample_factor,
                                                      disambiguate=True)

    if VERBOSE: print(f'Shift {shift}')
    moving = shift_image(moving, shift)

    shift_um = shift * spacing
    #Invert shifts
    shift_um = shift_um * -1

    ncc_score = compute_ncc(fixed, moving)
    if VERBOSE:print("NCC after alignment:", ncc_score)

    return shift_um, diffphase, ncc_score


def compute_shift_np_microns_multiscale(fixed, moving, spacing, initial_transform=None, shrink_factors=[2, 1],
                                        smoothing_sigmas=[1, 0]):
    """
    Multi-scale registration to compute translation (in microns) between 3D numpy arrays.

    Parameters:
        fixed (np.ndarray): Fixed image, shape (Z, Y, X)
        moving (np.ndarray): Moving image, shape (Z, Y, X)
        spacing (tuple): (z, y, x) voxel spacing in microns
        initial_transform: sitk transform as starting point for alignment,
        shrink_factors (list): Pyramid downsampling factors per level
        smoothing_sigmas (list): Gaussian smoothing sigmas per level

    Returns:
        np.ndarray: (
        [shift_z, shift_y, shift_x] in microns,
        calculated_sitk_transformation,
        normalized_cross_correlation_score
        )
    """

    if fixed.dtype != float:
        fixed = img_as_float32(fixed)
    if moving.dtype != float:
        moving = img_as_float32(moving)

    fixed = sitk.GetImageFromArray(fixed)
    moving = sitk.GetImageFromArray(moving)

    fixed.SetSpacing(spacing[::-1]) # swap to (x,y,z) for sitk
    moving.SetSpacing(spacing[::-1]) # swap to (x,y,z) for sitk

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
    if VERBOSE:
        def iteration_update(method):
            print(
                f"Level: {method.GetCurrentLevel()}, Iter: {method.GetOptimizerIteration()}, Metric: {method.GetMetricValue()}, Shift: {transform.GetOffset()}")

        method.AddCommand(sitk.sitkIterationEvent, lambda: iteration_update(method))

    final_transform = method.Execute(fixed, moving)
    offsets = final_transform.GetParameters()
    offsets = offsets[::-1] # Reverse axes for numpy i.e. (x,y,z --> z,y,x)

    # Resample moving to fixed space using final transform
    resampled_moving = sitk.Resample(moving, fixed, final_transform, sitk.sitkLinear, 0.0,
                                     moving.GetPixelID())

    # Convert both to NumPy arrays for NCC
    fixed_np = sitk.GetArrayFromImage(fixed)
    moving_np = sitk.GetArrayFromImage(resampled_moving)

    ncc_score = compute_ncc(fixed_np, moving_np)
    if VERBOSE: print("NCC after alignment:", ncc_score)


    return np.array(offsets), final_transform, ncc_score


def compute_ncc(fixed_np, moving_np):
    fixed_flat = fixed_np.flatten()
    moving_flat = moving_np.flatten()

    fixed_mean = np.mean(fixed_flat)
    moving_mean = np.mean(moving_flat)

    numerator = np.sum((fixed_flat - fixed_mean) * (moving_flat - moving_mean))
    denominator = np.sqrt(np.sum((fixed_flat - fixed_mean) ** 2) * np.sum((moving_flat - moving_mean) ** 2))

    return numerator / denominator if denominator != 0 else 0.0


from skimage import exposure
from skimage.exposure import match_histograms


def equalize_hist(img):
    img = (img - img.min()) / (img.max() - img.min())
    img = exposure.equalize_adapthist(img)
    return img

def match_hist(moving_image, fixed_image):
    matched_img = match_histograms(moving_image, fixed_image)
    return matched_img

def pre_process_images(fixed_image, moving_image):
    fixed_image = equalize_hist(fixed_image)
    moving_image = equalize_hist(moving_image)
    moving_image = match_hist(moving_image, fixed_image)
    return fixed_image, moving_image

def align_ims_files(directory_with_mesospim_metadata, directory_with_ims_tiles, channel: list=None, res_overide=None, preprocess:bool=False):


    directory_with_mesospim_metadata = ensure_path(directory_with_mesospim_metadata)
    directory_with_ims_tiles = ensure_path(directory_with_ims_tiles)

    # Collect all metadata from MesoSPIM acquisition directory and save to mesospim_metadata.json in the ims file dir
    metadata_by_channel = collect_all_metadata(directory_with_mesospim_metadata)
    grid_y, grid_x = get_first_entry(metadata_by_channel).get('grid_size')
    resolution_um = get_first_entry(metadata_by_channel).get('resolution')  # (z,y,x)
    overlap = get_first_entry(metadata_by_channel).get('overlap')
    # tile_shape = get_first_entry(metadata_by_channel).get('tile_shape')  # (z,y,x)
    stage_direction = get_first_entry(metadata_by_channel).get('stage_direction')

    # List IMS files in tile order
    ims_files = list(directory_with_ims_tiles.glob('*Tile*.ims'))
    ims_files = sort_list_of_paths_by_tile_number(ims_files)
    if VERBOSE == 2: print(ims_files)

    if not channel:
        sample = ims(ims_files[0])
        channel = sample.Channels
        channel = list(range(channel))
    elif isinstance(channel,int):
        channel = [channel]


    # Place files in a grid shape, as nested lists. Each list represents a row of files in the x-axis,
    # the list index is the y-axis location
    file_name_grid_layout = [[None]*grid_x for x in range(grid_y)]
    idx = 0
    for x in range(grid_x)[::stage_direction[1]]:
        for y in range(grid_y)[::stage_direction[0]]:
            current_file = ims_files[idx]
            file_name_grid_layout[y][x] = current_file
            idx += 1

    # Align 1 row (overs)
    overs = []
    for ch in channel:
        for row, current_row in enumerate(file_name_grid_layout):
            # current_row = file_name_grid_layout[5]
            for idx in range(len(current_row)):

                if idx == 0:
                    continue

                if VERBOSE: print(f'Aligning overs: row {row}, Column {idx}')
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

                    if VERBOSE: print(f'Aligning resolution level {res}')
                    fixed_tile.change_resolution_lock(res)
                    moving_tile.change_resolution_lock(res)

                    # spacing = [x * y for x, y in zip(moving_tile.resolution, resolution_um)]  # (z,y,x)
                    spacing = moving_tile.resolution
                    if VERBOSE == 2: print(f'ZYX Spacing {spacing}')

                    moving_start_idx = int(moving_tile.shape[-1] * overlap)
                    fixed_start_idx = -moving_start_idx

                    if VERBOSE: print(f'Reading fixed image: {fixed_fn}')
                    if res > 0:
                        fixed = fixed_tile[res, 0, ch, :, :, :]
                        fixed = fixed[:, :, fixed_start_idx:]
                    else:
                        fixed = fixed_tile[0, 0, ch, :, :, fixed_start_idx:]

                    if VERBOSE: print(f'Reading moving image: {moving_fn}')
                    if res > 0:
                        moving = moving_tile[res, 0, ch, :, :, :]
                        moving = moving[:, :, 0:moving_start_idx]
                    else:
                        moving = moving_tile[0, 0, ch, :, :, 0:moving_start_idx]

                    if preprocess:
                        if VERBOSE: print(f'Equalizing Images')
                        fixed, moving = pre_process_images(fixed, moving)

                    fixed = img_as_float32(fixed)
                    moving = img_as_float32(moving)


                    if VERBOSE: print(f'Compute Alignment')
                    # Shift_um is returned in numpy space (z,y,x)
                    # shift_um, initial_transform, correlation = compute_shift_np_microns_phase(fixed, moving, spacing)
                    shift_um, initial_transform, correlation = compute_shift_np_microns_multiscale2(fixed, moving, spacing,
                                                                                                   initial_transform=initial_transform,
                                                                                                   shrink_factors=[4, 2, 1],
                                                                                                   smoothing_sigmas=[2, 1,
                                                                                                                     0]
                                                                                                   )

                    pair = {
                        'fixed': fixed_fn,
                        'moving': moving_fn,
                        'channel': ch,
                        'x': float(shift_um[2]),
                        'y': float(shift_um[1]),
                        'z': float(shift_um[0]),
                        'corr': float(correlation),
                        'direction': 'over',
                        'shift_units': 'microns'
                    }

                    if VERBOSE == 2: print(pair)
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

                if VERBOSE: print(f'Aligning downs: row {row}, Column {idx}')
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

                    if VERBOSE: print(f'Aligning resolution level {res}')
                    fixed_tile.change_resolution_lock(res)
                    moving_tile.change_resolution_lock(res)

                    spacing = moving_tile.resolution
                    if VERBOSE: print(f'ZYX Spacing {spacing}')

                    moving_start_idx = int(moving_tile.shape[-1] * overlap)
                    fixed_start_idx = -moving_start_idx

                    if VERBOSE: print(f'Reading fixed image: {fixed_fn}')
                    if res > 0:
                        fixed = fixed_tile[res, 0, ch, :, :, :]
                        fixed = fixed[:, fixed_start_idx:, :]
                    else:
                        fixed = fixed_tile[0, 0, ch, :, fixed_start_idx:, :]

                    if VERBOSE: print(f'Reading moving image: {moving_fn}')
                    if res > 0:
                        moving = moving_tile[res, 0, ch, :, :, :]
                        moving = moving[:, 0:moving_start_idx, :]
                    else:
                        moving = moving_tile[0, 0, ch, :, 0:moving_start_idx, :]

                    if preprocess:
                        if VERBOSE: print(f'Equalizing Images')
                        fixed, moving = pre_process_images(fixed, moving)

                    fixed = img_as_float32(fixed)
                    moving = img_as_float32(moving)

                    if VERBOSE: print(f'Compute Alignment')
                    # shift_um, initial_transform, correlation = compute_shift_np_microns_phase(fixed, moving, spacing)
                    shift_um, initial_transform, correlation = compute_shift_np_microns_multiscale2(fixed, moving, spacing,
                                                                                                   initial_transform=initial_transform,
                                                                                                   shrink_factors=[4, 2, 1],
                                                                                                   smoothing_sigmas=[2, 1,
                                                                                                                     0]
                                                                                                   )

                    pair = {
                        'fixed': fixed_fn,
                        'moving': moving_fn,
                        'channel': ch,
                        'x': float(shift_um[2]),
                        'y': float(shift_um[1]),
                        'z': float(shift_um[0]),
                        'corr': float(correlation),
                        'direction': 'down',
                        'shift_units': 'microns'
                    }
                    if VERBOSE == 2: print(pair)
                    downs.append(pair)

                    if res_overide is not None:
                        break

    return overs, downs

@app.command()
def align(directory_with_mesospim_metadata:Path, directory_with_ims_tiles:Path,
          res_overide: int=RESOLUTION_LEVEL_FOR_ALIGN,
          correlation_threshold: float=CORRELATION_THRESHOLD_FOR_ALIGN):

    directory_with_ims_tiles = ensure_path(directory_with_ims_tiles)

    # Make an alignment directory
    align_dir = directory_with_ims_tiles / ALIGNMENT_DIRECTORY
    make_directories(align_dir)

    overs, downs = align_ims_files(directory_with_mesospim_metadata, directory_with_ims_tiles, res_overide=res_overide)

    # Extract sheet directionality information
    overs = annotate_with_sheet_direction(directory_with_mesospim_metadata, overs)
    downs = annotate_with_sheet_direction(directory_with_mesospim_metadata, downs)

    # Separate offsets by sheet directionality
    over_left, over_right = separate_by_sheet_direction(overs)
    down_left, down_right = separate_by_sheet_direction(downs)

    ## Save offset info
    file_output = align_dir / ALIGN_METRIC_OUTPUT_FILE_NAME

    median_offsets = {
        'overs': {
            'left': calculate_offsets(over_left, correlation=correlation_threshold),
            'right': calculate_offsets(over_right, correlation=correlation_threshold),
            'both': calculate_offsets(overs, correlation=correlation_threshold),
        },
        'downs': {
            'left': calculate_offsets(down_left, correlation=correlation_threshold),
            'right': calculate_offsets(down_right, correlation=correlation_threshold),
            'both': calculate_offsets(downs, correlation=correlation_threshold),
        },
        'correlation': correlation_threshold,
        'app': 'mesospim_align',
        'shift_units': 'microns',
        'outliers_removed': REMOVE_OUTLIERS,
        'metric': OFFSET_METRIC,

    }

    # Write to a JSON file
    dict_to_json_file(median_offsets, file_output)

    file_output = align_dir / ALIGN_ALL_OUTPUT_FILE_NAME

    all_offsets = {
        'overs': overs,
        'downs': downs,
        'app': 'mesospim_align',
    }

    # Write to a JSON file
    dict_to_json_file(all_offsets, file_output)

    return median_offsets, all_offsets



if __name__ == '__main__':
    app()

    # directory_with_mesospim_metadata = f'/CBI_FastStore/tmp/mesospim/brain'
    # directory_with_mesospim_metadata = f'/CBI_FastStore/tmp/mesospim/kidney'
    # directory_with_ims_tiles = f'/CBI_FastStore/tmp/mesospim/kidney/ims_files'
    #
    # # directory_with_mesospim_metadata = f'/CBI_FastStore/tmp/mesospim/knee'
    # # directory_with_ims_tiles = f'/CBI_FastStore/tmp/mesospim/knee/decon/ims_files'
    #
    # # median_offsets, all_offsets = align(directory_with_mesospim_metadata, directory_with_ims_tiles, res_overide=5)
    # median_offsets, all_offsets = align(directory_with_ims_tiles, directory_with_ims_tiles, res_overide=5)









