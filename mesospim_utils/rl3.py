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
from typing import Union

import numpy as np
import tifffile
import skimage
from skimage import img_as_float32, img_as_uint

from psf import get_psf
from metadata2 import collect_all_metadata, get_ch_entry_for_file_name, get_entry_for_file_name

from constants import ENV_PYTHON_LOC
from constants import DECON_SCRIPT as LOC_OF_THIS_SCRIPT


ims_conv_module_name = Path(LOC_OF_THIS_SCRIPT).parent / 'slurm.py'


app = typer.Typer()

@app.command()
def decon_dir(dir_loc: str, refractive_index: float, out_dir: str=None, out_file_type: str='.tif', file_type: str='.btf',
              queue_ims: bool=False, denoise_sigma: float=None, sharpen: bool=False,
              half_precision: bool=False, psf_shape: tuple[int,int,int]=(7,7,7), iterations: int=40, frames_per_chunk: int=100,
              num_parallel: int=8, run_slurm=True
              ):
    '''3D deconvolution of all files in a directory using the richardson-lucy method [executed on SLURM]'''
    import subprocess
    path = Path(dir_loc)
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = path / 'decon'
    log_dir = out_dir / 'logs'
    log_dir.parent.mkdir(parents=True, exist_ok=True)
    file_list = list(path.glob('*' + file_type))
    num_files = len(file_list)
    if queue_ims:
        num_files += 1

    SBATCH_ARG = '#SBATCH {}\n'
    to_run = ["sbatch", "-p gpu", "--gres=gpu:1", "-J decon", f'-o {log_dir} / %A_%a.log', f'--array=0-{num_files-1}']
    commands = "#!/bin/bash\n"
    commands += SBATCH_ARG.format('-p gpu')
    commands += SBATCH_ARG.format('--gres=gpu:1')
    commands += SBATCH_ARG.format('-J decon')
    commands += SBATCH_ARG.format(f'-o {log_dir}/%A_%a.log')
    commands += SBATCH_ARG.format(f'--array=0-{num_files-1}{"%" + str(num_parallel) if num_parallel>0 else ""}')
    commands += "\n"

    commands += "commands=("
    #Build each command
    for p in file_list:
        commands += '\n\t'
        commands += '"'
        commands += f'{ENV_PYTHON_LOC} -u {LOC_OF_THIS_SCRIPT} decon'
        commands += f' {p.as_posix()}'
        commands += f' {refractive_index}'
        commands += f' --out-location {out_dir / (p.stem + out_file_type)}'
        #commands += f'{" --queue-ims" if queue_ims else ""}'
        commands += f'{" --sharpen" if sharpen else ""}'
        commands += f'{f" --denoise-sigma {denoise_sigma}" if denoise_sigma else ""}'
        commands += f'{" --half-precision" if half_precision else " --no-half-precision"}'
        commands += f' --psf-shape {psf_shape[0]} {psf_shape[1]} {psf_shape[2]}'
        commands += f' --iterations {iterations}'
        commands += f' --frames-per-chunk {frames_per_chunk}'
        commands += '"'

    if queue_ims:
        # Extract resolution from mesospim metadata file if it exists
        res = (1,1,1)
        try:
            meta_dir = file_list[0].parent
            meta_entry = get_entry_for_file_name(meta_dir, file_list[0].name)
            x_res, y_res, z_res  = meta_entry.get('resolution')
            res = (z_res,y_res,x_res)
        except:
            pass

        ims_auto_queue = f'\n\t"sbatch -p compute -n 1 --mem 5G -o {log_dir}/%A.log --dependency=afterok:$SLURM_JOB_ID -J ims_queue --kill-on-invalid-dep=yes --wrap=\''
        # ims_auto_queue += f'{ENV_PYTHON_LOC} -u {ims_conv_module_name} convert-ims-dir-mesospim-tiles '
        ims_auto_queue += f'{ENV_PYTHON_LOC} -u {ims_conv_module_name} convert-ims-dir-mesospim-tiles-slurm-array '
        ims_auto_queue += f'{out_dir} '
        ims_auto_queue += f'--file-type {out_file_type} '
        ims_auto_queue += f'--res {res[0]} {res[1]} {res[2]}\'"'
        commands += ims_auto_queue

    commands += '\n)\n\n'
    commands += 'echo "Running command: ${commands[$SLURM_ARRAY_TASK_ID]}"\n'
    commands += 'eval "${commands[$SLURM_ARRAY_TASK_ID]}"'

    file_to_run = out_dir / 'slurm_array.sh'
    with open(file_to_run, 'w') as f:
        f.write(commands)

    if run_slurm:
        output = subprocess.run(f'sbatch {file_to_run}', shell=True, capture_output=True)
        prefix_len = len(b'Submitted batch job ')
        job_number = int(output.stdout[prefix_len:-1])
        print(f'SBATCH Job #: {job_number}')
    else:
        # Return the name of the sbatch script that was created.
        return file_to_run


def get_rl_model(psf, iterations=20, sigma=None):
    import torch
    import torch.nn.functional as F
    import torch.nn as nn
    class rl(nn.Module):
        def __init__(self, psf, iterations=20, sigma=None):
            super(rl, self).__init__()
            self.iterations = iterations
            if sigma and sigma > 0:
                print('Preparing PSF with smoothing kernel')
                psf[:, :] = T.functional.gaussian_blur(psf[0, 0], kernel_size=psf.shape[-2:],
                                                       sigma=(sigma, sigma))
            psf_flipped = torch.flip(psf, dims=(2, 3, 4))
            self.psf = nn.Parameter(psf, requires_grad=False)
            self.psf_flipped = nn.Parameter(psf_flipped, requires_grad=False)

        def forward(self, input_data):
            if sigma and sigma > 0:
                print('Preparing Input Data with smoothing kernel to reduce noise')
                input_data[:, :] = T.functional.gaussian_blur(input_data[0, 0], kernel_size=self.psf.shape[-2:],
                                                          sigma=(sigma, sigma))
            convolved = input_data.clone()
            relative_blur = input_data.clone()
            correction = input_data.clone()
            estimate = input_data.clone()

            print('Beginning deconvolution')
            with torch.no_grad():
                for i in range(self.iterations):
                    # Convolve estimate with PSF
                    convolved[:] = F.conv3d(estimate, self.psf, padding='same')

                    # Calculate the ratio of input data to the convolved data
                    relative_blur[:] = input_data / (convolved + 1e-12)

                    # Convolve the relative blur with the flipped PSF
                    correction[:] = F.conv3d(relative_blur, self.psf_flipped, padding='same')

                    # Update the estimate
                    estimate *= correction

                    print(f'Iteration {i+1} of {iterations}, Max: {estimate.max()}')

            return estimate

    return rl(psf, iterations=iterations, sigma=sigma)

def richardson_lucy_3d(input_data, psf, iterations=50, sigma=None):
    """
    Perform Richardson-Lucy deconvolution on 3D microscopy data.

    Args:
        input_data (torch.Tensor): Observed 3D microscopy data (B x C x Z x Y x X).
        psf (torch.Tensor): PSF kernel (1 x 1 x kZ x kY x kX).
        iterations (int): Number of iterations for deconvolution.

    Returns:
        torch.Tensor: Deconvolved 3D data.
    """
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    import torch.nn as nn

    # Ensure data and PSF are on the same device
    print('Loading data onto GPU')
    input_data = input_data.to('cuda')
    psf = psf.to('cuda')

    print(f'CUDA Input image dtype: {input_data.dtype}')
    print(f'CUDA PSF dtype: {psf.dtype}')

    with torch.no_grad():
        # Noise reduction
        if sigma and sigma > 0:
            print('Smoothing data and PSF to reduce noise')
            #gblur = T.GaussianBlur(kernel_size=(5, 5), sigma=sigma)
            #gblur = T.GaussianBlur(kernel_size=psf.shape[-2:], sigma=sigma)
            # input_data[:,:] = gblur(input_data[0,0])
            # psf[:,:] = gblur(psf[0,0])

            input_data[:,:] = T.functional.gaussian_blur(input_data[0,0], kernel_size=psf.shape[-2:], sigma=(sigma, sigma))
            psf[:, :] = T.functional.gaussian_blur(psf[0,0], kernel_size=psf.shape[-2:], sigma=(sigma, sigma))

        # Flip the PSF for convolution
        psf_flipped = torch.flip(psf, dims=(2, 3, 4))

        # Initialize the estimate with the input data
        #input_data = pad_reflect(input_data, padding_depth=tuple(psf.shape)[-1]*2)
        convolved = input_data.clone()
        relative_blur = input_data.clone()
        #correction = relative_blur.clone()
        estimate = input_data.clone()
        #estimate = torch.full_like(correction,0.5)

        def conv3d_with_reflect_pad(image,psf):
            _,_,z,y,x = psf.shape
            pad = [x,x,y,y,z,z] #Assume 5D tensor, pad only the z,y,x dims in x,y,z order (reversed)
            #print(f'PAD: {pad}')
            pad = [(x-1)//2 for x in pad]
            # print(f'PAD: {pad}')
            # print(f'SHAPE ORIG = {image.shape}')
            image = F.pad(image, pad, mode='reflect')
            # print(f'SHAPE PAD = {image.shape}')
            image = F.conv3d(image, psf, padding=0) #padding 0 is 'no padding' will return reduced array which will equal the origional input size.
            # print(f'SHAPE OUT = {out.shape}')
            return image

        print('Beginning deconvolution')
        #print(f"Iteration {0}/{iterations}, Correction Mean: {correction.mean().item()}, MIN: {estimate.min().item()}, MAX: {estimate.max().item()}")
        start_time = time_report(start_time=None, units='minutes')
        for i in range(iterations):
            itter_start = time_report(start_time=None, units='minutes')
            # Convolve estimate with PSF
            #convolved[:] = F.conv3d(estimate, psf, padding='same')
            convolved[:] = conv3d_with_reflect_pad(estimate, psf)

            # Calculate the ratio of input data to the convolved data
            convolved += 1e-12
            relative_blur[:] = input_data / convolved

            # Convolve the relative blur with the flipped PSF
            #correction[:] = F.conv3d(relative_blur, psf_flipped, padding='same')
            #relative_blur[:] = F.conv3d(relative_blur, psf_flipped, padding='same')
            relative_blur[:] = conv3d_with_reflect_pad(relative_blur, psf_flipped)

            # Update the estimate
            #estimate *= correction
            estimate *= relative_blur

            # Print progress
            if i % 1 == 0:
                torch.cuda.synchronize()
                #print(f"Iteration {i+1}/{iterations}, Time: {round(time_report(start_time=itter_start, to_print=False), 2)/60} Sec, MIN: {estimate.min().item()}, MAX: {estimate.max().item()}")
                print(f"Iteration {i+1}/{iterations}, Time: {round(time_report(start_time=itter_start, to_print=False)/60, 2)} Sec")

    # estimate = normalize_array(estimate, input_data.min(), input_data.max())
    # estimate = unpad(estimate, padding_depth=tuple(psf.shape)[-1]*2)
    print('Deconvolution Complete')
    time_report(start_time=start_time, to_print=True)
    return estimate

###################################################################################################################
## IMAGE PROCESSING FUNCTIONS ##
###################################################################################################################

def read_mesospim_btf(file_location: Union[str, Path], frame_start: int = None, frame_stop: int = None):

    #if frame_start or frame_stop:
    with tifffile.TiffFile(file_location) as tif:
        print(f'File contains {len(tif.series)} frames')
        stack = np.stack(tuple((x.asarray()[0] for x in tif.series[frame_start:frame_stop])))
        return stack

    # from io import BytesIO
    # with open(file_location, 'rb') as f:
    #     file_data = BytesIO(f.read())
    # with tifffile.TiffFile(file_data) as tif:
    #     print(f'File contains {len(tif.series)} frames')
    #     stack = np.stack(tuple((x.asarray()[0] for x in tif.series[frame_start:frame_stop])))
    #     return stack

    # with open(file_location, 'rb') as f:
    #    from io import BytesIO
    #    image_file = BytesIO(f.read())
    # with tifffile.TiffFile(image_file) as tif:
    #    print(f'File contains {len(tif.series)} frames')
    #    stack = np.stack(tuple((x.asarray()[0] for x in tif.series[frame_start:frame_stop])))
    #    return stack


from dask import delayed
import dask.array as da
class mesospim_btf_helper:

    def __init__(self,path: Union[str, Path]):
        self.path = path if isinstance(path,Path) else Path(path)
        self.tif = tifffile.TiffFile(self.path, mode='r')
        self.sample_data = self.tif.series[0].asarray()
        self.zdim = len(self.tif.series)

        self.build_lazy_array()

    def __getitem__(self, item):
        return self.lazy_array[item].compute()

    def build_lazy_array(self):
        print('Building Array')
        delayed_image_reads = [delayed(self.get_z_plane)(x) for x in range(self.zdim)]
        delayed_arrays = [da.from_delayed(x, shape=self.sample_data.shape, dtype=self.sample_data.dtype)[0] for x in delayed_image_reads]
        self.lazy_array = da.stack(delayed_arrays)
        print(self.lazy_array.shape)
        print(self.lazy_array)

    def get_z_plane(self,z_plane: int):
        return self.tif.series[z_plane].asarray()

    @property
    def shape(self):
        return self.lazy_array.shape

    @property
    def chunksize(self):
        return self.lazy_array.chunksize

    @property
    def nbytes(self):
        return self.lazy_array.nbytes

    @property
    def dtype(self):
        return self.lazy_array.dtype

    @property
    def chunks(self):
        return self.lazy_array.chunks

    @property
    def ndim(self):
        return self.lazy_array.ndim

    def __del__(self):
        del self.lazy_array
        self.tif.close()
        del self.tif


def mesospim_meta_data(metadata_dir: Path):
    return collect_all_metadata(metadata_dir)

def pad_reflect(input_tensor, padding_depth=0):
    # Calculate padding values
    pad_z, pad_y, pad_x = (padding_depth,) * 3

    # Pad the input tensor using reflection padding
    return F.pad(input_tensor, (pad_z, pad_z, pad_y, pad_y, pad_x, pad_x), mode='reflect')

    
def unpad(input_tensor, padding_depth=0):
    # Calculate padding values
    pad_z, pad_y, pad_x = (padding_depth,) * 3
    
    # Unpad the output tensor
    output = input_tensor[:, :, pad_z:-pad_z, pad_y:-pad_y, pad_x:-pad_x]

    return output

def unsharp(image, amount=2, sigma=0.2):
    # sharpened = original + (original − blurred) × amount.
    import torchvision.transforms as T
    blurred = image.clone()
    blurred[:,:] = T.functional.gaussian_blur(image[0,0], kernel_size=[7,7], sigma=(sigma, sigma))
    return image + (image - blurred) * amount

def normalize_array(image, min, max):
    '''Normalize an image between two values'''
    working = image - image.min()
    working[:] = working * ((max - min)/working.max())
    working += min
    return working


######################
## Helper functions ##
######################

def time_report(start_time=None, units='minutes', to_print=True):
    import time
    if start_time:
        diff_time = time.time() - start_time

        if units == 'minutes':
            diff_time = diff_time / 60

        if to_print:
            print(f'Total Time: {round(diff_time, 2)} {units}')
        return diff_time

    return time.time()

def save_image(file_name, array):
    print(f'Saving: {file_name}')
    skimage.io.imsave(file_name,array, plugin="tifffile", bigtiff=True)

    # import tifffile
    # tifffile.imwrite(
    #     'out_location',
    #     canvas,
    #     bigtiff=True,
    #     photometric='minisblack',
    #     metadata={'axes': 'ZYX'},
    # )


@app.command()
def decon(file_location: Path, refractive_index: float, out_location: Path=None, resolution: tuple[float,float,float]=None,
          emission_wavelength: int=None, na: float=None, ri: float=None,
          start_index: int=None, stop_index: int=None, save_pre_post: bool=False, queue_ims: bool=False,
          denoise_sigma: float=None, sharpen: bool=False, half_precision: bool=False,
          psf_shape: tuple[int,int,int]=(7,7,7), iterations: int=20, frames_per_chunk: int=75
          ):
    '''Deconvolution of a file using the richardson-lucy method'''

    start_time = time_report(start_time=None, to_print=True)

    FRAMES_PER_DECON = frames_per_chunk
    ITERATIONS = iterations
    PSF_SHAPE = psf_shape
    #ARRAY_PADDING = [[int((x*2)+1) for z in [1,1]] for x in PSF_SHAPE]
    SIGMA = denoise_sigma
    TMP_EXT = '.TMP'

    if not isinstance(file_location,Path):
        file_location = Path(file_location)
    print(f'Input File: {file_location}')

    if not out_location:
        print('Making out_location')
        out_location = file_location.parent / 'decon' / ('DECON_' + file_location.name)

    if not isinstance(out_location, Path):
        out_location = Path(out_location)
    print(f'Output File: {out_location}')
    # exit()
    out_location.parent.mkdir(parents=True, exist_ok=True)
    final_out_location = out_location
    # Make out location a .TMP file
    out_location = out_location.parent / (out_location.stem + TMP_EXT)

    if save_pre_post:
        pre_location = out_location.parent / ('PRE_' + out_location.stem + TMP_EXT)
        pre_location.parent.mkdir(parents=True, exist_ok=True)

    if not resolution:
        z_res = y_res = x_res = None

    if '.btf' in str(file_location):
        print('Is mesospim big tiff file')

        # Imaging parameters to be passes to psf
        na = 0.2
        # sample_ri = 1.47 #CUBIC
        sample_ri = refractive_index
        objective_immersion_ri_design = 1.000293 # Air
        objective_immersion_ri_actual = 1.000293 # Air
        objective_working_distance = 45 * 1000 #in microns
        coverslip_ri_design = 1.000293 # objective design
        coverslip_ri_actual = 1.515 # during experiment
        coverslip_thickness_actual = 1 * 1000 # in microns
        coverslip_thickness_design = 0

        psf_model = 'gaussian'  # Must be one of 'vectorial', 'scalar', 'gaussian'.

        # Extract imaging parameter from metadata file if it exists
        meta_dir = file_location.parent
        meta_dict = mesospim_meta_data(meta_dir)
        file_name = file_location.name
        meta_entry = get_entry_for_file_name(meta_dict,file_name)
        # ch, file_idx = get_ch_entry_for_file_name(meta_dict,file_location.name)
        # ch0 = list(meta_dict.keys())[file_idx]
        x_res, y_res, z_res = meta_entry.get('resolution')
        res = (z_res, y_res, x_res)

        emission_wavelength = meta_entry.get('emission_wavelength')


    assert all( (na, sample_ri, emission_wavelength, z_res, y_res, x_res) ), 'Some critical metadata parameters are not set'

    print(f'--- INPUT_LOCATION: {file_location}')
    print(f'--- OUT_LOCATION_TMP: {out_location}')
    print(f'--- OUT_LOCATION_FINAL: {final_out_location}')
    print(f'--- Depth of frames per run: {FRAMES_PER_DECON}')
    print(f'--- Iterations: {ITERATIONS}')
    print(f'--- NA: {na}')
    print(f'--- RI: {sample_ri}')
    print(f'--- Emission Wavelength: {emission_wavelength}')
    print(f'--- Lateral Resolution: X: {x_res}, Y: {y_res}')
    print(f'--- Z Steps: {z_res}')
    print(f'--- PSF Model: {psf_model}')

    if '.btf' in str(file_location):
        print('Opening File as mesospim_btf_helper')
        stack = mesospim_btf_helper(file_location)
        stack = stack.lazy_array
        stack = stack[start_index:stop_index]
        # print('Reading file')
        # stack = read_mesospim_btf(file_location, frame_start=start_index, frame_stop=stop_index)

    assert 'stack' in locals(), 'Image file has not been loaded, perhaps it is an unsupported format'

    print(f'FILE_SHAPE: {stack.shape}')

    print('Generating PSF')
    #psf = generate_psf_3d(PSF_SHAPE, (z_res,y_res,x_res), na, ri, emission_wavelength)
    #psf = generate_psf_3d_v2(PSF_SHAPE, (z_res, y_res, x_res), na, ri, emission_wavelength)
    #psf = generate_psf_3d_v3(PSF_SHAPE, (z_res, y_res, x_res), na, ri, emission_wavelength)
    psf = get_psf(
        z = PSF_SHAPE[0],
        nx = PSF_SHAPE[1],
        dxy = x_res,
        dz = z_res,
        NA = na,
        ns = sample_ri,
        ni = objective_immersion_ri_actual, # Immersion in air
        ni0 = objective_immersion_ri_design, # Air Obj
        wvl = emission_wavelength/1000,
        ti0 = objective_working_distance,
        tg = coverslip_thickness_actual, # coverslip thickness, experimental value (microns)
        tg0 = coverslip_thickness_design, # coverslip thickness, design value (microns)
        ng = coverslip_ri_actual, # coverslip refractive index, experimental value
        ng0 = coverslip_ri_design, # coverslip refractive index, design value
        model=psf_model, #Must be one of 'vectorial', 'scalar', 'gaussian'.
        normalize=True,
    )

    psf_image = out_location.parent / 'psf.tif'
    if not psf_image.is_file():
        save_image(psf_image, psf)

    # Pad array based on psf size
    ARRAY_PADDING = [[int((x * 2) + 1) for z in [1, 1]] for x in psf.shape]
    stack = np.pad(stack, ARRAY_PADDING, mode='reflect')
    print(f'PADDED_SHAPE: {stack.shape}')

    # Import pytorch dependencies
    print('Importing pytorch dependencies')
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    import torch.nn as nn

    if half_precision:
        psf = psf.astype(np.float16)
        psf = torch.tensor(psf, dtype=torch.float16).unsqueeze(0).unsqueeze(0)
    else:
        psf = torch.tensor(psf, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    print(f'--- PSF_SHAPE: {psf.shape}')

    start_z = 0
    stop_z = 0
    ideal_overlap_top = ARRAY_PADDING[0][0]
    ideal_overlap_btm = ARRAY_PADDING[0][1]

    canvas = np.zeros(stack.shape, dtype=stack.dtype)

    FRAMES_PER_DECON = FRAMES_PER_DECON - ideal_overlap_top - ideal_overlap_btm

    num_decon_chunks = len(range(0,stack.shape[0],FRAMES_PER_DECON))
    for idx, current in enumerate(range(0,stack.shape[0],FRAMES_PER_DECON)):
        chunk_start_time = time_report(start_time=None)

        max_frames_above = current
        max_frames_below = stack.shape[0] - current

        start_z = current - (ideal_overlap_top if max_frames_above > ideal_overlap_top else max_frames_above)
        start_chunk_idx = current-start_z
        stop_z = current + (ideal_overlap_btm if max_frames_below > ideal_overlap_btm else max_frames_below)
        new_end = stop_z + FRAMES_PER_DECON
        stop_z = new_end if new_end < stack.shape[0] else stack.shape[0]
        print(f'START: {start_z}, STOP: {stop_z}')
        print(f'Processing chunk {idx+1} of {num_decon_chunks}')

        # Reading in parallel show funky mix-ups with zlayers and image shift. This is attempt at loading 1 plane a time.
        to_decon_shape = stack[start_z:stop_z].shape
        to_decon = np.zeros(to_decon_shape, dtype=stack.dtype)
        print('Reading next chunk')
        for img_idx, img in enumerate(stack[start_z:stop_z]):
            to_decon[img_idx] = img.compute()
        # to_decon = stack[start_z:stop_z]
        # to_decon = to_decon.compute()
        to_decon = np.expand_dims(to_decon,0)
        to_decon = np.expand_dims(to_decon,0)
        to_decon = img_as_float32(to_decon)
        if half_precision:
            to_decon = to_decon.astype(np.float16)
        to_decon = torch.from_numpy(to_decon).cuda()
        #print(f"Prior Mean: {to_decon.mean()}, MIN: {to_decon.min()}, MAX: {to_decon.max()}")
        to_decon = richardson_lucy_3d(to_decon, psf, iterations=ITERATIONS, sigma=SIGMA)

        if sharpen:
            print('Unsharpening')
            to_decon = unsharp(to_decon, amount=2, sigma=2)

        to_decon = to_decon.cpu()
        to_decon = to_decon.numpy()

        to_decon = np.clip(to_decon, 0, 1)
        to_decon = img_as_uint(to_decon)
        to_decon = to_decon[0,0]

        to_decon = to_decon[start_chunk_idx:]

        canvas[current:stop_z] = to_decon

        torch.cuda.empty_cache()

        chunk_total_time = time_report(start_time=chunk_start_time, to_print=False)
        print(f'Chunk {idx+1} of {num_decon_chunks}: {round(chunk_total_time, 2)} minutes')

        accumulated_time = time_report(start_time=start_time, to_print=False)
        print(f'Accumulated Time: {round(accumulated_time, 2)} minutes')

    print('Deconvolution complete')

    # unpad canavs
    canvas = canvas[ARRAY_PADDING[0][0]:-ARRAY_PADDING[0][1], ARRAY_PADDING[1][0]:-ARRAY_PADDING[1][1],ARRAY_PADDING[2][0]:-ARRAY_PADDING[2][1]]

    print(f'Saving Deconvolved output: {out_location}')
    #save_image(out_location, canvas)
    with tifffile.TiffWriter(out_location, bigtiff=True) as tif:
        for img in canvas:
            tif.write(img[np.newaxis, ...], contiguous=False)

    if out_location.is_file():
        print(f'Renaming File: {out_location.name} to {final_out_location.name}')
        out_location.rename(final_out_location)

    if save_pre_post:
        stack = stack[ARRAY_PADDING[0][0]:-ARRAY_PADDING[0][1], ARRAY_PADDING[1][0]:-ARRAY_PADDING[1][1],
                ARRAY_PADDING[2][0]:-ARRAY_PADDING[2][1]]
        print(f'Saving Pre-Decon image: {pre_location}')
        with tifffile.TiffWriter(pre_location, bigtiff=True) as tif:
            for img in stack:
                tif.write(img[np.newaxis, ...], contiguous=False)
        if pre_location.is_file():
            new_name = pre_location.parent / (pre_location.stem + '.tif')
            print(f'Renaming File: {pre_location.name} to {new_name.name}')
            pre_location.rename(new_name)


    if queue_ims:
        #convert_ims(out_location, res=(z_res, y_res, x_res))
        convert_ims(out_location, res=(z_res, y_res, x_res))

    time_report(start_time=start_time, to_print=True)

if __name__ == "__main__":
    app()