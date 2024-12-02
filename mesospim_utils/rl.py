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
import os
import sys

import numpy as np
import tifffile
import skimage
from skimage import img_as_float32, img_as_uint

#######################################
# CHANGE THESE CONSTANTS IF NECESSARY #
#######################################
ENV_PYTHON_LOC = '/h20/home/lab/miniconda3/envs/decon/bin/python'
# Append -u so unbuffered outputs scroll in realtime to slurm out files
ENV_PYTHON_LOC = f'{ENV_PYTHON_LOC} -u'

LOC_OF_THIS_SCRIPT = '/CBI_FastStore/cbiPythonTools/mesospim_utils/mesospim_utils/rl.py'

WINE_INSTALL_LOC = '/h20/home/lab/src/wine/wine64'
IMARIS_CONVERTER_LOC = '/h20/home/lab/src/ImarisFileConverter 10.2.0/ImarisConvert.exe'

# Drive mappings for linux directories in wine for ims file concersions
MAPPINGS = { #linux_path:wine_drive_letter:
'/h20':'h:',
'/CBI_FastStore':'f:',
}

app = typer.Typer()

@app.command()
def convert_ims(file: str, res: tuple[float,float,float]=(1,1,1)):
    '''Convert a single file to ims format and spin off the process on slurm compute node [executed on SLURM]'''

    SLURM_PARTITION = 'compute'
    SLURM_CPUS = 24
    SLURM_JOB_LABEL = 'ims_conv'
    RAM = 64 #G
    res_z, res_y, res_x = res

    file = Path(file)
    
    out_file = file.parent / 'ims_files' / (file.stem + '.ims')
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    layout_path = out_file.parent / 'ims_convert_layouts' / (out_file.stem + '_layout.txt')
    layout_path.parent.mkdir(parents=True, exist_ok=True)
    
    log_location = out_file.parent / 'ims_convert_logs' / (out_file.stem + '.txt')
    log_location.parent.mkdir(parents=True, exist_ok=True)

    #line = f'''<?xml version = "1.0" encoding = "UTF-8"?>
    #<AllFileNames>
    #  <BaseDescription>TIFF (adjustable file series)</BaseDescription>
    #  <FileName>{path_to_wine_mappings(file)}</FileName>
    #  <AllFileNamesOfDataSet>
    #    <File mIndex="0" mName="{path_to_wine_mappings(file)}"/>
    #  </AllFileNamesOfDataSet>
    #  <NumberOfImages>1</NumberOfImages>
    #</AllFileNames>'''
    
    line = f'''<FileSeriesLayout>
    <ImageIndex name="{path_to_wine_mappings(file)}" x="0" y="0" z="0" c="0" t="0"/></FileSeriesLayout>'''

    print(line)
    print(file)
    print(out_file)
    with layout_path.open("w") as f:
        f.write(line)
    
    # Main ims converter command
    lines = f'{WINE_INSTALL_LOC} "{IMARIS_CONVERTER_LOC}" --voxelsizex {res_x} --voxelsizey {res_y} --voxelsizez {res_z} -i "{path_to_wine_mappings(file)}" -o "{path_to_wine_mappings(out_file).as_posix() + ".part"}" -il "{path_to_wine_mappings(layout_path)}" --logprogress --nthreads {SLURM_CPUS} --compression 6 -ps {RAM*1024} -of Imaris5 -a'
    
    # BASH script if/then statements to rename the .ims.part file to .ims
    lines = lines + f'\n\nif [ -f "{out_file}.part" ]; then\n  mv "{out_file}.part" "{out_file}"\n  echo "File renamed to {out_file}"\nelse\n  echo "File {out_file} does not exist."\nfi'
    
    # Sbatch command wrapping the 
    sbatch_cmd = f"sbatch -p {SLURM_PARTITION} -n {SLURM_CPUS} --mem={RAM}G -J {SLURM_JOB_LABEL} -o {log_location} --wrap='{lines}'"
    subprocess.run(sbatch_cmd, shell=True)
    print(lines)
    print(sbatch_cmd)

@app.command()
def ims_dir(dir_loc: str, file_type: str='.tif', res: tuple[float,float,float]=(1,1,1)):
    '''Convert all files in a directory to Imaris Files [executed on SLURM]'''
    path = Path(dir_loc)
    file_list = path.glob('*' + file_type)
    for p in file_list:
        convert_ims(p, res=res)


def decon_dir_OLD(dir_loc: str, out_dir: str=None, out_file_type: str='.tif', file_type: str='.btf',
              queue_ims: bool=False, sharpen: bool=False, denoise_sigma: float=None):
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
    cmd = ["sbatch", "-p gpu", "--gres=gpu:1", "-J decon"]
    file_list = path.glob('*' + file_type)
    #print(list(file_list))
    for p in file_list:
        to_run = cmd + [f'-o {log_dir / (p.stem + ".txt")}']
        to_run = to_run + [f'--wrap="{ENV_PYTHON_LOC} {LOC_OF_THIS_SCRIPT} decon']
        to_run = to_run + [p.as_posix()] 
        to_run = to_run + [f'--out-location {out_dir / (p.stem + out_file_type)}']
        to_run = to_run + [f'{" --queue-ims" if queue_ims else ""}']
        to_run = to_run + [f'{" --sharpen" if sharpen else ""}']
        to_run = to_run + [f'{" --denoise-sigma {denoise_sigma}" if denoise_sigma else ""}']
        to_run = to_run + ['"']
        print(to_run)
        subprocess.run(' '.join(to_run), shell=True)

@app.command()
def decon_dir(dir_loc: str, out_dir: str=None, out_file_type: str='.tif', file_type: str='.btf',
              queue_ims: bool=False, sharpen: bool=False, denoise_sigma: float=None, num_parallel: int=4):
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

    SBATCH_ARG = '#SBATCH {}'
    to_run = ["sbatch", "-p gpu", "--gres=gpu:1", "-J decon", f'-o {log_dir} / %A_%a.log', f'--array=0-{num_files-1}']
    commands = "#!/bin/bash\n"
    commands += SBATCH_ARG.format('-p gpu\n')
    commands += SBATCH_ARG.format('--gres=gpu:1\n')
    commands += SBATCH_ARG.format('-J decon\n')
    commands += SBATCH_ARG.format(f'-o {log_dir}/%A_%a.log\n')
    commands += SBATCH_ARG.format(f'--array=0-{num_files-1}{"%" + num_parallel if num_parallel>0 else ""}\n\n')

    commands += "commands=("
    #Build each command
    for p in file_list:
        commands += '\n\t'
        commands += '"'
        commands += f'{ENV_PYTHON_LOC} {LOC_OF_THIS_SCRIPT} decon '
        commands += f'{p.as_posix()} '
        commands += f'--out-location {out_dir / (p.stem + out_file_type)}'
        commands += f'{" --queue-ims" if queue_ims else ""}'
        commands += f'{" --sharpen" if sharpen else ""}'
        commands += f'{" --denoise-sigma {denoise_sigma}" if denoise_sigma else ""}'
        commands += '"'
    commands += '\n)\n\n'
    commands += 'echo "Running command: ${commands[$SLURM_ARRAY_TASK_ID]}"\n'
    commands += 'eval "${commands[$SLURM_ARRAY_TASK_ID]}"'

    file_to_run = out_dir / 'slurm_array.sh'
    with open(file_to_run, 'w') as f:
        f.write(commands)

    subprocess.run(f'sbatch {file_to_run}', shell=True)

@app.command()
def decon(file_location: Path, out_location: Path=None, resolution: tuple[float,float,float]=None,
          emission_wavelength: int=None, na: float=None, ri: float=None,
          start_index: int=None, stop_index: int=None, save_pre_post: bool=False, queue_ims: bool=False,
          denoise_sigma: float=None, sharpen: bool=False):
    '''Deconvolution of a file using the richardson-lucy method'''

    start_time = time_report(start_time=None, to_print=True)

    FRAMES_PER_DECON = 50
    ITERATIONS = 20
    PSF_SHAPE = (7,7,7) #Max size decreased automatically based on threshold value in enerate_psf_3d_v3 method
    #ARRAY_PADDING = [[int((x*2)+1) for z in [1,1]] for x in PSF_SHAPE]
    SIGMA = denoise_sigma
    
    if not isinstance(file_location,Path):
        file_location = Path(file_location)
    print(f'Input File: {file_location}')
    
    if not out_location:
        print('Making out_location')
        out_location = file_location.parent / 'decon' / ('DECON_' + file_location.stem + '.tif')

    if not isinstance(out_location, Path):
        out_location = Path(out_location)
    print(f'Output File: {out_location}')
    # exit()
    out_location.parent.mkdir(parents=True, exist_ok=True)

    if save_pre_post:
        pre_location = out_location.parent / ('PRE_' + out_location.stem + '.tif')
        pre_location.parent.mkdir(parents=True, exist_ok=True)
    
    if not resolution:
        z_res = y_res = x_res = None
    
    if '.btf' in str(file_location):
        print('Is mesospim big tiff file')
        
        na = 0.2
        ri = 1.0
        
        # Extract imaging parameter from metadata file if it exists
        meta_file = file_location.with_name(file_location.name + '_meta.txt')
        print(meta_file)
        if meta_file.is_file():
            with meta_file.open('r') as f:
                metadata = f.readlines()
            
            if not emission_wavelength:
                emission_wavelength = [x for x in metadata if '[Filter]' in x][0][9:12]
                emission_wavelength = int(emission_wavelength)
            
            if not resolution:
                xy_res = [x for x in metadata if '[Pixelsize in um]' in x][0][18:-1]
                xy_res = float(xy_res)
                y_res = x_res = xy_res
                
                z_res = [x for x in metadata if '[z_stepsize]' in x][0][13:-1]
                z_res = float(z_res)
                
    
    assert all( (na, ri, emission_wavelength, z_res, y_res, x_res) ), 'Some critical metadata parameters are not set'

    print(f'--- INPUT_LOCATION: {file_location}')
    print(f'--- OUT_LOCATION: {out_location}')
    print(f'--- Depth of chunks per run: {FRAMES_PER_DECON}')
    print(f'--- Iterations: {ITERATIONS}')
    print(f'--- NA: {na}')
    print(f'--- RI: {ri}')
    print(f'--- Emission Wavelength: {emission_wavelength}')
    print(f'--- Lateral Resolution: {xy_res}')
    print(f'--- Z Steps: {z_res}')

    if '.btf' in str(file_location):
        print('Reading file')
        stack = read_mesospim_btf(file_location, frame_start=start_index, frame_stop=stop_index)

    assert 'stack' in locals(), 'Image file has not been loaded, perhaps it is an unsupported format'

    print(f'FILE_SHAPE: {stack.shape}')
    
    print('Generating PSF')
    #psf = generate_psf_3d(PSF_SHAPE, (z_res,y_res,x_res), na, ri, emission_wavelength)
    #psf = generate_psf_3d_v2(PSF_SHAPE, (z_res, y_res, x_res), na, ri, emission_wavelength)
    psf = generate_psf_3d_v3(PSF_SHAPE, (z_res, y_res, x_res), na, ri, emission_wavelength)

    # Pad array based on psf size
    ARRAY_PADDING = [[int((x * 2) + 1) for z in [1, 1]] for x in psf.shape]
    stack = np.pad(stack, ARRAY_PADDING, mode='reflect')
    print(f'PADDED_SHAPE: {stack.shape}')

    # Import pytorch dependencies
    print('Importing pytorch dependencies')
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T

    psf = torch.tensor(psf, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    print(f'--- PSF_SHAPE: {psf.shape}')

    start_z = 0
    stop_z = 0
    ideal_overlap_top = ARRAY_PADDING[0][0]
    ideal_overlap_btm = ARRAY_PADDING[0][1]
    
    canvas = np.zeros_like(stack)
    
    num_decon_chunks = len(range(0,stack.shape[0],FRAMES_PER_DECON))
    for idx, current in enumerate(range(0,stack.shape[0],FRAMES_PER_DECON)):
        max_frames_above = current
        max_frames_below = stack.shape[0] - current
        
        start_z = current - (ideal_overlap_top if max_frames_above > ideal_overlap_top else max_frames_above)
        start_chunk_idx = current-start_z
        stop_z = current + (ideal_overlap_btm if max_frames_below > ideal_overlap_btm else max_frames_below)
        new_end = stop_z + FRAMES_PER_DECON
        stop_z = new_end if new_end < stack.shape[0] else stack.shape[0]
        print(f'START: {start_z}, STOP: {stop_z}')
        print(f'Processing chunk {idx+1} of {num_decon_chunks}')
        
        to_decon = stack[start_z:stop_z]
        to_decon = np.expand_dims(to_decon,0)
        to_decon = np.expand_dims(to_decon,0)
        to_decon = img_as_float32(to_decon)
        to_decon = torch.from_numpy(to_decon).cuda()
        #print(f"Prior Mean: {to_decon.mean()}, MIN: {to_decon.min()}, MAX: {to_decon.max()}")
        to_decon = richardson_lucy_3d(to_decon, psf, iterations=ITERATIONS, sigma=SIGMA)

        torch.cuda.empty_cache()
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
    
    print('Deconvolution complete')
    
    # unpad canavs
    canvas = canvas[ARRAY_PADDING[0][0]:-ARRAY_PADDING[0][1], ARRAY_PADDING[1][0]:-ARRAY_PADDING[1][1],ARRAY_PADDING[2][0]:-ARRAY_PADDING[2][1]]

    save_image(out_location, canvas)
    
    if save_pre_post:
        stack = stack[ARRAY_PADDING[0][0]:-ARRAY_PADDING[0][1], ARRAY_PADDING[1][0]:-ARRAY_PADDING[1][1],ARRAY_PADDING[2][0]:-ARRAY_PADDING[2][1]]
        save_image(pre_location, stack)

    if queue_ims:
        convert_ims(out_location, res=(z_res, y_res, x_res))

    time_report(start_time=start_time, to_print=True)


def read_mesospim_btf(file_location: Union[str, Path], frame_start: int=None, frame_stop: int=None):
    with tifffile.TiffFile(file_location) as tif:
        print(f'File contains {len(tif.series)} frames')
        stack = np.stack(tuple((x.asarray()[0] for x in tif.series[frame_start:frame_stop])))
        return stack
    
    #with open(file_location, 'rb') as f:
    #    from io import BytesIO
    #    image_file = BytesIO(f.read())
    #with tifffile.TiffFile(image_file) as tif:
    #    print(f'File contains {len(tif.series)} frames')
    #    stack = np.stack(tuple((x.asarray()[0] for x in tif.series[frame_start:frame_stop])))
    #    return stack



def generate_psf_3d(size, scale, NA, RI, wavelength):
    """
    Generate a 3D PSF based on optical parameters.
    
    Args:
        size (tuple): Shape of the PSF (D, H, W).
        scale (tuple): Scale of voxels in microns (z, y, x).
        NA (float): Numerical aperture of the system.
        RI (float): Refractive index of the medium.
        wavelength (float): Wavelength of light in microns.

    Returns:
        torch.Tensor: PSF with the given parameters.
    """
    z, y, x = size
    z_scale, y_scale, x_scale = scale

    # Calculate diffraction-limited resolution
    resolution_xy = 0.61 * wavelength / NA
    resolution_z = 2 * wavelength / (NA**2)

    # Create coordinate grids
    z_range = np.linspace(-(z // 2) * z_scale, (z // 2) * z_scale, z)
    y_range = np.linspace(-(y // 2) * y_scale, (y // 2) * y_scale, y)
    x_range = np.linspace(-(x // 2) * x_scale, (x // 2) * x_scale, x)
    zz, yy, xx = np.meshgrid(z_range+1, y_range+1, x_range+1, indexing="ij")

    # Generate Gaussian approximation of the PSF
    psf = np.exp(
        -((xx**2 + yy**2) / (2 * resolution_xy**2) + (zz**2) / (2 * resolution_z**2))
    )

    # Normalize PSF
    psf /= psf.sum()
    
    print(f'PSF SHAPE: {psf.shape}')
    print(psf)

    return psf
    #return torch.tensor(psf, dtype=torch.float32)

def generate_psf_3d_v2(size: tuple[int,int,int]=(3,3,3), scale: tuple[float,float,float]=(1,1,1),
        na: float=0.2, ri: float=1.0, wavelength: int=488):
    """
    size, scale, NA, RI, wavelength
    Estimate the 3D PSF for a microscopy system.

    Parameters:
    - na: Numerical Aperture (unitless)
    - wavelength: Emission wavelength (in nanometers)
    - resolution_xy: Pixel size in the xy-plane (in microns)
    - resolution_z: Pixel size in the z-dimension (in microns)
    - size: Tuple indicating the shape of the PSF (z, y, x)

    Returns:
    - psf: 3D PSF array
    """

    resolution_z, resolution_xy, _ = scale

    # Convert from nm to um
    wavelength /= 1000

    # Adjust na and wavelength for RI
    na = min(na, ri)
    wavelength = wavelength / ri

    # Compute FWHM in microns
    fwhm_xy = 0.61 * wavelength / na
    fwhm_z = 2 * wavelength / (na ** 2)

    # Convert FWHM to standard deviation in voxels
    sigma_xy = fwhm_xy / (2 * np.sqrt(2 * np.log(2))) / resolution_xy
    sigma_z = fwhm_z / (2 * np.sqrt(2 * np.log(2))) / resolution_z

    # Create a 3D Gaussian kernel
    z, y, x = size
    z_range = np.linspace(-(z // 2), z // 2, z)
    y_range = np.linspace(-(y // 2), y // 2, y)
    x_range = np.linspace(-(x // 2), x // 2, x)
    zz, yy, xx = np.meshgrid(z_range, y_range, x_range, indexing="ij")

    psf = np.exp(
        -((xx / sigma_xy) ** 2 + (yy / sigma_xy) ** 2 + (zz / sigma_z) ** 2) / 2
    )

    #psf /= psf.sum() # Normalize PSF (More accepted version)
    psf /= psf.max()  # Normalize PSF
    print(f'PSF SHAPE: {psf.shape}')
    #print(psf)
    for ii in psf[0,0]:
        print('\n\n')
    return psf

@app.command()
def generate_psf_3d_v3(size: tuple[int,int,int]=(3,3,3), scale: tuple[float,float,float]=(1,1,1),
        na: float=0.2, ri: float=1.0, wavelength: int=488):
    """
    Auto adjust size of PSF based on threshold
    size, scale, NA, RI, wavelength
    Estimate the 3D PSF for a microscopy system.

    Parameters:
    - na: Numerical Aperture (unitless)
    - wavelength: Emission wavelength (in nanometers)
    - resolution_xy: Pixel size in the xy-plane (in microns)
    - resolution_z: Pixel size in the z-dimension (in microns)
    - size: Tuple indicating the shape of the PSF (z, y, x)

    Returns:
    - psf: 3D PSF array
    """

    resolution_z, resolution_y, resolution_x = scale

    # Convert from nm to um
    wavelength /= 1000

    # Adjust na and wavelength for RI
    na = min(na, ri)
    wavelength = wavelength / ri

    # Compute FWHM in microns
    fwhm_xy = 0.61 * wavelength / na
    fwhm_z = 2 * wavelength / (na ** 2)

    # Convert FWHM to standard deviation in voxels
    sigma_y = fwhm_xy / (2 * np.sqrt(2 * np.log(2))) / resolution_y
    sigma_x = fwhm_xy / (2 * np.sqrt(2 * np.log(2))) / resolution_x
    sigma_z = fwhm_z / (2 * np.sqrt(2 * np.log(2))) / resolution_z

    # Create a 3D Gaussian kernel
    z, y, x = size

    z_range = np.linspace(-(z // 2), z // 2, z)
    y_range = np.linspace(-(y // 2), y // 2, y)
    x_range = np.linspace(-(x // 2), x // 2, x)
    # print(z_range)
    # print(y_range)
    # print(x_range)
    zz, yy, xx = np.meshgrid(z_range, y_range, x_range, indexing="ij")

    psf = np.exp(
        -((xx / sigma_x) ** 2 + (yy / sigma_y) ** 2 + (zz / sigma_z) ** 2) / 2
    )

    #psf /= psf.sum() # Normalize PSF (More accepted version)
    psf /= psf.max()  # Normalize PSF
    print(f'PSF SHAPE: {psf.shape}')
    #print(psf)
    THRESHOLD = 0.0001
    while (psf[0]<THRESHOLD).all() and (psf[-1]<THRESHOLD).all():
        psf = psf[1:]
        psf = psf[:-1]
    while (psf[:,0]<THRESHOLD).all() and (psf[:,-1]<THRESHOLD).all():
        psf = psf[:,1:]
        psf = psf[:,:-1]
    while (psf[:,:,0]<THRESHOLD).all() and (psf[:,:,-1]<THRESHOLD).all():
        psf = psf[:,:,1:]
        psf = psf[:,:,:-1]

    for ii in psf:
        #print(ii)
        print(np.round(ii, 4))
        print('\n')
    print(psf.shape)
    return psf

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

    # Ensure data and PSF are on the same device
    print('Loading data onto GPU')
    input_data = input_data.to('cuda')
    psf = psf.to('cuda')

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
    relative_blur = convolved.clone()
    correction = relative_blur.clone()
    estimate = correction.clone()
    #estimate = torch.full_like(correction,0.5)
    
    print('Beginning deconvolution')
    #print(f"Iteration {0}/{iterations}, Correction Mean: {correction.mean().item()}, MIN: {estimate.min().item()}, MAX: {estimate.max().item()}")
    start_time = time_report(start_time=None, units='minutes')
    for i in range(iterations):
        itter_start = time_report(start_time=None, units='minutes')
        # Convolve estimate with PSF
        convolved[:] = F.conv3d(estimate, psf, padding='same')
        #convolved[:] = F.conv3d(estimate, psf, padding_mode='reflect')

        # Calculate the ratio of input data to the convolved data
        relative_blur[:] = input_data / (convolved + 1e-12)

        # Convolve the relative blur with the flipped PSF
        correction[:] = F.conv3d(relative_blur, psf_flipped, padding='same')
        #correction[:] = F.conv3d(relative_blur, psf_flipped, padding_mode='reflect')

        # Update the estimate
        estimate *= correction

        # Print progress
        if i % 1 == 0:
            torch.cuda.synchronize()
            #print(f"Iteration {i+1}/{iterations}, Time: {round(time_report(start_time=itter_start, to_print=False), 2)/60} Sec, MIN: {estimate.min().item()}, MAX: {estimate.max().item()}")
            print(f"Iteration {i+1}/{iterations}, Time: {round(time_report(start_time=itter_start, to_print=False), 2)/60} Sec")

    # estimate = normalize_array(estimate, input_data.min(), input_data.max())
    # estimate = unpad(estimate, padding_depth=tuple(psf.shape)[-1]*2)
    print('Deconvolution Complete')
    time_report(start_time=start_time, to_print=True)
    return estimate

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
def path_to_wine_mappings(path: Path) -> Path:
    '''Convert a linux path to wine (windows) paths using MAPPINGS defined at top of script'''
    if isinstance(path, Path):
        path = path.as_posix()
    for key in MAPPINGS:
        path = path.replace(key,MAPPINGS[key])
    path = path.replace('/','\\')
    print(path)
    return Path(path)

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

# Example Usage
if __name__ == "__main__":
    
    app()

r"""
/h20/home/lab/src/wine/wine64 "/h20/home/lab/src/ImarisFileConverter 10.2.0/ImarisConvert.exe" --voxelsizex 1.0 --voxelsizey 1.0 --voxelsizez 1.0 -i "f:/toTest/mesospim/112524/decon/DECON_stujfos1_reimage_Tile7_Ch488_Sh0.tif" -o "f:/toTest/mesospim/112524/decon/DECON_stujfos1_reimage_Tile7_Ch488_Sh0.ims.part" -il "f:/toTest/mesospim/112524/decon/DECON_stujfos1_reimage_Tile7_Ch488_Sh0_layout.txt" -a

# Actual command running a single file conversion using the iamrisfilconverter on windows system
"C:\Program Files\Bitplane\Imaris 10.0.0\ImarisConvert.exe" -lp -ti 500 -i "Z:\toTest\mesospim\112524\decon\DECON_stujfos1_reimage_Tile23_Ch647_Sh0.tif" -ii 0 -nt 64 -of Imaris5 -o "z:\totest\mesospim\112524\decon\decon_stujfos1_reimage_tile23_ch647_sh0.ims.part" -vs "1.000 1.000 5.000" -c "0" -vsx 1.000 -vsy 1.000 -vsz 5.000 --crash-reporter-handler-pipe \\.\pipe\crashpad_20964_HXDBKGBIHRBJCLJT -il "C:\Users\awatson\AppData\Local\Temp\layout-20964-0.txt" -dcl "#FF0000" "#00FF00" "#0000FF" "#FFFFFF" -fsdx "_X;-X" -fsdy "_Y;-Y" -fsdz "_Z;_S;-Z" -fsdc "_C;_CH;_W;-C" -fsdt "_T;_TP;-T" -fsdf ""

Contents of the layout file referenced above: "C:\Users\awatson\AppData\Local\Temp\layout-20964-0.txt"
<FileSeriesLayout>
<ImageIndex name="Z:\toTest\mesospim\112524\decon\DECON_stujfos1_reimage_Tile0_Ch488_Sh0.tif" x="0" y="0" z="0" c="0" t="0"/></FileSeriesLayout>
"""