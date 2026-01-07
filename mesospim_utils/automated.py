import typer
from pathlib import Path
import subprocess
import shutil

from constants import ENV_PYTHON_LOC, LOCATION_OF_MESOSPIM_UTILS_INSTALL, ALIGNMENT_DIRECTORY
from metadata import collect_all_metadata, get_first_entry
from slurm import convert_ims_dir_mesospim_tiles_slurm_array, decon_dir, wrap_slurm, sbatch_depends, format_sbatch_wrap
from utils import ensure_path
from bigstitcher import (does_dir_contain_bigstitcher_metadata,
                         get_ome_zarr_directory_from_xml,
                         get_bigstitcher_omezarr_alignment_marco,
                         make_bigstitcher_slurm_dir_and_macro,
                         adjust_scale_in_bigstitcher_produced_ome_zarr)


mesospim_root_application = f'{ENV_PYTHON_LOC} -u {LOCATION_OF_MESOSPIM_UTILS_INSTALL}'

app = typer.Typer()

@app.command()
def automated_method_slurm(dir_loc: Path, refractive_index: float=None, iterations: int=20, frames_per_chunk: int=None, file_type: str = '.btf', decon: bool=True, num_parallel: int=None):
    '''
    Automate the processing of all data in a mesospim directory using slurm
    Stitching must be completed on windows which requires the separate stitch:run_windows_auto_stitch_client to be running
    
    This script is designed to run quickly, queueing everything in SLURM and letting slurm manage dependencies and 
    downstream processes.

    Default:
    decon
    ims_conversion
    stitching
    '''

    dir_loc = ensure_path(dir_loc)

    # Ensure that metadata json is produced which will be used by downstream processes
    metadata_by_channel = collect_all_metadata(dir_loc)
    first_metadata_entry = get_first_entry(metadata_by_channel)
    username = first_metadata_entry.get('username', "")

    if not refractive_index and decon:
        refractive_index = first_metadata_entry.get('refractive_index')

    omezarr_xml = does_dir_contain_bigstitcher_metadata(dir_loc)  # returns path to omezarr xml if found, else None
    omezarr_path = get_ome_zarr_directory_from_xml(omezarr_xml) # returns path to omezarr_data relative to xml, else None
    dir_loc = omezarr_path if omezarr_path else dir_loc # switch to omezarr path if found
    file_type = '.ome.zarr' if omezarr_path else file_type

    job_number = None
    out_dir = dir_loc

    if refractive_index and decon: # for now skip decon if omezarr
        ## Decon:
        print('Queueing DECON of MesoSPIM tiles on SLURM')
        out_file_type = '.ome.zarr' if omezarr_path else '.tif'
        job_number, out_dir = decon_dir(dir_loc, refractive_index, file_type=file_type, out_file_type=out_file_type, iterations=iterations, frames_per_chunk=frames_per_chunk, num_parallel=num_parallel)
        print((job_number, out_dir))
        file_type = out_file_type

    if not omezarr_path:
        # IMS Convert
        # Dependency process that kicks off IMS build following DECON
        print('Setting up script to manage IMS conversions after DECON')
        from constants import SLURM_PARAMETERS_FOR_DEPENDENCIES
        cmd = ''
        cmd += f'{mesospim_root_application}/automated.py ims-conv-then-align'
        cmd += f' {out_dir} {dir_loc} --file-type={file_type}'
        job_number = wrap_slurm(cmd, SLURM_PARAMETERS_FOR_DEPENDENCIES, out_dir,
                                after_slurm_jobs=[job_number] if job_number else None, username=username)
        print(f'Dependency process number: {job_number}')

    elif omezarr_path:
        # Dependency process that kicks off Bigstitcher alignment following DECON
        print('Setting up script to manage Bigstitcher conversions after DECON')
        from constants import SLURM_PARAMETERS_FOR_DEPENDENCIES
        cmd = ''
        cmd += f'{mesospim_root_application}/automated.py big-stitcher-align'
        cmd += f' {Path(out_dir).parent}'
        job_number = wrap_slurm(cmd, SLURM_PARAMETERS_FOR_DEPENDENCIES, out_dir,
                                after_slurm_jobs=[job_number] if job_number else None, username=username)
        print(f'Dependency process number: {job_number}')



@app.command()
def big_stitcher_align(dir_loc: Path):
    print('Setting up script to run omezarr alignment')
    from constants import SLURM_PARAMETERS_FOR_BIGSTITCHER
    from constants import SLURM_PARAMETERS_FOR_DEPENDENCIES
    from string_templates import BIGSTITCHER_ALIGN_OMEZARR_TEMPLATE

    print(f'Extracting metadata from {dir_loc}')
    metadata_by_channel = collect_all_metadata(dir_loc)
    first_metadata_entry = get_first_entry(metadata_by_channel)

    username = first_metadata_entry.get('username', "")

    bigstitcher_dir, fused_out_dir, macro_file = make_bigstitcher_slurm_dir_and_macro(dir_loc)
    cmd = BIGSTITCHER_ALIGN_OMEZARR_TEMPLATE.format(macro_file)

    job_number = None
    job_number = wrap_slurm(cmd, SLURM_PARAMETERS_FOR_BIGSTITCHER, bigstitcher_dir,
                            after_slurm_jobs=[job_number] if job_number else None, username=username)
    print(f'BigStitcher process number: {job_number}')

    cmd = f'{mesospim_root_application}/bigstitcher.py adjust-scale-in-bigstitcher-produced-ome-zarr'
    cmd += f' "{dir_loc}" "{fused_out_dir}"'
    job_number = wrap_slurm(cmd, SLURM_PARAMETERS_FOR_DEPENDENCIES, bigstitcher_dir,
                            after_slurm_jobs=[job_number] if job_number else None, username=username)
    print(f'BigStitcher Fix OME-Zarr Metadata Scale: {job_number}')



@app.command()
def ims_conv_then_align(dir_loc: Path, metadata_dir: Path, file_type: str='.tif', ims_convert: bool=True):

    # Collect all metadata from MesoSPIM acquisition directory and save to mesospim_metadata.json in the ims file dir
    print(f'Extracting metadata from {metadata_dir}')
    metadata_by_channel = collect_all_metadata(metadata_dir)
    first_metadata_entry = get_first_entry(metadata_by_channel)
    res = first_metadata_entry.get('resolution')
    print(f'Resolution of mesospim tiles: {res}')

    username = first_metadata_entry.get('username',"")

    job_number = None
    out_dir = dir_loc
    if ims_convert:
        # IMS Convert
        print('Setting queueing IMS conversions on SLURM')
        from slurm import convert_ims_dir_mesospim_tiles_slurm_array
        job_number, out_dir = convert_ims_dir_mesospim_tiles_slurm_array(dir_loc, file_type=file_type, res=(res.z,res.y,res.x))

    # Dependency process that kicks off alignment following IMS Convert
    print('Setting up script to manage alignment calculation after IMS conversion')
    from constants import SLURM_PARAMETERS_FOR_MESOSPIM_ALIGN
    cmd = ''
    cmd += f'{mesospim_root_application}/align_py.py'
    cmd += f' {metadata_dir} {out_dir}'
    if job_number:
        job_number = wrap_slurm(cmd, SLURM_PARAMETERS_FOR_MESOSPIM_ALIGN, out_dir / ALIGNMENT_DIRECTORY,
                                after_slurm_jobs=[job_number], username=username)
        print(f'Dependency process number: {job_number}')
    else:
        job_number = wrap_slurm(cmd, SLURM_PARAMETERS_FOR_MESOSPIM_ALIGN, out_dir / ALIGNMENT_DIRECTORY,
                                after_slurm_jobs=None, username=username)

    # Dependency process that kicks off windows resampling
    print('Setting up script to manage resampling after alignment')
    from constants import SLURM_PARAMETERS_FOR_DEPENDENCIES
    cmd = ''
    cmd += f'{mesospim_root_application}/resample_ims.py write-auto-resample-message'
    cmd += f' {metadata_dir} {out_dir} {job_number}'
    job_number = wrap_slurm(cmd, SLURM_PARAMETERS_FOR_DEPENDENCIES, out_dir / ALIGNMENT_DIRECTORY, after_slurm_jobs=[job_number], username=username)
    print(f'Dependency process number: {job_number}')



if __name__ == "__main__":
    app()

