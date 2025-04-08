import typer
from pathlib import Path
import subprocess

from constants import ENV_PYTHON_LOC, LOCATION_OF_MESOSPIM_UTILS_INSTALL
from metadata import collect_all_metadata, get_first_entry
from slurm import convert_ims_dir_mesospim_tiles_slurm_array, decon_dir, wrap_slurm, sbatch_depends, format_sbatch_wrap
from utils import ensure_path


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

    job_number = None
    out_dir = dir_loc
    if not refractive_index and decon:
        first_metadata_entry = get_first_entry(metadata_by_channel)
        refractive_index = first_metadata_entry.get('refractive_index')

    if refractive_index and decon:
        ## Decon:
        print('Queueing DECON of MesoSPIM tiles on SLURM')
        job_number, out_dir = decon_dir(dir_loc, refractive_index, iterations=iterations, frames_per_chunk=frames_per_chunk, num_parallel=num_parallel)
        print((job_number, out_dir))
        file_type = '.tif'

    # Dependency process that kicks off IMS build following DECON
    print('Setting up script to manage IMS conversions after DECON')
    from constants import SLURM_PARAMETERS_FOR_DEPENDENCIES
    cmd = ''
    cmd += f'{mesospim_root_application}/automated.py ims-conv-then-stitch'
    cmd += f' {out_dir} {dir_loc} --file-type={file_type}'
    job_number = wrap_slurm(cmd, SLURM_PARAMETERS_FOR_DEPENDENCIES, out_dir, after_slurm_jobs=[job_number] if job_number else None)
    print(f'Dependency process number: {job_number}')


@app.command()
def ims_conv_then_stitch(dir_loc: Path, metadata_dir: Path, file_type: str='.tif'):

    # Collect all metadata from MesoSPIM acquisition directory and save to mesospim_metadata.json in the ims file dir
    print(f'Extracting metadata from {metadata_dir}')
    metadata_by_channel = collect_all_metadata(metadata_dir)
    first_metadata_entry = get_first_entry(metadata_by_channel)
    res = first_metadata_entry.get('resolution')
    print(f'Resolution of mesospim tiles: {res}')
    
    
    # IMS Convert
    print('Setting queueing IMS conversions on SLURM')
    from slurm import convert_ims_dir_mesospim_tiles_slurm_array
    job_number, out_dir = convert_ims_dir_mesospim_tiles_slurm_array(dir_loc, file_type=file_type, res=(res.z,res.y,res.x))

    # Dependency process that kicks off IMS build following DECON
    print('Setting up script to manage Stitching after IMS conversion')
    from constants import SLURM_PARAMETERS_FOR_DEPENDENCIES
    cmd = ''
    cmd += f'{mesospim_root_application}/stitch.py write-auto-stitch-message'
    cmd += f' {metadata_dir} {out_dir} {job_number}'
    job_number = wrap_slurm(cmd, SLURM_PARAMETERS_FOR_DEPENDENCIES, out_dir, after_slurm_jobs=[job_number])
    print(f'Dependency process number: {job_number}')



if __name__ == "__main__":
    app()

