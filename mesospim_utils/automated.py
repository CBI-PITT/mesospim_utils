import typer
from pathlib import Path
import subprocess



from constants import ENV_PYTHON_LOC, LOCATION_OF_MESOSPIM_UTILS_INSTALL
from metadata2 import collect_all_metadata
from slurm import convert_ims_dir_mesospim_tiles_slurm_array
from utils import ensure_path



app = typer.Typer()

@app.command()
def automated_method_slurm(dir_loc: Path, file_type: str = '.btf', decon: bool=False, stitch=True, convert_ims=True):
    '''
    Automate the processing of all data in a mesospim directory using slurm
    Stitching must be completed on windows which requires the separate stitch:run_windows_auto_stitch_client to be running
    '''

    dir_loc = ensure_path(dir_loc)
    mesospim_root_application = f'{ENV_PYTHON_LOC} {LOCATION_OF_MESOSPIM_UTILS_INSTALL}'

    # Collect all metadata from MesoSPIM acquisition directory and save to mesospim_metadata.json in the ims file dir
    metadata_by_channel = collect_all_metadata(dir_loc)
    first_channel = list(metadata_by_channel.keys())[0]
    res = metadata_by_channel[first_channel][0]['resolution']
    print(res)


    out_dir = None
    depends_job_list = None
    if stitch or convert_ims:
        slurm_job_number = convert_ims_dir_mesospim_tiles_slurm_array(dir_loc, file_type, res=(res.z,res.y,res.x), after_slurm_jobs=depends_job_list)

    if stitch:
        from stitch3 import write_auto_stitch_message
        from slurm import get_job_number_from_slurm_out
        stitch_cmd = f'{mesospim_root_application}/stitch3.py write-auto-stitch-message {dir_loc} {dir_loc / "ims_files"} {slurm_job_number} --name-of-montage-file automated_montage.ims'

        stitch_cmd = f'sbatch --depend=afterok:{slurm_job_number} -p compute -n 1 -J auto_stitch --wrap="{stitch_cmd}"'
        output = subprocess.run(stitch_cmd, shell=True, capture_output=True, text=True, check=True)
        job_number = get_job_number_from_slurm_out(output)
        # write_auto_stitch_message(dir_loc, dir_loc / 'ims_files', job_number=slurm_job_number,
        #                           name_of_montage_file='automated_montage.ims',
        #                           skip_align=False, skip_resample=False,
        #                           build_scripts_only=False)

        print(stitch_cmd)
        print(job_number)



if __name__ == "__main__":
    app()





# @app.command()
# def automated_method_slurm(dir_loc: str, file_type: str = '.btf', decon: bool=False, stitch=True, convert_ims=True, **kwargs):
#     '''
#     Automate the processing of all data in a mesospim directory using slurm
#     Stitching must be completed on windows which requires the separate stitch:run_windows_auto_stitch_client to be running
#     '''
#
#
#     out_dir = None
#     depends_job_list = None
#     if decon:
#         assert 'refractive_index' in kwargs, 'refractive index not present'
#         refractive_index = kwargs.get('refractive_index')
#         iterations = kwargs.get('iterations') if kwargs.get('iterations') else 40
#         frames_per_chunk = kwargs.get('frames_per_chunk') if kwargs.get('frames_per_chunk') else 70
#         num_parallel = kwargs.get('num_parallel') if kwargs.get('num_parallel') else 8
#         psf_shape = kwargs.get('psf_shape') if kwargs.get('psf_shape') else (7,7,7)
#         run_slurm = False
#         out_dir = dir_loc / 'decon'
#
#         if stitch:
#             convert_ims = True
#
#         loc_of_sbatch_file = decon_dir(dir_loc=dir_loc, refractive_index=refractive_index, out_dir=out_dir, queue_ims=convert_ims,
#                   iterations=iterations, frames_per_chunk=frames_per_chunk, num_parallel=num_parallel,
#                   psf_shape=psf_shape, run_slurm=run_slurm)
#
#     if