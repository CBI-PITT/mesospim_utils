import typer

app = typer.Typer()

@app.command()
def automated_method(dir_loc: str, file_type: str = '.btf', decon: bool=True, stitch=True, convert_ims=True, **kwargs):
    '''
    Automate the processing of all data in a mesospim directory
    Manage the process via slurm with the exception of stitching which must be completed on windows
    '''


    out_dir = None
    if decon:
        assert 'refractive_index' in kwargs, 'refractive index not present'
        refractive_index = kwargs.get('refractive_index')
        iterations = kwargs.get('iterations') if kwargs.get('iterations') else 40
        frames_per_chunk = kwargs.get('frames_per_chunk') if kwargs.get('frames_per_chunk') else 70
        num_parallel = kwargs.get('num_parallel') if kwargs.get('num_parallel') else 8
        psf_shape = kwargs.get('psf_shape') if kwargs.get('psf_shape') else (7,7,7)
        run_slurm = False
        out_dir = dir_loc / 'decon'

        if stitch:
            convert_ims = True

        loc_of_sbatch_file = decon_dir(dir_loc=dir_loc, refractive_index=refractive_index, out_dir=out_dir, queue_ims=convert_ims,
                  iterations=iterations, frames_per_chunk=frames_per_chunk, num_parallel=num_parallel,
                  psf_shape=psf_shape, run_slurm=run_slurm)



if __name__ == "__main__":
    app()

