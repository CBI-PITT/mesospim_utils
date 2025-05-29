import typer
from pathlib import Path
import subprocess
import os


from imaris import convert_ims, nested_list_tile_files_sorted_by_color
from rl import decon
from utils import path_to_wine_mappings, ensure_path, get_user, get_file_size_gb

app = typer.Typer()


######################################################################################################################
####  DECON FUNCTIONS TO HANDLE SLURM SUBMISSION  ##################
######################################################################################################################

@app.command()
def decon_dir(dir_loc: str, refractive_index: float, out_dir: str=None, out_file_type: str='.tif', file_type: str='.btf',
              denoise_sigma: float=None, sharpen: bool=False,
              half_precision: bool=False, psf_shape: tuple[int,int,int]=(7,7,7), iterations: int=40, frames_per_chunk: int=None,
              num_parallel: int=None
              ):
    '''3D deconvolution of all files in a directory using the richardson-lucy method [executed on SLURM]'''
    import subprocess

    from constants import ENV_PYTHON_LOC
    from constants import DECON_SCRIPT

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

    # Dynamically determine how much RAM to allocate for SLURM to support decon, assumes all files are the same size.
    file_size_to_decon = get_file_size_gb(file_list[0])
    # Arbitrary multiplier by 2
    file_size_to_decon *= 2

    from constants import SLURM_PARAMETERS_DECON as PARAMS
    # Extract required parameters into simple names
    PARTITION = PARAMS.get('PARTITION')
    CPUS = PARAMS.get('CPUS')
    JOB_LABEL = PARAMS.get('JOB_LABEL')

    if PARAMS.get('RAM_GB'):
        # Effectively decon RAM_GB in config is a max value
        # If file_size_to_decon is less than PARAMS.get('RAM_GB') use file_size_to_decon
        # If the max size is used there is a change that the decon will fail with out of memory error, need to test
        RAM_GB = file_size_to_decon if file_size_to_decon <= PARAMS.get('RAM_GB') else PARAMS.get('RAM_GB')
    else:
        RAM_GB = PARAMS.get('RAM_GB')

    GRES = PARAMS.get('GRES')
    NICE = PARAMS.get('NICE')
    TIME_LIMIT = PARAMS.get('TIME_LIMIT')
    username = get_user(dir_loc)

    jobname = f'{username}' if username else ""
    jobname = f'{jobname}{"-" if jobname else ""}{JOB_LABEL}' if JOB_LABEL else jobname

    if num_parallel is None:
        PARALLEL_JOBS = PARAMS.get('PARALLEL_JOBS', 1)
    else:
        PARALLEL_JOBS = num_parallel

    SBATCH_ARG = '#SBATCH {}\n'
    # to_run = ["sbatch", "-p gpu", "--gres=gpu:1", "-J decon", f'-o {log_dir} / %A_%a.log', f'--array=0-{num_files-1}']
    commands = "#!/bin/bash\n"
    commands += SBATCH_ARG.format(f'-p {PARTITION}') if PARTITION is not None else ""
    commands += SBATCH_ARG.format(f'-n {CPUS}') if CPUS is not None else ""
    commands += SBATCH_ARG.format(f'--mem={RAM_GB}GB') if RAM_GB is not None else ""
    commands += SBATCH_ARG.format(f'--gres={GRES}') if GRES is not None else ""
    commands += SBATCH_ARG.format(f'-J {jobname}') if jobname else ""
    commands += SBATCH_ARG.format(f'--nice={NICE}') if NICE is not None else ""
    commands += SBATCH_ARG.format(f'-t {TIME_LIMIT}') if TIME_LIMIT is not None else ""
    commands += SBATCH_ARG.format(f'-o {log_dir}/%A_%a.log')
    commands += SBATCH_ARG.format(f'--array=0-{num_files - 1}{"%" + str(PARALLEL_JOBS) if PARALLEL_JOBS > 0 else ""}')
    commands += "\n"

    commands += "commands=("
    #Build each command
    for p in file_list:
        commands += '\n\t'
        commands += '"'
        commands += f'{ENV_PYTHON_LOC} -u {DECON_SCRIPT} decon'
        commands += f' {p.as_posix()}'
        commands += f' {refractive_index}'
        commands += f' --out-location {out_dir / (p.stem + out_file_type)}'
        #commands += f'{" --queue-ims" if queue_ims else ""}'
        commands += f'{" --sharpen" if sharpen else ""}'
        commands += f'{f" --denoise-sigma {denoise_sigma}" if denoise_sigma else ""}'
        commands += f'{" --half-precision" if half_precision else " --no-half-precision"}'
        commands += f' --psf-shape {psf_shape[0]} {psf_shape[1]} {psf_shape[2]}'
        commands += f' --iterations {iterations}'
        commands += f' --frames-per-chunk {frames_per_chunk}' if frames_per_chunk is not None else ''
        commands += '"'

    commands += '\n)\n\n'
    commands += 'echo "Running command: ${commands[$SLURM_ARRAY_TASK_ID]}"\n'
    commands += 'eval "${commands[$SLURM_ARRAY_TASK_ID]}"'

    file_to_run = out_dir / 'slurm_array.sh'
    with open(file_to_run, 'w') as f:
        f.write(commands)

    output = subprocess.run(f'sbatch {file_to_run}', shell=True, capture_output=True)
    prefix_len = len(b'Submitted batch job ')
    job_number = int(output.stdout[prefix_len:-1])
    print(f'SBATCH Job #: {job_number}')

    return job_number, out_dir


def make_sbatch_params(PARAMS, array_len=None):
    '''
    PARAMS is a dictionary with sbatch options
    Generally these are stored in the constants module
    '''
    # Extract required parameters into simple names
    PARTITION = PARAMS.get('PARTITION')
    CPUS = PARAMS.get('CPUS')
    JOB_LABEL = PARAMS.get('JOB_LABEL')
    RAM_GB = PARAMS.get('RAM_GB')
    GRES = PARAMS.get('GRES')
    PARALLEL_JOBS = PARAMS.get('PARALLEL_JOBS')
    NICE = PARAMS.get('NICE')
    TIME_LIMIT = PARAMS.get('TIME_LIMIT')

    # Build sbatch wrapper components
    sbatch_options = [
        f"-p {PARTITION}" if PARTITION is not None else "",
        f"-n {CPUS}" if CPUS is not None else "",
        f"--gres={GRES}" if GRES is not None else "",
        f"--mem={RAM_GB}G" if RAM_GB is not None else "",
        f"-J {JOB_LABEL}" if JOB_LABEL is not None else "",
        f"--array=0-{array_len - 1}" + (f"%{PARALLEL_JOBS}" if PARALLEL_JOBS else "") if array_len is not None else "",
        f"--nice={NICE}" if NICE is not None else "",
        f"--time={TIME_LIMIT}" if TIME_LIMIT is not None else "",
    ]

    # Join non-empty elements with a space
    sbatch_cmd = f"{' '.join(filter(None, sbatch_options))}"
    return sbatch_cmd

######################################################################################################################
####  IMARIS CONVERTER FUNCTIONS TO HANDLE SLURM SUBMISSION  ##################
######################################################################################################################

@app.command()
def convert_ims_dir_mesospim_tiles_slurm(dir_loc: Path, file_type: str='.btf', res: tuple[float,float,float]=(1,1,1),
                                after_slurm_jobs: list[int]=None):
    '''
    Convert all files in a directory to Imaris Files [executed on SLURM]
    The function assumes this is mesospim data.
    Tiles are grouped and sorted by color using the function
    nested_list_tile_files_sorted_by_color

    names are expected to be as follows: _Tile{NUMBER}_ with a number representing laser line

    Currently, resolution must be provided manually.

    Each imaris file is created using a different slurm job
    '''
    # from constants import SLURM_PARTITION, SLURM_CPUS, SLURM_RAM_MB, SLURM_JOB_LABEL, SLURM_RAM_MB, SLURM_GRES
    from constants import SLURM_PARAMETERS_IMARIS_CONVERTER as slurm_parameters_dictionary

    # after_slurm_jobs = parse_job_numbers(after_slurm_jobs)

    username = get_user(dir_loc)

    tiles = nested_list_tile_files_sorted_by_color(dir_loc=dir_loc, file_type=file_type)
    job_numbers = []
    for ii in tiles:
        current_script, log_location, out_dir = convert_ims(ii, res=res, run_conversion=False)
        job_number = wrap_slurm(current_script,
                                slurm_parameters_dictionary, log_location,
                                after_slurm_jobs=after_slurm_jobs, username=username)
        # job_number = wrap_slurm(current_script, SLURM_PARTITION, SLURM_CPUS, SLURM_RAM_MB, SLURM_JOB_LABEL, log_location,
        #                         after_slurm_jobs=None)
        job_numbers.append(job_number)

    return job_numbers, out_dir

@app.command()
def convert_ims_dir_mesospim_tiles_slurm_array(dir_loc: Path, file_type: str='.btf', res: tuple[float,float,float]=(1,1,1),
                                after_slurm_jobs: list[int]=None):
    '''
    Convert all files in a directory to Imaris Files [executed on SLURM]
    The function assumes this is mesospim data.
    Tiles are grouped and sorted by color using the function
    nested_list_tile_files_sorted_by_color

    names are expected to be as follows: _Tile{NUMBER}_ with a number representing laser line

    Currently, resolution must be provided manually.

    All imaris files are queued as a single slurm array
    '''
    # from constants import SLURM_PARTITION, SLURM_CPUS, SLURM_RAM_MB, SLURM_JOB_LABEL, SLURM_RAM_MB, SLURM_GRES, SLURM_PARALLEL_JOBS
    from constants import SLURM_PARAMETERS_IMARIS_CONVERTER as slurm_parameters_dictionary

    # after_slurm_jobs = parse_job_numbers(after_slurm_jobs)

    username = get_user(dir_loc)

    tiles = nested_list_tile_files_sorted_by_color(dir_loc=dir_loc, file_type=file_type)
    commands = []
    for ii in tiles:
        current_script, log_location, out_dir = convert_ims(ii, res=res, run_conversion=False)
        if current_script:
            commands.append(current_script)

    # job_number = submit_array(commands, dir_loc, SLURM_PARTITION, SLURM_CPUS, SLURM_RAM_MB, SLURM_JOB_LABEL, SLURM_GRES, log_location, SLURM_PARALLEL_JOBS, after_slurm_jobs = None)
    job_number = submit_array(commands, dir_loc, slurm_parameters_dictionary,
                              log_location, after_slurm_jobs=after_slurm_jobs, username=username)

    return job_number, out_dir



######################################################################################################################
####  HELPER FUNCTIONS TO HANDLE SLURM SUBMISSION  ##################
######################################################################################################################


def wrap_slurm(cmd: str, slurm_parameters_dictionary, log_location,
               after_slurm_jobs: list[int] = None, username: str = ""):
    '''
    Given a command (cmd) string, wrap the command in a sbatch script and submit it to slurm
    If the command should not run until after another job(s) pass these job number in a list to after_slurm_jobs
    '''

    # Sbatch command wrapping each ims build
    sbatch_cmd = format_sbatch_wrap(slurm_parameters_dictionary=slurm_parameters_dictionary, log_location=log_location, username=username)
    sbatch_cmd = sbatch_cmd.format(cmd)
    sbatch_cmd = sbatch_depends(sbatch_cmd, after_slurm_jobs)
    print(sbatch_cmd)

    # sbatch_cmd = f"sbatch{depends_statement}-p {SLURM_PARTITION} -n {SLURM_CPUS} --mem={SLURM_RAM_MB}G -J {SLURM_JOB_LABEL} -o {log_location} --wrap='{cmd}'"
    output = subprocess.run(sbatch_cmd, shell=True, capture_output=True, text=True, check=True)
    job_number = get_job_number_from_slurm_out(output)
    print(f'Submitted job: {job_number}')
    return job_number

def submit_array(cmd: list[str], location_for_sbatch_script, slurm_parameters_dictionary, log_location,
               after_slurm_jobs: list[int] = None, username: str = ""):
    '''
    Given a list of commands (cmd), wrap the commands in a sbatch script and submit it as an array to slurm
    If the command should not run until after another job(s) pass these job number in a list to after_slurm_jobs
    '''
    if not cmd:
        return
    ## Wrap all commands in a bash array 'commands' and call each element as a separate job in the SLURM array
    commands = 'commands=('
    for ii in cmd:
        commands = f'{commands}\n\t"{ii.replace("\n\n",";").replace("\n",";").replace("\"","\\\"")}' # Strip new lines and make single line commands
        commands = commands.replace('then;','then').replace('else;','else') # Ensure no ; after then or else
        commands += '"'
        # commands = commands.replace('"', '\\"')
        # commands = f'{commands}\n\t{ii}'
    commands = f'{commands}\n)'
    # commands += '\n\necho "Running command: ${commands[$SLURM_ARRAY_TASK_ID]}"'
    commands += '\neval "${commands[$SLURM_ARRAY_TASK_ID]}"'

    location_for_sbatch_script = ensure_path(location_for_sbatch_script)
    name_of_sbatch_script = location_for_sbatch_script / 'ims_conv_sbatch.sh'
    with open(name_of_sbatch_script, 'w') as f:
        f.write(commands)
    os.chmod(name_of_sbatch_script, 0o770)

    # Sbatch command wrapping each ims build
    sbatch_cmd = format_sbatch_wrap(slurm_parameters_dictionary=slurm_parameters_dictionary, log_location=log_location,
                                    array_len=len(cmd), bash=True, username=username)
    sbatch_cmd = sbatch_cmd.format(name_of_sbatch_script)
    sbatch_cmd = sbatch_depends(sbatch_cmd, after_slurm_jobs)
    print(sbatch_cmd)
    output = subprocess.run(sbatch_cmd, shell=True, capture_output=True, text=True, check=True)
    job_number = get_job_number_from_slurm_out(output)
    print(f'Submitted job: {job_number}')
    return job_number

def get_job_number_from_slurm_out(output):
    return int(output.stdout[20:-1])


@app.command()
def format_sbatch_wrap(slurm_parameters_dictionary: str, log_location:Path, array_len=None, bash=False, username=""):
    '''
    This function takes slurm_parameters_dictionary and outputs a sbatch command designed to wrap a single line command
    --wrap={commands}.  No commands are included, just the prologue required to configure parameters needed for slurm
    to allocate resources

    slurm_parameters_dictionary must include:

    GENERIC_slurm_parameters_dictionary = {
        PARTITION: 'compute,gpu', #multiple partitions can be specified with comma separation part1,par2
        CPUS: 24,
        JOB_LABEL: 'ims_conv',
        RAM_GB: 64,
        GRES: None, # Specific exactly as it would be in slurm eg. "gpu:1" or None
        PARALLEL_JOBS: 8,
        }

    returns: string with formed sbatch command with {} in the wrap argument. This can be used to insert commands
    using the .format() method.

    cmd_output.format(my_script_string_or_file_location)

    If the script requires bash to run, then indicate bash=True
    '''

    PARAMS = slurm_parameters_dictionary

    log_location = ensure_path(log_location)

    log_file = log_location.is_file()
    log_dir = log_location.is_dir()
    log_parent_exists = log_location.parent.is_dir()
    array = array_len is not None

    assert (log_file or log_dir or log_parent_exists), 'Log location does not exist'

    # Format log output file/dir
    if array and log_dir:
        log_location = log_location / '%A_%a.log'
    elif array and log_file:
        log_location = log_location.parent / '%A_%a.log'
    elif array and log_parent_exists:
        log_location = log_location.parent / '%A_%a.log'
    elif not array and log_dir:
        log_location = log_location / '%A.log'
    elif not array and log_file:
        pass
    elif not array and log_parent_exists:
        log_location = log_location.parent / '%A.log'

    # Extract required parameters into simple names
    PARTITION = PARAMS.get('PARTITION')
    CPUS = PARAMS.get('CPUS')
    JOB_LABEL = PARAMS.get('JOB_LABEL')
    RAM_GB = PARAMS.get('RAM_GB')
    GRES = PARAMS.get('GRES')
    PARALLEL_JOBS = PARAMS.get('PARALLEL_JOBS')
    NICE = PARAMS.get('NICE')
    TIME_LIMIT = PARAMS.get('TIME_LIMIT')

    jobname = f'{username}' if username else ""
    jobname = f'{jobname}{"-" if jobname else ""}{JOB_LABEL}' if JOB_LABEL else jobname

    # Build sbatch wrapper components
    sbatch_options = [
        f"-p {PARTITION}" if PARTITION is not None else "",
        f"-n {CPUS}" if CPUS is not None else "",
        f"--gres={GRES}" if GRES is not None else "",
        f"--mem={RAM_GB}G" if RAM_GB is not None else "",
        f"-J {jobname}" if jobname else "",
        f"--array=0-{array_len-1}" + (f"%{PARALLEL_JOBS}" if PARALLEL_JOBS else "") if array_len is not None else "",
        f"--nice={NICE}" if NICE is not None else "",
        f"--time={TIME_LIMIT}" if TIME_LIMIT is not None else "",
        f"-o {log_location}",
        "--wrap='bash -c \"{}\"'" if bash else "--wrap='{}'"
    ]

    # Join non-empty elements with a space
    sbatch_cmd = f"sbatch {' '.join(filter(None, sbatch_options))}"

    # print(sbatch_cmd.format('testtesttest'))
    return sbatch_cmd

def sbatch_depends(sbatch_script, list_of_job_ids:list[int]=None):

    if list_of_job_ids is None:
        return sbatch_script

    assert all( [isinstance(x, int) for x in list_of_job_ids] ), 'Job ID dependencies must be integers'

    ## For spinning off on slurm
    depends_statement = ''
    if isinstance(list_of_job_ids, list):
        for idx, job in enumerate(list_of_job_ids):
            if idx == 0:
                depends_statement = f' --depend=afterok:{job} --kill-on-invalid-dep=yes '
            else:
                depends_statement += f':{job}'
    depends_statement += ' '

    return f'sbatch{depends_statement}{sbatch_script[7:]}'

def parse_job_numbers(job_number_strings_comma_separated=None):
    if job_number_strings_comma_separated is None:
        return

    jobs_num_list = job_number_strings_comma_separated.split(',')
    jobs_num_list = [int(x) for x in jobs_num_list]
    return jobs_num_list



if __name__ == "__main__":
    app()
