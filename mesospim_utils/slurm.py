import typer
from pathlib import Path
import subprocess
import os
import math


from imaris import convert_ims, nested_list_tile_files_sorted_by_color
from utils import ensure_path, get_user, get_file_size_gb
from metadata import find_metadata_dir
from constants import DEV_SLURM_TOP_PRIORITY

app = typer.Typer()

######################################################################################################################
####  DECON FUNCTIONS TO HANDLE SLURM SUBMISSION  ##################
######################################################################################################################

@app.command()
def decon_dir(dir_loc: str, refractive_index: float, emission_wavelength: int=None, out_dir: str=None,
              out_file_type: str='.tif', file_type: str='.btf',
              denoise_sigma: float=None, sharpen: bool=False,
              half_precision: bool=False, psf_shape: tuple[int,int,int]=(7,7,7), iterations: int=40, frames_per_chunk: int=None,
              num_parallel: int=None
              ):
    '''3D deconvolution of all files in a directory using the richardson-lucy method [executed on SLURM]'''
    import subprocess

    from constants import ENV_PYTHON_LOC
    from constants import DECON_SCRIPT, MAX_VRAM

    path = ensure_path(dir_loc)
    # scripts_and_psf_dir = out_dir
    if out_dir:
        pass
        # scripts_and_psf_dir = out_dir
    else:
        if str(path).endswith('.ome.zarr'):
            # scripts_and_psf_dir = path.parent / 'decon'
            out_dir = path.parent / 'decon' / path.name
        else:
            out_dir = path / 'decon'
            # scripts_and_psf_dir = out_dir

    out_dir = ensure_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # scripts_and_psf_dir = ensure_path(scripts_and_psf_dir)
    # scripts_and_psf_dir.mkdir(parents=True, exist_ok=True)

    # log_dir = out_dir / 'logs'
    log_dir = get_slurm_log_location(path)
    file_list = list(path.glob('*' + file_type))

    # Dynamically determine how much RAM to allocate for SLURM to support decon, assumes all files are the same size.
    if Path(file_list[0]).is_file():
        file_size_to_decon = get_file_size_gb(file_list[0])
    elif str(file_list[0]).endswith('.ome.zarr'):
        from ome_zarr_multiscale_writer.zarr_reader import OmeZarrArray
        file_size_to_decon = OmeZarrArray(file_list[0]).nbytes / (1024 ** 3)  # ensure it's a valid zarr

    # Arbitrary multiplier by 2
    file_size_to_decon *= 2
    file_size_to_decon += 16  # Add 16GB overhead

    from constants import SLURM_PARAMETERS_DECON as PARAMS
    # Extract required parameters into simple names
    PARTITION = PARAMS.get('PARTITION')
    CPUS = PARAMS.get('CPUS')
    JOB_LABEL = PARAMS.get('JOB_LABEL')

    RAM_GB = file_size_to_decon
    if MAX_VRAM:
        MAX_VRAM_GB = MAX_VRAM // 1024
        RAM_GB = MAX_VRAM_GB if MAX_VRAM_GB < RAM_GB else RAM_GB
    if PARAMS.get('RAM_GB'):
        # Effectively decon RAM_GB in config is a max value
        # If file_size_to_decon is less than PARAMS.get('RAM_GB') use file_size_to_decon
        # If the max size is used there is a change that the decon will fail with out of memory error, need to test
        RAM_GB = RAM_GB if RAM_GB <= PARAMS.get('RAM_GB') else PARAMS.get('RAM_GB')
    RAM_GB = math.ceil(RAM_GB)

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

    files_not_done = [x for x in file_list if not (out_dir / (x.stem + out_file_type)).exists()]

    if files_not_done:
        SBATCH_ARG = '#SBATCH {}\n'
        commands = "#!/bin/bash\n"
        commands += SBATCH_ARG.format(f'-p {PARTITION}') if PARTITION is not None else ""
        commands += SBATCH_ARG.format(f'-n {CPUS}') if CPUS is not None else ""
        commands += SBATCH_ARG.format(f'--mem={RAM_GB}GB') if RAM_GB is not None else ""
        commands += SBATCH_ARG.format(f'--gres={GRES}') if GRES is not None else ""
        commands += SBATCH_ARG.format(f'-J {jobname}') if jobname else ""
        commands += SBATCH_ARG.format(f'--nice={NICE}') if NICE is not None else ""
        commands += SBATCH_ARG.format(f'-t {TIME_LIMIT}') if TIME_LIMIT is not None else ""
        commands += SBATCH_ARG.format(f'-o {log_dir}/%A_%a_decon.log')
        commands += SBATCH_ARG.format(f'--array=0-{len(files_not_done) - 1}{"%" + str(PARALLEL_JOBS) if PARALLEL_JOBS > 0 else ""}')
        commands += "\n"

        commands += "commands=("
        #Build each command
        for p in files_not_done:

            if str(p).endswith('.ome.zarr'):
                out_file_name = str(p.name).removesuffix('.ome.zarr') + out_file_type
            else:
                out_file_name = p.stem + out_file_type

            commands += '\n\t'
            commands += '"'
            commands += f'{ENV_PYTHON_LOC} -u {DECON_SCRIPT} decon'
            commands += f' {p.as_posix()}'
            commands += f' --refractive-index {refractive_index}'
            commands += f'{f" --emission-wavelength {emission_wavelength}" if emission_wavelength else ""}'
            commands += f' --out-location {out_dir / out_file_name}'
            #commands += f'{" --queue-ims" if queue_ims else ""}'
            commands += f'{" --sharpen" if sharpen else ""}'
            commands += f'{f" --denoise-sigma {denoise_sigma}" if denoise_sigma else ""}'
            commands += f'{" --half-precision" if half_precision else " --no-half-precision"}'
            commands += f' --psf-shape {psf_shape[0]} {psf_shape[1]} {psf_shape[2]}'
            commands += f' --iterations {iterations}'
            commands += f' --frames-per-chunk {frames_per_chunk}' if frames_per_chunk is not None else ''
            commands += '"'

        commands += '\n)\n\n'
        commands += 'cmd="${commands[$SLURM_ARRAY_TASK_ID]}"\n'
        commands += r'cmd="${cmd//(/\\(}"' + "\n"
        commands += r'cmd="${cmd//)/\\)}"' + "\n"
        commands += 'echo "Running command: $cmd"\n'
        commands += 'bash -lc "$cmd"'

        file_to_run = out_dir / 'slurm_array.sh'
        with open(file_to_run, 'w') as f:
            f.write(commands)

        output = subprocess.run(f'sbatch {file_to_run}', shell=True, capture_output=True)
        prefix_len = len(b'Submitted batch job ')
        job_number = int(output.stdout[prefix_len:-1])
        print(f'SBATCH Job #: {job_number}')
        if DEV_SLURM_TOP_PRIORITY:
            print(f'Setting job {job_number} to top priority')
            set_top_priority(job_number)
    else:
        job_number = None

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

def get_slurm_log_location(dir_loc: Path):
    '''
    Given a directory location, set the log directory to be at the same level as the mesospim metadata directory:
    {location_of_mestadata}/logs/

    If mesospim metadata directory is not found, set log directory to:
    {dir_loc}/logs/

    Create the directory if it does not exist
    '''
    metadata_dir = find_metadata_dir(dir_loc)
    if metadata_dir:
        log_dir = metadata_dir / 'logs_mesospim_utils'
    else:
        log_dir = dir_loc / 'logs_mesospim_utils'
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

def set_top_priority(job_number: int):
    '''
    For DEV purposes, set a job to top priority above all other jobs in the queue owned by the user.
    '''
    cmd = f'scontrol top {job_number}'
    try:
        output = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        print(output.stdout)
    except:
        print(f'Failed to set job {job_number} to top priority')


def wrap_slurm(cmd: str, slurm_parameters_dictionary, log_location,
               after_slurm_jobs: list[int] = None, username: str = "", log_prefix: str = "", log_suffix: str = ""):
    '''
    Given a command (cmd) string, wrap the command in a sbatch script and submit it to slurm
    If the command should not run until after another job(s) pass these job number in a list to after_slurm_jobs
    '''

    # Sbatch command wrapping each ims build
    sbatch_cmd = format_sbatch_wrap(slurm_parameters_dictionary=slurm_parameters_dictionary, log_location=log_location,
                                    username=username, log_prefix = log_prefix, log_suffix = log_suffix)
    sbatch_cmd = sbatch_cmd.format(cmd)
    sbatch_cmd = sbatch_depends(sbatch_cmd, after_slurm_jobs)
    print(sbatch_cmd)

    # sbatch_cmd = f"sbatch{depends_statement}-p {SLURM_PARTITION} -n {SLURM_CPUS} --mem={SLURM_RAM_MB}G -J {SLURM_JOB_LABEL} -o {log_location} --wrap='{cmd}'"
    output = subprocess.run(sbatch_cmd, shell=True, capture_output=True, text=True, check=True)
    job_number = get_job_number_from_slurm_out(output)
    print(f'Submitted job: {job_number}')
    if DEV_SLURM_TOP_PRIORITY:
        print(f'Setting job {job_number} to top priority')
        set_top_priority(job_number)
    return job_number

def submit_array(cmd: list[str], location_for_sbatch_script, slurm_parameters_dictionary, log_location,
               after_slurm_jobs: list[int] = None, username: str = "", log_prefix: str = "", log_suffix: str = ""):
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
    commands = f'{commands}\n\n)\n'
    # commands += '\n\necho "Running command: ${commands[$SLURM_ARRAY_TASK_ID]}"'
    commands += 'cmd="${commands[$SLURM_ARRAY_TASK_ID]}"\n'
    # commands += r'cmd="${cmd//(/\\(}"' + "\n"
    # commands += r'cmd="${cmd//)/\\)}"' + "\n"
    commands += 'echo "Running command: $cmd"\n'
    commands += 'eval "$cmd"'
    # commands += 'bash -lc "$cmd"'

    location_for_sbatch_script = ensure_path(location_for_sbatch_script)
    name_of_sbatch_script = location_for_sbatch_script / 'sbatch.sh'
    with open(name_of_sbatch_script, 'w') as f:
        f.write(commands)
    os.chmod(name_of_sbatch_script, 0o770)

    # Sbatch command wrapping each cmd build
    sbatch_cmd = format_sbatch_wrap(slurm_parameters_dictionary=slurm_parameters_dictionary, log_location=log_location,
                                    array_len=len(cmd), bash=True, username=username, log_prefix = log_prefix, log_suffix = log_suffix)
    sbatch_cmd = sbatch_cmd.format(name_of_sbatch_script)
    sbatch_cmd = sbatch_depends(sbatch_cmd, after_slurm_jobs)
    print(sbatch_cmd)
    output = subprocess.run(sbatch_cmd, shell=True, capture_output=True, text=True, check=True)
    job_number = get_job_number_from_slurm_out(output)
    print(f'Submitted job: {job_number}')
    if DEV_SLURM_TOP_PRIORITY:
        print(f'Setting job {job_number} to top priority')
        set_top_priority(job_number)
    return job_number

def get_job_number_from_slurm_out(output):
    return int(output.stdout[20:-1])


@app.command()
def format_sbatch_wrap(slurm_parameters_dictionary: str, log_location:Path, array_len=None, bash=False, username="",
                       log_prefix: str = "", log_suffix: str = ""
                       ):
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
        log_location = log_location / f'{log_prefix + '_' if log_prefix else ""}%A_%a{'_' + log_suffix if log_suffix else ""}.log'
    elif array and log_file:
        log_location = log_location.parent / f'{log_prefix + '_' if log_prefix else ""}%A_%a{'_' + log_suffix if log_suffix else ""}.log'
    elif array and log_parent_exists:
        log_location = log_location.parent / f'{log_prefix + '_' if log_prefix else ""}%A_%a{'_' + log_suffix if log_suffix else ""}.log'
    elif not array and log_dir:
        log_location = log_location / f'{log_prefix + '_' if log_prefix else ""}%A{'_' + log_suffix if log_suffix else ""}.log'
    elif not array and log_file:
        pass
    elif not array and log_parent_exists:
        log_location = log_location.parent / f'{log_prefix + '_' if log_prefix else ""}%A{'_' + log_suffix if log_suffix else ""}.log'

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
        f'-o "{log_location}"',
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
