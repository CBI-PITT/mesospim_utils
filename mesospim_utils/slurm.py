import typer
from pathlib import Path
import subprocess
import os


from imaris3 import convert_ims, nested_list_tile_files_sorted_by_color
from utils import path_to_wine_mappings, ensure_path

app = typer.Typer()


@app.command()
def convert_ims_dir_mesospim_tiles_slurm(dir_loc: Path, file_type: str='.btf', res: tuple[float,float,float]=(1,1,1),
                                after_slurm_jobs: str=None):
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

    after_slurm_jobs = parse_job_numbers(after_slurm_jobs)

    tiles = nested_list_tile_files_sorted_by_color(dir_loc=dir_loc, file_type=file_type)
    job_numbers = []
    for ii in tiles:
        current_script, log_location = convert_ims(ii, res=res, run_conversion=False)
        job_number = wrap_slurm(current_script,
                                slurm_parameters_dictionary, log_location,
                                after_slurm_jobs=after_slurm_jobs)
        # job_number = wrap_slurm(current_script, SLURM_PARTITION, SLURM_CPUS, SLURM_RAM_MB, SLURM_JOB_LABEL, log_location,
        #                         after_slurm_jobs=None)
        job_numbers.append(job_number)

    return job_numbers

@app.command()
def convert_ims_dir_mesospim_tiles_slurm_array(dir_loc: Path, file_type: str='.btf', res: tuple[float,float,float]=(1,1,1),
                                after_slurm_jobs: str=None):
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

    after_slurm_jobs = parse_job_numbers(after_slurm_jobs)

    tiles = nested_list_tile_files_sorted_by_color(dir_loc=dir_loc, file_type=file_type)
    commands = []
    for ii in tiles:
        current_script, log_location = convert_ims(ii, res=res, run_conversion=False)
        commands.append(current_script)

    # job_number = submit_array(commands, dir_loc, SLURM_PARTITION, SLURM_CPUS, SLURM_RAM_MB, SLURM_JOB_LABEL, SLURM_GRES, log_location, SLURM_PARALLEL_JOBS, after_slurm_jobs = None)
    job_number = submit_array(commands, dir_loc, slurm_parameters_dictionary,
                              log_location, after_slurm_jobs=after_slurm_jobs)

    return job_number



######################################################################################################################
####  HELPER FUNCTIONS TO HANDLE SLURM SUBMISSION  ##################
######################################################################################################################


def wrap_slurm(cmd: str, slurm_parameters_dictionary, log_location,
               after_slurm_jobs: list[int] = None):
    '''
    Given a command (cmd) string, wrap the command in a sbatch script and submit it to slurm
    If the command should not run until after another job(s) pass these job number in a list to after_slurm_jobs
    '''

    # Sbatch command wrapping each ims build
    sbatch_cmd = format_sbatch_wrap(slurm_parameters_dictionary=slurm_parameters_dictionary, log_location=log_location)
    sbatch_cmd = sbatch_cmd.format(cmd)
    sbatch_cmd = sbatch_depends(sbatch_cmd, after_slurm_jobs)
    print(sbatch_cmd)

    # sbatch_cmd = f"sbatch{depends_statement}-p {SLURM_PARTITION} -n {SLURM_CPUS} --mem={SLURM_RAM_MB}G -J {SLURM_JOB_LABEL} -o {log_location} --wrap='{cmd}'"
    output = subprocess.run(sbatch_cmd, shell=True, capture_output=True, text=True, check=True)
    job_number = get_job_number_from_slurm_out(output)
    print(f'Submitted job: {job_number}')
    return job_number

def submit_array(cmd: list[str], location_for_sbatch_script, slurm_parameters_dictionary, log_location,
               after_slurm_jobs: list[int] = None):
    '''
    Given a list of commands (cmd), wrap the commands in a sbatch script and submit it as an array to slurm
    If the command should not run until after another job(s) pass these job number in a list to after_slurm_jobs
    '''

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
                                    array_len=len(cmd), bash=True)
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
def format_sbatch_wrap(slurm_parameters_dictionary: str, log_location:Path, array_len=None, bash=False):
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
    # PARALLEL_JOBS = None

    # Build sbatch wrapper components
    sbatch_options = [
        f"-p {PARTITION}" if PARTITION is not None else "",
        f"-n {CPUS}" if CPUS is not None else "",
        f"--gres={GRES}" if GRES is not None else "",
        f"--mem={RAM_GB}G" if RAM_GB is not None else "",
        f"-J {JOB_LABEL}" if JOB_LABEL is not None else "",
        f"--array=0-{array_len-1}" + (f"%{PARALLEL_JOBS}" if PARALLEL_JOBS else "") if array_len is not None else "",
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
                depends_statement = f' --depend=afterok:{job}'
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


#######################################################################################################################
####  OLD CODE #############
#######################################################################################################################

# def submit_array(cmd: list[str], location_for_sbatch_script, SLURM_PARTITION, SLURM_CPUS, SLURM_RAM_MB, SLURM_JOB_LABEL, SLURM_GRES, log_location, parallel_jobs,
#                after_slurm_jobs: list[str] = None):
#     '''
#     Given a list of commands (cmd), wrap the commands in a sbatch script and submit it as an array to slurm
#     If the command should not run until after another job(s) pass these job number in a list to after_slurm_jobs
#     '''
#     ## For spinning off on slurm
#     depends_statement = ''
#     if after_slurm_jobs and isinstance(after_slurm_jobs, list):
#         for idx, job in enumerate(after_slurm_jobs):
#             if idx == 0:
#                 depends_statement = f' --depend=afterok:{job}'
#             else:
#                 depends_statement += f':{job}'
#     depends_statement += ' '
#
#     commands = 'commands=('
#     for ii in cmd:
#         commands = f'{commands}\n\t"{ii.replace("\n\n",";").replace("\n",";").replace("\"","\\\"")}'
#         commands = commands.replace('then;','then').replace('else;','else')
#         commands += '"'
#         # commands = commands.replace('"', '\\"')
#         # commands = f'{commands}\n\t{ii}'
#     commands = f'{commands}\n)'
#     commands += '\n\necho "Running command: ${commands[$SLURM_ARRAY_TASK_ID]}"'
#     commands += '\neval "${commands[$SLURM_ARRAY_TASK_ID]}"'
#
#     if not isinstance(location_for_sbatch_script, Path):
#         location_for_sbatch_script = Path(location_for_sbatch_script)
#     name_of_sbatch_script = location_for_sbatch_script / 'ims_conv_sbatch.sh'
#     with open(name_of_sbatch_script, 'w') as f:
#         f.write(commands)
#     os.chmod(name_of_sbatch_script, 0o770)
#
#     # Sbatch command wrapping each ims build
#     sbatch_cmd = f"sbatch{depends_statement}-p {SLURM_PARTITION} -n {SLURM_CPUS} {f'--gres={SLURM_GRES}' if SLURM_GRES is not None else ''} \
#     {f'--mem={SLURM_RAM_MB}G' if SLURM_RAM_MB is not None else ''} -J {SLURM_JOB_LABEL} -o {log_location.parent/'%A_%a.log'} --array=0-{len(cmd)}%{parallel_jobs}\
#     --wrap='bash -c {name_of_sbatch_script}'"
#     output = subprocess.run(sbatch_cmd, shell=True, capture_output=True, text=True, check=True)
#     job_number = get_job_number_from_slurm_out(output)
#     print(f'Submitted job: {job_number}')
#     return job_number

# def wrap_slurm(cmd: str, SLURM_PARTITION, SLURM_CPUS, SLURM_RAM_MB, SLURM_JOB_LABEL, log_location,
#                after_slurm_jobs: list[str] = None):
#     '''
#     Given a command (cmd) string, wrap the command in a sbatch script and submit it to slurm
#     If the command should not run until after another job(s) pass these job number in a list to after_slurm_jobs
#     '''
#     ## For spinning off on slurm
#     depends_statement = ''
#     if after_slurm_jobs and isinstance(after_slurm_jobs, list):
#         for idx, job in enumerate(after_slurm_jobs):
#             if idx == 0:
#                 depends_statement = f' --depend=afterok:{job}'
#             else:
#                 depends_statement += f':{job}'
#     depends_statement += ' '
#
#     # Sbatch command wrapping each ims build
#     sbatch_cmd = f"sbatch{depends_statement}-p {SLURM_PARTITION} -n {SLURM_CPUS} --mem={SLURM_RAM_MB}G -J {SLURM_JOB_LABEL} -o {log_location} --wrap='{cmd}'"
#     print(sbatch_cmd)
#     output = subprocess.run(sbatch_cmd, shell=True, capture_output=True, text=True, check=True)
#     job_number = get_job_number_from_slurm_out(output)
#     print(f'Submitted job: {job_number}')
#     return job_number

if __name__ == "__main__":
    app()