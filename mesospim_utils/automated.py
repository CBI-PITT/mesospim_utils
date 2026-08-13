import typer
from pathlib import Path
from typing import Annotated
import subprocess
import shutil

from numpy.f2py.auxfuncs import throw_error

from constants import ENV_PYTHON_LOC, LOCATION_OF_MESOSPIM_UTILS_INSTALL, ALIGNMENT_DIRECTORY, LOCATION_PYIMARISWRITER_ENV
from metadata import (
    collect_all_metadata,
    get_first_entry,
    determine_xyz_resolution,
    affine_microns_to_translation_zyx,
    get_entry_for_file_name,
    summarize_time_series,
    is_single_tile_single_channel_timeseries,
)
from slurm import (
    basicpy_dir,
    decon_dir,
    gain_correction_dir,
    wrap_slurm,
    submit_array,
    get_slurm_log_location,
    set_super_nice
)
from utils import ensure_path, common_prefix, strip_after
from imaris import convert_ims
from bigstitcher import (does_dir_contain_bigstitcher_metadata,
                         get_bigstitcher_omezarr_alignment_marco,
                         make_bigstitcher_slurm_dir_and_macro,
                         adjust_scale_in_bigstitcher_produced_ome_zarr,
                         get_ome_zarr_directory_from_xml,
                         get_reference_multiscale_tile_path,
                         get_bigstitcher_fused_output_path,
                         get_bigstitcher_tiff_series_output_path,
                         get_missing_tiff_planes_by_channel,
                         is_bigstitcher_tiff_series_complete,
                         is_tiff_generation_log_successful,
                         should_skip_bigstitcher_run)
from omezarr import OmeZarrV2Multiscale


mesospim_root_application = f'{ENV_PYTHON_LOC} -u {LOCATION_OF_MESOSPIM_UTILS_INSTALL}'

app = typer.Typer()


def shell_double_quote(value: str | Path) -> str:
    value = str(value)
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


def get_pyimariswriter_root_application() -> str:
    pyimaris_python = LOCATION_PYIMARISWRITER_ENV
    converter_script = LOCATION_OF_MESOSPIM_UTILS_INSTALL / 'omezarr_to_ims.py'

    cmd = f'PYIW_PYTHON={shell_double_quote(pyimaris_python)}; '
    cmd += 'PYIW_ENV_ROOT="$(dirname \"$(dirname \"$PYIW_PYTHON\")\")"; '
    cmd += 'PYIW_SITE_PACKAGES="$($PYIW_PYTHON -c "import sysconfig; print(sysconfig.get_paths()[\\"purelib\\"])")"; '
    cmd += 'export LD_LIBRARY_PATH="$PYIW_SITE_PACKAGES/PyImarisWriter:$PYIW_ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"; '
    cmd += f'$PYIW_PYTHON -u {shell_double_quote(converter_script)}'
    return cmd


def get_ims_channel_names_and_colors(metadata_by_channel: dict) -> tuple[list[str], list[tuple[float, float, float]]]:
    channel_names = []
    channel_colors = []

    for channel_key, channel_data in metadata_by_channel.items():
        first_tile = channel_data[0]
        channel_name = str(first_tile.get('channel_label', channel_key))

        channel_names.append(channel_name)
        channel_colors.append(tuple(first_tile.get('rgb_representation', (0.5, 0.5, 0.5))))

    return channel_names, channel_colors


def get_omezarr_output_directory_for_btf_conversion(dir_loc: Path, file_type: str = '.btf') -> Path:
    dir_loc = ensure_path(dir_loc)
    btf_file_list = list(dir_loc.glob(f'*{file_type}'))

    if len(btf_file_list) == 0:
        raise FileNotFoundError(f'No files found in {dir_loc} with file type {file_type}')

    prefix = common_prefix([f.name for f in btf_file_list])

    if prefix:
        prefix = strip_after(prefix, '_max')
    else:
        prefix = btf_file_list[0].name

    return dir_loc / 'ome_zarr' / f'{prefix}.ome.zarr'


def get_timeseries_omezarr_output_path(dir_loc: Path, metadata_by_channel: dict, file_type: str = '.btf') -> Path:
    dir_loc = ensure_path(dir_loc)
    summary = summarize_time_series(metadata_by_channel)
    base_name = summary.get('time_series_key') or get_omezarr_output_directory_for_btf_conversion(dir_loc, file_type=file_type).stem
    return dir_loc / 'ome_zarr' / f'{base_name}.ome.zarr'


def queue_direct_omezarr_to_ims(dir_loc: Path, omezarr_path: Path, metadata_by_channel: dict, after_job_number: int = None,
                                ims_resolution_level: int = 0, voxel_size_zyx: tuple[float, float, float] | None = None) -> int:
    from constants import SLURM_PARAMETERS_IMARIS_CONVERTER

    dir_loc = ensure_path(dir_loc)
    omezarr_path = ensure_path(omezarr_path)
    first_entry = get_first_entry(metadata_by_channel)
    username = first_entry.get('username', '')
    slurm_log_dir = get_slurm_log_location(dir_loc)

    if voxel_size_zyx is None:
        ome_zarr = OmeZarrV2Multiscale(omezarr_path)
        level_info = ome_zarr.get_level_zyx_info(ims_resolution_level)
        scale_zyx = level_info['scale_zyx']
        res_z, res_y, res_x = scale_zyx
    else:
        res_z, res_y, res_x = voxel_size_zyx
    channel_names, channel_colors = get_ims_channel_names_and_colors(metadata_by_channel)

    ims_out_file = omezarr_path.parent / f'{omezarr_path.name[:-9]}.ims'
    cmd = f'{get_pyimariswriter_root_application()} "{omezarr_path}" "{ims_out_file}"'
    cmd += f' --voxel-size-zyx-um {res_z} {res_y} {res_x}'
    cmd += ' --channel-names ' + ' '.join(f'"{name}"' for name in channel_names)
    cmd += ' --channel-colors ' + ' '.join(
        f'"{rgb[0]},{rgb[1]},{rgb[2]}"' for rgb in channel_colors
    )
    cmd += f' --level {ims_resolution_level}'

    return wrap_slurm(
        cmd,
        SLURM_PARAMETERS_IMARIS_CONVERTER,
        slurm_log_dir,
        after_slurm_jobs=[after_job_number] if after_job_number else None,
        username=username,
        log_suffix='omezarr_to_ims_timeseries',
    )


def convert_btf_timeseries_to_omezarr(dir_loc: Path, metadata_by_channel: dict, file_type: str = '.btf',
                                      after_slurm_jobs: list[int] = None, ims_resolution_level: int = 0,
                                      final_file_type: str = 'omezarr'):
    from constants import SLURM_PARAMETERS_OMEZARR
    import json

    dir_loc = ensure_path(dir_loc)
    summary = summarize_time_series(metadata_by_channel)
    if not is_single_tile_single_channel_timeseries(metadata_by_channel):
        raise ValueError('Time-series conversion currently supports only one tile and one channel')

    first_entry = get_first_entry(metadata_by_channel)
    username = first_entry.get('username', '')
    voxel_size = first_entry.get('resolution')
    slurm_log_dir = get_slurm_log_location(dir_loc)
    output_omezarr_path = get_timeseries_omezarr_output_path(dir_loc, metadata_by_channel, file_type=file_type)
    output_omezarr_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_path = output_omezarr_path.parent / f'{output_omezarr_path.stem}_timeseries_manifest.json'
    manifest = {
        'btf_paths': [str(entry.get('file_path')) for entry in summary['entries']],
        'timepoints': summary['timepoints'],
    }
    with manifest_path.open('w') as handle:
        json.dump(manifest, handle, indent=2)

    cmd = f'{mesospim_root_application}/omezarr.py convert-mesospim-btf-timeseries-to-omezarr'
    cmd += f' "{manifest_path}" "{output_omezarr_path}"'
    cmd += f' --voxel-size {voxel_size.z} {voxel_size.y} {voxel_size.x}'
    cmd += ' --ome-version 0.4'
    cmd += ' --generate-multiscales'

    job_number = wrap_slurm(
        cmd,
        SLURM_PARAMETERS_OMEZARR,
        slurm_log_dir,
        after_slurm_jobs=after_slurm_jobs,
        username=username,
        log_suffix='convert_btf_timeseries_to_omezarr',
    )

    if final_file_type.lower() == 'ims':
        job_number = queue_direct_omezarr_to_ims(
            dir_loc=dir_loc,
            omezarr_path=output_omezarr_path,
            metadata_by_channel=metadata_by_channel,
            after_job_number=job_number,
            ims_resolution_level=ims_resolution_level,
            voxel_size_zyx=(voxel_size.z, voxel_size.y, voxel_size.x),
        )
        print(f'Convert OME-Zarr time series to IMS File: {job_number}')

    return job_number, output_omezarr_path


def queue_bigstitcher_xml(dir_loc: Path, collection_dir: Path, after_job_number: int = None, supernice: bool = False):
    from constants import SLURM_PARAMETERS_FOR_DEPENDENCIES

    dir_loc = ensure_path(dir_loc)
    collection_dir = ensure_path(collection_dir)
    metadata_by_channel = collect_all_metadata(collection_dir)
    first_metadata_entry = get_first_entry(metadata_by_channel)
    username = first_metadata_entry.get('username', "")
    slurm_log_dir = get_slurm_log_location(dir_loc)

    xml_file_name = Path(collection_dir.as_posix() + '.xml')
    cmd = f'{mesospim_root_application}/bigstitcher.py mesospim-metadata-to-bigstitcher-xml'
    cmd += f' "{xml_file_name}"'
    cmd += f' --different-relative-zarr-path "{collection_dir.name}"'
    cmd += f' --modify-filename-in-xml .ome.zarr'

    return wrap_slurm(
        cmd,
        SLURM_PARAMETERS_FOR_DEPENDENCIES,
        slurm_log_dir,
        after_slurm_jobs=[after_job_number] if after_job_number else None,
        username=username,
        log_suffix='make_bigstitcher_xml',
    )


def queue_bigstitcher_alignment(dir_loc: Path, collection_dir: Path, final_file_type: str, ims_resolution_level: int = 0, after_job_number: int = None, supernice: bool = False):
    from constants import SLURM_PARAMETERS_FOR_BIGSTITCHER

    if final_file_type.lower() in {'ims', 'tiff'}:
        fused_file_type = 'omezarr'
    else:
        fused_file_type = final_file_type

    metadata_by_channel = collect_all_metadata(collection_dir)
    first_metadata_entry = get_first_entry(metadata_by_channel)
    username = first_metadata_entry.get('username', "")
    slurm_log_dir = get_slurm_log_location(dir_loc)

    cmd = f'{mesospim_root_application}/automated.py big-stitcher-align'
    cmd += f' {collection_dir.parent}'
    cmd += f' --fused-file-type {fused_file_type} --final-file-type {final_file_type}'
    cmd += f' --ims-resolution-level {ims_resolution_level}'
    if supernice:
        cmd += ' --supernice'

    return wrap_slurm(
        cmd,
        SLURM_PARAMETERS_FOR_BIGSTITCHER,
        slurm_log_dir,
        after_slurm_jobs=[after_job_number] if after_job_number else None,
        username=username,
        log_suffix='queue_bigstitcher',
    )

@app.command()
def automated_method_slurm(dir_loc: Path,
                           # Input options: None, .ome.zarr, .btf. If None, will auto-detect based on contents of dir_loc
                           file_type: Annotated[str,typer.Option(help="Input file type: .ome.zarr, .btf. Default is automatically detected")]=None,

                           # Final output options: omezarr, hdf5, ims, tiff
                           final_file_type: Annotated[str,typer.Option(help="Bigstitcher compatible output file format: omezarr, hdf5, ims, tiff")]='omezarr',

                           # Deconvolution Options: if decon==True, refractive_index is found in metadata or must be provided. No RI means no decon
                           decon: Annotated[bool,typer.Option(help="Deconvolution will proceed if refractive index is discovered in the metadata or provided manually")]=True,
                           objective: Annotated[str,typer.Option(help="Name of the microscope objective profile to use for deconvolution PSF parameters")]=None,
                           refractive_index: Annotated[float,typer.Option(help="Is discovered automatically in the metadata but can be provided manually")]=None,
                           iterations: Annotated[int,typer.Option(help="Deconvolution iterations")]=20,
                           frames_per_chunk: Annotated[int,typer.Option(help="How many z-planes are deconvolved at once. Best to let this be automatically determined")]=None,
                           num_parallel: Annotated[int,typer.Option(help="How many MesoSPIM tiles will be deconvolved in parallel on SLURM")]=None,
                           basicpy: Annotated[bool,typer.Option(help="Run BaSiCPy flat-field correction after deconvolution and before gain correction")]=False,
                           gain_correction: Annotated[bool,typer.Option(help="Run gain correction after deconvolution and BaSiCPy")]=False,
                           ims_resolution_level: Annotated[int,typer.Option(help="OME-Zarr multiscale resolution level to convert when --final-file-type ims or tiff")]=0,
                           supernice: Annotated[bool,typer.Option(help="Submit all downstream slurm jobs will elevated nice value")]=False
                           ):
    '''
    Automate the processing of all data in a mesospim directory using SLURM
    The only required argument is the directory location of the mesospim data.
    All other options will be determined automatically by the metadata or have defaults that should work for most use cases.

    This is expected to be run from the commandline on a client of a SLURM cluster with access to the mesospim data directory.
    The SLURM jobs will also run on the cluster and have access to the same data directory.

    Currently, supports data acquired from MesoSPIM in both omezarr and btf formats.
    BTF inputs are converted to tile OME-Zarr before any downstream processing.
    This method will perform deconvolution if RI information is found in the metadata.
    RI can be manually supplied to trigger deconvolution if it is not found in the metadata, or if the user wants to override the metadata information.
    Deconvolution can be skipped entirely by setting decon=False.

    After input normalization to tile OME-Zarr, the optional stage order is:
    deconvolution -> BaSiCPy -> gain correction.

    ** NOTE**
    This script is designed to run quickly, Everything is queued in SLURM.
    SLURM will manage all downstream dependencies.

    Default:
    deconvolution if RI information is found in metadata
    stitching via bigstitcher alignment of omezarr
    fusion of data into a single omezarr file
    '''

    if supernice:
        set_super_nice()

    from constants import SLURM_PARAMETERS_FOR_BIGSTITCHER, SLURM_PARAMETERS_FOR_DEPENDENCIES, SLURM_PARAMETERS_OMEZARR

    dir_loc = ensure_path(dir_loc)

    # Ensure that metadata json is produced which will be used by downstream processes
    metadata_by_channel = collect_all_metadata(dir_loc)
    first_metadata_entry = get_first_entry(metadata_by_channel)
    username = first_metadata_entry.get('username', "")

    if decon and not objective:
        from rl import validate_metadata_objective_parameters

        validate_metadata_objective_parameters(first_metadata_entry)

    if not refractive_index and decon:
        refractive_index = first_metadata_entry.get('refractive_index')

    # Determine file formats relevant for downstream processes based on desired output
    _, final_file_type = get_intermediate_file_type_for_bigstitcher_alignment(final_file_type)

    omezarr_xml = does_dir_contain_bigstitcher_metadata(dir_loc)  # returns path to omezarr xml if found, else None
    omezarr_path = get_ome_zarr_directory_from_xml(omezarr_xml) # returns path to omezarr_data relative to xml, else None
    dir_loc = omezarr_path if omezarr_path else dir_loc # switch to omezarr path if found

    if omezarr_path:
        file_type = '.ome.zarr'
    elif file_type is None and len(tuple(dir_loc.glob('*.btf'))) > 0:
        file_type = '.btf'
    else:
        raise FileNotFoundError(f"Supported file types [.ome.zarr, .btf] were not found: '{dir_loc}'")

    job_number = None
    slurm_log_dir = get_slurm_log_location(dir_loc)
    out_dir = dir_loc
    decon_ram_estimate_dir = out_dir

    if file_type == '.btf':
        print('Setting up script to convert BTF tiles to OME-Zarr before downstream processing')
        decon_ram_estimate_dir = dir_loc
        if is_single_tile_single_channel_timeseries(metadata_by_channel):
            print('Detected one-tile one-channel time-series BTF dataset; bypassing BigStitcher')
            job_number, out_dir = convert_btf_timeseries_to_omezarr(
                dir_loc,
                metadata_by_channel,
                file_type=file_type,
                after_slurm_jobs=[job_number] if job_number else None,
                ims_resolution_level=ims_resolution_level,
                final_file_type=final_file_type,
            )
            print(f'OME-Zarr time-series conversion process number: {job_number}')
            if final_file_type.lower() in {'omezarr', 'ims'}:
                return
            file_type = '.ome.zarr'
        else:
            out_dir = get_omezarr_output_directory_for_btf_conversion(dir_loc, file_type=file_type)
            job_number = convert_btf_tiles_to_omezarr_slurm_array(
                dir_loc,
                file_type=file_type,
                queue_alignment=False,
                final_file_type=final_file_type,
                after_slurm_jobs=[job_number] if job_number else None,
                supernice=supernice,
                make_xml=False,
                ims_resolution_level=ims_resolution_level,
            )
            print(f'OME-Zarr conversion process number: {job_number}')
            file_type = '.ome.zarr'
    else:
        decon_ram_estimate_dir = out_dir

    if refractive_index and decon:
        print('Queueing BigStitcher XML generation before DECON so decon workers can clone collection metadata')
        job_number = queue_bigstitcher_xml(dir_loc, out_dir, after_job_number=job_number, supernice=supernice)
        print(f'Queued pre-DECON BigStitcher XML build process number: {job_number}')

        print('Queueing DECON of MesoSPIM tiles on SLURM')
        job_number, out_dir = decon_dir(
            out_dir,
            refractive_index,
            objective=objective,
            file_type=file_type,
            out_file_type='.ome.zarr',
            iterations=iterations,
            frames_per_chunk=frames_per_chunk,
            num_parallel=num_parallel,
            after_slurm_jobs=[job_number] if job_number else None,
            ram_estimate_dir=decon_ram_estimate_dir,
        )
        print((job_number, out_dir))
        file_type = '.ome.zarr'

    if basicpy:
        print('Queueing BaSiCPy preprocessing on SLURM')
        job_number, out_dir = basicpy_dir(out_dir, after_slurm_jobs=[job_number] if job_number else None)
        print((job_number, out_dir))
        file_type = '.ome.zarr'

    if gain_correction:
        print('Queueing gain correction preprocessing on SLURM')
        job_number, out_dir = gain_correction_dir(out_dir, after_slurm_jobs=[job_number] if job_number else None)
        print((job_number, out_dir))
        file_type = '.ome.zarr'

    if file_type == '.ome.zarr':
        print('Setting up script to manage BigStitcher conversions after OME-Zarr preprocessing/deconvolution')
        job_number = queue_bigstitcher_xml(dir_loc, out_dir, after_job_number=job_number, supernice=supernice)
        print(f'Queued BigStitcher XML build process number: {job_number}')
        job_number = queue_bigstitcher_alignment(dir_loc, out_dir, final_file_type, ims_resolution_level=ims_resolution_level, after_job_number=job_number, supernice=supernice)
        print(f'Dependency process number: {job_number}')


@app.command()
def convert_btf_tiles_to_omezarr_slurm_array(dir_loc: Path, file_type: str='.btf', queue_alignment: bool=True, final_file_type: str='omezarr',
                                 after_slurm_jobs: list[int]=None, supernice: bool=False, make_xml: bool=True, ims_resolution_level: int=0):

    if supernice:
        set_super_nice()

    from constants import SLURM_PARAMETERS_FOR_BIGSTITCHER, SLURM_PARAMETERS_FOR_DEPENDENCIES, SLURM_PARAMETERS_OMEZARR
    import zarr

    btf_file_list = list(dir_loc.glob(f'*{file_type}'))

    if len(btf_file_list) == 0:
        raise FileNotFoundError(f'No files found in {dir_loc} with file type {file_type}')

    output_directory_for_omezarr_collection = get_omezarr_output_directory_for_btf_conversion(dir_loc, file_type=file_type)
    output_directory_for_omezarr_collection.mkdir(parents=True, exist_ok=True)

    # Create ome.zarr group to store all converted btf files in the collection
    zarr.open_group(output_directory_for_omezarr_collection, mode="a", zarr_version=2)

    print(f'Extracting metadata from {dir_loc}')
    metadata_by_channel = collect_all_metadata(dir_loc)
    first_metadata_entry = get_first_entry(metadata_by_channel)
    slurm_log_dir = get_slurm_log_location(dir_loc)

    username = first_metadata_entry.get('username', "")

    voxel_size = first_metadata_entry.get('resolution')  # z,y,x

    output_ome_zarr_list = [f'{output_directory_for_omezarr_collection / btf_file.name}.ome.zarr' for btf_file in btf_file_list]

    cmd_list = []
    for btf_file, output_dir in zip(btf_file_list, output_ome_zarr_list):
        file_metadata = get_entry_for_file_name(metadata_by_channel, btf_file.name)  # extract metadata for this file

        translation = affine_microns_to_translation_zyx(
            file_metadata.get('affine_microns')  # z,y,x
        )

        cmd = f'{mesospim_root_application}/omezarr.py convert-mesospim-btf-to-omezarr'
        cmd += f' "{btf_file}"'
        cmd += f' "{output_dir}"'
        cmd += f' --voxel-size {voxel_size.z} {voxel_size.y} {voxel_size.x}'
        cmd += f' --translation {translation.z} {translation.y} {translation.x}'
        cmd += f' --ome-version 0.4'
        cmd += f' --generate-multiscales'

        cmd_list.append(cmd)

    job_number = submit_array(cmd_list,
                     output_directory_for_omezarr_collection, SLURM_PARAMETERS_OMEZARR,
                     slurm_log_dir,
                     after_slurm_jobs=after_slurm_jobs, username=username, log_suffix=f'convert_btf_to_omezarr'
                               )

    print(f'OME-Zarr Conversion Array Job Number: {job_number}')


    if make_xml or queue_alignment:
        job_number = queue_bigstitcher_xml(dir_loc, output_directory_for_omezarr_collection, after_job_number=job_number, supernice=supernice)
        print(f'Queued BigStitcher XML build process number: {job_number}')

    if queue_alignment:
        # Determine file formats relevant for downstream processes based on desired output
        fused_file_type, final_file_type = get_intermediate_file_type_for_bigstitcher_alignment(final_file_type)
        cmd = f'{mesospim_root_application}/automated.py big-stitcher-align'
        cmd += f' {output_directory_for_omezarr_collection.parent}'
        cmd += f' --fused-file-type {fused_file_type} --final-file-type {final_file_type}'
        cmd += f' --ims-resolution-level {ims_resolution_level}'
        if supernice:
            cmd += f' --supernice'

        job_number = wrap_slurm(cmd, SLURM_PARAMETERS_FOR_BIGSTITCHER, slurm_log_dir,
                                after_slurm_jobs=[job_number] if job_number else None, username=username, log_suffix=f'queue_bigstitcher')

        print(f'Queued BigStitcher alignment process number: {job_number}')

    return job_number






def get_intermediate_file_type_for_bigstitcher_alignment(final_file_type: str):
    '''
    Given the desired final file type (omezarr, hdf5, ims, tiff) return the fused_file_type and final_file_type.
     fused_file_type is used for bigstitcher fusion.
     final_file_type is the final output format after bigstitcher alignment and fusion.
     '''

    if final_file_type.lower() == 'omezarr':
        return 'omezarr', 'omezarr'
    elif final_file_type.lower() == 'hdf5':
        return 'hdf5', 'hdf5'
    elif final_file_type.lower() == 'ims':
        return 'omezarr', 'ims'
    elif final_file_type.lower() == 'tiff':
        return 'omezarr', 'tiff'

    raise ValueError(f'Unsupported final file type for BigStitcher alignment: {final_file_type}')



@app.command()
def big_stitcher_align(dir_loc: Path, fused_file_type: str='omezarr', final_file_type: str='omezarr', ims_resolution_level: int=0, supernice: bool=False):

    if supernice:
        set_super_nice()

    print('Setting up script to run omezarr alignment')
    from constants import SLURM_PARAMETERS_FOR_BIGSTITCHER
    from constants import SLURM_PARAMETERS_FOR_DEPENDENCIES
    from string_templates import BIGSTITCHER_ALIGN_TEMPLATE

    print(f'Extracting metadata from {dir_loc}')
    metadata_by_channel = collect_all_metadata(dir_loc)
    first_metadata_entry = get_first_entry(metadata_by_channel)
    slurm_log_dir = get_slurm_log_location(dir_loc)

    username = first_metadata_entry.get('username', "")
    source_omezarr_xml = does_dir_contain_bigstitcher_metadata(dir_loc)
    source_omezarr_dir = get_ome_zarr_directory_from_xml(source_omezarr_xml) if source_omezarr_xml else None
    reference_tile_omezarr = None
    if fused_file_type.lower() == 'omezarr' and final_file_type.lower() in {'ims', 'tiff'} and source_omezarr_dir is None:
        raise FileNotFoundError(f'Could not locate source OME-Zarr directory from BigStitcher metadata in {dir_loc}')
    if source_omezarr_dir is not None:
        reference_tile_omezarr = get_reference_multiscale_tile_path(source_omezarr_dir)

    skip_bigstitcher, skip_reason = should_skip_bigstitcher_run(
        dir_loc,
        fused_file_type,
        final_file_type,
        slurm_log_dir,
        metadata_by_channel,
    )

    fused_out_dir_or_file = get_bigstitcher_fused_output_path(dir_loc, format=fused_file_type)
    tiff_series_out_dir = None
    if fused_file_type.lower() == 'omezarr' and final_file_type.lower() in {'ims', 'tiff'}:
        tiff_series_out_dir = get_bigstitcher_tiff_series_output_path(fused_out_dir_or_file)

    if skip_bigstitcher:
        print(f'Skipping BigStitcher rerun: {skip_reason}')
        job_number = None
    else:
        bigstitcher_dir, fused_out_dir_or_file, macro_file = make_bigstitcher_slurm_dir_and_macro(dir_loc, format=fused_file_type)
        cmd = BIGSTITCHER_ALIGN_TEMPLATE.format(macro_file)
        cmd += f' && [ -e "{fused_out_dir_or_file}" ]'
        cmd += f' && printf "BIGSTITCHER_SUCCESS: {Path(fused_out_dir_or_file).name}\\n"'

        job_number = None
        job_number = wrap_slurm(cmd, SLURM_PARAMETERS_FOR_BIGSTITCHER, slurm_log_dir,
                                after_slurm_jobs=[job_number] if job_number else None, username=username, log_suffix=f'align_fuse_bigstitcher')
        print(f'BigStitcher process number: {job_number}')

    if final_file_type.lower() == 'omezarr' and not final_file_type.lower() == 'ims':
        if skip_bigstitcher:
            return
        cmd = f'{mesospim_root_application}/bigstitcher.py adjust-scale-in-bigstitcher-produced-ome-zarr'
        cmd += f' "{dir_loc}" "{fused_out_dir_or_file}"'
        job_number = wrap_slurm(cmd, SLURM_PARAMETERS_FOR_DEPENDENCIES, slurm_log_dir,
                                after_slurm_jobs=[job_number] if job_number else None, username=username, log_suffix=f'fix_bigstitcher_omezarr_metadata')
        print(f'BigStitcher Fix OME-Zarr Metadata Scale: {job_number}')

    elif fused_file_type.lower() == 'hdf5' and final_file_type.lower() == 'ims':
        from constants import SLURM_PARAMETERS_IMARIS_CONVERTER
        metadata = collect_all_metadata(dir_loc)
        first_entry = get_first_entry(metadata)
        res = determine_xyz_resolution(first_entry) #zyx

        current_script, log_location, out_dir = convert_ims(fused_out_dir_or_file, res=res, run_conversion=False)
        job_number = wrap_slurm(current_script,
                                SLURM_PARAMETERS_IMARIS_CONVERTER, slurm_log_dir,
                                after_slurm_jobs=[job_number] if job_number else None, username=username, log_suffix=f'queue_hdf5_to_ims')
        print(f'BigStitcher HDF5 Convert to IMS: {job_number}')


    elif fused_file_type.lower() == 'omezarr' and final_file_type.lower() in {'ims', 'tiff'}:
        fused_out_dir_or_file = ensure_path(fused_out_dir_or_file)
        if final_file_type.lower() == 'tiff':
            tiff_series_dir_name = str(fused_out_dir_or_file.name[:-9]) + '_tiffstack'
            tiff_series_out_dir = fused_out_dir_or_file.parent / tiff_series_dir_name
            missing_tiff_planes_by_channel = get_missing_tiff_planes_by_channel(
                slurm_log_dir,
                tiff_series_out_dir,
                reference_tile_omezarr,
                num_channels=len(metadata_by_channel),
                resolution_level=ims_resolution_level,
            )

            if not missing_tiff_planes_by_channel:
                extraction_job_numbers = []
                print(f'Skipping OME-Zarr to TIFF extraction: complete TIFF stack with per-plane success markers already exists at {tiff_series_out_dir}')
            else:
                extraction_job_numbers = queue_omezarr_tiff_extraction_arrays(
                    fused_omezarr_directory=fused_out_dir_or_file,
                    reference_tile_omezarr_directory=reference_tile_omezarr,
                    num_channels=len(metadata_by_channel),
                    output_directory=tiff_series_out_dir,
                    resolution_level=ims_resolution_level,
                    slurm_log_dir=slurm_log_dir,
                    username=username,
                    after_slurm_jobs=[job_number] if job_number else None,
                    missing_z_by_channel=missing_tiff_planes_by_channel,
                    prefix='composite',
                )
                if extraction_job_numbers:
                    job_number = extraction_job_numbers[-1]
                total_missing_tiff_planes = sum(len(z_values) for z_values in missing_tiff_planes_by_channel.values())
                print(f'Convert OME-Zarr to Tiff Stack: {extraction_job_numbers} ({total_missing_tiff_planes} missing planes across {len(missing_tiff_planes_by_channel)} channels)')

            return

        from constants import SLURM_PARAMETERS_IMARIS_CONVERTER

        ome_zarr = OmeZarrV2Multiscale(reference_tile_omezarr)
        level_info = ome_zarr.get_level_zyx_info(ims_resolution_level)
        scale_zyx = level_info['scale_zyx']
        res_z, res_y, res_x = scale_zyx
        channel_names, channel_colors = get_ims_channel_names_and_colors(metadata_by_channel)

        ims_out_file = fused_out_dir_or_file.parent / f'{fused_out_dir_or_file.name[:-9]}.ims'
        cmd = f'{get_pyimariswriter_root_application()} "{fused_out_dir_or_file}" "{ims_out_file}"'
        cmd += f' --voxel-size-zyx-um {res_z} {res_y} {res_x}'
        cmd += ' --channel-names ' + ' '.join(f'"{name}"' for name in channel_names)
        cmd += ' --channel-colors ' + ' '.join(
            f'"{rgb[0]},{rgb[1]},{rgb[2]}"' for rgb in channel_colors
        )
        cmd += f' --level {ims_resolution_level}'

        job_number = wrap_slurm(
            cmd,
            SLURM_PARAMETERS_IMARIS_CONVERTER,
            slurm_log_dir,
            after_slurm_jobs=[job_number] if job_number else None,
            username=username,
            log_suffix='omezarr_to_ims',
        )
        print(f'Convert OME-Zarr to IMS File: {job_number}')


def queue_omezarr_tiff_extraction_arrays(
    fused_omezarr_directory: Path,
    reference_tile_omezarr_directory: Path,
    num_channels: int,
    output_directory: Path,
    resolution_level: int,
    slurm_log_dir: Path,
    username: str = '',
    after_slurm_jobs: list[int] = None,
    missing_z_by_channel: dict[int, list[int]] = None,
    prefix: str = 'composite',
    batch_size: int = 10,
):
    from constants import SLURM_PARAMETERS_IMARIS_CONVERTER

    fused_omezarr_directory = ensure_path(fused_omezarr_directory)
    reference_tile_omezarr_directory = ensure_path(reference_tile_omezarr_directory)
    output_directory = ensure_path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    ome_zarr = OmeZarrV2Multiscale(reference_tile_omezarr_directory)
    level_info = ome_zarr.get_level_zyx_info(resolution_level)
    z_layers = level_info['z_layers']
    job_numbers = []
    dependency_job_ids = after_slurm_jobs

    for channel in range(num_channels):
        commands = []
        z_values = missing_z_by_channel.get(channel, []) if missing_z_by_channel is not None else range(z_layers)
        batch_starts = sorted({(int(z) // batch_size) * batch_size for z in z_values})

        for start_z in batch_starts:
            cmd = f'{mesospim_root_application}/omezarr.py extract-tiff-plane-batch'
            cmd += f' "{fused_omezarr_directory}" "{output_directory}"'
            cmd += f' --resolution-level {resolution_level}'
            cmd += f' --channel {channel}'
            cmd += f' --start-z {start_z}'
            cmd += f' --batch-size {batch_size}'
            cmd += f' --prefix {prefix}'
            commands.append(cmd)

        if not commands:
            continue

        job_number = submit_array(
            commands,
            output_directory,
            SLURM_PARAMETERS_IMARIS_CONVERTER,
            slurm_log_dir,
            after_slurm_jobs=dependency_job_ids,
            username=username,
            log_suffix=f'omezarr_to_tiff_stack_c{channel:02d}',
        )
        job_numbers.append(job_number)

    return job_numbers



@app.command()
def ims_conv_then_align(dir_loc: Path, metadata_dir: Path, file_type: str='.tif', ims_convert: bool=True, supernice: bool=False):

    if supernice:
        set_super_nice()

    # Collect all metadata from MesoSPIM acquisition directory and save to mesospim_metadata.json in the ims file dir
    print(f'Extracting metadata from {metadata_dir}')
    metadata_by_channel = collect_all_metadata(metadata_dir)
    first_metadata_entry = get_first_entry(metadata_by_channel)
    slurm_log_dir = get_slurm_log_location(dir_loc)
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
        job_number = wrap_slurm(cmd, SLURM_PARAMETERS_FOR_MESOSPIM_ALIGN, slurm_log_dir,
                                after_slurm_jobs=[job_number], username=username)
        print(f'Dependency process number: {job_number}')
    else:
        job_number = wrap_slurm(cmd, SLURM_PARAMETERS_FOR_MESOSPIM_ALIGN, slurm_log_dir,
                                after_slurm_jobs=None, username=username)

    # Dependency process that kicks off windows resampling
    print('Setting up script to manage resampling after alignment')
    from constants import SLURM_PARAMETERS_FOR_DEPENDENCIES
    cmd = ''
    cmd += f'{mesospim_root_application}/resample_ims.py write-auto-resample-message'
    cmd += f' {metadata_dir} {out_dir} {job_number}'
    job_number = wrap_slurm(cmd, SLURM_PARAMETERS_FOR_DEPENDENCIES, slurm_log_dir, after_slurm_jobs=[job_number], username=username)
    print(f'Dependency process number: {job_number}')



if __name__ == "__main__":
    app()
