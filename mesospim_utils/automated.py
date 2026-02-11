import typer
from pathlib import Path
from typing import Annotated
import subprocess
import shutil

from numpy.f2py.auxfuncs import throw_error

from constants import ENV_PYTHON_LOC, LOCATION_OF_MESOSPIM_UTILS_INSTALL, ALIGNMENT_DIRECTORY
from metadata import (
    collect_all_metadata,
    get_first_entry,
    determine_xyz_resolution,
    affine_microns_to_translation_zyx,
    get_entry_for_file_name
)
from slurm import convert_ims_dir_mesospim_tiles_slurm_array, decon_dir, wrap_slurm, sbatch_depends, format_sbatch_wrap, submit_array
from utils import ensure_path, common_prefix, strip_after
from imaris import convert_ims
from bigstitcher import (does_dir_contain_bigstitcher_metadata,
                         get_ome_zarr_directory_from_xml,
                         get_bigstitcher_omezarr_alignment_marco,
                         make_bigstitcher_slurm_dir_and_macro,
                         adjust_scale_in_bigstitcher_produced_ome_zarr)


mesospim_root_application = f'{ENV_PYTHON_LOC} -u {LOCATION_OF_MESOSPIM_UTILS_INSTALL}'

app = typer.Typer()

@app.command()
def automated_method_slurm(dir_loc: Path,
                           # Input options: None, .ome.zarr, .btf. If None, will auto-detect based on contents of dir_loc
                           file_type: Annotated[str,typer.Option(help="Input file type: .ome.zarr, .btf. Default is automatically detected")]=None,

                           # Final output options: omezarr, hdf5, ims
                           final_file_type: Annotated[str,typer.Option(help="Bigstitcher compatible output file format: omezarr, hdf5, ims")]='omezarr',

                           # Deconvolution Options: if decon==True, refractive_index is found in metadata or must be provided. No RI means no decon
                           decon: Annotated[bool,typer.Option(help="Deconvolution will proceed if refractive index is discovered in the metadata or provided manually")]=True,
                           refractive_index: Annotated[float,typer.Option(help="Is discovered automatically in the metadata but can be provided manually")]=None,
                           iterations: Annotated[int,typer.Option(help="Deconvolution iterations")]=20,
                           frames_per_chunk: Annotated[int,typer.Option(help="How many z-planes are deconvolved at once. Best to let this be automatically determined")]=None,
                           num_parallel: Annotated[int,typer.Option(help="How many MesoSPIM tiles will be deconvolved in parallel on SLURM")]=None
                           ):
    '''
    Automate the processing of all data in a mesospim directory using SLURM
    The only required argument is the directory location of the mesospim data.
    All other options will be determined automatically by the metadata or have defaults that should work for most use cases.

    This is expected to be run from the commandline on a client of a SLURM cluster with access to the mesospim data directory.
    The SLURM jobs will also run on the cluster and have access to the same data directory.

    Currently, supports data acquired from MesoSPIM in both omezarr and btf formats.
    This method will first perform deconvolution if RI information is not found in the metadata.
    RI can be manually supplied to trigger deconvolution if it is not found in the metadata, or if the user wants to override the metadata information.
    Deconvolution can be skipped entirely by setting decon=False.

    If BTF files are discovered, decon will produce deconvolved BTF files, which will then be converted to multiscale omezarr format and stitched using bigstitcher.
    If OME-Zarr files are discovered, decon will be produced in multiscale omezarr format, and then stitched using bigstitcher.

    ** NOTE**
    This script is designed to run quickly, Everything is queued in SLURM.
    SLURM will manage all downstream dependencies.

    Default:
    deconvolution if RI information is found in metadata
    stitching via bigstitcher alignment of omezarr
    fusion of data into a single omezarr file
    '''

    dir_loc = ensure_path(dir_loc)

    # Ensure that metadata json is produced which will be used by downstream processes
    metadata_by_channel = collect_all_metadata(dir_loc)
    first_metadata_entry = get_first_entry(metadata_by_channel)
    username = first_metadata_entry.get('username', "")

    if not refractive_index and decon:
        refractive_index = first_metadata_entry.get('refractive_index')

    # Determine file formats relevant for downstream processes based on desired output
    fused_file_type, final_file_type = get_intermediate_file_type_for_bigstitcher_alignment(final_file_type)

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
    out_dir = dir_loc

    if refractive_index and decon: # for now skip decon if omezarr
        ## Decon:
        print('Queueing DECON of MesoSPIM tiles on SLURM')
        out_file_type = '.ome.zarr' if omezarr_path else '.btf'
        job_number, out_dir = decon_dir(dir_loc, refractive_index, file_type=file_type, out_file_type=out_file_type, iterations=iterations, frames_per_chunk=frames_per_chunk, num_parallel=num_parallel)
        print((job_number, out_dir))
        file_type = out_file_type

    if file_type in ('.btf','.tif'):
        # IMS Convert
        # Dependency process that kicks off IMS build following DECON
        print('Setting up script to manage IMS conversions after DECON')
        from constants import SLURM_PARAMETERS_OMEZARR
        cmd = ''
        cmd += f'{mesospim_root_application}/automated.py convert-btf-tiles-to-omezarr-slurm-array'
        cmd += f' {out_dir}'
        cmd += f' --file-type={file_type}'
        cmd += f' --queue-alignment'
        cmd += f' --final-file-type {final_file_type}'

        job_number = wrap_slurm(cmd, SLURM_PARAMETERS_OMEZARR, out_dir,
                                after_slurm_jobs=[job_number] if job_number else None, username=username)
        print(f'Dependency process number: {job_number}')

    elif file_type == '.ome.zarr':
        # Dependency process that kicks off Bigstitcher alignment following DECON
        # This process also creates a fused final montage image.
        # final_file_type can be omezarr or hdf5 or ims
        # if ims, a hdf5 file is built and then a ims conversion is initiated from the hdf5 file.
        print('Setting up script to manage Bigstitcher conversions after DECON')
        from constants import SLURM_PARAMETERS_FOR_DEPENDENCIES

        if final_file_type.lower() == 'ims':
            fused_file_type = 'hdf5'
        else:
            fused_file_type = final_file_type

        cmd = ''
        cmd += f'{mesospim_root_application}/automated.py big-stitcher-align'
        cmd += f' {Path(out_dir).parent}'
        cmd += f' --fused-file-type {fused_file_type} --final-file-type {final_file_type}'
        job_number = wrap_slurm(cmd, SLURM_PARAMETERS_FOR_DEPENDENCIES, out_dir,
                                after_slurm_jobs=[job_number] if job_number else None, username=username)
        print(f'Dependency process number: {job_number}')


@app.command()
def convert_btf_tiles_to_omezarr_slurm_array(dir_loc: Path, file_type: str='.btf', queue_alignment: bool=True, final_file_type: str='omezarr',
                                after_slurm_jobs: list[int]=None):

    from constants import SLURM_PARAMETERS_FOR_BIGSTITCHER, SLURM_PARAMETERS_FOR_DEPENDENCIES, SLURM_PARAMETERS_OMEZARR
    import zarr

    btf_file_list = list(dir_loc.glob(f'*{file_type}'))

    if len(btf_file_list) == 0:
        raise FileNotFoundError(f'No files found in {dir_loc} with file type {file_type}')

    prefix = common_prefix([f.name for f in btf_file_list])

    if prefix:
        # Remove trailing underscores or hyphens from prefix
        prefix = strip_after(prefix, '_max')
    else:
        # Use first file name as prefix
        prefix = btf_file_list[0].name

    # Create output directory for omezarr collection ./ome_zarr/{first_btf_file_name}/
    output_directory_for_omezarr_collection = dir_loc / 'ome_zarr' / f'{prefix}.ome.zarr'
    output_directory_for_omezarr_collection.mkdir(parents=True, exist_ok=True)

    # Create ome.zarr group to store all converted btf files in the collection
    zarr.open_group(output_directory_for_omezarr_collection, mode="a", zarr_version=2)

    print(f'Extracting metadata from {dir_loc}')
    metadata_by_channel = collect_all_metadata(dir_loc)
    first_metadata_entry = get_first_entry(metadata_by_channel)

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
                     output_directory_for_omezarr_collection,
                     after_slurm_jobs=None, username=username
                              )

    print(f'OME-Zarr Conversion Array Job Number: {job_number}')


    # Make bigstitcher xml for alignment and stitching of the converted omezarr files
    xml_file_name = Path(output_directory_for_omezarr_collection.as_posix() + '.xml')
    cmd = f'{mesospim_root_application}/bigstitcher.py mesospim-metadata-to-bigstitcher-xml'
    cmd += f' "{xml_file_name}"'
    cmd += f' --different-relative-zarr-path "{output_directory_for_omezarr_collection.name}"'
    cmd += f' --modify-filename-in-xml .ome.zarr'

    job_number = wrap_slurm(cmd, SLURM_PARAMETERS_FOR_DEPENDENCIES, output_directory_for_omezarr_collection,
                            after_slurm_jobs=[job_number] if job_number else None, username=username)

    print(f'Queued BigStitcher XML build process number: {job_number}')

    if queue_alignment:
        # Determine file formats relevant for downstream processes based on desired output
        fused_file_type, final_file_type = get_intermediate_file_type_for_bigstitcher_alignment(final_file_type)
        cmd = f'{mesospim_root_application}/automated.py big-stitcher-align'
        cmd += f' {output_directory_for_omezarr_collection.parent}'
        cmd += f' --fused-file-type {fused_file_type} --final-file-type {final_file_type}'

        job_number = wrap_slurm(cmd, SLURM_PARAMETERS_FOR_BIGSTITCHER, output_directory_for_omezarr_collection,
                                after_slurm_jobs=[job_number] if job_number else None, username=username)

        print(f'Queued BigStitcher alignment process number: {job_number}')






def get_intermediate_file_type_for_bigstitcher_alignment(final_file_type: str):
    '''
    Given the desired final file type (omezarr, hdf5, ims) return the fused_file_type and final_file_type.
     fused_file_type is used for bigstitcher fusion.
     final_file_type is the final output format after bigstitcher alignment and fusion.
     '''

    if final_file_type.lower() == 'omezarr':
        return 'omezarr', 'omezarr'
    elif final_file_type.lower() == 'hdf5':
        return 'hdf5', 'hdf5'
    elif final_file_type.lower() == 'ims':
        return 'hdf5', 'ims'



@app.command()
def big_stitcher_align(dir_loc: Path, fused_file_type: str='omezarr', final_file_type: str='omezarr'):
    print('Setting up script to run omezarr alignment')
    from constants import SLURM_PARAMETERS_FOR_BIGSTITCHER
    from constants import SLURM_PARAMETERS_FOR_DEPENDENCIES
    from string_templates import BIGSTITCHER_ALIGN_TEMPLATE

    print(f'Extracting metadata from {dir_loc}')
    metadata_by_channel = collect_all_metadata(dir_loc)
    first_metadata_entry = get_first_entry(metadata_by_channel)

    username = first_metadata_entry.get('username', "")

    bigstitcher_dir, fused_out_dir_or_file, macro_file = make_bigstitcher_slurm_dir_and_macro(dir_loc, format=fused_file_type)
    cmd = BIGSTITCHER_ALIGN_TEMPLATE.format(macro_file)

    job_number = None
    job_number = wrap_slurm(cmd, SLURM_PARAMETERS_FOR_BIGSTITCHER, bigstitcher_dir,
                            after_slurm_jobs=[job_number] if job_number else None, username=username)
    print(f'BigStitcher process number: {job_number}')

    if final_file_type.lower() == 'omezarr':
        cmd = f'{mesospim_root_application}/bigstitcher.py adjust-scale-in-bigstitcher-produced-ome-zarr'
        cmd += f' "{dir_loc}" "{fused_out_dir_or_file}"'
        job_number = wrap_slurm(cmd, SLURM_PARAMETERS_FOR_DEPENDENCIES, bigstitcher_dir,
                                after_slurm_jobs=[job_number] if job_number else None, username=username)
        print(f'BigStitcher Fix OME-Zarr Metadata Scale: {job_number}')

    elif fused_file_type.lower() == 'hdf5' and final_file_type.lower() == 'ims':
        from constants import SLURM_PARAMETERS_IMARIS_CONVERTER
        metadata = collect_all_metadata(dir_loc)
        first_entry = get_first_entry(metadata)
        res = determine_xyz_resolution(first_entry) #zyx

        current_script, log_location, out_dir = convert_ims(fused_out_dir_or_file, res=res, run_conversion=False)
        job_number = wrap_slurm(current_script,
                                SLURM_PARAMETERS_IMARIS_CONVERTER, log_location,
                                after_slurm_jobs=[job_number] if job_number else None, username=username)
        print(f'BigStitcher HDF5 Convert to IMS: {job_number}')



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

