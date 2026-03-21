'''
************************************************************************************************************************
Example Tile alignment macro for BigStitcher alignment of omezarr data, to be run headlessly in Fiji with BigStitcher installed
************************************************************************************************************************
run("Calculate pairwise shifts ...",
"select=I:/trash/reprocess-dont-delete/iyer-s/4CL59/093025/ome_zarr/a4flox_PBS_Mag8x_Tile.ome.zarr.xml \
process_angle=[All angles] \
process_channel=[All channels] \
process_illumination=[All illuminations] \
process_tile=[All tiles] \
process_timepoint=[All Timepoints] \
method=[Phase Correlation] \
show_expert_grouping_options \
how_to_treat_timepoints=[treat individually] \
how_to_treat_channels=group \
how_to_treat_illuminations=group \
how_to_treat_angles=[treat individually] \
how_to_treat_tiles=compare \
channels=[Average Channels] \
downsample_in_x=16 \
downsample_in_y=16 \
downsample_in_z=2");

************************************************************************************************************************
Example Global optimization macro for BigStitcher alignment of omezarr data, to be run headlessly in Fiji with BigStitcher installed
************************************************************************************************************************
run("Optimize globally and apply shifts ...",
"select=I:/trash/reprocess-dont-delete/iyer-s/4CL59/093025/ome_zarr/a4flox_PBS_Mag8x_Tile.ome.zarr.xml \
process_angle=[All angles] \
process_channel=[All channels] \
process_illumination=[All illuminations] \
process_tile=[All tiles] \
process_timepoint=[All Timepoints] \
relative=2.500 \
absolute=3.500 \
global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad links] \
show_expert_grouping_options \
how_to_treat_timepoints=[treat individually] \
how_to_treat_channels=group \
how_to_treat_illuminations=group how_to_treat_angles=[treat individually] \
how_to_treat_tiles=compare \
fix_group_0-0,");

************************************************************************************************************************
Example IPC refinement macro for BigStitcher alignment of omezarr data, to be run headlessly in Fiji with BigStitcher installed
Great for refining tile alignments after an initial global optimization,
And chromatic aberration correction
************************************************************************************************************************

run("ICP Refinement ...",
"select=I:/trash/reprocess-dont-delete/iyer-s/4CL59/093025/ome_zarr/a4flox_PBS_Mag8x_Tile.ome.zarr.xml \
process_angle=[All angles] \
process_channel=[All channels] \
process_illumination=[All illuminations] \
process_tile=[All tiles] \
process_timepoint=[All Timepoints] \
icp_refinement_type=[Simple (all together)] \
global_optimization_strategy=[Two-Round: Handle unconnected tiles, remove wrong links RELAXED (5.0x / 7.0px)] \
downsampling=[Downsampling 8/8/4] \
interest=[Average Threshold] \
icp_max_error=[Normal Adjustment (<5px)]");
************************************************************************************************************************
'''


'''
HOW TO USE THE BIGSTITCHER_ALIGN macro templates:

- format in order with:
    - input ome.zarr.xml path
    - downsample_in_x for alignment
    - downsample_in_y for alignment
    - downsample_in_z for alignment
    - output fused ome.zarr path
    - block_size_x for fused output
    - block_size_y for fused output
    - block_size_z for fused output
    - block_size_factor_x for processing fused output (block_size multiplier)
    - block_size_factor_y for processing fused output (block_size multiplier)
    - block_size_factor_z for processing fused output (block_size multiplier)
    - subsampling_factors string, e.g. { {1,1,1}, {2,2,2}, {4,4,4}, {8,8,8} }
    - downsample_refinement string for ICP refinement, e.g. "8/8/4", "1/1/1" (no downsampling) #X/Y/Z

BIGSTITCHER_ALIGN_OMEZARR_OUT.format(0,1,2,3,4,5,6,7,8,9,10,11,12)
'''


BIGSTITCHER_ALIGN_OMEZARR_OUT = '''
// This is a ImageJ macro file for running BigStitcher alignment
// It can be run headlessly in Fiji with BigStitcher installed

// --------------------------------------------------------------------
// STAGE 1: Tile alignment using channels=[Average Channels]
// --------------------------------------------------------------------

// select= xml file path to the ome.zarr.xml file
// downsample_in_x/y/z = downsampling factors for alignment (1 = no downsampling)

print("");
print("----------------------------------------------------------------------------------------");
print("Stage 1: Calculate pairwise shifts (Average Channels)");
print("----------------------------------------------------------------------------------------");
print("");

run("Calculate pairwise shifts ...",
"select='{0}' \
process_angle=[All angles] \
process_channel=[All channels] \
process_illumination=[All illuminations] \
process_tile=[All tiles] \
process_timepoint=[All Timepoints] \
method=[Phase Correlation] \
show_expert_grouping_options \
how_to_treat_timepoints=[treat individually] \
how_to_treat_channels=group \
how_to_treat_illuminations=group \
how_to_treat_angles=[treat individually] \
how_to_treat_tiles=compare \
channels=[Average Channels] \
downsample_in_x={1} \
downsample_in_y={2} \
downsample_in_z={3}");

// select= xml file path to the ome.zarr.xml file

print("");
print("----------------------------------------------------------------------------------------");
print("Stage 1: Optimize globally and apply shifts (tile geometry)");
print("----------------------------------------------------------------------------------------");
print("");

run("Optimize globally and apply shifts ...",
"select='{0}' \
process_angle=[All angles] \
process_channel=[All channels] \
process_illumination=[All illuminations] \
process_tile=[All tiles] \
process_timepoint=[All Timepoints] \
relative=2.500 \
absolute=3.500 \
global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad links] \
show_expert_grouping_options \
how_to_treat_timepoints=[treat individually] \
how_to_treat_channels=group \
how_to_treat_illuminations=group \
how_to_treat_angles=[treat individually] \
how_to_treat_tiles=compare \
fix_group_0-0");

// --------------------------------------------------------------------
// STAGE 2: Interest Point Refinement for tile and channel alignment
// --------------------------------------------------------------------

// select= xml file path to the ome.zarr.xml file
// downsample_in_x/y/z = downsampling factors for alignment (1 = no downsampling)

print("");
print("----------------------------------------------------------------------------------------");
print("Stage 2: Interest Point Refinement for tile and channel alignment");
print("----------------------------------------------------------------------------------------");
print("");

run("ICP Refinement ...",
"select='{0}' \
process_angle=[All angles] \
process_channel=[All channels] \
process_illumination=[All illuminations] \
process_tile=[All tiles] \
process_timepoint=[All Timepoints] \
icp_refinement_type=[Simple (chromatic abberation)] \
global_optimization_strategy=[Two-Round: Handle unconnected tiles, remove wrong links RELAXED (5.0x / 7.0px)] \
downsampling=[Downsampling {12}] \
interest=[Average Threshold] \
icp_max_error=[Relaxed (<10px)]");

//[Strict (<2px)]
//[Normal Adjustment (<5px)]
//[Relaxed (<10px)]
//[Very Relaxed (<20px)]


// --------------------------------------------------------------------
// STAGE 3: Image Fusion into OME-Zarr
// --------------------------------------------------------------------


// select= xml file path to the ome.zarr.xml file
// zarr_dataset_path= output path for fused ome.zarr file
// block_size_x/y/z = block size for fused output
// block_size_factor_x/y/z = block size factors for fused output
// subsampling_factors= downsampling factors for multiscale output.

print("");
print("----------------------------------------------------------------------------------------");
print("Creating Fused Dataset to OME-Zarr");
print("----------------------------------------------------------------------------------------");
print("");

run("Image Fusion",
"select='{0}' \
process_angle=[All angles] \
process_channel=[All channels] \
process_illumination=[All illuminations] \
process_tile=[All tiles] \
process_timepoint=[All Timepoints] \
bounding_box=[Currently Selected Views] \
downsampling=1 \
interpolation=[Linear Interpolation] \
fusion_type=[Avg, Blending] \
pixel_type=[16-bit unsigned integer] \
interest_points_for_non_rigid=[-= Disable Non-Rigid =-] \
preserve_original \
produce=[Each timepoint & channel] \
fused_image=[OME-ZARR/N5/HDF5 export using N5-API] \
define_input=[Auto-load from input data (values shown below)] \
export=OME-ZARR \
compression=Zstandard compression_level=5 \
create_multi-resolution \
store \
zarr_dataset_path='{4}' \
show_advanced_block_size_options \
block_size_x={5} \
block_size_y={6} \
block_size_z={7} \
block_size_factor_x={8} \
block_size_factor_y={9} \
block_size_factor_z={10} \
subsampling_factors=[{11}]");

// hard-exit JVM so Slurm releases resources
call("java.lang.System.exit", "0");
'''




BIGSTITCHER_ALIGN_HDF5_OUT = '''
// This is a ImageJ macro file for running BigStitcher alignment
// It can be run headlessly in Fiji with BigStitcher installed

// --------------------------------------------------------------------
// STAGE 1: Tile alignment using channels=[Average Channels]
// --------------------------------------------------------------------

// select= xml file path to the ome.zarr.xml file
// downsample_in_x/y/z = downsampling factors for alignment (1 = no downsampling)

print("");
print("----------------------------------------------------------------------------------------");
print("Stage 1: Calculate pairwise shifts (Average Channels)");
print("----------------------------------------------------------------------------------------");
print("");

run("Calculate pairwise shifts ...",
"select='{0}' \
process_angle=[All angles] \
process_channel=[All channels] \
process_illumination=[All illuminations] \
process_tile=[All tiles] \
process_timepoint=[All Timepoints] \
method=[Phase Correlation] \
show_expert_grouping_options \
how_to_treat_timepoints=[treat individually] \
how_to_treat_channels=group \
how_to_treat_illuminations=group \
how_to_treat_angles=[treat individually] \
how_to_treat_tiles=compare \
channels=[Average Channels] \
downsample_in_x={1} \
downsample_in_y={2} \
downsample_in_z={3}");

// select= xml file path to the ome.zarr.xml file

print("");
print("----------------------------------------------------------------------------------------");
print("Stage 1: Optimize globally and apply shifts (tile geometry)");
print("----------------------------------------------------------------------------------------");
print("");

run("Optimize globally and apply shifts ...",
"select='{0}' \
process_angle=[All angles] \
process_channel=[All channels] \
process_illumination=[All illuminations] \
process_tile=[All tiles] \
process_timepoint=[All Timepoints] \
relative=2.500 \
absolute=3.500 \
global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad links] \
show_expert_grouping_options \
how_to_treat_timepoints=[treat individually] \
how_to_treat_channels=group \
how_to_treat_illuminations=group \
how_to_treat_angles=[treat individually] \
how_to_treat_tiles=compare \
fix_group_0-0");

// --------------------------------------------------------------------
// STAGE 2: Interest Point Refinement for channel alignment
// --------------------------------------------------------------------

// select= xml file path to the ome.zarr.xml file
// downsample_in_x/y/z = downsampling factors for alignment (1 = no downsampling)

print("");
print("----------------------------------------------------------------------------------------");
print("Stage 2: Interest Point Refinement for channel alignment");
print("----------------------------------------------------------------------------------------");
print("");

run("ICP Refinement ...",
"select='{0}' \
process_angle=[All angles] \
process_channel=[All channels] \
process_illumination=[All illuminations] \
process_tile=[All tiles] \
process_timepoint=[All Timepoints] \
icp_refinement_type=[Simple (chromatic abberation)] \
global_optimization_strategy=[Two-Round: Handle unconnected tiles, remove wrong links RELAXED (5.0x / 7.0px)] \
downsampling=[Downsampling {12}] \
interest=[Average Threshold] \
icp_max_error=[Normal Adjustment (<5px)]");


// --------------------------------------------------------------------
// STAGE 3: Image Fusion into HDF5
// --------------------------------------------------------------------


// select= xml file path to the ome.zarr.xml file
// zarr_dataset_path= output path for fused ome.zarr file
// block_size_x/y/z = block size for fused output
// block_size_factor_x/y/z = block size factors for fused output
// subsampling_factors= downsampling factors for multiscale output.

print("");
print("----------------------------------------------------------------------------------------");
print("Creating Fused Dataset to HDF5");
print("----------------------------------------------------------------------------------------");
print("");

run("Image Fusion",
"select='{0}' \
process_angle=[All angles] \
process_channel=[All channels] \
process_illumination=[All illuminations] \
process_tile=[All tiles] \
process_timepoint=[All Timepoints] \
bounding_box=[Currently Selected Views] \
downsampling=1 \
interpolation=[Linear Interpolation] \
fusion_type=[Avg, Blending] \
pixel_type=[16-bit unsigned integer] \
interest_points_for_non_rigid=[-= Disable Non-Rigid =-] \
preserve_original \
produce=[Each timepoint & channel] \
fused_image=[OME-ZARR/N5/HDF5 export using N5-API] \
define_input=[Auto-load from input data (values shown below)] \
export=HDF5 \
compression=Zstandard compression_level=5 \
create_a_bdv/bigstitcher \
hdf5_file='{4}' \
xml_output_file={4}.xml \
show_advanced_block_size_options \
block_size_x={5} \
block_size_y={6} \
block_size_z={7} \
block_size_factor_x={8} \
block_size_factor_y={9} \
block_size_factor_z={10} \
subsampling_factors=[{11}]");

// hard-exit JVM so Slurm releases resources
call("java.lang.System.exit", "0");
'''


# BIGSTITCHER_ALIGN_OMEZARR_OUT = '''
# // This is a ImageJ macro file for running BigStitcher alignment
# // It can be run headlessly in Fiji with BigStitcher installed
#
# // --------------------------------------------------------------------
# // STAGE 1: Tile alignment using channels=[Average Channels]
# // --------------------------------------------------------------------
#
# // select= xml file path to the ome.zarr.xml file
# // downsample_in_x/y/z = downsampling factors for alignment (1 = no downsampling)
#
# print("");
# print("----------------------------------------------------------------------------------------");
# print("Stage 1: Calculate pairwise shifts (Average Channels)");
# print("----------------------------------------------------------------------------------------");
# print("");
#
# run("Calculate pairwise shifts ...",
# "select='{0}' \
# process_angle=[All angles] \
# process_channel=[All channels] \
# process_illumination=[All illuminations] \
# process_tile=[All tiles] \
# process_timepoint=[All Timepoints] \
# method=[Phase Correlation] \
# show_expert_grouping_options \
# how_to_treat_timepoints=[treat individually] \
# how_to_treat_channels=group \
# how_to_treat_illuminations=group \
# how_to_treat_angles=[treat individually] \
# how_to_treat_tiles=compare \
# channels=[Average Channels] \
# downsample_in_x={1} \
# downsample_in_y={2} \
# downsample_in_z={3}");
#
# // select= xml file path to the ome.zarr.xml file
#
# print("");
# print("----------------------------------------------------------------------------------------");
# print("Stage 1: Optimize globally and apply shifts (tile geometry)");
# print("----------------------------------------------------------------------------------------");
# print("");
#
# run("Optimize globally and apply shifts ...",
# "select='{0}' \
# process_angle=[All angles] \
# process_channel=[All channels] \
# process_illumination=[All illuminations] \
# process_tile=[All tiles] \
# process_timepoint=[All Timepoints] \
# relative=2.500 \
# absolute=3.500 \
# global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad links] \
# show_expert_grouping_options \
# how_to_treat_timepoints=[treat individually] \
# how_to_treat_channels=group \
# how_to_treat_illuminations=group \
# how_to_treat_angles=[treat individually] \
# how_to_treat_tiles=compare \
# fix_group_0-0");
#
# // --------------------------------------------------------------------
# // STAGE 2: Interest Point Refinement for tile and channel alignment
# // --------------------------------------------------------------------
#
# // select= xml file path to the ome.zarr.xml file
# // downsample_in_x/y/z = downsampling factors for alignment (1 = no downsampling)
#
# print("");
# print("----------------------------------------------------------------------------------------");
# print("Stage 2: Interest Point Refinement for tile and channel alignment");
# print("----------------------------------------------------------------------------------------");
# print("");
#
# run("ICP Refinement ...",
# "select='{0}' \
# process_angle=[All angles] \
# process_channel=[All channels] \
# process_illumination=[All illuminations] \
# process_tile=[All tiles] \
# process_timepoint=[All Timepoints] \
# icp_refinement_type=[Simple (all together)] \
# global_optimization_strategy=[Two-Round: Handle unconnected tiles, remove wrong links RELAXED (5.0x / 7.0px)] \
# downsampling=[Downsampling {12}] \
# interest=[Average Threshold] \
# icp_max_error=[Normal Adjustment (<5px)]");
#
#
# // --------------------------------------------------------------------
# // STAGE 3: Image Fusion into OME-Zarr
# // --------------------------------------------------------------------
#
#
# // select= xml file path to the ome.zarr.xml file
# // zarr_dataset_path= output path for fused ome.zarr file
# // block_size_x/y/z = block size for fused output
# // block_size_factor_x/y/z = block size factors for fused output
# // subsampling_factors= downsampling factors for multiscale output.
#
# print("");
# print("----------------------------------------------------------------------------------------");
# print("Creating Fused Dataset to OME-Zarr");
# print("----------------------------------------------------------------------------------------");
# print("");
#
# run("Image Fusion",
# "select='{0}' \
# process_angle=[All angles] \
# process_channel=[All channels] \
# process_illumination=[All illuminations] \
# process_tile=[All tiles] \
# process_timepoint=[All Timepoints] \
# bounding_box=[Currently Selected Views] \
# downsampling=1 \
# interpolation=[Linear Interpolation] \
# fusion_type=[Avg, Blending] \
# pixel_type=[16-bit unsigned integer] \
# interest_points_for_non_rigid=[-= Disable Non-Rigid =-] \
# preserve_original \
# produce=[Each timepoint & channel] \
# fused_image=[OME-ZARR/N5/HDF5 export using N5-API] \
# define_input=[Auto-load from input data (values shown below)] \
# export=OME-ZARR \
# compression=Zstandard compression_level=5 \
# create_multi-resolution \
# store \
# zarr_dataset_path='{4}' \
# show_advanced_block_size_options \
# block_size_x={5} \
# block_size_y={6} \
# block_size_z={7} \
# block_size_factor_x={8} \
# block_size_factor_y={9} \
# block_size_factor_z={10} \
# subsampling_factors=[{11}]");
#
# // hard-exit JVM so Slurm releases resources
# call("java.lang.System.exit", "0");
# '''
#
#
#
#
# BIGSTITCHER_ALIGN_HDF5_OUT = '''
# // This is a ImageJ macro file for running BigStitcher alignment
# // It can be run headlessly in Fiji with BigStitcher installed
#
# // --------------------------------------------------------------------
# // STAGE 1: Tile alignment using channels=[Average Channels]
# // --------------------------------------------------------------------
#
# // select= xml file path to the ome.zarr.xml file
# // downsample_in_x/y/z = downsampling factors for alignment (1 = no downsampling)
#
# print("");
# print("----------------------------------------------------------------------------------------");
# print("Stage 1: Calculate pairwise shifts (Average Channels)");
# print("----------------------------------------------------------------------------------------");
# print("");
#
# run("Calculate pairwise shifts ...",
# "select='{0}' \
# process_angle=[All angles] \
# process_channel=[All channels] \
# process_illumination=[All illuminations] \
# process_tile=[All tiles] \
# process_timepoint=[All Timepoints] \
# method=[Phase Correlation] \
# show_expert_grouping_options \
# how_to_treat_timepoints=[treat individually] \
# how_to_treat_channels=group \
# how_to_treat_illuminations=group \
# how_to_treat_angles=[treat individually] \
# how_to_treat_tiles=compare \
# channels=[Average Channels] \
# downsample_in_x={1} \
# downsample_in_y={2} \
# downsample_in_z={3}");
#
# // select= xml file path to the ome.zarr.xml file
#
# print("");
# print("----------------------------------------------------------------------------------------");
# print("Stage 1: Optimize globally and apply shifts (tile geometry)");
# print("----------------------------------------------------------------------------------------");
# print("");
#
# run("Optimize globally and apply shifts ...",
# "select='{0}' \
# process_angle=[All angles] \
# process_channel=[All channels] \
# process_illumination=[All illuminations] \
# process_tile=[All tiles] \
# process_timepoint=[All Timepoints] \
# relative=2.500 \
# absolute=3.500 \
# global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad links] \
# show_expert_grouping_options \
# how_to_treat_timepoints=[treat individually] \
# how_to_treat_channels=group \
# how_to_treat_illuminations=group \
# how_to_treat_angles=[treat individually] \
# how_to_treat_tiles=compare \
# fix_group_0-0");
#
# // --------------------------------------------------------------------
# // STAGE 2: Interest Point Refinement for tile and channel alignment
# // --------------------------------------------------------------------
#
# // select= xml file path to the ome.zarr.xml file
# // downsample_in_x/y/z = downsampling factors for alignment (1 = no downsampling)
#
# print("");
# print("----------------------------------------------------------------------------------------");
# print("Stage 2: Interest Point Refinement for tile and channel alignment");
# print("----------------------------------------------------------------------------------------");
# print("");
#
# run("ICP Refinement ...",
# "select='{0}' \
# process_angle=[All angles] \
# process_channel=[All channels] \
# process_illumination=[All illuminations] \
# process_tile=[All tiles] \
# process_timepoint=[All Timepoints] \
# icp_refinement_type=[Simple (all together)] \
# global_optimization_strategy=[Two-Round: Handle unconnected tiles, remove wrong links RELAXED (5.0x / 7.0px)] \
# downsampling=[Downsampling {12}] \
# interest=[Average Threshold] \
# icp_max_error=[Normal Adjustment (<5px)]");
#
#
# // --------------------------------------------------------------------
# // STAGE 3: Image Fusion into HDF5
# // --------------------------------------------------------------------
#
#
# // select= xml file path to the ome.zarr.xml file
# // zarr_dataset_path= output path for fused ome.zarr file
# // block_size_x/y/z = block size for fused output
# // block_size_factor_x/y/z = block size factors for fused output
# // subsampling_factors= downsampling factors for multiscale output.
#
# print("");
# print("----------------------------------------------------------------------------------------");
# print("Creating Fused Dataset to HDF5");
# print("----------------------------------------------------------------------------------------");
# print("");
#
# run("Image Fusion",
# "select='{0}' \
# process_angle=[All angles] \
# process_channel=[All channels] \
# process_illumination=[All illuminations] \
# process_tile=[All tiles] \
# process_timepoint=[All Timepoints] \
# bounding_box=[Currently Selected Views] \
# downsampling=1 \
# interpolation=[Linear Interpolation] \
# fusion_type=[Avg, Blending] \
# pixel_type=[16-bit unsigned integer] \
# interest_points_for_non_rigid=[-= Disable Non-Rigid =-] \
# preserve_original \
# produce=[Each timepoint & channel] \
# fused_image=[OME-ZARR/N5/HDF5 export using N5-API] \
# define_input=[Auto-load from input data (values shown below)] \
# export=HDF5 \
# compression=Zstandard compression_level=5 \
# create_a_bdv/bigstitcher \
# hdf5_file='{4}' \
# xml_output_file={4}.xml \
# show_advanced_block_size_options \
# block_size_x={5} \
# block_size_y={6} \
# block_size_z={7} \
# block_size_factor_x={8} \
# block_size_factor_y={9} \
# block_size_factor_z={10} \
# subsampling_factors=[{11}]");
#
# // hard-exit JVM so Slurm releases resources
# call("java.lang.System.exit", "0");
# '''


########################################################################################################################
## Depreciated scripts for BigStitcher alignment of omezarr data, to be run headlessly in Fiji with BigStitcher installed
## Replaced with scripts that include ICP refinement after global optimization of phase correlation tile alignment.
## ICP is helpful for refining tile alignments and for chromatic aberration correction (channel alignment).
########################################################################################################################

# BIGSTITCHER_ALIGN_OMEZARR_OUT = '''
# // This is a ImageJ macro file for running BigStitcher alignment
# // It can be run headlessly in Fiji with BigStitcher installed
#
# // --------------------------------------------------------------------
# // STAGE 1: Tile alignment using channels=[Average Channels]
# // --------------------------------------------------------------------
#
# // select= xml file path to the ome.zarr.xml file
# // downsample_in_x/y/z = downsampling factors for alignment (1 = no downsampling)
#
# print("");
# print("----------------------------------------------------------------------------------------");
# print("Stage 1: Calculate pairwise shifts (Average Channels)");
# print("----------------------------------------------------------------------------------------");
# print("");
#
# run("Calculate pairwise shifts ...",
# "select='{0}' \
# process_angle=[All angles] \
# process_channel=[All channels] \
# process_illumination=[All illuminations] \
# process_tile=[All tiles] \
# process_timepoint=[All Timepoints] \
# method=[Phase Correlation] \
# show_expert_grouping_options \
# how_to_treat_timepoints=[treat individually] \
# how_to_treat_channels=group \
# how_to_treat_illuminations=group \
# how_to_treat_angles=[treat individually] \
# how_to_treat_tiles=compare \
# channels=[Average Channels] \
# downsample_in_x={1} \
# downsample_in_y={2} \
# downsample_in_z={3}");
#
# // select= xml file path to the ome.zarr.xml file
#
# print("");
# print("----------------------------------------------------------------------------------------");
# print("Stage 1: Optimize globally and apply shifts (tile geometry)");
# print("----------------------------------------------------------------------------------------");
# print("");
#
# run("Optimize globally and apply shifts ...",
# "select='{0}' \
# process_angle=[All angles] \
# process_channel=[All channels] \
# process_illumination=[All illuminations] \
# process_tile=[All tiles] \
# process_timepoint=[All Timepoints] \
# relative=2.500 \
# absolute=3.500 \
# global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad links] \
# show_expert_grouping_options \
# how_to_treat_timepoints=[treat individually] \
# how_to_treat_channels=group \
# how_to_treat_illuminations=group \
# how_to_treat_angles=[treat individually] \
# how_to_treat_tiles=compare \
# fix_group_0-0");
#
# // --------------------------------------------------------------------
# // STAGE 2: Channel alignment with fixed tile positions
# // --------------------------------------------------------------------
#
# // select= xml file path to the ome.zarr.xml file
# // downsample_in_x/y/z = downsampling factors for alignment (1 = no downsampling)
#
# print("");
# print("----------------------------------------------------------------------------------------");
# print("Stage 2: Optimize per channel with fixed tile positions (Channel alignment)");
# print("----------------------------------------------------------------------------------------");
# print("");
#
# run("Calculate pairwise shifts ...",
# "select='{0}' \
# process_angle=[All angles] \
# process_channel=[All channels] \
# process_illumination=[All illuminations] \
# process_tile=[All tiles] \
# process_timepoint=[All Timepoints] \
# method=[Phase Correlation] \
# show_expert_grouping_options \
# how_to_treat_timepoints=[treat individually] \
# how_to_treat_channels=compare \
# how_to_treat_illuminations=group \
# how_to_treat_angles=[treat individually] \
# how_to_treat_tiles=group \
# illuminations=[Average Illuminations] \
# tiles=[Average Tiles] \
# downsample_in_x={1} \
# downsample_in_y={2} \
# downsample_in_z={3}");
#
#
# // select= xml file path to the ome.zarr.xml file
#
# print("");
# print("----------------------------------------------------------------------------------------");
# print("Stage 2: Optimize globally and apply shifts (Channel geometry)");
# print("----------------------------------------------------------------------------------------");
# print("");
#
# run("Optimize globally and apply shifts ...",
# "select='{0}' \
# process_angle=[All angles] \
# process_channel=[All channels] \
# process_illumination=[All illuminations] \
# process_tile=[All tiles] \
# process_timepoint=[All Timepoints] \
# relative=2.500 \
# absolute=3.500 \
# global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad links] \
# show_expert_grouping_options \
# how_to_treat_timepoints=[treat individually] \
# how_to_treat_channels=compare \
# how_to_treat_illuminations=group \
# how_to_treat_angles=[treat individually] \
# how_to_treat_tiles=group \
# fix_group_0-0");
#
#
# // --------------------------------------------------------------------
# // STAGE 3: Image Fusion into OME-Zarr
# // --------------------------------------------------------------------
#
#
# // select= xml file path to the ome.zarr.xml file
# // zarr_dataset_path= output path for fused ome.zarr file
# // block_size_x/y/z = block size for fused output
# // block_size_factor_x/y/z = block size factors for fused output
# // subsampling_factors= downsampling factors for multiscale output.
#
# print("");
# print("----------------------------------------------------------------------------------------");
# print("Creating Fused Dataset to OME-Zarr");
# print("----------------------------------------------------------------------------------------");
# print("");
#
# run("Image Fusion",
# "select='{0}' \
# process_angle=[All angles] \
# process_channel=[All channels] \
# process_illumination=[All illuminations] \
# process_tile=[All tiles] \
# process_timepoint=[All Timepoints] \
# bounding_box=[Currently Selected Views] \
# downsampling=1 \
# interpolation=[Linear Interpolation] \
# fusion_type=[Avg, Blending] \
# pixel_type=[16-bit unsigned integer] \
# interest_points_for_non_rigid=[-= Disable Non-Rigid =-] \
# preserve_original \
# produce=[Each timepoint & channel] \
# fused_image=[OME-ZARR/N5/HDF5 export using N5-API] \
# define_input=[Auto-load from input data (values shown below)] \
# export=OME-ZARR \
# compression=Zstandard compression_level=5 \
# create_multi-resolution \
# store \
# zarr_dataset_path='{4}' \
# show_advanced_block_size_options \
# block_size_x={5} \
# block_size_y={6} \
# block_size_z={7} \
# block_size_factor_x={8} \
# block_size_factor_y={9} \
# block_size_factor_z={10} \
# subsampling_factors=[{11}]");
#
# // hard-exit JVM so Slurm releases resources
# call("java.lang.System.exit", "0");
# '''




# BIGSTITCHER_ALIGN_HDF5_OUT = '''
# // This is a ImageJ macro file for running BigStitcher alignment
# // It can be run headlessly in Fiji with BigStitcher installed
#
# // --------------------------------------------------------------------
# // STAGE 1: Tile alignment using channels=[Average Channels]
# // --------------------------------------------------------------------
#
# // select= xml file path to the ome.zarr.xml file
# // downsample_in_x/y/z = downsampling factors for alignment (1 = no downsampling)
#
# print("");
# print("----------------------------------------------------------------------------------------");
# print("Stage 1: Calculate pairwise shifts (Average Channels)");
# print("----------------------------------------------------------------------------------------");
# print("");
#
# run("Calculate pairwise shifts ...",
# "select='{0}' \
# process_angle=[All angles] \
# process_channel=[All channels] \
# process_illumination=[All illuminations] \
# process_tile=[All tiles] \
# process_timepoint=[All Timepoints] \
# method=[Phase Correlation] \
# show_expert_grouping_options \
# how_to_treat_timepoints=[treat individually] \
# how_to_treat_channels=group \
# how_to_treat_illuminations=group \
# how_to_treat_angles=[treat individually] \
# how_to_treat_tiles=compare \
# channels=[Average Channels] \
# downsample_in_x={1} \
# downsample_in_y={2} \
# downsample_in_z={3}");
#
# // select= xml file path to the ome.zarr.xml file
#
# print("");
# print("----------------------------------------------------------------------------------------");
# print("Stage 1: Optimize globally and apply shifts (tile geometry)");
# print("----------------------------------------------------------------------------------------");
# print("");
#
# run("Optimize globally and apply shifts ...",
# "select='{0}' \
# process_angle=[All angles] \
# process_channel=[All channels] \
# process_illumination=[All illuminations] \
# process_tile=[All tiles] \
# process_timepoint=[All Timepoints] \
# relative=2.500 \
# absolute=3.500 \
# global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad links] \
# show_expert_grouping_options \
# how_to_treat_timepoints=[treat individually] \
# how_to_treat_channels=group \
# how_to_treat_illuminations=group \
# how_to_treat_angles=[treat individually] \
# how_to_treat_tiles=compare \
# fix_group_0-0");
#
# // --------------------------------------------------------------------
# // STAGE 2: Channel alignment with fixed tile positions
# // --------------------------------------------------------------------
#
# // select= xml file path to the ome.zarr.xml file
# // downsample_in_x/y/z = downsampling factors for alignment (1 = no downsampling)
#
# print("");
# print("----------------------------------------------------------------------------------------");
# print("Stage 2: Optimize per channel with fixed tile positions (Channel alignment)");
# print("----------------------------------------------------------------------------------------");
# print("");
#
# run("Calculate pairwise shifts ...",
# "select='{0}' \
# process_angle=[All angles] \
# process_channel=[All channels] \
# process_illumination=[All illuminations] \
# process_tile=[All tiles] \
# process_timepoint=[All Timepoints] \
# method=[Phase Correlation] \
# show_expert_grouping_options \
# how_to_treat_timepoints=[treat individually] \
# how_to_treat_channels=compare \
# how_to_treat_illuminations=group \
# how_to_treat_angles=[treat individually] \
# how_to_treat_tiles=group \
# illuminations=[Average Illuminations] \
# tiles=[Average Tiles] \
# downsample_in_x={1} \
# downsample_in_y={2} \
# downsample_in_z={3}");
#
#
# // select= xml file path to the ome.zarr.xml file
#
# print("");
# print("----------------------------------------------------------------------------------------");
# print("Stage 2: Optimize globally and apply shifts (Channel geometry)");
# print("----------------------------------------------------------------------------------------");
# print("");
#
# run("Optimize globally and apply shifts ...",
# "select='{0}' \
# process_angle=[All angles] \
# process_channel=[All channels] \
# process_illumination=[All illuminations] \
# process_tile=[All tiles] \
# process_timepoint=[All Timepoints] \
# relative=2.500 \
# absolute=3.500 \
# global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad links] \
# show_expert_grouping_options \
# how_to_treat_timepoints=[treat individually] \
# how_to_treat_channels=compare \
# how_to_treat_illuminations=group \
# how_to_treat_angles=[treat individually] \
# how_to_treat_tiles=group \
# fix_group_0-0");
#
#
# // --------------------------------------------------------------------
# // STAGE 3: Image Fusion into HDF5
# // --------------------------------------------------------------------
#
#
# // select= xml file path to the ome.zarr.xml file
# // zarr_dataset_path= output path for fused ome.zarr file
# // block_size_x/y/z = block size for fused output
# // block_size_factor_x/y/z = block size factors for fused output
# // subsampling_factors= downsampling factors for multiscale output.
#
# print("");
# print("----------------------------------------------------------------------------------------");
# print("Creating Fused Dataset to HDF5");
# print("----------------------------------------------------------------------------------------");
# print("");
#
# run("Image Fusion",
# "select='{0}' \
# process_angle=[All angles] \
# process_channel=[All channels] \
# process_illumination=[All illuminations] \
# process_tile=[All tiles] \
# process_timepoint=[All Timepoints] \
# bounding_box=[Currently Selected Views] \
# downsampling=1 \
# interpolation=[Linear Interpolation] \
# fusion_type=[Avg, Blending] \
# pixel_type=[16-bit unsigned integer] \
# interest_points_for_non_rigid=[-= Disable Non-Rigid =-] \
# preserve_original \
# produce=[Each timepoint & channel] \
# fused_image=[OME-ZARR/N5/HDF5 export using N5-API] \
# define_input=[Auto-load from input data (values shown below)] \
# export=HDF5 \
# compression=Zstandard compression_level=5 \
# create_a_bdv/bigstitcher \
# hdf5_file='{4}' \
# xml_output_file={4}.xml \
# show_advanced_block_size_options \
# block_size_x={5} \
# block_size_y={6} \
# block_size_z={7} \
# block_size_factor_x={8} \
# block_size_factor_y={9} \
# block_size_factor_z={10} \
# subsampling_factors=[{11}]");
#
# // hard-exit JVM so Slurm releases resources
# call("java.lang.System.exit", "0");
# '''