from pprint import pprint as print
import zarr
from pathlib import Path

from dask import __main__

from ome_zarr_multiscale_writer.zarr_reader import OmeZarrArray

# Set array zattrs
def set_zattrs_recursive(target: dict, source: dict):
    for key, value in source.items():
        if isinstance(value, dict):
            if key not in target:
                target[key] = {}
            set_zattrs_recursive(target[key], value)
        else:
            target[key] = value


# Open zarr tile group
origional_tile_path = '/CBI_FastStore/Acquire/MesoSPIM/alan-test/new/test_Mag2x_Ch488_Ch561_Ch638.ome.zarr/Mag2_Tile1_Ch561_Sh1_Rot0.ome.zarr'

# Open with OmeZarrArray to ensure compatibility
ome_zarr_input = OmeZarrArray(origional_tile_path)
origional_tile_path = Path(ome_zarr_input.store_path)  # get the actual zarr path
ome_zarr_input.resolution_level = 0  # set to full resolution
# ome_zarr_input.timepoint_lock = 0  # set to first timepoint (should be no timepoints for mesospim data, but this doesn't hurt).

# Get output zarr path and create output structure
out_dir = origional_tile_path.parents[1] / 'decon' / origional_tile_path.parts[-2] / origional_tile_path.parts[-1]
ome_zarr_output = ome_zarr_input.omezarr_like(out_dir) # create output zarr structure like input to write all deconned data

if __name__ ==  "__main__":
    pass


