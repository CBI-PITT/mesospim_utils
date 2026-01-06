from pprint import pprint as print
import zarr
from pathlib import Path

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
# origional_tile_path = '/CBI_FastStore/Acquire/MesoSPIM/alan-test/new/test_Mag2x_Ch488_Ch561_Ch638.ome.zarr/test_Mag2x_Ch488_Ch561_Ch638_montage.ome.zarr'
origional_tile_path = Path(origional_tile_path)
origional_tile_group = zarr.open(origional_tile_path)

# Extract zarr version
ZARR_VERSION = origional_tile_group.info._zarr_format

# Extract OME metadata
origional_ome_metadata = dict(origional_tile_group.attrs)

# Get full resolution array
full_res_array = origional_tile_group['0']
full_res_zattrs = dict(full_res_array.attrs)

# Get output zarr path
out_dir = origional_tile_path.parents[1] / 'decon' / origional_tile_path.parts[-2] / origional_tile_path.name
out_group = out_dir.parent
out_group.mkdir(parents=True, exist_ok=True)

# Open or create output zarr group
out_group = zarr.open_group(out_group, mode="a", zarr_version=ZARR_VERSION)
set_zattrs_recursive(out_group.attrs, origional_ome_metadata)

# Open or create full resolution array in output group
a = zarr.create(
    shape=full_res_array.shape,
    chunks=full_res_array.chunks,
    dtype=full_res_array.dtype,
    compressor=full_res_array.compressor,
    overwrite = False,
    store = out_group.store,  # same store as the group
    path = out_dir.name,  # create under store
    zarr_format = ZARR_VERSION,  # 2 or 3
    dimension_separator = "/",  # nested directories in .zarray
    )

set_zattrs_recursive(a.attrs, full_res_zattrs)

a[:] = full_res_array[:]
