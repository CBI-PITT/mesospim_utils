

import dask.array as da
import numpy as np
import tifffile
import os, sys
from pathlib import Path
from dask import delayed
from typing import Dict, List, Tuple, Optional
from collections import namedtuple

# Add directory of module to sys.path
current_file = Path(__file__).resolve()
parent_dir = current_file.parent.parent  # ../ relative to this script
sys.path.append(str(parent_dir))


from metadata import collect_all_metadata, get_first_entry, determine_sheet_direction_from_tile_number
from resample_ims import get_median_tile_offsets

from constants import USE_SEPARATE_ALIGN_DATA_PER_SHEET
from rl import mesospim_btf_helper

def get_offsets(directory_with_align_data: Path, metadata_by_channel: dict,
                           separate_sheet_stitch=USE_SEPARATE_ALIGN_DATA_PER_SHEET):

    median_tile_offsets = get_median_tile_offsets(directory_with_align_data)

    ## Build x,y,z coordinate grids
    metadata_entry = get_first_entry(metadata_by_channel)
    overlap = metadata_entry.get('overlap') # Proportion i.e. 0.1
    tile_size_um = metadata_entry.get('tile_size_um') # namedtuple i.e TileSizeUm(x=3200, y=3200, z=9500.0)
    grid_x, grid_y = metadata_entry.get('grid_size')

    x_min = np.zeros((grid_x, grid_y))
    y_min = np.zeros((grid_x, grid_y))
    z_min = np.zeros((grid_x, grid_y))

    tile = 0
    for x in range(grid_x):
        for y in range(grid_y):
            # print(f'Coords for: {(x,y)}')

            if separate_sheet_stitch:
                light_sheet_direction = determine_sheet_direction_from_tile_number(metadata_by_channel, tile)
            else:
                light_sheet_direction = 'both'

            if light_sheet_direction == 'left':
                overs = median_tile_offsets.get('overs').get('left')
                downs = median_tile_offsets.get('downs').get('left')

            elif light_sheet_direction == 'right':
                overs = median_tile_offsets.get('overs').get('right')
                downs = median_tile_offsets.get('downs').get('right')

            else:
                overs = median_tile_offsets.get('overs').get('both')
                downs = median_tile_offsets.get('downs').get('both')

            # Axes for overs/downs are inverted in x/y compared to imaris resampler.  The all x/y values are -x and -y
            x_min[x, y] = round((x * (tile_size_um.x * (1-overlap))) + (y * -downs.get('x')) + (x * -overs.get('x')))
            y_min[x, y] = round((y * (tile_size_um.y * (1-overlap))) + (y * -downs.get('y')) + (x * -overs.get('y')))
            z_min[x, y] = round(0 + (x * overs.get('z')) + (y * downs.get('z')))
            tile += 1

    x_max = x_min + tile_size_um.x
    y_max = y_min + tile_size_um.y
    z_max = z_min + tile_size_um.z

    return x_min, x_max, y_min, y_max, z_min, z_max

PixelSizeUM = namedtuple('PixelSizeUM', ['z', 'y', 'x'])


#############################################
##  MesoSPIM zarr store ##
#############################################

import os
import h5py
import shutil
import time
import numpy as np
import json
import itertools

# from zarr.errors import (
#     MetadataError,
#     BadCompressorError,
#     ContainsArrayError,
#     ContainsGroupError,
#     FSPathExistNotDir,
#     ReadOnlyError,
# )

from numcodecs.abc import Codec
from numcodecs.compat import (
    ensure_bytes,
    ensure_text,
    ensure_contiguous_ndarray
)
# from numcodecs.registry import codec_registry

from zarr.storage._local import LocalStore


PixelSizeUM = namedtuple('PixelSizeUM', ['z', 'y', 'x'])

class mesospim_zarr_store(LocalStore):
    """
    Zarr storage adapter for reading mesospim files
    """

    def __init__(self, mesospim_acquire_path, alignment_directory, file_type = '.btf',
                 normalize_keys=True, verbose=True, read_only=True):

        self.root = Path(mesospim_acquire_path)
        self.path = Path(mesospim_acquire_path)
        self.parent = self.path.parent
        self.alignment_directory = Path(alignment_directory)
        self.file_type = file_type
        self.normalize_keys = normalize_keys
        self.verbose = verbose  # bool or int >= 1
        self._read_only = True

        self.metadata_by_ch = collect_all_metadata(self.path)
        self._sample = get_first_entry(self.metadata_by_ch)

        self.offsets = get_offsets(self.alignment_directory, self.metadata_by_ch)
        # self.alignment_offsets_microns = alignment_offsets_microns

        self.pixel_size_microns = PixelSizeUM(
            z=self._sample.get('resolution').z,
            y=self._sample.get('resolution').y,
            x=self._sample.get('resolution').x
        )

        self.offsets_to_px()
        self.bounding_box = self._calculate_bounding_box()

        self._files = ['.zarray', '.zgroup', '.zattrs', '.zmetadata']

        self.Channels = len(self.metadata_by_ch)
        # self.chunks = self.ims.chunks
        # self.shape = self.ims.shape
        # self.dtype = self.ims.dtype
        # self.ndim = self.ims.ndim

    def get_tiff_info(self, idx):
        file = self._sample.get('file_path')
        with tifffile.TiffFile(file) as tif:
            return tif.pages[idx].asarray()
    def offsets_to_px(self):
        self.offsets = list(self.offsets)
        self.offsets[0] = self.offsets[0] / self.pixel_size_microns.x
        self.offsets[1] = self.offsets[1] / self.pixel_size_microns.x
        self.offsets[2] = self.offsets[2] / self.pixel_size_microns.y
        self.offsets[3] = self.offsets[3] / self.pixel_size_microns.y
        self.offsets[4] = self.offsets[4] / self.pixel_size_microns.z
        self.offsets[5] = self.offsets[5] / self.pixel_size_microns.z

        xmin = abs(self.offsets[0].min())
        ymin = abs(self.offsets[2].min())
        zmin = abs(self.offsets[4].min())

        self.offsets[0] += xmin
        self.offsets[1] += xmin
        self.offsets[2] += ymin
        self.offsets[3] += ymin
        self.offsets[4] += zmin
        self.offsets[5] += zmin

        self.offsets[0] = np.floor(self.offsets[0])
        self.offsets[1] = np.floor(self.offsets[1])
        self.offsets[2] = np.floor(self.offsets[2])
        self.offsets[3] = np.floor(self.offsets[3])
        self.offsets[4] = np.floor(self.offsets[4])
        self.offsets[5] = np.floor(self.offsets[5])

    def _calculate_bounding_box(self):
        min_x = int(self.offsets[0].min())
        max_x = int(self.offsets[1].max())
        min_y = int(self.offsets[2].min())
        max_y = int(self.offsets[3].max())
        min_z = int(self.offsets[4].min())
        max_z = int(self.offsets[5].max())
        self.bounding_box = (min_z, max_z, min_y, max_y, min_x, max_x)

        return (min_z, max_z, min_y, max_y, min_x, max_x)

    def open_ims(self):
        return ims.ims(self.path,
                       ResolutionLevelLock=self.ResolutionLevelLock,
                       write=self.writeable, squeeze_output=False)

    def _normalize_key(self, key):
        return key.lower() if self.normalize_keys else key

    def _get_pixel_index_from_key(self, key):
        '''
        Key is expected to be 5 dims
        Function returns a slice in pixel coordinates for the provided key
        '''
        key_split = key.split('.')
        key_split = [int(x) for x in key_split]

        index = []
        for idx, key_idx in enumerate(key_split):
            Start = self.chunks[idx] * key_idx
            Stop = Start + self.chunks[idx]
            Stop = Stop if Stop < self.shape[idx] else self.shape[idx]
            index.append((Start, Stop))

        return index

    def _fromfile(self, index):
        print(index)
        array = self.ims[
                self.ResolutionLevelLock,
                index[0][0]:index[0][1],
                index[1][0]:index[1][1],
                index[2][0]:index[2][1],
                index[3][0]:index[3][1],
                index[4][0]:index[4][1]
                ]
        print(array.shape)
        if array.shape == self.chunks:
            print(True)
            return array
        else:
            canvas = np.zeros(self.chunks, dtype=array.dtype)
            canvas[
            0:array.shape[0],
            0:array.shape[1],
            0:array.shape[2],
            0:array.shape[3],
            0:array.shape[4]
            ] = array
            return canvas

    def _get_zarray(self):

        if self.dtype == 'uint16':
            dtype = "<u2"
        elif self.dtype == 'uint8':
            dtype = "|u1"
        elif self.dtype == 'float32':
            dtype = "<f4"
        elif self.dtype == float:
            dtype = "<f8"

        zarray = {
            "chunks": [
                *self.chunks
            ],
            "compressor": None,
            "dtype": dtype,
            "fill_value": 0.0,
            "filters": None,
            "order": "C",
            "shape": [
                *self.shape
            ],
            "zarr_format": 2
        }
        return json.dumps(zarray, indent=2).encode('utf-8')

    def _tofile(self, key, data, file):
        """ Write data to a file
        """
        pass

    def _dset_from_dirStoreFilePath(self, key):
        '''
        filepath will include self.path + key ('0.1.2.3.4')
        Chunks will be sharded along the axis[-3] if the length is >= 3
        Otherwise chunks are sharded along axis 0.
        Key stored in the h5 file is the full key for each chunk ('0.1.2.3.4')
        '''

        _, key = os.path.split(key)

        key = self._normalize_key(key)

        if key in self._files:
            if key == '.zarray':
                return '.zarray'
            else:
                return None
        else:
            return key

    def __getitem__(self, key):

        if self.verbose:
            print('GET : {}'.format(key))

        dset = self._dset_from_dirStoreFilePath(key)
        # print(file)
        # print(dset)

        try:
            if dset is None:
                raise KeyError(key)
            if dset == '.zarray':
                return self._get_zarray()
            else:
                index = self._get_pixel_index_from_key(dset)
                return self._fromfile(index)
        except:
            raise KeyError(key)

    def __setitem__(self, key, value):

        # key = self._normalize_key(key)

        if self.verbose:
            print('SET : {}'.format(key))
            # print('SET VALUE : {}'.format(value))

        pass

    def __delitem__(self, key):

        '''
        Does not yet handle situation where directorystore path is provided
        as the key.
        '''

        if self.verbose == 2:
            print('__delitem__')
            print('DEL : {}'.format(key))

        pass

    def __contains__(self, key):

        if self.verbose == 2:
            print('__contains__')
            print('CON : {}'.format(key))

        dset = self._dset_from_dirStoreFilePath(key)
        # print(file)
        # print(dset)

        if dset == '.zarray':
            return True

        if self.verbose == 2:
            print('Store does not contain {}'.format(key))

        if dset is None:
            return False

        return True

    def __enter__(self):
        return self

    def keys(self):
        if self.verbose == 2:
            print('keys')
        if os.path.exists(self.path):
            yield from self._keys_fast()

    def _keys_fast(self):
        '''
        This will inspect each h5 file and yield keys in the form of paths.

        The paths must be translated into h5_file, key using the function:
            self._dset_from_dirStoreFilePath

        Only returns relative paths to store
        '''
        if self.verbose == 2:
            print('_keys_fast')
        yield '.zarray'
        chunk_num = []
        for idx in range(5):
            tmp = self.shape[idx] // self.chunks[idx]
            tmp = tmp if self.shape[idx] % self.chunks[idx] == 0 else tmp + 1
            chunk_num.append(tmp)

        for t, c, z, y, x in itertools.product(
                range(chunk_num[0]),
                range(chunk_num[1]),
                range(chunk_num[2]),
                range(chunk_num[3]),
                range(chunk_num[4])
        ):
            yield '{}.{}.{}.{}.{}'.format(t, c, z, y, x)

    def __iter__(self):
        if self.verbose == 2:
            print('__iter__')
        return self.keys()

    def __len__(self):
        if self.verbose == 2:
            print('__len__')
        return len(self.keys())


class VirtualMicroscopyGrid:
    def __init__(
        self,
        mesospim_acquisition_dir: Path,
        alignment_directory: Path,
        bigtiff_files: List[str],
        blending: str = 'max'  # options: 'sum', 'average', 'max'
    ):
        self.path = Path(mesospim_acquisition_dir)
        self.parent = self.path.parent
        self.alignment_directory = Path(alignment_directory)

        self.metadata_by_ch = collect_all_metadata(self.path)
        self._sample = get_first_entry(self.metadata_by_ch)

        self.bigtiff_files = bigtiff_files

        self.offsets = get_offsets(self.alignment_directory, self.metadata_by_ch)
        # self.alignment_offsets_microns = alignment_offsets_microns

        self.pixel_size_microns = PixelSizeUM(
            z=self._sample.get('resolution').z,
            y=self._sample.get('resolution').y,
            x=self._sample.get('resolution').x
            )

        self.offsets_to_px()
        self.bounding_box = self._calculate_bounding_box()
        self.blending = blending

        self.tile_infos = {}
        self.virtual_arrays = []
        self.full_virtual = None

    def offsets_to_px(self):
        self.offsets = list(self.offsets)
        self.offsets[0] = self.offsets[0] / self.pixel_size_microns.x
        self.offsets[1] = self.offsets[1] / self.pixel_size_microns.x
        self.offsets[2] = self.offsets[2] / self.pixel_size_microns.y
        self.offsets[3] = self.offsets[3] / self.pixel_size_microns.y
        self.offsets[4] = self.offsets[4] / self.pixel_size_microns.z
        self.offsets[5] = self.offsets[5] / self.pixel_size_microns.z

        xmin = abs(self.offsets[0].min())
        ymin = abs(self.offsets[2].min())
        zmin = abs(self.offsets[4].min())

        self.offsets[0] += xmin
        self.offsets[1] += xmin
        self.offsets[2] += ymin
        self.offsets[3] += ymin
        self.offsets[4] += zmin
        self.offsets[5] += zmin

        self.offsets[0] = np.floor(self.offsets[0])
        self.offsets[1] = np.floor(self.offsets[1])
        self.offsets[2] = np.floor(self.offsets[2])
        self.offsets[3] = np.floor(self.offsets[3])
        self.offsets[4] = np.floor(self.offsets[4])
        self.offsets[5] = np.floor(self.offsets[5])

    def _calculate_bounding_box(self):
        min_x = int(self.offsets[0].min())
        max_x = int(self.offsets[1].max())
        min_y = int(self.offsets[2].min())
        max_y = int(self.offsets[3].max())
        min_z = int(self.offsets[4].min())
        max_z = int(self.offsets[5].max())
        self.bounding_box = (min_z, max_z, min_y, max_y, min_x, max_x)

        return (min_z, max_z, min_y, max_y, min_x, max_x)
    def _build_virtual_array(self):
        # bounding_box = self._calculate_bounding_box()
        print(f'{self.bounding_box}')
        min_z, max_z, min_y, max_y, min_x, max_x = self.bounding_box
        canvas_shape = (max_z - min_z, max_y - min_y, max_x - min_x)
        print(f'{canvas_shape=}')

        # Find the max channel index
        max_channel = len(self.metadata_by_ch)-1

        channel_arrays = [[] for _ in range(max_channel + 1)]
        print(f'Channel Arrays Len: {len(channel_arrays)}')

        channel_idx = -1
        for ch in self.metadata_by_ch:
            channel_idx += 1
            for info in self.metadata_by_ch[ch]:
                yloc = info.get('grid_location').y
                xloc = info.get('grid_location').x

                x_offset = self.offsets[0][xloc, yloc]
                y_offset = self.offsets[2][xloc, yloc]
                z_offset = self.offsets[4][xloc, yloc]

                shape = (info['tile_shape'].z, info['tile_shape'].y, info['tile_shape'].x)
                print(f'{ch=}')

                # @delayed
                # def read_bigtiff(file=info['file_path']):
                #     with tifffile.TiffFile(file) as tif:
                #         stack = np.stack([page.asarray() for page in tif.pages])
                #     return stack

                tile_dask = mesospim_btf_helper(info['file_path']).lazy_array

                # tile_dask = da.from_delayed(
                #     read_bigtiff(),
                #     shape=shape,
                #     dtype='uint16'
                # )
                print(f'{shape=}')
                print(f'{tile_dask=}')

                z_pad_before = int(z_offset - min_z)
                y_pad_before = int(y_offset - min_y)
                x_pad_before = int(x_offset - min_x)

                pad_width = (
                    (z_pad_before, canvas_shape[0] - z_pad_before + shape[0]),
                    (y_pad_before, canvas_shape[1] - y_pad_before + shape[1]),
                    (x_pad_before, canvas_shape[2] - x_pad_before + shape[2])
                )
                print(f'{pad_width=}')

                tile_padded = da.pad(tile_dask, pad_width, mode='constant', constant_values=0)
                # tile_padded = tile_padded.rechunk(shape)
                channel_arrays[channel_idx].append(tile_padded)

        # Combine tiles per channel
        combined_channels = []
        for tiles in channel_arrays:
            if self.blending == 'sum':
                combined = da.sum(da.stack(tiles, axis=0), axis=0)
            elif self.blending == 'average':
                combined = da.mean(da.stack(tiles, axis=0), axis=0)
            elif self.blending == 'max':
                combined = da.max(da.stack(tiles, axis=0), axis=0)
            else:
                raise ValueError(f"Unknown blending mode {self.blending}")

            combined_channels.append(combined)

        # Stack channels into (C, Z, Y, X)
        self.full_virtual = da.stack(combined_channels, axis=0)

    def build(self):
        # self._load_metadata()
        self._build_virtual_array()

    def get_virtual_array(self) -> da.Array:
        if self.full_virtual is None:
            raise RuntimeError("Call build() first!")
        return self.full_virtual

    def save_virtual_zarr(self, path: str):
        if self.full_virtual is None:
            raise RuntimeError("Call build() first!")
        self.full_virtual.to_zarr(path, compute=False)  # Only metadata written


if __name__ == '__main__':
    mesospim_acquisition_dir = '/CBI_FastStore/tmp/mesospim/brain'
    alignment_directory = '/CBI_FastStore/tmp/mesospim/brain/ims_files/align'

    a = mesospim_zarr_store(mesospim_acquisition_dir, alignment_directory, file_type = '.btf', normalize_keys=True, verbose=True)