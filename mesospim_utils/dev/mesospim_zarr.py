

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
    grid_y, grid_x  = metadata_entry.get('grid_size')

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

    a = VirtualMicroscopyGrid(
            mesospim_acquisition_dir,
            alignment_directory,
            bigtiff_files = [],
            blending = 'average'  # options: 'sum', 'average', 'max'
    )