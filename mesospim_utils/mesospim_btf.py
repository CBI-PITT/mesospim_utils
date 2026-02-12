
"""
Helper class to read mesospim btf files as dask arrays
"""
from typing import Union
from pathlib import Path

import tifffile
import numpy as np
from dask import delayed
import dask.array as da

# mesospim_utils imports
try:
    from constants import VERBOSE
except ImportError:
    VERBOSE = 0

class mesospim_btf_helper:

    def __init__(self,path: Union[str, Path]):
        self.path = path if isinstance(path,Path) else Path(path)
        self.tif = tifffile.TiffFile(self.path, mode='r')
        self.sample_data = self.tif.series[0].asarray()
        self.zdim = len(self.tif.series)

        self.build_lazy_array()

    def __getitem__(self, item):
        return self.lazy_array[item].compute()

    def build_lazy_array(self):
        if VERBOSE: print('Building Array')
        delayed_image_reads = [delayed(self.get_z_plane)(x) for x in range(self.zdim)]
        delayed_arrays = [da.from_delayed(x, shape=self.sample_data.shape, dtype=self.sample_data.dtype)[0] for x in delayed_image_reads]
        self.lazy_array = da.stack(delayed_arrays)
        if VERBOSE > 1: print(self.lazy_array)

    def get_z_plane(self,z_plane: int):
        return self.tif.series[z_plane].asarray()

    def __iter__(self):
        for z in range(self.zdim):
            yield self.get_z_plane(z).squeeze()

    @property
    def shape(self):
        return self.lazy_array.shape

    @property
    def chunksize(self):
        return self.lazy_array.chunksize

    @property
    def nbytes(self):
        return self.lazy_array.nbytes

    @property
    def dtype(self):
        return self.lazy_array.dtype

    @property
    def chunks(self):
        return self.lazy_array.chunks

    @property
    def ndim(self):
        return self.lazy_array.ndim

    def __del__(self):
        del self.lazy_array
        self.tif.close()
        del self.tif