"""Exercise the bundled Linux writer against both OME-Zarr storage formats."""

import os
import platform
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr


@pytest.mark.skipif(
    sys.platform != 'linux' or platform.machine() != 'x86_64',
    reason='The bundled ImarisWriter runtime targets Linux x86-64',
)
@pytest.mark.parametrize('zarr_format', [2, 3])
def test_bundled_writer_converts_multichannel_omezarr(tmp_path, zarr_format):
    source_path = tmp_path / f'v{zarr_format}.ome.zarr'
    output_path = tmp_path / f'v{zarr_format}.ims'
    data = np.arange(660, dtype=np.uint16).reshape(1, 2, 3, 11, 10)
    root = zarr.open_group(str(source_path), mode='w', zarr_format=zarr_format)
    array_options = {
        'shape': data.shape,
        'dtype': data.dtype,
        'chunks': (1, 1, 2, 8, 8),
    }
    if zarr_format == 3:
        array_options['shards'] = (1, 1, 4, 16, 16)
        root.attrs['ome'] = {
            'version': '0.5',
            'multiscales': [{'version': '0.5', 'datasets': [{'path': '0'}]}],
        }
    else:
        root.attrs['multiscales'] = [{'version': '0.4', 'datasets': [{'path': '0'}]}]
    root.create_array('0', **array_options)[:] = data

    converter = Path(__file__).resolve().parents[1] / 'mesospim_utils' / 'omezarr_to_ims.py'
    clean_env = os.environ.copy()
    for name in ('LD_LIBRARY_PATH', 'MESOSPIM_PYIW_SITE_PACKAGES', 'CONDA_PREFIX', 'PYTHONPATH'):
        clean_env.pop(name, None)
    result = subprocess.run(
        [sys.executable, '-B', str(converter), str(source_path), str(output_path),
         '--voxel-size-zyx-um', '2', '1.2', '1.2',
         '--channel-names', 'green', 'red', '--channel-colors', '0,1,0', '1,0,0',
         '--reader-threads', '2', '--writer-threads', '2', '--max-prefetched-blocks', '4'],
        env=clean_env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert f'DIRECT_OMEZARR_TO_IMS_SUCCESS: {output_path.name}' in result.stdout

    with h5py.File(output_path, 'r') as ims:
        for channel in range(2):
            written = ims[f'DataSet/ResolutionLevel 0/TimePoint 0/Channel {channel}/Data']
            np.testing.assert_array_equal(written[:3, :11, :10], data[0, channel])
