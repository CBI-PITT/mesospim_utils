from __future__ import annotations

from dataclasses import dataclass
from gettext import translation
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
from pathlib import Path
import json

import typer

import zarr
import numpy as np
import tifffile
from zarr.codecs import BloscCodec, BloscShuffle

from ome_zarr_multiscale_writer.write import write_ome_zarr_multiscale
from ome_zarr_multiscale_writer.zarr_reader import OmeZarrArray
from ome_zarr_multiscale_writer.zarr_tools import _ensure_v2_compressor
from ome_zarr_multiscale_writer.zarr_reader import ZarrToOmeZarrConverter
from mesospim_btf import mesospim_btf_helper


# INIT typer cmdline interface
app = typer.Typer()


def validate_ome_zarr_multiscale(path: Path, probe_reads: bool = False) -> bool:
    """
    Validate that a path is a readable multiscale OME-Zarr dataset.

    This is intended as a lightweight completeness check before downstream
    processing skips a costly regeneration step.
    """
    try:
        path = Path(path)
        multiscale = OmeZarrV2Multiscale(path)

        if multiscale.num_levels <= 0:
            return False

        for level in range(multiscale.num_levels):
            arr = multiscale.open_level_array(level)
            if any(int(dim) <= 0 for dim in arr.shape):
                return False

            if probe_reads:
                probe = tuple(slice(0, 1) for _ in range(arr.ndim))
                _ = arr[probe]

        return True
    except Exception:
        return False


######################################################################################################################
####  OME-ZARR CONVERTER FUNCTIONS TO HANDLE SLURM SUBMISSION  ##################
######################################################################################################################

@app.command()
def convert_mesospim_btf_to_omezarr(
        mesospim_btf_path: str,
        output_omezarr_path: str,
        voxel_size: Optional[Tuple[float, float, float]] = (1,1,1),
        translation: Optional[Tuple[float, float, float]] = (0,0,0),
        ome_version: str="0.4",
        generate_multiscales: bool = True,
        start_chunks: Optional[Tuple[int, int, int]] = (256,256,256),
        end_chunks: Optional[Tuple[int, int, int]] = (256,256,256),
        compressor:str = "zstd",
        compression_level:int = 5,
        max_workers: Optional[int] = 8,

) -> None:
    """
    Convert a mesoSPIM BTF dataset to an OME-Zarr v2 multiscale dataset.

    Parameters
    ----------
    mesospim_btf_path :
        Path to the input mesoSPIM BTF dataset.
    output_omezarr_path :
        Path where the output OME-Zarr v2 dataset will be saved.
    """

    data = mesospim_btf_helper(mesospim_btf_path)

    # Function uses numpy-like object and iterates over the first axis to write multiscale ome-zarr
    write_ome_zarr_multiscale(
        data=data,
        path=output_omezarr_path,
        voxel_size=voxel_size,
        translation=translation,
        ome_version=ome_version,
        generate_multiscales=generate_multiscales,
        start_chunks=start_chunks,
        end_chunks=end_chunks,
        compressor=compressor,
        compression_level=compression_level,
        max_workers=max_workers
    )


def _get_btf_shape_and_dtype(path: Path) -> tuple[tuple[int, ...], np.dtype]:
    with tifffile.TiffFile(path, mode='r') as tif:
        sample_data = tif.series[0].asarray()
        zdim = len(tif.series)
        shape = (zdim,) + tuple(sample_data.shape[1:])
        return shape, sample_data.dtype


def _iter_btf_planes(path: Path):
    with tifffile.TiffFile(path, mode='r') as tif:
        for z_plane in range(len(tif.series)):
            yield tif.series[z_plane].asarray().squeeze()


def _iter_btf_volume_blocks(path: Path, z_block_size: int):
    with tifffile.TiffFile(path, mode='r') as tif:
        zdim = len(tif.series)
        if z_block_size <= 0:
            raise ValueError(f'z_block_size must be positive, got {z_block_size}')

        for z_start in range(0, zdim, z_block_size):
            z_stop = min(z_start + z_block_size, zdim)
            block_planes = [tif.series[z_index].asarray().squeeze() for z_index in range(z_start, z_stop)]
            yield z_start, np.stack(block_planes, axis=0)


def _write_single_level_timeseries_omezarr(
    btf_paths: Sequence[Path],
    output_omezarr_path: Path,
    voxel_size: tuple[float, float, float],
    start_chunks: tuple[int, int, int, int, int],
    compressor: str,
    compression_level: int,
    ome_version: str,
) -> None:
    first_shape, first_dtype = _get_btf_shape_and_dtype(btf_paths[0])
    t_size = len(btf_paths)
    z_size, y_size, x_size = first_shape

    if first_dtype != np.uint16:
        raise TypeError(f'Time-series OME-Zarr export expects uint16 source data, got {first_dtype}')

    output_omezarr_path.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(output_omezarr_path), mode='a', zarr_format=2)

    compressor_obj = None
    if compressor:
        compressor_obj = _ensure_v2_compressor(
            BloscCodec(
                cname=compressor,
                clevel=compression_level,
                shuffle=BloscShuffle.bitshuffle,
            )
        )

    chunks = tuple(int(value) for value in start_chunks)
    if len(chunks) != 5:
        raise ValueError(f'Expected 5D chunk shape for time-series export, got {chunks}')
    chunk_t, chunk_c, chunk_z, chunk_y, chunk_x = chunks
    if chunk_t != 1 or chunk_c != 1:
        raise ValueError(f'Time-series export expects chunks of 1 in t and c, got {chunks}')

    level0 = zarr.create(
        shape=(t_size, 1, z_size, y_size, x_size),
        chunks=chunks,
        dtype='uint16',
        compressor=compressor_obj,
        overwrite=True,
        store=root.store,
        path='0',
        zarr_format=2,
        dimension_separator='/',
    )
    level0.attrs['_ARRAY_DIMENSIONS'] = ['t', 'c', 'z', 'y', 'x']

    # Write chunk-aligned 3D blocks so the on-disk layout exactly matches the
    # chunks the direct IMS exporter will later read from the OME-Zarr.
    for time_index, btf_path in enumerate(btf_paths):
        for z_start, volume_block in _iter_btf_volume_blocks(btf_path, chunk_z):
            z_stop = z_start + volume_block.shape[0]

            for y_start in range(0, y_size, chunk_y):
                y_stop = min(y_start + chunk_y, y_size)
                for x_start in range(0, x_size, chunk_x):
                    x_stop = min(x_start + chunk_x, x_size)
                    level0[
                        time_index,
                        0,
                        z_start:z_stop,
                        y_start:y_stop,
                        x_start:x_stop,
                    ] = volume_block[:, y_start:y_stop, x_start:x_stop]

    converter = ZarrToOmeZarrConverter(str(output_omezarr_path), array_path='0', mode='r+')
    converter.convert(
        axes=[
            {'name': 't', 'type': 'time'},
            {'name': 'c', 'type': 'channel'},
            {'name': 'z', 'type': 'space', 'unit': 'micrometer'},
            {'name': 'y', 'type': 'space', 'unit': 'micrometer'},
            {'name': 'x', 'type': 'space', 'unit': 'micrometer'},
        ],
        voxel_size=voxel_size,
        ome_version=ome_version,
    )
@app.command()
def convert_mesospim_btf_timeseries_to_omezarr(
    input_manifest_json: Path,
    output_omezarr_path: Path,
    voxel_size: Optional[Tuple[float, float, float]] = (1, 1, 1),
    ome_version: str = '0.4',
    generate_multiscales: bool = True,
    start_chunks: Optional[Tuple[int, int, int, int, int]] = (1, 1, 64, 256, 256),
    end_chunks: Optional[Tuple[int, int, int, int, int]] = (1, 1, 64, 256, 256),
    compressor: str = 'zstd',
    compression_level: int = 5,
    max_workers: Optional[int] = 8,
) -> None:
    input_manifest_json = Path(input_manifest_json)
    output_omezarr_path = Path(output_omezarr_path)

    with input_manifest_json.open('r') as handle:
        manifest = json.load(handle)

    btf_paths = [Path(path) for path in manifest.get('btf_paths', [])]
    if not btf_paths:
        raise ValueError(f'No btf_paths were found in manifest {input_manifest_json}')

    first_shape, first_dtype = _get_btf_shape_and_dtype(btf_paths[0])
    if len(first_shape) != 3:
        raise ValueError(f'Expected 3D BTF stack at {btf_paths[0]}, got shape {first_shape}')

    for btf_path in btf_paths[1:]:
        shape, dtype = _get_btf_shape_and_dtype(btf_path)
        if shape != first_shape:
            raise ValueError(
                f'All timepoints must have the same shape. '
                f'Expected {first_shape}, got {shape} for {btf_path}'
            )
        if dtype != first_dtype:
            raise ValueError(
                f'All timepoints must have the same dtype. '
                f'Expected {first_dtype}, got {dtype} for {btf_path}'
            )

    _write_single_level_timeseries_omezarr(
        btf_paths=btf_paths,
        output_omezarr_path=output_omezarr_path,
        voxel_size=tuple(float(value) for value in voxel_size),
        start_chunks=tuple(int(value) for value in start_chunks),
        compressor=compressor,
        compression_level=compression_level,
        ome_version=ome_version,
    )




StoreLike = Union[str, MutableMapping[str, bytes]]
@dataclass
class VirtualChunkedArray:
    """
    Lightweight wrapper around a Zarr array that exposes *logical* chunking
    different from the underlying on-disk chunking.

    This is designed to be used with dask.array.from_array.

    Parameters
    ----------
    base_array : Any
        The underlying Zarr array (v2) object.
    logical_chunks : tuple[int, ...]
        Desired chunk shape to expose to Dask. Does *not* need to match the
        underlying Zarr chunking, but for performance it's usually best if each
        logical chunk is a multiple of the physical chunk sizes.
    """

    base_array: Any
    logical_chunks: Tuple[int, ...]

    # --- Array-like protocol for dask.from_array ---

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.base_array.shape)

    @property
    def dtype(self) -> np.dtype:
        return self.base_array.dtype

    @property
    def ndim(self) -> int:
        return self.base_array.ndim

    @property
    def chunks(self) -> Tuple[int, ...]:
        return self.logical_chunks

    def __getitem__(self, key):
        """
        Dask will call this with slice tuples corresponding to *logical*
        chunks. We just forward the slice directly to the underlying Zarr array.
        Zarr will pull and stitch however many physical chunks are needed.
        """
        return self.base_array[key]

    def __array__(self, dtype=None):
        """Allow np.asarray(view) to materialize the entire array if needed."""
        arr = np.asarray(self.base_array)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr


class OmeZarrV2Multiscale:
    """
    Reader for a v2 OME-Zarr multiscale dataset using the zarr v3 library.

    It:
      * Opens a v2 group via zarr.open_group(..., zarr_format=2)
      * Parses the 'multiscales' NGFF metadata
      * Lets you expose any multiscale level to Dask with arbitrary logical
        chunk sizes (e.g. turning (64, 128, 128) into (256, 256, 256)).

    Example
    -------
    >>> ms = OmeZarrV2Multiscale("path/to/data.zarr")
    >>> # get level-0 with logical chunks of (256, 256, 256)
    >>> d0 = ms.to_dask(level=0, logical_chunks=(256, 256, 256))
    >>> d0
    dask.array<from-array, shape=(...), chunksize=(256, 256, 256), dtype=...>
    """

    def __init__(
        self,
        store: StoreLike,
        *,
        group_path: str = "",
        multiscale_index: int = 0,
        open_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Parameters
        ----------
        store :
            Path to the root of the Zarr store (directory / URL) or a mapping-like store.
        group_path :
            Optional internal group path where the OME-Zarr image lives
            (e.g. '' for root, '0' for the first image, etc.).
        multiscale_index :
            Which entry of the 'multiscales' attribute to use (usually 0).
        open_kwargs :
            Extra keyword arguments forwarded to `zarr.open_group`, e.g.
            `storage_options` for fsspec-based stores.
        """
        open_kwargs = dict(open_kwargs or {})

        # Use zarr v3 API but explicitly tell it we're opening a v2 group.
        # See zarr v3 issue about opening v2 stores with zarr_format=2. :contentReference[oaicite:0]{index=0}
        self.root = zarr.open_group(
            store=store,
            mode="r",
            path=group_path or None,
            zarr_format=2,
            **open_kwargs,
        )

        attrs = dict(self.root.attrs)

        if "multiscales" not in attrs:
            raise ValueError("Group has no 'multiscales' attribute; not an OME-Zarr multiscale.")

        self._multiscales = attrs["multiscales"]
        if not self._multiscales:
            raise ValueError("OME-Zarr 'multiscales' attribute is empty.")

        if not (0 <= multiscale_index < len(self._multiscales)):
            raise IndexError(
                f"multiscale_index {multiscale_index} out of range for "
                f"{len(self._multiscales)} multiscales."
            )

        self._multiscale = self._multiscales[multiscale_index]
        self._datasets = self._multiscale.get("datasets", [])
        if not self._datasets:
            raise ValueError("Selected multiscale entry has no 'datasets' list.")

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def multiscales_metadata(self) -> Mapping[str, Any]:
        """Raw metadata for the selected multiscale entry."""
        return self._multiscale

    @property
    def num_levels(self) -> int:
        """Number of resolution levels in the multiscale pyramid."""
        return len(self._datasets)

    def level_paths(self) -> Tuple[str, ...]:
        """Return the Zarr paths for each multiscale level (e.g. ('0', '1', '2', ...))."""
        return tuple(ds["path"] for ds in self._datasets)

    def open_level_array(self, level: int = 0) -> Any:
        """
        Open a specific multiscale level as a Zarr array (v2).

        Parameters
        ----------
        level :
            Multiscale resolution level (0 = highest resolution).

        Returns
        -------
        zarr.core.Array (v2) or similar
        """
        if not (0 <= level < self.num_levels):
            raise IndexError(f"Level {level} out of range for {self.num_levels} levels.")

        path = self._datasets[level]["path"]
        arr = self.root[path]

        # Sanity check: OME-Zarr expects arrays here.
        if not hasattr(arr, "shape") or not hasattr(arr, "dtype"):
            raise TypeError(f"Object at path '{path}' is not a Zarr array.")
        return arr

    def get_level_shape(self, level: int = 0) -> Tuple[int, ...]:
        return tuple(self.open_level_array(level).shape)

    def get_level_scale(self, level: int = 0) -> Tuple[float, ...]:
        dataset = self._datasets[level]
        for transform in dataset.get('coordinateTransformations', []):
            if transform.get('type') == 'scale':
                return tuple(transform.get('scale', []))

        level_array = self.open_level_array(level)
        if hasattr(level_array, 'attrs'):
            for transform in level_array.attrs.get('coordinateTransformations', []):
                if transform.get('type') == 'scale':
                    return tuple(transform.get('scale', []))

        return tuple()

    def get_level_zyx_info(self, level: int = 0) -> dict[str, Tuple[int, ...] | int | Tuple[float, float, float]]:
        shape = self.get_level_shape(level)
        scale = self.get_level_scale(level)

        if len(shape) == 3:
            z_layers = int(shape[0])
            scale_zyx = tuple(scale[:3]) if len(scale) >= 3 else (1.0, 1.0, 1.0)
        elif len(shape) == 5:
            z_layers = int(shape[2])
            scale_zyx = tuple(scale[2:5]) if len(scale) >= 5 else (1.0, 1.0, 1.0)
        else:
            raise ValueError(f'Expected OME-Zarr level shape to be 3D or 5D, got {shape}')

        return {
            'shape': shape,
            'z_layers': z_layers,
            'scale_zyx': scale_zyx,
        }

    # ------------------------------------------------------------------
    # Chunked view + Dask
    # ------------------------------------------------------------------

    def get_level_view(
        self,
        level: int = 0,
        logical_chunks: Optional[Sequence[int]] = None,
    ) -> VirtualChunkedArray:
        """
        Wrap a multiscale level in a VirtualChunkedArray with desired logical chunks.

        Parameters
        ----------
        level :
            Multiscale level to open (0 = highest resolution).
        logical_chunks :
            Desired logical chunk shape, e.g. (256, 256, 256).
            If omitted, the Zarr-array's *physical* chunks are used.

        Returns
        -------
        VirtualChunkedArray
        """
        base = self.open_level_array(level)

        if logical_chunks is None:
            # Use existing Zarr chunks as the logical view.
            logical_chunks = getattr(base, "chunks", None)
            if logical_chunks is None:
                raise ValueError("Underlying array has no 'chunks' attribute; specify logical_chunks explicitly.")
        logical_chunks = tuple(int(c) for c in logical_chunks)

        # Optional: quick safety check that logical chunks are compatible with array shape
        if len(logical_chunks) != base.ndim:
            raise ValueError(
                f"logical_chunks ndim mismatch: got {len(logical_chunks)}, "
                f"but array has ndim={base.ndim}"
            )

        return VirtualChunkedArray(base_array=base, logical_chunks=logical_chunks)

    def to_dask(
        self,
        level: int = 0,
        logical_chunks: Optional[Sequence[int]] = None,
        **from_array_kwargs: Any,
    ) -> da.Array:
        """
        Expose a multiscale level as a Dask array with configurable logical chunks.

        Parameters
        ----------
        level :
            Multiscale level (0 = highest resolution).
        logical_chunks :
            Logical chunk shape to expose to Dask, e.g. (256, 256, 256).
            If None, uses the underlying Zarr chunks.
        **from_array_kwargs :
            Extra kwargs forwarded to dask.array.from_array (e.g. 'name', 'meta').

        Returns
        -------
        dask.array.Array
        """
        view = self.get_level_view(level=level, logical_chunks=logical_chunks)
        return da.from_array(view, chunks=view.chunks, **from_array_kwargs)

@app.command()
def extract_tiff_series(ome_zarr_directory: Path, output_directory: Path, prefix: str=None) -> None:
    if not prefix:
        prefix = ome_zarr_directory.name[:-9] # Strip .ome.zarr
    ome_zarr = OmeZarrArray(ome_zarr_directory)
    ome_zarr.to_tiff_stack(output_directory, basename=prefix)
    return

@app.command()
def extract_single_tiff_plane(
    ome_zarr_directory: Path,
    output_directory: Path,
    resolution_level: int = 0,
    channel: int = 0,
    z: int = 0,
    prefix: str = None,
) -> None:
    ome_zarr_directory = Path(ome_zarr_directory)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    if not prefix:
        prefix = ome_zarr_directory.name[:-9] # Strip .ome.zarr

    ome_zarr = OmeZarrV2Multiscale(ome_zarr_directory)
    level_array = ome_zarr.open_level_array(resolution_level)

    if len(level_array.shape) != 5:
        raise ValueError(
            f"Expected a 5D OME-Zarr array shaped (t, c, z, y, x), got {level_array.shape}"
        )

    plane = da.array(level_array)[0, channel, z, :, :].compute()
    out_file = output_directory / (
        f'{prefix}_r{resolution_level:02d}_t00_c{channel:02d}_z{z:04d}.tif'
    )
    tifffile.imwrite(out_file, plane)
    return


@app.command()
def extract_tiff_plane_batch(
    ome_zarr_directory: Path,
    output_directory: Path,
    resolution_level: int = 0,
    channel: int = 0,
    start_z: int = 0,
    batch_size: int = 10,
    prefix: str = None,
) -> None:
    ome_zarr_directory = Path(ome_zarr_directory)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    if not prefix:
        prefix = ome_zarr_directory.name[:-9] # Strip .ome.zarr

    ome_zarr = OmeZarrV2Multiscale(ome_zarr_directory)
    level_array = ome_zarr.open_level_array(resolution_level)

    if len(level_array.shape) != 5:
        raise ValueError(
            f"Expected a 5D OME-Zarr array shaped (t, c, z, y, x), got {level_array.shape}"
        )

    z_layers = level_array.shape[2]
    if start_z < 0 or start_z >= z_layers:
        raise ValueError(f"start_z {start_z} is out of range for {z_layers} z-layers")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    stop_z = min(start_z + batch_size, z_layers)
    planes = da.array(level_array)[0, channel, start_z:stop_z, :, :].compute()

    for z_offset, plane in enumerate(planes):
        z = start_z + z_offset
        out_file = output_directory / (
            f'{prefix}_r{resolution_level:02d}_t00_c{channel:02d}_z{z:04d}.tif'
        )
        if out_file.exists() and out_file.stat().st_size > 0:
            print(f'OMEZARR_TO_TIFF_SUCCESS: {output_directory.name} r{resolution_level:02d} c{channel:02d} z{z:04d}')
            continue

        tifffile.imwrite(out_file, plane)
        print(f'OMEZARR_TO_TIFF_SUCCESS: {output_directory.name} r{resolution_level:02d} c{channel:02d} z{z:04d}')

    return

@app.command()
def test_func():
    print('Test function in omezarr.py')

if __name__ == "__main__":
    app()
    # path = r"Z:\test_data\mesospim\omezarr\embryo-ome-zarr\Mag8x_Ch488_Ch561_Ch640_montage.ome.zarr"
    #
    # ms = OmeZarrV2Multiscale(path)
    #
    # # Represent it to Dask as (256, 256, 256)-chunked:
    # # d0 = ms.to_dask(level=0, logical_chunks=(512, 1024, 1024))
    # print(d0.chunks)
    # # -> ((256, 256, ...), (256, 256, ...), (256, 256, ...))
