from __future__ import annotations

from dataclasses import dataclass
from gettext import translation
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import typer

import zarr
import dask.array as da
import numpy as np

from ome_zarr_multiscale_writer.write import write_ome_zarr_multiscale
from mesospim_btf import mesospim_btf_helper


# INIT typer cmdline interface
app = typer.Typer()


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