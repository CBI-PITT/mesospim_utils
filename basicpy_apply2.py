#!/h20/home/lab/miniconda3/envs/basicpy-cuda/bin/python

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import zarr
from basicpy import BaSiC


TILE_RE = re.compile(
    r"^(?P<mag>[^_]+)_"
    r"Tile(?P<tile>\d+)_"
    r"(?P<channel>[^_]+)_"
    r"(?P<filter>[^_]+)_"
    r"Sh(?P<sh>[01])_"
    r"Rot(?P<rot>[-+]?\d+(?:\.\d+)?)"
    r"\.ome\.zarr$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BaSiCPy correction on MesoSPIM tiled OME-Zarr data."
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Input folder containing tile .ome.zarr directories.",
    )

    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Output folder for corrected tile .ome.zarr directories.",
    )

    parser.add_argument(
        "--rows",
        required=True,
        type=int,
        help="Number of tile rows.",
    )

    parser.add_argument(
        "--cols",
        required=True,
        type=int,
        help="Number of tile columns.",
    )

    parser.add_argument(
        "--fit-tile",
        type=int,
        default=None,
        help=(
            "Tile index to fit BaSiCPy on. "
            "Default: center-ish tile, computed as (rows * cols - 1) // 2."
        ),
    )

    parser.add_argument(
        "--levels",
        nargs="*",
        default=None,
        help=(
            "Resolution levels to process, e.g. --levels 5 4 3. "
            "Default: all numeric levels found in the fit tile, processed coarse-to-fine."
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="BaSiCPy device. Default: cuda.",
    )

    parser.add_argument(
        "--fitting-mode",
        default="approximate",
        choices=["approximate", "ladmap"],
        help="BaSiCPy fitting mode. Default: approximate.",
    )

    parser.add_argument(
        "--chunk-z",
        type=int,
        default=8,
        help="Output chunk size in Z. Default: 8.",
    )

    parser.add_argument(
        "--chunk-y",
        type=int,
        default=512,
        help="Output chunk size in Y. Default: 512.",
    )

    parser.add_argument(
        "--chunk-x",
        type=int,
        default=512,
        help="Output chunk size in X. Default: 512.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output levels.",
    )

    return parser.parse_args()


def copy_attrs(src, dst):
    dst.attrs.clear()
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def safe_chunks(shape, chunk_z=8, chunk_y=512, chunk_x=512):
    """
    Make chunks small enough for Blosc.

    Assumes array shape is ZYX.
    """
    z, y, x = shape
    return (
        min(chunk_z, z),
        min(chunk_y, y),
        min(chunk_x, x),
    )


def discover_tiles(src_base):
    """
    Discover MesoSPIM tile OME-Zarr folders.

    Returns:
        records_by_combo:
            {
                (channel, filter): {
                    tile_index: {
                        "path": Path,
                        "name": str,
                        "mag": str,
                        "channel": str,
                        "filter": str,
                        "sh": int,
                        "rot": str,
                    }
                }
            }
    """
    records_by_combo = defaultdict(dict)

    for path in sorted(src_base.glob("*.ome.zarr")):
        match = TILE_RE.match(path.name)
        if match is None:
            print(f"Skipping unrecognized folder name: {path.name}")
            continue

        info = match.groupdict()
        tile = int(info["tile"])
        channel = info["channel"]
        filt = info["filter"]
        combo = (channel, filt)

        if tile in records_by_combo[combo]:
            raise RuntimeError(
                f"Duplicate tile {tile} for channel/filter {combo}: "
                f"{records_by_combo[combo][tile]['path']} and {path}"
            )

        records_by_combo[combo][tile] = {
            "path": path,
            "name": path.name,
            "mag": info["mag"],
            "tile": tile,
            "channel": channel,
            "filter": filt,
            "sh": int(info["sh"]),
            "rot": info["rot"],
        }

    if not records_by_combo:
        raise RuntimeError(f"No matching .ome.zarr tile folders found in {src_base}")

    return records_by_combo


def numeric_levels(root):
    """
    Return numeric OME-Zarr multiscale levels as strings.

    Processes coarse-to-fine by default, e.g. ["5", "4", ..., "0"].
    """
    levels = []

    for key in root.array_keys():
        if str(key).isdigit():
            levels.append(str(key))

    if not levels:
        raise RuntimeError("No numeric multiscale levels found in OME-Zarr group.")

    return sorted(levels, key=lambda x: int(x), reverse=True)


def create_output_array(out_root, level, src_arr, chunks, overwrite):
    if level in out_root:
        if overwrite:
            del out_root[level]
        else:
            raise RuntimeError(
                f"Output level {level} already exists. "
                f"Use --overwrite to replace it."
            )

    kwargs = dict(
        name=level,
        shape=src_arr.shape,
        chunks=chunks,
        dtype=np.uint16,
        overwrite=True,
    )

    # zarr v2 arrays usually have .compressor.
    # Some newer zarr configurations may not expose it the same way.
    compressor = getattr(src_arr, "compressor", None)
    if compressor is not None:
        kwargs["compressor"] = compressor

    out_arr = out_root.create_dataset(**kwargs)
    copy_attrs(src_arr, out_arr)
    return out_arr


def choose_fit_tile(tile_records, requested_fit_tile, default_fit_tile):
    if requested_fit_tile is not None:
        if requested_fit_tile not in tile_records:
            raise RuntimeError(
                f"Requested --fit-tile {requested_fit_tile} is missing for "
                f"{tile_records[next(iter(tile_records))]['channel']} / "
                f"{tile_records[next(iter(tile_records))]['filter']}."
            )
        return requested_fit_tile

    if default_fit_tile in tile_records:
        return default_fit_tile

    available = sorted(tile_records)
    fallback = available[len(available) // 2]

    print(
        f"Default fit tile {default_fit_tile} is missing. "
        f"Using available tile {fallback} instead."
    )

    return fallback


def main():
    args = parse_args()

    src_base = args.input
    out_base = args.output
    out_base.mkdir(parents=True, exist_ok=True)

    expected_n_tiles = args.rows * args.cols
    default_fit_tile = (expected_n_tiles - 1) // 2

    records_by_combo = discover_tiles(src_base)

    print("Discovered channel/filter combinations:")
    for channel, filt in sorted(records_by_combo):
        n_tiles = len(records_by_combo[(channel, filt)])
        print(f"  {channel} / {filt}: {n_tiles} tiles")

    for (channel, filt), tile_records in sorted(records_by_combo.items()):
        print()
        print(f"Channel: {channel}  Filter: {filt}")

        missing_tiles = sorted(set(range(expected_n_tiles)) - set(tile_records))
        extra_tiles = sorted(t for t in tile_records if t >= expected_n_tiles)

        if missing_tiles:
            print(f"  WARNING: missing expected tiles: {missing_tiles}")

        if extra_tiles:
            print(f"  WARNING: found tile indices outside rows*cols: {extra_tiles}")

        fit_tile = choose_fit_tile(
            tile_records=tile_records,
            requested_fit_tile=args.fit_tile,
            default_fit_tile=default_fit_tile,
        )

        fit_record = tile_records[fit_tile]
        fit_path = fit_record["path"]

        fit_root = zarr.open_group(str(fit_path), mode="r")

        if args.levels is None:
            levels = numeric_levels(fit_root)
        else:
            levels = [str(level) for level in args.levels]

        print(f"  Fit tile: {fit_tile}")
        print(f"  Levels: {levels}")

        for level in levels:
            print(f"\tlevel {level}")

            if level not in fit_root:
                raise RuntimeError(f"Level {level} not found in fit tile {fit_path}")

            fit_arr = fit_root[level]

            print("\tloading fit data")
            orig = fit_arr[:].astype(np.float32)
            orig = np.nan_to_num(orig) + 1

            basic = BaSiC(
                fitting_mode=args.fitting_mode,
                device=args.device,
            )

            print("\tfitting BaSiCPy")
            basic.fit(orig)

            for tile in sorted(tile_records):
                record = tile_records[tile]

                print(
                    f"\t\tprocessing tile {tile} "
                    f"Sh{record['sh']} Rot{record['rot']} "
                    f"{record['channel']} {record['filter']}"
                )

                src_path = record["path"]
                out_path = out_base / record["name"]

                src_root = zarr.open_group(str(src_path), mode="r")

                if level not in src_root:
                    print(f"\t\tWARNING: level {level} missing in {src_path.name}; skipping")
                    continue

                src_arr = src_root[level]

                out_root = zarr.open_group(str(out_path), mode="a")

                # Copy root OME-Zarr metadata.
                copy_attrs(src_root, out_root)

                chunks = safe_chunks(
                    src_arr.shape,
                    chunk_z=args.chunk_z,
                    chunk_y=args.chunk_y,
                    chunk_x=args.chunk_x,
                )

                out_arr = create_output_array(
                    out_root=out_root,
                    level=level,
                    src_arr=src_arr,
                    chunks=chunks,
                    overwrite=args.overwrite,
                )

                data = src_arr[:].astype(np.float32)
                data = np.nan_to_num(data) + 1

                corrected_data = basic.transform(data)
                corrected_uint16 = np.clip(corrected_data, 0, 65535).astype(np.uint16)

                out_arr[:] = corrected_uint16

            print("\tdone")

    print()
    print("All done.")


if __name__ == "__main__":
    main()
