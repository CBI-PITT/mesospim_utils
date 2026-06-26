from pathlib import Path
import argparse
import re
from collections import defaultdict

import numpy as np
import zarr
from scipy.optimize import least_squares


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
        description="Estimate and apply per-tile gain correction to MesoSPIM OME-Zarr tiles."
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
        help="Output folder for gain-corrected tile .ome.zarr directories.",
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
        "--overlap",
        type=float,
        default=0.10,
        help="Fractional tile overlap used for gain estimation. Default: 0.10.",
    )

    parser.add_argument(
        "--levels",
        nargs="*",
        default=None,
        help=(
            "Resolution levels to process, e.g. --levels 5 4 3. "
            "Default: all numeric levels found in the first available tile, coarse-to-fine."
        ),
    )

    parser.add_argument(
        "--projection-percentile",
        type=float,
        default=50.0,
        help="Percentile projection along Z used for gain estimation. Default: 50.",
    )

    parser.add_argument(
        "--low-percentile",
        type=float,
        default=40.0,
        help="Lower percentile for overlap-mask thresholding. Default: 40.",
    )

    parser.add_argument(
        "--high-percentile",
        type=float,
        default=98.0,
        help="Upper percentile for overlap-mask thresholding. Default: 98.",
    )

    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.10,
        help=(
            "Minimum fraction of valid pixels required in an overlap pair. "
            "Default: 0.10."
        ),
    )

    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.5,
        help="Reject overlap ratios below this value. Default: 0.5.",
    )

    parser.add_argument(
        "--max-ratio",
        type=float,
        default=2.0,
        help="Reject overlap ratios above this value. Default: 2.0.",
    )

    parser.add_argument(
        "--min-gain",
        type=float,
        default=0.75,
        help="Clip final gains below this value. Default: 0.75.",
    )

    parser.add_argument(
        "--max-gain",
        type=float,
        default=1.35,
        help="Clip final gains above this value. Default: 1.35.",
    )

    parser.add_argument(
        "--anchor-tile",
        type=int,
        default=0,
        help="Tile whose log-gain is weakly anchored to zero. Default: 0.",
    )

    parser.add_argument(
        "--anchor-weight",
        type=float,
        default=1000.0,
        help="Weight for the gain anchor constraint. Default: 1000.",
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

    Assumes shape is ZYX.
    """
    z, y, x = shape
    return (
        min(chunk_z, z),
        min(chunk_y, y),
        min(chunk_x, x),
    )


def tile_id(row, col, rows):
    """
    MesoSPIM tile numbering convention used in your original script.

    For rows=4, cols=3:

        col 0: tile 0, 1, 2, 3
        col 1: tile 4, 5, 6, 7
        col 2: tile 8, 9, 10, 11
    """
    return col * rows + row


def discover_tiles(src_base):
    """
    Discover MesoSPIM tile OME-Zarr folders.

    Returns:
        {
            (channel, filter): {
                tile_index: {
                    "path": Path,
                    "name": str,
                    "mag": str,
                    "tile": int,
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

    Default order is coarse-to-fine, e.g. ["5", "4", ..., "0"].
    """
    levels = []

    for key in root.array_keys():
        if str(key).isdigit():
            levels.append(str(key))

    if not levels:
        raise RuntimeError("No numeric multiscale levels found in OME-Zarr group.")

    return sorted(levels, key=lambda x: int(x), reverse=True)


def robust_overlap_ratio(
    a_strip,
    b_strip,
    eps=1e-6,
    low_percentile=40.0,
    high_percentile=98.0,
    min_valid_fraction=0.10,
    min_ratio=0.5,
    max_ratio=2.0,
):
    a = a_strip.astype(np.float32, copy=False)
    b = b_strip.astype(np.float32, copy=False)

    a_low, a_high = np.percentile(a, [low_percentile, high_percentile])
    b_low, b_high = np.percentile(b, [low_percentile, high_percentile])

    mask = (
        (a > a_low)
        & (b > b_low)
        & (a < a_high)
        & (b < b_high)
    )

    npx = int(mask.sum())

    if npx < int(a.size * min_valid_fraction):
        return None, npx

    ratio = np.median(a[mask] / (b[mask] + eps))

    if not np.isfinite(ratio):
        return None, npx

    if ratio < min_ratio or ratio > max_ratio:
        return None, npx

    return float(ratio), npx


def make_residual_function(pairs, n_tiles, anchor_tile, anchor_weight):
    def residual(x):
        res = []

        for a, b, ratio, npx in pairs:
            w = np.sqrt(npx)

            # Measured:
            #     intensity_a / intensity_b = ratio
            #
            # Desired corrected intensities:
            #     gain_a * intensity_a ~= gain_b * intensity_b
            #
            # Therefore:
            #     gain_b / gain_a ~= ratio
            #     log_gain_b - log_gain_a ~= log(ratio)
            res.append(w * ((x[b] - x[a]) - np.log(ratio)))

        # Anchor one tile so the solution is not arbitrary.
        if 0 <= anchor_tile < n_tiles:
            res.append(anchor_weight * x[anchor_tile])
        else:
            res.append(anchor_weight * x[0])

        return np.array(res, dtype=np.float64)

    return residual


def estimate_pairs(
    projs,
    rows,
    cols,
    overlap,
    low_percentile,
    high_percentile,
    min_valid_fraction,
    min_ratio,
    max_ratio,
):
    H, W = projs[0].shape

    oy = int(H * overlap)
    ox = int(W * overlap)

    if oy < 1:
        raise RuntimeError(
            f"Y overlap is {oy} pixels. Increase --overlap or check image height."
        )

    if ox < 1:
        raise RuntimeError(
            f"X overlap is {ox} pixels. Increase --overlap or check image width."
        )

    print("Tile projection size:", H, W)
    print("Overlap pixels Y/X:", oy, ox)

    pairs = []

    print("\n\tVertical overlap ratios:")

    for r in range(rows - 1):
        for c in range(cols):
            a = tile_id(r, c, rows)
            b = tile_id(r + 1, c, rows)

            strip_a = projs[a][-oy:, :]
            strip_b = projs[b][:oy, :]

            ratio, npx = robust_overlap_ratio(
                strip_a,
                strip_b,
                low_percentile=low_percentile,
                high_percentile=high_percentile,
                min_valid_fraction=min_valid_fraction,
                min_ratio=min_ratio,
                max_ratio=max_ratio,
            )

            if ratio is None:
                print(f"\t\tSkipping vertical pair {a}->{b}, valid pixels={npx}")
            else:
                print(
                    f"\t\tVertical pair {a}->{b}: "
                    f"ratio={ratio:.3f}, valid pixels={npx}"
                )
                pairs.append((a, b, ratio, npx))

    print("\n\tHorizontal overlap ratios:")

    for r in range(rows):
        for c in range(cols - 1):
            a = tile_id(r, c, rows)
            b = tile_id(r, c + 1, rows)

            strip_a = projs[a][:, -ox:]
            strip_b = projs[b][:, :ox]

            ratio, npx = robust_overlap_ratio(
                strip_a,
                strip_b,
                low_percentile=low_percentile,
                high_percentile=high_percentile,
                min_valid_fraction=min_valid_fraction,
                min_ratio=min_ratio,
                max_ratio=max_ratio,
            )

            if ratio is None:
                print(f"\t\tSkipping horizontal pair {a}->{b}, valid pixels={npx}")
            else:
                print(
                    f"\t\tHorizontal pair {a}->{b}: "
                    f"ratio={ratio:.3f}, valid pixels={npx}"
                )
                pairs.append((a, b, ratio, npx))

    return pairs


def solve_gains(
    pairs,
    n_tiles,
    anchor_tile=0,
    anchor_weight=1000.0,
    min_gain=0.75,
    max_gain=1.35,
):
    if len(pairs) == 0:
        raise RuntimeError("No valid overlap pairs found.")

    residual = make_residual_function(
        pairs=pairs,
        n_tiles=n_tiles,
        anchor_tile=anchor_tile,
        anchor_weight=anchor_weight,
    )

    x0 = np.zeros(n_tiles, dtype=np.float64)
    sol = least_squares(residual, x0)

    gains = np.exp(sol.x)

    # Normalize global brightness.
    gains = gains / np.median(gains)

    # Prevent one bad overlap from creating extreme changes.
    gains = np.clip(gains, min_gain, max_gain)

    return gains


def create_output_array(out_root, level, src_arr, chunks, dtype, overwrite):
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
        dtype=dtype,
        overwrite=True,
    )

    compressor = getattr(src_arr, "compressor", None)
    if compressor is not None:
        kwargs["compressor"] = compressor

    out_arr = out_root.create_dataset(**kwargs)
    copy_attrs(src_arr, out_arr)

    return out_arr


def apply_gain_to_stack(stack, gain, out_dtype):
    corrected = stack.astype(np.float32) * gain

    if np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        corrected = np.clip(corrected, info.min, info.max).astype(out_dtype)
    else:
        corrected = corrected.astype(out_dtype)

    return corrected


def validate_tile_set(tile_records, rows, cols, channel, filt):
    expected_n_tiles = rows * cols
    expected_tiles = set(range(expected_n_tiles))
    observed_tiles = set(tile_records)

    missing_tiles = sorted(expected_tiles - observed_tiles)
    extra_tiles = sorted(t for t in observed_tiles if t >= expected_n_tiles)

    if missing_tiles:
        raise RuntimeError(
            f"Missing expected tiles for {channel} / {filt}: {missing_tiles}"
        )

    if extra_tiles:
        print(
            f"WARNING: found tile indices outside rows*cols for "
            f"{channel} / {filt}: {extra_tiles}"
        )


def main():
    args = parse_args()

    src_base = args.input
    out_base = args.output
    out_base.mkdir(parents=True, exist_ok=True)

    n_tiles = args.rows * args.cols

    records_by_combo = discover_tiles(src_base)

    print("Discovered channel/filter combinations:")
    for channel, filt in sorted(records_by_combo):
        print(f"  {channel} / {filt}: {len(records_by_combo[(channel, filt)])} tiles")

    for (channel, filt), tile_records in sorted(records_by_combo.items()):
        print()
        print(f"Channel: {channel}  Filter: {filt}")

        validate_tile_set(
            tile_records=tile_records,
            rows=args.rows,
            cols=args.cols,
            channel=channel,
            filt=filt,
        )

        first_tile = tile_records[min(tile_records)]
        first_root = zarr.open_group(str(first_tile["path"]), mode="r")

        if args.levels is None:
            levels = numeric_levels(first_root)
        else:
            levels = [str(level) for level in args.levels]

        print(f"Levels: {levels}")

        for level in levels:
            print()
            print(f"\tlevel {level}")

            # ------------------------------------------------------------
            # 1. Make robust 2D projections for gain estimation
            # ------------------------------------------------------------
            print("\n\tMaking projections...")

            projs = [None] * n_tiles

            for tile in range(n_tiles):
                record = tile_records[tile]
                src_path = record["path"]

                print(
                    f"\t\tprocessing tile {tile} "
                    f"Sh{record['sh']} Rot{record['rot']}"
                )

                src_root = zarr.open_group(str(src_path), mode="r")

                if level not in src_root:
                    raise RuntimeError(f"Level {level} not found in {src_path}")

                src_arr = src_root[level]
                stack = src_arr[:].astype(np.float32)

                proj = np.percentile(
                    stack,
                    args.projection_percentile,
                    axis=0,
                )

                projs[tile] = proj

            # ------------------------------------------------------------
            # 2. Estimate pairwise overlap ratios
            # ------------------------------------------------------------
            pairs = estimate_pairs(
                projs=projs,
                rows=args.rows,
                cols=args.cols,
                overlap=args.overlap,
                low_percentile=args.low_percentile,
                high_percentile=args.high_percentile,
                min_valid_fraction=args.min_valid_fraction,
                min_ratio=args.min_ratio,
                max_ratio=args.max_ratio,
            )

            if len(pairs) == 0:
                raise RuntimeError(
                    f"No valid overlap pairs found for {channel} / {filt}, level {level}."
                )

            # ------------------------------------------------------------
            # 3. Solve tile gains in log-space
            # ------------------------------------------------------------
            gains = solve_gains(
                pairs=pairs,
                n_tiles=n_tiles,
                anchor_tile=args.anchor_tile,
                anchor_weight=args.anchor_weight,
                min_gain=args.min_gain,
                max_gain=args.max_gain,
            )

            print("\n\tEstimated gains:")
            for i, g in enumerate(gains):
                print(f"\t\t{i:02d}  gain={g:.4f}")

            # ------------------------------------------------------------
            # 4. Apply gains to full 3D stacks
            # ------------------------------------------------------------
            print("\n\tApplying gains...")

            for tile in range(n_tiles):
                record = tile_records[tile]

                print(
                    f"\t\tprocessing tile {tile} "
                    f"gain={gains[tile]:.4f} "
                    f"Sh{record['sh']} Rot{record['rot']}"
                )

                src_path = record["path"]
                out_path = out_base / record["name"]

                src_root = zarr.open_group(str(src_path), mode="r")

                if level not in src_root:
                    raise RuntimeError(f"Level {level} not found in {src_path}")

                src_arr = src_root[level]
                src_dtype = src_arr.dtype

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
                    dtype=src_dtype,
                    overwrite=args.overwrite,
                )

                stack = src_arr[:]
                corrected = apply_gain_to_stack(
                    stack=stack,
                    gain=gains[tile],
                    out_dtype=src_dtype,
                )

                out_arr[:] = corrected

            print("\n\tDone.")

    print()
    print("All done.")


if __name__ == "__main__":
    main()
