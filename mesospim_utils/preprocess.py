from __future__ import annotations

import argparse
from collections import defaultdict
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import zarr

from constants import LOCATION_BASICPY_ENV, LOCATION_OF_MESOSPIM_UTILS_INSTALL
from metadata import collect_all_metadata, get_first_entry
from utils import ensure_path


TILE_RE = re.compile(
    r"^(?P<mag>[^_]+)_"
    r"Tile(?P<tile>\d+)_"
    r"(?P<channel>[^_]+)"
    r"(?:_(?P<filter>.*?))?_"
    r"Sh(?P<sh>[01])_"
    r"Rot(?P<rot>[-+]?\d+(?:\.\d+)?)"
    r"\.ome\.zarr$"
)

CHANNEL_ONLY_FILTER = '__channel_only__'


def normalize_filter_name(filter_name):
    if filter_name is None:
        return CHANNEL_ONLY_FILTER

    filter_name = str(filter_name).strip()
    if not filter_name:
        return CHANNEL_ONLY_FILTER

    return filter_name


def build_metadata_combo_lookup(metadata_by_channel):
    combo_lookup = {}

    for channel_key, entries in metadata_by_channel.items():
        for entry in entries:
            channel = str(entry.get('channel', channel_key))
            filter_name = normalize_filter_name(entry.get('CFG', {}).get('Filter'))

            file_name = entry.get('file_name')
            if file_name:
                combo_lookup[Path(str(file_name)).name] = (channel, filter_name)

    return combo_lookup


def copy_attrs(src, dst):
    dst.attrs.clear()
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def discover_tiles(src_base: Path, metadata_by_channel=None):
    records_by_combo = defaultdict(dict)
    src_base = ensure_path(src_base)

    if metadata_by_channel is None:
        metadata_by_channel = collect_all_metadata(src_base)

    metadata_combo_lookup = build_metadata_combo_lookup(metadata_by_channel)

    for path in sorted(src_base.glob('*.ome.zarr')):
        match = TILE_RE.match(path.name)
        if match is None:
            print(f'Skipping unrecognized folder name: {path.name}')
            continue

        info = match.groupdict()
        tile = int(info['tile'])
        channel, filt = metadata_combo_lookup.get(
            path.name,
            (str(info['channel']), normalize_filter_name(info.get('filter'))),
        )
        combo = (channel, filt)

        if tile in records_by_combo[combo]:
            raise RuntimeError(
                f'Duplicate tile {tile} for channel/filter {combo}: '
                f'{records_by_combo[combo][tile]["path"]} and {path}'
            )

        records_by_combo[combo][tile] = {
            'path': path,
            'name': path.name,
            'mag': info['mag'],
            'tile': tile,
            'channel': channel,
            'filter': filt,
            'sh': int(info['sh']),
            'rot': info['rot'],
        }

    if not records_by_combo:
        raise RuntimeError(f'No matching .ome.zarr tile folders found in {src_base}')

    return records_by_combo


def discover_channel_filter_combinations_from_metadata(metadata_by_channel):
    combinations = set()

    for channel_key, entries in metadata_by_channel.items():
        for entry in entries:
            channel = str(entry.get('channel', channel_key))
            filter_name = normalize_filter_name(entry.get('CFG', {}).get('Filter'))
            combinations.add((channel, filter_name))

    if not combinations:
        raise RuntimeError('No channel/filter combinations could be derived from metadata.')

    return sorted(combinations)


def choose_fit_tile(tile_records, requested_fit_tile, default_fit_tile, rows, cols):
    if requested_fit_tile is not None:
        if requested_fit_tile not in tile_records:
            raise RuntimeError(
                f'Requested --fit-tile {requested_fit_tile} is missing for '
                f'{tile_records[next(iter(tile_records))]["channel"]} / '
                f'{tile_records[next(iter(tile_records))]["filter"]}.'
            )
        return requested_fit_tile

    compressed_fit_tile = choose_scored_lowres_compressed_tile(
        tile_records,
        rows=rows,
        cols=cols,
    )
    if compressed_fit_tile is not None:
        return compressed_fit_tile

    if default_fit_tile in tile_records:
        print(
            f'Compression is not enabled for the lowest multiscale level. '
            f'Using default fit tile {default_fit_tile}.'
        )
        return default_fit_tile

    available = sorted(tile_records)
    fallback = available[len(available) // 2]

    print(
        f'Compression is not enabled for the lowest multiscale level and '
        f'default fit tile {default_fit_tile} is missing. '
        f'Using available tile {fallback} instead.'
    )

    return fallback


def get_folder_size(folder_path):
    total_size = 0

    try:
        for entry in os.scandir(folder_path):
            if entry.is_file(follow_symlinks=False):
                total_size += entry.stat(follow_symlinks=False).st_size
            elif entry.is_dir(follow_symlinks=False):
                total_size += get_folder_size(entry.path)
    except PermissionError:
        pass

    return total_size


def get_lowest_resolution_dataset_path(tile_path: Path) -> str:
    tile_path = ensure_path(tile_path)
    tile_root = zarr.open_group(str(tile_path), mode='r', zarr_version=2)
    multiscales = tile_root.attrs.get('multiscales')

    if not multiscales:
        raise RuntimeError(f'No multiscales metadata found in tile {tile_path}')

    datasets = multiscales[0].get('datasets', [])
    if len(datasets) == 0:
        raise RuntimeError(f'No multiscale datasets found in tile {tile_path}')

    dataset_path = datasets[-1].get('path')
    if not dataset_path:
        raise RuntimeError(f'Lowest multiscale dataset path is missing in tile {tile_path}')

    return str(dataset_path)


def is_lowest_resolution_dataset_compressed(tile_path: Path, dataset_path: str) -> bool:
    tile_path = ensure_path(tile_path)
    tile_root = zarr.open_group(str(tile_path), mode='r', zarr_version=2)
    dataset = tile_root[dataset_path]
    return getattr(dataset, 'compressor', None) is not None


def center_proximity_score(tile: int, rows: int, cols: int) -> float:
    row = tile % rows
    col = tile // rows

    center_row = (rows - 1) / 2.0
    center_col = (cols - 1) / 2.0
    distance = ((row - center_row) ** 2 + (col - center_col) ** 2) ** 0.5

    max_distance = (center_row ** 2 + center_col ** 2) ** 0.5
    if max_distance == 0:
        return 1.0

    return 1.0 - ((distance / max_distance) ** 2)


def normalize_scores(values_by_tile):
    values = list(values_by_tile.values())
    min_value = min(values)
    max_value = max(values)

    if max_value == min_value:
        return {tile: 1.0 for tile in values_by_tile}

    return {
        tile: (value - min_value) / (max_value - min_value)
        for tile, value in values_by_tile.items()
    }


def choose_scored_lowres_compressed_tile(tile_records, rows: int, cols: int):
    dataset_sizes = {}

    for tile in sorted(tile_records):
        record = tile_records[tile]
        dataset_path = get_lowest_resolution_dataset_path(record['path'])

        if not is_lowest_resolution_dataset_compressed(record['path'], dataset_path):
            return None

        dataset_dir = record['path'] / dataset_path
        dataset_sizes[tile] = get_folder_size(dataset_dir)
        print(f'Lowest-resolution dataset size for tile {tile}: {dataset_sizes[tile]} bytes')

    size_scores = normalize_scores(dataset_sizes)
    center_scores = {
        tile: center_proximity_score(tile, rows=rows, cols=cols)
        for tile in tile_records
    }

    best_tile = None
    best_score = None
    best_center_score = None

    for tile in sorted(tile_records):
        center_score = center_scores[tile]
        size_score = size_scores[tile]
        total_score = (0.6 * center_score) + (0.4 * size_score)
        print(
            f'Fit-tile score for tile {tile}: '
            f'center={center_score:.3f} size={size_score:.3f} total={total_score:.3f}'
        )

        if (
            best_score is None
            or total_score > best_score
            or (
                total_score == best_score
                and center_score > best_center_score
            )
        ):
            best_tile = tile
            best_score = total_score
            best_center_score = center_score

    if best_tile is not None:
        print(
            f'Using tile {best_tile} for BaSiCPy fit based on weighted '
            f'center/size score ({best_score:.3f}; weights center=0.6 size=0.4).'
        )

    return best_tile


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
            weight = np.sqrt(npx)
            res.append(weight * ((x[b] - x[a]) - np.log(ratio)))

        if 0 <= anchor_tile < n_tiles:
            res.append(anchor_weight * x[anchor_tile])
        else:
            res.append(anchor_weight * x[0])

        return np.array(res, dtype=np.float64)

    return residual


def tile_id(row, col, rows):
    return col * rows + row


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
    height, width = projs[0].shape

    oy = int(height * overlap)
    ox = int(width * overlap)

    if oy < 1:
        raise RuntimeError(
            f'Y overlap is {oy} pixels. Increase overlap or check image height.'
        )

    if ox < 1:
        raise RuntimeError(
            f'X overlap is {ox} pixels. Increase overlap or check image width.'
        )

    print('Tile projection size:', height, width)
    print('Overlap pixels Y/X:', oy, ox)

    pairs = []

    print('\n\tVertical overlap ratios:')
    for row in range(rows - 1):
        for col in range(cols):
            a = tile_id(row, col, rows)
            b = tile_id(row + 1, col, rows)

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
                print(f'\t\tSkipping vertical pair {a}->{b}, valid pixels={npx}')
            else:
                print(f'\t\tVertical pair {a}->{b}: ratio={ratio:.3f}, valid pixels={npx}')
                pairs.append((a, b, ratio, npx))

    print('\n\tHorizontal overlap ratios:')
    for row in range(rows):
        for col in range(cols - 1):
            a = tile_id(row, col, rows)
            b = tile_id(row, col + 1, rows)

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
                print(f'\t\tSkipping horizontal pair {a}->{b}, valid pixels={npx}')
            else:
                print(f'\t\tHorizontal pair {a}->{b}: ratio={ratio:.3f}, valid pixels={npx}')
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
    from scipy.optimize import least_squares

    if len(pairs) == 0:
        raise RuntimeError('No valid overlap pairs found.')

    residual = make_residual_function(
        pairs=pairs,
        n_tiles=n_tiles,
        anchor_tile=anchor_tile,
        anchor_weight=anchor_weight,
    )

    x0 = np.zeros(n_tiles, dtype=np.float64)
    sol = least_squares(residual, x0)

    gains = np.exp(sol.x)
    gains = gains / np.median(gains)
    gains = np.clip(gains, min_gain, max_gain)

    return gains


def validate_tile_set(tile_records, rows, cols, channel, filt):
    expected_n_tiles = rows * cols
    expected_tiles = set(range(expected_n_tiles))
    observed_tiles = set(tile_records)

    missing_tiles = sorted(expected_tiles - observed_tiles)
    extra_tiles = sorted(tile for tile in observed_tiles if tile >= expected_n_tiles)

    if missing_tiles:
        raise RuntimeError(
            f'Missing expected tiles for {channel} / {filt}: {missing_tiles}'
        )

    if extra_tiles:
        print(
            f'WARNING: found tile indices outside rows*cols for '
            f'{channel} / {filt}: {extra_tiles}'
        )


def prepare_output_collection(input_collection: Path, output_collection: Path):
    input_collection = ensure_path(input_collection)
    output_collection = ensure_path(output_collection)
    output_collection.mkdir(parents=True, exist_ok=True)
    zarr.open_group(str(output_collection), mode='a', zarr_version=2)

    for name in ('.zgroup', '.zattrs'):
        src = input_collection / name
        dst = output_collection / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def write_corrected_tile(src_path: Path, out_path: Path, corrected_data: np.ndarray, overwrite: bool):
    from ome_zarr_multiscale_writer.zarr_reader import OmeZarrArray

    src_path = ensure_path(src_path)
    out_path = ensure_path(out_path)

    if overwrite and out_path.exists():
        shutil.rmtree(out_path)

    input_omezarr = OmeZarrArray(src_path)
    out_ome_zarr = input_omezarr.omezarr_like(out_path)

    if str(src_path.parent).endswith('.ome.zarr'):
        zgroup_file = src_path.parent / '.zgroup'
        zattrs_file = src_path.parent / '.zattrs'
        if zgroup_file.is_file() and not (out_path.parent / '.zgroup').is_file():
            shutil.copy2(zgroup_file, out_path.parent / '.zgroup')
        if zattrs_file.is_file() and not (out_path.parent / '.zattrs').is_file():
            shutil.copy2(zattrs_file, out_path.parent / '.zattrs')

    out_ome_zarr.mode = 'a'
    out_ome_zarr[:] = corrected_data
    out_ome_zarr.create_multiscales(async_close=False)


def resolve_grid_and_overlap(input_collection: Path):
    metadata_by_channel = collect_all_metadata(input_collection)
    first_entry = get_first_entry(metadata_by_channel)
    grid_size = first_entry.get('grid_size')
    overlap = first_entry.get('overlap')

    return grid_size.y, grid_size.x, overlap


def middle_tile_col_major(rows: int, cols: int) -> int:
    """
    Return the tile index closest to the center of a rectangular grid.

    Tile numbering is column-major:
        tile_index = col * rows + row

    If multiple tiles are equally close to the grid center,
    return the one with the lowest tile number.
    """

    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must both be >= 1")

    # For even dimensions, the center falls between tiles.
    # Choosing floor gives the lower row/column among equally central options,
    # which also gives the lowest tile number in column-major order.
    center_row = (rows - 1) // 2
    center_col = (cols - 1) // 2

    return center_col * rows + center_row


def process_basicpy_group(
    input_collection: Path,
    output_collection: Path,
    channel: str,
    filter_name: str,
    fit_tile: int | None,
    device: str,
    fitting_mode: str,
    overwrite: bool,
):
    rows, cols, _ = resolve_grid_and_overlap(input_collection)
    expected_n_tiles = rows * cols
    default_fit_tile = middle_tile_col_major(rows, cols)

    records_by_combo = discover_tiles(input_collection)
    combo = (channel, filter_name)
    if combo not in records_by_combo:
        raise RuntimeError(f'Channel/filter combination not found: {channel} / {filter_name}')

    tile_records = records_by_combo[combo]
    validate_tile_set(tile_records, rows, cols, channel, filter_name)

    fit_tile = choose_fit_tile(tile_records, fit_tile, default_fit_tile, rows, cols)
    print("=============== fit tile =============", fit_tile)

    prepare_output_collection(input_collection, output_collection)
    temp_dir = output_collection / '.basicpy_tmp'
    temp_dir.mkdir(parents=True, exist_ok=True)

    for tile in sorted(tile_records):
        record = tile_records[tile]
        temp_output = temp_dir / f'{record["name"]}.npy'
        if temp_output.exists():
            temp_output.unlink()

        cmd = [
            str(LOCATION_BASICPY_ENV),
            '-u',
            str(Path(LOCATION_OF_MESOSPIM_UTILS_INSTALL) / 'basicpy_worker.py'),
            '--input', str(input_collection),
            '--temp-output', str(temp_output),
            '--channel', channel,
            '--filter', filter_name,
            '--fit-tile', str(fit_tile),
            '--target-tile', str(tile),
            '--device', device,
            '--fitting-mode', fitting_mode,
        ]
        subprocess.run(cmd, check=True)

        corrected_uint16 = np.load(temp_output, allow_pickle=False)
        try:
            write_corrected_tile(
                record['path'],
                output_collection / record['name'],
                corrected_uint16,
                overwrite,
            )
        finally:
            if temp_output.exists():
                temp_output.unlink()

    if temp_dir.exists() and not any(temp_dir.iterdir()):
        temp_dir.rmdir()


def apply_gain_to_stack(stack, gain, out_dtype):
    corrected = stack.astype(np.float32) * gain

    if np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        corrected = np.clip(corrected, info.min, info.max).astype(out_dtype)
    else:
        corrected = corrected.astype(out_dtype)

    return corrected


def process_gain_correction_group(
    input_collection: Path,
    output_collection: Path,
    channel: str,
    filter_name: str,
    overlap_override: float | None,
    projection_percentile: float,
    low_percentile: float,
    high_percentile: float,
    min_valid_fraction: float,
    min_ratio: float,
    max_ratio: float,
    min_gain: float,
    max_gain: float,
    anchor_tile: int,
    anchor_weight: float,
    overwrite: bool,
):
    rows, cols, overlap = resolve_grid_and_overlap(input_collection)
    if overlap_override is not None:
        overlap = overlap_override

    n_tiles = rows * cols
    records_by_combo = discover_tiles(input_collection)
    combo = (channel, filter_name)
    if combo not in records_by_combo:
        raise RuntimeError(f'Channel/filter combination not found: {channel} / {filter_name}')

    tile_records = records_by_combo[combo]
    validate_tile_set(tile_records, rows, cols, channel, filter_name)

    print(f'Channel: {channel}  Filter: {filter_name}')
    print('Levels: [\'0\']')

    print('\n\tMaking level 0 projections...')
    projs = [None] * n_tiles
    source_dtypes = {}

    for tile in range(n_tiles):
        record = tile_records[tile]
        print(f'\t\tprocessing tile {tile} Sh{record["sh"]} Rot{record["rot"]}')

        src_root = zarr.open_group(str(record['path']), mode='r')
        src_arr = src_root['0']
        source_dtypes[tile] = src_arr.dtype
        stack = src_arr[:].astype(np.float32)
        proj = np.percentile(stack, projection_percentile, axis=0)
        projs[tile] = proj

    pairs = estimate_pairs(
        projs=projs,
        rows=rows,
        cols=cols,
        overlap=overlap,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        min_valid_fraction=min_valid_fraction,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
    )

    if len(pairs) == 0:
        raise RuntimeError(
            f'No valid overlap pairs found for {channel} / {filter_name}, level 0.'
        )

    gains = solve_gains(
        pairs=pairs,
        n_tiles=n_tiles,
        anchor_tile=anchor_tile,
        anchor_weight=anchor_weight,
        min_gain=min_gain,
        max_gain=max_gain,
    )

    print('\n\tEstimated gains:')
    for idx, gain in enumerate(gains):
        print(f'\t\t{idx:02d}  gain={gain:.4f}')

    print('\n\tApplying gains to level 0 and rebuilding multiscales...')
    prepare_output_collection(input_collection, output_collection)

    for tile in range(n_tiles):
        record = tile_records[tile]
        print(
            f'\t\tprocessing tile {tile} '
            f'gain={gains[tile]:.4f} '
            f'Sh{record["sh"]} Rot{record["rot"]}'
        )

        src_root = zarr.open_group(str(record['path']), mode='r')
        src_arr = src_root['0']
        stack = src_arr[:]
        corrected = apply_gain_to_stack(stack, gains[tile], source_dtypes[tile])

        write_corrected_tile(
            record['path'],
            output_collection / record['name'],
            corrected,
            overwrite,
        )


def basicpy_apply(args):
    process_basicpy_group(
        ensure_path(args.input),
        ensure_path(args.output),
        args.channel,
        args.filter_name,
        args.fit_tile,
        args.device,
        args.fitting_mode,
        args.overwrite,
    )
    print('\nAll done.')


def gain_correction_apply(args):
    process_gain_correction_group(
        ensure_path(args.input),
        ensure_path(args.output),
        args.channel,
        args.filter_name,
        args.overlap,
        args.projection_percentile,
        args.low_percentile,
        args.high_percentile,
        args.min_valid_fraction,
        args.min_ratio,
        args.max_ratio,
        args.min_gain,
        args.max_gain,
        args.anchor_tile,
        args.anchor_weight,
        args.overwrite,
    )
    print('\nAll done.')


def build_parser():
    parser = argparse.ArgumentParser(description='Preprocessing commands for MesoSPIM tile OME-Zarr collections.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    basicpy_parser = subparsers.add_parser('basicpy-apply', help='Run BaSiCPy flat-field correction on a single channel/filter group.')
    basicpy_parser.add_argument('--input', '-i', required=True, type=Path, help='Input folder containing tile .ome.zarr directories.')
    basicpy_parser.add_argument('--output', '-o', required=True, type=Path, help='Output folder for corrected tile .ome.zarr directories.')
    basicpy_parser.add_argument('--channel', required=True, help='Channel name to process.')
    basicpy_parser.add_argument('--filter', dest='filter_name', required=True, help='Filter name to process.')
    basicpy_parser.add_argument('--fit-tile', type=int, default=None, help='Tile index to fit BaSiCPy on. Defaults to a weighted center/size score on the compressed lowest-resolution dataset directory, or the center-ish tile if compression is not used.')
    basicpy_parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], help='BaSiCPy device.')
    basicpy_parser.add_argument('--fitting-mode', default='approximate', choices=['approximate', 'ladmap'], help='BaSiCPy fitting mode.')
    basicpy_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output tile directories.')
    basicpy_parser.set_defaults(func=basicpy_apply)

    gain_parser = subparsers.add_parser('gain-correction-apply', help='Run gain correction on a single channel/filter group.')
    gain_parser.add_argument('--input', '-i', required=True, type=Path, help='Input folder containing tile .ome.zarr directories.')
    gain_parser.add_argument('--output', '-o', required=True, type=Path, help='Output folder for gain-corrected tile .ome.zarr directories.')
    gain_parser.add_argument('--channel', required=True, help='Channel name to process.')
    gain_parser.add_argument('--filter', dest='filter_name', required=True, help='Filter name to process.')
    gain_parser.add_argument('--overlap', type=float, default=None, help='Optional overlap override. Defaults to metadata overlap.')
    gain_parser.add_argument('--projection-percentile', type=float, default=50.0, help='Percentile projection along Z used for gain estimation.')
    gain_parser.add_argument('--low-percentile', type=float, default=40.0, help='Lower percentile for overlap-mask thresholding.')
    gain_parser.add_argument('--high-percentile', type=float, default=98.0, help='Upper percentile for overlap-mask thresholding.')
    gain_parser.add_argument('--min-valid-fraction', type=float, default=0.10, help='Minimum valid overlap fraction.')
    gain_parser.add_argument('--min-ratio', type=float, default=0.5, help='Reject overlap ratios below this value.')
    gain_parser.add_argument('--max-ratio', type=float, default=2.0, help='Reject overlap ratios above this value.')
    gain_parser.add_argument('--min-gain', type=float, default=0.75, help='Clip final gains below this value.')
    gain_parser.add_argument('--max-gain', type=float, default=1.35, help='Clip final gains above this value.')
    gain_parser.add_argument('--anchor-tile', type=int, default=0, help='Tile whose log-gain is weakly anchored to zero.')
    gain_parser.add_argument('--anchor-weight', type=float, default=1000.0, help='Weight for the gain anchor constraint.')
    gain_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output tile directories.')
    gain_parser.set_defaults(func=gain_correction_apply)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
