from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import zarr

from constants import LOCATION_BASICPY_ENV, LOCATION_OF_MESOSPIM_UTILS_INSTALL
from metadata import collect_all_metadata, get_first_entry, get_anchor_tile_entry
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


def get_grid_formula_offset(entry, anchor_entry):
    gy0, gx0 = anchor_entry.get('grid_location')
    z0 = anchor_entry['POSITION']['z_start']
    z_step = anchor_entry['POSITION']['z_stepsize']

    gy, gx = entry.get('grid_location')
    overlap = entry.get('overlap')
    nz, ny, nx = entry.get('tile_shape')

    x_offset = (gx - gx0) * nx * (1 - overlap)
    y_offset = (gy - gy0) * ny * (1 - overlap)
    z_offset = (entry['POSITION']['z_start'] - z0) / z_step

    return z_offset, y_offset, x_offset


def get_tile_offsets_zyx(input_collection: Path, tile_records: dict) -> dict:
    '''Per-tile (z, y, x) offsets in level-0 voxels.

    Primary source is the per-tile annotated `affine_voxel` translation, which
    is the same value written into the BigStitcher XML 'Translation to Regular
    Grid' transform. Falls back to the grid-layout formula per tile if the
    annotated translation is missing.
    '''
    metadata_by_channel = collect_all_metadata(input_collection)
    anchor_entry = get_anchor_tile_entry(metadata_by_channel)

    entries_by_tile_number = {}
    for _, entries in metadata_by_channel.items():
        for entry in entries:
            tile_number = entry.get('tile_number')
            if tile_number is None:
                continue
            entries_by_tile_number.setdefault(int(tile_number), entry)

    offsets = {}
    for tile in tile_records:
        entry = entries_by_tile_number.get(tile)
        if entry is None:
            raise RuntimeError(
                f'No metadata entry found for tile {tile}; '
                f'cannot resolve tile offsets for gain correction.'
            )

        affine_voxel = entry.get('affine_voxel')
        if affine_voxel is not None:
            offsets[tile] = (
                float(affine_voxel[0][-1]),
                float(affine_voxel[1][-1]),
                float(affine_voxel[2][-1]),
            )
        else:
            offsets[tile] = get_grid_formula_offset(entry, anchor_entry)

    return offsets


def generate_pair_specs(rows: int, cols: int) -> list[tuple[int, int, int, str]]:
    '''Neighbor pairs for the tile grid, column-major tile_id(row, col, rows).

    (a, b, array_axis, kind): axis 1 = y-neighbors (vertical), axis 2 = x-neighbors.
    '''
    pair_specs = []

    for row in range(rows - 1):
        for col in range(cols):
            a = tile_id(row, col, rows)
            b = tile_id(row + 1, col, rows)
            pair_specs.append((a, b, 1, 'vertical'))

    for row in range(rows):
        for col in range(cols - 1):
            a = tile_id(row, col, rows)
            b = tile_id(row, col + 1, rows)
            pair_specs.append((a, b, 2, 'horizontal'))

    return pair_specs


def get_tile_level_paths(tile_path: Path) -> list[str]:
    tile_root = zarr.open_group(str(tile_path), mode='r', zarr_version=2)
    multiscales = tile_root.attrs.get('multiscales')

    if not multiscales:
        raise RuntimeError(f'No multiscales metadata found in tile {tile_path}')

    datasets = multiscales[0].get('datasets', [])
    level_paths = [dataset.get('path') for dataset in datasets if dataset.get('path')]

    if not level_paths:
        raise RuntimeError(f'No multiscale datasets found in tile {tile_path}')

    return level_paths


def choose_estimation_level(tile_records: dict, requested_level: str) -> str:
    '''Multiscale level used for gain estimation: the requested level if every
    tile has it, otherwise the highest common non-zero level.'''
    level_sets = [
        set(get_tile_level_paths(tile_records[tile]['path']))
        for tile in sorted(tile_records)
    ]
    common_levels = set.intersection(*level_sets)

    requested = str(requested_level)
    if requested in common_levels:
        return requested

    nonzero = sorted((level for level in common_levels if level != '0'), key=int)
    if nonzero:
        fallback = nonzero[-1]
        print(
            f'Estimation level {requested} is not present in every tile. '
            f'Using the highest common level {fallback} instead.'
        )
        return fallback

    raise RuntimeError(
        f'No usable multiscale level found for gain estimation '
        f'(requested {requested}); available common levels: {sorted(common_levels, key=int)}.'
    )


def load_estimation_level(tile_path: Path, level: str) -> np.ndarray:
    tile_root = zarr.open_group(str(tile_path), mode='r')
    return np.asarray(tile_root[level])


def tile_id(row, col, rows):
    return col * rows + row


def overlap_interval(off_a: float, off_b: float, size: int):
    start = max(off_a, off_b)
    end = min(off_a + size, off_b + size)
    if end - start < 8:
        return None
    return start, end


def local_slice(offset: float, start: float, end: float) -> slice:
    return slice(int(round(start - offset)), int(round(end - offset)))


def estimate_pair_shift(
    mp_a: np.ndarray,
    mp_b: np.ndarray,
    axis: int,
    shift_search: int,
    low_percentile: float,
) -> int:
    '''Integer residual shift of b relative to a along `axis` of the 2D
    max-projections: peak of the masked zero-mean normalized cross-correlation
    (crop-based search). NCC is invariant to the intensity mismatch between
    tiles, unlike SSD, which locks onto the intensity gradient across the
    strip. 0 means the tile offsets are exact.'''
    n = mp_a.shape[axis]
    a = mp_a.astype(np.float32, copy=False)
    b = mp_b.astype(np.float32, copy=False)
    mask0 = (a > np.percentile(a, low_percentile)) & (b > np.percentile(b, low_percentile))
    best_sh, best_v = 0, -np.inf
    for sh in range(-shift_search, shift_search + 1):
        lo = max(0, sh)
        hi = min(n, n + sh)
        if hi - lo < 8:
            continue
        sl_a = [slice(None)] * 2
        sl_b = [slice(None)] * 2
        sl_a[axis] = slice(lo, hi)
        sl_b[axis] = slice(lo - sh, hi - sh)
        m = mask0[tuple(sl_a)]
        if m.sum() < 100:
            continue
        aa = a[tuple(sl_a)][m]
        bb = b[tuple(sl_b)][m]
        aa_z = aa - aa.mean()
        bb_z = bb - bb.mean()
        v = float((aa_z * bb_z).sum() / (np.linalg.norm(aa_z) * np.linalg.norm(bb_z) + 1e-9))
        if v > best_v:
            best_sh, best_v = sh, v
    return best_sh


def load_pair_strips(tile_arrays, offsets, a, b, axis, shift_search, low_percentile):
    '''Overlap strips for tiles a, b with residual-shift compensation.

    axis: 1 (y-neighbors) or 2 (x-neighbors). Returns
    (strip_a, strip_b, sh, pair_start, strip_start) where pair_start is the
    shift-crop start and strip_start the uncropped strip's start, both along
    the pair axis in tile-a local coords, or None if no overlap.
    '''
    interval = overlap_interval(offsets[a][axis], offsets[b][axis], tile_arrays[a].shape[axis])
    if interval is None:
        return None
    start, end = interval
    strip_start = int(round(start - offsets[a][axis]))
    sl_a = [slice(None)] * 3
    sl_b = [slice(None)] * 3
    sl_a[axis] = local_slice(offsets[a][axis], start, end)
    sl_b[axis] = local_slice(offsets[b][axis], start, end)
    strip_a = tile_arrays[a][tuple(sl_a)]
    strip_b = tile_arrays[b][tuple(sl_b)]

    sh = estimate_pair_shift(
        strip_a.max(axis=0),
        strip_b.max(axis=0),
        0 if axis == 1 else 1,
        shift_search,
        low_percentile,
    )
    pair_start = 0
    if sh != 0:
        n = strip_a.shape[axis]
        lo = max(0, sh)
        hi = min(n, n + sh)
        sl_a = [slice(None)] * 3
        sl_b = [slice(None)] * 3
        sl_a[axis] = slice(lo, hi)
        sl_b[axis] = slice(lo - sh, hi - sh)
        strip_a = strip_a[tuple(sl_a)]
        strip_b = strip_b[tuple(sl_b)]
        pair_start = lo
    return strip_a, strip_b, sh, pair_start, strip_start


def solve_log_fields(
    tile_arrays,
    offsets,
    pair_specs,
    cell,
    smooth_lambda,
    shift_search,
    low_percentile,
    high_percentile,
    min_valid_fraction,
    min_ratio,
    max_ratio,
    anchor_tile,
    anchor_weight,
):
    '''Sparse regularized least squares for per-tile (z, y) log-gain fields on a
    global cell grid.

    Data terms: (L_b - L_a) = log(ratio) per (z, y) cell of each pairwise
    overlap, ratios computed by robust masked medians with the tiles' offsets
    mapped into global coords. Smoothness terms weakly penalize differences
    between adjacent cells, and the anchor tile's field is pinned to zero.
    Returns (log_fields dict, (z0, y0, n_zc, n_yc), pair_stats dict).'''
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr

    n_tiles = len(tile_arrays)
    nz, ny, _ = tile_arrays[0].shape

    z_offs = [offsets[i][0] for i in tile_arrays]
    y_offs = [offsets[i][1] for i in tile_arrays]
    z0 = min(z_offs)
    y0 = min(y_offs)
    gz1 = max(o + nz for o in z_offs)
    gy1 = max(o + ny for o in y_offs)
    n_zc = int(np.ceil((gz1 - z0) / cell))
    n_yc = int(np.ceil((gy1 - y0) / cell))
    print(f'\tGlobal field grid: z_cells={n_zc}, y_cells={n_yc} (origin z={z0:.1f}, y={y0:.1f})')

    def cell_idx(tile, zc, yc):
        return tile * n_zc * n_yc + zc * n_yc + yc

    matrix_rows, matrix_cols, vals, rhs = [], [], [], []
    row = 0
    pair_stats = {}
    for a, b, axis, kind in pair_specs:
        pair = load_pair_strips(tile_arrays, offsets, a, b, axis, shift_search, low_percentile)
        if pair is None:
            print(f'\tPair {a}-{b} ({kind}): no overlap')
            continue
        strip_a, strip_b, sh, pair_start, strip_start = pair

        # Vertical pairs: the pair axis is y, so the strip covers only the
        # y-overlap and cell centers must be mapped through the strip's
        # tile-local start plus the shift-crop start. Horizontal pairs: the
        # pair axis is x, so strip y is the full tile extent starting at 0.
        y_pad = (strip_start + pair_start) if axis == 1 else 0

        n_strip_z, n_strip_y = strip_a.shape[0], strip_a.shape[1]
        n_cells_z = int(np.ceil(n_strip_z / cell))
        n_cells_y = int(np.ceil(n_strip_y / cell))

        cell_ratios = []
        for zc in range(n_cells_z):
            for yc in range(n_cells_y):
                zs, ze = zc * cell, min((zc + 1) * cell, n_strip_z)
                ys, ye = yc * cell, min((yc + 1) * cell, n_strip_y)
                ratio, npx = robust_overlap_ratio(
                    strip_a[zs:ze, ys:ye],
                    strip_b[zs:ze, ys:ye],
                    low_percentile=low_percentile,
                    high_percentile=high_percentile,
                    min_valid_fraction=min_valid_fraction,
                    min_ratio=min_ratio,
                    max_ratio=max_ratio,
                )
                if ratio is None:
                    continue
                # Cell center in tile-a local coords -> global cell indices.
                gz_center = (zs + (ze - zs) / 2.0) + offsets[a][0] - z0
                gy_center = (y_pad + ys + (ye - ys) / 2.0) + offsets[a][1] - y0
                gzc = min(n_zc - 1, max(0, int(gz_center / cell)))
                gyc = min(n_yc - 1, max(0, int(gy_center / cell)))
                matrix_rows.append(row); matrix_cols.append(cell_idx(b, gzc, gyc)); vals.append(1.0)
                matrix_rows.append(row); matrix_cols.append(cell_idx(a, gzc, gyc)); vals.append(-1.0)
                rhs.append(np.log(ratio))
                row += 1
                cell_ratios.append(ratio)

        if cell_ratios:
            pair_stats[(a, b)] = {
                'kind': kind,
                'shift': sh,
                'n_cells': len(cell_ratios),
                'median_ratio': float(np.median(cell_ratios)),
            }
            print(
                f'\tPair {a}-{b} ({kind}): shift={sh:+d}px, {len(cell_ratios)} valid cells, '
                f'median ratio={np.median(cell_ratios):.4f}'
            )
        else:
            print(f'\tPair {a}-{b} ({kind}): shift={sh:+d}px, no valid cells')

    n_unknowns = n_tiles * n_zc * n_yc

    # Smoothness terms.
    for tile in range(n_tiles):
        for zc in range(n_zc):
            for yc in range(n_yc - 1):
                matrix_rows.append(row); matrix_cols.append(cell_idx(tile, zc, yc + 1)); vals.append(smooth_lambda)
                matrix_rows.append(row); matrix_cols.append(cell_idx(tile, zc, yc)); vals.append(-smooth_lambda)
                rhs.append(0.0)
                row += 1
        for zc in range(n_zc - 1):
            for yc in range(n_yc):
                matrix_rows.append(row); matrix_cols.append(cell_idx(tile, zc + 1, yc)); vals.append(smooth_lambda)
                matrix_rows.append(row); matrix_cols.append(cell_idx(tile, zc, yc)); vals.append(-smooth_lambda)
                rhs.append(0.0)
                row += 1

    # Anchor: anchor tile's field pinned to zero.
    for zc in range(n_zc):
        for yc in range(n_yc):
            matrix_rows.append(row); matrix_cols.append(cell_idx(anchor_tile, zc, yc)); vals.append(anchor_weight)
            rhs.append(0.0)
            row += 1

    A = coo_matrix((vals, (matrix_rows, matrix_cols)), shape=(row, n_unknowns)).tocsr()
    b_vec = np.array(rhs, dtype=np.float64)
    print(f'\tSolving sparse LS: {A.shape[0]} rows x {n_unknowns} unknowns ...')
    solution = lsqr(A, b_vec, atol=1e-10, btol=1e-10, iter_lim=20000)[0]

    log_fields = {
        tile: solution[cell_idx(tile, 0, 0): cell_idx(tile, 0, 0) + n_zc * n_yc].reshape(n_zc, n_yc)
        for tile in range(n_tiles)
    }
    grid = (z0, y0, n_zc, n_yc)
    return log_fields, grid, pair_stats


def field_gain_map_for_tile(shape, offsets_tile, log_field, grid, cell) -> np.ndarray:
    '''exp of the global log-gain field sampled at a tile's local coords
    (both in estimation-level units).'''
    from scipy.ndimage import map_coordinates

    z0, y0, n_zc, n_yc = grid
    gz = np.clip((np.arange(shape[0]) + offsets_tile[0] - z0) / cell, 0, n_zc - 1)
    gy = np.clip((np.arange(shape[1]) + offsets_tile[1] - y0) / cell, 0, n_yc - 1)
    coords = np.meshgrid(gz, gy, indexing='ij')
    return np.exp(map_coordinates(log_field, coords, order=1, mode='nearest')).astype(np.float32)


def solve_residual_scalars(
    tile_arrays,
    field_gains,
    offsets,
    pair_specs,
    anchor_tile,
    anchor_weight,
    low_percentile,
    high_percentile,
    min_valid_fraction,
    min_ratio,
    max_ratio,
    min_gain,
    max_gain,
):
    '''Stage-2 scalar gains balancing the residual maxip seam ratios of the
    field-corrected tiles. The median-based fields cannot capture the
    signal-dependent mismatch of the brightest maxip pixels, so the residual
    ratios are balanced by a least-squares over the tile graph, anchored and
    normalized at the anchor tile.'''
    from scipy.optimize import least_squares

    n_tiles = len(tile_arrays)
    maxips = {
        tile: (tile_arrays[tile].astype(np.float32) * field_gains[tile][:, :, None]).max(axis=0)
        for tile in tile_arrays
    }

    rho = {}
    for a, b, axis, kind in pair_specs:
        ia = max(offsets[a][axis], offsets[b][axis])
        ib = min(offsets[a][axis] + tile_arrays[a].shape[axis],
                 offsets[b][axis] + tile_arrays[b].shape[axis])
        sa = slice(int(round(ia - offsets[a][axis])), int(round(ib - offsets[a][axis])))
        sb = slice(int(round(ia - offsets[b][axis])), int(round(ib - offsets[b][axis])))
        strip_a = maxips[a][sa, :] if axis == 1 else maxips[a][:, sa]
        strip_b = maxips[b][sb, :] if axis == 1 else maxips[b][:, sb]
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
            print(f'\tResidual pair {a}-{b} ({kind}): skipped, valid pixels={npx}')
            continue
        rho[(a, b)] = ratio
        print(f'\tResidual pair {a}-{b} ({kind}): {ratio:.4f}')

    if not rho:
        raise RuntimeError('No valid residual maxip ratios found for the scalar stage.')

    def residual(x):
        res = [((x[b] - x[a]) - np.log(rho[(a, b)])) for (a, b) in rho]
        res.append(anchor_weight * x[anchor_tile])
        return np.array(res, dtype=np.float64)

    sol = least_squares(residual, np.zeros(n_tiles))
    scalars = np.exp(sol.x)
    scalars = scalars / scalars[anchor_tile]

    return np.clip(scalars, min_gain, max_gain)


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


def tile_output_has_multiscales(tile_path: Path) -> bool:
    zattrs_path = ensure_path(tile_path) / '.zattrs'
    if not zattrs_path.is_file():
        return False

    try:
        zattrs = json.loads(zattrs_path.read_text())
    except (OSError, IOError, json.JSONDecodeError):
        return False

    multiscales = zattrs.get('multiscales')
    if not multiscales:
        return False

    datasets = multiscales[0].get('datasets', [])
    if len(datasets) == 0:
        return False

    for dataset in datasets:
        dataset_path = dataset.get('path')
        if not dataset_path:
            return False
        if not (tile_path / dataset_path).exists():
            return False

    return True


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


def level0_gain_map_for_tile(
    src_root,
    level: str,
    offsets_tile: tuple,
    log_field: np.ndarray,
    grid: tuple,
    cell: int,
    resid_scalar: float,
) -> np.ndarray:
    '''(z, y) gain map for a tile at level 0, from the estimation-level log
    field. The field grid is in estimation-level units; the level-0/level-L
    shape ratio gives the coordinate mapping.'''
    from scipy.ndimage import map_coordinates

    base_shape = src_root['0'].shape
    level_shape = src_root[level].shape
    z_div = base_shape[0] / level_shape[0]
    y_div = base_shape[1] / level_shape[1]

    z0, y0, n_zc, n_yc = grid
    gz = np.clip(((np.arange(base_shape[0]) + offsets_tile[0]) / z_div - z0) / cell, 0, n_zc - 1)
    gy = np.clip(((np.arange(base_shape[1]) + offsets_tile[1]) / y_div - y0) / cell, 0, n_yc - 1)
    coords = np.meshgrid(gz, gy, indexing='ij')

    field = np.exp(map_coordinates(log_field, coords, order=1, mode='nearest'))
    return (field * resid_scalar).astype(np.float32)


def write_corrected_tile_with_gain_map(
    src_path: Path,
    out_path: Path,
    gain_map_zy: np.ndarray,
    overwrite: bool,
):
    '''Write a gain-corrected tile: level 0 is read and written chunk-wise with
    the (z, y) gain map applied, then multiscales are rebuilt from the
    corrected level 0. Mirrors write_corrected_tile but bounds memory.'''
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

    src_root = zarr.open_group(str(src_path), mode='r')
    src_arr = src_root['0']
    shape = src_arr.shape
    cz, cy, cx = src_arr.chunks

    for z in range(0, shape[0], cz):
        for y in range(0, shape[1], cy):
            for x in range(0, shape[2], cx):
                sl = (
                    slice(z, min(z + cz, shape[0])),
                    slice(y, min(y + cy, shape[1])),
                    slice(x, min(x + cx, shape[2])),
                )
                stack = np.asarray(src_arr[sl])
                corrected = apply_gain_to_stack(stack, gain_map_zy[sl[0], sl[1], None], stack.dtype)
                out_ome_zarr[sl] = corrected

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
    group_temp_dir_name = re.sub(r'[^A-Za-z0-9._-]+', '_', f'{channel}_{filter_name}')
    group_temp_dir = temp_dir / group_temp_dir_name
    group_temp_dir.mkdir(parents=True, exist_ok=True)

    for tile in sorted(tile_records):
        record = tile_records[tile]
        out_tile_path = output_collection / record['name']
        if not overwrite and tile_output_has_multiscales(out_tile_path):
            print(f'\tskipping completed tile {tile} {record["name"]}')
            continue

        temp_output = group_temp_dir / f'{record["name"]}.npy'
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
                out_tile_path,
                corrected_uint16,
                overwrite,
            )
        finally:
            if temp_output.exists():
                temp_output.unlink()

    if group_temp_dir.exists() and not any(group_temp_dir.iterdir()):
        group_temp_dir.rmdir()

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


def save_gain_estimation_npz(
    output_collection: Path,
    channel: str,
    filter_name: str,
    level: str,
    cell: int,
    grid: tuple,
    offsets_voxel: dict,
    log_fields: dict,
    residual_scalars: np.ndarray,
    pair_stats: dict,
    tile_names: dict,
) -> Path:
    '''Save the estimated fields/gains for QC, mirroring the reference
    estimated_gains.npz artifacts.'''
    group_key = re.sub(r'[^A-Za-z0-9._-]+', '_', f'{channel}_{filter_name}')
    out_path = output_collection / f'.gains_{group_key}.npz'

    tiles = sorted(tile_names)
    payload = {
        'channel': np.array(channel),
        'filter': np.array(filter_name),
        'level': np.array(level),
        'cell': np.array(cell),
        'grid': np.array(grid, dtype=np.float64),
        'tile_indices': np.array(tiles),
        'tile_names': np.array([tile_names[tile] for tile in tiles]),
        'offsets_zyx_voxel': np.array([offsets_voxel[tile] for tile in tiles], dtype=np.float64),
        'resid_scalar': np.array([residual_scalars[tile] for tile in tiles], dtype=np.float64),
    }
    for tile in tiles:
        payload[f'log_field_t{tile}'] = log_fields[tile]

    pairs = sorted(pair_stats)
    payload['pair_a'] = np.array([pair[0] for pair in pairs])
    payload['pair_b'] = np.array([pair[1] for pair in pairs])
    payload['pair_kind'] = np.array([pair_stats[pair]['kind'] for pair in pairs])
    payload['pair_shift'] = np.array([pair_stats[pair]['shift'] for pair in pairs])
    payload['pair_n_cells'] = np.array([pair_stats[pair]['n_cells'] for pair in pairs])
    payload['pair_median_ratio'] = np.array([pair_stats[pair]['median_ratio'] for pair in pairs])

    np.savez(out_path, **payload)
    print(f'\tSaved gain estimation artifact: {out_path}')
    return out_path


def process_gain_correction_group(
    input_collection: Path,
    output_collection: Path,
    channel: str,
    filter_name: str,
    estimation_level: str,
    cell: int,
    smooth_lambda: float,
    shift_search: int,
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
    '''Two-stage gain correction for one channel/filter group.

    Stage 1 solves per-tile (z, y) log-gain fields by sparse regularized least
    squares over all neighbor pairs, using robust masked median ratios in the
    overlap intervals implied by the tile offsets (with masked-NCC residual
    shift compensation). Stage 2 balances the residual maxip seam ratios of
    the field-corrected tiles with scalar gains. The final gain per tile is
    field x residual scalar, applied to level 0 chunk-wise before multiscales
    are rebuilt.'''
    rows, cols, _ = resolve_grid_and_overlap(input_collection)

    records_by_combo = discover_tiles(input_collection)
    combo = (channel, filter_name)
    if combo not in records_by_combo:
        raise RuntimeError(f'Channel/filter combination not found: {channel} / {filter_name}')

    tile_records = records_by_combo[combo]
    validate_tile_set(tile_records, rows, cols, channel, filter_name)

    print(f'Channel: {channel}  Filter: {filter_name}')
    print('Gain correction: two-stage (log-gain fields + residual scalars)')

    offsets_voxel = get_tile_offsets_zyx(input_collection, tile_records)
    pair_specs = generate_pair_specs(rows, cols)
    level = choose_estimation_level(tile_records, estimation_level)
    print(f'Estimation level: {level}')

    tile_arrays = {}
    for tile in sorted(tile_records):
        record = tile_records[tile]
        print(f'\tLoading tile {tile} Sh{record["sh"]} Rot{record["rot"]}: {record["name"]}')
        tile_arrays[tile] = load_estimation_level(record['path'], level)

    # Offsets are annotated in level-0 voxels; map them into estimation-level
    # units using divisors derived from the level-0/level-L shape ratio.
    src_root = zarr.open_group(str(tile_records[0]['path']), mode='r')
    base_shape = src_root['0'].shape
    level_shape = tile_arrays[0].shape
    z_div = base_shape[0] / level_shape[0]
    y_div = base_shape[1] / level_shape[1]
    x_div = base_shape[2] / level_shape[2]
    print(f'Level-0 to level-{level} divisors (z, y, x): {z_div:.1f} {y_div:.1f} {x_div:.1f}')
    offsets = {
        tile: (o[0] / z_div, o[1] / y_div, o[2] / x_div)
        for tile, o in offsets_voxel.items()
    }

    print('\n\tStage 1: estimating (z, y) log-gain fields ...')
    log_fields, grid, pair_stats = solve_log_fields(
        tile_arrays=tile_arrays,
        offsets=offsets,
        pair_specs=pair_specs,
        cell=cell,
        smooth_lambda=smooth_lambda,
        shift_search=shift_search,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        min_valid_fraction=min_valid_fraction,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        anchor_tile=anchor_tile,
        anchor_weight=anchor_weight,
    )

    field_gains = {
        tile: field_gain_map_for_tile(tile_arrays[tile].shape, offsets[tile], log_fields[tile], grid, cell)
        for tile in tile_records
    }
    for tile in sorted(tile_records):
        g = field_gains[tile]
        print(f'\tTile {tile} field gain: min={g.min():.3f} median={np.median(g):.3f} max={g.max():.3f}')

    print('\n\tStage 2: estimating residual scalar gains ...')
    residual_scalars = solve_residual_scalars(
        tile_arrays=tile_arrays,
        field_gains=field_gains,
        offsets=offsets,
        pair_specs=pair_specs,
        anchor_tile=anchor_tile,
        anchor_weight=anchor_weight,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        min_valid_fraction=min_valid_fraction,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        min_gain=min_gain,
        max_gain=max_gain,
    )

    print('\n\tEstimated gains (field x residual scalar):')
    for tile in sorted(tile_records):
        s = residual_scalars[tile]
        g = field_gains[tile]
        print(
            f'\t\t{tile:02d}  scalar={s:.4f}  '
            f'field min/med/max={g.min():.3f}/{np.median(g):.3f}/{g.max():.3f}'
        )

    print('\n\tApplying gain maps to level 0 and rebuilding multiscales...')
    prepare_output_collection(input_collection, output_collection)

    save_gain_estimation_npz(
        output_collection=output_collection,
        channel=channel,
        filter_name=filter_name,
        level=level,
        cell=cell,
        grid=grid,
        offsets_voxel=offsets_voxel,
        log_fields=log_fields,
        residual_scalars=residual_scalars,
        pair_stats=pair_stats,
        tile_names={tile: tile_records[tile]['name'] for tile in tile_records},
    )

    for tile in sorted(tile_records):
        record = tile_records[tile]
        print(f'\t\tprocessing tile {tile} Sh{record["sh"]} Rot{record["rot"]}')

        src_root = zarr.open_group(str(record['path']), mode='r')
        gain_map = level0_gain_map_for_tile(
            src_root,
            level,
            offsets_voxel[tile],
            log_fields[tile],
            grid,
            cell,
            residual_scalars[tile],
        )
        print(
            f'\t\ttile {tile} level 0 gain map: '
            f'min={gain_map.min():.3f} median={np.median(gain_map):.3f} max={gain_map.max():.3f}'
        )

        write_corrected_tile_with_gain_map(
            record['path'],
            output_collection / record['name'],
            gain_map,
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
        args.estimation_level,
        args.cell,
        args.smooth_lambda,
        args.shift_search,
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

    gain_parser = subparsers.add_parser('gain-correction-apply', help='Run two-stage (log-gain field + residual scalar) gain correction on a single channel/filter group.')
    gain_parser.add_argument('--input', '-i', required=True, type=Path, help='Input folder containing tile .ome.zarr directories.')
    gain_parser.add_argument('--output', '-o', required=True, type=Path, help='Output folder for gain-corrected tile .ome.zarr directories.')
    gain_parser.add_argument('--channel', required=True, help='Channel name to process.')
    gain_parser.add_argument('--filter', dest='filter_name', required=True, help='Filter name to process.')
    gain_parser.add_argument('--estimation-level', default='2', help='Multiscale level used to estimate the gain fields. Defaults to 2, falling back to the highest common level when absent.')
    gain_parser.add_argument('--cell', type=int, default=16, help='(z, y) cell size in estimation-level pixels for the gain field grid.')
    gain_parser.add_argument('--smooth-lambda', type=float, default=0.5, help='Smoothness weight for the log-gain field least squares.')
    gain_parser.add_argument('--shift-search', type=int, default=20, help='Residual shift search range in estimation-level pixels along each pair axis.')
    gain_parser.add_argument('--low-percentile', type=float, default=40.0, help='Lower percentile for overlap-mask thresholding.')
    gain_parser.add_argument('--high-percentile', type=float, default=98.0, help='Upper percentile for overlap-mask thresholding.')
    gain_parser.add_argument('--min-valid-fraction', type=float, default=0.10, help='Minimum valid overlap fraction.')
    gain_parser.add_argument('--min-ratio', type=float, default=0.5, help='Reject overlap ratios below this value.')
    gain_parser.add_argument('--max-ratio', type=float, default=2.0, help='Reject overlap ratios above this value.')
    gain_parser.add_argument('--min-gain', type=float, default=0.75, help='Clip residual scalar gains below this value.')
    gain_parser.add_argument('--max-gain', type=float, default=1.35, help='Clip residual scalar gains above this value.')
    gain_parser.add_argument('--anchor-tile', type=int, default=0, help='Tile whose log-gain field is anchored to zero and whose scalar gain is normalized to one.')
    gain_parser.add_argument('--anchor-weight', type=float, default=1000.0, help='Weight for the gain field anchor constraint.')
    gain_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output tile directories.')
    gain_parser.set_defaults(func=gain_correction_apply)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
