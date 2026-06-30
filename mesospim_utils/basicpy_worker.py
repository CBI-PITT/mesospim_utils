import argparse
from collections import defaultdict
from pathlib import Path
import re

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


def discover_tiles(src_base: Path):
    records_by_combo = defaultdict(dict)

    for path in sorted(src_base.glob('*.ome.zarr')):
        match = TILE_RE.match(path.name)
        if match is None:
            continue

        info = match.groupdict()
        tile = int(info['tile'])
        combo = (info['channel'], info['filter'])
        records_by_combo[combo][tile] = {
            'path': path,
            'name': path.name,
            'sh': int(info['sh']),
            'rot': info['rot'],
            'channel': info['channel'],
            'filter': info['filter'],
        }

    return records_by_combo


def build_parser():
    parser = argparse.ArgumentParser(description='Run BaSiCPy on a single channel/filter group and write one temporary corrected tile array.')
    parser.add_argument('--input', '-i', required=True, type=Path, help='Input folder containing tile .ome.zarr directories.')
    parser.add_argument('--temp-output', required=True, type=Path, help='Temporary output .npy path on the destination filesystem.')
    parser.add_argument('--channel', required=True, help='Channel name to process.')
    parser.add_argument('--filter', dest='filter_name', required=True, help='Filter name to process.')
    parser.add_argument('--fit-tile', required=True, type=int, help='Tile index to fit BaSiCPy on.')
    parser.add_argument('--target-tile', required=True, type=int, help='Tile index to correct and write.')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], help='BaSiCPy device.')
    parser.add_argument('--fitting-mode', default='approximate', choices=['approximate', 'ladmap'], help='BaSiCPy fitting mode.')
    return parser


def main():
    args = build_parser().parse_args()

    records_by_combo = discover_tiles(args.input)
    combo = (args.channel, args.filter_name)
    if combo not in records_by_combo:
        raise RuntimeError(f'Channel/filter combination not found: {args.channel} / {args.filter_name}')

    tile_records = records_by_combo[combo]
    if args.fit_tile not in tile_records:
        raise RuntimeError(f'Fit tile {args.fit_tile} not found for {args.channel} / {args.filter_name}')

    if args.target_tile not in tile_records:
        raise RuntimeError(f'Target tile {args.target_tile} not found for {args.channel} / {args.filter_name}')

    args.temp_output.parent.mkdir(parents=True, exist_ok=True)

    fit_path = tile_records[args.fit_tile]['path']
    fit_root = zarr.open_group(str(fit_path), mode='r')
    fit_arr = fit_root['0']

    print(f'Channel: {args.channel}  Filter: {args.filter_name}')
    print(f'  Fit tile: {args.fit_tile}')
    print('\tloading level 0 fit data')

    orig = fit_arr[:].astype(np.float32)
    orig = np.nan_to_num(orig) + 1

    basic = BaSiC(fitting_mode=args.fitting_mode, device=args.device)
    print('\tfitting BaSiCPy on level 0')
    basic.fit(orig)

    record = tile_records[args.target_tile]
    print(
        f'\tprocessing tile {args.target_tile} '
        f'Sh{record["sh"]} Rot{record["rot"]} '
        f'{record["channel"]} {record["filter"]}'
    )

    src_root = zarr.open_group(str(record['path']), mode='r')
    src_arr = src_root['0']
    data = src_arr[:].astype(np.float32)
    data = np.nan_to_num(data) + 1

    corrected_data = basic.transform(data)
    corrected_uint16 = np.clip(corrected_data, 0, 65535).astype(np.uint16)
    np.save(args.temp_output, corrected_uint16, allow_pickle=False)

    print('\nAll done.')


if __name__ == '__main__':
    main()
