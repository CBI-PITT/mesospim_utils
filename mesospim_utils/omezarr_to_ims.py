from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
import argparse
import math
import time

import numpy as np
import zarr

from PyImarisWriter import PyImarisWriter as PW


DEFAULT_CHANNEL_COLORS = [
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.5, 0.0),
    (0.5, 0.0, 1.0),
    (0.0, 0.7, 0.7),
    (1.0, 1.0, 1.0),
]


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


class ProgressCallback(PW.CallbackClass):
    def __init__(self):
        super().__init__()
        self.previous_percent = -1

    def RecordProgress(self, progress, total_bytes_written):
        percent = int(progress * 100)
        if percent > self.previous_percent:
            self.previous_percent = percent
            print(
                f"[writer] {percent:3d}% | {total_bytes_written / 1024**3:.3f} GiB written",
                flush=True,
            )


def parse_channel_color(value: str) -> tuple[float, float, float]:
    parts = value.split(',')
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f'Expected channel color as r,g,b floats, got {value}'
        )

    rgb = tuple(float(part) for part in parts)
    if not all(0.0 <= component <= 1.0 for component in rgb):
        raise argparse.ArgumentTypeError(
            f'Channel color values must be in [0, 1], got {value}'
        )
    return rgb


def read_zarr_block(
    source,
    time_index: int,
    source_channel_index: int,
    output_channel_index: int,
    bx: int,
    by: int,
    bz: int,
    block_x: int,
    block_y: int,
    block_z: int,
    width: int,
    height: int,
    depth: int,
):
    start = time.perf_counter()

    x0 = bx * block_x
    y0 = by * block_y
    z0 = bz * block_z

    x1 = min(x0 + block_x, width)
    y1 = min(y0 + block_y, height)
    z1 = min(z0 + block_z, depth)

    valid_x = x1 - x0
    valid_y = y1 - y0
    valid_z = z1 - z0

    block = np.zeros((block_z, block_y, block_x), dtype=np.uint16)
    block[:valid_z, :valid_y, :valid_x] = source[
        time_index,
        source_channel_index,
        z0:z1,
        y0:y1,
        x0:x1,
    ]

    return (output_channel_index, bx, by, bz), np.ascontiguousarray(block), time.perf_counter() - start


def omezarr_to_ims_multichannel_parallel(
    input_path: str | Path,
    output_path: str | Path,
    voxel_size_zyx_um: tuple[float, float, float],
    channel_names: list[str],
    channel_colors: list[tuple[float, float, float]] | None = None,
    time_index: int = 0,
    level: int = 0,
    reader_threads: int = 16,
    writer_threads: int = 16,
    max_prefetched_blocks: int = 32,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.suffix.lower() != '.ims':
        output_path = output_path.with_suffix('.ims')

    if output_path.exists() and output_path.stat().st_size > 0:
        log(f'Skipping conversion because IMS already exists: {output_path}')
        return

    working_path = output_path.with_suffix('.ims.part')
    if working_path.exists():
        working_path.unlink()

    root = zarr.open_group(str(input_path), mode='r')
    source = root[str(level)]

    if source.ndim != 5:
        raise ValueError(f'Expected TCZYX data with 5 dimensions, got {source.shape}')

    size_t, size_c, depth, height, width = source.shape
    if not 0 <= time_index < size_t:
        raise IndexError(f'time_index={time_index}, but source has T={size_t}')

    channel_names = list(channel_names)
    if not channel_names:
        raise ValueError('At least one channel name must be supplied')
    if len(channel_names) > len(DEFAULT_CHANNEL_COLORS):
        raise ValueError(
            f'A maximum of {len(DEFAULT_CHANNEL_COLORS)} channels is supported; received {len(channel_names)}'
        )
    if len(channel_names) > size_c:
        raise ValueError(
            f'{len(channel_names)} channel names were supplied, but the OME-Zarr contains only {size_c} channels'
        )

    if channel_colors is None:
        channel_colors = DEFAULT_CHANNEL_COLORS[:len(channel_names)]
    else:
        channel_colors = list(channel_colors)
        if len(channel_colors) != len(channel_names):
            raise ValueError('channel_colors must have the same length as channel_names')

    if source.dtype.kind != 'u' or source.dtype.itemsize != 2:
        raise TypeError(f'This converter expects uint16 source data, got {source.dtype}')
    if len(source.chunks) != 5:
        raise ValueError(f'Expected five-dimensional chunks, got {source.chunks}')

    _, chunk_c, block_z, block_y, block_x = source.chunks
    if chunk_c != 1:
        log(
            f'Warning: source C chunk is {chunk_c}. Reading channels separately may decode the same source chunk repeatedly.'
        )

    blocks_x = math.ceil(width / block_x)
    blocks_y = math.ceil(height / block_y)
    blocks_z = math.ceil(depth / block_z)
    blocks_per_channel = blocks_x * blocks_y * blocks_z
    expected_blocks = len(channel_names) * blocks_per_channel

    log(f'Source shape: {source.shape}')
    log(f'Source chunks: {source.chunks}')
    log(f'Source dtype: {source.dtype}')
    log(f'Output channel names: {channel_names}')
    log(f'Writer block: Z={block_z}, Y={block_y}, X={block_x}')

    image_size = PW.ImageSize(x=width, y=height, z=depth, c=len(channel_names), t=1)
    sample_size = PW.ImageSize(x=1, y=1, z=1, c=1, t=1)
    block_size = PW.ImageSize(x=block_x, y=block_y, z=block_z, c=1, t=1)
    dimension_sequence = PW.DimensionSequence('x', 'y', 'z', 'c', 't')

    options = PW.Options()
    options.mNumberOfThreads = writer_threads
    options.mCompressionAlgorithmType = PW.eCompressionAlgorithmGzipLevel2
    options.mEnableLogProgress = True

    working_path.parent.mkdir(parents=True, exist_ok=True)
    callback = ProgressCallback()
    converter = None
    conversion_start = time.perf_counter()

    try:
        converter = PW.ImageConverter(
            'uint16',
            image_size,
            sample_size,
            dimension_sequence,
            block_size,
            str(working_path),
            options,
            'MesoSPIM direct OME-Zarr to IMS converter',
            '1.0',
            callback,
        )

        block_coordinates = [
            (source_channel_index, output_channel_index, bx, by, bz)
            for output_channel_index, source_channel_index in enumerate(range(len(channel_names)))
            for bz in range(blocks_z)
            for by in range(blocks_y)
            for bx in range(blocks_x)
        ]
        coordinate_iterator = iter(block_coordinates)

        copied_blocks = 0
        copied_per_channel = [0 for _ in channel_names]
        total_worker_read_time = 0.0
        total_copy_time = 0.0

        with ThreadPoolExecutor(max_workers=reader_threads, thread_name_prefix='zarr-reader') as executor:
            pending = {}

            def submit_one() -> bool:
                try:
                    source_channel_index, output_channel_index, bx, by, bz = next(coordinate_iterator)
                except StopIteration:
                    return False

                future = executor.submit(
                    read_zarr_block,
                    source,
                    time_index,
                    source_channel_index,
                    output_channel_index,
                    bx,
                    by,
                    bz,
                    block_x,
                    block_y,
                    block_z,
                    width,
                    height,
                    depth,
                )
                pending[future] = (source_channel_index, output_channel_index, bx, by, bz)
                return True

            for _ in range(max_prefetched_blocks):
                if not submit_one():
                    break

            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    submitted = pending.pop(future)
                    try:
                        (output_channel_index, bx, by, bz), block, read_elapsed = future.result()
                    except Exception as error:
                        source_channel_index, output_channel_index, bx, by, bz = submitted
                        raise RuntimeError(
                            'Failed reading block '
                            f'source_channel={source_channel_index}, output_channel={output_channel_index}, '
                            f'bx={bx}, by={by}, bz={bz}'
                        ) from error

                    total_worker_read_time += read_elapsed
                    block_index = PW.ImageSize(x=bx, y=by, z=bz, c=output_channel_index, t=0)
                    if not converter.NeedCopyBlock(block_index):
                        raise RuntimeError(
                            f'Writer rejected block channel={output_channel_index}, bx={bx}, by={by}, bz={bz}'
                        )

                    copy_start = time.perf_counter()
                    converter.CopyBlock(block, block_index)
                    total_copy_time += time.perf_counter() - copy_start

                    copied_blocks += 1
                    copied_per_channel[output_channel_index] += 1
                    if copied_blocks == 1 or copied_blocks % 50 == 0 or copied_blocks == expected_blocks:
                        wall_elapsed = time.perf_counter() - conversion_start
                        channel_progress = ', '.join(
                            f'C{channel_number}={count}/{blocks_per_channel}'
                            for channel_number, count in enumerate(copied_per_channel)
                        )
                        log(
                            f'Copied {copied_blocks}/{expected_blocks} | {channel_progress} | '
                            f'wall={wall_elapsed:.1f}s | '
                            f'mean reader={total_worker_read_time / copied_blocks:.3f}s | '
                            f'mean CopyBlock={total_copy_time / copied_blocks:.3f}s'
                        )

                    del block
                    submit_one()

        if copied_blocks != expected_blocks:
            raise RuntimeError(f'Copied {copied_blocks} blocks, expected {expected_blocks}')

        voxel_z, voxel_y, voxel_x = voxel_size_zyx_um
        image_extents = PW.ImageExtents(0.0, 0.0, 0.0, width * voxel_x, height * voxel_y, depth * voxel_z)
        parameters = PW.Parameters()
        for output_channel_index, channel_name in enumerate(channel_names):
            parameters.set_channel_name(output_channel_index, channel_name)

        color_infos = []
        for red, green, blue in channel_colors:
            color_info = PW.ColorInfo()
            color_info.set_base_color(PW.Color(float(red), float(green), float(blue), 1.0))
            color_infos.append(color_info)

        log('Calling Finish()')
        converter.Finish(image_extents, parameters, [datetime.now()], color_infos, True)
    finally:
        if converter is not None:
            log('Destroying converter')
            converter.Destroy()
            log('Converter destroyed')

    working_path.replace(output_path)
    if not output_path.exists():
        raise RuntimeError(f'Conversion completed but output is missing: {output_path}')

    total_seconds = time.perf_counter() - conversion_start
    log(
        f'Created {output_path} | {output_path.stat().st_size / 1024**3:.3f} GiB | total time={total_seconds:.1f} seconds'
    )
    print(f'DIRECT_OMEZARR_TO_IMS_SUCCESS: {output_path.name}', flush=True)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Convert a multichannel TCZYX OME-Zarr dataset directly to an IMS file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('input_path', type=Path, help='Path to the input OME-Zarr directory.')
    parser.add_argument('output_path', type=Path, help='Path to the output IMS file.')
    parser.add_argument(
        '--voxel-size-zyx-um',
        type=float,
        nargs=3,
        required=True,
        metavar=('Z', 'Y', 'X'),
        help='Voxel size in micrometers in Z Y X order.',
    )
    parser.add_argument(
        '--channel-names',
        nargs='+',
        required=True,
        metavar='NAME',
        help='Channel names in source-channel order.',
    )
    parser.add_argument(
        '--channel-colors',
        nargs='*',
        type=parse_channel_color,
        metavar='R,G,B',
        help='Optional channel colors in source-channel order, one r,g,b triple per channel.',
    )
    parser.add_argument('--time-index', type=int, default=0, help='OME-Zarr time index to export.')
    parser.add_argument('--level', type=int, default=0, help='OME-Zarr multiscale level to read.')
    parser.add_argument('--reader-threads', type=int, default=16, help='Number of parallel Zarr reader threads.')
    parser.add_argument('--writer-threads', type=int, default=16, help='Number of PyImarisWriter threads.')
    parser.add_argument('--max-prefetched-blocks', type=int, default=32, help='Maximum prefetched blocks.')
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if any(value <= 0 for value in args.voxel_size_zyx_um):
        parser.error('All values supplied to --voxel-size-zyx-um must be greater than zero')
    if args.reader_threads < 1 or args.writer_threads < 1 or args.max_prefetched_blocks < 1:
        parser.error('Thread and prefetch arguments must all be positive integers')
    if args.channel_colors and len(args.channel_colors) != len(args.channel_names):
        parser.error('--channel-colors must match the number of --channel-names')

    omezarr_to_ims_multichannel_parallel(
        input_path=args.input_path,
        output_path=args.output_path,
        voxel_size_zyx_um=tuple(args.voxel_size_zyx_um),
        channel_names=args.channel_names,
        channel_colors=args.channel_colors,
        time_index=args.time_index,
        level=args.level,
        reader_threads=args.reader_threads,
        writer_threads=args.writer_threads,
        max_prefetched_blocks=args.max_prefetched_blocks,
    )


if __name__ == '__main__':
    main()
