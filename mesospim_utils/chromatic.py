import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import median

import numpy as np
import typer
import zarr

from constants import (
    BIGSTITCHER_CHROMATIC_MAX_SHIFT_UM,
    BIGSTITCHER_CHROMATIC_QUALITY_THRESHOLD,
    BIGSTITCHER_CHROMATIC_REFERENCE_CHANNEL,
)
from bigstitcher import does_dir_contain_bigstitcher_metadata, get_ome_zarr_directory_from_xml
from utils import ensure_path

CHROMATIC_TRANSFORM_NAME = 'Chromatic Correction'
# Measurement pyramid level: the finest level whose y extent fits this bound.
# Coarser levels lose the chromatic shift to sub-pixel; finer levels cost RAM.
MEASUREMENT_MAX_Y_PX = 1500
MEASUREMENT_N_PLANES = 48

app = typer.Typer()


def _phase_corr_2d(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    '''Lateral shift of b relative to a by phase correlation.

    Returns (dy, dx, quality) where (dy, dx) is the shift to APPLY to b so its
    content aligns with a: the cross-power spectrum F(a)*conj(F(b)) peaks at
    minus the displacement of b, i.e. at the correction itself.
    '''
    fa = np.fft.rfft2(a - a.mean())
    fb = np.fft.rfft2(b - b.mean())
    corr = np.fft.fftshift(np.fft.irfft2(fa * np.conj(fb), s=a.shape))
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    dy = peak[0] - a.shape[0] // 2
    dx = peak[1] - a.shape[1] // 2
    denom = float(np.linalg.norm(a - a.mean()) * np.linalg.norm(b - b.mean()))
    quality = float(corr.max()) / denom if denom > 0 else 0.0
    return int(dy), int(dx), quality


def _corr_1d(a: np.ndarray, b: np.ndarray) -> tuple[int, float]:
    '''Axial shift of b relative to a, same sign convention as _phase_corr_2d.'''
    fa = np.fft.rfft(a - a.mean())
    fb = np.fft.rfft(b - b.mean())
    corr = np.fft.fftshift(np.fft.irfft(fa * np.conj(fb), n=a.shape[0]))
    peak = int(np.argmax(corr))
    dz = peak - a.shape[0] // 2
    denom = float(np.linalg.norm(a - a.mean()) * np.linalg.norm(b - b.mean()))
    quality = float(corr.max()) / denom if denom > 0 else 0.0
    return int(dz), quality


def _pick_level(tile_dir: Path, max_y: int = MEASUREMENT_MAX_Y_PX):
    '''Finest pyramid level (lowest number) whose y extent fits max_y.'''
    levels = sorted(
        (d for d in tile_dir.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    for d in levels:
        try:
            arr = zarr.open(str(d), mode='r')
        except Exception:
            continue
        if arr.ndim == 3 and arr.shape[1] <= max_y:
            return d, tuple(arr.shape)
    return None, None


def _read_tile_block(arr, n_planes: int = MEASUREMENT_N_PLANES) -> np.ndarray:
    '''Mid-z z-crop of a (z, y, x) tile array as float32.'''
    nz = arr.shape[0]
    z0 = max(0, (nz - n_planes) // 2)
    z1 = min(nz, z0 + n_planes)
    return np.asarray(arr[z0:z1, :, :], dtype=np.float32)


def measure_inter_channel_shift(xml_path: Path, reference_channel: int, quality_threshold: float):
    '''Measure the global inter-channel shift from the deconved tiles.

    For every tile number present in both the reference channel and channel i,
    phase-correlates a mid-z crop plane by plane (lateral) and the mean z
    profiles (axial), then takes the median across tiles. Returns per-channel
    shifts in raw level-0 pixels plus a diagnostics dict.
    '''
    xml_path = ensure_path(xml_path)
    root = ET.parse(xml_path).getroot()
    collection_dir = get_ome_zarr_directory_from_xml(xml_path)
    if collection_dir is None or not collection_dir.exists():
        raise FileNotFoundError(f'no readable OME-Zarr collection for {xml_path}')

    # BigStitcher XML maps setup id -> channel/tile attributes and zgroup id -> tile file
    setups = {}
    for s in root.findall('.//ViewSetup'):
        attrs = s.find('attributes')
        setups[s.find('id').text] = (int(attrs.find('channel').text), int(attrs.find('tile').text))
    zgroup_paths = {z.attrib['setup']: z.attrib['path'] for z in root.iter('zgroup')}
    views = {
        (setups[sid][0], setups[sid][1]): collection_dir / zgroup_paths[sid]
        for sid in zgroup_paths if sid in setups
    }
    channels = sorted({ch for ch, tile in views})
    if len(channels) < 2:
        raise ValueError(f'only {len(channels)} channel(s) in {xml_path}; nothing to correct')

    # ViewSetup voxelSize lists X Y Z um
    voxel_xyz = [float(v) for v in root.find('.//ViewSetup/voxelSize/size').text.split()]

    ref_tiles = {t for c, t in views if c == reference_channel}
    results = {}
    for ch in channels:
        if ch == reference_channel:
            continue
        ch_tiles = {t for c, t in views if c == ch}
        per_tile = []
        level_shape = None
        level0_shape = None
        for tile in sorted(ref_tiles & ch_tiles):
            ref_dir = views[(reference_channel, tile)]
            ch_dir = views[(ch, tile)]
            level_dir, ref_level_shape = _pick_level(ref_dir)
            if level_dir is None:
                print(f'  tile {tile}: no pyramid level with y <= {MEASUREMENT_MAX_Y_PX}, skipped')
                continue
            if level_shape is None:
                level_shape, level0_shape = ref_level_shape, tuple(zarr.open(str(ref_dir / '0'), mode='r').shape)
            ch_arr = zarr.open(str(ch_dir / level_dir.name), mode='r')
            ref_block = _read_tile_block(zarr.open(str(ref_dir / level_dir.name), mode='r'))
            ch_block = _read_tile_block(ch_arr)
            if ref_block.shape != ch_block.shape:
                print(f'  tile {tile}: shape mismatch {ref_block.shape} vs {ch_block.shape}, skipped')
                continue
            dys, dxs, qs = [], [], []
            for p in range(ref_block.shape[0]):
                dy, dx, q = _phase_corr_2d(ref_block[p], ch_block[p])
                if q >= quality_threshold:
                    dys.append(dy)
                    dxs.append(dx)
                    qs.append(q)
            if not dys:
                print(f'  tile {tile}: no plane above quality {quality_threshold}, skipped')
                continue
            dz, qz = _corr_1d(ref_block.mean(axis=(1, 2)), ch_block.mean(axis=(1, 2)))
            per_tile.append({
                'tile': tile,
                'dy': median(dys), 'dx': median(dxs), 'q': median(qs),
                'dz': dz, 'qz': qz, 'n_planes': len(dys),
            })
            print(f"  tile {tile}: dy={per_tile[-1]['dy']} dx={per_tile[-1]['dx']} "
                  f"dz={dz} q={per_tile[-1]['q']:.3f} ({per_tile[-1]['n_planes']} planes)")
        if not per_tile:
            print(f'channel {ch}: no usable tile pairs, no correction measured')
            continue
        # measured level -> level-0 (raw) pixels
        down_y = level0_shape[1] / level_shape[1]
        down_x = level0_shape[2] / level_shape[2]
        down_z = level0_shape[0] / level_shape[0]
        results[ch] = {
            'dy': median([t['dy'] for t in per_tile]) * down_y,
            'dx': median([t['dx'] for t in per_tile]) * down_x,
            'dz': median([t['dz'] for t in per_tile]) * down_z,
            'quality': median([t['q'] for t in per_tile]),
            'n_tiles': len(per_tile),
            'tiles': per_tile,
        }
        print(f"channel {ch}: dy={results[ch]['dy']} dx={results[ch]['dx']} dz={results[ch]['dz']} raw px "
              f"(q={results[ch]['quality']:.3f}, {len(per_tile)} tiles)")
    return results, {
        'level_shape': level_shape, 'level0_shape': level0_shape, 'voxel_xyz': voxel_xyz,
    }


def apply_chromatic_translations(xml_path: Path, shifts: dict, z_ratio_of_xy: float) -> int:
    '''Insert a per-channel Chromatic Correction ViewTransform into the XML.

    shifts: {channel: {'dy': raw px, 'dx': raw px, 'dz': raw z-planes}}. The
    transform is prepended to each affected ViewRegistration (pure
    translations commute, so document position does not change the result);
    the z translation is expressed in the global units the calibration
    transform defines (1 z-plane = z_ratio_of_xy lateral pixels). Returns the
    number of registrations corrected. Registrations already carrying a
    Chromatic Correction transform are left untouched (idempotent re-runs).
    '''
    tree = ET.parse(xml_path)
    root = tree.getroot()
    setups = {
        s.find('id').text: int(s.find('attributes/channel').text)
        for s in root.findall('.//ViewSetup')
    }
    registrations = root.find('ViewRegistrations')
    n_corrected = 0
    for reg in registrations.findall('ViewRegistration'):
        ch = setups[reg.attrib['setup']]
        if ch not in shifts:
            continue
        if any(
            vt.find('name') is not None and vt.find('name').text == CHROMATIC_TRANSFORM_NAME
            for vt in reg.findall('ViewTransform')
        ):
            continue
        s = shifts[ch]
        vt = ET.Element('ViewTransform', {'type': 'affine'})
        name = ET.SubElement(vt, 'name')
        name.text = CHROMATIC_TRANSFORM_NAME
        affine = ET.SubElement(vt, 'affine')
        affine.text = (
            f'1.0 0.0 0.0 {s["dx"]} 0.0 1.0 0.0 {s["dy"]} 0.0 0.0 1.0 {s["dz"] * z_ratio_of_xy}'
        )
        reg.insert(0, vt)
        n_corrected += 1
    tree.write(xml_path)
    return n_corrected


@app.command()
def apply_chromatic_correction(
    dir_loc: Path,
    reference_channel: int = None,
    quality_threshold: float = None,
    max_shift_um: float = None,
    dry_run: bool = False,
):
    '''Measure the inter-channel shift on the deconved tiles and apply
    per-channel translations to the BigStitcher XML before fusion.

    dir_loc: directory holding the BigStitcher XML (the alignment output dir).
    '''
    reference_channel = BIGSTITCHER_CHROMATIC_REFERENCE_CHANNEL if reference_channel is None else reference_channel
    quality_threshold = BIGSTITCHER_CHROMATIC_QUALITY_THRESHOLD if quality_threshold is None else quality_threshold
    max_shift_um = BIGSTITCHER_CHROMATIC_MAX_SHIFT_UM if max_shift_um is None else max_shift_um

    xml_path = does_dir_contain_bigstitcher_metadata(dir_loc)
    if xml_path is None:
        raise FileNotFoundError(f'no bigstitcher xml under {dir_loc}')
    print(f'Chromatic correction: measuring inter-channel shift in {xml_path}')

    results, diag = measure_inter_channel_shift(xml_path, reference_channel, quality_threshold)
    if not results:
        print('Chromatic correction: nothing measured, XML left unchanged')
        return

    voxel_xyz = diag['voxel_xyz']
    max_px = max_shift_um / voxel_xyz[1]  # lateral sanity bound in raw px
    applied = {}
    for ch, s in sorted(results.items()):
        shift_um = (s['dy'] * voxel_xyz[1], s['dx'] * voxel_xyz[0])
        if max(abs(s['dy']), abs(s['dx'])) > max_px:
            print(f'channel {ch}: measured shift {shift_um[0]:.2f}/{shift_um[1]:.2f} um exceeds '
                  f'{max_shift_um} um bound, skipping this channel')
            continue
        applied[ch] = s
        print(f'channel {ch}: applying dy={s["dy"]} dx={s["dx"]} dz={s["dz"]} raw px '
              f'({shift_um[0]:.2f} um y, {shift_um[1]:.2f} um x, ref channel {reference_channel})')
    if not applied:
        print('Chromatic correction: no channel passed the sanity bound, XML left unchanged')
        return

    z_ratio = voxel_xyz[2] / voxel_xyz[0]
    if dry_run:
        print('dry run: XML left unchanged')
        return
    n = apply_chromatic_translations(xml_path, applied, z_ratio)
    print(f'Chromatic correction: wrote {CHROMATIC_TRANSFORM_NAME} transforms into {n} registrations of {xml_path}')


if __name__ == '__main__':
    app()
