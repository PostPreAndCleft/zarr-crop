"""Generate a throwaway OME-Zarr for testing the crop workflow end to end.

The point of this dataset is to make mistakes *visible* rather than plausible,
so that a crop drawn in Neuroglancer can be checked by eye:

* **Asymmetric shape.** Every axis is a different length, so an axis-order or
  transpose bug produces a wrong shape and fails loudly instead of quietly
  returning the wrong region.
* **Countable stripes.** Moving out from the origin you cross 3 sheets along Z,
  2 along Y and 1 along X, so you can always tell which axis you are scrolling.
* **Landmark spheres** at documented s0 coordinates, each a different size and
  brightness. Draw a box round one, run the crop, and check that exactly that
  sphere came along.

  Caveat when checking by pixel value: exact values only survive at **s0**. The
  coarser levels are block means, so voxels on a sphere's rim are blends of its
  value with the background and can land exactly on *another* sphere's value. A
  handful of stray matching voxels at s1/s2 is expected and does not mean the
  crop is wrong -- verified on this dataset, where an s2 crop of sphere E holds
  1 voxel of C's value and 2 of D's, all on E's edge. Judge by the large blob
  (or compare against a direct slice of the same level), not by whether a value
  appears at all.
* **Metadata matching production.** scale 8/16/32 nm with translation 0/4/12 nm
  is the half-voxel convention verified against the real CellMap dataset
  jrc_amphiuma-means-liver-1 -- so this exercises the s0-to-coarse-level
  conversion for real, not as a toy.

Multiscales metadata goes on the container root, matching how Fileglancer served
`mito001.zarr/` in a real Neuroglancer link, which is what lets Neuroglancer open
the container directly.

Writes both a zarr v2 and a zarr v3 copy, since Fileglancer served v2 and S3
served v3 in the real link, and the Amira plugin handles both.

Usage:
    python tools/make_test_dataset.py [OUTPUT_DIR]

OUTPUT_DIR defaults to /Volumes/andy_crop_test. Nothing is overwritten: the
script refuses to run if a target container already exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from crop_tool import zarr_io  # noqa: E402

DEFAULT_OUT = Path('/Volumes/andy_crop_test')

#: s0 shape, (z, y, x). Deliberately all different.
SHAPE = (512, 384, 256)

#: Voxel size in nanometres at s0, and the downsampling factor per level.
S0_SCALE_NM = 8.0
LEVELS = 3

CHUNK = 64
DTYPE = 'uint8'

#: Intensity budget, kept separated so features stay distinguishable.
BACKGROUND_LO, BACKGROUND_HI = 20, 90
STRIPE_VALUE = 150
CORNER_VALUE = 255

#: (name, (z, y, x) centre at s0, radius in voxels, intensity)
SPHERES = [
    ('A', (100, 100, 64), 22, 170),
    ('B', (200, 180, 180), 18, 190),
    ('C', (320, 90, 100), 25, 210),
    ('D', (420, 280, 70), 20, 230),
    ('E', (256, 300, 190), 16, 245),
]

#: Stripes per axis: crossing N sheets tells you which axis you are on.
STRIPES = {0: 3, 1: 2, 2: 1}   # axis index (z, y, x) -> sheet count
STRIPE_THICKNESS = 6
STRIPE_GAP = 14
STRIPE_START = 8


def build_volume(shape=SHAPE) -> np.ndarray:
    """Build the s0 volume."""
    nz, ny, nx = shape
    zz, yy, xx = np.meshgrid(
        np.arange(nz), np.arange(ny), np.arange(nx), indexing='ij')

    # Low-frequency texture, different period per axis so the pattern itself is
    # asymmetric. Amplitude sits inside the background band.
    span = BACKGROUND_HI - BACKGROUND_LO
    texture = (
        np.sin(zz / 37.0) * 0.4
        + np.sin(yy / 23.0) * 0.35
        + np.sin(xx / 15.0) * 0.25
    )
    volume = BACKGROUND_LO + (texture + 1.0) / 2.0 * span

    # A little grain so it reads as image data rather than a test card.
    rng = np.random.default_rng(0)
    volume += rng.normal(0.0, 3.0, size=shape)
    volume = np.clip(volume, 0, 255)

    # Countable sheets near each origin face.
    for axis, count in STRIPES.items():
        for n in range(count):
            lo = STRIPE_START + n * (STRIPE_THICKNESS + STRIPE_GAP)
            hi = lo + STRIPE_THICKNESS
            index = [slice(None)] * 3
            index[axis] = slice(lo, hi)
            volume[tuple(index)] = STRIPE_VALUE

    # Solid cube at the origin corner, so (0, 0, 0) is unmistakable.
    volume[0:24, 0:24, 0:24] = CORNER_VALUE

    # Landmark spheres.
    for name, (cz, cy, cx), radius, value in SPHERES:
        mask = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2
        volume[mask] = value

    return volume.astype(DTYPE)


def downsample(volume: np.ndarray) -> np.ndarray:
    """2x block mean along every axis. Requires even dimensions."""
    nz, ny, nx = volume.shape
    if nz % 2 or ny % 2 or nx % 2:
        raise ValueError('shape {0} is not evenly divisible by 2'.format(volume.shape))
    blocks = volume.astype(np.float32).reshape(nz // 2, 2, ny // 2, 2, nx // 2, 2)
    return blocks.mean(axis=(1, 3, 5)).round().clip(0, 255).astype(DTYPE)


def level_metadata(count=LEVELS, base_scale=S0_SCALE_NM):
    """Per-level scale and translation.

    Uses the convention verified on real CellMap data: a coarse level's first
    voxel centre sits at the centre of the block of s0 voxels it covers, so

        translation_n = (scale_n - scale_0) / 2

    giving 0, 4, 12 nm for 8, 16, 32 nm levels. This offset is exactly what a
    naive "divide the s0 index by 2" gets wrong.
    """
    out = []
    for n in range(count):
        scale = base_scale * (2 ** n)
        translation = (scale - base_scale) / 2.0
        out.append(('s{0}'.format(n), [scale] * 3, [translation] * 3))
    return out


def multiscales_entry(levels, name):
    entry = {
        'name': name,
        'axes': [{'name': a, 'type': 'space', 'unit': 'nanometer'}
                 for a in ('z', 'y', 'x')],
        'coordinateTransformations': [{'type': 'scale', 'scale': [1.0, 1.0, 1.0]}],
        'datasets': [
            {
                'path': path,
                'coordinateTransformations': [
                    {'type': 'scale', 'scale': [float(v) for v in scale]},
                    {'type': 'translation', 'translation': [float(v) for v in translation]},
                ],
            }
            for path, scale, translation in levels
        ],
    }
    return entry


def write_container(path: Path, pyramid, levels, zarr_format: int) -> None:
    """Write one .zarr container with the pyramid and root multiscales."""
    from ome_zarr_models.v04.image import Multiscale as MultiscaleV04
    from ome_zarr_models.v05.image import Multiscale as MultiscaleV05

    zarr_io.ensure_group_metadata(str(path), zarr_format)

    for (level_path, _, _), data in zip(levels, pyramid):
        # Reuse the package's writer so the on-disk conventions (zstd level 3,
        # dimension_names, v2 separator) match everything else we produce.
        array = zarr_io.create_array(
            container=str(path), name=level_path, shape=data.shape,
            dtype=data.dtype, zarr_format=zarr_format, chunks=[CHUNK] * 3)
        array[...] = data
        print('    {0:3s} shape {1!s:20s} chunks {2}'.format(
            level_path, data.shape, array.chunks))

    entry = multiscales_entry(levels, name=path.name[:-len('.zarr')])
    if zarr_format == 3:
        MultiscaleV05.model_validate(entry)
    else:
        entry['version'] = '0.4'
        MultiscaleV04.model_validate(entry)
    zarr_io.write_multiscales(str(path), entry, zarr_format)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out_dir = Path(argv[0]) if argv else DEFAULT_OUT

    if not out_dir.is_dir():
        print('error: {0} is not a directory. Is the share mounted?'.format(out_dir),
              file=sys.stderr)
        return 1

    targets = {fmt: out_dir / 'crop_test_em_v{0}.zarr'.format(fmt) for fmt in (3, 2)}
    existing = [str(p) for p in targets.values() if p.exists()]
    if existing:
        print('error: refusing to overwrite existing container(s):', file=sys.stderr)
        for p in existing:
            print('  {0}'.format(p), file=sys.stderr)
        print('Remove them first, or pass a different output directory.', file=sys.stderr)
        return 1

    levels = level_metadata()

    print('building s0 volume {0} {1} ...'.format(SHAPE, DTYPE))
    pyramid = [build_volume()]
    for _ in range(LEVELS - 1):
        pyramid.append(downsample(pyramid[-1]))

    total = sum(a.nbytes for a in pyramid)
    print('pyramid: {0}  ({1:.1f} MB per copy)'.format(
        [a.shape for a in pyramid], total / 1e6))

    for fmt, path in targets.items():
        print('\nwriting zarr v{0} -> {1}'.format(fmt, path))
        write_container(path, pyramid, levels, fmt)

    # -- manifest -----------------------------------------------------------
    print('\n' + '=' * 68)
    print('LANDMARK MANIFEST (s0 voxel indices, z y x)')
    print('=' * 68)
    print('{0:6s} {1:>18s} {2:>8s} {3:>6s}   {4}'.format(
        'sphere', 'centre (z,y,x)', 'radius', 'value', 'suggested box (s0)'))
    for name, centre, radius, value in SPHERES:
        pad = radius + 8
        lo = [max(0, c - pad) for c in centre]
        hi = [min(s, c + pad) for c, s in zip(centre, SHAPE)]
        print('{0:6s} {1!s:>18s} {2:>8d} {3:>6d}   min {4} max {5}'.format(
            name, centre, radius, value,
            ','.join(str(v) for v in lo), ','.join(str(v) for v in hi)))

    print('\nlevels:')
    for path, scale, translation in levels:
        print('  {0}  scale {1} nm  translation {2} nm'.format(
            path, scale[0], translation[0]))

    print('\nstripes from each origin face: {0} along Z, {1} along Y, {2} along X'
          .format(STRIPES[0], STRIPES[1], STRIPES[2]))
    print('solid {0}-valued cube at the origin corner, 24 voxels per side'
          .format(CORNER_VALUE))
    print('\nnote: sphere values above are exact at s0 only. s1/s2 are block means,')
    print('so rim voxels blend with the background and a few can coincide with')
    print("another sphere's value. Judge a crop by its large blob, not by whether")
    print('a value appears at all.')

    print('\nnext: in Fileglancer, navigate to this folder using the Linux/cluster')
    print('path it displays (not /Volumes/...), create a data link so Neuroglancer')
    print('can reach it over HTTP, open it, draw a box round a sphere, and copy')
    print('the link back. Avoid opening the .zarr in Finder -- it scatters')
    print('.DS_Store files through the chunk directories.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
