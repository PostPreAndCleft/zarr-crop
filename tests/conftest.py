"""Synthetic OME-Zarr fixtures.

The coordinate-encoding trick is taken from the plugin's own
``test_data/make_test_zarr_datasets.py``: every voxel holds a value encoding
its own storage index,

    value[i, j, k] = i * 10000 + j * 100 + k

so a mis-slice or a bad transpose is obvious on inspection rather than showing
up as an opaque array mismatch. Axis sizes must stay under 100.

Added here beyond the plugin's cases: multi-level containers (s0/s1/s2) with
the half-voxel ``translation`` offsets that real OME-NGFF pyramids carry. Those
are what the s0-coordinate conversion has to get right, and the plugin's
single-level fixtures cannot exercise them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

NANO = 'nanometer'


def coord_encoded(shape) -> np.ndarray:
    """Array where each voxel encodes its own index tuple (see module docstring)."""
    if any(s >= 100 for s in shape):
        raise ValueError('Coordinate encoding requires axes smaller than 100.')
    indices = np.indices(shape)
    out = np.zeros(shape, dtype=np.int64)
    ndim = len(shape)
    for k in range(ndim):
        out += indices[k] * (100 ** (ndim - 1 - k))
    return out.astype('uint32')


def axes_metadata(axes_spec):
    out = []
    for ax_type, name in axes_spec:
        ax = {'name': name, 'type': ax_type}
        if ax_type == 'space':
            ax['unit'] = NANO
        elif ax_type == 'time':
            ax['unit'] = 'second'
        out.append(ax)
    return out


def write_container(path, levels, axes_spec, zarr_format=3, include_axes=True,
                    group='em', dtype='uint32', data_by_level=None):
    """Write a .zarr container holding one multiscale group.

    ``levels`` is a list of dicts with ``path``, ``shape``, ``scale`` and
    ``translation`` (all in storage axis order). Returns a dict of level path ->
    the array data written, so tests can compute expectations from the source
    data rather than by re-reading it through the code under test.
    """
    path = Path(path)
    root = zarr.open_group(str(path), mode='w', zarr_format=zarr_format)
    grp = root.create_group(group)

    written = {}
    for level in levels:
        shape = tuple(level['shape'])
        if data_by_level and level['path'] in data_by_level:
            data = data_by_level[level['path']].astype(dtype)
        else:
            data = coord_encoded(shape).astype(dtype)
        chunks = tuple(min(s, 32) for s in shape)
        kwargs = dict(shape=shape, dtype=dtype, chunks=chunks)
        if zarr_format == 3 and include_axes:
            kwargs['dimension_names'] = [n for _, n in axes_spec]
        arr = grp.create_array(level['path'], **kwargs)
        arr[:] = data
        written[level['path']] = data

    entry = {
        'name': 'test',
        'datasets': [
            {
                'path': level['path'],
                'coordinateTransformations': [
                    {'type': 'scale', 'scale': [float(v) for v in level['scale']]},
                    {'type': 'translation',
                     'translation': [float(v) for v in level['translation']]},
                ],
            }
            for level in levels
        ],
        'coordinateTransformations': [
            {'type': 'scale', 'scale': [1.0] * len(levels[0]['shape'])},
        ],
    }
    if include_axes:
        entry['axes'] = axes_metadata(axes_spec)

    if zarr_format == 3:
        grp.attrs.update({'ome': {'version': '0.5', 'multiscales': [entry]}})
    else:
        entry['version'] = '0.4'
        grp.attrs.update({'multiscales': [entry]})
    return written


# ---------------------------------------------------------------------------
# The pyramid fixture: three levels, zyx storage, half-voxel level offsets.
#
# s0 scale (10, 20, 30) nm, translation (100, 200, 300)
# s1 scale (20, 40, 60), s2 scale (40, 80, 120)
#
# Each coarser level's translation follows the usual OME-NGFF convention of
# placing the level's first voxel centre at the centre of the block of s0
# voxels it covers:  t_n = t_0 + (scale_n - scale_0) / 2
#   s1 -> (105, 210, 315)
#   s2 -> (115, 230, 345)
# These offsets are exactly what a plain "divide by 2" gets wrong.
# ---------------------------------------------------------------------------

ZYX = [('space', 'z'), ('space', 'y'), ('space', 'x')]
S0_SCALE = (10.0, 20.0, 30.0)
S0_TRANSLATION = (100.0, 200.0, 300.0)

PYRAMID_LEVELS = [
    dict(path='s0', shape=(80, 30, 40), scale=S0_SCALE, translation=S0_TRANSLATION),
    dict(path='s1', shape=(40, 15, 20), scale=(20.0, 40.0, 60.0),
         translation=(105.0, 210.0, 315.0)),
    dict(path='s2', shape=(20, 8, 10), scale=(40.0, 80.0, 120.0),
         translation=(115.0, 230.0, 345.0)),
]


@pytest.fixture
def pyramid_v3(tmp_path):
    """Three-level zyx pyramid, zarr v3 / OME-NGFF 0.5."""
    path = tmp_path / 'src_v3.zarr'
    data = write_container(path, PYRAMID_LEVELS, ZYX, zarr_format=3)
    return dict(container=path, data=data, levels=PYRAMID_LEVELS, group='em')


@pytest.fixture
def pyramid_v2(tmp_path):
    """Three-level zyx pyramid, zarr v2 / OME-NGFF 0.4."""
    path = tmp_path / 'src_v2.zarr'
    data = write_container(path, PYRAMID_LEVELS, ZYX, zarr_format=2)
    return dict(container=path, data=data, levels=PYRAMID_LEVELS, group='em')


# ---------------------------------------------------------------------------
# The same logical volume stored two ways, to test axis-order independence.
# ---------------------------------------------------------------------------

VOLUME_ZYX = coord_encoded((20, 15, 10))


@pytest.fixture
def same_volume_two_orders(tmp_path):
    """One volume written twice: once zyx, once xyz.

    Both describe identical physical content, so a crop of the same region must
    produce byte-identical (z, y, x) output from either.
    """
    zyx_path = tmp_path / 'as_zyx.zarr'
    write_container(
        zyx_path,
        [dict(path='s0', shape=(20, 15, 10), scale=(10.0, 20.0, 30.0),
              translation=(100.0, 200.0, 300.0))],
        ZYX, zarr_format=3, data_by_level={'s0': VOLUME_ZYX})

    xyz_path = tmp_path / 'as_xyz.zarr'
    write_container(
        xyz_path,
        [dict(path='s0', shape=(10, 15, 20), scale=(30.0, 20.0, 10.0),
              translation=(300.0, 200.0, 100.0))],
        [('space', 'x'), ('space', 'y'), ('space', 'z')], zarr_format=3,
        data_by_level={'s0': VOLUME_ZYX.transpose(2, 1, 0)})

    return dict(zyx=zyx_path, xyz=xyz_path, volume=VOLUME_ZYX)


@pytest.fixture
def four_d_czyx(tmp_path):
    """4D (c, z, y, x) source, for --channel selection."""
    path = tmp_path / 'src_4d.zarr'
    axes = [('channel', 'c')] + ZYX
    levels = [dict(path='s0', shape=(3, 20, 15, 10), scale=(1.0, 10.0, 20.0, 30.0),
                   translation=(0.0, 100.0, 200.0, 300.0))]
    data = write_container(path, levels, axes, zarr_format=3)
    return dict(container=path, data=data['s0'])


@pytest.fixture
def no_axes_meta(tmp_path):
    """3D source with no axis metadata, exercising the (z, y, x) fallback."""
    path = tmp_path / 'src_no_axes.zarr'
    levels = [dict(path='s0', shape=(20, 15, 10), scale=(10.0, 20.0, 30.0),
                   translation=(100.0, 200.0, 300.0))]
    data = write_container(path, levels, ZYX, zarr_format=3, include_axes=False)
    return dict(container=path, data=data['s0'])
