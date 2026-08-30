"""Zarr access for the crop tool.

Most of this file is ported from the Amira plugin
(janelia-cellmap/amira_python_extensions, ``src/extensions/zarr/``) with the
Amira API removed. Keeping the ported helpers recognizably close to their
originals is deliberate: fixes made to the plugin should be easy to carry
across, and vice versa.

Two substitutions relative to the plugin:

* ``open_ts_array`` / ``create_ts_array`` used tensorstore; here they are
  ``open_array`` / ``create_array`` built on zarr-python. The on-disk result is
  the same, verified field by field against the plugin's tensorstore specs.
  Zarr v2 output is byte-identical in its metadata. Zarr v3 output differs only
  in that zarr-python writes two spec defaults explicitly that tensorstore
  leaves implicit (``bytes.configuration.endian = "little"`` and
  ``zstd.configuration.checksum = false``); both libraries read either form.
* ``get_resolution_and_offset`` popped up Amira dialogs on the no-metadata
  path. Here it takes a ``warn`` callback so the caller decides how to report.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numcodecs
import numpy as np
import zarr
from zarr.codecs import BytesCodec, ZstdCodec

_container_extension = '.zarr'

#: Chunk shape ceiling, matching the plugin's ``create_ts_array``. Actual
#: chunks are clamped per axis to the array shape (see :func:`create_array`).
DEFAULT_CHUNK = 128

#: Canonical on-disk axis order for output, matching ``ZarrWrite``.
OUTPUT_AXES = ('z', 'y', 'x')


# ---------------------------------------------------------------------------
# Ported verbatim from the plugin
# ---------------------------------------------------------------------------

def split_path_at_container(path: str):
    # check whether a path contains a valid file path to a container file, and if so which container format it is
    result = None, None
    pathobj = Path(path)
    if pathobj.suffix == _container_extension:
        result = [path, '']
    else:
        for parent in pathobj.parents:
            if parent.suffix == _container_extension:
                result = path.split(parent.suffix)
                result[0] += parent.suffix
    return result


def is_zarr_array(path: str) -> bool:
    try:
        node = zarr.open(str(path), mode='r')
        return isinstance(node, zarr.Array)
    except Exception:
        return False


def find_multiscales(group_path: str, container_root: str):
    """Walk up from group_path to container_root, return first OME-Zarr multiscales list found.

    Reads raw attrs based on zarr format:
      zarr v3 -> attrs['ome']['multiscales']  (OME-NGFF 0.5)
      zarr v2 -> attrs['multiscales']         (OME-NGFF 0.4)

    Returns (multiscales, ms_dir, errors).
    """
    import os

    errors = []
    path = group_path
    while True:
        try:
            grp = zarr.open_group(str(path), mode='r')
            attrs = dict(grp.attrs)
            if grp.metadata.zarr_format == 3:
                ome = attrs.get('ome')
                ms = ome.get('multiscales') if isinstance(ome, dict) else None
            else:
                ms = attrs.get('multiscales')
            if ms:
                return ms, path, errors
            errors.append((path, 'NoMultiscales', 'no multiscales attribute in this group'))
        except Exception as e:
            errors.append((path, type(e).__name__, str(e)))
        if os.path.normpath(path) == os.path.normpath(container_root):
            return None, path, errors
        path = str(Path(path).parent)


def classify_axes(axes_list, ndim):
    """Classify OME-NGFF axes by type.

    Returns ((axis_types, axis_names, spatial_indices, storage_axes, t_idx,
    c_idx, assumed_layout), None) on success, or (None, error_message).

    Fallbacks (when axis type metadata is absent):
      ndim == 3 -> all axes assumed spatial (z, y, x)
      ndim == 4 -> assume (c, z, y, x) for legacy compatibility
      ndim == 5 -> rejected
    """
    axis_types = [ax.get('type') for ax in axes_list] if axes_list else [None] * ndim
    axis_names = [ax.get('name', '') for ax in axes_list] if axes_list else []
    assumed_layout = None

    if not all(t in ('time', 'channel', 'space') for t in axis_types):
        if ndim == 5:
            return None, '5D arrays require axis type metadata (time/channel/space).'
        elif ndim == 4:
            axis_types = ['channel', 'space', 'space', 'space']
            if not axis_names:
                axis_names = ['c', 'z', 'y', 'x']
            assumed_layout = '(c, z, y, x)'
        else:
            axis_types = ['space', 'space', 'space']
            if not axis_names:
                axis_names = ['z', 'y', 'x']
            assumed_layout = '(z, y, x)'

    t_indices = [i for i, t in enumerate(axis_types) if t == 'time']
    c_indices = [i for i, t in enumerate(axis_types) if t == 'channel']
    spatial_indices = [i for i, t in enumerate(axis_types) if t == 'space']

    if len(t_indices) > 1:
        return None, 'Multiple time axes are not supported.'
    if len(c_indices) > 1:
        return None, 'Multiple channel axes are not supported.'
    if len(spatial_indices) != 3:
        return None, 'Expected exactly 3 spatial axes, got {0}.'.format(len(spatial_indices))

    storage_axes = [axis_names[i] for i in spatial_indices]
    if set(storage_axes) != {'x', 'y', 'z'}:
        return None, 'Spatial axes must be named x, y, z (got {0}).'.format(storage_axes)

    t_idx = t_indices[0] if t_indices else None
    c_idx = c_indices[0] if c_indices else None
    return (axis_types, axis_names, spatial_indices, storage_axes, t_idx, c_idx, assumed_layout), None


# ---------------------------------------------------------------------------
# zarr-python replacements for the tensorstore helpers
# ---------------------------------------------------------------------------

def open_array(path: str, mode: str = 'r') -> zarr.Array:
    """Open an existing zarr array, v2 or v3, defaulting to read-only.

    The crop tool never opens a source with anything but ``mode='r'``.
    """
    node = zarr.open(str(path), mode=mode)
    if not isinstance(node, zarr.Array):
        raise ValueError('{0} is a zarr group, not an array'.format(path))
    return node


def ensure_group_metadata(path: str, zarr_format: int) -> zarr.Group:
    """Create a zarr group if the directory lacks one."""
    return zarr.open_group(str(path), mode='a', zarr_format=zarr_format)


def create_array(container: str, name: str, shape, dtype, zarr_format: int,
                 chunks: Optional[Sequence[int]] = None) -> zarr.Array:
    """Create the output array, matching the plugin's on-disk format.

    ``chunks`` defaults to :data:`DEFAULT_CHUNK` per axis and is clamped per
    axis to ``shape`` -- a 40-slice-deep crop gets a 40-deep chunk rather than
    a 128-deep one padded with fill value.
    """
    ceiling = [DEFAULT_CHUNK] * len(shape) if chunks is None else list(chunks)
    resolved = tuple(int(min(c, s)) for c, s in zip(ceiling, shape))

    if zarr_format == 3:
        return zarr.create_array(
            store=str(container), name=name, shape=tuple(shape), dtype=dtype,
            chunks=resolved,
            serializer=BytesCodec(),
            compressors=ZstdCodec(level=3),
            dimension_names=OUTPUT_AXES,
            zarr_format=3,
        )
    return zarr.create_array(
        store=str(container), name=name, shape=tuple(shape), dtype=dtype,
        chunks=resolved,
        compressors=numcodecs.Zstd(level=3),
        chunk_key_encoding={'name': 'v2', 'configuration': {'separator': '/'}},
        zarr_format=2,
    )


# ---------------------------------------------------------------------------
# Multiscales metadata
# ---------------------------------------------------------------------------

@dataclass
class Transform:
    """A dataset's coordinateTransformations, in storage axis order."""
    scale: list
    translation: list

    def physical(self, axis_index: int, index: float) -> float:
        return self.translation[axis_index] + index * self.scale[axis_index]

    def index(self, axis_index: int, physical: float) -> float:
        return (physical - self.translation[axis_index]) / self.scale[axis_index]


def transform_for(multiscales, ms_dir: str, array_dir: str, ndim: int,
                  warn: Optional[Callable[[str], None]] = None):
    """Return (Transform, units) for the dataset entry matching ``array_dir``.

    Ported from ``get_resolution_and_offset``; the Amira error dialogs are
    replaced by the ``warn`` callback.
    """
    import os

    def _warn(msg):
        if warn is not None:
            warn(msg)

    scale = [1.0] * ndim
    translation = [0.0] * ndim
    units = ['nanometer'] * ndim

    if multiscales is None:
        _warn('No OME-Zarr multiscales metadata found. Falling back to '
              'scale={0}, translation={1}.'.format(scale, translation))
        return Transform(scale, translation), units

    ms = multiscales[0]
    axes = ms.get('axes') or []
    if axes:
        units = [ax.get('unit') or ('' if ax.get('type') == 'channel' else 'nanometer')
                 for ax in axes]

    try:
        rel_path = str(Path(array_dir).relative_to(ms_dir))
    except ValueError:
        rel_path = os.path.basename(array_dir)

    for ds in ms.get('datasets') or []:
        if ds.get('path', '').lstrip('/') == rel_path.lstrip('/'):
            for ct in ds.get('coordinateTransformations') or []:
                if 'scale' in ct:
                    scale = [float(v) for v in ct['scale']]
                elif 'translation' in ct:
                    translation = [float(v) for v in ct['translation']]
            return Transform(scale, translation), units

    _warn('No multiscales dataset entry matches {0!r}. Falling back to '
          'scale={1}, translation={2}.'.format(rel_path, scale, translation))
    return Transform(scale, translation), units


def transform_for_reference_level(multiscales, ndim: int):
    """Return (Transform, dataset_path) for the finest level in ``multiscales``.

    OME-NGFF requires ``datasets`` to be ordered from highest to lowest
    resolution, so entry 0 is the s0 level that Fileglancer reports
    coordinates in.
    """
    if not multiscales:
        return None, None
    datasets = multiscales[0].get('datasets') or []
    if not datasets:
        return None, None
    ds = datasets[0]
    scale = [1.0] * ndim
    translation = [0.0] * ndim
    for ct in ds.get('coordinateTransformations') or []:
        if 'scale' in ct:
            scale = [float(v) for v in ct['scale']]
        elif 'translation' in ct:
            translation = [float(v) for v in ct['translation']]
    return Transform(scale, translation), ds.get('path')


def build_multiscales_entry(ds_name: str, scale_zyx, translation_zyx,
                            unit: str, zarr_format: int, name: str = 'amira_export'):
    """Build and validate the multiscales attribute, as ``ZarrWrite`` does.

    Where ``ZarrWrite`` recovered these numbers by string-scraping Amira's
    ``VoxelSize`` and ``PhysicalSize`` GUI labels, they are passed in directly
    here, computed from the source metadata.
    """
    from ome_zarr_models.v04.image import Multiscale as MultiscaleV04
    from ome_zarr_models.v05.image import Multiscale as MultiscaleV05

    entry = {
        'axes': [{'name': axis, 'type': 'space', 'unit': unit} for axis in OUTPUT_AXES],
        'coordinateTransformations': [{'scale': [1.0, 1.0, 1.0], 'type': 'scale'}],
        'datasets': [
            {
                'coordinateTransformations': [
                    {'scale': [float(v) for v in scale_zyx], 'type': 'scale'},
                    {'translation': [float(v) for v in translation_zyx], 'type': 'translation'},
                ],
                'path': ds_name,
            }
        ],
        'name': name,
    }

    if zarr_format == 3:
        MultiscaleV05.model_validate(entry)
    else:
        entry['version'] = '0.4'
        MultiscaleV04.model_validate(entry)
    return entry


def write_multiscales(group_dir: str, entry: dict, zarr_format: int) -> None:
    """Attach a multiscales entry to a group, as ``ZarrWrite`` does."""
    grp = zarr.open_group(str(group_dir), mode='a', zarr_format=zarr_format)
    if zarr_format == 3:
        grp.attrs.update({'ome': {'version': '0.5', 'multiscales': [entry]}})
    else:
        grp.attrs.update({'multiscales': [entry]})


# ---------------------------------------------------------------------------
# Reading a region
# ---------------------------------------------------------------------------

def aligned_slabs(start: int, stop: int, chunk: int, depth: int):
    """Split ``[start, stop)`` into slabs, aligned to the chunk grid.

    Interior boundaries are placed on absolute multiples of ``chunk`` wherever
    the depth budget allows, so no source chunk is decoded by more than one slab.
    An ROI rarely begins on a chunk boundary, so the first slab is the short one
    and every later slab is aligned.

    ``depth`` is a hard ceiling: if a single chunk is larger than the budget,
    slabs of ``depth`` are emitted unaligned rather than blowing the budget.
    """
    chunk = max(1, int(chunk))
    depth = max(1, int(depth))
    position = start
    while position < stop:
        if depth < chunk:
            end = position + depth
        else:
            boundary = ((position // chunk) + 1) * chunk
            if boundary - position > depth:
                end = position + depth
            else:
                whole = (depth - (boundary - position)) // chunk
                end = boundary + whole * chunk
        end = min(end, stop)
        if end <= position:            # defensive: always make progress
            end = min(position + depth, stop)
        yield position, end
        position = end


def copy_region(src: zarr.Array, dst: zarr.Array, full_index, z_axis: int,
                z_start: int, z_stop: int, perm, slab: int,
                chunk_z: int = 1) -> None:
    """Copy the ROI from ``src`` into ``dst``, transposing to (z, y, x).

    Reads in slabs along the source axis that maps to output z, so peak memory
    is bounded by the slab rather than by the whole ROI. ``full_index`` holds
    the per-axis selection (slices for spatial axes, ints for time/channel);
    the z entry is replaced per slab. Slab boundaries follow the source's chunk
    grid -- see :func:`aligned_slabs`.
    """
    for start, stop in aligned_slabs(z_start, z_stop, chunk_z, slab):
        index = list(full_index)
        index[z_axis] = slice(start, stop)
        raw = src[tuple(index)]
        dst[start - z_start:stop - z_start] = np.transpose(raw, perm)
