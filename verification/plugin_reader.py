"""The Amira plugin's reader functions, copied verbatim.

These come from ``ZarrRead.pyscro`` in
janelia-cellmap/amira_python_extensions, unchanged apart from dropping the
``hx_message`` calls that need a running Amira. Keeping them verbatim is the
point: ``check_plugin_compat.py`` imports them so its comparison runs the
plugin's actual code path rather than a paraphrase of it.

Do not "clean up" this file. If the plugin changes, re-copy it.

Can also be run directly to inspect one array the way ZarrRead would see it:

    python plugin_reader.py <container.zarr> <group> <level>
"""
import os
import sys
from pathlib import Path

import numpy as np
import tensorstore as ts
import zarr

# ---- verbatim from ZarrRead.pyscro -----------------------------------------

_container_extension = '.zarr'


def is_zarr_array(path: str) -> bool:
    try:
        node = zarr.open(str(path), mode='r')
        return isinstance(node, zarr.Array)
    except Exception:
        return False


def open_ts_array(path: str):
    try:
        node = zarr.open(str(path), mode='r')
        if not isinstance(node, zarr.Array):
            raise ValueError(f'{path} is a zarr group, not an array')
        driver = 'zarr3' if node.metadata.zarr_format == 3 else 'zarr'
    except Exception as e:
        raise ValueError(f'No zarr array found at {path}') from e
    return ts.open({
        'driver': driver,
        'kvstore': {'driver': 'file', 'path': str(path)},
    }, write=False).result()


def find_multiscales(group_path: str, container_root: str):
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
    axis_types = [ax.get('type') for ax in axes_list] if axes_list else [None] * ndim
    axis_names = [ax.get('name', '') for ax in axes_list] if axes_list else []
    assumed_layout = None
    if not all(t in ('time', 'channel', 'space') for t in axis_types):
        if ndim == 5:
            return None, '5D arrays require axis type metadata.'
        elif ndim == 4:
            axis_types = ['channel', 'space', 'space', 'space']
            axis_names = axis_names or ['c', 'z', 'y', 'x']
            assumed_layout = '(c, z, y, x)'
        else:
            axis_types = ['space'] * 3
            axis_names = axis_names or ['z', 'y', 'x']
            assumed_layout = '(z, y, x)'
    spatial_indices = [i for i, t in enumerate(axis_types) if t == 'space']
    storage_axes = [axis_names[i] for i in spatial_indices]
    if set(storage_axes) != {'x', 'y', 'z'}:
        return None, 'Spatial axes must be named x, y, z (got {0}).'.format(storage_axes)
    return (axis_types, axis_names, spatial_indices, storage_axes, None, None,
            assumed_layout), None


def get_resolution_and_offset(array_dir, ndim, find_result, axis_types=None):
    voxel_size = [1.0] * ndim
    offset = [0.0] * ndim
    units = ['nanometer'] * ndim
    multiscales, ms_dir, errors = find_result
    if multiscales is None:
        return voxel_size, offset, units
    ms = multiscales[0]
    axes = ms.get('axes') or []
    if axes:
        units = [ax.get('unit') or 'nanometer' for ax in axes]
    try:
        rel_path = str(Path(array_dir).relative_to(ms_dir))
    except ValueError:
        rel_path = os.path.basename(array_dir)
    for ds in ms.get('datasets') or []:
        if ds.get('path', '').lstrip('/') == rel_path.lstrip('/'):
            for ct in ds.get('coordinateTransformations') or []:
                if 'scale' in ct:
                    voxel_size = list(ct['scale'])
                elif 'translation' in ct:
                    offset = list(ct['translation'])
            return voxel_size, offset, units
    return voxel_size, offset, units


# ---- drive it the way ZarrRead.compute() does -------------------------------

def main():
    container, group, level = sys.argv[1], sys.argv[2], sys.argv[3]
    array_dir = os.path.join(container, group, level)

    print('reading {0} with the plugin\'s tensorstore path'.format(array_dir))
    assert is_zarr_array(array_dir), 'plugin does not recognize this as a zarr array!'

    dataset = open_ts_array(array_dir)
    ndim = len(dataset.shape)
    find_result = find_multiscales(os.path.join(container, group), container)
    multiscales, ms_dir, errors = find_result
    assert multiscales is not None, 'plugin found no multiscales: {0}'.format(errors)

    axes = multiscales[0].get('axes') or []
    classified, err = classify_axes(axes, ndim)
    assert classified is not None, err
    axis_types, axis_names, spatial_indices, storage_axes, _, _, assumed = classified

    resolution, offset, units = get_resolution_and_offset(
        array_dir, ndim, find_result, axis_types=axis_types)

    # ZarrRead transposes storage order to Amira's (x, y, z)
    slices = {ax: slice(0, dataset.shape[spatial_indices[storage_axes.index(ax)]])
              for ax in ('x', 'y', 'z')}
    ts_slices = [slices[axis_names[i]] for i in range(ndim)]
    raw = dataset[tuple(ts_slices)].read().result()
    perm = [storage_axes.index(ax) for ax in ('x', 'y', 'z')]
    array = np.transpose(raw, perm)

    resolution_xyz = [resolution[storage_axes.index(ax)] for ax in ('x', 'y', 'z')]
    offset_xyz = [offset[storage_axes.index(ax)] for ax in ('x', 'y', 'z')]
    shape_native = [(s - 1) * r for s, r in zip(array.shape, resolution_xyz)]
    bbox_starts = tuple(r * slices[ax].start + o
                        for r, ax, o in zip(resolution_xyz, ('x', 'y', 'z'), offset_xyz))
    bbox_stops = tuple(o + s for o, s in zip(bbox_starts, shape_native))

    print()
    print('  dtype          {0}'.format(dataset.dtype.numpy_dtype))
    print('  storage shape  {0}  axes ({1})'.format(
        tuple(dataset.shape), ', '.join(axis_names)))
    print('  Amira shape    {0}  (x, y, z)'.format(array.shape))
    print('  assumed layout {0}'.format(assumed))
    print('  voxel size     {0}  (x, y, z) {1}'.format(resolution_xyz, units[0]))
    print('  offset         {0}  (x, y, z)'.format(offset_xyz))
    print('  bounding box   {0} -> {1}'.format(
        [round(v, 3) for v in bbox_starts], [round(v, 3) for v in bbox_stops]))
    print()

    # Cross-check the actual voxel values against a direct zarr read.
    direct = zarr.open(array_dir, mode='r')[...]
    assert np.array_equal(array, np.transpose(direct, perm)), 'voxel mismatch!'
    print('  voxel values match a direct zarr read: OK')
    print('  sample corner value: {0}'.format(array[0, 0, 0]))


if __name__ == '__main__':
    main()
