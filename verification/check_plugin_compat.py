"""Check that zarr-crop output is usable by the Amira plugin.

This is the proof behind the compatibility claim in the README, and it is
deliberately kept out of the pytest suite because it needs tensorstore, which
the tool itself does not depend on. Run it after changing anything in
``zarr_io.py``.

It answers the question that actually matters to annotators: if I cut a crop
with zarr-crop, will opening it in Amira give me the same thing as the old
Amira-only workflow? Two routes are compared:

  A) The old way -- ``ZarrRead`` the SOURCE level with X/Y/Z limits set to the
     ROI. This is what Amira holds in memory after protocol step 3.
  B) The new way -- ``zarr-crop`` writes a crop container, then ``ZarrRead``
     reads that crop whole.

Equal voxels prove the data is right; an equal Amira bounding box proves the
crop is positioned right, which is what makes it line up in Neuroglancer
(protocol step 6).

``plugin_reader.py`` next to this file holds the plugin's reader functions
copied verbatim from ``ZarrRead.pyscro``, so route A is genuinely the plugin's
code path and not a paraphrase of it.

Setup -- a separate env, because tensorstore is deliberately not a dependency of
the tool. It needs the tool's own deps as well, since it runs the CLI:

    micromamba create -n crop_verify -c conda-forge -y \\
        python=3.12 zarr=3.1.5 numpy numcodecs tensorstore pytest
    micromamba run -n crop_verify python -m pip install "ome-zarr-models==1.7"
    micromamba run -n crop_verify python verification/check_plugin_compat.py

(pytest is only there because the shared fixtures in ``tests/conftest.py``
import it. ome-zarr-models is pip-only.)
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / 'tests'))

from plugin_reader import (  # noqa: E402
    classify_axes, find_multiscales, get_resolution_and_offset, is_zarr_array,
    open_ts_array,
)


def plugin_read(container, group, level, limits=None):
    """Reproduce ``ZarrRead.compute()``.

    Returns (array in Amira's (x, y, z) order, bbox_starts, bbox_stops).
    ``limits`` maps axis name -> (start, stop) in the level's index space, as
    the X/Y/Z limit ports do; None reads the whole array.
    """
    array_dir = os.path.join(container, group, level)
    assert is_zarr_array(array_dir), '{0} is not a zarr array to the plugin'.format(array_dir)

    dataset = open_ts_array(array_dir)
    ndim = len(dataset.shape)
    find_result = find_multiscales(os.path.join(container, group), container)
    multiscales = find_result[0]
    assert multiscales is not None, 'plugin found no multiscales for {0}'.format(array_dir)

    classified, err = classify_axes(multiscales[0].get('axes') or [], ndim)
    assert classified is not None, err
    axis_types, axis_names, spatial_indices, storage_axes, _, _, _ = classified
    resolution, offset, units = get_resolution_and_offset(
        array_dir, ndim, find_result, axis_types=axis_types)

    slices = {}
    for ax in ('x', 'y', 'z'):
        size = dataset.shape[spatial_indices[storage_axes.index(ax)]]
        slices[ax] = slice(*limits[ax]) if (limits and ax in limits) else slice(0, size)

    raw = dataset[tuple(slices[axis_names[i]] for i in range(ndim))].read().result()
    perm = [storage_axes.index(ax) for ax in ('x', 'y', 'z')]
    array = np.transpose(raw, perm)

    resolution_xyz = [resolution[storage_axes.index(ax)] for ax in ('x', 'y', 'z')]
    offset_xyz = [offset[storage_axes.index(ax)] for ax in ('x', 'y', 'z')]
    slices_xyz = [slices[ax] for ax in ('x', 'y', 'z')]
    shape_native = [(s - 1) * r for s, r in zip(array.shape, resolution_xyz)]
    starts = tuple(r * s.start + o
                   for r, s, o in zip(resolution_xyz, slices_xyz, offset_xyz))
    stops = tuple(o + s for o, s in zip(starts, shape_native))
    return array, starts, stops


def run_crop(args):
    """Invoke the CLI in this repo, in whatever env has it installed."""
    cmd = [sys.executable, '-c',
           'import sys; from crop_tool.cli import main; sys.exit(main(sys.argv[1:]))'] + args
    env = dict(os.environ, PYTHONPATH=str(REPO / 'src'))
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise SystemExit('zarr-crop failed:\n{0}\n{1}'.format(result.stdout, result.stderr))
    return result.stdout


def compare(label, src, group, level, limits, crop_container, crop_group):
    print('=' * 72)
    print(label)
    print('=' * 72)
    a_arr, a_start, a_stop = plugin_read(src, group, level, limits)
    b_arr, b_start, b_stop = plugin_read(crop_container, crop_group, 's0')

    print('  A) ZarrRead source, limits {0}'.format(
        {k: list(v) for k, v in limits.items()}))
    print('       shape {0}  bbox {1} -> {2}'.format(
        a_arr.shape, [round(v, 3) for v in a_start], [round(v, 3) for v in a_stop]))
    print('  B) ZarrRead the zarr-crop output')
    print('       shape {0}  bbox {1} -> {2}'.format(
        b_arr.shape, [round(v, 3) for v in b_start], [round(v, 3) for v in b_stop]))

    assert a_arr.shape == b_arr.shape, 'shape differs'
    assert np.array_equal(a_arr, b_arr), 'voxel values differ'
    assert a_start == b_start, 'bbox origin differs: {0} vs {1}'.format(a_start, b_start)
    assert a_stop == b_stop, 'bbox extent differs: {0} vs {1}'.format(a_stop, b_stop)
    print('  => identical voxels and identical Amira bounding box: OK\n')


def main():
    from conftest import PYRAMID_LEVELS, ZYX, write_container

    tmp = Path(tempfile.mkdtemp(prefix='crop_verify_'))
    try:
        for fmt, label in ((3, 'v3'), (2, 'v2')):
            src = tmp / 'src_{0}.zarr'.format(label)
            write_container(src, PYRAMID_LEVELS, ZYX, zarr_format=fmt)

            # Cut a coarse level from s0 coordinates -- the case the protocol
            # currently handles by guessing a divisor.
            out = tmp / 'crop_{0}.zarr'.format(label)
            run_crop(['--source', str(src / 'em' / 's2'), '--out', str(out),
                      '--min', '8,4,8', '--max', '40,28,32',
                      '--axis-order', 'zyx', '--roi-space', 's0'])
            compare('Zarr {0}: cropping s2 from s0 coordinates'.format(label),
                    str(src), 'em', 's2',
                    {'z': (1, 10), 'y': (0, 7), 'x': (1, 8)},
                    str(out), out.name[:-len('.zarr')])

            # And the plain s0 case.
            out0 = tmp / 'crop_{0}_s0.zarr'.format(label)
            run_crop(['--source', str(src / 'em' / 's0'), '--out', str(out0),
                      '--min', '4,6,8', '--max', '12,14,16',
                      '--axis-order', 'zyx', '--roi-space', 's0'])
            compare('Zarr {0}: cropping s0'.format(label),
                    str(src), 'em', 's0',
                    {'z': (4, 12), 'y': (6, 14), 'x': (8, 16)},
                    str(out0), out0.name[:-len('.zarr')])

        print('Plugin compatibility holds for zarr v2 and v3.')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
