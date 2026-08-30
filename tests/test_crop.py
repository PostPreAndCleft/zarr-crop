"""End-to-end tests: run the CLI, then check the crop and its metadata."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from crop_tool import cli


def crop(args) -> int:
    return cli.main([str(a) for a in args])


def _group_name(out_path, name=None):
    """Default group name matches the CLI's: the --out filename minus .zarr."""
    return name if name is not None else Path(out_path).name[:-len('.zarr')]


def read_output(out_path, name=None):
    return zarr.open(str(Path(out_path) / _group_name(out_path, name) / 's0'), mode='r')


def read_multiscales(out_path, name=None):
    grp = zarr.open_group(str(Path(out_path) / _group_name(out_path, name)), mode='r')
    attrs = dict(grp.attrs)
    if grp.metadata.zarr_format == 3:
        return attrs['ome']['multiscales'][0]
    return attrs['multiscales'][0]


def dataset_transform(entry):
    scale = translation = None
    for ct in entry['datasets'][0]['coordinateTransformations']:
        if ct['type'] == 'scale':
            scale = ct['scale']
        elif ct['type'] == 'translation':
            translation = ct['translation']
    return scale, translation


# ---------------------------------------------------------------------------
# The data actually landing in the right place
# ---------------------------------------------------------------------------

def test_crop_at_s0_matches_a_plain_numpy_slice(pyramid_v3, tmp_path):
    out = tmp_path / 'crop.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0',
                 '--out', out, '--min', '4,6,8', '--max', '12,14,16',
                 '--axis-order', 'zyx', '--roi-space', 's0']) == 0

    expected = pyramid_v3['data']['s0'][4:12, 6:14, 8:16]
    assert np.array_equal(read_output(out)[...], expected)


def test_crop_at_s1_using_s0_coords(pyramid_v3, tmp_path):
    """The divisor test.

    s0 ROI z=[4,12) maps to s1 [1, 6) once the half-voxel level offset is taken
    into account (worked out by hand in test_roi.py). y=[6,14) -> [2, 7),
    x=[8,16) -> [3, 8). Naive halving would have given z=[2, 6), silently
    shifting the crop.
    """
    out = tmp_path / 'crop.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's1',
                 '--out', out, '--min', '4,6,8', '--max', '12,14,16',
                 '--axis-order', 'zyx', '--roi-space', 's0']) == 0

    expected = pyramid_v3['data']['s1'][1:6, 2:7, 3:8]
    written = read_output(out)[...]
    assert written.shape == (5, 5, 5)
    assert np.array_equal(written, expected)

    naive = pyramid_v3['data']['s1'][2:6, 3:7, 4:8]
    assert not np.array_equal(written, naive)


def test_crop_at_s2_using_s0_coords(pyramid_v3, tmp_path):
    """physical(z=8) = 180; index_s2 = (180 - 115)/40 = 1.625 -> 1.
    physical(z=40) = 500; index_s2 = (500 - 115)/40 = 9.625 -> 10.
    y: (200+80-230)/80 = 0.625 -> 0 ; (200+560-230)/80 = 6.625 -> 7.
    x: (300+240-345)/120 = 1.625 -> 1 ; (300+960-345)/120 = 7.625 -> 8.
    """
    out = tmp_path / 'crop.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's2',
                 '--out', out, '--min', '8,4,8', '--max', '40,28,32',
                 '--axis-order', 'zyx', '--roi-space', 's0']) == 0

    expected = pyramid_v3['data']['s2'][1:10, 0:7, 1:8]
    assert np.array_equal(read_output(out)[...], expected)


def test_storage_axis_order_does_not_change_the_result(same_volume_two_orders, tmp_path):
    """A zyx-stored and an xyz-stored copy of one volume must crop identically."""
    results = {}
    for label in ('zyx', 'xyz'):
        out = tmp_path / 'crop_{0}.zarr'.format(label)
        assert crop(['--source', same_volume_two_orders[label] / 'em' / 's0',
                     '--out', out, '--min', '2,3,4', '--max', '8,9,10',
                     '--axis-order', 'zyx', '--roi-space', 's0']) == 0
        results[label] = read_output(out)[...]

    assert np.array_equal(results['zyx'], results['xyz'])
    # and both equal the plain slice of the reference volume
    assert np.array_equal(results['zyx'],
                          same_volume_two_orders['volume'][2:8, 3:9, 4:10])


def test_crop_spanning_multiple_slabs(pyramid_v3, tmp_path):
    """The slab loop must stitch correctly; SLAB is 64 and this ROI is 80 deep."""
    out = tmp_path / 'crop.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0',
                 '--out', out, '--min', '0,0,0', '--max', '80,30,40',
                 '--axis-order', 'zyx', '--roi-space', 'level']) == 0

    assert np.array_equal(read_output(out)[...], pyramid_v3['data']['s0'])


def test_channel_selection(four_d_czyx, tmp_path):
    out = tmp_path / 'crop.zarr'
    assert crop(['--source', four_d_czyx['container'] / 'em' / 's0',
                 '--out', out, '--min', '2,3,4', '--max', '8,9,10',
                 '--axis-order', 'zyx', '--roi-space', 'level',
                 '--channel', '2']) == 0

    expected = four_d_czyx['data'][2, 2:8, 3:9, 4:10]
    assert np.array_equal(read_output(out)[...], expected)


def test_channel_out_of_range_rejected(four_d_czyx, tmp_path):
    assert crop(['--source', four_d_czyx['container'] / 'em' / 's0',
                 '--out', tmp_path / 'crop.zarr', '--min', '2,3,4',
                 '--max', '8,9,10', '--axis-order', 'zyx',
                 '--roi-space', 'level', '--channel', '9']) == 1


def test_channel_on_3d_source_rejected(pyramid_v3, tmp_path):
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0',
                 '--out', tmp_path / 'crop.zarr', '--min', '2,3,4',
                 '--max', '8,9,10', '--axis-order', 'zyx',
                 '--roi-space', 'level', '--channel', '1']) == 1


def test_missing_axis_metadata_falls_back_to_zyx(no_axes_meta, tmp_path):
    out = tmp_path / 'crop.zarr'
    assert crop(['--source', no_axes_meta['container'] / 'em' / 's0',
                 '--out', out, '--min', '2,3,4', '--max', '8,9,10',
                 '--axis-order', 'zyx', '--roi-space', 'level']) == 0
    assert np.array_equal(read_output(out)[...], no_axes_meta['data'][2:8, 3:9, 4:10])


# ---------------------------------------------------------------------------
# Output metadata
# ---------------------------------------------------------------------------

def test_output_scale_and_translation(pyramid_v3, tmp_path):
    """Hand-worked expectation, matching what the Amira round-trip produces.

    ZarrRead computes bbox_starts = scale * slice.start + translation, and
    ZarrWrite writes that back as the output translation. For s0 start
    (z, y, x) = (4, 6, 8):
        z: 100 + 4 * 10 = 140
        y: 200 + 6 * 20 = 320
        x: 300 + 8 * 30 = 540
    Scale is carried over from the level unchanged.
    """
    out = tmp_path / 'crop.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0',
                 '--out', out, '--min', '4,6,8', '--max', '12,14,16',
                 '--axis-order', 'zyx', '--roi-space', 's0']) == 0

    scale, translation = dataset_transform(read_multiscales(out))
    assert scale == [10.0, 20.0, 30.0]
    assert translation == [140.0, 320.0, 540.0]


def test_output_translation_when_cropping_a_coarser_level(pyramid_v3, tmp_path):
    """s1 start index 1 -> 105 + 1 * 20 = 125 on z, 210 + 2 * 40 = 290 on y,
    315 + 3 * 60 = 495 on x."""
    out = tmp_path / 'crop.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's1',
                 '--out', out, '--min', '4,6,8', '--max', '12,14,16',
                 '--axis-order', 'zyx', '--roi-space', 's0']) == 0

    scale, translation = dataset_transform(read_multiscales(out))
    assert scale == [20.0, 40.0, 60.0]
    assert translation == [125.0, 290.0, 495.0]


@pytest.mark.parametrize('level', ['s0', 's1', 's2'])
def test_written_crop_always_covers_the_requested_region(pyramid_v3, tmp_path, level):
    """The guarantee a user relies on: the crop contains the ROI they drew.

    Cutting a coarser level cannot land on the exact ROI boundary, so bounds are
    floored and ceiled outward. That must always produce a superset of the
    requested physical region -- never a crop that clips it.
    """
    s0_scale, s0_translation = (10.0, 20.0, 30.0), (100.0, 200.0, 300.0)
    start_idx, stop_idx = (8, 4, 8), (40, 28, 32)
    want_lo = [t + i * s for t, i, s in zip(s0_translation, start_idx, s0_scale)]
    want_hi = [t + i * s for t, i, s in zip(s0_translation, stop_idx, s0_scale)]

    out = tmp_path / 'crop.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / level, '--out', out,
                 '--min', '8,4,8', '--max', '40,28,32',
                 '--axis-order', 'zyx', '--roi-space', 's0']) == 0

    scale, translation = dataset_transform(read_multiscales(out))
    shape = read_output(out).shape
    got_lo = translation
    got_hi = [t + s * n for t, s, n in zip(translation, scale, shape)]

    for axis, lo, hi, glo, ghi in zip('zyx', want_lo, want_hi, got_lo, got_hi):
        assert glo <= lo, '{0}: crop starts at {1}, inside the ROI at {2}'.format(
            axis, glo, lo)
        assert ghi >= hi, '{0}: crop ends at {1}, inside the ROI at {2}'.format(
            axis, ghi, hi)


def test_output_axes_are_always_zyx(pyramid_v3, tmp_path):
    out = tmp_path / 'crop.zarr'
    crop(['--source', pyramid_v3['container'] / 'em' / 's0', '--out', out,
          '--min', '4,6,8', '--max', '12,14,16', '--axis-order', 'zyx'])
    entry = read_multiscales(out)
    assert [ax['name'] for ax in entry['axes']] == ['z', 'y', 'x']
    assert all(ax['type'] == 'space' for ax in entry['axes'])
    assert all(ax['unit'] == 'nanometer' for ax in entry['axes'])


def test_metadata_validates_under_ome_zarr_models(pyramid_v3, pyramid_v2, tmp_path):
    from ome_zarr_models.v04.image import Multiscale as M4
    from ome_zarr_models.v05.image import Multiscale as M5

    out3 = tmp_path / 'crop3.zarr'
    crop(['--source', pyramid_v3['container'] / 'em' / 's0', '--out', out3,
          '--min', '4,6,8', '--max', '12,14,16', '--axis-order', 'zyx'])
    M5.model_validate(read_multiscales(out3))

    out2 = tmp_path / 'crop2.zarr'
    crop(['--source', pyramid_v2['container'] / 'em' / 's0', '--out', out2,
          '--min', '4,6,8', '--max', '12,14,16', '--axis-order', 'zyx'])
    entry = read_multiscales(out2)
    assert entry['version'] == '0.4'
    M4.model_validate(entry)


def test_output_format_defaults_to_source_format(pyramid_v2, tmp_path):
    out = tmp_path / 'crop.zarr'
    crop(['--source', pyramid_v2['container'] / 'em' / 's0', '--out', out,
          '--min', '4,6,8', '--max', '12,14,16', '--axis-order', 'zyx'])
    assert read_output(out).metadata.zarr_format == 2
    assert (out / 'crop' / 's0' / '.zarray').is_file()


def test_output_format_override(pyramid_v2, tmp_path):
    out = tmp_path / 'crop.zarr'
    crop(['--source', pyramid_v2['container'] / 'em' / 's0', '--out', out,
          '--min', '4,6,8', '--max', '12,14,16', '--axis-order', 'zyx',
          '--zarr-format', '3'])
    assert read_output(out).metadata.zarr_format == 3


def test_on_disk_format_matches_the_plugin_spec_v2(pyramid_v2, tmp_path):
    """The plugin's tensorstore v2 spec: zstd level 3, dimension_separator '/'."""
    out = tmp_path / 'crop.zarr'
    crop(['--source', pyramid_v2['container'] / 'em' / 's0', '--out', out,
          '--min', '4,6,8', '--max', '12,14,16', '--axis-order', 'zyx'])
    meta = json.loads((out / 'crop' / 's0' / '.zarray').read_text())
    assert meta['compressor'] == {'id': 'zstd', 'level': 3}
    assert meta['dimension_separator'] == '/'


def test_on_disk_format_matches_the_plugin_spec_v3(pyramid_v3, tmp_path):
    """The plugin's tensorstore v3 spec: bytes + zstd level 3, dimension_names zyx.

    zarr-python writes the two spec defaults tensorstore leaves implicit
    (endian little, checksum false); everything else must match.
    """
    out = tmp_path / 'crop.zarr'
    crop(['--source', pyramid_v3['container'] / 'em' / 's0', '--out', out,
          '--min', '4,6,8', '--max', '12,14,16', '--axis-order', 'zyx'])
    meta = json.loads((out / 'crop' / 's0' / 'zarr.json').read_text())
    assert [c['name'] for c in meta['codecs']] == ['bytes', 'zstd']
    zstd = [c for c in meta['codecs'] if c['name'] == 'zstd'][0]
    assert zstd['configuration']['level'] == 3
    assert meta['dimension_names'] == ['z', 'y', 'x']
    assert meta['chunk_grid']['configuration']['chunk_shape'] == [8, 8, 8]


def test_chunks_clamped_to_crop_shape(pyramid_v3, tmp_path):
    """A crop smaller than 128 must not get a 128-wide chunk grid."""
    out = tmp_path / 'crop.zarr'
    crop(['--source', pyramid_v3['container'] / 'em' / 's0', '--out', out,
          '--min', '0,0,0', '--max', '80,30,40', '--axis-order', 'zyx',
          '--roi-space', 'level'])
    assert read_output(out).chunks == (80, 30, 40)


def test_explicit_chunks(pyramid_v3, tmp_path):
    out = tmp_path / 'crop.zarr'
    crop(['--source', pyramid_v3['container'] / 'em' / 's0', '--out', out,
          '--min', '0,0,0', '--max', '80,30,40', '--axis-order', 'zyx',
          '--roi-space', 'level', '--chunks', '16,16,16'])
    assert read_output(out).chunks == (16, 16, 16)


def test_dtype_is_preserved_not_narrowed(pyramid_v3, tmp_path):
    """ZarrRead would have narrowed uint32 to uint16 for Amira's sake."""
    out = tmp_path / 'crop.zarr'
    crop(['--source', pyramid_v3['container'] / 'em' / 's0', '--out', out,
          '--min', '4,6,8', '--max', '12,14,16', '--axis-order', 'zyx'])
    assert read_output(out).dtype == np.dtype('uint32')


def test_high_uint32_values_survive(tmp_path):
    """The concrete risk of narrowing: values above 65535 would have wrapped."""
    from conftest import ZYX, write_container

    src = tmp_path / 'big_values.zarr'
    data = np.full((4, 4, 4), 4_000_000_000, dtype='uint32')
    write_container(src, [dict(path='s0', shape=(4, 4, 4), scale=(1.0, 1.0, 1.0),
                               translation=(0.0, 0.0, 0.0))],
                    ZYX, zarr_format=3, data_by_level={'s0': data})

    out = tmp_path / 'crop.zarr'
    assert crop(['--source', src / 'em' / 's0', '--out', out, '--min', '0,0,0',
                 '--max', '4,4,4', '--axis-order', 'zyx', '--roi-space', 'level']) == 0
    written = read_output(out)[...]
    assert written.dtype == np.dtype('uint32')
    assert written.min() == 4_000_000_000
    assert np.array_equal(written, data)


# ---------------------------------------------------------------------------
# The source must never be touched
# ---------------------------------------------------------------------------

def tree_fingerprint(root: Path):
    """Hash every file under root, with its size and mtime."""
    out = {}
    for path in sorted(Path(root).rglob('*')):
        if path.is_file():
            stat = path.stat()
            out[str(path.relative_to(root))] = (
                hashlib.sha256(path.read_bytes()).hexdigest(), stat.st_size, stat.st_mtime_ns)
    return out


def test_source_is_byte_identical_after_a_run(pyramid_v3, tmp_path):
    before = tree_fingerprint(pyramid_v3['container'])
    assert before  # guard against fingerprinting nothing

    out = tmp_path / 'crop.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0', '--out', out,
                 '--min', '4,6,8', '--max', '12,14,16', '--axis-order', 'zyx']) == 0

    assert tree_fingerprint(pyramid_v3['container']) == before


def test_output_inside_source_container_is_refused(pyramid_v3, capsys):
    inside = pyramid_v3['container'] / 'sneaky.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0', '--out', inside,
                 '--min', '4,6,8', '--max', '12,14,16', '--axis-order', 'zyx']) == 1
    assert 'never writes into raw data' in capsys.readouterr().err
    assert not inside.exists()


def test_output_equal_to_source_container_is_refused(pyramid_v3):
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0',
                 '--out', pyramid_v3['container'],
                 '--min', '4,6,8', '--max', '12,14,16', '--axis-order', 'zyx']) == 1


def test_dry_run_writes_nothing(pyramid_v3, tmp_path):
    out = tmp_path / 'crop.zarr'
    before = tree_fingerprint(pyramid_v3['container'])
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0', '--out', out,
                 '--min', '4,6,8', '--max', '12,14,16', '--axis-order', 'zyx',
                 '--dry-run']) == 0
    assert not out.exists()
    assert tree_fingerprint(pyramid_v3['container']) == before


def test_existing_crop_is_not_overwritten(pyramid_v3, tmp_path):
    out = tmp_path / 'crop.zarr'
    args = ['--source', pyramid_v3['container'] / 'em' / 's0', '--out', out,
            '--min', '4,6,8', '--max', '12,14,16', '--axis-order', 'zyx']
    assert crop(args) == 0
    assert crop(args) == 1


# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------

def test_axis_order_is_required_with_min_max(pyramid_v3, tmp_path, capsys):
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0',
                 '--out', tmp_path / 'crop.zarr',
                 '--min', '4,6,8', '--max', '12,14,16']) == 1
    assert 'axis-order' in capsys.readouterr().err


def test_out_must_end_in_zarr(pyramid_v3, tmp_path):
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0',
                 '--out', tmp_path / 'crop_folder', '--min', '4,6,8',
                 '--max', '12,14,16', '--axis-order', 'zyx']) == 1


def test_source_must_be_an_array_not_a_group(pyramid_v3, tmp_path, capsys):
    assert crop(['--source', pyramid_v3['container'] / 'em',
                 '--out', tmp_path / 'crop.zarr', '--min', '4,6,8',
                 '--max', '12,14,16', '--axis-order', 'zyx']) == 1
    assert 'not a zarr array' in capsys.readouterr().err


def test_source_outside_a_zarr_container_is_rejected(tmp_path):
    plain = tmp_path / 'not_zarr'
    plain.mkdir()
    assert crop(['--source', plain, '--out', tmp_path / 'crop.zarr',
                 '--min', '4,6,8', '--max', '12,14,16', '--axis-order', 'zyx']) == 1


def test_both_roi_sources_rejected(pyramid_v3, tmp_path):
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0',
                 '--out', tmp_path / 'crop.zarr', '--min', '4,6,8',
                 '--max', '12,14,16', '--axis-order', 'zyx',
                 '--neuroglancer', '-']) == 1


def test_no_roi_given_rejected(pyramid_v3, tmp_path):
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0',
                 '--out', tmp_path / 'crop.zarr']) == 1


def test_neuroglancer_file_input_end_to_end(pyramid_v3, tmp_path):
    """The realistic path: paste the Neuroglancer state, get the crop."""
    state = {
        'dimensions': {'x': [8e-9, 'm'], 'y': [8e-9, 'm'], 'z': [8e-9, 'm']},
        'layers': [{
            'type': 'annotation',
            'annotations': [{
                'type': 'axis_aligned_bounding_box',
                # xyz order, and deliberately unsorted
                'pointA': [16.0, 14.0, 12.0],
                'pointB': [8.0, 6.0, 4.0],
            }],
        }],
    }
    state_file = tmp_path / 'state.json'
    state_file.write_text(json.dumps(state))

    out = tmp_path / 'crop.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0', '--out', out,
                 '--neuroglancer', state_file, '--roi-space', 's0']) == 0

    expected = pyramid_v3['data']['s0'][4:12, 6:14, 8:16]
    assert np.array_equal(read_output(out)[...], expected)


def test_name_defaults_to_out_stem(pyramid_v3, tmp_path):
    out = tmp_path / 'Pancreas5_soma_7.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0', '--out', out,
                 '--min', '4,6,8', '--max', '12,14,16', '--axis-order', 'zyx']) == 0
    assert (out / 'Pancreas5_soma_7' / 's0').is_dir()


def test_explicit_name(pyramid_v3, tmp_path):
    out = tmp_path / 'crops.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0', '--out', out,
                 '--name', 'Pancreas5_soma_7', '--min', '4,6,8', '--max', '12,14,16',
                 '--axis-order', 'zyx']) == 0
    assert (out / 'Pancreas5_soma_7' / 's0').is_dir()


def test_summary_reports_the_bbox(pyramid_v3, tmp_path, capsys):
    """The printed summary is what gets pasted into the GitHub issue."""
    crop(['--source', pyramid_v3['container'] / 'em' / 's0',
          '--out', tmp_path / 'crop.zarr', '--min', '4,6,8', '--max', '12,14,16',
          '--axis-order', 'zyx', '--dry-run'])
    out = capsys.readouterr().out
    assert 'nothing written' in out
    assert '140.0' in out  # z translation
    assert 'index [4:12]' in out
