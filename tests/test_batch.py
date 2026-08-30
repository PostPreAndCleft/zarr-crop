"""Cropping from a Neuroglancer link, one or many boxes at a time."""
from __future__ import annotations

import json

import numpy as np
import pytest
import zarr

from crop_tool import cli
from test_crop import dataset_transform, read_multiscales

# The pyramid fixture's s0 transform: scale (10, 20, 30) nm, translation
# (100, 200, 300) nm, in (z, y, x). The non-zero translation is what makes these
# tests meaningful -- see test_link_coordinates_go_through_physical_space.
S0_SCALE = {'z': 10.0, 'y': 20.0, 'x': 30.0}
S0_TRANSLATION = {'z': 100.0, 'y': 200.0, 'x': 300.0}


def crop(args) -> int:
    return cli.main([str(a) for a in args])


def read(out_path, name):
    return zarr.open(str(out_path / name / 's0'), mode='r')


def groups(out_path):
    return sorted(p.name for p in out_path.iterdir() if p.is_dir())


def voxel_to_state_coord(axis, index):
    """Where Neuroglancer would put voxel ``index`` in a space whose unit equals
    that axis's s0 voxel size.

    A voxel's physical position is ``translation + index * scale``, and the
    coordinate space's unit is ``scale`` nm, so the coordinate is
    ``translation / scale + index`` -- i.e. offset from the index by
    ``translation / scale``. Ignoring that offset is precisely the bug this
    conversion avoids.
    """
    return S0_TRANSLATION[axis] / S0_SCALE[axis] + index


def make_state(boxes_zyx, scale_nm=None, descriptions=None, level=None,
               unitless=False, source_url=None):
    """Build a Neuroglancer state whose coordinate space matches the fixture.

    ``boxes_zyx`` is a list of ((z0, y0, x0), (z1, y1, x1)) in *state* units.
    ``level`` makes the image layer point at that level of a container, as
    Fileglancer does when you open a level folder rather than the container root;
    ``unitless`` drops the unit from ``dimensions``, which is what happens in that
    case because a bare array carries no OME metadata.
    """
    scale_nm = scale_nm or S0_SCALE
    annotations = []
    for i, (lo, hi) in enumerate(boxes_zyx):
        ann = {
            'type': 'axis_aligned_bounding_box',
            # Deliberately reversed, as Neuroglancer emits draw order.
            'pointA': list(hi),
            'pointB': list(lo),
            'id': 'box{0}'.format(i),
        }
        if descriptions and i < len(descriptions) and descriptions[i]:
            ann['description'] = descriptions[i]
        annotations.append(ann)
    if unitless:
        dimensions = {a: [1, ''] for a in ('z', 'y', 'x')}
    else:
        dimensions = {a: [scale_nm[a] * 1e-9, 'm'] for a in ('z', 'y', 'x')}

    if source_url is None:
        source_url = 'https://s3/unused.zarr/'
        if level:
            source_url += '{0}/'.format(level)
        source_url += '|zarr3:'

    return {
        'dimensions': dimensions,
        'layers': [
            {'type': 'image', 'source': source_url, 'name': level or 'em'},
            {'type': 'annotation', 'source': {'url': 'local://annotations'},
             'name': 'boxes', 'annotations': annotations},
        ],
    }


def state_box_for_indices(lo_idx, hi_idx):
    """A state-unit box that should resolve back to the given s0 indices."""
    axes = ('z', 'y', 'x')
    return (tuple(voxel_to_state_coord(a, i) for a, i in zip(axes, lo_idx)),
            tuple(voxel_to_state_coord(a, i) for a, i in zip(axes, hi_idx)))


# ---------------------------------------------------------------------------
# The coordinate-space conversion
# ---------------------------------------------------------------------------

def test_link_coordinates_go_through_physical_space(pyramid_v3, tmp_path):
    """Link coordinates are not s0 indices when the source has a translation.

    The fixture's s0 translation is (100, 200, 300) nm with scale (10, 20, 30),
    so Neuroglancer coordinates sit 10 units above the voxel index on every axis.
    Treating them as indices would shift the crop by 10 voxels; converting
    through physical units lands on the right voxels.
    """
    lo, hi = state_box_for_indices((4, 6, 8), (12, 14, 16))
    assert lo == (14.0, 16.0, 18.0)      # not (4, 6, 8) -- the offset is real
    state = make_state([(lo, hi)])
    link = tmp_path / 'state.json'
    link.write_text(json.dumps(state))

    out = tmp_path / 'crops.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0', '--out', out,
                 '--neuroglancer', link, '--name', 'c']) == 0

    expected = pyramid_v3['data']['s0'][4:12, 6:14, 8:16]
    assert np.array_equal(read(out, 'c')[...], expected)


def test_forcing_s0_space_reproduces_the_naive_reading(pyramid_v3, tmp_path):
    """--roi-space s0 overrides the conversion, and gives a different answer."""
    lo, hi = state_box_for_indices((4, 6, 8), (12, 14, 16))
    link = tmp_path / 'state.json'
    link.write_text(json.dumps(make_state([(lo, hi)])))

    out = tmp_path / 'crops.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0', '--out', out,
                 '--neuroglancer', link, '--name', 'c', '--roi-space', 's0']) == 0

    # Coordinates taken literally as indices: [14:22] etc.
    expected = pyramid_v3['data']['s0'][14:22, 16:24, 18:26]
    assert np.array_equal(read(out, 'c')[...], expected)


def test_link_crop_from_a_coarser_level(pyramid_v3, tmp_path):
    """The same pasted link, cutting s1 instead of s0."""
    lo, hi = state_box_for_indices((4, 6, 8), (12, 14, 16))
    link = tmp_path / 'state.json'
    link.write_text(json.dumps(make_state([(lo, hi)])))

    out = tmp_path / 'crops.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's1', '--out', out,
                 '--neuroglancer', link, '--name', 'c']) == 0

    # Same physical region as test_crop_at_s1_using_s0_coords: s1 [1:6, 2:7, 3:8].
    assert np.array_equal(read(out, 'c')[...],
                          pyramid_v3['data']['s1'][1:6, 2:7, 3:8])


def test_link_without_unit_scale_falls_back_to_indices(pyramid_v3, tmp_path, capsys):
    state = make_state([((4, 6, 8), (12, 14, 16))])
    state['dimensions'] = {a: [1, ''] for a in ('z', 'y', 'x')}
    link = tmp_path / 'state.json'
    link.write_text(json.dumps(state))

    out = tmp_path / 'crops.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0', '--out', out,
                 '--neuroglancer', link, '--name', 'c']) == 0
    assert 'treated as s0 indices' in capsys.readouterr().err
    assert np.array_equal(read(out, 'c')[...],
                          pyramid_v3['data']['s0'][4:12, 6:14, 8:16])


# ---------------------------------------------------------------------------
# A box drawn on a level is cut from that level
#
# Opening a level folder rather than the container root gives Neuroglancer a bare
# array: no OME metadata, so no units, and the coordinate space is that level's
# raw voxel indices. The crop should follow the level it was drawn on.
# ---------------------------------------------------------------------------

def s1_link(tmp_path, boxes_s1):
    link = tmp_path / 'state.json'
    link.write_text(json.dumps(make_state(boxes_s1, level='s1', unitless=True)))
    return link


def test_box_drawn_on_s1_is_cut_from_s1(pyramid_v3, tmp_path):
    """The default: what you drew is what you get, at the resolution you drew it.

    Cutting the drawn level is the identity -- no conversion, whatever the level's
    scale happens to be.
    """
    link = s1_link(tmp_path, [((4, 3, 5), (12, 8, 10))])
    out = tmp_path / 'crops.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's1', '--out', out,
                 '--neuroglancer', link, '--name', 'c']) == 0
    assert np.array_equal(read(out, 'c')[...],
                          pyramid_v3['data']['s1'][4:12, 3:8, 5:10])


def test_no_warning_when_the_level_is_known(pyramid_v3, tmp_path, capsys):
    link = s1_link(tmp_path, [((4, 3, 5), (12, 8, 10))])
    crop(['--source', pyramid_v3['container'] / 'em' / 's1',
          '--out', tmp_path / 'crops.zarr', '--neuroglancer', link,
          '--name', 'c', '--dry-run'])
    captured = capsys.readouterr()
    assert 'treated as s0 indices' not in captured.err
    assert "drawn on 's1', cutting 's1'" in captured.out


def test_box_drawn_on_s1_can_be_cut_from_s0(pyramid_v3, tmp_path):
    """The capability this adds: convert the drawn level's indices to another.

    Hand-worked for the fixture (s1 scale 20/40/60, translation 105/210/315;
    s0 scale 10/20/30, translation 100/200/300):
      z: 105 + 4*20  = 185 -> (185-100)/10 = 8.5  -> 8
         105 + 12*20 = 345 -> (345-100)/10 = 24.5 -> 25
      y: 210 + 3*40  = 330 -> (330-200)/20 = 6.5  -> 6
         210 + 8*40  = 530 -> (530-200)/20 = 16.5 -> 17
      x: 315 + 5*60  = 615 -> (615-300)/30 = 10.5 -> 10
         315 + 10*60 = 915 -> (915-300)/30 = 20.5 -> 21
    """
    link = s1_link(tmp_path, [((4, 3, 5), (12, 8, 10))])
    out = tmp_path / 'crops.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0', '--out', out,
                 '--neuroglancer', link, '--name', 'c']) == 0

    written = read(out, 'c')[...]
    assert written.shape == (17, 11, 11)
    assert np.array_equal(written, pyramid_v3['data']['s0'][8:25, 6:17, 10:21])

    # Neither wrong answer may sneak through: the raw numbers, or naive halving.
    assert not np.array_equal(
        written, pyramid_v3['data']['s0'][4:12, 3:8, 5:10])
    assert not np.array_equal(
        written, pyramid_v3['data']['s0'][8:24, 6:16, 10:20])


def test_converted_crop_covers_the_same_physical_region(pyramid_v3, tmp_path):
    """Cutting s1 and s0 from one link must describe the same place."""
    link = s1_link(tmp_path, [((4, 3, 5), (12, 8, 10))])
    boxes = {}
    for level in ('s1', 's0'):
        out = tmp_path / 'crops_{0}.zarr'.format(level)
        assert crop(['--source', pyramid_v3['container'] / 'em' / level,
                     '--out', out, '--neuroglancer', link, '--name', 'c']) == 0
        scale, translation = dataset_transform(read_multiscales(out, 'c'))
        shape = read(out, 'c').shape
        boxes[level] = (translation,
                        [t + s * n for t, s, n in zip(translation, scale, shape)])

    # s0 has finer voxels, so its cover is tighter; both must contain the s1 box's
    # interior and agree to within one s1 voxel on every face.
    for axis, s1_scale in enumerate((20.0, 40.0, 60.0)):
        assert abs(boxes['s1'][0][axis] - boxes['s0'][0][axis]) <= s1_scale
        assert abs(boxes['s1'][1][axis] - boxes['s0'][1][axis]) <= s1_scale


def test_unitless_link_with_unrecognizable_level_falls_back_to_s0(
        pyramid_v3, tmp_path, capsys):
    """Only guess when there is nothing better; say so when guessing."""
    link = tmp_path / 'state.json'
    link.write_text(json.dumps(make_state(
        [((4, 6, 8), (12, 14, 16))], unitless=True,
        source_url='https://s3/unused.zarr/em/|zarr3:')))
    out = tmp_path / 'crops.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0', '--out', out,
                 '--neuroglancer', link, '--name', 'c']) == 0
    assert 'treated as s0 indices' in capsys.readouterr().err
    assert np.array_equal(read(out, 'c')[...],
                          pyramid_v3['data']['s0'][4:12, 6:14, 8:16])


def test_units_take_precedence_over_the_drawn_level(pyramid_v3, tmp_path):
    """A container-root link carries units; that path is unaffected by all this."""
    lo, hi = state_box_for_indices((4, 6, 8), (12, 14, 16))
    link = tmp_path / 'state.json'
    link.write_text(json.dumps(make_state([(lo, hi)], level='s1')))
    out = tmp_path / 'crops.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0', '--out', out,
                 '--neuroglancer', link, '--name', 'c']) == 0
    assert np.array_equal(read(out, 'c')[...],
                          pyramid_v3['data']['s0'][4:12, 6:14, 8:16])


def test_roi_space_still_overrides_the_drawn_level(pyramid_v3, tmp_path):
    link = s1_link(tmp_path, [((4, 3, 5), (12, 8, 10))])
    out = tmp_path / 'crops.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0', '--out', out,
                 '--neuroglancer', link, '--name', 'c',
                 '--roi-space', 'level']) == 0
    assert np.array_equal(read(out, 'c')[...],
                          pyramid_v3['data']['s0'][4:12, 3:8, 5:10])


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------

def test_three_boxes_become_three_sibling_groups(pyramid_v3, tmp_path):
    boxes = [state_box_for_indices((0, 0, 0), (8, 8, 8)),
             state_box_for_indices((10, 10, 10), (20, 20, 20)),
             state_box_for_indices((30, 12, 4), (40, 20, 12))]
    link = tmp_path / 'state.json'
    link.write_text(json.dumps(make_state(boxes)))

    out = tmp_path / 'crops.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0', '--out', out,
                 '--neuroglancer', link, '--name-prefix', 'soma']) == 0

    assert groups(out) == ['soma_1', 'soma_2', 'soma_3']
    data = pyramid_v3['data']['s0']
    assert np.array_equal(read(out, 'soma_1')[...], data[0:8, 0:8, 0:8])
    assert np.array_equal(read(out, 'soma_2')[...], data[10:20, 10:20, 10:20])
    assert np.array_equal(read(out, 'soma_3')[...], data[30:40, 12:20, 4:12])


def test_start_number_continues_a_series(pyramid_v3, tmp_path):
    boxes = [state_box_for_indices((0, 0, 0), (8, 8, 8)),
             state_box_for_indices((10, 10, 10), (20, 20, 20))]
    link = tmp_path / 'state.json'
    link.write_text(json.dumps(make_state(boxes)))

    out = tmp_path / 'crops.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0', '--out', out,
                 '--neuroglancer', link, '--name-prefix', 'Pancreas5_soma',
                 '--start-number', '7']) == 0
    assert groups(out) == ['Pancreas5_soma_7', 'Pancreas5_soma_8']


def test_a_bad_box_does_not_cost_the_batch(pyramid_v3, tmp_path, capsys):
    """One out-of-bounds box is skipped; its neighbours are still written."""
    boxes = [state_box_for_indices((0, 0, 0), (8, 8, 8)),
             state_box_for_indices((5000, 5000, 5000), (5100, 5100, 5100)),
             state_box_for_indices((10, 10, 10), (20, 20, 20))]
    link = tmp_path / 'state.json'
    link.write_text(json.dumps(make_state(boxes)))

    out = tmp_path / 'crops.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0', '--out', out,
                 '--neuroglancer', link, '--name-prefix', 'soma']) == 1

    assert groups(out) == ['soma_1', 'soma_3']
    printed = capsys.readouterr().out
    assert '2 of 3' in printed
    assert 'soma_2' in printed
    data = pyramid_v3['data']['s0']
    assert np.array_equal(read(out, 'soma_1')[...], data[0:8, 0:8, 0:8])
    assert np.array_equal(read(out, 'soma_3')[...], data[10:20, 10:20, 10:20])


def test_existing_group_is_not_overwritten_in_a_batch(pyramid_v3, tmp_path):
    link = tmp_path / 'state.json'
    link.write_text(json.dumps(make_state(
        [state_box_for_indices((0, 0, 0), (8, 8, 8))])))
    out = tmp_path / 'crops.zarr'
    args = ['--source', pyramid_v3['container'] / 'em' / 's0', '--out', out,
            '--neuroglancer', link, '--name-prefix', 'soma']
    assert crop(args) == 0
    assert crop(args) == 1                       # refuses, does not clobber
    assert groups(out) == ['soma_1']


def test_name_with_several_boxes_is_rejected(pyramid_v3, tmp_path, capsys):
    boxes = [state_box_for_indices((0, 0, 0), (8, 8, 8)),
             state_box_for_indices((10, 10, 10), (20, 20, 20))]
    link = tmp_path / 'state.json'
    link.write_text(json.dumps(make_state(boxes)))
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0',
                 '--out', tmp_path / 'crops.zarr', '--neuroglancer', link,
                 '--name', 'only-one']) == 1
    assert '--name-prefix' in capsys.readouterr().err


def test_several_boxes_default_to_the_out_stem_as_prefix(pyramid_v3, tmp_path):
    boxes = [state_box_for_indices((0, 0, 0), (8, 8, 8)),
             state_box_for_indices((10, 10, 10), (20, 20, 20))]
    link = tmp_path / 'state.json'
    link.write_text(json.dumps(make_state(boxes)))
    out = tmp_path / 'mycrops.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0', '--out', out,
                 '--neuroglancer', link]) == 0
    assert groups(out) == ['mycrops_1', 'mycrops_2']


def test_box_description_is_reported(pyramid_v3, tmp_path, capsys):
    """Descriptions do not name the crop, but they are echoed for context."""
    link = tmp_path / 'state.json'
    link.write_text(json.dumps(make_state(
        [state_box_for_indices((0, 0, 0), (8, 8, 8))],
        descriptions=['left soma'])))
    crop(['--source', pyramid_v3['container'] / 'em' / 's0',
          '--out', tmp_path / 'crops.zarr', '--neuroglancer', link,
          '--name-prefix', 'soma', '--dry-run'])
    assert 'left soma' in capsys.readouterr().out


def test_no_boxes_in_the_link(pyramid_v3, tmp_path, capsys):
    state = make_state([])
    link = tmp_path / 'state.json'
    link.write_text(json.dumps(state))
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0',
                 '--out', tmp_path / 'crops.zarr', '--neuroglancer', link]) == 1
    assert 'No bounding-box' in capsys.readouterr().err


def test_dry_run_writes_nothing_for_a_batch(pyramid_v3, tmp_path):
    boxes = [state_box_for_indices((0, 0, 0), (8, 8, 8)),
             state_box_for_indices((10, 10, 10), (20, 20, 20))]
    link = tmp_path / 'state.json'
    link.write_text(json.dumps(make_state(boxes)))
    out = tmp_path / 'crops.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0', '--out', out,
                 '--neuroglancer', link, '--name-prefix', 'soma',
                 '--dry-run']) == 0
    assert not out.exists()


# ---------------------------------------------------------------------------
# Link as literal text, and source resolution
# ---------------------------------------------------------------------------

def test_link_can_be_passed_as_text_not_a_file(pyramid_v3, tmp_path):
    from urllib.parse import quote
    state = make_state([state_box_for_indices((4, 6, 8), (12, 14, 16))])
    url = 'https://neuroglancer-demo.appspot.com/#!' + quote(json.dumps(state))

    out = tmp_path / 'crops.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0', '--out', out,
                 '--neuroglancer', url, '--name', 'c']) == 0
    assert np.array_equal(read(out, 'c')[...],
                          pyramid_v3['data']['s0'][4:12, 6:14, 8:16])


def test_unresolvable_source_asks_for_source(pyramid_v3, tmp_path, capsys):
    """An S3-backed layer has no filesystem path, so --source is required."""
    link = tmp_path / 'state.json'
    link.write_text(json.dumps(make_state(
        [state_box_for_indices((0, 0, 0), (8, 8, 8))])))
    assert crop(['--out', tmp_path / 'crops.zarr', '--neuroglancer', link]) == 1
    err = capsys.readouterr().err
    assert 'no filesystem path' in err
    assert '--source' in err


def test_derived_path_that_does_not_exist_is_reported(tmp_path, capsys):
    """The Fileglancer URL->path rule is inferred, so a miss must be loud."""
    state = make_state([((0, 0, 0), (8, 8, 8))])
    state['layers'][0]['source'] = (
        'https://fileglancer.int.janelia.org/files/KEY/nope/missing.zarr/|zarr2:')
    link = tmp_path / 'state.json'
    link.write_text(json.dumps(state))
    assert crop(['--out', tmp_path / 'crops.zarr', '--neuroglancer', link]) == 1
    err = capsys.readouterr().err
    assert '/nope/missing.zarr' in err
    assert 'does not exist' in err


def test_source_immutable_across_a_batch(pyramid_v3, tmp_path):
    from test_crop import tree_fingerprint
    before = tree_fingerprint(pyramid_v3['container'])
    boxes = [state_box_for_indices((0, 0, 0), (8, 8, 8)),
             state_box_for_indices((10, 10, 10), (20, 20, 20))]
    link = tmp_path / 'state.json'
    link.write_text(json.dumps(make_state(boxes)))
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0',
                 '--out', tmp_path / 'crops.zarr', '--neuroglancer', link,
                 '--name-prefix', 'soma']) == 0
    assert tree_fingerprint(pyramid_v3['container']) == before


# ---------------------------------------------------------------------------
# Slab behavior on a real chunk grid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('max_memory', [0.001, 0.05, 100.0])
def test_result_is_independent_of_the_memory_budget(pyramid_v3, tmp_path, max_memory):
    """Slabbing is an IO strategy; it must not change the output."""
    out = tmp_path / 'crops.zarr'
    assert crop(['--source', pyramid_v3['container'] / 'em' / 's0', '--out', out,
                 '--min', '0,0,0', '--max', '80,30,40', '--axis-order', 'zyx',
                 '--roi-space', 'level', '--name', 'c',
                 '--max-memory', max_memory]) == 0
    assert np.array_equal(read(out, 'c')[...], pyramid_v3['data']['s0'])
