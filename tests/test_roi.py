"""ROI parsing and coordinate conversion."""
from __future__ import annotations

import json
from urllib.parse import quote

import pytest

from crop_tool import roi as roi_mod
from crop_tool.zarr_io import Transform

AXIS_INDEX = {'z': 0, 'y': 1, 'x': 2}
SIZES = {'z': 80, 'y': 30, 'x': 40}
STORAGE = ['z', 'y', 'x']


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

def test_parse_flags_respects_axis_order():
    """The same numbers mean different things under xyz and zyx."""
    as_xyz = roi_mod.parse_flags('1,2,3', '4,5,6', 'xyz', 's0')
    as_zyx = roi_mod.parse_flags('1,2,3', '4,5,6', 'zyx', 's0')
    assert (as_xyz.start['x'], as_xyz.start['z']) == (1.0, 3.0)
    assert (as_zyx.start['x'], as_zyx.start['z']) == (3.0, 1.0)


@pytest.mark.parametrize('order', ['xy', 'xyzz', 'abc', 'xxy', ''])
def test_bad_axis_order_rejected(order):
    with pytest.raises(roi_mod.RoiError, match='permutation'):
        roi_mod.normalize_axis_order(order)


@pytest.mark.parametrize('text', ['1,2', '1,2,3,4', '1,two,3'])
def test_bad_triples_rejected(text):
    with pytest.raises(roi_mod.RoiError):
        roi_mod.parse_flags(text, '9,9,9', 'zyx', 's0')


# ---------------------------------------------------------------------------
# Neuroglancer input
# ---------------------------------------------------------------------------

BOX = {
    'type': 'axis_aligned_bounding_box',
    # Deliberately not sorted: pointB is below pointA on every axis. These are
    # in xyz order, matching the state's dimensions below.
    'pointA': [40.0, 25.0, 12.0],
    'pointB': [10.0, 5.0, 4.0],
    'id': 'abc123',
}

STATE = {
    'dimensions': {'x': [8e-9, 'm'], 'y': [8e-9, 'm'], 'z': [8e-9, 'm']},
    'layers': [
        {'type': 'image', 'name': 'em'},
        {'type': 'annotation', 'name': 'annotations', 'annotations': [BOX]},
    ],
}


def test_unsorted_points_become_min_max():
    """pointA/pointB come in draw order; they must be reduced per axis."""
    roi = roi_mod.parse_neuroglancer(json.dumps(STATE), 's0')
    assert roi.start == {'x': 10.0, 'y': 5.0, 'z': 4.0}
    assert roi.stop == {'x': 40.0, 'y': 25.0, 'z': 12.0}


def test_all_input_forms_agree():
    """Bare annotation, full state, and long URL must give the same ROI."""
    from_state = roi_mod.parse_neuroglancer(json.dumps(STATE), 's0')
    from_bare = roi_mod.parse_neuroglancer(json.dumps(BOX), 's0', axis_order='xyz')
    url = 'https://neuroglancer-demo.appspot.com/#!' + quote(json.dumps(STATE))
    from_url = roi_mod.parse_neuroglancer(url, 's0')

    assert from_state.start == from_bare.start == from_url.start
    assert from_state.stop == from_bare.stop == from_url.stop


def test_axis_order_read_from_state_dimensions():
    """A zyx-dimensioned state must not be read as xyz."""
    state = dict(STATE)
    state['dimensions'] = {'z': [8e-9, 'm'], 'y': [8e-9, 'm'], 'x': [8e-9, 'm']}
    roi = roi_mod.parse_neuroglancer(json.dumps(state), 's0')
    # pointA/pointB are now interpreted z, y, x.
    assert roi.start == {'z': 10.0, 'y': 5.0, 'x': 4.0}


def test_bare_annotation_without_axis_order_is_an_error():
    with pytest.raises(roi_mod.RoiError, match='axis order'):
        roi_mod.parse_neuroglancer(json.dumps(BOX), 's0')


def test_shortened_link_gives_actionable_error():
    with pytest.raises(roi_mod.RoiError, match='shortened'):
        roi_mod.parse_neuroglancer('https://ngl.short/#!abc123XYZ', 's0')


def test_no_bounding_box_found():
    state = {'dimensions': {'x': [1, 'm'], 'y': [1, 'm'], 'z': [1, 'm']},
             'layers': [{'type': 'image', 'name': 'em'}]}
    with pytest.raises(roi_mod.RoiError, match='No bounding-box'):
        roi_mod.parse_neuroglancer(json.dumps(state), 's0')


def test_multiple_boxes_rejected():
    state = json.loads(json.dumps(STATE))
    state['layers'][1]['annotations'].append(dict(BOX, id='second'))
    with pytest.raises(roi_mod.RoiError, match='one crop per run'):
        roi_mod.parse_neuroglancer(json.dumps(state), 's0')


# ---------------------------------------------------------------------------
# Coordinate conversion -- the divisor problem
#
# Expected values below are worked out by hand rather than by re-running the
# implementation's formula, so the test is a genuine check.
# ---------------------------------------------------------------------------

S0 = Transform(scale=[10.0, 20.0, 30.0], translation=[100.0, 200.0, 300.0])
S1 = Transform(scale=[20.0, 40.0, 60.0], translation=[105.0, 210.0, 315.0])


def resolve(roi, level, reference=S0):
    return roi_mod.to_level_indices(roi, level, reference, STORAGE, AXIS_INDEX, SIZES)


def test_s0_coords_at_s0_are_the_identity():
    roi = roi_mod.parse_flags('4,6,8', '12,14,16', 'zyx', 's0')
    out = resolve(roi, S0)
    assert out.start == {'z': 4, 'y': 6, 'x': 8}
    assert out.stop == {'z': 12, 'y': 14, 'x': 16}


def test_s0_coords_convert_to_s1_absorbing_the_half_voxel_offset():
    """Hand-worked: z start 4 in s0 lands at 1.75 in s1, not at 2.

    physical(z=4) = 100 + 4 * 10 = 140
    index_s1      = (140 - 105) / 20 = 1.75  -> floor -> 1
    physical(z=12) = 100 + 12 * 10 = 220
    index_s1      = (220 - 105) / 20 = 5.75  -> ceil  -> 6

    A plain "divide by 2" would have said [2, 6) and silently shifted the crop
    by one s1 voxel. Same arithmetic on the other axes:
      y: (200 + 6*20 - 210)/40  = 2.75 -> 2 ; (200 + 14*20 - 210)/40 = 6.75 -> 7
      x: (300 + 8*30 - 315)/60  = 3.75 -> 3 ; (300 + 16*30 - 315)/60 = 7.75 -> 8
    """
    roi = roi_mod.parse_flags('4,6,8', '12,14,16', 'zyx', 's0')
    out = resolve(roi, S1)
    assert out.start == {'z': 1, 'y': 2, 'x': 3}
    assert out.stop == {'z': 6, 'y': 7, 'x': 8}


def test_naive_division_would_have_disagreed():
    """Guard the point of the exercise: the naive answer is genuinely different."""
    roi = roi_mod.parse_flags('4,6,8', '12,14,16', 'zyx', 's0')
    exact = resolve(roi, S1)
    naive = {'z': 4 // 2, 'y': 6 // 2, 'x': 8 // 2}
    assert exact.start != naive


def test_level_space_passes_through_untouched():
    """--roi-space level is plugin parity: no conversion at all."""
    roi = roi_mod.parse_flags('4,6,8', '12,14,16', 'zyx', 'level')
    out = resolve(roi, S1)
    assert out.start == {'z': 4, 'y': 6, 'x': 8}
    assert out.stop == {'z': 12, 'y': 14, 'x': 16}


def test_physical_space():
    """physical(z) = 140 -> s0 index 4; 220 -> 12."""
    roi = roi_mod.parse_flags('140,320,540', '220,480,780', 'zyx', 'physical')
    out = resolve(roi, S0)
    assert out.start == {'z': 4, 'y': 6, 'x': 8}
    assert out.stop == {'z': 12, 'y': 14, 'x': 16}


def test_bounds_are_floored_and_ceiled_to_cover_the_roi():
    """A fractional ROI must never shave voxels off the requested region."""
    roi = roi_mod.Roi(start={'z': 4.6, 'y': 0.2, 'x': 0.9},
                      stop={'z': 10.1, 'y': 5.5, 'x': 6.5}, space='level')
    out = resolve(roi, S0)
    assert out.start == {'z': 4, 'y': 0, 'x': 0}
    assert out.stop == {'z': 11, 'y': 6, 'x': 7}


def test_float_noise_does_not_add_a_voxel():
    """Unit conversion is lossy in binary; ceil() must not amplify that.

    A Neuroglancer space of 3e-8 m is 30.000000000000004 nm, not 30. Converting a
    box through it yields a stop index like 16.000000000000004, and a naive ceil
    would return 17 -- silently making every crop a voxel too big on that axis.
    """
    noisy_scale = 3e-8 * 1e9          # 30.000000000000004, as the real code sees it
    assert noisy_scale != 30.0
    physical_stop = 26.0 * noisy_scale

    roi = roi_mod.Roi(start={'z': 100.0, 'y': 320.0, 'x': 540.0},
                      stop={'z': 220.0, 'y': 480.0, 'x': physical_stop},
                      space='physical')
    out = resolve(roi, S0)
    assert out.stop['x'] == 16, 'float noise leaked into the index'


def test_genuinely_fractional_bounds_still_expand():
    """The tolerance must not swallow real fractions."""
    roi = roi_mod.Roi(start={'z': 4.0, 'y': 6.0, 'x': 8.0},
                      stop={'z': 12.5, 'y': 14.0, 'x': 16.0}, space='level')
    assert resolve(roi, S0).stop['z'] == 13


# ---------------------------------------------------------------------------
# Regression: the real Fileglancer link, drawn on s1
#
# A link made by opening the s1 folder of the sample dataset on
# /Volumes/projtechres/andy_crop_test. The dataset's real transforms are
# scale 8/16/32 nm with translation 0/4/12 nm. Box 1 was drawn round sphere C.
# The numbers below were recorded from a verified run and checked against the
# data: 7442 voxels of sphere C's value at s1, 65267 at s0.
# ---------------------------------------------------------------------------

SAMPLE_S0 = Transform(scale=[8.0] * 3, translation=[0.0] * 3)
SAMPLE_S1 = Transform(scale=[16.0] * 3, translation=[4.0] * 3)
SAMPLE_SIZES_S1 = {'z': 256, 'y': 192, 'x': 128}
SAMPLE_SIZES_S0 = {'z': 512, 'y': 384, 'x': 256}

REAL_BOX = roi_mod.Roi(
    start={'z': 143.7732391357422, 'y': 30.888975143432617, 'x': 35.90521240234375},
    stop={'z': 175.7732391357422, 'y': 59.32911682128906, 'x': 64.66036224365234},
    space='drawn', source='neuroglancer')


def test_real_s1_link_cut_from_s1_is_the_identity():
    out = roi_mod.to_level_indices(REAL_BOX, SAMPLE_S1, SAMPLE_S1, STORAGE,
                                  AXIS_INDEX, SAMPLE_SIZES_S1)
    assert (out.start, out.stop) == (
        {'z': 143, 'y': 30, 'x': 35}, {'z': 176, 'y': 60, 'x': 65})


def test_real_s1_link_cut_from_s0_converts():
    """Hand-worked: s1 index 143.7732 is physical 4 + 143.7732*16 = 2304.37 nm,
    so s0 index 2304.37/8 = 288.05 -> 288. Stop 175.7732 -> 4 + 2812.37 =
    2816.37 -> /8 = 352.05 -> 353.
    """
    out = roi_mod.to_level_indices(REAL_BOX, SAMPLE_S0, SAMPLE_S1, STORAGE,
                                   AXIS_INDEX, SAMPLE_SIZES_S0)
    assert out.start['z'] == 288 and out.stop['z'] == 353
    assert out.start['y'] == 62 and out.stop['y'] == 120
    assert out.start['x'] == 72 and out.stop['x'] == 130

    # The two wrong answers this whole change exists to prevent.
    assert out.start['z'] != 143, 'raw drawn-level numbers used as s0 indices'
    assert out.start['z'] != 71, 'naive halving'


def test_real_link_conversion_preserves_the_physical_region():
    """Both cuts must describe the same place, within one s1 voxel per face."""
    at_s1 = roi_mod.to_level_indices(REAL_BOX, SAMPLE_S1, SAMPLE_S1, STORAGE,
                                     AXIS_INDEX, SAMPLE_SIZES_S1)
    at_s0 = roi_mod.to_level_indices(REAL_BOX, SAMPLE_S0, SAMPLE_S1, STORAGE,
                                     AXIS_INDEX, SAMPLE_SIZES_S0)
    for axis in STORAGE:
        i = AXIS_INDEX[axis]
        assert abs(SAMPLE_S1.physical(i, at_s1.start[axis])
                   - SAMPLE_S0.physical(i, at_s0.start[axis])) <= 16.0
        assert abs(SAMPLE_S1.physical(i, at_s1.stop[axis])
                   - SAMPLE_S0.physical(i, at_s0.stop[axis])) <= 16.0


def test_out_of_bounds_is_clamped_and_reported():
    roi = roi_mod.parse_flags('0,0,0', '999,14,16', 'zyx', 'level')
    out = resolve(roi, S0)
    assert out.stop['z'] == SIZES['z']
    assert 'z' in out.clamped


def test_inverted_roi_rejected():
    roi = roi_mod.parse_flags('12,6,8', '4,14,16', 'zyx', 'level')
    with pytest.raises(roi_mod.RoiError, match='empty along Z'):
        resolve(roi, S0)


def test_roi_entirely_outside_array_rejected():
    roi = roi_mod.parse_flags('500,0,0', '600,14,16', 'zyx', 'level')
    with pytest.raises(roi_mod.RoiError, match='does not overlap'):
        resolve(roi, S0)


def test_s0_space_without_multiscales_is_an_actionable_error():
    roi = roi_mod.parse_flags('4,6,8', '12,14,16', 'zyx', 's0')
    with pytest.raises(roi_mod.RoiError, match='--roi-space level'):
        roi_mod.to_level_indices(roi, S0, None, STORAGE, AXIS_INDEX, SIZES)
