"""Neuroglancer state parsing.

The regression cases at the bottom come from a real link for the public CellMap
dataset ``jrc_amphiuma-means-liver-1``, cross-checked against that dataset's
published OME metadata.
"""
from __future__ import annotations

import json
from urllib.parse import quote

import pytest

from crop_tool import neuroglancer as ng
from crop_tool.zarr_io import aligned_slabs


# ---------------------------------------------------------------------------
# Source URLs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('raw,url,fmt', [
    ('https://h/a.zarr/|zarr3:', 'https://h/a.zarr', 3),
    ('https://h/a.zarr/|zarr2:', 'https://h/a.zarr', 2),
    ('https://h/a.zarr|zarr:', 'https://h/a.zarr', None),
    ('zarr://https://h/a.zarr', 'https://h/a.zarr', None),
    ('zarr2://https://h/a.zarr', 'https://h/a.zarr', 2),
    ('zarr3://https://h/a.zarr/', 'https://h/a.zarr', 3),
    ('https://h/a.zarr', 'https://h/a.zarr', None),
])
def test_source_url_forms(raw, url, fmt):
    """Both the documented zarr:// prefix and the pipe suffix real links use."""
    parsed = ng.parse_source(raw)
    assert (parsed.url, parsed.zarr_format) == (url, fmt)


def test_source_from_dict_and_list():
    assert ng.parse_source({'url': 'local://annotations'}).is_local
    assert ng.parse_source(['zarr://https://h/a.zarr']).url == 'https://h/a.zarr'
    assert ng.parse_source(None) is None
    assert ng.parse_source('') is None


# ---------------------------------------------------------------------------
# Fileglancer path derivation
# ---------------------------------------------------------------------------

def test_fileglancer_url_yields_a_filesystem_path():
    url = ('https://fileglancer.int.janelia.org/files/ekDViMrqI9F60zgt/groups/'
           'cellmap/cellmap/annotations/amira/jrc_amphiuma-means-liver-1/'
           'mito001/mito001.zarr')
    assert str(ng.fileglancer_path(url)) == (
        '/groups/cellmap/cellmap/annotations/amira/jrc_amphiuma-means-liver-1/'
        'mito001/mito001.zarr')


def test_fileglancer_rule_works_for_other_mounts():
    url = 'https://fileglancer.int.janelia.org/files/AbCd1234/nrs/projtechres/x.zarr'
    assert str(ng.fileglancer_path(url)) == '/nrs/projtechres/x.zarr'


@pytest.mark.parametrize('url', [
    'https://s3.janelia.org/cellmap/x.zarr',          # object storage, no path
    'https://fileglancer.int.janelia.org/api/things',  # not a /files/ URL
    'local://annotations',
    'not a url at all',
])
def test_non_fileglancer_urls_yield_no_path(url):
    assert ng.fileglancer_path(url) is None


# ---------------------------------------------------------------------------
# Coordinate space
# ---------------------------------------------------------------------------

def test_coordinate_space_reads_order_and_scale():
    state = {'dimensions': {'z': [8e-9, 'm'], 'y': [8e-9, 'm'], 'x': [8e-9, 'm']}}
    space = ng.coordinate_space(state)
    assert space.order == 'zyx'
    assert space.nm_per_unit == {'z': 8.0, 'y': 8.0, 'x': 8.0}
    assert space.has_scale


def test_coordinate_space_order_is_not_assumed():
    """xyz and zyx states must not be conflated -- the protocol's stated hazard."""
    xyz = ng.coordinate_space(
        {'dimensions': {'x': [1, 'nm'], 'y': [1, 'nm'], 'z': [1, 'nm']}})
    assert xyz.order == 'xyz'


def test_coordinate_space_anisotropic_units():
    state = {'dimensions': {'z': [30, 'nm'], 'y': [0.02, 'um'], 'x': [1e-8, 'm']}}
    space = ng.coordinate_space(state)
    assert space.nm_per_unit == {'z': 30.0, 'y': 20.0, 'x': 10.0}


def test_coordinate_space_without_units_has_no_scale():
    space = ng.coordinate_space(
        {'dimensions': {'z': [1, ''], 'y': [1, ''], 'x': [1, '']}})
    assert space is not None and not space.has_scale


def test_coordinate_space_ignores_non_spatial_dimensions():
    state = {'dimensions': {'t': [1, 's'], 'z': [1, 'nm'], 'y': [1, 'nm'],
                            'x': [1, 'nm']}}
    assert ng.coordinate_space(state).order == 'zyx'


def test_coordinate_space_missing_or_partial_is_none():
    assert ng.coordinate_space({}) is None
    assert ng.coordinate_space({'dimensions': {'z': [1, 'nm'], 'y': [1, 'nm']}}) is None


@pytest.mark.parametrize('unit,nm', [
    ('m', 1e9), ('mm', 1e6), ('um', 1e3), ('µm', 1e3), ('nm', 1.0),
    ('nanometer', 1.0), ('NM', 1.0), ('angstrom', 0.1),
])
def test_unit_conversion(unit, nm):
    assert ng.unit_to_nm(unit) == nm


@pytest.mark.parametrize('unit', ['', None, 'furlong', 'parsec'])
def test_unrecognized_units(unit):
    assert ng.unit_to_nm(unit) is None


# ---------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------

def box(a, b, **extra):
    return dict(type='axis_aligned_bounding_box', pointA=a, pointB=b, **extra)


def test_boxes_sorts_draw_order_per_axis():
    """pointA/pointB arrive in draw order, not sorted."""
    found = ng.boxes(box([40, 25, 12], [10, 5, 4]), 'zyx')
    assert len(found) == 1
    assert found[0].start == {'z': 10.0, 'y': 5.0, 'x': 4.0}
    assert found[0].stop == {'z': 40.0, 'y': 25.0, 'x': 12.0}


def test_multiple_boxes_are_all_returned_in_order():
    state = {'layers': [{'type': 'annotation', 'name': 'a', 'annotations': [
        box([0, 0, 0], [10, 10, 10], id='one'),
        box([20, 20, 20], [30, 30, 30], id='two', description='soma'),
    ]}]}
    found = ng.boxes(state, 'zyx')
    assert [b.id for b in found] == ['one', 'two']
    assert [b.description for b in found] == [None, 'soma']
    assert all(b.layer == 'a' for b in found)


def test_boxes_across_several_layers():
    state = {'layers': [
        {'type': 'annotation', 'name': 'first', 'annotations': [box([0]*3, [1]*3)]},
        {'type': 'image', 'name': 'em'},
        {'type': 'annotation', 'name': 'second', 'annotations': [box([2]*3, [3]*3)]},
    ]}
    found = ng.boxes(state, 'zyx')
    assert [b.layer for b in found] == ['first', 'second']


def test_non_box_annotations_are_ignored():
    state = {'layers': [{'type': 'annotation', 'annotations': [
        {'type': 'point', 'point': [1, 2, 3], 'id': 'p'},
        box([0]*3, [1]*3, id='b'),
    ]}]}
    assert [b.id for b in ng.boxes(state, 'zyx')] == ['b']


def test_too_few_coordinates_is_an_error():
    with pytest.raises(ng.NeuroglancerError, match='coordinate space'):
        ng.boxes(box([1, 2], [3, 4]), 'zyx')


# ---------------------------------------------------------------------------
# Layer selection
# ---------------------------------------------------------------------------

def _three_layer_state():
    return {
        'dimensions': {'z': [8e-9, 'm'], 'y': [8e-9, 'm'], 'x': [8e-9, 'm']},
        'layers': [
            {'type': 'image', 'source': 'https://s3/x.zarr/|zarr3:', 'name': 'em'},
            {'type': 'segmentation', 'source': 'https://fg/files/k/groups/l/s.zarr/|zarr2:',
             'name': 'seg'},
            {'type': 'annotation', 'source': {'url': 'local://annotations'},
             'name': 'ann', 'annotations': [box([0]*3, [1]*3)]},
        ],
    }


def test_image_layer_is_chosen_over_segmentation_and_local():
    assert ng.pick_image_layer(_three_layer_state()).name == 'em'


def test_named_layer_can_be_a_segmentation():
    picked = ng.pick_image_layer(_three_layer_state(), name='seg')
    assert picked.type == 'segmentation'
    assert ng.fileglancer_path(picked.source.url) is not None


def test_unknown_layer_name_lists_what_is_available():
    with pytest.raises(ng.NeuroglancerError, match="'em'"):
        ng.pick_image_layer(_three_layer_state(), name='nope')


def test_no_image_layer():
    state = {'layers': [{'type': 'annotation', 'source': {'url': 'local://a'},
                         'name': 'ann'}]}
    with pytest.raises(ng.NeuroglancerError, match='No image layer'):
        ng.pick_image_layer(state)


def test_several_image_layers_needs_disambiguation():
    state = {'layers': [
        {'type': 'image', 'source': 'https://s3/a.zarr', 'name': 'a'},
        {'type': 'image', 'source': 'https://s3/b.zarr', 'name': 'b'},
    ]}
    with pytest.raises(ng.NeuroglancerError, match='--layer'):
        ng.pick_image_layer(state)


def test_layer_failure_is_carried_not_raised():
    """A link with no usable source layer is still usable with --source."""
    state = {'dimensions': {'z': [1, 'nm'], 'y': [1, 'nm'], 'x': [1, 'nm']},
             'layers': [{'type': 'annotation', 'source': {'url': 'local://a'},
                         'name': 'ann', 'annotations': [box([0]*3, [4]*3)]}]}
    parsed = ng.parse(json.dumps(state))
    assert parsed.image_layer is None
    assert 'No image layer' in parsed.layer_error
    assert len(parsed.boxes) == 1


# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------

def test_json_url_and_state_agree():
    state = _three_layer_state()
    from_json = ng.parse(json.dumps(state))
    from_url = ng.parse('https://neuroglancer-demo.appspot.com/#!' +
                        quote(json.dumps(state)))
    assert from_json.boxes[0].start == from_url.boxes[0].start
    assert from_json.space.order == from_url.space.order


def test_shortened_link_gives_actionable_error():
    with pytest.raises(ng.NeuroglancerError, match='shortened'):
        ng.extract_state('https://fileglancer.int.janelia.org/ng/#!abc123XYZ')


def test_garbage_input():
    with pytest.raises(ng.NeuroglancerError, match='Expected Neuroglancer'):
        ng.extract_state('hello')
    with pytest.raises(ng.NeuroglancerError, match='parse'):
        ng.extract_state('{not json')


def test_missing_dimensions_requires_axis_order():
    state = {'layers': [{'type': 'annotation', 'annotations': [box([0]*3, [1]*3)]}]}
    with pytest.raises(ng.NeuroglancerError, match='axis order'):
        ng.parse(json.dumps(state))
    parsed = ng.parse(json.dumps(state), axis_order='zyx')
    assert parsed.space.order == 'zyx'
    assert not parsed.space.has_scale


# ---------------------------------------------------------------------------
# Regression: the real CellMap link
# ---------------------------------------------------------------------------

REAL_STATE = {
    'dimensions': {'z': [8e-9, 'm'], 'y': [8e-9, 'm'], 'x': [8e-9, 'm']},
    'position': [9879.6357421875, 3483.72119140625, 6946],
    'layers': [
        {'type': 'image',
         'source': ('https://s3.janelia.org/cellmap/jrc_amphiuma-means-liver-1/'
                    'jrc_amphiuma-means-liver-1.zarr/recon-1/em/fibsem-uint8/|zarr3:'),
         'name': 'fibsem-uint8'},
        {'type': 'segmentation',
         'source': ('https://fileglancer.int.janelia.org/files/ekDViMrqI9F60zgt/'
                    'groups/cellmap/cellmap/annotations/amira/'
                    'jrc_amphiuma-means-liver-1/mito001/mito001.zarr/|zarr2:'),
         'name': 'mito001.zarr'},
        {'type': 'annotation', 'source': {'url': 'local://annotations'},
         'name': 'test',
         'annotations': [{
             'pointA': [9464.19140625, 2790.916748046875, 6715],
             'pointB': [10986.34765625, 4438.87060546875, 6955.00048828125],
             'type': 'axis_aligned_bounding_box',
             'id': 'a169f6a15ae262bb43ea6bf10461f3a364ef4ac4'}]},
    ],
}


def test_real_link_parses_as_recorded():
    parsed = ng.parse(json.dumps(REAL_STATE))

    # Axis order was zyx, not xyz. Reading it as xyz would put x at 10986,
    # outside the real array's x extent of 10037.
    assert parsed.space.order == 'zyx'
    assert parsed.space.nm_per_unit == {'z': 8.0, 'y': 8.0, 'x': 8.0}

    assert parsed.image_layer.name == 'fibsem-uint8'
    assert parsed.image_layer.source.zarr_format == 3
    # Object storage: no filesystem path, so --source is required for this one.
    assert ng.fileglancer_path(parsed.image_layer.source.url) is None

    assert len(parsed.boxes) == 1
    b = parsed.boxes[0]
    assert b.start['z'] == pytest.approx(9464.19140625)
    assert b.stop['x'] == pytest.approx(6955.00048828125)

    # The dataset's real s0 voxel size is 8 nm, matching the space scale, so
    # these coordinates happen to equal s0 indices.
    assert b.start['z'] * parsed.space.nm_per_unit['z'] == pytest.approx(75713.53125)


# ---------------------------------------------------------------------------
# Chunk-aligned slabs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('start,stop,chunk,depth', [
    (0, 100, 64, 64), (71, 88, 64, 64), (0, 512, 64, 128), (5, 7, 64, 64),
    (9464, 10987, 64, 256), (13, 31, 8, 3), (0, 1, 1, 1), (100, 200, 7, 50),
])
def test_slabs_cover_the_range_exactly(start, stop, chunk, depth):
    slabs = list(aligned_slabs(start, stop, chunk, depth))
    assert slabs[0][0] == start
    assert slabs[-1][1] == stop
    for (_, a), (b, _) in zip(slabs, slabs[1:]):
        assert a == b                      # contiguous, no gaps or overlaps
    assert all(a < b for a, b in slabs)    # always makes progress
    assert all(b - a <= max(depth, 1) for a, b in slabs)   # budget respected


def test_interior_boundaries_land_on_chunk_boundaries():
    """The ROI rarely starts on a chunk boundary, so the first slab is the short
    one and every later boundary is aligned -- otherwise a source chunk would be
    decoded by two different slabs."""
    slabs = list(aligned_slabs(71, 400, chunk=64, depth=64))
    assert slabs[0] == (71, 128)
    for _, end in slabs[:-1]:
        assert end % 64 == 0


def test_depth_allows_several_whole_chunks():
    slabs = list(aligned_slabs(10, 500, chunk=64, depth=192))
    assert slabs[0] == (10, 192)           # to the boundary, plus two whole chunks
    assert slabs[1] == (192, 384)
    for _, end in slabs[:-1]:
        assert end % 64 == 0


def test_chunk_larger_than_budget_respects_the_budget():
    """A hard memory ceiling wins over alignment."""
    slabs = list(aligned_slabs(0, 100, chunk=1024, depth=16))
    assert all(b - a <= 16 for a, b in slabs)
    assert slabs[-1][1] == 100
