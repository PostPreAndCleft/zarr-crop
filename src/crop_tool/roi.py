"""ROI parsing and coordinate-space conversion.

This module holds the piece that removes the protocol's worst failure mode.
``crop-making-protocol.html`` documents the current practice for cutting at a
level below s0 as: "start dividing by 2; if the crop doesn't light up in the
right place, try 4, then 6 (or 8)". That guesswork exists because Amira's X/Y/Z
limit boxes are indices into the level being read, while Fileglancer reports s0
coordinates. Every level's ``scale`` and ``translation`` are recorded in the
OME-NGFF multiscales metadata, so the conversion is exact -- see
:func:`to_level_indices`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
from urllib.parse import unquote

SPATIAL = ('x', 'y', 'z')

#: Accepted values for --roi-space.
#:
#: ``s0``       indices at the finest level, as Fileglancer reports them
#: ``drawn``    indices of the level a Neuroglancer link was drawn on
#: ``level``    indices of the array being cropped (plugin parity)
#: ``physical`` scaled units
#:
#: ``s0`` and ``drawn`` share the same arithmetic and differ only in which
#: dataset entry is supplied as the reference transform.
SPACES = ('s0', 'drawn', 'level', 'physical')


class RoiError(ValueError):
    """Raised for malformed or unusable ROI input."""


@dataclass
class Roi:
    """An ROI as given by the user, before conversion to array indices.

    ``start`` and ``stop`` map axis name -> coordinate, in whatever space
    ``space`` names. Values are floats because Neuroglancer emits floats.
    """
    start: Dict[str, float]
    stop: Dict[str, float]
    space: str
    source: str = 'flags'

    def __post_init__(self):
        if self.space not in SPACES:
            raise RoiError('Unknown ROI space {0!r}; expected one of {1}.'.format(
                self.space, ', '.join(SPACES)))
        missing = [a for a in SPATIAL if a not in self.start or a not in self.stop]
        if missing:
            raise RoiError('ROI is missing axes: {0}.'.format(', '.join(missing)))


def _parse_triple(text: str, axis_order: str, what: str) -> Dict[str, float]:
    parts = [p.strip() for p in str(text).split(',')]
    if len(parts) != 3:
        raise RoiError(
            '{0} must be three comma-separated numbers (got {1!r}).'.format(what, text))
    try:
        values = [float(p) for p in parts]
    except ValueError:
        raise RoiError('{0} contains a non-numeric value: {1!r}.'.format(what, text))
    return dict(zip(axis_order, values))


def normalize_axis_order(order: str) -> str:
    """Validate an axis-order string like 'zyx' and return it lowercased."""
    cleaned = str(order).strip().lower().replace(',', '').replace(' ', '')
    if sorted(cleaned) != ['x', 'y', 'z']:
        raise RoiError(
            'Axis order must be some permutation of x, y and z (got {0!r}).'.format(order))
    return cleaned


def parse_flags(min_text: str, max_text: str, axis_order: str, space: str) -> Roi:
    """Build an ROI from --min/--max, interpreted in ``axis_order``."""
    order = normalize_axis_order(axis_order)
    lo = _parse_triple(min_text, order, '--min')
    hi = _parse_triple(max_text, order, '--max')
    return Roi(start=lo, stop=hi, space=space, source='flags')


# ---------------------------------------------------------------------------
# Neuroglancer input
# ---------------------------------------------------------------------------

_BBOX_TYPES = ('axis_aligned_bounding_box', 'AXIS_ALIGNED_BOUNDING_BOX')


def _state_axis_order(state: dict) -> Optional[str]:
    """Recover spatial axis order from a Neuroglancer state's dimensions.

    A state carries ``dimensions`` as an ordered mapping, e.g.
    ``{"x": [8e-9, "m"], "y": [8e-9, "m"], "z": [8e-9, "m"]}``. Annotation
    point coordinates are in that order. Dimension names may carry a trailing
    marker such as ``x'`` for local dimensions, so names are normalized.
    """
    dims = state.get('dimensions')
    if not isinstance(dims, dict):
        return None
    names = [re.sub(r"[^xyz]", '', str(k).lower()) for k in dims.keys()]
    spatial = [n for n in names if n in SPATIAL]
    if sorted(spatial) != ['x', 'y', 'z']:
        return None
    return ''.join(spatial)


def _iter_annotations(payload):
    """Yield candidate annotation dicts from any accepted JSON shape."""
    if isinstance(payload, dict):
        if 'pointA' in payload and 'pointB' in payload:
            yield payload
        for layer in payload.get('layers') or []:
            if isinstance(layer, dict):
                for ann in layer.get('annotations') or []:
                    if isinstance(ann, dict):
                        yield ann
        for ann in payload.get('annotations') or []:
            if isinstance(ann, dict):
                yield ann
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield from _iter_annotations(item)


def _extract_json(text: str) -> dict:
    """Pull the JSON state out of raw JSON or a long Neuroglancer URL."""
    stripped = text.strip()
    if stripped.startswith('{') or stripped.startswith('['):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as e:
            raise RoiError('Could not parse the input as JSON: {0}.'.format(e))

    if '#!' in stripped:
        fragment = stripped.split('#!', 1)[1]
        decoded = unquote(fragment).strip()
        if not decoded.startswith('{'):
            raise RoiError(
                'This looks like a shortened Neuroglancer link, whose state lives on a '
                'server rather than in the URL. Open it, then either paste the full '
                '(long) URL or copy the bounding-box annotation JSON.')
        try:
            return json.loads(decoded)
        except json.JSONDecodeError as e:
            raise RoiError('Could not parse the URL fragment as JSON: {0}.'.format(e))

    raise RoiError(
        'Expected Neuroglancer JSON or a long Neuroglancer URL containing "#!". '
        'Shortened links are not supported -- paste the long URL or the annotation JSON.')


def parse_neuroglancer(text: str, space: str, axis_order: Optional[str] = None) -> Roi:
    """Build an ROI from Neuroglancer output.

    Accepts a bare ``axis_aligned_bounding_box`` annotation, a list of
    annotations, a full Neuroglancer state, or a long ``#!``-fragment URL.
    Axis order comes from the state's ``dimensions`` when present, otherwise
    ``axis_order`` must be supplied.

    ``pointA``/``pointB`` are in draw order, not sorted, so they are reduced to
    a per-axis min/max rather than assumed to be low/high corners.
    """
    payload = _extract_json(text)

    order = None
    if isinstance(payload, dict):
        order = _state_axis_order(payload)
    if order is None:
        if axis_order is None:
            raise RoiError(
                'Could not determine axis order from the Neuroglancer input '
                '(no usable "dimensions" key). Pass --axis-order explicitly.')
        order = normalize_axis_order(axis_order)

    boxes = [a for a in _iter_annotations(payload)
             if a.get('type') in _BBOX_TYPES or ('pointA' in a and 'pointB' in a)]
    if not boxes:
        raise RoiError('No bounding-box annotation found in the Neuroglancer input. '
                       'Draw one with the bounding-box tool, then copy its JSON.')
    if len(boxes) > 1:
        raise RoiError(
            'Found {0} bounding-box annotations; this tool makes one crop per run. '
            'Copy just the annotation you want.'.format(len(boxes)))

    box = boxes[0]
    a, b = box['pointA'], box['pointB']
    if len(a) < len(order) or len(b) < len(order):
        raise RoiError('Annotation points have {0} and {1} coordinates, need at least '
                       '{2}.'.format(len(a), len(b), len(order)))

    start, stop = {}, {}
    for position, axis in enumerate(order):
        lo, hi = float(a[position]), float(b[position])
        start[axis], stop[axis] = min(lo, hi), max(lo, hi)
    return Roi(start=start, stop=stop, space=space, source='neuroglancer')


# ---------------------------------------------------------------------------
# Conversion to array indices
# ---------------------------------------------------------------------------

@dataclass
class ResolvedRoi:
    """Integer index bounds in a specific level's index space."""
    start: Dict[str, int]
    stop: Dict[str, int]
    clamped: Sequence[str]

    def shape(self, axes: Sequence[str]) -> Tuple[int, ...]:
        return tuple(self.stop[a] - self.start[a] for a in axes)


#: How close to a whole index counts as being on it, before flooring/ceiling.
#: Unit conversions are lossy in binary -- a Neuroglancer space of 3e-8 m is
#: 30.000000000000004 nm, not 30 -- and ceil() would turn that noise into a whole
#: extra slice. A millionth of a voxel is far below any real intent and far above
#: double-precision error for realistic index magnitudes.
INDEX_TOLERANCE = 1e-6


def _snap(value: float, tolerance: float = INDEX_TOLERANCE) -> float:
    """Round ``value`` to a whole number when it is within ``tolerance`` of one."""
    nearest = round(value)
    return float(nearest) if abs(value - nearest) <= tolerance else value


def to_level_indices(roi: Roi, level, reference, storage_axes: Sequence[str],
                     axis_index: Dict[str, int], sizes: Dict[str, int]) -> ResolvedRoi:
    """Convert an ROI into integer indices of the level being cropped.

    ``level`` is the target level's :class:`~crop_tool.zarr_io.Transform`;
    ``reference`` is the transform the input coordinates are expressed in -- the
    s0 dataset entry for ``space='s0'``, or the drawn level's entry for
    ``space='drawn'``. ``axis_index`` maps an axis name to its position in the
    full array.

    The conversion routes through physical space::

        physical = reference.translation[a] + index * reference.scale[a]
        index_L  = (physical - level.translation[a]) / level.scale[a]

    Going through physical coordinates rather than a bare scale ratio is what
    makes this exact: it absorbs the sub-voxel ``translation`` offsets that
    OME-NGFF levels commonly carry, which dividing by 2/4/8 cannot.

    Bounds are floored on start and ceiled on stop, so the written crop always
    covers the requested region rather than shaving a voxel off an edge.
    """
    import math

    start: Dict[str, int] = {}
    stop: Dict[str, int] = {}
    clamped = []

    for axis in storage_axes:
        i = axis_index[axis]
        lo_in, hi_in = roi.start[axis], roi.stop[axis]

        if roi.space == 'level':
            lo, hi = lo_in, hi_in
        else:
            if roi.space == 'physical':
                lo_phys, hi_phys = lo_in, hi_in
            else:
                if reference is None:
                    raise RoiError(
                        'ROI given in {0} coordinates, but there is no reference '
                        'transform to convert from -- the source has no multiscales '
                        'metadata. Re-run with --roi-space level and indices of the '
                        'array you are cropping.'.format(roi.space))
                lo_phys = reference.physical(i, lo_in)
                hi_phys = reference.physical(i, hi_in)
            lo = level.index(i, lo_phys)
            hi = level.index(i, hi_phys)

        lo_i, hi_i = int(math.floor(_snap(lo))), int(math.ceil(_snap(hi)))
        if hi_i <= lo_i:
            raise RoiError(
                'ROI is empty along {0}: resolved to [{1}, {2}) in the index space of '
                'the array being cropped. Check the axis order and --roi-space.'.format(
                    axis.upper(), lo_i, hi_i))

        size = sizes[axis]
        bounded_lo, bounded_hi = max(0, lo_i), min(size, hi_i)
        if (bounded_lo, bounded_hi) != (lo_i, hi_i):
            clamped.append(axis)
        if bounded_hi <= bounded_lo:
            raise RoiError(
                'ROI does not overlap the array along {0}: resolved to [{1}, {2}) but '
                'the axis has size {3}.'.format(axis.upper(), lo_i, hi_i, size))

        start[axis], stop[axis] = bounded_lo, bounded_hi

    return ResolvedRoi(start=start, stop=stop, clamped=tuple(clamped))
