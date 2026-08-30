"""Understanding a Neuroglancer state.

Everything that knows about Neuroglancer's JSON lives here, so that
:mod:`crop_tool.roi` stays a coordinate-maths module. This handles state
extraction, layer source URLs, the coordinate space, and bounding-box
annotations.

The shapes handled here were derived from a real link for the CellMap dataset
``jrc_amphiuma-means-liver-1``, cross-checked against that dataset's published
OME metadata. Notably:

* Source URLs use the pipe form, ``https://host/path/|zarr3:``, as well as the
  ``zarr://URL`` form in the Neuroglancer docs. Both are accepted.
* ``dimensions`` is an *ordered* mapping like ``{"z": [8e-9, "m"], ...}``, giving
  both the axis order and the size of one coordinate unit. Its key order is the
  order of annotation point coordinates -- and it was ``zyx``, not ``xyz``, which
  is exactly the hazard the crop protocol warns about.
* Annotation coordinates are therefore in units of that space, which equal s0
  indices only when the space's scale matches the s0 voxel size. It did in that
  link (8 nm both), but that is a coincidence, so coordinates are always
  converted through physical units rather than assumed to be indices.
* A Fileglancer-served URL carries the filesystem path:
  ``https://fileglancer.../files/<key>/groups/cellmap/...`` corresponds to
  ``/groups/cellmap/...``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

SPATIAL = ('x', 'y', 'z')

_BBOX_TYPES = ('axis_aligned_bounding_box', 'AXIS_ALIGNED_BOUNDING_BOX')

#: Length units that may appear in a Neuroglancer coordinate space or in
#: OME-NGFF axis metadata, as nanometres.
_UNIT_NM = {
    'm': 1e9, 'meter': 1e9, 'metre': 1e9,
    'mm': 1e6, 'millimeter': 1e6, 'millimetre': 1e6,
    'um': 1e3, 'µm': 1e3, 'μm': 1e3, 'micrometer': 1e3, 'micrometre': 1e3,
    'nm': 1.0, 'nanometer': 1.0, 'nanometre': 1.0,
    'a': 0.1, 'angstrom': 0.1, 'å': 0.1,
}


class NeuroglancerError(ValueError):
    """Raised for state input that cannot be used."""


def unit_to_nm(unit: Optional[str]) -> Optional[float]:
    """Nanometres per ``unit``, or None if the unit is absent or unrecognized."""
    if not unit:
        return None
    return _UNIT_NM.get(str(unit).strip().lower())


# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------

def extract_state(text: str) -> dict:
    """Pull the JSON state out of raw JSON or a long Neuroglancer URL."""
    stripped = str(text).strip()
    if stripped.startswith('{') or stripped.startswith('['):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as e:
            raise NeuroglancerError('Could not parse the input as JSON: {0}.'.format(e))

    if '#!' in stripped:
        fragment = stripped.split('#!', 1)[1]
        decoded = unquote(fragment).strip()
        if decoded.startswith('{'):
            try:
                return json.loads(decoded)
            except json.JSONDecodeError as e:
                raise NeuroglancerError(
                    'Could not parse the URL fragment as JSON: {0}.'.format(e))
        raise NeuroglancerError(
            'This looks like a shortened Neuroglancer link -- its state lives on a '
            'server rather than in the URL, so there is nothing to decode. Open it '
            'in Neuroglancer, then paste either the full (long) URL from the address '
            'bar or the bounding-box annotation JSON.')

    raise NeuroglancerError(
        'Expected Neuroglancer JSON or a long Neuroglancer URL containing "#!". '
        'Shortened links are not supported -- paste the long URL or the JSON.')


# ---------------------------------------------------------------------------
# Layer sources
# ---------------------------------------------------------------------------

@dataclass
class LayerSource:
    """A parsed layer source URL."""
    url: str
    zarr_format: Optional[int] = None
    raw: str = ''

    @property
    def is_local(self) -> bool:
        return self.url.startswith('local://')

    @property
    def container_url(self) -> Optional[str]:
        """The ``.zarr`` container part of the URL, if there is one."""
        marker = '.zarr'
        index = self.url.find(marker)
        if index < 0:
            return None
        return self.url[:index + len(marker)]

    @property
    def inner_path(self) -> str:
        """Whatever the URL names inside the container, e.g. ``em/s1``."""
        container = self.container_url
        if container is None:
            return ''
        return self.url[len(container):].strip('/')

    @property
    def leaf(self) -> str:
        """The last path segment, which is the level name when one is named."""
        return self.inner_path.rsplit('/', 1)[-1] if self.inner_path else ''


def parse_source(raw) -> Optional[LayerSource]:
    """Parse a layer ``source`` entry.

    Accepts a bare string, a ``{"url": ...}`` dict (as annotation layers use), or
    a list of either, taking the first usable one. Understands both
    ``zarr://URL`` / ``zarr2://`` / ``zarr3://`` prefixes and the
    ``URL|zarr2:`` / ``URL|zarr3:`` pipe suffix.
    """
    if isinstance(raw, list):
        for item in raw:
            parsed = parse_source(item)
            if parsed is not None:
                return parsed
        return None
    if isinstance(raw, dict):
        return parse_source(raw.get('url'))
    if not isinstance(raw, str) or not raw:
        return None

    text = raw.strip()
    fmt = None

    # Pipe suffix, e.g. "https://host/a.zarr/|zarr3:"
    if '|' in text:
        head, _, tail = text.rpartition('|')
        match = re.match(r'^zarr(2|3)?:?$', tail.strip(), re.IGNORECASE)
        if match:
            text = head
            if match.group(1):
                fmt = int(match.group(1))

    # Scheme prefix, e.g. "zarr://https://host/a.zarr"
    match = re.match(r'^zarr(2|3)?://', text, re.IGNORECASE)
    if match:
        text = text[match.end():]
        if match.group(1):
            fmt = int(match.group(1))

    return LayerSource(url=text.rstrip('/'), zarr_format=fmt, raw=raw)


@dataclass
class Layer:
    name: str
    type: str
    source: Optional[LayerSource]


def layers(state: dict) -> List[Layer]:
    out = []
    for entry in state.get('layers') or []:
        if not isinstance(entry, dict):
            continue
        out.append(Layer(
            name=str(entry.get('name') or ''),
            type=str(entry.get('type') or ''),
            source=parse_source(entry.get('source')),
        ))
    return out


def pick_image_layer(state: dict, name: Optional[str] = None) -> Layer:
    """Choose the layer to crop.

    Defaults to the single ``type: "image"`` layer. ``name`` selects a layer
    explicitly, which is how you crop a segmentation layer or disambiguate when
    a state holds several images.
    """
    found = layers(state)
    if name is not None:
        matches = [layer for layer in found if layer.name == name]
        if not matches:
            raise NeuroglancerError(
                'No layer named {0!r}. Layers in this link: {1}.'.format(
                    name, ', '.join('{0!r} ({1})'.format(l.name, l.type)
                                    for l in found) or 'none'))
        return matches[0]

    images = [layer for layer in found
              if layer.type == 'image' and layer.source and not layer.source.is_local]
    if not images:
        raise NeuroglancerError(
            'No image layer with a remote source found in this link. Layers: {0}. '
            'Pass --layer NAME to choose one, or --source to name the array '
            'directly.'.format(', '.join('{0!r} ({1})'.format(l.name, l.type)
                                         for l in found) or 'none'))
    if len(images) > 1:
        raise NeuroglancerError(
            'This link has {0} image layers ({1}). Pass --layer NAME to choose '
            'one.'.format(len(images), ', '.join(repr(l.name) for l in images)))
    return images[0]


def fileglancer_path(url: str) -> Optional[Path]:
    """Filesystem path for a Fileglancer-served URL, or None.

    Fileglancer serves file-share content at ``/files/<key>/<path>``, where
    ``<path>`` is the share path without its leading slash. So

        https://fileglancer.int.janelia.org/files/ekDViMrqI9F60zgt/groups/cellmap/x

    corresponds to ``/groups/cellmap/x``.

    Inferred from one example, so callers must confirm the path exists rather
    than trusting it -- see ``resolve_source_path`` in the CLI.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if not parsed.scheme.startswith('http'):
        return None
    match = re.match(r'^/files/[^/]+/(.+)$', parsed.path)
    if not match:
        return None
    return Path('/' + match.group(1).strip('/'))


# ---------------------------------------------------------------------------
# Coordinate space
# ---------------------------------------------------------------------------

@dataclass
class CoordinateSpace:
    """The state's spatial coordinate space.

    ``order`` is the spatial axis order as it appears in ``dimensions``, which is
    also the order of annotation point coordinates. ``nm_per_unit`` gives the
    physical size of one coordinate unit per axis, or None when the state does
    not say.
    """
    order: str
    nm_per_unit: Dict[str, Optional[float]] = field(default_factory=dict)

    @property
    def has_scale(self) -> bool:
        return all(self.nm_per_unit.get(a) for a in self.order)


def _clean_axis_name(key: str) -> str:
    """Normalize a dimension name; Neuroglancer marks local dims as e.g. ``x'``."""
    return re.sub(r"[^xyz]", '', str(key).lower())


def coordinate_space(state: dict) -> Optional[CoordinateSpace]:
    """Read ``dimensions`` into a :class:`CoordinateSpace`, or None."""
    dims = state.get('dimensions')
    if not isinstance(dims, dict):
        return None

    order, nm = [], {}
    for key, value in dims.items():
        axis = _clean_axis_name(key)
        if axis not in SPATIAL or axis in order:
            continue
        order.append(axis)
        scale_nm = None
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            factor = unit_to_nm(value[1])
            if factor is not None:
                try:
                    scale_nm = float(value[0]) * factor
                except (TypeError, ValueError):
                    scale_nm = None
        nm[axis] = scale_nm

    if sorted(order) != ['x', 'y', 'z']:
        return None
    return CoordinateSpace(order=''.join(order), nm_per_unit=nm)


# ---------------------------------------------------------------------------
# Bounding boxes
# ---------------------------------------------------------------------------

@dataclass
class Box:
    """One bounding box, in the state's coordinate units."""
    start: Dict[str, float]
    stop: Dict[str, float]
    description: Optional[str] = None
    id: Optional[str] = None
    layer: Optional[str] = None


def _iter_annotation_entries(payload):
    """Yield (annotation, layer_name) from any accepted JSON shape."""
    if isinstance(payload, dict):
        if 'pointA' in payload and 'pointB' in payload:
            yield payload, None
        for entry in payload.get('layers') or []:
            if isinstance(entry, dict):
                name = entry.get('name')
                for ann in entry.get('annotations') or []:
                    if isinstance(ann, dict):
                        yield ann, name
        for ann in payload.get('annotations') or []:
            if isinstance(ann, dict):
                yield ann, payload.get('name')
    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_annotation_entries(item)


def boxes(payload, order: str) -> List[Box]:
    """Collect every bounding-box annotation, in the order they appear.

    ``pointA``/``pointB`` are in draw order rather than sorted, so each axis is
    reduced to a min/max pair.
    """
    out = []
    for ann, layer_name in _iter_annotation_entries(payload):
        if ann.get('type') not in _BBOX_TYPES and not (
                'pointA' in ann and 'pointB' in ann):
            continue
        a, b = ann.get('pointA'), ann.get('pointB')
        if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
            continue
        if len(a) < len(order) or len(b) < len(order):
            raise NeuroglancerError(
                'A bounding box has {0} and {1} coordinates but the coordinate '
                'space has {2} spatial axes.'.format(len(a), len(b), len(order)))
        start, stop = {}, {}
        for position, axis in enumerate(order):
            lo, hi = float(a[position]), float(b[position])
            start[axis], stop[axis] = min(lo, hi), max(lo, hi)
        description = ann.get('description')
        out.append(Box(
            start=start, stop=stop,
            description=str(description).strip() if description else None,
            id=ann.get('id'), layer=layer_name))
    return out


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

@dataclass
class ParsedState:
    raw: dict
    space: Optional[CoordinateSpace]
    boxes: List[Box]
    image_layer: Optional[Layer] = None
    layer_error: Optional[str] = None


def parse(text: str, layer_name: Optional[str] = None,
          axis_order: Optional[str] = None) -> ParsedState:
    """Parse a link or state into a coordinate space, boxes, and a source layer.

    Failing to identify a source layer is *not* fatal -- the boxes are still
    usable with an explicit ``--source`` -- so that error is carried on the
    result rather than raised.
    """
    state = extract_state(text)
    space = coordinate_space(state) if isinstance(state, dict) else None

    if space is None:
        if axis_order is None:
            raise NeuroglancerError(
                'Could not determine axis order from the input (no usable '
                '"dimensions" key). Pass --axis-order explicitly. Note that '
                'Neuroglancer reports ZYX for some datasets and XYZ for others, '
                'so there is no safe default.')
        order = ''.join(c for c in str(axis_order).strip().lower() if c in SPATIAL)
        if sorted(order) != ['x', 'y', 'z']:
            raise NeuroglancerError(
                'Axis order must be a permutation of x, y and z (got '
                '{0!r}).'.format(axis_order))
        space = CoordinateSpace(order=order, nm_per_unit={a: None for a in order})

    found = boxes(state, space.order)

    image_layer, layer_error = None, None
    if isinstance(state, dict):
        try:
            image_layer = pick_image_layer(state, layer_name)
        except NeuroglancerError as e:
            layer_error = str(e)

    return ParsedState(raw=state if isinstance(state, dict) else {}, space=space,
                       boxes=found, image_layer=image_layer, layer_error=layer_error)
