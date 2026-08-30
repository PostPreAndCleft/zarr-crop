"""Command-line entry point: cut crops out of an OME-Zarr and write them as a
new OME-Zarr container.

This replaces the Amira leg of the crop-making protocol (steps 3-5 of
``crop-making-protocol.html``). It never modifies the source: the source array
is opened read-only, and the run is refused outright if the output would land
inside the source container.

An ROI can come from ``--min``/``--max``, or from a Neuroglancer link. A link
holding several bounding boxes yields several crops as sibling groups of one
output container.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import neuroglancer as ng
from . import roi as roi_mod
from . import zarr_io

#: Default ceiling on how much of the ROI is held in memory at once, in bytes.
DEFAULT_MAX_MEMORY = 512 * 1024 * 1024


class CropError(RuntimeError):
    """A user-facing failure; reported without a traceback."""


def warn(message: str) -> None:
    print('warning: {0}'.format(message), file=sys.stderr)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='zarr-crop',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Cut crops from an OME-Zarr volume and write them as a new '
                    '.zarr container, without going through Amira.',
        epilog="""\
examples:
  # ROI typed in, cutting from s0
  zarr-crop --source /nrs/lab/FuncEworm/recon.zarr/em/s0 \\
            --out    /nrs/lab/crops/FuncEworm_soma_7.zarr \\
            --min 1024,2048,512 --max 1536,2560,768 --axis-order xyz

  # same s0 numbers, cutting from s2: no divisor math needed
  zarr-crop --source /nrs/lab/FuncEworm/recon.zarr/em/s2 \\
            --out    /nrs/lab/crops/FuncEworm_soma_7.zarr \\
            --min 1024,2048,512 --max 1536,2560,768 --axis-order xyz

  # paste a Neuroglancer link; every box in it becomes a crop
  zarr-crop --neuroglancer 'https://neuroglancer-demo.appspot.com/#!{...}' \\
            --source /nrs/lab/FuncEworm/recon.zarr/em/s0 \\
            --out /nrs/lab/crops.zarr --name-prefix Pancreas5_soma --start-number 7
""")

    parser.add_argument('--source', default=None,
                        help='Path to the source zarr ARRAY (a level, e.g. '
                             '.../recon.zarr/em/s0). Optional when '
                             '--neuroglancer supplies a resolvable source.')
    parser.add_argument('--out', required=True,
                        help='Output .zarr container. Created if absent.')

    naming = parser.add_argument_group('naming')
    naming.add_argument('--name', default=None,
                        help='Group name for a single crop '
                             '(default: the --out filename without .zarr).')
    naming.add_argument('--name-prefix', default=None,
                        help='Number the crops as <prefix>_<n>. Implied when a '
                             'link holds more than one box.')
    naming.add_argument('--start-number', type=int, default=1,
                        help='First number for --name-prefix (default 1). Use this '
                             'to continue an existing series.')

    roi_group = parser.add_argument_group('ROI')
    roi_group.add_argument('--min', dest='roi_min',
                           help='Lower corner as three comma-separated numbers.')
    roi_group.add_argument('--max', dest='roi_max',
                           help='Upper corner as three comma-separated numbers.')
    roi_group.add_argument('--neuroglancer', dest='neuroglancer',
                           help='A Neuroglancer link or state JSON: pass the text '
                                'directly, a file path, or "-" for stdin.')
    roi_group.add_argument('--layer', default=None,
                           help='Name of the link layer to crop (default: the only '
                                'image layer).')
    roi_group.add_argument('--axis-order', default=None,
                           help='Order of the --min/--max values, e.g. xyz or zyx. '
                                'Required with --min/--max; with --neuroglancer it '
                                'is read from the state when available.')
    roi_group.add_argument('--roi-space', default=None, choices=roi_mod.SPACES,
                           help='Coordinate space of the ROI values. With '
                                '--min/--max the default is "s0": indices at the '
                                'finest level, as Fileglancer reports them, '
                                'converted automatically for whichever level '
                                '--source names. With --neuroglancer the default is '
                                'read from the link -- physical units when the state '
                                'gives a scale, otherwise "drawn", meaning indices '
                                'of the level the box was drawn on. A box drawn on '
                                's1 is cut from s1 unless --source says otherwise, '
                                'in which case the coordinates are converted.')

    array_group = parser.add_argument_group('array selection')
    array_group.add_argument('--channel', type=int, default=0,
                             help='Channel index for 4D/5D sources (default 0).')
    array_group.add_argument('--time', type=int, default=0,
                             help='Time index for 4D/5D sources (default 0).')

    out_group = parser.add_argument_group('output')
    out_group.add_argument('--zarr-format', choices=('2', '3'), default=None,
                           help="Output zarr format. Defaults to the source's.")
    out_group.add_argument('--chunks', default=None,
                           help='Chunk shape as z,y,x (default 128,128,128). Clamped '
                                'per axis to the crop shape.')
    out_group.add_argument('--unit', default=None,
                           help="Spatial unit for the output metadata (default: the "
                                "source's, else nanometer).")
    out_group.add_argument('--max-memory', type=float, default=None,
                           help='Approximate read budget in MB (default {0:.0f}). '
                                'Larger ROIs are copied in slabs aligned to the '
                                "source's chunk grid.".format(
                                    DEFAULT_MAX_MEMORY / 1e6))
    out_group.add_argument('--dry-run', action='store_true',
                           help='Report what would be written, then exit without '
                                'creating anything.')
    return parser


def parse_chunks(text: Optional[str]) -> Optional[Sequence[int]]:
    if text is None:
        return None
    parts = [p.strip() for p in text.split(',')]
    if len(parts) != 3:
        raise CropError('--chunks must be three comma-separated integers (z,y,x).')
    try:
        values = [int(p) for p in parts]
    except ValueError:
        raise CropError('--chunks must be integers, got {0!r}.'.format(text))
    if any(v < 1 for v in values):
        raise CropError('--chunks values must be positive.')
    return values


def read_link_text(value: str) -> str:
    """Accept the link text itself, a file path, or ``-`` for stdin."""
    if value == '-':
        return sys.stdin.read()
    stripped = value.strip()
    if stripped.startswith('{') or stripped.startswith('[') or '#!' in stripped \
            or stripped.startswith('http'):
        return stripped
    path = Path(value)
    if path.is_file():
        return path.read_text()
    raise CropError(
        'Could not read --neuroglancer {0!r}: not a file, and it does not look '
        'like a Neuroglancer link or JSON state.'.format(value))


# ---------------------------------------------------------------------------
# Source inspection
# ---------------------------------------------------------------------------

class Source:
    """Everything we need to know about the source level."""

    def __init__(self, source_path: str):
        container, dataset = zarr_io.split_path_at_container(source_path)
        if container is None:
            raise CropError(
                '{0} is not inside a .zarr container. Point --source at a level '
                'directory such as .../recon.zarr/em/s0.'.format(source_path))
        self.container = container
        self.dataset = dataset
        self.array_dir = container + dataset

        if not zarr_io.is_zarr_array(self.array_dir):
            raise CropError(
                '{0} is not a zarr array. Point --source at a level (e.g. '
                '.../s0), not at the container or an intermediate '
                'group.'.format(self.array_dir))

        self.array = zarr_io.open_array(self.array_dir, mode='r')
        self.zarr_format = 3 if self.array.metadata.zarr_format == 3 else 2
        self.ndim = len(self.array.shape)
        if self.ndim not in (3, 4, 5):
            raise CropError(
                'Only 3D, 4D and 5D arrays are supported (got {0}D).'.format(self.ndim))

        parent = container if dataset == '' else str(Path(self.array_dir).parent)
        self.multiscales, self.ms_dir, ms_errors = zarr_io.find_multiscales(parent, container)
        if self.multiscales is None:
            warn('No OME-Zarr multiscales metadata found above {0}. Falling back to '
                 'unit scale and zero offset; s0 coordinates cannot be converted.'
                 .format(self.array_dir))
            for path, kind, message in ms_errors:
                warn('  {0}: {1}: {2}'.format(path, kind, message))

        axes = (self.multiscales[0].get('axes') or []) if self.multiscales else []
        classified, error = zarr_io.classify_axes(axes, self.ndim)
        if classified is None:
            raise CropError('Axis layout error: {0}'.format(error))
        (self.axis_types, self.axis_names, self.spatial_indices, self.storage_axes,
         self.t_axis, self.c_axis, assumed) = classified
        if assumed:
            warn('Axis types are absent from the metadata; assumed layout {0}. '
                 'Verify the crop lands where you expect.'.format(assumed))

        self.transform, self.units = zarr_io.transform_for(
            self.multiscales, self.ms_dir, self.array_dir, self.ndim, warn=warn)
        self.reference, self.reference_path = zarr_io.transform_for_reference_level(
            self.multiscales, self.ndim)

        try:
            self.rel_path = str(Path(self.array_dir).relative_to(self.ms_dir))
        except (ValueError, TypeError):
            self.rel_path = Path(self.array_dir).name

        self.axis_index = {name: self.spatial_indices[i]
                           for i, name in enumerate(self.storage_axes)}
        self.sizes = {name: self.array.shape[self.axis_index[name]]
                      for name in self.storage_axes}

    def level_paths(self) -> List[str]:
        """Dataset paths in this container's multiscales, finest first."""
        if not self.multiscales:
            return []
        return [d.get('path', '') for d in (self.multiscales[0].get('datasets') or [])]

    def transform_for_level(self, level_path: str):
        """The :class:`~crop_tool.zarr_io.Transform` for a named level, or None."""
        if not self.multiscales:
            return None
        for dataset in self.multiscales[0].get('datasets') or []:
            if dataset.get('path', '').strip('/') == str(level_path).strip('/'):
                scale = [1.0] * self.ndim
                translation = [0.0] * self.ndim
                for ct in dataset.get('coordinateTransformations') or []:
                    if 'scale' in ct:
                        scale = [float(v) for v in ct['scale']]
                    elif 'translation' in ct:
                        translation = [float(v) for v in ct['translation']]
                return zarr_io.Transform(scale, translation)
        return None

    def spatial_unit(self) -> str:
        for i in self.spatial_indices:
            if i < len(self.units) and self.units[i]:
                return self.units[i]
        return 'nanometer'

    def validate_channel_time(self, channel: int, time: int) -> None:
        """Same rules as ZarrRead's channel/time port validation."""
        if self.c_axis is None:
            if channel != 0:
                raise CropError('This array has no channel axis; --channel must be 0.')
        else:
            top = self.array.shape[self.c_axis] - 1
            if not 0 <= channel <= top:
                raise CropError('--channel out of range; choose within 0-{0}.'.format(top))
        if self.t_axis is None:
            if time != 0:
                raise CropError('This array has no time axis; --time must be 0.')
        else:
            top = self.array.shape[self.t_axis] - 1
            if not 0 <= time <= top:
                raise CropError('--time out of range; choose within 0-{0}.'.format(top))


def check_output_outside_source(out: str, source_container: str) -> None:
    """Refuse to write anywhere inside the source container.

    The tool's one hard guarantee is that it never alters raw collaborator
    data, so this is checked before anything is created.
    """
    out_abs = Path(os.path.abspath(out))
    src_abs = Path(os.path.abspath(source_container))
    if out_abs == src_abs or src_abs in out_abs.parents:
        raise CropError(
            'Refusing to run: --out ({0}) is inside the source container ({1}). '
            'This tool never writes into raw data. Choose an output path outside '
            'it.'.format(out_abs, src_abs))


def resolve_source_path(args, parsed: Optional[ng.ParsedState]) -> str:
    """Work out which array to crop.

    ``--source`` always wins. Otherwise the link's chosen layer is used, which
    only yields a filesystem path for Fileglancer-served URLs; object-storage
    URLs (e.g. s3.janelia.org) carry no path, so those need ``--source``.
    """
    if args.source:
        return args.source
    if parsed is None:
        raise CropError('No ROI given. Pass --min and --max, or --neuroglancer.')
    if parsed.image_layer is None:
        raise CropError(
            'Could not pick a source layer from the link, so --source is required. '
            '{0}'.format(parsed.layer_error or ''))

    url = parsed.image_layer.source.url
    path = ng.fileglancer_path(url)
    if path is None:
        raise CropError(
            'The link\'s layer {0!r} is served from {1}, which carries no filesystem '
            'path (only Fileglancer /files/<key>/ URLs do). Pass --source with the '
            'path to the level you want to crop.'.format(
                parsed.image_layer.name, url))

    if not path.exists():
        hint = ''
        leaf = parsed.image_layer.source.leaf
        if leaf:
            hint = (' The link was drawn on level {0!r}, so --source should normally '
                    'name that level of your local copy.'.format(leaf))
        raise CropError(
            'Derived the path {0} from the link\'s layer {1!r} (url {2}), but that '
            'path does not exist. Either the share is not mounted here under that '
            'path, or the URL-to-path rule does not hold for this host. Pass '
            '--source explicitly.{3}'.format(
                path, parsed.image_layer.name, url, hint))

    # The URL usually names the level that was drawn on; crop that by default.
    if zarr_io.is_zarr_array(str(path)):
        return str(path)
    for candidate in ('s0', '0'):
        if zarr_io.is_zarr_array(str(path / candidate)):
            print('resolved source from link: {0}'.format(path / candidate))
            return str(path / candidate)
    raise CropError(
        'Derived the path {0} from the link, but it is not a zarr array and holds '
        'no s0 level. Pass --source with the exact level to crop.'.format(path))


def drawn_level(parsed: Optional[ng.ParsedState], source: Source) -> Optional[str]:
    """The level the boxes were drawn on, if the link names a recognizable one.

    Neuroglancer's coordinate space for a layer opened directly on a level array
    is that level's voxel indices. The level name is in the layer URL, so it can
    be recovered even when the URL's filesystem path is unusable here.
    """
    if parsed is None or parsed.image_layer is None or not parsed.image_layer.source:
        return None
    leaf = parsed.image_layer.source.leaf
    if leaf and leaf in source.level_paths():
        return leaf
    return None


# ---------------------------------------------------------------------------
# Turning input into a list of jobs
# ---------------------------------------------------------------------------

class Job:
    def __init__(self, name: str, roi: roi_mod.Roi, box: Optional[ng.Box] = None):
        self.name = name
        self.roi = roi
        self.box = box


def box_to_roi(box: ng.Box, space: ng.CoordinateSpace, source: Source,
              forced_space: Optional[str], drawn: Optional[str]) -> roi_mod.Roi:
    """Convert a Neuroglancer box into an ROI the resolver can place.

    Annotation coordinates are in units of the state's coordinate space, and
    which space that is depends on what the link opened:

    * **A level array** (e.g. ``.../crop.zarr/s1``) has no OME metadata, so the
      space is that level's raw voxel indices with no unit. A box drawn there is
      cut from that level by default, and converting it to a different level goes
      through physical space using the drawn level's own transform.
    * **A multiscale group** carries units, so coordinates convert to physical
      units directly. This is exact whether or not the state's scale happens to
      match the source's s0 voxel size.

    Falls back to treating coordinates as s0 indices only when neither applies,
    and says so.
    """
    if forced_space is not None and forced_space != 'drawn':
        return roi_mod.Roi(start=dict(box.start), stop=dict(box.stop),
                           space=forced_space, source='neuroglancer')

    source_nm = ng.unit_to_nm(source.spatial_unit())
    if space.has_scale and source_nm is not None:
        start, stop = {}, {}
        for axis in ('z', 'y', 'x'):
            factor = space.nm_per_unit[axis] / source_nm
            start[axis] = box.start[axis] * factor
            stop[axis] = box.stop[axis] * factor
        return roi_mod.Roi(start=start, stop=stop, space='physical',
                           source='neuroglancer')

    # No unit scale in the state. If the link named a level, its indices are that
    # level's, which is both the sensible default target and the right reference
    # for converting to any other level.
    if drawn is not None:
        return roi_mod.Roi(start=dict(box.start), stop=dict(box.stop),
                           space='drawn', source='neuroglancer')

    warn("The link's coordinate space gives no unit scale and its layer does not "
         'name a level of this container, so its coordinates are being treated as '
         's0 indices. Pass --roi-space to override.')
    return roi_mod.Roi(start=dict(box.start), stop=dict(box.stop),
                       space='s0', source='neuroglancer')


def build_jobs(args, source: Source, parsed: Optional[ng.ParsedState],
               out_path: Path, drawn: Optional[str] = None) -> List[Job]:
    """Resolve the ROI input into one job per crop."""
    if parsed is None:
        if args.roi_min is None or args.roi_max is None:
            raise CropError('--min and --max must be given together.')
        if args.axis_order is None:
            raise CropError(
                'With --min/--max you must pass --axis-order (e.g. xyz or zyx). '
                'Neuroglancer reports XYZ for some datasets and ZYX for others, so '
                'there is no safe default.')
        roi = roi_mod.parse_flags(args.roi_min, args.roi_max, args.axis_order,
                                  args.roi_space or 's0')
        rois = [(roi, None)]
    else:
        if not parsed.boxes:
            raise CropError(
                'No bounding-box annotation found in the link. Draw one with the '
                'bounding-box tool, then copy the link again.')
        rois = [(box_to_roi(box, parsed.space, source, args.roi_space, drawn), box)
                for box in parsed.boxes]

    stem = out_path.name[:-len('.zarr')]
    count = len(rois)

    if args.name_prefix:
        prefix = args.name_prefix
    elif args.name:
        if count > 1:
            raise CropError(
                '--name names a single crop, but this link holds {0} boxes. Use '
                '--name-prefix to number them.'.format(count))
        prefix = None
    else:
        prefix = None if count == 1 else stem

    jobs = []
    for index, (roi, box) in enumerate(rois):
        if prefix is not None:
            name = '{0}_{1}'.format(prefix, args.start_number + index)
        else:
            name = args.name or stem
        jobs.append(Job(name=name, roi=roi, box=box))

    seen = set()
    for job in jobs:
        if job.name in seen:
            raise CropError('Duplicate crop name {0!r}.'.format(job.name))
        seen.add(job.name)
    return jobs


# ---------------------------------------------------------------------------
# Writing one crop
# ---------------------------------------------------------------------------

def choose_slab(source: Source, shape, budget_bytes: float) -> int:
    """Slab depth along output z, aligned to the source's chunk grid.

    Reading in slabs keeps memory bounded, but an arbitrary slab depth makes the
    reader decode the same source chunk more than once. Picking a multiple of the
    source's chunk depth (and letting :func:`zarr_io.aligned_slabs` place the
    boundaries on absolute chunk boundaries) avoids that.
    """
    z_axis = source.axis_index['z']
    chunks = getattr(source.array, 'chunks', None)
    chunk_z = int(chunks[z_axis]) if chunks else 1
    chunk_z = max(1, chunk_z)

    per_slice = max(1, int(np.prod(shape[1:])) * source.array.dtype.itemsize)
    affordable = max(1, int(budget_bytes // per_slice))
    if affordable < chunk_z:
        return affordable
    return (affordable // chunk_z) * chunk_z


class Result:
    def __init__(self, job: Job, ok: bool, detail: str = '',
                 array_dir: Optional[Path] = None):
        self.job = job
        self.ok = ok
        self.detail = detail
        self.array_dir = array_dir


def reference_for(source: Source, roi: roi_mod.Roi, drawn: Optional[str]):
    """The transform the ROI's coordinates are expressed in.

    For ``drawn`` that is the level the box was drawn on; otherwise the finest
    level, which is what ``s0`` coordinates mean.
    """
    if roi.space == 'drawn' and drawn is not None:
        return source.transform_for_level(drawn) or source.reference
    return source.reference


def write_crop(source: Source, job: Job, out_path: Path, out_format: int,
               chunks, unit: str, args, drawn: Optional[str] = None) -> Result:
    """Resolve, validate and write one crop."""
    resolved = roi_mod.to_level_indices(
        job.roi, source.transform, reference_for(source, job.roi, drawn),
        source.storage_axes, source.axis_index, source.sizes)
    if resolved.clamped:
        warn('{0}: ROI clamped to the array bounds along {1}.'.format(
            job.name, ', '.join(a.upper() for a in resolved.clamped)))

    shape = resolved.shape(zarr_io.OUTPUT_AXES)
    scale_zyx = [source.transform.scale[source.axis_index[a]]
                 for a in zarr_io.OUTPUT_AXES]
    translation_zyx = [
        source.transform.physical(source.axis_index[a], resolved.start[a])
        for a in zarr_io.OUTPUT_AXES
    ]

    report_crop(source, job, resolved, shape, scale_zyx, translation_zyx,
                out_path, out_format, unit, drawn)

    if args.dry_run:
        return Result(job, True, 'dry run')

    array_dir = out_path / job.name / 's0'
    if zarr_io.is_zarr_array(str(array_dir)):
        raise CropError(
            '{0} already exists. Choose a different --name/--name-prefix or '
            '--start-number rather than overwriting an existing crop.'.format(array_dir))

    zarr_io.ensure_group_metadata(str(out_path), out_format)
    zarr_io.ensure_group_metadata(str(out_path / job.name), out_format)
    dst = zarr_io.create_array(
        container=str(out_path), name='{0}/s0'.format(job.name), shape=shape,
        dtype=source.array.dtype, zarr_format=out_format, chunks=chunks)

    full_index = [slice(None)] * source.ndim
    for i, ax_type in enumerate(source.axis_types):
        if ax_type == 'time':
            full_index[i] = args.time
        elif ax_type == 'channel':
            full_index[i] = args.channel
        else:
            axis = source.axis_names[i]
            full_index[i] = slice(resolved.start[axis], resolved.stop[axis])

    budget = (args.max_memory * 1e6) if args.max_memory else DEFAULT_MAX_MEMORY
    perm = [source.storage_axes.index(a) for a in zarr_io.OUTPUT_AXES]
    zarr_io.copy_region(
        src=source.array, dst=dst, full_index=full_index,
        z_axis=source.axis_index['z'], z_start=resolved.start['z'],
        z_stop=resolved.stop['z'], perm=perm,
        slab=choose_slab(source, shape, budget),
        chunk_z=(getattr(source.array, 'chunks', None) or [1])[source.axis_index['z']])

    entry = zarr_io.build_multiscales_entry(
        's0', scale_zyx, translation_zyx, unit, out_format)
    zarr_io.write_multiscales(str(out_path / job.name), entry, out_format)
    return Result(job, True, array_dir=array_dir)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_source(source: Source, unit: str) -> None:
    print('source      {0}'.format(source.array_dir))
    print('  shape     {0} dtype {1} axes ({2}) zarr v{3}'.format(
        tuple(source.array.shape), source.array.dtype,
        ', '.join(source.axis_names), source.zarr_format))
    print('  chunks    {0}{1}'.format(
        getattr(source.array, 'chunks', None),
        '  shards {0}'.format(source.array.shards)
        if getattr(source.array, 'shards', None) else ''))
    print('  scale     {0} {1}'.format(source.transform.scale, unit))
    print('  offset    {0} {1}'.format(source.transform.translation, unit))


def report_crop(source, job, resolved, shape, scale_zyx, translation_zyx,
                out_path, out_format, unit, drawn=None) -> None:
    """Print one crop's summary; the tail is for pasting into the GitHub issue."""
    label = job.name
    if job.box is not None and job.box.description:
        label += '  (box described {0!r})'.format(job.box.description)
    print('\n--- {0}'.format(label))
    print('  ROI       from {0}, in {1} coordinates'.format(
        job.roi.source, job.roi.space))
    for axis in ('z', 'y', 'x'):
        print('    {0}       {1:g} -> {2:g}   index [{3}:{4}]'.format(
            axis.upper(), job.roi.start[axis], job.roi.stop[axis],
            resolved.start[axis], resolved.stop[axis]))

    # Always say which level the coordinates came from and which is being cut,
    # so the conversion is never an invisible assumption.
    if job.roi.space in ('s0', 'drawn'):
        origin = drawn if job.roi.space == 'drawn' else source.reference_path
        if origin and origin == source.rel_path:
            print('    drawn on {0!r}, cutting {0!r} -- no conversion needed'.format(
                origin))
        elif origin:
            print('    drawn on {0!r}, cutting {1!r} -- converted via the '
                  'multiscales scale/translation'.format(origin, source.rel_path))
    nbytes = int(np.prod(shape)) * source.array.dtype.itemsize
    print('  output    {0}'.format(out_path / job.name / 's0'))
    print('    shape   {0} (z, y, x) dtype {1} zarr v{2}'.format(
        shape, source.array.dtype, out_format))
    print('    size    {0:,} voxels, {1:.1f} MB'.format(int(np.prod(shape)),
                                                        nbytes / 1e6))
    print('    scale   {0} {1} (z, y, x)'.format(scale_zyx, unit))
    print('    offset  {0} {1} (z, y, x)'.format(translation_zyx, unit))
    extent = [t + s * n for t, s, n in zip(translation_zyx, scale_zyx, shape)]
    print('    bbox    {0} -> {1} {2}'.format(
        [round(v, 3) for v in translation_zyx], [round(v, 3) for v in extent], unit))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    has_flags = args.roi_min is not None or args.roi_max is not None
    if has_flags and args.neuroglancer:
        raise CropError('Give either --min/--max or --neuroglancer, not both.')
    if not has_flags and not args.neuroglancer:
        raise CropError('No ROI given. Pass --min and --max, or --neuroglancer.')

    parsed = None
    if args.neuroglancer:
        parsed = ng.parse(read_link_text(args.neuroglancer), layer_name=args.layer,
                          axis_order=args.axis_order)

    out_path = Path(args.out)
    if out_path.suffix != '.zarr':
        raise CropError(
            '--out must name a folder ending in .zarr (got {0}). This matches the '
            'plugin and is what Neuroglancer expects.'.format(out_path.name))

    source = Source(resolve_source_path(args, parsed))
    source.validate_channel_time(args.channel, args.time)
    check_output_outside_source(args.out, source.container)

    out_format = int(args.zarr_format) if args.zarr_format else source.zarr_format
    chunks = parse_chunks(args.chunks)
    unit = args.unit or source.spatial_unit()

    drawn = drawn_level(parsed, source)
    jobs = build_jobs(args, source, parsed, out_path, drawn)

    report_source(source, unit)
    if parsed is not None:
        print('  link      {0} box(es), coordinate space ({1}){2}'.format(
            len(parsed.boxes), ', '.join(parsed.space.order),
            '' if parsed.space.has_scale else ', no unit scale given'))
        if parsed.space.has_scale:
            print('            {0} nm per coordinate unit'.format(
                [parsed.space.nm_per_unit[a] for a in parsed.space.order]))
        if drawn:
            print('  drawn on  level {0!r}{1}'.format(
                drawn, '' if drawn == source.rel_path
                else ' (cutting {0!r})'.format(source.rel_path)))

    results: List[Result] = []
    for job in jobs:
        try:
            results.append(write_crop(source, job, out_path, out_format, chunks,
                                      unit, args, drawn))
        except (CropError, roi_mod.RoiError) as e:
            # One bad box should not cost the rest of the batch.
            print('  FAILED: {0}'.format(e))
            results.append(Result(job, False, str(e)))

    written = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    if len(jobs) > 1 or failed:
        print('\n{0} of {1} crop(s) {2}'.format(
            len(written), len(jobs), 'planned' if args.dry_run else 'written'))
        for r in failed:
            print('  failed  {0}: {1}'.format(r.job.name, r.detail))
    if args.dry_run:
        print('\ndry run: nothing written.')
    else:
        for r in written:
            if r.array_dir:
                print('wrote {0}'.format(r.array_dir))

    return 1 if failed else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        return run(argv)
    except (CropError, roi_mod.RoiError, ng.NeuroglancerError) as e:
        print('error: {0}'.format(e), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
