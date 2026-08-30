# zarr-crop

Cut a crop out of a collaborator's OME-Zarr tomogram and write it as a new
`.zarr` container — without Amira.

This replaces steps 3–5 of [crop-making-protocol.html](crop-making-protocol.html)
(*Make the crop in Amira* → *Clean up the crop edges* → *Export the crop to
Zarr*). Everything before and after is unchanged: you still pick the ROI in
Fileglancer/Neuroglancer, and annotators still work in Amira, opening the crop
with `ZarrRead` and exporting labels with `ZarrWrite` exactly as before.

**The source is never modified.** It is opened read-only, and the tool refuses
to run at all if the output path resolves inside the source container.

## Why

The Amira leg of the protocol is a GUI wrapper around a slice-and-copy, and it
carries two avoidable problems:

**The divisor guessing game.** Fileglancer reports `s0` coordinates, but Amira's
X/Y/Z limit boxes are indices into whichever level you loaded. The protocol's
documented workaround is to "start dividing by 2; if the crop doesn't light up
in the right place, try 4, then 6 (or 8)" — and to keep a per-team spreadsheet
of which divisor each team uses. None of that is necessary: every level's
`scale` and `translation` are recorded in the OME-NGFF multiscales metadata, so
the conversion is exact. Pass `s0` numbers, cut from any level, and the tool
does the arithmetic.

It is also more accurate than dividing. Pyramid levels usually carry a
half-voxel `translation` offset, because a coarse voxel's centre sits at the
centre of the block of fine voxels it covers. `zarr-crop` routes coordinates
through physical space:

```
physical = translation_ref[axis] + index * scale_ref[axis]
index_L  = (physical - translation_L[axis]) / scale_L[axis]
```

which absorbs that offset. Dividing by 2 does not, and silently shifts the crop
by a voxel. See `test_s0_coords_convert_to_s1_absorbing_the_half_voxel_offset`
in [tests/test_roi.py](tests/test_roi.py) for the worked arithmetic.

**Axis order.** The protocol flags that coordinates come out `XYZ` on some
datasets and `ZYX` on others. `--axis-order` is required with `--min`/`--max`
precisely because there is no safe default; with `--neuroglancer` the order is
read from the state's `dimensions`.

## Setup

Uses micromamba against conda-forge — the same approach as
`cellmap-segmentation-challenge`, and no Anaconda `defaults` channel, so no
commercial-licensing question.

If you don't have micromamba:

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

Then create the environment and install the tool:

```bash
micromamba create -f environment.yml -y
micromamba run -n crop_tool python -m pip install -e . --no-deps
```

Check it:

```bash
micromamba run -n crop_tool python -m pytest -q
```

## Usage

Point `--source` at a **level** (an array such as `.../recon.zarr/em/s0`), not
at the container root or an intermediate group.

Cutting from `s0`, with an ROI read off Fileglancer in XYZ order:

```bash
micromamba run -n crop_tool zarr-crop \
  --source /nrs/lab/FuncEworm/recon.zarr/em/s0 \
  --out    /nrs/lab/crops/FuncEworm_soma_7.zarr \
  --min 1024,2048,512 --max 1536,2560,768 --axis-order xyz
```

Cutting from `s2` using **the same `s0` numbers** — no division by hand:

```bash
micromamba run -n crop_tool zarr-crop \
  --source /nrs/lab/FuncEworm/recon.zarr/em/s2 \
  --out    /nrs/lab/crops/FuncEworm_soma_7.zarr \
  --min 1024,2048,512 --max 1536,2560,768 --axis-order xyz
```

### Pasting a Neuroglancer link

Draw one or more bounding boxes in Neuroglancer, then hand the link straight to
the tool — no numbers get retyped, and no axis order to get wrong:

```bash
micromamba run -n crop_tool zarr-crop \
  --neuroglancer 'https://neuroglancer-demo.appspot.com/#!{...}' \
  --source /nrs/lab/FuncEworm/recon.zarr/em/s0 \
  --out    /nrs/lab/crops.zarr \
  --name-prefix Pancreas5_soma --start-number 7
```

`--neuroglancer` takes the link text itself, a file containing it, or `-` for
stdin (so `pbpaste | zarr-crop … --neuroglancer -` works).

**Every box in the link becomes a crop**, written as sibling groups of one output
container — `crops.zarr/Pancreas5_soma_7/s0`, `_8`, `_9`, … `--start-number`
continues an existing series so nothing needs renaming afterwards. If one box is
bad (outside the volume, say), the rest are still written and the run exits
non-zero naming the failure.

What the tool takes from the link:

| From the state | Used for |
|---|---|
| `dimensions` key order | Axis order — `zyx` in some datasets, `xyz` in others |
| `dimensions` scale + unit | Converting coordinates to physical units |
| bounding-box annotations | One ROI per box, across all annotation layers |
| the image layer's `source` | The source array, when it resolves to a path |
| the level named in that URL | Which level to cut, and what the coordinates mean |

### Which level gets cut

**A box drawn on a level is cut from that level.** If you open `…/crop.zarr/s1`
in Fileglancer and draw there, the crop comes out of `s1` — `--source` is
optional, and nothing is converted. To cut a *different* level, name it with
`--source` and the coordinates are converted for you:

```bash
# box drawn on s1, but cut at full resolution
zarr-crop --neuroglancer 'LINK' --source .../crop.zarr/s0 --out crops.zarr --name-prefix soma
```

This matters because opening a level folder gives Neuroglancer a bare array with
no OME metadata, so its coordinate space is that level's raw voxel indices with
no unit. Treating those as `s0` indices would silently halve an `s1` box.

Opening the **container root** instead gives Neuroglancer the multiscale group,
so the state carries units and coordinates convert through **physical units** —
exact whether or not the space's scale matches the s0 voxel size, and correct even
when the dataset has a non-zero `translation` (where assuming s0 indices would be
wrong).

The summary always prints which level was drawn on and which is being cut, so the
conversion is never an invisible assumption. `--roi-space` overrides everything:
`drawn`, `s0`, `level` (indices of `--source`, plugin parity), or `physical`.

**Source resolution.** `--source` always wins. Otherwise the tool uses the
link's image layer (`--layer NAME` to pick another), which yields a filesystem
path only for Fileglancer-served URLs: `…/files/<key>/groups/lab/x.zarr` maps to
`/groups/lab/x.zarr`. Object-storage URLs such as `https://s3.janelia.org/…`
carry no path, so those need `--source`. A derived path that doesn't exist is
reported rather than assumed.

Shortened Neuroglancer links can't be used — their state lives on a server, so
there is nothing in the URL to decode. Paste the long URL from the address bar,
or the annotation JSON.

Add `--dry-run` to see the resolved indices, output shape and physical bounding
box without writing anything. Worth doing on the first crop of a new dataset.

### Output layout

Matching `ZarrWrite`, the crop lands at `<out>.zarr/<name>/s0`, where `<name>`
defaults to the `--out` filename without `.zarr`. Data is written `(z, y, x)`
on disk regardless of the source's storage order, which is the OME-NGFF
canonical order and what the plugin produces.

Naming follows the protocol's convention — `DatasetName (what's annotated) N`,
e.g. `--out .../Pancreas5_soma_7.zarr`.

### ROI coordinate spaces

| `--roi-space` | Meaning |
|---|---|
| `s0` (default) | Indices at the finest level, as Fileglancer reports them. Converted automatically for whichever level `--source` names. |
| `level` | Indices of `--source` itself — what Amira's limit boxes take. Use this to reproduce old behavior, or when the source has no multiscales metadata. |
| `physical` | Scaled units (usually nanometres). |

Bounds are floored on the low side and ceiled on the high side, so the crop
always **covers** the region you asked for rather than clipping an edge. When
cutting a coarse level the crop can therefore be slightly larger than the ROI;
it is never smaller.

### Trimming crop edges

Protocol step 4 (shifting the volume to drop unwanted end slices) is just a
smaller ROI here — re-run with adjusted `--min`/`--max`. Use `--dry-run` to
check the shape before writing.

## Differences from the Amira plugin

The tool is a deliberate behavioral match for
[amira_python_extensions](https://github.com/janelia-cellmap/amira_python_extensions)'
`ZarrRead` + `ZarrWrite` round-trip, with these differences:

| | Plugin | Here | Why |
|---|---|---|---|
| **dtype** | Narrows uint32/uint64→uint16, int64→int32 | Preserved exactly | The narrowing exists only because Amira's `HxUniformScalarField3` cannot hold those types. Fileglancer and Neuroglancer read Zarr directly and handle them natively. Narrowing uint32→uint16 silently wraps values above 65535. |
| **Output zarr format** | Menu, defaults to v3 | Defaults to the source's format | More predictable across a dataset. Override with `--zarr-format`. |
| **Chunk shape** | Always 128³ | `min(128, dim)` per axis | A 40-slice-deep crop gets a 40-deep chunk instead of a 128-deep one mostly full of fill value. Override with `--chunks`. |
| **IO library** | tensorstore | zarr-python | Per Yurii's suggestion: one less dependency, since zarr-python is needed for the metadata regardless. All zarr access is confined to `zarr_io.py`. |
| **Coordinate handling** | Indices of the loaded level | `s0` by default, converted from metadata | Removes the divisor guesswork. `--roi-space level` restores the old behavior. |

On-disk format parity was checked field by field against the plugin's
tensorstore specs. Zarr **v2** metadata is identical (`compressor
{"id":"zstd","level":3}`, `dimension_separator "/"`). Zarr **v3** differs only
in that zarr-python writes two spec defaults that tensorstore leaves implicit —
`bytes.configuration.endian = "little"` and `zstd.configuration.checksum =
false`. Both libraries read either form; see
`test_on_disk_format_matches_the_plugin_spec_v3`.

### Verifying compatibility yourself

[verification/check_plugin_compat.py](verification/check_plugin_compat.py)
checks the claim that matters to annotators — that opening a `zarr-crop` output
in Amira gives the same thing as the old Amira-only workflow. It reads crops
using the plugin's own `ZarrRead` functions, copied verbatim into
[verification/plugin_reader.py](verification/plugin_reader.py), and compares
against `ZarrRead`-ing the same ROI straight from the source. Equal voxels mean
the data is right; an equal Amira bounding box means the crop is *positioned*
right, which is what makes it line up in Neuroglancer.

It lives outside the pytest suite because it needs tensorstore, which the tool
does not depend on. Setup instructions are in the script's docstring. Currently
passing for zarr v2 and v3, cropping both `s0` and `s2`. Re-run it after
changing `zarr_io.py`.

## Status

156/156 tests pass (`pytest`). No known bugs or incomplete features. Open items
before production sign-off:

- **License** — not yet set; pending a decision from Janelia Scientific
  Computing.
- **CI** — no automated test runner configured yet; tests are currently run
  manually.
- **Plugin round-trip verification** — `verification/check_plugin_compat.py`
  needs `tensorstore` (not a runtime dependency) and is run manually; it is not
  part of the `pytest` suite or any CI.

## Scope

Does **not** do: ROI selection (that stays in Fileglancer/Neuroglancer),
multiple crops per run, downsampled pyramids in the output (the plugin writes a
single `s0` too), label creation, or anything downstream of step 5. Shortened
Neuroglancer links are not supported since their state lives on a server — paste
the long URL or the annotation JSON.

## Layout

```
src/crop_tool/
├── zarr_io.py       all zarr access; helpers ported from the plugin
├── roi.py           coordinate-space conversion and ROI resolution
├── neuroglancer.py  state, layer sources, coordinate space, boxes
└── cli.py           argument handling, validation, slab-wise copy, summary
tests/
├── conftest.py           synthetic OME-Zarr fixtures
├── test_roi.py           coordinate maths
├── test_crop.py          end-to-end, metadata, source immutability
├── test_neuroglancer.py  state parsing, chunk-aligned slabs
└── test_batch.py         links, multi-box batches, source resolution
tools/
└── make_test_dataset.py  generates a throwaway dataset to test against
verification/
├── plugin_reader.py       ZarrRead's functions, copied verbatim
└── check_plugin_compat.py Amira round-trip equivalence (needs tensorstore)
```

### Making a dataset to test against

`tools/make_test_dataset.py` writes a throwaway OME-Zarr (both v2 and v3) built
so mistakes are visible: every axis a different length, countable stripes per
axis, and landmark spheres at documented coordinates. Point Fileglancer at it,
draw boxes round the spheres, and check the right sphere lands in each crop.

```bash
micromamba run -n crop_tool python tools/make_test_dataset.py /path/to/scratch
```

The ported helpers in `zarr_io.py` are kept recognizably close to their
originals on purpose, so fixes can be carried between this tool and the plugin.

Test fixtures use the plugin's own coordinate-encoding trick (each voxel holds
a value encoding its own index), so a mis-slice or bad transpose is visible on
inspection. The multi-level fixtures with half-voxel offsets are additions —
the plugin's own fixtures are single-level and cannot exercise the conversion.
