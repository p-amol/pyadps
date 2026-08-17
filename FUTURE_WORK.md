# Future Work

Ideas and deferred work for `pyadps`, not yet scheduled. Unlike `CHANGELOG.md`
(what shipped), this tracks what hasn't been built yet. Move an entry to
`CHANGELOG.md`'s `[Unreleased]` section once it's actually implemented, and
delete it from here.

## Processing

### `end_cell_option` variant that doesn't clamp to the surface

`regrid()`'s `end_cell_option="cell"` clamps the grid's shallow boundary to
not cross the surface (see the `if last_grid_depth_calc < 0` clamp for
upward-looking instruments). A new option would skip that clamp and let the
grid extend into physically-above-surface (negative depth) bins using the
raw calculated extent instead - useful for diagnostics/QA, to see the full
raw cell range without early clamping.

### Beam-coordinate processing

`pyadps` currently only supports Earth-coordinate processing. Several
signal-quality functions' docstrings (`signal_quality.md`, "Future Work"
section) note that genuine per-ping, per-beam auto-detection of a bad beam
(rather than requiring `beam_ignore` to name it) is only solvable in raw,
single-ping beam-coordinate data, which pyadps doesn't currently read or
process. Adding beam-coordinate support would be a prerequisite for that,
and opens up other beam-level analysis this package can't currently do.

## Autoprocess / CLI

### CSV output support in `autoprocess()`

`autoprocess()`/`pyadps-auto`/the Add-Ons Auto Processing tool only ever
produce NetCDF (documented as a deliberate scope limit - `config.ini`
records which components were exported but not the output *format*). CSV
export is currently only available interactively via the Write File page.
Adding config-driven CSV output would remove that gap, if there's demand
for fully non-interactive CSV reprocessing.
