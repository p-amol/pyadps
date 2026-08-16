# Future Work

Ideas and deferred work for `pyadps`, not yet scheduled. Unlike `CHANGELOG.md`
(what shipped), this tracks what hasn't been built yet. Move an entry to
`CHANGELOG.md`'s `[Unreleased]` section once it's actually implemented, and
delete it from here.

## Processing

### Manual surface/boundary-layer trim after regridding (`trim_surface`)

Regridded data can retain a shallow (or deep) boundary depth bin that's only
*partially* masked by `cut_bins_side_lobe()` — some ensembles' geometric
side-lobe cutoff falls just above the bin, some just below, because the
cutoff depends on `transducer_depth`, which drifts over a deployment. The
formula-driven cutoff doesn't always line up with where the data is actually
contaminated: verified on a real file (`GD15A000.000`) that the ensembles the
formula kept as "valid" at the boundary bin had *higher* echo intensity
(more contaminated) than the ones it rejected.

Proposed: a new function/QC step, `trim_surface`, that lets the user
visually compare a candidate boundary depth cell against its clean
neighbors (time series of speed + echo intensity/correlation) and manually
mask it out.

Design decided so far (2026-08-16 discussion):
- **Masks, doesn't drop** the depth level — consistent with every other cut
  function (`cut_bins_side_lobe`, `cut_bins_manual`, `trim_ensembles`).
- **Velocity-only by default** (matches the "never touch echo/correlation/
  percent_good" mask convention from `get_export_dataset()`/`regrid()`), but
  with an **opt-in "apply to all variables" toggle** — legitimate here
  specifically because the contamination can be in the raw diagnostic
  itself (elevated echo intensity), not just a velocity-derived QC flag.
  This toggle should be a reusable pattern, not one-off for this tool.
- **Cell picker defaults to the boundary cell + next two deeper cells**,
  but the user can select any three.
- **Lives on the Velocity Processing page**, not Profile Operations —
  regrid's result is already `st.session_state`-cached (confirmed in
  `07_Profile_Operations.py`: `runner.regrid()` only runs inside the
  "Preview Regrid" button handler), so this isn't about avoiding
  recomputation. It's a workflow separation: Profile Operations is a
  tune/preview/finalize page people naturally loop back into; putting the
  boundary check on a later page keeps it decoupled from that loop, once
  the regridded dataset is frozen.
- Reuse the chart style from the verification artifact built during that
  session (stacked time-series line charts, one per depth cell, synced
  crosshair/tooltip).

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
